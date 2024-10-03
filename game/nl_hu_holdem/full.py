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
struct Union9;
__device__ void block_matmul_44(float * v0, float * v1, int v2, float * v3);
__device__ void block_row_map_45(float * v0, int v1, float * v2);
__device__ int int_range_46(int v0, int v1, curandStatePhilox4_32_10_t & v2);
struct Tuple14;
struct Tuple15;
struct Tuple16;
struct Tuple17;
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
__device__ void play_loop_31(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut0 & v3, Union4 v4);
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
__device__ void f_52(unsigned char * v0, unsigned long long v1, Union3 v2, static_array_list<Union6,128> v3, static_array<Union2,2> v4, Union7 v5);
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
struct Union2_0 { // CallingMachine
};
struct Union2_1 { // Computer
};
struct Union2_2 { // Human
};
struct Union2_3 { // Random
};
struct Union2 {
    union {
        Union2_0 case0; // CallingMachine
        Union2_1 case1; // Computer
        Union2_2 case2; // Human
        Union2_3 case3; // Random
    };
    unsigned char tag{255};
    __device__ Union2() {}
    __device__ Union2(Union2_0 t) : tag(0), case0(t) {} // CallingMachine
    __device__ Union2(Union2_1 t) : tag(1), case1(t) {} // Computer
    __device__ Union2(Union2_2 t) : tag(2), case2(t) {} // Human
    __device__ Union2(Union2_3 t) : tag(3), case3(t) {} // Random
    __device__ Union2(Union2 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(x.case0); break; // CallingMachine
            case 1: new (&this->case1) Union2_1(x.case1); break; // Computer
            case 2: new (&this->case2) Union2_2(x.case2); break; // Human
            case 3: new (&this->case3) Union2_3(x.case3); break; // Random
        }
    }
    __device__ Union2(Union2 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(std::move(x.case0)); break; // CallingMachine
            case 1: new (&this->case1) Union2_1(std::move(x.case1)); break; // Computer
            case 2: new (&this->case2) Union2_2(std::move(x.case2)); break; // Human
            case 3: new (&this->case3) Union2_3(std::move(x.case3)); break; // Random
        }
    }
    __device__ Union2 & operator=(Union2 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // CallingMachine
                case 1: this->case1 = x.case1; break; // Computer
                case 2: this->case2 = x.case2; break; // Human
                case 3: this->case3 = x.case3; break; // Random
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
                case 0: this->case0 = std::move(x.case0); break; // CallingMachine
                case 1: this->case1 = std::move(x.case1); break; // Computer
                case 2: this->case2 = std::move(x.case2); break; // Human
                case 3: this->case3 = std::move(x.case3); break; // Random
            }
        } else {
            this->~Union2();
            new (this) Union2{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union2() {
        switch(this->tag){
            case 0: this->case0.~Union2_0(); break; // CallingMachine
            case 1: this->case1.~Union2_1(); break; // Computer
            case 2: this->case2.~Union2_2(); break; // Human
            case 3: this->case3.~Union2_3(); break; // Random
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
struct Union0_3 { // StartTrainingVsRando
};
struct Union0_4 { // StartTrainingVsSelf
};
struct Union0 {
    union {
        Union0_0 case0; // ActionSelected
        Union0_1 case1; // PlayerChanged
        Union0_2 case2; // StartGame
        Union0_3 case3; // StartTrainingVsRando
        Union0_4 case4; // StartTrainingVsSelf
    };
    unsigned char tag{255};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // ActionSelected
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // PlayerChanged
    __device__ Union0(Union0_2 t) : tag(2), case2(t) {} // StartGame
    __device__ Union0(Union0_3 t) : tag(3), case3(t) {} // StartTrainingVsRando
    __device__ Union0(Union0_4 t) : tag(4), case4(t) {} // StartTrainingVsSelf
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // ActionSelected
            case 1: new (&this->case1) Union0_1(x.case1); break; // PlayerChanged
            case 2: new (&this->case2) Union0_2(x.case2); break; // StartGame
            case 3: new (&this->case3) Union0_3(x.case3); break; // StartTrainingVsRando
            case 4: new (&this->case4) Union0_4(x.case4); break; // StartTrainingVsSelf
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // ActionSelected
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // PlayerChanged
            case 2: new (&this->case2) Union0_2(std::move(x.case2)); break; // StartGame
            case 3: new (&this->case3) Union0_3(std::move(x.case3)); break; // StartTrainingVsRando
            case 4: new (&this->case4) Union0_4(std::move(x.case4)); break; // StartTrainingVsSelf
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // ActionSelected
                case 1: this->case1 = x.case1; break; // PlayerChanged
                case 2: this->case2 = x.case2; break; // StartGame
                case 3: this->case3 = x.case3; break; // StartTrainingVsRando
                case 4: this->case4 = x.case4; break; // StartTrainingVsSelf
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
                case 3: this->case3 = std::move(x.case3); break; // StartTrainingVsRando
                case 4: this->case4 = std::move(x.case4); break; // StartTrainingVsSelf
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
            case 3: this->case3.~Union0_3(); break; // StartTrainingVsRando
            case 4: this->case4.~Union0_4(); break; // StartTrainingVsSelf
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
    static_array_list<Union6,128> v2;
    static_array<Union2,2> v3;
    Union7 v4;
    __device__ Tuple1() = default;
    __device__ Tuple1(unsigned long long t0, Union3 t1, static_array_list<Union6,128> t2, static_array<Union2,2> t3, Union7 t4) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4) {}
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
    static_array_list<Union6,128> v2;
    static_array<Union2,2> v3;
    curandStatePhilox4_32_10_t v4;
    Union7 v5;
    __device__ StackMut0() = default;
    __device__ StackMut0(unsigned long long t0, Union3 t1, static_array_list<Union6,128> t2, static_array<Union2,2> t3, curandStatePhilox4_32_10_t t4, Union7 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
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
struct Tuple14 {
    int v0;
    float v1;
    __device__ Tuple14() = default;
    __device__ Tuple14(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Closure2 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple15 {
    float v0;
    bool v1;
    __device__ Tuple15() = default;
    __device__ Tuple15(float t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ Tuple15 operator()(Tuple15 tup0, Tuple15 tup1){
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
                return Tuple15{v5, true};
            } else {
                return Tuple15{v0, v1};
            }
        } else {
            if (v3){
                return Tuple15{v2, v3};
            } else {
                return Tuple15{v0, v1};
            }
        }
    }
};
struct Tuple16 {
    float v0;
    int v1;
    __device__ Tuple16() = default;
    __device__ Tuple16(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
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
struct Tuple17 {
    int v0;
    bool v1;
    __device__ Tuple17() = default;
    __device__ Tuple17(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure5 {
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
        case 3: {
            f_3(v2);
            return Union2{Union2_3{}};
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
        case 3: {
            f_3(v2);
            return Union0{Union0_3{}};
            break;
        }
        case 4: {
            f_3(v2);
            return Union0{Union0_4{}};
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
    static_array_list<Union6,128> v10;
    v10 = static_array_list<Union6,128>{};
    int v12;
    v12 = f_20(v0);
    v10.unsafe_set_length(v12);
    int v13;
    v13 = v10.length;
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
            while (while_method_2(v25)){
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
            while (while_method_3(v15)){
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
    while (while_method_1(v34)){
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
__device__ bool player_can_act_37(int v0, static_array<static_array<unsigned char,2>,2> v1, static_array<int,2> v2, int v3, static_array<int,2> v4, Union5 v5){
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
    Tuple4 tmp19 = Tuple4{1, v21};
    v23 = tmp19.v0; v24 = tmp19.v1;
    while (while_method_0(v23)){
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
    Tuple4 tmp20 = Tuple4{0, 0};
    v36 = tmp20.v0; v37 = tmp20.v1;
    while (while_method_0(v36)){
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
            while (while_method_2(v25)){
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
            while (while_method_3(v15)){
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
    Tuple13 tmp29 = Tuple13{0, v3};
    v8 = tmp29.v0; v9 = tmp29.v1;
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
    Tuple13 tmp30 = Tuple13{0, v3};
    v8 = tmp30.v0; v9 = tmp30.v1;
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
    v1 = v0 < 32;
    return v1;
}
__device__ inline bool while_method_11(int v0){
    bool v1;
    v1 = v0 < 8;
    return v1;
}
__device__ inline bool while_method_12(int v0){
    bool v1;
    v1 = v0 < 64;
    return v1;
}
__device__ inline bool while_method_13(int v0){
    bool v1;
    v1 = v0 < 16;
    return v1;
}
__device__ void block_matmul_44(float * v0, float * v1, int v2, float * v3){
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 24);
    int v5;
    v5 = 524288 * v4;
    int v6;
    v6 = blockIdx.x;
    assert("Tensor range check" && 0 <= v6 && v6 < 24);
    int v7;
    v7 = 16384 * v6;
    cuda::pipeline<cuda::thread_scope_thread> v8 = cuda::make_pipeline();
    extern __shared__ unsigned char v9[];
    float * v10;
    v10 = reinterpret_cast<float *>(&v9[0ull]);
    float * v12;
    v12 = reinterpret_cast<float *>(&v9[36864ull]);
    float * v14;
    v14 = reinterpret_cast<float *>(&v9[0ull]);
    int v16;
    v16 = threadIdx.x;
    int v17;
    v17 = v16 / 32;
    bool v18;
    v18 = 0 <= v17;
    bool v19;
    v19 = v18 == false;
    if (v19){
        assert("The index needs to be zero or positive." && v18);
    } else {
    }
    int v21;
    v21 = v17 % 4;
    int v22;
    v22 = v17 / 4;
    bool v23;
    v23 = v22 < 2;
    bool v24;
    v24 = v23 == false;
    if (v24){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v23);
    } else {
    }
    assert("Tensor range check" && 0 <= v22 && v22 < 2);
    assert("Tensor range check" && 0 <= v21 && v21 < 4);
    int v26;
    v26 = 16 * v21;
    int v27;
    v27 = 9216 * v22;
    int v28;
    v28 = v27 + v26;
    float * v29;
    v29 = v14+v28;
    assert("Tensor range check" && 0 <= v22 && v22 < 2);
    int v31;
    v31 = 4608 * v22;
    int v32;
    v32 = threadIdx.x;
    int v33;
    v33 = v32 % 32;
    bool v34;
    v34 = 0 <= v33;
    bool v35;
    v35 = v34 == false;
    if (v35){
        assert("The index needs to be zero or positive." && v34);
    } else {
    }
    int v37;
    v37 = v33 % 4;
    int v38;
    v38 = v33 / 4;
    bool v39;
    v39 = v38 < 8;
    bool v40;
    v40 = v39 == false;
    if (v40){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v39);
    } else {
    }
    assert("Tensor range check" && 0 <= v38 && v38 < 8);
    assert("Tensor range check" && 0 <= v37 && v37 < 4);
    int v42;
    v42 = v37 + v31;
    int v43;
    v43 = 36 * v38;
    int v44;
    v44 = v43 + v42;
    float * v45;
    v45 = v10+v44;
    assert("Tensor range check" && 0 <= v21 && v21 < 4);
    int v47;
    v47 = 576 * v21;
    int v48;
    v48 = threadIdx.x;
    int v49;
    v49 = v48 % 32;
    bool v50;
    v50 = 0 <= v49;
    bool v51;
    v51 = v50 == false;
    if (v51){
        assert("The index needs to be zero or positive." && v50);
    } else {
    }
    int v53;
    v53 = v49 % 4;
    int v54;
    v54 = v49 / 4;
    bool v55;
    v55 = v54 < 8;
    bool v56;
    v56 = v55 == false;
    if (v56){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v55);
    } else {
    }
    assert("Tensor range check" && 0 <= v54 && v54 < 8);
    assert("Tensor range check" && 0 <= v53 && v53 < 4);
    int v58;
    v58 = v53 + v47;
    int v59;
    v59 = 36 * v54;
    int v60;
    v60 = v59 + v58;
    float * v61;
    v61 = v12+v60;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> v63[8];
    int v64;
    v64 = 0;
    while (while_method_6(v64)){
        int v66;
        v66 = 0;
        while (while_method_6(v66)){
            assert("Tensor range check" && 0 <= v64 && v64 < 1);
            assert("Tensor range check" && 0 <= v66 && v66 < 1);
            int v68;
            v68 = 64 * v66;
            int v69;
            v69 = v68 + v7;
            int v70;
            v70 = 16384 * v64;
            int v71;
            v71 = v70 + v69;
            float * v72;
            v72 = v0+v71;
            // Pushing the loop unrolling to: 0
            int v74;
            v74 = 0;
            #pragma unroll
            while (while_method_11(v74)){
                int v76;
                v76 = 0;
                #pragma unroll
                while (while_method_6(v76)){
                    assert("Tensor range check" && 0 <= v74 && v74 < 8);
                    assert("Tensor range check" && 0 <= v76 && v76 < 1);
                    int v78;
                    v78 = v74 + v76;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v79 = v63[v78];
                    wmma::fill_fragment(v79, 0.0f);
                    v76 += 1 ;
                }
                v74 += 1 ;
            }
            // Poping the loop unrolling to: 0
            int v80;
            v80 = 0;
            while (while_method_12(v80)){
                int v82;
                v82 = v80 + 1;
                bool v83;
                v83 = v80 == 0;
                int v84;
                v84 = v80 % 2;
                bool v85;
                v85 = 0 <= v80;
                bool v86;
                v86 = v85 == false;
                if (v86){
                    assert("The index needs to be zero or positive." && v85);
                } else {
                }
                bool v88;
                v88 = v80 < 64;
                bool v89;
                v89 = v88 == false;
                if (v89){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v88);
                } else {
                }
                bool v91;
                v91 = v82 < 64;
                Union9 v97;
                if (v91){
                    bool v92;
                    v92 = 0 <= v82;
                    bool v93;
                    v93 = v92 == false;
                    if (v93){
                        assert("The index needs to be zero or positive." && v92);
                    } else {
                    }
                    v97 = Union9{Union9_1{v82}};
                } else {
                    v97 = Union9{Union9_0{}};
                }
                assert("Tensor range check" && 0 <= v64 && v64 < 1);
                int v98;
                v98 = 524288 * v64;
                int v99;
                v99 = v98 + v5;
                assert("Tensor range check" && 0 <= v80 && v80 < 64);
                int v100;
                v100 = 32 * v80;
                int v101;
                v101 = v100 + v99;
                float * v102;
                v102 = v3+v101;
                assert("Tensor range check" && 0 <= v66 && v66 < 1);
                int v104;
                v104 = 131072 * v66;
                int v105;
                v105 = v104 + v2;
                if (v83){
                    assert("Tensor range check" && 0 <= v80 && v80 < 64);
                    int v106;
                    v106 = v100 + v105;
                    float * v107;
                    v107 = v1+v106;
                    // Pushing the loop unrolling to: 0
                    v8.producer_acquire();
                    int v109;
                    v109 = threadIdx.x;
                    bool v110;
                    v110 = 0 <= v109;
                    bool v111;
                    v111 = v110 == false;
                    if (v111){
                        assert("The index needs to be zero or positive." && v110);
                    } else {
                    }
                    int v113;
                    v113 = v109 % 8;
                    int v114;
                    v114 = v109 / 8;
                    bool v115;
                    v115 = v114 < 32;
                    bool v116;
                    v116 = v115 == false;
                    if (v116){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v115);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v114 && v114 < 32);
                    assert("Tensor range check" && 0 <= v113 && v113 < 8);
                    int v118;
                    v118 = 4 * v113;
                    int v119;
                    v119 = 36 * v114;
                    int v120;
                    v120 = v119 + v118;
                    int v121;
                    v121 = 2048 * v114;
                    int v122;
                    v122 = v121 + v118;
                    float * v123;
                    v123 = v12+v120;
                    float * v125;
                    v125 = v107+v122;
                    int v127;
                    v127 = 0;
                    #pragma unroll
                    while (while_method_0(v127)){
                        int v129;
                        v129 = 0;
                        #pragma unroll
                        while (while_method_6(v129)){
                            assert("Tensor range check" && 0 <= v127 && v127 < 2);
                            assert("Tensor range check" && 0 <= v129 && v129 < 1);
                            int v131;
                            v131 = 32 * v129;
                            int v132;
                            v132 = 1152 * v127;
                            int v133;
                            v133 = v132 + v131;
                            int v134;
                            v134 = 65536 * v127;
                            int v135;
                            v135 = v134 + v131;
                            constexpr int v136 = sizeof(float) * 4;
                            assert("Pointer alignment check" && (unsigned long long)(v125 + v135) % v136 == 0 && (unsigned long long)(v123 + v133) % v136 == 0);
                            cuda::memcpy_async(v123 + v133, v125 + v135, cuda::aligned_size_t<v136>(v136), v8);
                            v129 += 1 ;
                        }
                        v127 += 1 ;
                    }
                    v8.producer_commit();
                    // Poping the loop unrolling to: 0
                } else {
                }
                // Pushing the loop unrolling to: 0
                int v137;
                v137 = threadIdx.x;
                bool v138;
                v138 = 0 <= v137;
                bool v139;
                v139 = v138 == false;
                if (v139){
                    assert("The index needs to be zero or positive." && v138);
                } else {
                }
                int v141;
                v141 = v137 % 8;
                int v142;
                v142 = v137 / 8;
                bool v143;
                v143 = v142 < 32;
                bool v144;
                v144 = v143 == false;
                if (v144){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v143);
                } else {
                }
                assert("Tensor range check" && 0 <= v142 && v142 < 32);
                assert("Tensor range check" && 0 <= v141 && v141 < 8);
                int v146;
                v146 = 4 * v141;
                int v147;
                v147 = 36 * v142;
                int v148;
                v148 = v147 + v146;
                int v149;
                v149 = 2048 * v142;
                int v150;
                v150 = v149 + v146;
                float * v151;
                v151 = v10+v148;
                float * v153;
                v153 = v102+v150;
                int v155;
                v155 = 0;
                #pragma unroll
                while (while_method_11(v155)){
                    int v157;
                    v157 = 0;
                    #pragma unroll
                    while (while_method_6(v157)){
                        assert("Tensor range check" && 0 <= v155 && v155 < 8);
                        assert("Tensor range check" && 0 <= v157 && v157 < 1);
                        int v159;
                        v159 = 32 * v157;
                        int v160;
                        v160 = 1152 * v155;
                        int v161;
                        v161 = v160 + v159;
                        int v162;
                        v162 = 65536 * v155;
                        int v163;
                        v163 = v162 + v159;
                        int4* v164;
                        v164 = reinterpret_cast<int4*>(v153 + v163);
                        int4* v165;
                        v165 = reinterpret_cast<int4*>(v151 + v161);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v164) % 16 == 0 && reinterpret_cast<unsigned long long>(v165) % 16 == 0);
                        *v165 = *v164;
                        v157 += 1 ;
                    }
                    v155 += 1 ;
                }
                // Poping the loop unrolling to: 0
                wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v166[1];
                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v167[4];
                cuda::pipeline_consumer_wait_prior<0>(v8);;
                __syncthreads();
                // Pushing the loop unrolling to: 0
                int v168;
                v168 = 0;
                #pragma unroll
                while (while_method_6(v168)){
                    int v170;
                    v170 = 0;
                    #pragma unroll
                    while (while_method_3(v170)){
                        assert("Tensor range check" && 0 <= v168 && v168 < 1);
                        assert("Tensor range check" && 0 <= v170 && v170 < 4);
                        int v172;
                        v172 = 4 * v168;
                        int v173;
                        v173 = v172 + v170;
                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v174 = v167[v173];
                        assert("Tensor range check" && 0 <= v168 && v168 < 1);
                        int v175;
                        v175 = 576 * v168;
                        assert("Tensor range check" && 0 <= v170 && v170 < 4);
                        int v176;
                        v176 = 8 * v170;
                        int v177;
                        v177 = v176 + v175;
                        int v178;
                        v178 = 0;
                        #pragma unroll
                        while (while_method_0(v178)){
                            int v180;
                            v180 = 0;
                            #pragma unroll
                            while (while_method_0(v180)){
                                assert("Tensor range check" && 0 <= v178 && v178 < 2);
                                assert("Tensor range check" && 0 <= v180 && v180 < 2);
                                int v182;
                                v182 = 4 * v180;
                                int v183;
                                v183 = v182 + v177;
                                int v184;
                                v184 = 288 * v178;
                                int v185;
                                v185 = v184 + v183;
                                float v186;
                                v186 = v61[v185];
                                bool v187;
                                v187 = 0 <= v180;
                                bool v189;
                                if (v187){
                                    bool v188;
                                    v188 = v180 < 2;
                                    v189 = v188;
                                } else {
                                    v189 = false;
                                }
                                bool v190;
                                v190 = v189 == false;
                                if (v190){
                                    assert("The indices should be inside the range of the dimension." && v189);
                                } else {
                                }
                                bool v192;
                                v192 = 0 <= v178;
                                bool v194;
                                if (v192){
                                    bool v193;
                                    v193 = v178 < 2;
                                    v194 = v193;
                                } else {
                                    v194 = false;
                                }
                                bool v195;
                                v195 = v194 == false;
                                if (v195){
                                    assert("The indices should be inside the range of the dimension." && v194);
                                } else {
                                }
                                int v197;
                                v197 = v178 * 2;
                                int v198;
                                v198 = v180 + v197;
                                v174.x[v198] = wmma::__float_to_tf32(v186);
                                v180 += 1 ;
                            }
                            v178 += 1 ;
                        }
                        v170 += 1 ;
                    }
                    v168 += 1 ;
                }
                // Poping the loop unrolling to: 0
                v8.consumer_release();
                switch (v97.tag) {
                    case 0: { // None
                        break;
                    }
                    case 1: { // Some
                        int v199 = v97.case1.v0;
                        assert("Tensor range check" && 0 <= v199 && v199 < 64);
                        int v200;
                        v200 = 32 * v199;
                        int v201;
                        v201 = v200 + v105;
                        float * v202;
                        v202 = v1+v201;
                        __syncthreads();
                        // Pushing the loop unrolling to: 0
                        v8.producer_acquire();
                        int v204;
                        v204 = threadIdx.x;
                        bool v205;
                        v205 = 0 <= v204;
                        bool v206;
                        v206 = v205 == false;
                        if (v206){
                            assert("The index needs to be zero or positive." && v205);
                        } else {
                        }
                        int v208;
                        v208 = v204 % 8;
                        int v209;
                        v209 = v204 / 8;
                        bool v210;
                        v210 = v209 < 32;
                        bool v211;
                        v211 = v210 == false;
                        if (v211){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v210);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v209 && v209 < 32);
                        assert("Tensor range check" && 0 <= v208 && v208 < 8);
                        int v213;
                        v213 = 4 * v208;
                        int v214;
                        v214 = 36 * v209;
                        int v215;
                        v215 = v214 + v213;
                        int v216;
                        v216 = 2048 * v209;
                        int v217;
                        v217 = v216 + v213;
                        float * v218;
                        v218 = v12+v215;
                        float * v220;
                        v220 = v202+v217;
                        int v222;
                        v222 = 0;
                        #pragma unroll
                        while (while_method_0(v222)){
                            int v224;
                            v224 = 0;
                            #pragma unroll
                            while (while_method_6(v224)){
                                assert("Tensor range check" && 0 <= v222 && v222 < 2);
                                assert("Tensor range check" && 0 <= v224 && v224 < 1);
                                int v226;
                                v226 = 32 * v224;
                                int v227;
                                v227 = 1152 * v222;
                                int v228;
                                v228 = v227 + v226;
                                int v229;
                                v229 = 65536 * v222;
                                int v230;
                                v230 = v229 + v226;
                                constexpr int v231 = sizeof(float) * 4;
                                assert("Pointer alignment check" && (unsigned long long)(v220 + v230) % v231 == 0 && (unsigned long long)(v218 + v228) % v231 == 0);
                                cuda::memcpy_async(v218 + v228, v220 + v230, cuda::aligned_size_t<v231>(v231), v8);
                                v224 += 1 ;
                            }
                            v222 += 1 ;
                        }
                        v8.producer_commit();
                        // Poping the loop unrolling to: 0
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                // Pushing the loop unrolling to: 0
                int v232;
                v232 = 0;
                #pragma unroll
                while (while_method_11(v232)){
                    int v234;
                    v234 = 0;
                    #pragma unroll
                    while (while_method_3(v234)){
                        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v236 = v166[0];
                        assert("Tensor range check" && 0 <= v232 && v232 < 8);
                        int v237;
                        v237 = 576 * v232;
                        assert("Tensor range check" && 0 <= v234 && v234 < 4);
                        int v238;
                        v238 = 8 * v234;
                        int v239;
                        v239 = v238 + v237;
                        int v240;
                        v240 = 0;
                        #pragma unroll
                        while (while_method_0(v240)){
                            int v242;
                            v242 = 0;
                            #pragma unroll
                            while (while_method_0(v242)){
                                assert("Tensor range check" && 0 <= v240 && v240 < 2);
                                assert("Tensor range check" && 0 <= v242 && v242 < 2);
                                int v244;
                                v244 = 288 * v242;
                                int v245;
                                v245 = v244 + v239;
                                int v246;
                                v246 = 4 * v240;
                                int v247;
                                v247 = v246 + v245;
                                float v248;
                                v248 = v45[v247];
                                bool v249;
                                v249 = 0 <= v242;
                                bool v251;
                                if (v249){
                                    bool v250;
                                    v250 = v242 < 2;
                                    v251 = v250;
                                } else {
                                    v251 = false;
                                }
                                bool v252;
                                v252 = v251 == false;
                                if (v252){
                                    assert("The indices should be inside the range of the dimension." && v251);
                                } else {
                                }
                                bool v254;
                                v254 = 0 <= v240;
                                bool v256;
                                if (v254){
                                    bool v255;
                                    v255 = v240 < 2;
                                    v256 = v255;
                                } else {
                                    v256 = false;
                                }
                                bool v257;
                                v257 = v256 == false;
                                if (v257){
                                    assert("The indices should be inside the range of the dimension." && v256);
                                } else {
                                }
                                int v259;
                                v259 = v240 * 2;
                                int v260;
                                v260 = v242 + v259;
                                v236.x[v260] = wmma::__float_to_tf32(v248);
                                v242 += 1 ;
                            }
                            v240 += 1 ;
                        }
                        int v261;
                        v261 = 0;
                        #pragma unroll
                        while (while_method_6(v261)){
                            assert("Tensor range check" && 0 <= v232 && v232 < 8);
                            assert("Tensor range check" && 0 <= v261 && v261 < 1);
                            int v263;
                            v263 = v232 + v261;
                            wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v264 = v63[v263];
                            assert("Tensor range check" && 0 <= v261 && v261 < 1);
                            assert("Tensor range check" && 0 <= v234 && v234 < 4);
                            int v265;
                            v265 = 4 * v261;
                            int v266;
                            v266 = v265 + v234;
                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v267 = v167[v266];
                            wmma::mma_sync(v264, v236, v267, v264);
                            v261 += 1 ;
                        }
                        v234 += 1 ;
                    }
                    v232 += 1 ;
                }
                // Poping the loop unrolling to: 0
                __syncthreads();
                v80 = v82;
            }
            // Pushing the loop unrolling to: 0
            int v268;
            v268 = 0;
            #pragma unroll
            while (while_method_11(v268)){
                int v270;
                v270 = 0;
                #pragma unroll
                while (while_method_6(v270)){
                    assert("Tensor range check" && 0 <= v268 && v268 < 8);
                    assert("Tensor range check" && 0 <= v270 && v270 < 1);
                    int v272;
                    v272 = v268 + v270;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v273 = v63[v272];
                    assert("Tensor range check" && 0 <= v268 && v268 < 8);
                    assert("Tensor range check" && 0 <= v270 && v270 < 1);
                    int v274;
                    v274 = 16 * v270;
                    int v275;
                    v275 = 1152 * v268;
                    int v276;
                    v276 = v275 + v274;
                    float * v277;
                    v277 = v29+v276;
                    wmma::store_matrix_sync(v277, v273, 72, wmma::mem_row_major);
                    v270 += 1 ;
                }
                v268 += 1 ;
            }
            // Poping the loop unrolling to: 0
            __syncthreads();
            // Pushing the loop unrolling to: 0
            int v279;
            v279 = threadIdx.x;
            bool v280;
            v280 = 0 <= v279;
            bool v281;
            v281 = v280 == false;
            if (v281){
                assert("The index needs to be zero or positive." && v280);
            } else {
            }
            int v283;
            v283 = v279 % 16;
            int v284;
            v284 = v279 / 16;
            bool v285;
            v285 = v284 < 16;
            bool v286;
            v286 = v285 == false;
            if (v286){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v285);
            } else {
            }
            assert("Tensor range check" && 0 <= v284 && v284 < 16);
            assert("Tensor range check" && 0 <= v283 && v283 < 16);
            int v288;
            v288 = 4 * v283;
            int v289;
            v289 = 64 * v284;
            int v290;
            v290 = v289 + v288;
            int v291;
            v291 = 72 * v284;
            int v292;
            v292 = v291 + v288;
            float * v293;
            v293 = v72+v290;
            float * v295;
            v295 = v14+v292;
            int v297;
            v297 = 0;
            #pragma unroll
            while (while_method_13(v297)){
                int v299;
                v299 = 0;
                #pragma unroll
                while (while_method_6(v299)){
                    assert("Tensor range check" && 0 <= v297 && v297 < 16);
                    assert("Tensor range check" && 0 <= v299 && v299 < 1);
                    int v301;
                    v301 = 64 * v299;
                    int v302;
                    v302 = 1024 * v297;
                    int v303;
                    v303 = v302 + v301;
                    int v304;
                    v304 = 1152 * v297;
                    int v305;
                    v305 = v304 + v301;
                    int4* v306;
                    v306 = reinterpret_cast<int4*>(v295 + v305);
                    int4* v307;
                    v307 = reinterpret_cast<int4*>(v293 + v303);
                    assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v306) % 16 == 0 && reinterpret_cast<unsigned long long>(v307) % 16 == 0);
                    *v307 = *v306;
                    v299 += 1 ;
                }
                v297 += 1 ;
            }
            // Poping the loop unrolling to: 0
            __syncthreads();
            v66 += 1 ;
        }
        v64 += 1 ;
    }
    return ;
}
__device__ void block_row_map_45(float * v0, int v1, float * v2){
    int v3;
    v3 = blockIdx.x;
    assert("Tensor range check" && 0 <= v3 && v3 < 24);
    int v4;
    v4 = 16384 * v3;
    int v5;
    v5 = blockIdx.x;
    assert("Tensor range check" && 0 <= v5 && v5 < 24);
    int v6;
    v6 = 16384 * v5;
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
    v12 = v8 % 16;
    int v13;
    v13 = v8 / 16;
    bool v14;
    v14 = v13 < 16;
    bool v15;
    v15 = v14 == false;
    if (v15){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v14);
    } else {
    }
    assert("Tensor range check" && 0 <= v13 && v13 < 16);
    assert("Tensor range check" && 0 <= v12 && v12 < 16);
    int v17;
    v17 = 4 * v12;
    int v18;
    v18 = v17 + v4;
    int v19;
    v19 = 64 * v13;
    int v20;
    v20 = v19 + v18;
    assert("Tensor range check" && 0 <= v13 && v13 < 16);
    assert("Tensor range check" && 0 <= v12 && v12 < 16);
    int v21;
    v21 = v17 + v7;
    int v22;
    v22 = v19 + v21;
    int v23;
    v23 = 0;
    while (while_method_13(v23)){
        assert("Tensor range check" && 0 <= v23 && v23 < 16);
        int v25;
        v25 = 1024 * v23;
        int v26;
        v26 = v25 + v20;
        float v27[4];
        int v28[4];
        int v29;
        v29 = 0;
        while (while_method_6(v29)){
            assert("Tensor range check" && 0 <= v29 && v29 < 1);
            int v31;
            v31 = 4 * v29;
            assert("Tensor range check" && 0 <= v29 && v29 < 1);
            int v32;
            v32 = 64 * v29;
            int v33;
            v33 = v32 + v26;
            int4* v34;
            v34 = reinterpret_cast<int4*>(v2 + v33);
            int4* v35;
            v35 = reinterpret_cast<int4*>(v27 + v31);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v34) % 16 == 0 && reinterpret_cast<unsigned long long>(v35) % 16 == 0);
            *v35 = *v34;
            v29 += 1 ;
        }
        int v36;
        v36 = 0;
        while (while_method_6(v36)){
            int v38;
            v38 = 0;
            while (while_method_3(v38)){
                bool v40;
                v40 = 0 <= v38;
                bool v42;
                if (v40){
                    bool v41;
                    v41 = v38 < 4;
                    v42 = v41;
                } else {
                    v42 = false;
                }
                bool v43;
                v43 = v42 == false;
                if (v43){
                    assert("The indices should be inside the range of the dimension." && v42);
                } else {
                }
                bool v45;
                v45 = 0 <= v12;
                bool v47;
                if (v45){
                    bool v46;
                    v46 = v12 < 16;
                    v47 = v46;
                } else {
                    v47 = false;
                }
                bool v48;
                v48 = v47 == false;
                if (v48){
                    assert("The indices should be inside the range of the dimension." && v47);
                } else {
                }
                int v50;
                v50 = v12 * 4;
                int v51;
                v51 = v38 + v50;
                bool v52;
                v52 = 0 <= v36;
                bool v54;
                if (v52){
                    bool v53;
                    v53 = v36 < 1;
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
                v57 = v36 * 64;
                int v58;
                v58 = v51 + v57;
                assert("Tensor range check" && 0 <= v36 && v36 < 1);
                assert("Tensor range check" && 0 <= v38 && v38 < 4);
                int v59;
                v59 = 4 * v36;
                int v60;
                v60 = v59 + v38;
                v28[v60] = v58;
                v38 += 1 ;
            }
            v36 += 1 ;
        }
        bool v61;
        v61 = 0 <= v13;
        bool v62;
        v62 = v61 && v14;
        bool v63;
        v63 = v62 == false;
        if (v63){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v62);
        } else {
        }
        bool v65;
        v65 = 0 <= v23;
        bool v67;
        if (v65){
            bool v66;
            v66 = v23 < 16;
            v67 = v66;
        } else {
            v67 = false;
        }
        bool v68;
        v68 = v67 == false;
        if (v68){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v67);
        } else {
        }
        int v70;
        v70 = v23 * 16;
        int v71;
        v71 = v70 + v13;
        bool v72[4];
        int v73;
        v73 = 0;
        while (while_method_6(v73)){
            int v75;
            v75 = 0;
            while (while_method_3(v75)){
                assert("Tensor range check" && 0 <= v73 && v73 < 1);
                assert("Tensor range check" && 0 <= v75 && v75 < 4);
                int v77;
                v77 = 4 * v73;
                int v78;
                v78 = v77 + v75;
                float v79;
                v79 = v27[v78];
                int v80;
                v80 = v28[v78];
                bool v81;
                v81 = v80 < 3;
                assert("Tensor range check" && 0 <= v73 && v73 < 1);
                assert("Tensor range check" && 0 <= v75 && v75 < 4);
                v72[v78] = v81;
                v75 += 1 ;
            }
            v73 += 1 ;
        }
        float v82[4];
        int v83;
        v83 = 0;
        while (while_method_6(v83)){
            int v85;
            v85 = 0;
            while (while_method_3(v85)){
                assert("Tensor range check" && 0 <= v83 && v83 < 1);
                assert("Tensor range check" && 0 <= v85 && v85 < 4);
                int v87;
                v87 = 4 * v83;
                int v88;
                v88 = v87 + v85;
                float v89;
                v89 = v27[v88];
                bool v90;
                v90 = v72[v88];
                float v91;
                if (v90){
                    v91 = v89;
                } else {
                    v91 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v83 && v83 < 1);
                assert("Tensor range check" && 0 <= v85 && v85 < 4);
                v82[v88] = v91;
                v85 += 1 ;
            }
            v83 += 1 ;
        }
        float v92;
        v92 = 0.0f;
        int v93;
        v93 = 0;
        while (while_method_6(v93)){
            int v95;
            v95 = 0;
            while (while_method_3(v95)){
                assert("Tensor range check" && 0 <= v93 && v93 < 1);
                assert("Tensor range check" && 0 <= v95 && v95 < 4);
                int v97;
                v97 = 4 * v93;
                int v98;
                v98 = v97 + v95;
                float v99;
                v99 = v82[v98];
                float v100;
                v100 = v92 + v99;
                v92 = v100;
                v95 += 1 ;
            }
            v93 += 1 ;
        }
        auto v101 = cooperative_groups::coalesced_threads();
        int v102;
        v102 = threadIdx.x;
        int v103;
        v103 = v102 / 16;
        auto v104 = cooperative_groups::labeled_partition(v101,v103);
        Closure0 v105{};
        float v106;
        v106 = cooperative_groups::reduce(v104, v92, v105);
        int v107[4];
        int v108;
        v108 = 0;
        while (while_method_6(v108)){
            int v110;
            v110 = 0;
            while (while_method_3(v110)){
                assert("Tensor range check" && 0 <= v108 && v108 < 1);
                assert("Tensor range check" && 0 <= v110 && v110 < 4);
                int v112;
                v112 = 4 * v108;
                int v113;
                v113 = v112 + v110;
                bool v114;
                v114 = v72[v113];
                int v115;
                if (v114){
                    v115 = 1;
                } else {
                    v115 = 0;
                }
                assert("Tensor range check" && 0 <= v108 && v108 < 1);
                assert("Tensor range check" && 0 <= v110 && v110 < 4);
                v107[v113] = v115;
                v110 += 1 ;
            }
            v108 += 1 ;
        }
        int v116;
        v116 = 0;
        int v117;
        v117 = 0;
        while (while_method_6(v117)){
            int v119;
            v119 = 0;
            while (while_method_3(v119)){
                assert("Tensor range check" && 0 <= v117 && v117 < 1);
                assert("Tensor range check" && 0 <= v119 && v119 < 4);
                int v121;
                v121 = 4 * v117;
                int v122;
                v122 = v121 + v119;
                int v123;
                v123 = v107[v122];
                int v124;
                v124 = v116 + v123;
                v116 = v124;
                v119 += 1 ;
            }
            v117 += 1 ;
        }
        auto v125 = cooperative_groups::coalesced_threads();
        int v126;
        v126 = threadIdx.x;
        int v127;
        v127 = v126 / 16;
        auto v128 = cooperative_groups::labeled_partition(v125,v127);
        Closure1 v129{};
        int v130;
        v130 = cooperative_groups::reduce(v128, v116, v129);
        float v131;
        v131 = (float)v130;
        float v132;
        v132 = v106 / v131;
        float v133[4];
        int v134;
        v134 = 0;
        while (while_method_6(v134)){
            int v136;
            v136 = 0;
            while (while_method_3(v136)){
                assert("Tensor range check" && 0 <= v134 && v134 < 1);
                assert("Tensor range check" && 0 <= v136 && v136 < 4);
                int v138;
                v138 = 4 * v134;
                int v139;
                v139 = v138 + v136;
                float v140;
                v140 = v27[v139];
                bool v141;
                v141 = v72[v139];
                float v142;
                if (v141){
                    v142 = v140;
                } else {
                    v142 = -1.0f / 0.0f;
                }
                float v143;
                v143 = v142 - v132;
                float v144;
                v144 = exp(v143);
                assert("Tensor range check" && 0 <= v134 && v134 < 1);
                assert("Tensor range check" && 0 <= v136 && v136 < 4);
                v133[v139] = v144;
                v136 += 1 ;
            }
            v134 += 1 ;
        }
        float v145;
        v145 = 0.0f;
        int v146;
        v146 = 0;
        while (while_method_6(v146)){
            int v148;
            v148 = 0;
            while (while_method_3(v148)){
                assert("Tensor range check" && 0 <= v146 && v146 < 1);
                assert("Tensor range check" && 0 <= v148 && v148 < 4);
                int v150;
                v150 = 4 * v146;
                int v151;
                v151 = v150 + v148;
                float v152;
                v152 = v133[v151];
                float v153;
                v153 = v145 + v152;
                v145 = v153;
                v148 += 1 ;
            }
            v146 += 1 ;
        }
        auto v154 = cooperative_groups::coalesced_threads();
        int v155;
        v155 = threadIdx.x;
        int v156;
        v156 = v155 / 16;
        auto v157 = cooperative_groups::labeled_partition(v154,v156);
        float v158;
        v158 = cooperative_groups::reduce(v157, v145, v105);
        float v159[4];
        int v160;
        v160 = 0;
        while (while_method_6(v160)){
            int v162;
            v162 = 0;
            while (while_method_3(v162)){
                assert("Tensor range check" && 0 <= v160 && v160 < 1);
                assert("Tensor range check" && 0 <= v162 && v162 < 4);
                int v164;
                v164 = 4 * v160;
                int v165;
                v165 = v164 + v162;
                float v166;
                v166 = v133[v165];
                float v167;
                v167 = v166 / v158;
                assert("Tensor range check" && 0 <= v160 && v160 < 1);
                assert("Tensor range check" && 0 <= v162 && v162 < 4);
                v159[v165] = v167;
                v162 += 1 ;
            }
            v160 += 1 ;
        }
        assert("Tensor range check" && 0 <= v23 && v23 < 16);
        int v168;
        v168 = v25 + v22;
        int v169;
        v169 = 0;
        while (while_method_6(v169)){
            assert("Tensor range check" && 0 <= v169 && v169 < 1);
            int v171;
            v171 = 64 * v169;
            int v172;
            v172 = v171 + v168;
            assert("Tensor range check" && 0 <= v169 && v169 < 1);
            int v173;
            v173 = 4 * v169;
            int4* v174;
            v174 = reinterpret_cast<int4*>(v159 + v173);
            int4* v175;
            v175 = reinterpret_cast<int4*>(v0 + v172);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v174) % 16 == 0 && reinterpret_cast<unsigned long long>(v175) % 16 == 0);
            *v175 = *v174;
            v169 += 1 ;
        }
        v23 += 1 ;
    }
    __syncthreads();
    return ;
}
__device__ int int_range_46(int v0, int v1, curandStatePhilox4_32_10_t & v2){
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
            return loop_50(v0, v1, v11);
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
    return loop_50(v2, v36, v37);
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
        Union1 v11; float v12;
        Tuple18 tmp47 = v0[v4];
        v11 = tmp47.v0; v12 = tmp47.v1;
        v2[v4] = v12;
        v4 += 1 ;
    }
    int v15;
    v15 = sample_discrete__48(v2, v1);
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
    Union1 v21; float v22;
    Tuple18 tmp48 = v0[v15];
    v21 = tmp48.v0; v22 = tmp48.v1;
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
__device__ Tuple0 score_51(static_array<unsigned char,7> v0){
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
    Tuple19 tmp55 = Tuple19{true, 1};
    v14 = tmp55.v0; v15 = tmp55.v1;
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
            Tuple20 tmp56 = Tuple20{v17, v21, v17};
            v26 = tmp56.v0; v27 = tmp56.v1; v28 = tmp56.v2;
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
                    Union11 v71;
                    if (v65){
                        v71 = Union11{Union11_2{}};
                    } else {
                        bool v67;
                        v67 = v63 > v64;
                        if (v67){
                            v71 = Union11{Union11_1{}};
                        } else {
                            v71 = Union11{Union11_0{}};
                        }
                    }
                    Union11 v81;
                    switch (v71.tag) {
                        case 0: { // Eq
                            unsigned char v72;
                            v72 = v47 % 4u;
                            unsigned char v73;
                            v73 = v62 % 4u;
                            bool v74;
                            v74 = v72 < v73;
                            if (v74){
                                v81 = Union11{Union11_2{}};
                            } else {
                                bool v76;
                                v76 = v72 > v73;
                                if (v76){
                                    v81 = Union11{Union11_1{}};
                                } else {
                                    v81 = Union11{Union11_0{}};
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
    Tuple21 tmp57 = Tuple21{0, 0, 12u};
    v132 = tmp57.v0; v133 = tmp57.v1; v134 = tmp57.v2;
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
    Union12 v202;
    if (v196){
        v202 = Union12{Union12_1{v130}};
    } else {
        bool v198;
        v198 = v133 == 5;
        if (v198){
            v202 = Union12{Union12_1{v130}};
        } else {
            v202 = Union12{Union12_0{}};
        }
    }
    static_array<unsigned char,5> v203;
    int v205; int v206; unsigned char v207;
    Tuple21 tmp58 = Tuple21{0, 0, 12u};
    v205 = tmp58.v0; v206 = tmp58.v1; v207 = tmp58.v2;
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
    Union12 v275;
    if (v269){
        v275 = Union12{Union12_1{v203}};
    } else {
        bool v271;
        v271 = v206 == 5;
        if (v271){
            v275 = Union12{Union12_1{v203}};
        } else {
            v275 = Union12{Union12_0{}};
        }
    }
    Union12 v312;
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
                    Union11 v278;
                    v278 = Union11{Union11_0{}};
                    int v279; Union11 v280;
                    Tuple22 tmp59 = Tuple22{0, v278};
                    v279 = tmp59.v0; v280 = tmp59.v1;
                    while (while_method_2(v279)){
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
                        Union11 v305;
                        switch (v280.tag) {
                            case 0: { // Eq
                                unsigned char v295;
                                v295 = v287 / 4u;
                                unsigned char v296;
                                v296 = v293 / 4u;
                                bool v297;
                                v297 = v295 < v296;
                                if (v297){
                                    v305 = Union11{Union11_2{}};
                                } else {
                                    bool v299;
                                    v299 = v295 > v296;
                                    if (v299){
                                        v305 = Union11{Union11_1{}};
                                    } else {
                                        v305 = Union11{Union11_0{}};
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
                    v312 = Union12{Union12_1{v307}};
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
    Tuple21 tmp60 = Tuple21{0, 0, 12u};
    v315 = tmp60.v0; v316 = tmp60.v1; v317 = tmp60.v2;
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
    Union12 v385;
    if (v379){
        v385 = Union12{Union12_1{v313}};
    } else {
        bool v381;
        v381 = v316 == 5;
        if (v381){
            v385 = Union12{Union12_1{v313}};
        } else {
            v385 = Union12{Union12_0{}};
        }
    }
    Union12 v422;
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
                    Union11 v388;
                    v388 = Union11{Union11_0{}};
                    int v389; Union11 v390;
                    Tuple22 tmp61 = Tuple22{0, v388};
                    v389 = tmp61.v0; v390 = tmp61.v1;
                    while (while_method_2(v389)){
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
                        Union11 v415;
                        switch (v390.tag) {
                            case 0: { // Eq
                                unsigned char v405;
                                v405 = v397 / 4u;
                                unsigned char v406;
                                v406 = v403 / 4u;
                                bool v407;
                                v407 = v405 < v406;
                                if (v407){
                                    v415 = Union11{Union11_2{}};
                                } else {
                                    bool v409;
                                    v409 = v405 > v406;
                                    if (v409){
                                        v415 = Union11{Union11_1{}};
                                    } else {
                                        v415 = Union11{Union11_0{}};
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
                    v422 = Union12{Union12_1{v417}};
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
    Tuple21 tmp62 = Tuple21{0, 0, 12u};
    v425 = tmp62.v0; v426 = tmp62.v1; v427 = tmp62.v2;
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
    Union12 v495;
    if (v489){
        v495 = Union12{Union12_1{v423}};
    } else {
        bool v491;
        v491 = v426 == 5;
        if (v491){
            v495 = Union12{Union12_1{v423}};
        } else {
            v495 = Union12{Union12_0{}};
        }
    }
    Union12 v532;
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
                    Union11 v498;
                    v498 = Union11{Union11_0{}};
                    int v499; Union11 v500;
                    Tuple22 tmp63 = Tuple22{0, v498};
                    v499 = tmp63.v0; v500 = tmp63.v1;
                    while (while_method_2(v499)){
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
                        Union11 v525;
                        switch (v500.tag) {
                            case 0: { // Eq
                                unsigned char v515;
                                v515 = v507 / 4u;
                                unsigned char v516;
                                v516 = v513 / 4u;
                                bool v517;
                                v517 = v515 < v516;
                                if (v517){
                                    v525 = Union11{Union11_2{}};
                                } else {
                                    bool v519;
                                    v519 = v515 > v516;
                                    if (v519){
                                        v525 = Union11{Union11_1{}};
                                    } else {
                                        v525 = Union11{Union11_0{}};
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
                    v532 = Union12{Union12_1{v527}};
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
            Tuple23 tmp64 = Tuple23{0, 0, 0, 12u};
            v538 = tmp64.v0; v539 = tmp64.v1; v540 = tmp64.v2; v541 = tmp64.v3;
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
            Union13 v577;
            if (v561){
                int v562;
                v562 = 0;
                while (while_method_1(v562)){
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
                v577 = Union13{Union13_1{v534, v536}};
            } else {
                v577 = Union13{Union13_0{}};
            }
            Union12 v615;
            switch (v577.tag) {
                case 0: { // None
                    v615 = Union12{Union12_0{}};
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
                    while (while_method_3(v593)){
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
                    v615 = Union12{Union12_1{v591}};
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
                    Tuple23 tmp65 = Tuple23{0, 0, 0, 12u};
                    v621 = tmp65.v0; v622 = tmp65.v1; v623 = tmp65.v2; v624 = tmp65.v3;
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
                    Union14 v660;
                    if (v644){
                        int v645;
                        v645 = 0;
                        while (while_method_3(v645)){
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
                        v660 = Union14{Union14_1{v617, v619}};
                    } else {
                        v660 = Union14{Union14_0{}};
                    }
                    Union12 v736;
                    switch (v660.tag) {
                        case 0: { // None
                            v736 = Union12{Union12_0{}};
                            break;
                        }
                        case 1: { // Some
                            static_array<unsigned char,3> v661 = v660.case1.v0; static_array<unsigned char,4> v662 = v660.case1.v1;
                            static_array<unsigned char,2> v663;
                            static_array<unsigned char,2> v665;
                            int v667; int v668; int v669; unsigned char v670;
                            Tuple23 tmp66 = Tuple23{0, 0, 0, 12u};
                            v667 = tmp66.v0; v668 = tmp66.v1; v669 = tmp66.v2; v670 = tmp66.v3;
                            while (while_method_3(v667)){
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
                            Union15 v706;
                            if (v690){
                                int v691;
                                v691 = 0;
                                while (while_method_0(v691)){
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
                                v706 = Union15{Union15_1{v663, v665}};
                            } else {
                                v706 = Union15{Union15_0{}};
                            }
                            switch (v706.tag) {
                                case 0: { // None
                                    v736 = Union12{Union12_0{}};
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,2> v707 = v706.case1.v0; static_array<unsigned char,2> v708 = v706.case1.v1;
                                    static_array<unsigned char,5> v709;
                                    int v711;
                                    v711 = 0;
                                    while (while_method_1(v711)){
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
                                    while (while_method_0(v720)){
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
                                    v736 = Union12{Union12_1{v709}};
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
                            Tuple4 tmp67 = Tuple4{0, 0};
                            v740 = tmp67.v0; v741 = tmp67.v1;
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
                            Union12 v759;
                            if (v756){
                                v759 = Union12{Union12_1{v738}};
                            } else {
                                v759 = Union12{Union12_0{}};
                            }
                            static_array<unsigned char,5> v760;
                            int v762; int v763;
                            Tuple4 tmp68 = Tuple4{0, 0};
                            v762 = tmp68.v0; v763 = tmp68.v1;
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
                            Union12 v781;
                            if (v778){
                                v781 = Union12{Union12_1{v760}};
                            } else {
                                v781 = Union12{Union12_0{}};
                            }
                            Union12 v818;
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
                                            Union11 v784;
                                            v784 = Union11{Union11_0{}};
                                            int v785; Union11 v786;
                                            Tuple22 tmp69 = Tuple22{0, v784};
                                            v785 = tmp69.v0; v786 = tmp69.v1;
                                            while (while_method_2(v785)){
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
                                                Union11 v811;
                                                switch (v786.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v801;
                                                        v801 = v793 / 4u;
                                                        unsigned char v802;
                                                        v802 = v799 / 4u;
                                                        bool v803;
                                                        v803 = v801 < v802;
                                                        if (v803){
                                                            v811 = Union11{Union11_2{}};
                                                        } else {
                                                            bool v805;
                                                            v805 = v801 > v802;
                                                            if (v805){
                                                                v811 = Union11{Union11_1{}};
                                                            } else {
                                                                v811 = Union11{Union11_0{}};
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
                                            v818 = Union12{Union12_1{v813}};
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
                            Tuple4 tmp70 = Tuple4{0, 0};
                            v821 = tmp70.v0; v822 = tmp70.v1;
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
                            Union12 v840;
                            if (v837){
                                v840 = Union12{Union12_1{v819}};
                            } else {
                                v840 = Union12{Union12_0{}};
                            }
                            Union12 v877;
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
                                            Union11 v843;
                                            v843 = Union11{Union11_0{}};
                                            int v844; Union11 v845;
                                            Tuple22 tmp71 = Tuple22{0, v843};
                                            v844 = tmp71.v0; v845 = tmp71.v1;
                                            while (while_method_2(v844)){
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
                                                Union11 v870;
                                                switch (v845.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v860;
                                                        v860 = v852 / 4u;
                                                        unsigned char v861;
                                                        v861 = v858 / 4u;
                                                        bool v862;
                                                        v862 = v860 < v861;
                                                        if (v862){
                                                            v870 = Union11{Union11_2{}};
                                                        } else {
                                                            bool v864;
                                                            v864 = v860 > v861;
                                                            if (v864){
                                                                v870 = Union11{Union11_1{}};
                                                            } else {
                                                                v870 = Union11{Union11_0{}};
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
                                            v877 = Union12{Union12_1{v872}};
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
                            Tuple4 tmp72 = Tuple4{0, 0};
                            v880 = tmp72.v0; v881 = tmp72.v1;
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
                            Union12 v899;
                            if (v896){
                                v899 = Union12{Union12_1{v878}};
                            } else {
                                v899 = Union12{Union12_0{}};
                            }
                            Union12 v936;
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
                                            Union11 v902;
                                            v902 = Union11{Union11_0{}};
                                            int v903; Union11 v904;
                                            Tuple22 tmp73 = Tuple22{0, v902};
                                            v903 = tmp73.v0; v904 = tmp73.v1;
                                            while (while_method_2(v903)){
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
                                                Union11 v929;
                                                switch (v904.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v919;
                                                        v919 = v911 / 4u;
                                                        unsigned char v920;
                                                        v920 = v917 / 4u;
                                                        bool v921;
                                                        v921 = v919 < v920;
                                                        if (v921){
                                                            v929 = Union11{Union11_2{}};
                                                        } else {
                                                            bool v923;
                                                            v923 = v919 > v920;
                                                            if (v923){
                                                                v929 = Union11{Union11_1{}};
                                                            } else {
                                                                v929 = Union11{Union11_0{}};
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
                                            v936 = Union12{Union12_1{v931}};
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
                                    Tuple21 tmp74 = Tuple21{0, 0, 12u};
                                    v940 = tmp74.v0; v941 = tmp74.v1; v942 = tmp74.v2;
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
                                    Union12 v980;
                                    if (v974){
                                        v980 = Union12{Union12_1{v938}};
                                    } else {
                                        bool v976;
                                        v976 = v941 == 5;
                                        if (v976){
                                            v980 = Union12{Union12_1{v938}};
                                        } else {
                                            v980 = Union12{Union12_0{}};
                                        }
                                    }
                                    switch (v980.tag) {
                                        case 0: { // None
                                            static_array<unsigned char,3> v982;
                                            static_array<unsigned char,4> v984;
                                            int v986; int v987; int v988; unsigned char v989;
                                            Tuple23 tmp75 = Tuple23{0, 0, 0, 12u};
                                            v986 = tmp75.v0; v987 = tmp75.v1; v988 = tmp75.v2; v989 = tmp75.v3;
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
                                            Union14 v1025;
                                            if (v1009){
                                                int v1010;
                                                v1010 = 0;
                                                while (while_method_3(v1010)){
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
                                                v1025 = Union14{Union14_1{v982, v984}};
                                            } else {
                                                v1025 = Union14{Union14_0{}};
                                            }
                                            Union12 v1063;
                                            switch (v1025.tag) {
                                                case 0: { // None
                                                    v1063 = Union12{Union12_0{}};
                                                    break;
                                                }
                                                case 1: { // Some
                                                    static_array<unsigned char,3> v1026 = v1025.case1.v0; static_array<unsigned char,4> v1027 = v1025.case1.v1;
                                                    static_array<unsigned char,2> v1028;
                                                    int v1030;
                                                    v1030 = 0;
                                                    while (while_method_0(v1030)){
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
                                                    while (while_method_1(v1041)){
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
                                                    while (while_method_0(v1050)){
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
                                                    v1063 = Union12{Union12_1{v1039}};
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
                                                    Tuple23 tmp76 = Tuple23{0, 0, 0, 12u};
                                                    v1069 = tmp76.v0; v1070 = tmp76.v1; v1071 = tmp76.v2; v1072 = tmp76.v3;
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
                                                    Union16 v1108;
                                                    if (v1092){
                                                        int v1093;
                                                        v1093 = 0;
                                                        while (while_method_2(v1093)){
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
                                                        v1108 = Union16{Union16_1{v1065, v1067}};
                                                    } else {
                                                        v1108 = Union16{Union16_0{}};
                                                    }
                                                    Union12 v1205;
                                                    switch (v1108.tag) {
                                                        case 0: { // None
                                                            v1205 = Union12{Union12_0{}};
                                                            break;
                                                        }
                                                        case 1: { // Some
                                                            static_array<unsigned char,2> v1109 = v1108.case1.v0; static_array<unsigned char,5> v1110 = v1108.case1.v1;
                                                            static_array<unsigned char,2> v1111;
                                                            static_array<unsigned char,3> v1113;
                                                            int v1115; int v1116; int v1117; unsigned char v1118;
                                                            Tuple23 tmp77 = Tuple23{0, 0, 0, 12u};
                                                            v1115 = tmp77.v0; v1116 = tmp77.v1; v1117 = tmp77.v2; v1118 = tmp77.v3;
                                                            while (while_method_2(v1115)){
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
                                                            Union17 v1154;
                                                            if (v1138){
                                                                int v1139;
                                                                v1139 = 0;
                                                                while (while_method_1(v1139)){
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
                                                                v1154 = Union17{Union17_1{v1111, v1113}};
                                                            } else {
                                                                v1154 = Union17{Union17_0{}};
                                                            }
                                                            switch (v1154.tag) {
                                                                case 0: { // None
                                                                    v1205 = Union12{Union12_0{}};
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
                                                                    while (while_method_0(v1170)){
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
                                                                    while (while_method_0(v1179)){
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
                                                                    v1205 = Union12{Union12_1{v1168}};
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
                                                            Tuple23 tmp78 = Tuple23{0, 0, 0, 12u};
                                                            v1211 = tmp78.v0; v1212 = tmp78.v1; v1213 = tmp78.v2; v1214 = tmp78.v3;
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
                                                            Union16 v1250;
                                                            if (v1234){
                                                                int v1235;
                                                                v1235 = 0;
                                                                while (while_method_2(v1235)){
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
                                                                v1250 = Union16{Union16_1{v1207, v1209}};
                                                            } else {
                                                                v1250 = Union16{Union16_0{}};
                                                            }
                                                            Union12 v1288;
                                                            switch (v1250.tag) {
                                                                case 0: { // None
                                                                    v1288 = Union12{Union12_0{}};
                                                                    break;
                                                                }
                                                                case 1: { // Some
                                                                    static_array<unsigned char,2> v1251 = v1250.case1.v0; static_array<unsigned char,5> v1252 = v1250.case1.v1;
                                                                    static_array<unsigned char,3> v1253;
                                                                    int v1255;
                                                                    v1255 = 0;
                                                                    while (while_method_1(v1255)){
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
                                                                    while (while_method_0(v1266)){
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
                                                                    while (while_method_1(v1275)){
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
                                                                    v1288 = Union12{Union12_1{v1264}};
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
                                                                    while (while_method_2(v1292)){
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
__device__ void play_loop_31(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut0 & v3, Union4 v4){
    static_array_list<Union6,128> & v5 = v3.v2;
    unsigned long long & v6 = v3.v0;
    Union3 v7;
    v7 = Union3{Union3_1{v4}};
    Union3 v8;
    v8 = v7;
    while (while_method_5(v8)){
        Union3 v2166;
        switch (v8.tag) {
            case 0: { // None
                v2166 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v10 = v8.case1.v0;
                switch (v10.tag) {
                    case 0: { // G_Flop
                        int v2024 = v10.case0.v0; static_array<static_array<unsigned char,2>,2> v2025 = v10.case0.v1; static_array<int,2> v2026 = v10.case0.v2; int v2027 = v10.case0.v3; static_array<int,2> v2028 = v10.case0.v4; Union5 v2029 = v10.case0.v5;
                        curandStatePhilox4_32_10_t & v2030 = v3.v4;
                        curandStatePhilox4_32_10_t & v2031 = v2030;
                        static_array<unsigned char,3> v2032; unsigned long long v2033;
                        Tuple8 tmp18 = draw_cards_32(v2031, v6);
                        v2032 = tmp18.v0; v2033 = tmp18.v1;
                        v3.v0 = v2033;
                        static_array_list<unsigned char,5> v2034;
                        v2034 = get_community_cards_35(v2029, v2032);
                        Union6 v2035;
                        v2035 = Union6{Union6_0{v2034}};
                        v5.push(v2035);
                        Union5 v2038;
                        switch (v2029.tag) {
                            case 1: { // Preflop
                                v2038 = Union5{Union5_0{v2032}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in flop.");
                                __trap();
                            }
                        }
                        int v2039;
                        v2039 = 2;
                        int v2040;
                        v2040 = 0;
                        Union4 v2041;
                        v2041 = try_round_36(v2039, v2025, v2026, v2040, v2028, v2038);
                        v2166 = Union3{Union3_1{v2041}};
                        break;
                    }
                    case 1: { // G_Fold
                        int v11 = v10.case1.v0; static_array<static_array<unsigned char,2>,2> v12 = v10.case1.v1; static_array<int,2> v13 = v10.case1.v2; int v14 = v10.case1.v3; static_array<int,2> v15 = v10.case1.v4; Union5 v16 = v10.case1.v5;
                        int v17;
                        v17 = v14 % 2;
                        bool v18;
                        v18 = 0 <= v17;
                        bool v20;
                        if (v18){
                            bool v19;
                            v19 = v17 < 2;
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
                        int v23;
                        v23 = v13[v17];
                        int v25;
                        v25 = v14 + 1;
                        int v26;
                        v26 = v25 % 2;
                        Union6 v27;
                        v27 = Union6{Union6_1{v23, v26}};
                        v5.push(v27);
                        Union7 v28;
                        v28 = Union7{Union7_1{v11, v12, v13, v14, v15, v16}};
                        v3.v5 = v28;
                        Union3 v29;
                        v29 = Union3{Union3_0{}};
                        v3.v1 = v29;
                        v2166 = Union3{Union3_0{}};
                        break;
                    }
                    case 2: { // G_Preflop
                        curandStatePhilox4_32_10_t & v2125 = v3.v4;
                        curandStatePhilox4_32_10_t & v2126 = v2125;
                        static_array<unsigned char,2> v2127; unsigned long long v2128;
                        Tuple11 tmp23 = draw_cards_39(v2126, v6);
                        v2127 = tmp23.v0; v2128 = tmp23.v1;
                        v3.v0 = v2128;
                        curandStatePhilox4_32_10_t & v2129 = v3.v4;
                        curandStatePhilox4_32_10_t & v2130 = v2129;
                        static_array<unsigned char,2> v2131; unsigned long long v2132;
                        Tuple11 tmp24 = draw_cards_39(v2130, v6);
                        v2131 = tmp24.v0; v2132 = tmp24.v1;
                        v3.v0 = v2132;
                        Union6 v2133;
                        v2133 = Union6{Union6_3{0, v2127}};
                        v5.push(v2133);
                        Union6 v2134;
                        v2134 = Union6{Union6_3{1, v2131}};
                        v5.push(v2134);
                        static_array<static_array<unsigned char,2>,2> v2135;
                        v2135[0] = v2127;
                        v2135[1] = v2131;
                        static_array<int,2> v2137;
                        v2137[0] = 2;
                        v2137[1] = 1;
                        static_array<int,2> v2139;
                        int v2141;
                        v2141 = 0;
                        while (while_method_0(v2141)){
                            bool v2143;
                            v2143 = 0 <= v2141;
                            bool v2145;
                            if (v2143){
                                bool v2144;
                                v2144 = v2141 < 2;
                                v2145 = v2144;
                            } else {
                                v2145 = false;
                            }
                            bool v2146;
                            v2146 = v2145 == false;
                            if (v2146){
                                assert("Index must be in range." && v2145);
                            } else {
                            }
                            int v2148;
                            v2148 = v2137[v2141];
                            int v2150;
                            v2150 = 100 - v2148;
                            v2139[v2141] = v2150;
                            v2141 += 1 ;
                        }
                        int v2151;
                        v2151 = 2;
                        int v2152;
                        v2152 = 0;
                        Union5 v2153;
                        v2153 = Union5{Union5_1{}};
                        Union4 v2154;
                        v2154 = try_round_36(v2151, v2135, v2137, v2152, v2139, v2153);
                        v2166 = Union3{Union3_1{v2154}};
                        break;
                    }
                    case 3: { // G_River
                        int v2084 = v10.case3.v0; static_array<static_array<unsigned char,2>,2> v2085 = v10.case3.v1; static_array<int,2> v2086 = v10.case3.v2; int v2087 = v10.case3.v3; static_array<int,2> v2088 = v10.case3.v4; Union5 v2089 = v10.case3.v5;
                        curandStatePhilox4_32_10_t & v2090 = v3.v4;
                        curandStatePhilox4_32_10_t & v2091 = v2090;
                        static_array<unsigned char,1> v2092; unsigned long long v2093;
                        Tuple12 tmp27 = draw_cards_40(v2091, v6);
                        v2092 = tmp27.v0; v2093 = tmp27.v1;
                        v3.v0 = v2093;
                        static_array_list<unsigned char,5> v2094;
                        v2094 = get_community_cards_41(v2089, v2092);
                        Union6 v2095;
                        v2095 = Union6{Union6_0{v2094}};
                        v5.push(v2095);
                        Union5 v2120;
                        switch (v2089.tag) {
                            case 3: { // Turn
                                static_array<unsigned char,4> v2096 = v2089.case3.v0;
                                static_array<unsigned char,5> v2097;
                                int v2099;
                                v2099 = 0;
                                while (while_method_3(v2099)){
                                    bool v2101;
                                    v2101 = 0 <= v2099;
                                    bool v2103;
                                    if (v2101){
                                        bool v2102;
                                        v2102 = v2099 < 4;
                                        v2103 = v2102;
                                    } else {
                                        v2103 = false;
                                    }
                                    bool v2104;
                                    v2104 = v2103 == false;
                                    if (v2104){
                                        assert("Index must be in range." && v2103);
                                    } else {
                                    }
                                    unsigned char v2106;
                                    v2106 = v2096[v2099];
                                    v2097[v2099] = v2106;
                                    v2099 += 1 ;
                                }
                                int v2108;
                                v2108 = 0;
                                while (while_method_6(v2108)){
                                    bool v2110;
                                    v2110 = 0 <= v2108;
                                    bool v2112;
                                    if (v2110){
                                        bool v2111;
                                        v2111 = v2108 < 1;
                                        v2112 = v2111;
                                    } else {
                                        v2112 = false;
                                    }
                                    bool v2113;
                                    v2113 = v2112 == false;
                                    if (v2113){
                                        assert("Index must be in range." && v2112);
                                    } else {
                                    }
                                    unsigned char v2115;
                                    v2115 = v2092[v2108];
                                    int v2117;
                                    v2117 = 4 + v2108;
                                    v2097[v2117] = v2115;
                                    v2108 += 1 ;
                                }
                                v2120 = Union5{Union5_2{v2097}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in river.");
                                __trap();
                            }
                        }
                        int v2121;
                        v2121 = 2;
                        int v2122;
                        v2122 = 0;
                        Union4 v2123;
                        v2123 = try_round_36(v2121, v2085, v2086, v2122, v2088, v2120);
                        v2166 = Union3{Union3_1{v2123}};
                        break;
                    }
                    case 4: { // G_Round
                        int v145 = v10.case4.v0; static_array<static_array<unsigned char,2>,2> v146 = v10.case4.v1; static_array<int,2> v147 = v10.case4.v2; int v148 = v10.case4.v3; static_array<int,2> v149 = v10.case4.v4; Union5 v150 = v10.case4.v5;
                        int v151;
                        v151 = v148 % 2;
                        static_array<Union2,2> & v152 = v3.v3;
                        bool v153;
                        v153 = 0 <= v151;
                        bool v155;
                        if (v153){
                            bool v154;
                            v154 = v151 < 2;
                            v155 = v154;
                        } else {
                            v155 = false;
                        }
                        bool v156;
                        v156 = v155 == false;
                        if (v156){
                            assert("Index must be in range." && v155);
                        } else {
                        }
                        Union2 v158;
                        v158 = v152[v151];
                        switch (v158.tag) {
                            case 0: { // CallingMachine
                                Union1 v1600;
                                v1600 = Union1{Union1_1{}};
                                Union6 v1601;
                                v1601 = Union6{Union6_2{v151, v1600}};
                                v5.push(v1601);
                                static_array<int,2> v1602;
                                int v1604;
                                v1604 = 0;
                                while (while_method_0(v1604)){
                                    bool v1606;
                                    v1606 = 0 <= v1604;
                                    bool v1608;
                                    if (v1606){
                                        bool v1607;
                                        v1607 = v1604 < 2;
                                        v1608 = v1607;
                                    } else {
                                        v1608 = false;
                                    }
                                    bool v1609;
                                    v1609 = v1608 == false;
                                    if (v1609){
                                        assert("Index must be in range." && v1608);
                                    } else {
                                    }
                                    int v1611;
                                    v1611 = v149[v1604];
                                    bool v1614;
                                    if (v1606){
                                        bool v1613;
                                        v1613 = v1604 < 2;
                                        v1614 = v1613;
                                    } else {
                                        v1614 = false;
                                    }
                                    bool v1615;
                                    v1615 = v1614 == false;
                                    if (v1615){
                                        assert("Index must be in range." && v1614);
                                    } else {
                                    }
                                    int v1617;
                                    v1617 = v147[v1604];
                                    int v1619;
                                    v1619 = v1611 + v1617;
                                    v1602[v1604] = v1619;
                                    v1604 += 1 ;
                                }
                                int v1620;
                                v1620 = v147[0];
                                int v1622; int v1623;
                                Tuple4 tmp28 = Tuple4{1, v1620};
                                v1622 = tmp28.v0; v1623 = tmp28.v1;
                                while (while_method_0(v1622)){
                                    bool v1625;
                                    v1625 = 0 <= v1622;
                                    bool v1627;
                                    if (v1625){
                                        bool v1626;
                                        v1626 = v1622 < 2;
                                        v1627 = v1626;
                                    } else {
                                        v1627 = false;
                                    }
                                    bool v1628;
                                    v1628 = v1627 == false;
                                    if (v1628){
                                        assert("Index must be in range." && v1627);
                                    } else {
                                    }
                                    int v1630;
                                    v1630 = v147[v1622];
                                    bool v1632;
                                    v1632 = v1623 >= v1630;
                                    int v1633;
                                    if (v1632){
                                        v1633 = v1623;
                                    } else {
                                        v1633 = v1630;
                                    }
                                    v1623 = v1633;
                                    v1622 += 1 ;
                                }
                                bool v1635;
                                if (v153){
                                    bool v1634;
                                    v1634 = v151 < 2;
                                    v1635 = v1634;
                                } else {
                                    v1635 = false;
                                }
                                bool v1636;
                                v1636 = v1635 == false;
                                if (v1636){
                                    assert("Index must be in range." && v1635);
                                } else {
                                }
                                int v1638;
                                v1638 = v1602[v151];
                                bool v1640;
                                v1640 = v1623 < v1638;
                                int v1641;
                                if (v1640){
                                    v1641 = v1623;
                                } else {
                                    v1641 = v1638;
                                }
                                static_array<int,2> v1642;
                                int v1644;
                                v1644 = 0;
                                while (while_method_0(v1644)){
                                    bool v1646;
                                    v1646 = 0 <= v1644;
                                    bool v1648;
                                    if (v1646){
                                        bool v1647;
                                        v1647 = v1644 < 2;
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
                                    v1651 = v147[v1644];
                                    bool v1653;
                                    v1653 = v151 == v1644;
                                    int v1654;
                                    if (v1653){
                                        v1654 = v1641;
                                    } else {
                                        v1654 = v1651;
                                    }
                                    v1642[v1644] = v1654;
                                    v1644 += 1 ;
                                }
                                static_array<int,2> v1655;
                                int v1657;
                                v1657 = 0;
                                while (while_method_0(v1657)){
                                    bool v1659;
                                    v1659 = 0 <= v1657;
                                    bool v1661;
                                    if (v1659){
                                        bool v1660;
                                        v1660 = v1657 < 2;
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
                                    v1664 = v1602[v1657];
                                    bool v1667;
                                    if (v1659){
                                        bool v1666;
                                        v1666 = v1657 < 2;
                                        v1667 = v1666;
                                    } else {
                                        v1667 = false;
                                    }
                                    bool v1668;
                                    v1668 = v1667 == false;
                                    if (v1668){
                                        assert("Index must be in range." && v1667);
                                    } else {
                                    }
                                    int v1670;
                                    v1670 = v1642[v1657];
                                    int v1672;
                                    v1672 = v1664 - v1670;
                                    v1655[v1657] = v1672;
                                    v1657 += 1 ;
                                }
                                bool v1673;
                                v1673 = v151 < 2;
                                Union4 v1677;
                                if (v1673){
                                    int v1674;
                                    v1674 = v148 + 1;
                                    v1677 = try_round_36(v145, v146, v1642, v1674, v1655, v150);
                                } else {
                                    v1677 = go_next_street_38(v145, v146, v1642, v148, v1655, v150);
                                }
                                v2166 = Union3{Union3_1{v1677}};
                                break;
                            }
                            case 1: { // Computer
                                static_array_list<Union6,128> & v160 = v3.v2;
                                curandStatePhilox4_32_10_t & v161 = v3.v4;
                                curandStatePhilox4_32_10_t & v162 = v161;
                                float * v163;
                                v163 = reinterpret_cast<float *>(&v1[51904512ull]);
                                float * v165;
                                v165 = reinterpret_cast<float *>(&v1[0ull]);
                                float * v167;
                                v167 = reinterpret_cast<float *>(&v1[0ull]);
                                int v169;
                                v169 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v169 && v169 < 24);
                                int v170;
                                v170 = 524288 * v169;
                                int v171;
                                v171 = threadIdx.x;
                                int v172;
                                v172 = v171;
                                while (while_method_7(v172)){
                                    bool v174;
                                    v174 = 0 <= v172;
                                    bool v175;
                                    v175 = v174 == false;
                                    if (v175){
                                        assert("The index needs to be zero or positive." && v174);
                                    } else {
                                    }
                                    int v177;
                                    v177 = v172 % 2048;
                                    int v178;
                                    v178 = v172 / 2048;
                                    bool v179;
                                    v179 = v178 < 256;
                                    bool v180;
                                    v180 = v179 == false;
                                    if (v180){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v179);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v178 && v178 < 256);
                                    assert("Tensor range check" && 0 <= v177 && v177 < 2048);
                                    int v182;
                                    v182 = v177 + v170;
                                    int v183;
                                    v183 = 2048 * v178;
                                    int v184;
                                    v184 = v183 + v182;
                                    v167[v184] = 0.0f;
                                    v172 += 256 ;
                                }
                                __syncthreads();
                                int v185;
                                v185 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v185 && v185 < 256);
                                int v186;
                                v186 = 2048 * v185;
                                int v187;
                                v187 = v186 + v170;
                                int v188;
                                v188 = v160.length;
                                bool v189;
                                v189 = 128 >= v188;
                                bool v190;
                                v190 = v189 == false;
                                if (v190){
                                    assert("The type level dimension has to equal the value passed at runtime into create." && v189);
                                } else {
                                }
                                static_array_list<Union8,128> v192;
                                v192 = static_array_list<Union8,128>{};
                                v192.unsafe_set_length(v188);
                                int v194;
                                v194 = 0;
                                while (while_method_4(v188, v194)){
                                    Union6 v196;
                                    v196 = v160[v194];
                                    Union8 v202;
                                    switch (v196.tag) {
                                        case 2: { // PlayerAction
                                            int v198 = v196.case2.v0; Union1 v199 = v196.case2.v1;
                                            v202 = Union8{Union8_1{v199}};
                                            break;
                                        }
                                        default: {
                                            v202 = Union8{Union8_0{}};
                                        }
                                    }
                                    v192[v194] = v202;
                                    v194 += 1 ;
                                }
                                static_array<int,2> v203;
                                int v205;
                                v205 = 0;
                                while (while_method_0(v205)){
                                    int v207;
                                    v207 = v205 + v148;
                                    int v208;
                                    v208 = v207 % 2;
                                    bool v209;
                                    v209 = 0 <= v208;
                                    bool v211;
                                    if (v209){
                                        bool v210;
                                        v210 = v208 < 2;
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
                                    int v214;
                                    v214 = v147[v208];
                                    v203[v205] = v214;
                                    v205 += 1 ;
                                }
                                static_array<int,2> v216;
                                int v218;
                                v218 = 0;
                                while (while_method_0(v218)){
                                    int v220;
                                    v220 = v218 + v148;
                                    int v221;
                                    v221 = v220 % 2;
                                    bool v222;
                                    v222 = 0 <= v221;
                                    bool v224;
                                    if (v222){
                                        bool v223;
                                        v223 = v221 < 2;
                                        v224 = v223;
                                    } else {
                                        v224 = false;
                                    }
                                    bool v225;
                                    v225 = v224 == false;
                                    if (v225){
                                        assert("Index must be in range." && v224);
                                    } else {
                                    }
                                    int v227;
                                    v227 = v149[v221];
                                    v216[v218] = v227;
                                    v218 += 1 ;
                                }
                                bool v230;
                                if (v153){
                                    bool v229;
                                    v229 = v151 < 2;
                                    v230 = v229;
                                } else {
                                    v230 = false;
                                }
                                bool v231;
                                v231 = v230 == false;
                                if (v231){
                                    assert("Index must be in range." && v230);
                                } else {
                                }
                                static_array<unsigned char,2> v233;
                                v233 = v146[v151];
                                static_array_list<unsigned char,5> v235;
                                v235 = static_array_list<unsigned char,5>{};
                                switch (v150.tag) {
                                    case 0: { // Flop
                                        static_array<unsigned char,3> v237 = v150.case0.v0;
                                        int v238;
                                        v238 = 0;
                                        while (while_method_1(v238)){
                                            bool v240;
                                            v240 = 0 <= v238;
                                            bool v242;
                                            if (v240){
                                                bool v241;
                                                v241 = v238 < 3;
                                                v242 = v241;
                                            } else {
                                                v242 = false;
                                            }
                                            bool v243;
                                            v243 = v242 == false;
                                            if (v243){
                                                assert("Index must be in range." && v242);
                                            } else {
                                            }
                                            unsigned char v245;
                                            v245 = v237[v238];
                                            v235.push(v245);
                                            v238 += 1 ;
                                        }
                                        break;
                                    }
                                    case 1: { // Preflop
                                        break;
                                    }
                                    case 2: { // River
                                        static_array<unsigned char,5> v257 = v150.case2.v0;
                                        int v258;
                                        v258 = 0;
                                        while (while_method_2(v258)){
                                            bool v260;
                                            v260 = 0 <= v258;
                                            bool v262;
                                            if (v260){
                                                bool v261;
                                                v261 = v258 < 5;
                                                v262 = v261;
                                            } else {
                                                v262 = false;
                                            }
                                            bool v263;
                                            v263 = v262 == false;
                                            if (v263){
                                                assert("Index must be in range." && v262);
                                            } else {
                                            }
                                            unsigned char v265;
                                            v265 = v257[v258];
                                            v235.push(v265);
                                            v258 += 1 ;
                                        }
                                        break;
                                    }
                                    case 3: { // Turn
                                        static_array<unsigned char,4> v247 = v150.case3.v0;
                                        int v248;
                                        v248 = 0;
                                        while (while_method_3(v248)){
                                            bool v250;
                                            v250 = 0 <= v248;
                                            bool v252;
                                            if (v250){
                                                bool v251;
                                                v251 = v248 < 4;
                                                v252 = v251;
                                            } else {
                                                v252 = false;
                                            }
                                            bool v253;
                                            v253 = v252 == false;
                                            if (v253){
                                                assert("Index must be in range." && v252);
                                            } else {
                                            }
                                            unsigned char v255;
                                            v255 = v247[v248];
                                            v235.push(v255);
                                            v248 += 1 ;
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                float * v267;
                                v267 = v167+v187;
                                int v269;
                                v269 = v192.length;
                                bool v270;
                                v270 = v269 == 0;
                                if (v270){
                                    v267[0] = 1.0f;
                                } else {
                                }
                                int v271;
                                v271 = v192.length;
                                int v272;
                                v272 = 0;
                                while (while_method_4(v271, v272)){
                                    Union8 v274;
                                    v274 = v192[v272];
                                    int v276;
                                    v276 = v272 * 14;
                                    int v277;
                                    v277 = 1 + v276;
                                    switch (v274.tag) {
                                        case 0: { // None
                                            v267[v277] = 1.0f;
                                            break;
                                        }
                                        case 1: { // Some
                                            Union1 v278 = v274.case1.v0;
                                            int v279;
                                            v279 = v277 + 1;
                                            switch (v278.tag) {
                                                case 0: { // A_All_In
                                                    v267[v279] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // A_Call
                                                    int v280;
                                                    v280 = v279 + 1;
                                                    v267[v280] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // A_Fold
                                                    int v281;
                                                    v281 = v279 + 2;
                                                    v267[v281] = 1.0f;
                                                    break;
                                                }
                                                case 3: { // A_Raise
                                                    int v282 = v278.case3.v0;
                                                    int v283;
                                                    v283 = v279 + 3;
                                                    bool v284;
                                                    v284 = 1 <= v282;
                                                    bool v286;
                                                    if (v284){
                                                        bool v285;
                                                        v285 = v282 < 1023;
                                                        v286 = v285;
                                                    } else {
                                                        v286 = false;
                                                    }
                                                    bool v287;
                                                    v287 = v286 == false;
                                                    if (v287){
                                                        assert("Pickle failure. The input is out of the bounds of the given range." && v286);
                                                    } else {
                                                    }
                                                    int v289;
                                                    v289 = v282 - 1;
                                                    unsigned int v290;
                                                    v290 = (unsigned int)v289;
                                                    method_42(v290, v267, v283);
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
                                    v272 += 1 ;
                                }
                                int v291;
                                v291 = 0;
                                while (while_method_0(v291)){
                                    // 222;
                                    bool v293;
                                    v293 = 0 <= v291;
                                    bool v295;
                                    if (v293){
                                        bool v294;
                                        v294 = v291 < 2;
                                        v295 = v294;
                                    } else {
                                        v295 = false;
                                    }
                                    bool v296;
                                    v296 = v295 == false;
                                    if (v296){
                                        assert("Index must be in range." && v295);
                                    } else {
                                    }
                                    int v298;
                                    v298 = v203[v291];
                                    int v300;
                                    v300 = v291 * 11;
                                    int v301;
                                    v301 = 1794 + v300;
                                    bool v302;
                                    v302 = 0 <= v298;
                                    bool v304;
                                    if (v302){
                                        bool v303;
                                        v303 = v298 < 1023;
                                        v304 = v303;
                                    } else {
                                        v304 = false;
                                    }
                                    bool v305;
                                    v305 = v304 == false;
                                    if (v305){
                                        assert("Pickle failure. The input is out of the bounds of the given range." && v304);
                                    } else {
                                    }
                                    unsigned int v307;
                                    v307 = (unsigned int)v298;
                                    method_43(v307, v267, v301);
                                    v291 += 1 ;
                                }
                                int v308;
                                v308 = 0;
                                while (while_method_0(v308)){
                                    // 222;
                                    bool v310;
                                    v310 = 0 <= v308;
                                    bool v312;
                                    if (v310){
                                        bool v311;
                                        v311 = v308 < 2;
                                        v312 = v311;
                                    } else {
                                        v312 = false;
                                    }
                                    bool v313;
                                    v313 = v312 == false;
                                    if (v313){
                                        assert("Index must be in range." && v312);
                                    } else {
                                    }
                                    int v315;
                                    v315 = v216[v308];
                                    int v317;
                                    v317 = v308 * 11;
                                    int v318;
                                    v318 = 1817 + v317;
                                    bool v319;
                                    v319 = 0 <= v315;
                                    bool v321;
                                    if (v319){
                                        bool v320;
                                        v320 = v315 < 1023;
                                        v321 = v320;
                                    } else {
                                        v321 = false;
                                    }
                                    bool v322;
                                    v322 = v321 == false;
                                    if (v322){
                                        assert("Pickle failure. The input is out of the bounds of the given range." && v321);
                                    } else {
                                    }
                                    unsigned int v324;
                                    v324 = (unsigned int)v315;
                                    method_43(v324, v267, v318);
                                    v308 += 1 ;
                                }
                                int v325;
                                v325 = 0;
                                while (while_method_0(v325)){
                                    // 222;
                                    bool v327;
                                    v327 = 0 <= v325;
                                    bool v329;
                                    if (v327){
                                        bool v328;
                                        v328 = v325 < 2;
                                        v329 = v328;
                                    } else {
                                        v329 = false;
                                    }
                                    bool v330;
                                    v330 = v329 == false;
                                    if (v330){
                                        assert("Index must be in range." && v329);
                                    } else {
                                    }
                                    unsigned char v332;
                                    v332 = v233[v325];
                                    int v334;
                                    v334 = v325 * 17;
                                    int v335;
                                    v335 = 1840 + v334;
                                    unsigned char v336;
                                    v336 = v332 % 4u;
                                    int v337;
                                    v337 = (int)v336;
                                    unsigned char v338;
                                    v338 = v332 / 4u;
                                    int v339;
                                    v339 = (int)v338;
                                    unsigned int v340;
                                    v340 = (unsigned int)v337;
                                    int v341;
                                    v341 = (int)v340;
                                    bool v342;
                                    v342 = v341 < 4;
                                    bool v343;
                                    v343 = v342 == false;
                                    if (v343){
                                        assert("Pickle failure. Int value out of bounds." && v342);
                                    } else {
                                    }
                                    int v345;
                                    v345 = v335 + v341;
                                    v267[v345] = 1.0f;
                                    int v346;
                                    v346 = v335 + 4;
                                    unsigned int v347;
                                    v347 = (unsigned int)v339;
                                    int v348;
                                    v348 = (int)v347;
                                    bool v349;
                                    v349 = v348 < 13;
                                    bool v350;
                                    v350 = v349 == false;
                                    if (v350){
                                        assert("Pickle failure. Int value out of bounds." && v349);
                                    } else {
                                    }
                                    int v352;
                                    v352 = v346 + v348;
                                    v267[v352] = 1.0f;
                                    v325 += 1 ;
                                }
                                int v353;
                                v353 = v235.length;
                                bool v354;
                                v354 = v353 == 0;
                                if (v354){
                                    v267[1874] = 1.0f;
                                } else {
                                }
                                int v355;
                                v355 = v235.length;
                                int v356;
                                v356 = 0;
                                while (while_method_4(v355, v356)){
                                    unsigned char v358;
                                    v358 = v235[v356];
                                    int v360;
                                    v360 = v356 * 17;
                                    int v361;
                                    v361 = 1875 + v360;
                                    unsigned char v362;
                                    v362 = v358 % 4u;
                                    int v363;
                                    v363 = (int)v362;
                                    unsigned char v364;
                                    v364 = v358 / 4u;
                                    int v365;
                                    v365 = (int)v364;
                                    unsigned int v366;
                                    v366 = (unsigned int)v363;
                                    int v367;
                                    v367 = (int)v366;
                                    bool v368;
                                    v368 = v367 < 4;
                                    bool v369;
                                    v369 = v368 == false;
                                    if (v369){
                                        assert("Pickle failure. Int value out of bounds." && v368);
                                    } else {
                                    }
                                    int v371;
                                    v371 = v361 + v367;
                                    v267[v371] = 1.0f;
                                    int v372;
                                    v372 = v361 + 4;
                                    unsigned int v373;
                                    v373 = (unsigned int)v365;
                                    int v374;
                                    v374 = (int)v373;
                                    bool v375;
                                    v375 = v374 < 13;
                                    bool v376;
                                    v376 = v375 == false;
                                    if (v376){
                                        assert("Pickle failure. Int value out of bounds." && v375);
                                    } else {
                                    }
                                    int v378;
                                    v378 = v372 + v374;
                                    v267[v378] = 1.0f;
                                    v356 += 1 ;
                                }
                                __syncthreads();
                                int v379;
                                v379 = 0;
                                while (while_method_10(v379)){
                                    float * v381;
                                    v381 = reinterpret_cast<float *>(&v1[51904512ull]);
                                    assert("Tensor range check" && 0 <= v379 && v379 < 32);
                                    int v383;
                                    v383 = 393216 * v379;
                                    float * v384;
                                    v384 = reinterpret_cast<float *>(&v1[0ull]);
                                    float * v386;
                                    v386 = reinterpret_cast<float *>(&v0[0ull]);
                                    assert("Tensor range check" && 0 <= v379 && v379 < 32);
                                    int v388;
                                    v388 = 131072 * v379;
                                    float * v389;
                                    v389 = reinterpret_cast<float *>(&v1[50331648ull]);
                                    block_matmul_44(v389, v386, v388, v384);
                                    block_row_map_45(v381, v383, v389);
                                    v379 += 1 ;
                                }
                                __syncthreads();
                                int v391;
                                v391 = 0;
                                int v392;
                                v392 = 32;
                                int v393;
                                v393 = int_range_46(v392, v391, v162);
                                float * v394;
                                v394 = reinterpret_cast<float *>(&v1[51904512ull]);
                                assert("Tensor range check" && 0 <= v393 && v393 < 32);
                                int v396;
                                v396 = 393216 * v393;
                                int v397;
                                v397 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v397 && v397 < 24);
                                int v398;
                                v398 = 16384 * v397;
                                int v399;
                                v399 = v398 + v396;
                                int v400;
                                v400 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v400 && v400 < 256);
                                int v401;
                                v401 = 64 * v400;
                                int v402;
                                v402 = v401 + v399;
                                float * v403;
                                v403 = v394+v402;
                                int v405;
                                v405 = sizeof(float *);
                                unsigned long long v406;
                                v406 = (unsigned long long)v405;
                                unsigned long long v407;
                                v407 = 256ull * v406;
                                unsigned long long v408;
                                v408 = v407 + 16ull;
                                unsigned long long v409;
                                v409 = v408 - 1ull;
                                unsigned long long v410;
                                v410 = v409 % 16ull;
                                unsigned long long v411;
                                v411 = v409 - v410;
                                unsigned long long v412;
                                v412 = v411 + 1024ull;
                                bool v413;
                                v413 = v412 <= 98304ull;
                                bool v414;
                                v414 = v413 == false;
                                if (v414){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v413);
                                } else {
                                }
                                extern __shared__ unsigned char v416[];
                                bool v417;
                                v417 = v412 <= v412;
                                bool v418;
                                v418 = v417 == false;
                                if (v418){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v417);
                                } else {
                                }
                                float * * v420;
                                v420 = reinterpret_cast<float * *>(&v416[0ull]);
                                int * v422;
                                v422 = reinterpret_cast<int *>(&v416[v411]);
                                int v424;
                                v424 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v424 && v424 < 256);
                                v420[v424] = v403;
                                __syncthreads();
                                bool v425;
                                v425 = 0 <= v424;
                                bool v426;
                                v426 = v425 == false;
                                if (v426){
                                    assert("The index needs to be zero or positive." && v425);
                                } else {
                                }
                                int v428;
                                v428 = v424 % 16;
                                int v429;
                                v429 = v424 / 16;
                                bool v430;
                                v430 = v429 < 16;
                                bool v431;
                                v431 = v430 == false;
                                if (v431){
                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v430);
                                } else {
                                }
                                assert("Tensor range check" && 0 <= v429 && v429 < 16);
                                int v433;
                                v433 = 0;
                                while (while_method_13(v433)){
                                    bool v435;
                                    v435 = 0 <= v429;
                                    bool v436;
                                    v436 = v435 && v430;
                                    bool v437;
                                    v437 = v436 == false;
                                    if (v437){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v436);
                                    } else {
                                    }
                                    bool v439;
                                    v439 = 0 <= v433;
                                    bool v441;
                                    if (v439){
                                        bool v440;
                                        v440 = v433 < 16;
                                        v441 = v440;
                                    } else {
                                        v441 = false;
                                    }
                                    bool v442;
                                    v442 = v441 == false;
                                    if (v442){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v441);
                                    } else {
                                    }
                                    int v444;
                                    v444 = v433 * 16;
                                    int v445;
                                    v445 = v444 + v429;
                                    assert("Tensor range check" && 0 <= v433 && v433 < 16);
                                    int v446;
                                    v446 = 16 * v433;
                                    int v447;
                                    v447 = v446 + v429;
                                    float * v448;
                                    v448 = v420[v447];
                                    int v449;
                                    v449 = blockIdx.x;
                                    int v450;
                                    v450 = v449 * 256;
                                    int v451;
                                    v451 = v450 + v445;
                                    assert("Tensor range check" && 0 <= v428 && v428 < 16);
                                    int v452;
                                    v452 = 4 * v428;
                                    float v453[4];
                                    int v454[4];
                                    int v455;
                                    v455 = 0;
                                    while (while_method_6(v455)){
                                        assert("Tensor range check" && 0 <= v455 && v455 < 1);
                                        int v457;
                                        v457 = 4 * v455;
                                        assert("Tensor range check" && 0 <= v455 && v455 < 1);
                                        int v458;
                                        v458 = 64 * v455;
                                        int v459;
                                        v459 = v458 + v452;
                                        int4* v460;
                                        v460 = reinterpret_cast<int4*>(v448 + v459);
                                        int4* v461;
                                        v461 = reinterpret_cast<int4*>(v453 + v457);
                                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v460) % 16 == 0 && reinterpret_cast<unsigned long long>(v461) % 16 == 0);
                                        *v461 = *v460;
                                        v455 += 1 ;
                                    }
                                    int v462;
                                    v462 = 0;
                                    while (while_method_6(v462)){
                                        int v464;
                                        v464 = 0;
                                        while (while_method_3(v464)){
                                            bool v466;
                                            v466 = 0 <= v464;
                                            bool v468;
                                            if (v466){
                                                bool v467;
                                                v467 = v464 < 4;
                                                v468 = v467;
                                            } else {
                                                v468 = false;
                                            }
                                            bool v469;
                                            v469 = v468 == false;
                                            if (v469){
                                                assert("The indices should be inside the range of the dimension." && v468);
                                            } else {
                                            }
                                            bool v471;
                                            v471 = 0 <= v428;
                                            bool v473;
                                            if (v471){
                                                bool v472;
                                                v472 = v428 < 16;
                                                v473 = v472;
                                            } else {
                                                v473 = false;
                                            }
                                            bool v474;
                                            v474 = v473 == false;
                                            if (v474){
                                                assert("The indices should be inside the range of the dimension." && v473);
                                            } else {
                                            }
                                            int v476;
                                            v476 = v428 * 4;
                                            int v477;
                                            v477 = v464 + v476;
                                            bool v478;
                                            v478 = 0 <= v462;
                                            bool v480;
                                            if (v478){
                                                bool v479;
                                                v479 = v462 < 1;
                                                v480 = v479;
                                            } else {
                                                v480 = false;
                                            }
                                            bool v481;
                                            v481 = v480 == false;
                                            if (v481){
                                                assert("The indices should be inside the range of the dimension." && v480);
                                            } else {
                                            }
                                            int v483;
                                            v483 = v462 * 64;
                                            int v484;
                                            v484 = v477 + v483;
                                            assert("Tensor range check" && 0 <= v462 && v462 < 1);
                                            assert("Tensor range check" && 0 <= v464 && v464 < 4);
                                            int v485;
                                            v485 = 4 * v462;
                                            int v486;
                                            v486 = v485 + v464;
                                            v454[v486] = v484;
                                            v464 += 1 ;
                                        }
                                        v462 += 1 ;
                                    }
                                    float v487[4];
                                    float v488;
                                    v488 = 0.0f;
                                    int v489;
                                    v489 = 0;
                                    while (while_method_6(v489)){
                                        assert("Tensor range check" && 0 <= v489 && v489 < 1);
                                        int v491;
                                        v491 = 4 * v489;
                                        assert("Tensor range check" && 0 <= v489 && v489 < 1);
                                        int v492; float v493;
                                        Tuple14 tmp31 = Tuple14{0, 0.0f};
                                        v492 = tmp31.v0; v493 = tmp31.v1;
                                        while (while_method_3(v492)){
                                            assert("Tensor range check" && 0 <= v492 && v492 < 4);
                                            int v495;
                                            v495 = v492 + v491;
                                            float v496;
                                            v496 = v453[v495];
                                            float v497;
                                            v497 = v493 + v496;
                                            v493 = v497;
                                            v492 += 1 ;
                                        }
                                        auto v498 = cooperative_groups::coalesced_threads();
                                        int v499;
                                        v499 = threadIdx.x;
                                        int v500;
                                        v500 = v499 / 16;
                                        auto v501 = cooperative_groups::labeled_partition(v498,v500);
                                        Closure2 v502{};
                                        float v503;
                                        v503 = cooperative_groups::inclusive_scan(v501, v493, v502);
                                        float v504;
                                        v504 = v501.shfl_up(v503,1);
                                        bool v505;
                                        v505 = v501.thread_rank() == 0;
                                        float v506;
                                        if (v505){
                                            v506 = 0.0f;
                                        } else {
                                            v506 = v504;
                                        }
                                        float v507;
                                        v507 = v501.shfl(v503,v501.num_threads()-1);
                                        float v508;
                                        v508 = v488 + v506;
                                        int v509; float v510;
                                        Tuple14 tmp32 = Tuple14{0, v508};
                                        v509 = tmp32.v0; v510 = tmp32.v1;
                                        while (while_method_3(v509)){
                                            assert("Tensor range check" && 0 <= v509 && v509 < 4);
                                            int v512;
                                            v512 = v509 + v491;
                                            float v513;
                                            v513 = v453[v512];
                                            float v514;
                                            v514 = v510 + v513;
                                            assert("Tensor range check" && 0 <= v509 && v509 < 4);
                                            v487[v512] = v514;
                                            v510 = v514;
                                            v509 += 1 ;
                                        }
                                        float v515;
                                        v515 = v488 + v507;
                                        v488 = v515;
                                        v489 += 1 ;
                                    }
                                    float v516[4];
                                    bool v517[4];
                                    int v518;
                                    v518 = 0;
                                    while (while_method_6(v518)){
                                        int v520;
                                        v520 = 0;
                                        while (while_method_3(v520)){
                                            assert("Tensor range check" && 0 <= v518 && v518 < 1);
                                            assert("Tensor range check" && 0 <= v520 && v520 < 4);
                                            int v522;
                                            v522 = 4 * v518;
                                            int v523;
                                            v523 = v522 + v520;
                                            float v524;
                                            v524 = v487[v523];
                                            float v525;
                                            v525 = v453[v523];
                                            bool v526;
                                            v526 = v525 > 0.0f;
                                            assert("Tensor range check" && 0 <= v518 && v518 < 1);
                                            assert("Tensor range check" && 0 <= v520 && v520 < 4);
                                            v516[v523] = v524;
                                            v517[v523] = v526;
                                            v520 += 1 ;
                                        }
                                        v518 += 1 ;
                                    }
                                    float v527; bool v528;
                                    Tuple15 tmp33 = Tuple15{-1.0f / 0.0f, false};
                                    v527 = tmp33.v0; v528 = tmp33.v1;
                                    int v529;
                                    v529 = 0;
                                    while (while_method_6(v529)){
                                        int v531;
                                        v531 = 0;
                                        while (while_method_3(v531)){
                                            assert("Tensor range check" && 0 <= v529 && v529 < 1);
                                            assert("Tensor range check" && 0 <= v531 && v531 < 4);
                                            int v533;
                                            v533 = 4 * v529;
                                            int v534;
                                            v534 = v533 + v531;
                                            float v535;
                                            v535 = v516[v534];
                                            bool v536;
                                            v536 = v517[v534];
                                            float v543; bool v544;
                                            if (v528){
                                                if (v536){
                                                    bool v537;
                                                    v537 = v527 >= v535;
                                                    float v538;
                                                    if (v537){
                                                        v538 = v527;
                                                    } else {
                                                        v538 = v535;
                                                    }
                                                    v543 = v538; v544 = true;
                                                } else {
                                                    v543 = v527; v544 = v528;
                                                }
                                            } else {
                                                if (v536){
                                                    v543 = v535; v544 = v536;
                                                } else {
                                                    v543 = v527; v544 = v528;
                                                }
                                            }
                                            v527 = v543;
                                            v528 = v544;
                                            v531 += 1 ;
                                        }
                                        v529 += 1 ;
                                    }
                                    auto v545 = cooperative_groups::coalesced_threads();
                                    int v546;
                                    v546 = threadIdx.x;
                                    int v547;
                                    v547 = v546 / 16;
                                    auto v548 = cooperative_groups::labeled_partition(v545,v547);
                                    Closure3 v549{};
                                    float v550; bool v551;
                                    Tuple15 tmp34 = cooperative_groups::reduce(v548, Tuple15{v527, v528}, v549);
                                    v550 = tmp34.v0; v551 = tmp34.v1;
                                    bool v552;
                                    v552 = v551 == false;
                                    if (v552){
                                        assert("The local reduce must be true." && v551);
                                    } else {
                                    }
                                    float v554[4];
                                    int v555[4];
                                    int v556;
                                    v556 = 0;
                                    while (while_method_6(v556)){
                                        int v558;
                                        v558 = 0;
                                        while (while_method_3(v558)){
                                            assert("Tensor range check" && 0 <= v556 && v556 < 1);
                                            assert("Tensor range check" && 0 <= v558 && v558 < 4);
                                            int v560;
                                            v560 = 4 * v556;
                                            int v561;
                                            v561 = v560 + v558;
                                            int v562;
                                            v562 = v454[v561];
                                            float v563;
                                            v563 = curand_uniform(&v162);
                                            assert("Tensor range check" && 0 <= v556 && v556 < 1);
                                            assert("Tensor range check" && 0 <= v558 && v558 < 4);
                                            v554[v561] = v563;
                                            v555[v561] = v562;
                                            v558 += 1 ;
                                        }
                                        v556 += 1 ;
                                    }
                                    float v564; int v565;
                                    Tuple16 tmp35 = Tuple16{0.0f, 2147483647};
                                    v564 = tmp35.v0; v565 = tmp35.v1;
                                    int v566;
                                    v566 = 0;
                                    while (while_method_6(v566)){
                                        int v568;
                                        v568 = 0;
                                        while (while_method_3(v568)){
                                            assert("Tensor range check" && 0 <= v566 && v566 < 1);
                                            assert("Tensor range check" && 0 <= v568 && v568 < 4);
                                            int v570;
                                            v570 = 4 * v566;
                                            int v571;
                                            v571 = v570 + v568;
                                            float v572;
                                            v572 = v554[v571];
                                            int v573;
                                            v573 = v555[v571];
                                            bool v574;
                                            v574 = v565 < v573;
                                            float v575; int v576;
                                            if (v574){
                                                v575 = v564; v576 = v565;
                                            } else {
                                                v575 = v572; v576 = v573;
                                            }
                                            v564 = v575;
                                            v565 = v576;
                                            v568 += 1 ;
                                        }
                                        v566 += 1 ;
                                    }
                                    auto v577 = cooperative_groups::coalesced_threads();
                                    int v578;
                                    v578 = threadIdx.x;
                                    int v579;
                                    v579 = v578 / 16;
                                    auto v580 = cooperative_groups::labeled_partition(v577,v579);
                                    Closure4 v581{};
                                    float v582; int v583;
                                    Tuple16 tmp36 = cooperative_groups::reduce(v580, Tuple16{v564, v565}, v581);
                                    v582 = tmp36.v0; v583 = tmp36.v1;
                                    float v584;
                                    v584 = v550 * v582;
                                    int v585[4];
                                    bool v586[4];
                                    int v587;
                                    v587 = 0;
                                    while (while_method_6(v587)){
                                        int v589;
                                        v589 = 0;
                                        while (while_method_3(v589)){
                                            assert("Tensor range check" && 0 <= v587 && v587 < 1);
                                            assert("Tensor range check" && 0 <= v589 && v589 < 4);
                                            int v591;
                                            v591 = 4 * v587;
                                            int v592;
                                            v592 = v591 + v589;
                                            float v593;
                                            v593 = v516[v592];
                                            bool v594;
                                            v594 = v517[v592];
                                            int v595;
                                            v595 = v454[v592];
                                            int v598; bool v599;
                                            if (v594){
                                                float v596;
                                                v596 = v593 - v584;
                                                bool v597;
                                                v597 = v596 >= 0.0f;
                                                v598 = v595; v599 = v597;
                                            } else {
                                                v598 = 2147483647; v599 = false;
                                            }
                                            assert("Tensor range check" && 0 <= v587 && v587 < 1);
                                            assert("Tensor range check" && 0 <= v589 && v589 < 4);
                                            v585[v592] = v598;
                                            v586[v592] = v599;
                                            v589 += 1 ;
                                        }
                                        v587 += 1 ;
                                    }
                                    int v600; bool v601;
                                    Tuple17 tmp37 = Tuple17{2147483647, false};
                                    v600 = tmp37.v0; v601 = tmp37.v1;
                                    int v602;
                                    v602 = 0;
                                    while (while_method_6(v602)){
                                        int v604;
                                        v604 = 0;
                                        while (while_method_3(v604)){
                                            assert("Tensor range check" && 0 <= v602 && v602 < 1);
                                            assert("Tensor range check" && 0 <= v604 && v604 < 4);
                                            int v606;
                                            v606 = 4 * v602;
                                            int v607;
                                            v607 = v606 + v604;
                                            int v608;
                                            v608 = v585[v607];
                                            bool v609;
                                            v609 = v586[v607];
                                            int v616; bool v617;
                                            if (v601){
                                                if (v609){
                                                    bool v610;
                                                    v610 = v600 < v608;
                                                    int v611;
                                                    if (v610){
                                                        v611 = v600;
                                                    } else {
                                                        v611 = v608;
                                                    }
                                                    v616 = v611; v617 = true;
                                                } else {
                                                    v616 = v600; v617 = v601;
                                                }
                                            } else {
                                                if (v609){
                                                    v616 = v608; v617 = v609;
                                                } else {
                                                    v616 = v600; v617 = v601;
                                                }
                                            }
                                            v600 = v616;
                                            v601 = v617;
                                            v604 += 1 ;
                                        }
                                        v602 += 1 ;
                                    }
                                    auto v618 = cooperative_groups::coalesced_threads();
                                    int v619;
                                    v619 = threadIdx.x;
                                    int v620;
                                    v620 = v619 / 16;
                                    auto v621 = cooperative_groups::labeled_partition(v618,v620);
                                    Closure5 v622{};
                                    int v623; bool v624;
                                    Tuple17 tmp38 = cooperative_groups::reduce(v621, Tuple17{v600, v601}, v622);
                                    v623 = tmp38.v0; v624 = tmp38.v1;
                                    bool v625;
                                    v625 = v624 == false;
                                    if (v625){
                                        assert("The local reduce must be true." && v624);
                                    } else {
                                    }
                                    int v627;
                                    v627 = 0;
                                    while (while_method_6(v627)){
                                        assert("Tensor range check" && 0 <= v627 && v627 < 1);
                                        assert("Tensor range check" && 0 <= v627 && v627 < 1);
                                        v627 += 1 ;
                                    }
                                    assert("Tensor range check" && 0 <= v445 && v445 < 256);
                                    v422[v445] = v623;
                                    v433 += 1 ;
                                }
                                __syncthreads();
                                assert("Tensor range check" && 0 <= v424 && v424 < 256);
                                int v629;
                                v629 = v422[v424];
                                __syncthreads();
                                bool v630;
                                v630 = 0 == v629;
                                Union10 v639;
                                if (v630){
                                    v639 = Union10{Union10_1{}};
                                } else {
                                    bool v632;
                                    v632 = 1 == v629;
                                    if (v632){
                                        v639 = Union10{Union10_0{}};
                                    } else {
                                        bool v634;
                                        v634 = 2 == v629;
                                        if (v634){
                                            v639 = Union10{Union10_2{1, 1}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the NL Holdem model.");
                                            __trap();
                                        }
                                    }
                                }
                                Union1 v767;
                                switch (v639.tag) {
                                    case 0: { // AA_Call
                                        v767 = Union1{Union1_1{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v640;
                                        v640 = v147[0];
                                        int v642; int v643;
                                        Tuple4 tmp39 = Tuple4{1, v640};
                                        v642 = tmp39.v0; v643 = tmp39.v1;
                                        while (while_method_0(v642)){
                                            bool v645;
                                            v645 = 0 <= v642;
                                            bool v647;
                                            if (v645){
                                                bool v646;
                                                v646 = v642 < 2;
                                                v647 = v646;
                                            } else {
                                                v647 = false;
                                            }
                                            bool v648;
                                            v648 = v647 == false;
                                            if (v648){
                                                assert("Index must be in range." && v647);
                                            } else {
                                            }
                                            int v650;
                                            v650 = v147[v642];
                                            bool v652;
                                            v652 = v643 >= v650;
                                            int v653;
                                            if (v652){
                                                v653 = v643;
                                            } else {
                                                v653 = v650;
                                            }
                                            v643 = v653;
                                            v642 += 1 ;
                                        }
                                        bool v655;
                                        if (v153){
                                            bool v654;
                                            v654 = v151 < 2;
                                            v655 = v654;
                                        } else {
                                            v655 = false;
                                        }
                                        bool v656;
                                        v656 = v655 == false;
                                        if (v656){
                                            assert("Index must be in range." && v655);
                                        } else {
                                        }
                                        int v658;
                                        v658 = v147[v151];
                                        bool v660;
                                        v660 = v658 == v643;
                                        if (v660){
                                            v767 = Union1{Union1_1{}};
                                        } else {
                                            v767 = Union1{Union1_2{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        int v665 = v639.case2.v0; int v666 = v639.case2.v1;
                                        static_array<int,2> v667;
                                        int v669;
                                        v669 = 0;
                                        while (while_method_0(v669)){
                                            bool v671;
                                            v671 = 0 <= v669;
                                            bool v673;
                                            if (v671){
                                                bool v672;
                                                v672 = v669 < 2;
                                                v673 = v672;
                                            } else {
                                                v673 = false;
                                            }
                                            bool v674;
                                            v674 = v673 == false;
                                            if (v674){
                                                assert("Index must be in range." && v673);
                                            } else {
                                            }
                                            int v676;
                                            v676 = v149[v669];
                                            bool v679;
                                            if (v671){
                                                bool v678;
                                                v678 = v669 < 2;
                                                v679 = v678;
                                            } else {
                                                v679 = false;
                                            }
                                            bool v680;
                                            v680 = v679 == false;
                                            if (v680){
                                                assert("Index must be in range." && v679);
                                            } else {
                                            }
                                            int v682;
                                            v682 = v147[v669];
                                            int v684;
                                            v684 = v676 + v682;
                                            v667[v669] = v684;
                                            v669 += 1 ;
                                        }
                                        int v685;
                                        v685 = v147[0];
                                        int v687; int v688;
                                        Tuple4 tmp40 = Tuple4{1, v685};
                                        v687 = tmp40.v0; v688 = tmp40.v1;
                                        while (while_method_0(v687)){
                                            bool v690;
                                            v690 = 0 <= v687;
                                            bool v692;
                                            if (v690){
                                                bool v691;
                                                v691 = v687 < 2;
                                                v692 = v691;
                                            } else {
                                                v692 = false;
                                            }
                                            bool v693;
                                            v693 = v692 == false;
                                            if (v693){
                                                assert("Index must be in range." && v692);
                                            } else {
                                            }
                                            int v695;
                                            v695 = v147[v687];
                                            bool v697;
                                            v697 = v688 >= v695;
                                            int v698;
                                            if (v697){
                                                v698 = v688;
                                            } else {
                                                v698 = v695;
                                            }
                                            v688 = v698;
                                            v687 += 1 ;
                                        }
                                        bool v700;
                                        if (v153){
                                            bool v699;
                                            v699 = v151 < 2;
                                            v700 = v699;
                                        } else {
                                            v700 = false;
                                        }
                                        bool v701;
                                        v701 = v700 == false;
                                        if (v701){
                                            assert("Index must be in range." && v700);
                                        } else {
                                        }
                                        int v703;
                                        v703 = v667[v151];
                                        bool v705;
                                        v705 = v688 < v703;
                                        int v706;
                                        if (v705){
                                            v706 = v688;
                                        } else {
                                            v706 = v703;
                                        }
                                        static_array<int,2> v707;
                                        int v709;
                                        v709 = 0;
                                        while (while_method_0(v709)){
                                            bool v711;
                                            v711 = 0 <= v709;
                                            bool v713;
                                            if (v711){
                                                bool v712;
                                                v712 = v709 < 2;
                                                v713 = v712;
                                            } else {
                                                v713 = false;
                                            }
                                            bool v714;
                                            v714 = v713 == false;
                                            if (v714){
                                                assert("Index must be in range." && v713);
                                            } else {
                                            }
                                            int v716;
                                            v716 = v147[v709];
                                            bool v718;
                                            v718 = v151 == v709;
                                            int v719;
                                            if (v718){
                                                v719 = v706;
                                            } else {
                                                v719 = v716;
                                            }
                                            v707[v709] = v719;
                                            v709 += 1 ;
                                        }
                                        int v720;
                                        v720 = v707[0];
                                        int v722; int v723;
                                        Tuple4 tmp41 = Tuple4{1, v720};
                                        v722 = tmp41.v0; v723 = tmp41.v1;
                                        while (while_method_0(v722)){
                                            bool v725;
                                            v725 = 0 <= v722;
                                            bool v727;
                                            if (v725){
                                                bool v726;
                                                v726 = v722 < 2;
                                                v727 = v726;
                                            } else {
                                                v727 = false;
                                            }
                                            bool v728;
                                            v728 = v727 == false;
                                            if (v728){
                                                assert("Index must be in range." && v727);
                                            } else {
                                            }
                                            int v730;
                                            v730 = v707[v722];
                                            int v732;
                                            v732 = v723 + v730;
                                            v723 = v732;
                                            v722 += 1 ;
                                        }
                                        static_array<int,2> v733;
                                        int v735;
                                        v735 = 0;
                                        while (while_method_0(v735)){
                                            bool v737;
                                            v737 = 0 <= v735;
                                            bool v739;
                                            if (v737){
                                                bool v738;
                                                v738 = v735 < 2;
                                                v739 = v738;
                                            } else {
                                                v739 = false;
                                            }
                                            bool v740;
                                            v740 = v739 == false;
                                            if (v740){
                                                assert("Index must be in range." && v739);
                                            } else {
                                            }
                                            int v742;
                                            v742 = v667[v735];
                                            bool v745;
                                            if (v737){
                                                bool v744;
                                                v744 = v735 < 2;
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
                                            int v748;
                                            v748 = v707[v735];
                                            int v750;
                                            v750 = v742 - v748;
                                            v733[v735] = v750;
                                            v735 += 1 ;
                                        }
                                        int v751;
                                        v751 = v665 * v723;
                                        int v752;
                                        v752 = v751 / v666;
                                        bool v753;
                                        v753 = v145 >= v752;
                                        int v754;
                                        if (v753){
                                            v754 = v145;
                                        } else {
                                            v754 = v752;
                                        }
                                        bool v756;
                                        if (v153){
                                            bool v755;
                                            v755 = v151 < 2;
                                            v756 = v755;
                                        } else {
                                            v756 = false;
                                        }
                                        bool v757;
                                        v757 = v756 == false;
                                        if (v757){
                                            assert("Index must be in range." && v756);
                                        } else {
                                        }
                                        int v759;
                                        v759 = v733[v151];
                                        bool v761;
                                        v761 = v754 >= v759;
                                        if (v761){
                                            v767 = Union1{Union1_0{}};
                                        } else {
                                            v767 = Union1{Union1_3{v754}};
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                int v768;
                                v768 = sizeof(Union1);
                                unsigned long long v769;
                                v769 = (unsigned long long)v768;
                                bool v770;
                                v770 = v769 <= 98304ull;
                                bool v771;
                                v771 = v770 == false;
                                if (v771){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v770);
                                } else {
                                }
                                extern __shared__ unsigned char v773[];
                                bool v774;
                                v774 = v769 <= v769;
                                bool v775;
                                v775 = v774 == false;
                                if (v775){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v774);
                                } else {
                                }
                                Union1 * v777;
                                v777 = reinterpret_cast<Union1 *>(&v773[0ull]);
                                int v779;
                                v779 = threadIdx.x;
                                bool v780;
                                v780 = v779 == 0;
                                if (v780){
                                    v777[0] = v767;
                                } else {
                                }
                                __syncthreads();
                                Union1 v781;
                                v781 = v777[0];
                                __syncthreads();
                                Union6 v782;
                                v782 = Union6{Union6_2{v151, v781}};
                                v5.push(v782);
                                Union4 v1110;
                                switch (v781.tag) {
                                    case 0: { // A_All_In
                                        static_array<int,2> v986;
                                        int v988;
                                        v988 = 0;
                                        while (while_method_0(v988)){
                                            bool v990;
                                            v990 = 0 <= v988;
                                            bool v992;
                                            if (v990){
                                                bool v991;
                                                v991 = v988 < 2;
                                                v992 = v991;
                                            } else {
                                                v992 = false;
                                            }
                                            bool v993;
                                            v993 = v992 == false;
                                            if (v993){
                                                assert("Index must be in range." && v992);
                                            } else {
                                            }
                                            int v995;
                                            v995 = v149[v988];
                                            bool v998;
                                            if (v990){
                                                bool v997;
                                                v997 = v988 < 2;
                                                v998 = v997;
                                            } else {
                                                v998 = false;
                                            }
                                            bool v999;
                                            v999 = v998 == false;
                                            if (v999){
                                                assert("Index must be in range." && v998);
                                            } else {
                                            }
                                            int v1001;
                                            v1001 = v147[v988];
                                            int v1003;
                                            v1003 = v995 + v1001;
                                            v986[v988] = v1003;
                                            v988 += 1 ;
                                        }
                                        int v1004;
                                        v1004 = v147[0];
                                        int v1006; int v1007;
                                        Tuple4 tmp42 = Tuple4{1, v1004};
                                        v1006 = tmp42.v0; v1007 = tmp42.v1;
                                        while (while_method_0(v1006)){
                                            bool v1009;
                                            v1009 = 0 <= v1006;
                                            bool v1011;
                                            if (v1009){
                                                bool v1010;
                                                v1010 = v1006 < 2;
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
                                            v1014 = v147[v1006];
                                            bool v1016;
                                            v1016 = v1007 >= v1014;
                                            int v1017;
                                            if (v1016){
                                                v1017 = v1007;
                                            } else {
                                                v1017 = v1014;
                                            }
                                            v1007 = v1017;
                                            v1006 += 1 ;
                                        }
                                        bool v1019;
                                        if (v153){
                                            bool v1018;
                                            v1018 = v151 < 2;
                                            v1019 = v1018;
                                        } else {
                                            v1019 = false;
                                        }
                                        bool v1020;
                                        v1020 = v1019 == false;
                                        if (v1020){
                                            assert("Index must be in range." && v1019);
                                        } else {
                                        }
                                        int v1022;
                                        v1022 = v986[v151];
                                        bool v1024;
                                        v1024 = v1007 < v1022;
                                        int v1025;
                                        if (v1024){
                                            v1025 = v1007;
                                        } else {
                                            v1025 = v1022;
                                        }
                                        static_array<int,2> v1026;
                                        int v1028;
                                        v1028 = 0;
                                        while (while_method_0(v1028)){
                                            bool v1030;
                                            v1030 = 0 <= v1028;
                                            bool v1032;
                                            if (v1030){
                                                bool v1031;
                                                v1031 = v1028 < 2;
                                                v1032 = v1031;
                                            } else {
                                                v1032 = false;
                                            }
                                            bool v1033;
                                            v1033 = v1032 == false;
                                            if (v1033){
                                                assert("Index must be in range." && v1032);
                                            } else {
                                            }
                                            int v1035;
                                            v1035 = v147[v1028];
                                            bool v1037;
                                            v1037 = v151 == v1028;
                                            int v1038;
                                            if (v1037){
                                                v1038 = v1025;
                                            } else {
                                                v1038 = v1035;
                                            }
                                            v1026[v1028] = v1038;
                                            v1028 += 1 ;
                                        }
                                        static_array<int,2> v1039;
                                        int v1041;
                                        v1041 = 0;
                                        while (while_method_0(v1041)){
                                            bool v1043;
                                            v1043 = 0 <= v1041;
                                            bool v1045;
                                            if (v1043){
                                                bool v1044;
                                                v1044 = v1041 < 2;
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
                                            int v1048;
                                            v1048 = v986[v1041];
                                            bool v1051;
                                            if (v1043){
                                                bool v1050;
                                                v1050 = v1041 < 2;
                                                v1051 = v1050;
                                            } else {
                                                v1051 = false;
                                            }
                                            bool v1052;
                                            v1052 = v1051 == false;
                                            if (v1052){
                                                assert("Index must be in range." && v1051);
                                            } else {
                                            }
                                            int v1054;
                                            v1054 = v1026[v1041];
                                            int v1056;
                                            v1056 = v1048 - v1054;
                                            v1039[v1041] = v1056;
                                            v1041 += 1 ;
                                        }
                                        bool v1058;
                                        if (v153){
                                            bool v1057;
                                            v1057 = v151 < 2;
                                            v1058 = v1057;
                                        } else {
                                            v1058 = false;
                                        }
                                        bool v1059;
                                        v1059 = v1058 == false;
                                        if (v1059){
                                            assert("Index must be in range." && v1058);
                                        } else {
                                        }
                                        int v1061;
                                        v1061 = v1039[v151];
                                        int v1063;
                                        v1063 = v1007 + v1061;
                                        bool v1065;
                                        if (v153){
                                            bool v1064;
                                            v1064 = v151 < 2;
                                            v1065 = v1064;
                                        } else {
                                            v1065 = false;
                                        }
                                        bool v1066;
                                        v1066 = v1065 == false;
                                        if (v1066){
                                            assert("Index must be in range." && v1065);
                                        } else {
                                        }
                                        int v1068;
                                        v1068 = v986[v151];
                                        bool v1070;
                                        v1070 = v1063 < v1068;
                                        int v1071;
                                        if (v1070){
                                            v1071 = v1063;
                                        } else {
                                            v1071 = v1068;
                                        }
                                        static_array<int,2> v1072;
                                        int v1074;
                                        v1074 = 0;
                                        while (while_method_0(v1074)){
                                            bool v1076;
                                            v1076 = 0 <= v1074;
                                            bool v1078;
                                            if (v1076){
                                                bool v1077;
                                                v1077 = v1074 < 2;
                                                v1078 = v1077;
                                            } else {
                                                v1078 = false;
                                            }
                                            bool v1079;
                                            v1079 = v1078 == false;
                                            if (v1079){
                                                assert("Index must be in range." && v1078);
                                            } else {
                                            }
                                            int v1081;
                                            v1081 = v147[v1074];
                                            bool v1083;
                                            v1083 = v151 == v1074;
                                            int v1084;
                                            if (v1083){
                                                v1084 = v1071;
                                            } else {
                                                v1084 = v1081;
                                            }
                                            v1072[v1074] = v1084;
                                            v1074 += 1 ;
                                        }
                                        static_array<int,2> v1085;
                                        int v1087;
                                        v1087 = 0;
                                        while (while_method_0(v1087)){
                                            bool v1089;
                                            v1089 = 0 <= v1087;
                                            bool v1091;
                                            if (v1089){
                                                bool v1090;
                                                v1090 = v1087 < 2;
                                                v1091 = v1090;
                                            } else {
                                                v1091 = false;
                                            }
                                            bool v1092;
                                            v1092 = v1091 == false;
                                            if (v1092){
                                                assert("Index must be in range." && v1091);
                                            } else {
                                            }
                                            int v1094;
                                            v1094 = v986[v1087];
                                            bool v1097;
                                            if (v1089){
                                                bool v1096;
                                                v1096 = v1087 < 2;
                                                v1097 = v1096;
                                            } else {
                                                v1097 = false;
                                            }
                                            bool v1098;
                                            v1098 = v1097 == false;
                                            if (v1098){
                                                assert("Index must be in range." && v1097);
                                            } else {
                                            }
                                            int v1100;
                                            v1100 = v1072[v1087];
                                            int v1102;
                                            v1102 = v1094 - v1100;
                                            v1085[v1087] = v1102;
                                            v1087 += 1 ;
                                        }
                                        bool v1103;
                                        v1103 = v1061 >= v145;
                                        int v1104;
                                        if (v1103){
                                            v1104 = v1061;
                                        } else {
                                            v1104 = v145;
                                        }
                                        int v1105;
                                        v1105 = v148 + 1;
                                        v1110 = try_round_36(v1104, v146, v1072, v1105, v1085, v150);
                                        break;
                                    }
                                    case 1: { // A_Call
                                        static_array<int,2> v784;
                                        int v786;
                                        v786 = 0;
                                        while (while_method_0(v786)){
                                            bool v788;
                                            v788 = 0 <= v786;
                                            bool v790;
                                            if (v788){
                                                bool v789;
                                                v789 = v786 < 2;
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
                                            int v793;
                                            v793 = v149[v786];
                                            bool v796;
                                            if (v788){
                                                bool v795;
                                                v795 = v786 < 2;
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
                                            int v799;
                                            v799 = v147[v786];
                                            int v801;
                                            v801 = v793 + v799;
                                            v784[v786] = v801;
                                            v786 += 1 ;
                                        }
                                        int v802;
                                        v802 = v147[0];
                                        int v804; int v805;
                                        Tuple4 tmp43 = Tuple4{1, v802};
                                        v804 = tmp43.v0; v805 = tmp43.v1;
                                        while (while_method_0(v804)){
                                            bool v807;
                                            v807 = 0 <= v804;
                                            bool v809;
                                            if (v807){
                                                bool v808;
                                                v808 = v804 < 2;
                                                v809 = v808;
                                            } else {
                                                v809 = false;
                                            }
                                            bool v810;
                                            v810 = v809 == false;
                                            if (v810){
                                                assert("Index must be in range." && v809);
                                            } else {
                                            }
                                            int v812;
                                            v812 = v147[v804];
                                            bool v814;
                                            v814 = v805 >= v812;
                                            int v815;
                                            if (v814){
                                                v815 = v805;
                                            } else {
                                                v815 = v812;
                                            }
                                            v805 = v815;
                                            v804 += 1 ;
                                        }
                                        bool v817;
                                        if (v153){
                                            bool v816;
                                            v816 = v151 < 2;
                                            v817 = v816;
                                        } else {
                                            v817 = false;
                                        }
                                        bool v818;
                                        v818 = v817 == false;
                                        if (v818){
                                            assert("Index must be in range." && v817);
                                        } else {
                                        }
                                        int v820;
                                        v820 = v784[v151];
                                        bool v822;
                                        v822 = v805 < v820;
                                        int v823;
                                        if (v822){
                                            v823 = v805;
                                        } else {
                                            v823 = v820;
                                        }
                                        static_array<int,2> v824;
                                        int v826;
                                        v826 = 0;
                                        while (while_method_0(v826)){
                                            bool v828;
                                            v828 = 0 <= v826;
                                            bool v830;
                                            if (v828){
                                                bool v829;
                                                v829 = v826 < 2;
                                                v830 = v829;
                                            } else {
                                                v830 = false;
                                            }
                                            bool v831;
                                            v831 = v830 == false;
                                            if (v831){
                                                assert("Index must be in range." && v830);
                                            } else {
                                            }
                                            int v833;
                                            v833 = v147[v826];
                                            bool v835;
                                            v835 = v151 == v826;
                                            int v836;
                                            if (v835){
                                                v836 = v823;
                                            } else {
                                                v836 = v833;
                                            }
                                            v824[v826] = v836;
                                            v826 += 1 ;
                                        }
                                        static_array<int,2> v837;
                                        int v839;
                                        v839 = 0;
                                        while (while_method_0(v839)){
                                            bool v841;
                                            v841 = 0 <= v839;
                                            bool v843;
                                            if (v841){
                                                bool v842;
                                                v842 = v839 < 2;
                                                v843 = v842;
                                            } else {
                                                v843 = false;
                                            }
                                            bool v844;
                                            v844 = v843 == false;
                                            if (v844){
                                                assert("Index must be in range." && v843);
                                            } else {
                                            }
                                            int v846;
                                            v846 = v784[v839];
                                            bool v849;
                                            if (v841){
                                                bool v848;
                                                v848 = v839 < 2;
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
                                            int v852;
                                            v852 = v824[v839];
                                            int v854;
                                            v854 = v846 - v852;
                                            v837[v839] = v854;
                                            v839 += 1 ;
                                        }
                                        bool v855;
                                        v855 = v151 < 2;
                                        if (v855){
                                            int v856;
                                            v856 = v148 + 1;
                                            v1110 = try_round_36(v145, v146, v824, v856, v837, v150);
                                        } else {
                                            v1110 = go_next_street_38(v145, v146, v824, v148, v837, v150);
                                        }
                                        break;
                                    }
                                    case 2: { // A_Fold
                                        v1110 = Union4{Union4_1{v145, v146, v147, v148, v149, v150}};
                                        break;
                                    }
                                    case 3: { // A_Raise
                                        int v860 = v781.case3.v0;
                                        bool v861;
                                        v861 = v145 <= v860;
                                        bool v862;
                                        v862 = v861 == false;
                                        if (v862){
                                            assert("The raise amount must match the minimum." && v861);
                                        } else {
                                        }
                                        static_array<int,2> v864;
                                        int v866;
                                        v866 = 0;
                                        while (while_method_0(v866)){
                                            bool v868;
                                            v868 = 0 <= v866;
                                            bool v870;
                                            if (v868){
                                                bool v869;
                                                v869 = v866 < 2;
                                                v870 = v869;
                                            } else {
                                                v870 = false;
                                            }
                                            bool v871;
                                            v871 = v870 == false;
                                            if (v871){
                                                assert("Index must be in range." && v870);
                                            } else {
                                            }
                                            int v873;
                                            v873 = v149[v866];
                                            bool v876;
                                            if (v868){
                                                bool v875;
                                                v875 = v866 < 2;
                                                v876 = v875;
                                            } else {
                                                v876 = false;
                                            }
                                            bool v877;
                                            v877 = v876 == false;
                                            if (v877){
                                                assert("Index must be in range." && v876);
                                            } else {
                                            }
                                            int v879;
                                            v879 = v147[v866];
                                            int v881;
                                            v881 = v873 + v879;
                                            v864[v866] = v881;
                                            v866 += 1 ;
                                        }
                                        int v882;
                                        v882 = v147[0];
                                        int v884; int v885;
                                        Tuple4 tmp44 = Tuple4{1, v882};
                                        v884 = tmp44.v0; v885 = tmp44.v1;
                                        while (while_method_0(v884)){
                                            bool v887;
                                            v887 = 0 <= v884;
                                            bool v889;
                                            if (v887){
                                                bool v888;
                                                v888 = v884 < 2;
                                                v889 = v888;
                                            } else {
                                                v889 = false;
                                            }
                                            bool v890;
                                            v890 = v889 == false;
                                            if (v890){
                                                assert("Index must be in range." && v889);
                                            } else {
                                            }
                                            int v892;
                                            v892 = v147[v884];
                                            bool v894;
                                            v894 = v885 >= v892;
                                            int v895;
                                            if (v894){
                                                v895 = v885;
                                            } else {
                                                v895 = v892;
                                            }
                                            v885 = v895;
                                            v884 += 1 ;
                                        }
                                        bool v897;
                                        if (v153){
                                            bool v896;
                                            v896 = v151 < 2;
                                            v897 = v896;
                                        } else {
                                            v897 = false;
                                        }
                                        bool v898;
                                        v898 = v897 == false;
                                        if (v898){
                                            assert("Index must be in range." && v897);
                                        } else {
                                        }
                                        int v900;
                                        v900 = v864[v151];
                                        bool v902;
                                        v902 = v885 < v900;
                                        int v903;
                                        if (v902){
                                            v903 = v885;
                                        } else {
                                            v903 = v900;
                                        }
                                        static_array<int,2> v904;
                                        int v906;
                                        v906 = 0;
                                        while (while_method_0(v906)){
                                            bool v908;
                                            v908 = 0 <= v906;
                                            bool v910;
                                            if (v908){
                                                bool v909;
                                                v909 = v906 < 2;
                                                v910 = v909;
                                            } else {
                                                v910 = false;
                                            }
                                            bool v911;
                                            v911 = v910 == false;
                                            if (v911){
                                                assert("Index must be in range." && v910);
                                            } else {
                                            }
                                            int v913;
                                            v913 = v147[v906];
                                            bool v915;
                                            v915 = v151 == v906;
                                            int v916;
                                            if (v915){
                                                v916 = v903;
                                            } else {
                                                v916 = v913;
                                            }
                                            v904[v906] = v916;
                                            v906 += 1 ;
                                        }
                                        static_array<int,2> v917;
                                        int v919;
                                        v919 = 0;
                                        while (while_method_0(v919)){
                                            bool v921;
                                            v921 = 0 <= v919;
                                            bool v923;
                                            if (v921){
                                                bool v922;
                                                v922 = v919 < 2;
                                                v923 = v922;
                                            } else {
                                                v923 = false;
                                            }
                                            bool v924;
                                            v924 = v923 == false;
                                            if (v924){
                                                assert("Index must be in range." && v923);
                                            } else {
                                            }
                                            int v926;
                                            v926 = v864[v919];
                                            bool v929;
                                            if (v921){
                                                bool v928;
                                                v928 = v919 < 2;
                                                v929 = v928;
                                            } else {
                                                v929 = false;
                                            }
                                            bool v930;
                                            v930 = v929 == false;
                                            if (v930){
                                                assert("Index must be in range." && v929);
                                            } else {
                                            }
                                            int v932;
                                            v932 = v904[v919];
                                            int v934;
                                            v934 = v926 - v932;
                                            v917[v919] = v934;
                                            v919 += 1 ;
                                        }
                                        bool v936;
                                        if (v153){
                                            bool v935;
                                            v935 = v151 < 2;
                                            v936 = v935;
                                        } else {
                                            v936 = false;
                                        }
                                        bool v937;
                                        v937 = v936 == false;
                                        if (v937){
                                            assert("Index must be in range." && v936);
                                        } else {
                                        }
                                        int v939;
                                        v939 = v917[v151];
                                        bool v941;
                                        v941 = v860 < v939;
                                        bool v942;
                                        v942 = v941 == false;
                                        if (v942){
                                            assert("The raise amount must be less than the stack size after calling." && v941);
                                        } else {
                                        }
                                        int v944;
                                        v944 = v885 + v860;
                                        bool v946;
                                        if (v153){
                                            bool v945;
                                            v945 = v151 < 2;
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
                                        int v949;
                                        v949 = v864[v151];
                                        bool v951;
                                        v951 = v944 < v949;
                                        int v952;
                                        if (v951){
                                            v952 = v944;
                                        } else {
                                            v952 = v949;
                                        }
                                        static_array<int,2> v953;
                                        int v955;
                                        v955 = 0;
                                        while (while_method_0(v955)){
                                            bool v957;
                                            v957 = 0 <= v955;
                                            bool v959;
                                            if (v957){
                                                bool v958;
                                                v958 = v955 < 2;
                                                v959 = v958;
                                            } else {
                                                v959 = false;
                                            }
                                            bool v960;
                                            v960 = v959 == false;
                                            if (v960){
                                                assert("Index must be in range." && v959);
                                            } else {
                                            }
                                            int v962;
                                            v962 = v147[v955];
                                            bool v964;
                                            v964 = v151 == v955;
                                            int v965;
                                            if (v964){
                                                v965 = v952;
                                            } else {
                                                v965 = v962;
                                            }
                                            v953[v955] = v965;
                                            v955 += 1 ;
                                        }
                                        static_array<int,2> v966;
                                        int v968;
                                        v968 = 0;
                                        while (while_method_0(v968)){
                                            bool v970;
                                            v970 = 0 <= v968;
                                            bool v972;
                                            if (v970){
                                                bool v971;
                                                v971 = v968 < 2;
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
                                            v975 = v864[v968];
                                            bool v978;
                                            if (v970){
                                                bool v977;
                                                v977 = v968 < 2;
                                                v978 = v977;
                                            } else {
                                                v978 = false;
                                            }
                                            bool v979;
                                            v979 = v978 == false;
                                            if (v979){
                                                assert("Index must be in range." && v978);
                                            } else {
                                            }
                                            int v981;
                                            v981 = v953[v968];
                                            int v983;
                                            v983 = v975 - v981;
                                            v966[v968] = v983;
                                            v968 += 1 ;
                                        }
                                        int v984;
                                        v984 = v148 + 1;
                                        v1110 = try_round_36(v860, v146, v953, v984, v966, v150);
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                v2166 = Union3{Union3_1{v1110}};
                                break;
                            }
                            case 2: { // Human
                                Union7 v1112;
                                v1112 = Union7{Union7_2{v145, v146, v147, v148, v149, v150}};
                                v3.v5 = v1112;
                                Union3 v1113;
                                v1113 = Union3{Union3_1{v10}};
                                v3.v1 = v1113;
                                v2166 = Union3{Union3_0{}};
                                break;
                            }
                            case 3: { // Random
                                curandStatePhilox4_32_10_t & v1115 = v3.v4;
                                curandStatePhilox4_32_10_t & v1116 = v1115;
                                static_array<int,2> v1117;
                                int v1119;
                                v1119 = 0;
                                while (while_method_0(v1119)){
                                    bool v1121;
                                    v1121 = 0 <= v1119;
                                    bool v1123;
                                    if (v1121){
                                        bool v1122;
                                        v1122 = v1119 < 2;
                                        v1123 = v1122;
                                    } else {
                                        v1123 = false;
                                    }
                                    bool v1124;
                                    v1124 = v1123 == false;
                                    if (v1124){
                                        assert("Index must be in range." && v1123);
                                    } else {
                                    }
                                    int v1126;
                                    v1126 = v149[v1119];
                                    bool v1129;
                                    if (v1121){
                                        bool v1128;
                                        v1128 = v1119 < 2;
                                        v1129 = v1128;
                                    } else {
                                        v1129 = false;
                                    }
                                    bool v1130;
                                    v1130 = v1129 == false;
                                    if (v1130){
                                        assert("Index must be in range." && v1129);
                                    } else {
                                    }
                                    int v1132;
                                    v1132 = v147[v1119];
                                    int v1134;
                                    v1134 = v1126 + v1132;
                                    v1117[v1119] = v1134;
                                    v1119 += 1 ;
                                }
                                int v1135;
                                v1135 = v147[0];
                                int v1137; int v1138;
                                Tuple4 tmp45 = Tuple4{1, v1135};
                                v1137 = tmp45.v0; v1138 = tmp45.v1;
                                while (while_method_0(v1137)){
                                    bool v1140;
                                    v1140 = 0 <= v1137;
                                    bool v1142;
                                    if (v1140){
                                        bool v1141;
                                        v1141 = v1137 < 2;
                                        v1142 = v1141;
                                    } else {
                                        v1142 = false;
                                    }
                                    bool v1143;
                                    v1143 = v1142 == false;
                                    if (v1143){
                                        assert("Index must be in range." && v1142);
                                    } else {
                                    }
                                    int v1145;
                                    v1145 = v147[v1137];
                                    bool v1147;
                                    v1147 = v1138 >= v1145;
                                    int v1148;
                                    if (v1147){
                                        v1148 = v1138;
                                    } else {
                                        v1148 = v1145;
                                    }
                                    v1138 = v1148;
                                    v1137 += 1 ;
                                }
                                bool v1150;
                                if (v153){
                                    bool v1149;
                                    v1149 = v151 < 2;
                                    v1150 = v1149;
                                } else {
                                    v1150 = false;
                                }
                                bool v1151;
                                v1151 = v1150 == false;
                                if (v1151){
                                    assert("Index must be in range." && v1150);
                                } else {
                                }
                                int v1153;
                                v1153 = v1117[v151];
                                bool v1155;
                                v1155 = v1138 < v1153;
                                int v1156;
                                if (v1155){
                                    v1156 = v1138;
                                } else {
                                    v1156 = v1153;
                                }
                                static_array<int,2> v1157;
                                int v1159;
                                v1159 = 0;
                                while (while_method_0(v1159)){
                                    bool v1161;
                                    v1161 = 0 <= v1159;
                                    bool v1163;
                                    if (v1161){
                                        bool v1162;
                                        v1162 = v1159 < 2;
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
                                    int v1166;
                                    v1166 = v147[v1159];
                                    bool v1168;
                                    v1168 = v151 == v1159;
                                    int v1169;
                                    if (v1168){
                                        v1169 = v1156;
                                    } else {
                                        v1169 = v1166;
                                    }
                                    v1157[v1159] = v1169;
                                    v1159 += 1 ;
                                }
                                int v1170;
                                v1170 = v1157[0];
                                int v1172; int v1173;
                                Tuple4 tmp46 = Tuple4{1, v1170};
                                v1172 = tmp46.v0; v1173 = tmp46.v1;
                                while (while_method_0(v1172)){
                                    bool v1175;
                                    v1175 = 0 <= v1172;
                                    bool v1177;
                                    if (v1175){
                                        bool v1176;
                                        v1176 = v1172 < 2;
                                        v1177 = v1176;
                                    } else {
                                        v1177 = false;
                                    }
                                    bool v1178;
                                    v1178 = v1177 == false;
                                    if (v1178){
                                        assert("Index must be in range." && v1177);
                                    } else {
                                    }
                                    int v1180;
                                    v1180 = v1157[v1172];
                                    int v1182;
                                    v1182 = v1173 + v1180;
                                    v1173 = v1182;
                                    v1172 += 1 ;
                                }
                                static_array<int,2> v1183;
                                int v1185;
                                v1185 = 0;
                                while (while_method_0(v1185)){
                                    bool v1187;
                                    v1187 = 0 <= v1185;
                                    bool v1189;
                                    if (v1187){
                                        bool v1188;
                                        v1188 = v1185 < 2;
                                        v1189 = v1188;
                                    } else {
                                        v1189 = false;
                                    }
                                    bool v1190;
                                    v1190 = v1189 == false;
                                    if (v1190){
                                        assert("Index must be in range." && v1189);
                                    } else {
                                    }
                                    int v1192;
                                    v1192 = v1117[v1185];
                                    bool v1195;
                                    if (v1187){
                                        bool v1194;
                                        v1194 = v1185 < 2;
                                        v1195 = v1194;
                                    } else {
                                        v1195 = false;
                                    }
                                    bool v1196;
                                    v1196 = v1195 == false;
                                    if (v1196){
                                        assert("Index must be in range." && v1195);
                                    } else {
                                    }
                                    int v1198;
                                    v1198 = v1157[v1185];
                                    int v1200;
                                    v1200 = v1192 - v1198;
                                    v1183[v1185] = v1200;
                                    v1185 += 1 ;
                                }
                                bool v1202;
                                if (v153){
                                    bool v1201;
                                    v1201 = v151 < 2;
                                    v1202 = v1201;
                                } else {
                                    v1202 = false;
                                }
                                bool v1203;
                                v1203 = v1202 == false;
                                if (v1203){
                                    assert("Index must be in range." && v1202);
                                } else {
                                }
                                int v1205;
                                v1205 = v147[v151];
                                bool v1207;
                                v1207 = v1205 < v1138;
                                float v1208;
                                if (v1207){
                                    v1208 = 1.0f;
                                } else {
                                    v1208 = 0.0f;
                                }
                                int v1209;
                                v1209 = v1173 / 3;
                                bool v1210;
                                v1210 = v145 <= v1209;
                                bool v1218;
                                if (v1210){
                                    bool v1212;
                                    if (v153){
                                        bool v1211;
                                        v1211 = v151 < 2;
                                        v1212 = v1211;
                                    } else {
                                        v1212 = false;
                                    }
                                    bool v1213;
                                    v1213 = v1212 == false;
                                    if (v1213){
                                        assert("Index must be in range." && v1212);
                                    } else {
                                    }
                                    int v1215;
                                    v1215 = v1183[v151];
                                    bool v1217;
                                    v1217 = v1209 < v1215;
                                    v1218 = v1217;
                                } else {
                                    v1218 = false;
                                }
                                float v1219;
                                if (v1218){
                                    v1219 = 1.0f;
                                } else {
                                    v1219 = 0.0f;
                                }
                                int v1220;
                                v1220 = v1173 / 2;
                                bool v1221;
                                v1221 = v145 <= v1220;
                                bool v1229;
                                if (v1221){
                                    bool v1223;
                                    if (v153){
                                        bool v1222;
                                        v1222 = v151 < 2;
                                        v1223 = v1222;
                                    } else {
                                        v1223 = false;
                                    }
                                    bool v1224;
                                    v1224 = v1223 == false;
                                    if (v1224){
                                        assert("Index must be in range." && v1223);
                                    } else {
                                    }
                                    int v1226;
                                    v1226 = v1183[v151];
                                    bool v1228;
                                    v1228 = v1220 < v1226;
                                    v1229 = v1228;
                                } else {
                                    v1229 = false;
                                }
                                float v1230;
                                if (v1229){
                                    v1230 = 1.0f;
                                } else {
                                    v1230 = 0.0f;
                                }
                                bool v1231;
                                v1231 = v145 <= v1173;
                                bool v1239;
                                if (v1231){
                                    bool v1233;
                                    if (v153){
                                        bool v1232;
                                        v1232 = v151 < 2;
                                        v1233 = v1232;
                                    } else {
                                        v1233 = false;
                                    }
                                    bool v1234;
                                    v1234 = v1233 == false;
                                    if (v1234){
                                        assert("Index must be in range." && v1233);
                                    } else {
                                    }
                                    int v1236;
                                    v1236 = v1183[v151];
                                    bool v1238;
                                    v1238 = v1173 < v1236;
                                    v1239 = v1238;
                                } else {
                                    v1239 = false;
                                }
                                float v1240;
                                if (v1239){
                                    v1240 = 1.0f;
                                } else {
                                    v1240 = 0.0f;
                                }
                                static_array<Tuple18,6> v1241;
                                Union1 v1243;
                                v1243 = Union1{Union1_2{}};
                                v1241[0] = Tuple18{v1243, v1208};
                                Union1 v1245;
                                v1245 = Union1{Union1_1{}};
                                v1241[1] = Tuple18{v1245, 4.0f};
                                Union1 v1247;
                                v1247 = Union1{Union1_3{v1209}};
                                v1241[2] = Tuple18{v1247, v1219};
                                Union1 v1249;
                                v1249 = Union1{Union1_3{v1220}};
                                v1241[3] = Tuple18{v1249, v1230};
                                Union1 v1251;
                                v1251 = Union1{Union1_3{v1173}};
                                v1241[4] = Tuple18{v1251, v1240};
                                Union1 v1253;
                                v1253 = Union1{Union1_0{}};
                                v1241[5] = Tuple18{v1253, 1.0f};
                                Union1 v1255;
                                v1255 = sample_discrete_47(v1241, v1116);
                                int v1256;
                                v1256 = sizeof(Union1);
                                unsigned long long v1257;
                                v1257 = (unsigned long long)v1256;
                                bool v1258;
                                v1258 = v1257 <= 98304ull;
                                bool v1259;
                                v1259 = v1258 == false;
                                if (v1259){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v1258);
                                } else {
                                }
                                extern __shared__ unsigned char v1261[];
                                bool v1262;
                                v1262 = v1257 <= v1257;
                                bool v1263;
                                v1263 = v1262 == false;
                                if (v1263){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v1262);
                                } else {
                                }
                                Union1 * v1265;
                                v1265 = reinterpret_cast<Union1 *>(&v1261[0ull]);
                                int v1267;
                                v1267 = threadIdx.x;
                                bool v1268;
                                v1268 = v1267 == 0;
                                if (v1268){
                                    v1265[0] = v1255;
                                } else {
                                }
                                __syncthreads();
                                Union1 v1269;
                                v1269 = v1265[0];
                                __syncthreads();
                                Union6 v1270;
                                v1270 = Union6{Union6_2{v151, v1269}};
                                v5.push(v1270);
                                Union4 v1598;
                                switch (v1269.tag) {
                                    case 0: { // A_All_In
                                        static_array<int,2> v1474;
                                        int v1476;
                                        v1476 = 0;
                                        while (while_method_0(v1476)){
                                            bool v1478;
                                            v1478 = 0 <= v1476;
                                            bool v1480;
                                            if (v1478){
                                                bool v1479;
                                                v1479 = v1476 < 2;
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
                                            v1483 = v149[v1476];
                                            bool v1486;
                                            if (v1478){
                                                bool v1485;
                                                v1485 = v1476 < 2;
                                                v1486 = v1485;
                                            } else {
                                                v1486 = false;
                                            }
                                            bool v1487;
                                            v1487 = v1486 == false;
                                            if (v1487){
                                                assert("Index must be in range." && v1486);
                                            } else {
                                            }
                                            int v1489;
                                            v1489 = v147[v1476];
                                            int v1491;
                                            v1491 = v1483 + v1489;
                                            v1474[v1476] = v1491;
                                            v1476 += 1 ;
                                        }
                                        int v1492;
                                        v1492 = v147[0];
                                        int v1494; int v1495;
                                        Tuple4 tmp49 = Tuple4{1, v1492};
                                        v1494 = tmp49.v0; v1495 = tmp49.v1;
                                        while (while_method_0(v1494)){
                                            bool v1497;
                                            v1497 = 0 <= v1494;
                                            bool v1499;
                                            if (v1497){
                                                bool v1498;
                                                v1498 = v1494 < 2;
                                                v1499 = v1498;
                                            } else {
                                                v1499 = false;
                                            }
                                            bool v1500;
                                            v1500 = v1499 == false;
                                            if (v1500){
                                                assert("Index must be in range." && v1499);
                                            } else {
                                            }
                                            int v1502;
                                            v1502 = v147[v1494];
                                            bool v1504;
                                            v1504 = v1495 >= v1502;
                                            int v1505;
                                            if (v1504){
                                                v1505 = v1495;
                                            } else {
                                                v1505 = v1502;
                                            }
                                            v1495 = v1505;
                                            v1494 += 1 ;
                                        }
                                        bool v1507;
                                        if (v153){
                                            bool v1506;
                                            v1506 = v151 < 2;
                                            v1507 = v1506;
                                        } else {
                                            v1507 = false;
                                        }
                                        bool v1508;
                                        v1508 = v1507 == false;
                                        if (v1508){
                                            assert("Index must be in range." && v1507);
                                        } else {
                                        }
                                        int v1510;
                                        v1510 = v1474[v151];
                                        bool v1512;
                                        v1512 = v1495 < v1510;
                                        int v1513;
                                        if (v1512){
                                            v1513 = v1495;
                                        } else {
                                            v1513 = v1510;
                                        }
                                        static_array<int,2> v1514;
                                        int v1516;
                                        v1516 = 0;
                                        while (while_method_0(v1516)){
                                            bool v1518;
                                            v1518 = 0 <= v1516;
                                            bool v1520;
                                            if (v1518){
                                                bool v1519;
                                                v1519 = v1516 < 2;
                                                v1520 = v1519;
                                            } else {
                                                v1520 = false;
                                            }
                                            bool v1521;
                                            v1521 = v1520 == false;
                                            if (v1521){
                                                assert("Index must be in range." && v1520);
                                            } else {
                                            }
                                            int v1523;
                                            v1523 = v147[v1516];
                                            bool v1525;
                                            v1525 = v151 == v1516;
                                            int v1526;
                                            if (v1525){
                                                v1526 = v1513;
                                            } else {
                                                v1526 = v1523;
                                            }
                                            v1514[v1516] = v1526;
                                            v1516 += 1 ;
                                        }
                                        static_array<int,2> v1527;
                                        int v1529;
                                        v1529 = 0;
                                        while (while_method_0(v1529)){
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
                                            v1536 = v1474[v1529];
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
                                            v1542 = v1514[v1529];
                                            int v1544;
                                            v1544 = v1536 - v1542;
                                            v1527[v1529] = v1544;
                                            v1529 += 1 ;
                                        }
                                        bool v1546;
                                        if (v153){
                                            bool v1545;
                                            v1545 = v151 < 2;
                                            v1546 = v1545;
                                        } else {
                                            v1546 = false;
                                        }
                                        bool v1547;
                                        v1547 = v1546 == false;
                                        if (v1547){
                                            assert("Index must be in range." && v1546);
                                        } else {
                                        }
                                        int v1549;
                                        v1549 = v1527[v151];
                                        int v1551;
                                        v1551 = v1495 + v1549;
                                        bool v1553;
                                        if (v153){
                                            bool v1552;
                                            v1552 = v151 < 2;
                                            v1553 = v1552;
                                        } else {
                                            v1553 = false;
                                        }
                                        bool v1554;
                                        v1554 = v1553 == false;
                                        if (v1554){
                                            assert("Index must be in range." && v1553);
                                        } else {
                                        }
                                        int v1556;
                                        v1556 = v1474[v151];
                                        bool v1558;
                                        v1558 = v1551 < v1556;
                                        int v1559;
                                        if (v1558){
                                            v1559 = v1551;
                                        } else {
                                            v1559 = v1556;
                                        }
                                        static_array<int,2> v1560;
                                        int v1562;
                                        v1562 = 0;
                                        while (while_method_0(v1562)){
                                            bool v1564;
                                            v1564 = 0 <= v1562;
                                            bool v1566;
                                            if (v1564){
                                                bool v1565;
                                                v1565 = v1562 < 2;
                                                v1566 = v1565;
                                            } else {
                                                v1566 = false;
                                            }
                                            bool v1567;
                                            v1567 = v1566 == false;
                                            if (v1567){
                                                assert("Index must be in range." && v1566);
                                            } else {
                                            }
                                            int v1569;
                                            v1569 = v147[v1562];
                                            bool v1571;
                                            v1571 = v151 == v1562;
                                            int v1572;
                                            if (v1571){
                                                v1572 = v1559;
                                            } else {
                                                v1572 = v1569;
                                            }
                                            v1560[v1562] = v1572;
                                            v1562 += 1 ;
                                        }
                                        static_array<int,2> v1573;
                                        int v1575;
                                        v1575 = 0;
                                        while (while_method_0(v1575)){
                                            bool v1577;
                                            v1577 = 0 <= v1575;
                                            bool v1579;
                                            if (v1577){
                                                bool v1578;
                                                v1578 = v1575 < 2;
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
                                            v1582 = v1474[v1575];
                                            bool v1585;
                                            if (v1577){
                                                bool v1584;
                                                v1584 = v1575 < 2;
                                                v1585 = v1584;
                                            } else {
                                                v1585 = false;
                                            }
                                            bool v1586;
                                            v1586 = v1585 == false;
                                            if (v1586){
                                                assert("Index must be in range." && v1585);
                                            } else {
                                            }
                                            int v1588;
                                            v1588 = v1560[v1575];
                                            int v1590;
                                            v1590 = v1582 - v1588;
                                            v1573[v1575] = v1590;
                                            v1575 += 1 ;
                                        }
                                        bool v1591;
                                        v1591 = v1549 >= v145;
                                        int v1592;
                                        if (v1591){
                                            v1592 = v1549;
                                        } else {
                                            v1592 = v145;
                                        }
                                        int v1593;
                                        v1593 = v148 + 1;
                                        v1598 = try_round_36(v1592, v146, v1560, v1593, v1573, v150);
                                        break;
                                    }
                                    case 1: { // A_Call
                                        static_array<int,2> v1272;
                                        int v1274;
                                        v1274 = 0;
                                        while (while_method_0(v1274)){
                                            bool v1276;
                                            v1276 = 0 <= v1274;
                                            bool v1278;
                                            if (v1276){
                                                bool v1277;
                                                v1277 = v1274 < 2;
                                                v1278 = v1277;
                                            } else {
                                                v1278 = false;
                                            }
                                            bool v1279;
                                            v1279 = v1278 == false;
                                            if (v1279){
                                                assert("Index must be in range." && v1278);
                                            } else {
                                            }
                                            int v1281;
                                            v1281 = v149[v1274];
                                            bool v1284;
                                            if (v1276){
                                                bool v1283;
                                                v1283 = v1274 < 2;
                                                v1284 = v1283;
                                            } else {
                                                v1284 = false;
                                            }
                                            bool v1285;
                                            v1285 = v1284 == false;
                                            if (v1285){
                                                assert("Index must be in range." && v1284);
                                            } else {
                                            }
                                            int v1287;
                                            v1287 = v147[v1274];
                                            int v1289;
                                            v1289 = v1281 + v1287;
                                            v1272[v1274] = v1289;
                                            v1274 += 1 ;
                                        }
                                        int v1290;
                                        v1290 = v147[0];
                                        int v1292; int v1293;
                                        Tuple4 tmp50 = Tuple4{1, v1290};
                                        v1292 = tmp50.v0; v1293 = tmp50.v1;
                                        while (while_method_0(v1292)){
                                            bool v1295;
                                            v1295 = 0 <= v1292;
                                            bool v1297;
                                            if (v1295){
                                                bool v1296;
                                                v1296 = v1292 < 2;
                                                v1297 = v1296;
                                            } else {
                                                v1297 = false;
                                            }
                                            bool v1298;
                                            v1298 = v1297 == false;
                                            if (v1298){
                                                assert("Index must be in range." && v1297);
                                            } else {
                                            }
                                            int v1300;
                                            v1300 = v147[v1292];
                                            bool v1302;
                                            v1302 = v1293 >= v1300;
                                            int v1303;
                                            if (v1302){
                                                v1303 = v1293;
                                            } else {
                                                v1303 = v1300;
                                            }
                                            v1293 = v1303;
                                            v1292 += 1 ;
                                        }
                                        bool v1305;
                                        if (v153){
                                            bool v1304;
                                            v1304 = v151 < 2;
                                            v1305 = v1304;
                                        } else {
                                            v1305 = false;
                                        }
                                        bool v1306;
                                        v1306 = v1305 == false;
                                        if (v1306){
                                            assert("Index must be in range." && v1305);
                                        } else {
                                        }
                                        int v1308;
                                        v1308 = v1272[v151];
                                        bool v1310;
                                        v1310 = v1293 < v1308;
                                        int v1311;
                                        if (v1310){
                                            v1311 = v1293;
                                        } else {
                                            v1311 = v1308;
                                        }
                                        static_array<int,2> v1312;
                                        int v1314;
                                        v1314 = 0;
                                        while (while_method_0(v1314)){
                                            bool v1316;
                                            v1316 = 0 <= v1314;
                                            bool v1318;
                                            if (v1316){
                                                bool v1317;
                                                v1317 = v1314 < 2;
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
                                            int v1321;
                                            v1321 = v147[v1314];
                                            bool v1323;
                                            v1323 = v151 == v1314;
                                            int v1324;
                                            if (v1323){
                                                v1324 = v1311;
                                            } else {
                                                v1324 = v1321;
                                            }
                                            v1312[v1314] = v1324;
                                            v1314 += 1 ;
                                        }
                                        static_array<int,2> v1325;
                                        int v1327;
                                        v1327 = 0;
                                        while (while_method_0(v1327)){
                                            bool v1329;
                                            v1329 = 0 <= v1327;
                                            bool v1331;
                                            if (v1329){
                                                bool v1330;
                                                v1330 = v1327 < 2;
                                                v1331 = v1330;
                                            } else {
                                                v1331 = false;
                                            }
                                            bool v1332;
                                            v1332 = v1331 == false;
                                            if (v1332){
                                                assert("Index must be in range." && v1331);
                                            } else {
                                            }
                                            int v1334;
                                            v1334 = v1272[v1327];
                                            bool v1337;
                                            if (v1329){
                                                bool v1336;
                                                v1336 = v1327 < 2;
                                                v1337 = v1336;
                                            } else {
                                                v1337 = false;
                                            }
                                            bool v1338;
                                            v1338 = v1337 == false;
                                            if (v1338){
                                                assert("Index must be in range." && v1337);
                                            } else {
                                            }
                                            int v1340;
                                            v1340 = v1312[v1327];
                                            int v1342;
                                            v1342 = v1334 - v1340;
                                            v1325[v1327] = v1342;
                                            v1327 += 1 ;
                                        }
                                        bool v1343;
                                        v1343 = v151 < 2;
                                        if (v1343){
                                            int v1344;
                                            v1344 = v148 + 1;
                                            v1598 = try_round_36(v145, v146, v1312, v1344, v1325, v150);
                                        } else {
                                            v1598 = go_next_street_38(v145, v146, v1312, v148, v1325, v150);
                                        }
                                        break;
                                    }
                                    case 2: { // A_Fold
                                        v1598 = Union4{Union4_1{v145, v146, v147, v148, v149, v150}};
                                        break;
                                    }
                                    case 3: { // A_Raise
                                        int v1348 = v1269.case3.v0;
                                        bool v1349;
                                        v1349 = v145 <= v1348;
                                        bool v1350;
                                        v1350 = v1349 == false;
                                        if (v1350){
                                            assert("The raise amount must match the minimum." && v1349);
                                        } else {
                                        }
                                        static_array<int,2> v1352;
                                        int v1354;
                                        v1354 = 0;
                                        while (while_method_0(v1354)){
                                            bool v1356;
                                            v1356 = 0 <= v1354;
                                            bool v1358;
                                            if (v1356){
                                                bool v1357;
                                                v1357 = v1354 < 2;
                                                v1358 = v1357;
                                            } else {
                                                v1358 = false;
                                            }
                                            bool v1359;
                                            v1359 = v1358 == false;
                                            if (v1359){
                                                assert("Index must be in range." && v1358);
                                            } else {
                                            }
                                            int v1361;
                                            v1361 = v149[v1354];
                                            bool v1364;
                                            if (v1356){
                                                bool v1363;
                                                v1363 = v1354 < 2;
                                                v1364 = v1363;
                                            } else {
                                                v1364 = false;
                                            }
                                            bool v1365;
                                            v1365 = v1364 == false;
                                            if (v1365){
                                                assert("Index must be in range." && v1364);
                                            } else {
                                            }
                                            int v1367;
                                            v1367 = v147[v1354];
                                            int v1369;
                                            v1369 = v1361 + v1367;
                                            v1352[v1354] = v1369;
                                            v1354 += 1 ;
                                        }
                                        int v1370;
                                        v1370 = v147[0];
                                        int v1372; int v1373;
                                        Tuple4 tmp51 = Tuple4{1, v1370};
                                        v1372 = tmp51.v0; v1373 = tmp51.v1;
                                        while (while_method_0(v1372)){
                                            bool v1375;
                                            v1375 = 0 <= v1372;
                                            bool v1377;
                                            if (v1375){
                                                bool v1376;
                                                v1376 = v1372 < 2;
                                                v1377 = v1376;
                                            } else {
                                                v1377 = false;
                                            }
                                            bool v1378;
                                            v1378 = v1377 == false;
                                            if (v1378){
                                                assert("Index must be in range." && v1377);
                                            } else {
                                            }
                                            int v1380;
                                            v1380 = v147[v1372];
                                            bool v1382;
                                            v1382 = v1373 >= v1380;
                                            int v1383;
                                            if (v1382){
                                                v1383 = v1373;
                                            } else {
                                                v1383 = v1380;
                                            }
                                            v1373 = v1383;
                                            v1372 += 1 ;
                                        }
                                        bool v1385;
                                        if (v153){
                                            bool v1384;
                                            v1384 = v151 < 2;
                                            v1385 = v1384;
                                        } else {
                                            v1385 = false;
                                        }
                                        bool v1386;
                                        v1386 = v1385 == false;
                                        if (v1386){
                                            assert("Index must be in range." && v1385);
                                        } else {
                                        }
                                        int v1388;
                                        v1388 = v1352[v151];
                                        bool v1390;
                                        v1390 = v1373 < v1388;
                                        int v1391;
                                        if (v1390){
                                            v1391 = v1373;
                                        } else {
                                            v1391 = v1388;
                                        }
                                        static_array<int,2> v1392;
                                        int v1394;
                                        v1394 = 0;
                                        while (while_method_0(v1394)){
                                            bool v1396;
                                            v1396 = 0 <= v1394;
                                            bool v1398;
                                            if (v1396){
                                                bool v1397;
                                                v1397 = v1394 < 2;
                                                v1398 = v1397;
                                            } else {
                                                v1398 = false;
                                            }
                                            bool v1399;
                                            v1399 = v1398 == false;
                                            if (v1399){
                                                assert("Index must be in range." && v1398);
                                            } else {
                                            }
                                            int v1401;
                                            v1401 = v147[v1394];
                                            bool v1403;
                                            v1403 = v151 == v1394;
                                            int v1404;
                                            if (v1403){
                                                v1404 = v1391;
                                            } else {
                                                v1404 = v1401;
                                            }
                                            v1392[v1394] = v1404;
                                            v1394 += 1 ;
                                        }
                                        static_array<int,2> v1405;
                                        int v1407;
                                        v1407 = 0;
                                        while (while_method_0(v1407)){
                                            bool v1409;
                                            v1409 = 0 <= v1407;
                                            bool v1411;
                                            if (v1409){
                                                bool v1410;
                                                v1410 = v1407 < 2;
                                                v1411 = v1410;
                                            } else {
                                                v1411 = false;
                                            }
                                            bool v1412;
                                            v1412 = v1411 == false;
                                            if (v1412){
                                                assert("Index must be in range." && v1411);
                                            } else {
                                            }
                                            int v1414;
                                            v1414 = v1352[v1407];
                                            bool v1417;
                                            if (v1409){
                                                bool v1416;
                                                v1416 = v1407 < 2;
                                                v1417 = v1416;
                                            } else {
                                                v1417 = false;
                                            }
                                            bool v1418;
                                            v1418 = v1417 == false;
                                            if (v1418){
                                                assert("Index must be in range." && v1417);
                                            } else {
                                            }
                                            int v1420;
                                            v1420 = v1392[v1407];
                                            int v1422;
                                            v1422 = v1414 - v1420;
                                            v1405[v1407] = v1422;
                                            v1407 += 1 ;
                                        }
                                        bool v1424;
                                        if (v153){
                                            bool v1423;
                                            v1423 = v151 < 2;
                                            v1424 = v1423;
                                        } else {
                                            v1424 = false;
                                        }
                                        bool v1425;
                                        v1425 = v1424 == false;
                                        if (v1425){
                                            assert("Index must be in range." && v1424);
                                        } else {
                                        }
                                        int v1427;
                                        v1427 = v1405[v151];
                                        bool v1429;
                                        v1429 = v1348 < v1427;
                                        bool v1430;
                                        v1430 = v1429 == false;
                                        if (v1430){
                                            assert("The raise amount must be less than the stack size after calling." && v1429);
                                        } else {
                                        }
                                        int v1432;
                                        v1432 = v1373 + v1348;
                                        bool v1434;
                                        if (v153){
                                            bool v1433;
                                            v1433 = v151 < 2;
                                            v1434 = v1433;
                                        } else {
                                            v1434 = false;
                                        }
                                        bool v1435;
                                        v1435 = v1434 == false;
                                        if (v1435){
                                            assert("Index must be in range." && v1434);
                                        } else {
                                        }
                                        int v1437;
                                        v1437 = v1352[v151];
                                        bool v1439;
                                        v1439 = v1432 < v1437;
                                        int v1440;
                                        if (v1439){
                                            v1440 = v1432;
                                        } else {
                                            v1440 = v1437;
                                        }
                                        static_array<int,2> v1441;
                                        int v1443;
                                        v1443 = 0;
                                        while (while_method_0(v1443)){
                                            bool v1445;
                                            v1445 = 0 <= v1443;
                                            bool v1447;
                                            if (v1445){
                                                bool v1446;
                                                v1446 = v1443 < 2;
                                                v1447 = v1446;
                                            } else {
                                                v1447 = false;
                                            }
                                            bool v1448;
                                            v1448 = v1447 == false;
                                            if (v1448){
                                                assert("Index must be in range." && v1447);
                                            } else {
                                            }
                                            int v1450;
                                            v1450 = v147[v1443];
                                            bool v1452;
                                            v1452 = v151 == v1443;
                                            int v1453;
                                            if (v1452){
                                                v1453 = v1440;
                                            } else {
                                                v1453 = v1450;
                                            }
                                            v1441[v1443] = v1453;
                                            v1443 += 1 ;
                                        }
                                        static_array<int,2> v1454;
                                        int v1456;
                                        v1456 = 0;
                                        while (while_method_0(v1456)){
                                            bool v1458;
                                            v1458 = 0 <= v1456;
                                            bool v1460;
                                            if (v1458){
                                                bool v1459;
                                                v1459 = v1456 < 2;
                                                v1460 = v1459;
                                            } else {
                                                v1460 = false;
                                            }
                                            bool v1461;
                                            v1461 = v1460 == false;
                                            if (v1461){
                                                assert("Index must be in range." && v1460);
                                            } else {
                                            }
                                            int v1463;
                                            v1463 = v1352[v1456];
                                            bool v1466;
                                            if (v1458){
                                                bool v1465;
                                                v1465 = v1456 < 2;
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
                                            v1469 = v1441[v1456];
                                            int v1471;
                                            v1471 = v1463 - v1469;
                                            v1454[v1456] = v1471;
                                            v1456 += 1 ;
                                        }
                                        int v1472;
                                        v1472 = v148 + 1;
                                        v1598 = try_round_36(v1348, v146, v1441, v1472, v1454, v150);
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                v2166 = Union3{Union3_1{v1598}};
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        break;
                    }
                    case 5: { // G_Round'
                        int v1683 = v10.case5.v0; static_array<static_array<unsigned char,2>,2> v1684 = v10.case5.v1; static_array<int,2> v1685 = v10.case5.v2; int v1686 = v10.case5.v3; static_array<int,2> v1687 = v10.case5.v4; Union5 v1688 = v10.case5.v5; Union1 v1689 = v10.case5.v6;
                        int v1690;
                        v1690 = v1686 % 2;
                        Union6 v1691;
                        v1691 = Union6{Union6_2{v1690, v1689}};
                        v5.push(v1691);
                        Union4 v2022;
                        switch (v1689.tag) {
                            case 0: { // A_All_In
                                static_array<int,2> v1897;
                                int v1899;
                                v1899 = 0;
                                while (while_method_0(v1899)){
                                    bool v1901;
                                    v1901 = 0 <= v1899;
                                    bool v1903;
                                    if (v1901){
                                        bool v1902;
                                        v1902 = v1899 < 2;
                                        v1903 = v1902;
                                    } else {
                                        v1903 = false;
                                    }
                                    bool v1904;
                                    v1904 = v1903 == false;
                                    if (v1904){
                                        assert("Index must be in range." && v1903);
                                    } else {
                                    }
                                    int v1906;
                                    v1906 = v1687[v1899];
                                    bool v1909;
                                    if (v1901){
                                        bool v1908;
                                        v1908 = v1899 < 2;
                                        v1909 = v1908;
                                    } else {
                                        v1909 = false;
                                    }
                                    bool v1910;
                                    v1910 = v1909 == false;
                                    if (v1910){
                                        assert("Index must be in range." && v1909);
                                    } else {
                                    }
                                    int v1912;
                                    v1912 = v1685[v1899];
                                    int v1914;
                                    v1914 = v1906 + v1912;
                                    v1897[v1899] = v1914;
                                    v1899 += 1 ;
                                }
                                int v1915;
                                v1915 = v1685[0];
                                int v1917; int v1918;
                                Tuple4 tmp52 = Tuple4{1, v1915};
                                v1917 = tmp52.v0; v1918 = tmp52.v1;
                                while (while_method_0(v1917)){
                                    bool v1920;
                                    v1920 = 0 <= v1917;
                                    bool v1922;
                                    if (v1920){
                                        bool v1921;
                                        v1921 = v1917 < 2;
                                        v1922 = v1921;
                                    } else {
                                        v1922 = false;
                                    }
                                    bool v1923;
                                    v1923 = v1922 == false;
                                    if (v1923){
                                        assert("Index must be in range." && v1922);
                                    } else {
                                    }
                                    int v1925;
                                    v1925 = v1685[v1917];
                                    bool v1927;
                                    v1927 = v1918 >= v1925;
                                    int v1928;
                                    if (v1927){
                                        v1928 = v1918;
                                    } else {
                                        v1928 = v1925;
                                    }
                                    v1918 = v1928;
                                    v1917 += 1 ;
                                }
                                bool v1929;
                                v1929 = 0 <= v1690;
                                bool v1931;
                                if (v1929){
                                    bool v1930;
                                    v1930 = v1690 < 2;
                                    v1931 = v1930;
                                } else {
                                    v1931 = false;
                                }
                                bool v1932;
                                v1932 = v1931 == false;
                                if (v1932){
                                    assert("Index must be in range." && v1931);
                                } else {
                                }
                                int v1934;
                                v1934 = v1897[v1690];
                                bool v1936;
                                v1936 = v1918 < v1934;
                                int v1937;
                                if (v1936){
                                    v1937 = v1918;
                                } else {
                                    v1937 = v1934;
                                }
                                static_array<int,2> v1938;
                                int v1940;
                                v1940 = 0;
                                while (while_method_0(v1940)){
                                    bool v1942;
                                    v1942 = 0 <= v1940;
                                    bool v1944;
                                    if (v1942){
                                        bool v1943;
                                        v1943 = v1940 < 2;
                                        v1944 = v1943;
                                    } else {
                                        v1944 = false;
                                    }
                                    bool v1945;
                                    v1945 = v1944 == false;
                                    if (v1945){
                                        assert("Index must be in range." && v1944);
                                    } else {
                                    }
                                    int v1947;
                                    v1947 = v1685[v1940];
                                    bool v1949;
                                    v1949 = v1690 == v1940;
                                    int v1950;
                                    if (v1949){
                                        v1950 = v1937;
                                    } else {
                                        v1950 = v1947;
                                    }
                                    v1938[v1940] = v1950;
                                    v1940 += 1 ;
                                }
                                static_array<int,2> v1951;
                                int v1953;
                                v1953 = 0;
                                while (while_method_0(v1953)){
                                    bool v1955;
                                    v1955 = 0 <= v1953;
                                    bool v1957;
                                    if (v1955){
                                        bool v1956;
                                        v1956 = v1953 < 2;
                                        v1957 = v1956;
                                    } else {
                                        v1957 = false;
                                    }
                                    bool v1958;
                                    v1958 = v1957 == false;
                                    if (v1958){
                                        assert("Index must be in range." && v1957);
                                    } else {
                                    }
                                    int v1960;
                                    v1960 = v1897[v1953];
                                    bool v1963;
                                    if (v1955){
                                        bool v1962;
                                        v1962 = v1953 < 2;
                                        v1963 = v1962;
                                    } else {
                                        v1963 = false;
                                    }
                                    bool v1964;
                                    v1964 = v1963 == false;
                                    if (v1964){
                                        assert("Index must be in range." && v1963);
                                    } else {
                                    }
                                    int v1966;
                                    v1966 = v1938[v1953];
                                    int v1968;
                                    v1968 = v1960 - v1966;
                                    v1951[v1953] = v1968;
                                    v1953 += 1 ;
                                }
                                bool v1970;
                                if (v1929){
                                    bool v1969;
                                    v1969 = v1690 < 2;
                                    v1970 = v1969;
                                } else {
                                    v1970 = false;
                                }
                                bool v1971;
                                v1971 = v1970 == false;
                                if (v1971){
                                    assert("Index must be in range." && v1970);
                                } else {
                                }
                                int v1973;
                                v1973 = v1951[v1690];
                                int v1975;
                                v1975 = v1918 + v1973;
                                bool v1977;
                                if (v1929){
                                    bool v1976;
                                    v1976 = v1690 < 2;
                                    v1977 = v1976;
                                } else {
                                    v1977 = false;
                                }
                                bool v1978;
                                v1978 = v1977 == false;
                                if (v1978){
                                    assert("Index must be in range." && v1977);
                                } else {
                                }
                                int v1980;
                                v1980 = v1897[v1690];
                                bool v1982;
                                v1982 = v1975 < v1980;
                                int v1983;
                                if (v1982){
                                    v1983 = v1975;
                                } else {
                                    v1983 = v1980;
                                }
                                static_array<int,2> v1984;
                                int v1986;
                                v1986 = 0;
                                while (while_method_0(v1986)){
                                    bool v1988;
                                    v1988 = 0 <= v1986;
                                    bool v1990;
                                    if (v1988){
                                        bool v1989;
                                        v1989 = v1986 < 2;
                                        v1990 = v1989;
                                    } else {
                                        v1990 = false;
                                    }
                                    bool v1991;
                                    v1991 = v1990 == false;
                                    if (v1991){
                                        assert("Index must be in range." && v1990);
                                    } else {
                                    }
                                    int v1993;
                                    v1993 = v1685[v1986];
                                    bool v1995;
                                    v1995 = v1690 == v1986;
                                    int v1996;
                                    if (v1995){
                                        v1996 = v1983;
                                    } else {
                                        v1996 = v1993;
                                    }
                                    v1984[v1986] = v1996;
                                    v1986 += 1 ;
                                }
                                static_array<int,2> v1997;
                                int v1999;
                                v1999 = 0;
                                while (while_method_0(v1999)){
                                    bool v2001;
                                    v2001 = 0 <= v1999;
                                    bool v2003;
                                    if (v2001){
                                        bool v2002;
                                        v2002 = v1999 < 2;
                                        v2003 = v2002;
                                    } else {
                                        v2003 = false;
                                    }
                                    bool v2004;
                                    v2004 = v2003 == false;
                                    if (v2004){
                                        assert("Index must be in range." && v2003);
                                    } else {
                                    }
                                    int v2006;
                                    v2006 = v1897[v1999];
                                    bool v2009;
                                    if (v2001){
                                        bool v2008;
                                        v2008 = v1999 < 2;
                                        v2009 = v2008;
                                    } else {
                                        v2009 = false;
                                    }
                                    bool v2010;
                                    v2010 = v2009 == false;
                                    if (v2010){
                                        assert("Index must be in range." && v2009);
                                    } else {
                                    }
                                    int v2012;
                                    v2012 = v1984[v1999];
                                    int v2014;
                                    v2014 = v2006 - v2012;
                                    v1997[v1999] = v2014;
                                    v1999 += 1 ;
                                }
                                bool v2015;
                                v2015 = v1973 >= v1683;
                                int v2016;
                                if (v2015){
                                    v2016 = v1973;
                                } else {
                                    v2016 = v1683;
                                }
                                int v2017;
                                v2017 = v1686 + 1;
                                v2022 = try_round_36(v2016, v1684, v1984, v2017, v1997, v1688);
                                break;
                            }
                            case 1: { // A_Call
                                static_array<int,2> v1693;
                                int v1695;
                                v1695 = 0;
                                while (while_method_0(v1695)){
                                    bool v1697;
                                    v1697 = 0 <= v1695;
                                    bool v1699;
                                    if (v1697){
                                        bool v1698;
                                        v1698 = v1695 < 2;
                                        v1699 = v1698;
                                    } else {
                                        v1699 = false;
                                    }
                                    bool v1700;
                                    v1700 = v1699 == false;
                                    if (v1700){
                                        assert("Index must be in range." && v1699);
                                    } else {
                                    }
                                    int v1702;
                                    v1702 = v1687[v1695];
                                    bool v1705;
                                    if (v1697){
                                        bool v1704;
                                        v1704 = v1695 < 2;
                                        v1705 = v1704;
                                    } else {
                                        v1705 = false;
                                    }
                                    bool v1706;
                                    v1706 = v1705 == false;
                                    if (v1706){
                                        assert("Index must be in range." && v1705);
                                    } else {
                                    }
                                    int v1708;
                                    v1708 = v1685[v1695];
                                    int v1710;
                                    v1710 = v1702 + v1708;
                                    v1693[v1695] = v1710;
                                    v1695 += 1 ;
                                }
                                int v1711;
                                v1711 = v1685[0];
                                int v1713; int v1714;
                                Tuple4 tmp53 = Tuple4{1, v1711};
                                v1713 = tmp53.v0; v1714 = tmp53.v1;
                                while (while_method_0(v1713)){
                                    bool v1716;
                                    v1716 = 0 <= v1713;
                                    bool v1718;
                                    if (v1716){
                                        bool v1717;
                                        v1717 = v1713 < 2;
                                        v1718 = v1717;
                                    } else {
                                        v1718 = false;
                                    }
                                    bool v1719;
                                    v1719 = v1718 == false;
                                    if (v1719){
                                        assert("Index must be in range." && v1718);
                                    } else {
                                    }
                                    int v1721;
                                    v1721 = v1685[v1713];
                                    bool v1723;
                                    v1723 = v1714 >= v1721;
                                    int v1724;
                                    if (v1723){
                                        v1724 = v1714;
                                    } else {
                                        v1724 = v1721;
                                    }
                                    v1714 = v1724;
                                    v1713 += 1 ;
                                }
                                bool v1725;
                                v1725 = 0 <= v1690;
                                bool v1727;
                                if (v1725){
                                    bool v1726;
                                    v1726 = v1690 < 2;
                                    v1727 = v1726;
                                } else {
                                    v1727 = false;
                                }
                                bool v1728;
                                v1728 = v1727 == false;
                                if (v1728){
                                    assert("Index must be in range." && v1727);
                                } else {
                                }
                                int v1730;
                                v1730 = v1693[v1690];
                                bool v1732;
                                v1732 = v1714 < v1730;
                                int v1733;
                                if (v1732){
                                    v1733 = v1714;
                                } else {
                                    v1733 = v1730;
                                }
                                static_array<int,2> v1734;
                                int v1736;
                                v1736 = 0;
                                while (while_method_0(v1736)){
                                    bool v1738;
                                    v1738 = 0 <= v1736;
                                    bool v1740;
                                    if (v1738){
                                        bool v1739;
                                        v1739 = v1736 < 2;
                                        v1740 = v1739;
                                    } else {
                                        v1740 = false;
                                    }
                                    bool v1741;
                                    v1741 = v1740 == false;
                                    if (v1741){
                                        assert("Index must be in range." && v1740);
                                    } else {
                                    }
                                    int v1743;
                                    v1743 = v1685[v1736];
                                    bool v1745;
                                    v1745 = v1690 == v1736;
                                    int v1746;
                                    if (v1745){
                                        v1746 = v1733;
                                    } else {
                                        v1746 = v1743;
                                    }
                                    v1734[v1736] = v1746;
                                    v1736 += 1 ;
                                }
                                static_array<int,2> v1747;
                                int v1749;
                                v1749 = 0;
                                while (while_method_0(v1749)){
                                    bool v1751;
                                    v1751 = 0 <= v1749;
                                    bool v1753;
                                    if (v1751){
                                        bool v1752;
                                        v1752 = v1749 < 2;
                                        v1753 = v1752;
                                    } else {
                                        v1753 = false;
                                    }
                                    bool v1754;
                                    v1754 = v1753 == false;
                                    if (v1754){
                                        assert("Index must be in range." && v1753);
                                    } else {
                                    }
                                    int v1756;
                                    v1756 = v1693[v1749];
                                    bool v1759;
                                    if (v1751){
                                        bool v1758;
                                        v1758 = v1749 < 2;
                                        v1759 = v1758;
                                    } else {
                                        v1759 = false;
                                    }
                                    bool v1760;
                                    v1760 = v1759 == false;
                                    if (v1760){
                                        assert("Index must be in range." && v1759);
                                    } else {
                                    }
                                    int v1762;
                                    v1762 = v1734[v1749];
                                    int v1764;
                                    v1764 = v1756 - v1762;
                                    v1747[v1749] = v1764;
                                    v1749 += 1 ;
                                }
                                bool v1765;
                                v1765 = v1690 < 2;
                                if (v1765){
                                    int v1766;
                                    v1766 = v1686 + 1;
                                    v2022 = try_round_36(v1683, v1684, v1734, v1766, v1747, v1688);
                                } else {
                                    v2022 = go_next_street_38(v1683, v1684, v1734, v1686, v1747, v1688);
                                }
                                break;
                            }
                            case 2: { // A_Fold
                                v2022 = Union4{Union4_1{v1683, v1684, v1685, v1686, v1687, v1688}};
                                break;
                            }
                            case 3: { // A_Raise
                                int v1770 = v1689.case3.v0;
                                bool v1771;
                                v1771 = v1683 <= v1770;
                                bool v1772;
                                v1772 = v1771 == false;
                                if (v1772){
                                    assert("The raise amount must match the minimum." && v1771);
                                } else {
                                }
                                static_array<int,2> v1774;
                                int v1776;
                                v1776 = 0;
                                while (while_method_0(v1776)){
                                    bool v1778;
                                    v1778 = 0 <= v1776;
                                    bool v1780;
                                    if (v1778){
                                        bool v1779;
                                        v1779 = v1776 < 2;
                                        v1780 = v1779;
                                    } else {
                                        v1780 = false;
                                    }
                                    bool v1781;
                                    v1781 = v1780 == false;
                                    if (v1781){
                                        assert("Index must be in range." && v1780);
                                    } else {
                                    }
                                    int v1783;
                                    v1783 = v1687[v1776];
                                    bool v1786;
                                    if (v1778){
                                        bool v1785;
                                        v1785 = v1776 < 2;
                                        v1786 = v1785;
                                    } else {
                                        v1786 = false;
                                    }
                                    bool v1787;
                                    v1787 = v1786 == false;
                                    if (v1787){
                                        assert("Index must be in range." && v1786);
                                    } else {
                                    }
                                    int v1789;
                                    v1789 = v1685[v1776];
                                    int v1791;
                                    v1791 = v1783 + v1789;
                                    v1774[v1776] = v1791;
                                    v1776 += 1 ;
                                }
                                int v1792;
                                v1792 = v1685[0];
                                int v1794; int v1795;
                                Tuple4 tmp54 = Tuple4{1, v1792};
                                v1794 = tmp54.v0; v1795 = tmp54.v1;
                                while (while_method_0(v1794)){
                                    bool v1797;
                                    v1797 = 0 <= v1794;
                                    bool v1799;
                                    if (v1797){
                                        bool v1798;
                                        v1798 = v1794 < 2;
                                        v1799 = v1798;
                                    } else {
                                        v1799 = false;
                                    }
                                    bool v1800;
                                    v1800 = v1799 == false;
                                    if (v1800){
                                        assert("Index must be in range." && v1799);
                                    } else {
                                    }
                                    int v1802;
                                    v1802 = v1685[v1794];
                                    bool v1804;
                                    v1804 = v1795 >= v1802;
                                    int v1805;
                                    if (v1804){
                                        v1805 = v1795;
                                    } else {
                                        v1805 = v1802;
                                    }
                                    v1795 = v1805;
                                    v1794 += 1 ;
                                }
                                bool v1806;
                                v1806 = 0 <= v1690;
                                bool v1808;
                                if (v1806){
                                    bool v1807;
                                    v1807 = v1690 < 2;
                                    v1808 = v1807;
                                } else {
                                    v1808 = false;
                                }
                                bool v1809;
                                v1809 = v1808 == false;
                                if (v1809){
                                    assert("Index must be in range." && v1808);
                                } else {
                                }
                                int v1811;
                                v1811 = v1774[v1690];
                                bool v1813;
                                v1813 = v1795 < v1811;
                                int v1814;
                                if (v1813){
                                    v1814 = v1795;
                                } else {
                                    v1814 = v1811;
                                }
                                static_array<int,2> v1815;
                                int v1817;
                                v1817 = 0;
                                while (while_method_0(v1817)){
                                    bool v1819;
                                    v1819 = 0 <= v1817;
                                    bool v1821;
                                    if (v1819){
                                        bool v1820;
                                        v1820 = v1817 < 2;
                                        v1821 = v1820;
                                    } else {
                                        v1821 = false;
                                    }
                                    bool v1822;
                                    v1822 = v1821 == false;
                                    if (v1822){
                                        assert("Index must be in range." && v1821);
                                    } else {
                                    }
                                    int v1824;
                                    v1824 = v1685[v1817];
                                    bool v1826;
                                    v1826 = v1690 == v1817;
                                    int v1827;
                                    if (v1826){
                                        v1827 = v1814;
                                    } else {
                                        v1827 = v1824;
                                    }
                                    v1815[v1817] = v1827;
                                    v1817 += 1 ;
                                }
                                static_array<int,2> v1828;
                                int v1830;
                                v1830 = 0;
                                while (while_method_0(v1830)){
                                    bool v1832;
                                    v1832 = 0 <= v1830;
                                    bool v1834;
                                    if (v1832){
                                        bool v1833;
                                        v1833 = v1830 < 2;
                                        v1834 = v1833;
                                    } else {
                                        v1834 = false;
                                    }
                                    bool v1835;
                                    v1835 = v1834 == false;
                                    if (v1835){
                                        assert("Index must be in range." && v1834);
                                    } else {
                                    }
                                    int v1837;
                                    v1837 = v1774[v1830];
                                    bool v1840;
                                    if (v1832){
                                        bool v1839;
                                        v1839 = v1830 < 2;
                                        v1840 = v1839;
                                    } else {
                                        v1840 = false;
                                    }
                                    bool v1841;
                                    v1841 = v1840 == false;
                                    if (v1841){
                                        assert("Index must be in range." && v1840);
                                    } else {
                                    }
                                    int v1843;
                                    v1843 = v1815[v1830];
                                    int v1845;
                                    v1845 = v1837 - v1843;
                                    v1828[v1830] = v1845;
                                    v1830 += 1 ;
                                }
                                bool v1847;
                                if (v1806){
                                    bool v1846;
                                    v1846 = v1690 < 2;
                                    v1847 = v1846;
                                } else {
                                    v1847 = false;
                                }
                                bool v1848;
                                v1848 = v1847 == false;
                                if (v1848){
                                    assert("Index must be in range." && v1847);
                                } else {
                                }
                                int v1850;
                                v1850 = v1828[v1690];
                                bool v1852;
                                v1852 = v1770 < v1850;
                                bool v1853;
                                v1853 = v1852 == false;
                                if (v1853){
                                    assert("The raise amount must be less than the stack size after calling." && v1852);
                                } else {
                                }
                                int v1855;
                                v1855 = v1795 + v1770;
                                bool v1857;
                                if (v1806){
                                    bool v1856;
                                    v1856 = v1690 < 2;
                                    v1857 = v1856;
                                } else {
                                    v1857 = false;
                                }
                                bool v1858;
                                v1858 = v1857 == false;
                                if (v1858){
                                    assert("Index must be in range." && v1857);
                                } else {
                                }
                                int v1860;
                                v1860 = v1774[v1690];
                                bool v1862;
                                v1862 = v1855 < v1860;
                                int v1863;
                                if (v1862){
                                    v1863 = v1855;
                                } else {
                                    v1863 = v1860;
                                }
                                static_array<int,2> v1864;
                                int v1866;
                                v1866 = 0;
                                while (while_method_0(v1866)){
                                    bool v1868;
                                    v1868 = 0 <= v1866;
                                    bool v1870;
                                    if (v1868){
                                        bool v1869;
                                        v1869 = v1866 < 2;
                                        v1870 = v1869;
                                    } else {
                                        v1870 = false;
                                    }
                                    bool v1871;
                                    v1871 = v1870 == false;
                                    if (v1871){
                                        assert("Index must be in range." && v1870);
                                    } else {
                                    }
                                    int v1873;
                                    v1873 = v1685[v1866];
                                    bool v1875;
                                    v1875 = v1690 == v1866;
                                    int v1876;
                                    if (v1875){
                                        v1876 = v1863;
                                    } else {
                                        v1876 = v1873;
                                    }
                                    v1864[v1866] = v1876;
                                    v1866 += 1 ;
                                }
                                static_array<int,2> v1877;
                                int v1879;
                                v1879 = 0;
                                while (while_method_0(v1879)){
                                    bool v1881;
                                    v1881 = 0 <= v1879;
                                    bool v1883;
                                    if (v1881){
                                        bool v1882;
                                        v1882 = v1879 < 2;
                                        v1883 = v1882;
                                    } else {
                                        v1883 = false;
                                    }
                                    bool v1884;
                                    v1884 = v1883 == false;
                                    if (v1884){
                                        assert("Index must be in range." && v1883);
                                    } else {
                                    }
                                    int v1886;
                                    v1886 = v1774[v1879];
                                    bool v1889;
                                    if (v1881){
                                        bool v1888;
                                        v1888 = v1879 < 2;
                                        v1889 = v1888;
                                    } else {
                                        v1889 = false;
                                    }
                                    bool v1890;
                                    v1890 = v1889 == false;
                                    if (v1890){
                                        assert("Index must be in range." && v1889);
                                    } else {
                                    }
                                    int v1892;
                                    v1892 = v1864[v1879];
                                    int v1894;
                                    v1894 = v1886 - v1892;
                                    v1877[v1879] = v1894;
                                    v1879 += 1 ;
                                }
                                int v1895;
                                v1895 = v1686 + 1;
                                v2022 = try_round_36(v1770, v1684, v1864, v1895, v1877, v1688);
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        v2166 = Union3{Union3_1{v2022}};
                        break;
                    }
                    case 6: { // G_Showdown
                        int v31 = v10.case6.v0; static_array<static_array<unsigned char,2>,2> v32 = v10.case6.v1; static_array<int,2> v33 = v10.case6.v2; int v34 = v10.case6.v3; static_array<int,2> v35 = v10.case6.v4; Union5 v36 = v10.case6.v5;
                        static_array<unsigned char,5> v39;
                        switch (v36.tag) {
                            case 2: { // River
                                static_array<unsigned char,5> v37 = v36.case2.v0;
                                v39 = v37;
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in showdown.");
                                __trap();
                            }
                        }
                        static_array<unsigned char,2> v40;
                        v40 = v32[0];
                        static_array<unsigned char,7> v42;
                        int v44;
                        v44 = 0;
                        while (while_method_0(v44)){
                            bool v46;
                            v46 = 0 <= v44;
                            bool v48;
                            if (v46){
                                bool v47;
                                v47 = v44 < 2;
                                v48 = v47;
                            } else {
                                v48 = false;
                            }
                            bool v49;
                            v49 = v48 == false;
                            if (v49){
                                assert("Index must be in range." && v48);
                            } else {
                            }
                            unsigned char v51;
                            v51 = v40[v44];
                            v42[v44] = v51;
                            v44 += 1 ;
                        }
                        int v53;
                        v53 = 0;
                        while (while_method_2(v53)){
                            bool v55;
                            v55 = 0 <= v53;
                            bool v57;
                            if (v55){
                                bool v56;
                                v56 = v53 < 5;
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
                            v60 = v39[v53];
                            int v62;
                            v62 = 2 + v53;
                            v42[v62] = v60;
                            v53 += 1 ;
                        }
                        static_array<unsigned char,5> v63; char v64;
                        Tuple0 tmp79 = score_51(v42);
                        v63 = tmp79.v0; v64 = tmp79.v1;
                        static_array<unsigned char,2> v65;
                        v65 = v32[1];
                        static_array<unsigned char,7> v67;
                        int v69;
                        v69 = 0;
                        while (while_method_0(v69)){
                            bool v71;
                            v71 = 0 <= v69;
                            bool v73;
                            if (v71){
                                bool v72;
                                v72 = v69 < 2;
                                v73 = v72;
                            } else {
                                v73 = false;
                            }
                            bool v74;
                            v74 = v73 == false;
                            if (v74){
                                assert("Index must be in range." && v73);
                            } else {
                            }
                            unsigned char v76;
                            v76 = v65[v69];
                            v67[v69] = v76;
                            v69 += 1 ;
                        }
                        int v78;
                        v78 = 0;
                        while (while_method_2(v78)){
                            bool v80;
                            v80 = 0 <= v78;
                            bool v82;
                            if (v80){
                                bool v81;
                                v81 = v78 < 5;
                                v82 = v81;
                            } else {
                                v82 = false;
                            }
                            bool v83;
                            v83 = v82 == false;
                            if (v83){
                                assert("Index must be in range." && v82);
                            } else {
                            }
                            unsigned char v85;
                            v85 = v39[v78];
                            int v87;
                            v87 = 2 + v78;
                            v67[v87] = v85;
                            v78 += 1 ;
                        }
                        static_array<unsigned char,5> v88; char v89;
                        Tuple0 tmp80 = score_51(v67);
                        v88 = tmp80.v0; v89 = tmp80.v1;
                        int v90;
                        v90 = v34 % 2;
                        bool v91;
                        v91 = 0 <= v90;
                        bool v93;
                        if (v91){
                            bool v92;
                            v92 = v90 < 2;
                            v93 = v92;
                        } else {
                            v93 = false;
                        }
                        bool v94;
                        v94 = v93 == false;
                        if (v94){
                            assert("Index must be in range." && v93);
                        } else {
                        }
                        int v96;
                        v96 = v33[v90];
                        bool v98;
                        v98 = v64 < v89;
                        Union11 v104;
                        if (v98){
                            v104 = Union11{Union11_2{}};
                        } else {
                            bool v100;
                            v100 = v64 > v89;
                            if (v100){
                                v104 = Union11{Union11_1{}};
                            } else {
                                v104 = Union11{Union11_0{}};
                            }
                        }
                        Union11 v132;
                        switch (v104.tag) {
                            case 0: { // Eq
                                Union11 v105;
                                v105 = Union11{Union11_0{}};
                                int v106;
                                v106 = 0;
                                while (while_method_2(v106)){
                                    bool v108;
                                    v108 = 0 <= v106;
                                    bool v110;
                                    if (v108){
                                        bool v109;
                                        v109 = v106 < 5;
                                        v110 = v109;
                                    } else {
                                        v110 = false;
                                    }
                                    bool v111;
                                    v111 = v110 == false;
                                    if (v111){
                                        assert("Index must be in range." && v110);
                                    } else {
                                    }
                                    unsigned char v113;
                                    v113 = v63[v106];
                                    bool v116;
                                    if (v108){
                                        bool v115;
                                        v115 = v106 < 5;
                                        v116 = v115;
                                    } else {
                                        v116 = false;
                                    }
                                    bool v117;
                                    v117 = v116 == false;
                                    if (v117){
                                        assert("Index must be in range." && v116);
                                    } else {
                                    }
                                    unsigned char v119;
                                    v119 = v88[v106];
                                    unsigned char v121;
                                    v121 = v113 / 4u;
                                    unsigned char v122;
                                    v122 = v119 / 4u;
                                    bool v123;
                                    v123 = v121 < v122;
                                    Union11 v129;
                                    if (v123){
                                        v129 = Union11{Union11_2{}};
                                    } else {
                                        bool v125;
                                        v125 = v121 > v122;
                                        if (v125){
                                            v129 = Union11{Union11_1{}};
                                        } else {
                                            v129 = Union11{Union11_0{}};
                                        }
                                    }
                                    bool v130;
                                    switch (v129.tag) {
                                        case 0: { // Eq
                                            v130 = true;
                                            break;
                                        }
                                        default: {
                                            v130 = false;
                                        }
                                    }
                                    bool v131;
                                    v131 = v130 == false;
                                    if (v131){
                                        v105 = v129;
                                        break;
                                    } else {
                                    }
                                    v106 += 1 ;
                                }
                                v132 = v105;
                                break;
                            }
                            default: {
                                v132 = v104;
                            }
                        }
                        int v137; int v138;
                        switch (v132.tag) {
                            case 0: { // Eq
                                v137 = 0; v138 = -1;
                                break;
                            }
                            case 1: { // Gt
                                v137 = v96; v138 = 0;
                                break;
                            }
                            case 2: { // Lt
                                v137 = v96; v138 = 1;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        static_array<Tuple0,2> v139;
                        v139[0] = Tuple0{v63, v64};
                        v139[1] = Tuple0{v88, v89};
                        Union6 v141;
                        v141 = Union6{Union6_4{v137, v139, v138}};
                        v5.push(v141);
                        Union7 v142;
                        v142 = Union7{Union7_1{v31, v32, v33, v34, v35, v36}};
                        v3.v5 = v142;
                        Union3 v143;
                        v143 = Union3{Union3_0{}};
                        v3.v1 = v143;
                        v2166 = Union3{Union3_0{}};
                        break;
                    }
                    case 7: { // G_Turn
                        int v2043 = v10.case7.v0; static_array<static_array<unsigned char,2>,2> v2044 = v10.case7.v1; static_array<int,2> v2045 = v10.case7.v2; int v2046 = v10.case7.v3; static_array<int,2> v2047 = v10.case7.v4; Union5 v2048 = v10.case7.v5;
                        curandStatePhilox4_32_10_t & v2049 = v3.v4;
                        curandStatePhilox4_32_10_t & v2050 = v2049;
                        static_array<unsigned char,1> v2051; unsigned long long v2052;
                        Tuple12 tmp81 = draw_cards_40(v2050, v6);
                        v2051 = tmp81.v0; v2052 = tmp81.v1;
                        v3.v0 = v2052;
                        static_array_list<unsigned char,5> v2053;
                        v2053 = get_community_cards_41(v2048, v2051);
                        Union6 v2054;
                        v2054 = Union6{Union6_0{v2053}};
                        v5.push(v2054);
                        Union5 v2079;
                        switch (v2048.tag) {
                            case 0: { // Flop
                                static_array<unsigned char,3> v2055 = v2048.case0.v0;
                                static_array<unsigned char,4> v2056;
                                int v2058;
                                v2058 = 0;
                                while (while_method_1(v2058)){
                                    bool v2060;
                                    v2060 = 0 <= v2058;
                                    bool v2062;
                                    if (v2060){
                                        bool v2061;
                                        v2061 = v2058 < 3;
                                        v2062 = v2061;
                                    } else {
                                        v2062 = false;
                                    }
                                    bool v2063;
                                    v2063 = v2062 == false;
                                    if (v2063){
                                        assert("Index must be in range." && v2062);
                                    } else {
                                    }
                                    unsigned char v2065;
                                    v2065 = v2055[v2058];
                                    v2056[v2058] = v2065;
                                    v2058 += 1 ;
                                }
                                int v2067;
                                v2067 = 0;
                                while (while_method_6(v2067)){
                                    bool v2069;
                                    v2069 = 0 <= v2067;
                                    bool v2071;
                                    if (v2069){
                                        bool v2070;
                                        v2070 = v2067 < 1;
                                        v2071 = v2070;
                                    } else {
                                        v2071 = false;
                                    }
                                    bool v2072;
                                    v2072 = v2071 == false;
                                    if (v2072){
                                        assert("Index must be in range." && v2071);
                                    } else {
                                    }
                                    unsigned char v2074;
                                    v2074 = v2051[v2067];
                                    int v2076;
                                    v2076 = 3 + v2067;
                                    v2056[v2076] = v2074;
                                    v2067 += 1 ;
                                }
                                v2079 = Union5{Union5_3{v2056}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in turn.");
                                __trap();
                            }
                        }
                        int v2080;
                        v2080 = 2;
                        int v2081;
                        v2081 = 0;
                        Union4 v2082;
                        v2082 = try_round_36(v2080, v2044, v2045, v2081, v2047, v2079);
                        v2166 = Union3{Union3_1{v2082}};
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
        v8 = v2166;
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
        bool v7;
        v7 = 0 <= v2;
        bool v9;
        if (v7){
            bool v8;
            v8 = v2 < 2;
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
        v12 = v1[v2];
        f_60(v5, v12);
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
        bool v7;
        v7 = 0 <= v2;
        bool v9;
        if (v7){
            bool v8;
            v8 = v2 < 3;
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
        v12 = v1[v2];
        f_60(v5, v12);
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
        bool v7;
        v7 = 0 <= v2;
        bool v9;
        if (v7){
            bool v8;
            v8 = v2 < 5;
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
        v12 = v1[v2];
        f_60(v5, v12);
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
        bool v7;
        v7 = 0 <= v2;
        bool v9;
        if (v7){
            bool v8;
            v8 = v2 < 4;
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
        v12 = v1[v2];
        f_60(v5, v12);
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
        bool v16;
        v16 = 0 <= v9;
        bool v18;
        if (v16){
            bool v17;
            v17 = v9 < 2;
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
        static_array<unsigned char,2> v21;
        v21 = v2[v9];
        f_59(v14, v21);
        v9 += 1 ;
    }
    int v23;
    v23 = 0;
    while (while_method_0(v23)){
        unsigned long long v25;
        v25 = (unsigned long long)v23;
        unsigned long long v26;
        v26 = v25 * 4ull;
        unsigned long long v27;
        v27 = 8ull + v26;
        unsigned char * v28;
        v28 = (unsigned char *)(v0+v27);
        bool v30;
        v30 = 0 <= v23;
        bool v32;
        if (v30){
            bool v31;
            v31 = v23 < 2;
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
        v35 = v3[v23];
        f_57(v28, v35);
        v23 += 1 ;
    }
    int * v37;
    v37 = (int *)(v0+16ull);
    v37[0] = v4;
    int v39;
    v39 = 0;
    while (while_method_0(v39)){
        unsigned long long v41;
        v41 = (unsigned long long)v39;
        unsigned long long v42;
        v42 = v41 * 4ull;
        unsigned long long v43;
        v43 = 20ull + v42;
        unsigned char * v44;
        v44 = (unsigned char *)(v0+v43);
        bool v46;
        v46 = 0 <= v39;
        bool v48;
        if (v46){
            bool v47;
            v47 = v39 < 2;
            v48 = v47;
        } else {
            v48 = false;
        }
        bool v49;
        v49 = v48 == false;
        if (v49){
            assert("Index must be in range." && v48);
        } else {
        }
        int v51;
        v51 = v5[v39];
        f_57(v44, v51);
        v39 += 1 ;
    }
    int v53;
    v53 = v6.tag;
    f_62(v0, v53);
    unsigned char * v54;
    v54 = (unsigned char *)(v0+32ull);
    switch (v6.tag) {
        case 0: { // Flop
            static_array<unsigned char,3> v56 = v6.case0.v0;
            return f_63(v54, v56);
            break;
        }
        case 1: { // Preflop
            return f_55(v54);
            break;
        }
        case 2: { // River
            static_array<unsigned char,5> v57 = v6.case2.v0;
            return f_64(v54, v57);
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4> v58 = v6.case3.v0;
            return f_65(v54, v58);
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
        bool v17;
        v17 = 0 <= v10;
        bool v19;
        if (v17){
            bool v18;
            v18 = v10 < 2;
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
        static_array<unsigned char,2> v22;
        v22 = v2[v10];
        f_59(v15, v22);
        v10 += 1 ;
    }
    int v24;
    v24 = 0;
    while (while_method_0(v24)){
        unsigned long long v26;
        v26 = (unsigned long long)v24;
        unsigned long long v27;
        v27 = v26 * 4ull;
        unsigned long long v28;
        v28 = 8ull + v27;
        unsigned char * v29;
        v29 = (unsigned char *)(v0+v28);
        bool v31;
        v31 = 0 <= v24;
        bool v33;
        if (v31){
            bool v32;
            v32 = v24 < 2;
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
        int v36;
        v36 = v3[v24];
        f_57(v29, v36);
        v24 += 1 ;
    }
    int * v38;
    v38 = (int *)(v0+16ull);
    v38[0] = v4;
    int v40;
    v40 = 0;
    while (while_method_0(v40)){
        unsigned long long v42;
        v42 = (unsigned long long)v40;
        unsigned long long v43;
        v43 = v42 * 4ull;
        unsigned long long v44;
        v44 = 20ull + v43;
        unsigned char * v45;
        v45 = (unsigned char *)(v0+v44);
        bool v47;
        v47 = 0 <= v40;
        bool v49;
        if (v47){
            bool v48;
            v48 = v40 < 2;
            v49 = v48;
        } else {
            v49 = false;
        }
        bool v50;
        v50 = v49 == false;
        if (v50){
            assert("Index must be in range." && v49);
        } else {
        }
        int v52;
        v52 = v5[v40];
        f_57(v45, v52);
        v40 += 1 ;
    }
    int v54;
    v54 = v6.tag;
    f_62(v0, v54);
    unsigned char * v55;
    v55 = (unsigned char *)(v0+32ull);
    switch (v6.tag) {
        case 0: { // Flop
            static_array<unsigned char,3> v57 = v6.case0.v0;
            f_63(v55, v57);
            break;
        }
        case 1: { // Preflop
            f_55(v55);
            break;
        }
        case 2: { // River
            static_array<unsigned char,5> v58 = v6.case2.v0;
            f_64(v55, v58);
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4> v59 = v6.case3.v0;
            f_65(v55, v59);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v60;
    v60 = v7.tag;
    f_67(v0, v60);
    unsigned char * v61;
    v61 = (unsigned char *)(v0+44ull);
    switch (v7.tag) {
        case 0: { // A_All_In
            return f_55(v61);
            break;
        }
        case 1: { // A_Call
            return f_55(v61);
            break;
        }
        case 2: { // A_Fold
            return f_55(v61);
            break;
        }
        case 3: { // A_Raise
            int v63 = v7.case3.v0;
            return f_57(v61, v63);
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
        bool v11;
        v11 = 0 <= v5;
        bool v13;
        if (v11){
            bool v12;
            v12 = v5 < 2;
            v13 = v12;
        } else {
            v13 = false;
        }
        bool v14;
        v14 = v13 == false;
        if (v14){
            assert("Index must be in range." && v13);
        } else {
        }
        unsigned char v16;
        v16 = v2[v5];
        f_60(v9, v16);
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
        bool v8;
        v8 = 0 <= v3;
        bool v10;
        if (v8){
            bool v9;
            v9 = v3 < 5;
            v10 = v9;
        } else {
            v10 = false;
        }
        bool v11;
        v11 = v10 == false;
        if (v11){
            assert("Index must be in range." && v10);
        } else {
        }
        unsigned char v13;
        v13 = v1[v3];
        f_60(v6, v13);
        v3 += 1 ;
    }
    char * v15;
    v15 = (char *)(v0+5ull);
    v15[0] = v2;
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
        bool v13;
        v13 = 0 <= v6;
        bool v15;
        if (v13){
            bool v14;
            v14 = v6 < 2;
            v15 = v14;
        } else {
            v15 = false;
        }
        bool v16;
        v16 = v15 == false;
        if (v16){
            assert("Index must be in range." && v15);
        } else {
        }
        static_array<unsigned char,5> v18; char v19;
        Tuple0 tmp82 = v2[v6];
        v18 = tmp82.v0; v19 = tmp82.v1;
        f_76(v11, v18, v19);
        v6 += 1 ;
    }
    int * v22;
    v22 = (int *)(v0+24ull);
    v22[0] = v3;
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
        case 0: { // CallingMachine
            return f_55(v3);
            break;
        }
        case 1: { // Computer
            return f_55(v3);
            break;
        }
        case 2: { // Human
            return f_55(v3);
            break;
        }
        case 3: { // Random
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
__device__ void f_52(unsigned char * v0, unsigned long long v1, Union3 v2, static_array_list<Union6,128> v3, static_array<Union2,2> v4, Union7 v5){
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
    v10 = v3.length;
    f_68(v0, v10);
    int v11;
    v11 = v3.length;
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
        bool v28;
        v28 = 0 <= v21;
        bool v30;
        if (v28){
            bool v29;
            v29 = v21 < 2;
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
        Union2 v33;
        v33 = v4[v21];
        f_78(v26, v33);
        v21 += 1 ;
    }
    int v35;
    v35 = v5.tag;
    f_79(v0, v35);
    unsigned char * v36;
    v36 = (unsigned char *)(v0+6256ull);
    switch (v5.tag) {
        case 0: { // GameNotStarted
            return f_55(v36);
            break;
        }
        case 1: { // GameOver
            int v38 = v5.case1.v0; static_array<static_array<unsigned char,2>,2> v39 = v5.case1.v1; static_array<int,2> v40 = v5.case1.v2; int v41 = v5.case1.v3; static_array<int,2> v42 = v5.case1.v4; Union5 v43 = v5.case1.v5;
            return f_58(v36, v38, v39, v40, v41, v42, v43);
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            int v44 = v5.case2.v0; static_array<static_array<unsigned char,2>,2> v45 = v5.case2.v1; static_array<int,2> v46 = v5.case2.v2; int v47 = v5.case2.v3; static_array<int,2> v48 = v5.case2.v4; Union5 v49 = v5.case2.v5;
            return f_58(v36, v44, v45, v46, v47, v48, v49);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1, unsigned char * v2, unsigned char * v3, unsigned char * v4) {
    Union0 v5;
    v5 = f_0(v1);
    unsigned long long v6; Union3 v7; static_array_list<Union6,128> v8; static_array<Union2,2> v9; Union7 v10;
    Tuple1 tmp15 = f_6(v0);
    v6 = tmp15.v0; v7 = tmp15.v1; v8 = tmp15.v2; v9 = tmp15.v3; v10 = tmp15.v4;
    unsigned long long v11;
    v11 = clock64();
    int v12;
    v12 = threadIdx.x;
    int v13;
    v13 = blockIdx.x;
    int v14;
    v14 = v13 * 256;
    int v15;
    v15 = v12 + v14;
    unsigned long long v16;
    v16 = (unsigned long long)v15;
    curandStatePhilox4_32_10_t v17;
    curand_init(v11,v16,0ull,&v17);
    curandStatePhilox4_32_10_t & v18 = v17;
    StackMut0 v19{v6, v7, v8, v9, v18, v10};
    Union3 v56;
    switch (v5.tag) {
        case 0: { // ActionSelected
            Union1 v34 = v5.case0.v0;
            Union3 & v35 = v19.v1;
            switch (v35.tag) {
                case 0: { // None
                    printf("%s\n", "The game hasn't been started in ActionSelected.");
                    __trap();
                    break;
                }
                case 1: { // Some
                    Union4 v36 = v35.case1.v0;
                    switch (v36.tag) {
                        case 4: { // G_Round
                            int v37 = v36.case4.v0; static_array<static_array<unsigned char,2>,2> v38 = v36.case4.v1; static_array<int,2> v39 = v36.case4.v2; int v40 = v36.case4.v3; static_array<int,2> v41 = v36.case4.v4; Union5 v42 = v36.case4.v5;
                            Union4 v43;
                            v43 = Union4{Union4_5{v37, v38, v39, v40, v41, v42, v34}};
                            v56 = Union3{Union3_1{v43}};
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
            static_array<Union2,2> v32 = v5.case1.v0;
            v19.v3 = v32;
            v56 = Union3{Union3_0{}};
            break;
        }
        case 2: { // StartGame
            static_array<Union2,2> v20;
            Union2 v22;
            v22 = Union2{Union2_1{}};
            v20[0] = v22;
            Union2 v24;
            v24 = Union2{Union2_2{}};
            v20[1] = v24;
            static_array_list<Union6,128> v26;
            v26 = static_array_list<Union6,128>{};
            Union7 v28;
            v28 = Union7{Union7_0{}};
            v19.v5 = v28;
            Union3 v29;
            v29 = Union3{Union3_0{}};
            v19.v1 = v29;
            v19.v0 = 4503599627370495ull;
            v19.v2 = v26;
            Union4 v30;
            v30 = Union4{Union4_2{}};
            v56 = Union3{Union3_1{v30}};
            break;
        }
        case 3: { // StartTrainingVsRando
            printf("%s\n", "Training is not supported in the `event_loop_play` function.");
            __trap();
            break;
        }
        case 4: { // StartTrainingVsSelf
            printf("%s\n", "Training is not supported in the `event_loop_play` function.");
            __trap();
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    switch (v56.tag) {
        case 0: { // None
            break;
        }
        case 1: { // Some
            Union4 v57 = v56.case1.v0;
            play_loop_31(v2, v3, v4, v19, v57);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v58;
    v58 = threadIdx.x;
    int v59;
    v59 = blockIdx.x;
    int v60;
    v60 = v59 * 256;
    int v61;
    v61 = v58 + v60;
    bool v62;
    v62 = v61 == 0;
    if (v62){
        Union7 & v63 = v19.v5;
        static_array<Union2,2> & v64 = v19.v3;
        static_array_list<Union6,128> & v65 = v19.v2;
        Union3 & v66 = v19.v1;
        unsigned long long & v67 = v19.v0;
        return f_52(v0, v67, v66, v65, v64, v63);
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
class US0_3(NamedTuple): # StartTrainingVsRando
    tag = 3
class US0_4(NamedTuple): # StartTrainingVsSelf
    tag = 4
US0 = Union[US0_0, US0_1, US0_2, US0_3, US0_4]
class US2_0(NamedTuple): # CallingMachine
    tag = 0
class US2_1(NamedTuple): # Computer
    tag = 1
class US2_2(NamedTuple): # Human
    tag = 2
class US2_3(NamedTuple): # Random
    tag = 3
US2 = Union[US2_0, US2_1, US2_2, US2_3]
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
        v3, v4, v5, v6, v7, v8, v9, v10 = method8(v1)
        v11 = cp.empty(16,dtype=cp.uint8)
        v12 = cp.empty(6304,dtype=cp.uint8)
        method46(v12, v3, v4, v5, v6, v7)
        del v3, v4, v5, v6, v7
        v15 = "{}\n"
        v16 = "Going to run the NL Holdem full kernel."
        print(v15.format(v16),end="")
        del v15, v16
        v17 = time.perf_counter()
        v18 = []
        match v2:
            case US0_0(_): # ActionSelected
                method78(v11, v2)
                v33 = cp.cuda.Device().attributes['MultiProcessorCount']
                v34 = v33 == 24
                del v33
                v35 = v34 == False
                if v35:
                    v36 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v34, v36
                    del v36
                else:
                    pass
                del v34, v35
                v37 = 0
                v38 = raw_module.get_function(f"entry{v37}")
                del v37
                v38.max_dynamic_shared_size_bytes = 98304 
                print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
                v38((24,),(256,),(v12, v11, v8, v9, v10),shared_mem=98304)
                del v38
            case US0_1(_): # PlayerChanged
                method78(v11, v2)
                v26 = cp.cuda.Device().attributes['MultiProcessorCount']
                v27 = v26 == 24
                del v26
                v28 = v27 == False
                if v28:
                    v29 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v27, v29
                    del v29
                else:
                    pass
                del v27, v28
                v30 = 0
                v31 = raw_module.get_function(f"entry{v30}")
                del v30
                v31.max_dynamic_shared_size_bytes = 98304 
                print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
                v31((24,),(256,),(v12, v11, v8, v9, v10),shared_mem=98304)
                del v31
            case US0_2(): # StartGame
                method78(v11, v2)
                v19 = cp.cuda.Device().attributes['MultiProcessorCount']
                v20 = v19 == 24
                del v19
                v21 = v20 == False
                if v21:
                    v22 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v20, v22
                    del v22
                else:
                    pass
                del v20, v21
                v23 = 0
                v24 = raw_module.get_function(f"entry{v23}")
                del v23
                v24.max_dynamic_shared_size_bytes = 98304 
                print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
                v24((24,),(256,),(v12, v11, v8, v9, v10),shared_mem=98304)
                del v24
            case US0_3(): # StartTrainingVsRando
                raise Exception("TOTO")
            case US0_4(): # StartTrainingVsSelf
                raise Exception("TOTO")
            case t:
                raise Exception(f'Pattern matching miss. Got: {t}')
        del v2, v11
        cp.cuda.get_current_stream().synchronize()
        v39 = time.perf_counter()
        v42 = "{}"
        v43 = "The time it took to run the kernel (in seconds) is: "
        print(v42.format(v43),end="")
        del v42, v43
        v44 = v39 - v17
        del v17, v39
        v47 = "{:.6f}\n"
        print(v47.format(v44),end="")
        del v44, v47
        v48, v49, v50, v51, v52 = method81(v12)
        del v12
        return method109(v48, v49, v50, v51, v52, v8, v9, v10, v18)
    return inner
def Closure1():
    def inner() -> object:
        v0 = cp.empty(0,dtype=cp.uint8)
        v1 = cp.empty(102236160,dtype=cp.uint8)
        v2 = cp.empty(16777216,dtype=cp.uint8)
        v4 = v1[0:0+4*12582912].view(cp.float32)
        del v4
        v6 = v2[0:0+4*4194304].view(cp.float32)
        v8 = v1[51904512:51904512+4*12582912].view(cp.float32)
        del v8
        v9 = cp.random.normal(0.0,0.00048828125,4194304,dtype=cp.float32) # type: ignore
        cp.copyto(v6[0:0+4194304],v9[0:0+4194304])
        del v6, v9
        v11 = static_array(2)
        v13 = US2_1()
        v11[0] = v13
        del v13
        v15 = US2_2()
        v11[1] = v15
        del v15
        v17 = static_array_list(128)
        v18 = 4503599627370495
        v19 = US3_0()
        v20 = US6_0()
        return method158(v18, v19, v17, v11, v20, v2, v1, v0)
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
    v3 = "CallingMachine" == v1
    if v3:
        del v1, v3
        method3(v2)
        del v2
        return US2_0()
    else:
        del v3
        v5 = "Computer" == v1
        if v5:
            del v1, v5
            method3(v2)
            del v2
            return US2_1()
        else:
            del v5
            v7 = "Human" == v1
            if v7:
                del v1, v7
                method3(v2)
                del v2
                return US2_2()
            else:
                del v7
                v9 = "Random" == v1
                if v9:
                    del v1, v9
                    method3(v2)
                    del v2
                    return US2_3()
                else:
                    del v2, v9
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
def method43(v0 : object) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    v1 = v0[0] # type: ignore
    v2 = method44(v1)
    del v1
    v3 = v0[1] # type: ignore
    v4 = method44(v3)
    del v3
    v5 = v0[2] # type: ignore
    v6 = method44(v5)
    del v5
    v7 = v0[3] # type: ignore
    del v0
    method3(v7)
    del v7
    return v2, v4, v6
def method42(v0 : object) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    v1, v2, v3 = method43(v0)
    del v0
    return v1, v2, v3
def method41(v0 : object) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    v1, v2, v3 = method42(v0)
    del v0
    return v1, v2, v3
def method40(v0 : object) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    v1 = v0["model_ptrs"] # type: ignore
    del v0
    v2, v3, v4 = method41(v1)
    del v1
    return v2, v3, v4
def method9(v0 : object) -> Tuple[u64, US3, static_array_list, static_array, US6, cp.ndarray, cp.ndarray, cp.ndarray]:
    v1 = v0["game"] # type: ignore
    v2, v3, v4, v5, v6 = method10(v1)
    del v1
    v7 = v0["neural"] # type: ignore
    del v0
    v8, v9, v10 = method40(v7)
    del v7
    return v2, v3, v4, v5, v6, v8, v9, v10
def method8(v0 : object) -> Tuple[u64, US3, static_array_list, static_array, US6, cp.ndarray, cp.ndarray, cp.ndarray]:
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
        case US2_0(): # CallingMachine
            del v1
            return method49(v4)
        case US2_1(): # Computer
            del v1
            return method49(v4)
        case US2_2(): # Human
            del v1
            return method49(v4)
        case US2_3(): # Random
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
        v8 = 0 <= v2
        if v8:
            v9 = v2 < 2
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
        v14 = v1[v2]
        method76(v7, v14)
        del v7, v14
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
        case US0_3(): # StartTrainingVsRando
            del v1
            return method49(v4)
        case US0_4(): # StartTrainingVsSelf
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
    elif v1 == 3:
        del v1
        method84(v3)
        del v3
        return US2_3()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method108(v0 : cp.ndarray) -> i32:
    v2 = v0[6248:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method81(v0 : cp.ndarray) -> Tuple[u64, US3, static_array_list, static_array, US6]:
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
    v11 = static_array_list(128)
    v12 = method97(v0)
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
        v11 = method123(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method121(v0 : static_array) -> object:
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
        v11 = method122(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method125(v0 : static_array) -> object:
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
def method127(v0 : static_array) -> object:
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
        v11 = method123(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method128(v0 : static_array) -> object:
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
        v11 = method123(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method129(v0 : static_array) -> object:
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
        v11 = method123(v10)
        del v10
        v1.append(v11)
        del v11
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
        v13 = method141(v11, v12)
        del v11, v12
        v1.append(v13)
        del v13
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
def method133(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
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
        case US2_0(): # CallingMachine
            del v0
            v1 = method117()
            v2 = "CallingMachine"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US2_1(): # Computer
            del v0
            v4 = method117()
            v5 = "Computer"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US2_2(): # Human
            del v0
            v7 = method117()
            v8 = "Human"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US2_3(): # Random
            del v0
            v10 = method117()
            v11 = "Random"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method144(v0 : static_array) -> object:
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
        v11 = method145(v10)
        del v10
        v1.append(v11)
        del v11
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
def method132(v0 : static_array_list, v1 : static_array, v2 : US6) -> object:
    v3 = method133(v0)
    del v0
    v4 = method144(v1)
    del v1
    v5 = method146(v2)
    del v2
    v6 = {'messages': v3, 'pl_type': v4, 'ui_game_state': v5}
    del v3, v4, v5
    return v6
def method112(v0 : u64, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US6) -> object:
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
def method150(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    v3 = []
    v4 = method151(v0)
    del v0
    v3.append(v4)
    del v4
    v5 = method151(v1)
    del v1
    v3.append(v5)
    del v5
    v6 = method151(v2)
    del v2
    v3.append(v6)
    del v6
    v7 = method117()
    v3.append(v7)
    del v7
    v8 = v3
    del v3
    return v8
def method149(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    return method150(v0, v1, v2)
def method148(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    return method149(v0, v1, v2)
def method147(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    v3 = method148(v0, v1, v2)
    del v0, v1, v2
    v4 = {'model_ptrs': v3}
    del v3
    return v4
def method111(v0 : u64, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US6, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray) -> object:
    v8 = method112(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    v9 = method147(v5, v6, v7)
    del v5, v6, v7
    v10 = {'game': v8, 'neural': v9}
    del v8, v9
    return v10
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
def method110(v0 : u64, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US6, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : list) -> object:
    v9 = []
    v10 = method111(v0, v1, v2, v3, v4, v5, v6, v7)
    del v0, v1, v2, v3, v4, v5, v6, v7
    v9.append(v10)
    del v10
    v11 = method153(v8)
    del v8
    v9.append(v11)
    del v11
    v12 = v9
    del v9
    return v12
def method109(v0 : u64, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US6, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : list) -> object:
    v9 = method110(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8
    return v9
def method158(v0 : u64, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US6, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray) -> object:
    v8 = method111(v0, v1, v2, v3, v4, v5, v6, v7)
    del v0, v1, v2, v3, v4, v5, v6, v7
    return v8
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
