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
struct Union6;
struct Union5;
struct Union4;
struct Union3;
struct Union7;
struct Union8;
struct Tuple0;
__device__ unsigned int f_7(unsigned char * v0);
__device__ int f_8(unsigned char * v0);
struct Tuple1;
__device__ Union6 f_11(unsigned char * v0);
__device__ Tuple1 f_10(unsigned char * v0);
struct Tuple2;
__device__ int f_13(unsigned char * v0);
__device__ Tuple2 f_12(unsigned char * v0);
__device__ Union4 f_9(unsigned char * v0);
__device__ int f_14(unsigned char * v0);
struct Tuple3;
__device__ Tuple3 f_16(unsigned char * v0);
struct Tuple4;
__device__ Tuple4 f_17(unsigned char * v0);
struct Tuple5;
__device__ Tuple5 f_18(unsigned char * v0);
__device__ Union7 f_15(unsigned char * v0);
__device__ int f_19(unsigned char * v0);
__device__ Tuple0 f_6(unsigned char * v0);
struct StackMut0;
struct Tuple6;
__device__ unsigned int loop_21(unsigned int v0, curandStatePhilox4_32_10_t & v1);
__device__ Tuple6 draw_card_20(curandStatePhilox4_32_10_t & v0, unsigned int v1);
struct Tuple7;
struct Union9;
struct Union10;
__device__ int int_range_22(int v0, int v1, curandStatePhilox4_32_10_t & v2);
struct Union11;
__device__ void block_matmul_23(float * v0, float * v1, int v2, float * v3);
__device__ void block_map_24(float * v0, int v1, float * v2);
__device__ void block_row_map_25(float * v0, int v1, float * v2);
struct Tuple8;
struct Tuple9;
struct Tuple10;
struct Tuple11;
struct Union12;
struct Union13;
__device__ int tag_27(Union6 v0);
__device__ bool is_pair_28(int v0, int v1);
__device__ Tuple7 order_29(int v0, int v1);
__device__ Union13 compare_hands_26(Union5 v0, bool v1, static_array<Union6,2> v2, int v3, static_array<int,2> v4, int v5);
__device__ void f_31(unsigned char * v0, unsigned int v1);
__device__ void f_32(unsigned char * v0, int v1);
__device__ void f_33(unsigned char * v0);
__device__ void f_35(unsigned char * v0, int v1);
__device__ void f_37(unsigned char * v0, Union6 v1);
__device__ void f_36(unsigned char * v0, Union5 v1, bool v2, static_array<Union6,2> v3, int v4, static_array<int,2> v5, int v6);
__device__ void f_39(unsigned char * v0, int v1);
__device__ void f_38(unsigned char * v0, Union5 v1, bool v2, static_array<Union6,2> v3, int v4, static_array<int,2> v5, int v6, Union1 v7);
__device__ void f_34(unsigned char * v0, Union4 v1);
__device__ void f_40(unsigned char * v0, int v1);
__device__ void f_42(unsigned char * v0, int v1, Union1 v2);
__device__ void f_43(unsigned char * v0, int v1, Union6 v2);
__device__ void f_44(unsigned char * v0, static_array<Union6,2> v1, int v2, int v3);
__device__ void f_41(unsigned char * v0, Union7 v1);
__device__ void f_45(unsigned char * v0, Union2 v1);
__device__ void f_46(unsigned char * v0, int v1);
__device__ void f_30(unsigned char * v0, unsigned int v1, Union3 v2, static_array_list<Union7,32> v3, static_array<Union2,2> v4, Union8 v5);
struct StackMut1;
struct Union14;
__device__ void method_47(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut1 & v3, int v4, Union4 v5);
struct Tuple12;
struct Tuple13;
__device__ void method_48(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut1 & v3, Union4 v4);
__device__ void method_49(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut1 & v3, Union4 v4);
struct Union1_0 { // Call
};
struct Union1_1 { // Fold
};
struct Union1_2 { // Raise
};
struct Union1 {
    union {
        Union1_0 case0; // Call
        Union1_1 case1; // Fold
        Union1_2 case2; // Raise
    };
    unsigned char tag{255};
    __device__ Union1() {}
    __device__ Union1(Union1_0 t) : tag(0), case0(t) {} // Call
    __device__ Union1(Union1_1 t) : tag(1), case1(t) {} // Fold
    __device__ Union1(Union1_2 t) : tag(2), case2(t) {} // Raise
    __device__ Union1(Union1 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(x.case0); break; // Call
            case 1: new (&this->case1) Union1_1(x.case1); break; // Fold
            case 2: new (&this->case2) Union1_2(x.case2); break; // Raise
        }
    }
    __device__ Union1(Union1 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // Call
            case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // Fold
            case 2: new (&this->case2) Union1_2(std::move(x.case2)); break; // Raise
        }
    }
    __device__ Union1 & operator=(Union1 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Call
                case 1: this->case1 = x.case1; break; // Fold
                case 2: this->case2 = x.case2; break; // Raise
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
                case 0: this->case0 = std::move(x.case0); break; // Call
                case 1: this->case1 = std::move(x.case1); break; // Fold
                case 2: this->case2 = std::move(x.case2); break; // Raise
            }
        } else {
            this->~Union1();
            new (this) Union1{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union1() {
        switch(this->tag){
            case 0: this->case0.~Union1_0(); break; // Call
            case 1: this->case1.~Union1_1(); break; // Fold
            case 2: this->case2.~Union1_2(); break; // Raise
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
struct Union6_0 { // Jack
};
struct Union6_1 { // King
};
struct Union6_2 { // Queen
};
struct Union6 {
    union {
        Union6_0 case0; // Jack
        Union6_1 case1; // King
        Union6_2 case2; // Queen
    };
    unsigned char tag{255};
    __device__ Union6() {}
    __device__ Union6(Union6_0 t) : tag(0), case0(t) {} // Jack
    __device__ Union6(Union6_1 t) : tag(1), case1(t) {} // King
    __device__ Union6(Union6_2 t) : tag(2), case2(t) {} // Queen
    __device__ Union6(Union6 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(x.case0); break; // Jack
            case 1: new (&this->case1) Union6_1(x.case1); break; // King
            case 2: new (&this->case2) Union6_2(x.case2); break; // Queen
        }
    }
    __device__ Union6(Union6 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(std::move(x.case0)); break; // Jack
            case 1: new (&this->case1) Union6_1(std::move(x.case1)); break; // King
            case 2: new (&this->case2) Union6_2(std::move(x.case2)); break; // Queen
        }
    }
    __device__ Union6 & operator=(Union6 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Jack
                case 1: this->case1 = x.case1; break; // King
                case 2: this->case2 = x.case2; break; // Queen
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
                case 0: this->case0 = std::move(x.case0); break; // Jack
                case 1: this->case1 = std::move(x.case1); break; // King
                case 2: this->case2 = std::move(x.case2); break; // Queen
            }
        } else {
            this->~Union6();
            new (this) Union6{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union6() {
        switch(this->tag){
            case 0: this->case0.~Union6_0(); break; // Jack
            case 1: this->case1.~Union6_1(); break; // King
            case 2: this->case2.~Union6_2(); break; // Queen
        }
        this->tag = 255;
    }
};
struct Union5_0 { // None
};
struct Union5_1 { // Some
    Union6 v0;
    __device__ Union5_1(Union6 t0) : v0(t0) {}
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
struct Union4_0 { // ChanceCommunityCard
    Union5 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_0(Union5 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_0() = delete;
};
struct Union4_1 { // ChanceInit
};
struct Union4_2 { // Round
    Union5 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_2(Union5 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_2() = delete;
};
struct Union4_3 { // RoundWithAction
    Union5 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    Union1 v6;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_3(Union5 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union4_3() = delete;
};
struct Union4_4 { // TerminalCall
    Union5 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_4(Union5 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_4() = delete;
};
struct Union4_5 { // TerminalFold
    Union5 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_5(Union5 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_5() = delete;
};
struct Union4 {
    union {
        Union4_0 case0; // ChanceCommunityCard
        Union4_1 case1; // ChanceInit
        Union4_2 case2; // Round
        Union4_3 case3; // RoundWithAction
        Union4_4 case4; // TerminalCall
        Union4_5 case5; // TerminalFold
    };
    unsigned char tag{255};
    __device__ Union4() {}
    __device__ Union4(Union4_0 t) : tag(0), case0(t) {} // ChanceCommunityCard
    __device__ Union4(Union4_1 t) : tag(1), case1(t) {} // ChanceInit
    __device__ Union4(Union4_2 t) : tag(2), case2(t) {} // Round
    __device__ Union4(Union4_3 t) : tag(3), case3(t) {} // RoundWithAction
    __device__ Union4(Union4_4 t) : tag(4), case4(t) {} // TerminalCall
    __device__ Union4(Union4_5 t) : tag(5), case5(t) {} // TerminalFold
    __device__ Union4(Union4 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(x.case0); break; // ChanceCommunityCard
            case 1: new (&this->case1) Union4_1(x.case1); break; // ChanceInit
            case 2: new (&this->case2) Union4_2(x.case2); break; // Round
            case 3: new (&this->case3) Union4_3(x.case3); break; // RoundWithAction
            case 4: new (&this->case4) Union4_4(x.case4); break; // TerminalCall
            case 5: new (&this->case5) Union4_5(x.case5); break; // TerminalFold
        }
    }
    __device__ Union4(Union4 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(std::move(x.case0)); break; // ChanceCommunityCard
            case 1: new (&this->case1) Union4_1(std::move(x.case1)); break; // ChanceInit
            case 2: new (&this->case2) Union4_2(std::move(x.case2)); break; // Round
            case 3: new (&this->case3) Union4_3(std::move(x.case3)); break; // RoundWithAction
            case 4: new (&this->case4) Union4_4(std::move(x.case4)); break; // TerminalCall
            case 5: new (&this->case5) Union4_5(std::move(x.case5)); break; // TerminalFold
        }
    }
    __device__ Union4 & operator=(Union4 & x) {
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
            this->~Union4();
            new (this) Union4{x};
        }
        return *this;
    }
    __device__ Union4 & operator=(Union4 && x) {
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
            this->~Union4();
            new (this) Union4{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union4() {
        switch(this->tag){
            case 0: this->case0.~Union4_0(); break; // ChanceCommunityCard
            case 1: this->case1.~Union4_1(); break; // ChanceInit
            case 2: this->case2.~Union4_2(); break; // Round
            case 3: this->case3.~Union4_3(); break; // RoundWithAction
            case 4: this->case4.~Union4_4(); break; // TerminalCall
            case 5: this->case5.~Union4_5(); break; // TerminalFold
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
struct Union7_0 { // CommunityCardIs
    Union6 v0;
    __device__ Union7_0(Union6 t0) : v0(t0) {}
    __device__ Union7_0() = delete;
};
struct Union7_1 { // PlayerAction
    Union1 v1;
    int v0;
    __device__ Union7_1(int t0, Union1 t1) : v0(t0), v1(t1) {}
    __device__ Union7_1() = delete;
};
struct Union7_2 { // PlayerGotCard
    Union6 v1;
    int v0;
    __device__ Union7_2(int t0, Union6 t1) : v0(t0), v1(t1) {}
    __device__ Union7_2() = delete;
};
struct Union7_3 { // Showdown
    static_array<Union6,2> v0;
    int v1;
    int v2;
    __device__ Union7_3(static_array<Union6,2> t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
    __device__ Union7_3() = delete;
};
struct Union7 {
    union {
        Union7_0 case0; // CommunityCardIs
        Union7_1 case1; // PlayerAction
        Union7_2 case2; // PlayerGotCard
        Union7_3 case3; // Showdown
    };
    unsigned char tag{255};
    __device__ Union7() {}
    __device__ Union7(Union7_0 t) : tag(0), case0(t) {} // CommunityCardIs
    __device__ Union7(Union7_1 t) : tag(1), case1(t) {} // PlayerAction
    __device__ Union7(Union7_2 t) : tag(2), case2(t) {} // PlayerGotCard
    __device__ Union7(Union7_3 t) : tag(3), case3(t) {} // Showdown
    __device__ Union7(Union7 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(x.case0); break; // CommunityCardIs
            case 1: new (&this->case1) Union7_1(x.case1); break; // PlayerAction
            case 2: new (&this->case2) Union7_2(x.case2); break; // PlayerGotCard
            case 3: new (&this->case3) Union7_3(x.case3); break; // Showdown
        }
    }
    __device__ Union7(Union7 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(std::move(x.case0)); break; // CommunityCardIs
            case 1: new (&this->case1) Union7_1(std::move(x.case1)); break; // PlayerAction
            case 2: new (&this->case2) Union7_2(std::move(x.case2)); break; // PlayerGotCard
            case 3: new (&this->case3) Union7_3(std::move(x.case3)); break; // Showdown
        }
    }
    __device__ Union7 & operator=(Union7 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // CommunityCardIs
                case 1: this->case1 = x.case1; break; // PlayerAction
                case 2: this->case2 = x.case2; break; // PlayerGotCard
                case 3: this->case3 = x.case3; break; // Showdown
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
                case 0: this->case0 = std::move(x.case0); break; // CommunityCardIs
                case 1: this->case1 = std::move(x.case1); break; // PlayerAction
                case 2: this->case2 = std::move(x.case2); break; // PlayerGotCard
                case 3: this->case3 = std::move(x.case3); break; // Showdown
            }
        } else {
            this->~Union7();
            new (this) Union7{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union7() {
        switch(this->tag){
            case 0: this->case0.~Union7_0(); break; // CommunityCardIs
            case 1: this->case1.~Union7_1(); break; // PlayerAction
            case 2: this->case2.~Union7_2(); break; // PlayerGotCard
            case 3: this->case3.~Union7_3(); break; // Showdown
        }
        this->tag = 255;
    }
};
struct Union8_0 { // GameNotStarted
};
struct Union8_1 { // GameOver
    Union5 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union8_1(Union5 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union8_1() = delete;
};
struct Union8_2 { // WaitingForActionFromPlayerId
    Union5 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union8_2(Union5 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union8_2() = delete;
};
struct Union8 {
    union {
        Union8_0 case0; // GameNotStarted
        Union8_1 case1; // GameOver
        Union8_2 case2; // WaitingForActionFromPlayerId
    };
    unsigned char tag{255};
    __device__ Union8() {}
    __device__ Union8(Union8_0 t) : tag(0), case0(t) {} // GameNotStarted
    __device__ Union8(Union8_1 t) : tag(1), case1(t) {} // GameOver
    __device__ Union8(Union8_2 t) : tag(2), case2(t) {} // WaitingForActionFromPlayerId
    __device__ Union8(Union8 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(x.case0); break; // GameNotStarted
            case 1: new (&this->case1) Union8_1(x.case1); break; // GameOver
            case 2: new (&this->case2) Union8_2(x.case2); break; // WaitingForActionFromPlayerId
        }
    }
    __device__ Union8(Union8 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(std::move(x.case0)); break; // GameNotStarted
            case 1: new (&this->case1) Union8_1(std::move(x.case1)); break; // GameOver
            case 2: new (&this->case2) Union8_2(std::move(x.case2)); break; // WaitingForActionFromPlayerId
        }
    }
    __device__ Union8 & operator=(Union8 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // GameNotStarted
                case 1: this->case1 = x.case1; break; // GameOver
                case 2: this->case2 = x.case2; break; // WaitingForActionFromPlayerId
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
                case 0: this->case0 = std::move(x.case0); break; // GameNotStarted
                case 1: this->case1 = std::move(x.case1); break; // GameOver
                case 2: this->case2 = std::move(x.case2); break; // WaitingForActionFromPlayerId
            }
        } else {
            this->~Union8();
            new (this) Union8{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union8() {
        switch(this->tag){
            case 0: this->case0.~Union8_0(); break; // GameNotStarted
            case 1: this->case1.~Union8_1(); break; // GameOver
            case 2: this->case2.~Union8_2(); break; // WaitingForActionFromPlayerId
        }
        this->tag = 255;
    }
};
struct Tuple0 {
    Union3 v1;
    static_array_list<Union7,32> v2;
    static_array<Union2,2> v3;
    Union8 v4;
    unsigned int v0;
    __device__ Tuple0() = default;
    __device__ Tuple0(unsigned int t0, Union3 t1, static_array_list<Union7,32> t2, static_array<Union2,2> t3, Union8 t4) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4) {}
};
struct Tuple1 {
    Union5 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(Union5 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Tuple2 {
    Union5 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    Union1 v6;
    int v3;
    int v5;
    bool v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(Union5 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
};
struct Tuple3 {
    Union1 v1;
    int v0;
    __device__ Tuple3() = default;
    __device__ Tuple3(int t0, Union1 t1) : v0(t0), v1(t1) {}
};
struct Tuple4 {
    Union6 v1;
    int v0;
    __device__ Tuple4() = default;
    __device__ Tuple4(int t0, Union6 t1) : v0(t0), v1(t1) {}
};
struct Tuple5 {
    static_array<Union6,2> v0;
    int v1;
    int v2;
    __device__ Tuple5() = default;
    __device__ Tuple5(static_array<Union6,2> t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
};
struct StackMut0 {
    Union3 v1;
    static_array_list<Union7,32> v2;
    static_array<Union2,2> v3;
    curandStatePhilox4_32_10_t v4;
    Union8 v5;
    unsigned int v0;
    __device__ StackMut0() = default;
    __device__ StackMut0(unsigned int t0, Union3 t1, static_array_list<Union7,32> t2, static_array<Union2,2> t3, curandStatePhilox4_32_10_t t4, Union8 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Tuple6 {
    Union6 v0;
    unsigned int v1;
    __device__ Tuple6() = default;
    __device__ Tuple6(Union6 t0, unsigned int t1) : v0(t0), v1(t1) {}
};
struct Tuple7 {
    int v0;
    int v1;
    __device__ Tuple7() = default;
    __device__ Tuple7(int t0, int t1) : v0(t0), v1(t1) {}
};
struct Union9_0 { // C1of2
    Union1 v0;
    __device__ Union9_0(Union1 t0) : v0(t0) {}
    __device__ Union9_0() = delete;
};
struct Union9_1 { // C2of2
    Union6 v0;
    __device__ Union9_1(Union6 t0) : v0(t0) {}
    __device__ Union9_1() = delete;
};
struct Union9 {
    union {
        Union9_0 case0; // C1of2
        Union9_1 case1; // C2of2
    };
    unsigned char tag{255};
    __device__ Union9() {}
    __device__ Union9(Union9_0 t) : tag(0), case0(t) {} // C1of2
    __device__ Union9(Union9_1 t) : tag(1), case1(t) {} // C2of2
    __device__ Union9(Union9 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union9_0(x.case0); break; // C1of2
            case 1: new (&this->case1) Union9_1(x.case1); break; // C2of2
        }
    }
    __device__ Union9(Union9 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union9_0(std::move(x.case0)); break; // C1of2
            case 1: new (&this->case1) Union9_1(std::move(x.case1)); break; // C2of2
        }
    }
    __device__ Union9 & operator=(Union9 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // C1of2
                case 1: this->case1 = x.case1; break; // C2of2
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
                case 0: this->case0 = std::move(x.case0); break; // C1of2
                case 1: this->case1 = std::move(x.case1); break; // C2of2
            }
        } else {
            this->~Union9();
            new (this) Union9{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union9() {
        switch(this->tag){
            case 0: this->case0.~Union9_0(); break; // C1of2
            case 1: this->case1.~Union9_1(); break; // C2of2
        }
        this->tag = 255;
    }
};
struct Union10_0 { // None
};
struct Union10_1 { // Some
    Union9 v0;
    __device__ Union10_1(Union9 t0) : v0(t0) {}
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
    int v0;
    __device__ Union11_1(int t0) : v0(t0) {}
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
struct Tuple8 {
    int v0;
    float v1;
    __device__ Tuple8() = default;
    __device__ Tuple8(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Closure2 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple9 {
    float v0;
    bool v1;
    __device__ Tuple9() = default;
    __device__ Tuple9(float t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ Tuple9 operator()(Tuple9 tup0, Tuple9 tup1){
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
                return Tuple9{v5, true};
            } else {
                return Tuple9{v0, v1};
            }
        } else {
            if (v3){
                return Tuple9{v2, v3};
            } else {
                return Tuple9{v0, v1};
            }
        }
    }
};
struct Tuple10 {
    float v0;
    int v1;
    __device__ Tuple10() = default;
    __device__ Tuple10(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
    __device__ Tuple10 operator()(Tuple10 tup0, Tuple10 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple10{v0, v1};
        } else {
            return Tuple10{v2, v3};
        }
    }
};
struct Tuple11 {
    int v0;
    bool v1;
    __device__ Tuple11() = default;
    __device__ Tuple11(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure5 {
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
struct Closure6 {
    int v0;
    __device__ Tuple10 operator()(Tuple10 tup0, Tuple10 tup1){
        int & v0 = this->v0;
        float v1 = tup0.v0; int v2 = tup0.v1; float v3 = tup1.v0; int v4 = tup1.v1;
        bool v5;
        v5 = v2 == v0;
        if (v5){
            return Tuple10{v1, v2};
        } else {
            bool v6;
            v6 = v4 == v0;
            if (v6){
                return Tuple10{v3, v4};
            } else {
                return Tuple10{v1, v2};
            }
        }
    }
    __device__ Closure6(int _v0) : v0(_v0) { }
};
struct Union12_0 { // AA_Call
};
struct Union12_1 { // AA_Fold
};
struct Union12_2 { // AA_Raise
};
struct Union12 {
    union {
        Union12_0 case0; // AA_Call
        Union12_1 case1; // AA_Fold
        Union12_2 case2; // AA_Raise
    };
    unsigned char tag{255};
    __device__ Union12() {}
    __device__ Union12(Union12_0 t) : tag(0), case0(t) {} // AA_Call
    __device__ Union12(Union12_1 t) : tag(1), case1(t) {} // AA_Fold
    __device__ Union12(Union12_2 t) : tag(2), case2(t) {} // AA_Raise
    __device__ Union12(Union12 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union12_0(x.case0); break; // AA_Call
            case 1: new (&this->case1) Union12_1(x.case1); break; // AA_Fold
            case 2: new (&this->case2) Union12_2(x.case2); break; // AA_Raise
        }
    }
    __device__ Union12(Union12 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union12_0(std::move(x.case0)); break; // AA_Call
            case 1: new (&this->case1) Union12_1(std::move(x.case1)); break; // AA_Fold
            case 2: new (&this->case2) Union12_2(std::move(x.case2)); break; // AA_Raise
        }
    }
    __device__ Union12 & operator=(Union12 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // AA_Call
                case 1: this->case1 = x.case1; break; // AA_Fold
                case 2: this->case2 = x.case2; break; // AA_Raise
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
                case 0: this->case0 = std::move(x.case0); break; // AA_Call
                case 1: this->case1 = std::move(x.case1); break; // AA_Fold
                case 2: this->case2 = std::move(x.case2); break; // AA_Raise
            }
        } else {
            this->~Union12();
            new (this) Union12{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union12() {
        switch(this->tag){
            case 0: this->case0.~Union12_0(); break; // AA_Call
            case 1: this->case1.~Union12_1(); break; // AA_Fold
            case 2: this->case2.~Union12_2(); break; // AA_Raise
        }
        this->tag = 255;
    }
};
struct Union13_0 { // Eq
};
struct Union13_1 { // Gt
};
struct Union13_2 { // Lt
};
struct Union13 {
    union {
        Union13_0 case0; // Eq
        Union13_1 case1; // Gt
        Union13_2 case2; // Lt
    };
    unsigned char tag{255};
    __device__ Union13() {}
    __device__ Union13(Union13_0 t) : tag(0), case0(t) {} // Eq
    __device__ Union13(Union13_1 t) : tag(1), case1(t) {} // Gt
    __device__ Union13(Union13_2 t) : tag(2), case2(t) {} // Lt
    __device__ Union13(Union13 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union13_0(x.case0); break; // Eq
            case 1: new (&this->case1) Union13_1(x.case1); break; // Gt
            case 2: new (&this->case2) Union13_2(x.case2); break; // Lt
        }
    }
    __device__ Union13(Union13 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union13_0(std::move(x.case0)); break; // Eq
            case 1: new (&this->case1) Union13_1(std::move(x.case1)); break; // Gt
            case 2: new (&this->case2) Union13_2(std::move(x.case2)); break; // Lt
        }
    }
    __device__ Union13 & operator=(Union13 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Eq
                case 1: this->case1 = x.case1; break; // Gt
                case 2: this->case2 = x.case2; break; // Lt
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
                case 0: this->case0 = std::move(x.case0); break; // Eq
                case 1: this->case1 = std::move(x.case1); break; // Gt
                case 2: this->case2 = std::move(x.case2); break; // Lt
            }
        } else {
            this->~Union13();
            new (this) Union13{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union13() {
        switch(this->tag){
            case 0: this->case0.~Union13_0(); break; // Eq
            case 1: this->case1.~Union13_1(); break; // Gt
            case 2: this->case2.~Union13_2(); break; // Lt
        }
        this->tag = 255;
    }
};
struct StackMut1 {
    cooperative_groups::grid_group v1;
    static_array_list<Union7,32> v2;
    static_array<Union2,2> v3;
    static_array<float,2> v4;
    curandStatePhilox4_32_10_t v5;
    unsigned int v0;
    __device__ StackMut1() = default;
    __device__ StackMut1(unsigned int t0, cooperative_groups::grid_group t1, static_array_list<Union7,32> t2, static_array<Union2,2> t3, static_array<float,2> t4, curandStatePhilox4_32_10_t t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Union14_0 { // T_game_chance_community_card
    Union5 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    Union6 v6;
    int v3;
    int v5;
    bool v1;
    __device__ Union14_0(Union5 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5, Union6 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union14_0() = delete;
};
struct Union14_1 { // T_game_chance_init
    Union6 v0;
    Union6 v1;
    __device__ Union14_1(Union6 t0, Union6 t1) : v0(t0), v1(t1) {}
    __device__ Union14_1() = delete;
};
struct Union14_2 { // T_game_round
    Union5 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    Union1 v6;
    int v3;
    int v5;
    bool v1;
    __device__ Union14_2(Union5 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union14_2() = delete;
};
struct Union14_3 { // T_none
};
struct Union14 {
    union {
        Union14_0 case0; // T_game_chance_community_card
        Union14_1 case1; // T_game_chance_init
        Union14_2 case2; // T_game_round
        Union14_3 case3; // T_none
    };
    unsigned char tag{255};
    __device__ Union14() {}
    __device__ Union14(Union14_0 t) : tag(0), case0(t) {} // T_game_chance_community_card
    __device__ Union14(Union14_1 t) : tag(1), case1(t) {} // T_game_chance_init
    __device__ Union14(Union14_2 t) : tag(2), case2(t) {} // T_game_round
    __device__ Union14(Union14_3 t) : tag(3), case3(t) {} // T_none
    __device__ Union14(Union14 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union14_0(x.case0); break; // T_game_chance_community_card
            case 1: new (&this->case1) Union14_1(x.case1); break; // T_game_chance_init
            case 2: new (&this->case2) Union14_2(x.case2); break; // T_game_round
            case 3: new (&this->case3) Union14_3(x.case3); break; // T_none
        }
    }
    __device__ Union14(Union14 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union14_0(std::move(x.case0)); break; // T_game_chance_community_card
            case 1: new (&this->case1) Union14_1(std::move(x.case1)); break; // T_game_chance_init
            case 2: new (&this->case2) Union14_2(std::move(x.case2)); break; // T_game_round
            case 3: new (&this->case3) Union14_3(std::move(x.case3)); break; // T_none
        }
    }
    __device__ Union14 & operator=(Union14 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // T_game_chance_community_card
                case 1: this->case1 = x.case1; break; // T_game_chance_init
                case 2: this->case2 = x.case2; break; // T_game_round
                case 3: this->case3 = x.case3; break; // T_none
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
                case 0: this->case0 = std::move(x.case0); break; // T_game_chance_community_card
                case 1: this->case1 = std::move(x.case1); break; // T_game_chance_init
                case 2: this->case2 = std::move(x.case2); break; // T_game_round
                case 3: this->case3 = std::move(x.case3); break; // T_none
            }
        } else {
            this->~Union14();
            new (this) Union14{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union14() {
        switch(this->tag){
            case 0: this->case0.~Union14_0(); break; // T_game_chance_community_card
            case 1: this->case1.~Union14_1(); break; // T_game_chance_init
            case 2: this->case2.~Union14_2(); break; // T_game_round
            case 3: this->case3.~Union14_3(); break; // T_none
        }
        this->tag = 255;
    }
};
struct Tuple12 {
    double v1;
    int v0;
    __device__ Tuple12() = default;
    __device__ Tuple12(int t0, double t1) : v0(t0), v1(t1) {}
};
struct Tuple13 {
    double v2;
    float v1;
    int v0;
    __device__ Tuple13() = default;
    __device__ Tuple13(int t0, float t1, double t2) : v0(t0), v1(t1), v2(t2) {}
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
__device__ unsigned int f_7(unsigned char * v0){
    unsigned int * v1;
    v1 = (unsigned int *)(v0+0ull);
    unsigned int v3;
    v3 = v1[0];
    return v3;
}
__device__ int f_8(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+4ull);
    int v3;
    v3 = v1[0];
    return v3;
}
__device__ Union6 f_11(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    switch (v1) {
        case 0: {
            f_3(v2);
            return Union6{Union6_0{}};
            break;
        }
        case 1: {
            f_3(v2);
            return Union6{Union6_1{}};
            break;
        }
        case 2: {
            f_3(v2);
            return Union6{Union6_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
}
__device__ Tuple1 f_10(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    Union5 v8;
    switch (v1) {
        case 0: {
            f_3(v2);
            v8 = Union5{Union5_0{}};
            break;
        }
        case 1: {
            Union6 v6;
            v6 = f_11(v2);
            v8 = Union5{Union5_1{v6}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
    bool * v9;
    v9 = (bool *)(v0+8ull);
    bool v11;
    v11 = v9[0];
    static_array<Union6,2> v12;
    int v14;
    v14 = 0;
    while (while_method_0(v14)){
        unsigned long long v16;
        v16 = (unsigned long long)v14;
        unsigned long long v17;
        v17 = v16 * 4ull;
        unsigned long long v18;
        v18 = 12ull + v17;
        unsigned char * v19;
        v19 = (unsigned char *)(v0+v18);
        Union6 v21;
        v21 = f_11(v19);
        v12[v14] = v21;
        v14 += 1 ;
    }
    int * v22;
    v22 = (int *)(v0+20ull);
    int v24;
    v24 = v22[0];
    static_array<int,2> v25;
    int v27;
    v27 = 0;
    while (while_method_0(v27)){
        unsigned long long v29;
        v29 = (unsigned long long)v27;
        unsigned long long v30;
        v30 = v29 * 4ull;
        unsigned long long v31;
        v31 = 24ull + v30;
        unsigned char * v32;
        v32 = (unsigned char *)(v0+v31);
        int v34;
        v34 = f_1(v32);
        v25[v27] = v34;
        v27 += 1 ;
    }
    int * v35;
    v35 = (int *)(v0+32ull);
    int v37;
    v37 = v35[0];
    return Tuple1{v8, v11, v12, v24, v25, v37};
}
__device__ int f_13(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+36ull);
    int v3;
    v3 = v1[0];
    return v3;
}
__device__ Tuple2 f_12(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    Union5 v8;
    switch (v1) {
        case 0: {
            f_3(v2);
            v8 = Union5{Union5_0{}};
            break;
        }
        case 1: {
            Union6 v6;
            v6 = f_11(v2);
            v8 = Union5{Union5_1{v6}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
    bool * v9;
    v9 = (bool *)(v0+8ull);
    bool v11;
    v11 = v9[0];
    static_array<Union6,2> v12;
    int v14;
    v14 = 0;
    while (while_method_0(v14)){
        unsigned long long v16;
        v16 = (unsigned long long)v14;
        unsigned long long v17;
        v17 = v16 * 4ull;
        unsigned long long v18;
        v18 = 12ull + v17;
        unsigned char * v19;
        v19 = (unsigned char *)(v0+v18);
        Union6 v21;
        v21 = f_11(v19);
        v12[v14] = v21;
        v14 += 1 ;
    }
    int * v22;
    v22 = (int *)(v0+20ull);
    int v24;
    v24 = v22[0];
    static_array<int,2> v25;
    int v27;
    v27 = 0;
    while (while_method_0(v27)){
        unsigned long long v29;
        v29 = (unsigned long long)v27;
        unsigned long long v30;
        v30 = v29 * 4ull;
        unsigned long long v31;
        v31 = 24ull + v30;
        unsigned char * v32;
        v32 = (unsigned char *)(v0+v31);
        int v34;
        v34 = f_1(v32);
        v25[v27] = v34;
        v27 += 1 ;
    }
    int * v35;
    v35 = (int *)(v0+32ull);
    int v37;
    v37 = v35[0];
    int v38;
    v38 = f_13(v0);
    unsigned char * v39;
    v39 = (unsigned char *)(v0+40ull);
    Union1 v45;
    switch (v38) {
        case 0: {
            f_3(v39);
            v45 = Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v39);
            v45 = Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v39);
            v45 = Union1{Union1_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
    return Tuple2{v8, v11, v12, v24, v25, v37, v45};
}
__device__ Union4 f_9(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            Union5 v5; bool v6; static_array<Union6,2> v7; int v8; static_array<int,2> v9; int v10;
            Tuple1 tmp0 = f_10(v2);
            v5 = tmp0.v0; v6 = tmp0.v1; v7 = tmp0.v2; v8 = tmp0.v3; v9 = tmp0.v4; v10 = tmp0.v5;
            return Union4{Union4_0{v5, v6, v7, v8, v9, v10}};
            break;
        }
        case 1: {
            f_3(v2);
            return Union4{Union4_1{}};
            break;
        }
        case 2: {
            Union5 v13; bool v14; static_array<Union6,2> v15; int v16; static_array<int,2> v17; int v18;
            Tuple1 tmp1 = f_10(v2);
            v13 = tmp1.v0; v14 = tmp1.v1; v15 = tmp1.v2; v16 = tmp1.v3; v17 = tmp1.v4; v18 = tmp1.v5;
            return Union4{Union4_2{v13, v14, v15, v16, v17, v18}};
            break;
        }
        case 3: {
            Union5 v20; bool v21; static_array<Union6,2> v22; int v23; static_array<int,2> v24; int v25; Union1 v26;
            Tuple2 tmp2 = f_12(v2);
            v20 = tmp2.v0; v21 = tmp2.v1; v22 = tmp2.v2; v23 = tmp2.v3; v24 = tmp2.v4; v25 = tmp2.v5; v26 = tmp2.v6;
            return Union4{Union4_3{v20, v21, v22, v23, v24, v25, v26}};
            break;
        }
        case 4: {
            Union5 v28; bool v29; static_array<Union6,2> v30; int v31; static_array<int,2> v32; int v33;
            Tuple1 tmp3 = f_10(v2);
            v28 = tmp3.v0; v29 = tmp3.v1; v30 = tmp3.v2; v31 = tmp3.v3; v32 = tmp3.v4; v33 = tmp3.v5;
            return Union4{Union4_4{v28, v29, v30, v31, v32, v33}};
            break;
        }
        case 5: {
            Union5 v35; bool v36; static_array<Union6,2> v37; int v38; static_array<int,2> v39; int v40;
            Tuple1 tmp4 = f_10(v2);
            v35 = tmp4.v0; v36 = tmp4.v1; v37 = tmp4.v2; v38 = tmp4.v3; v39 = tmp4.v4; v40 = tmp4.v5;
            return Union4{Union4_5{v35, v36, v37, v38, v39, v40}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
}
__device__ int f_14(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+80ull);
    int v3;
    v3 = v1[0];
    return v3;
}
__device__ inline bool while_method_1(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ Tuple3 f_16(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0];
    int v4;
    v4 = f_8(v0);
    unsigned char * v5;
    v5 = (unsigned char *)(v0+8ull);
    Union1 v11;
    switch (v4) {
        case 0: {
            f_3(v5);
            v11 = Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v5);
            v11 = Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v5);
            v11 = Union1{Union1_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
    return Tuple3{v3, v11};
}
__device__ Tuple4 f_17(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0];
    int v4;
    v4 = f_8(v0);
    unsigned char * v5;
    v5 = (unsigned char *)(v0+8ull);
    Union6 v11;
    switch (v4) {
        case 0: {
            f_3(v5);
            v11 = Union6{Union6_0{}};
            break;
        }
        case 1: {
            f_3(v5);
            v11 = Union6{Union6_1{}};
            break;
        }
        case 2: {
            f_3(v5);
            v11 = Union6{Union6_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
    return Tuple4{v3, v11};
}
__device__ Tuple5 f_18(unsigned char * v0){
    static_array<Union6,2> v1;
    int v3;
    v3 = 0;
    while (while_method_0(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned long long v6;
        v6 = v5 * 4ull;
        unsigned char * v7;
        v7 = (unsigned char *)(v0+v6);
        Union6 v9;
        v9 = f_11(v7);
        v1[v3] = v9;
        v3 += 1 ;
    }
    int * v10;
    v10 = (int *)(v0+8ull);
    int v12;
    v12 = v10[0];
    int * v13;
    v13 = (int *)(v0+12ull);
    int v15;
    v15 = v13[0];
    return Tuple5{v1, v12, v15};
}
__device__ Union7 f_15(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            Union6 v5;
            v5 = f_11(v2);
            return Union7{Union7_0{v5}};
            break;
        }
        case 1: {
            int v7; Union1 v8;
            Tuple3 tmp5 = f_16(v2);
            v7 = tmp5.v0; v8 = tmp5.v1;
            return Union7{Union7_1{v7, v8}};
            break;
        }
        case 2: {
            int v10; Union6 v11;
            Tuple4 tmp6 = f_17(v2);
            v10 = tmp6.v0; v11 = tmp6.v1;
            return Union7{Union7_2{v10, v11}};
            break;
        }
        case 3: {
            static_array<Union6,2> v13; int v14; int v15;
            Tuple5 tmp7 = f_18(v2);
            v13 = tmp7.v0; v14 = tmp7.v1; v15 = tmp7.v2;
            return Union7{Union7_3{v13, v14, v15}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
}
__device__ int f_19(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+1128ull);
    int v3;
    v3 = v1[0];
    return v3;
}
__device__ Tuple0 f_6(unsigned char * v0){
    unsigned int v1;
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
    static_array_list<Union7,32> v10;
    v10 = static_array_list<Union7,32>{};
    int v12;
    v12 = f_14(v0);
    v10.unsafe_set_length(v12);
    int v13;
    v13 = v10.length;
    int v14;
    v14 = 0;
    while (while_method_1(v13, v14)){
        unsigned long long v16;
        v16 = (unsigned long long)v14;
        unsigned long long v17;
        v17 = v16 * 32ull;
        unsigned long long v18;
        v18 = 96ull + v17;
        unsigned char * v19;
        v19 = (unsigned char *)(v0+v18);
        Union7 v21;
        v21 = f_15(v19);
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
        v28 = 1120ull + v27;
        unsigned char * v29;
        v29 = (unsigned char *)(v0+v28);
        Union2 v31;
        v31 = f_5(v29);
        v22[v24] = v31;
        v24 += 1 ;
    }
    int v32;
    v32 = f_19(v0);
    unsigned char * v33;
    v33 = (unsigned char *)(v0+1136ull);
    Union8 v51;
    switch (v32) {
        case 0: {
            f_3(v33);
            v51 = Union8{Union8_0{}};
            break;
        }
        case 1: {
            Union5 v37; bool v38; static_array<Union6,2> v39; int v40; static_array<int,2> v41; int v42;
            Tuple1 tmp8 = f_10(v33);
            v37 = tmp8.v0; v38 = tmp8.v1; v39 = tmp8.v2; v40 = tmp8.v3; v41 = tmp8.v4; v42 = tmp8.v5;
            v51 = Union8{Union8_1{v37, v38, v39, v40, v41, v42}};
            break;
        }
        case 2: {
            Union5 v44; bool v45; static_array<Union6,2> v46; int v47; static_array<int,2> v48; int v49;
            Tuple1 tmp9 = f_10(v33);
            v44 = tmp9.v0; v45 = tmp9.v1; v46 = tmp9.v2; v47 = tmp9.v3; v48 = tmp9.v4; v49 = tmp9.v5;
            v51 = Union8{Union8_2{v44, v45, v46, v47, v48, v49}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
    return Tuple0{v1, v9, v10, v22, v51};
}
__device__ inline bool while_method_2(Union3 v0){
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
__device__ unsigned int loop_21(unsigned int v0, curandStatePhilox4_32_10_t & v1){
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
        return loop_21(v0, v1);
    }
}
__device__ Tuple6 draw_card_20(curandStatePhilox4_32_10_t & v0, unsigned int v1){
    int v2;
    v2 = __popc(v1);
    unsigned int v3;
    v3 = (unsigned int)v2;
    unsigned int v4;
    v4 = loop_21(v3, v0);
    int v5;
    v5 = (int)v4;
    int v6;
    v6 = __popc(v1);
    bool v7;
    v7 = v5 < v6;
    unsigned int v12;
    if (v7){
        int v8;
        v8 = v5 + 1;
        unsigned int v9;
        v9 = __fns(v1,0u,v8);
        v12 = v9;
    } else {
        int v10;
        v10 = v5 - v6;
        printf("%s\n", "Cannot find the n-th set bit.");
        __trap();
    }
    bool v13;
    v13 = 0u == v12;
    Union6 v31;
    if (v13){
        v31 = Union6{Union6_1{}};
    } else {
        bool v15;
        v15 = 1u == v12;
        if (v15){
            v31 = Union6{Union6_1{}};
        } else {
            bool v17;
            v17 = 2u == v12;
            if (v17){
                v31 = Union6{Union6_2{}};
            } else {
                bool v19;
                v19 = 3u == v12;
                if (v19){
                    v31 = Union6{Union6_2{}};
                } else {
                    bool v21;
                    v21 = 4u == v12;
                    if (v21){
                        v31 = Union6{Union6_0{}};
                    } else {
                        bool v23;
                        v23 = 5u == v12;
                        if (v23){
                            v31 = Union6{Union6_0{}};
                        } else {
                            printf("%s\n", "Invalid int in int_to_card.");
                            __trap();
                        }
                    }
                }
            }
        }
    }
    int v32;
    v32 = (int)v12;
    unsigned int v33;
    v33 = 1u << v32;
    unsigned int v34;
    v34 = v1 ^ v33;
    return Tuple6{v31, v34};
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 32768;
    return v1;
}
__device__ int int_range_22(int v0, int v1, curandStatePhilox4_32_10_t & v2){
    int v3;
    v3 = v0 - v1;
    unsigned int v4;
    v4 = (unsigned int)v3;
    unsigned int v5;
    v5 = loop_21(v4, v2);
    unsigned int v6;
    v6 = (unsigned int)v1;
    unsigned int v7;
    v7 = v5 + v6;
    int v8;
    v8 = (int)v7;
    return v8;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 1;
    return v1;
}
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 8;
    return v1;
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 4;
    return v1;
}
__device__ inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 4;
    return v1;
}
__device__ inline bool while_method_8(int v0){
    bool v1;
    v1 = v0 < 16;
    return v1;
}
__device__ void block_matmul_23(float * v0, float * v1, int v2, float * v3){
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 24);
    int v5;
    v5 = 32768 * v4;
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
    while (while_method_4(v64)){
        int v66;
        v66 = 0;
        while (while_method_4(v66)){
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
            while (while_method_5(v74)){
                int v76;
                v76 = 0;
                #pragma unroll
                while (while_method_4(v76)){
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
            while (while_method_6(v80)){
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
                v88 = v80 < 4;
                bool v89;
                v89 = v88 == false;
                if (v89){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v88);
                } else {
                }
                bool v91;
                v91 = v82 < 4;
                Union11 v97;
                if (v91){
                    bool v92;
                    v92 = 0 <= v82;
                    bool v93;
                    v93 = v92 == false;
                    if (v93){
                        assert("The index needs to be zero or positive." && v92);
                    } else {
                    }
                    v97 = Union11{Union11_1{v82}};
                } else {
                    v97 = Union11{Union11_0{}};
                }
                assert("Tensor range check" && 0 <= v64 && v64 < 1);
                int v98;
                v98 = 32768 * v64;
                int v99;
                v99 = v98 + v5;
                assert("Tensor range check" && 0 <= v80 && v80 < 4);
                int v100;
                v100 = 32 * v80;
                int v101;
                v101 = v100 + v99;
                float * v102;
                v102 = v3+v101;
                assert("Tensor range check" && 0 <= v66 && v66 < 1);
                int v104;
                v104 = 8192 * v66;
                int v105;
                v105 = v104 + v2;
                if (v83){
                    assert("Tensor range check" && 0 <= v80 && v80 < 4);
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
                    v121 = 128 * v114;
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
                        while (while_method_4(v129)){
                            assert("Tensor range check" && 0 <= v127 && v127 < 2);
                            assert("Tensor range check" && 0 <= v129 && v129 < 1);
                            int v131;
                            v131 = 32 * v129;
                            int v132;
                            v132 = 1152 * v127;
                            int v133;
                            v133 = v132 + v131;
                            int v134;
                            v134 = 4096 * v127;
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
                v149 = 128 * v142;
                int v150;
                v150 = v149 + v146;
                float * v151;
                v151 = v10+v148;
                float * v153;
                v153 = v102+v150;
                int v155;
                v155 = 0;
                #pragma unroll
                while (while_method_5(v155)){
                    int v157;
                    v157 = 0;
                    #pragma unroll
                    while (while_method_4(v157)){
                        assert("Tensor range check" && 0 <= v155 && v155 < 8);
                        assert("Tensor range check" && 0 <= v157 && v157 < 1);
                        int v159;
                        v159 = 32 * v157;
                        int v160;
                        v160 = 1152 * v155;
                        int v161;
                        v161 = v160 + v159;
                        int v162;
                        v162 = 4096 * v155;
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
                while (while_method_4(v168)){
                    int v170;
                    v170 = 0;
                    #pragma unroll
                    while (while_method_7(v170)){
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
                        assert("Tensor range check" && 0 <= v199 && v199 < 4);
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
                        v216 = 128 * v209;
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
                            while (while_method_4(v224)){
                                assert("Tensor range check" && 0 <= v222 && v222 < 2);
                                assert("Tensor range check" && 0 <= v224 && v224 < 1);
                                int v226;
                                v226 = 32 * v224;
                                int v227;
                                v227 = 1152 * v222;
                                int v228;
                                v228 = v227 + v226;
                                int v229;
                                v229 = 4096 * v222;
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
                while (while_method_5(v232)){
                    int v234;
                    v234 = 0;
                    #pragma unroll
                    while (while_method_7(v234)){
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
                        while (while_method_4(v261)){
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
            while (while_method_5(v268)){
                int v270;
                v270 = 0;
                #pragma unroll
                while (while_method_4(v270)){
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
            while (while_method_8(v297)){
                int v299;
                v299 = 0;
                #pragma unroll
                while (while_method_4(v299)){
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
__device__ inline bool while_method_9(int v0){
    bool v1;
    v1 = v0 < 4096;
    return v1;
}
__device__ void block_map_24(float * v0, int v1, float * v2){
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
    int v9;
    v9 = v8;
    while (while_method_9(v9)){
        bool v11;
        v11 = 0 <= v9;
        bool v12;
        v12 = v11 == false;
        if (v12){
            assert("The index needs to be zero or positive." && v11);
        } else {
        }
        int v14;
        v14 = v9 % 16;
        int v15;
        v15 = v9 / 16;
        bool v16;
        v16 = v15 < 256;
        bool v17;
        v17 = v16 == false;
        if (v17){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v16);
        } else {
        }
        assert("Tensor range check" && 0 <= v15 && v15 < 256);
        assert("Tensor range check" && 0 <= v14 && v14 < 16);
        int v19;
        v19 = 4 * v14;
        int v20;
        v20 = v19 + v4;
        int v21;
        v21 = 64 * v15;
        int v22;
        v22 = v21 + v20;
        assert("Tensor range check" && 0 <= v15 && v15 < 256);
        assert("Tensor range check" && 0 <= v14 && v14 < 16);
        int v23;
        v23 = v19 + v7;
        int v24;
        v24 = v21 + v23;
        float v25[4];
        float v26[4];
        int4* v27;
        v27 = reinterpret_cast<int4*>(v2 + v22);
        int4* v28;
        v28 = reinterpret_cast<int4*>(v25 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v27) % 16 == 0 && reinterpret_cast<unsigned long long>(v28) % 16 == 0);
        *v28 = *v27;
        // Pushing the loop unrolling to: 0
        int v29;
        v29 = 0;
        #pragma unroll
        while (while_method_7(v29)){
            assert("Tensor range check" && 0 <= v29 && v29 < 4);
            float v31;
            v31 = v25[v29];
            assert("Tensor range check" && 0 <= v29 && v29 < 4);
            v26[v29] = v31;
            v29 += 1 ;
        }
        // Poping the loop unrolling to: 0
        int4* v32;
        v32 = reinterpret_cast<int4*>(v26 + 0);
        int4* v33;
        v33 = reinterpret_cast<int4*>(v0 + v24);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v32) % 16 == 0 && reinterpret_cast<unsigned long long>(v33) % 16 == 0);
        *v33 = *v32;
        v9 += 256 ;
    }
    __syncthreads();
    return ;
}
__device__ void block_row_map_25(float * v0, int v1, float * v2){
    int v3;
    v3 = blockIdx.x;
    assert("Tensor range check" && 0 <= v3 && v3 < 24);
    int v4;
    v4 = 16384 * v3;
    int v5;
    v5 = v4 + v1;
    int v6;
    v6 = blockIdx.x;
    assert("Tensor range check" && 0 <= v6 && v6 < 24);
    int v7;
    v7 = 16384 * v6;
    int v8;
    v8 = v7 + v1;
    int v9;
    v9 = threadIdx.x;
    bool v10;
    v10 = 0 <= v9;
    bool v11;
    v11 = v10 == false;
    if (v11){
        assert("The index needs to be zero or positive." && v10);
    } else {
    }
    int v13;
    v13 = v9 % 16;
    int v14;
    v14 = v9 / 16;
    bool v15;
    v15 = v14 < 16;
    bool v16;
    v16 = v15 == false;
    if (v16){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v15);
    } else {
    }
    assert("Tensor range check" && 0 <= v14 && v14 < 16);
    assert("Tensor range check" && 0 <= v13 && v13 < 16);
    int v18;
    v18 = 4 * v13;
    int v19;
    v19 = v18 + v5;
    int v20;
    v20 = 64 * v14;
    int v21;
    v21 = v20 + v19;
    assert("Tensor range check" && 0 <= v14 && v14 < 16);
    assert("Tensor range check" && 0 <= v13 && v13 < 16);
    int v22;
    v22 = v18 + v8;
    int v23;
    v23 = v20 + v22;
    int v24;
    v24 = 0;
    while (while_method_8(v24)){
        assert("Tensor range check" && 0 <= v24 && v24 < 16);
        int v26;
        v26 = 1024 * v24;
        int v27;
        v27 = v26 + v21;
        float v28[4];
        int v29[4];
        int v30;
        v30 = 0;
        while (while_method_4(v30)){
            assert("Tensor range check" && 0 <= v30 && v30 < 1);
            int v32;
            v32 = 4 * v30;
            assert("Tensor range check" && 0 <= v30 && v30 < 1);
            int v33;
            v33 = 64 * v30;
            int v34;
            v34 = v33 + v27;
            int4* v35;
            v35 = reinterpret_cast<int4*>(v2 + v34);
            int4* v36;
            v36 = reinterpret_cast<int4*>(v28 + v32);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v35) % 16 == 0 && reinterpret_cast<unsigned long long>(v36) % 16 == 0);
            *v36 = *v35;
            v30 += 1 ;
        }
        int v37;
        v37 = 0;
        while (while_method_4(v37)){
            int v39;
            v39 = 0;
            while (while_method_7(v39)){
                bool v41;
                v41 = 0 <= v39;
                bool v43;
                if (v41){
                    bool v42;
                    v42 = v39 < 4;
                    v43 = v42;
                } else {
                    v43 = false;
                }
                bool v44;
                v44 = v43 == false;
                if (v44){
                    assert("The indices should be inside the range of the dimension." && v43);
                } else {
                }
                bool v46;
                v46 = 0 <= v13;
                bool v48;
                if (v46){
                    bool v47;
                    v47 = v13 < 16;
                    v48 = v47;
                } else {
                    v48 = false;
                }
                bool v49;
                v49 = v48 == false;
                if (v49){
                    assert("The indices should be inside the range of the dimension." && v48);
                } else {
                }
                int v51;
                v51 = v13 * 4;
                int v52;
                v52 = v39 + v51;
                bool v53;
                v53 = 0 <= v37;
                bool v55;
                if (v53){
                    bool v54;
                    v54 = v37 < 1;
                    v55 = v54;
                } else {
                    v55 = false;
                }
                bool v56;
                v56 = v55 == false;
                if (v56){
                    assert("The indices should be inside the range of the dimension." && v55);
                } else {
                }
                int v58;
                v58 = v37 * 64;
                int v59;
                v59 = v52 + v58;
                assert("Tensor range check" && 0 <= v37 && v37 < 1);
                assert("Tensor range check" && 0 <= v39 && v39 < 4);
                int v60;
                v60 = 4 * v37;
                int v61;
                v61 = v60 + v39;
                v29[v61] = v59;
                v39 += 1 ;
            }
            v37 += 1 ;
        }
        bool v62;
        v62 = 0 <= v14;
        bool v63;
        v63 = v62 && v15;
        bool v64;
        v64 = v63 == false;
        if (v64){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v63);
        } else {
        }
        bool v66;
        v66 = 0 <= v24;
        bool v68;
        if (v66){
            bool v67;
            v67 = v24 < 16;
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
        v71 = v24 * 16;
        int v72;
        v72 = v71 + v14;
        bool v73[4];
        int v74;
        v74 = 0;
        while (while_method_4(v74)){
            int v76;
            v76 = 0;
            while (while_method_7(v76)){
                assert("Tensor range check" && 0 <= v74 && v74 < 1);
                assert("Tensor range check" && 0 <= v76 && v76 < 4);
                int v78;
                v78 = 4 * v74;
                int v79;
                v79 = v78 + v76;
                float v80;
                v80 = v28[v79];
                int v81;
                v81 = v29[v79];
                bool v82;
                v82 = v81 < 3;
                assert("Tensor range check" && 0 <= v74 && v74 < 1);
                assert("Tensor range check" && 0 <= v76 && v76 < 4);
                v73[v79] = v82;
                v76 += 1 ;
            }
            v74 += 1 ;
        }
        float v83[4];
        int v84;
        v84 = 0;
        while (while_method_4(v84)){
            int v86;
            v86 = 0;
            while (while_method_7(v86)){
                assert("Tensor range check" && 0 <= v84 && v84 < 1);
                assert("Tensor range check" && 0 <= v86 && v86 < 4);
                int v88;
                v88 = 4 * v84;
                int v89;
                v89 = v88 + v86;
                float v90;
                v90 = v28[v89];
                bool v91;
                v91 = v73[v89];
                float v92;
                if (v91){
                    v92 = v90;
                } else {
                    v92 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v84 && v84 < 1);
                assert("Tensor range check" && 0 <= v86 && v86 < 4);
                v83[v89] = v92;
                v86 += 1 ;
            }
            v84 += 1 ;
        }
        float v93;
        v93 = 0.0f;
        int v94;
        v94 = 0;
        while (while_method_4(v94)){
            int v96;
            v96 = 0;
            while (while_method_7(v96)){
                assert("Tensor range check" && 0 <= v94 && v94 < 1);
                assert("Tensor range check" && 0 <= v96 && v96 < 4);
                int v98;
                v98 = 4 * v94;
                int v99;
                v99 = v98 + v96;
                float v100;
                v100 = v83[v99];
                float v101;
                v101 = v93 + v100;
                v93 = v101;
                v96 += 1 ;
            }
            v94 += 1 ;
        }
        auto v102 = cooperative_groups::coalesced_threads();
        int v103;
        v103 = threadIdx.x;
        int v104;
        v104 = v103 / 16;
        auto v105 = cooperative_groups::labeled_partition(v102,v104);
        Closure0 v106{};
        float v107;
        v107 = cooperative_groups::reduce(v105, v93, v106);
        int v108[4];
        int v109;
        v109 = 0;
        while (while_method_4(v109)){
            int v111;
            v111 = 0;
            while (while_method_7(v111)){
                assert("Tensor range check" && 0 <= v109 && v109 < 1);
                assert("Tensor range check" && 0 <= v111 && v111 < 4);
                int v113;
                v113 = 4 * v109;
                int v114;
                v114 = v113 + v111;
                bool v115;
                v115 = v73[v114];
                int v116;
                if (v115){
                    v116 = 1;
                } else {
                    v116 = 0;
                }
                assert("Tensor range check" && 0 <= v109 && v109 < 1);
                assert("Tensor range check" && 0 <= v111 && v111 < 4);
                v108[v114] = v116;
                v111 += 1 ;
            }
            v109 += 1 ;
        }
        int v117;
        v117 = 0;
        int v118;
        v118 = 0;
        while (while_method_4(v118)){
            int v120;
            v120 = 0;
            while (while_method_7(v120)){
                assert("Tensor range check" && 0 <= v118 && v118 < 1);
                assert("Tensor range check" && 0 <= v120 && v120 < 4);
                int v122;
                v122 = 4 * v118;
                int v123;
                v123 = v122 + v120;
                int v124;
                v124 = v108[v123];
                int v125;
                v125 = v117 + v124;
                v117 = v125;
                v120 += 1 ;
            }
            v118 += 1 ;
        }
        auto v126 = cooperative_groups::coalesced_threads();
        int v127;
        v127 = threadIdx.x;
        int v128;
        v128 = v127 / 16;
        auto v129 = cooperative_groups::labeled_partition(v126,v128);
        Closure1 v130{};
        int v131;
        v131 = cooperative_groups::reduce(v129, v117, v130);
        float v132;
        v132 = (float)v131;
        float v133;
        v133 = v107 / v132;
        float v134[4];
        int v135;
        v135 = 0;
        while (while_method_4(v135)){
            int v137;
            v137 = 0;
            while (while_method_7(v137)){
                assert("Tensor range check" && 0 <= v135 && v135 < 1);
                assert("Tensor range check" && 0 <= v137 && v137 < 4);
                int v139;
                v139 = 4 * v135;
                int v140;
                v140 = v139 + v137;
                float v141;
                v141 = v28[v140];
                bool v142;
                v142 = v73[v140];
                float v143;
                if (v142){
                    v143 = v141;
                } else {
                    v143 = -1.0f / 0.0f;
                }
                float v144;
                v144 = v143 - v133;
                float v145;
                v145 = exp(v144);
                assert("Tensor range check" && 0 <= v135 && v135 < 1);
                assert("Tensor range check" && 0 <= v137 && v137 < 4);
                v134[v140] = v145;
                v137 += 1 ;
            }
            v135 += 1 ;
        }
        float v146;
        v146 = 0.0f;
        int v147;
        v147 = 0;
        while (while_method_4(v147)){
            int v149;
            v149 = 0;
            while (while_method_7(v149)){
                assert("Tensor range check" && 0 <= v147 && v147 < 1);
                assert("Tensor range check" && 0 <= v149 && v149 < 4);
                int v151;
                v151 = 4 * v147;
                int v152;
                v152 = v151 + v149;
                float v153;
                v153 = v134[v152];
                float v154;
                v154 = v146 + v153;
                v146 = v154;
                v149 += 1 ;
            }
            v147 += 1 ;
        }
        auto v155 = cooperative_groups::coalesced_threads();
        int v156;
        v156 = threadIdx.x;
        int v157;
        v157 = v156 / 16;
        auto v158 = cooperative_groups::labeled_partition(v155,v157);
        float v159;
        v159 = cooperative_groups::reduce(v158, v146, v106);
        float v160[4];
        int v161;
        v161 = 0;
        while (while_method_4(v161)){
            int v163;
            v163 = 0;
            while (while_method_7(v163)){
                assert("Tensor range check" && 0 <= v161 && v161 < 1);
                assert("Tensor range check" && 0 <= v163 && v163 < 4);
                int v165;
                v165 = 4 * v161;
                int v166;
                v166 = v165 + v163;
                float v167;
                v167 = v134[v166];
                float v168;
                v168 = v167 / v159;
                assert("Tensor range check" && 0 <= v161 && v161 < 1);
                assert("Tensor range check" && 0 <= v163 && v163 < 4);
                v160[v166] = v168;
                v163 += 1 ;
            }
            v161 += 1 ;
        }
        assert("Tensor range check" && 0 <= v24 && v24 < 16);
        int v169;
        v169 = v26 + v23;
        int v170;
        v170 = 0;
        while (while_method_4(v170)){
            assert("Tensor range check" && 0 <= v170 && v170 < 1);
            int v172;
            v172 = 64 * v170;
            int v173;
            v173 = v172 + v169;
            assert("Tensor range check" && 0 <= v170 && v170 < 1);
            int v174;
            v174 = 4 * v170;
            int4* v175;
            v175 = reinterpret_cast<int4*>(v160 + v174);
            int4* v176;
            v176 = reinterpret_cast<int4*>(v0 + v173);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v175) % 16 == 0 && reinterpret_cast<unsigned long long>(v176) % 16 == 0);
            *v176 = *v175;
            v170 += 1 ;
        }
        v24 += 1 ;
    }
    __syncthreads();
    return ;
}
__device__ int tag_27(Union6 v0){
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
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ bool is_pair_28(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
__device__ Tuple7 order_29(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple7{v1, v0};
    } else {
        return Tuple7{v0, v1};
    }
}
__device__ Union13 compare_hands_26(Union5 v0, bool v1, static_array<Union6,2> v2, int v3, static_array<int,2> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            __trap();
            break;
        }
        case 1: { // Some
            Union6 v7 = v0.case1.v0;
            int v8;
            v8 = tag_27(v7);
            Union6 v9;
            v9 = v2[0];
            int v11;
            v11 = tag_27(v9);
            Union6 v12;
            v12 = v2[1];
            int v14;
            v14 = tag_27(v12);
            bool v15;
            v15 = is_pair_28(v8, v11);
            bool v16;
            v16 = is_pair_28(v8, v14);
            if (v15){
                if (v16){
                    bool v17;
                    v17 = v11 < v14;
                    if (v17){
                        return Union13{Union13_2{}};
                    } else {
                        bool v19;
                        v19 = v11 > v14;
                        if (v19){
                            return Union13{Union13_1{}};
                        } else {
                            return Union13{Union13_0{}};
                        }
                    }
                } else {
                    return Union13{Union13_1{}};
                }
            } else {
                if (v16){
                    return Union13{Union13_2{}};
                } else {
                    int v27; int v28;
                    Tuple7 tmp35 = order_29(v8, v11);
                    v27 = tmp35.v0; v28 = tmp35.v1;
                    int v29; int v30;
                    Tuple7 tmp36 = order_29(v8, v14);
                    v29 = tmp36.v0; v30 = tmp36.v1;
                    bool v31;
                    v31 = v27 < v29;
                    Union13 v37;
                    if (v31){
                        v37 = Union13{Union13_2{}};
                    } else {
                        bool v33;
                        v33 = v27 > v29;
                        if (v33){
                            v37 = Union13{Union13_1{}};
                        } else {
                            v37 = Union13{Union13_0{}};
                        }
                    }
                    bool v38;
                    switch (v37.tag) {
                        case 0: { // Eq
                            v38 = true;
                            break;
                        }
                        default: {
                            v38 = false;
                        }
                    }
                    if (v38){
                        bool v39;
                        v39 = v28 < v30;
                        if (v39){
                            return Union13{Union13_2{}};
                        } else {
                            bool v41;
                            v41 = v28 > v30;
                            if (v41){
                                return Union13{Union13_1{}};
                            } else {
                                return Union13{Union13_0{}};
                            }
                        }
                    } else {
                        return v37;
                    }
                }
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_31(unsigned char * v0, unsigned int v1){
    unsigned int * v2;
    v2 = (unsigned int *)(v0+0ull);
    v2[0] = v1;
    return ;
}
__device__ void f_32(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+4ull);
    v2[0] = v1;
    return ;
}
__device__ void f_33(unsigned char * v0){
    return ;
}
__device__ void f_35(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+0ull);
    v2[0] = v1;
    return ;
}
__device__ void f_37(unsigned char * v0, Union6 v1){
    int v2;
    v2 = v1.tag;
    f_35(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // Jack
            return f_33(v3);
            break;
        }
        case 1: { // King
            return f_33(v3);
            break;
        }
        case 2: { // Queen
            return f_33(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_36(unsigned char * v0, Union5 v1, bool v2, static_array<Union6,2> v3, int v4, static_array<int,2> v5, int v6){
    int v7;
    v7 = v1.tag;
    f_35(v0, v7);
    unsigned char * v8;
    v8 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // None
            f_33(v8);
            break;
        }
        case 1: { // Some
            Union6 v10 = v1.case1.v0;
            f_37(v8, v10);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    bool * v11;
    v11 = (bool *)(v0+8ull);
    v11[0] = v2;
    int v13;
    v13 = 0;
    while (while_method_0(v13)){
        unsigned long long v15;
        v15 = (unsigned long long)v13;
        unsigned long long v16;
        v16 = v15 * 4ull;
        unsigned long long v17;
        v17 = 12ull + v16;
        unsigned char * v18;
        v18 = (unsigned char *)(v0+v17);
        bool v20;
        v20 = 0 <= v13;
        bool v22;
        if (v20){
            bool v21;
            v21 = v13 < 2;
            v22 = v21;
        } else {
            v22 = false;
        }
        bool v23;
        v23 = v22 == false;
        if (v23){
            assert("Index must be in range." && v22);
        } else {
        }
        Union6 v25;
        v25 = v3[v13];
        f_37(v18, v25);
        v13 += 1 ;
    }
    int * v27;
    v27 = (int *)(v0+20ull);
    v27[0] = v4;
    int v29;
    v29 = 0;
    while (while_method_0(v29)){
        unsigned long long v31;
        v31 = (unsigned long long)v29;
        unsigned long long v32;
        v32 = v31 * 4ull;
        unsigned long long v33;
        v33 = 24ull + v32;
        unsigned char * v34;
        v34 = (unsigned char *)(v0+v33);
        bool v36;
        v36 = 0 <= v29;
        bool v38;
        if (v36){
            bool v37;
            v37 = v29 < 2;
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
        int v41;
        v41 = v5[v29];
        f_35(v34, v41);
        v29 += 1 ;
    }
    int * v43;
    v43 = (int *)(v0+32ull);
    v43[0] = v6;
    return ;
}
__device__ void f_39(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+36ull);
    v2[0] = v1;
    return ;
}
__device__ void f_38(unsigned char * v0, Union5 v1, bool v2, static_array<Union6,2> v3, int v4, static_array<int,2> v5, int v6, Union1 v7){
    int v8;
    v8 = v1.tag;
    f_35(v0, v8);
    unsigned char * v9;
    v9 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // None
            f_33(v9);
            break;
        }
        case 1: { // Some
            Union6 v11 = v1.case1.v0;
            f_37(v9, v11);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    bool * v12;
    v12 = (bool *)(v0+8ull);
    v12[0] = v2;
    int v14;
    v14 = 0;
    while (while_method_0(v14)){
        unsigned long long v16;
        v16 = (unsigned long long)v14;
        unsigned long long v17;
        v17 = v16 * 4ull;
        unsigned long long v18;
        v18 = 12ull + v17;
        unsigned char * v19;
        v19 = (unsigned char *)(v0+v18);
        bool v21;
        v21 = 0 <= v14;
        bool v23;
        if (v21){
            bool v22;
            v22 = v14 < 2;
            v23 = v22;
        } else {
            v23 = false;
        }
        bool v24;
        v24 = v23 == false;
        if (v24){
            assert("Index must be in range." && v23);
        } else {
        }
        Union6 v26;
        v26 = v3[v14];
        f_37(v19, v26);
        v14 += 1 ;
    }
    int * v28;
    v28 = (int *)(v0+20ull);
    v28[0] = v4;
    int v30;
    v30 = 0;
    while (while_method_0(v30)){
        unsigned long long v32;
        v32 = (unsigned long long)v30;
        unsigned long long v33;
        v33 = v32 * 4ull;
        unsigned long long v34;
        v34 = 24ull + v33;
        unsigned char * v35;
        v35 = (unsigned char *)(v0+v34);
        bool v37;
        v37 = 0 <= v30;
        bool v39;
        if (v37){
            bool v38;
            v38 = v30 < 2;
            v39 = v38;
        } else {
            v39 = false;
        }
        bool v40;
        v40 = v39 == false;
        if (v40){
            assert("Index must be in range." && v39);
        } else {
        }
        int v42;
        v42 = v5[v30];
        f_35(v35, v42);
        v30 += 1 ;
    }
    int * v44;
    v44 = (int *)(v0+32ull);
    v44[0] = v6;
    int v46;
    v46 = v7.tag;
    f_39(v0, v46);
    unsigned char * v47;
    v47 = (unsigned char *)(v0+40ull);
    switch (v7.tag) {
        case 0: { // Call
            return f_33(v47);
            break;
        }
        case 1: { // Fold
            return f_33(v47);
            break;
        }
        case 2: { // Raise
            return f_33(v47);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_34(unsigned char * v0, Union4 v1){
    int v2;
    v2 = v1.tag;
    f_35(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // ChanceCommunityCard
            Union5 v5 = v1.case0.v0; bool v6 = v1.case0.v1; static_array<Union6,2> v7 = v1.case0.v2; int v8 = v1.case0.v3; static_array<int,2> v9 = v1.case0.v4; int v10 = v1.case0.v5;
            return f_36(v3, v5, v6, v7, v8, v9, v10);
            break;
        }
        case 1: { // ChanceInit
            return f_33(v3);
            break;
        }
        case 2: { // Round
            Union5 v11 = v1.case2.v0; bool v12 = v1.case2.v1; static_array<Union6,2> v13 = v1.case2.v2; int v14 = v1.case2.v3; static_array<int,2> v15 = v1.case2.v4; int v16 = v1.case2.v5;
            return f_36(v3, v11, v12, v13, v14, v15, v16);
            break;
        }
        case 3: { // RoundWithAction
            Union5 v17 = v1.case3.v0; bool v18 = v1.case3.v1; static_array<Union6,2> v19 = v1.case3.v2; int v20 = v1.case3.v3; static_array<int,2> v21 = v1.case3.v4; int v22 = v1.case3.v5; Union1 v23 = v1.case3.v6;
            return f_38(v3, v17, v18, v19, v20, v21, v22, v23);
            break;
        }
        case 4: { // TerminalCall
            Union5 v24 = v1.case4.v0; bool v25 = v1.case4.v1; static_array<Union6,2> v26 = v1.case4.v2; int v27 = v1.case4.v3; static_array<int,2> v28 = v1.case4.v4; int v29 = v1.case4.v5;
            return f_36(v3, v24, v25, v26, v27, v28, v29);
            break;
        }
        case 5: { // TerminalFold
            Union5 v30 = v1.case5.v0; bool v31 = v1.case5.v1; static_array<Union6,2> v32 = v1.case5.v2; int v33 = v1.case5.v3; static_array<int,2> v34 = v1.case5.v4; int v35 = v1.case5.v5;
            return f_36(v3, v30, v31, v32, v33, v34, v35);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_40(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+80ull);
    v2[0] = v1;
    return ;
}
__device__ void f_42(unsigned char * v0, int v1, Union1 v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0] = v1;
    int v5;
    v5 = v2.tag;
    f_32(v0, v5);
    unsigned char * v6;
    v6 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // Call
            return f_33(v6);
            break;
        }
        case 1: { // Fold
            return f_33(v6);
            break;
        }
        case 2: { // Raise
            return f_33(v6);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_43(unsigned char * v0, int v1, Union6 v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0] = v1;
    int v5;
    v5 = v2.tag;
    f_32(v0, v5);
    unsigned char * v6;
    v6 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // Jack
            return f_33(v6);
            break;
        }
        case 1: { // King
            return f_33(v6);
            break;
        }
        case 2: { // Queen
            return f_33(v6);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_44(unsigned char * v0, static_array<Union6,2> v1, int v2, int v3){
    int v4;
    v4 = 0;
    while (while_method_0(v4)){
        unsigned long long v6;
        v6 = (unsigned long long)v4;
        unsigned long long v7;
        v7 = v6 * 4ull;
        unsigned char * v8;
        v8 = (unsigned char *)(v0+v7);
        bool v10;
        v10 = 0 <= v4;
        bool v12;
        if (v10){
            bool v11;
            v11 = v4 < 2;
            v12 = v11;
        } else {
            v12 = false;
        }
        bool v13;
        v13 = v12 == false;
        if (v13){
            assert("Index must be in range." && v12);
        } else {
        }
        Union6 v15;
        v15 = v1[v4];
        f_37(v8, v15);
        v4 += 1 ;
    }
    int * v17;
    v17 = (int *)(v0+8ull);
    v17[0] = v2;
    int * v19;
    v19 = (int *)(v0+12ull);
    v19[0] = v3;
    return ;
}
__device__ void f_41(unsigned char * v0, Union7 v1){
    int v2;
    v2 = v1.tag;
    f_35(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // CommunityCardIs
            Union6 v5 = v1.case0.v0;
            return f_37(v3, v5);
            break;
        }
        case 1: { // PlayerAction
            int v6 = v1.case1.v0; Union1 v7 = v1.case1.v1;
            return f_42(v3, v6, v7);
            break;
        }
        case 2: { // PlayerGotCard
            int v8 = v1.case2.v0; Union6 v9 = v1.case2.v1;
            return f_43(v3, v8, v9);
            break;
        }
        case 3: { // Showdown
            static_array<Union6,2> v10 = v1.case3.v0; int v11 = v1.case3.v1; int v12 = v1.case3.v2;
            return f_44(v3, v10, v11, v12);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_45(unsigned char * v0, Union2 v1){
    int v2;
    v2 = v1.tag;
    f_35(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // Computer
            return f_33(v3);
            break;
        }
        case 1: { // Human
            return f_33(v3);
            break;
        }
        case 2: { // Random
            return f_33(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_46(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+1128ull);
    v2[0] = v1;
    return ;
}
__device__ void f_30(unsigned char * v0, unsigned int v1, Union3 v2, static_array_list<Union7,32> v3, static_array<Union2,2> v4, Union8 v5){
    f_31(v0, v1);
    int v6;
    v6 = v2.tag;
    f_32(v0, v6);
    unsigned char * v7;
    v7 = (unsigned char *)(v0+16ull);
    switch (v2.tag) {
        case 0: { // None
            f_33(v7);
            break;
        }
        case 1: { // Some
            Union4 v9 = v2.case1.v0;
            f_34(v7, v9);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v10;
    v10 = v3.length;
    f_40(v0, v10);
    int v11;
    v11 = v3.length;
    int v12;
    v12 = 0;
    while (while_method_1(v11, v12)){
        unsigned long long v14;
        v14 = (unsigned long long)v12;
        unsigned long long v15;
        v15 = v14 * 32ull;
        unsigned long long v16;
        v16 = 96ull + v15;
        unsigned char * v17;
        v17 = (unsigned char *)(v0+v16);
        Union7 v19;
        v19 = v3[v12];
        f_41(v17, v19);
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
        v25 = 1120ull + v24;
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
        f_45(v26, v33);
        v21 += 1 ;
    }
    int v35;
    v35 = v5.tag;
    f_46(v0, v35);
    unsigned char * v36;
    v36 = (unsigned char *)(v0+1136ull);
    switch (v5.tag) {
        case 0: { // GameNotStarted
            return f_33(v36);
            break;
        }
        case 1: { // GameOver
            Union5 v38 = v5.case1.v0; bool v39 = v5.case1.v1; static_array<Union6,2> v40 = v5.case1.v2; int v41 = v5.case1.v3; static_array<int,2> v42 = v5.case1.v4; int v43 = v5.case1.v5;
            return f_36(v36, v38, v39, v40, v41, v42, v43);
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            Union5 v44 = v5.case2.v0; bool v45 = v5.case2.v1; static_array<Union6,2> v46 = v5.case2.v2; int v47 = v5.case2.v3; static_array<int,2> v48 = v5.case2.v4; int v49 = v5.case2.v5;
            return f_36(v36, v44, v45, v46, v47, v48, v49);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ inline bool while_method_10(int v0){
    bool v1;
    v1 = v0 < 32;
    return v1;
}
__device__ inline bool while_method_11(Union3 v0){
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
__device__ void method_47(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut1 & v3, int v4, Union4 v5){
    v3.v0 = 63u;
    static_array<float,2> v6;
    v6[0] = 0.0f;
    v6[1] = 0.0f;
    v3.v4 = v6;
    static_array_list<Union7,32> & v8 = v3.v2;
    v8.unsafe_set_length(0);
    static_array<Union2,2> v9;
    Union2 v11;
    v11 = Union2{Union2_0{}};
    v9[0] = v11;
    Union2 v13;
    v13 = Union2{Union2_0{}};
    v9[1] = v13;
    int v15;
    v15 = v4 ^ 1;
    Union2 v16;
    v16 = Union2{Union2_2{}};
    v9[v15] = v16;
    v3.v3 = v9;
    static_array_list<Union7,32> & v18 = v3.v2;
    Union3 v19;
    v19 = Union3{Union3_1{v5}};
    Union3 v20;
    v20 = v19;
    while (while_method_11(v20)){
        Union3 v792;
        switch (v20.tag) {
            case 0: { // None
                v792 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v22 = v20.case1.v0;
                Union14 v632;
                switch (v22.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v601 = v22.case0.v0; bool v602 = v22.case0.v1; static_array<Union6,2> v603 = v22.case0.v2; int v604 = v22.case0.v3; static_array<int,2> v605 = v22.case0.v4; int v606 = v22.case0.v5;
                        curandStatePhilox4_32_10_t & v607 = v3.v5;
                        curandStatePhilox4_32_10_t & v608 = v607;
                        unsigned int & v609 = v3.v0;
                        Union6 v610; unsigned int v611;
                        Tuple6 tmp37 = draw_card_20(v608, v609);
                        v610 = tmp37.v0; v611 = tmp37.v1;
                        v3.v0 = v611;
                        Union7 v612;
                        v612 = Union7{Union7_0{v610}};
                        v18.push(v612);
                        v632 = Union14{Union14_0{v601, v602, v603, v604, v605, v606, v610}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v614 = v3.v5;
                        curandStatePhilox4_32_10_t & v615 = v614;
                        unsigned int & v616 = v3.v0;
                        Union6 v617; unsigned int v618;
                        Tuple6 tmp38 = draw_card_20(v615, v616);
                        v617 = tmp38.v0; v618 = tmp38.v1;
                        v3.v0 = v618;
                        curandStatePhilox4_32_10_t & v619 = v3.v5;
                        curandStatePhilox4_32_10_t & v620 = v619;
                        unsigned int & v621 = v3.v0;
                        Union6 v622; unsigned int v623;
                        Tuple6 tmp39 = draw_card_20(v620, v621);
                        v622 = tmp39.v0; v623 = tmp39.v1;
                        v3.v0 = v623;
                        Union7 v624;
                        v624 = Union7{Union7_2{0, v617}};
                        v18.push(v624);
                        Union7 v625;
                        v625 = Union7{Union7_2{1, v622}};
                        v18.push(v625);
                        v632 = Union14{Union14_1{v617, v622}};
                        break;
                    }
                    case 2: { // Round
                        Union5 v72 = v22.case2.v0; bool v73 = v22.case2.v1; static_array<Union6,2> v74 = v22.case2.v2; int v75 = v22.case2.v3; static_array<int,2> v76 = v22.case2.v4; int v77 = v22.case2.v5;
                        static_array<Union2,2> & v78 = v3.v3;
                        bool v79;
                        v79 = 0 <= v75;
                        bool v81;
                        if (v79){
                            bool v80;
                            v80 = v75 < 2;
                            v81 = v80;
                        } else {
                            v81 = false;
                        }
                        bool v82;
                        v82 = v81 == false;
                        if (v82){
                            assert("Index must be in range." && v81);
                        } else {
                        }
                        Union2 v84;
                        v84 = v78[v75];
                        Union1 v589;
                        switch (v84.tag) {
                            case 0: { // Computer
                                static_array_list<Union7,32> & v87 = v3.v2;
                                curandStatePhilox4_32_10_t & v88 = v3.v5;
                                curandStatePhilox4_32_10_t & v89 = v88;
                                float * v90;
                                v90 = reinterpret_cast<float *>(&v1[55050240ull]);
                                float * v92;
                                v92 = reinterpret_cast<float *>(&v1[0ull]);
                                float * v94;
                                v94 = reinterpret_cast<float *>(&v1[0ull]);
                                int v96;
                                v96 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v96 && v96 < 24);
                                int v97;
                                v97 = 32768 * v96;
                                int v98;
                                v98 = threadIdx.x;
                                int v99;
                                v99 = v98;
                                while (while_method_3(v99)){
                                    bool v101;
                                    v101 = 0 <= v99;
                                    bool v102;
                                    v102 = v101 == false;
                                    if (v102){
                                        assert("The index needs to be zero or positive." && v101);
                                    } else {
                                    }
                                    int v104;
                                    v104 = v99 % 128;
                                    int v105;
                                    v105 = v99 / 128;
                                    bool v106;
                                    v106 = v105 < 256;
                                    bool v107;
                                    v107 = v106 == false;
                                    if (v107){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v106);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v105 && v105 < 256);
                                    assert("Tensor range check" && 0 <= v104 && v104 < 128);
                                    int v109;
                                    v109 = v104 + v97;
                                    int v110;
                                    v110 = 128 * v105;
                                    int v111;
                                    v111 = v110 + v109;
                                    v94[v111] = 0.0f;
                                    v99 += 256 ;
                                }
                                __syncthreads();
                                int v112;
                                v112 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v112 && v112 < 256);
                                int v113;
                                v113 = 128 * v112;
                                int v114;
                                v114 = v113 + v97;
                                static_array_list<Union9,10> v115;
                                v115 = static_array_list<Union9,10>{};
                                int v117;
                                v117 = v87.length;
                                int v118;
                                v118 = 0;
                                while (while_method_1(v117, v118)){
                                    Union7 v120;
                                    v120 = v87[v118];
                                    Union10 v139;
                                    switch (v120.tag) {
                                        case 0: { // CommunityCardIs
                                            Union6 v129 = v120.case0.v0;
                                            Union9 v130;
                                            v130 = Union9{Union9_1{v129}};
                                            v139 = Union10{Union10_1{v130}};
                                            break;
                                        }
                                        case 1: { // PlayerAction
                                            int v132 = v120.case1.v0; Union1 v133 = v120.case1.v1;
                                            Union9 v134;
                                            v134 = Union9{Union9_0{v133}};
                                            v139 = Union10{Union10_1{v134}};
                                            break;
                                        }
                                        case 2: { // PlayerGotCard
                                            int v122 = v120.case2.v0; Union6 v123 = v120.case2.v1;
                                            bool v124;
                                            v124 = v122 == v75;
                                            if (v124){
                                                Union9 v125;
                                                v125 = Union9{Union9_1{v123}};
                                                v139 = Union10{Union10_1{v125}};
                                            } else {
                                                v139 = Union10{Union10_0{}};
                                            }
                                            break;
                                        }
                                        default: {
                                            v139 = Union10{Union10_0{}};
                                        }
                                    }
                                    switch (v139.tag) {
                                        case 0: { // None
                                            break;
                                        }
                                        case 1: { // Some
                                            Union9 v140 = v139.case1.v0;
                                            v115.push(v140);
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false); __trap();
                                        }
                                    }
                                    v118 += 1 ;
                                }
                                float * v141;
                                v141 = v94+v114;
                                int v143;
                                v143 = v115.length;
                                bool v144;
                                v144 = v143 == 0;
                                if (v144){
                                    v141[0] = 1.0f;
                                } else {
                                }
                                int v145;
                                v145 = v115.length;
                                int v146;
                                v146 = 0;
                                while (while_method_1(v145, v146)){
                                    Union9 v148;
                                    v148 = v115[v146];
                                    int v150;
                                    v150 = v146 * 6;
                                    int v151;
                                    v151 = 1 + v150;
                                    switch (v148.tag) {
                                        case 0: { // C1of2
                                            Union1 v152 = v148.case0.v0;
                                            switch (v152.tag) {
                                                case 0: { // Call
                                                    v141[v151] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // Fold
                                                    int v153;
                                                    v153 = v151 + 1;
                                                    v141[v153] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // Raise
                                                    int v154;
                                                    v154 = v151 + 2;
                                                    v141[v154] = 1.0f;
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false); __trap();
                                                }
                                            }
                                            break;
                                        }
                                        case 1: { // C2of2
                                            Union6 v155 = v148.case1.v0;
                                            int v156;
                                            v156 = v151 + 3;
                                            switch (v155.tag) {
                                                case 0: { // Jack
                                                    v141[v156] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // King
                                                    int v157;
                                                    v157 = v156 + 1;
                                                    v141[v157] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // Queen
                                                    int v158;
                                                    v158 = v156 + 2;
                                                    v141[v158] = 1.0f;
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
                                    v146 += 1 ;
                                }
                                __syncthreads();
                                int v159;
                                v159 = 0;
                                while (while_method_10(v159)){
                                    float * v161;
                                    v161 = reinterpret_cast<float *>(&v1[55050240ull]);
                                    assert("Tensor range check" && 0 <= v159 && v159 < 32);
                                    int v163;
                                    v163 = 393216 * v159;
                                    float * v164;
                                    v164 = reinterpret_cast<float *>(&v1[4718592ull]);
                                    assert("Tensor range check" && 0 <= v159 && v159 < 32);
                                    float * v166;
                                    v166 = reinterpret_cast<float *>(&v1[0ull]);
                                    float * v168;
                                    v168 = reinterpret_cast<float *>(&v0[0ull]);
                                    float * v170;
                                    v170 = reinterpret_cast<float *>(&v2[0ull]);
                                    assert("Tensor range check" && 0 <= v159 && v159 < 32);
                                    int v172;
                                    v172 = 8192 * v159;
                                    float * v173;
                                    v173 = reinterpret_cast<float *>(&v1[3145728ull]);
                                    block_matmul_23(v173, v168, v172, v166);
                                    block_map_24(v164, v163, v173);
                                    block_row_map_25(v161, v163, v164);
                                    int * v175;
                                    v175 = reinterpret_cast<int *>(&v0[1048576ull]);
                                    float * v177;
                                    v177 = reinterpret_cast<float *>(&v0[1048592ull]);
                                    float * v179;
                                    v179 = reinterpret_cast<float *>(&v0[1048720ull]);
                                    double * v181;
                                    v181 = reinterpret_cast<double *>(&v1[105381888ull]);
                                    double * v183;
                                    v183 = reinterpret_cast<double *>(&v1[108527616ull]);
                                    v159 += 1 ;
                                }
                                __syncthreads();
                                int * v185;
                                v185 = reinterpret_cast<int *>(&v0[1048576ull]);
                                float * v187;
                                v187 = reinterpret_cast<float *>(&v0[1048592ull]);
                                float * v189;
                                v189 = reinterpret_cast<float *>(&v0[1048720ull]);
                                int v191;
                                v191 = v185[0];
                                float * v192;
                                v192 = reinterpret_cast<float *>(&v1[55050240ull]);
                                assert("Tensor range check" && 0 <= v191 && v191 < 32);
                                int v194;
                                v194 = 393216 * v191;
                                int v195;
                                v195 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v195 && v195 < 24);
                                int v196;
                                v196 = 16384 * v195;
                                int v197;
                                v197 = v196 + v194;
                                int v198;
                                v198 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v198 && v198 < 256);
                                int v199;
                                v199 = 64 * v198;
                                int v200;
                                v200 = v199 + v197;
                                float * v201;
                                v201 = v192+v200;
                                int v203;
                                v203 = sizeof(float *);
                                unsigned long long v204;
                                v204 = (unsigned long long)v203;
                                unsigned long long v205;
                                v205 = 256ull * v204;
                                unsigned long long v206;
                                v206 = v205 + 16ull;
                                unsigned long long v207;
                                v207 = v206 - 1ull;
                                unsigned long long v208;
                                v208 = v207 % 16ull;
                                unsigned long long v209;
                                v209 = v207 - v208;
                                unsigned long long v210;
                                v210 = v209 + 1024ull;
                                unsigned long long v211;
                                v211 = v210 + 16ull;
                                unsigned long long v212;
                                v212 = v211 - 1ull;
                                unsigned long long v213;
                                v213 = v212 % 16ull;
                                unsigned long long v214;
                                v214 = v212 - v213;
                                unsigned long long v215;
                                v215 = v214 + 1024ull;
                                bool v216;
                                v216 = v215 <= 98304ull;
                                bool v217;
                                v217 = v216 == false;
                                if (v217){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v216);
                                } else {
                                }
                                extern __shared__ unsigned char v219[];
                                bool v220;
                                v220 = v215 <= v215;
                                bool v221;
                                v221 = v220 == false;
                                if (v221){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v220);
                                } else {
                                }
                                float * * v223;
                                v223 = reinterpret_cast<float * *>(&v219[0ull]);
                                float * v225;
                                v225 = reinterpret_cast<float *>(&v219[v209]);
                                int * v227;
                                v227 = reinterpret_cast<int *>(&v219[v214]);
                                int v229;
                                v229 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v229 && v229 < 256);
                                v223[v229] = v201;
                                __syncthreads();
                                bool v230;
                                v230 = 0 <= v229;
                                bool v231;
                                v231 = v230 == false;
                                if (v231){
                                    assert("The index needs to be zero or positive." && v230);
                                } else {
                                }
                                int v233;
                                v233 = v229 % 16;
                                int v234;
                                v234 = v229 / 16;
                                bool v235;
                                v235 = v234 < 16;
                                bool v236;
                                v236 = v235 == false;
                                if (v236){
                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v235);
                                } else {
                                }
                                assert("Tensor range check" && 0 <= v234 && v234 < 16);
                                int v238;
                                v238 = 0;
                                while (while_method_8(v238)){
                                    bool v240;
                                    v240 = 0 <= v234;
                                    bool v241;
                                    v241 = v240 && v235;
                                    bool v242;
                                    v242 = v241 == false;
                                    if (v242){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v241);
                                    } else {
                                    }
                                    bool v244;
                                    v244 = 0 <= v238;
                                    bool v246;
                                    if (v244){
                                        bool v245;
                                        v245 = v238 < 16;
                                        v246 = v245;
                                    } else {
                                        v246 = false;
                                    }
                                    bool v247;
                                    v247 = v246 == false;
                                    if (v247){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v246);
                                    } else {
                                    }
                                    int v249;
                                    v249 = v238 * 16;
                                    int v250;
                                    v250 = v249 + v234;
                                    assert("Tensor range check" && 0 <= v238 && v238 < 16);
                                    int v251;
                                    v251 = 16 * v238;
                                    int v252;
                                    v252 = v251 + v234;
                                    float * v253;
                                    v253 = v223[v252];
                                    int v254;
                                    v254 = blockIdx.x;
                                    int v255;
                                    v255 = v254 * 256;
                                    int v256;
                                    v256 = v255 + v250;
                                    assert("Tensor range check" && 0 <= v233 && v233 < 16);
                                    int v257;
                                    v257 = 4 * v233;
                                    float v258[4];
                                    int v259[4];
                                    int v260;
                                    v260 = 0;
                                    while (while_method_4(v260)){
                                        assert("Tensor range check" && 0 <= v260 && v260 < 1);
                                        int v262;
                                        v262 = 4 * v260;
                                        assert("Tensor range check" && 0 <= v260 && v260 < 1);
                                        int v263;
                                        v263 = 64 * v260;
                                        int v264;
                                        v264 = v263 + v257;
                                        int4* v265;
                                        v265 = reinterpret_cast<int4*>(v253 + v264);
                                        int4* v266;
                                        v266 = reinterpret_cast<int4*>(v258 + v262);
                                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v265) % 16 == 0 && reinterpret_cast<unsigned long long>(v266) % 16 == 0);
                                        *v266 = *v265;
                                        v260 += 1 ;
                                    }
                                    int v267;
                                    v267 = 0;
                                    while (while_method_4(v267)){
                                        int v269;
                                        v269 = 0;
                                        while (while_method_7(v269)){
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
                                                assert("The indices should be inside the range of the dimension." && v273);
                                            } else {
                                            }
                                            bool v276;
                                            v276 = 0 <= v233;
                                            bool v278;
                                            if (v276){
                                                bool v277;
                                                v277 = v233 < 16;
                                                v278 = v277;
                                            } else {
                                                v278 = false;
                                            }
                                            bool v279;
                                            v279 = v278 == false;
                                            if (v279){
                                                assert("The indices should be inside the range of the dimension." && v278);
                                            } else {
                                            }
                                            int v281;
                                            v281 = v233 * 4;
                                            int v282;
                                            v282 = v269 + v281;
                                            bool v283;
                                            v283 = 0 <= v267;
                                            bool v285;
                                            if (v283){
                                                bool v284;
                                                v284 = v267 < 1;
                                                v285 = v284;
                                            } else {
                                                v285 = false;
                                            }
                                            bool v286;
                                            v286 = v285 == false;
                                            if (v286){
                                                assert("The indices should be inside the range of the dimension." && v285);
                                            } else {
                                            }
                                            int v288;
                                            v288 = v267 * 64;
                                            int v289;
                                            v289 = v282 + v288;
                                            assert("Tensor range check" && 0 <= v267 && v267 < 1);
                                            assert("Tensor range check" && 0 <= v269 && v269 < 4);
                                            int v290;
                                            v290 = 4 * v267;
                                            int v291;
                                            v291 = v290 + v269;
                                            v259[v291] = v289;
                                            v269 += 1 ;
                                        }
                                        v267 += 1 ;
                                    }
                                    float v292[4];
                                    float v293;
                                    v293 = 0.0f;
                                    int v294;
                                    v294 = 0;
                                    while (while_method_4(v294)){
                                        assert("Tensor range check" && 0 <= v294 && v294 < 1);
                                        int v296;
                                        v296 = 4 * v294;
                                        assert("Tensor range check" && 0 <= v294 && v294 < 1);
                                        int v297; float v298;
                                        Tuple8 tmp40 = Tuple8{0, 0.0f};
                                        v297 = tmp40.v0; v298 = tmp40.v1;
                                        while (while_method_7(v297)){
                                            assert("Tensor range check" && 0 <= v297 && v297 < 4);
                                            int v300;
                                            v300 = v297 + v296;
                                            float v301;
                                            v301 = v258[v300];
                                            float v302;
                                            v302 = v298 + v301;
                                            v298 = v302;
                                            v297 += 1 ;
                                        }
                                        auto v303 = cooperative_groups::coalesced_threads();
                                        int v304;
                                        v304 = threadIdx.x;
                                        int v305;
                                        v305 = v304 / 16;
                                        auto v306 = cooperative_groups::labeled_partition(v303,v305);
                                        Closure2 v307{};
                                        float v308;
                                        v308 = cooperative_groups::inclusive_scan(v306, v298, v307);
                                        float v309;
                                        v309 = v306.shfl_up(v308,1);
                                        bool v310;
                                        v310 = v306.thread_rank() == 0;
                                        float v311;
                                        if (v310){
                                            v311 = 0.0f;
                                        } else {
                                            v311 = v309;
                                        }
                                        float v312;
                                        v312 = v306.shfl(v308,v306.num_threads()-1);
                                        float v313;
                                        v313 = v293 + v311;
                                        int v314; float v315;
                                        Tuple8 tmp41 = Tuple8{0, v313};
                                        v314 = tmp41.v0; v315 = tmp41.v1;
                                        while (while_method_7(v314)){
                                            assert("Tensor range check" && 0 <= v314 && v314 < 4);
                                            int v317;
                                            v317 = v314 + v296;
                                            float v318;
                                            v318 = v258[v317];
                                            float v319;
                                            v319 = v315 + v318;
                                            assert("Tensor range check" && 0 <= v314 && v314 < 4);
                                            v292[v317] = v319;
                                            v315 = v319;
                                            v314 += 1 ;
                                        }
                                        float v320;
                                        v320 = v293 + v312;
                                        v293 = v320;
                                        v294 += 1 ;
                                    }
                                    float v321[4];
                                    bool v322[4];
                                    int v323;
                                    v323 = 0;
                                    while (while_method_4(v323)){
                                        int v325;
                                        v325 = 0;
                                        while (while_method_7(v325)){
                                            assert("Tensor range check" && 0 <= v323 && v323 < 1);
                                            assert("Tensor range check" && 0 <= v325 && v325 < 4);
                                            int v327;
                                            v327 = 4 * v323;
                                            int v328;
                                            v328 = v327 + v325;
                                            float v329;
                                            v329 = v292[v328];
                                            float v330;
                                            v330 = v258[v328];
                                            bool v331;
                                            v331 = v330 > 0.0f;
                                            assert("Tensor range check" && 0 <= v323 && v323 < 1);
                                            assert("Tensor range check" && 0 <= v325 && v325 < 4);
                                            v321[v328] = v329;
                                            v322[v328] = v331;
                                            v325 += 1 ;
                                        }
                                        v323 += 1 ;
                                    }
                                    float v332; bool v333;
                                    Tuple9 tmp42 = Tuple9{-1.0f / 0.0f, false};
                                    v332 = tmp42.v0; v333 = tmp42.v1;
                                    int v334;
                                    v334 = 0;
                                    while (while_method_4(v334)){
                                        int v336;
                                        v336 = 0;
                                        while (while_method_7(v336)){
                                            assert("Tensor range check" && 0 <= v334 && v334 < 1);
                                            assert("Tensor range check" && 0 <= v336 && v336 < 4);
                                            int v338;
                                            v338 = 4 * v334;
                                            int v339;
                                            v339 = v338 + v336;
                                            float v340;
                                            v340 = v321[v339];
                                            bool v341;
                                            v341 = v322[v339];
                                            float v348; bool v349;
                                            if (v333){
                                                if (v341){
                                                    bool v342;
                                                    v342 = v332 >= v340;
                                                    float v343;
                                                    if (v342){
                                                        v343 = v332;
                                                    } else {
                                                        v343 = v340;
                                                    }
                                                    v348 = v343; v349 = true;
                                                } else {
                                                    v348 = v332; v349 = v333;
                                                }
                                            } else {
                                                if (v341){
                                                    v348 = v340; v349 = v341;
                                                } else {
                                                    v348 = v332; v349 = v333;
                                                }
                                            }
                                            v332 = v348;
                                            v333 = v349;
                                            v336 += 1 ;
                                        }
                                        v334 += 1 ;
                                    }
                                    auto v350 = cooperative_groups::coalesced_threads();
                                    int v351;
                                    v351 = threadIdx.x;
                                    int v352;
                                    v352 = v351 / 16;
                                    auto v353 = cooperative_groups::labeled_partition(v350,v352);
                                    Closure3 v354{};
                                    float v355; bool v356;
                                    Tuple9 tmp43 = cooperative_groups::reduce(v353, Tuple9{v332, v333}, v354);
                                    v355 = tmp43.v0; v356 = tmp43.v1;
                                    bool v357;
                                    v357 = v356 == false;
                                    if (v357){
                                        assert("The local reduce must be true." && v356);
                                    } else {
                                    }
                                    float v359[4];
                                    int v360[4];
                                    int v361;
                                    v361 = 0;
                                    while (while_method_4(v361)){
                                        int v363;
                                        v363 = 0;
                                        while (while_method_7(v363)){
                                            assert("Tensor range check" && 0 <= v361 && v361 < 1);
                                            assert("Tensor range check" && 0 <= v363 && v363 < 4);
                                            int v365;
                                            v365 = 4 * v361;
                                            int v366;
                                            v366 = v365 + v363;
                                            int v367;
                                            v367 = v259[v366];
                                            float v368;
                                            v368 = curand_uniform(&v89);
                                            assert("Tensor range check" && 0 <= v361 && v361 < 1);
                                            assert("Tensor range check" && 0 <= v363 && v363 < 4);
                                            v359[v366] = v368;
                                            v360[v366] = v367;
                                            v363 += 1 ;
                                        }
                                        v361 += 1 ;
                                    }
                                    float v369; int v370;
                                    Tuple10 tmp44 = Tuple10{0.0f, 2147483647};
                                    v369 = tmp44.v0; v370 = tmp44.v1;
                                    int v371;
                                    v371 = 0;
                                    while (while_method_4(v371)){
                                        int v373;
                                        v373 = 0;
                                        while (while_method_7(v373)){
                                            assert("Tensor range check" && 0 <= v371 && v371 < 1);
                                            assert("Tensor range check" && 0 <= v373 && v373 < 4);
                                            int v375;
                                            v375 = 4 * v371;
                                            int v376;
                                            v376 = v375 + v373;
                                            float v377;
                                            v377 = v359[v376];
                                            int v378;
                                            v378 = v360[v376];
                                            bool v379;
                                            v379 = v370 < v378;
                                            float v380; int v381;
                                            if (v379){
                                                v380 = v369; v381 = v370;
                                            } else {
                                                v380 = v377; v381 = v378;
                                            }
                                            v369 = v380;
                                            v370 = v381;
                                            v373 += 1 ;
                                        }
                                        v371 += 1 ;
                                    }
                                    auto v382 = cooperative_groups::coalesced_threads();
                                    int v383;
                                    v383 = threadIdx.x;
                                    int v384;
                                    v384 = v383 / 16;
                                    auto v385 = cooperative_groups::labeled_partition(v382,v384);
                                    Closure4 v386{};
                                    float v387; int v388;
                                    Tuple10 tmp45 = cooperative_groups::reduce(v385, Tuple10{v369, v370}, v386);
                                    v387 = tmp45.v0; v388 = tmp45.v1;
                                    float v389;
                                    v389 = v355 * v387;
                                    int v390[4];
                                    bool v391[4];
                                    int v392;
                                    v392 = 0;
                                    while (while_method_4(v392)){
                                        int v394;
                                        v394 = 0;
                                        while (while_method_7(v394)){
                                            assert("Tensor range check" && 0 <= v392 && v392 < 1);
                                            assert("Tensor range check" && 0 <= v394 && v394 < 4);
                                            int v396;
                                            v396 = 4 * v392;
                                            int v397;
                                            v397 = v396 + v394;
                                            float v398;
                                            v398 = v321[v397];
                                            bool v399;
                                            v399 = v322[v397];
                                            int v400;
                                            v400 = v259[v397];
                                            int v403; bool v404;
                                            if (v399){
                                                float v401;
                                                v401 = v398 - v389;
                                                bool v402;
                                                v402 = v401 >= 0.0f;
                                                v403 = v400; v404 = v402;
                                            } else {
                                                v403 = 2147483647; v404 = false;
                                            }
                                            assert("Tensor range check" && 0 <= v392 && v392 < 1);
                                            assert("Tensor range check" && 0 <= v394 && v394 < 4);
                                            v390[v397] = v403;
                                            v391[v397] = v404;
                                            v394 += 1 ;
                                        }
                                        v392 += 1 ;
                                    }
                                    int v405; bool v406;
                                    Tuple11 tmp46 = Tuple11{2147483647, false};
                                    v405 = tmp46.v0; v406 = tmp46.v1;
                                    int v407;
                                    v407 = 0;
                                    while (while_method_4(v407)){
                                        int v409;
                                        v409 = 0;
                                        while (while_method_7(v409)){
                                            assert("Tensor range check" && 0 <= v407 && v407 < 1);
                                            assert("Tensor range check" && 0 <= v409 && v409 < 4);
                                            int v411;
                                            v411 = 4 * v407;
                                            int v412;
                                            v412 = v411 + v409;
                                            int v413;
                                            v413 = v390[v412];
                                            bool v414;
                                            v414 = v391[v412];
                                            int v421; bool v422;
                                            if (v406){
                                                if (v414){
                                                    bool v415;
                                                    v415 = v405 < v413;
                                                    int v416;
                                                    if (v415){
                                                        v416 = v405;
                                                    } else {
                                                        v416 = v413;
                                                    }
                                                    v421 = v416; v422 = true;
                                                } else {
                                                    v421 = v405; v422 = v406;
                                                }
                                            } else {
                                                if (v414){
                                                    v421 = v413; v422 = v414;
                                                } else {
                                                    v421 = v405; v422 = v406;
                                                }
                                            }
                                            v405 = v421;
                                            v406 = v422;
                                            v409 += 1 ;
                                        }
                                        v407 += 1 ;
                                    }
                                    auto v423 = cooperative_groups::coalesced_threads();
                                    int v424;
                                    v424 = threadIdx.x;
                                    int v425;
                                    v425 = v424 / 16;
                                    auto v426 = cooperative_groups::labeled_partition(v423,v425);
                                    Closure5 v427{};
                                    int v428; bool v429;
                                    Tuple11 tmp47 = cooperative_groups::reduce(v426, Tuple11{v405, v406}, v427);
                                    v428 = tmp47.v0; v429 = tmp47.v1;
                                    bool v430;
                                    v430 = v429 == false;
                                    if (v430){
                                        assert("The local reduce must be true." && v429);
                                    } else {
                                    }
                                    float v432; int v433;
                                    Tuple10 tmp48 = Tuple10{0.0f, 2147483647};
                                    v432 = tmp48.v0; v433 = tmp48.v1;
                                    int v434;
                                    v434 = 0;
                                    while (while_method_4(v434)){
                                        int v436;
                                        v436 = 0;
                                        while (while_method_7(v436)){
                                            assert("Tensor range check" && 0 <= v434 && v434 < 1);
                                            assert("Tensor range check" && 0 <= v436 && v436 < 4);
                                            int v438;
                                            v438 = 4 * v434;
                                            int v439;
                                            v439 = v438 + v436;
                                            float v440;
                                            v440 = v258[v439];
                                            int v441;
                                            v441 = v259[v439];
                                            bool v442;
                                            v442 = v433 == v428;
                                            float v446; int v447;
                                            if (v442){
                                                v446 = v432; v447 = v433;
                                            } else {
                                                bool v443;
                                                v443 = v441 == v428;
                                                if (v443){
                                                    v446 = v440; v447 = v441;
                                                } else {
                                                    v446 = v432; v447 = v433;
                                                }
                                            }
                                            v432 = v446;
                                            v433 = v447;
                                            v436 += 1 ;
                                        }
                                        v434 += 1 ;
                                    }
                                    auto v448 = cooperative_groups::coalesced_threads();
                                    int v449;
                                    v449 = threadIdx.x;
                                    int v450;
                                    v450 = v449 / 16;
                                    auto v451 = cooperative_groups::labeled_partition(v448,v450);
                                    Closure6 v452{v428};
                                    float v453; int v454;
                                    Tuple10 tmp49 = cooperative_groups::reduce(v451, Tuple10{v432, v433}, v452);
                                    v453 = tmp49.v0; v454 = tmp49.v1;
                                    bool v455;
                                    v455 = v454 == 2147483647;
                                    bool v456;
                                    v456 = v455 != true;
                                    bool v457;
                                    v457 = v456 == false;
                                    if (v457){
                                        assert("Expected a valid action id in get_prob." && v456);
                                    } else {
                                    }
                                    int v459;
                                    v459 = 0;
                                    while (while_method_4(v459)){
                                        assert("Tensor range check" && 0 <= v459 && v459 < 1);
                                        assert("Tensor range check" && 0 <= v459 && v459 < 1);
                                        v459 += 1 ;
                                    }
                                    assert("Tensor range check" && 0 <= v250 && v250 < 256);
                                    v225[v250] = v453;
                                    v227[v250] = v428;
                                    v238 += 1 ;
                                }
                                __syncthreads();
                                assert("Tensor range check" && 0 <= v229 && v229 < 256);
                                float v461;
                                v461 = v225[v229];
                                int v462;
                                v462 = v227[v229];
                                __syncthreads();
                                extern __shared__ unsigned char v463[];
                                float * v464;
                                v464 = reinterpret_cast<float *>(&v463[0ull]);
                                int * v466;
                                v466 = reinterpret_cast<int *>(&v463[16ull]);
                                int v468;
                                v468 = threadIdx.x;
                                bool v469;
                                v469 = v468 == 0;
                                if (v469){
                                    v464[0] = v461;
                                    v466[0] = v462;
                                } else {
                                }
                                __syncthreads();
                                float v470;
                                v470 = v464[0];
                                int v471;
                                v471 = v466[0];
                                __syncthreads();
                                double * v472;
                                v472 = reinterpret_cast<double *>(&v1[105381888ull]);
                                double * v474;
                                v474 = reinterpret_cast<double *>(&v1[108527616ull]);
                                int v476;
                                v476 = threadIdx.x;
                                int v477;
                                v477 = blockIdx.x;
                                int v478;
                                v478 = v477 * 256;
                                int v479;
                                v479 = v476 + v478;
                                int v480;
                                v480 = 0;
                                while (while_method_10(v480)){
                                    float * v482;
                                    v482 = reinterpret_cast<float *>(&v1[55050240ull]);
                                    int v484;
                                    v484 = blockIdx.x;
                                    int v485;
                                    v485 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v480 && v480 < 32);
                                    assert("Tensor range check" && 0 <= v484 && v484 < 24);
                                    assert("Tensor range check" && 0 <= v485 && v485 < 256);
                                    assert("Tensor range check" && 0 <= v471 && v471 < 64);
                                    int v486;
                                    v486 = 64 * v485;
                                    int v487;
                                    v487 = v486 + v471;
                                    int v488;
                                    v488 = 16384 * v484;
                                    int v489;
                                    v489 = v488 + v487;
                                    int v490;
                                    v490 = 393216 * v480;
                                    int v491;
                                    v491 = v490 + v489;
                                    float v492;
                                    v492 = v482[v491];
                                    double v493;
                                    v493 = (double)v470;
                                    double v494;
                                    v494 = log(v493);
                                    double v495;
                                    v495 = (double)v492;
                                    double v496;
                                    v496 = log(v495);
                                    assert("Tensor range check" && 0 <= v480 && v480 < 32);
                                    assert("Tensor range check" && 0 <= v479 && v479 < 6144);
                                    assert("Tensor range check" && 0 <= v75 && v75 < 2);
                                    int v497;
                                    v497 = 2 * v479;
                                    int v498;
                                    v498 = v497 + v75;
                                    int v499;
                                    v499 = 12288 * v480;
                                    int v500;
                                    v500 = v499 + v498;
                                    double v501;
                                    v501 = v472[v500];
                                    double v502;
                                    v502 = v474[v500];
                                    double v503;
                                    v503 = v496 + v501;
                                    double v504;
                                    v504 = v494 + v502;
                                    assert("Tensor range check" && 0 <= v480 && v480 < 32);
                                    assert("Tensor range check" && 0 <= v479 && v479 < 6144);
                                    assert("Tensor range check" && 0 <= v75 && v75 < 2);
                                    v472[v500] = v503;
                                    v474[v500] = v504;
                                    v480 += 1 ;
                                }
                                bool v505;
                                v505 = 0 == v471;
                                Union12 v514;
                                if (v505){
                                    v514 = Union12{Union12_1{}};
                                } else {
                                    bool v507;
                                    v507 = 1 == v471;
                                    if (v507){
                                        v514 = Union12{Union12_0{}};
                                    } else {
                                        bool v509;
                                        v509 = 2 == v471;
                                        if (v509){
                                            v514 = Union12{Union12_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v514.tag) {
                                    case 0: { // AA_Call
                                        v589 = Union1{Union1_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v515;
                                        v515 = v76[0];
                                        int v517; int v518;
                                        Tuple7 tmp50 = Tuple7{1, v515};
                                        v517 = tmp50.v0; v518 = tmp50.v1;
                                        while (while_method_0(v517)){
                                            bool v520;
                                            v520 = 0 <= v517;
                                            bool v522;
                                            if (v520){
                                                bool v521;
                                                v521 = v517 < 2;
                                                v522 = v521;
                                            } else {
                                                v522 = false;
                                            }
                                            bool v523;
                                            v523 = v522 == false;
                                            if (v523){
                                                assert("Index must be in range." && v522);
                                            } else {
                                            }
                                            int v525;
                                            v525 = v76[v517];
                                            bool v527;
                                            v527 = v518 >= v525;
                                            int v528;
                                            if (v527){
                                                v528 = v518;
                                            } else {
                                                v528 = v525;
                                            }
                                            v518 = v528;
                                            v517 += 1 ;
                                        }
                                        bool v530;
                                        if (v79){
                                            bool v529;
                                            v529 = v75 < 2;
                                            v530 = v529;
                                        } else {
                                            v530 = false;
                                        }
                                        bool v531;
                                        v531 = v530 == false;
                                        if (v531){
                                            assert("Index must be in range." && v530);
                                        } else {
                                        }
                                        int v533;
                                        v533 = v76[v75];
                                        bool v535;
                                        v535 = v533 == v518;
                                        if (v535){
                                            v589 = Union1{Union1_0{}};
                                        } else {
                                            v589 = Union1{Union1_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v540;
                                        v540 = v77 > 0;
                                        if (v540){
                                            v589 = Union1{Union1_2{}};
                                        } else {
                                            v589 = Union1{Union1_0{}};
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
                                curandStatePhilox4_32_10_t & v547 = v3.v5;
                                curandStatePhilox4_32_10_t & v548 = v547;
                                static_array_list<Union1,3> v549;
                                v549 = static_array_list<Union1,3>{};
                                v549.unsafe_set_length(1);
                                Union1 v551;
                                v551 = Union1{Union1_0{}};
                                v549[0] = v551;
                                int v553;
                                v553 = v76[0];
                                int v555;
                                v555 = v76[1];
                                bool v557;
                                v557 = v553 == v555;
                                bool v558;
                                v558 = v557 != true;
                                if (v558){
                                    Union1 v559;
                                    v559 = Union1{Union1_1{}};
                                    v549.push(v559);
                                } else {
                                }
                                bool v560;
                                v560 = v77 > 0;
                                if (v560){
                                    Union1 v561;
                                    v561 = Union1{Union1_2{}};
                                    v549.push(v561);
                                } else {
                                }
                                int v562;
                                v562 = v549.length;
                                int v563;
                                v563 = v562 - 1;
                                int v564;
                                v564 = 0;
                                while (while_method_1(v563, v564)){
                                    int v566;
                                    v566 = v549.length;
                                    int v567;
                                    v567 = int_range_22(v566, v564, v548);
                                    Union1 v568;
                                    v568 = v549[v564];
                                    Union1 v570;
                                    v570 = v549[v567];
                                    v549[v564] = v570;
                                    v549[v567] = v568;
                                    v564 += 1 ;
                                }
                                Union1 v572;
                                v572 = v549.pop();
                                int v573;
                                v573 = sizeof(Union1);
                                unsigned long long v574;
                                v574 = (unsigned long long)v573;
                                bool v575;
                                v575 = v574 <= 98304ull;
                                bool v576;
                                v576 = v575 == false;
                                if (v576){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v575);
                                } else {
                                }
                                extern __shared__ unsigned char v578[];
                                bool v579;
                                v579 = v574 <= v574;
                                bool v580;
                                v580 = v579 == false;
                                if (v580){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v579);
                                } else {
                                }
                                Union1 * v582;
                                v582 = reinterpret_cast<Union1 *>(&v578[0ull]);
                                int v584;
                                v584 = threadIdx.x;
                                bool v585;
                                v585 = v584 == 0;
                                if (v585){
                                    v582[0] = v572;
                                } else {
                                }
                                __syncthreads();
                                Union1 v586;
                                v586 = v582[0];
                                __syncthreads();
                                v589 = v586;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union7 v590;
                        v590 = Union7{Union7_1{v75, v589}};
                        v18.push(v590);
                        v632 = Union14{Union14_2{v72, v73, v74, v75, v76, v77, v589}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v592 = v22.case3.v0; bool v593 = v22.case3.v1; static_array<Union6,2> v594 = v22.case3.v2; int v595 = v22.case3.v3; static_array<int,2> v596 = v22.case3.v4; int v597 = v22.case3.v5; Union1 v598 = v22.case3.v6;
                        Union7 v599;
                        v599 = Union7{Union7_1{v595, v598}};
                        v18.push(v599);
                        v632 = Union14{Union14_2{v592, v593, v594, v595, v596, v597, v598}};
                        break;
                    }
                    case 4: { // TerminalCall
                        Union5 v43 = v22.case4.v0; bool v44 = v22.case4.v1; static_array<Union6,2> v45 = v22.case4.v2; int v46 = v22.case4.v3; static_array<int,2> v47 = v22.case4.v4; int v48 = v22.case4.v5;
                        bool v49;
                        v49 = 0 <= v46;
                        bool v51;
                        if (v49){
                            bool v50;
                            v50 = v46 < 2;
                            v51 = v50;
                        } else {
                            v51 = false;
                        }
                        bool v52;
                        v52 = v51 == false;
                        if (v52){
                            assert("Index must be in range." && v51);
                        } else {
                        }
                        int v54;
                        v54 = v47[v46];
                        Union13 v56;
                        v56 = compare_hands_26(v43, v44, v45, v46, v47, v48);
                        int v61; int v62;
                        switch (v56.tag) {
                            case 0: { // Eq
                                v61 = 0; v62 = -1;
                                break;
                            }
                            case 1: { // Gt
                                v61 = v54; v62 = 0;
                                break;
                            }
                            case 2: { // Lt
                                v61 = v54; v62 = 1;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        int v63;
                        v63 = -v62;
                        bool v64;
                        v64 = v62 >= v63;
                        int v65;
                        if (v64){
                            v65 = v62;
                        } else {
                            v65 = v63;
                        }
                        float v66;
                        v66 = (float)v61;
                        static_array<float,2> & v67 = v3.v4;
                        v67[v65] = v66;
                        int v68;
                        v68 = v65 ^ 1;
                        float v69;
                        v69 = -v66;
                        v67[v68] = v69;
                        Union7 v70;
                        v70 = Union7{Union7_3{v45, v61, v62}};
                        v18.push(v70);
                        v632 = Union14{Union14_3{}};
                        break;
                    }
                    case 5: { // TerminalFold
                        Union5 v23 = v22.case5.v0; bool v24 = v22.case5.v1; static_array<Union6,2> v25 = v22.case5.v2; int v26 = v22.case5.v3; static_array<int,2> v27 = v22.case5.v4; int v28 = v22.case5.v5;
                        bool v29;
                        v29 = 0 <= v26;
                        bool v31;
                        if (v29){
                            bool v30;
                            v30 = v26 < 2;
                            v31 = v30;
                        } else {
                            v31 = false;
                        }
                        bool v32;
                        v32 = v31 == false;
                        if (v32){
                            assert("Index must be in range." && v31);
                        } else {
                        }
                        int v34;
                        v34 = v27[v26];
                        int v36;
                        v36 = -v34;
                        float v37;
                        v37 = (float)v36;
                        static_array<float,2> & v38 = v3.v4;
                        v38[v26] = v37;
                        int v39;
                        v39 = v26 ^ 1;
                        float v40;
                        v40 = -v37;
                        v38[v39] = v40;
                        Union7 v41;
                        v41 = Union7{Union7_3{v25, v34, v39}};
                        v18.push(v41);
                        v632 = Union14{Union14_3{}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                switch (v632.tag) {
                    case 0: { // T_game_chance_community_card
                        Union5 v634 = v632.case0.v0; bool v635 = v632.case0.v1; static_array<Union6,2> v636 = v632.case0.v2; int v637 = v632.case0.v3; static_array<int,2> v638 = v632.case0.v4; int v639 = v632.case0.v5; Union6 v640 = v632.case0.v6;
                        int v641;
                        v641 = 2;
                        int v642; int v643;
                        Tuple7 tmp51 = Tuple7{0, 0};
                        v642 = tmp51.v0; v643 = tmp51.v1;
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
                            v650 = v638[v642];
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
                        static_array<int,2> v654;
                        int v656;
                        v656 = 0;
                        while (while_method_0(v656)){
                            v654[v656] = v643;
                            v656 += 1 ;
                        }
                        Union5 v658;
                        v658 = Union5{Union5_1{v640}};
                        Union4 v659;
                        v659 = Union4{Union4_2{v658, true, v636, 0, v654, v641}};
                        v792 = Union3{Union3_1{v659}};
                        break;
                    }
                    case 1: { // T_game_chance_init
                        Union6 v661 = v632.case1.v0; Union6 v662 = v632.case1.v1;
                        int v663;
                        v663 = 2;
                        static_array<int,2> v664;
                        v664[0] = 1;
                        v664[1] = 1;
                        static_array<Union6,2> v666;
                        v666[0] = v661;
                        v666[1] = v662;
                        Union5 v668;
                        v668 = Union5{Union5_0{}};
                        Union4 v669;
                        v669 = Union4{Union4_2{v668, true, v666, 0, v664, v663}};
                        v792 = Union3{Union3_1{v669}};
                        break;
                    }
                    case 2: { // T_game_round
                        Union5 v671 = v632.case2.v0; bool v672 = v632.case2.v1; static_array<Union6,2> v673 = v632.case2.v2; int v674 = v632.case2.v3; static_array<int,2> v675 = v632.case2.v4; int v676 = v632.case2.v5; Union1 v677 = v632.case2.v6;
                        Union4 v784;
                        switch (v671.tag) {
                            case 0: { // None
                                switch (v677.tag) {
                                    case 0: { // Call
                                        if (v672){
                                            int v740;
                                            v740 = v674 ^ 1;
                                            v784 = Union4{Union4_2{v671, false, v673, v740, v675, v676}};
                                        } else {
                                            v784 = Union4{Union4_0{v671, v672, v673, v674, v675, v676}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v784 = Union4{Union4_5{v671, v672, v673, v674, v675, v676}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v744;
                                        v744 = v676 > 0;
                                        if (v744){
                                            int v745;
                                            v745 = v674 ^ 1;
                                            int v746;
                                            v746 = -1 + v676;
                                            int v747; int v748;
                                            Tuple7 tmp52 = Tuple7{0, 0};
                                            v747 = tmp52.v0; v748 = tmp52.v1;
                                            while (while_method_0(v747)){
                                                bool v750;
                                                v750 = 0 <= v747;
                                                bool v752;
                                                if (v750){
                                                    bool v751;
                                                    v751 = v747 < 2;
                                                    v752 = v751;
                                                } else {
                                                    v752 = false;
                                                }
                                                bool v753;
                                                v753 = v752 == false;
                                                if (v753){
                                                    assert("Index must be in range." && v752);
                                                } else {
                                                }
                                                int v755;
                                                v755 = v675[v747];
                                                bool v757;
                                                v757 = v748 >= v755;
                                                int v758;
                                                if (v757){
                                                    v758 = v748;
                                                } else {
                                                    v758 = v755;
                                                }
                                                v748 = v758;
                                                v747 += 1 ;
                                            }
                                            static_array<int,2> v759;
                                            int v761;
                                            v761 = 0;
                                            while (while_method_0(v761)){
                                                v759[v761] = v748;
                                                v761 += 1 ;
                                            }
                                            static_array<int,2> v763;
                                            int v765;
                                            v765 = 0;
                                            while (while_method_0(v765)){
                                                bool v767;
                                                v767 = 0 <= v765;
                                                bool v769;
                                                if (v767){
                                                    bool v768;
                                                    v768 = v765 < 2;
                                                    v769 = v768;
                                                } else {
                                                    v769 = false;
                                                }
                                                bool v770;
                                                v770 = v769 == false;
                                                if (v770){
                                                    assert("Index must be in range." && v769);
                                                } else {
                                                }
                                                int v772;
                                                v772 = v759[v765];
                                                bool v774;
                                                v774 = v765 == v674;
                                                int v776;
                                                if (v774){
                                                    int v775;
                                                    v775 = v772 + 2;
                                                    v776 = v775;
                                                } else {
                                                    v776 = v772;
                                                }
                                                v763[v765] = v776;
                                                v765 += 1 ;
                                            }
                                            v784 = Union4{Union4_2{v671, false, v673, v745, v763, v746}};
                                        } else {
                                            printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                            __trap();
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                break;
                            }
                            case 1: { // Some
                                Union6 v678 = v671.case1.v0;
                                switch (v677.tag) {
                                    case 0: { // Call
                                        if (v672){
                                            int v680;
                                            v680 = v674 ^ 1;
                                            v784 = Union4{Union4_2{v671, false, v673, v680, v675, v676}};
                                        } else {
                                            int v682; int v683;
                                            Tuple7 tmp53 = Tuple7{0, 0};
                                            v682 = tmp53.v0; v683 = tmp53.v1;
                                            while (while_method_0(v682)){
                                                bool v685;
                                                v685 = 0 <= v682;
                                                bool v687;
                                                if (v685){
                                                    bool v686;
                                                    v686 = v682 < 2;
                                                    v687 = v686;
                                                } else {
                                                    v687 = false;
                                                }
                                                bool v688;
                                                v688 = v687 == false;
                                                if (v688){
                                                    assert("Index must be in range." && v687);
                                                } else {
                                                }
                                                int v690;
                                                v690 = v675[v682];
                                                bool v692;
                                                v692 = v683 >= v690;
                                                int v693;
                                                if (v692){
                                                    v693 = v683;
                                                } else {
                                                    v693 = v690;
                                                }
                                                v683 = v693;
                                                v682 += 1 ;
                                            }
                                            static_array<int,2> v694;
                                            int v696;
                                            v696 = 0;
                                            while (while_method_0(v696)){
                                                v694[v696] = v683;
                                                v696 += 1 ;
                                            }
                                            v784 = Union4{Union4_4{v671, v672, v673, v674, v694, v676}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v784 = Union4{Union4_5{v671, v672, v673, v674, v675, v676}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v700;
                                        v700 = v676 > 0;
                                        if (v700){
                                            int v701;
                                            v701 = v674 ^ 1;
                                            int v702;
                                            v702 = -1 + v676;
                                            int v703; int v704;
                                            Tuple7 tmp54 = Tuple7{0, 0};
                                            v703 = tmp54.v0; v704 = tmp54.v1;
                                            while (while_method_0(v703)){
                                                bool v706;
                                                v706 = 0 <= v703;
                                                bool v708;
                                                if (v706){
                                                    bool v707;
                                                    v707 = v703 < 2;
                                                    v708 = v707;
                                                } else {
                                                    v708 = false;
                                                }
                                                bool v709;
                                                v709 = v708 == false;
                                                if (v709){
                                                    assert("Index must be in range." && v708);
                                                } else {
                                                }
                                                int v711;
                                                v711 = v675[v703];
                                                bool v713;
                                                v713 = v704 >= v711;
                                                int v714;
                                                if (v713){
                                                    v714 = v704;
                                                } else {
                                                    v714 = v711;
                                                }
                                                v704 = v714;
                                                v703 += 1 ;
                                            }
                                            static_array<int,2> v715;
                                            int v717;
                                            v717 = 0;
                                            while (while_method_0(v717)){
                                                v715[v717] = v704;
                                                v717 += 1 ;
                                            }
                                            static_array<int,2> v719;
                                            int v721;
                                            v721 = 0;
                                            while (while_method_0(v721)){
                                                bool v723;
                                                v723 = 0 <= v721;
                                                bool v725;
                                                if (v723){
                                                    bool v724;
                                                    v724 = v721 < 2;
                                                    v725 = v724;
                                                } else {
                                                    v725 = false;
                                                }
                                                bool v726;
                                                v726 = v725 == false;
                                                if (v726){
                                                    assert("Index must be in range." && v725);
                                                } else {
                                                }
                                                int v728;
                                                v728 = v715[v721];
                                                bool v730;
                                                v730 = v721 == v674;
                                                int v732;
                                                if (v730){
                                                    int v731;
                                                    v731 = v728 + 4;
                                                    v732 = v731;
                                                } else {
                                                    v732 = v728;
                                                }
                                                v719[v721] = v732;
                                                v721 += 1 ;
                                            }
                                            v784 = Union4{Union4_2{v671, false, v673, v701, v719, v702}};
                                        } else {
                                            printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                            __trap();
                                        }
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
                        v792 = Union3{Union3_1{v784}};
                        break;
                    }
                    case 3: { // T_none
                        v792 = Union3{Union3_0{}};
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
        v20 = v792;
    }
    return ;
}
__device__ inline bool while_method_12(int v0){
    bool v1;
    v1 = v0 < 16384;
    return v1;
}
__device__ inline bool while_method_13(int v0){
    bool v1;
    v1 = v0 < 256;
    return v1;
}
__device__ void method_48(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut1 & v3, Union4 v4){
    v3.v0 = 63u;
    static_array<float,2> v5;
    v5[0] = 0.0f;
    v5[1] = 0.0f;
    v3.v4 = v5;
    static_array_list<Union7,32> & v7 = v3.v2;
    v7.unsafe_set_length(0);
    static_array<Union2,2> v8;
    Union2 v10;
    v10 = Union2{Union2_0{}};
    v8[0] = v10;
    Union2 v12;
    v12 = Union2{Union2_0{}};
    v8[1] = v12;
    v3.v3 = v8;
    static_array_list<Union7,32> & v14 = v3.v2;
    Union3 v15;
    v15 = Union3{Union3_1{v4}};
    Union3 v16;
    v16 = v15;
    while (while_method_11(v16)){
        Union3 v788;
        switch (v16.tag) {
            case 0: { // None
                v788 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v18 = v16.case1.v0;
                Union14 v628;
                switch (v18.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v597 = v18.case0.v0; bool v598 = v18.case0.v1; static_array<Union6,2> v599 = v18.case0.v2; int v600 = v18.case0.v3; static_array<int,2> v601 = v18.case0.v4; int v602 = v18.case0.v5;
                        curandStatePhilox4_32_10_t & v603 = v3.v5;
                        curandStatePhilox4_32_10_t & v604 = v603;
                        unsigned int & v605 = v3.v0;
                        Union6 v606; unsigned int v607;
                        Tuple6 tmp58 = draw_card_20(v604, v605);
                        v606 = tmp58.v0; v607 = tmp58.v1;
                        v3.v0 = v607;
                        Union7 v608;
                        v608 = Union7{Union7_0{v606}};
                        v14.push(v608);
                        v628 = Union14{Union14_0{v597, v598, v599, v600, v601, v602, v606}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v610 = v3.v5;
                        curandStatePhilox4_32_10_t & v611 = v610;
                        unsigned int & v612 = v3.v0;
                        Union6 v613; unsigned int v614;
                        Tuple6 tmp59 = draw_card_20(v611, v612);
                        v613 = tmp59.v0; v614 = tmp59.v1;
                        v3.v0 = v614;
                        curandStatePhilox4_32_10_t & v615 = v3.v5;
                        curandStatePhilox4_32_10_t & v616 = v615;
                        unsigned int & v617 = v3.v0;
                        Union6 v618; unsigned int v619;
                        Tuple6 tmp60 = draw_card_20(v616, v617);
                        v618 = tmp60.v0; v619 = tmp60.v1;
                        v3.v0 = v619;
                        Union7 v620;
                        v620 = Union7{Union7_2{0, v613}};
                        v14.push(v620);
                        Union7 v621;
                        v621 = Union7{Union7_2{1, v618}};
                        v14.push(v621);
                        v628 = Union14{Union14_1{v613, v618}};
                        break;
                    }
                    case 2: { // Round
                        Union5 v68 = v18.case2.v0; bool v69 = v18.case2.v1; static_array<Union6,2> v70 = v18.case2.v2; int v71 = v18.case2.v3; static_array<int,2> v72 = v18.case2.v4; int v73 = v18.case2.v5;
                        static_array<Union2,2> & v74 = v3.v3;
                        bool v75;
                        v75 = 0 <= v71;
                        bool v77;
                        if (v75){
                            bool v76;
                            v76 = v71 < 2;
                            v77 = v76;
                        } else {
                            v77 = false;
                        }
                        bool v78;
                        v78 = v77 == false;
                        if (v78){
                            assert("Index must be in range." && v77);
                        } else {
                        }
                        Union2 v80;
                        v80 = v74[v71];
                        Union1 v585;
                        switch (v80.tag) {
                            case 0: { // Computer
                                static_array_list<Union7,32> & v83 = v3.v2;
                                curandStatePhilox4_32_10_t & v84 = v3.v5;
                                curandStatePhilox4_32_10_t & v85 = v84;
                                float * v86;
                                v86 = reinterpret_cast<float *>(&v1[55050240ull]);
                                float * v88;
                                v88 = reinterpret_cast<float *>(&v1[0ull]);
                                float * v90;
                                v90 = reinterpret_cast<float *>(&v1[0ull]);
                                int v92;
                                v92 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v92 && v92 < 24);
                                int v93;
                                v93 = 32768 * v92;
                                int v94;
                                v94 = threadIdx.x;
                                int v95;
                                v95 = v94;
                                while (while_method_3(v95)){
                                    bool v97;
                                    v97 = 0 <= v95;
                                    bool v98;
                                    v98 = v97 == false;
                                    if (v98){
                                        assert("The index needs to be zero or positive." && v97);
                                    } else {
                                    }
                                    int v100;
                                    v100 = v95 % 128;
                                    int v101;
                                    v101 = v95 / 128;
                                    bool v102;
                                    v102 = v101 < 256;
                                    bool v103;
                                    v103 = v102 == false;
                                    if (v103){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v102);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v101 && v101 < 256);
                                    assert("Tensor range check" && 0 <= v100 && v100 < 128);
                                    int v105;
                                    v105 = v100 + v93;
                                    int v106;
                                    v106 = 128 * v101;
                                    int v107;
                                    v107 = v106 + v105;
                                    v90[v107] = 0.0f;
                                    v95 += 256 ;
                                }
                                __syncthreads();
                                int v108;
                                v108 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v108 && v108 < 256);
                                int v109;
                                v109 = 128 * v108;
                                int v110;
                                v110 = v109 + v93;
                                static_array_list<Union9,10> v111;
                                v111 = static_array_list<Union9,10>{};
                                int v113;
                                v113 = v83.length;
                                int v114;
                                v114 = 0;
                                while (while_method_1(v113, v114)){
                                    Union7 v116;
                                    v116 = v83[v114];
                                    Union10 v135;
                                    switch (v116.tag) {
                                        case 0: { // CommunityCardIs
                                            Union6 v125 = v116.case0.v0;
                                            Union9 v126;
                                            v126 = Union9{Union9_1{v125}};
                                            v135 = Union10{Union10_1{v126}};
                                            break;
                                        }
                                        case 1: { // PlayerAction
                                            int v128 = v116.case1.v0; Union1 v129 = v116.case1.v1;
                                            Union9 v130;
                                            v130 = Union9{Union9_0{v129}};
                                            v135 = Union10{Union10_1{v130}};
                                            break;
                                        }
                                        case 2: { // PlayerGotCard
                                            int v118 = v116.case2.v0; Union6 v119 = v116.case2.v1;
                                            bool v120;
                                            v120 = v118 == v71;
                                            if (v120){
                                                Union9 v121;
                                                v121 = Union9{Union9_1{v119}};
                                                v135 = Union10{Union10_1{v121}};
                                            } else {
                                                v135 = Union10{Union10_0{}};
                                            }
                                            break;
                                        }
                                        default: {
                                            v135 = Union10{Union10_0{}};
                                        }
                                    }
                                    switch (v135.tag) {
                                        case 0: { // None
                                            break;
                                        }
                                        case 1: { // Some
                                            Union9 v136 = v135.case1.v0;
                                            v111.push(v136);
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false); __trap();
                                        }
                                    }
                                    v114 += 1 ;
                                }
                                float * v137;
                                v137 = v90+v110;
                                int v139;
                                v139 = v111.length;
                                bool v140;
                                v140 = v139 == 0;
                                if (v140){
                                    v137[0] = 1.0f;
                                } else {
                                }
                                int v141;
                                v141 = v111.length;
                                int v142;
                                v142 = 0;
                                while (while_method_1(v141, v142)){
                                    Union9 v144;
                                    v144 = v111[v142];
                                    int v146;
                                    v146 = v142 * 6;
                                    int v147;
                                    v147 = 1 + v146;
                                    switch (v144.tag) {
                                        case 0: { // C1of2
                                            Union1 v148 = v144.case0.v0;
                                            switch (v148.tag) {
                                                case 0: { // Call
                                                    v137[v147] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // Fold
                                                    int v149;
                                                    v149 = v147 + 1;
                                                    v137[v149] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // Raise
                                                    int v150;
                                                    v150 = v147 + 2;
                                                    v137[v150] = 1.0f;
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false); __trap();
                                                }
                                            }
                                            break;
                                        }
                                        case 1: { // C2of2
                                            Union6 v151 = v144.case1.v0;
                                            int v152;
                                            v152 = v147 + 3;
                                            switch (v151.tag) {
                                                case 0: { // Jack
                                                    v137[v152] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // King
                                                    int v153;
                                                    v153 = v152 + 1;
                                                    v137[v153] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // Queen
                                                    int v154;
                                                    v154 = v152 + 2;
                                                    v137[v154] = 1.0f;
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
                                    v142 += 1 ;
                                }
                                __syncthreads();
                                int v155;
                                v155 = 0;
                                while (while_method_10(v155)){
                                    float * v157;
                                    v157 = reinterpret_cast<float *>(&v1[55050240ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 32);
                                    int v159;
                                    v159 = 393216 * v155;
                                    float * v160;
                                    v160 = reinterpret_cast<float *>(&v1[4718592ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 32);
                                    float * v162;
                                    v162 = reinterpret_cast<float *>(&v1[0ull]);
                                    float * v164;
                                    v164 = reinterpret_cast<float *>(&v0[0ull]);
                                    float * v166;
                                    v166 = reinterpret_cast<float *>(&v2[0ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 32);
                                    int v168;
                                    v168 = 8192 * v155;
                                    float * v169;
                                    v169 = reinterpret_cast<float *>(&v1[3145728ull]);
                                    block_matmul_23(v169, v164, v168, v162);
                                    block_map_24(v160, v159, v169);
                                    block_row_map_25(v157, v159, v160);
                                    int * v171;
                                    v171 = reinterpret_cast<int *>(&v0[1048576ull]);
                                    float * v173;
                                    v173 = reinterpret_cast<float *>(&v0[1048592ull]);
                                    float * v175;
                                    v175 = reinterpret_cast<float *>(&v0[1048720ull]);
                                    double * v177;
                                    v177 = reinterpret_cast<double *>(&v1[105381888ull]);
                                    double * v179;
                                    v179 = reinterpret_cast<double *>(&v1[108527616ull]);
                                    v155 += 1 ;
                                }
                                __syncthreads();
                                int * v181;
                                v181 = reinterpret_cast<int *>(&v0[1048576ull]);
                                float * v183;
                                v183 = reinterpret_cast<float *>(&v0[1048592ull]);
                                float * v185;
                                v185 = reinterpret_cast<float *>(&v0[1048720ull]);
                                int v187;
                                v187 = v181[0];
                                float * v188;
                                v188 = reinterpret_cast<float *>(&v1[55050240ull]);
                                assert("Tensor range check" && 0 <= v187 && v187 < 32);
                                int v190;
                                v190 = 393216 * v187;
                                int v191;
                                v191 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v191 && v191 < 24);
                                int v192;
                                v192 = 16384 * v191;
                                int v193;
                                v193 = v192 + v190;
                                int v194;
                                v194 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v194 && v194 < 256);
                                int v195;
                                v195 = 64 * v194;
                                int v196;
                                v196 = v195 + v193;
                                float * v197;
                                v197 = v188+v196;
                                int v199;
                                v199 = sizeof(float *);
                                unsigned long long v200;
                                v200 = (unsigned long long)v199;
                                unsigned long long v201;
                                v201 = 256ull * v200;
                                unsigned long long v202;
                                v202 = v201 + 16ull;
                                unsigned long long v203;
                                v203 = v202 - 1ull;
                                unsigned long long v204;
                                v204 = v203 % 16ull;
                                unsigned long long v205;
                                v205 = v203 - v204;
                                unsigned long long v206;
                                v206 = v205 + 1024ull;
                                unsigned long long v207;
                                v207 = v206 + 16ull;
                                unsigned long long v208;
                                v208 = v207 - 1ull;
                                unsigned long long v209;
                                v209 = v208 % 16ull;
                                unsigned long long v210;
                                v210 = v208 - v209;
                                unsigned long long v211;
                                v211 = v210 + 1024ull;
                                bool v212;
                                v212 = v211 <= 98304ull;
                                bool v213;
                                v213 = v212 == false;
                                if (v213){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v212);
                                } else {
                                }
                                extern __shared__ unsigned char v215[];
                                bool v216;
                                v216 = v211 <= v211;
                                bool v217;
                                v217 = v216 == false;
                                if (v217){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v216);
                                } else {
                                }
                                float * * v219;
                                v219 = reinterpret_cast<float * *>(&v215[0ull]);
                                float * v221;
                                v221 = reinterpret_cast<float *>(&v215[v205]);
                                int * v223;
                                v223 = reinterpret_cast<int *>(&v215[v210]);
                                int v225;
                                v225 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v225 && v225 < 256);
                                v219[v225] = v197;
                                __syncthreads();
                                bool v226;
                                v226 = 0 <= v225;
                                bool v227;
                                v227 = v226 == false;
                                if (v227){
                                    assert("The index needs to be zero or positive." && v226);
                                } else {
                                }
                                int v229;
                                v229 = v225 % 16;
                                int v230;
                                v230 = v225 / 16;
                                bool v231;
                                v231 = v230 < 16;
                                bool v232;
                                v232 = v231 == false;
                                if (v232){
                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v231);
                                } else {
                                }
                                assert("Tensor range check" && 0 <= v230 && v230 < 16);
                                int v234;
                                v234 = 0;
                                while (while_method_8(v234)){
                                    bool v236;
                                    v236 = 0 <= v230;
                                    bool v237;
                                    v237 = v236 && v231;
                                    bool v238;
                                    v238 = v237 == false;
                                    if (v238){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v237);
                                    } else {
                                    }
                                    bool v240;
                                    v240 = 0 <= v234;
                                    bool v242;
                                    if (v240){
                                        bool v241;
                                        v241 = v234 < 16;
                                        v242 = v241;
                                    } else {
                                        v242 = false;
                                    }
                                    bool v243;
                                    v243 = v242 == false;
                                    if (v243){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v242);
                                    } else {
                                    }
                                    int v245;
                                    v245 = v234 * 16;
                                    int v246;
                                    v246 = v245 + v230;
                                    assert("Tensor range check" && 0 <= v234 && v234 < 16);
                                    int v247;
                                    v247 = 16 * v234;
                                    int v248;
                                    v248 = v247 + v230;
                                    float * v249;
                                    v249 = v219[v248];
                                    int v250;
                                    v250 = blockIdx.x;
                                    int v251;
                                    v251 = v250 * 256;
                                    int v252;
                                    v252 = v251 + v246;
                                    assert("Tensor range check" && 0 <= v229 && v229 < 16);
                                    int v253;
                                    v253 = 4 * v229;
                                    float v254[4];
                                    int v255[4];
                                    int v256;
                                    v256 = 0;
                                    while (while_method_4(v256)){
                                        assert("Tensor range check" && 0 <= v256 && v256 < 1);
                                        int v258;
                                        v258 = 4 * v256;
                                        assert("Tensor range check" && 0 <= v256 && v256 < 1);
                                        int v259;
                                        v259 = 64 * v256;
                                        int v260;
                                        v260 = v259 + v253;
                                        int4* v261;
                                        v261 = reinterpret_cast<int4*>(v249 + v260);
                                        int4* v262;
                                        v262 = reinterpret_cast<int4*>(v254 + v258);
                                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v261) % 16 == 0 && reinterpret_cast<unsigned long long>(v262) % 16 == 0);
                                        *v262 = *v261;
                                        v256 += 1 ;
                                    }
                                    int v263;
                                    v263 = 0;
                                    while (while_method_4(v263)){
                                        int v265;
                                        v265 = 0;
                                        while (while_method_7(v265)){
                                            bool v267;
                                            v267 = 0 <= v265;
                                            bool v269;
                                            if (v267){
                                                bool v268;
                                                v268 = v265 < 4;
                                                v269 = v268;
                                            } else {
                                                v269 = false;
                                            }
                                            bool v270;
                                            v270 = v269 == false;
                                            if (v270){
                                                assert("The indices should be inside the range of the dimension." && v269);
                                            } else {
                                            }
                                            bool v272;
                                            v272 = 0 <= v229;
                                            bool v274;
                                            if (v272){
                                                bool v273;
                                                v273 = v229 < 16;
                                                v274 = v273;
                                            } else {
                                                v274 = false;
                                            }
                                            bool v275;
                                            v275 = v274 == false;
                                            if (v275){
                                                assert("The indices should be inside the range of the dimension." && v274);
                                            } else {
                                            }
                                            int v277;
                                            v277 = v229 * 4;
                                            int v278;
                                            v278 = v265 + v277;
                                            bool v279;
                                            v279 = 0 <= v263;
                                            bool v281;
                                            if (v279){
                                                bool v280;
                                                v280 = v263 < 1;
                                                v281 = v280;
                                            } else {
                                                v281 = false;
                                            }
                                            bool v282;
                                            v282 = v281 == false;
                                            if (v282){
                                                assert("The indices should be inside the range of the dimension." && v281);
                                            } else {
                                            }
                                            int v284;
                                            v284 = v263 * 64;
                                            int v285;
                                            v285 = v278 + v284;
                                            assert("Tensor range check" && 0 <= v263 && v263 < 1);
                                            assert("Tensor range check" && 0 <= v265 && v265 < 4);
                                            int v286;
                                            v286 = 4 * v263;
                                            int v287;
                                            v287 = v286 + v265;
                                            v255[v287] = v285;
                                            v265 += 1 ;
                                        }
                                        v263 += 1 ;
                                    }
                                    float v288[4];
                                    float v289;
                                    v289 = 0.0f;
                                    int v290;
                                    v290 = 0;
                                    while (while_method_4(v290)){
                                        assert("Tensor range check" && 0 <= v290 && v290 < 1);
                                        int v292;
                                        v292 = 4 * v290;
                                        assert("Tensor range check" && 0 <= v290 && v290 < 1);
                                        int v293; float v294;
                                        Tuple8 tmp61 = Tuple8{0, 0.0f};
                                        v293 = tmp61.v0; v294 = tmp61.v1;
                                        while (while_method_7(v293)){
                                            assert("Tensor range check" && 0 <= v293 && v293 < 4);
                                            int v296;
                                            v296 = v293 + v292;
                                            float v297;
                                            v297 = v254[v296];
                                            float v298;
                                            v298 = v294 + v297;
                                            v294 = v298;
                                            v293 += 1 ;
                                        }
                                        auto v299 = cooperative_groups::coalesced_threads();
                                        int v300;
                                        v300 = threadIdx.x;
                                        int v301;
                                        v301 = v300 / 16;
                                        auto v302 = cooperative_groups::labeled_partition(v299,v301);
                                        Closure2 v303{};
                                        float v304;
                                        v304 = cooperative_groups::inclusive_scan(v302, v294, v303);
                                        float v305;
                                        v305 = v302.shfl_up(v304,1);
                                        bool v306;
                                        v306 = v302.thread_rank() == 0;
                                        float v307;
                                        if (v306){
                                            v307 = 0.0f;
                                        } else {
                                            v307 = v305;
                                        }
                                        float v308;
                                        v308 = v302.shfl(v304,v302.num_threads()-1);
                                        float v309;
                                        v309 = v289 + v307;
                                        int v310; float v311;
                                        Tuple8 tmp62 = Tuple8{0, v309};
                                        v310 = tmp62.v0; v311 = tmp62.v1;
                                        while (while_method_7(v310)){
                                            assert("Tensor range check" && 0 <= v310 && v310 < 4);
                                            int v313;
                                            v313 = v310 + v292;
                                            float v314;
                                            v314 = v254[v313];
                                            float v315;
                                            v315 = v311 + v314;
                                            assert("Tensor range check" && 0 <= v310 && v310 < 4);
                                            v288[v313] = v315;
                                            v311 = v315;
                                            v310 += 1 ;
                                        }
                                        float v316;
                                        v316 = v289 + v308;
                                        v289 = v316;
                                        v290 += 1 ;
                                    }
                                    float v317[4];
                                    bool v318[4];
                                    int v319;
                                    v319 = 0;
                                    while (while_method_4(v319)){
                                        int v321;
                                        v321 = 0;
                                        while (while_method_7(v321)){
                                            assert("Tensor range check" && 0 <= v319 && v319 < 1);
                                            assert("Tensor range check" && 0 <= v321 && v321 < 4);
                                            int v323;
                                            v323 = 4 * v319;
                                            int v324;
                                            v324 = v323 + v321;
                                            float v325;
                                            v325 = v288[v324];
                                            float v326;
                                            v326 = v254[v324];
                                            bool v327;
                                            v327 = v326 > 0.0f;
                                            assert("Tensor range check" && 0 <= v319 && v319 < 1);
                                            assert("Tensor range check" && 0 <= v321 && v321 < 4);
                                            v317[v324] = v325;
                                            v318[v324] = v327;
                                            v321 += 1 ;
                                        }
                                        v319 += 1 ;
                                    }
                                    float v328; bool v329;
                                    Tuple9 tmp63 = Tuple9{-1.0f / 0.0f, false};
                                    v328 = tmp63.v0; v329 = tmp63.v1;
                                    int v330;
                                    v330 = 0;
                                    while (while_method_4(v330)){
                                        int v332;
                                        v332 = 0;
                                        while (while_method_7(v332)){
                                            assert("Tensor range check" && 0 <= v330 && v330 < 1);
                                            assert("Tensor range check" && 0 <= v332 && v332 < 4);
                                            int v334;
                                            v334 = 4 * v330;
                                            int v335;
                                            v335 = v334 + v332;
                                            float v336;
                                            v336 = v317[v335];
                                            bool v337;
                                            v337 = v318[v335];
                                            float v344; bool v345;
                                            if (v329){
                                                if (v337){
                                                    bool v338;
                                                    v338 = v328 >= v336;
                                                    float v339;
                                                    if (v338){
                                                        v339 = v328;
                                                    } else {
                                                        v339 = v336;
                                                    }
                                                    v344 = v339; v345 = true;
                                                } else {
                                                    v344 = v328; v345 = v329;
                                                }
                                            } else {
                                                if (v337){
                                                    v344 = v336; v345 = v337;
                                                } else {
                                                    v344 = v328; v345 = v329;
                                                }
                                            }
                                            v328 = v344;
                                            v329 = v345;
                                            v332 += 1 ;
                                        }
                                        v330 += 1 ;
                                    }
                                    auto v346 = cooperative_groups::coalesced_threads();
                                    int v347;
                                    v347 = threadIdx.x;
                                    int v348;
                                    v348 = v347 / 16;
                                    auto v349 = cooperative_groups::labeled_partition(v346,v348);
                                    Closure3 v350{};
                                    float v351; bool v352;
                                    Tuple9 tmp64 = cooperative_groups::reduce(v349, Tuple9{v328, v329}, v350);
                                    v351 = tmp64.v0; v352 = tmp64.v1;
                                    bool v353;
                                    v353 = v352 == false;
                                    if (v353){
                                        assert("The local reduce must be true." && v352);
                                    } else {
                                    }
                                    float v355[4];
                                    int v356[4];
                                    int v357;
                                    v357 = 0;
                                    while (while_method_4(v357)){
                                        int v359;
                                        v359 = 0;
                                        while (while_method_7(v359)){
                                            assert("Tensor range check" && 0 <= v357 && v357 < 1);
                                            assert("Tensor range check" && 0 <= v359 && v359 < 4);
                                            int v361;
                                            v361 = 4 * v357;
                                            int v362;
                                            v362 = v361 + v359;
                                            int v363;
                                            v363 = v255[v362];
                                            float v364;
                                            v364 = curand_uniform(&v85);
                                            assert("Tensor range check" && 0 <= v357 && v357 < 1);
                                            assert("Tensor range check" && 0 <= v359 && v359 < 4);
                                            v355[v362] = v364;
                                            v356[v362] = v363;
                                            v359 += 1 ;
                                        }
                                        v357 += 1 ;
                                    }
                                    float v365; int v366;
                                    Tuple10 tmp65 = Tuple10{0.0f, 2147483647};
                                    v365 = tmp65.v0; v366 = tmp65.v1;
                                    int v367;
                                    v367 = 0;
                                    while (while_method_4(v367)){
                                        int v369;
                                        v369 = 0;
                                        while (while_method_7(v369)){
                                            assert("Tensor range check" && 0 <= v367 && v367 < 1);
                                            assert("Tensor range check" && 0 <= v369 && v369 < 4);
                                            int v371;
                                            v371 = 4 * v367;
                                            int v372;
                                            v372 = v371 + v369;
                                            float v373;
                                            v373 = v355[v372];
                                            int v374;
                                            v374 = v356[v372];
                                            bool v375;
                                            v375 = v366 < v374;
                                            float v376; int v377;
                                            if (v375){
                                                v376 = v365; v377 = v366;
                                            } else {
                                                v376 = v373; v377 = v374;
                                            }
                                            v365 = v376;
                                            v366 = v377;
                                            v369 += 1 ;
                                        }
                                        v367 += 1 ;
                                    }
                                    auto v378 = cooperative_groups::coalesced_threads();
                                    int v379;
                                    v379 = threadIdx.x;
                                    int v380;
                                    v380 = v379 / 16;
                                    auto v381 = cooperative_groups::labeled_partition(v378,v380);
                                    Closure4 v382{};
                                    float v383; int v384;
                                    Tuple10 tmp66 = cooperative_groups::reduce(v381, Tuple10{v365, v366}, v382);
                                    v383 = tmp66.v0; v384 = tmp66.v1;
                                    float v385;
                                    v385 = v351 * v383;
                                    int v386[4];
                                    bool v387[4];
                                    int v388;
                                    v388 = 0;
                                    while (while_method_4(v388)){
                                        int v390;
                                        v390 = 0;
                                        while (while_method_7(v390)){
                                            assert("Tensor range check" && 0 <= v388 && v388 < 1);
                                            assert("Tensor range check" && 0 <= v390 && v390 < 4);
                                            int v392;
                                            v392 = 4 * v388;
                                            int v393;
                                            v393 = v392 + v390;
                                            float v394;
                                            v394 = v317[v393];
                                            bool v395;
                                            v395 = v318[v393];
                                            int v396;
                                            v396 = v255[v393];
                                            int v399; bool v400;
                                            if (v395){
                                                float v397;
                                                v397 = v394 - v385;
                                                bool v398;
                                                v398 = v397 >= 0.0f;
                                                v399 = v396; v400 = v398;
                                            } else {
                                                v399 = 2147483647; v400 = false;
                                            }
                                            assert("Tensor range check" && 0 <= v388 && v388 < 1);
                                            assert("Tensor range check" && 0 <= v390 && v390 < 4);
                                            v386[v393] = v399;
                                            v387[v393] = v400;
                                            v390 += 1 ;
                                        }
                                        v388 += 1 ;
                                    }
                                    int v401; bool v402;
                                    Tuple11 tmp67 = Tuple11{2147483647, false};
                                    v401 = tmp67.v0; v402 = tmp67.v1;
                                    int v403;
                                    v403 = 0;
                                    while (while_method_4(v403)){
                                        int v405;
                                        v405 = 0;
                                        while (while_method_7(v405)){
                                            assert("Tensor range check" && 0 <= v403 && v403 < 1);
                                            assert("Tensor range check" && 0 <= v405 && v405 < 4);
                                            int v407;
                                            v407 = 4 * v403;
                                            int v408;
                                            v408 = v407 + v405;
                                            int v409;
                                            v409 = v386[v408];
                                            bool v410;
                                            v410 = v387[v408];
                                            int v417; bool v418;
                                            if (v402){
                                                if (v410){
                                                    bool v411;
                                                    v411 = v401 < v409;
                                                    int v412;
                                                    if (v411){
                                                        v412 = v401;
                                                    } else {
                                                        v412 = v409;
                                                    }
                                                    v417 = v412; v418 = true;
                                                } else {
                                                    v417 = v401; v418 = v402;
                                                }
                                            } else {
                                                if (v410){
                                                    v417 = v409; v418 = v410;
                                                } else {
                                                    v417 = v401; v418 = v402;
                                                }
                                            }
                                            v401 = v417;
                                            v402 = v418;
                                            v405 += 1 ;
                                        }
                                        v403 += 1 ;
                                    }
                                    auto v419 = cooperative_groups::coalesced_threads();
                                    int v420;
                                    v420 = threadIdx.x;
                                    int v421;
                                    v421 = v420 / 16;
                                    auto v422 = cooperative_groups::labeled_partition(v419,v421);
                                    Closure5 v423{};
                                    int v424; bool v425;
                                    Tuple11 tmp68 = cooperative_groups::reduce(v422, Tuple11{v401, v402}, v423);
                                    v424 = tmp68.v0; v425 = tmp68.v1;
                                    bool v426;
                                    v426 = v425 == false;
                                    if (v426){
                                        assert("The local reduce must be true." && v425);
                                    } else {
                                    }
                                    float v428; int v429;
                                    Tuple10 tmp69 = Tuple10{0.0f, 2147483647};
                                    v428 = tmp69.v0; v429 = tmp69.v1;
                                    int v430;
                                    v430 = 0;
                                    while (while_method_4(v430)){
                                        int v432;
                                        v432 = 0;
                                        while (while_method_7(v432)){
                                            assert("Tensor range check" && 0 <= v430 && v430 < 1);
                                            assert("Tensor range check" && 0 <= v432 && v432 < 4);
                                            int v434;
                                            v434 = 4 * v430;
                                            int v435;
                                            v435 = v434 + v432;
                                            float v436;
                                            v436 = v254[v435];
                                            int v437;
                                            v437 = v255[v435];
                                            bool v438;
                                            v438 = v429 == v424;
                                            float v442; int v443;
                                            if (v438){
                                                v442 = v428; v443 = v429;
                                            } else {
                                                bool v439;
                                                v439 = v437 == v424;
                                                if (v439){
                                                    v442 = v436; v443 = v437;
                                                } else {
                                                    v442 = v428; v443 = v429;
                                                }
                                            }
                                            v428 = v442;
                                            v429 = v443;
                                            v432 += 1 ;
                                        }
                                        v430 += 1 ;
                                    }
                                    auto v444 = cooperative_groups::coalesced_threads();
                                    int v445;
                                    v445 = threadIdx.x;
                                    int v446;
                                    v446 = v445 / 16;
                                    auto v447 = cooperative_groups::labeled_partition(v444,v446);
                                    Closure6 v448{v424};
                                    float v449; int v450;
                                    Tuple10 tmp70 = cooperative_groups::reduce(v447, Tuple10{v428, v429}, v448);
                                    v449 = tmp70.v0; v450 = tmp70.v1;
                                    bool v451;
                                    v451 = v450 == 2147483647;
                                    bool v452;
                                    v452 = v451 != true;
                                    bool v453;
                                    v453 = v452 == false;
                                    if (v453){
                                        assert("Expected a valid action id in get_prob." && v452);
                                    } else {
                                    }
                                    int v455;
                                    v455 = 0;
                                    while (while_method_4(v455)){
                                        assert("Tensor range check" && 0 <= v455 && v455 < 1);
                                        assert("Tensor range check" && 0 <= v455 && v455 < 1);
                                        v455 += 1 ;
                                    }
                                    assert("Tensor range check" && 0 <= v246 && v246 < 256);
                                    v221[v246] = v449;
                                    v223[v246] = v424;
                                    v234 += 1 ;
                                }
                                __syncthreads();
                                assert("Tensor range check" && 0 <= v225 && v225 < 256);
                                float v457;
                                v457 = v221[v225];
                                int v458;
                                v458 = v223[v225];
                                __syncthreads();
                                extern __shared__ unsigned char v459[];
                                float * v460;
                                v460 = reinterpret_cast<float *>(&v459[0ull]);
                                int * v462;
                                v462 = reinterpret_cast<int *>(&v459[16ull]);
                                int v464;
                                v464 = threadIdx.x;
                                bool v465;
                                v465 = v464 == 0;
                                if (v465){
                                    v460[0] = v457;
                                    v462[0] = v458;
                                } else {
                                }
                                __syncthreads();
                                float v466;
                                v466 = v460[0];
                                int v467;
                                v467 = v462[0];
                                __syncthreads();
                                double * v468;
                                v468 = reinterpret_cast<double *>(&v1[105381888ull]);
                                double * v470;
                                v470 = reinterpret_cast<double *>(&v1[108527616ull]);
                                int v472;
                                v472 = threadIdx.x;
                                int v473;
                                v473 = blockIdx.x;
                                int v474;
                                v474 = v473 * 256;
                                int v475;
                                v475 = v472 + v474;
                                int v476;
                                v476 = 0;
                                while (while_method_10(v476)){
                                    float * v478;
                                    v478 = reinterpret_cast<float *>(&v1[55050240ull]);
                                    int v480;
                                    v480 = blockIdx.x;
                                    int v481;
                                    v481 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v476 && v476 < 32);
                                    assert("Tensor range check" && 0 <= v480 && v480 < 24);
                                    assert("Tensor range check" && 0 <= v481 && v481 < 256);
                                    assert("Tensor range check" && 0 <= v467 && v467 < 64);
                                    int v482;
                                    v482 = 64 * v481;
                                    int v483;
                                    v483 = v482 + v467;
                                    int v484;
                                    v484 = 16384 * v480;
                                    int v485;
                                    v485 = v484 + v483;
                                    int v486;
                                    v486 = 393216 * v476;
                                    int v487;
                                    v487 = v486 + v485;
                                    float v488;
                                    v488 = v478[v487];
                                    double v489;
                                    v489 = (double)v466;
                                    double v490;
                                    v490 = log(v489);
                                    double v491;
                                    v491 = (double)v488;
                                    double v492;
                                    v492 = log(v491);
                                    assert("Tensor range check" && 0 <= v476 && v476 < 32);
                                    assert("Tensor range check" && 0 <= v475 && v475 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    int v493;
                                    v493 = 2 * v475;
                                    int v494;
                                    v494 = v493 + v71;
                                    int v495;
                                    v495 = 12288 * v476;
                                    int v496;
                                    v496 = v495 + v494;
                                    double v497;
                                    v497 = v468[v496];
                                    double v498;
                                    v498 = v470[v496];
                                    double v499;
                                    v499 = v492 + v497;
                                    double v500;
                                    v500 = v490 + v498;
                                    assert("Tensor range check" && 0 <= v476 && v476 < 32);
                                    assert("Tensor range check" && 0 <= v475 && v475 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    v468[v496] = v499;
                                    v470[v496] = v500;
                                    v476 += 1 ;
                                }
                                bool v501;
                                v501 = 0 == v467;
                                Union12 v510;
                                if (v501){
                                    v510 = Union12{Union12_1{}};
                                } else {
                                    bool v503;
                                    v503 = 1 == v467;
                                    if (v503){
                                        v510 = Union12{Union12_0{}};
                                    } else {
                                        bool v505;
                                        v505 = 2 == v467;
                                        if (v505){
                                            v510 = Union12{Union12_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v510.tag) {
                                    case 0: { // AA_Call
                                        v585 = Union1{Union1_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v511;
                                        v511 = v72[0];
                                        int v513; int v514;
                                        Tuple7 tmp71 = Tuple7{1, v511};
                                        v513 = tmp71.v0; v514 = tmp71.v1;
                                        while (while_method_0(v513)){
                                            bool v516;
                                            v516 = 0 <= v513;
                                            bool v518;
                                            if (v516){
                                                bool v517;
                                                v517 = v513 < 2;
                                                v518 = v517;
                                            } else {
                                                v518 = false;
                                            }
                                            bool v519;
                                            v519 = v518 == false;
                                            if (v519){
                                                assert("Index must be in range." && v518);
                                            } else {
                                            }
                                            int v521;
                                            v521 = v72[v513];
                                            bool v523;
                                            v523 = v514 >= v521;
                                            int v524;
                                            if (v523){
                                                v524 = v514;
                                            } else {
                                                v524 = v521;
                                            }
                                            v514 = v524;
                                            v513 += 1 ;
                                        }
                                        bool v526;
                                        if (v75){
                                            bool v525;
                                            v525 = v71 < 2;
                                            v526 = v525;
                                        } else {
                                            v526 = false;
                                        }
                                        bool v527;
                                        v527 = v526 == false;
                                        if (v527){
                                            assert("Index must be in range." && v526);
                                        } else {
                                        }
                                        int v529;
                                        v529 = v72[v71];
                                        bool v531;
                                        v531 = v529 == v514;
                                        if (v531){
                                            v585 = Union1{Union1_0{}};
                                        } else {
                                            v585 = Union1{Union1_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v536;
                                        v536 = v73 > 0;
                                        if (v536){
                                            v585 = Union1{Union1_2{}};
                                        } else {
                                            v585 = Union1{Union1_0{}};
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
                                curandStatePhilox4_32_10_t & v543 = v3.v5;
                                curandStatePhilox4_32_10_t & v544 = v543;
                                static_array_list<Union1,3> v545;
                                v545 = static_array_list<Union1,3>{};
                                v545.unsafe_set_length(1);
                                Union1 v547;
                                v547 = Union1{Union1_0{}};
                                v545[0] = v547;
                                int v549;
                                v549 = v72[0];
                                int v551;
                                v551 = v72[1];
                                bool v553;
                                v553 = v549 == v551;
                                bool v554;
                                v554 = v553 != true;
                                if (v554){
                                    Union1 v555;
                                    v555 = Union1{Union1_1{}};
                                    v545.push(v555);
                                } else {
                                }
                                bool v556;
                                v556 = v73 > 0;
                                if (v556){
                                    Union1 v557;
                                    v557 = Union1{Union1_2{}};
                                    v545.push(v557);
                                } else {
                                }
                                int v558;
                                v558 = v545.length;
                                int v559;
                                v559 = v558 - 1;
                                int v560;
                                v560 = 0;
                                while (while_method_1(v559, v560)){
                                    int v562;
                                    v562 = v545.length;
                                    int v563;
                                    v563 = int_range_22(v562, v560, v544);
                                    Union1 v564;
                                    v564 = v545[v560];
                                    Union1 v566;
                                    v566 = v545[v563];
                                    v545[v560] = v566;
                                    v545[v563] = v564;
                                    v560 += 1 ;
                                }
                                Union1 v568;
                                v568 = v545.pop();
                                int v569;
                                v569 = sizeof(Union1);
                                unsigned long long v570;
                                v570 = (unsigned long long)v569;
                                bool v571;
                                v571 = v570 <= 98304ull;
                                bool v572;
                                v572 = v571 == false;
                                if (v572){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v571);
                                } else {
                                }
                                extern __shared__ unsigned char v574[];
                                bool v575;
                                v575 = v570 <= v570;
                                bool v576;
                                v576 = v575 == false;
                                if (v576){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v575);
                                } else {
                                }
                                Union1 * v578;
                                v578 = reinterpret_cast<Union1 *>(&v574[0ull]);
                                int v580;
                                v580 = threadIdx.x;
                                bool v581;
                                v581 = v580 == 0;
                                if (v581){
                                    v578[0] = v568;
                                } else {
                                }
                                __syncthreads();
                                Union1 v582;
                                v582 = v578[0];
                                __syncthreads();
                                v585 = v582;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union7 v586;
                        v586 = Union7{Union7_1{v71, v585}};
                        v14.push(v586);
                        v628 = Union14{Union14_2{v68, v69, v70, v71, v72, v73, v585}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v588 = v18.case3.v0; bool v589 = v18.case3.v1; static_array<Union6,2> v590 = v18.case3.v2; int v591 = v18.case3.v3; static_array<int,2> v592 = v18.case3.v4; int v593 = v18.case3.v5; Union1 v594 = v18.case3.v6;
                        Union7 v595;
                        v595 = Union7{Union7_1{v591, v594}};
                        v14.push(v595);
                        v628 = Union14{Union14_2{v588, v589, v590, v591, v592, v593, v594}};
                        break;
                    }
                    case 4: { // TerminalCall
                        Union5 v39 = v18.case4.v0; bool v40 = v18.case4.v1; static_array<Union6,2> v41 = v18.case4.v2; int v42 = v18.case4.v3; static_array<int,2> v43 = v18.case4.v4; int v44 = v18.case4.v5;
                        bool v45;
                        v45 = 0 <= v42;
                        bool v47;
                        if (v45){
                            bool v46;
                            v46 = v42 < 2;
                            v47 = v46;
                        } else {
                            v47 = false;
                        }
                        bool v48;
                        v48 = v47 == false;
                        if (v48){
                            assert("Index must be in range." && v47);
                        } else {
                        }
                        int v50;
                        v50 = v43[v42];
                        Union13 v52;
                        v52 = compare_hands_26(v39, v40, v41, v42, v43, v44);
                        int v57; int v58;
                        switch (v52.tag) {
                            case 0: { // Eq
                                v57 = 0; v58 = -1;
                                break;
                            }
                            case 1: { // Gt
                                v57 = v50; v58 = 0;
                                break;
                            }
                            case 2: { // Lt
                                v57 = v50; v58 = 1;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        int v59;
                        v59 = -v58;
                        bool v60;
                        v60 = v58 >= v59;
                        int v61;
                        if (v60){
                            v61 = v58;
                        } else {
                            v61 = v59;
                        }
                        float v62;
                        v62 = (float)v57;
                        static_array<float,2> & v63 = v3.v4;
                        v63[v61] = v62;
                        int v64;
                        v64 = v61 ^ 1;
                        float v65;
                        v65 = -v62;
                        v63[v64] = v65;
                        Union7 v66;
                        v66 = Union7{Union7_3{v41, v57, v58}};
                        v14.push(v66);
                        v628 = Union14{Union14_3{}};
                        break;
                    }
                    case 5: { // TerminalFold
                        Union5 v19 = v18.case5.v0; bool v20 = v18.case5.v1; static_array<Union6,2> v21 = v18.case5.v2; int v22 = v18.case5.v3; static_array<int,2> v23 = v18.case5.v4; int v24 = v18.case5.v5;
                        bool v25;
                        v25 = 0 <= v22;
                        bool v27;
                        if (v25){
                            bool v26;
                            v26 = v22 < 2;
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
                        int v30;
                        v30 = v23[v22];
                        int v32;
                        v32 = -v30;
                        float v33;
                        v33 = (float)v32;
                        static_array<float,2> & v34 = v3.v4;
                        v34[v22] = v33;
                        int v35;
                        v35 = v22 ^ 1;
                        float v36;
                        v36 = -v33;
                        v34[v35] = v36;
                        Union7 v37;
                        v37 = Union7{Union7_3{v21, v30, v35}};
                        v14.push(v37);
                        v628 = Union14{Union14_3{}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                switch (v628.tag) {
                    case 0: { // T_game_chance_community_card
                        Union5 v630 = v628.case0.v0; bool v631 = v628.case0.v1; static_array<Union6,2> v632 = v628.case0.v2; int v633 = v628.case0.v3; static_array<int,2> v634 = v628.case0.v4; int v635 = v628.case0.v5; Union6 v636 = v628.case0.v6;
                        int v637;
                        v637 = 2;
                        int v638; int v639;
                        Tuple7 tmp72 = Tuple7{0, 0};
                        v638 = tmp72.v0; v639 = tmp72.v1;
                        while (while_method_0(v638)){
                            bool v641;
                            v641 = 0 <= v638;
                            bool v643;
                            if (v641){
                                bool v642;
                                v642 = v638 < 2;
                                v643 = v642;
                            } else {
                                v643 = false;
                            }
                            bool v644;
                            v644 = v643 == false;
                            if (v644){
                                assert("Index must be in range." && v643);
                            } else {
                            }
                            int v646;
                            v646 = v634[v638];
                            bool v648;
                            v648 = v639 >= v646;
                            int v649;
                            if (v648){
                                v649 = v639;
                            } else {
                                v649 = v646;
                            }
                            v639 = v649;
                            v638 += 1 ;
                        }
                        static_array<int,2> v650;
                        int v652;
                        v652 = 0;
                        while (while_method_0(v652)){
                            v650[v652] = v639;
                            v652 += 1 ;
                        }
                        Union5 v654;
                        v654 = Union5{Union5_1{v636}};
                        Union4 v655;
                        v655 = Union4{Union4_2{v654, true, v632, 0, v650, v637}};
                        v788 = Union3{Union3_1{v655}};
                        break;
                    }
                    case 1: { // T_game_chance_init
                        Union6 v657 = v628.case1.v0; Union6 v658 = v628.case1.v1;
                        int v659;
                        v659 = 2;
                        static_array<int,2> v660;
                        v660[0] = 1;
                        v660[1] = 1;
                        static_array<Union6,2> v662;
                        v662[0] = v657;
                        v662[1] = v658;
                        Union5 v664;
                        v664 = Union5{Union5_0{}};
                        Union4 v665;
                        v665 = Union4{Union4_2{v664, true, v662, 0, v660, v659}};
                        v788 = Union3{Union3_1{v665}};
                        break;
                    }
                    case 2: { // T_game_round
                        Union5 v667 = v628.case2.v0; bool v668 = v628.case2.v1; static_array<Union6,2> v669 = v628.case2.v2; int v670 = v628.case2.v3; static_array<int,2> v671 = v628.case2.v4; int v672 = v628.case2.v5; Union1 v673 = v628.case2.v6;
                        Union4 v780;
                        switch (v667.tag) {
                            case 0: { // None
                                switch (v673.tag) {
                                    case 0: { // Call
                                        if (v668){
                                            int v736;
                                            v736 = v670 ^ 1;
                                            v780 = Union4{Union4_2{v667, false, v669, v736, v671, v672}};
                                        } else {
                                            v780 = Union4{Union4_0{v667, v668, v669, v670, v671, v672}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v780 = Union4{Union4_5{v667, v668, v669, v670, v671, v672}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v740;
                                        v740 = v672 > 0;
                                        if (v740){
                                            int v741;
                                            v741 = v670 ^ 1;
                                            int v742;
                                            v742 = -1 + v672;
                                            int v743; int v744;
                                            Tuple7 tmp73 = Tuple7{0, 0};
                                            v743 = tmp73.v0; v744 = tmp73.v1;
                                            while (while_method_0(v743)){
                                                bool v746;
                                                v746 = 0 <= v743;
                                                bool v748;
                                                if (v746){
                                                    bool v747;
                                                    v747 = v743 < 2;
                                                    v748 = v747;
                                                } else {
                                                    v748 = false;
                                                }
                                                bool v749;
                                                v749 = v748 == false;
                                                if (v749){
                                                    assert("Index must be in range." && v748);
                                                } else {
                                                }
                                                int v751;
                                                v751 = v671[v743];
                                                bool v753;
                                                v753 = v744 >= v751;
                                                int v754;
                                                if (v753){
                                                    v754 = v744;
                                                } else {
                                                    v754 = v751;
                                                }
                                                v744 = v754;
                                                v743 += 1 ;
                                            }
                                            static_array<int,2> v755;
                                            int v757;
                                            v757 = 0;
                                            while (while_method_0(v757)){
                                                v755[v757] = v744;
                                                v757 += 1 ;
                                            }
                                            static_array<int,2> v759;
                                            int v761;
                                            v761 = 0;
                                            while (while_method_0(v761)){
                                                bool v763;
                                                v763 = 0 <= v761;
                                                bool v765;
                                                if (v763){
                                                    bool v764;
                                                    v764 = v761 < 2;
                                                    v765 = v764;
                                                } else {
                                                    v765 = false;
                                                }
                                                bool v766;
                                                v766 = v765 == false;
                                                if (v766){
                                                    assert("Index must be in range." && v765);
                                                } else {
                                                }
                                                int v768;
                                                v768 = v755[v761];
                                                bool v770;
                                                v770 = v761 == v670;
                                                int v772;
                                                if (v770){
                                                    int v771;
                                                    v771 = v768 + 2;
                                                    v772 = v771;
                                                } else {
                                                    v772 = v768;
                                                }
                                                v759[v761] = v772;
                                                v761 += 1 ;
                                            }
                                            v780 = Union4{Union4_2{v667, false, v669, v741, v759, v742}};
                                        } else {
                                            printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                            __trap();
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                break;
                            }
                            case 1: { // Some
                                Union6 v674 = v667.case1.v0;
                                switch (v673.tag) {
                                    case 0: { // Call
                                        if (v668){
                                            int v676;
                                            v676 = v670 ^ 1;
                                            v780 = Union4{Union4_2{v667, false, v669, v676, v671, v672}};
                                        } else {
                                            int v678; int v679;
                                            Tuple7 tmp74 = Tuple7{0, 0};
                                            v678 = tmp74.v0; v679 = tmp74.v1;
                                            while (while_method_0(v678)){
                                                bool v681;
                                                v681 = 0 <= v678;
                                                bool v683;
                                                if (v681){
                                                    bool v682;
                                                    v682 = v678 < 2;
                                                    v683 = v682;
                                                } else {
                                                    v683 = false;
                                                }
                                                bool v684;
                                                v684 = v683 == false;
                                                if (v684){
                                                    assert("Index must be in range." && v683);
                                                } else {
                                                }
                                                int v686;
                                                v686 = v671[v678];
                                                bool v688;
                                                v688 = v679 >= v686;
                                                int v689;
                                                if (v688){
                                                    v689 = v679;
                                                } else {
                                                    v689 = v686;
                                                }
                                                v679 = v689;
                                                v678 += 1 ;
                                            }
                                            static_array<int,2> v690;
                                            int v692;
                                            v692 = 0;
                                            while (while_method_0(v692)){
                                                v690[v692] = v679;
                                                v692 += 1 ;
                                            }
                                            v780 = Union4{Union4_4{v667, v668, v669, v670, v690, v672}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v780 = Union4{Union4_5{v667, v668, v669, v670, v671, v672}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v696;
                                        v696 = v672 > 0;
                                        if (v696){
                                            int v697;
                                            v697 = v670 ^ 1;
                                            int v698;
                                            v698 = -1 + v672;
                                            int v699; int v700;
                                            Tuple7 tmp75 = Tuple7{0, 0};
                                            v699 = tmp75.v0; v700 = tmp75.v1;
                                            while (while_method_0(v699)){
                                                bool v702;
                                                v702 = 0 <= v699;
                                                bool v704;
                                                if (v702){
                                                    bool v703;
                                                    v703 = v699 < 2;
                                                    v704 = v703;
                                                } else {
                                                    v704 = false;
                                                }
                                                bool v705;
                                                v705 = v704 == false;
                                                if (v705){
                                                    assert("Index must be in range." && v704);
                                                } else {
                                                }
                                                int v707;
                                                v707 = v671[v699];
                                                bool v709;
                                                v709 = v700 >= v707;
                                                int v710;
                                                if (v709){
                                                    v710 = v700;
                                                } else {
                                                    v710 = v707;
                                                }
                                                v700 = v710;
                                                v699 += 1 ;
                                            }
                                            static_array<int,2> v711;
                                            int v713;
                                            v713 = 0;
                                            while (while_method_0(v713)){
                                                v711[v713] = v700;
                                                v713 += 1 ;
                                            }
                                            static_array<int,2> v715;
                                            int v717;
                                            v717 = 0;
                                            while (while_method_0(v717)){
                                                bool v719;
                                                v719 = 0 <= v717;
                                                bool v721;
                                                if (v719){
                                                    bool v720;
                                                    v720 = v717 < 2;
                                                    v721 = v720;
                                                } else {
                                                    v721 = false;
                                                }
                                                bool v722;
                                                v722 = v721 == false;
                                                if (v722){
                                                    assert("Index must be in range." && v721);
                                                } else {
                                                }
                                                int v724;
                                                v724 = v711[v717];
                                                bool v726;
                                                v726 = v717 == v670;
                                                int v728;
                                                if (v726){
                                                    int v727;
                                                    v727 = v724 + 4;
                                                    v728 = v727;
                                                } else {
                                                    v728 = v724;
                                                }
                                                v715[v717] = v728;
                                                v717 += 1 ;
                                            }
                                            v780 = Union4{Union4_2{v667, false, v669, v697, v715, v698}};
                                        } else {
                                            printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                            __trap();
                                        }
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
                        v788 = Union3{Union3_1{v780}};
                        break;
                    }
                    case 3: { // T_none
                        v788 = Union3{Union3_0{}};
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
        v16 = v788;
    }
    return ;
}
__device__ void method_49(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut1 & v3, Union4 v4){
    v3.v0 = 63u;
    static_array<float,2> v5;
    v5[0] = 0.0f;
    v5[1] = 0.0f;
    v3.v4 = v5;
    static_array_list<Union7,32> & v7 = v3.v2;
    v7.unsafe_set_length(0);
    static_array<Union2,2> v8;
    Union2 v10;
    v10 = Union2{Union2_0{}};
    v8[0] = v10;
    Union2 v12;
    v12 = Union2{Union2_0{}};
    v8[1] = v12;
    v3.v3 = v8;
    static_array_list<Union7,32> & v14 = v3.v2;
    Union3 v15;
    v15 = Union3{Union3_1{v4}};
    Union3 v16;
    v16 = v15;
    while (while_method_11(v16)){
        Union3 v796;
        switch (v16.tag) {
            case 0: { // None
                v796 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v18 = v16.case1.v0;
                Union14 v636;
                switch (v18.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v605 = v18.case0.v0; bool v606 = v18.case0.v1; static_array<Union6,2> v607 = v18.case0.v2; int v608 = v18.case0.v3; static_array<int,2> v609 = v18.case0.v4; int v610 = v18.case0.v5;
                        curandStatePhilox4_32_10_t & v611 = v3.v5;
                        curandStatePhilox4_32_10_t & v612 = v611;
                        unsigned int & v613 = v3.v0;
                        Union6 v614; unsigned int v615;
                        Tuple6 tmp78 = draw_card_20(v612, v613);
                        v614 = tmp78.v0; v615 = tmp78.v1;
                        v3.v0 = v615;
                        Union7 v616;
                        v616 = Union7{Union7_0{v614}};
                        v14.push(v616);
                        v636 = Union14{Union14_0{v605, v606, v607, v608, v609, v610, v614}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v618 = v3.v5;
                        curandStatePhilox4_32_10_t & v619 = v618;
                        unsigned int & v620 = v3.v0;
                        Union6 v621; unsigned int v622;
                        Tuple6 tmp79 = draw_card_20(v619, v620);
                        v621 = tmp79.v0; v622 = tmp79.v1;
                        v3.v0 = v622;
                        curandStatePhilox4_32_10_t & v623 = v3.v5;
                        curandStatePhilox4_32_10_t & v624 = v623;
                        unsigned int & v625 = v3.v0;
                        Union6 v626; unsigned int v627;
                        Tuple6 tmp80 = draw_card_20(v624, v625);
                        v626 = tmp80.v0; v627 = tmp80.v1;
                        v3.v0 = v627;
                        Union7 v628;
                        v628 = Union7{Union7_2{0, v621}};
                        v14.push(v628);
                        Union7 v629;
                        v629 = Union7{Union7_2{1, v626}};
                        v14.push(v629);
                        v636 = Union14{Union14_1{v621, v626}};
                        break;
                    }
                    case 2: { // Round
                        Union5 v68 = v18.case2.v0; bool v69 = v18.case2.v1; static_array<Union6,2> v70 = v18.case2.v2; int v71 = v18.case2.v3; static_array<int,2> v72 = v18.case2.v4; int v73 = v18.case2.v5;
                        static_array<Union2,2> & v74 = v3.v3;
                        bool v75;
                        v75 = 0 <= v71;
                        bool v77;
                        if (v75){
                            bool v76;
                            v76 = v71 < 2;
                            v77 = v76;
                        } else {
                            v77 = false;
                        }
                        bool v78;
                        v78 = v77 == false;
                        if (v78){
                            assert("Index must be in range." && v77);
                        } else {
                        }
                        Union2 v80;
                        v80 = v74[v71];
                        Union1 v593;
                        switch (v80.tag) {
                            case 0: { // Computer
                                static_array_list<Union7,32> & v83 = v3.v2;
                                curandStatePhilox4_32_10_t & v84 = v3.v5;
                                curandStatePhilox4_32_10_t & v85 = v84;
                                float * v86;
                                v86 = reinterpret_cast<float *>(&v1[55050240ull]);
                                float * v88;
                                v88 = reinterpret_cast<float *>(&v1[0ull]);
                                float * v90;
                                v90 = reinterpret_cast<float *>(&v1[0ull]);
                                int v92;
                                v92 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v92 && v92 < 24);
                                int v93;
                                v93 = 32768 * v92;
                                int v94;
                                v94 = threadIdx.x;
                                int v95;
                                v95 = v94;
                                while (while_method_3(v95)){
                                    bool v97;
                                    v97 = 0 <= v95;
                                    bool v98;
                                    v98 = v97 == false;
                                    if (v98){
                                        assert("The index needs to be zero or positive." && v97);
                                    } else {
                                    }
                                    int v100;
                                    v100 = v95 % 128;
                                    int v101;
                                    v101 = v95 / 128;
                                    bool v102;
                                    v102 = v101 < 256;
                                    bool v103;
                                    v103 = v102 == false;
                                    if (v103){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v102);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v101 && v101 < 256);
                                    assert("Tensor range check" && 0 <= v100 && v100 < 128);
                                    int v105;
                                    v105 = v100 + v93;
                                    int v106;
                                    v106 = 128 * v101;
                                    int v107;
                                    v107 = v106 + v105;
                                    v90[v107] = 0.0f;
                                    v95 += 256 ;
                                }
                                __syncthreads();
                                int v108;
                                v108 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v108 && v108 < 256);
                                int v109;
                                v109 = 128 * v108;
                                int v110;
                                v110 = v109 + v93;
                                static_array_list<Union9,10> v111;
                                v111 = static_array_list<Union9,10>{};
                                int v113;
                                v113 = v83.length;
                                int v114;
                                v114 = 0;
                                while (while_method_1(v113, v114)){
                                    Union7 v116;
                                    v116 = v83[v114];
                                    Union10 v135;
                                    switch (v116.tag) {
                                        case 0: { // CommunityCardIs
                                            Union6 v125 = v116.case0.v0;
                                            Union9 v126;
                                            v126 = Union9{Union9_1{v125}};
                                            v135 = Union10{Union10_1{v126}};
                                            break;
                                        }
                                        case 1: { // PlayerAction
                                            int v128 = v116.case1.v0; Union1 v129 = v116.case1.v1;
                                            Union9 v130;
                                            v130 = Union9{Union9_0{v129}};
                                            v135 = Union10{Union10_1{v130}};
                                            break;
                                        }
                                        case 2: { // PlayerGotCard
                                            int v118 = v116.case2.v0; Union6 v119 = v116.case2.v1;
                                            bool v120;
                                            v120 = v118 == v71;
                                            if (v120){
                                                Union9 v121;
                                                v121 = Union9{Union9_1{v119}};
                                                v135 = Union10{Union10_1{v121}};
                                            } else {
                                                v135 = Union10{Union10_0{}};
                                            }
                                            break;
                                        }
                                        default: {
                                            v135 = Union10{Union10_0{}};
                                        }
                                    }
                                    switch (v135.tag) {
                                        case 0: { // None
                                            break;
                                        }
                                        case 1: { // Some
                                            Union9 v136 = v135.case1.v0;
                                            v111.push(v136);
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false); __trap();
                                        }
                                    }
                                    v114 += 1 ;
                                }
                                float * v137;
                                v137 = v90+v110;
                                int v139;
                                v139 = v111.length;
                                bool v140;
                                v140 = v139 == 0;
                                if (v140){
                                    v137[0] = 1.0f;
                                } else {
                                }
                                int v141;
                                v141 = v111.length;
                                int v142;
                                v142 = 0;
                                while (while_method_1(v141, v142)){
                                    Union9 v144;
                                    v144 = v111[v142];
                                    int v146;
                                    v146 = v142 * 6;
                                    int v147;
                                    v147 = 1 + v146;
                                    switch (v144.tag) {
                                        case 0: { // C1of2
                                            Union1 v148 = v144.case0.v0;
                                            switch (v148.tag) {
                                                case 0: { // Call
                                                    v137[v147] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // Fold
                                                    int v149;
                                                    v149 = v147 + 1;
                                                    v137[v149] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // Raise
                                                    int v150;
                                                    v150 = v147 + 2;
                                                    v137[v150] = 1.0f;
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false); __trap();
                                                }
                                            }
                                            break;
                                        }
                                        case 1: { // C2of2
                                            Union6 v151 = v144.case1.v0;
                                            int v152;
                                            v152 = v147 + 3;
                                            switch (v151.tag) {
                                                case 0: { // Jack
                                                    v137[v152] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // King
                                                    int v153;
                                                    v153 = v152 + 1;
                                                    v137[v153] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // Queen
                                                    int v154;
                                                    v154 = v152 + 2;
                                                    v137[v154] = 1.0f;
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
                                    v142 += 1 ;
                                }
                                __syncthreads();
                                int v155;
                                v155 = 0;
                                while (while_method_10(v155)){
                                    float * v157;
                                    v157 = reinterpret_cast<float *>(&v1[55050240ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 32);
                                    int v159;
                                    v159 = 393216 * v155;
                                    float * v160;
                                    v160 = reinterpret_cast<float *>(&v1[4718592ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 32);
                                    float * v162;
                                    v162 = reinterpret_cast<float *>(&v1[0ull]);
                                    float * v164;
                                    v164 = reinterpret_cast<float *>(&v0[0ull]);
                                    float * v166;
                                    v166 = reinterpret_cast<float *>(&v2[0ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 32);
                                    int v168;
                                    v168 = 8192 * v155;
                                    float * v169;
                                    v169 = reinterpret_cast<float *>(&v1[3145728ull]);
                                    block_matmul_23(v169, v164, v168, v162);
                                    block_map_24(v160, v159, v169);
                                    block_row_map_25(v157, v159, v160);
                                    int * v171;
                                    v171 = reinterpret_cast<int *>(&v0[1048576ull]);
                                    float * v173;
                                    v173 = reinterpret_cast<float *>(&v0[1048592ull]);
                                    float * v175;
                                    v175 = reinterpret_cast<float *>(&v0[1048720ull]);
                                    double * v177;
                                    v177 = reinterpret_cast<double *>(&v1[105381888ull]);
                                    double * v179;
                                    v179 = reinterpret_cast<double *>(&v1[108527616ull]);
                                    v155 += 1 ;
                                }
                                __syncthreads();
                                int * v181;
                                v181 = reinterpret_cast<int *>(&v0[1048576ull]);
                                float * v183;
                                v183 = reinterpret_cast<float *>(&v0[1048592ull]);
                                float * v185;
                                v185 = reinterpret_cast<float *>(&v0[1048720ull]);
                                int v187;
                                v187 = 0;
                                int v188;
                                v188 = 32;
                                int v189;
                                v189 = int_range_22(v188, v187, v85);
                                extern __shared__ unsigned char v190[];
                                int * v191;
                                v191 = reinterpret_cast<int *>(&v190[0ull]);
                                int v193;
                                v193 = threadIdx.x;
                                bool v194;
                                v194 = v193 == 0;
                                if (v194){
                                    v191[0] = v189;
                                } else {
                                }
                                __syncthreads();
                                int v195;
                                v195 = v191[0];
                                __syncthreads();
                                float * v196;
                                v196 = reinterpret_cast<float *>(&v1[55050240ull]);
                                assert("Tensor range check" && 0 <= v195 && v195 < 32);
                                int v198;
                                v198 = 393216 * v195;
                                int v199;
                                v199 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v199 && v199 < 24);
                                int v200;
                                v200 = 16384 * v199;
                                int v201;
                                v201 = v200 + v198;
                                int v202;
                                v202 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v202 && v202 < 256);
                                int v203;
                                v203 = 64 * v202;
                                int v204;
                                v204 = v203 + v201;
                                float * v205;
                                v205 = v196+v204;
                                int v207;
                                v207 = sizeof(float *);
                                unsigned long long v208;
                                v208 = (unsigned long long)v207;
                                unsigned long long v209;
                                v209 = 256ull * v208;
                                unsigned long long v210;
                                v210 = v209 + 16ull;
                                unsigned long long v211;
                                v211 = v210 - 1ull;
                                unsigned long long v212;
                                v212 = v211 % 16ull;
                                unsigned long long v213;
                                v213 = v211 - v212;
                                unsigned long long v214;
                                v214 = v213 + 1024ull;
                                unsigned long long v215;
                                v215 = v214 + 16ull;
                                unsigned long long v216;
                                v216 = v215 - 1ull;
                                unsigned long long v217;
                                v217 = v216 % 16ull;
                                unsigned long long v218;
                                v218 = v216 - v217;
                                unsigned long long v219;
                                v219 = v218 + 1024ull;
                                bool v220;
                                v220 = v219 <= 98304ull;
                                bool v221;
                                v221 = v220 == false;
                                if (v221){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v220);
                                } else {
                                }
                                extern __shared__ unsigned char v223[];
                                bool v224;
                                v224 = v219 <= v219;
                                bool v225;
                                v225 = v224 == false;
                                if (v225){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v224);
                                } else {
                                }
                                float * * v227;
                                v227 = reinterpret_cast<float * *>(&v223[0ull]);
                                float * v229;
                                v229 = reinterpret_cast<float *>(&v223[v213]);
                                int * v231;
                                v231 = reinterpret_cast<int *>(&v223[v218]);
                                int v233;
                                v233 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v233 && v233 < 256);
                                v227[v233] = v205;
                                __syncthreads();
                                bool v234;
                                v234 = 0 <= v233;
                                bool v235;
                                v235 = v234 == false;
                                if (v235){
                                    assert("The index needs to be zero or positive." && v234);
                                } else {
                                }
                                int v237;
                                v237 = v233 % 16;
                                int v238;
                                v238 = v233 / 16;
                                bool v239;
                                v239 = v238 < 16;
                                bool v240;
                                v240 = v239 == false;
                                if (v240){
                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v239);
                                } else {
                                }
                                assert("Tensor range check" && 0 <= v238 && v238 < 16);
                                int v242;
                                v242 = 0;
                                while (while_method_8(v242)){
                                    bool v244;
                                    v244 = 0 <= v238;
                                    bool v245;
                                    v245 = v244 && v239;
                                    bool v246;
                                    v246 = v245 == false;
                                    if (v246){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v245);
                                    } else {
                                    }
                                    bool v248;
                                    v248 = 0 <= v242;
                                    bool v250;
                                    if (v248){
                                        bool v249;
                                        v249 = v242 < 16;
                                        v250 = v249;
                                    } else {
                                        v250 = false;
                                    }
                                    bool v251;
                                    v251 = v250 == false;
                                    if (v251){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v250);
                                    } else {
                                    }
                                    int v253;
                                    v253 = v242 * 16;
                                    int v254;
                                    v254 = v253 + v238;
                                    assert("Tensor range check" && 0 <= v242 && v242 < 16);
                                    int v255;
                                    v255 = 16 * v242;
                                    int v256;
                                    v256 = v255 + v238;
                                    float * v257;
                                    v257 = v227[v256];
                                    int v258;
                                    v258 = blockIdx.x;
                                    int v259;
                                    v259 = v258 * 256;
                                    int v260;
                                    v260 = v259 + v254;
                                    assert("Tensor range check" && 0 <= v237 && v237 < 16);
                                    int v261;
                                    v261 = 4 * v237;
                                    float v262[4];
                                    int v263[4];
                                    int v264;
                                    v264 = 0;
                                    while (while_method_4(v264)){
                                        assert("Tensor range check" && 0 <= v264 && v264 < 1);
                                        int v266;
                                        v266 = 4 * v264;
                                        assert("Tensor range check" && 0 <= v264 && v264 < 1);
                                        int v267;
                                        v267 = 64 * v264;
                                        int v268;
                                        v268 = v267 + v261;
                                        int4* v269;
                                        v269 = reinterpret_cast<int4*>(v257 + v268);
                                        int4* v270;
                                        v270 = reinterpret_cast<int4*>(v262 + v266);
                                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v269) % 16 == 0 && reinterpret_cast<unsigned long long>(v270) % 16 == 0);
                                        *v270 = *v269;
                                        v264 += 1 ;
                                    }
                                    int v271;
                                    v271 = 0;
                                    while (while_method_4(v271)){
                                        int v273;
                                        v273 = 0;
                                        while (while_method_7(v273)){
                                            bool v275;
                                            v275 = 0 <= v273;
                                            bool v277;
                                            if (v275){
                                                bool v276;
                                                v276 = v273 < 4;
                                                v277 = v276;
                                            } else {
                                                v277 = false;
                                            }
                                            bool v278;
                                            v278 = v277 == false;
                                            if (v278){
                                                assert("The indices should be inside the range of the dimension." && v277);
                                            } else {
                                            }
                                            bool v280;
                                            v280 = 0 <= v237;
                                            bool v282;
                                            if (v280){
                                                bool v281;
                                                v281 = v237 < 16;
                                                v282 = v281;
                                            } else {
                                                v282 = false;
                                            }
                                            bool v283;
                                            v283 = v282 == false;
                                            if (v283){
                                                assert("The indices should be inside the range of the dimension." && v282);
                                            } else {
                                            }
                                            int v285;
                                            v285 = v237 * 4;
                                            int v286;
                                            v286 = v273 + v285;
                                            bool v287;
                                            v287 = 0 <= v271;
                                            bool v289;
                                            if (v287){
                                                bool v288;
                                                v288 = v271 < 1;
                                                v289 = v288;
                                            } else {
                                                v289 = false;
                                            }
                                            bool v290;
                                            v290 = v289 == false;
                                            if (v290){
                                                assert("The indices should be inside the range of the dimension." && v289);
                                            } else {
                                            }
                                            int v292;
                                            v292 = v271 * 64;
                                            int v293;
                                            v293 = v286 + v292;
                                            assert("Tensor range check" && 0 <= v271 && v271 < 1);
                                            assert("Tensor range check" && 0 <= v273 && v273 < 4);
                                            int v294;
                                            v294 = 4 * v271;
                                            int v295;
                                            v295 = v294 + v273;
                                            v263[v295] = v293;
                                            v273 += 1 ;
                                        }
                                        v271 += 1 ;
                                    }
                                    float v296[4];
                                    float v297;
                                    v297 = 0.0f;
                                    int v298;
                                    v298 = 0;
                                    while (while_method_4(v298)){
                                        assert("Tensor range check" && 0 <= v298 && v298 < 1);
                                        int v300;
                                        v300 = 4 * v298;
                                        assert("Tensor range check" && 0 <= v298 && v298 < 1);
                                        int v301; float v302;
                                        Tuple8 tmp81 = Tuple8{0, 0.0f};
                                        v301 = tmp81.v0; v302 = tmp81.v1;
                                        while (while_method_7(v301)){
                                            assert("Tensor range check" && 0 <= v301 && v301 < 4);
                                            int v304;
                                            v304 = v301 + v300;
                                            float v305;
                                            v305 = v262[v304];
                                            float v306;
                                            v306 = v302 + v305;
                                            v302 = v306;
                                            v301 += 1 ;
                                        }
                                        auto v307 = cooperative_groups::coalesced_threads();
                                        int v308;
                                        v308 = threadIdx.x;
                                        int v309;
                                        v309 = v308 / 16;
                                        auto v310 = cooperative_groups::labeled_partition(v307,v309);
                                        Closure2 v311{};
                                        float v312;
                                        v312 = cooperative_groups::inclusive_scan(v310, v302, v311);
                                        float v313;
                                        v313 = v310.shfl_up(v312,1);
                                        bool v314;
                                        v314 = v310.thread_rank() == 0;
                                        float v315;
                                        if (v314){
                                            v315 = 0.0f;
                                        } else {
                                            v315 = v313;
                                        }
                                        float v316;
                                        v316 = v310.shfl(v312,v310.num_threads()-1);
                                        float v317;
                                        v317 = v297 + v315;
                                        int v318; float v319;
                                        Tuple8 tmp82 = Tuple8{0, v317};
                                        v318 = tmp82.v0; v319 = tmp82.v1;
                                        while (while_method_7(v318)){
                                            assert("Tensor range check" && 0 <= v318 && v318 < 4);
                                            int v321;
                                            v321 = v318 + v300;
                                            float v322;
                                            v322 = v262[v321];
                                            float v323;
                                            v323 = v319 + v322;
                                            assert("Tensor range check" && 0 <= v318 && v318 < 4);
                                            v296[v321] = v323;
                                            v319 = v323;
                                            v318 += 1 ;
                                        }
                                        float v324;
                                        v324 = v297 + v316;
                                        v297 = v324;
                                        v298 += 1 ;
                                    }
                                    float v325[4];
                                    bool v326[4];
                                    int v327;
                                    v327 = 0;
                                    while (while_method_4(v327)){
                                        int v329;
                                        v329 = 0;
                                        while (while_method_7(v329)){
                                            assert("Tensor range check" && 0 <= v327 && v327 < 1);
                                            assert("Tensor range check" && 0 <= v329 && v329 < 4);
                                            int v331;
                                            v331 = 4 * v327;
                                            int v332;
                                            v332 = v331 + v329;
                                            float v333;
                                            v333 = v296[v332];
                                            float v334;
                                            v334 = v262[v332];
                                            bool v335;
                                            v335 = v334 > 0.0f;
                                            assert("Tensor range check" && 0 <= v327 && v327 < 1);
                                            assert("Tensor range check" && 0 <= v329 && v329 < 4);
                                            v325[v332] = v333;
                                            v326[v332] = v335;
                                            v329 += 1 ;
                                        }
                                        v327 += 1 ;
                                    }
                                    float v336; bool v337;
                                    Tuple9 tmp83 = Tuple9{-1.0f / 0.0f, false};
                                    v336 = tmp83.v0; v337 = tmp83.v1;
                                    int v338;
                                    v338 = 0;
                                    while (while_method_4(v338)){
                                        int v340;
                                        v340 = 0;
                                        while (while_method_7(v340)){
                                            assert("Tensor range check" && 0 <= v338 && v338 < 1);
                                            assert("Tensor range check" && 0 <= v340 && v340 < 4);
                                            int v342;
                                            v342 = 4 * v338;
                                            int v343;
                                            v343 = v342 + v340;
                                            float v344;
                                            v344 = v325[v343];
                                            bool v345;
                                            v345 = v326[v343];
                                            float v352; bool v353;
                                            if (v337){
                                                if (v345){
                                                    bool v346;
                                                    v346 = v336 >= v344;
                                                    float v347;
                                                    if (v346){
                                                        v347 = v336;
                                                    } else {
                                                        v347 = v344;
                                                    }
                                                    v352 = v347; v353 = true;
                                                } else {
                                                    v352 = v336; v353 = v337;
                                                }
                                            } else {
                                                if (v345){
                                                    v352 = v344; v353 = v345;
                                                } else {
                                                    v352 = v336; v353 = v337;
                                                }
                                            }
                                            v336 = v352;
                                            v337 = v353;
                                            v340 += 1 ;
                                        }
                                        v338 += 1 ;
                                    }
                                    auto v354 = cooperative_groups::coalesced_threads();
                                    int v355;
                                    v355 = threadIdx.x;
                                    int v356;
                                    v356 = v355 / 16;
                                    auto v357 = cooperative_groups::labeled_partition(v354,v356);
                                    Closure3 v358{};
                                    float v359; bool v360;
                                    Tuple9 tmp84 = cooperative_groups::reduce(v357, Tuple9{v336, v337}, v358);
                                    v359 = tmp84.v0; v360 = tmp84.v1;
                                    bool v361;
                                    v361 = v360 == false;
                                    if (v361){
                                        assert("The local reduce must be true." && v360);
                                    } else {
                                    }
                                    float v363[4];
                                    int v364[4];
                                    int v365;
                                    v365 = 0;
                                    while (while_method_4(v365)){
                                        int v367;
                                        v367 = 0;
                                        while (while_method_7(v367)){
                                            assert("Tensor range check" && 0 <= v365 && v365 < 1);
                                            assert("Tensor range check" && 0 <= v367 && v367 < 4);
                                            int v369;
                                            v369 = 4 * v365;
                                            int v370;
                                            v370 = v369 + v367;
                                            int v371;
                                            v371 = v263[v370];
                                            float v372;
                                            v372 = curand_uniform(&v85);
                                            assert("Tensor range check" && 0 <= v365 && v365 < 1);
                                            assert("Tensor range check" && 0 <= v367 && v367 < 4);
                                            v363[v370] = v372;
                                            v364[v370] = v371;
                                            v367 += 1 ;
                                        }
                                        v365 += 1 ;
                                    }
                                    float v373; int v374;
                                    Tuple10 tmp85 = Tuple10{0.0f, 2147483647};
                                    v373 = tmp85.v0; v374 = tmp85.v1;
                                    int v375;
                                    v375 = 0;
                                    while (while_method_4(v375)){
                                        int v377;
                                        v377 = 0;
                                        while (while_method_7(v377)){
                                            assert("Tensor range check" && 0 <= v375 && v375 < 1);
                                            assert("Tensor range check" && 0 <= v377 && v377 < 4);
                                            int v379;
                                            v379 = 4 * v375;
                                            int v380;
                                            v380 = v379 + v377;
                                            float v381;
                                            v381 = v363[v380];
                                            int v382;
                                            v382 = v364[v380];
                                            bool v383;
                                            v383 = v374 < v382;
                                            float v384; int v385;
                                            if (v383){
                                                v384 = v373; v385 = v374;
                                            } else {
                                                v384 = v381; v385 = v382;
                                            }
                                            v373 = v384;
                                            v374 = v385;
                                            v377 += 1 ;
                                        }
                                        v375 += 1 ;
                                    }
                                    auto v386 = cooperative_groups::coalesced_threads();
                                    int v387;
                                    v387 = threadIdx.x;
                                    int v388;
                                    v388 = v387 / 16;
                                    auto v389 = cooperative_groups::labeled_partition(v386,v388);
                                    Closure4 v390{};
                                    float v391; int v392;
                                    Tuple10 tmp86 = cooperative_groups::reduce(v389, Tuple10{v373, v374}, v390);
                                    v391 = tmp86.v0; v392 = tmp86.v1;
                                    float v393;
                                    v393 = v359 * v391;
                                    int v394[4];
                                    bool v395[4];
                                    int v396;
                                    v396 = 0;
                                    while (while_method_4(v396)){
                                        int v398;
                                        v398 = 0;
                                        while (while_method_7(v398)){
                                            assert("Tensor range check" && 0 <= v396 && v396 < 1);
                                            assert("Tensor range check" && 0 <= v398 && v398 < 4);
                                            int v400;
                                            v400 = 4 * v396;
                                            int v401;
                                            v401 = v400 + v398;
                                            float v402;
                                            v402 = v325[v401];
                                            bool v403;
                                            v403 = v326[v401];
                                            int v404;
                                            v404 = v263[v401];
                                            int v407; bool v408;
                                            if (v403){
                                                float v405;
                                                v405 = v402 - v393;
                                                bool v406;
                                                v406 = v405 >= 0.0f;
                                                v407 = v404; v408 = v406;
                                            } else {
                                                v407 = 2147483647; v408 = false;
                                            }
                                            assert("Tensor range check" && 0 <= v396 && v396 < 1);
                                            assert("Tensor range check" && 0 <= v398 && v398 < 4);
                                            v394[v401] = v407;
                                            v395[v401] = v408;
                                            v398 += 1 ;
                                        }
                                        v396 += 1 ;
                                    }
                                    int v409; bool v410;
                                    Tuple11 tmp87 = Tuple11{2147483647, false};
                                    v409 = tmp87.v0; v410 = tmp87.v1;
                                    int v411;
                                    v411 = 0;
                                    while (while_method_4(v411)){
                                        int v413;
                                        v413 = 0;
                                        while (while_method_7(v413)){
                                            assert("Tensor range check" && 0 <= v411 && v411 < 1);
                                            assert("Tensor range check" && 0 <= v413 && v413 < 4);
                                            int v415;
                                            v415 = 4 * v411;
                                            int v416;
                                            v416 = v415 + v413;
                                            int v417;
                                            v417 = v394[v416];
                                            bool v418;
                                            v418 = v395[v416];
                                            int v425; bool v426;
                                            if (v410){
                                                if (v418){
                                                    bool v419;
                                                    v419 = v409 < v417;
                                                    int v420;
                                                    if (v419){
                                                        v420 = v409;
                                                    } else {
                                                        v420 = v417;
                                                    }
                                                    v425 = v420; v426 = true;
                                                } else {
                                                    v425 = v409; v426 = v410;
                                                }
                                            } else {
                                                if (v418){
                                                    v425 = v417; v426 = v418;
                                                } else {
                                                    v425 = v409; v426 = v410;
                                                }
                                            }
                                            v409 = v425;
                                            v410 = v426;
                                            v413 += 1 ;
                                        }
                                        v411 += 1 ;
                                    }
                                    auto v427 = cooperative_groups::coalesced_threads();
                                    int v428;
                                    v428 = threadIdx.x;
                                    int v429;
                                    v429 = v428 / 16;
                                    auto v430 = cooperative_groups::labeled_partition(v427,v429);
                                    Closure5 v431{};
                                    int v432; bool v433;
                                    Tuple11 tmp88 = cooperative_groups::reduce(v430, Tuple11{v409, v410}, v431);
                                    v432 = tmp88.v0; v433 = tmp88.v1;
                                    bool v434;
                                    v434 = v433 == false;
                                    if (v434){
                                        assert("The local reduce must be true." && v433);
                                    } else {
                                    }
                                    float v436; int v437;
                                    Tuple10 tmp89 = Tuple10{0.0f, 2147483647};
                                    v436 = tmp89.v0; v437 = tmp89.v1;
                                    int v438;
                                    v438 = 0;
                                    while (while_method_4(v438)){
                                        int v440;
                                        v440 = 0;
                                        while (while_method_7(v440)){
                                            assert("Tensor range check" && 0 <= v438 && v438 < 1);
                                            assert("Tensor range check" && 0 <= v440 && v440 < 4);
                                            int v442;
                                            v442 = 4 * v438;
                                            int v443;
                                            v443 = v442 + v440;
                                            float v444;
                                            v444 = v262[v443];
                                            int v445;
                                            v445 = v263[v443];
                                            bool v446;
                                            v446 = v437 == v432;
                                            float v450; int v451;
                                            if (v446){
                                                v450 = v436; v451 = v437;
                                            } else {
                                                bool v447;
                                                v447 = v445 == v432;
                                                if (v447){
                                                    v450 = v444; v451 = v445;
                                                } else {
                                                    v450 = v436; v451 = v437;
                                                }
                                            }
                                            v436 = v450;
                                            v437 = v451;
                                            v440 += 1 ;
                                        }
                                        v438 += 1 ;
                                    }
                                    auto v452 = cooperative_groups::coalesced_threads();
                                    int v453;
                                    v453 = threadIdx.x;
                                    int v454;
                                    v454 = v453 / 16;
                                    auto v455 = cooperative_groups::labeled_partition(v452,v454);
                                    Closure6 v456{v432};
                                    float v457; int v458;
                                    Tuple10 tmp90 = cooperative_groups::reduce(v455, Tuple10{v436, v437}, v456);
                                    v457 = tmp90.v0; v458 = tmp90.v1;
                                    bool v459;
                                    v459 = v458 == 2147483647;
                                    bool v460;
                                    v460 = v459 != true;
                                    bool v461;
                                    v461 = v460 == false;
                                    if (v461){
                                        assert("Expected a valid action id in get_prob." && v460);
                                    } else {
                                    }
                                    int v463;
                                    v463 = 0;
                                    while (while_method_4(v463)){
                                        assert("Tensor range check" && 0 <= v463 && v463 < 1);
                                        assert("Tensor range check" && 0 <= v463 && v463 < 1);
                                        v463 += 1 ;
                                    }
                                    assert("Tensor range check" && 0 <= v254 && v254 < 256);
                                    v229[v254] = v457;
                                    v231[v254] = v432;
                                    v242 += 1 ;
                                }
                                __syncthreads();
                                assert("Tensor range check" && 0 <= v233 && v233 < 256);
                                float v465;
                                v465 = v229[v233];
                                int v466;
                                v466 = v231[v233];
                                __syncthreads();
                                extern __shared__ unsigned char v467[];
                                float * v468;
                                v468 = reinterpret_cast<float *>(&v467[0ull]);
                                int * v470;
                                v470 = reinterpret_cast<int *>(&v467[16ull]);
                                int v472;
                                v472 = threadIdx.x;
                                bool v473;
                                v473 = v472 == 0;
                                if (v473){
                                    v468[0] = v465;
                                    v470[0] = v466;
                                } else {
                                }
                                __syncthreads();
                                float v474;
                                v474 = v468[0];
                                int v475;
                                v475 = v470[0];
                                __syncthreads();
                                double * v476;
                                v476 = reinterpret_cast<double *>(&v1[105381888ull]);
                                double * v478;
                                v478 = reinterpret_cast<double *>(&v1[108527616ull]);
                                int v480;
                                v480 = threadIdx.x;
                                int v481;
                                v481 = blockIdx.x;
                                int v482;
                                v482 = v481 * 256;
                                int v483;
                                v483 = v480 + v482;
                                int v484;
                                v484 = 0;
                                while (while_method_10(v484)){
                                    float * v486;
                                    v486 = reinterpret_cast<float *>(&v1[55050240ull]);
                                    int v488;
                                    v488 = blockIdx.x;
                                    int v489;
                                    v489 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v484 && v484 < 32);
                                    assert("Tensor range check" && 0 <= v488 && v488 < 24);
                                    assert("Tensor range check" && 0 <= v489 && v489 < 256);
                                    assert("Tensor range check" && 0 <= v475 && v475 < 64);
                                    int v490;
                                    v490 = 64 * v489;
                                    int v491;
                                    v491 = v490 + v475;
                                    int v492;
                                    v492 = 16384 * v488;
                                    int v493;
                                    v493 = v492 + v491;
                                    int v494;
                                    v494 = 393216 * v484;
                                    int v495;
                                    v495 = v494 + v493;
                                    float v496;
                                    v496 = v486[v495];
                                    double v497;
                                    v497 = (double)v474;
                                    double v498;
                                    v498 = log(v497);
                                    double v499;
                                    v499 = (double)v496;
                                    double v500;
                                    v500 = log(v499);
                                    assert("Tensor range check" && 0 <= v484 && v484 < 32);
                                    assert("Tensor range check" && 0 <= v483 && v483 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    int v501;
                                    v501 = 2 * v483;
                                    int v502;
                                    v502 = v501 + v71;
                                    int v503;
                                    v503 = 12288 * v484;
                                    int v504;
                                    v504 = v503 + v502;
                                    double v505;
                                    v505 = v476[v504];
                                    double v506;
                                    v506 = v478[v504];
                                    double v507;
                                    v507 = v500 + v505;
                                    double v508;
                                    v508 = v498 + v506;
                                    assert("Tensor range check" && 0 <= v484 && v484 < 32);
                                    assert("Tensor range check" && 0 <= v483 && v483 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    v476[v504] = v507;
                                    v478[v504] = v508;
                                    v484 += 1 ;
                                }
                                bool v509;
                                v509 = 0 == v475;
                                Union12 v518;
                                if (v509){
                                    v518 = Union12{Union12_1{}};
                                } else {
                                    bool v511;
                                    v511 = 1 == v475;
                                    if (v511){
                                        v518 = Union12{Union12_0{}};
                                    } else {
                                        bool v513;
                                        v513 = 2 == v475;
                                        if (v513){
                                            v518 = Union12{Union12_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v518.tag) {
                                    case 0: { // AA_Call
                                        v593 = Union1{Union1_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v519;
                                        v519 = v72[0];
                                        int v521; int v522;
                                        Tuple7 tmp91 = Tuple7{1, v519};
                                        v521 = tmp91.v0; v522 = tmp91.v1;
                                        while (while_method_0(v521)){
                                            bool v524;
                                            v524 = 0 <= v521;
                                            bool v526;
                                            if (v524){
                                                bool v525;
                                                v525 = v521 < 2;
                                                v526 = v525;
                                            } else {
                                                v526 = false;
                                            }
                                            bool v527;
                                            v527 = v526 == false;
                                            if (v527){
                                                assert("Index must be in range." && v526);
                                            } else {
                                            }
                                            int v529;
                                            v529 = v72[v521];
                                            bool v531;
                                            v531 = v522 >= v529;
                                            int v532;
                                            if (v531){
                                                v532 = v522;
                                            } else {
                                                v532 = v529;
                                            }
                                            v522 = v532;
                                            v521 += 1 ;
                                        }
                                        bool v534;
                                        if (v75){
                                            bool v533;
                                            v533 = v71 < 2;
                                            v534 = v533;
                                        } else {
                                            v534 = false;
                                        }
                                        bool v535;
                                        v535 = v534 == false;
                                        if (v535){
                                            assert("Index must be in range." && v534);
                                        } else {
                                        }
                                        int v537;
                                        v537 = v72[v71];
                                        bool v539;
                                        v539 = v537 == v522;
                                        if (v539){
                                            v593 = Union1{Union1_0{}};
                                        } else {
                                            v593 = Union1{Union1_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v544;
                                        v544 = v73 > 0;
                                        if (v544){
                                            v593 = Union1{Union1_2{}};
                                        } else {
                                            v593 = Union1{Union1_0{}};
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
                                curandStatePhilox4_32_10_t & v551 = v3.v5;
                                curandStatePhilox4_32_10_t & v552 = v551;
                                static_array_list<Union1,3> v553;
                                v553 = static_array_list<Union1,3>{};
                                v553.unsafe_set_length(1);
                                Union1 v555;
                                v555 = Union1{Union1_0{}};
                                v553[0] = v555;
                                int v557;
                                v557 = v72[0];
                                int v559;
                                v559 = v72[1];
                                bool v561;
                                v561 = v557 == v559;
                                bool v562;
                                v562 = v561 != true;
                                if (v562){
                                    Union1 v563;
                                    v563 = Union1{Union1_1{}};
                                    v553.push(v563);
                                } else {
                                }
                                bool v564;
                                v564 = v73 > 0;
                                if (v564){
                                    Union1 v565;
                                    v565 = Union1{Union1_2{}};
                                    v553.push(v565);
                                } else {
                                }
                                int v566;
                                v566 = v553.length;
                                int v567;
                                v567 = v566 - 1;
                                int v568;
                                v568 = 0;
                                while (while_method_1(v567, v568)){
                                    int v570;
                                    v570 = v553.length;
                                    int v571;
                                    v571 = int_range_22(v570, v568, v552);
                                    Union1 v572;
                                    v572 = v553[v568];
                                    Union1 v574;
                                    v574 = v553[v571];
                                    v553[v568] = v574;
                                    v553[v571] = v572;
                                    v568 += 1 ;
                                }
                                Union1 v576;
                                v576 = v553.pop();
                                int v577;
                                v577 = sizeof(Union1);
                                unsigned long long v578;
                                v578 = (unsigned long long)v577;
                                bool v579;
                                v579 = v578 <= 98304ull;
                                bool v580;
                                v580 = v579 == false;
                                if (v580){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v579);
                                } else {
                                }
                                extern __shared__ unsigned char v582[];
                                bool v583;
                                v583 = v578 <= v578;
                                bool v584;
                                v584 = v583 == false;
                                if (v584){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v583);
                                } else {
                                }
                                Union1 * v586;
                                v586 = reinterpret_cast<Union1 *>(&v582[0ull]);
                                int v588;
                                v588 = threadIdx.x;
                                bool v589;
                                v589 = v588 == 0;
                                if (v589){
                                    v586[0] = v576;
                                } else {
                                }
                                __syncthreads();
                                Union1 v590;
                                v590 = v586[0];
                                __syncthreads();
                                v593 = v590;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union7 v594;
                        v594 = Union7{Union7_1{v71, v593}};
                        v14.push(v594);
                        v636 = Union14{Union14_2{v68, v69, v70, v71, v72, v73, v593}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v596 = v18.case3.v0; bool v597 = v18.case3.v1; static_array<Union6,2> v598 = v18.case3.v2; int v599 = v18.case3.v3; static_array<int,2> v600 = v18.case3.v4; int v601 = v18.case3.v5; Union1 v602 = v18.case3.v6;
                        Union7 v603;
                        v603 = Union7{Union7_1{v599, v602}};
                        v14.push(v603);
                        v636 = Union14{Union14_2{v596, v597, v598, v599, v600, v601, v602}};
                        break;
                    }
                    case 4: { // TerminalCall
                        Union5 v39 = v18.case4.v0; bool v40 = v18.case4.v1; static_array<Union6,2> v41 = v18.case4.v2; int v42 = v18.case4.v3; static_array<int,2> v43 = v18.case4.v4; int v44 = v18.case4.v5;
                        bool v45;
                        v45 = 0 <= v42;
                        bool v47;
                        if (v45){
                            bool v46;
                            v46 = v42 < 2;
                            v47 = v46;
                        } else {
                            v47 = false;
                        }
                        bool v48;
                        v48 = v47 == false;
                        if (v48){
                            assert("Index must be in range." && v47);
                        } else {
                        }
                        int v50;
                        v50 = v43[v42];
                        Union13 v52;
                        v52 = compare_hands_26(v39, v40, v41, v42, v43, v44);
                        int v57; int v58;
                        switch (v52.tag) {
                            case 0: { // Eq
                                v57 = 0; v58 = -1;
                                break;
                            }
                            case 1: { // Gt
                                v57 = v50; v58 = 0;
                                break;
                            }
                            case 2: { // Lt
                                v57 = v50; v58 = 1;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        int v59;
                        v59 = -v58;
                        bool v60;
                        v60 = v58 >= v59;
                        int v61;
                        if (v60){
                            v61 = v58;
                        } else {
                            v61 = v59;
                        }
                        float v62;
                        v62 = (float)v57;
                        static_array<float,2> & v63 = v3.v4;
                        v63[v61] = v62;
                        int v64;
                        v64 = v61 ^ 1;
                        float v65;
                        v65 = -v62;
                        v63[v64] = v65;
                        Union7 v66;
                        v66 = Union7{Union7_3{v41, v57, v58}};
                        v14.push(v66);
                        v636 = Union14{Union14_3{}};
                        break;
                    }
                    case 5: { // TerminalFold
                        Union5 v19 = v18.case5.v0; bool v20 = v18.case5.v1; static_array<Union6,2> v21 = v18.case5.v2; int v22 = v18.case5.v3; static_array<int,2> v23 = v18.case5.v4; int v24 = v18.case5.v5;
                        bool v25;
                        v25 = 0 <= v22;
                        bool v27;
                        if (v25){
                            bool v26;
                            v26 = v22 < 2;
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
                        int v30;
                        v30 = v23[v22];
                        int v32;
                        v32 = -v30;
                        float v33;
                        v33 = (float)v32;
                        static_array<float,2> & v34 = v3.v4;
                        v34[v22] = v33;
                        int v35;
                        v35 = v22 ^ 1;
                        float v36;
                        v36 = -v33;
                        v34[v35] = v36;
                        Union7 v37;
                        v37 = Union7{Union7_3{v21, v30, v35}};
                        v14.push(v37);
                        v636 = Union14{Union14_3{}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                switch (v636.tag) {
                    case 0: { // T_game_chance_community_card
                        Union5 v638 = v636.case0.v0; bool v639 = v636.case0.v1; static_array<Union6,2> v640 = v636.case0.v2; int v641 = v636.case0.v3; static_array<int,2> v642 = v636.case0.v4; int v643 = v636.case0.v5; Union6 v644 = v636.case0.v6;
                        int v645;
                        v645 = 2;
                        int v646; int v647;
                        Tuple7 tmp92 = Tuple7{0, 0};
                        v646 = tmp92.v0; v647 = tmp92.v1;
                        while (while_method_0(v646)){
                            bool v649;
                            v649 = 0 <= v646;
                            bool v651;
                            if (v649){
                                bool v650;
                                v650 = v646 < 2;
                                v651 = v650;
                            } else {
                                v651 = false;
                            }
                            bool v652;
                            v652 = v651 == false;
                            if (v652){
                                assert("Index must be in range." && v651);
                            } else {
                            }
                            int v654;
                            v654 = v642[v646];
                            bool v656;
                            v656 = v647 >= v654;
                            int v657;
                            if (v656){
                                v657 = v647;
                            } else {
                                v657 = v654;
                            }
                            v647 = v657;
                            v646 += 1 ;
                        }
                        static_array<int,2> v658;
                        int v660;
                        v660 = 0;
                        while (while_method_0(v660)){
                            v658[v660] = v647;
                            v660 += 1 ;
                        }
                        Union5 v662;
                        v662 = Union5{Union5_1{v644}};
                        Union4 v663;
                        v663 = Union4{Union4_2{v662, true, v640, 0, v658, v645}};
                        v796 = Union3{Union3_1{v663}};
                        break;
                    }
                    case 1: { // T_game_chance_init
                        Union6 v665 = v636.case1.v0; Union6 v666 = v636.case1.v1;
                        int v667;
                        v667 = 2;
                        static_array<int,2> v668;
                        v668[0] = 1;
                        v668[1] = 1;
                        static_array<Union6,2> v670;
                        v670[0] = v665;
                        v670[1] = v666;
                        Union5 v672;
                        v672 = Union5{Union5_0{}};
                        Union4 v673;
                        v673 = Union4{Union4_2{v672, true, v670, 0, v668, v667}};
                        v796 = Union3{Union3_1{v673}};
                        break;
                    }
                    case 2: { // T_game_round
                        Union5 v675 = v636.case2.v0; bool v676 = v636.case2.v1; static_array<Union6,2> v677 = v636.case2.v2; int v678 = v636.case2.v3; static_array<int,2> v679 = v636.case2.v4; int v680 = v636.case2.v5; Union1 v681 = v636.case2.v6;
                        Union4 v788;
                        switch (v675.tag) {
                            case 0: { // None
                                switch (v681.tag) {
                                    case 0: { // Call
                                        if (v676){
                                            int v744;
                                            v744 = v678 ^ 1;
                                            v788 = Union4{Union4_2{v675, false, v677, v744, v679, v680}};
                                        } else {
                                            v788 = Union4{Union4_0{v675, v676, v677, v678, v679, v680}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v788 = Union4{Union4_5{v675, v676, v677, v678, v679, v680}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v748;
                                        v748 = v680 > 0;
                                        if (v748){
                                            int v749;
                                            v749 = v678 ^ 1;
                                            int v750;
                                            v750 = -1 + v680;
                                            int v751; int v752;
                                            Tuple7 tmp93 = Tuple7{0, 0};
                                            v751 = tmp93.v0; v752 = tmp93.v1;
                                            while (while_method_0(v751)){
                                                bool v754;
                                                v754 = 0 <= v751;
                                                bool v756;
                                                if (v754){
                                                    bool v755;
                                                    v755 = v751 < 2;
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
                                                v759 = v679[v751];
                                                bool v761;
                                                v761 = v752 >= v759;
                                                int v762;
                                                if (v761){
                                                    v762 = v752;
                                                } else {
                                                    v762 = v759;
                                                }
                                                v752 = v762;
                                                v751 += 1 ;
                                            }
                                            static_array<int,2> v763;
                                            int v765;
                                            v765 = 0;
                                            while (while_method_0(v765)){
                                                v763[v765] = v752;
                                                v765 += 1 ;
                                            }
                                            static_array<int,2> v767;
                                            int v769;
                                            v769 = 0;
                                            while (while_method_0(v769)){
                                                bool v771;
                                                v771 = 0 <= v769;
                                                bool v773;
                                                if (v771){
                                                    bool v772;
                                                    v772 = v769 < 2;
                                                    v773 = v772;
                                                } else {
                                                    v773 = false;
                                                }
                                                bool v774;
                                                v774 = v773 == false;
                                                if (v774){
                                                    assert("Index must be in range." && v773);
                                                } else {
                                                }
                                                int v776;
                                                v776 = v763[v769];
                                                bool v778;
                                                v778 = v769 == v678;
                                                int v780;
                                                if (v778){
                                                    int v779;
                                                    v779 = v776 + 2;
                                                    v780 = v779;
                                                } else {
                                                    v780 = v776;
                                                }
                                                v767[v769] = v780;
                                                v769 += 1 ;
                                            }
                                            v788 = Union4{Union4_2{v675, false, v677, v749, v767, v750}};
                                        } else {
                                            printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                            __trap();
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                break;
                            }
                            case 1: { // Some
                                Union6 v682 = v675.case1.v0;
                                switch (v681.tag) {
                                    case 0: { // Call
                                        if (v676){
                                            int v684;
                                            v684 = v678 ^ 1;
                                            v788 = Union4{Union4_2{v675, false, v677, v684, v679, v680}};
                                        } else {
                                            int v686; int v687;
                                            Tuple7 tmp94 = Tuple7{0, 0};
                                            v686 = tmp94.v0; v687 = tmp94.v1;
                                            while (while_method_0(v686)){
                                                bool v689;
                                                v689 = 0 <= v686;
                                                bool v691;
                                                if (v689){
                                                    bool v690;
                                                    v690 = v686 < 2;
                                                    v691 = v690;
                                                } else {
                                                    v691 = false;
                                                }
                                                bool v692;
                                                v692 = v691 == false;
                                                if (v692){
                                                    assert("Index must be in range." && v691);
                                                } else {
                                                }
                                                int v694;
                                                v694 = v679[v686];
                                                bool v696;
                                                v696 = v687 >= v694;
                                                int v697;
                                                if (v696){
                                                    v697 = v687;
                                                } else {
                                                    v697 = v694;
                                                }
                                                v687 = v697;
                                                v686 += 1 ;
                                            }
                                            static_array<int,2> v698;
                                            int v700;
                                            v700 = 0;
                                            while (while_method_0(v700)){
                                                v698[v700] = v687;
                                                v700 += 1 ;
                                            }
                                            v788 = Union4{Union4_4{v675, v676, v677, v678, v698, v680}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v788 = Union4{Union4_5{v675, v676, v677, v678, v679, v680}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v704;
                                        v704 = v680 > 0;
                                        if (v704){
                                            int v705;
                                            v705 = v678 ^ 1;
                                            int v706;
                                            v706 = -1 + v680;
                                            int v707; int v708;
                                            Tuple7 tmp95 = Tuple7{0, 0};
                                            v707 = tmp95.v0; v708 = tmp95.v1;
                                            while (while_method_0(v707)){
                                                bool v710;
                                                v710 = 0 <= v707;
                                                bool v712;
                                                if (v710){
                                                    bool v711;
                                                    v711 = v707 < 2;
                                                    v712 = v711;
                                                } else {
                                                    v712 = false;
                                                }
                                                bool v713;
                                                v713 = v712 == false;
                                                if (v713){
                                                    assert("Index must be in range." && v712);
                                                } else {
                                                }
                                                int v715;
                                                v715 = v679[v707];
                                                bool v717;
                                                v717 = v708 >= v715;
                                                int v718;
                                                if (v717){
                                                    v718 = v708;
                                                } else {
                                                    v718 = v715;
                                                }
                                                v708 = v718;
                                                v707 += 1 ;
                                            }
                                            static_array<int,2> v719;
                                            int v721;
                                            v721 = 0;
                                            while (while_method_0(v721)){
                                                v719[v721] = v708;
                                                v721 += 1 ;
                                            }
                                            static_array<int,2> v723;
                                            int v725;
                                            v725 = 0;
                                            while (while_method_0(v725)){
                                                bool v727;
                                                v727 = 0 <= v725;
                                                bool v729;
                                                if (v727){
                                                    bool v728;
                                                    v728 = v725 < 2;
                                                    v729 = v728;
                                                } else {
                                                    v729 = false;
                                                }
                                                bool v730;
                                                v730 = v729 == false;
                                                if (v730){
                                                    assert("Index must be in range." && v729);
                                                } else {
                                                }
                                                int v732;
                                                v732 = v719[v725];
                                                bool v734;
                                                v734 = v725 == v678;
                                                int v736;
                                                if (v734){
                                                    int v735;
                                                    v735 = v732 + 4;
                                                    v736 = v735;
                                                } else {
                                                    v736 = v732;
                                                }
                                                v723[v725] = v736;
                                                v725 += 1 ;
                                            }
                                            v788 = Union4{Union4_2{v675, false, v677, v705, v723, v706}};
                                        } else {
                                            printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                            __trap();
                                        }
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
                        v796 = Union3{Union3_1{v788}};
                        break;
                    }
                    case 3: { // T_none
                        v796 = Union3{Union3_0{}};
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
        v16 = v796;
    }
    return ;
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1, unsigned char * v2, unsigned char * v3, unsigned char * v4) {
    Union0 v5;
    v5 = f_0(v1);
    unsigned int v6; Union3 v7; static_array_list<Union7,32> v8; static_array<Union2,2> v9; Union8 v10;
    Tuple0 tmp10 = f_6(v0);
    v6 = tmp10.v0; v7 = tmp10.v1; v8 = tmp10.v2; v9 = tmp10.v3; v10 = tmp10.v4;
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
                        case 2: { // Round
                            Union5 v37 = v36.case2.v0; bool v38 = v36.case2.v1; static_array<Union6,2> v39 = v36.case2.v2; int v40 = v36.case2.v3; static_array<int,2> v41 = v36.case2.v4; int v42 = v36.case2.v5;
                            Union4 v43;
                            v43 = Union4{Union4_3{v37, v38, v39, v40, v41, v42, v34}};
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
            v22 = Union2{Union2_0{}};
            v20[0] = v22;
            Union2 v24;
            v24 = Union2{Union2_1{}};
            v20[1] = v24;
            static_array_list<Union7,32> v26;
            v26 = static_array_list<Union7,32>{};
            Union8 v28;
            v28 = Union8{Union8_0{}};
            v19.v5 = v28;
            Union3 v29;
            v29 = Union3{Union3_0{}};
            v19.v1 = v29;
            v19.v0 = 63u;
            v19.v2 = v26;
            Union4 v30;
            v30 = Union4{Union4_1{}};
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
            static_array_list<Union7,32> & v58 = v19.v2;
            Union3 v59;
            v59 = Union3{Union3_1{v57}};
            Union3 v60;
            v60 = v59;
            while (while_method_2(v60)){
                Union3 v988;
                switch (v60.tag) {
                    case 0: { // None
                        v988 = Union3{Union3_0{}};
                        break;
                    }
                    case 1: { // Some
                        Union4 v62 = v60.case1.v0;
                        switch (v62.tag) {
                            case 0: { // ChanceCommunityCard
                                Union5 v928 = v62.case0.v0; bool v929 = v62.case0.v1; static_array<Union6,2> v930 = v62.case0.v2; int v931 = v62.case0.v3; static_array<int,2> v932 = v62.case0.v4; int v933 = v62.case0.v5;
                                curandStatePhilox4_32_10_t & v934 = v19.v4;
                                curandStatePhilox4_32_10_t & v935 = v934;
                                unsigned int & v936 = v19.v0;
                                Union6 v937; unsigned int v938;
                                Tuple6 tmp11 = draw_card_20(v935, v936);
                                v937 = tmp11.v0; v938 = tmp11.v1;
                                v19.v0 = v938;
                                Union7 v939;
                                v939 = Union7{Union7_0{v937}};
                                v58.push(v939);
                                int v940;
                                v940 = 2;
                                int v941; int v942;
                                Tuple7 tmp12 = Tuple7{0, 0};
                                v941 = tmp12.v0; v942 = tmp12.v1;
                                while (while_method_0(v941)){
                                    bool v944;
                                    v944 = 0 <= v941;
                                    bool v946;
                                    if (v944){
                                        bool v945;
                                        v945 = v941 < 2;
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
                                    v949 = v932[v941];
                                    bool v951;
                                    v951 = v942 >= v949;
                                    int v952;
                                    if (v951){
                                        v952 = v942;
                                    } else {
                                        v952 = v949;
                                    }
                                    v942 = v952;
                                    v941 += 1 ;
                                }
                                static_array<int,2> v953;
                                int v955;
                                v955 = 0;
                                while (while_method_0(v955)){
                                    v953[v955] = v942;
                                    v955 += 1 ;
                                }
                                Union5 v957;
                                v957 = Union5{Union5_1{v937}};
                                Union4 v958;
                                v958 = Union4{Union4_2{v957, true, v930, 0, v953, v940}};
                                v988 = Union3{Union3_1{v958}};
                                break;
                            }
                            case 1: { // ChanceInit
                                curandStatePhilox4_32_10_t & v960 = v19.v4;
                                curandStatePhilox4_32_10_t & v961 = v960;
                                unsigned int & v962 = v19.v0;
                                Union6 v963; unsigned int v964;
                                Tuple6 tmp13 = draw_card_20(v961, v962);
                                v963 = tmp13.v0; v964 = tmp13.v1;
                                v19.v0 = v964;
                                curandStatePhilox4_32_10_t & v965 = v19.v4;
                                curandStatePhilox4_32_10_t & v966 = v965;
                                unsigned int & v967 = v19.v0;
                                Union6 v968; unsigned int v969;
                                Tuple6 tmp14 = draw_card_20(v966, v967);
                                v968 = tmp14.v0; v969 = tmp14.v1;
                                v19.v0 = v969;
                                Union7 v970;
                                v970 = Union7{Union7_2{0, v963}};
                                v58.push(v970);
                                Union7 v971;
                                v971 = Union7{Union7_2{1, v968}};
                                v58.push(v971);
                                int v972;
                                v972 = 2;
                                static_array<int,2> v973;
                                v973[0] = 1;
                                v973[1] = 1;
                                static_array<Union6,2> v975;
                                v975[0] = v963;
                                v975[1] = v968;
                                Union5 v977;
                                v977 = Union5{Union5_0{}};
                                Union4 v978;
                                v978 = Union4{Union4_2{v977, true, v975, 0, v973, v972}};
                                v988 = Union3{Union3_1{v978}};
                                break;
                            }
                            case 2: { // Round
                                Union5 v105 = v62.case2.v0; bool v106 = v62.case2.v1; static_array<Union6,2> v107 = v62.case2.v2; int v108 = v62.case2.v3; static_array<int,2> v109 = v62.case2.v4; int v110 = v62.case2.v5;
                                static_array<Union2,2> & v111 = v19.v3;
                                bool v112;
                                v112 = 0 <= v108;
                                bool v114;
                                if (v112){
                                    bool v113;
                                    v113 = v108 < 2;
                                    v114 = v113;
                                } else {
                                    v114 = false;
                                }
                                bool v115;
                                v115 = v114 == false;
                                if (v115){
                                    assert("Index must be in range." && v114);
                                } else {
                                }
                                Union2 v117;
                                v117 = v111[v108];
                                switch (v117.tag) {
                                    case 0: { // Computer
                                        static_array_list<Union7,32> & v119 = v19.v2;
                                        curandStatePhilox4_32_10_t & v120 = v19.v4;
                                        curandStatePhilox4_32_10_t & v121 = v120;
                                        float * v122;
                                        v122 = reinterpret_cast<float *>(&v3[55050240ull]);
                                        float * v124;
                                        v124 = reinterpret_cast<float *>(&v3[0ull]);
                                        float * v126;
                                        v126 = reinterpret_cast<float *>(&v3[0ull]);
                                        int v128;
                                        v128 = blockIdx.x;
                                        assert("Tensor range check" && 0 <= v128 && v128 < 24);
                                        int v129;
                                        v129 = 32768 * v128;
                                        int v130;
                                        v130 = threadIdx.x;
                                        int v131;
                                        v131 = v130;
                                        while (while_method_3(v131)){
                                            bool v133;
                                            v133 = 0 <= v131;
                                            bool v134;
                                            v134 = v133 == false;
                                            if (v134){
                                                assert("The index needs to be zero or positive." && v133);
                                            } else {
                                            }
                                            int v136;
                                            v136 = v131 % 128;
                                            int v137;
                                            v137 = v131 / 128;
                                            bool v138;
                                            v138 = v137 < 256;
                                            bool v139;
                                            v139 = v138 == false;
                                            if (v139){
                                                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v138);
                                            } else {
                                            }
                                            assert("Tensor range check" && 0 <= v137 && v137 < 256);
                                            assert("Tensor range check" && 0 <= v136 && v136 < 128);
                                            int v141;
                                            v141 = v136 + v129;
                                            int v142;
                                            v142 = 128 * v137;
                                            int v143;
                                            v143 = v142 + v141;
                                            v126[v143] = 0.0f;
                                            v131 += 256 ;
                                        }
                                        __syncthreads();
                                        int v144;
                                        v144 = threadIdx.x;
                                        assert("Tensor range check" && 0 <= v144 && v144 < 256);
                                        int v145;
                                        v145 = 128 * v144;
                                        int v146;
                                        v146 = v145 + v129;
                                        static_array_list<Union9,10> v147;
                                        v147 = static_array_list<Union9,10>{};
                                        int v149;
                                        v149 = v119.length;
                                        int v150;
                                        v150 = 0;
                                        while (while_method_1(v149, v150)){
                                            Union7 v152;
                                            v152 = v119[v150];
                                            Union10 v171;
                                            switch (v152.tag) {
                                                case 0: { // CommunityCardIs
                                                    Union6 v161 = v152.case0.v0;
                                                    Union9 v162;
                                                    v162 = Union9{Union9_1{v161}};
                                                    v171 = Union10{Union10_1{v162}};
                                                    break;
                                                }
                                                case 1: { // PlayerAction
                                                    int v164 = v152.case1.v0; Union1 v165 = v152.case1.v1;
                                                    Union9 v166;
                                                    v166 = Union9{Union9_0{v165}};
                                                    v171 = Union10{Union10_1{v166}};
                                                    break;
                                                }
                                                case 2: { // PlayerGotCard
                                                    int v154 = v152.case2.v0; Union6 v155 = v152.case2.v1;
                                                    bool v156;
                                                    v156 = v154 == v108;
                                                    if (v156){
                                                        Union9 v157;
                                                        v157 = Union9{Union9_1{v155}};
                                                        v171 = Union10{Union10_1{v157}};
                                                    } else {
                                                        v171 = Union10{Union10_0{}};
                                                    }
                                                    break;
                                                }
                                                default: {
                                                    v171 = Union10{Union10_0{}};
                                                }
                                            }
                                            switch (v171.tag) {
                                                case 0: { // None
                                                    break;
                                                }
                                                case 1: { // Some
                                                    Union9 v172 = v171.case1.v0;
                                                    v147.push(v172);
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false); __trap();
                                                }
                                            }
                                            v150 += 1 ;
                                        }
                                        float * v173;
                                        v173 = v126+v146;
                                        int v175;
                                        v175 = v147.length;
                                        bool v176;
                                        v176 = v175 == 0;
                                        if (v176){
                                            v173[0] = 1.0f;
                                        } else {
                                        }
                                        int v177;
                                        v177 = v147.length;
                                        int v178;
                                        v178 = 0;
                                        while (while_method_1(v177, v178)){
                                            Union9 v180;
                                            v180 = v147[v178];
                                            int v182;
                                            v182 = v178 * 6;
                                            int v183;
                                            v183 = 1 + v182;
                                            switch (v180.tag) {
                                                case 0: { // C1of2
                                                    Union1 v184 = v180.case0.v0;
                                                    switch (v184.tag) {
                                                        case 0: { // Call
                                                            v173[v183] = 1.0f;
                                                            break;
                                                        }
                                                        case 1: { // Fold
                                                            int v185;
                                                            v185 = v183 + 1;
                                                            v173[v185] = 1.0f;
                                                            break;
                                                        }
                                                        case 2: { // Raise
                                                            int v186;
                                                            v186 = v183 + 2;
                                                            v173[v186] = 1.0f;
                                                            break;
                                                        }
                                                        default: {
                                                            assert("Invalid tag." && false); __trap();
                                                        }
                                                    }
                                                    break;
                                                }
                                                case 1: { // C2of2
                                                    Union6 v187 = v180.case1.v0;
                                                    int v188;
                                                    v188 = v183 + 3;
                                                    switch (v187.tag) {
                                                        case 0: { // Jack
                                                            v173[v188] = 1.0f;
                                                            break;
                                                        }
                                                        case 1: { // King
                                                            int v189;
                                                            v189 = v188 + 1;
                                                            v173[v189] = 1.0f;
                                                            break;
                                                        }
                                                        case 2: { // Queen
                                                            int v190;
                                                            v190 = v188 + 2;
                                                            v173[v190] = 1.0f;
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
                                            v178 += 1 ;
                                        }
                                        __syncthreads();
                                        int v191;
                                        v191 = 0;
                                        int v192;
                                        v192 = 32;
                                        int v193;
                                        v193 = int_range_22(v192, v191, v121);
                                        extern __shared__ unsigned char v194[];
                                        int * v195;
                                        v195 = reinterpret_cast<int *>(&v194[0ull]);
                                        int v197;
                                        v197 = threadIdx.x;
                                        bool v198;
                                        v198 = v197 == 0;
                                        if (v198){
                                            v195[0] = v193;
                                        } else {
                                        }
                                        __syncthreads();
                                        int v199;
                                        v199 = v195[0];
                                        __syncthreads();
                                        float * v200;
                                        v200 = reinterpret_cast<float *>(&v3[55050240ull]);
                                        assert("Tensor range check" && 0 <= v199 && v199 < 32);
                                        int v202;
                                        v202 = 393216 * v199;
                                        float * v203;
                                        v203 = reinterpret_cast<float *>(&v3[4718592ull]);
                                        assert("Tensor range check" && 0 <= v199 && v199 < 32);
                                        float * v205;
                                        v205 = reinterpret_cast<float *>(&v3[0ull]);
                                        float * v207;
                                        v207 = reinterpret_cast<float *>(&v2[0ull]);
                                        float * v209;
                                        v209 = reinterpret_cast<float *>(&v4[0ull]);
                                        assert("Tensor range check" && 0 <= v199 && v199 < 32);
                                        int v211;
                                        v211 = 8192 * v199;
                                        float * v212;
                                        v212 = reinterpret_cast<float *>(&v3[3145728ull]);
                                        block_matmul_23(v212, v207, v211, v205);
                                        block_map_24(v203, v202, v212);
                                        block_row_map_25(v200, v202, v203);
                                        int * v214;
                                        v214 = reinterpret_cast<int *>(&v2[1048576ull]);
                                        float * v216;
                                        v216 = reinterpret_cast<float *>(&v2[1048592ull]);
                                        float * v218;
                                        v218 = reinterpret_cast<float *>(&v2[1048720ull]);
                                        double * v220;
                                        v220 = reinterpret_cast<double *>(&v3[105381888ull]);
                                        double * v222;
                                        v222 = reinterpret_cast<double *>(&v3[108527616ull]);
                                        __syncthreads();
                                        float * v224;
                                        v224 = reinterpret_cast<float *>(&v3[55050240ull]);
                                        assert("Tensor range check" && 0 <= v199 && v199 < 32);
                                        int v226;
                                        v226 = blockIdx.x;
                                        assert("Tensor range check" && 0 <= v226 && v226 < 24);
                                        int v227;
                                        v227 = 16384 * v226;
                                        int v228;
                                        v228 = v227 + v202;
                                        int v229;
                                        v229 = threadIdx.x;
                                        assert("Tensor range check" && 0 <= v229 && v229 < 256);
                                        int v230;
                                        v230 = 64 * v229;
                                        int v231;
                                        v231 = v230 + v228;
                                        float * v232;
                                        v232 = v224+v231;
                                        int v234;
                                        v234 = sizeof(float *);
                                        unsigned long long v235;
                                        v235 = (unsigned long long)v234;
                                        unsigned long long v236;
                                        v236 = 256ull * v235;
                                        unsigned long long v237;
                                        v237 = v236 + 16ull;
                                        unsigned long long v238;
                                        v238 = v237 - 1ull;
                                        unsigned long long v239;
                                        v239 = v238 % 16ull;
                                        unsigned long long v240;
                                        v240 = v238 - v239;
                                        unsigned long long v241;
                                        v241 = v240 + 1024ull;
                                        unsigned long long v242;
                                        v242 = v241 + 16ull;
                                        unsigned long long v243;
                                        v243 = v242 - 1ull;
                                        unsigned long long v244;
                                        v244 = v243 % 16ull;
                                        unsigned long long v245;
                                        v245 = v243 - v244;
                                        unsigned long long v246;
                                        v246 = v245 + 1024ull;
                                        bool v247;
                                        v247 = v246 <= 98304ull;
                                        bool v248;
                                        v248 = v247 == false;
                                        if (v248){
                                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v247);
                                        } else {
                                        }
                                        extern __shared__ unsigned char v250[];
                                        bool v251;
                                        v251 = v246 <= v246;
                                        bool v252;
                                        v252 = v251 == false;
                                        if (v252){
                                            assert("The length of the partition has to be less than or equal to the length of the base array." && v251);
                                        } else {
                                        }
                                        float * * v254;
                                        v254 = reinterpret_cast<float * *>(&v250[0ull]);
                                        float * v256;
                                        v256 = reinterpret_cast<float *>(&v250[v240]);
                                        int * v258;
                                        v258 = reinterpret_cast<int *>(&v250[v245]);
                                        int v260;
                                        v260 = threadIdx.x;
                                        assert("Tensor range check" && 0 <= v260 && v260 < 256);
                                        v254[v260] = v232;
                                        __syncthreads();
                                        bool v261;
                                        v261 = 0 <= v260;
                                        bool v262;
                                        v262 = v261 == false;
                                        if (v262){
                                            assert("The index needs to be zero or positive." && v261);
                                        } else {
                                        }
                                        int v264;
                                        v264 = v260 % 16;
                                        int v265;
                                        v265 = v260 / 16;
                                        bool v266;
                                        v266 = v265 < 16;
                                        bool v267;
                                        v267 = v266 == false;
                                        if (v267){
                                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v266);
                                        } else {
                                        }
                                        assert("Tensor range check" && 0 <= v265 && v265 < 16);
                                        int v269;
                                        v269 = 0;
                                        while (while_method_8(v269)){
                                            bool v271;
                                            v271 = 0 <= v265;
                                            bool v272;
                                            v272 = v271 && v266;
                                            bool v273;
                                            v273 = v272 == false;
                                            if (v273){
                                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v272);
                                            } else {
                                            }
                                            bool v275;
                                            v275 = 0 <= v269;
                                            bool v277;
                                            if (v275){
                                                bool v276;
                                                v276 = v269 < 16;
                                                v277 = v276;
                                            } else {
                                                v277 = false;
                                            }
                                            bool v278;
                                            v278 = v277 == false;
                                            if (v278){
                                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v277);
                                            } else {
                                            }
                                            int v280;
                                            v280 = v269 * 16;
                                            int v281;
                                            v281 = v280 + v265;
                                            assert("Tensor range check" && 0 <= v269 && v269 < 16);
                                            int v282;
                                            v282 = 16 * v269;
                                            int v283;
                                            v283 = v282 + v265;
                                            float * v284;
                                            v284 = v254[v283];
                                            int v285;
                                            v285 = blockIdx.x;
                                            int v286;
                                            v286 = v285 * 256;
                                            int v287;
                                            v287 = v286 + v281;
                                            assert("Tensor range check" && 0 <= v264 && v264 < 16);
                                            int v288;
                                            v288 = 4 * v264;
                                            float v289[4];
                                            int v290[4];
                                            int v291;
                                            v291 = 0;
                                            while (while_method_4(v291)){
                                                assert("Tensor range check" && 0 <= v291 && v291 < 1);
                                                int v293;
                                                v293 = 4 * v291;
                                                assert("Tensor range check" && 0 <= v291 && v291 < 1);
                                                int v294;
                                                v294 = 64 * v291;
                                                int v295;
                                                v295 = v294 + v288;
                                                int4* v296;
                                                v296 = reinterpret_cast<int4*>(v284 + v295);
                                                int4* v297;
                                                v297 = reinterpret_cast<int4*>(v289 + v293);
                                                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v296) % 16 == 0 && reinterpret_cast<unsigned long long>(v297) % 16 == 0);
                                                *v297 = *v296;
                                                v291 += 1 ;
                                            }
                                            int v298;
                                            v298 = 0;
                                            while (while_method_4(v298)){
                                                int v300;
                                                v300 = 0;
                                                while (while_method_7(v300)){
                                                    bool v302;
                                                    v302 = 0 <= v300;
                                                    bool v304;
                                                    if (v302){
                                                        bool v303;
                                                        v303 = v300 < 4;
                                                        v304 = v303;
                                                    } else {
                                                        v304 = false;
                                                    }
                                                    bool v305;
                                                    v305 = v304 == false;
                                                    if (v305){
                                                        assert("The indices should be inside the range of the dimension." && v304);
                                                    } else {
                                                    }
                                                    bool v307;
                                                    v307 = 0 <= v264;
                                                    bool v309;
                                                    if (v307){
                                                        bool v308;
                                                        v308 = v264 < 16;
                                                        v309 = v308;
                                                    } else {
                                                        v309 = false;
                                                    }
                                                    bool v310;
                                                    v310 = v309 == false;
                                                    if (v310){
                                                        assert("The indices should be inside the range of the dimension." && v309);
                                                    } else {
                                                    }
                                                    int v312;
                                                    v312 = v264 * 4;
                                                    int v313;
                                                    v313 = v300 + v312;
                                                    bool v314;
                                                    v314 = 0 <= v298;
                                                    bool v316;
                                                    if (v314){
                                                        bool v315;
                                                        v315 = v298 < 1;
                                                        v316 = v315;
                                                    } else {
                                                        v316 = false;
                                                    }
                                                    bool v317;
                                                    v317 = v316 == false;
                                                    if (v317){
                                                        assert("The indices should be inside the range of the dimension." && v316);
                                                    } else {
                                                    }
                                                    int v319;
                                                    v319 = v298 * 64;
                                                    int v320;
                                                    v320 = v313 + v319;
                                                    assert("Tensor range check" && 0 <= v298 && v298 < 1);
                                                    assert("Tensor range check" && 0 <= v300 && v300 < 4);
                                                    int v321;
                                                    v321 = 4 * v298;
                                                    int v322;
                                                    v322 = v321 + v300;
                                                    v290[v322] = v320;
                                                    v300 += 1 ;
                                                }
                                                v298 += 1 ;
                                            }
                                            float v323[4];
                                            float v324;
                                            v324 = 0.0f;
                                            int v325;
                                            v325 = 0;
                                            while (while_method_4(v325)){
                                                assert("Tensor range check" && 0 <= v325 && v325 < 1);
                                                int v327;
                                                v327 = 4 * v325;
                                                assert("Tensor range check" && 0 <= v325 && v325 < 1);
                                                int v328; float v329;
                                                Tuple8 tmp15 = Tuple8{0, 0.0f};
                                                v328 = tmp15.v0; v329 = tmp15.v1;
                                                while (while_method_7(v328)){
                                                    assert("Tensor range check" && 0 <= v328 && v328 < 4);
                                                    int v331;
                                                    v331 = v328 + v327;
                                                    float v332;
                                                    v332 = v289[v331];
                                                    float v333;
                                                    v333 = v329 + v332;
                                                    v329 = v333;
                                                    v328 += 1 ;
                                                }
                                                auto v334 = cooperative_groups::coalesced_threads();
                                                int v335;
                                                v335 = threadIdx.x;
                                                int v336;
                                                v336 = v335 / 16;
                                                auto v337 = cooperative_groups::labeled_partition(v334,v336);
                                                Closure2 v338{};
                                                float v339;
                                                v339 = cooperative_groups::inclusive_scan(v337, v329, v338);
                                                float v340;
                                                v340 = v337.shfl_up(v339,1);
                                                bool v341;
                                                v341 = v337.thread_rank() == 0;
                                                float v342;
                                                if (v341){
                                                    v342 = 0.0f;
                                                } else {
                                                    v342 = v340;
                                                }
                                                float v343;
                                                v343 = v337.shfl(v339,v337.num_threads()-1);
                                                float v344;
                                                v344 = v324 + v342;
                                                int v345; float v346;
                                                Tuple8 tmp16 = Tuple8{0, v344};
                                                v345 = tmp16.v0; v346 = tmp16.v1;
                                                while (while_method_7(v345)){
                                                    assert("Tensor range check" && 0 <= v345 && v345 < 4);
                                                    int v348;
                                                    v348 = v345 + v327;
                                                    float v349;
                                                    v349 = v289[v348];
                                                    float v350;
                                                    v350 = v346 + v349;
                                                    assert("Tensor range check" && 0 <= v345 && v345 < 4);
                                                    v323[v348] = v350;
                                                    v346 = v350;
                                                    v345 += 1 ;
                                                }
                                                float v351;
                                                v351 = v324 + v343;
                                                v324 = v351;
                                                v325 += 1 ;
                                            }
                                            float v352[4];
                                            bool v353[4];
                                            int v354;
                                            v354 = 0;
                                            while (while_method_4(v354)){
                                                int v356;
                                                v356 = 0;
                                                while (while_method_7(v356)){
                                                    assert("Tensor range check" && 0 <= v354 && v354 < 1);
                                                    assert("Tensor range check" && 0 <= v356 && v356 < 4);
                                                    int v358;
                                                    v358 = 4 * v354;
                                                    int v359;
                                                    v359 = v358 + v356;
                                                    float v360;
                                                    v360 = v323[v359];
                                                    float v361;
                                                    v361 = v289[v359];
                                                    bool v362;
                                                    v362 = v361 > 0.0f;
                                                    assert("Tensor range check" && 0 <= v354 && v354 < 1);
                                                    assert("Tensor range check" && 0 <= v356 && v356 < 4);
                                                    v352[v359] = v360;
                                                    v353[v359] = v362;
                                                    v356 += 1 ;
                                                }
                                                v354 += 1 ;
                                            }
                                            float v363; bool v364;
                                            Tuple9 tmp17 = Tuple9{-1.0f / 0.0f, false};
                                            v363 = tmp17.v0; v364 = tmp17.v1;
                                            int v365;
                                            v365 = 0;
                                            while (while_method_4(v365)){
                                                int v367;
                                                v367 = 0;
                                                while (while_method_7(v367)){
                                                    assert("Tensor range check" && 0 <= v365 && v365 < 1);
                                                    assert("Tensor range check" && 0 <= v367 && v367 < 4);
                                                    int v369;
                                                    v369 = 4 * v365;
                                                    int v370;
                                                    v370 = v369 + v367;
                                                    float v371;
                                                    v371 = v352[v370];
                                                    bool v372;
                                                    v372 = v353[v370];
                                                    float v379; bool v380;
                                                    if (v364){
                                                        if (v372){
                                                            bool v373;
                                                            v373 = v363 >= v371;
                                                            float v374;
                                                            if (v373){
                                                                v374 = v363;
                                                            } else {
                                                                v374 = v371;
                                                            }
                                                            v379 = v374; v380 = true;
                                                        } else {
                                                            v379 = v363; v380 = v364;
                                                        }
                                                    } else {
                                                        if (v372){
                                                            v379 = v371; v380 = v372;
                                                        } else {
                                                            v379 = v363; v380 = v364;
                                                        }
                                                    }
                                                    v363 = v379;
                                                    v364 = v380;
                                                    v367 += 1 ;
                                                }
                                                v365 += 1 ;
                                            }
                                            auto v381 = cooperative_groups::coalesced_threads();
                                            int v382;
                                            v382 = threadIdx.x;
                                            int v383;
                                            v383 = v382 / 16;
                                            auto v384 = cooperative_groups::labeled_partition(v381,v383);
                                            Closure3 v385{};
                                            float v386; bool v387;
                                            Tuple9 tmp18 = cooperative_groups::reduce(v384, Tuple9{v363, v364}, v385);
                                            v386 = tmp18.v0; v387 = tmp18.v1;
                                            bool v388;
                                            v388 = v387 == false;
                                            if (v388){
                                                assert("The local reduce must be true." && v387);
                                            } else {
                                            }
                                            float v390[4];
                                            int v391[4];
                                            int v392;
                                            v392 = 0;
                                            while (while_method_4(v392)){
                                                int v394;
                                                v394 = 0;
                                                while (while_method_7(v394)){
                                                    assert("Tensor range check" && 0 <= v392 && v392 < 1);
                                                    assert("Tensor range check" && 0 <= v394 && v394 < 4);
                                                    int v396;
                                                    v396 = 4 * v392;
                                                    int v397;
                                                    v397 = v396 + v394;
                                                    int v398;
                                                    v398 = v290[v397];
                                                    float v399;
                                                    v399 = curand_uniform(&v121);
                                                    assert("Tensor range check" && 0 <= v392 && v392 < 1);
                                                    assert("Tensor range check" && 0 <= v394 && v394 < 4);
                                                    v390[v397] = v399;
                                                    v391[v397] = v398;
                                                    v394 += 1 ;
                                                }
                                                v392 += 1 ;
                                            }
                                            float v400; int v401;
                                            Tuple10 tmp19 = Tuple10{0.0f, 2147483647};
                                            v400 = tmp19.v0; v401 = tmp19.v1;
                                            int v402;
                                            v402 = 0;
                                            while (while_method_4(v402)){
                                                int v404;
                                                v404 = 0;
                                                while (while_method_7(v404)){
                                                    assert("Tensor range check" && 0 <= v402 && v402 < 1);
                                                    assert("Tensor range check" && 0 <= v404 && v404 < 4);
                                                    int v406;
                                                    v406 = 4 * v402;
                                                    int v407;
                                                    v407 = v406 + v404;
                                                    float v408;
                                                    v408 = v390[v407];
                                                    int v409;
                                                    v409 = v391[v407];
                                                    bool v410;
                                                    v410 = v401 < v409;
                                                    float v411; int v412;
                                                    if (v410){
                                                        v411 = v400; v412 = v401;
                                                    } else {
                                                        v411 = v408; v412 = v409;
                                                    }
                                                    v400 = v411;
                                                    v401 = v412;
                                                    v404 += 1 ;
                                                }
                                                v402 += 1 ;
                                            }
                                            auto v413 = cooperative_groups::coalesced_threads();
                                            int v414;
                                            v414 = threadIdx.x;
                                            int v415;
                                            v415 = v414 / 16;
                                            auto v416 = cooperative_groups::labeled_partition(v413,v415);
                                            Closure4 v417{};
                                            float v418; int v419;
                                            Tuple10 tmp20 = cooperative_groups::reduce(v416, Tuple10{v400, v401}, v417);
                                            v418 = tmp20.v0; v419 = tmp20.v1;
                                            float v420;
                                            v420 = v386 * v418;
                                            int v421[4];
                                            bool v422[4];
                                            int v423;
                                            v423 = 0;
                                            while (while_method_4(v423)){
                                                int v425;
                                                v425 = 0;
                                                while (while_method_7(v425)){
                                                    assert("Tensor range check" && 0 <= v423 && v423 < 1);
                                                    assert("Tensor range check" && 0 <= v425 && v425 < 4);
                                                    int v427;
                                                    v427 = 4 * v423;
                                                    int v428;
                                                    v428 = v427 + v425;
                                                    float v429;
                                                    v429 = v352[v428];
                                                    bool v430;
                                                    v430 = v353[v428];
                                                    int v431;
                                                    v431 = v290[v428];
                                                    int v434; bool v435;
                                                    if (v430){
                                                        float v432;
                                                        v432 = v429 - v420;
                                                        bool v433;
                                                        v433 = v432 >= 0.0f;
                                                        v434 = v431; v435 = v433;
                                                    } else {
                                                        v434 = 2147483647; v435 = false;
                                                    }
                                                    assert("Tensor range check" && 0 <= v423 && v423 < 1);
                                                    assert("Tensor range check" && 0 <= v425 && v425 < 4);
                                                    v421[v428] = v434;
                                                    v422[v428] = v435;
                                                    v425 += 1 ;
                                                }
                                                v423 += 1 ;
                                            }
                                            int v436; bool v437;
                                            Tuple11 tmp21 = Tuple11{2147483647, false};
                                            v436 = tmp21.v0; v437 = tmp21.v1;
                                            int v438;
                                            v438 = 0;
                                            while (while_method_4(v438)){
                                                int v440;
                                                v440 = 0;
                                                while (while_method_7(v440)){
                                                    assert("Tensor range check" && 0 <= v438 && v438 < 1);
                                                    assert("Tensor range check" && 0 <= v440 && v440 < 4);
                                                    int v442;
                                                    v442 = 4 * v438;
                                                    int v443;
                                                    v443 = v442 + v440;
                                                    int v444;
                                                    v444 = v421[v443];
                                                    bool v445;
                                                    v445 = v422[v443];
                                                    int v452; bool v453;
                                                    if (v437){
                                                        if (v445){
                                                            bool v446;
                                                            v446 = v436 < v444;
                                                            int v447;
                                                            if (v446){
                                                                v447 = v436;
                                                            } else {
                                                                v447 = v444;
                                                            }
                                                            v452 = v447; v453 = true;
                                                        } else {
                                                            v452 = v436; v453 = v437;
                                                        }
                                                    } else {
                                                        if (v445){
                                                            v452 = v444; v453 = v445;
                                                        } else {
                                                            v452 = v436; v453 = v437;
                                                        }
                                                    }
                                                    v436 = v452;
                                                    v437 = v453;
                                                    v440 += 1 ;
                                                }
                                                v438 += 1 ;
                                            }
                                            auto v454 = cooperative_groups::coalesced_threads();
                                            int v455;
                                            v455 = threadIdx.x;
                                            int v456;
                                            v456 = v455 / 16;
                                            auto v457 = cooperative_groups::labeled_partition(v454,v456);
                                            Closure5 v458{};
                                            int v459; bool v460;
                                            Tuple11 tmp22 = cooperative_groups::reduce(v457, Tuple11{v436, v437}, v458);
                                            v459 = tmp22.v0; v460 = tmp22.v1;
                                            bool v461;
                                            v461 = v460 == false;
                                            if (v461){
                                                assert("The local reduce must be true." && v460);
                                            } else {
                                            }
                                            float v463; int v464;
                                            Tuple10 tmp23 = Tuple10{0.0f, 2147483647};
                                            v463 = tmp23.v0; v464 = tmp23.v1;
                                            int v465;
                                            v465 = 0;
                                            while (while_method_4(v465)){
                                                int v467;
                                                v467 = 0;
                                                while (while_method_7(v467)){
                                                    assert("Tensor range check" && 0 <= v465 && v465 < 1);
                                                    assert("Tensor range check" && 0 <= v467 && v467 < 4);
                                                    int v469;
                                                    v469 = 4 * v465;
                                                    int v470;
                                                    v470 = v469 + v467;
                                                    float v471;
                                                    v471 = v289[v470];
                                                    int v472;
                                                    v472 = v290[v470];
                                                    bool v473;
                                                    v473 = v464 == v459;
                                                    float v477; int v478;
                                                    if (v473){
                                                        v477 = v463; v478 = v464;
                                                    } else {
                                                        bool v474;
                                                        v474 = v472 == v459;
                                                        if (v474){
                                                            v477 = v471; v478 = v472;
                                                        } else {
                                                            v477 = v463; v478 = v464;
                                                        }
                                                    }
                                                    v463 = v477;
                                                    v464 = v478;
                                                    v467 += 1 ;
                                                }
                                                v465 += 1 ;
                                            }
                                            auto v479 = cooperative_groups::coalesced_threads();
                                            int v480;
                                            v480 = threadIdx.x;
                                            int v481;
                                            v481 = v480 / 16;
                                            auto v482 = cooperative_groups::labeled_partition(v479,v481);
                                            Closure6 v483{v459};
                                            float v484; int v485;
                                            Tuple10 tmp24 = cooperative_groups::reduce(v482, Tuple10{v463, v464}, v483);
                                            v484 = tmp24.v0; v485 = tmp24.v1;
                                            bool v486;
                                            v486 = v485 == 2147483647;
                                            bool v487;
                                            v487 = v486 != true;
                                            bool v488;
                                            v488 = v487 == false;
                                            if (v488){
                                                assert("Expected a valid action id in get_prob." && v487);
                                            } else {
                                            }
                                            int v490;
                                            v490 = 0;
                                            while (while_method_4(v490)){
                                                assert("Tensor range check" && 0 <= v490 && v490 < 1);
                                                assert("Tensor range check" && 0 <= v490 && v490 < 1);
                                                v490 += 1 ;
                                            }
                                            assert("Tensor range check" && 0 <= v281 && v281 < 256);
                                            v256[v281] = v484;
                                            v258[v281] = v459;
                                            v269 += 1 ;
                                        }
                                        __syncthreads();
                                        assert("Tensor range check" && 0 <= v260 && v260 < 256);
                                        float v492;
                                        v492 = v256[v260];
                                        int v493;
                                        v493 = v258[v260];
                                        __syncthreads();
                                        bool v494;
                                        v494 = 0 == v493;
                                        Union12 v503;
                                        if (v494){
                                            v503 = Union12{Union12_1{}};
                                        } else {
                                            bool v496;
                                            v496 = 1 == v493;
                                            if (v496){
                                                v503 = Union12{Union12_0{}};
                                            } else {
                                                bool v498;
                                                v498 = 2 == v493;
                                                if (v498){
                                                    v503 = Union12{Union12_2{}};
                                                } else {
                                                    printf("%s\n", "Invalid output id in the Leduc model.");
                                                    __trap();
                                                }
                                            }
                                        }
                                        Union1 v535;
                                        switch (v503.tag) {
                                            case 0: { // AA_Call
                                                v535 = Union1{Union1_0{}};
                                                break;
                                            }
                                            case 1: { // AA_Fold
                                                int v504;
                                                v504 = v109[0];
                                                int v506; int v507;
                                                Tuple7 tmp25 = Tuple7{1, v504};
                                                v506 = tmp25.v0; v507 = tmp25.v1;
                                                while (while_method_0(v506)){
                                                    bool v509;
                                                    v509 = 0 <= v506;
                                                    bool v511;
                                                    if (v509){
                                                        bool v510;
                                                        v510 = v506 < 2;
                                                        v511 = v510;
                                                    } else {
                                                        v511 = false;
                                                    }
                                                    bool v512;
                                                    v512 = v511 == false;
                                                    if (v512){
                                                        assert("Index must be in range." && v511);
                                                    } else {
                                                    }
                                                    int v514;
                                                    v514 = v109[v506];
                                                    bool v516;
                                                    v516 = v507 >= v514;
                                                    int v517;
                                                    if (v516){
                                                        v517 = v507;
                                                    } else {
                                                        v517 = v514;
                                                    }
                                                    v507 = v517;
                                                    v506 += 1 ;
                                                }
                                                bool v519;
                                                if (v112){
                                                    bool v518;
                                                    v518 = v108 < 2;
                                                    v519 = v518;
                                                } else {
                                                    v519 = false;
                                                }
                                                bool v520;
                                                v520 = v519 == false;
                                                if (v520){
                                                    assert("Index must be in range." && v519);
                                                } else {
                                                }
                                                int v522;
                                                v522 = v109[v108];
                                                bool v524;
                                                v524 = v522 == v507;
                                                if (v524){
                                                    v535 = Union1{Union1_0{}};
                                                } else {
                                                    v535 = Union1{Union1_1{}};
                                                }
                                                break;
                                            }
                                            case 2: { // AA_Raise
                                                bool v529;
                                                v529 = v110 > 0;
                                                if (v529){
                                                    v535 = Union1{Union1_2{}};
                                                } else {
                                                    v535 = Union1{Union1_0{}};
                                                }
                                                break;
                                            }
                                            default: {
                                                assert("Invalid tag." && false); __trap();
                                            }
                                        }
                                        int v536;
                                        v536 = sizeof(Union1);
                                        unsigned long long v537;
                                        v537 = (unsigned long long)v536;
                                        bool v538;
                                        v538 = v537 <= 98304ull;
                                        bool v539;
                                        v539 = v538 == false;
                                        if (v539){
                                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v538);
                                        } else {
                                        }
                                        extern __shared__ unsigned char v541[];
                                        bool v542;
                                        v542 = v537 <= v537;
                                        bool v543;
                                        v543 = v542 == false;
                                        if (v543){
                                            assert("The length of the partition has to be less than or equal to the length of the base array." && v542);
                                        } else {
                                        }
                                        Union1 * v545;
                                        v545 = reinterpret_cast<Union1 *>(&v541[0ull]);
                                        int v547;
                                        v547 = threadIdx.x;
                                        bool v548;
                                        v548 = v547 == 0;
                                        if (v548){
                                            v545[0] = v535;
                                        } else {
                                        }
                                        __syncthreads();
                                        Union1 v549;
                                        v549 = v545[0];
                                        __syncthreads();
                                        Union7 v550;
                                        v550 = Union7{Union7_1{v108, v549}};
                                        v58.push(v550);
                                        Union4 v657;
                                        switch (v105.tag) {
                                            case 0: { // None
                                                switch (v549.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v613;
                                                            v613 = v108 ^ 1;
                                                            v657 = Union4{Union4_2{v105, false, v107, v613, v109, v110}};
                                                        } else {
                                                            v657 = Union4{Union4_0{v105, v106, v107, v108, v109, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v657 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        bool v617;
                                                        v617 = v110 > 0;
                                                        if (v617){
                                                            int v618;
                                                            v618 = v108 ^ 1;
                                                            int v619;
                                                            v619 = -1 + v110;
                                                            int v620; int v621;
                                                            Tuple7 tmp26 = Tuple7{0, 0};
                                                            v620 = tmp26.v0; v621 = tmp26.v1;
                                                            while (while_method_0(v620)){
                                                                bool v623;
                                                                v623 = 0 <= v620;
                                                                bool v625;
                                                                if (v623){
                                                                    bool v624;
                                                                    v624 = v620 < 2;
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
                                                                v628 = v109[v620];
                                                                bool v630;
                                                                v630 = v621 >= v628;
                                                                int v631;
                                                                if (v630){
                                                                    v631 = v621;
                                                                } else {
                                                                    v631 = v628;
                                                                }
                                                                v621 = v631;
                                                                v620 += 1 ;
                                                            }
                                                            static_array<int,2> v632;
                                                            int v634;
                                                            v634 = 0;
                                                            while (while_method_0(v634)){
                                                                v632[v634] = v621;
                                                                v634 += 1 ;
                                                            }
                                                            static_array<int,2> v636;
                                                            int v638;
                                                            v638 = 0;
                                                            while (while_method_0(v638)){
                                                                bool v640;
                                                                v640 = 0 <= v638;
                                                                bool v642;
                                                                if (v640){
                                                                    bool v641;
                                                                    v641 = v638 < 2;
                                                                    v642 = v641;
                                                                } else {
                                                                    v642 = false;
                                                                }
                                                                bool v643;
                                                                v643 = v642 == false;
                                                                if (v643){
                                                                    assert("Index must be in range." && v642);
                                                                } else {
                                                                }
                                                                int v645;
                                                                v645 = v632[v638];
                                                                bool v647;
                                                                v647 = v638 == v108;
                                                                int v649;
                                                                if (v647){
                                                                    int v648;
                                                                    v648 = v645 + 2;
                                                                    v649 = v648;
                                                                } else {
                                                                    v649 = v645;
                                                                }
                                                                v636[v638] = v649;
                                                                v638 += 1 ;
                                                            }
                                                            v657 = Union4{Union4_2{v105, false, v107, v618, v636, v619}};
                                                        } else {
                                                            printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                                            __trap();
                                                        }
                                                        break;
                                                    }
                                                    default: {
                                                        assert("Invalid tag." && false); __trap();
                                                    }
                                                }
                                                break;
                                            }
                                            case 1: { // Some
                                                Union6 v551 = v105.case1.v0;
                                                switch (v549.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v553;
                                                            v553 = v108 ^ 1;
                                                            v657 = Union4{Union4_2{v105, false, v107, v553, v109, v110}};
                                                        } else {
                                                            int v555; int v556;
                                                            Tuple7 tmp27 = Tuple7{0, 0};
                                                            v555 = tmp27.v0; v556 = tmp27.v1;
                                                            while (while_method_0(v555)){
                                                                bool v558;
                                                                v558 = 0 <= v555;
                                                                bool v560;
                                                                if (v558){
                                                                    bool v559;
                                                                    v559 = v555 < 2;
                                                                    v560 = v559;
                                                                } else {
                                                                    v560 = false;
                                                                }
                                                                bool v561;
                                                                v561 = v560 == false;
                                                                if (v561){
                                                                    assert("Index must be in range." && v560);
                                                                } else {
                                                                }
                                                                int v563;
                                                                v563 = v109[v555];
                                                                bool v565;
                                                                v565 = v556 >= v563;
                                                                int v566;
                                                                if (v565){
                                                                    v566 = v556;
                                                                } else {
                                                                    v566 = v563;
                                                                }
                                                                v556 = v566;
                                                                v555 += 1 ;
                                                            }
                                                            static_array<int,2> v567;
                                                            int v569;
                                                            v569 = 0;
                                                            while (while_method_0(v569)){
                                                                v567[v569] = v556;
                                                                v569 += 1 ;
                                                            }
                                                            v657 = Union4{Union4_4{v105, v106, v107, v108, v567, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v657 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        bool v573;
                                                        v573 = v110 > 0;
                                                        if (v573){
                                                            int v574;
                                                            v574 = v108 ^ 1;
                                                            int v575;
                                                            v575 = -1 + v110;
                                                            int v576; int v577;
                                                            Tuple7 tmp28 = Tuple7{0, 0};
                                                            v576 = tmp28.v0; v577 = tmp28.v1;
                                                            while (while_method_0(v576)){
                                                                bool v579;
                                                                v579 = 0 <= v576;
                                                                bool v581;
                                                                if (v579){
                                                                    bool v580;
                                                                    v580 = v576 < 2;
                                                                    v581 = v580;
                                                                } else {
                                                                    v581 = false;
                                                                }
                                                                bool v582;
                                                                v582 = v581 == false;
                                                                if (v582){
                                                                    assert("Index must be in range." && v581);
                                                                } else {
                                                                }
                                                                int v584;
                                                                v584 = v109[v576];
                                                                bool v586;
                                                                v586 = v577 >= v584;
                                                                int v587;
                                                                if (v586){
                                                                    v587 = v577;
                                                                } else {
                                                                    v587 = v584;
                                                                }
                                                                v577 = v587;
                                                                v576 += 1 ;
                                                            }
                                                            static_array<int,2> v588;
                                                            int v590;
                                                            v590 = 0;
                                                            while (while_method_0(v590)){
                                                                v588[v590] = v577;
                                                                v590 += 1 ;
                                                            }
                                                            static_array<int,2> v592;
                                                            int v594;
                                                            v594 = 0;
                                                            while (while_method_0(v594)){
                                                                bool v596;
                                                                v596 = 0 <= v594;
                                                                bool v598;
                                                                if (v596){
                                                                    bool v597;
                                                                    v597 = v594 < 2;
                                                                    v598 = v597;
                                                                } else {
                                                                    v598 = false;
                                                                }
                                                                bool v599;
                                                                v599 = v598 == false;
                                                                if (v599){
                                                                    assert("Index must be in range." && v598);
                                                                } else {
                                                                }
                                                                int v601;
                                                                v601 = v588[v594];
                                                                bool v603;
                                                                v603 = v594 == v108;
                                                                int v605;
                                                                if (v603){
                                                                    int v604;
                                                                    v604 = v601 + 4;
                                                                    v605 = v604;
                                                                } else {
                                                                    v605 = v601;
                                                                }
                                                                v592[v594] = v605;
                                                                v594 += 1 ;
                                                            }
                                                            v657 = Union4{Union4_2{v105, false, v107, v574, v592, v575}};
                                                        } else {
                                                            printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                                            __trap();
                                                        }
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
                                        v988 = Union3{Union3_1{v657}};
                                        break;
                                    }
                                    case 1: { // Human
                                        Union8 v659;
                                        v659 = Union8{Union8_2{v105, v106, v107, v108, v109, v110}};
                                        v19.v5 = v659;
                                        Union3 v660;
                                        v660 = Union3{Union3_1{v62}};
                                        v19.v1 = v660;
                                        v988 = Union3{Union3_0{}};
                                        break;
                                    }
                                    case 2: { // Random
                                        curandStatePhilox4_32_10_t & v662 = v19.v4;
                                        curandStatePhilox4_32_10_t & v663 = v662;
                                        static_array_list<Union1,3> v664;
                                        v664 = static_array_list<Union1,3>{};
                                        v664.unsafe_set_length(1);
                                        Union1 v666;
                                        v666 = Union1{Union1_0{}};
                                        v664[0] = v666;
                                        int v668;
                                        v668 = v109[0];
                                        int v670;
                                        v670 = v109[1];
                                        bool v672;
                                        v672 = v668 == v670;
                                        bool v673;
                                        v673 = v672 != true;
                                        if (v673){
                                            Union1 v674;
                                            v674 = Union1{Union1_1{}};
                                            v664.push(v674);
                                        } else {
                                        }
                                        bool v675;
                                        v675 = v110 > 0;
                                        if (v675){
                                            Union1 v676;
                                            v676 = Union1{Union1_2{}};
                                            v664.push(v676);
                                        } else {
                                        }
                                        int v677;
                                        v677 = v664.length;
                                        int v678;
                                        v678 = v677 - 1;
                                        int v679;
                                        v679 = 0;
                                        while (while_method_1(v678, v679)){
                                            int v681;
                                            v681 = v664.length;
                                            int v682;
                                            v682 = int_range_22(v681, v679, v663);
                                            Union1 v683;
                                            v683 = v664[v679];
                                            Union1 v685;
                                            v685 = v664[v682];
                                            v664[v679] = v685;
                                            v664[v682] = v683;
                                            v679 += 1 ;
                                        }
                                        Union1 v687;
                                        v687 = v664.pop();
                                        int v688;
                                        v688 = sizeof(Union1);
                                        unsigned long long v689;
                                        v689 = (unsigned long long)v688;
                                        bool v690;
                                        v690 = v689 <= 98304ull;
                                        bool v691;
                                        v691 = v690 == false;
                                        if (v691){
                                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v690);
                                        } else {
                                        }
                                        extern __shared__ unsigned char v693[];
                                        bool v694;
                                        v694 = v689 <= v689;
                                        bool v695;
                                        v695 = v694 == false;
                                        if (v695){
                                            assert("The length of the partition has to be less than or equal to the length of the base array." && v694);
                                        } else {
                                        }
                                        Union1 * v697;
                                        v697 = reinterpret_cast<Union1 *>(&v693[0ull]);
                                        int v699;
                                        v699 = threadIdx.x;
                                        bool v700;
                                        v700 = v699 == 0;
                                        if (v700){
                                            v697[0] = v687;
                                        } else {
                                        }
                                        __syncthreads();
                                        Union1 v701;
                                        v701 = v697[0];
                                        __syncthreads();
                                        Union7 v702;
                                        v702 = Union7{Union7_1{v108, v701}};
                                        v58.push(v702);
                                        Union4 v807;
                                        switch (v105.tag) {
                                            case 0: { // None
                                                switch (v701.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v764;
                                                            v764 = v108 ^ 1;
                                                            v807 = Union4{Union4_2{v105, false, v107, v764, v109, v110}};
                                                        } else {
                                                            v807 = Union4{Union4_0{v105, v106, v107, v108, v109, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v807 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        if (v675){
                                                            int v768;
                                                            v768 = v108 ^ 1;
                                                            int v769;
                                                            v769 = -1 + v110;
                                                            int v770; int v771;
                                                            Tuple7 tmp29 = Tuple7{0, 0};
                                                            v770 = tmp29.v0; v771 = tmp29.v1;
                                                            while (while_method_0(v770)){
                                                                bool v773;
                                                                v773 = 0 <= v770;
                                                                bool v775;
                                                                if (v773){
                                                                    bool v774;
                                                                    v774 = v770 < 2;
                                                                    v775 = v774;
                                                                } else {
                                                                    v775 = false;
                                                                }
                                                                bool v776;
                                                                v776 = v775 == false;
                                                                if (v776){
                                                                    assert("Index must be in range." && v775);
                                                                } else {
                                                                }
                                                                int v778;
                                                                v778 = v109[v770];
                                                                bool v780;
                                                                v780 = v771 >= v778;
                                                                int v781;
                                                                if (v780){
                                                                    v781 = v771;
                                                                } else {
                                                                    v781 = v778;
                                                                }
                                                                v771 = v781;
                                                                v770 += 1 ;
                                                            }
                                                            static_array<int,2> v782;
                                                            int v784;
                                                            v784 = 0;
                                                            while (while_method_0(v784)){
                                                                v782[v784] = v771;
                                                                v784 += 1 ;
                                                            }
                                                            static_array<int,2> v786;
                                                            int v788;
                                                            v788 = 0;
                                                            while (while_method_0(v788)){
                                                                bool v790;
                                                                v790 = 0 <= v788;
                                                                bool v792;
                                                                if (v790){
                                                                    bool v791;
                                                                    v791 = v788 < 2;
                                                                    v792 = v791;
                                                                } else {
                                                                    v792 = false;
                                                                }
                                                                bool v793;
                                                                v793 = v792 == false;
                                                                if (v793){
                                                                    assert("Index must be in range." && v792);
                                                                } else {
                                                                }
                                                                int v795;
                                                                v795 = v782[v788];
                                                                bool v797;
                                                                v797 = v788 == v108;
                                                                int v799;
                                                                if (v797){
                                                                    int v798;
                                                                    v798 = v795 + 2;
                                                                    v799 = v798;
                                                                } else {
                                                                    v799 = v795;
                                                                }
                                                                v786[v788] = v799;
                                                                v788 += 1 ;
                                                            }
                                                            v807 = Union4{Union4_2{v105, false, v107, v768, v786, v769}};
                                                        } else {
                                                            printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                                            __trap();
                                                        }
                                                        break;
                                                    }
                                                    default: {
                                                        assert("Invalid tag." && false); __trap();
                                                    }
                                                }
                                                break;
                                            }
                                            case 1: { // Some
                                                Union6 v703 = v105.case1.v0;
                                                switch (v701.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v705;
                                                            v705 = v108 ^ 1;
                                                            v807 = Union4{Union4_2{v105, false, v107, v705, v109, v110}};
                                                        } else {
                                                            int v707; int v708;
                                                            Tuple7 tmp30 = Tuple7{0, 0};
                                                            v707 = tmp30.v0; v708 = tmp30.v1;
                                                            while (while_method_0(v707)){
                                                                bool v710;
                                                                v710 = 0 <= v707;
                                                                bool v712;
                                                                if (v710){
                                                                    bool v711;
                                                                    v711 = v707 < 2;
                                                                    v712 = v711;
                                                                } else {
                                                                    v712 = false;
                                                                }
                                                                bool v713;
                                                                v713 = v712 == false;
                                                                if (v713){
                                                                    assert("Index must be in range." && v712);
                                                                } else {
                                                                }
                                                                int v715;
                                                                v715 = v109[v707];
                                                                bool v717;
                                                                v717 = v708 >= v715;
                                                                int v718;
                                                                if (v717){
                                                                    v718 = v708;
                                                                } else {
                                                                    v718 = v715;
                                                                }
                                                                v708 = v718;
                                                                v707 += 1 ;
                                                            }
                                                            static_array<int,2> v719;
                                                            int v721;
                                                            v721 = 0;
                                                            while (while_method_0(v721)){
                                                                v719[v721] = v708;
                                                                v721 += 1 ;
                                                            }
                                                            v807 = Union4{Union4_4{v105, v106, v107, v108, v719, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v807 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        if (v675){
                                                            int v725;
                                                            v725 = v108 ^ 1;
                                                            int v726;
                                                            v726 = -1 + v110;
                                                            int v727; int v728;
                                                            Tuple7 tmp31 = Tuple7{0, 0};
                                                            v727 = tmp31.v0; v728 = tmp31.v1;
                                                            while (while_method_0(v727)){
                                                                bool v730;
                                                                v730 = 0 <= v727;
                                                                bool v732;
                                                                if (v730){
                                                                    bool v731;
                                                                    v731 = v727 < 2;
                                                                    v732 = v731;
                                                                } else {
                                                                    v732 = false;
                                                                }
                                                                bool v733;
                                                                v733 = v732 == false;
                                                                if (v733){
                                                                    assert("Index must be in range." && v732);
                                                                } else {
                                                                }
                                                                int v735;
                                                                v735 = v109[v727];
                                                                bool v737;
                                                                v737 = v728 >= v735;
                                                                int v738;
                                                                if (v737){
                                                                    v738 = v728;
                                                                } else {
                                                                    v738 = v735;
                                                                }
                                                                v728 = v738;
                                                                v727 += 1 ;
                                                            }
                                                            static_array<int,2> v739;
                                                            int v741;
                                                            v741 = 0;
                                                            while (while_method_0(v741)){
                                                                v739[v741] = v728;
                                                                v741 += 1 ;
                                                            }
                                                            static_array<int,2> v743;
                                                            int v745;
                                                            v745 = 0;
                                                            while (while_method_0(v745)){
                                                                bool v747;
                                                                v747 = 0 <= v745;
                                                                bool v749;
                                                                if (v747){
                                                                    bool v748;
                                                                    v748 = v745 < 2;
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
                                                                v752 = v739[v745];
                                                                bool v754;
                                                                v754 = v745 == v108;
                                                                int v756;
                                                                if (v754){
                                                                    int v755;
                                                                    v755 = v752 + 4;
                                                                    v756 = v755;
                                                                } else {
                                                                    v756 = v752;
                                                                }
                                                                v743[v745] = v756;
                                                                v745 += 1 ;
                                                            }
                                                            v807 = Union4{Union4_2{v105, false, v107, v725, v743, v726}};
                                                        } else {
                                                            printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                                            __trap();
                                                        }
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
                                        v988 = Union3{Union3_1{v807}};
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                break;
                            }
                            case 3: { // RoundWithAction
                                Union5 v812 = v62.case3.v0; bool v813 = v62.case3.v1; static_array<Union6,2> v814 = v62.case3.v2; int v815 = v62.case3.v3; static_array<int,2> v816 = v62.case3.v4; int v817 = v62.case3.v5; Union1 v818 = v62.case3.v6;
                                Union7 v819;
                                v819 = Union7{Union7_1{v815, v818}};
                                v58.push(v819);
                                Union4 v926;
                                switch (v812.tag) {
                                    case 0: { // None
                                        switch (v818.tag) {
                                            case 0: { // Call
                                                if (v813){
                                                    int v882;
                                                    v882 = v815 ^ 1;
                                                    v926 = Union4{Union4_2{v812, false, v814, v882, v816, v817}};
                                                } else {
                                                    v926 = Union4{Union4_0{v812, v813, v814, v815, v816, v817}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v926 = Union4{Union4_5{v812, v813, v814, v815, v816, v817}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                bool v886;
                                                v886 = v817 > 0;
                                                if (v886){
                                                    int v887;
                                                    v887 = v815 ^ 1;
                                                    int v888;
                                                    v888 = -1 + v817;
                                                    int v889; int v890;
                                                    Tuple7 tmp32 = Tuple7{0, 0};
                                                    v889 = tmp32.v0; v890 = tmp32.v1;
                                                    while (while_method_0(v889)){
                                                        bool v892;
                                                        v892 = 0 <= v889;
                                                        bool v894;
                                                        if (v892){
                                                            bool v893;
                                                            v893 = v889 < 2;
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
                                                        v897 = v816[v889];
                                                        bool v899;
                                                        v899 = v890 >= v897;
                                                        int v900;
                                                        if (v899){
                                                            v900 = v890;
                                                        } else {
                                                            v900 = v897;
                                                        }
                                                        v890 = v900;
                                                        v889 += 1 ;
                                                    }
                                                    static_array<int,2> v901;
                                                    int v903;
                                                    v903 = 0;
                                                    while (while_method_0(v903)){
                                                        v901[v903] = v890;
                                                        v903 += 1 ;
                                                    }
                                                    static_array<int,2> v905;
                                                    int v907;
                                                    v907 = 0;
                                                    while (while_method_0(v907)){
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
                                                            assert("Index must be in range." && v911);
                                                        } else {
                                                        }
                                                        int v914;
                                                        v914 = v901[v907];
                                                        bool v916;
                                                        v916 = v907 == v815;
                                                        int v918;
                                                        if (v916){
                                                            int v917;
                                                            v917 = v914 + 2;
                                                            v918 = v917;
                                                        } else {
                                                            v918 = v914;
                                                        }
                                                        v905[v907] = v918;
                                                        v907 += 1 ;
                                                    }
                                                    v926 = Union4{Union4_2{v812, false, v814, v887, v905, v888}};
                                                } else {
                                                    printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                                    __trap();
                                                }
                                                break;
                                            }
                                            default: {
                                                assert("Invalid tag." && false); __trap();
                                            }
                                        }
                                        break;
                                    }
                                    case 1: { // Some
                                        Union6 v820 = v812.case1.v0;
                                        switch (v818.tag) {
                                            case 0: { // Call
                                                if (v813){
                                                    int v822;
                                                    v822 = v815 ^ 1;
                                                    v926 = Union4{Union4_2{v812, false, v814, v822, v816, v817}};
                                                } else {
                                                    int v824; int v825;
                                                    Tuple7 tmp33 = Tuple7{0, 0};
                                                    v824 = tmp33.v0; v825 = tmp33.v1;
                                                    while (while_method_0(v824)){
                                                        bool v827;
                                                        v827 = 0 <= v824;
                                                        bool v829;
                                                        if (v827){
                                                            bool v828;
                                                            v828 = v824 < 2;
                                                            v829 = v828;
                                                        } else {
                                                            v829 = false;
                                                        }
                                                        bool v830;
                                                        v830 = v829 == false;
                                                        if (v830){
                                                            assert("Index must be in range." && v829);
                                                        } else {
                                                        }
                                                        int v832;
                                                        v832 = v816[v824];
                                                        bool v834;
                                                        v834 = v825 >= v832;
                                                        int v835;
                                                        if (v834){
                                                            v835 = v825;
                                                        } else {
                                                            v835 = v832;
                                                        }
                                                        v825 = v835;
                                                        v824 += 1 ;
                                                    }
                                                    static_array<int,2> v836;
                                                    int v838;
                                                    v838 = 0;
                                                    while (while_method_0(v838)){
                                                        v836[v838] = v825;
                                                        v838 += 1 ;
                                                    }
                                                    v926 = Union4{Union4_4{v812, v813, v814, v815, v836, v817}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v926 = Union4{Union4_5{v812, v813, v814, v815, v816, v817}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                bool v842;
                                                v842 = v817 > 0;
                                                if (v842){
                                                    int v843;
                                                    v843 = v815 ^ 1;
                                                    int v844;
                                                    v844 = -1 + v817;
                                                    int v845; int v846;
                                                    Tuple7 tmp34 = Tuple7{0, 0};
                                                    v845 = tmp34.v0; v846 = tmp34.v1;
                                                    while (while_method_0(v845)){
                                                        bool v848;
                                                        v848 = 0 <= v845;
                                                        bool v850;
                                                        if (v848){
                                                            bool v849;
                                                            v849 = v845 < 2;
                                                            v850 = v849;
                                                        } else {
                                                            v850 = false;
                                                        }
                                                        bool v851;
                                                        v851 = v850 == false;
                                                        if (v851){
                                                            assert("Index must be in range." && v850);
                                                        } else {
                                                        }
                                                        int v853;
                                                        v853 = v816[v845];
                                                        bool v855;
                                                        v855 = v846 >= v853;
                                                        int v856;
                                                        if (v855){
                                                            v856 = v846;
                                                        } else {
                                                            v856 = v853;
                                                        }
                                                        v846 = v856;
                                                        v845 += 1 ;
                                                    }
                                                    static_array<int,2> v857;
                                                    int v859;
                                                    v859 = 0;
                                                    while (while_method_0(v859)){
                                                        v857[v859] = v846;
                                                        v859 += 1 ;
                                                    }
                                                    static_array<int,2> v861;
                                                    int v863;
                                                    v863 = 0;
                                                    while (while_method_0(v863)){
                                                        bool v865;
                                                        v865 = 0 <= v863;
                                                        bool v867;
                                                        if (v865){
                                                            bool v866;
                                                            v866 = v863 < 2;
                                                            v867 = v866;
                                                        } else {
                                                            v867 = false;
                                                        }
                                                        bool v868;
                                                        v868 = v867 == false;
                                                        if (v868){
                                                            assert("Index must be in range." && v867);
                                                        } else {
                                                        }
                                                        int v870;
                                                        v870 = v857[v863];
                                                        bool v872;
                                                        v872 = v863 == v815;
                                                        int v874;
                                                        if (v872){
                                                            int v873;
                                                            v873 = v870 + 4;
                                                            v874 = v873;
                                                        } else {
                                                            v874 = v870;
                                                        }
                                                        v861[v863] = v874;
                                                        v863 += 1 ;
                                                    }
                                                    v926 = Union4{Union4_2{v812, false, v814, v843, v861, v844}};
                                                } else {
                                                    printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                                    __trap();
                                                }
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
                                v988 = Union3{Union3_1{v926}};
                                break;
                            }
                            case 4: { // TerminalCall
                                Union5 v81 = v62.case4.v0; bool v82 = v62.case4.v1; static_array<Union6,2> v83 = v62.case4.v2; int v84 = v62.case4.v3; static_array<int,2> v85 = v62.case4.v4; int v86 = v62.case4.v5;
                                bool v87;
                                v87 = 0 <= v84;
                                bool v89;
                                if (v87){
                                    bool v88;
                                    v88 = v84 < 2;
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
                                int v92;
                                v92 = v85[v84];
                                Union13 v94;
                                v94 = compare_hands_26(v81, v82, v83, v84, v85, v86);
                                int v99; int v100;
                                switch (v94.tag) {
                                    case 0: { // Eq
                                        v99 = 0; v100 = -1;
                                        break;
                                    }
                                    case 1: { // Gt
                                        v99 = v92; v100 = 0;
                                        break;
                                    }
                                    case 2: { // Lt
                                        v99 = v92; v100 = 1;
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                Union7 v101;
                                v101 = Union7{Union7_3{v83, v99, v100}};
                                v58.push(v101);
                                Union8 v102;
                                v102 = Union8{Union8_1{v81, v82, v83, v84, v85, v86}};
                                v19.v5 = v102;
                                Union3 v103;
                                v103 = Union3{Union3_0{}};
                                v19.v1 = v103;
                                v988 = Union3{Union3_0{}};
                                break;
                            }
                            case 5: { // TerminalFold
                                Union5 v63 = v62.case5.v0; bool v64 = v62.case5.v1; static_array<Union6,2> v65 = v62.case5.v2; int v66 = v62.case5.v3; static_array<int,2> v67 = v62.case5.v4; int v68 = v62.case5.v5;
                                bool v69;
                                v69 = 0 <= v66;
                                bool v71;
                                if (v69){
                                    bool v70;
                                    v70 = v66 < 2;
                                    v71 = v70;
                                } else {
                                    v71 = false;
                                }
                                bool v72;
                                v72 = v71 == false;
                                if (v72){
                                    assert("Index must be in range." && v71);
                                } else {
                                }
                                int v74;
                                v74 = v67[v66];
                                int v76;
                                v76 = v66 ^ 1;
                                Union7 v77;
                                v77 = Union7{Union7_3{v65, v74, v76}};
                                v58.push(v77);
                                Union8 v78;
                                v78 = Union8{Union8_1{v63, v64, v65, v66, v67, v68}};
                                v19.v5 = v78;
                                Union3 v79;
                                v79 = Union3{Union3_0{}};
                                v19.v1 = v79;
                                v988 = Union3{Union3_0{}};
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
                v60 = v988;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v989;
    v989 = threadIdx.x;
    int v990;
    v990 = blockIdx.x;
    int v991;
    v991 = v990 * 256;
    int v992;
    v992 = v989 + v991;
    bool v993;
    v993 = v992 == 0;
    if (v993){
        Union8 & v994 = v19.v5;
        static_array<Union2,2> & v995 = v19.v3;
        static_array_list<Union7,32> & v996 = v19.v2;
        Union3 & v997 = v19.v1;
        unsigned int & v998 = v19.v0;
        return f_30(v0, v998, v997, v996, v995, v994);
    } else {
        return ;
    }
}
extern "C" __global__ void entry1(unsigned char * v0, unsigned char * v1, unsigned char * v2, float * v3, float * v4, float * v5) {
    auto v6 = cooperative_groups::this_grid();
    unsigned long long v7;
    v7 = clock64();
    int v8;
    v8 = threadIdx.x;
    int v9;
    v9 = blockIdx.x;
    int v10;
    v10 = v9 * 256;
    int v11;
    v11 = v8 + v10;
    unsigned long long v12;
    v12 = (unsigned long long)v11;
    curandStatePhilox4_32_10_t v13;
    curand_init(v7,v12,0ull,&v13);
    static_array<Union2,2> v14;
    Union2 v16;
    v16 = Union2{Union2_2{}};
    v14[0] = v16;
    Union2 v18;
    v18 = Union2{Union2_2{}};
    v14[1] = v18;
    static_array_list<Union7,32> v20;
    v20 = static_array_list<Union7,32>{};
    static_array<float,2> v22;
    v22[0] = 0.0f;
    v22[1] = 0.0f;
    cooperative_groups::grid_group & v24 = v6;
    curandStatePhilox4_32_10_t & v25 = v13;
    StackMut1 v26{63u, v24, v20, v14, v22, v25};
    int v27;
    v27 = 0;
    while (while_method_10(v27)){
        int v29;
        v29 = 0;
        while (while_method_5(v29)){
            int v31;
            v31 = 0;
            while (while_method_0(v31)){
                Union4 v33;
                v33 = Union4{Union4_1{}};
                method_47(v0, v1, v2, v26, v31, v33);
                static_array<float,2> & v34 = v26.v4;
                bool v35;
                v35 = 0 <= v31;
                bool v37;
                if (v35){
                    bool v36;
                    v36 = v31 < 2;
                    v37 = v36;
                } else {
                    v37 = false;
                }
                bool v38;
                v38 = v37 == false;
                if (v38){
                    assert("Index must be in range." && v37);
                } else {
                }
                float v40;
                v40 = v34[v31];
                double * v42;
                v42 = reinterpret_cast<double *>(&v1[105381888ull]);
                double * v44;
                v44 = reinterpret_cast<double *>(&v1[108527616ull]);
                int v46;
                v46 = threadIdx.x;
                int v47;
                v47 = blockIdx.x;
                int v48;
                v48 = v47 * 256;
                int v49;
                v49 = v46 + v48;
                assert("Tensor range check" && 0 <= v49 && v49 < 6144);
                int v50;
                v50 = 2 * v49;
                double v51[2];
                int v52;
                v52 = 0;
                while (while_method_0(v52)){
                    int v54; double v55;
                    Tuple12 tmp55 = Tuple12{0, 0.0};
                    v54 = tmp55.v0; v55 = tmp55.v1;
                    while (while_method_10(v54)){
                        assert("Tensor range check" && 0 <= v54 && v54 < 32);
                        assert("Tensor range check" && 0 <= v52 && v52 < 2);
                        int v57;
                        v57 = v52 + v50;
                        int v58;
                        v58 = 12288 * v54;
                        int v59;
                        v59 = v58 + v57;
                        double v60;
                        v60 = v42[v59];
                        double v61;
                        v61 = v44[v59];
                        double v62;
                        v62 = v60 - v61;
                        double v63;
                        v63 = exp(v62);
                        double v64;
                        v64 = v55 + v63;
                        v55 = v64;
                        v54 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v52 && v52 < 2);
                    v51[v52] = v55;
                    v52 += 1 ;
                }
                double v65;
                v65 = 1.0;
                int v66;
                v66 = 0;
                while (while_method_0(v66)){
                    assert("Tensor range check" && 0 <= v66 && v66 < 2);
                    double v68;
                    v68 = v51[v66];
                    double v69;
                    v69 = v65 * v68;
                    v65 = v69;
                    v66 += 1 ;
                }
                double v70[64];
                int v71;
                v71 = 0;
                while (while_method_10(v71)){
                    int v73;
                    v73 = 0;
                    while (while_method_0(v73)){
                        assert("Tensor range check" && 0 <= v73 && v73 < 2);
                        double v75;
                        v75 = v51[v73];
                        double v76;
                        v76 = v65 / v75;
                        assert("Tensor range check" && 0 <= v71 && v71 < 32);
                        assert("Tensor range check" && 0 <= v73 && v73 < 2);
                        int v77;
                        v77 = v73 + v50;
                        int v78;
                        v78 = 12288 * v71;
                        int v79;
                        v79 = v78 + v77;
                        double v80;
                        v80 = v42[v79];
                        double v81;
                        v81 = v44[v79];
                        double v82;
                        v82 = v80 - v81;
                        double v83;
                        v83 = exp(v82);
                        double v84;
                        v84 = v76 * v83;
                        assert("Tensor range check" && 0 <= v71 && v71 < 32);
                        assert("Tensor range check" && 0 <= v73 && v73 < 2);
                        int v85;
                        v85 = 2 * v71;
                        int v86;
                        v86 = v85 + v73;
                        v70[v86] = v84;
                        v73 += 1 ;
                    }
                    v71 += 1 ;
                }
                int v87;
                v87 = 0;
                while (while_method_10(v87)){
                    assert("Tensor range check" && 0 <= v87 && v87 < 32);
                    assert("Tensor range check" && 0 <= v31 && v31 < 2);
                    int v89;
                    v89 = 2 * v87;
                    int v90;
                    v90 = v89 + v31;
                    double v91;
                    v91 = v70[v90];
                    float v92;
                    v92 = (float)v91;
                    float v93;
                    v93 = v40 * v92;
                    assert("Tensor range check" && 0 <= v87 && v87 < 32);
                    assert("Tensor range check" && 0 <= v27 && v27 < 32);
                    int v94;
                    v94 = 32 * v87;
                    int v95;
                    v95 = v94 + v27;
                    float * v96;
                    v96 = v3+v95;
                    float * v98;
                    v98 = v4+v95;
                    float v100;
                    v100 = atomicAdd(v96,v93);
                    float v101;
                    v101 = atomicAdd(v98,v92);
                    v87 += 1 ;
                }
                static_array<float,2> & v102 = v26.v4;
                float * v103;
                v103 = reinterpret_cast<float *>(&v1[55050240ull]);
                int * v105;
                v105 = reinterpret_cast<int *>(&v0[1048576ull]);
                float * v107;
                v107 = reinterpret_cast<float *>(&v0[1048592ull]);
                float * v109;
                v109 = reinterpret_cast<float *>(&v0[1048720ull]);
                double * v111;
                v111 = reinterpret_cast<double *>(&v1[105381888ull]);
                double * v113;
                v113 = reinterpret_cast<double *>(&v1[108527616ull]);
                int v115;
                v115 = threadIdx.x;
                int v116;
                v116 = blockIdx.x;
                int v117;
                v117 = v116 * 256;
                int v118;
                v118 = v115 + v117;
                assert("Tensor range check" && 0 <= v118 && v118 < 6144);
                int v119;
                v119 = 2 * v118;
                double * v120;
                v120 = v111+v119;
                double * v122;
                v122 = v113+v119;
                float v124[2];
                int v125;
                v125 = 0;
                while (while_method_0(v125)){
                    bool v127;
                    v127 = 0 <= v125;
                    bool v129;
                    if (v127){
                        bool v128;
                        v128 = v125 < 2;
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
                    float v132;
                    v132 = v102[v125];
                    assert("Tensor range check" && 0 <= v125 && v125 < 2);
                    v124[v125] = v132;
                    v125 += 1 ;
                }
                double v134[2];
                int v135;
                v135 = 0;
                while (while_method_0(v135)){
                    int v137; double v138;
                    Tuple12 tmp56 = Tuple12{0, 0.0};
                    v137 = tmp56.v0; v138 = tmp56.v1;
                    while (while_method_10(v137)){
                        assert("Tensor range check" && 0 <= v137 && v137 < 32);
                        assert("Tensor range check" && 0 <= v135 && v135 < 2);
                        int v140;
                        v140 = 12288 * v137;
                        int v141;
                        v141 = v140 + v135;
                        double v142;
                        v142 = v120[v141];
                        double v143;
                        v143 = v122[v141];
                        double v144;
                        v144 = v142 - v143;
                        double v145;
                        v145 = exp(v144);
                        double v146;
                        v146 = v138 + v145;
                        v138 = v146;
                        v137 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v135 && v135 < 2);
                    v134[v135] = v138;
                    v135 += 1 ;
                }
                double v147;
                v147 = 1.0;
                int v148;
                v148 = 0;
                while (while_method_0(v148)){
                    assert("Tensor range check" && 0 <= v148 && v148 < 2);
                    double v150;
                    v150 = v134[v148];
                    double v151;
                    v151 = v147 * v150;
                    v147 = v151;
                    v148 += 1 ;
                }
                double v152[64];
                int v153;
                v153 = 0;
                while (while_method_10(v153)){
                    int v155;
                    v155 = 0;
                    while (while_method_0(v155)){
                        assert("Tensor range check" && 0 <= v155 && v155 < 2);
                        double v157;
                        v157 = v134[v155];
                        double v158;
                        v158 = v147 / v157;
                        assert("Tensor range check" && 0 <= v153 && v153 < 32);
                        assert("Tensor range check" && 0 <= v155 && v155 < 2);
                        int v159;
                        v159 = 12288 * v153;
                        int v160;
                        v160 = v159 + v155;
                        double v161;
                        v161 = v120[v160];
                        double v162;
                        v162 = v122[v160];
                        double v163;
                        v163 = v161 - v162;
                        double v164;
                        v164 = exp(v163);
                        double v165;
                        v165 = v158 * v164;
                        assert("Tensor range check" && 0 <= v153 && v153 < 32);
                        assert("Tensor range check" && 0 <= v155 && v155 < 2);
                        int v166;
                        v166 = 2 * v153;
                        int v167;
                        v167 = v166 + v155;
                        v152[v167] = v165;
                        v155 += 1 ;
                    }
                    v153 += 1 ;
                }
                float v168[32];
                float v169[32];
                int v170;
                v170 = 0;
                while (while_method_10(v170)){
                    int v172; float v173; double v174;
                    Tuple13 tmp57 = Tuple13{0, 0.0f, 0.0};
                    v172 = tmp57.v0; v173 = tmp57.v1; v174 = tmp57.v2;
                    while (while_method_0(v172)){
                        assert("Tensor range check" && 0 <= v170 && v170 < 32);
                        assert("Tensor range check" && 0 <= v172 && v172 < 2);
                        int v176;
                        v176 = 2 * v170;
                        int v177;
                        v177 = v176 + v172;
                        double v178;
                        v178 = v152[v177];
                        assert("Tensor range check" && 0 <= v172 && v172 < 2);
                        float v179;
                        v179 = v124[v172];
                        float v180;
                        v180 = (float)v178;
                        float v181;
                        v181 = v180 * v179;
                        float v182;
                        v182 = v173 + v181;
                        double v183;
                        v183 = v174 + v178;
                        v173 = v182;
                        v174 = v183;
                        v172 += 1 ;
                    }
                    float v184;
                    v184 = (float)v174;
                    assert("Tensor range check" && 0 <= v170 && v170 < 32);
                    v168[v170] = v173;
                    v169[v170] = v184;
                    v170 += 1 ;
                }
                int v185;
                v185 = 0;
                while (while_method_10(v185)){
                    assert("Tensor range check" && 0 <= v185 && v185 < 32);
                    float v187;
                    v187 = v168[v185];
                    float v188;
                    v188 = v169[v185];
                    float v189;
                    v189 = v187 * v188;
                    assert("Tensor range check" && 0 <= v185 && v185 < 32);
                    float * v190;
                    v190 = v107+v185;
                    float * v192;
                    v192 = v109+v185;
                    float v194;
                    v194 = atomicAdd(v190,v189);
                    float v195;
                    v195 = atomicAdd(v192,v188);
                    v185 += 1 ;
                }
                int v196;
                v196 = threadIdx.x;
                int v197;
                v197 = blockIdx.x;
                int v198;
                v198 = v197 * 256;
                int v199;
                v199 = v196 + v198;
                int v200;
                v200 = 0;
                while (while_method_10(v200)){
                    assert("Tensor range check" && 0 <= v200 && v200 < 32);
                    int v202;
                    v202 = 12288 * v200;
                    assert("Tensor range check" && 0 <= v199 && v199 < 6144);
                    int v203;
                    v203 = 2 * v199;
                    int v204;
                    v204 = v203 + v202;
                    double * v205;
                    v205 = v111+v204;
                    double * v207;
                    v207 = v113+v204;
                    double * v209;
                    v209 = v111+v204;
                    double * v211;
                    v211 = v113+v204;
                    int v213;
                    v213 = sizeof(double *);
                    unsigned long long v214;
                    v214 = (unsigned long long)v213;
                    unsigned long long v215;
                    v215 = 256ull * v214;
                    unsigned long long v216;
                    v216 = v215 + 16ull;
                    unsigned long long v217;
                    v217 = v216 - 1ull;
                    unsigned long long v218;
                    v218 = v217 % 16ull;
                    unsigned long long v219;
                    v219 = v217 - v218;
                    unsigned long long v220;
                    v220 = v219 + v215;
                    unsigned long long v221;
                    v221 = v220 + 16ull;
                    unsigned long long v222;
                    v222 = v221 - 1ull;
                    unsigned long long v223;
                    v223 = v222 % 16ull;
                    unsigned long long v224;
                    v224 = v222 - v223;
                    unsigned long long v225;
                    v225 = v224 + v215;
                    unsigned long long v226;
                    v226 = v225 + 16ull;
                    unsigned long long v227;
                    v227 = v226 - 1ull;
                    unsigned long long v228;
                    v228 = v227 % 16ull;
                    unsigned long long v229;
                    v229 = v227 - v228;
                    unsigned long long v230;
                    v230 = v229 + v215;
                    bool v231;
                    v231 = v230 <= 98304ull;
                    bool v232;
                    v232 = v231 == false;
                    if (v232){
                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v231);
                    } else {
                    }
                    extern __shared__ unsigned char v234[];
                    bool v235;
                    v235 = v230 <= v230;
                    bool v236;
                    v236 = v235 == false;
                    if (v236){
                        assert("The length of the partition has to be less than or equal to the length of the base array." && v235);
                    } else {
                    }
                    double * * v238;
                    v238 = reinterpret_cast<double * *>(&v234[0ull]);
                    double * * v240;
                    v240 = reinterpret_cast<double * *>(&v234[v219]);
                    double * * v242;
                    v242 = reinterpret_cast<double * *>(&v234[v224]);
                    double * * v244;
                    v244 = reinterpret_cast<double * *>(&v234[v229]);
                    int v246;
                    v246 = threadIdx.x;
                    assert("Tensor range check" && 0 <= v246 && v246 < 256);
                    v238[v246] = v205;
                    v240[v246] = v207;
                    v242[v246] = v209;
                    v244[v246] = v211;
                    __syncthreads();
                    bool v247;
                    v247 = 0 <= v246;
                    bool v248;
                    v248 = v247 == false;
                    if (v248){
                        assert("The index needs to be zero or positive." && v247);
                    } else {
                    }
                    int v250;
                    v250 = v246 % 1;
                    bool v251;
                    v251 = v246 < 256;
                    bool v252;
                    v252 = v251 == false;
                    if (v252){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v251);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v246 && v246 < 256);
                    int v254;
                    v254 = 0;
                    while (while_method_4(v254)){
                        bool v256;
                        v256 = v247 && v251;
                        bool v257;
                        v257 = v256 == false;
                        if (v257){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v256);
                        } else {
                        }
                        bool v259;
                        v259 = 0 <= v254;
                        bool v261;
                        if (v259){
                            bool v260;
                            v260 = v254 < 1;
                            v261 = v260;
                        } else {
                            v261 = false;
                        }
                        bool v262;
                        v262 = v261 == false;
                        if (v262){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v261);
                        } else {
                        }
                        int v264;
                        v264 = v254 * 256;
                        int v265;
                        v265 = v264 + v246;
                        assert("Tensor range check" && 0 <= v254 && v254 < 1);
                        int v266;
                        v266 = 256 * v254;
                        int v267;
                        v267 = v266 + v246;
                        double * v268;
                        v268 = v238[v267];
                        double * v269;
                        v269 = v240[v267];
                        double * v270;
                        v270 = v242[v267];
                        double * v271;
                        v271 = v244[v267];
                        int v272;
                        v272 = blockIdx.x;
                        int v273;
                        v273 = v272 * 256;
                        int v274;
                        v274 = v273 + v265;
                        assert("Tensor range check" && 0 <= v250 && v250 < 1);
                        int v275;
                        v275 = 2 * v250;
                        double v276[2];
                        double v277[2];
                        int v278[2];
                        int v279;
                        v279 = 0;
                        while (while_method_4(v279)){
                            assert("Tensor range check" && 0 <= v279 && v279 < 1);
                            int v281;
                            v281 = 2 * v279;
                            assert("Tensor range check" && 0 <= v279 && v279 < 1);
                            int v282;
                            v282 = v281 + v275;
                            int4* v283;
                            v283 = reinterpret_cast<int4*>(v268 + v282);
                            int4* v284;
                            v284 = reinterpret_cast<int4*>(v276 + v281);
                            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v283) % 16 == 0 && reinterpret_cast<unsigned long long>(v284) % 16 == 0);
                            *v284 = *v283;
                            int4* v285;
                            v285 = reinterpret_cast<int4*>(v269 + v282);
                            int4* v286;
                            v286 = reinterpret_cast<int4*>(v277 + v281);
                            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v285) % 16 == 0 && reinterpret_cast<unsigned long long>(v286) % 16 == 0);
                            *v286 = *v285;
                            v279 += 1 ;
                        }
                        int v287;
                        v287 = 0;
                        while (while_method_4(v287)){
                            int v289;
                            v289 = 0;
                            while (while_method_0(v289)){
                                bool v291;
                                v291 = 0 <= v289;
                                bool v293;
                                if (v291){
                                    bool v292;
                                    v292 = v289 < 2;
                                    v293 = v292;
                                } else {
                                    v293 = false;
                                }
                                bool v294;
                                v294 = v293 == false;
                                if (v294){
                                    assert("The indices should be inside the range of the dimension." && v293);
                                } else {
                                }
                                bool v296;
                                v296 = 0 <= v250;
                                bool v298;
                                if (v296){
                                    bool v297;
                                    v297 = v250 < 1;
                                    v298 = v297;
                                } else {
                                    v298 = false;
                                }
                                bool v299;
                                v299 = v298 == false;
                                if (v299){
                                    assert("The indices should be inside the range of the dimension." && v298);
                                } else {
                                }
                                int v301;
                                v301 = v250 * 2;
                                int v302;
                                v302 = v289 + v301;
                                bool v303;
                                v303 = 0 <= v287;
                                bool v305;
                                if (v303){
                                    bool v304;
                                    v304 = v287 < 1;
                                    v305 = v304;
                                } else {
                                    v305 = false;
                                }
                                bool v306;
                                v306 = v305 == false;
                                if (v306){
                                    assert("The indices should be inside the range of the dimension." && v305);
                                } else {
                                }
                                int v308;
                                v308 = v287 * 2;
                                int v309;
                                v309 = v302 + v308;
                                assert("Tensor range check" && 0 <= v287 && v287 < 1);
                                assert("Tensor range check" && 0 <= v289 && v289 < 2);
                                int v310;
                                v310 = 2 * v287;
                                int v311;
                                v311 = v310 + v289;
                                v278[v311] = v309;
                                v289 += 1 ;
                            }
                            v287 += 1 ;
                        }
                        double v312[2];
                        double v313[2];
                        int v314;
                        v314 = 0;
                        while (while_method_4(v314)){
                            int v316;
                            v316 = 0;
                            while (while_method_0(v316)){
                                assert("Tensor range check" && 0 <= v314 && v314 < 1);
                                assert("Tensor range check" && 0 <= v316 && v316 < 2);
                                int v318;
                                v318 = 2 * v314;
                                int v319;
                                v319 = v318 + v316;
                                double v320;
                                v320 = v276[v319];
                                double v321;
                                v321 = v277[v319];
                                assert("Tensor range check" && 0 <= v314 && v314 < 1);
                                assert("Tensor range check" && 0 <= v316 && v316 < 2);
                                v312[v319] = 0.0;
                                v313[v319] = 0.0;
                                v316 += 1 ;
                            }
                            v314 += 1 ;
                        }
                        int v322;
                        v322 = 0;
                        while (while_method_4(v322)){
                            assert("Tensor range check" && 0 <= v322 && v322 < 1);
                            int v324;
                            v324 = 2 * v322;
                            int v325;
                            v325 = v324 + v275;
                            assert("Tensor range check" && 0 <= v322 && v322 < 1);
                            int4* v326;
                            v326 = reinterpret_cast<int4*>(v312 + v324);
                            int4* v327;
                            v327 = reinterpret_cast<int4*>(v270 + v325);
                            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v326) % 16 == 0 && reinterpret_cast<unsigned long long>(v327) % 16 == 0);
                            *v327 = *v326;
                            int4* v328;
                            v328 = reinterpret_cast<int4*>(v313 + v324);
                            int4* v329;
                            v329 = reinterpret_cast<int4*>(v271 + v325);
                            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v328) % 16 == 0 && reinterpret_cast<unsigned long long>(v329) % 16 == 0);
                            *v329 = *v328;
                            v322 += 1 ;
                        }
                        assert("Tensor range check" && 0 <= v265 && v265 < 256);
                        v254 += 1 ;
                    }
                    __syncthreads();
                    assert("Tensor range check" && 0 <= v246 && v246 < 256);
                    __syncthreads();
                    v200 += 1 ;
                }
                v31 += 1 ;
            }
            v29 += 1 ;
        }
        cooperative_groups::grid_group & v330 = v26.v1;
        cooperative_groups::grid_group & v331 = v330;
        curandStatePhilox4_32_10_t & v332 = v26.v5;
        curandStatePhilox4_32_10_t & v333 = v332;
        float * v334;
        v334 = reinterpret_cast<float *>(&v0[0ull]);
        float * v336;
        v336 = reinterpret_cast<float *>(&v2[0ull]);
        float * v338;
        v338 = reinterpret_cast<float *>(&v1[55050240ull]);
        int * v340;
        v340 = reinterpret_cast<int *>(&v0[1048576ull]);
        float * v342;
        v342 = reinterpret_cast<float *>(&v0[1048592ull]);
        float * v344;
        v344 = reinterpret_cast<float *>(&v0[1048720ull]);
        double * v346;
        v346 = reinterpret_cast<double *>(&v1[105381888ull]);
        double * v348;
        v348 = reinterpret_cast<double *>(&v1[108527616ull]);
        v331.sync() ;
        int v350;
        v350 = threadIdx.x;
        int v351;
        v351 = blockIdx.x;
        int v352;
        v352 = v351 * 256;
        int v353;
        v353 = v350 + v352;
        bool v354;
        v354 = v353 == 0;
        if (v354){
            int v355;
            v355 = 0;
            int v356;
            v356 = 32;
            int v357;
            v357 = int_range_22(v356, v355, v333);
            v340[0] = v357;
        } else {
        }
        __syncwarp();
        extern __shared__ unsigned char v358[];
        float * v359;
        v359 = reinterpret_cast<float *>(&v358[0ull]);
        int v361;
        v361 = blockIdx.x;
        int v362;
        v362 = v361;
        while (while_method_8(v362)){
            bool v364;
            v364 = 0 <= v362;
            bool v365;
            v365 = v364 == false;
            if (v365){
                assert("The index needs to be zero or positive." && v364);
            } else {
            }
            int v367;
            v367 = v362 % 16;
            int v368;
            v368 = v362 / 16;
            bool v369;
            v369 = v368 < 1;
            bool v370;
            v370 = v369 == false;
            if (v370){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v369);
            } else {
            }
            assert("Tensor range check" && 0 <= v368 && v368 < 1);
            assert("Tensor range check" && 0 <= v367 && v367 < 16);
            int v372;
            v372 = 512 * v367;
            int v373;
            v373 = 262144 * v368;
            int v374;
            v374 = v373 + v372;
            int v375;
            v375 = 16384 * v367;
            int v376;
            v376 = 32 * v368;
            int v377;
            v377 = v376 + v375;
            int v378;
            v378 = threadIdx.x;
            int v379;
            v379 = v378;
            while (while_method_12(v379)){
                bool v381;
                v381 = 0 <= v379;
                bool v382;
                v382 = v381 == false;
                if (v382){
                    assert("The index needs to be zero or positive." && v381);
                } else {
                }
                int v384;
                v384 = v379 % 512;
                int v385;
                v385 = v379 / 512;
                bool v386;
                v386 = v385 < 32;
                bool v387;
                v387 = v386 == false;
                if (v387){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v386);
                } else {
                }
                assert("Tensor range check" && 0 <= v385 && v385 < 32);
                assert("Tensor range check" && 0 <= v384 && v384 < 512);
                int v389;
                v389 = v384 + v374;
                int v390;
                v390 = 8192 * v385;
                int v391;
                v391 = v390 + v389;
                float v392;
                v392 = v334[v391];
                assert("Tensor range check" && 0 <= v385 && v385 < 32);
                assert("Tensor range check" && 0 <= v384 && v384 < 512);
                int v393;
                v393 = 513 * v385;
                int v394;
                v394 = v393 + v384;
                v359[v394] = v392;
                v379 += 256 ;
            }
            __syncthreads();
            int v395;
            v395 = threadIdx.x;
            int v396;
            v396 = v395;
            while (while_method_12(v396)){
                bool v398;
                v398 = 0 <= v396;
                bool v399;
                v399 = v398 == false;
                if (v399){
                    assert("The index needs to be zero or positive." && v398);
                } else {
                }
                int v401;
                v401 = v396 % 32;
                int v402;
                v402 = v396 / 32;
                bool v403;
                v403 = v402 < 512;
                bool v404;
                v404 = v403 == false;
                if (v404){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v403);
                } else {
                }
                assert("Tensor range check" && 0 <= v402 && v402 < 512);
                assert("Tensor range check" && 0 <= v401 && v401 < 32);
                int v406;
                v406 = 513 * v401;
                int v407;
                v407 = v402 + v406;
                float v408;
                v408 = v359[v407];
                assert("Tensor range check" && 0 <= v402 && v402 < 512);
                assert("Tensor range check" && 0 <= v401 && v401 < 32);
                int v409;
                v409 = v401 + v377;
                int v410;
                v410 = 32 * v402;
                int v411;
                v411 = v410 + v409;
                v336[v411] = v408;
                v396 += 256 ;
            }
            __syncthreads();
            v362 += 24 ;
        }
        v331.sync() ;
        int v412;
        v412 = threadIdx.x;
        bool v413;
        v413 = 0 <= v412;
        bool v414;
        v414 = v413 == false;
        if (v414){
            assert("The index needs to be zero or positive." && v413);
        } else {
        }
        int v416;
        v416 = v412 % 8;
        int v417;
        v417 = v412 / 8;
        int v418;
        v418 = v417 % 32;
        int v419;
        v419 = v417 / 32;
        bool v420;
        v420 = v419 < 1;
        bool v421;
        v421 = v420 == false;
        if (v421){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v420);
        } else {
        }
        assert("Tensor range check" && 0 <= v419 && v419 < 1);
        assert("Tensor range check" && 0 <= v418 && v418 < 32);
        assert("Tensor range check" && 0 <= v416 && v416 < 8);
        int v423;
        v423 = 4 * v416;
        int v424;
        v424 = 32 * v418;
        int v425;
        v425 = v424 + v423;
        int v426;
        v426 = 4096 * v419;
        int v427;
        v427 = v426 + v425;
        assert("Tensor range check" && 0 <= v419 && v419 < 1);
        assert("Tensor range check" && 0 <= v418 && v418 < 32);
        assert("Tensor range check" && 0 <= v416 && v416 < 8);
        int v428;
        v428 = blockIdx.x;
        int v429;
        v429 = v428;
        while (while_method_13(v429)){
            bool v431;
            v431 = 0 <= v429;
            bool v432;
            v432 = v431 == false;
            if (v432){
                assert("The index needs to be zero or positive." && v431);
            } else {
            }
            int v434;
            v434 = v429 % 4;
            int v435;
            v435 = v429 / 4;
            bool v436;
            v436 = v435 < 64;
            bool v437;
            v437 = v436 == false;
            if (v437){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v436);
            } else {
            }
            assert("Tensor range check" && 0 <= v435 && v435 < 64);
            assert("Tensor range check" && 0 <= v434 && v434 < 4);
            int v439;
            v439 = 1024 * v434;
            int v440;
            v440 = v439 + v427;
            int v441;
            v441 = 4096 * v435;
            int v442;
            v442 = v441 + v440;
            float v443[4];
            int v444[4];
            int v445;
            v445 = 0;
            while (while_method_4(v445)){
                assert("Tensor range check" && 0 <= v445 && v445 < 1);
                int v447;
                v447 = 4 * v445;
                assert("Tensor range check" && 0 <= v445 && v445 < 1);
                int v448;
                v448 = 32 * v445;
                int v449;
                v449 = v448 + v442;
                int4* v450;
                v450 = reinterpret_cast<int4*>(v336 + v449);
                int4* v451;
                v451 = reinterpret_cast<int4*>(v443 + v447);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v450) % 16 == 0 && reinterpret_cast<unsigned long long>(v451) % 16 == 0);
                *v451 = *v450;
                v445 += 1 ;
            }
            int v452;
            v452 = 0;
            while (while_method_4(v452)){
                int v454;
                v454 = 0;
                while (while_method_7(v454)){
                    bool v456;
                    v456 = 0 <= v454;
                    bool v458;
                    if (v456){
                        bool v457;
                        v457 = v454 < 4;
                        v458 = v457;
                    } else {
                        v458 = false;
                    }
                    bool v459;
                    v459 = v458 == false;
                    if (v459){
                        assert("The indices should be inside the range of the dimension." && v458);
                    } else {
                    }
                    bool v461;
                    v461 = 0 <= v416;
                    bool v463;
                    if (v461){
                        bool v462;
                        v462 = v416 < 8;
                        v463 = v462;
                    } else {
                        v463 = false;
                    }
                    bool v464;
                    v464 = v463 == false;
                    if (v464){
                        assert("The indices should be inside the range of the dimension." && v463);
                    } else {
                    }
                    int v466;
                    v466 = v416 * 4;
                    int v467;
                    v467 = v454 + v466;
                    bool v468;
                    v468 = 0 <= v452;
                    bool v470;
                    if (v468){
                        bool v469;
                        v469 = v452 < 1;
                        v470 = v469;
                    } else {
                        v470 = false;
                    }
                    bool v471;
                    v471 = v470 == false;
                    if (v471){
                        assert("The indices should be inside the range of the dimension." && v470);
                    } else {
                    }
                    int v473;
                    v473 = v452 * 32;
                    int v474;
                    v474 = v467 + v473;
                    assert("Tensor range check" && 0 <= v452 && v452 < 1);
                    assert("Tensor range check" && 0 <= v454 && v454 < 4);
                    int v475;
                    v475 = 4 * v452;
                    int v476;
                    v476 = v475 + v454;
                    v444[v476] = v474;
                    v454 += 1 ;
                }
                v452 += 1 ;
            }
            bool v477;
            v477 = 0 <= v419;
            bool v478;
            v478 = v477 && v420;
            bool v479;
            v479 = v478 == false;
            if (v479){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v478);
            } else {
            }
            bool v481;
            v481 = 0 <= v418;
            bool v483;
            if (v481){
                bool v482;
                v482 = v418 < 32;
                v483 = v482;
            } else {
                v483 = false;
            }
            bool v484;
            v484 = v483 == false;
            if (v484){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v483);
            } else {
            }
            bool v486;
            v486 = 0 <= v435;
            bool v487;
            v487 = v486 && v436;
            bool v488;
            v488 = v487 == false;
            if (v488){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v487);
            } else {
            }
            bool v490;
            v490 = 0 <= v434;
            bool v492;
            if (v490){
                bool v491;
                v491 = v434 < 4;
                v492 = v491;
            } else {
                v492 = false;
            }
            bool v493;
            v493 = v492 == false;
            if (v493){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v492);
            } else {
            }
            int v495;
            v495 = v434 * 32;
            int v496;
            v496 = v435 + v419;
            int v497;
            v497 = v495 + v418;
            float v498[4];
            int v499;
            v499 = 0;
            while (while_method_4(v499)){
                int v501;
                v501 = 0;
                while (while_method_7(v501)){
                    assert("Tensor range check" && 0 <= v499 && v499 < 1);
                    assert("Tensor range check" && 0 <= v501 && v501 < 4);
                    int v503;
                    v503 = 4 * v499;
                    int v504;
                    v504 = v503 + v501;
                    int v505;
                    v505 = v444[v504];
                    assert("Tensor range check" && 0 <= v505 && v505 < 32);
                    float v506;
                    v506 = v342[v505];
                    float v507;
                    v507 = v344[v505];
                    bool v508;
                    v508 = v507 == 0.0f;
                    bool v509;
                    v509 = v508 != true;
                    float v511;
                    if (v509){
                        float v510;
                        v510 = v506 / v507;
                        v511 = v510;
                    } else {
                        v511 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v499 && v499 < 1);
                    assert("Tensor range check" && 0 <= v501 && v501 < 4);
                    v498[v504] = v511;
                    v501 += 1 ;
                }
                v499 += 1 ;
            }
            float v512;
            v512 = 0.0f;
            int v513;
            v513 = 0;
            while (while_method_4(v513)){
                int v515;
                v515 = 0;
                while (while_method_7(v515)){
                    assert("Tensor range check" && 0 <= v513 && v513 < 1);
                    assert("Tensor range check" && 0 <= v515 && v515 < 4);
                    int v517;
                    v517 = 4 * v513;
                    int v518;
                    v518 = v517 + v515;
                    float v519;
                    v519 = v498[v518];
                    float v520;
                    v520 = v512 + v519;
                    v512 = v520;
                    v515 += 1 ;
                }
                v513 += 1 ;
            }
            auto v521 = cooperative_groups::coalesced_threads();
            int v522;
            v522 = threadIdx.x;
            int v523;
            v523 = v522 / 8;
            auto v524 = cooperative_groups::labeled_partition(v521,v523);
            Closure0 v525{};
            float v526;
            v526 = cooperative_groups::reduce(v524, v512, v525);
            float v527;
            v527 = v526 / 32.0f;
            bool v528[4];
            int v529;
            v529 = 0;
            while (while_method_4(v529)){
                int v531;
                v531 = 0;
                while (while_method_7(v531)){
                    assert("Tensor range check" && 0 <= v529 && v529 < 1);
                    assert("Tensor range check" && 0 <= v531 && v531 < 4);
                    int v533;
                    v533 = 4 * v529;
                    int v534;
                    v534 = v533 + v531;
                    float v535;
                    v535 = v498[v534];
                    bool v536;
                    v536 = v535 >= v527;
                    assert("Tensor range check" && 0 <= v529 && v529 < 1);
                    assert("Tensor range check" && 0 <= v531 && v531 < 4);
                    v528[v534] = v536;
                    v531 += 1 ;
                }
                v529 += 1 ;
            }
            int v537[4];
            int v538;
            v538 = 0;
            while (while_method_4(v538)){
                int v540;
                v540 = 0;
                while (while_method_7(v540)){
                    assert("Tensor range check" && 0 <= v538 && v538 < 1);
                    assert("Tensor range check" && 0 <= v540 && v540 < 4);
                    int v542;
                    v542 = 4 * v538;
                    int v543;
                    v543 = v542 + v540;
                    bool v544;
                    v544 = v528[v543];
                    int v545;
                    if (v544){
                        v545 = 1;
                    } else {
                        v545 = 0;
                    }
                    assert("Tensor range check" && 0 <= v538 && v538 < 1);
                    assert("Tensor range check" && 0 <= v540 && v540 < 4);
                    v537[v543] = v545;
                    v540 += 1 ;
                }
                v538 += 1 ;
            }
            int v546;
            v546 = 0;
            int v547;
            v547 = 0;
            while (while_method_4(v547)){
                int v549;
                v549 = 0;
                while (while_method_7(v549)){
                    assert("Tensor range check" && 0 <= v547 && v547 < 1);
                    assert("Tensor range check" && 0 <= v549 && v549 < 4);
                    int v551;
                    v551 = 4 * v547;
                    int v552;
                    v552 = v551 + v549;
                    int v553;
                    v553 = v537[v552];
                    int v554;
                    v554 = v546 + v553;
                    v546 = v554;
                    v549 += 1 ;
                }
                v547 += 1 ;
            }
            auto v555 = cooperative_groups::coalesced_threads();
            int v556;
            v556 = threadIdx.x;
            int v557;
            v557 = v556 / 8;
            auto v558 = cooperative_groups::labeled_partition(v555,v557);
            Closure1 v559{};
            int v560;
            v560 = cooperative_groups::reduce(v558, v546, v559);
            float v561;
            v561 = (float)v560;
            float v562[4];
            int v563;
            v563 = 0;
            while (while_method_4(v563)){
                int v565;
                v565 = 0;
                while (while_method_7(v565)){
                    assert("Tensor range check" && 0 <= v563 && v563 < 1);
                    assert("Tensor range check" && 0 <= v565 && v565 < 4);
                    int v567;
                    v567 = 4 * v563;
                    int v568;
                    v568 = v567 + v565;
                    float v569;
                    v569 = v443[v568];
                    bool v570;
                    v570 = v528[v568];
                    float v571;
                    if (v570){
                        v571 = v569;
                    } else {
                        v571 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v563 && v563 < 1);
                    assert("Tensor range check" && 0 <= v565 && v565 < 4);
                    v562[v568] = v571;
                    v565 += 1 ;
                }
                v563 += 1 ;
            }
            float v572;
            v572 = 0.0f;
            int v573;
            v573 = 0;
            while (while_method_4(v573)){
                int v575;
                v575 = 0;
                while (while_method_7(v575)){
                    assert("Tensor range check" && 0 <= v573 && v573 < 1);
                    assert("Tensor range check" && 0 <= v575 && v575 < 4);
                    int v577;
                    v577 = 4 * v573;
                    int v578;
                    v578 = v577 + v575;
                    float v579;
                    v579 = v562[v578];
                    float v580;
                    v580 = v572 + v579;
                    v572 = v580;
                    v575 += 1 ;
                }
                v573 += 1 ;
            }
            auto v581 = cooperative_groups::coalesced_threads();
            int v582;
            v582 = threadIdx.x;
            int v583;
            v583 = v582 / 8;
            auto v584 = cooperative_groups::labeled_partition(v581,v583);
            float v585;
            v585 = cooperative_groups::reduce(v584, v572, v525);
            float v586;
            v586 = v585 / v561;
            float v587[4];
            int v588;
            v588 = 0;
            while (while_method_4(v588)){
                int v590;
                v590 = 0;
                while (while_method_7(v590)){
                    assert("Tensor range check" && 0 <= v588 && v588 < 1);
                    assert("Tensor range check" && 0 <= v590 && v590 < 4);
                    int v592;
                    v592 = 4 * v588;
                    int v593;
                    v593 = v592 + v590;
                    float v594;
                    v594 = v443[v593];
                    float v595;
                    v595 = v594 - v586;
                    float v596;
                    v596 = v595 * v595;
                    assert("Tensor range check" && 0 <= v588 && v588 < 1);
                    assert("Tensor range check" && 0 <= v590 && v590 < 4);
                    v587[v593] = v596;
                    v590 += 1 ;
                }
                v588 += 1 ;
            }
            float v597[4];
            int v598;
            v598 = 0;
            while (while_method_4(v598)){
                int v600;
                v600 = 0;
                while (while_method_7(v600)){
                    assert("Tensor range check" && 0 <= v598 && v598 < 1);
                    assert("Tensor range check" && 0 <= v600 && v600 < 4);
                    int v602;
                    v602 = 4 * v598;
                    int v603;
                    v603 = v602 + v600;
                    float v604;
                    v604 = v587[v603];
                    bool v605;
                    v605 = v528[v603];
                    float v606;
                    if (v605){
                        v606 = v604;
                    } else {
                        v606 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v598 && v598 < 1);
                    assert("Tensor range check" && 0 <= v600 && v600 < 4);
                    v597[v603] = v606;
                    v600 += 1 ;
                }
                v598 += 1 ;
            }
            float v607;
            v607 = 0.0f;
            int v608;
            v608 = 0;
            while (while_method_4(v608)){
                int v610;
                v610 = 0;
                while (while_method_7(v610)){
                    assert("Tensor range check" && 0 <= v608 && v608 < 1);
                    assert("Tensor range check" && 0 <= v610 && v610 < 4);
                    int v612;
                    v612 = 4 * v608;
                    int v613;
                    v613 = v612 + v610;
                    float v614;
                    v614 = v597[v613];
                    float v615;
                    v615 = v607 + v614;
                    v607 = v615;
                    v610 += 1 ;
                }
                v608 += 1 ;
            }
            auto v616 = cooperative_groups::coalesced_threads();
            int v617;
            v617 = threadIdx.x;
            int v618;
            v618 = v617 / 8;
            auto v619 = cooperative_groups::labeled_partition(v616,v618);
            float v620;
            v620 = cooperative_groups::reduce(v619, v607, v525);
            float v621;
            v621 = v620 / v561;
            float v622;
            v622 = sqrt(v621);
            bool v623;
            v623 = v561 > 1.0f;
            float v627;
            if (v623){
                float v624;
                v624 = v622 * v561;
                float v625;
                v625 = v561 - 1.0f;
                float v626;
                v626 = v624 / v625;
                v627 = v626;
            } else {
                v627 = 0.0f;
            }
            float v628[4];
            int v629;
            v629 = 0;
            while (while_method_4(v629)){
                int v631;
                v631 = 0;
                while (while_method_7(v631)){
                    assert("Tensor range check" && 0 <= v629 && v629 < 1);
                    assert("Tensor range check" && 0 <= v631 && v631 < 4);
                    int v633;
                    v633 = 4 * v629;
                    int v634;
                    v634 = v633 + v631;
                    float v635;
                    v635 = v443[v634];
                    bool v636;
                    v636 = v528[v634];
                    float v637;
                    v637 = curand_normal(&v333);
                    float v638;
                    v638 = v627 + 0.3f;
                    float v639;
                    v639 = v637 * v638;
                    float v640;
                    v640 = v639 + v586;
                    float v641;
                    if (v636){
                        v641 = v635;
                    } else {
                        v641 = v640;
                    }
                    assert("Tensor range check" && 0 <= v629 && v629 < 1);
                    assert("Tensor range check" && 0 <= v631 && v631 < 4);
                    v628[v634] = v641;
                    v631 += 1 ;
                }
                v629 += 1 ;
            }
            assert("Tensor range check" && 0 <= v435 && v435 < 64);
            assert("Tensor range check" && 0 <= v434 && v434 < 4);
            int v642;
            v642 = 0;
            while (while_method_4(v642)){
                assert("Tensor range check" && 0 <= v642 && v642 < 1);
                int v644;
                v644 = 32 * v642;
                int v645;
                v645 = v644 + v442;
                assert("Tensor range check" && 0 <= v642 && v642 < 1);
                int v646;
                v646 = 4 * v642;
                int4* v647;
                v647 = reinterpret_cast<int4*>(v628 + v646);
                int4* v648;
                v648 = reinterpret_cast<int4*>(v336 + v645);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v647) % 16 == 0 && reinterpret_cast<unsigned long long>(v648) % 16 == 0);
                *v648 = *v647;
                v642 += 1 ;
            }
            v429 += 24 ;
        }
        v331.sync() ;
        extern __shared__ unsigned char v649[];
        float * v650;
        v650 = reinterpret_cast<float *>(&v649[0ull]);
        int v652;
        v652 = blockIdx.x;
        int v653;
        v653 = v652;
        while (while_method_8(v653)){
            bool v655;
            v655 = 0 <= v653;
            bool v656;
            v656 = v655 == false;
            if (v656){
                assert("The index needs to be zero or positive." && v655);
            } else {
            }
            int v658;
            v658 = v653 % 1;
            bool v659;
            v659 = v653 < 16;
            bool v660;
            v660 = v659 == false;
            if (v660){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v659);
            } else {
            }
            assert("Tensor range check" && 0 <= v653 && v653 < 16);
            assert("Tensor range check" && 0 <= v658 && v658 < 1);
            int v662;
            v662 = 32 * v658;
            int v663;
            v663 = 16384 * v653;
            int v664;
            v664 = v663 + v662;
            int v665;
            v665 = 262144 * v658;
            int v666;
            v666 = 512 * v653;
            int v667;
            v667 = v666 + v665;
            int v668;
            v668 = threadIdx.x;
            int v669;
            v669 = v668;
            while (while_method_12(v669)){
                bool v671;
                v671 = 0 <= v669;
                bool v672;
                v672 = v671 == false;
                if (v672){
                    assert("The index needs to be zero or positive." && v671);
                } else {
                }
                int v674;
                v674 = v669 % 32;
                int v675;
                v675 = v669 / 32;
                bool v676;
                v676 = v675 < 512;
                bool v677;
                v677 = v676 == false;
                if (v677){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v676);
                } else {
                }
                assert("Tensor range check" && 0 <= v675 && v675 < 512);
                assert("Tensor range check" && 0 <= v674 && v674 < 32);
                int v679;
                v679 = v674 + v664;
                int v680;
                v680 = 32 * v675;
                int v681;
                v681 = v680 + v679;
                float v682;
                v682 = v336[v681];
                assert("Tensor range check" && 0 <= v675 && v675 < 512);
                assert("Tensor range check" && 0 <= v674 && v674 < 32);
                int v683;
                v683 = 33 * v675;
                int v684;
                v684 = v683 + v674;
                v650[v684] = v682;
                v669 += 256 ;
            }
            __syncthreads();
            int v685;
            v685 = threadIdx.x;
            int v686;
            v686 = v685;
            while (while_method_12(v686)){
                bool v688;
                v688 = 0 <= v686;
                bool v689;
                v689 = v688 == false;
                if (v689){
                    assert("The index needs to be zero or positive." && v688);
                } else {
                }
                int v691;
                v691 = v686 % 512;
                int v692;
                v692 = v686 / 512;
                bool v693;
                v693 = v692 < 32;
                bool v694;
                v694 = v693 == false;
                if (v694){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v693);
                } else {
                }
                assert("Tensor range check" && 0 <= v692 && v692 < 32);
                assert("Tensor range check" && 0 <= v691 && v691 < 512);
                int v696;
                v696 = 33 * v691;
                int v697;
                v697 = v692 + v696;
                float v698;
                v698 = v650[v697];
                assert("Tensor range check" && 0 <= v692 && v692 < 32);
                assert("Tensor range check" && 0 <= v691 && v691 < 512);
                int v699;
                v699 = v691 + v667;
                int v700;
                v700 = 8192 * v692;
                int v701;
                v701 = v700 + v699;
                v334[v701] = v698;
                v686 += 256 ;
            }
            __syncthreads();
            v653 += 24 ;
        }
        int v702;
        v702 = threadIdx.x;
        int v703;
        v703 = blockIdx.x;
        int v704;
        v704 = v703 * 256;
        int v705;
        v705 = v702 + v704;
        int v706;
        v706 = v705;
        while (while_method_5(v706)){
            bool v708;
            v708 = 0 <= v706;
            bool v709;
            v709 = v708 == false;
            if (v709){
                assert("The index needs to be zero or positive." && v708);
            } else {
            }
            bool v711;
            v711 = v706 < 8;
            bool v712;
            v712 = v711 == false;
            if (v712){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v711);
            } else {
            }
            assert("Tensor range check" && 0 <= v706 && v706 < 8);
            int v714;
            v714 = 4 * v706;
            assert("Tensor range check" && 0 <= v706 && v706 < 8);
            float v715[4];
            float v716[4];
            float v717[4];
            float v718[4];
            int4* v719;
            v719 = reinterpret_cast<int4*>(v342 + v714);
            int4* v720;
            v720 = reinterpret_cast<int4*>(v715 + 0);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v719) % 16 == 0 && reinterpret_cast<unsigned long long>(v720) % 16 == 0);
            *v720 = *v719;
            int4* v721;
            v721 = reinterpret_cast<int4*>(v344 + v714);
            int4* v722;
            v722 = reinterpret_cast<int4*>(v716 + 0);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v721) % 16 == 0 && reinterpret_cast<unsigned long long>(v722) % 16 == 0);
            *v722 = *v721;
            // Pushing the loop unrolling to: 0
            int v723;
            v723 = 0;
            #pragma unroll
            while (while_method_7(v723)){
                assert("Tensor range check" && 0 <= v723 && v723 < 4);
                float v725;
                v725 = v715[v723];
                float v726;
                v726 = v716[v723];
                assert("Tensor range check" && 0 <= v723 && v723 < 4);
                v717[v723] = 0.0f;
                v718[v723] = 0.0f;
                v723 += 1 ;
            }
            // Poping the loop unrolling to: 0
            int4* v727;
            v727 = reinterpret_cast<int4*>(v717 + 0);
            int4* v728;
            v728 = reinterpret_cast<int4*>(v342 + v714);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v727) % 16 == 0 && reinterpret_cast<unsigned long long>(v728) % 16 == 0);
            *v728 = *v727;
            int4* v729;
            v729 = reinterpret_cast<int4*>(v718 + 0);
            int4* v730;
            v730 = reinterpret_cast<int4*>(v344 + v714);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v729) % 16 == 0 && reinterpret_cast<unsigned long long>(v730) % 16 == 0);
            *v730 = *v729;
            v706 += 6144 ;
        }
        v331.sync() ;
        v27 += 1 ;
    }
    cooperative_groups::grid_group & v731 = v26.v1;
    cooperative_groups::grid_group & v732 = v731;
    int v733;
    v733 = threadIdx.x;
    int v734;
    v734 = blockIdx.x;
    int v735;
    v735 = v734 * 256;
    int v736;
    v736 = v733 + v735;
    int v737;
    v737 = v736;
    while (while_method_13(v737)){
        bool v739;
        v739 = 0 <= v737;
        bool v740;
        v740 = v739 == false;
        if (v740){
            assert("The index needs to be zero or positive." && v739);
        } else {
        }
        int v742;
        v742 = v737 % 8;
        int v743;
        v743 = v737 / 8;
        bool v744;
        v744 = v743 < 32;
        bool v745;
        v745 = v744 == false;
        if (v745){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v744);
        } else {
        }
        assert("Tensor range check" && 0 <= v743 && v743 < 32);
        assert("Tensor range check" && 0 <= v742 && v742 < 8);
        int v747;
        v747 = 4 * v742;
        int v748;
        v748 = 32 * v743;
        int v749;
        v749 = v748 + v747;
        assert("Tensor range check" && 0 <= v743 && v743 < 32);
        assert("Tensor range check" && 0 <= v742 && v742 < 8);
        float v750[4];
        float v751[4];
        float v752[4];
        int4* v753;
        v753 = reinterpret_cast<int4*>(v3 + v749);
        int4* v754;
        v754 = reinterpret_cast<int4*>(v750 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v753) % 16 == 0 && reinterpret_cast<unsigned long long>(v754) % 16 == 0);
        *v754 = *v753;
        int4* v755;
        v755 = reinterpret_cast<int4*>(v4 + v749);
        int4* v756;
        v756 = reinterpret_cast<int4*>(v751 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v755) % 16 == 0 && reinterpret_cast<unsigned long long>(v756) % 16 == 0);
        *v756 = *v755;
        // Pushing the loop unrolling to: 0
        int v757;
        v757 = 0;
        #pragma unroll
        while (while_method_7(v757)){
            assert("Tensor range check" && 0 <= v757 && v757 < 4);
            float v759;
            v759 = v750[v757];
            float v760;
            v760 = v751[v757];
            bool v761;
            v761 = v760 == 0.0f;
            bool v762;
            v762 = v761 != true;
            float v764;
            if (v762){
                float v763;
                v763 = v759 / v760;
                v764 = v763;
            } else {
                v764 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v757 && v757 < 4);
            v752[v757] = v764;
            v757 += 1 ;
        }
        // Poping the loop unrolling to: 0
        int4* v765;
        v765 = reinterpret_cast<int4*>(v752 + 0);
        int4* v766;
        v766 = reinterpret_cast<int4*>(v5 + v749);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v765) % 16 == 0 && reinterpret_cast<unsigned long long>(v766) % 16 == 0);
        *v766 = *v765;
        v737 += 6144 ;
    }
    v732.sync() ;
    return ;
}
extern "C" __global__ void entry2(unsigned char * v0, unsigned char * v1, unsigned char * v2, float * v3, float * v4, float * v5) {
    auto v6 = cooperative_groups::this_grid();
    unsigned long long v7;
    v7 = clock64();
    int v8;
    v8 = threadIdx.x;
    int v9;
    v9 = blockIdx.x;
    int v10;
    v10 = v9 * 256;
    int v11;
    v11 = v8 + v10;
    unsigned long long v12;
    v12 = (unsigned long long)v11;
    curandStatePhilox4_32_10_t v13;
    curand_init(v7,v12,0ull,&v13);
    static_array<Union2,2> v14;
    Union2 v16;
    v16 = Union2{Union2_2{}};
    v14[0] = v16;
    Union2 v18;
    v18 = Union2{Union2_2{}};
    v14[1] = v18;
    static_array_list<Union7,32> v20;
    v20 = static_array_list<Union7,32>{};
    static_array<float,2> v22;
    v22[0] = 0.0f;
    v22[1] = 0.0f;
    cooperative_groups::grid_group & v24 = v6;
    curandStatePhilox4_32_10_t & v25 = v13;
    StackMut1 v26{63u, v24, v20, v14, v22, v25};
    int v27;
    v27 = 0;
    while (while_method_10(v27)){
        int v29;
        v29 = 0;
        while (while_method_5(v29)){
            Union4 v31;
            v31 = Union4{Union4_1{}};
            method_48(v0, v1, v2, v26, v31);
            static_array<float,2> & v32 = v26.v4;
            float * v33;
            v33 = reinterpret_cast<float *>(&v1[55050240ull]);
            int * v35;
            v35 = reinterpret_cast<int *>(&v0[1048576ull]);
            float * v37;
            v37 = reinterpret_cast<float *>(&v0[1048592ull]);
            float * v39;
            v39 = reinterpret_cast<float *>(&v0[1048720ull]);
            double * v41;
            v41 = reinterpret_cast<double *>(&v1[105381888ull]);
            double * v43;
            v43 = reinterpret_cast<double *>(&v1[108527616ull]);
            int v45;
            v45 = threadIdx.x;
            int v46;
            v46 = blockIdx.x;
            int v47;
            v47 = v46 * 256;
            int v48;
            v48 = v45 + v47;
            assert("Tensor range check" && 0 <= v48 && v48 < 6144);
            int v49;
            v49 = 2 * v48;
            double * v50;
            v50 = v41+v49;
            double * v52;
            v52 = v43+v49;
            float v54[2];
            int v55;
            v55 = 0;
            while (while_method_0(v55)){
                bool v57;
                v57 = 0 <= v55;
                bool v59;
                if (v57){
                    bool v58;
                    v58 = v55 < 2;
                    v59 = v58;
                } else {
                    v59 = false;
                }
                bool v60;
                v60 = v59 == false;
                if (v60){
                    assert("Index must be in range." && v59);
                } else {
                }
                float v62;
                v62 = v32[v55];
                assert("Tensor range check" && 0 <= v55 && v55 < 2);
                v54[v55] = v62;
                v55 += 1 ;
            }
            double v64[2];
            int v65;
            v65 = 0;
            while (while_method_0(v65)){
                int v67; double v68;
                Tuple12 tmp76 = Tuple12{0, 0.0};
                v67 = tmp76.v0; v68 = tmp76.v1;
                while (while_method_10(v67)){
                    assert("Tensor range check" && 0 <= v67 && v67 < 32);
                    assert("Tensor range check" && 0 <= v65 && v65 < 2);
                    int v70;
                    v70 = 12288 * v67;
                    int v71;
                    v71 = v70 + v65;
                    double v72;
                    v72 = v50[v71];
                    double v73;
                    v73 = v52[v71];
                    double v74;
                    v74 = v72 - v73;
                    double v75;
                    v75 = exp(v74);
                    double v76;
                    v76 = v68 + v75;
                    v68 = v76;
                    v67 += 1 ;
                }
                assert("Tensor range check" && 0 <= v65 && v65 < 2);
                v64[v65] = v68;
                v65 += 1 ;
            }
            double v77;
            v77 = 1.0;
            int v78;
            v78 = 0;
            while (while_method_0(v78)){
                assert("Tensor range check" && 0 <= v78 && v78 < 2);
                double v80;
                v80 = v64[v78];
                double v81;
                v81 = v77 * v80;
                v77 = v81;
                v78 += 1 ;
            }
            double v82[64];
            int v83;
            v83 = 0;
            while (while_method_10(v83)){
                int v85;
                v85 = 0;
                while (while_method_0(v85)){
                    assert("Tensor range check" && 0 <= v85 && v85 < 2);
                    double v87;
                    v87 = v64[v85];
                    double v88;
                    v88 = v77 / v87;
                    assert("Tensor range check" && 0 <= v83 && v83 < 32);
                    assert("Tensor range check" && 0 <= v85 && v85 < 2);
                    int v89;
                    v89 = 12288 * v83;
                    int v90;
                    v90 = v89 + v85;
                    double v91;
                    v91 = v50[v90];
                    double v92;
                    v92 = v52[v90];
                    double v93;
                    v93 = v91 - v92;
                    double v94;
                    v94 = exp(v93);
                    double v95;
                    v95 = v88 * v94;
                    assert("Tensor range check" && 0 <= v83 && v83 < 32);
                    assert("Tensor range check" && 0 <= v85 && v85 < 2);
                    int v96;
                    v96 = 2 * v83;
                    int v97;
                    v97 = v96 + v85;
                    v82[v97] = v95;
                    v85 += 1 ;
                }
                v83 += 1 ;
            }
            float v98[32];
            float v99[32];
            int v100;
            v100 = 0;
            while (while_method_10(v100)){
                int v102; float v103; double v104;
                Tuple13 tmp77 = Tuple13{0, 0.0f, 0.0};
                v102 = tmp77.v0; v103 = tmp77.v1; v104 = tmp77.v2;
                while (while_method_0(v102)){
                    assert("Tensor range check" && 0 <= v100 && v100 < 32);
                    assert("Tensor range check" && 0 <= v102 && v102 < 2);
                    int v106;
                    v106 = 2 * v100;
                    int v107;
                    v107 = v106 + v102;
                    double v108;
                    v108 = v82[v107];
                    assert("Tensor range check" && 0 <= v102 && v102 < 2);
                    float v109;
                    v109 = v54[v102];
                    float v110;
                    v110 = (float)v108;
                    float v111;
                    v111 = v110 * v109;
                    float v112;
                    v112 = v103 + v111;
                    double v113;
                    v113 = v104 + v108;
                    v103 = v112;
                    v104 = v113;
                    v102 += 1 ;
                }
                float v114;
                v114 = (float)v104;
                assert("Tensor range check" && 0 <= v100 && v100 < 32);
                v98[v100] = v103;
                v99[v100] = v114;
                v100 += 1 ;
            }
            int v115;
            v115 = 0;
            while (while_method_10(v115)){
                assert("Tensor range check" && 0 <= v115 && v115 < 32);
                float v117;
                v117 = v98[v115];
                float v118;
                v118 = v99[v115];
                float v119;
                v119 = v117 * v118;
                assert("Tensor range check" && 0 <= v115 && v115 < 32);
                float * v120;
                v120 = v37+v115;
                float * v122;
                v122 = v39+v115;
                float v124;
                v124 = atomicAdd(v120,v119);
                float v125;
                v125 = atomicAdd(v122,v118);
                v115 += 1 ;
            }
            int v126;
            v126 = threadIdx.x;
            int v127;
            v127 = blockIdx.x;
            int v128;
            v128 = v127 * 256;
            int v129;
            v129 = v126 + v128;
            int v130;
            v130 = 0;
            while (while_method_10(v130)){
                assert("Tensor range check" && 0 <= v130 && v130 < 32);
                int v132;
                v132 = 12288 * v130;
                assert("Tensor range check" && 0 <= v129 && v129 < 6144);
                int v133;
                v133 = 2 * v129;
                int v134;
                v134 = v133 + v132;
                double * v135;
                v135 = v41+v134;
                double * v137;
                v137 = v43+v134;
                double * v139;
                v139 = v41+v134;
                double * v141;
                v141 = v43+v134;
                int v143;
                v143 = sizeof(double *);
                unsigned long long v144;
                v144 = (unsigned long long)v143;
                unsigned long long v145;
                v145 = 256ull * v144;
                unsigned long long v146;
                v146 = v145 + 16ull;
                unsigned long long v147;
                v147 = v146 - 1ull;
                unsigned long long v148;
                v148 = v147 % 16ull;
                unsigned long long v149;
                v149 = v147 - v148;
                unsigned long long v150;
                v150 = v149 + v145;
                unsigned long long v151;
                v151 = v150 + 16ull;
                unsigned long long v152;
                v152 = v151 - 1ull;
                unsigned long long v153;
                v153 = v152 % 16ull;
                unsigned long long v154;
                v154 = v152 - v153;
                unsigned long long v155;
                v155 = v154 + v145;
                unsigned long long v156;
                v156 = v155 + 16ull;
                unsigned long long v157;
                v157 = v156 - 1ull;
                unsigned long long v158;
                v158 = v157 % 16ull;
                unsigned long long v159;
                v159 = v157 - v158;
                unsigned long long v160;
                v160 = v159 + v145;
                bool v161;
                v161 = v160 <= 98304ull;
                bool v162;
                v162 = v161 == false;
                if (v162){
                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v161);
                } else {
                }
                extern __shared__ unsigned char v164[];
                bool v165;
                v165 = v160 <= v160;
                bool v166;
                v166 = v165 == false;
                if (v166){
                    assert("The length of the partition has to be less than or equal to the length of the base array." && v165);
                } else {
                }
                double * * v168;
                v168 = reinterpret_cast<double * *>(&v164[0ull]);
                double * * v170;
                v170 = reinterpret_cast<double * *>(&v164[v149]);
                double * * v172;
                v172 = reinterpret_cast<double * *>(&v164[v154]);
                double * * v174;
                v174 = reinterpret_cast<double * *>(&v164[v159]);
                int v176;
                v176 = threadIdx.x;
                assert("Tensor range check" && 0 <= v176 && v176 < 256);
                v168[v176] = v135;
                v170[v176] = v137;
                v172[v176] = v139;
                v174[v176] = v141;
                __syncthreads();
                bool v177;
                v177 = 0 <= v176;
                bool v178;
                v178 = v177 == false;
                if (v178){
                    assert("The index needs to be zero or positive." && v177);
                } else {
                }
                int v180;
                v180 = v176 % 1;
                bool v181;
                v181 = v176 < 256;
                bool v182;
                v182 = v181 == false;
                if (v182){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v181);
                } else {
                }
                assert("Tensor range check" && 0 <= v176 && v176 < 256);
                int v184;
                v184 = 0;
                while (while_method_4(v184)){
                    bool v186;
                    v186 = v177 && v181;
                    bool v187;
                    v187 = v186 == false;
                    if (v187){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v186);
                    } else {
                    }
                    bool v189;
                    v189 = 0 <= v184;
                    bool v191;
                    if (v189){
                        bool v190;
                        v190 = v184 < 1;
                        v191 = v190;
                    } else {
                        v191 = false;
                    }
                    bool v192;
                    v192 = v191 == false;
                    if (v192){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v191);
                    } else {
                    }
                    int v194;
                    v194 = v184 * 256;
                    int v195;
                    v195 = v194 + v176;
                    assert("Tensor range check" && 0 <= v184 && v184 < 1);
                    int v196;
                    v196 = 256 * v184;
                    int v197;
                    v197 = v196 + v176;
                    double * v198;
                    v198 = v168[v197];
                    double * v199;
                    v199 = v170[v197];
                    double * v200;
                    v200 = v172[v197];
                    double * v201;
                    v201 = v174[v197];
                    int v202;
                    v202 = blockIdx.x;
                    int v203;
                    v203 = v202 * 256;
                    int v204;
                    v204 = v203 + v195;
                    assert("Tensor range check" && 0 <= v180 && v180 < 1);
                    int v205;
                    v205 = 2 * v180;
                    double v206[2];
                    double v207[2];
                    int v208[2];
                    int v209;
                    v209 = 0;
                    while (while_method_4(v209)){
                        assert("Tensor range check" && 0 <= v209 && v209 < 1);
                        int v211;
                        v211 = 2 * v209;
                        assert("Tensor range check" && 0 <= v209 && v209 < 1);
                        int v212;
                        v212 = v211 + v205;
                        int4* v213;
                        v213 = reinterpret_cast<int4*>(v198 + v212);
                        int4* v214;
                        v214 = reinterpret_cast<int4*>(v206 + v211);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v213) % 16 == 0 && reinterpret_cast<unsigned long long>(v214) % 16 == 0);
                        *v214 = *v213;
                        int4* v215;
                        v215 = reinterpret_cast<int4*>(v199 + v212);
                        int4* v216;
                        v216 = reinterpret_cast<int4*>(v207 + v211);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v215) % 16 == 0 && reinterpret_cast<unsigned long long>(v216) % 16 == 0);
                        *v216 = *v215;
                        v209 += 1 ;
                    }
                    int v217;
                    v217 = 0;
                    while (while_method_4(v217)){
                        int v219;
                        v219 = 0;
                        while (while_method_0(v219)){
                            bool v221;
                            v221 = 0 <= v219;
                            bool v223;
                            if (v221){
                                bool v222;
                                v222 = v219 < 2;
                                v223 = v222;
                            } else {
                                v223 = false;
                            }
                            bool v224;
                            v224 = v223 == false;
                            if (v224){
                                assert("The indices should be inside the range of the dimension." && v223);
                            } else {
                            }
                            bool v226;
                            v226 = 0 <= v180;
                            bool v228;
                            if (v226){
                                bool v227;
                                v227 = v180 < 1;
                                v228 = v227;
                            } else {
                                v228 = false;
                            }
                            bool v229;
                            v229 = v228 == false;
                            if (v229){
                                assert("The indices should be inside the range of the dimension." && v228);
                            } else {
                            }
                            int v231;
                            v231 = v180 * 2;
                            int v232;
                            v232 = v219 + v231;
                            bool v233;
                            v233 = 0 <= v217;
                            bool v235;
                            if (v233){
                                bool v234;
                                v234 = v217 < 1;
                                v235 = v234;
                            } else {
                                v235 = false;
                            }
                            bool v236;
                            v236 = v235 == false;
                            if (v236){
                                assert("The indices should be inside the range of the dimension." && v235);
                            } else {
                            }
                            int v238;
                            v238 = v217 * 2;
                            int v239;
                            v239 = v232 + v238;
                            assert("Tensor range check" && 0 <= v217 && v217 < 1);
                            assert("Tensor range check" && 0 <= v219 && v219 < 2);
                            int v240;
                            v240 = 2 * v217;
                            int v241;
                            v241 = v240 + v219;
                            v208[v241] = v239;
                            v219 += 1 ;
                        }
                        v217 += 1 ;
                    }
                    double v242[2];
                    double v243[2];
                    int v244;
                    v244 = 0;
                    while (while_method_4(v244)){
                        int v246;
                        v246 = 0;
                        while (while_method_0(v246)){
                            assert("Tensor range check" && 0 <= v244 && v244 < 1);
                            assert("Tensor range check" && 0 <= v246 && v246 < 2);
                            int v248;
                            v248 = 2 * v244;
                            int v249;
                            v249 = v248 + v246;
                            double v250;
                            v250 = v206[v249];
                            double v251;
                            v251 = v207[v249];
                            assert("Tensor range check" && 0 <= v244 && v244 < 1);
                            assert("Tensor range check" && 0 <= v246 && v246 < 2);
                            v242[v249] = 0.0;
                            v243[v249] = 0.0;
                            v246 += 1 ;
                        }
                        v244 += 1 ;
                    }
                    int v252;
                    v252 = 0;
                    while (while_method_4(v252)){
                        assert("Tensor range check" && 0 <= v252 && v252 < 1);
                        int v254;
                        v254 = 2 * v252;
                        int v255;
                        v255 = v254 + v205;
                        assert("Tensor range check" && 0 <= v252 && v252 < 1);
                        int4* v256;
                        v256 = reinterpret_cast<int4*>(v242 + v254);
                        int4* v257;
                        v257 = reinterpret_cast<int4*>(v200 + v255);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v256) % 16 == 0 && reinterpret_cast<unsigned long long>(v257) % 16 == 0);
                        *v257 = *v256;
                        int4* v258;
                        v258 = reinterpret_cast<int4*>(v243 + v254);
                        int4* v259;
                        v259 = reinterpret_cast<int4*>(v201 + v255);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v258) % 16 == 0 && reinterpret_cast<unsigned long long>(v259) % 16 == 0);
                        *v259 = *v258;
                        v252 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v195 && v195 < 256);
                    v184 += 1 ;
                }
                __syncthreads();
                assert("Tensor range check" && 0 <= v176 && v176 < 256);
                __syncthreads();
                v130 += 1 ;
            }
            Union4 v260;
            v260 = Union4{Union4_1{}};
            method_49(v0, v1, v2, v26, v260);
            double * v261;
            v261 = reinterpret_cast<double *>(&v1[105381888ull]);
            double * v263;
            v263 = reinterpret_cast<double *>(&v1[108527616ull]);
            int v265;
            v265 = threadIdx.x;
            int v266;
            v266 = blockIdx.x;
            int v267;
            v267 = v266 * 256;
            int v268;
            v268 = v265 + v267;
            assert("Tensor range check" && 0 <= v268 && v268 < 6144);
            int v269;
            v269 = 2 * v268;
            static_array<float,2> & v270 = v26.v4;
            float v271[2];
            int v272;
            v272 = 0;
            while (while_method_0(v272)){
                bool v274;
                v274 = 0 <= v272;
                bool v276;
                if (v274){
                    bool v275;
                    v275 = v272 < 2;
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
                float v279;
                v279 = v270[v272];
                assert("Tensor range check" && 0 <= v272 && v272 < 2);
                v271[v272] = v279;
                v272 += 1 ;
            }
            double v281[2];
            int v282;
            v282 = 0;
            while (while_method_0(v282)){
                int v284; double v285;
                Tuple12 tmp96 = Tuple12{0, 0.0};
                v284 = tmp96.v0; v285 = tmp96.v1;
                while (while_method_10(v284)){
                    assert("Tensor range check" && 0 <= v284 && v284 < 32);
                    assert("Tensor range check" && 0 <= v282 && v282 < 2);
                    int v287;
                    v287 = v282 + v269;
                    int v288;
                    v288 = 12288 * v284;
                    int v289;
                    v289 = v288 + v287;
                    double v290;
                    v290 = v261[v289];
                    double v291;
                    v291 = v263[v289];
                    double v292;
                    v292 = v290 - v291;
                    double v293;
                    v293 = exp(v292);
                    double v294;
                    v294 = v285 + v293;
                    v285 = v294;
                    v284 += 1 ;
                }
                assert("Tensor range check" && 0 <= v282 && v282 < 2);
                v281[v282] = v285;
                v282 += 1 ;
            }
            double v295;
            v295 = 1.0;
            int v296;
            v296 = 0;
            while (while_method_0(v296)){
                assert("Tensor range check" && 0 <= v296 && v296 < 2);
                double v298;
                v298 = v281[v296];
                double v299;
                v299 = v295 * v298;
                v295 = v299;
                v296 += 1 ;
            }
            double v300[64];
            int v301;
            v301 = 0;
            while (while_method_10(v301)){
                int v303;
                v303 = 0;
                while (while_method_0(v303)){
                    assert("Tensor range check" && 0 <= v303 && v303 < 2);
                    double v305;
                    v305 = v281[v303];
                    double v306;
                    v306 = v295 / v305;
                    assert("Tensor range check" && 0 <= v301 && v301 < 32);
                    assert("Tensor range check" && 0 <= v303 && v303 < 2);
                    int v307;
                    v307 = v303 + v269;
                    int v308;
                    v308 = 12288 * v301;
                    int v309;
                    v309 = v308 + v307;
                    double v310;
                    v310 = v261[v309];
                    double v311;
                    v311 = v263[v309];
                    double v312;
                    v312 = v310 - v311;
                    double v313;
                    v313 = exp(v312);
                    double v314;
                    v314 = v306 * v313;
                    assert("Tensor range check" && 0 <= v301 && v301 < 32);
                    assert("Tensor range check" && 0 <= v303 && v303 < 2);
                    int v315;
                    v315 = 2 * v301;
                    int v316;
                    v316 = v315 + v303;
                    v300[v316] = v314;
                    v303 += 1 ;
                }
                v301 += 1 ;
            }
            float v317[32];
            float v318[32];
            int v319;
            v319 = 0;
            while (while_method_10(v319)){
                int v321; float v322; double v323;
                Tuple13 tmp97 = Tuple13{0, 0.0f, 0.0};
                v321 = tmp97.v0; v322 = tmp97.v1; v323 = tmp97.v2;
                while (while_method_0(v321)){
                    assert("Tensor range check" && 0 <= v319 && v319 < 32);
                    assert("Tensor range check" && 0 <= v321 && v321 < 2);
                    int v325;
                    v325 = 2 * v319;
                    int v326;
                    v326 = v325 + v321;
                    double v327;
                    v327 = v300[v326];
                    assert("Tensor range check" && 0 <= v321 && v321 < 2);
                    float v328;
                    v328 = v271[v321];
                    float v329;
                    v329 = (float)v327;
                    float v330;
                    v330 = v329 * v328;
                    float v331;
                    v331 = v322 + v330;
                    double v332;
                    v332 = v323 + v327;
                    v322 = v331;
                    v323 = v332;
                    v321 += 1 ;
                }
                float v333;
                v333 = (float)v323;
                assert("Tensor range check" && 0 <= v319 && v319 < 32);
                v317[v319] = v322;
                v318[v319] = v333;
                v319 += 1 ;
            }
            int v334;
            v334 = 0;
            while (while_method_10(v334)){
                assert("Tensor range check" && 0 <= v334 && v334 < 32);
                float v336;
                v336 = v317[v334];
                float v337;
                v337 = v318[v334];
                assert("Tensor range check" && 0 <= v334 && v334 < 2);
                assert("Tensor range check" && 0 <= v27 && v27 < 32);
                int v338;
                v338 = 32 * v334;
                int v339;
                v339 = v338 + v27;
                float * v340;
                v340 = v3+v339;
                float * v342;
                v342 = v4+v339;
                float v344;
                v344 = atomicAdd(v340,v336);
                float v345;
                v345 = atomicAdd(v342,v337);
                v334 += 1 ;
            }
            double * v346;
            v346 = reinterpret_cast<double *>(&v1[105381888ull]);
            double * v348;
            v348 = reinterpret_cast<double *>(&v1[108527616ull]);
            int v350;
            v350 = threadIdx.x;
            int v351;
            v351 = blockIdx.x;
            int v352;
            v352 = v351 * 256;
            int v353;
            v353 = v350 + v352;
            int v354;
            v354 = 0;
            while (while_method_10(v354)){
                assert("Tensor range check" && 0 <= v354 && v354 < 32);
                int v356;
                v356 = 12288 * v354;
                assert("Tensor range check" && 0 <= v353 && v353 < 6144);
                int v357;
                v357 = 2 * v353;
                int v358;
                v358 = v357 + v356;
                double * v359;
                v359 = v346+v358;
                double * v361;
                v361 = v348+v358;
                double * v363;
                v363 = v346+v358;
                double * v365;
                v365 = v348+v358;
                int v367;
                v367 = sizeof(double *);
                unsigned long long v368;
                v368 = (unsigned long long)v367;
                unsigned long long v369;
                v369 = 256ull * v368;
                unsigned long long v370;
                v370 = v369 + 16ull;
                unsigned long long v371;
                v371 = v370 - 1ull;
                unsigned long long v372;
                v372 = v371 % 16ull;
                unsigned long long v373;
                v373 = v371 - v372;
                unsigned long long v374;
                v374 = v373 + v369;
                unsigned long long v375;
                v375 = v374 + 16ull;
                unsigned long long v376;
                v376 = v375 - 1ull;
                unsigned long long v377;
                v377 = v376 % 16ull;
                unsigned long long v378;
                v378 = v376 - v377;
                unsigned long long v379;
                v379 = v378 + v369;
                unsigned long long v380;
                v380 = v379 + 16ull;
                unsigned long long v381;
                v381 = v380 - 1ull;
                unsigned long long v382;
                v382 = v381 % 16ull;
                unsigned long long v383;
                v383 = v381 - v382;
                unsigned long long v384;
                v384 = v383 + v369;
                bool v385;
                v385 = v384 <= 98304ull;
                bool v386;
                v386 = v385 == false;
                if (v386){
                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v385);
                } else {
                }
                extern __shared__ unsigned char v388[];
                bool v389;
                v389 = v384 <= v384;
                bool v390;
                v390 = v389 == false;
                if (v390){
                    assert("The length of the partition has to be less than or equal to the length of the base array." && v389);
                } else {
                }
                double * * v392;
                v392 = reinterpret_cast<double * *>(&v388[0ull]);
                double * * v394;
                v394 = reinterpret_cast<double * *>(&v388[v373]);
                double * * v396;
                v396 = reinterpret_cast<double * *>(&v388[v378]);
                double * * v398;
                v398 = reinterpret_cast<double * *>(&v388[v383]);
                int v400;
                v400 = threadIdx.x;
                assert("Tensor range check" && 0 <= v400 && v400 < 256);
                v392[v400] = v359;
                v394[v400] = v361;
                v396[v400] = v363;
                v398[v400] = v365;
                __syncthreads();
                bool v401;
                v401 = 0 <= v400;
                bool v402;
                v402 = v401 == false;
                if (v402){
                    assert("The index needs to be zero or positive." && v401);
                } else {
                }
                int v404;
                v404 = v400 % 1;
                bool v405;
                v405 = v400 < 256;
                bool v406;
                v406 = v405 == false;
                if (v406){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v405);
                } else {
                }
                assert("Tensor range check" && 0 <= v400 && v400 < 256);
                int v408;
                v408 = 0;
                while (while_method_4(v408)){
                    bool v410;
                    v410 = v401 && v405;
                    bool v411;
                    v411 = v410 == false;
                    if (v411){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v410);
                    } else {
                    }
                    bool v413;
                    v413 = 0 <= v408;
                    bool v415;
                    if (v413){
                        bool v414;
                        v414 = v408 < 1;
                        v415 = v414;
                    } else {
                        v415 = false;
                    }
                    bool v416;
                    v416 = v415 == false;
                    if (v416){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v415);
                    } else {
                    }
                    int v418;
                    v418 = v408 * 256;
                    int v419;
                    v419 = v418 + v400;
                    assert("Tensor range check" && 0 <= v408 && v408 < 1);
                    int v420;
                    v420 = 256 * v408;
                    int v421;
                    v421 = v420 + v400;
                    double * v422;
                    v422 = v392[v421];
                    double * v423;
                    v423 = v394[v421];
                    double * v424;
                    v424 = v396[v421];
                    double * v425;
                    v425 = v398[v421];
                    int v426;
                    v426 = blockIdx.x;
                    int v427;
                    v427 = v426 * 256;
                    int v428;
                    v428 = v427 + v419;
                    assert("Tensor range check" && 0 <= v404 && v404 < 1);
                    int v429;
                    v429 = 2 * v404;
                    double v430[2];
                    double v431[2];
                    int v432[2];
                    int v433;
                    v433 = 0;
                    while (while_method_4(v433)){
                        assert("Tensor range check" && 0 <= v433 && v433 < 1);
                        int v435;
                        v435 = 2 * v433;
                        assert("Tensor range check" && 0 <= v433 && v433 < 1);
                        int v436;
                        v436 = v435 + v429;
                        int4* v437;
                        v437 = reinterpret_cast<int4*>(v422 + v436);
                        int4* v438;
                        v438 = reinterpret_cast<int4*>(v430 + v435);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v437) % 16 == 0 && reinterpret_cast<unsigned long long>(v438) % 16 == 0);
                        *v438 = *v437;
                        int4* v439;
                        v439 = reinterpret_cast<int4*>(v423 + v436);
                        int4* v440;
                        v440 = reinterpret_cast<int4*>(v431 + v435);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v439) % 16 == 0 && reinterpret_cast<unsigned long long>(v440) % 16 == 0);
                        *v440 = *v439;
                        v433 += 1 ;
                    }
                    int v441;
                    v441 = 0;
                    while (while_method_4(v441)){
                        int v443;
                        v443 = 0;
                        while (while_method_0(v443)){
                            bool v445;
                            v445 = 0 <= v443;
                            bool v447;
                            if (v445){
                                bool v446;
                                v446 = v443 < 2;
                                v447 = v446;
                            } else {
                                v447 = false;
                            }
                            bool v448;
                            v448 = v447 == false;
                            if (v448){
                                assert("The indices should be inside the range of the dimension." && v447);
                            } else {
                            }
                            bool v450;
                            v450 = 0 <= v404;
                            bool v452;
                            if (v450){
                                bool v451;
                                v451 = v404 < 1;
                                v452 = v451;
                            } else {
                                v452 = false;
                            }
                            bool v453;
                            v453 = v452 == false;
                            if (v453){
                                assert("The indices should be inside the range of the dimension." && v452);
                            } else {
                            }
                            int v455;
                            v455 = v404 * 2;
                            int v456;
                            v456 = v443 + v455;
                            bool v457;
                            v457 = 0 <= v441;
                            bool v459;
                            if (v457){
                                bool v458;
                                v458 = v441 < 1;
                                v459 = v458;
                            } else {
                                v459 = false;
                            }
                            bool v460;
                            v460 = v459 == false;
                            if (v460){
                                assert("The indices should be inside the range of the dimension." && v459);
                            } else {
                            }
                            int v462;
                            v462 = v441 * 2;
                            int v463;
                            v463 = v456 + v462;
                            assert("Tensor range check" && 0 <= v441 && v441 < 1);
                            assert("Tensor range check" && 0 <= v443 && v443 < 2);
                            int v464;
                            v464 = 2 * v441;
                            int v465;
                            v465 = v464 + v443;
                            v432[v465] = v463;
                            v443 += 1 ;
                        }
                        v441 += 1 ;
                    }
                    double v466[2];
                    double v467[2];
                    int v468;
                    v468 = 0;
                    while (while_method_4(v468)){
                        int v470;
                        v470 = 0;
                        while (while_method_0(v470)){
                            assert("Tensor range check" && 0 <= v468 && v468 < 1);
                            assert("Tensor range check" && 0 <= v470 && v470 < 2);
                            int v472;
                            v472 = 2 * v468;
                            int v473;
                            v473 = v472 + v470;
                            double v474;
                            v474 = v430[v473];
                            double v475;
                            v475 = v431[v473];
                            assert("Tensor range check" && 0 <= v468 && v468 < 1);
                            assert("Tensor range check" && 0 <= v470 && v470 < 2);
                            v466[v473] = 0.0;
                            v467[v473] = 0.0;
                            v470 += 1 ;
                        }
                        v468 += 1 ;
                    }
                    int v476;
                    v476 = 0;
                    while (while_method_4(v476)){
                        assert("Tensor range check" && 0 <= v476 && v476 < 1);
                        int v478;
                        v478 = 2 * v476;
                        int v479;
                        v479 = v478 + v429;
                        assert("Tensor range check" && 0 <= v476 && v476 < 1);
                        int4* v480;
                        v480 = reinterpret_cast<int4*>(v466 + v478);
                        int4* v481;
                        v481 = reinterpret_cast<int4*>(v424 + v479);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v480) % 16 == 0 && reinterpret_cast<unsigned long long>(v481) % 16 == 0);
                        *v481 = *v480;
                        int4* v482;
                        v482 = reinterpret_cast<int4*>(v467 + v478);
                        int4* v483;
                        v483 = reinterpret_cast<int4*>(v425 + v479);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v482) % 16 == 0 && reinterpret_cast<unsigned long long>(v483) % 16 == 0);
                        *v483 = *v482;
                        v476 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v419 && v419 < 256);
                    v408 += 1 ;
                }
                __syncthreads();
                assert("Tensor range check" && 0 <= v400 && v400 < 256);
                __syncthreads();
                v354 += 1 ;
            }
            v29 += 1 ;
        }
        cooperative_groups::grid_group & v484 = v26.v1;
        cooperative_groups::grid_group & v485 = v484;
        curandStatePhilox4_32_10_t & v486 = v26.v5;
        curandStatePhilox4_32_10_t & v487 = v486;
        float * v488;
        v488 = reinterpret_cast<float *>(&v0[0ull]);
        float * v490;
        v490 = reinterpret_cast<float *>(&v2[0ull]);
        float * v492;
        v492 = reinterpret_cast<float *>(&v1[55050240ull]);
        int * v494;
        v494 = reinterpret_cast<int *>(&v0[1048576ull]);
        float * v496;
        v496 = reinterpret_cast<float *>(&v0[1048592ull]);
        float * v498;
        v498 = reinterpret_cast<float *>(&v0[1048720ull]);
        double * v500;
        v500 = reinterpret_cast<double *>(&v1[105381888ull]);
        double * v502;
        v502 = reinterpret_cast<double *>(&v1[108527616ull]);
        v485.sync() ;
        int v504;
        v504 = threadIdx.x;
        int v505;
        v505 = blockIdx.x;
        int v506;
        v506 = v505 * 256;
        int v507;
        v507 = v504 + v506;
        bool v508;
        v508 = v507 == 0;
        if (v508){
            int v509;
            v509 = 0;
            int v510;
            v510 = 32;
            int v511;
            v511 = int_range_22(v510, v509, v487);
            v494[0] = v511;
        } else {
        }
        __syncwarp();
        extern __shared__ unsigned char v512[];
        float * v513;
        v513 = reinterpret_cast<float *>(&v512[0ull]);
        int v515;
        v515 = blockIdx.x;
        int v516;
        v516 = v515;
        while (while_method_8(v516)){
            bool v518;
            v518 = 0 <= v516;
            bool v519;
            v519 = v518 == false;
            if (v519){
                assert("The index needs to be zero or positive." && v518);
            } else {
            }
            int v521;
            v521 = v516 % 16;
            int v522;
            v522 = v516 / 16;
            bool v523;
            v523 = v522 < 1;
            bool v524;
            v524 = v523 == false;
            if (v524){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v523);
            } else {
            }
            assert("Tensor range check" && 0 <= v522 && v522 < 1);
            assert("Tensor range check" && 0 <= v521 && v521 < 16);
            int v526;
            v526 = 512 * v521;
            int v527;
            v527 = 262144 * v522;
            int v528;
            v528 = v527 + v526;
            int v529;
            v529 = 16384 * v521;
            int v530;
            v530 = 32 * v522;
            int v531;
            v531 = v530 + v529;
            int v532;
            v532 = threadIdx.x;
            int v533;
            v533 = v532;
            while (while_method_12(v533)){
                bool v535;
                v535 = 0 <= v533;
                bool v536;
                v536 = v535 == false;
                if (v536){
                    assert("The index needs to be zero or positive." && v535);
                } else {
                }
                int v538;
                v538 = v533 % 512;
                int v539;
                v539 = v533 / 512;
                bool v540;
                v540 = v539 < 32;
                bool v541;
                v541 = v540 == false;
                if (v541){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v540);
                } else {
                }
                assert("Tensor range check" && 0 <= v539 && v539 < 32);
                assert("Tensor range check" && 0 <= v538 && v538 < 512);
                int v543;
                v543 = v538 + v528;
                int v544;
                v544 = 8192 * v539;
                int v545;
                v545 = v544 + v543;
                float v546;
                v546 = v488[v545];
                assert("Tensor range check" && 0 <= v539 && v539 < 32);
                assert("Tensor range check" && 0 <= v538 && v538 < 512);
                int v547;
                v547 = 513 * v539;
                int v548;
                v548 = v547 + v538;
                v513[v548] = v546;
                v533 += 256 ;
            }
            __syncthreads();
            int v549;
            v549 = threadIdx.x;
            int v550;
            v550 = v549;
            while (while_method_12(v550)){
                bool v552;
                v552 = 0 <= v550;
                bool v553;
                v553 = v552 == false;
                if (v553){
                    assert("The index needs to be zero or positive." && v552);
                } else {
                }
                int v555;
                v555 = v550 % 32;
                int v556;
                v556 = v550 / 32;
                bool v557;
                v557 = v556 < 512;
                bool v558;
                v558 = v557 == false;
                if (v558){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v557);
                } else {
                }
                assert("Tensor range check" && 0 <= v556 && v556 < 512);
                assert("Tensor range check" && 0 <= v555 && v555 < 32);
                int v560;
                v560 = 513 * v555;
                int v561;
                v561 = v556 + v560;
                float v562;
                v562 = v513[v561];
                assert("Tensor range check" && 0 <= v556 && v556 < 512);
                assert("Tensor range check" && 0 <= v555 && v555 < 32);
                int v563;
                v563 = v555 + v531;
                int v564;
                v564 = 32 * v556;
                int v565;
                v565 = v564 + v563;
                v490[v565] = v562;
                v550 += 256 ;
            }
            __syncthreads();
            v516 += 24 ;
        }
        v485.sync() ;
        int v566;
        v566 = threadIdx.x;
        bool v567;
        v567 = 0 <= v566;
        bool v568;
        v568 = v567 == false;
        if (v568){
            assert("The index needs to be zero or positive." && v567);
        } else {
        }
        int v570;
        v570 = v566 % 8;
        int v571;
        v571 = v566 / 8;
        int v572;
        v572 = v571 % 32;
        int v573;
        v573 = v571 / 32;
        bool v574;
        v574 = v573 < 1;
        bool v575;
        v575 = v574 == false;
        if (v575){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v574);
        } else {
        }
        assert("Tensor range check" && 0 <= v573 && v573 < 1);
        assert("Tensor range check" && 0 <= v572 && v572 < 32);
        assert("Tensor range check" && 0 <= v570 && v570 < 8);
        int v577;
        v577 = 4 * v570;
        int v578;
        v578 = 32 * v572;
        int v579;
        v579 = v578 + v577;
        int v580;
        v580 = 4096 * v573;
        int v581;
        v581 = v580 + v579;
        assert("Tensor range check" && 0 <= v573 && v573 < 1);
        assert("Tensor range check" && 0 <= v572 && v572 < 32);
        assert("Tensor range check" && 0 <= v570 && v570 < 8);
        int v582;
        v582 = blockIdx.x;
        int v583;
        v583 = v582;
        while (while_method_13(v583)){
            bool v585;
            v585 = 0 <= v583;
            bool v586;
            v586 = v585 == false;
            if (v586){
                assert("The index needs to be zero or positive." && v585);
            } else {
            }
            int v588;
            v588 = v583 % 4;
            int v589;
            v589 = v583 / 4;
            bool v590;
            v590 = v589 < 64;
            bool v591;
            v591 = v590 == false;
            if (v591){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v590);
            } else {
            }
            assert("Tensor range check" && 0 <= v589 && v589 < 64);
            assert("Tensor range check" && 0 <= v588 && v588 < 4);
            int v593;
            v593 = 1024 * v588;
            int v594;
            v594 = v593 + v581;
            int v595;
            v595 = 4096 * v589;
            int v596;
            v596 = v595 + v594;
            float v597[4];
            int v598[4];
            int v599;
            v599 = 0;
            while (while_method_4(v599)){
                assert("Tensor range check" && 0 <= v599 && v599 < 1);
                int v601;
                v601 = 4 * v599;
                assert("Tensor range check" && 0 <= v599 && v599 < 1);
                int v602;
                v602 = 32 * v599;
                int v603;
                v603 = v602 + v596;
                int4* v604;
                v604 = reinterpret_cast<int4*>(v490 + v603);
                int4* v605;
                v605 = reinterpret_cast<int4*>(v597 + v601);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v604) % 16 == 0 && reinterpret_cast<unsigned long long>(v605) % 16 == 0);
                *v605 = *v604;
                v599 += 1 ;
            }
            int v606;
            v606 = 0;
            while (while_method_4(v606)){
                int v608;
                v608 = 0;
                while (while_method_7(v608)){
                    bool v610;
                    v610 = 0 <= v608;
                    bool v612;
                    if (v610){
                        bool v611;
                        v611 = v608 < 4;
                        v612 = v611;
                    } else {
                        v612 = false;
                    }
                    bool v613;
                    v613 = v612 == false;
                    if (v613){
                        assert("The indices should be inside the range of the dimension." && v612);
                    } else {
                    }
                    bool v615;
                    v615 = 0 <= v570;
                    bool v617;
                    if (v615){
                        bool v616;
                        v616 = v570 < 8;
                        v617 = v616;
                    } else {
                        v617 = false;
                    }
                    bool v618;
                    v618 = v617 == false;
                    if (v618){
                        assert("The indices should be inside the range of the dimension." && v617);
                    } else {
                    }
                    int v620;
                    v620 = v570 * 4;
                    int v621;
                    v621 = v608 + v620;
                    bool v622;
                    v622 = 0 <= v606;
                    bool v624;
                    if (v622){
                        bool v623;
                        v623 = v606 < 1;
                        v624 = v623;
                    } else {
                        v624 = false;
                    }
                    bool v625;
                    v625 = v624 == false;
                    if (v625){
                        assert("The indices should be inside the range of the dimension." && v624);
                    } else {
                    }
                    int v627;
                    v627 = v606 * 32;
                    int v628;
                    v628 = v621 + v627;
                    assert("Tensor range check" && 0 <= v606 && v606 < 1);
                    assert("Tensor range check" && 0 <= v608 && v608 < 4);
                    int v629;
                    v629 = 4 * v606;
                    int v630;
                    v630 = v629 + v608;
                    v598[v630] = v628;
                    v608 += 1 ;
                }
                v606 += 1 ;
            }
            bool v631;
            v631 = 0 <= v573;
            bool v632;
            v632 = v631 && v574;
            bool v633;
            v633 = v632 == false;
            if (v633){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v632);
            } else {
            }
            bool v635;
            v635 = 0 <= v572;
            bool v637;
            if (v635){
                bool v636;
                v636 = v572 < 32;
                v637 = v636;
            } else {
                v637 = false;
            }
            bool v638;
            v638 = v637 == false;
            if (v638){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v637);
            } else {
            }
            bool v640;
            v640 = 0 <= v589;
            bool v641;
            v641 = v640 && v590;
            bool v642;
            v642 = v641 == false;
            if (v642){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v641);
            } else {
            }
            bool v644;
            v644 = 0 <= v588;
            bool v646;
            if (v644){
                bool v645;
                v645 = v588 < 4;
                v646 = v645;
            } else {
                v646 = false;
            }
            bool v647;
            v647 = v646 == false;
            if (v647){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v646);
            } else {
            }
            int v649;
            v649 = v588 * 32;
            int v650;
            v650 = v589 + v573;
            int v651;
            v651 = v649 + v572;
            float v652[4];
            int v653;
            v653 = 0;
            while (while_method_4(v653)){
                int v655;
                v655 = 0;
                while (while_method_7(v655)){
                    assert("Tensor range check" && 0 <= v653 && v653 < 1);
                    assert("Tensor range check" && 0 <= v655 && v655 < 4);
                    int v657;
                    v657 = 4 * v653;
                    int v658;
                    v658 = v657 + v655;
                    int v659;
                    v659 = v598[v658];
                    assert("Tensor range check" && 0 <= v659 && v659 < 32);
                    float v660;
                    v660 = v496[v659];
                    float v661;
                    v661 = v498[v659];
                    bool v662;
                    v662 = v661 == 0.0f;
                    bool v663;
                    v663 = v662 != true;
                    float v665;
                    if (v663){
                        float v664;
                        v664 = v660 / v661;
                        v665 = v664;
                    } else {
                        v665 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v653 && v653 < 1);
                    assert("Tensor range check" && 0 <= v655 && v655 < 4);
                    v652[v658] = v665;
                    v655 += 1 ;
                }
                v653 += 1 ;
            }
            float v666;
            v666 = 0.0f;
            int v667;
            v667 = 0;
            while (while_method_4(v667)){
                int v669;
                v669 = 0;
                while (while_method_7(v669)){
                    assert("Tensor range check" && 0 <= v667 && v667 < 1);
                    assert("Tensor range check" && 0 <= v669 && v669 < 4);
                    int v671;
                    v671 = 4 * v667;
                    int v672;
                    v672 = v671 + v669;
                    float v673;
                    v673 = v652[v672];
                    float v674;
                    v674 = v666 + v673;
                    v666 = v674;
                    v669 += 1 ;
                }
                v667 += 1 ;
            }
            auto v675 = cooperative_groups::coalesced_threads();
            int v676;
            v676 = threadIdx.x;
            int v677;
            v677 = v676 / 8;
            auto v678 = cooperative_groups::labeled_partition(v675,v677);
            Closure0 v679{};
            float v680;
            v680 = cooperative_groups::reduce(v678, v666, v679);
            float v681;
            v681 = v680 / 32.0f;
            bool v682[4];
            int v683;
            v683 = 0;
            while (while_method_4(v683)){
                int v685;
                v685 = 0;
                while (while_method_7(v685)){
                    assert("Tensor range check" && 0 <= v683 && v683 < 1);
                    assert("Tensor range check" && 0 <= v685 && v685 < 4);
                    int v687;
                    v687 = 4 * v683;
                    int v688;
                    v688 = v687 + v685;
                    float v689;
                    v689 = v652[v688];
                    bool v690;
                    v690 = v689 >= v681;
                    assert("Tensor range check" && 0 <= v683 && v683 < 1);
                    assert("Tensor range check" && 0 <= v685 && v685 < 4);
                    v682[v688] = v690;
                    v685 += 1 ;
                }
                v683 += 1 ;
            }
            int v691[4];
            int v692;
            v692 = 0;
            while (while_method_4(v692)){
                int v694;
                v694 = 0;
                while (while_method_7(v694)){
                    assert("Tensor range check" && 0 <= v692 && v692 < 1);
                    assert("Tensor range check" && 0 <= v694 && v694 < 4);
                    int v696;
                    v696 = 4 * v692;
                    int v697;
                    v697 = v696 + v694;
                    bool v698;
                    v698 = v682[v697];
                    int v699;
                    if (v698){
                        v699 = 1;
                    } else {
                        v699 = 0;
                    }
                    assert("Tensor range check" && 0 <= v692 && v692 < 1);
                    assert("Tensor range check" && 0 <= v694 && v694 < 4);
                    v691[v697] = v699;
                    v694 += 1 ;
                }
                v692 += 1 ;
            }
            int v700;
            v700 = 0;
            int v701;
            v701 = 0;
            while (while_method_4(v701)){
                int v703;
                v703 = 0;
                while (while_method_7(v703)){
                    assert("Tensor range check" && 0 <= v701 && v701 < 1);
                    assert("Tensor range check" && 0 <= v703 && v703 < 4);
                    int v705;
                    v705 = 4 * v701;
                    int v706;
                    v706 = v705 + v703;
                    int v707;
                    v707 = v691[v706];
                    int v708;
                    v708 = v700 + v707;
                    v700 = v708;
                    v703 += 1 ;
                }
                v701 += 1 ;
            }
            auto v709 = cooperative_groups::coalesced_threads();
            int v710;
            v710 = threadIdx.x;
            int v711;
            v711 = v710 / 8;
            auto v712 = cooperative_groups::labeled_partition(v709,v711);
            Closure1 v713{};
            int v714;
            v714 = cooperative_groups::reduce(v712, v700, v713);
            float v715;
            v715 = (float)v714;
            float v716[4];
            int v717;
            v717 = 0;
            while (while_method_4(v717)){
                int v719;
                v719 = 0;
                while (while_method_7(v719)){
                    assert("Tensor range check" && 0 <= v717 && v717 < 1);
                    assert("Tensor range check" && 0 <= v719 && v719 < 4);
                    int v721;
                    v721 = 4 * v717;
                    int v722;
                    v722 = v721 + v719;
                    float v723;
                    v723 = v597[v722];
                    bool v724;
                    v724 = v682[v722];
                    float v725;
                    if (v724){
                        v725 = v723;
                    } else {
                        v725 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v717 && v717 < 1);
                    assert("Tensor range check" && 0 <= v719 && v719 < 4);
                    v716[v722] = v725;
                    v719 += 1 ;
                }
                v717 += 1 ;
            }
            float v726;
            v726 = 0.0f;
            int v727;
            v727 = 0;
            while (while_method_4(v727)){
                int v729;
                v729 = 0;
                while (while_method_7(v729)){
                    assert("Tensor range check" && 0 <= v727 && v727 < 1);
                    assert("Tensor range check" && 0 <= v729 && v729 < 4);
                    int v731;
                    v731 = 4 * v727;
                    int v732;
                    v732 = v731 + v729;
                    float v733;
                    v733 = v716[v732];
                    float v734;
                    v734 = v726 + v733;
                    v726 = v734;
                    v729 += 1 ;
                }
                v727 += 1 ;
            }
            auto v735 = cooperative_groups::coalesced_threads();
            int v736;
            v736 = threadIdx.x;
            int v737;
            v737 = v736 / 8;
            auto v738 = cooperative_groups::labeled_partition(v735,v737);
            float v739;
            v739 = cooperative_groups::reduce(v738, v726, v679);
            float v740;
            v740 = v739 / v715;
            float v741[4];
            int v742;
            v742 = 0;
            while (while_method_4(v742)){
                int v744;
                v744 = 0;
                while (while_method_7(v744)){
                    assert("Tensor range check" && 0 <= v742 && v742 < 1);
                    assert("Tensor range check" && 0 <= v744 && v744 < 4);
                    int v746;
                    v746 = 4 * v742;
                    int v747;
                    v747 = v746 + v744;
                    float v748;
                    v748 = v597[v747];
                    float v749;
                    v749 = v748 - v740;
                    float v750;
                    v750 = v749 * v749;
                    assert("Tensor range check" && 0 <= v742 && v742 < 1);
                    assert("Tensor range check" && 0 <= v744 && v744 < 4);
                    v741[v747] = v750;
                    v744 += 1 ;
                }
                v742 += 1 ;
            }
            float v751[4];
            int v752;
            v752 = 0;
            while (while_method_4(v752)){
                int v754;
                v754 = 0;
                while (while_method_7(v754)){
                    assert("Tensor range check" && 0 <= v752 && v752 < 1);
                    assert("Tensor range check" && 0 <= v754 && v754 < 4);
                    int v756;
                    v756 = 4 * v752;
                    int v757;
                    v757 = v756 + v754;
                    float v758;
                    v758 = v741[v757];
                    bool v759;
                    v759 = v682[v757];
                    float v760;
                    if (v759){
                        v760 = v758;
                    } else {
                        v760 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v752 && v752 < 1);
                    assert("Tensor range check" && 0 <= v754 && v754 < 4);
                    v751[v757] = v760;
                    v754 += 1 ;
                }
                v752 += 1 ;
            }
            float v761;
            v761 = 0.0f;
            int v762;
            v762 = 0;
            while (while_method_4(v762)){
                int v764;
                v764 = 0;
                while (while_method_7(v764)){
                    assert("Tensor range check" && 0 <= v762 && v762 < 1);
                    assert("Tensor range check" && 0 <= v764 && v764 < 4);
                    int v766;
                    v766 = 4 * v762;
                    int v767;
                    v767 = v766 + v764;
                    float v768;
                    v768 = v751[v767];
                    float v769;
                    v769 = v761 + v768;
                    v761 = v769;
                    v764 += 1 ;
                }
                v762 += 1 ;
            }
            auto v770 = cooperative_groups::coalesced_threads();
            int v771;
            v771 = threadIdx.x;
            int v772;
            v772 = v771 / 8;
            auto v773 = cooperative_groups::labeled_partition(v770,v772);
            float v774;
            v774 = cooperative_groups::reduce(v773, v761, v679);
            float v775;
            v775 = v774 / v715;
            float v776;
            v776 = sqrt(v775);
            bool v777;
            v777 = v715 > 1.0f;
            float v781;
            if (v777){
                float v778;
                v778 = v776 * v715;
                float v779;
                v779 = v715 - 1.0f;
                float v780;
                v780 = v778 / v779;
                v781 = v780;
            } else {
                v781 = 0.0f;
            }
            float v782[4];
            int v783;
            v783 = 0;
            while (while_method_4(v783)){
                int v785;
                v785 = 0;
                while (while_method_7(v785)){
                    assert("Tensor range check" && 0 <= v783 && v783 < 1);
                    assert("Tensor range check" && 0 <= v785 && v785 < 4);
                    int v787;
                    v787 = 4 * v783;
                    int v788;
                    v788 = v787 + v785;
                    float v789;
                    v789 = v597[v788];
                    bool v790;
                    v790 = v682[v788];
                    float v791;
                    v791 = curand_normal(&v487);
                    float v792;
                    v792 = v781 + 0.3f;
                    float v793;
                    v793 = v791 * v792;
                    float v794;
                    v794 = v793 + v740;
                    float v795;
                    if (v790){
                        v795 = v789;
                    } else {
                        v795 = v794;
                    }
                    assert("Tensor range check" && 0 <= v783 && v783 < 1);
                    assert("Tensor range check" && 0 <= v785 && v785 < 4);
                    v782[v788] = v795;
                    v785 += 1 ;
                }
                v783 += 1 ;
            }
            assert("Tensor range check" && 0 <= v589 && v589 < 64);
            assert("Tensor range check" && 0 <= v588 && v588 < 4);
            int v796;
            v796 = 0;
            while (while_method_4(v796)){
                assert("Tensor range check" && 0 <= v796 && v796 < 1);
                int v798;
                v798 = 32 * v796;
                int v799;
                v799 = v798 + v596;
                assert("Tensor range check" && 0 <= v796 && v796 < 1);
                int v800;
                v800 = 4 * v796;
                int4* v801;
                v801 = reinterpret_cast<int4*>(v782 + v800);
                int4* v802;
                v802 = reinterpret_cast<int4*>(v490 + v799);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v801) % 16 == 0 && reinterpret_cast<unsigned long long>(v802) % 16 == 0);
                *v802 = *v801;
                v796 += 1 ;
            }
            v583 += 24 ;
        }
        v485.sync() ;
        extern __shared__ unsigned char v803[];
        float * v804;
        v804 = reinterpret_cast<float *>(&v803[0ull]);
        int v806;
        v806 = blockIdx.x;
        int v807;
        v807 = v806;
        while (while_method_8(v807)){
            bool v809;
            v809 = 0 <= v807;
            bool v810;
            v810 = v809 == false;
            if (v810){
                assert("The index needs to be zero or positive." && v809);
            } else {
            }
            int v812;
            v812 = v807 % 1;
            bool v813;
            v813 = v807 < 16;
            bool v814;
            v814 = v813 == false;
            if (v814){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v813);
            } else {
            }
            assert("Tensor range check" && 0 <= v807 && v807 < 16);
            assert("Tensor range check" && 0 <= v812 && v812 < 1);
            int v816;
            v816 = 32 * v812;
            int v817;
            v817 = 16384 * v807;
            int v818;
            v818 = v817 + v816;
            int v819;
            v819 = 262144 * v812;
            int v820;
            v820 = 512 * v807;
            int v821;
            v821 = v820 + v819;
            int v822;
            v822 = threadIdx.x;
            int v823;
            v823 = v822;
            while (while_method_12(v823)){
                bool v825;
                v825 = 0 <= v823;
                bool v826;
                v826 = v825 == false;
                if (v826){
                    assert("The index needs to be zero or positive." && v825);
                } else {
                }
                int v828;
                v828 = v823 % 32;
                int v829;
                v829 = v823 / 32;
                bool v830;
                v830 = v829 < 512;
                bool v831;
                v831 = v830 == false;
                if (v831){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v830);
                } else {
                }
                assert("Tensor range check" && 0 <= v829 && v829 < 512);
                assert("Tensor range check" && 0 <= v828 && v828 < 32);
                int v833;
                v833 = v828 + v818;
                int v834;
                v834 = 32 * v829;
                int v835;
                v835 = v834 + v833;
                float v836;
                v836 = v490[v835];
                assert("Tensor range check" && 0 <= v829 && v829 < 512);
                assert("Tensor range check" && 0 <= v828 && v828 < 32);
                int v837;
                v837 = 33 * v829;
                int v838;
                v838 = v837 + v828;
                v804[v838] = v836;
                v823 += 256 ;
            }
            __syncthreads();
            int v839;
            v839 = threadIdx.x;
            int v840;
            v840 = v839;
            while (while_method_12(v840)){
                bool v842;
                v842 = 0 <= v840;
                bool v843;
                v843 = v842 == false;
                if (v843){
                    assert("The index needs to be zero or positive." && v842);
                } else {
                }
                int v845;
                v845 = v840 % 512;
                int v846;
                v846 = v840 / 512;
                bool v847;
                v847 = v846 < 32;
                bool v848;
                v848 = v847 == false;
                if (v848){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v847);
                } else {
                }
                assert("Tensor range check" && 0 <= v846 && v846 < 32);
                assert("Tensor range check" && 0 <= v845 && v845 < 512);
                int v850;
                v850 = 33 * v845;
                int v851;
                v851 = v846 + v850;
                float v852;
                v852 = v804[v851];
                assert("Tensor range check" && 0 <= v846 && v846 < 32);
                assert("Tensor range check" && 0 <= v845 && v845 < 512);
                int v853;
                v853 = v845 + v821;
                int v854;
                v854 = 8192 * v846;
                int v855;
                v855 = v854 + v853;
                v488[v855] = v852;
                v840 += 256 ;
            }
            __syncthreads();
            v807 += 24 ;
        }
        int v856;
        v856 = threadIdx.x;
        int v857;
        v857 = blockIdx.x;
        int v858;
        v858 = v857 * 256;
        int v859;
        v859 = v856 + v858;
        int v860;
        v860 = v859;
        while (while_method_5(v860)){
            bool v862;
            v862 = 0 <= v860;
            bool v863;
            v863 = v862 == false;
            if (v863){
                assert("The index needs to be zero or positive." && v862);
            } else {
            }
            bool v865;
            v865 = v860 < 8;
            bool v866;
            v866 = v865 == false;
            if (v866){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v865);
            } else {
            }
            assert("Tensor range check" && 0 <= v860 && v860 < 8);
            int v868;
            v868 = 4 * v860;
            assert("Tensor range check" && 0 <= v860 && v860 < 8);
            float v869[4];
            float v870[4];
            float v871[4];
            float v872[4];
            int4* v873;
            v873 = reinterpret_cast<int4*>(v496 + v868);
            int4* v874;
            v874 = reinterpret_cast<int4*>(v869 + 0);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v873) % 16 == 0 && reinterpret_cast<unsigned long long>(v874) % 16 == 0);
            *v874 = *v873;
            int4* v875;
            v875 = reinterpret_cast<int4*>(v498 + v868);
            int4* v876;
            v876 = reinterpret_cast<int4*>(v870 + 0);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v875) % 16 == 0 && reinterpret_cast<unsigned long long>(v876) % 16 == 0);
            *v876 = *v875;
            // Pushing the loop unrolling to: 0
            int v877;
            v877 = 0;
            #pragma unroll
            while (while_method_7(v877)){
                assert("Tensor range check" && 0 <= v877 && v877 < 4);
                float v879;
                v879 = v869[v877];
                float v880;
                v880 = v870[v877];
                assert("Tensor range check" && 0 <= v877 && v877 < 4);
                v871[v877] = 0.0f;
                v872[v877] = 0.0f;
                v877 += 1 ;
            }
            // Poping the loop unrolling to: 0
            int4* v881;
            v881 = reinterpret_cast<int4*>(v871 + 0);
            int4* v882;
            v882 = reinterpret_cast<int4*>(v496 + v868);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v881) % 16 == 0 && reinterpret_cast<unsigned long long>(v882) % 16 == 0);
            *v882 = *v881;
            int4* v883;
            v883 = reinterpret_cast<int4*>(v872 + 0);
            int4* v884;
            v884 = reinterpret_cast<int4*>(v498 + v868);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v883) % 16 == 0 && reinterpret_cast<unsigned long long>(v884) % 16 == 0);
            *v884 = *v883;
            v860 += 6144 ;
        }
        v485.sync() ;
        v27 += 1 ;
    }
    cooperative_groups::grid_group & v885 = v26.v1;
    cooperative_groups::grid_group & v886 = v885;
    int v887;
    v887 = threadIdx.x;
    int v888;
    v888 = blockIdx.x;
    int v889;
    v889 = v888 * 256;
    int v890;
    v890 = v887 + v889;
    int v891;
    v891 = v890;
    while (while_method_8(v891)){
        bool v893;
        v893 = 0 <= v891;
        bool v894;
        v894 = v893 == false;
        if (v894){
            assert("The index needs to be zero or positive." && v893);
        } else {
        }
        int v896;
        v896 = v891 % 8;
        int v897;
        v897 = v891 / 8;
        bool v898;
        v898 = v897 < 2;
        bool v899;
        v899 = v898 == false;
        if (v899){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v898);
        } else {
        }
        assert("Tensor range check" && 0 <= v897 && v897 < 2);
        assert("Tensor range check" && 0 <= v896 && v896 < 8);
        int v901;
        v901 = 4 * v896;
        int v902;
        v902 = 32 * v897;
        int v903;
        v903 = v902 + v901;
        assert("Tensor range check" && 0 <= v897 && v897 < 2);
        assert("Tensor range check" && 0 <= v896 && v896 < 8);
        float v904[4];
        float v905[4];
        float v906[4];
        int4* v907;
        v907 = reinterpret_cast<int4*>(v3 + v903);
        int4* v908;
        v908 = reinterpret_cast<int4*>(v904 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v907) % 16 == 0 && reinterpret_cast<unsigned long long>(v908) % 16 == 0);
        *v908 = *v907;
        int4* v909;
        v909 = reinterpret_cast<int4*>(v4 + v903);
        int4* v910;
        v910 = reinterpret_cast<int4*>(v905 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v909) % 16 == 0 && reinterpret_cast<unsigned long long>(v910) % 16 == 0);
        *v910 = *v909;
        // Pushing the loop unrolling to: 0
        int v911;
        v911 = 0;
        #pragma unroll
        while (while_method_7(v911)){
            assert("Tensor range check" && 0 <= v911 && v911 < 4);
            float v913;
            v913 = v904[v911];
            float v914;
            v914 = v905[v911];
            bool v915;
            v915 = v914 == 0.0f;
            bool v916;
            v916 = v915 != true;
            float v918;
            if (v916){
                float v917;
                v917 = v913 / v914;
                v918 = v917;
            } else {
                v918 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v911 && v911 < 4);
            v906[v911] = v918;
            v911 += 1 ;
        }
        // Poping the loop unrolling to: 0
        int4* v919;
        v919 = reinterpret_cast<int4*>(v906 + 0);
        int4* v920;
        v920 = reinterpret_cast<int4*>(v5 + v903);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v919) % 16 == 0 && reinterpret_cast<unsigned long long>(v920) % 16 == 0);
        *v920 = *v919;
        v891 += 6144 ;
    }
    v886.sync() ;
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
class US1_0(NamedTuple): # Call
    tag = 0
class US1_1(NamedTuple): # Fold
    tag = 1
class US1_2(NamedTuple): # Raise
    tag = 2
US1 = Union[US1_0, US1_1, US1_2]
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
class US6_0(NamedTuple): # Jack
    tag = 0
class US6_1(NamedTuple): # King
    tag = 1
class US6_2(NamedTuple): # Queen
    tag = 2
US6 = Union[US6_0, US6_1, US6_2]
class US5_0(NamedTuple): # None
    tag = 0
class US5_1(NamedTuple): # Some
    v0 : US6
    tag = 1
US5 = Union[US5_0, US5_1]
class US4_0(NamedTuple): # ChanceCommunityCard
    v0 : US5
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 0
class US4_1(NamedTuple): # ChanceInit
    tag = 1
class US4_2(NamedTuple): # Round
    v0 : US5
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 2
class US4_3(NamedTuple): # RoundWithAction
    v0 : US5
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    v6 : US1
    tag = 3
class US4_4(NamedTuple): # TerminalCall
    v0 : US5
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 4
class US4_5(NamedTuple): # TerminalFold
    v0 : US5
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 5
US4 = Union[US4_0, US4_1, US4_2, US4_3, US4_4, US4_5]
class US3_0(NamedTuple): # None
    tag = 0
class US3_1(NamedTuple): # Some
    v0 : US4
    tag = 1
US3 = Union[US3_0, US3_1]
class US7_0(NamedTuple): # GameNotStarted
    tag = 0
class US7_1(NamedTuple): # GameOver
    v0 : US5
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 1
class US7_2(NamedTuple): # WaitingForActionFromPlayerId
    v0 : US5
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 2
US7 = Union[US7_0, US7_1, US7_2]
class US8_0(NamedTuple): # CommunityCardIs
    v0 : US6
    tag = 0
class US8_1(NamedTuple): # PlayerAction
    v0 : i32
    v1 : US1
    tag = 1
class US8_2(NamedTuple): # PlayerGotCard
    v0 : i32
    v1 : US6
    tag = 2
class US8_3(NamedTuple): # Showdown
    v0 : static_array
    v1 : i32
    v2 : i32
    tag = 3
US8 = Union[US8_0, US8_1, US8_2, US8_3]
class US9_0(NamedTuple): # AddRewardsRando
    v0 : list
    tag = 0
class US9_1(NamedTuple): # AddRewardsSelf
    v0 : list
    tag = 1
US9 = Union[US9_0, US9_1]
def Closure0():
    def inner(v0 : object, v1 : object) -> object:
        v2 = method0(v0)
        v3, v4, v5, v6, v7, v8, v9, v10 = method7(v1)
        v11 = cp.empty(16,dtype=cp.uint8)
        v12 = cp.empty(1184,dtype=cp.uint8)
        method36(v12, v3, v4, v5, v6, v7)
        del v3, v4, v5, v6, v7
        v15 = "{}\n"
        v16 = "Going to run the Leduc full kernel."
        print(v15.format(v16),end="")
        del v15, v16
        v17 = time.perf_counter()
        v18 = []
        match v2:
            case US0_0(_): # ActionSelected
                method54(v11, v2)
                v77 = cp.cuda.Device().attributes['MultiProcessorCount']
                v78 = v77 == 24
                del v77
                v79 = v78 == False
                if v79:
                    v80 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v78, v80
                    del v80
                else:
                    pass
                del v78, v79
                v81 = 0
                v82 = raw_module.get_function(f"entry{v81}")
                del v81
                v82.max_dynamic_shared_size_bytes = 98304 
                print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
                v82((24,),(256,),(v12, v11, v8, v9, v10),shared_mem=98304)
                del v82
            case US0_1(_): # PlayerChanged
                method54(v11, v2)
                v70 = cp.cuda.Device().attributes['MultiProcessorCount']
                v71 = v70 == 24
                del v70
                v72 = v71 == False
                if v72:
                    v73 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v71, v73
                    del v73
                else:
                    pass
                del v71, v72
                v74 = 0
                v75 = raw_module.get_function(f"entry{v74}")
                del v74
                v75.max_dynamic_shared_size_bytes = 98304 
                print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
                v75((24,),(256,),(v12, v11, v8, v9, v10),shared_mem=98304)
                del v75
            case US0_2(): # StartGame
                method54(v11, v2)
                v63 = cp.cuda.Device().attributes['MultiProcessorCount']
                v64 = v63 == 24
                del v63
                v65 = v64 == False
                if v65:
                    v66 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v64, v66
                    del v66
                else:
                    pass
                del v64, v65
                v67 = 0
                v68 = raw_module.get_function(f"entry{v67}")
                del v67
                v68.max_dynamic_shared_size_bytes = 98304 
                print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
                v68((24,),(256,),(v12, v11, v8, v9, v10),shared_mem=98304)
                del v68
            case US0_3(): # StartTrainingVsRando
                v19 = cp.zeros(1024,dtype=cp.float32) # type: ignore
                v20 = cp.zeros(1024,dtype=cp.float32) # type: ignore
                v21 = cp.empty(1024,dtype=cp.float32)
                v22 = cp.cuda.Device().attributes['MultiProcessorCount']
                v23 = v22 == 24
                del v22
                v24 = v23 == False
                if v24:
                    v25 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v23, v25
                    del v25
                else:
                    pass
                del v23, v24
                v26 = 1
                v27 = raw_module.get_function(f"entry{v26}")
                del v26
                v27.max_dynamic_shared_size_bytes = 98304 
                print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
                v27((24,),(256,),(v8, v9, v10, v19, v20, v21),shared_mem=98304)
                del v19, v20, v27
                v28 = []
                v30 = v21[0:]
                del v21
                v31 = v30.get()
                del v30
                v32 = 0
                while method57(v32):
                    v34 = []
                    v35 = 0
                    while method57(v35):
                        assert 0 <= v32 < 32, 'Tensor range check'
                        assert 0 <= v35 < 32, 'Tensor range check'
                        v37 = 32 * v32
                        v38 = v37 + v35
                        del v37
                        v39 = v31[v38].item()
                        del v38
                        v34.append(v39)
                        del v39
                        v35 += 1 
                    del v35
                    v28.append(v34)
                    del v34
                    v32 += 1 
                del v31, v32
                v40 = US9_0(v28)
                del v28
                v18.append(v40)
                del v40
            case US0_4(): # StartTrainingVsSelf
                v41 = cp.zeros(64,dtype=cp.float32) # type: ignore
                v42 = cp.zeros(64,dtype=cp.float32) # type: ignore
                v43 = cp.empty(64,dtype=cp.float32)
                v44 = cp.cuda.Device().attributes['MultiProcessorCount']
                v45 = v44 == 24
                del v44
                v46 = v45 == False
                if v46:
                    v47 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v45, v47
                    del v47
                else:
                    pass
                del v45, v46
                v48 = 2
                v49 = raw_module.get_function(f"entry{v48}")
                del v48
                v49.max_dynamic_shared_size_bytes = 98304 
                print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
                v49((24,),(256,),(v8, v9, v10, v41, v42, v43),shared_mem=98304)
                del v41, v42, v49
                v50 = []
                v52 = v43[0:]
                del v43
                v53 = v52.get()
                del v52
                v54 = 0
                while method44(v54):
                    v56 = []
                    v57 = 0
                    while method57(v57):
                        assert 0 <= v54 < 2, 'Tensor range check'
                        assert 0 <= v57 < 32, 'Tensor range check'
                        v59 = 32 * v54
                        v60 = v59 + v57
                        del v59
                        v61 = v53[v60].item()
                        del v60
                        v56.append(v61)
                        del v61
                        v57 += 1 
                    del v57
                    v50.append(v56)
                    del v56
                    v54 += 1 
                del v53, v54
                v62 = US9_1(v50)
                del v50
                v18.append(v62)
                del v62
            case t:
                raise Exception(f'Pattern matching miss. Got: {t}')
        del v2, v11
        cp.cuda.get_current_stream().synchronize()
        v83 = time.perf_counter()
        v86 = "{}"
        v87 = "The time it took to run the kernel (in seconds) is: "
        print(v86.format(v87),end="")
        del v86, v87
        v88 = v83 - v17
        del v17, v83
        v91 = "{:.6f}\n"
        print(v91.format(v88),end="")
        del v88, v91
        v92, v93, v94, v95, v96 = method58(v12)
        del v12
        return method75(v92, v93, v94, v95, v96, v8, v9, v10, v18)
    return inner
def Closure1():
    def inner() -> object:
        v0 = cp.empty(1048576,dtype=cp.uint8)
        v1 = cp.empty(111673344,dtype=cp.uint8)
        v2 = cp.empty(1048848,dtype=cp.uint8)
        v4 = v1[0:0+4*786432].view(cp.float32)
        del v4
        v6 = v2[0:0+4*262144].view(cp.float32)
        v8 = v0[0:0+4*262144].view(cp.float32)
        del v8
        v10 = v1[4718592:4718592+4*12582912].view(cp.float32)
        del v10
        v12 = v1[55050240:55050240+4*12582912].view(cp.float32)
        del v12
        v14 = v2[1048576:1048576+4*1].view(cp.int32)
        v16 = v2[1048592:1048592+4*32].view(cp.float32)
        v18 = v2[1048720:1048720+4*32].view(cp.float32)
        v20 = v1[105381888:105381888+8*393216].view(cp.float64)
        v22 = v1[108527616:108527616+8*393216].view(cp.float64)
        v23 = cp.random.normal(0.0,0.001953125,262144,dtype=cp.float32) # type: ignore
        cp.copyto(v6[0:0+262144],v23[0:0+262144])
        del v6, v23
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
        v25 = static_array(2)
        v27 = US2_0()
        v25[0] = v27
        del v27
        v29 = US2_1()
        v25[1] = v29
        del v29
        v31 = static_array_list(32)
        v32 = 63
        v33 = US3_0()
        v34 = US7_0()
        return method114(v32, v33, v31, v25, v34, v2, v1, v0)
    return inner
def method3(v0 : object) -> None:
    assert v0 == [], f'Expected an unit type. Got: {v0}'
    del v0
    return 
def method2(v0 : object) -> US1:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "Call" == v1
    if v3:
        del v1, v3
        method3(v2)
        del v2
        return US1_0()
    else:
        del v3
        v5 = "Fold" == v1
        if v5:
            del v1, v5
            method3(v2)
            del v2
            return US1_1()
        else:
            del v5
            v7 = "Raise" == v1
            if v7:
                del v1, v7
                method3(v2)
                del v2
                return US1_2()
            else:
                del v2, v7
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method5(v0 : i32, v1 : i32) -> bool:
    v2 = v1 < v0
    del v0, v1
    return v2
def method6(v0 : object) -> US2:
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
def method4(v0 : object) -> static_array:
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
    while method5(v1, v7):
        v9 = v0[v7]
        v10 = method6(v9)
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
            v7 = method4(v2)
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
def method12(v0 : object) -> u32:
    assert isinstance(v0,u32), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method11(v0 : object) -> u32:
    v1 = method12(v0)
    del v0
    return v1
def method17(v0 : object) -> US6:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "Jack" == v1
    if v3:
        del v1, v3
        method3(v2)
        del v2
        return US6_0()
    else:
        del v3
        v5 = "King" == v1
        if v5:
            del v1, v5
            method3(v2)
            del v2
            return US6_1()
        else:
            del v5
            v7 = "Queen" == v1
            if v7:
                del v1, v7
                method3(v2)
                del v2
                return US6_2()
            else:
                del v2, v7
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method16(v0 : object) -> US5:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "None" == v1
    if v3:
        del v1, v3
        method3(v2)
        del v2
        return US5_0()
    else:
        del v3
        v5 = "Some" == v1
        if v5:
            del v1, v5
            v6 = method17(v2)
            del v2
            return US5_1(v6)
        else:
            del v2, v5
            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
            del v1
            raise Exception("Error")
def method18(v0 : object) -> bool:
    assert isinstance(v0,bool), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method19(v0 : object) -> static_array:
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
    while method5(v1, v7):
        v9 = v0[v7]
        v10 = method17(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method20(v0 : object) -> i32:
    assert isinstance(v0,i32), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
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
    while method5(v1, v7):
        v9 = v0[v7]
        v10 = method20(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method15(v0 : object) -> Tuple[US5, bool, static_array, i32, static_array, i32]:
    v1 = v0["community_card"] # type: ignore
    v2 = method16(v1)
    del v1
    v3 = v0["is_button_s_first_move"] # type: ignore
    v4 = method18(v3)
    del v3
    v5 = v0["pl_card"] # type: ignore
    v6 = method19(v5)
    del v5
    v7 = v0["player_turn"] # type: ignore
    v8 = method20(v7)
    del v7
    v9 = v0["pot"] # type: ignore
    v10 = method21(v9)
    del v9
    v11 = v0["raises_left"] # type: ignore
    del v0
    v12 = method20(v11)
    del v11
    return v2, v4, v6, v8, v10, v12
def method22(v0 : object) -> Tuple[US5, bool, static_array, i32, static_array, i32, US1]:
    v1 = v0[0] # type: ignore
    v2, v3, v4, v5, v6, v7 = method15(v1)
    del v1
    v8 = v0[1] # type: ignore
    del v0
    v9 = method2(v8)
    del v8
    return v2, v3, v4, v5, v6, v7, v9
def method14(v0 : object) -> US4:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "ChanceCommunityCard" == v1
    if v3:
        del v1, v3
        v4, v5, v6, v7, v8, v9 = method15(v2)
        del v2
        return US4_0(v4, v5, v6, v7, v8, v9)
    else:
        del v3
        v11 = "ChanceInit" == v1
        if v11:
            del v1, v11
            method3(v2)
            del v2
            return US4_1()
        else:
            del v11
            v13 = "Round" == v1
            if v13:
                del v1, v13
                v14, v15, v16, v17, v18, v19 = method15(v2)
                del v2
                return US4_2(v14, v15, v16, v17, v18, v19)
            else:
                del v13
                v21 = "RoundWithAction" == v1
                if v21:
                    del v1, v21
                    v22, v23, v24, v25, v26, v27, v28 = method22(v2)
                    del v2
                    return US4_3(v22, v23, v24, v25, v26, v27, v28)
                else:
                    del v21
                    v30 = "TerminalCall" == v1
                    if v30:
                        del v1, v30
                        v31, v32, v33, v34, v35, v36 = method15(v2)
                        del v2
                        return US4_4(v31, v32, v33, v34, v35, v36)
                    else:
                        del v30
                        v38 = "TerminalFold" == v1
                        if v38:
                            del v1, v38
                            v39, v40, v41, v42, v43, v44 = method15(v2)
                            del v2
                            return US4_5(v39, v40, v41, v42, v43, v44)
                        else:
                            del v2, v38
                            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                            del v1
                            raise Exception("Error")
def method13(v0 : object) -> US3:
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
            v6 = method14(v2)
            del v2
            return US3_1(v6)
        else:
            del v2, v5
            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
            del v1
            raise Exception("Error")
def method10(v0 : object) -> Tuple[u32, US3]:
    v1 = v0["deck"] # type: ignore
    v2 = method11(v1)
    del v1
    v3 = v0["game"] # type: ignore
    del v0
    v4 = method13(v3)
    del v3
    return v2, v4
def method26(v0 : object) -> Tuple[i32, US1]:
    v1 = v0[0] # type: ignore
    v2 = method20(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method2(v3)
    del v3
    return v2, v4
def method27(v0 : object) -> Tuple[i32, US6]:
    v1 = v0[0] # type: ignore
    v2 = method20(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method17(v3)
    del v3
    return v2, v4
def method28(v0 : object) -> Tuple[static_array, i32, i32]:
    v1 = v0["cards_shown"] # type: ignore
    v2 = method19(v1)
    del v1
    v3 = v0["chips_won"] # type: ignore
    v4 = method20(v3)
    del v3
    v5 = v0["winner_id"] # type: ignore
    del v0
    v6 = method20(v5)
    del v5
    return v2, v4, v6
def method25(v0 : object) -> US8:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "CommunityCardIs" == v1
    if v3:
        del v1, v3
        v4 = method17(v2)
        del v2
        return US8_0(v4)
    else:
        del v3
        v6 = "PlayerAction" == v1
        if v6:
            del v1, v6
            v7, v8 = method26(v2)
            del v2
            return US8_1(v7, v8)
        else:
            del v6
            v10 = "PlayerGotCard" == v1
            if v10:
                del v1, v10
                v11, v12 = method27(v2)
                del v2
                return US8_2(v11, v12)
            else:
                del v10
                v14 = "Showdown" == v1
                if v14:
                    del v1, v14
                    v15, v16, v17 = method28(v2)
                    del v2
                    return US8_3(v15, v16, v17)
                else:
                    del v2, v14
                    raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                    del v1
                    raise Exception("Error")
def method24(v0 : object) -> static_array_list:
    v1 = len(v0) # type: ignore
    assert (32 >= v1), f'The length of the original object has to be greater than or equal to the static array dimension.\nExpected: 32\nGot: {v1} '
    del v1
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v2 = len(v0) # type: ignore
    v3 = 32 >= v2
    v4 = v3 == False
    if v4:
        v5 = "The type level dimension has to equal the value passed at runtime into create."
        assert v3, v5
        del v5
    else:
        pass
    del v3, v4
    v7 = static_array_list(32)
    v7.unsafe_set_length(v2)
    v8 = 0
    while method5(v2, v8):
        v10 = v0[v8]
        v11 = method25(v10)
        del v10
        v7[v8] = v11
        del v11
        v8 += 1 
    del v0, v2, v8
    return v7
def method29(v0 : object) -> US7:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "GameNotStarted" == v1
    if v3:
        del v1, v3
        method3(v2)
        del v2
        return US7_0()
    else:
        del v3
        v5 = "GameOver" == v1
        if v5:
            del v1, v5
            v6, v7, v8, v9, v10, v11 = method15(v2)
            del v2
            return US7_1(v6, v7, v8, v9, v10, v11)
        else:
            del v5
            v13 = "WaitingForActionFromPlayerId" == v1
            if v13:
                del v1, v13
                v14, v15, v16, v17, v18, v19 = method15(v2)
                del v2
                return US7_2(v14, v15, v16, v17, v18, v19)
            else:
                del v2, v13
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method23(v0 : object) -> Tuple[static_array_list, static_array, US7]:
    v1 = v0["messages"] # type: ignore
    v2 = method24(v1)
    del v1
    v3 = v0["pl_type"] # type: ignore
    v4 = method4(v3)
    del v3
    v5 = v0["ui_game_state"] # type: ignore
    del v0
    v6 = method29(v5)
    del v5
    return v2, v4, v6
def method9(v0 : object) -> Tuple[u32, US3, static_array_list, static_array, US7]:
    v1 = v0["private"] # type: ignore
    v2, v3 = method10(v1)
    del v1
    v4 = v0["public"] # type: ignore
    del v0
    v5, v6, v7 = method23(v4)
    del v4
    return v2, v3, v5, v6, v7
def method35(v0 : object) -> cp.ndarray:
    assert isinstance(v0,cp.ndarray), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method34(v0 : object) -> cp.ndarray:
    v1 = method35(v0)
    del v0
    return v1
def method33(v0 : object) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    v1 = v0[0] # type: ignore
    v2 = method34(v1)
    del v1
    v3 = v0[1] # type: ignore
    v4 = method34(v3)
    del v3
    v5 = v0[2] # type: ignore
    v6 = method34(v5)
    del v5
    v7 = v0[3] # type: ignore
    del v0
    method3(v7)
    del v7
    return v2, v4, v6
def method32(v0 : object) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    v1, v2, v3 = method33(v0)
    del v0
    return v1, v2, v3
def method31(v0 : object) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    v1, v2, v3 = method32(v0)
    del v0
    return v1, v2, v3
def method30(v0 : object) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    v1 = v0["model_ptrs"] # type: ignore
    del v0
    v2, v3, v4 = method31(v1)
    del v1
    return v2, v3, v4
def method8(v0 : object) -> Tuple[u32, US3, static_array_list, static_array, US7, cp.ndarray, cp.ndarray, cp.ndarray]:
    v1 = v0["game"] # type: ignore
    v2, v3, v4, v5, v6 = method9(v1)
    del v1
    v7 = v0["neural"] # type: ignore
    del v0
    v8, v9, v10 = method30(v7)
    del v7
    return v2, v3, v4, v5, v6, v8, v9, v10
def method7(v0 : object) -> Tuple[u32, US3, static_array_list, static_array, US7, cp.ndarray, cp.ndarray, cp.ndarray]:
    return method8(v0)
def method37(v0 : cp.ndarray, v1 : u32) -> None:
    v3 = v0[0:].view(cp.uint32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method38(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[4:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method39(v0 : cp.ndarray) -> None:
    del v0
    return 
def method41(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[0:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method43(v0 : cp.ndarray, v1 : US6) -> None:
    v2 = v1.tag
    method41(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US6_0(): # Jack
            del v1
            return method39(v4)
        case US6_1(): # King
            del v1
            return method39(v4)
        case US6_2(): # Queen
            del v1
            return method39(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method44(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method42(v0 : cp.ndarray, v1 : US5, v2 : bool, v3 : static_array, v4 : i32, v5 : static_array, v6 : i32) -> None:
    v7 = v1.tag
    method41(v0, v7)
    del v7
    v9 = v0[4:].view(cp.uint8)
    match v1:
        case US5_0(): # None
            method39(v9)
        case US5_1(v10): # Some
            method43(v9, v10)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v1, v9
    v12 = v0[8:].view(cp.bool_)
    v12[0] = v2
    del v2, v12
    v13 = 0
    while method44(v13):
        v15 = u64(v13)
        v16 = v15 * 4
        del v15
        v17 = 12 + v16
        del v16
        v19 = v0[v17:].view(cp.uint8)
        del v17
        v20 = 0 <= v13
        if v20:
            v21 = v13 < 2
            v22 = v21
        else:
            v22 = False
        del v20
        v23 = v22 == False
        if v23:
            v24 = "Index must be in range."
            assert v22, v24
            del v24
        else:
            pass
        del v22, v23
        v26 = v3[v13]
        method43(v19, v26)
        del v19, v26
        v13 += 1 
    del v3, v13
    v28 = v0[20:].view(cp.int32)
    v28[0] = v4
    del v4, v28
    v29 = 0
    while method44(v29):
        v31 = u64(v29)
        v32 = v31 * 4
        del v31
        v33 = 24 + v32
        del v32
        v35 = v0[v33:].view(cp.uint8)
        del v33
        v36 = 0 <= v29
        if v36:
            v37 = v29 < 2
            v38 = v37
        else:
            v38 = False
        del v36
        v39 = v38 == False
        if v39:
            v40 = "Index must be in range."
            assert v38, v40
            del v40
        else:
            pass
        del v38, v39
        v42 = v5[v29]
        method41(v35, v42)
        del v35, v42
        v29 += 1 
    del v5, v29
    v44 = v0[32:].view(cp.int32)
    del v0
    v44[0] = v6
    del v6, v44
    return 
def method46(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[36:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method45(v0 : cp.ndarray, v1 : US5, v2 : bool, v3 : static_array, v4 : i32, v5 : static_array, v6 : i32, v7 : US1) -> None:
    v8 = v1.tag
    method41(v0, v8)
    del v8
    v10 = v0[4:].view(cp.uint8)
    match v1:
        case US5_0(): # None
            method39(v10)
        case US5_1(v11): # Some
            method43(v10, v11)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v1, v10
    v13 = v0[8:].view(cp.bool_)
    v13[0] = v2
    del v2, v13
    v14 = 0
    while method44(v14):
        v16 = u64(v14)
        v17 = v16 * 4
        del v16
        v18 = 12 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v21 = 0 <= v14
        if v21:
            v22 = v14 < 2
            v23 = v22
        else:
            v23 = False
        del v21
        v24 = v23 == False
        if v24:
            v25 = "Index must be in range."
            assert v23, v25
            del v25
        else:
            pass
        del v23, v24
        v27 = v3[v14]
        method43(v20, v27)
        del v20, v27
        v14 += 1 
    del v3, v14
    v29 = v0[20:].view(cp.int32)
    v29[0] = v4
    del v4, v29
    v30 = 0
    while method44(v30):
        v32 = u64(v30)
        v33 = v32 * 4
        del v32
        v34 = 24 + v33
        del v33
        v36 = v0[v34:].view(cp.uint8)
        del v34
        v37 = 0 <= v30
        if v37:
            v38 = v30 < 2
            v39 = v38
        else:
            v39 = False
        del v37
        v40 = v39 == False
        if v40:
            v41 = "Index must be in range."
            assert v39, v41
            del v41
        else:
            pass
        del v39, v40
        v43 = v5[v30]
        method41(v36, v43)
        del v36, v43
        v30 += 1 
    del v5, v30
    v45 = v0[32:].view(cp.int32)
    v45[0] = v6
    del v6, v45
    v46 = v7.tag
    method46(v0, v46)
    del v46
    v48 = v0[40:].view(cp.uint8)
    del v0
    match v7:
        case US1_0(): # Call
            del v7
            return method39(v48)
        case US1_1(): # Fold
            del v7
            return method39(v48)
        case US1_2(): # Raise
            del v7
            return method39(v48)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method40(v0 : cp.ndarray, v1 : US4) -> None:
    v2 = v1.tag
    method41(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US4_0(v5, v6, v7, v8, v9, v10): # ChanceCommunityCard
            del v1
            return method42(v4, v5, v6, v7, v8, v9, v10)
        case US4_1(): # ChanceInit
            del v1
            return method39(v4)
        case US4_2(v11, v12, v13, v14, v15, v16): # Round
            del v1
            return method42(v4, v11, v12, v13, v14, v15, v16)
        case US4_3(v17, v18, v19, v20, v21, v22, v23): # RoundWithAction
            del v1
            return method45(v4, v17, v18, v19, v20, v21, v22, v23)
        case US4_4(v24, v25, v26, v27, v28, v29): # TerminalCall
            del v1
            return method42(v4, v24, v25, v26, v27, v28, v29)
        case US4_5(v30, v31, v32, v33, v34, v35): # TerminalFold
            del v1
            return method42(v4, v30, v31, v32, v33, v34, v35)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method47(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[80:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method49(v0 : cp.ndarray, v1 : i32, v2 : US1) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = v2.tag
    method38(v0, v5)
    del v5
    v7 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US1_0(): # Call
            del v2
            return method39(v7)
        case US1_1(): # Fold
            del v2
            return method39(v7)
        case US1_2(): # Raise
            del v2
            return method39(v7)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method50(v0 : cp.ndarray, v1 : i32, v2 : US6) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = v2.tag
    method38(v0, v5)
    del v5
    v7 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US6_0(): # Jack
            del v2
            return method39(v7)
        case US6_1(): # King
            del v2
            return method39(v7)
        case US6_2(): # Queen
            del v2
            return method39(v7)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method51(v0 : cp.ndarray, v1 : static_array, v2 : i32, v3 : i32) -> None:
    v4 = 0
    while method44(v4):
        v6 = u64(v4)
        v7 = v6 * 4
        del v6
        v9 = v0[v7:].view(cp.uint8)
        del v7
        v10 = 0 <= v4
        if v10:
            v11 = v4 < 2
            v12 = v11
        else:
            v12 = False
        del v10
        v13 = v12 == False
        if v13:
            v14 = "Index must be in range."
            assert v12, v14
            del v14
        else:
            pass
        del v12, v13
        v16 = v1[v4]
        method43(v9, v16)
        del v9, v16
        v4 += 1 
    del v1, v4
    v18 = v0[8:].view(cp.int32)
    v18[0] = v2
    del v2, v18
    v20 = v0[12:].view(cp.int32)
    del v0
    v20[0] = v3
    del v3, v20
    return 
def method48(v0 : cp.ndarray, v1 : US8) -> None:
    v2 = v1.tag
    method41(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US8_0(v5): # CommunityCardIs
            del v1
            return method43(v4, v5)
        case US8_1(v6, v7): # PlayerAction
            del v1
            return method49(v4, v6, v7)
        case US8_2(v8, v9): # PlayerGotCard
            del v1
            return method50(v4, v8, v9)
        case US8_3(v10, v11, v12): # Showdown
            del v1
            return method51(v4, v10, v11, v12)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method52(v0 : cp.ndarray, v1 : US2) -> None:
    v2 = v1.tag
    method41(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US2_0(): # Computer
            del v1
            return method39(v4)
        case US2_1(): # Human
            del v1
            return method39(v4)
        case US2_2(): # Random
            del v1
            return method39(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method53(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[1128:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method36(v0 : cp.ndarray, v1 : u32, v2 : US3, v3 : static_array_list, v4 : static_array, v5 : US7) -> None:
    method37(v0, v1)
    del v1
    v6 = v2.tag
    method38(v0, v6)
    del v6
    v8 = v0[16:].view(cp.uint8)
    match v2:
        case US3_0(): # None
            method39(v8)
        case US3_1(v9): # Some
            method40(v8, v9)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v2, v8
    v10 = v3.length
    method47(v0, v10)
    del v10
    v11 = v3.length
    v12 = 0
    while method5(v11, v12):
        v14 = u64(v12)
        v15 = v14 * 32
        del v14
        v16 = 96 + v15
        del v15
        v18 = v0[v16:].view(cp.uint8)
        del v16
        v20 = v3[v12]
        method48(v18, v20)
        del v18, v20
        v12 += 1 
    del v3, v11, v12
    v21 = 0
    while method44(v21):
        v23 = u64(v21)
        v24 = v23 * 4
        del v23
        v25 = 1120 + v24
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
        method52(v27, v34)
        del v27, v34
        v21 += 1 
    del v4, v21
    v35 = v5.tag
    method53(v0, v35)
    del v35
    v37 = v0[1136:].view(cp.uint8)
    del v0
    match v5:
        case US7_0(): # GameNotStarted
            del v5
            return method39(v37)
        case US7_1(v38, v39, v40, v41, v42, v43): # GameOver
            del v5
            return method42(v37, v38, v39, v40, v41, v42, v43)
        case US7_2(v44, v45, v46, v47, v48, v49): # WaitingForActionFromPlayerId
            del v5
            return method42(v37, v44, v45, v46, v47, v48, v49)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method55(v0 : cp.ndarray, v1 : US1) -> None:
    v2 = v1.tag
    method41(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US1_0(): # Call
            del v1
            return method39(v4)
        case US1_1(): # Fold
            del v1
            return method39(v4)
        case US1_2(): # Raise
            del v1
            return method39(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method56(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method44(v2):
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
        method52(v7, v14)
        del v7, v14
        v2 += 1 
    del v0, v1, v2
    return 
def method54(v0 : cp.ndarray, v1 : US0) -> None:
    v2 = v1.tag
    method41(v0, v2)
    del v2
    v4 = v0[8:].view(cp.uint8)
    del v0
    match v1:
        case US0_0(v5): # ActionSelected
            del v1
            return method55(v4, v5)
        case US0_1(v6): # PlayerChanged
            del v1
            return method56(v4, v6)
        case US0_2(): # StartGame
            del v1
            return method39(v4)
        case US0_3(): # StartTrainingVsRando
            del v1
            return method39(v4)
        case US0_4(): # StartTrainingVsSelf
            del v1
            return method39(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method57(v0 : i32) -> bool:
    v1 = v0 < 32
    del v0
    return v1
def method59(v0 : cp.ndarray) -> u32:
    v2 = v0[0:].view(cp.uint32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method60(v0 : cp.ndarray) -> i32:
    v2 = v0[4:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method61(v0 : cp.ndarray) -> None:
    del v0
    return 
def method63(v0 : cp.ndarray) -> i32:
    v2 = v0[0:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method65(v0 : cp.ndarray) -> US6:
    v1 = method63(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method61(v3)
        del v3
        return US6_0()
    elif v1 == 1:
        del v1
        method61(v3)
        del v3
        return US6_1()
    elif v1 == 2:
        del v1
        method61(v3)
        del v3
        return US6_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method64(v0 : cp.ndarray) -> Tuple[US5, bool, static_array, i32, static_array, i32]:
    v1 = method63(v0)
    v3 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method61(v3)
        v8 = US5_0()
    elif v1 == 1:
        v6 = method65(v3)
        v8 = US5_1(v6)
    else:
        raise Exception("Invalid tag.")
    del v1, v3
    v10 = v0[8:].view(cp.bool_)
    v11 = v10[0].item()
    del v10
    v13 = static_array(2)
    v14 = 0
    while method44(v14):
        v16 = u64(v14)
        v17 = v16 * 4
        del v16
        v18 = 12 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v21 = method65(v20)
        del v20
        v13[v14] = v21
        del v21
        v14 += 1 
    del v14
    v23 = v0[20:].view(cp.int32)
    v24 = v23[0].item()
    del v23
    v26 = static_array(2)
    v27 = 0
    while method44(v27):
        v29 = u64(v27)
        v30 = v29 * 4
        del v29
        v31 = 24 + v30
        del v30
        v33 = v0[v31:].view(cp.uint8)
        del v31
        v34 = method63(v33)
        del v33
        v26[v27] = v34
        del v34
        v27 += 1 
    del v27
    v36 = v0[32:].view(cp.int32)
    del v0
    v37 = v36[0].item()
    del v36
    return v8, v11, v13, v24, v26, v37
def method67(v0 : cp.ndarray) -> i32:
    v2 = v0[36:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method66(v0 : cp.ndarray) -> Tuple[US5, bool, static_array, i32, static_array, i32, US1]:
    v1 = method63(v0)
    v3 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method61(v3)
        v8 = US5_0()
    elif v1 == 1:
        v6 = method65(v3)
        v8 = US5_1(v6)
    else:
        raise Exception("Invalid tag.")
    del v1, v3
    v10 = v0[8:].view(cp.bool_)
    v11 = v10[0].item()
    del v10
    v13 = static_array(2)
    v14 = 0
    while method44(v14):
        v16 = u64(v14)
        v17 = v16 * 4
        del v16
        v18 = 12 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v21 = method65(v20)
        del v20
        v13[v14] = v21
        del v21
        v14 += 1 
    del v14
    v23 = v0[20:].view(cp.int32)
    v24 = v23[0].item()
    del v23
    v26 = static_array(2)
    v27 = 0
    while method44(v27):
        v29 = u64(v27)
        v30 = v29 * 4
        del v29
        v31 = 24 + v30
        del v30
        v33 = v0[v31:].view(cp.uint8)
        del v31
        v34 = method63(v33)
        del v33
        v26[v27] = v34
        del v34
        v27 += 1 
    del v27
    v36 = v0[32:].view(cp.int32)
    v37 = v36[0].item()
    del v36
    v38 = method67(v0)
    v40 = v0[40:].view(cp.uint8)
    del v0
    if v38 == 0:
        method61(v40)
        v45 = US1_0()
    elif v38 == 1:
        method61(v40)
        v45 = US1_1()
    elif v38 == 2:
        method61(v40)
        v45 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v38, v40
    return v8, v11, v13, v24, v26, v37, v45
def method62(v0 : cp.ndarray) -> US4:
    v1 = method63(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5, v6, v7, v8, v9, v10 = method64(v3)
        del v3
        return US4_0(v5, v6, v7, v8, v9, v10)
    elif v1 == 1:
        del v1
        method61(v3)
        del v3
        return US4_1()
    elif v1 == 2:
        del v1
        v13, v14, v15, v16, v17, v18 = method64(v3)
        del v3
        return US4_2(v13, v14, v15, v16, v17, v18)
    elif v1 == 3:
        del v1
        v20, v21, v22, v23, v24, v25, v26 = method66(v3)
        del v3
        return US4_3(v20, v21, v22, v23, v24, v25, v26)
    elif v1 == 4:
        del v1
        v28, v29, v30, v31, v32, v33 = method64(v3)
        del v3
        return US4_4(v28, v29, v30, v31, v32, v33)
    elif v1 == 5:
        del v1
        v35, v36, v37, v38, v39, v40 = method64(v3)
        del v3
        return US4_5(v35, v36, v37, v38, v39, v40)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method68(v0 : cp.ndarray) -> i32:
    v2 = v0[80:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method70(v0 : cp.ndarray) -> Tuple[i32, US1]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method60(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method61(v6)
        v11 = US1_0()
    elif v4 == 1:
        method61(v6)
        v11 = US1_1()
    elif v4 == 2:
        method61(v6)
        v11 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v11
def method71(v0 : cp.ndarray) -> Tuple[i32, US6]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method60(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method61(v6)
        v11 = US6_0()
    elif v4 == 1:
        method61(v6)
        v11 = US6_1()
    elif v4 == 2:
        method61(v6)
        v11 = US6_2()
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v11
def method72(v0 : cp.ndarray) -> Tuple[static_array, i32, i32]:
    v2 = static_array(2)
    v3 = 0
    while method44(v3):
        v5 = u64(v3)
        v6 = v5 * 4
        del v5
        v8 = v0[v6:].view(cp.uint8)
        del v6
        v9 = method65(v8)
        del v8
        v2[v3] = v9
        del v9
        v3 += 1 
    del v3
    v11 = v0[8:].view(cp.int32)
    v12 = v11[0].item()
    del v11
    v14 = v0[12:].view(cp.int32)
    del v0
    v15 = v14[0].item()
    del v14
    return v2, v12, v15
def method69(v0 : cp.ndarray) -> US8:
    v1 = method63(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5 = method65(v3)
        del v3
        return US8_0(v5)
    elif v1 == 1:
        del v1
        v7, v8 = method70(v3)
        del v3
        return US8_1(v7, v8)
    elif v1 == 2:
        del v1
        v10, v11 = method71(v3)
        del v3
        return US8_2(v10, v11)
    elif v1 == 3:
        del v1
        v13, v14, v15 = method72(v3)
        del v3
        return US8_3(v13, v14, v15)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method73(v0 : cp.ndarray) -> US2:
    v1 = method63(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method61(v3)
        del v3
        return US2_0()
    elif v1 == 1:
        del v1
        method61(v3)
        del v3
        return US2_1()
    elif v1 == 2:
        del v1
        method61(v3)
        del v3
        return US2_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method74(v0 : cp.ndarray) -> i32:
    v2 = v0[1128:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method58(v0 : cp.ndarray) -> Tuple[u32, US3, static_array_list, static_array, US7]:
    v1 = method59(v0)
    v2 = method60(v0)
    v4 = v0[16:].view(cp.uint8)
    if v2 == 0:
        method61(v4)
        v9 = US3_0()
    elif v2 == 1:
        v7 = method62(v4)
        v9 = US3_1(v7)
    else:
        raise Exception("Invalid tag.")
    del v2, v4
    v11 = static_array_list(32)
    v12 = method68(v0)
    v11.unsafe_set_length(v12)
    del v12
    v13 = v11.length
    v14 = 0
    while method5(v13, v14):
        v16 = u64(v14)
        v17 = v16 * 32
        del v16
        v18 = 96 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v21 = method69(v20)
        del v20
        v11[v14] = v21
        del v21
        v14 += 1 
    del v13, v14
    v23 = static_array(2)
    v24 = 0
    while method44(v24):
        v26 = u64(v24)
        v27 = v26 * 4
        del v26
        v28 = 1120 + v27
        del v27
        v30 = v0[v28:].view(cp.uint8)
        del v28
        v31 = method73(v30)
        del v30
        v23[v24] = v31
        del v31
        v24 += 1 
    del v24
    v32 = method74(v0)
    v34 = v0[1136:].view(cp.uint8)
    del v0
    if v32 == 0:
        method61(v34)
        v51 = US7_0()
    elif v32 == 1:
        v37, v38, v39, v40, v41, v42 = method64(v34)
        v51 = US7_1(v37, v38, v39, v40, v41, v42)
    elif v32 == 2:
        v44, v45, v46, v47, v48, v49 = method64(v34)
        v51 = US7_2(v44, v45, v46, v47, v48, v49)
    else:
        raise Exception("Invalid tag.")
    del v32, v34
    return v1, v9, v11, v23, v51
def method81(v0 : u32) -> object:
    v1 = v0
    del v0
    return v1
def method80(v0 : u32) -> object:
    return method81(v0)
def method83() -> object:
    v0 = []
    return v0
def method87(v0 : US6) -> object:
    match v0:
        case US6_0(): # Jack
            del v0
            v1 = method83()
            v2 = "Jack"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US6_1(): # King
            del v0
            v4 = method83()
            v5 = "King"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US6_2(): # Queen
            del v0
            v7 = method83()
            v8 = "Queen"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method86(v0 : US5) -> object:
    match v0:
        case US5_0(): # None
            del v0
            v1 = method83()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US5_1(v4): # Some
            del v0
            v5 = method87(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method88(v0 : bool) -> object:
    v1 = v0
    del v0
    return v1
def method89(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method44(v2):
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
        v11 = method87(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method90(v0 : i32) -> object:
    v1 = v0
    del v0
    return v1
def method91(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method44(v2):
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
        v11 = method90(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method85(v0 : US5, v1 : bool, v2 : static_array, v3 : i32, v4 : static_array, v5 : i32) -> object:
    v6 = method86(v0)
    del v0
    v7 = method88(v1)
    del v1
    v8 = method89(v2)
    del v2
    v9 = method90(v3)
    del v3
    v10 = method91(v4)
    del v4
    v11 = method90(v5)
    del v5
    v12 = {'community_card': v6, 'is_button_s_first_move': v7, 'pl_card': v8, 'player_turn': v9, 'pot': v10, 'raises_left': v11}
    del v6, v7, v8, v9, v10, v11
    return v12
def method93(v0 : US1) -> object:
    match v0:
        case US1_0(): # Call
            del v0
            v1 = method83()
            v2 = "Call"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US1_1(): # Fold
            del v0
            v4 = method83()
            v5 = "Fold"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US1_2(): # Raise
            del v0
            v7 = method83()
            v8 = "Raise"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method92(v0 : US5, v1 : bool, v2 : static_array, v3 : i32, v4 : static_array, v5 : i32, v6 : US1) -> object:
    v7 = []
    v8 = method85(v0, v1, v2, v3, v4, v5)
    del v0, v1, v2, v3, v4, v5
    v7.append(v8)
    del v8
    v9 = method93(v6)
    del v6
    v7.append(v9)
    del v9
    v10 = v7
    del v7
    return v10
def method84(v0 : US4) -> object:
    match v0:
        case US4_0(v1, v2, v3, v4, v5, v6): # ChanceCommunityCard
            del v0
            v7 = method85(v1, v2, v3, v4, v5, v6)
            del v1, v2, v3, v4, v5, v6
            v8 = "ChanceCommunityCard"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US4_1(): # ChanceInit
            del v0
            v10 = method83()
            v11 = "ChanceInit"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US4_2(v13, v14, v15, v16, v17, v18): # Round
            del v0
            v19 = method85(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "Round"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case US4_3(v22, v23, v24, v25, v26, v27, v28): # RoundWithAction
            del v0
            v29 = method92(v22, v23, v24, v25, v26, v27, v28)
            del v22, v23, v24, v25, v26, v27, v28
            v30 = "RoundWithAction"
            v31 = [v30,v29]
            del v29, v30
            return v31
        case US4_4(v32, v33, v34, v35, v36, v37): # TerminalCall
            del v0
            v38 = method85(v32, v33, v34, v35, v36, v37)
            del v32, v33, v34, v35, v36, v37
            v39 = "TerminalCall"
            v40 = [v39,v38]
            del v38, v39
            return v40
        case US4_5(v41, v42, v43, v44, v45, v46): # TerminalFold
            del v0
            v47 = method85(v41, v42, v43, v44, v45, v46)
            del v41, v42, v43, v44, v45, v46
            v48 = "TerminalFold"
            v49 = [v48,v47]
            del v47, v48
            return v49
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method82(v0 : US3) -> object:
    match v0:
        case US3_0(): # None
            del v0
            v1 = method83()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US3_1(v4): # Some
            del v0
            v5 = method84(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method79(v0 : u32, v1 : US3) -> object:
    v2 = method80(v0)
    del v0
    v3 = method82(v1)
    del v1
    v4 = {'deck': v2, 'game': v3}
    del v2, v3
    return v4
def method97(v0 : i32, v1 : US1) -> object:
    v2 = []
    v3 = method90(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method93(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method98(v0 : i32, v1 : US6) -> object:
    v2 = []
    v3 = method90(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method87(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method99(v0 : static_array, v1 : i32, v2 : i32) -> object:
    v3 = method89(v0)
    del v0
    v4 = method90(v1)
    del v1
    v5 = method90(v2)
    del v2
    v6 = {'cards_shown': v3, 'chips_won': v4, 'winner_id': v5}
    del v3, v4, v5
    return v6
def method96(v0 : US8) -> object:
    match v0:
        case US8_0(v1): # CommunityCardIs
            del v0
            v2 = method87(v1)
            del v1
            v3 = "CommunityCardIs"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US8_1(v5, v6): # PlayerAction
            del v0
            v7 = method97(v5, v6)
            del v5, v6
            v8 = "PlayerAction"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US8_2(v10, v11): # PlayerGotCard
            del v0
            v12 = method98(v10, v11)
            del v10, v11
            v13 = "PlayerGotCard"
            v14 = [v13,v12]
            del v12, v13
            return v14
        case US8_3(v15, v16, v17): # Showdown
            del v0
            v18 = method99(v15, v16, v17)
            del v15, v16, v17
            v19 = "Showdown"
            v20 = [v19,v18]
            del v18, v19
            return v20
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method95(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method5(v2, v3):
        v6 = v0[v3]
        v7 = method96(v6)
        del v6
        v1.append(v7)
        del v7
        v3 += 1 
    del v0, v2, v3
    return v1
def method101(v0 : US2) -> object:
    match v0:
        case US2_0(): # Computer
            del v0
            v1 = method83()
            v2 = "Computer"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US2_1(): # Human
            del v0
            v4 = method83()
            v5 = "Human"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US2_2(): # Random
            del v0
            v7 = method83()
            v8 = "Random"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method100(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method44(v2):
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
        v11 = method101(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method102(v0 : US7) -> object:
    match v0:
        case US7_0(): # GameNotStarted
            del v0
            v1 = method83()
            v2 = "GameNotStarted"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US7_1(v4, v5, v6, v7, v8, v9): # GameOver
            del v0
            v10 = method85(v4, v5, v6, v7, v8, v9)
            del v4, v5, v6, v7, v8, v9
            v11 = "GameOver"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US7_2(v13, v14, v15, v16, v17, v18): # WaitingForActionFromPlayerId
            del v0
            v19 = method85(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "WaitingForActionFromPlayerId"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method94(v0 : static_array_list, v1 : static_array, v2 : US7) -> object:
    v3 = method95(v0)
    del v0
    v4 = method100(v1)
    del v1
    v5 = method102(v2)
    del v2
    v6 = {'messages': v3, 'pl_type': v4, 'ui_game_state': v5}
    del v3, v4, v5
    return v6
def method78(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7) -> object:
    v5 = method79(v0, v1)
    del v0, v1
    v6 = method94(v2, v3, v4)
    del v2, v3, v4
    v7 = {'private': v5, 'public': v6}
    del v5, v6
    return v7
def method108(v0 : cp.ndarray) -> object:
    v1 = v0
    del v0
    return v1
def method107(v0 : cp.ndarray) -> object:
    return method108(v0)
def method106(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    v3 = []
    v4 = method107(v0)
    del v0
    v3.append(v4)
    del v4
    v5 = method107(v1)
    del v1
    v3.append(v5)
    del v5
    v6 = method107(v2)
    del v2
    v3.append(v6)
    del v6
    v7 = method83()
    v3.append(v7)
    del v7
    v8 = v3
    del v3
    return v8
def method105(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    return method106(v0, v1, v2)
def method104(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    return method105(v0, v1, v2)
def method103(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    v3 = method104(v0, v1, v2)
    del v0, v1, v2
    v4 = {'model_ptrs': v3}
    del v3
    return v4
def method77(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray) -> object:
    v8 = method78(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    v9 = method103(v5, v6, v7)
    del v5, v6, v7
    v10 = {'game': v8, 'neural': v9}
    del v8, v9
    return v10
def method113(v0 : f32) -> object:
    v1 = v0
    del v0
    return v1
def method112(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method5(v2, v3):
        v5 = v0[v3]
        v6 = method113(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method111(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method5(v2, v3):
        v5 = v0[v3]
        v6 = method112(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method110(v0 : US9) -> object:
    match v0:
        case US9_0(v1): # AddRewardsRando
            del v0
            v2 = method111(v1)
            del v1
            v3 = "AddRewardsRando"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US9_1(v5): # AddRewardsSelf
            del v0
            v6 = method111(v5)
            del v5
            v7 = "AddRewardsSelf"
            v8 = [v7,v6]
            del v6, v7
            return v8
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method109(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method5(v2, v3):
        v5 = v0[v3]
        v6 = method110(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method76(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : list) -> object:
    v9 = []
    v10 = method77(v0, v1, v2, v3, v4, v5, v6, v7)
    del v0, v1, v2, v3, v4, v5, v6, v7
    v9.append(v10)
    del v10
    v11 = method109(v8)
    del v8
    v9.append(v11)
    del v11
    v12 = v9
    del v9
    return v12
def method75(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : list) -> object:
    v9 = method76(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8
    return v9
def method114(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray) -> object:
    v8 = method77(v0, v1, v2, v3, v4, v5, v6, v7)
    del v0, v1, v2, v3, v4, v5, v6, v7
    return v8
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
