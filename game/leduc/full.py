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
__device__ void block_row_map_24(float * v0, int v1, float * v2);
struct Tuple8;
struct Tuple9;
struct Tuple10;
struct Tuple11;
struct Union12;
struct Union13;
__device__ int tag_26(Union6 v0);
__device__ bool is_pair_27(int v0, int v1);
__device__ Tuple7 order_28(int v0, int v1);
__device__ Union13 compare_hands_25(Union5 v0, bool v1, static_array<Union6,2> v2, int v3, static_array<int,2> v4, int v5);
__device__ void f_30(unsigned char * v0, unsigned int v1);
__device__ void f_31(unsigned char * v0, int v1);
__device__ void f_32(unsigned char * v0);
__device__ void f_34(unsigned char * v0, int v1);
__device__ void f_36(unsigned char * v0, Union6 v1);
__device__ void f_35(unsigned char * v0, Union5 v1, bool v2, static_array<Union6,2> v3, int v4, static_array<int,2> v5, int v6);
__device__ void f_38(unsigned char * v0, int v1);
__device__ void f_37(unsigned char * v0, Union5 v1, bool v2, static_array<Union6,2> v3, int v4, static_array<int,2> v5, int v6, Union1 v7);
__device__ void f_33(unsigned char * v0, Union4 v1);
__device__ void f_39(unsigned char * v0, int v1);
__device__ void f_41(unsigned char * v0, int v1, Union1 v2);
__device__ void f_42(unsigned char * v0, int v1, Union6 v2);
__device__ void f_43(unsigned char * v0, static_array<Union6,2> v1, int v2, int v3);
__device__ void f_40(unsigned char * v0, Union7 v1);
__device__ void f_44(unsigned char * v0, Union2 v1);
__device__ void f_45(unsigned char * v0, int v1);
__device__ void f_29(unsigned char * v0, unsigned int v1, Union3 v2, static_array_list<Union7,32> v3, static_array<Union2,2> v4, Union8 v5);
struct StackMut1;
struct Union14;
__device__ void method_46(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut1 & v3, int v4, Union4 v5);
struct Tuple12;
__device__ void method_47(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut1 & v3, Union4 v4);
__device__ void method_48(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut1 & v3, Union4 v4);
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
__device__ void block_row_map_24(float * v0, int v1, float * v2){
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
    while (while_method_8(v23)){
        assert("Tensor range check" && 0 <= v23 && v23 < 16);
        int v25;
        v25 = 1024 * v23;
        int v26;
        v26 = v25 + v20;
        float v27[4];
        int v28[4];
        int v29;
        v29 = 0;
        while (while_method_4(v29)){
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
        while (while_method_4(v36)){
            int v38;
            v38 = 0;
            while (while_method_7(v38)){
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
        while (while_method_4(v73)){
            int v75;
            v75 = 0;
            while (while_method_7(v75)){
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
        while (while_method_4(v83)){
            int v85;
            v85 = 0;
            while (while_method_7(v85)){
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
        while (while_method_4(v93)){
            int v95;
            v95 = 0;
            while (while_method_7(v95)){
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
        while (while_method_4(v108)){
            int v110;
            v110 = 0;
            while (while_method_7(v110)){
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
        while (while_method_4(v117)){
            int v119;
            v119 = 0;
            while (while_method_7(v119)){
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
        while (while_method_4(v134)){
            int v136;
            v136 = 0;
            while (while_method_7(v136)){
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
        while (while_method_4(v146)){
            int v148;
            v148 = 0;
            while (while_method_7(v148)){
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
        while (while_method_4(v160)){
            int v162;
            v162 = 0;
            while (while_method_7(v162)){
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
        while (while_method_4(v169)){
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
__device__ int tag_26(Union6 v0){
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
__device__ bool is_pair_27(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
__device__ Tuple7 order_28(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple7{v1, v0};
    } else {
        return Tuple7{v0, v1};
    }
}
__device__ Union13 compare_hands_25(Union5 v0, bool v1, static_array<Union6,2> v2, int v3, static_array<int,2> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            __trap();
            break;
        }
        case 1: { // Some
            Union6 v7 = v0.case1.v0;
            int v8;
            v8 = tag_26(v7);
            Union6 v9;
            v9 = v2[0];
            int v11;
            v11 = tag_26(v9);
            Union6 v12;
            v12 = v2[1];
            int v14;
            v14 = tag_26(v12);
            bool v15;
            v15 = is_pair_27(v8, v11);
            bool v16;
            v16 = is_pair_27(v8, v14);
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
                    Tuple7 tmp35 = order_28(v8, v11);
                    v27 = tmp35.v0; v28 = tmp35.v1;
                    int v29; int v30;
                    Tuple7 tmp36 = order_28(v8, v14);
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
__device__ void f_30(unsigned char * v0, unsigned int v1){
    unsigned int * v2;
    v2 = (unsigned int *)(v0+0ull);
    v2[0] = v1;
    return ;
}
__device__ void f_31(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+4ull);
    v2[0] = v1;
    return ;
}
__device__ void f_32(unsigned char * v0){
    return ;
}
__device__ void f_34(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+0ull);
    v2[0] = v1;
    return ;
}
__device__ void f_36(unsigned char * v0, Union6 v1){
    int v2;
    v2 = v1.tag;
    f_34(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // Jack
            return f_32(v3);
            break;
        }
        case 1: { // King
            return f_32(v3);
            break;
        }
        case 2: { // Queen
            return f_32(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_35(unsigned char * v0, Union5 v1, bool v2, static_array<Union6,2> v3, int v4, static_array<int,2> v5, int v6){
    int v7;
    v7 = v1.tag;
    f_34(v0, v7);
    unsigned char * v8;
    v8 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // None
            f_32(v8);
            break;
        }
        case 1: { // Some
            Union6 v10 = v1.case1.v0;
            f_36(v8, v10);
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
        f_36(v18, v25);
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
        f_34(v34, v41);
        v29 += 1 ;
    }
    int * v43;
    v43 = (int *)(v0+32ull);
    v43[0] = v6;
    return ;
}
__device__ void f_38(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+36ull);
    v2[0] = v1;
    return ;
}
__device__ void f_37(unsigned char * v0, Union5 v1, bool v2, static_array<Union6,2> v3, int v4, static_array<int,2> v5, int v6, Union1 v7){
    int v8;
    v8 = v1.tag;
    f_34(v0, v8);
    unsigned char * v9;
    v9 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // None
            f_32(v9);
            break;
        }
        case 1: { // Some
            Union6 v11 = v1.case1.v0;
            f_36(v9, v11);
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
        f_36(v19, v26);
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
        f_34(v35, v42);
        v30 += 1 ;
    }
    int * v44;
    v44 = (int *)(v0+32ull);
    v44[0] = v6;
    int v46;
    v46 = v7.tag;
    f_38(v0, v46);
    unsigned char * v47;
    v47 = (unsigned char *)(v0+40ull);
    switch (v7.tag) {
        case 0: { // Call
            return f_32(v47);
            break;
        }
        case 1: { // Fold
            return f_32(v47);
            break;
        }
        case 2: { // Raise
            return f_32(v47);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_33(unsigned char * v0, Union4 v1){
    int v2;
    v2 = v1.tag;
    f_34(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // ChanceCommunityCard
            Union5 v5 = v1.case0.v0; bool v6 = v1.case0.v1; static_array<Union6,2> v7 = v1.case0.v2; int v8 = v1.case0.v3; static_array<int,2> v9 = v1.case0.v4; int v10 = v1.case0.v5;
            return f_35(v3, v5, v6, v7, v8, v9, v10);
            break;
        }
        case 1: { // ChanceInit
            return f_32(v3);
            break;
        }
        case 2: { // Round
            Union5 v11 = v1.case2.v0; bool v12 = v1.case2.v1; static_array<Union6,2> v13 = v1.case2.v2; int v14 = v1.case2.v3; static_array<int,2> v15 = v1.case2.v4; int v16 = v1.case2.v5;
            return f_35(v3, v11, v12, v13, v14, v15, v16);
            break;
        }
        case 3: { // RoundWithAction
            Union5 v17 = v1.case3.v0; bool v18 = v1.case3.v1; static_array<Union6,2> v19 = v1.case3.v2; int v20 = v1.case3.v3; static_array<int,2> v21 = v1.case3.v4; int v22 = v1.case3.v5; Union1 v23 = v1.case3.v6;
            return f_37(v3, v17, v18, v19, v20, v21, v22, v23);
            break;
        }
        case 4: { // TerminalCall
            Union5 v24 = v1.case4.v0; bool v25 = v1.case4.v1; static_array<Union6,2> v26 = v1.case4.v2; int v27 = v1.case4.v3; static_array<int,2> v28 = v1.case4.v4; int v29 = v1.case4.v5;
            return f_35(v3, v24, v25, v26, v27, v28, v29);
            break;
        }
        case 5: { // TerminalFold
            Union5 v30 = v1.case5.v0; bool v31 = v1.case5.v1; static_array<Union6,2> v32 = v1.case5.v2; int v33 = v1.case5.v3; static_array<int,2> v34 = v1.case5.v4; int v35 = v1.case5.v5;
            return f_35(v3, v30, v31, v32, v33, v34, v35);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_39(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+80ull);
    v2[0] = v1;
    return ;
}
__device__ void f_41(unsigned char * v0, int v1, Union1 v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0] = v1;
    int v5;
    v5 = v2.tag;
    f_31(v0, v5);
    unsigned char * v6;
    v6 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // Call
            return f_32(v6);
            break;
        }
        case 1: { // Fold
            return f_32(v6);
            break;
        }
        case 2: { // Raise
            return f_32(v6);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_42(unsigned char * v0, int v1, Union6 v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0] = v1;
    int v5;
    v5 = v2.tag;
    f_31(v0, v5);
    unsigned char * v6;
    v6 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // Jack
            return f_32(v6);
            break;
        }
        case 1: { // King
            return f_32(v6);
            break;
        }
        case 2: { // Queen
            return f_32(v6);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_43(unsigned char * v0, static_array<Union6,2> v1, int v2, int v3){
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
        f_36(v8, v15);
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
__device__ void f_40(unsigned char * v0, Union7 v1){
    int v2;
    v2 = v1.tag;
    f_34(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // CommunityCardIs
            Union6 v5 = v1.case0.v0;
            return f_36(v3, v5);
            break;
        }
        case 1: { // PlayerAction
            int v6 = v1.case1.v0; Union1 v7 = v1.case1.v1;
            return f_41(v3, v6, v7);
            break;
        }
        case 2: { // PlayerGotCard
            int v8 = v1.case2.v0; Union6 v9 = v1.case2.v1;
            return f_42(v3, v8, v9);
            break;
        }
        case 3: { // Showdown
            static_array<Union6,2> v10 = v1.case3.v0; int v11 = v1.case3.v1; int v12 = v1.case3.v2;
            return f_43(v3, v10, v11, v12);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_44(unsigned char * v0, Union2 v1){
    int v2;
    v2 = v1.tag;
    f_34(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // Computer
            return f_32(v3);
            break;
        }
        case 1: { // Human
            return f_32(v3);
            break;
        }
        case 2: { // Random
            return f_32(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_45(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+1128ull);
    v2[0] = v1;
    return ;
}
__device__ void f_29(unsigned char * v0, unsigned int v1, Union3 v2, static_array_list<Union7,32> v3, static_array<Union2,2> v4, Union8 v5){
    f_30(v0, v1);
    int v6;
    v6 = v2.tag;
    f_31(v0, v6);
    unsigned char * v7;
    v7 = (unsigned char *)(v0+16ull);
    switch (v2.tag) {
        case 0: { // None
            f_32(v7);
            break;
        }
        case 1: { // Some
            Union4 v9 = v2.case1.v0;
            f_33(v7, v9);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v10;
    v10 = v3.length;
    f_39(v0, v10);
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
        f_40(v17, v19);
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
        f_44(v26, v33);
        v21 += 1 ;
    }
    int v35;
    v35 = v5.tag;
    f_45(v0, v35);
    unsigned char * v36;
    v36 = (unsigned char *)(v0+1136ull);
    switch (v5.tag) {
        case 0: { // GameNotStarted
            return f_32(v36);
            break;
        }
        case 1: { // GameOver
            Union5 v38 = v5.case1.v0; bool v39 = v5.case1.v1; static_array<Union6,2> v40 = v5.case1.v2; int v41 = v5.case1.v3; static_array<int,2> v42 = v5.case1.v4; int v43 = v5.case1.v5;
            return f_35(v36, v38, v39, v40, v41, v42, v43);
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            Union5 v44 = v5.case2.v0; bool v45 = v5.case2.v1; static_array<Union6,2> v46 = v5.case2.v2; int v47 = v5.case2.v3; static_array<int,2> v48 = v5.case2.v4; int v49 = v5.case2.v5;
            return f_35(v36, v44, v45, v46, v47, v48, v49);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ inline bool while_method_9(int v0){
    bool v1;
    v1 = v0 < 32;
    return v1;
}
__device__ inline bool while_method_10(Union3 v0){
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
__device__ void method_46(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut1 & v3, int v4, Union4 v5){
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
    while (while_method_10(v20)){
        Union3 v788;
        switch (v20.tag) {
            case 0: { // None
                v788 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v22 = v20.case1.v0;
                Union14 v628;
                switch (v22.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v597 = v22.case0.v0; bool v598 = v22.case0.v1; static_array<Union6,2> v599 = v22.case0.v2; int v600 = v22.case0.v3; static_array<int,2> v601 = v22.case0.v4; int v602 = v22.case0.v5;
                        curandStatePhilox4_32_10_t & v603 = v3.v5;
                        curandStatePhilox4_32_10_t & v604 = v603;
                        unsigned int & v605 = v3.v0;
                        Union6 v606; unsigned int v607;
                        Tuple6 tmp37 = draw_card_20(v604, v605);
                        v606 = tmp37.v0; v607 = tmp37.v1;
                        v3.v0 = v607;
                        Union7 v608;
                        v608 = Union7{Union7_0{v606}};
                        v18.push(v608);
                        v628 = Union14{Union14_0{v597, v598, v599, v600, v601, v602, v606}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v610 = v3.v5;
                        curandStatePhilox4_32_10_t & v611 = v610;
                        unsigned int & v612 = v3.v0;
                        Union6 v613; unsigned int v614;
                        Tuple6 tmp38 = draw_card_20(v611, v612);
                        v613 = tmp38.v0; v614 = tmp38.v1;
                        v3.v0 = v614;
                        curandStatePhilox4_32_10_t & v615 = v3.v5;
                        curandStatePhilox4_32_10_t & v616 = v615;
                        unsigned int & v617 = v3.v0;
                        Union6 v618; unsigned int v619;
                        Tuple6 tmp39 = draw_card_20(v616, v617);
                        v618 = tmp39.v0; v619 = tmp39.v1;
                        v3.v0 = v619;
                        Union7 v620;
                        v620 = Union7{Union7_2{0, v613}};
                        v18.push(v620);
                        Union7 v621;
                        v621 = Union7{Union7_2{1, v618}};
                        v18.push(v621);
                        v628 = Union14{Union14_1{v613, v618}};
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
                        Union1 v585;
                        switch (v84.tag) {
                            case 0: { // Computer
                                static_array_list<Union7,32> & v87 = v3.v2;
                                curandStatePhilox4_32_10_t & v88 = v3.v5;
                                curandStatePhilox4_32_10_t & v89 = v88;
                                float * v90;
                                v90 = reinterpret_cast<float *>(&v1[4718592ull]);
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
                                while (while_method_7(v159)){
                                    float * v161;
                                    v161 = reinterpret_cast<float *>(&v1[4718592ull]);
                                    assert("Tensor range check" && 0 <= v159 && v159 < 4);
                                    int v163;
                                    v163 = 393216 * v159;
                                    float * v164;
                                    v164 = reinterpret_cast<float *>(&v1[0ull]);
                                    float * v166;
                                    v166 = reinterpret_cast<float *>(&v0[0ull]);
                                    assert("Tensor range check" && 0 <= v159 && v159 < 4);
                                    int v168;
                                    v168 = 8192 * v159;
                                    float * v169;
                                    v169 = reinterpret_cast<float *>(&v1[3145728ull]);
                                    block_matmul_23(v169, v166, v168, v164);
                                    block_row_map_24(v161, v163, v169);
                                    int * v171;
                                    v171 = reinterpret_cast<int *>(&v0[131072ull]);
                                    float * v173;
                                    v173 = reinterpret_cast<float *>(&v0[131088ull]);
                                    float * v175;
                                    v175 = reinterpret_cast<float *>(&v0[131104ull]);
                                    double * v177;
                                    v177 = reinterpret_cast<double *>(&v1[11010048ull]);
                                    double * v179;
                                    v179 = reinterpret_cast<double *>(&v1[11403264ull]);
                                    v159 += 1 ;
                                }
                                __syncthreads();
                                int * v181;
                                v181 = reinterpret_cast<int *>(&v0[131072ull]);
                                float * v183;
                                v183 = reinterpret_cast<float *>(&v0[131088ull]);
                                float * v185;
                                v185 = reinterpret_cast<float *>(&v0[131104ull]);
                                int v187;
                                v187 = v181[0];
                                float * v188;
                                v188 = reinterpret_cast<float *>(&v1[4718592ull]);
                                assert("Tensor range check" && 0 <= v187 && v187 < 4);
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
                                        Tuple8 tmp40 = Tuple8{0, 0.0f};
                                        v293 = tmp40.v0; v294 = tmp40.v1;
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
                                        Tuple8 tmp41 = Tuple8{0, v309};
                                        v310 = tmp41.v0; v311 = tmp41.v1;
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
                                    Tuple9 tmp42 = Tuple9{-1.0f / 0.0f, false};
                                    v328 = tmp42.v0; v329 = tmp42.v1;
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
                                    Tuple9 tmp43 = cooperative_groups::reduce(v349, Tuple9{v328, v329}, v350);
                                    v351 = tmp43.v0; v352 = tmp43.v1;
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
                                            v364 = curand_uniform(&v89);
                                            assert("Tensor range check" && 0 <= v357 && v357 < 1);
                                            assert("Tensor range check" && 0 <= v359 && v359 < 4);
                                            v355[v362] = v364;
                                            v356[v362] = v363;
                                            v359 += 1 ;
                                        }
                                        v357 += 1 ;
                                    }
                                    float v365; int v366;
                                    Tuple10 tmp44 = Tuple10{0.0f, 2147483647};
                                    v365 = tmp44.v0; v366 = tmp44.v1;
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
                                    Tuple10 tmp45 = cooperative_groups::reduce(v381, Tuple10{v365, v366}, v382);
                                    v383 = tmp45.v0; v384 = tmp45.v1;
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
                                    Tuple11 tmp46 = Tuple11{2147483647, false};
                                    v401 = tmp46.v0; v402 = tmp46.v1;
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
                                    Tuple11 tmp47 = cooperative_groups::reduce(v422, Tuple11{v401, v402}, v423);
                                    v424 = tmp47.v0; v425 = tmp47.v1;
                                    bool v426;
                                    v426 = v425 == false;
                                    if (v426){
                                        assert("The local reduce must be true." && v425);
                                    } else {
                                    }
                                    float v428; int v429;
                                    Tuple10 tmp48 = Tuple10{0.0f, 2147483647};
                                    v428 = tmp48.v0; v429 = tmp48.v1;
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
                                    Tuple10 tmp49 = cooperative_groups::reduce(v447, Tuple10{v428, v429}, v448);
                                    v449 = tmp49.v0; v450 = tmp49.v1;
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
                                v468 = reinterpret_cast<double *>(&v1[11010048ull]);
                                double * v470;
                                v470 = reinterpret_cast<double *>(&v1[11403264ull]);
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
                                while (while_method_7(v476)){
                                    float * v478;
                                    v478 = reinterpret_cast<float *>(&v1[4718592ull]);
                                    int v480;
                                    v480 = blockIdx.x;
                                    int v481;
                                    v481 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v476 && v476 < 4);
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
                                    assert("Tensor range check" && 0 <= v476 && v476 < 4);
                                    assert("Tensor range check" && 0 <= v475 && v475 < 6144);
                                    assert("Tensor range check" && 0 <= v75 && v75 < 2);
                                    int v493;
                                    v493 = 2 * v475;
                                    int v494;
                                    v494 = v493 + v75;
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
                                    assert("Tensor range check" && 0 <= v476 && v476 < 4);
                                    assert("Tensor range check" && 0 <= v475 && v475 < 6144);
                                    assert("Tensor range check" && 0 <= v75 && v75 < 2);
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
                                        v511 = v76[0];
                                        int v513; int v514;
                                        Tuple7 tmp50 = Tuple7{1, v511};
                                        v513 = tmp50.v0; v514 = tmp50.v1;
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
                                            v521 = v76[v513];
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
                                        if (v79){
                                            bool v525;
                                            v525 = v75 < 2;
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
                                        v529 = v76[v75];
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
                                        v536 = v77 > 0;
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
                                v549 = v76[0];
                                int v551;
                                v551 = v76[1];
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
                                v556 = v77 > 0;
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
                        v586 = Union7{Union7_1{v75, v585}};
                        v18.push(v586);
                        v628 = Union14{Union14_2{v72, v73, v74, v75, v76, v77, v585}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v588 = v22.case3.v0; bool v589 = v22.case3.v1; static_array<Union6,2> v590 = v22.case3.v2; int v591 = v22.case3.v3; static_array<int,2> v592 = v22.case3.v4; int v593 = v22.case3.v5; Union1 v594 = v22.case3.v6;
                        Union7 v595;
                        v595 = Union7{Union7_1{v591, v594}};
                        v18.push(v595);
                        v628 = Union14{Union14_2{v588, v589, v590, v591, v592, v593, v594}};
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
                        v56 = compare_hands_25(v43, v44, v45, v46, v47, v48);
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
                        v628 = Union14{Union14_3{}};
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
                        Tuple7 tmp51 = Tuple7{0, 0};
                        v638 = tmp51.v0; v639 = tmp51.v1;
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
                                            Tuple7 tmp52 = Tuple7{0, 0};
                                            v743 = tmp52.v0; v744 = tmp52.v1;
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
                                            Tuple7 tmp53 = Tuple7{0, 0};
                                            v678 = tmp53.v0; v679 = tmp53.v1;
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
                                            Tuple7 tmp54 = Tuple7{0, 0};
                                            v699 = tmp54.v0; v700 = tmp54.v1;
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
        v20 = v788;
    }
    return ;
}
__device__ void method_47(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut1 & v3, Union4 v4){
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
    while (while_method_10(v16)){
        Union3 v784;
        switch (v16.tag) {
            case 0: { // None
                v784 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v18 = v16.case1.v0;
                Union14 v624;
                switch (v18.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v593 = v18.case0.v0; bool v594 = v18.case0.v1; static_array<Union6,2> v595 = v18.case0.v2; int v596 = v18.case0.v3; static_array<int,2> v597 = v18.case0.v4; int v598 = v18.case0.v5;
                        curandStatePhilox4_32_10_t & v599 = v3.v5;
                        curandStatePhilox4_32_10_t & v600 = v599;
                        unsigned int & v601 = v3.v0;
                        Union6 v602; unsigned int v603;
                        Tuple6 tmp59 = draw_card_20(v600, v601);
                        v602 = tmp59.v0; v603 = tmp59.v1;
                        v3.v0 = v603;
                        Union7 v604;
                        v604 = Union7{Union7_0{v602}};
                        v14.push(v604);
                        v624 = Union14{Union14_0{v593, v594, v595, v596, v597, v598, v602}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v606 = v3.v5;
                        curandStatePhilox4_32_10_t & v607 = v606;
                        unsigned int & v608 = v3.v0;
                        Union6 v609; unsigned int v610;
                        Tuple6 tmp60 = draw_card_20(v607, v608);
                        v609 = tmp60.v0; v610 = tmp60.v1;
                        v3.v0 = v610;
                        curandStatePhilox4_32_10_t & v611 = v3.v5;
                        curandStatePhilox4_32_10_t & v612 = v611;
                        unsigned int & v613 = v3.v0;
                        Union6 v614; unsigned int v615;
                        Tuple6 tmp61 = draw_card_20(v612, v613);
                        v614 = tmp61.v0; v615 = tmp61.v1;
                        v3.v0 = v615;
                        Union7 v616;
                        v616 = Union7{Union7_2{0, v609}};
                        v14.push(v616);
                        Union7 v617;
                        v617 = Union7{Union7_2{1, v614}};
                        v14.push(v617);
                        v624 = Union14{Union14_1{v609, v614}};
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
                        Union1 v581;
                        switch (v80.tag) {
                            case 0: { // Computer
                                static_array_list<Union7,32> & v83 = v3.v2;
                                curandStatePhilox4_32_10_t & v84 = v3.v5;
                                curandStatePhilox4_32_10_t & v85 = v84;
                                float * v86;
                                v86 = reinterpret_cast<float *>(&v1[4718592ull]);
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
                                while (while_method_7(v155)){
                                    float * v157;
                                    v157 = reinterpret_cast<float *>(&v1[4718592ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 4);
                                    int v159;
                                    v159 = 393216 * v155;
                                    float * v160;
                                    v160 = reinterpret_cast<float *>(&v1[0ull]);
                                    float * v162;
                                    v162 = reinterpret_cast<float *>(&v0[0ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 4);
                                    int v164;
                                    v164 = 8192 * v155;
                                    float * v165;
                                    v165 = reinterpret_cast<float *>(&v1[3145728ull]);
                                    block_matmul_23(v165, v162, v164, v160);
                                    block_row_map_24(v157, v159, v165);
                                    int * v167;
                                    v167 = reinterpret_cast<int *>(&v0[131072ull]);
                                    float * v169;
                                    v169 = reinterpret_cast<float *>(&v0[131088ull]);
                                    float * v171;
                                    v171 = reinterpret_cast<float *>(&v0[131104ull]);
                                    double * v173;
                                    v173 = reinterpret_cast<double *>(&v1[11010048ull]);
                                    double * v175;
                                    v175 = reinterpret_cast<double *>(&v1[11403264ull]);
                                    v155 += 1 ;
                                }
                                __syncthreads();
                                int * v177;
                                v177 = reinterpret_cast<int *>(&v0[131072ull]);
                                float * v179;
                                v179 = reinterpret_cast<float *>(&v0[131088ull]);
                                float * v181;
                                v181 = reinterpret_cast<float *>(&v0[131104ull]);
                                int v183;
                                v183 = v177[0];
                                float * v184;
                                v184 = reinterpret_cast<float *>(&v1[4718592ull]);
                                assert("Tensor range check" && 0 <= v183 && v183 < 4);
                                int v186;
                                v186 = 393216 * v183;
                                int v187;
                                v187 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v187 && v187 < 24);
                                int v188;
                                v188 = 16384 * v187;
                                int v189;
                                v189 = v188 + v186;
                                int v190;
                                v190 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v190 && v190 < 256);
                                int v191;
                                v191 = 64 * v190;
                                int v192;
                                v192 = v191 + v189;
                                float * v193;
                                v193 = v184+v192;
                                int v195;
                                v195 = sizeof(float *);
                                unsigned long long v196;
                                v196 = (unsigned long long)v195;
                                unsigned long long v197;
                                v197 = 256ull * v196;
                                unsigned long long v198;
                                v198 = v197 + 16ull;
                                unsigned long long v199;
                                v199 = v198 - 1ull;
                                unsigned long long v200;
                                v200 = v199 % 16ull;
                                unsigned long long v201;
                                v201 = v199 - v200;
                                unsigned long long v202;
                                v202 = v201 + 1024ull;
                                unsigned long long v203;
                                v203 = v202 + 16ull;
                                unsigned long long v204;
                                v204 = v203 - 1ull;
                                unsigned long long v205;
                                v205 = v204 % 16ull;
                                unsigned long long v206;
                                v206 = v204 - v205;
                                unsigned long long v207;
                                v207 = v206 + 1024ull;
                                bool v208;
                                v208 = v207 <= 98304ull;
                                bool v209;
                                v209 = v208 == false;
                                if (v209){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v208);
                                } else {
                                }
                                extern __shared__ unsigned char v211[];
                                bool v212;
                                v212 = v207 <= v207;
                                bool v213;
                                v213 = v212 == false;
                                if (v213){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v212);
                                } else {
                                }
                                float * * v215;
                                v215 = reinterpret_cast<float * *>(&v211[0ull]);
                                float * v217;
                                v217 = reinterpret_cast<float *>(&v211[v201]);
                                int * v219;
                                v219 = reinterpret_cast<int *>(&v211[v206]);
                                int v221;
                                v221 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v221 && v221 < 256);
                                v215[v221] = v193;
                                __syncthreads();
                                bool v222;
                                v222 = 0 <= v221;
                                bool v223;
                                v223 = v222 == false;
                                if (v223){
                                    assert("The index needs to be zero or positive." && v222);
                                } else {
                                }
                                int v225;
                                v225 = v221 % 16;
                                int v226;
                                v226 = v221 / 16;
                                bool v227;
                                v227 = v226 < 16;
                                bool v228;
                                v228 = v227 == false;
                                if (v228){
                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v227);
                                } else {
                                }
                                assert("Tensor range check" && 0 <= v226 && v226 < 16);
                                int v230;
                                v230 = 0;
                                while (while_method_8(v230)){
                                    bool v232;
                                    v232 = 0 <= v226;
                                    bool v233;
                                    v233 = v232 && v227;
                                    bool v234;
                                    v234 = v233 == false;
                                    if (v234){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v233);
                                    } else {
                                    }
                                    bool v236;
                                    v236 = 0 <= v230;
                                    bool v238;
                                    if (v236){
                                        bool v237;
                                        v237 = v230 < 16;
                                        v238 = v237;
                                    } else {
                                        v238 = false;
                                    }
                                    bool v239;
                                    v239 = v238 == false;
                                    if (v239){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v238);
                                    } else {
                                    }
                                    int v241;
                                    v241 = v230 * 16;
                                    int v242;
                                    v242 = v241 + v226;
                                    assert("Tensor range check" && 0 <= v230 && v230 < 16);
                                    int v243;
                                    v243 = 16 * v230;
                                    int v244;
                                    v244 = v243 + v226;
                                    float * v245;
                                    v245 = v215[v244];
                                    int v246;
                                    v246 = blockIdx.x;
                                    int v247;
                                    v247 = v246 * 256;
                                    int v248;
                                    v248 = v247 + v242;
                                    assert("Tensor range check" && 0 <= v225 && v225 < 16);
                                    int v249;
                                    v249 = 4 * v225;
                                    float v250[4];
                                    int v251[4];
                                    int v252;
                                    v252 = 0;
                                    while (while_method_4(v252)){
                                        assert("Tensor range check" && 0 <= v252 && v252 < 1);
                                        int v254;
                                        v254 = 4 * v252;
                                        assert("Tensor range check" && 0 <= v252 && v252 < 1);
                                        int v255;
                                        v255 = 64 * v252;
                                        int v256;
                                        v256 = v255 + v249;
                                        int4* v257;
                                        v257 = reinterpret_cast<int4*>(v245 + v256);
                                        int4* v258;
                                        v258 = reinterpret_cast<int4*>(v250 + v254);
                                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v257) % 16 == 0 && reinterpret_cast<unsigned long long>(v258) % 16 == 0);
                                        *v258 = *v257;
                                        v252 += 1 ;
                                    }
                                    int v259;
                                    v259 = 0;
                                    while (while_method_4(v259)){
                                        int v261;
                                        v261 = 0;
                                        while (while_method_7(v261)){
                                            bool v263;
                                            v263 = 0 <= v261;
                                            bool v265;
                                            if (v263){
                                                bool v264;
                                                v264 = v261 < 4;
                                                v265 = v264;
                                            } else {
                                                v265 = false;
                                            }
                                            bool v266;
                                            v266 = v265 == false;
                                            if (v266){
                                                assert("The indices should be inside the range of the dimension." && v265);
                                            } else {
                                            }
                                            bool v268;
                                            v268 = 0 <= v225;
                                            bool v270;
                                            if (v268){
                                                bool v269;
                                                v269 = v225 < 16;
                                                v270 = v269;
                                            } else {
                                                v270 = false;
                                            }
                                            bool v271;
                                            v271 = v270 == false;
                                            if (v271){
                                                assert("The indices should be inside the range of the dimension." && v270);
                                            } else {
                                            }
                                            int v273;
                                            v273 = v225 * 4;
                                            int v274;
                                            v274 = v261 + v273;
                                            bool v275;
                                            v275 = 0 <= v259;
                                            bool v277;
                                            if (v275){
                                                bool v276;
                                                v276 = v259 < 1;
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
                                            int v280;
                                            v280 = v259 * 64;
                                            int v281;
                                            v281 = v274 + v280;
                                            assert("Tensor range check" && 0 <= v259 && v259 < 1);
                                            assert("Tensor range check" && 0 <= v261 && v261 < 4);
                                            int v282;
                                            v282 = 4 * v259;
                                            int v283;
                                            v283 = v282 + v261;
                                            v251[v283] = v281;
                                            v261 += 1 ;
                                        }
                                        v259 += 1 ;
                                    }
                                    float v284[4];
                                    float v285;
                                    v285 = 0.0f;
                                    int v286;
                                    v286 = 0;
                                    while (while_method_4(v286)){
                                        assert("Tensor range check" && 0 <= v286 && v286 < 1);
                                        int v288;
                                        v288 = 4 * v286;
                                        assert("Tensor range check" && 0 <= v286 && v286 < 1);
                                        int v289; float v290;
                                        Tuple8 tmp62 = Tuple8{0, 0.0f};
                                        v289 = tmp62.v0; v290 = tmp62.v1;
                                        while (while_method_7(v289)){
                                            assert("Tensor range check" && 0 <= v289 && v289 < 4);
                                            int v292;
                                            v292 = v289 + v288;
                                            float v293;
                                            v293 = v250[v292];
                                            float v294;
                                            v294 = v290 + v293;
                                            v290 = v294;
                                            v289 += 1 ;
                                        }
                                        auto v295 = cooperative_groups::coalesced_threads();
                                        int v296;
                                        v296 = threadIdx.x;
                                        int v297;
                                        v297 = v296 / 16;
                                        auto v298 = cooperative_groups::labeled_partition(v295,v297);
                                        Closure2 v299{};
                                        float v300;
                                        v300 = cooperative_groups::inclusive_scan(v298, v290, v299);
                                        float v301;
                                        v301 = v298.shfl_up(v300,1);
                                        bool v302;
                                        v302 = v298.thread_rank() == 0;
                                        float v303;
                                        if (v302){
                                            v303 = 0.0f;
                                        } else {
                                            v303 = v301;
                                        }
                                        float v304;
                                        v304 = v298.shfl(v300,v298.num_threads()-1);
                                        float v305;
                                        v305 = v285 + v303;
                                        int v306; float v307;
                                        Tuple8 tmp63 = Tuple8{0, v305};
                                        v306 = tmp63.v0; v307 = tmp63.v1;
                                        while (while_method_7(v306)){
                                            assert("Tensor range check" && 0 <= v306 && v306 < 4);
                                            int v309;
                                            v309 = v306 + v288;
                                            float v310;
                                            v310 = v250[v309];
                                            float v311;
                                            v311 = v307 + v310;
                                            assert("Tensor range check" && 0 <= v306 && v306 < 4);
                                            v284[v309] = v311;
                                            v307 = v311;
                                            v306 += 1 ;
                                        }
                                        float v312;
                                        v312 = v285 + v304;
                                        v285 = v312;
                                        v286 += 1 ;
                                    }
                                    float v313[4];
                                    bool v314[4];
                                    int v315;
                                    v315 = 0;
                                    while (while_method_4(v315)){
                                        int v317;
                                        v317 = 0;
                                        while (while_method_7(v317)){
                                            assert("Tensor range check" && 0 <= v315 && v315 < 1);
                                            assert("Tensor range check" && 0 <= v317 && v317 < 4);
                                            int v319;
                                            v319 = 4 * v315;
                                            int v320;
                                            v320 = v319 + v317;
                                            float v321;
                                            v321 = v284[v320];
                                            float v322;
                                            v322 = v250[v320];
                                            bool v323;
                                            v323 = v322 > 0.0f;
                                            assert("Tensor range check" && 0 <= v315 && v315 < 1);
                                            assert("Tensor range check" && 0 <= v317 && v317 < 4);
                                            v313[v320] = v321;
                                            v314[v320] = v323;
                                            v317 += 1 ;
                                        }
                                        v315 += 1 ;
                                    }
                                    float v324; bool v325;
                                    Tuple9 tmp64 = Tuple9{-1.0f / 0.0f, false};
                                    v324 = tmp64.v0; v325 = tmp64.v1;
                                    int v326;
                                    v326 = 0;
                                    while (while_method_4(v326)){
                                        int v328;
                                        v328 = 0;
                                        while (while_method_7(v328)){
                                            assert("Tensor range check" && 0 <= v326 && v326 < 1);
                                            assert("Tensor range check" && 0 <= v328 && v328 < 4);
                                            int v330;
                                            v330 = 4 * v326;
                                            int v331;
                                            v331 = v330 + v328;
                                            float v332;
                                            v332 = v313[v331];
                                            bool v333;
                                            v333 = v314[v331];
                                            float v340; bool v341;
                                            if (v325){
                                                if (v333){
                                                    bool v334;
                                                    v334 = v324 >= v332;
                                                    float v335;
                                                    if (v334){
                                                        v335 = v324;
                                                    } else {
                                                        v335 = v332;
                                                    }
                                                    v340 = v335; v341 = true;
                                                } else {
                                                    v340 = v324; v341 = v325;
                                                }
                                            } else {
                                                if (v333){
                                                    v340 = v332; v341 = v333;
                                                } else {
                                                    v340 = v324; v341 = v325;
                                                }
                                            }
                                            v324 = v340;
                                            v325 = v341;
                                            v328 += 1 ;
                                        }
                                        v326 += 1 ;
                                    }
                                    auto v342 = cooperative_groups::coalesced_threads();
                                    int v343;
                                    v343 = threadIdx.x;
                                    int v344;
                                    v344 = v343 / 16;
                                    auto v345 = cooperative_groups::labeled_partition(v342,v344);
                                    Closure3 v346{};
                                    float v347; bool v348;
                                    Tuple9 tmp65 = cooperative_groups::reduce(v345, Tuple9{v324, v325}, v346);
                                    v347 = tmp65.v0; v348 = tmp65.v1;
                                    bool v349;
                                    v349 = v348 == false;
                                    if (v349){
                                        assert("The local reduce must be true." && v348);
                                    } else {
                                    }
                                    float v351[4];
                                    int v352[4];
                                    int v353;
                                    v353 = 0;
                                    while (while_method_4(v353)){
                                        int v355;
                                        v355 = 0;
                                        while (while_method_7(v355)){
                                            assert("Tensor range check" && 0 <= v353 && v353 < 1);
                                            assert("Tensor range check" && 0 <= v355 && v355 < 4);
                                            int v357;
                                            v357 = 4 * v353;
                                            int v358;
                                            v358 = v357 + v355;
                                            int v359;
                                            v359 = v251[v358];
                                            float v360;
                                            v360 = curand_uniform(&v85);
                                            assert("Tensor range check" && 0 <= v353 && v353 < 1);
                                            assert("Tensor range check" && 0 <= v355 && v355 < 4);
                                            v351[v358] = v360;
                                            v352[v358] = v359;
                                            v355 += 1 ;
                                        }
                                        v353 += 1 ;
                                    }
                                    float v361; int v362;
                                    Tuple10 tmp66 = Tuple10{0.0f, 2147483647};
                                    v361 = tmp66.v0; v362 = tmp66.v1;
                                    int v363;
                                    v363 = 0;
                                    while (while_method_4(v363)){
                                        int v365;
                                        v365 = 0;
                                        while (while_method_7(v365)){
                                            assert("Tensor range check" && 0 <= v363 && v363 < 1);
                                            assert("Tensor range check" && 0 <= v365 && v365 < 4);
                                            int v367;
                                            v367 = 4 * v363;
                                            int v368;
                                            v368 = v367 + v365;
                                            float v369;
                                            v369 = v351[v368];
                                            int v370;
                                            v370 = v352[v368];
                                            bool v371;
                                            v371 = v362 < v370;
                                            float v372; int v373;
                                            if (v371){
                                                v372 = v361; v373 = v362;
                                            } else {
                                                v372 = v369; v373 = v370;
                                            }
                                            v361 = v372;
                                            v362 = v373;
                                            v365 += 1 ;
                                        }
                                        v363 += 1 ;
                                    }
                                    auto v374 = cooperative_groups::coalesced_threads();
                                    int v375;
                                    v375 = threadIdx.x;
                                    int v376;
                                    v376 = v375 / 16;
                                    auto v377 = cooperative_groups::labeled_partition(v374,v376);
                                    Closure4 v378{};
                                    float v379; int v380;
                                    Tuple10 tmp67 = cooperative_groups::reduce(v377, Tuple10{v361, v362}, v378);
                                    v379 = tmp67.v0; v380 = tmp67.v1;
                                    float v381;
                                    v381 = v347 * v379;
                                    int v382[4];
                                    bool v383[4];
                                    int v384;
                                    v384 = 0;
                                    while (while_method_4(v384)){
                                        int v386;
                                        v386 = 0;
                                        while (while_method_7(v386)){
                                            assert("Tensor range check" && 0 <= v384 && v384 < 1);
                                            assert("Tensor range check" && 0 <= v386 && v386 < 4);
                                            int v388;
                                            v388 = 4 * v384;
                                            int v389;
                                            v389 = v388 + v386;
                                            float v390;
                                            v390 = v313[v389];
                                            bool v391;
                                            v391 = v314[v389];
                                            int v392;
                                            v392 = v251[v389];
                                            int v395; bool v396;
                                            if (v391){
                                                float v393;
                                                v393 = v390 - v381;
                                                bool v394;
                                                v394 = v393 >= 0.0f;
                                                v395 = v392; v396 = v394;
                                            } else {
                                                v395 = 2147483647; v396 = false;
                                            }
                                            assert("Tensor range check" && 0 <= v384 && v384 < 1);
                                            assert("Tensor range check" && 0 <= v386 && v386 < 4);
                                            v382[v389] = v395;
                                            v383[v389] = v396;
                                            v386 += 1 ;
                                        }
                                        v384 += 1 ;
                                    }
                                    int v397; bool v398;
                                    Tuple11 tmp68 = Tuple11{2147483647, false};
                                    v397 = tmp68.v0; v398 = tmp68.v1;
                                    int v399;
                                    v399 = 0;
                                    while (while_method_4(v399)){
                                        int v401;
                                        v401 = 0;
                                        while (while_method_7(v401)){
                                            assert("Tensor range check" && 0 <= v399 && v399 < 1);
                                            assert("Tensor range check" && 0 <= v401 && v401 < 4);
                                            int v403;
                                            v403 = 4 * v399;
                                            int v404;
                                            v404 = v403 + v401;
                                            int v405;
                                            v405 = v382[v404];
                                            bool v406;
                                            v406 = v383[v404];
                                            int v413; bool v414;
                                            if (v398){
                                                if (v406){
                                                    bool v407;
                                                    v407 = v397 < v405;
                                                    int v408;
                                                    if (v407){
                                                        v408 = v397;
                                                    } else {
                                                        v408 = v405;
                                                    }
                                                    v413 = v408; v414 = true;
                                                } else {
                                                    v413 = v397; v414 = v398;
                                                }
                                            } else {
                                                if (v406){
                                                    v413 = v405; v414 = v406;
                                                } else {
                                                    v413 = v397; v414 = v398;
                                                }
                                            }
                                            v397 = v413;
                                            v398 = v414;
                                            v401 += 1 ;
                                        }
                                        v399 += 1 ;
                                    }
                                    auto v415 = cooperative_groups::coalesced_threads();
                                    int v416;
                                    v416 = threadIdx.x;
                                    int v417;
                                    v417 = v416 / 16;
                                    auto v418 = cooperative_groups::labeled_partition(v415,v417);
                                    Closure5 v419{};
                                    int v420; bool v421;
                                    Tuple11 tmp69 = cooperative_groups::reduce(v418, Tuple11{v397, v398}, v419);
                                    v420 = tmp69.v0; v421 = tmp69.v1;
                                    bool v422;
                                    v422 = v421 == false;
                                    if (v422){
                                        assert("The local reduce must be true." && v421);
                                    } else {
                                    }
                                    float v424; int v425;
                                    Tuple10 tmp70 = Tuple10{0.0f, 2147483647};
                                    v424 = tmp70.v0; v425 = tmp70.v1;
                                    int v426;
                                    v426 = 0;
                                    while (while_method_4(v426)){
                                        int v428;
                                        v428 = 0;
                                        while (while_method_7(v428)){
                                            assert("Tensor range check" && 0 <= v426 && v426 < 1);
                                            assert("Tensor range check" && 0 <= v428 && v428 < 4);
                                            int v430;
                                            v430 = 4 * v426;
                                            int v431;
                                            v431 = v430 + v428;
                                            float v432;
                                            v432 = v250[v431];
                                            int v433;
                                            v433 = v251[v431];
                                            bool v434;
                                            v434 = v425 == v420;
                                            float v438; int v439;
                                            if (v434){
                                                v438 = v424; v439 = v425;
                                            } else {
                                                bool v435;
                                                v435 = v433 == v420;
                                                if (v435){
                                                    v438 = v432; v439 = v433;
                                                } else {
                                                    v438 = v424; v439 = v425;
                                                }
                                            }
                                            v424 = v438;
                                            v425 = v439;
                                            v428 += 1 ;
                                        }
                                        v426 += 1 ;
                                    }
                                    auto v440 = cooperative_groups::coalesced_threads();
                                    int v441;
                                    v441 = threadIdx.x;
                                    int v442;
                                    v442 = v441 / 16;
                                    auto v443 = cooperative_groups::labeled_partition(v440,v442);
                                    Closure6 v444{v420};
                                    float v445; int v446;
                                    Tuple10 tmp71 = cooperative_groups::reduce(v443, Tuple10{v424, v425}, v444);
                                    v445 = tmp71.v0; v446 = tmp71.v1;
                                    bool v447;
                                    v447 = v446 == 2147483647;
                                    bool v448;
                                    v448 = v447 != true;
                                    bool v449;
                                    v449 = v448 == false;
                                    if (v449){
                                        assert("Expected a valid action id in get_prob." && v448);
                                    } else {
                                    }
                                    int v451;
                                    v451 = 0;
                                    while (while_method_4(v451)){
                                        assert("Tensor range check" && 0 <= v451 && v451 < 1);
                                        assert("Tensor range check" && 0 <= v451 && v451 < 1);
                                        v451 += 1 ;
                                    }
                                    assert("Tensor range check" && 0 <= v242 && v242 < 256);
                                    v217[v242] = v445;
                                    v219[v242] = v420;
                                    v230 += 1 ;
                                }
                                __syncthreads();
                                assert("Tensor range check" && 0 <= v221 && v221 < 256);
                                float v453;
                                v453 = v217[v221];
                                int v454;
                                v454 = v219[v221];
                                __syncthreads();
                                extern __shared__ unsigned char v455[];
                                float * v456;
                                v456 = reinterpret_cast<float *>(&v455[0ull]);
                                int * v458;
                                v458 = reinterpret_cast<int *>(&v455[16ull]);
                                int v460;
                                v460 = threadIdx.x;
                                bool v461;
                                v461 = v460 == 0;
                                if (v461){
                                    v456[0] = v453;
                                    v458[0] = v454;
                                } else {
                                }
                                __syncthreads();
                                float v462;
                                v462 = v456[0];
                                int v463;
                                v463 = v458[0];
                                __syncthreads();
                                double * v464;
                                v464 = reinterpret_cast<double *>(&v1[11010048ull]);
                                double * v466;
                                v466 = reinterpret_cast<double *>(&v1[11403264ull]);
                                int v468;
                                v468 = threadIdx.x;
                                int v469;
                                v469 = blockIdx.x;
                                int v470;
                                v470 = v469 * 256;
                                int v471;
                                v471 = v468 + v470;
                                int v472;
                                v472 = 0;
                                while (while_method_7(v472)){
                                    float * v474;
                                    v474 = reinterpret_cast<float *>(&v1[4718592ull]);
                                    int v476;
                                    v476 = blockIdx.x;
                                    int v477;
                                    v477 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v472 && v472 < 4);
                                    assert("Tensor range check" && 0 <= v476 && v476 < 24);
                                    assert("Tensor range check" && 0 <= v477 && v477 < 256);
                                    assert("Tensor range check" && 0 <= v463 && v463 < 64);
                                    int v478;
                                    v478 = 64 * v477;
                                    int v479;
                                    v479 = v478 + v463;
                                    int v480;
                                    v480 = 16384 * v476;
                                    int v481;
                                    v481 = v480 + v479;
                                    int v482;
                                    v482 = 393216 * v472;
                                    int v483;
                                    v483 = v482 + v481;
                                    float v484;
                                    v484 = v474[v483];
                                    double v485;
                                    v485 = (double)v462;
                                    double v486;
                                    v486 = log(v485);
                                    double v487;
                                    v487 = (double)v484;
                                    double v488;
                                    v488 = log(v487);
                                    assert("Tensor range check" && 0 <= v472 && v472 < 4);
                                    assert("Tensor range check" && 0 <= v471 && v471 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    int v489;
                                    v489 = 2 * v471;
                                    int v490;
                                    v490 = v489 + v71;
                                    int v491;
                                    v491 = 12288 * v472;
                                    int v492;
                                    v492 = v491 + v490;
                                    double v493;
                                    v493 = v464[v492];
                                    double v494;
                                    v494 = v466[v492];
                                    double v495;
                                    v495 = v488 + v493;
                                    double v496;
                                    v496 = v486 + v494;
                                    assert("Tensor range check" && 0 <= v472 && v472 < 4);
                                    assert("Tensor range check" && 0 <= v471 && v471 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    v464[v492] = v495;
                                    v466[v492] = v496;
                                    v472 += 1 ;
                                }
                                bool v497;
                                v497 = 0 == v463;
                                Union12 v506;
                                if (v497){
                                    v506 = Union12{Union12_1{}};
                                } else {
                                    bool v499;
                                    v499 = 1 == v463;
                                    if (v499){
                                        v506 = Union12{Union12_0{}};
                                    } else {
                                        bool v501;
                                        v501 = 2 == v463;
                                        if (v501){
                                            v506 = Union12{Union12_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v506.tag) {
                                    case 0: { // AA_Call
                                        v581 = Union1{Union1_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v507;
                                        v507 = v72[0];
                                        int v509; int v510;
                                        Tuple7 tmp72 = Tuple7{1, v507};
                                        v509 = tmp72.v0; v510 = tmp72.v1;
                                        while (while_method_0(v509)){
                                            bool v512;
                                            v512 = 0 <= v509;
                                            bool v514;
                                            if (v512){
                                                bool v513;
                                                v513 = v509 < 2;
                                                v514 = v513;
                                            } else {
                                                v514 = false;
                                            }
                                            bool v515;
                                            v515 = v514 == false;
                                            if (v515){
                                                assert("Index must be in range." && v514);
                                            } else {
                                            }
                                            int v517;
                                            v517 = v72[v509];
                                            bool v519;
                                            v519 = v510 >= v517;
                                            int v520;
                                            if (v519){
                                                v520 = v510;
                                            } else {
                                                v520 = v517;
                                            }
                                            v510 = v520;
                                            v509 += 1 ;
                                        }
                                        bool v522;
                                        if (v75){
                                            bool v521;
                                            v521 = v71 < 2;
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
                                        v525 = v72[v71];
                                        bool v527;
                                        v527 = v525 == v510;
                                        if (v527){
                                            v581 = Union1{Union1_0{}};
                                        } else {
                                            v581 = Union1{Union1_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v532;
                                        v532 = v73 > 0;
                                        if (v532){
                                            v581 = Union1{Union1_2{}};
                                        } else {
                                            v581 = Union1{Union1_0{}};
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
                                curandStatePhilox4_32_10_t & v539 = v3.v5;
                                curandStatePhilox4_32_10_t & v540 = v539;
                                static_array_list<Union1,3> v541;
                                v541 = static_array_list<Union1,3>{};
                                v541.unsafe_set_length(1);
                                Union1 v543;
                                v543 = Union1{Union1_0{}};
                                v541[0] = v543;
                                int v545;
                                v545 = v72[0];
                                int v547;
                                v547 = v72[1];
                                bool v549;
                                v549 = v545 == v547;
                                bool v550;
                                v550 = v549 != true;
                                if (v550){
                                    Union1 v551;
                                    v551 = Union1{Union1_1{}};
                                    v541.push(v551);
                                } else {
                                }
                                bool v552;
                                v552 = v73 > 0;
                                if (v552){
                                    Union1 v553;
                                    v553 = Union1{Union1_2{}};
                                    v541.push(v553);
                                } else {
                                }
                                int v554;
                                v554 = v541.length;
                                int v555;
                                v555 = v554 - 1;
                                int v556;
                                v556 = 0;
                                while (while_method_1(v555, v556)){
                                    int v558;
                                    v558 = v541.length;
                                    int v559;
                                    v559 = int_range_22(v558, v556, v540);
                                    Union1 v560;
                                    v560 = v541[v556];
                                    Union1 v562;
                                    v562 = v541[v559];
                                    v541[v556] = v562;
                                    v541[v559] = v560;
                                    v556 += 1 ;
                                }
                                Union1 v564;
                                v564 = v541.pop();
                                int v565;
                                v565 = sizeof(Union1);
                                unsigned long long v566;
                                v566 = (unsigned long long)v565;
                                bool v567;
                                v567 = v566 <= 98304ull;
                                bool v568;
                                v568 = v567 == false;
                                if (v568){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v567);
                                } else {
                                }
                                extern __shared__ unsigned char v570[];
                                bool v571;
                                v571 = v566 <= v566;
                                bool v572;
                                v572 = v571 == false;
                                if (v572){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v571);
                                } else {
                                }
                                Union1 * v574;
                                v574 = reinterpret_cast<Union1 *>(&v570[0ull]);
                                int v576;
                                v576 = threadIdx.x;
                                bool v577;
                                v577 = v576 == 0;
                                if (v577){
                                    v574[0] = v564;
                                } else {
                                }
                                __syncthreads();
                                Union1 v578;
                                v578 = v574[0];
                                __syncthreads();
                                v581 = v578;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union7 v582;
                        v582 = Union7{Union7_1{v71, v581}};
                        v14.push(v582);
                        v624 = Union14{Union14_2{v68, v69, v70, v71, v72, v73, v581}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v584 = v18.case3.v0; bool v585 = v18.case3.v1; static_array<Union6,2> v586 = v18.case3.v2; int v587 = v18.case3.v3; static_array<int,2> v588 = v18.case3.v4; int v589 = v18.case3.v5; Union1 v590 = v18.case3.v6;
                        Union7 v591;
                        v591 = Union7{Union7_1{v587, v590}};
                        v14.push(v591);
                        v624 = Union14{Union14_2{v584, v585, v586, v587, v588, v589, v590}};
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
                        v52 = compare_hands_25(v39, v40, v41, v42, v43, v44);
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
                        v624 = Union14{Union14_3{}};
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
                        v624 = Union14{Union14_3{}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                switch (v624.tag) {
                    case 0: { // T_game_chance_community_card
                        Union5 v626 = v624.case0.v0; bool v627 = v624.case0.v1; static_array<Union6,2> v628 = v624.case0.v2; int v629 = v624.case0.v3; static_array<int,2> v630 = v624.case0.v4; int v631 = v624.case0.v5; Union6 v632 = v624.case0.v6;
                        int v633;
                        v633 = 2;
                        int v634; int v635;
                        Tuple7 tmp73 = Tuple7{0, 0};
                        v634 = tmp73.v0; v635 = tmp73.v1;
                        while (while_method_0(v634)){
                            bool v637;
                            v637 = 0 <= v634;
                            bool v639;
                            if (v637){
                                bool v638;
                                v638 = v634 < 2;
                                v639 = v638;
                            } else {
                                v639 = false;
                            }
                            bool v640;
                            v640 = v639 == false;
                            if (v640){
                                assert("Index must be in range." && v639);
                            } else {
                            }
                            int v642;
                            v642 = v630[v634];
                            bool v644;
                            v644 = v635 >= v642;
                            int v645;
                            if (v644){
                                v645 = v635;
                            } else {
                                v645 = v642;
                            }
                            v635 = v645;
                            v634 += 1 ;
                        }
                        static_array<int,2> v646;
                        int v648;
                        v648 = 0;
                        while (while_method_0(v648)){
                            v646[v648] = v635;
                            v648 += 1 ;
                        }
                        Union5 v650;
                        v650 = Union5{Union5_1{v632}};
                        Union4 v651;
                        v651 = Union4{Union4_2{v650, true, v628, 0, v646, v633}};
                        v784 = Union3{Union3_1{v651}};
                        break;
                    }
                    case 1: { // T_game_chance_init
                        Union6 v653 = v624.case1.v0; Union6 v654 = v624.case1.v1;
                        int v655;
                        v655 = 2;
                        static_array<int,2> v656;
                        v656[0] = 1;
                        v656[1] = 1;
                        static_array<Union6,2> v658;
                        v658[0] = v653;
                        v658[1] = v654;
                        Union5 v660;
                        v660 = Union5{Union5_0{}};
                        Union4 v661;
                        v661 = Union4{Union4_2{v660, true, v658, 0, v656, v655}};
                        v784 = Union3{Union3_1{v661}};
                        break;
                    }
                    case 2: { // T_game_round
                        Union5 v663 = v624.case2.v0; bool v664 = v624.case2.v1; static_array<Union6,2> v665 = v624.case2.v2; int v666 = v624.case2.v3; static_array<int,2> v667 = v624.case2.v4; int v668 = v624.case2.v5; Union1 v669 = v624.case2.v6;
                        Union4 v776;
                        switch (v663.tag) {
                            case 0: { // None
                                switch (v669.tag) {
                                    case 0: { // Call
                                        if (v664){
                                            int v732;
                                            v732 = v666 ^ 1;
                                            v776 = Union4{Union4_2{v663, false, v665, v732, v667, v668}};
                                        } else {
                                            v776 = Union4{Union4_0{v663, v664, v665, v666, v667, v668}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v776 = Union4{Union4_5{v663, v664, v665, v666, v667, v668}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v736;
                                        v736 = v668 > 0;
                                        if (v736){
                                            int v737;
                                            v737 = v666 ^ 1;
                                            int v738;
                                            v738 = -1 + v668;
                                            int v739; int v740;
                                            Tuple7 tmp74 = Tuple7{0, 0};
                                            v739 = tmp74.v0; v740 = tmp74.v1;
                                            while (while_method_0(v739)){
                                                bool v742;
                                                v742 = 0 <= v739;
                                                bool v744;
                                                if (v742){
                                                    bool v743;
                                                    v743 = v739 < 2;
                                                    v744 = v743;
                                                } else {
                                                    v744 = false;
                                                }
                                                bool v745;
                                                v745 = v744 == false;
                                                if (v745){
                                                    assert("Index must be in range." && v744);
                                                } else {
                                                }
                                                int v747;
                                                v747 = v667[v739];
                                                bool v749;
                                                v749 = v740 >= v747;
                                                int v750;
                                                if (v749){
                                                    v750 = v740;
                                                } else {
                                                    v750 = v747;
                                                }
                                                v740 = v750;
                                                v739 += 1 ;
                                            }
                                            static_array<int,2> v751;
                                            int v753;
                                            v753 = 0;
                                            while (while_method_0(v753)){
                                                v751[v753] = v740;
                                                v753 += 1 ;
                                            }
                                            static_array<int,2> v755;
                                            int v757;
                                            v757 = 0;
                                            while (while_method_0(v757)){
                                                bool v759;
                                                v759 = 0 <= v757;
                                                bool v761;
                                                if (v759){
                                                    bool v760;
                                                    v760 = v757 < 2;
                                                    v761 = v760;
                                                } else {
                                                    v761 = false;
                                                }
                                                bool v762;
                                                v762 = v761 == false;
                                                if (v762){
                                                    assert("Index must be in range." && v761);
                                                } else {
                                                }
                                                int v764;
                                                v764 = v751[v757];
                                                bool v766;
                                                v766 = v757 == v666;
                                                int v768;
                                                if (v766){
                                                    int v767;
                                                    v767 = v764 + 2;
                                                    v768 = v767;
                                                } else {
                                                    v768 = v764;
                                                }
                                                v755[v757] = v768;
                                                v757 += 1 ;
                                            }
                                            v776 = Union4{Union4_2{v663, false, v665, v737, v755, v738}};
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
                                Union6 v670 = v663.case1.v0;
                                switch (v669.tag) {
                                    case 0: { // Call
                                        if (v664){
                                            int v672;
                                            v672 = v666 ^ 1;
                                            v776 = Union4{Union4_2{v663, false, v665, v672, v667, v668}};
                                        } else {
                                            int v674; int v675;
                                            Tuple7 tmp75 = Tuple7{0, 0};
                                            v674 = tmp75.v0; v675 = tmp75.v1;
                                            while (while_method_0(v674)){
                                                bool v677;
                                                v677 = 0 <= v674;
                                                bool v679;
                                                if (v677){
                                                    bool v678;
                                                    v678 = v674 < 2;
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
                                                v682 = v667[v674];
                                                bool v684;
                                                v684 = v675 >= v682;
                                                int v685;
                                                if (v684){
                                                    v685 = v675;
                                                } else {
                                                    v685 = v682;
                                                }
                                                v675 = v685;
                                                v674 += 1 ;
                                            }
                                            static_array<int,2> v686;
                                            int v688;
                                            v688 = 0;
                                            while (while_method_0(v688)){
                                                v686[v688] = v675;
                                                v688 += 1 ;
                                            }
                                            v776 = Union4{Union4_4{v663, v664, v665, v666, v686, v668}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v776 = Union4{Union4_5{v663, v664, v665, v666, v667, v668}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v692;
                                        v692 = v668 > 0;
                                        if (v692){
                                            int v693;
                                            v693 = v666 ^ 1;
                                            int v694;
                                            v694 = -1 + v668;
                                            int v695; int v696;
                                            Tuple7 tmp76 = Tuple7{0, 0};
                                            v695 = tmp76.v0; v696 = tmp76.v1;
                                            while (while_method_0(v695)){
                                                bool v698;
                                                v698 = 0 <= v695;
                                                bool v700;
                                                if (v698){
                                                    bool v699;
                                                    v699 = v695 < 2;
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
                                                v703 = v667[v695];
                                                bool v705;
                                                v705 = v696 >= v703;
                                                int v706;
                                                if (v705){
                                                    v706 = v696;
                                                } else {
                                                    v706 = v703;
                                                }
                                                v696 = v706;
                                                v695 += 1 ;
                                            }
                                            static_array<int,2> v707;
                                            int v709;
                                            v709 = 0;
                                            while (while_method_0(v709)){
                                                v707[v709] = v696;
                                                v709 += 1 ;
                                            }
                                            static_array<int,2> v711;
                                            int v713;
                                            v713 = 0;
                                            while (while_method_0(v713)){
                                                bool v715;
                                                v715 = 0 <= v713;
                                                bool v717;
                                                if (v715){
                                                    bool v716;
                                                    v716 = v713 < 2;
                                                    v717 = v716;
                                                } else {
                                                    v717 = false;
                                                }
                                                bool v718;
                                                v718 = v717 == false;
                                                if (v718){
                                                    assert("Index must be in range." && v717);
                                                } else {
                                                }
                                                int v720;
                                                v720 = v707[v713];
                                                bool v722;
                                                v722 = v713 == v666;
                                                int v724;
                                                if (v722){
                                                    int v723;
                                                    v723 = v720 + 4;
                                                    v724 = v723;
                                                } else {
                                                    v724 = v720;
                                                }
                                                v711[v713] = v724;
                                                v713 += 1 ;
                                            }
                                            v776 = Union4{Union4_2{v663, false, v665, v693, v711, v694}};
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
                        v784 = Union3{Union3_1{v776}};
                        break;
                    }
                    case 3: { // T_none
                        v784 = Union3{Union3_0{}};
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
        v16 = v784;
    }
    return ;
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
    while (while_method_10(v16)){
        Union3 v792;
        switch (v16.tag) {
            case 0: { // None
                v792 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v18 = v16.case1.v0;
                Union14 v632;
                switch (v18.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v601 = v18.case0.v0; bool v602 = v18.case0.v1; static_array<Union6,2> v603 = v18.case0.v2; int v604 = v18.case0.v3; static_array<int,2> v605 = v18.case0.v4; int v606 = v18.case0.v5;
                        curandStatePhilox4_32_10_t & v607 = v3.v5;
                        curandStatePhilox4_32_10_t & v608 = v607;
                        unsigned int & v609 = v3.v0;
                        Union6 v610; unsigned int v611;
                        Tuple6 tmp77 = draw_card_20(v608, v609);
                        v610 = tmp77.v0; v611 = tmp77.v1;
                        v3.v0 = v611;
                        Union7 v612;
                        v612 = Union7{Union7_0{v610}};
                        v14.push(v612);
                        v632 = Union14{Union14_0{v601, v602, v603, v604, v605, v606, v610}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v614 = v3.v5;
                        curandStatePhilox4_32_10_t & v615 = v614;
                        unsigned int & v616 = v3.v0;
                        Union6 v617; unsigned int v618;
                        Tuple6 tmp78 = draw_card_20(v615, v616);
                        v617 = tmp78.v0; v618 = tmp78.v1;
                        v3.v0 = v618;
                        curandStatePhilox4_32_10_t & v619 = v3.v5;
                        curandStatePhilox4_32_10_t & v620 = v619;
                        unsigned int & v621 = v3.v0;
                        Union6 v622; unsigned int v623;
                        Tuple6 tmp79 = draw_card_20(v620, v621);
                        v622 = tmp79.v0; v623 = tmp79.v1;
                        v3.v0 = v623;
                        Union7 v624;
                        v624 = Union7{Union7_2{0, v617}};
                        v14.push(v624);
                        Union7 v625;
                        v625 = Union7{Union7_2{1, v622}};
                        v14.push(v625);
                        v632 = Union14{Union14_1{v617, v622}};
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
                        Union1 v589;
                        switch (v80.tag) {
                            case 0: { // Computer
                                static_array_list<Union7,32> & v83 = v3.v2;
                                curandStatePhilox4_32_10_t & v84 = v3.v5;
                                curandStatePhilox4_32_10_t & v85 = v84;
                                float * v86;
                                v86 = reinterpret_cast<float *>(&v1[4718592ull]);
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
                                while (while_method_7(v155)){
                                    float * v157;
                                    v157 = reinterpret_cast<float *>(&v1[4718592ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 4);
                                    int v159;
                                    v159 = 393216 * v155;
                                    float * v160;
                                    v160 = reinterpret_cast<float *>(&v1[0ull]);
                                    float * v162;
                                    v162 = reinterpret_cast<float *>(&v0[0ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 4);
                                    int v164;
                                    v164 = 8192 * v155;
                                    float * v165;
                                    v165 = reinterpret_cast<float *>(&v1[3145728ull]);
                                    block_matmul_23(v165, v162, v164, v160);
                                    block_row_map_24(v157, v159, v165);
                                    int * v167;
                                    v167 = reinterpret_cast<int *>(&v0[131072ull]);
                                    float * v169;
                                    v169 = reinterpret_cast<float *>(&v0[131088ull]);
                                    float * v171;
                                    v171 = reinterpret_cast<float *>(&v0[131104ull]);
                                    double * v173;
                                    v173 = reinterpret_cast<double *>(&v1[11010048ull]);
                                    double * v175;
                                    v175 = reinterpret_cast<double *>(&v1[11403264ull]);
                                    v155 += 1 ;
                                }
                                __syncthreads();
                                int * v177;
                                v177 = reinterpret_cast<int *>(&v0[131072ull]);
                                float * v179;
                                v179 = reinterpret_cast<float *>(&v0[131088ull]);
                                float * v181;
                                v181 = reinterpret_cast<float *>(&v0[131104ull]);
                                int v183;
                                v183 = 0;
                                int v184;
                                v184 = 4;
                                int v185;
                                v185 = int_range_22(v184, v183, v85);
                                extern __shared__ unsigned char v186[];
                                int * v187;
                                v187 = reinterpret_cast<int *>(&v186[0ull]);
                                int v189;
                                v189 = threadIdx.x;
                                bool v190;
                                v190 = v189 == 0;
                                if (v190){
                                    v187[0] = v185;
                                } else {
                                }
                                __syncthreads();
                                int v191;
                                v191 = v187[0];
                                __syncthreads();
                                float * v192;
                                v192 = reinterpret_cast<float *>(&v1[4718592ull]);
                                assert("Tensor range check" && 0 <= v191 && v191 < 4);
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
                                        Tuple8 tmp80 = Tuple8{0, 0.0f};
                                        v297 = tmp80.v0; v298 = tmp80.v1;
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
                                        Tuple8 tmp81 = Tuple8{0, v313};
                                        v314 = tmp81.v0; v315 = tmp81.v1;
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
                                    Tuple9 tmp82 = Tuple9{-1.0f / 0.0f, false};
                                    v332 = tmp82.v0; v333 = tmp82.v1;
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
                                    Tuple9 tmp83 = cooperative_groups::reduce(v353, Tuple9{v332, v333}, v354);
                                    v355 = tmp83.v0; v356 = tmp83.v1;
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
                                            v368 = curand_uniform(&v85);
                                            assert("Tensor range check" && 0 <= v361 && v361 < 1);
                                            assert("Tensor range check" && 0 <= v363 && v363 < 4);
                                            v359[v366] = v368;
                                            v360[v366] = v367;
                                            v363 += 1 ;
                                        }
                                        v361 += 1 ;
                                    }
                                    float v369; int v370;
                                    Tuple10 tmp84 = Tuple10{0.0f, 2147483647};
                                    v369 = tmp84.v0; v370 = tmp84.v1;
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
                                    Tuple10 tmp85 = cooperative_groups::reduce(v385, Tuple10{v369, v370}, v386);
                                    v387 = tmp85.v0; v388 = tmp85.v1;
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
                                    Tuple11 tmp86 = Tuple11{2147483647, false};
                                    v405 = tmp86.v0; v406 = tmp86.v1;
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
                                    Tuple11 tmp87 = cooperative_groups::reduce(v426, Tuple11{v405, v406}, v427);
                                    v428 = tmp87.v0; v429 = tmp87.v1;
                                    bool v430;
                                    v430 = v429 == false;
                                    if (v430){
                                        assert("The local reduce must be true." && v429);
                                    } else {
                                    }
                                    float v432; int v433;
                                    Tuple10 tmp88 = Tuple10{0.0f, 2147483647};
                                    v432 = tmp88.v0; v433 = tmp88.v1;
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
                                    Tuple10 tmp89 = cooperative_groups::reduce(v451, Tuple10{v432, v433}, v452);
                                    v453 = tmp89.v0; v454 = tmp89.v1;
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
                                v472 = reinterpret_cast<double *>(&v1[11010048ull]);
                                double * v474;
                                v474 = reinterpret_cast<double *>(&v1[11403264ull]);
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
                                while (while_method_7(v480)){
                                    float * v482;
                                    v482 = reinterpret_cast<float *>(&v1[4718592ull]);
                                    int v484;
                                    v484 = blockIdx.x;
                                    int v485;
                                    v485 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v480 && v480 < 4);
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
                                    assert("Tensor range check" && 0 <= v480 && v480 < 4);
                                    assert("Tensor range check" && 0 <= v479 && v479 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    int v497;
                                    v497 = 2 * v479;
                                    int v498;
                                    v498 = v497 + v71;
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
                                    assert("Tensor range check" && 0 <= v480 && v480 < 4);
                                    assert("Tensor range check" && 0 <= v479 && v479 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
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
                                        v515 = v72[0];
                                        int v517; int v518;
                                        Tuple7 tmp90 = Tuple7{1, v515};
                                        v517 = tmp90.v0; v518 = tmp90.v1;
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
                                            v525 = v72[v517];
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
                                        if (v75){
                                            bool v529;
                                            v529 = v71 < 2;
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
                                        v533 = v72[v71];
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
                                        v540 = v73 > 0;
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
                                v553 = v72[0];
                                int v555;
                                v555 = v72[1];
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
                                v560 = v73 > 0;
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
                        v590 = Union7{Union7_1{v71, v589}};
                        v14.push(v590);
                        v632 = Union14{Union14_2{v68, v69, v70, v71, v72, v73, v589}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v592 = v18.case3.v0; bool v593 = v18.case3.v1; static_array<Union6,2> v594 = v18.case3.v2; int v595 = v18.case3.v3; static_array<int,2> v596 = v18.case3.v4; int v597 = v18.case3.v5; Union1 v598 = v18.case3.v6;
                        Union7 v599;
                        v599 = Union7{Union7_1{v595, v598}};
                        v14.push(v599);
                        v632 = Union14{Union14_2{v592, v593, v594, v595, v596, v597, v598}};
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
                        v52 = compare_hands_25(v39, v40, v41, v42, v43, v44);
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
                        v632 = Union14{Union14_3{}};
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
                        Tuple7 tmp91 = Tuple7{0, 0};
                        v642 = tmp91.v0; v643 = tmp91.v1;
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
                                            Tuple7 tmp92 = Tuple7{0, 0};
                                            v747 = tmp92.v0; v748 = tmp92.v1;
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
                                            Tuple7 tmp93 = Tuple7{0, 0};
                                            v682 = tmp93.v0; v683 = tmp93.v1;
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
                                            Tuple7 tmp94 = Tuple7{0, 0};
                                            v703 = tmp94.v0; v704 = tmp94.v1;
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
        v16 = v792;
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
                Union3 v984;
                switch (v60.tag) {
                    case 0: { // None
                        v984 = Union3{Union3_0{}};
                        break;
                    }
                    case 1: { // Some
                        Union4 v62 = v60.case1.v0;
                        switch (v62.tag) {
                            case 0: { // ChanceCommunityCard
                                Union5 v924 = v62.case0.v0; bool v925 = v62.case0.v1; static_array<Union6,2> v926 = v62.case0.v2; int v927 = v62.case0.v3; static_array<int,2> v928 = v62.case0.v4; int v929 = v62.case0.v5;
                                curandStatePhilox4_32_10_t & v930 = v19.v4;
                                curandStatePhilox4_32_10_t & v931 = v930;
                                unsigned int & v932 = v19.v0;
                                Union6 v933; unsigned int v934;
                                Tuple6 tmp11 = draw_card_20(v931, v932);
                                v933 = tmp11.v0; v934 = tmp11.v1;
                                v19.v0 = v934;
                                Union7 v935;
                                v935 = Union7{Union7_0{v933}};
                                v58.push(v935);
                                int v936;
                                v936 = 2;
                                int v937; int v938;
                                Tuple7 tmp12 = Tuple7{0, 0};
                                v937 = tmp12.v0; v938 = tmp12.v1;
                                while (while_method_0(v937)){
                                    bool v940;
                                    v940 = 0 <= v937;
                                    bool v942;
                                    if (v940){
                                        bool v941;
                                        v941 = v937 < 2;
                                        v942 = v941;
                                    } else {
                                        v942 = false;
                                    }
                                    bool v943;
                                    v943 = v942 == false;
                                    if (v943){
                                        assert("Index must be in range." && v942);
                                    } else {
                                    }
                                    int v945;
                                    v945 = v928[v937];
                                    bool v947;
                                    v947 = v938 >= v945;
                                    int v948;
                                    if (v947){
                                        v948 = v938;
                                    } else {
                                        v948 = v945;
                                    }
                                    v938 = v948;
                                    v937 += 1 ;
                                }
                                static_array<int,2> v949;
                                int v951;
                                v951 = 0;
                                while (while_method_0(v951)){
                                    v949[v951] = v938;
                                    v951 += 1 ;
                                }
                                Union5 v953;
                                v953 = Union5{Union5_1{v933}};
                                Union4 v954;
                                v954 = Union4{Union4_2{v953, true, v926, 0, v949, v936}};
                                v984 = Union3{Union3_1{v954}};
                                break;
                            }
                            case 1: { // ChanceInit
                                curandStatePhilox4_32_10_t & v956 = v19.v4;
                                curandStatePhilox4_32_10_t & v957 = v956;
                                unsigned int & v958 = v19.v0;
                                Union6 v959; unsigned int v960;
                                Tuple6 tmp13 = draw_card_20(v957, v958);
                                v959 = tmp13.v0; v960 = tmp13.v1;
                                v19.v0 = v960;
                                curandStatePhilox4_32_10_t & v961 = v19.v4;
                                curandStatePhilox4_32_10_t & v962 = v961;
                                unsigned int & v963 = v19.v0;
                                Union6 v964; unsigned int v965;
                                Tuple6 tmp14 = draw_card_20(v962, v963);
                                v964 = tmp14.v0; v965 = tmp14.v1;
                                v19.v0 = v965;
                                Union7 v966;
                                v966 = Union7{Union7_2{0, v959}};
                                v58.push(v966);
                                Union7 v967;
                                v967 = Union7{Union7_2{1, v964}};
                                v58.push(v967);
                                int v968;
                                v968 = 2;
                                static_array<int,2> v969;
                                v969[0] = 1;
                                v969[1] = 1;
                                static_array<Union6,2> v971;
                                v971[0] = v959;
                                v971[1] = v964;
                                Union5 v973;
                                v973 = Union5{Union5_0{}};
                                Union4 v974;
                                v974 = Union4{Union4_2{v973, true, v971, 0, v969, v968}};
                                v984 = Union3{Union3_1{v974}};
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
                                        v122 = reinterpret_cast<float *>(&v3[4718592ull]);
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
                                        v192 = 4;
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
                                        v200 = reinterpret_cast<float *>(&v3[4718592ull]);
                                        assert("Tensor range check" && 0 <= v199 && v199 < 4);
                                        int v202;
                                        v202 = 393216 * v199;
                                        float * v203;
                                        v203 = reinterpret_cast<float *>(&v3[0ull]);
                                        float * v205;
                                        v205 = reinterpret_cast<float *>(&v2[0ull]);
                                        assert("Tensor range check" && 0 <= v199 && v199 < 4);
                                        int v207;
                                        v207 = 8192 * v199;
                                        float * v208;
                                        v208 = reinterpret_cast<float *>(&v3[3145728ull]);
                                        block_matmul_23(v208, v205, v207, v203);
                                        block_row_map_24(v200, v202, v208);
                                        int * v210;
                                        v210 = reinterpret_cast<int *>(&v2[131072ull]);
                                        float * v212;
                                        v212 = reinterpret_cast<float *>(&v2[131088ull]);
                                        float * v214;
                                        v214 = reinterpret_cast<float *>(&v2[131104ull]);
                                        double * v216;
                                        v216 = reinterpret_cast<double *>(&v3[11010048ull]);
                                        double * v218;
                                        v218 = reinterpret_cast<double *>(&v3[11403264ull]);
                                        __syncthreads();
                                        float * v220;
                                        v220 = reinterpret_cast<float *>(&v3[4718592ull]);
                                        assert("Tensor range check" && 0 <= v199 && v199 < 4);
                                        int v222;
                                        v222 = blockIdx.x;
                                        assert("Tensor range check" && 0 <= v222 && v222 < 24);
                                        int v223;
                                        v223 = 16384 * v222;
                                        int v224;
                                        v224 = v223 + v202;
                                        int v225;
                                        v225 = threadIdx.x;
                                        assert("Tensor range check" && 0 <= v225 && v225 < 256);
                                        int v226;
                                        v226 = 64 * v225;
                                        int v227;
                                        v227 = v226 + v224;
                                        float * v228;
                                        v228 = v220+v227;
                                        int v230;
                                        v230 = sizeof(float *);
                                        unsigned long long v231;
                                        v231 = (unsigned long long)v230;
                                        unsigned long long v232;
                                        v232 = 256ull * v231;
                                        unsigned long long v233;
                                        v233 = v232 + 16ull;
                                        unsigned long long v234;
                                        v234 = v233 - 1ull;
                                        unsigned long long v235;
                                        v235 = v234 % 16ull;
                                        unsigned long long v236;
                                        v236 = v234 - v235;
                                        unsigned long long v237;
                                        v237 = v236 + 1024ull;
                                        unsigned long long v238;
                                        v238 = v237 + 16ull;
                                        unsigned long long v239;
                                        v239 = v238 - 1ull;
                                        unsigned long long v240;
                                        v240 = v239 % 16ull;
                                        unsigned long long v241;
                                        v241 = v239 - v240;
                                        unsigned long long v242;
                                        v242 = v241 + 1024ull;
                                        bool v243;
                                        v243 = v242 <= 98304ull;
                                        bool v244;
                                        v244 = v243 == false;
                                        if (v244){
                                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v243);
                                        } else {
                                        }
                                        extern __shared__ unsigned char v246[];
                                        bool v247;
                                        v247 = v242 <= v242;
                                        bool v248;
                                        v248 = v247 == false;
                                        if (v248){
                                            assert("The length of the partition has to be less than or equal to the length of the base array." && v247);
                                        } else {
                                        }
                                        float * * v250;
                                        v250 = reinterpret_cast<float * *>(&v246[0ull]);
                                        float * v252;
                                        v252 = reinterpret_cast<float *>(&v246[v236]);
                                        int * v254;
                                        v254 = reinterpret_cast<int *>(&v246[v241]);
                                        int v256;
                                        v256 = threadIdx.x;
                                        assert("Tensor range check" && 0 <= v256 && v256 < 256);
                                        v250[v256] = v228;
                                        __syncthreads();
                                        bool v257;
                                        v257 = 0 <= v256;
                                        bool v258;
                                        v258 = v257 == false;
                                        if (v258){
                                            assert("The index needs to be zero or positive." && v257);
                                        } else {
                                        }
                                        int v260;
                                        v260 = v256 % 16;
                                        int v261;
                                        v261 = v256 / 16;
                                        bool v262;
                                        v262 = v261 < 16;
                                        bool v263;
                                        v263 = v262 == false;
                                        if (v263){
                                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v262);
                                        } else {
                                        }
                                        assert("Tensor range check" && 0 <= v261 && v261 < 16);
                                        int v265;
                                        v265 = 0;
                                        while (while_method_8(v265)){
                                            bool v267;
                                            v267 = 0 <= v261;
                                            bool v268;
                                            v268 = v267 && v262;
                                            bool v269;
                                            v269 = v268 == false;
                                            if (v269){
                                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v268);
                                            } else {
                                            }
                                            bool v271;
                                            v271 = 0 <= v265;
                                            bool v273;
                                            if (v271){
                                                bool v272;
                                                v272 = v265 < 16;
                                                v273 = v272;
                                            } else {
                                                v273 = false;
                                            }
                                            bool v274;
                                            v274 = v273 == false;
                                            if (v274){
                                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v273);
                                            } else {
                                            }
                                            int v276;
                                            v276 = v265 * 16;
                                            int v277;
                                            v277 = v276 + v261;
                                            assert("Tensor range check" && 0 <= v265 && v265 < 16);
                                            int v278;
                                            v278 = 16 * v265;
                                            int v279;
                                            v279 = v278 + v261;
                                            float * v280;
                                            v280 = v250[v279];
                                            int v281;
                                            v281 = blockIdx.x;
                                            int v282;
                                            v282 = v281 * 256;
                                            int v283;
                                            v283 = v282 + v277;
                                            assert("Tensor range check" && 0 <= v260 && v260 < 16);
                                            int v284;
                                            v284 = 4 * v260;
                                            float v285[4];
                                            int v286[4];
                                            int v287;
                                            v287 = 0;
                                            while (while_method_4(v287)){
                                                assert("Tensor range check" && 0 <= v287 && v287 < 1);
                                                int v289;
                                                v289 = 4 * v287;
                                                assert("Tensor range check" && 0 <= v287 && v287 < 1);
                                                int v290;
                                                v290 = 64 * v287;
                                                int v291;
                                                v291 = v290 + v284;
                                                int4* v292;
                                                v292 = reinterpret_cast<int4*>(v280 + v291);
                                                int4* v293;
                                                v293 = reinterpret_cast<int4*>(v285 + v289);
                                                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v292) % 16 == 0 && reinterpret_cast<unsigned long long>(v293) % 16 == 0);
                                                *v293 = *v292;
                                                v287 += 1 ;
                                            }
                                            int v294;
                                            v294 = 0;
                                            while (while_method_4(v294)){
                                                int v296;
                                                v296 = 0;
                                                while (while_method_7(v296)){
                                                    bool v298;
                                                    v298 = 0 <= v296;
                                                    bool v300;
                                                    if (v298){
                                                        bool v299;
                                                        v299 = v296 < 4;
                                                        v300 = v299;
                                                    } else {
                                                        v300 = false;
                                                    }
                                                    bool v301;
                                                    v301 = v300 == false;
                                                    if (v301){
                                                        assert("The indices should be inside the range of the dimension." && v300);
                                                    } else {
                                                    }
                                                    bool v303;
                                                    v303 = 0 <= v260;
                                                    bool v305;
                                                    if (v303){
                                                        bool v304;
                                                        v304 = v260 < 16;
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
                                                    v308 = v260 * 4;
                                                    int v309;
                                                    v309 = v296 + v308;
                                                    bool v310;
                                                    v310 = 0 <= v294;
                                                    bool v312;
                                                    if (v310){
                                                        bool v311;
                                                        v311 = v294 < 1;
                                                        v312 = v311;
                                                    } else {
                                                        v312 = false;
                                                    }
                                                    bool v313;
                                                    v313 = v312 == false;
                                                    if (v313){
                                                        assert("The indices should be inside the range of the dimension." && v312);
                                                    } else {
                                                    }
                                                    int v315;
                                                    v315 = v294 * 64;
                                                    int v316;
                                                    v316 = v309 + v315;
                                                    assert("Tensor range check" && 0 <= v294 && v294 < 1);
                                                    assert("Tensor range check" && 0 <= v296 && v296 < 4);
                                                    int v317;
                                                    v317 = 4 * v294;
                                                    int v318;
                                                    v318 = v317 + v296;
                                                    v286[v318] = v316;
                                                    v296 += 1 ;
                                                }
                                                v294 += 1 ;
                                            }
                                            float v319[4];
                                            float v320;
                                            v320 = 0.0f;
                                            int v321;
                                            v321 = 0;
                                            while (while_method_4(v321)){
                                                assert("Tensor range check" && 0 <= v321 && v321 < 1);
                                                int v323;
                                                v323 = 4 * v321;
                                                assert("Tensor range check" && 0 <= v321 && v321 < 1);
                                                int v324; float v325;
                                                Tuple8 tmp15 = Tuple8{0, 0.0f};
                                                v324 = tmp15.v0; v325 = tmp15.v1;
                                                while (while_method_7(v324)){
                                                    assert("Tensor range check" && 0 <= v324 && v324 < 4);
                                                    int v327;
                                                    v327 = v324 + v323;
                                                    float v328;
                                                    v328 = v285[v327];
                                                    float v329;
                                                    v329 = v325 + v328;
                                                    v325 = v329;
                                                    v324 += 1 ;
                                                }
                                                auto v330 = cooperative_groups::coalesced_threads();
                                                int v331;
                                                v331 = threadIdx.x;
                                                int v332;
                                                v332 = v331 / 16;
                                                auto v333 = cooperative_groups::labeled_partition(v330,v332);
                                                Closure2 v334{};
                                                float v335;
                                                v335 = cooperative_groups::inclusive_scan(v333, v325, v334);
                                                float v336;
                                                v336 = v333.shfl_up(v335,1);
                                                bool v337;
                                                v337 = v333.thread_rank() == 0;
                                                float v338;
                                                if (v337){
                                                    v338 = 0.0f;
                                                } else {
                                                    v338 = v336;
                                                }
                                                float v339;
                                                v339 = v333.shfl(v335,v333.num_threads()-1);
                                                float v340;
                                                v340 = v320 + v338;
                                                int v341; float v342;
                                                Tuple8 tmp16 = Tuple8{0, v340};
                                                v341 = tmp16.v0; v342 = tmp16.v1;
                                                while (while_method_7(v341)){
                                                    assert("Tensor range check" && 0 <= v341 && v341 < 4);
                                                    int v344;
                                                    v344 = v341 + v323;
                                                    float v345;
                                                    v345 = v285[v344];
                                                    float v346;
                                                    v346 = v342 + v345;
                                                    assert("Tensor range check" && 0 <= v341 && v341 < 4);
                                                    v319[v344] = v346;
                                                    v342 = v346;
                                                    v341 += 1 ;
                                                }
                                                float v347;
                                                v347 = v320 + v339;
                                                v320 = v347;
                                                v321 += 1 ;
                                            }
                                            float v348[4];
                                            bool v349[4];
                                            int v350;
                                            v350 = 0;
                                            while (while_method_4(v350)){
                                                int v352;
                                                v352 = 0;
                                                while (while_method_7(v352)){
                                                    assert("Tensor range check" && 0 <= v350 && v350 < 1);
                                                    assert("Tensor range check" && 0 <= v352 && v352 < 4);
                                                    int v354;
                                                    v354 = 4 * v350;
                                                    int v355;
                                                    v355 = v354 + v352;
                                                    float v356;
                                                    v356 = v319[v355];
                                                    float v357;
                                                    v357 = v285[v355];
                                                    bool v358;
                                                    v358 = v357 > 0.0f;
                                                    assert("Tensor range check" && 0 <= v350 && v350 < 1);
                                                    assert("Tensor range check" && 0 <= v352 && v352 < 4);
                                                    v348[v355] = v356;
                                                    v349[v355] = v358;
                                                    v352 += 1 ;
                                                }
                                                v350 += 1 ;
                                            }
                                            float v359; bool v360;
                                            Tuple9 tmp17 = Tuple9{-1.0f / 0.0f, false};
                                            v359 = tmp17.v0; v360 = tmp17.v1;
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
                                                    float v367;
                                                    v367 = v348[v366];
                                                    bool v368;
                                                    v368 = v349[v366];
                                                    float v375; bool v376;
                                                    if (v360){
                                                        if (v368){
                                                            bool v369;
                                                            v369 = v359 >= v367;
                                                            float v370;
                                                            if (v369){
                                                                v370 = v359;
                                                            } else {
                                                                v370 = v367;
                                                            }
                                                            v375 = v370; v376 = true;
                                                        } else {
                                                            v375 = v359; v376 = v360;
                                                        }
                                                    } else {
                                                        if (v368){
                                                            v375 = v367; v376 = v368;
                                                        } else {
                                                            v375 = v359; v376 = v360;
                                                        }
                                                    }
                                                    v359 = v375;
                                                    v360 = v376;
                                                    v363 += 1 ;
                                                }
                                                v361 += 1 ;
                                            }
                                            auto v377 = cooperative_groups::coalesced_threads();
                                            int v378;
                                            v378 = threadIdx.x;
                                            int v379;
                                            v379 = v378 / 16;
                                            auto v380 = cooperative_groups::labeled_partition(v377,v379);
                                            Closure3 v381{};
                                            float v382; bool v383;
                                            Tuple9 tmp18 = cooperative_groups::reduce(v380, Tuple9{v359, v360}, v381);
                                            v382 = tmp18.v0; v383 = tmp18.v1;
                                            bool v384;
                                            v384 = v383 == false;
                                            if (v384){
                                                assert("The local reduce must be true." && v383);
                                            } else {
                                            }
                                            float v386[4];
                                            int v387[4];
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
                                                    int v394;
                                                    v394 = v286[v393];
                                                    float v395;
                                                    v395 = curand_uniform(&v121);
                                                    assert("Tensor range check" && 0 <= v388 && v388 < 1);
                                                    assert("Tensor range check" && 0 <= v390 && v390 < 4);
                                                    v386[v393] = v395;
                                                    v387[v393] = v394;
                                                    v390 += 1 ;
                                                }
                                                v388 += 1 ;
                                            }
                                            float v396; int v397;
                                            Tuple10 tmp19 = Tuple10{0.0f, 2147483647};
                                            v396 = tmp19.v0; v397 = tmp19.v1;
                                            int v398;
                                            v398 = 0;
                                            while (while_method_4(v398)){
                                                int v400;
                                                v400 = 0;
                                                while (while_method_7(v400)){
                                                    assert("Tensor range check" && 0 <= v398 && v398 < 1);
                                                    assert("Tensor range check" && 0 <= v400 && v400 < 4);
                                                    int v402;
                                                    v402 = 4 * v398;
                                                    int v403;
                                                    v403 = v402 + v400;
                                                    float v404;
                                                    v404 = v386[v403];
                                                    int v405;
                                                    v405 = v387[v403];
                                                    bool v406;
                                                    v406 = v397 < v405;
                                                    float v407; int v408;
                                                    if (v406){
                                                        v407 = v396; v408 = v397;
                                                    } else {
                                                        v407 = v404; v408 = v405;
                                                    }
                                                    v396 = v407;
                                                    v397 = v408;
                                                    v400 += 1 ;
                                                }
                                                v398 += 1 ;
                                            }
                                            auto v409 = cooperative_groups::coalesced_threads();
                                            int v410;
                                            v410 = threadIdx.x;
                                            int v411;
                                            v411 = v410 / 16;
                                            auto v412 = cooperative_groups::labeled_partition(v409,v411);
                                            Closure4 v413{};
                                            float v414; int v415;
                                            Tuple10 tmp20 = cooperative_groups::reduce(v412, Tuple10{v396, v397}, v413);
                                            v414 = tmp20.v0; v415 = tmp20.v1;
                                            float v416;
                                            v416 = v382 * v414;
                                            int v417[4];
                                            bool v418[4];
                                            int v419;
                                            v419 = 0;
                                            while (while_method_4(v419)){
                                                int v421;
                                                v421 = 0;
                                                while (while_method_7(v421)){
                                                    assert("Tensor range check" && 0 <= v419 && v419 < 1);
                                                    assert("Tensor range check" && 0 <= v421 && v421 < 4);
                                                    int v423;
                                                    v423 = 4 * v419;
                                                    int v424;
                                                    v424 = v423 + v421;
                                                    float v425;
                                                    v425 = v348[v424];
                                                    bool v426;
                                                    v426 = v349[v424];
                                                    int v427;
                                                    v427 = v286[v424];
                                                    int v430; bool v431;
                                                    if (v426){
                                                        float v428;
                                                        v428 = v425 - v416;
                                                        bool v429;
                                                        v429 = v428 >= 0.0f;
                                                        v430 = v427; v431 = v429;
                                                    } else {
                                                        v430 = 2147483647; v431 = false;
                                                    }
                                                    assert("Tensor range check" && 0 <= v419 && v419 < 1);
                                                    assert("Tensor range check" && 0 <= v421 && v421 < 4);
                                                    v417[v424] = v430;
                                                    v418[v424] = v431;
                                                    v421 += 1 ;
                                                }
                                                v419 += 1 ;
                                            }
                                            int v432; bool v433;
                                            Tuple11 tmp21 = Tuple11{2147483647, false};
                                            v432 = tmp21.v0; v433 = tmp21.v1;
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
                                                    int v440;
                                                    v440 = v417[v439];
                                                    bool v441;
                                                    v441 = v418[v439];
                                                    int v448; bool v449;
                                                    if (v433){
                                                        if (v441){
                                                            bool v442;
                                                            v442 = v432 < v440;
                                                            int v443;
                                                            if (v442){
                                                                v443 = v432;
                                                            } else {
                                                                v443 = v440;
                                                            }
                                                            v448 = v443; v449 = true;
                                                        } else {
                                                            v448 = v432; v449 = v433;
                                                        }
                                                    } else {
                                                        if (v441){
                                                            v448 = v440; v449 = v441;
                                                        } else {
                                                            v448 = v432; v449 = v433;
                                                        }
                                                    }
                                                    v432 = v448;
                                                    v433 = v449;
                                                    v436 += 1 ;
                                                }
                                                v434 += 1 ;
                                            }
                                            auto v450 = cooperative_groups::coalesced_threads();
                                            int v451;
                                            v451 = threadIdx.x;
                                            int v452;
                                            v452 = v451 / 16;
                                            auto v453 = cooperative_groups::labeled_partition(v450,v452);
                                            Closure5 v454{};
                                            int v455; bool v456;
                                            Tuple11 tmp22 = cooperative_groups::reduce(v453, Tuple11{v432, v433}, v454);
                                            v455 = tmp22.v0; v456 = tmp22.v1;
                                            bool v457;
                                            v457 = v456 == false;
                                            if (v457){
                                                assert("The local reduce must be true." && v456);
                                            } else {
                                            }
                                            float v459; int v460;
                                            Tuple10 tmp23 = Tuple10{0.0f, 2147483647};
                                            v459 = tmp23.v0; v460 = tmp23.v1;
                                            int v461;
                                            v461 = 0;
                                            while (while_method_4(v461)){
                                                int v463;
                                                v463 = 0;
                                                while (while_method_7(v463)){
                                                    assert("Tensor range check" && 0 <= v461 && v461 < 1);
                                                    assert("Tensor range check" && 0 <= v463 && v463 < 4);
                                                    int v465;
                                                    v465 = 4 * v461;
                                                    int v466;
                                                    v466 = v465 + v463;
                                                    float v467;
                                                    v467 = v285[v466];
                                                    int v468;
                                                    v468 = v286[v466];
                                                    bool v469;
                                                    v469 = v460 == v455;
                                                    float v473; int v474;
                                                    if (v469){
                                                        v473 = v459; v474 = v460;
                                                    } else {
                                                        bool v470;
                                                        v470 = v468 == v455;
                                                        if (v470){
                                                            v473 = v467; v474 = v468;
                                                        } else {
                                                            v473 = v459; v474 = v460;
                                                        }
                                                    }
                                                    v459 = v473;
                                                    v460 = v474;
                                                    v463 += 1 ;
                                                }
                                                v461 += 1 ;
                                            }
                                            auto v475 = cooperative_groups::coalesced_threads();
                                            int v476;
                                            v476 = threadIdx.x;
                                            int v477;
                                            v477 = v476 / 16;
                                            auto v478 = cooperative_groups::labeled_partition(v475,v477);
                                            Closure6 v479{v455};
                                            float v480; int v481;
                                            Tuple10 tmp24 = cooperative_groups::reduce(v478, Tuple10{v459, v460}, v479);
                                            v480 = tmp24.v0; v481 = tmp24.v1;
                                            bool v482;
                                            v482 = v481 == 2147483647;
                                            bool v483;
                                            v483 = v482 != true;
                                            bool v484;
                                            v484 = v483 == false;
                                            if (v484){
                                                assert("Expected a valid action id in get_prob." && v483);
                                            } else {
                                            }
                                            int v486;
                                            v486 = 0;
                                            while (while_method_4(v486)){
                                                assert("Tensor range check" && 0 <= v486 && v486 < 1);
                                                assert("Tensor range check" && 0 <= v486 && v486 < 1);
                                                v486 += 1 ;
                                            }
                                            assert("Tensor range check" && 0 <= v277 && v277 < 256);
                                            v252[v277] = v480;
                                            v254[v277] = v455;
                                            v265 += 1 ;
                                        }
                                        __syncthreads();
                                        assert("Tensor range check" && 0 <= v256 && v256 < 256);
                                        float v488;
                                        v488 = v252[v256];
                                        int v489;
                                        v489 = v254[v256];
                                        __syncthreads();
                                        bool v490;
                                        v490 = 0 == v489;
                                        Union12 v499;
                                        if (v490){
                                            v499 = Union12{Union12_1{}};
                                        } else {
                                            bool v492;
                                            v492 = 1 == v489;
                                            if (v492){
                                                v499 = Union12{Union12_0{}};
                                            } else {
                                                bool v494;
                                                v494 = 2 == v489;
                                                if (v494){
                                                    v499 = Union12{Union12_2{}};
                                                } else {
                                                    printf("%s\n", "Invalid output id in the Leduc model.");
                                                    __trap();
                                                }
                                            }
                                        }
                                        Union1 v531;
                                        switch (v499.tag) {
                                            case 0: { // AA_Call
                                                v531 = Union1{Union1_0{}};
                                                break;
                                            }
                                            case 1: { // AA_Fold
                                                int v500;
                                                v500 = v109[0];
                                                int v502; int v503;
                                                Tuple7 tmp25 = Tuple7{1, v500};
                                                v502 = tmp25.v0; v503 = tmp25.v1;
                                                while (while_method_0(v502)){
                                                    bool v505;
                                                    v505 = 0 <= v502;
                                                    bool v507;
                                                    if (v505){
                                                        bool v506;
                                                        v506 = v502 < 2;
                                                        v507 = v506;
                                                    } else {
                                                        v507 = false;
                                                    }
                                                    bool v508;
                                                    v508 = v507 == false;
                                                    if (v508){
                                                        assert("Index must be in range." && v507);
                                                    } else {
                                                    }
                                                    int v510;
                                                    v510 = v109[v502];
                                                    bool v512;
                                                    v512 = v503 >= v510;
                                                    int v513;
                                                    if (v512){
                                                        v513 = v503;
                                                    } else {
                                                        v513 = v510;
                                                    }
                                                    v503 = v513;
                                                    v502 += 1 ;
                                                }
                                                bool v515;
                                                if (v112){
                                                    bool v514;
                                                    v514 = v108 < 2;
                                                    v515 = v514;
                                                } else {
                                                    v515 = false;
                                                }
                                                bool v516;
                                                v516 = v515 == false;
                                                if (v516){
                                                    assert("Index must be in range." && v515);
                                                } else {
                                                }
                                                int v518;
                                                v518 = v109[v108];
                                                bool v520;
                                                v520 = v518 == v503;
                                                if (v520){
                                                    v531 = Union1{Union1_0{}};
                                                } else {
                                                    v531 = Union1{Union1_1{}};
                                                }
                                                break;
                                            }
                                            case 2: { // AA_Raise
                                                bool v525;
                                                v525 = v110 > 0;
                                                if (v525){
                                                    v531 = Union1{Union1_2{}};
                                                } else {
                                                    v531 = Union1{Union1_0{}};
                                                }
                                                break;
                                            }
                                            default: {
                                                assert("Invalid tag." && false); __trap();
                                            }
                                        }
                                        int v532;
                                        v532 = sizeof(Union1);
                                        unsigned long long v533;
                                        v533 = (unsigned long long)v532;
                                        bool v534;
                                        v534 = v533 <= 98304ull;
                                        bool v535;
                                        v535 = v534 == false;
                                        if (v535){
                                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v534);
                                        } else {
                                        }
                                        extern __shared__ unsigned char v537[];
                                        bool v538;
                                        v538 = v533 <= v533;
                                        bool v539;
                                        v539 = v538 == false;
                                        if (v539){
                                            assert("The length of the partition has to be less than or equal to the length of the base array." && v538);
                                        } else {
                                        }
                                        Union1 * v541;
                                        v541 = reinterpret_cast<Union1 *>(&v537[0ull]);
                                        int v543;
                                        v543 = threadIdx.x;
                                        bool v544;
                                        v544 = v543 == 0;
                                        if (v544){
                                            v541[0] = v531;
                                        } else {
                                        }
                                        __syncthreads();
                                        Union1 v545;
                                        v545 = v541[0];
                                        __syncthreads();
                                        Union7 v546;
                                        v546 = Union7{Union7_1{v108, v545}};
                                        v58.push(v546);
                                        Union4 v653;
                                        switch (v105.tag) {
                                            case 0: { // None
                                                switch (v545.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v609;
                                                            v609 = v108 ^ 1;
                                                            v653 = Union4{Union4_2{v105, false, v107, v609, v109, v110}};
                                                        } else {
                                                            v653 = Union4{Union4_0{v105, v106, v107, v108, v109, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v653 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        bool v613;
                                                        v613 = v110 > 0;
                                                        if (v613){
                                                            int v614;
                                                            v614 = v108 ^ 1;
                                                            int v615;
                                                            v615 = -1 + v110;
                                                            int v616; int v617;
                                                            Tuple7 tmp26 = Tuple7{0, 0};
                                                            v616 = tmp26.v0; v617 = tmp26.v1;
                                                            while (while_method_0(v616)){
                                                                bool v619;
                                                                v619 = 0 <= v616;
                                                                bool v621;
                                                                if (v619){
                                                                    bool v620;
                                                                    v620 = v616 < 2;
                                                                    v621 = v620;
                                                                } else {
                                                                    v621 = false;
                                                                }
                                                                bool v622;
                                                                v622 = v621 == false;
                                                                if (v622){
                                                                    assert("Index must be in range." && v621);
                                                                } else {
                                                                }
                                                                int v624;
                                                                v624 = v109[v616];
                                                                bool v626;
                                                                v626 = v617 >= v624;
                                                                int v627;
                                                                if (v626){
                                                                    v627 = v617;
                                                                } else {
                                                                    v627 = v624;
                                                                }
                                                                v617 = v627;
                                                                v616 += 1 ;
                                                            }
                                                            static_array<int,2> v628;
                                                            int v630;
                                                            v630 = 0;
                                                            while (while_method_0(v630)){
                                                                v628[v630] = v617;
                                                                v630 += 1 ;
                                                            }
                                                            static_array<int,2> v632;
                                                            int v634;
                                                            v634 = 0;
                                                            while (while_method_0(v634)){
                                                                bool v636;
                                                                v636 = 0 <= v634;
                                                                bool v638;
                                                                if (v636){
                                                                    bool v637;
                                                                    v637 = v634 < 2;
                                                                    v638 = v637;
                                                                } else {
                                                                    v638 = false;
                                                                }
                                                                bool v639;
                                                                v639 = v638 == false;
                                                                if (v639){
                                                                    assert("Index must be in range." && v638);
                                                                } else {
                                                                }
                                                                int v641;
                                                                v641 = v628[v634];
                                                                bool v643;
                                                                v643 = v634 == v108;
                                                                int v645;
                                                                if (v643){
                                                                    int v644;
                                                                    v644 = v641 + 2;
                                                                    v645 = v644;
                                                                } else {
                                                                    v645 = v641;
                                                                }
                                                                v632[v634] = v645;
                                                                v634 += 1 ;
                                                            }
                                                            v653 = Union4{Union4_2{v105, false, v107, v614, v632, v615}};
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
                                                Union6 v547 = v105.case1.v0;
                                                switch (v545.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v549;
                                                            v549 = v108 ^ 1;
                                                            v653 = Union4{Union4_2{v105, false, v107, v549, v109, v110}};
                                                        } else {
                                                            int v551; int v552;
                                                            Tuple7 tmp27 = Tuple7{0, 0};
                                                            v551 = tmp27.v0; v552 = tmp27.v1;
                                                            while (while_method_0(v551)){
                                                                bool v554;
                                                                v554 = 0 <= v551;
                                                                bool v556;
                                                                if (v554){
                                                                    bool v555;
                                                                    v555 = v551 < 2;
                                                                    v556 = v555;
                                                                } else {
                                                                    v556 = false;
                                                                }
                                                                bool v557;
                                                                v557 = v556 == false;
                                                                if (v557){
                                                                    assert("Index must be in range." && v556);
                                                                } else {
                                                                }
                                                                int v559;
                                                                v559 = v109[v551];
                                                                bool v561;
                                                                v561 = v552 >= v559;
                                                                int v562;
                                                                if (v561){
                                                                    v562 = v552;
                                                                } else {
                                                                    v562 = v559;
                                                                }
                                                                v552 = v562;
                                                                v551 += 1 ;
                                                            }
                                                            static_array<int,2> v563;
                                                            int v565;
                                                            v565 = 0;
                                                            while (while_method_0(v565)){
                                                                v563[v565] = v552;
                                                                v565 += 1 ;
                                                            }
                                                            v653 = Union4{Union4_4{v105, v106, v107, v108, v563, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v653 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        bool v569;
                                                        v569 = v110 > 0;
                                                        if (v569){
                                                            int v570;
                                                            v570 = v108 ^ 1;
                                                            int v571;
                                                            v571 = -1 + v110;
                                                            int v572; int v573;
                                                            Tuple7 tmp28 = Tuple7{0, 0};
                                                            v572 = tmp28.v0; v573 = tmp28.v1;
                                                            while (while_method_0(v572)){
                                                                bool v575;
                                                                v575 = 0 <= v572;
                                                                bool v577;
                                                                if (v575){
                                                                    bool v576;
                                                                    v576 = v572 < 2;
                                                                    v577 = v576;
                                                                } else {
                                                                    v577 = false;
                                                                }
                                                                bool v578;
                                                                v578 = v577 == false;
                                                                if (v578){
                                                                    assert("Index must be in range." && v577);
                                                                } else {
                                                                }
                                                                int v580;
                                                                v580 = v109[v572];
                                                                bool v582;
                                                                v582 = v573 >= v580;
                                                                int v583;
                                                                if (v582){
                                                                    v583 = v573;
                                                                } else {
                                                                    v583 = v580;
                                                                }
                                                                v573 = v583;
                                                                v572 += 1 ;
                                                            }
                                                            static_array<int,2> v584;
                                                            int v586;
                                                            v586 = 0;
                                                            while (while_method_0(v586)){
                                                                v584[v586] = v573;
                                                                v586 += 1 ;
                                                            }
                                                            static_array<int,2> v588;
                                                            int v590;
                                                            v590 = 0;
                                                            while (while_method_0(v590)){
                                                                bool v592;
                                                                v592 = 0 <= v590;
                                                                bool v594;
                                                                if (v592){
                                                                    bool v593;
                                                                    v593 = v590 < 2;
                                                                    v594 = v593;
                                                                } else {
                                                                    v594 = false;
                                                                }
                                                                bool v595;
                                                                v595 = v594 == false;
                                                                if (v595){
                                                                    assert("Index must be in range." && v594);
                                                                } else {
                                                                }
                                                                int v597;
                                                                v597 = v584[v590];
                                                                bool v599;
                                                                v599 = v590 == v108;
                                                                int v601;
                                                                if (v599){
                                                                    int v600;
                                                                    v600 = v597 + 4;
                                                                    v601 = v600;
                                                                } else {
                                                                    v601 = v597;
                                                                }
                                                                v588[v590] = v601;
                                                                v590 += 1 ;
                                                            }
                                                            v653 = Union4{Union4_2{v105, false, v107, v570, v588, v571}};
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
                                        v984 = Union3{Union3_1{v653}};
                                        break;
                                    }
                                    case 1: { // Human
                                        Union8 v655;
                                        v655 = Union8{Union8_2{v105, v106, v107, v108, v109, v110}};
                                        v19.v5 = v655;
                                        Union3 v656;
                                        v656 = Union3{Union3_1{v62}};
                                        v19.v1 = v656;
                                        v984 = Union3{Union3_0{}};
                                        break;
                                    }
                                    case 2: { // Random
                                        curandStatePhilox4_32_10_t & v658 = v19.v4;
                                        curandStatePhilox4_32_10_t & v659 = v658;
                                        static_array_list<Union1,3> v660;
                                        v660 = static_array_list<Union1,3>{};
                                        v660.unsafe_set_length(1);
                                        Union1 v662;
                                        v662 = Union1{Union1_0{}};
                                        v660[0] = v662;
                                        int v664;
                                        v664 = v109[0];
                                        int v666;
                                        v666 = v109[1];
                                        bool v668;
                                        v668 = v664 == v666;
                                        bool v669;
                                        v669 = v668 != true;
                                        if (v669){
                                            Union1 v670;
                                            v670 = Union1{Union1_1{}};
                                            v660.push(v670);
                                        } else {
                                        }
                                        bool v671;
                                        v671 = v110 > 0;
                                        if (v671){
                                            Union1 v672;
                                            v672 = Union1{Union1_2{}};
                                            v660.push(v672);
                                        } else {
                                        }
                                        int v673;
                                        v673 = v660.length;
                                        int v674;
                                        v674 = v673 - 1;
                                        int v675;
                                        v675 = 0;
                                        while (while_method_1(v674, v675)){
                                            int v677;
                                            v677 = v660.length;
                                            int v678;
                                            v678 = int_range_22(v677, v675, v659);
                                            Union1 v679;
                                            v679 = v660[v675];
                                            Union1 v681;
                                            v681 = v660[v678];
                                            v660[v675] = v681;
                                            v660[v678] = v679;
                                            v675 += 1 ;
                                        }
                                        Union1 v683;
                                        v683 = v660.pop();
                                        int v684;
                                        v684 = sizeof(Union1);
                                        unsigned long long v685;
                                        v685 = (unsigned long long)v684;
                                        bool v686;
                                        v686 = v685 <= 98304ull;
                                        bool v687;
                                        v687 = v686 == false;
                                        if (v687){
                                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v686);
                                        } else {
                                        }
                                        extern __shared__ unsigned char v689[];
                                        bool v690;
                                        v690 = v685 <= v685;
                                        bool v691;
                                        v691 = v690 == false;
                                        if (v691){
                                            assert("The length of the partition has to be less than or equal to the length of the base array." && v690);
                                        } else {
                                        }
                                        Union1 * v693;
                                        v693 = reinterpret_cast<Union1 *>(&v689[0ull]);
                                        int v695;
                                        v695 = threadIdx.x;
                                        bool v696;
                                        v696 = v695 == 0;
                                        if (v696){
                                            v693[0] = v683;
                                        } else {
                                        }
                                        __syncthreads();
                                        Union1 v697;
                                        v697 = v693[0];
                                        __syncthreads();
                                        Union7 v698;
                                        v698 = Union7{Union7_1{v108, v697}};
                                        v58.push(v698);
                                        Union4 v803;
                                        switch (v105.tag) {
                                            case 0: { // None
                                                switch (v697.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v760;
                                                            v760 = v108 ^ 1;
                                                            v803 = Union4{Union4_2{v105, false, v107, v760, v109, v110}};
                                                        } else {
                                                            v803 = Union4{Union4_0{v105, v106, v107, v108, v109, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v803 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        if (v671){
                                                            int v764;
                                                            v764 = v108 ^ 1;
                                                            int v765;
                                                            v765 = -1 + v110;
                                                            int v766; int v767;
                                                            Tuple7 tmp29 = Tuple7{0, 0};
                                                            v766 = tmp29.v0; v767 = tmp29.v1;
                                                            while (while_method_0(v766)){
                                                                bool v769;
                                                                v769 = 0 <= v766;
                                                                bool v771;
                                                                if (v769){
                                                                    bool v770;
                                                                    v770 = v766 < 2;
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
                                                                v774 = v109[v766];
                                                                bool v776;
                                                                v776 = v767 >= v774;
                                                                int v777;
                                                                if (v776){
                                                                    v777 = v767;
                                                                } else {
                                                                    v777 = v774;
                                                                }
                                                                v767 = v777;
                                                                v766 += 1 ;
                                                            }
                                                            static_array<int,2> v778;
                                                            int v780;
                                                            v780 = 0;
                                                            while (while_method_0(v780)){
                                                                v778[v780] = v767;
                                                                v780 += 1 ;
                                                            }
                                                            static_array<int,2> v782;
                                                            int v784;
                                                            v784 = 0;
                                                            while (while_method_0(v784)){
                                                                bool v786;
                                                                v786 = 0 <= v784;
                                                                bool v788;
                                                                if (v786){
                                                                    bool v787;
                                                                    v787 = v784 < 2;
                                                                    v788 = v787;
                                                                } else {
                                                                    v788 = false;
                                                                }
                                                                bool v789;
                                                                v789 = v788 == false;
                                                                if (v789){
                                                                    assert("Index must be in range." && v788);
                                                                } else {
                                                                }
                                                                int v791;
                                                                v791 = v778[v784];
                                                                bool v793;
                                                                v793 = v784 == v108;
                                                                int v795;
                                                                if (v793){
                                                                    int v794;
                                                                    v794 = v791 + 2;
                                                                    v795 = v794;
                                                                } else {
                                                                    v795 = v791;
                                                                }
                                                                v782[v784] = v795;
                                                                v784 += 1 ;
                                                            }
                                                            v803 = Union4{Union4_2{v105, false, v107, v764, v782, v765}};
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
                                                Union6 v699 = v105.case1.v0;
                                                switch (v697.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v701;
                                                            v701 = v108 ^ 1;
                                                            v803 = Union4{Union4_2{v105, false, v107, v701, v109, v110}};
                                                        } else {
                                                            int v703; int v704;
                                                            Tuple7 tmp30 = Tuple7{0, 0};
                                                            v703 = tmp30.v0; v704 = tmp30.v1;
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
                                                                v711 = v109[v703];
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
                                                            v803 = Union4{Union4_4{v105, v106, v107, v108, v715, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v803 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        if (v671){
                                                            int v721;
                                                            v721 = v108 ^ 1;
                                                            int v722;
                                                            v722 = -1 + v110;
                                                            int v723; int v724;
                                                            Tuple7 tmp31 = Tuple7{0, 0};
                                                            v723 = tmp31.v0; v724 = tmp31.v1;
                                                            while (while_method_0(v723)){
                                                                bool v726;
                                                                v726 = 0 <= v723;
                                                                bool v728;
                                                                if (v726){
                                                                    bool v727;
                                                                    v727 = v723 < 2;
                                                                    v728 = v727;
                                                                } else {
                                                                    v728 = false;
                                                                }
                                                                bool v729;
                                                                v729 = v728 == false;
                                                                if (v729){
                                                                    assert("Index must be in range." && v728);
                                                                } else {
                                                                }
                                                                int v731;
                                                                v731 = v109[v723];
                                                                bool v733;
                                                                v733 = v724 >= v731;
                                                                int v734;
                                                                if (v733){
                                                                    v734 = v724;
                                                                } else {
                                                                    v734 = v731;
                                                                }
                                                                v724 = v734;
                                                                v723 += 1 ;
                                                            }
                                                            static_array<int,2> v735;
                                                            int v737;
                                                            v737 = 0;
                                                            while (while_method_0(v737)){
                                                                v735[v737] = v724;
                                                                v737 += 1 ;
                                                            }
                                                            static_array<int,2> v739;
                                                            int v741;
                                                            v741 = 0;
                                                            while (while_method_0(v741)){
                                                                bool v743;
                                                                v743 = 0 <= v741;
                                                                bool v745;
                                                                if (v743){
                                                                    bool v744;
                                                                    v744 = v741 < 2;
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
                                                                v748 = v735[v741];
                                                                bool v750;
                                                                v750 = v741 == v108;
                                                                int v752;
                                                                if (v750){
                                                                    int v751;
                                                                    v751 = v748 + 4;
                                                                    v752 = v751;
                                                                } else {
                                                                    v752 = v748;
                                                                }
                                                                v739[v741] = v752;
                                                                v741 += 1 ;
                                                            }
                                                            v803 = Union4{Union4_2{v105, false, v107, v721, v739, v722}};
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
                                        v984 = Union3{Union3_1{v803}};
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                break;
                            }
                            case 3: { // RoundWithAction
                                Union5 v808 = v62.case3.v0; bool v809 = v62.case3.v1; static_array<Union6,2> v810 = v62.case3.v2; int v811 = v62.case3.v3; static_array<int,2> v812 = v62.case3.v4; int v813 = v62.case3.v5; Union1 v814 = v62.case3.v6;
                                Union7 v815;
                                v815 = Union7{Union7_1{v811, v814}};
                                v58.push(v815);
                                Union4 v922;
                                switch (v808.tag) {
                                    case 0: { // None
                                        switch (v814.tag) {
                                            case 0: { // Call
                                                if (v809){
                                                    int v878;
                                                    v878 = v811 ^ 1;
                                                    v922 = Union4{Union4_2{v808, false, v810, v878, v812, v813}};
                                                } else {
                                                    v922 = Union4{Union4_0{v808, v809, v810, v811, v812, v813}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v922 = Union4{Union4_5{v808, v809, v810, v811, v812, v813}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                bool v882;
                                                v882 = v813 > 0;
                                                if (v882){
                                                    int v883;
                                                    v883 = v811 ^ 1;
                                                    int v884;
                                                    v884 = -1 + v813;
                                                    int v885; int v886;
                                                    Tuple7 tmp32 = Tuple7{0, 0};
                                                    v885 = tmp32.v0; v886 = tmp32.v1;
                                                    while (while_method_0(v885)){
                                                        bool v888;
                                                        v888 = 0 <= v885;
                                                        bool v890;
                                                        if (v888){
                                                            bool v889;
                                                            v889 = v885 < 2;
                                                            v890 = v889;
                                                        } else {
                                                            v890 = false;
                                                        }
                                                        bool v891;
                                                        v891 = v890 == false;
                                                        if (v891){
                                                            assert("Index must be in range." && v890);
                                                        } else {
                                                        }
                                                        int v893;
                                                        v893 = v812[v885];
                                                        bool v895;
                                                        v895 = v886 >= v893;
                                                        int v896;
                                                        if (v895){
                                                            v896 = v886;
                                                        } else {
                                                            v896 = v893;
                                                        }
                                                        v886 = v896;
                                                        v885 += 1 ;
                                                    }
                                                    static_array<int,2> v897;
                                                    int v899;
                                                    v899 = 0;
                                                    while (while_method_0(v899)){
                                                        v897[v899] = v886;
                                                        v899 += 1 ;
                                                    }
                                                    static_array<int,2> v901;
                                                    int v903;
                                                    v903 = 0;
                                                    while (while_method_0(v903)){
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
                                                        v910 = v897[v903];
                                                        bool v912;
                                                        v912 = v903 == v811;
                                                        int v914;
                                                        if (v912){
                                                            int v913;
                                                            v913 = v910 + 2;
                                                            v914 = v913;
                                                        } else {
                                                            v914 = v910;
                                                        }
                                                        v901[v903] = v914;
                                                        v903 += 1 ;
                                                    }
                                                    v922 = Union4{Union4_2{v808, false, v810, v883, v901, v884}};
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
                                        Union6 v816 = v808.case1.v0;
                                        switch (v814.tag) {
                                            case 0: { // Call
                                                if (v809){
                                                    int v818;
                                                    v818 = v811 ^ 1;
                                                    v922 = Union4{Union4_2{v808, false, v810, v818, v812, v813}};
                                                } else {
                                                    int v820; int v821;
                                                    Tuple7 tmp33 = Tuple7{0, 0};
                                                    v820 = tmp33.v0; v821 = tmp33.v1;
                                                    while (while_method_0(v820)){
                                                        bool v823;
                                                        v823 = 0 <= v820;
                                                        bool v825;
                                                        if (v823){
                                                            bool v824;
                                                            v824 = v820 < 2;
                                                            v825 = v824;
                                                        } else {
                                                            v825 = false;
                                                        }
                                                        bool v826;
                                                        v826 = v825 == false;
                                                        if (v826){
                                                            assert("Index must be in range." && v825);
                                                        } else {
                                                        }
                                                        int v828;
                                                        v828 = v812[v820];
                                                        bool v830;
                                                        v830 = v821 >= v828;
                                                        int v831;
                                                        if (v830){
                                                            v831 = v821;
                                                        } else {
                                                            v831 = v828;
                                                        }
                                                        v821 = v831;
                                                        v820 += 1 ;
                                                    }
                                                    static_array<int,2> v832;
                                                    int v834;
                                                    v834 = 0;
                                                    while (while_method_0(v834)){
                                                        v832[v834] = v821;
                                                        v834 += 1 ;
                                                    }
                                                    v922 = Union4{Union4_4{v808, v809, v810, v811, v832, v813}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v922 = Union4{Union4_5{v808, v809, v810, v811, v812, v813}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                bool v838;
                                                v838 = v813 > 0;
                                                if (v838){
                                                    int v839;
                                                    v839 = v811 ^ 1;
                                                    int v840;
                                                    v840 = -1 + v813;
                                                    int v841; int v842;
                                                    Tuple7 tmp34 = Tuple7{0, 0};
                                                    v841 = tmp34.v0; v842 = tmp34.v1;
                                                    while (while_method_0(v841)){
                                                        bool v844;
                                                        v844 = 0 <= v841;
                                                        bool v846;
                                                        if (v844){
                                                            bool v845;
                                                            v845 = v841 < 2;
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
                                                        v849 = v812[v841];
                                                        bool v851;
                                                        v851 = v842 >= v849;
                                                        int v852;
                                                        if (v851){
                                                            v852 = v842;
                                                        } else {
                                                            v852 = v849;
                                                        }
                                                        v842 = v852;
                                                        v841 += 1 ;
                                                    }
                                                    static_array<int,2> v853;
                                                    int v855;
                                                    v855 = 0;
                                                    while (while_method_0(v855)){
                                                        v853[v855] = v842;
                                                        v855 += 1 ;
                                                    }
                                                    static_array<int,2> v857;
                                                    int v859;
                                                    v859 = 0;
                                                    while (while_method_0(v859)){
                                                        bool v861;
                                                        v861 = 0 <= v859;
                                                        bool v863;
                                                        if (v861){
                                                            bool v862;
                                                            v862 = v859 < 2;
                                                            v863 = v862;
                                                        } else {
                                                            v863 = false;
                                                        }
                                                        bool v864;
                                                        v864 = v863 == false;
                                                        if (v864){
                                                            assert("Index must be in range." && v863);
                                                        } else {
                                                        }
                                                        int v866;
                                                        v866 = v853[v859];
                                                        bool v868;
                                                        v868 = v859 == v811;
                                                        int v870;
                                                        if (v868){
                                                            int v869;
                                                            v869 = v866 + 4;
                                                            v870 = v869;
                                                        } else {
                                                            v870 = v866;
                                                        }
                                                        v857[v859] = v870;
                                                        v859 += 1 ;
                                                    }
                                                    v922 = Union4{Union4_2{v808, false, v810, v839, v857, v840}};
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
                                v984 = Union3{Union3_1{v922}};
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
                                v94 = compare_hands_25(v81, v82, v83, v84, v85, v86);
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
                                v984 = Union3{Union3_0{}};
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
                                v984 = Union3{Union3_0{}};
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
                v60 = v984;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v985;
    v985 = threadIdx.x;
    int v986;
    v986 = blockIdx.x;
    int v987;
    v987 = v986 * 256;
    int v988;
    v988 = v985 + v987;
    bool v989;
    v989 = v988 == 0;
    if (v989){
        Union8 & v990 = v19.v5;
        static_array<Union2,2> & v991 = v19.v3;
        static_array_list<Union7,32> & v992 = v19.v2;
        Union3 & v993 = v19.v1;
        unsigned int & v994 = v19.v0;
        return f_29(v0, v994, v993, v992, v991, v990);
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
    while (while_method_9(v27)){
        int v29;
        v29 = 0;
        while (while_method_5(v29)){
            int v31;
            v31 = 0;
            while (while_method_0(v31)){
                Union4 v33;
                v33 = Union4{Union4_1{}};
                method_46(v0, v1, v2, v26, v31, v33);
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
                v42 = reinterpret_cast<double *>(&v1[11010048ull]);
                double * v44;
                v44 = reinterpret_cast<double *>(&v1[11403264ull]);
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
                int v51; double v52;
                Tuple12 tmp55 = Tuple12{0, 1.0};
                v51 = tmp55.v0; v52 = tmp55.v1;
                while (while_method_0(v51)){
                    assert("Tensor range check" && 0 <= v51 && v51 < 2);
                    int v54;
                    v54 = v51 + v50;
                    int v55; double v56;
                    Tuple12 tmp56 = Tuple12{0, 0.0};
                    v55 = tmp56.v0; v56 = tmp56.v1;
                    while (while_method_7(v55)){
                        assert("Tensor range check" && 0 <= v55 && v55 < 4);
                        int v58;
                        v58 = 12288 * v55;
                        int v59;
                        v59 = v58 + v54;
                        double v60;
                        v60 = v42[v59];
                        double v61;
                        v61 = v44[v59];
                        double v62;
                        v62 = v60 - v61;
                        double v63;
                        v63 = exp(v62);
                        double v64;
                        v64 = v56 + v63;
                        v56 = v64;
                        v55 += 1 ;
                    }
                    double v65;
                    v65 = v52 * v56;
                    v52 = v65;
                    v51 += 1 ;
                }
                float v66;
                v66 = (float)v52;
                int v67;
                v67 = 0;
                while (while_method_7(v67)){
                    double * v69;
                    v69 = reinterpret_cast<double *>(&v1[11010048ull]);
                    double * v71;
                    v71 = reinterpret_cast<double *>(&v1[11403264ull]);
                    int v73;
                    v73 = threadIdx.x;
                    int v74;
                    v74 = blockIdx.x;
                    int v75;
                    v75 = v74 * 256;
                    int v76;
                    v76 = v73 + v75;
                    assert("Tensor range check" && 0 <= v76 && v76 < 6144);
                    int v77;
                    v77 = 2 * v76;
                    int v78; double v79;
                    Tuple12 tmp57 = Tuple12{0, 1.0};
                    v78 = tmp57.v0; v79 = tmp57.v1;
                    while (while_method_0(v78)){
                        assert("Tensor range check" && 0 <= v78 && v78 < 2);
                        int v81;
                        v81 = v78 + v77;
                        int v82; double v83;
                        Tuple12 tmp58 = Tuple12{0, 0.0};
                        v82 = tmp58.v0; v83 = tmp58.v1;
                        while (while_method_7(v82)){
                            assert("Tensor range check" && 0 <= v82 && v82 < 4);
                            int v85;
                            v85 = 12288 * v82;
                            int v86;
                            v86 = v85 + v81;
                            double v87;
                            v87 = v69[v86];
                            double v88;
                            v88 = v71[v86];
                            double v89;
                            v89 = v87 - v88;
                            double v90;
                            v90 = exp(v89);
                            bool v91;
                            v91 = v67 == v82;
                            bool v92;
                            v92 = v91 != true;
                            double v93;
                            if (v92){
                                v93 = v90;
                            } else {
                                v93 = 0.0;
                            }
                            double v94;
                            v94 = v83 + v93;
                            v83 = v94;
                            v82 += 1 ;
                        }
                        double v95;
                        v95 = v79 * v83;
                        v79 = v95;
                        v78 += 1 ;
                    }
                    float v96;
                    v96 = (float)v79;
                    float v97;
                    v97 = v66 - v96;
                    float v98;
                    v98 = v40 * v97;
                    assert("Tensor range check" && 0 <= v67 && v67 < 4);
                    assert("Tensor range check" && 0 <= v27 && v27 < 32);
                    int v99;
                    v99 = 32 * v67;
                    int v100;
                    v100 = v99 + v27;
                    float * v101;
                    v101 = v3+v100;
                    float * v103;
                    v103 = v4+v100;
                    float v105;
                    v105 = atomicAdd(v101,v98);
                    float v106;
                    v106 = atomicAdd(v103,v97);
                    v67 += 1 ;
                }
                static_array<float,2> & v107 = v26.v4;
                float * v108;
                v108 = reinterpret_cast<float *>(&v1[4718592ull]);
                int * v110;
                v110 = reinterpret_cast<int *>(&v0[131072ull]);
                float * v112;
                v112 = reinterpret_cast<float *>(&v0[131088ull]);
                float * v114;
                v114 = reinterpret_cast<float *>(&v0[131104ull]);
                double * v116;
                v116 = reinterpret_cast<double *>(&v1[11010048ull]);
                double * v118;
                v118 = reinterpret_cast<double *>(&v1[11403264ull]);
                int v120;
                v120 = 0;
                while (while_method_7(v120)){
                    assert("Tensor range check" && 0 <= v120 && v120 < 4);
                    int v122;
                    v122 = 12288 * v120;
                    int v123;
                    v123 = threadIdx.x;
                    assert("Tensor range check" && 0 <= v123 && v123 < 6144);
                    int v124;
                    v124 = 2 * v123;
                    int v125;
                    v125 = v124 + v122;
                    double * v126;
                    v126 = v116+v125;
                    double * v128;
                    v128 = v118+v125;
                    int v130;
                    v130 = 0;
                    while (while_method_0(v130)){
                        bool v132;
                        v132 = 0 <= v130;
                        bool v134;
                        if (v132){
                            bool v133;
                            v133 = v130 < 2;
                            v134 = v133;
                        } else {
                            v134 = false;
                        }
                        bool v135;
                        v135 = v134 == false;
                        if (v135){
                            assert("Index must be in range." && v134);
                        } else {
                        }
                        float v137;
                        v137 = v107[v130];
                        assert("Tensor range check" && 0 <= v130 && v130 < 2);
                        double v139;
                        v139 = v126[v130];
                        double v140;
                        v140 = v128[v130];
                        double v141;
                        v141 = v139 - v140;
                        double v142;
                        v142 = exp(v141);
                        float v143;
                        v143 = (float)v142;
                        float v144;
                        v144 = v137 * v143;
                        assert("Tensor range check" && 0 <= v120 && v120 < 4);
                        float * v145;
                        v145 = v112+v120;
                        float * v147;
                        v147 = v114+v120;
                        float v149;
                        v149 = atomicAdd(v145,v144);
                        float v150;
                        v150 = atomicAdd(v147,v143);
                        v130 += 1 ;
                    }
                    v120 += 1 ;
                }
                v31 += 1 ;
            }
            v29 += 1 ;
        }
        cooperative_groups::grid_group & v151 = v26.v1;
        cooperative_groups::grid_group & v152 = v151;
        curandStatePhilox4_32_10_t & v153 = v26.v5;
        curandStatePhilox4_32_10_t & v154 = v153;
        float * v155;
        v155 = reinterpret_cast<float *>(&v1[4718592ull]);
        int * v157;
        v157 = reinterpret_cast<int *>(&v0[131072ull]);
        float * v159;
        v159 = reinterpret_cast<float *>(&v0[131088ull]);
        float * v161;
        v161 = reinterpret_cast<float *>(&v0[131104ull]);
        double * v163;
        v163 = reinterpret_cast<double *>(&v1[11010048ull]);
        double * v165;
        v165 = reinterpret_cast<double *>(&v1[11403264ull]);
        v152.sync() ;
        v152.sync() ;
        v152.sync() ;
        v27 += 1 ;
    }
    cooperative_groups::grid_group & v167 = v26.v1;
    cooperative_groups::grid_group & v168 = v167;
    int v169;
    v169 = threadIdx.x;
    int v170;
    v170 = blockIdx.x;
    int v171;
    v171 = v170 * 256;
    int v172;
    v172 = v169 + v171;
    int v173;
    v173 = v172;
    while (while_method_9(v173)){
        bool v175;
        v175 = 0 <= v173;
        bool v176;
        v176 = v175 == false;
        if (v176){
            assert("The index needs to be zero or positive." && v175);
        } else {
        }
        int v178;
        v178 = v173 % 8;
        int v179;
        v179 = v173 / 8;
        bool v180;
        v180 = v179 < 4;
        bool v181;
        v181 = v180 == false;
        if (v181){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v180);
        } else {
        }
        assert("Tensor range check" && 0 <= v179 && v179 < 4);
        assert("Tensor range check" && 0 <= v178 && v178 < 8);
        int v183;
        v183 = 4 * v178;
        int v184;
        v184 = 32 * v179;
        int v185;
        v185 = v184 + v183;
        assert("Tensor range check" && 0 <= v179 && v179 < 4);
        assert("Tensor range check" && 0 <= v178 && v178 < 8);
        float v186[4];
        float v187[4];
        float v188[4];
        int4* v189;
        v189 = reinterpret_cast<int4*>(v3 + v185);
        int4* v190;
        v190 = reinterpret_cast<int4*>(v186 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v189) % 16 == 0 && reinterpret_cast<unsigned long long>(v190) % 16 == 0);
        *v190 = *v189;
        int4* v191;
        v191 = reinterpret_cast<int4*>(v4 + v185);
        int4* v192;
        v192 = reinterpret_cast<int4*>(v187 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v191) % 16 == 0 && reinterpret_cast<unsigned long long>(v192) % 16 == 0);
        *v192 = *v191;
        // Pushing the loop unrolling to: 0
        int v193;
        v193 = 0;
        #pragma unroll
        while (while_method_7(v193)){
            assert("Tensor range check" && 0 <= v193 && v193 < 4);
            float v195;
            v195 = v186[v193];
            float v196;
            v196 = v187[v193];
            bool v197;
            v197 = v196 == 0.0f;
            bool v198;
            v198 = v197 != true;
            float v200;
            if (v198){
                float v199;
                v199 = v195 / v196;
                v200 = v199;
            } else {
                v200 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v193 && v193 < 4);
            v188[v193] = v200;
            v193 += 1 ;
        }
        // Poping the loop unrolling to: 0
        int4* v201;
        v201 = reinterpret_cast<int4*>(v188 + 0);
        int4* v202;
        v202 = reinterpret_cast<int4*>(v5 + v185);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v201) % 16 == 0 && reinterpret_cast<unsigned long long>(v202) % 16 == 0);
        *v202 = *v201;
        v173 += 6144 ;
    }
    v168.sync() ;
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
    while (while_method_9(v27)){
        int v29;
        v29 = 0;
        while (while_method_5(v29)){
            Union4 v31;
            v31 = Union4{Union4_1{}};
            method_47(v0, v1, v2, v26, v31);
            static_array<float,2> & v32 = v26.v4;
            float * v33;
            v33 = reinterpret_cast<float *>(&v1[4718592ull]);
            int * v35;
            v35 = reinterpret_cast<int *>(&v0[131072ull]);
            float * v37;
            v37 = reinterpret_cast<float *>(&v0[131088ull]);
            float * v39;
            v39 = reinterpret_cast<float *>(&v0[131104ull]);
            double * v41;
            v41 = reinterpret_cast<double *>(&v1[11010048ull]);
            double * v43;
            v43 = reinterpret_cast<double *>(&v1[11403264ull]);
            int v45;
            v45 = 0;
            while (while_method_7(v45)){
                assert("Tensor range check" && 0 <= v45 && v45 < 4);
                int v47;
                v47 = 12288 * v45;
                int v48;
                v48 = threadIdx.x;
                assert("Tensor range check" && 0 <= v48 && v48 < 6144);
                int v49;
                v49 = 2 * v48;
                int v50;
                v50 = v49 + v47;
                double * v51;
                v51 = v41+v50;
                double * v53;
                v53 = v43+v50;
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
                    double v64;
                    v64 = v51[v55];
                    double v65;
                    v65 = v53[v55];
                    double v66;
                    v66 = v64 - v65;
                    double v67;
                    v67 = exp(v66);
                    float v68;
                    v68 = (float)v67;
                    float v69;
                    v69 = v62 * v68;
                    assert("Tensor range check" && 0 <= v45 && v45 < 4);
                    float * v70;
                    v70 = v37+v45;
                    float * v72;
                    v72 = v39+v45;
                    float v74;
                    v74 = atomicAdd(v70,v69);
                    float v75;
                    v75 = atomicAdd(v72,v68);
                    v55 += 1 ;
                }
                v45 += 1 ;
            }
            Union4 v76;
            v76 = Union4{Union4_1{}};
            method_48(v0, v1, v2, v26, v76);
            double * v77;
            v77 = reinterpret_cast<double *>(&v1[11010048ull]);
            double * v79;
            v79 = reinterpret_cast<double *>(&v1[11403264ull]);
            int v81;
            v81 = threadIdx.x;
            int v82;
            v82 = blockIdx.x;
            int v83;
            v83 = v82 * 256;
            int v84;
            v84 = v81 + v83;
            assert("Tensor range check" && 0 <= v84 && v84 < 6144);
            int v85;
            v85 = 2 * v84;
            int v86; double v87;
            Tuple12 tmp95 = Tuple12{0, 1.0};
            v86 = tmp95.v0; v87 = tmp95.v1;
            while (while_method_0(v86)){
                assert("Tensor range check" && 0 <= v86 && v86 < 2);
                int v89;
                v89 = v86 + v85;
                int v90; double v91;
                Tuple12 tmp96 = Tuple12{0, 0.0};
                v90 = tmp96.v0; v91 = tmp96.v1;
                while (while_method_7(v90)){
                    assert("Tensor range check" && 0 <= v90 && v90 < 4);
                    int v93;
                    v93 = 12288 * v90;
                    int v94;
                    v94 = v93 + v89;
                    double v95;
                    v95 = v77[v94];
                    double v96;
                    v96 = v79[v94];
                    double v97;
                    v97 = v95 - v96;
                    double v98;
                    v98 = exp(v97);
                    double v99;
                    v99 = v91 + v98;
                    v91 = v99;
                    v90 += 1 ;
                }
                double v100;
                v100 = v87 * v91;
                v87 = v100;
                v86 += 1 ;
            }
            float v101;
            v101 = (float)v87;
            int v102;
            v102 = 0;
            while (while_method_0(v102)){
                static_array<float,2> & v104 = v26.v4;
                bool v105;
                v105 = 0 <= v102;
                bool v107;
                if (v105){
                    bool v106;
                    v106 = v102 < 2;
                    v107 = v106;
                } else {
                    v107 = false;
                }
                bool v108;
                v108 = v107 == false;
                if (v108){
                    assert("Index must be in range." && v107);
                } else {
                }
                float v110;
                v110 = v104[v102];
                float v112;
                v112 = v110 * v101;
                assert("Tensor range check" && 0 <= v102 && v102 < 2);
                assert("Tensor range check" && 0 <= v27 && v27 < 32);
                int v113;
                v113 = 32 * v102;
                int v114;
                v114 = v113 + v27;
                float * v115;
                v115 = v3+v114;
                float * v117;
                v117 = v4+v114;
                float v119;
                v119 = atomicAdd(v115,v112);
                float v120;
                v120 = atomicAdd(v117,v101);
                v102 += 1 ;
            }
            double * v121;
            v121 = reinterpret_cast<double *>(&v1[11010048ull]);
            double * v123;
            v123 = reinterpret_cast<double *>(&v1[11403264ull]);
            int v125;
            v125 = 0;
            while (while_method_7(v125)){
                int v127;
                v127 = threadIdx.x;
                int v128;
                v128 = blockIdx.x;
                int v129;
                v129 = v128 * 256;
                int v130;
                v130 = v127 + v129;
                assert("Tensor range check" && 0 <= v125 && v125 < 4);
                int v131;
                v131 = 12288 * v125;
                assert("Tensor range check" && 0 <= v130 && v130 < 6144);
                int v132;
                v132 = 2 * v130;
                int v133;
                v133 = v132 + v131;
                double * v134;
                v134 = v121+v133;
                double * v136;
                v136 = v123+v133;
                double * v138;
                v138 = v134+0;
                double * v140;
                v140 = v136+0;
                double * v142;
                v142 = v134+0;
                double * v144;
                v144 = v136+0;
                int v146;
                v146 = sizeof(double *);
                unsigned long long v147;
                v147 = (unsigned long long)v146;
                unsigned long long v148;
                v148 = 256ull * v147;
                unsigned long long v149;
                v149 = v148 + 16ull;
                unsigned long long v150;
                v150 = v149 - 1ull;
                unsigned long long v151;
                v151 = v150 % 16ull;
                unsigned long long v152;
                v152 = v150 - v151;
                unsigned long long v153;
                v153 = v152 + v148;
                unsigned long long v154;
                v154 = v153 + 16ull;
                unsigned long long v155;
                v155 = v154 - 1ull;
                unsigned long long v156;
                v156 = v155 % 16ull;
                unsigned long long v157;
                v157 = v155 - v156;
                unsigned long long v158;
                v158 = v157 + v148;
                unsigned long long v159;
                v159 = v158 + 16ull;
                unsigned long long v160;
                v160 = v159 - 1ull;
                unsigned long long v161;
                v161 = v160 % 16ull;
                unsigned long long v162;
                v162 = v160 - v161;
                unsigned long long v163;
                v163 = v162 + v148;
                bool v164;
                v164 = v163 <= 98304ull;
                bool v165;
                v165 = v164 == false;
                if (v165){
                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v164);
                } else {
                }
                extern __shared__ unsigned char v167[];
                bool v168;
                v168 = v163 <= v163;
                bool v169;
                v169 = v168 == false;
                if (v169){
                    assert("The length of the partition has to be less than or equal to the length of the base array." && v168);
                } else {
                }
                double * * v171;
                v171 = reinterpret_cast<double * *>(&v167[0ull]);
                double * * v173;
                v173 = reinterpret_cast<double * *>(&v167[v152]);
                double * * v175;
                v175 = reinterpret_cast<double * *>(&v167[v157]);
                double * * v177;
                v177 = reinterpret_cast<double * *>(&v167[v162]);
                int v179;
                v179 = threadIdx.x;
                assert("Tensor range check" && 0 <= v179 && v179 < 256);
                v171[v179] = v138;
                v173[v179] = v140;
                v175[v179] = v142;
                v177[v179] = v144;
                __syncthreads();
                bool v180;
                v180 = 0 <= v179;
                bool v181;
                v181 = v180 == false;
                if (v181){
                    assert("The index needs to be zero or positive." && v180);
                } else {
                }
                int v183;
                v183 = v179 % 1;
                bool v184;
                v184 = v179 < 256;
                bool v185;
                v185 = v184 == false;
                if (v185){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v184);
                } else {
                }
                assert("Tensor range check" && 0 <= v179 && v179 < 256);
                int v187;
                v187 = 0;
                while (while_method_4(v187)){
                    bool v189;
                    v189 = v180 && v184;
                    bool v190;
                    v190 = v189 == false;
                    if (v190){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v189);
                    } else {
                    }
                    bool v192;
                    v192 = 0 <= v187;
                    bool v194;
                    if (v192){
                        bool v193;
                        v193 = v187 < 1;
                        v194 = v193;
                    } else {
                        v194 = false;
                    }
                    bool v195;
                    v195 = v194 == false;
                    if (v195){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v194);
                    } else {
                    }
                    int v197;
                    v197 = v187 * 256;
                    int v198;
                    v198 = v197 + v179;
                    assert("Tensor range check" && 0 <= v187 && v187 < 1);
                    int v199;
                    v199 = 256 * v187;
                    int v200;
                    v200 = v199 + v179;
                    double * v201;
                    v201 = v171[v200];
                    double * v202;
                    v202 = v173[v200];
                    double * v203;
                    v203 = v175[v200];
                    double * v204;
                    v204 = v177[v200];
                    int v205;
                    v205 = blockIdx.x;
                    int v206;
                    v206 = v205 * 256;
                    int v207;
                    v207 = v206 + v198;
                    assert("Tensor range check" && 0 <= v183 && v183 < 1);
                    int v208;
                    v208 = 2 * v183;
                    double v209[2];
                    double v210[2];
                    int v211[2];
                    int v212;
                    v212 = 0;
                    while (while_method_4(v212)){
                        assert("Tensor range check" && 0 <= v212 && v212 < 1);
                        int v214;
                        v214 = 2 * v212;
                        assert("Tensor range check" && 0 <= v212 && v212 < 1);
                        int v215;
                        v215 = v214 + v208;
                        int4* v216;
                        v216 = reinterpret_cast<int4*>(v201 + v215);
                        int4* v217;
                        v217 = reinterpret_cast<int4*>(v209 + v214);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v216) % 16 == 0 && reinterpret_cast<unsigned long long>(v217) % 16 == 0);
                        *v217 = *v216;
                        int4* v218;
                        v218 = reinterpret_cast<int4*>(v202 + v215);
                        int4* v219;
                        v219 = reinterpret_cast<int4*>(v210 + v214);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v218) % 16 == 0 && reinterpret_cast<unsigned long long>(v219) % 16 == 0);
                        *v219 = *v218;
                        v212 += 1 ;
                    }
                    int v220;
                    v220 = 0;
                    while (while_method_4(v220)){
                        int v222;
                        v222 = 0;
                        while (while_method_0(v222)){
                            bool v224;
                            v224 = 0 <= v222;
                            bool v226;
                            if (v224){
                                bool v225;
                                v225 = v222 < 2;
                                v226 = v225;
                            } else {
                                v226 = false;
                            }
                            bool v227;
                            v227 = v226 == false;
                            if (v227){
                                assert("The indices should be inside the range of the dimension." && v226);
                            } else {
                            }
                            bool v229;
                            v229 = 0 <= v183;
                            bool v231;
                            if (v229){
                                bool v230;
                                v230 = v183 < 1;
                                v231 = v230;
                            } else {
                                v231 = false;
                            }
                            bool v232;
                            v232 = v231 == false;
                            if (v232){
                                assert("The indices should be inside the range of the dimension." && v231);
                            } else {
                            }
                            int v234;
                            v234 = v183 * 2;
                            int v235;
                            v235 = v222 + v234;
                            bool v236;
                            v236 = 0 <= v220;
                            bool v238;
                            if (v236){
                                bool v237;
                                v237 = v220 < 1;
                                v238 = v237;
                            } else {
                                v238 = false;
                            }
                            bool v239;
                            v239 = v238 == false;
                            if (v239){
                                assert("The indices should be inside the range of the dimension." && v238);
                            } else {
                            }
                            int v241;
                            v241 = v220 * 2;
                            int v242;
                            v242 = v235 + v241;
                            assert("Tensor range check" && 0 <= v220 && v220 < 1);
                            assert("Tensor range check" && 0 <= v222 && v222 < 2);
                            int v243;
                            v243 = 2 * v220;
                            int v244;
                            v244 = v243 + v222;
                            v211[v244] = v242;
                            v222 += 1 ;
                        }
                        v220 += 1 ;
                    }
                    double v245[2];
                    double v246[2];
                    int v247;
                    v247 = 0;
                    while (while_method_4(v247)){
                        int v249;
                        v249 = 0;
                        while (while_method_0(v249)){
                            assert("Tensor range check" && 0 <= v247 && v247 < 1);
                            assert("Tensor range check" && 0 <= v249 && v249 < 2);
                            int v251;
                            v251 = 2 * v247;
                            int v252;
                            v252 = v251 + v249;
                            double v253;
                            v253 = v209[v252];
                            double v254;
                            v254 = v210[v252];
                            assert("Tensor range check" && 0 <= v247 && v247 < 1);
                            assert("Tensor range check" && 0 <= v249 && v249 < 2);
                            v245[v252] = 0.0;
                            v246[v252] = 0.0;
                            v249 += 1 ;
                        }
                        v247 += 1 ;
                    }
                    int v255;
                    v255 = 0;
                    while (while_method_4(v255)){
                        assert("Tensor range check" && 0 <= v255 && v255 < 1);
                        int v257;
                        v257 = 2 * v255;
                        int v258;
                        v258 = v257 + v208;
                        assert("Tensor range check" && 0 <= v255 && v255 < 1);
                        int4* v259;
                        v259 = reinterpret_cast<int4*>(v245 + v257);
                        int4* v260;
                        v260 = reinterpret_cast<int4*>(v203 + v258);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v259) % 16 == 0 && reinterpret_cast<unsigned long long>(v260) % 16 == 0);
                        *v260 = *v259;
                        int4* v261;
                        v261 = reinterpret_cast<int4*>(v246 + v257);
                        int4* v262;
                        v262 = reinterpret_cast<int4*>(v204 + v258);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v261) % 16 == 0 && reinterpret_cast<unsigned long long>(v262) % 16 == 0);
                        *v262 = *v261;
                        v255 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v198 && v198 < 256);
                    v187 += 1 ;
                }
                __syncthreads();
                assert("Tensor range check" && 0 <= v179 && v179 < 256);
                __syncthreads();
                v125 += 1 ;
            }
            v29 += 1 ;
        }
        cooperative_groups::grid_group & v263 = v26.v1;
        cooperative_groups::grid_group & v264 = v263;
        curandStatePhilox4_32_10_t & v265 = v26.v5;
        curandStatePhilox4_32_10_t & v266 = v265;
        float * v267;
        v267 = reinterpret_cast<float *>(&v1[4718592ull]);
        int * v269;
        v269 = reinterpret_cast<int *>(&v0[131072ull]);
        float * v271;
        v271 = reinterpret_cast<float *>(&v0[131088ull]);
        float * v273;
        v273 = reinterpret_cast<float *>(&v0[131104ull]);
        double * v275;
        v275 = reinterpret_cast<double *>(&v1[11010048ull]);
        double * v277;
        v277 = reinterpret_cast<double *>(&v1[11403264ull]);
        v264.sync() ;
        v264.sync() ;
        v264.sync() ;
        v27 += 1 ;
    }
    cooperative_groups::grid_group & v279 = v26.v1;
    cooperative_groups::grid_group & v280 = v279;
    int v281;
    v281 = threadIdx.x;
    int v282;
    v282 = blockIdx.x;
    int v283;
    v283 = v282 * 256;
    int v284;
    v284 = v281 + v283;
    int v285;
    v285 = v284;
    while (while_method_8(v285)){
        bool v287;
        v287 = 0 <= v285;
        bool v288;
        v288 = v287 == false;
        if (v288){
            assert("The index needs to be zero or positive." && v287);
        } else {
        }
        int v290;
        v290 = v285 % 8;
        int v291;
        v291 = v285 / 8;
        bool v292;
        v292 = v291 < 2;
        bool v293;
        v293 = v292 == false;
        if (v293){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v292);
        } else {
        }
        assert("Tensor range check" && 0 <= v291 && v291 < 2);
        assert("Tensor range check" && 0 <= v290 && v290 < 8);
        int v295;
        v295 = 4 * v290;
        int v296;
        v296 = 32 * v291;
        int v297;
        v297 = v296 + v295;
        assert("Tensor range check" && 0 <= v291 && v291 < 2);
        assert("Tensor range check" && 0 <= v290 && v290 < 8);
        float v298[4];
        float v299[4];
        float v300[4];
        int4* v301;
        v301 = reinterpret_cast<int4*>(v3 + v297);
        int4* v302;
        v302 = reinterpret_cast<int4*>(v298 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v301) % 16 == 0 && reinterpret_cast<unsigned long long>(v302) % 16 == 0);
        *v302 = *v301;
        int4* v303;
        v303 = reinterpret_cast<int4*>(v4 + v297);
        int4* v304;
        v304 = reinterpret_cast<int4*>(v299 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v303) % 16 == 0 && reinterpret_cast<unsigned long long>(v304) % 16 == 0);
        *v304 = *v303;
        // Pushing the loop unrolling to: 0
        int v305;
        v305 = 0;
        #pragma unroll
        while (while_method_7(v305)){
            assert("Tensor range check" && 0 <= v305 && v305 < 4);
            float v307;
            v307 = v298[v305];
            float v308;
            v308 = v299[v305];
            bool v309;
            v309 = v308 == 0.0f;
            bool v310;
            v310 = v309 != true;
            float v312;
            if (v310){
                float v311;
                v311 = v307 / v308;
                v312 = v311;
            } else {
                v312 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v305 && v305 < 4);
            v300[v305] = v312;
            v305 += 1 ;
        }
        // Poping the loop unrolling to: 0
        int4* v313;
        v313 = reinterpret_cast<int4*>(v300 + 0);
        int4* v314;
        v314 = reinterpret_cast<int4*>(v5 + v297);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v313) % 16 == 0 && reinterpret_cast<unsigned long long>(v314) % 16 == 0);
        *v314 = *v313;
        v285 += 6144 ;
    }
    v280.sync() ;
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
                v19 = cp.zeros(128,dtype=cp.float32) # type: ignore
                v20 = cp.zeros(128,dtype=cp.float32) # type: ignore
                v21 = cp.empty(128,dtype=cp.float32)
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
                    while method58(v35):
                        assert 0 <= v32 < 4, 'Tensor range check'
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
                    while method58(v57):
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
        v92, v93, v94, v95, v96 = method59(v12)
        del v12
        return method76(v92, v93, v94, v95, v96, v8, v9, v10, v18)
    return inner
def Closure1():
    def inner() -> object:
        v0 = cp.empty(0,dtype=cp.uint8)
        v1 = cp.empty(11796480,dtype=cp.uint8)
        v2 = cp.empty(131120,dtype=cp.uint8)
        v4 = v1[0:0+4*786432].view(cp.float32)
        del v4
        v6 = v2[0:0+4*32768].view(cp.float32)
        v8 = v1[4718592:4718592+4*1572864].view(cp.float32)
        del v8
        v10 = v2[131072:131072+4*1].view(cp.int32)
        v12 = v2[131088:131088+4*4].view(cp.float32)
        v14 = v2[131104:131104+4*4].view(cp.float32)
        v16 = v1[11010048:11010048+8*49152].view(cp.float64)
        v18 = v1[11403264:11403264+8*49152].view(cp.float64)
        v19 = cp.random.normal(0.0,0.0055242716,32768,dtype=cp.float32) # type: ignore
        cp.copyto(v6[0:0+32768],v19[0:0+32768])
        del v6, v19
        v10[:] = 0
        del v10
        v12[:] = 0
        del v12
        v14[:] = 0
        del v14
        v16[:] = 0
        del v16
        v18[:] = 0
        del v18
        v21 = static_array(2)
        v23 = US2_0()
        v21[0] = v23
        del v23
        v25 = US2_1()
        v21[1] = v25
        del v25
        v27 = static_array_list(32)
        v28 = 63
        v29 = US3_0()
        v30 = US7_0()
        return method115(v28, v29, v27, v21, v30, v2, v1, v0)
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
    v1 = v0 < 4
    del v0
    return v1
def method58(v0 : i32) -> bool:
    v1 = v0 < 32
    del v0
    return v1
def method60(v0 : cp.ndarray) -> u32:
    v2 = v0[0:].view(cp.uint32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method61(v0 : cp.ndarray) -> i32:
    v2 = v0[4:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method62(v0 : cp.ndarray) -> None:
    del v0
    return 
def method64(v0 : cp.ndarray) -> i32:
    v2 = v0[0:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method66(v0 : cp.ndarray) -> US6:
    v1 = method64(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method62(v3)
        del v3
        return US6_0()
    elif v1 == 1:
        del v1
        method62(v3)
        del v3
        return US6_1()
    elif v1 == 2:
        del v1
        method62(v3)
        del v3
        return US6_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method65(v0 : cp.ndarray) -> Tuple[US5, bool, static_array, i32, static_array, i32]:
    v1 = method64(v0)
    v3 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method62(v3)
        v8 = US5_0()
    elif v1 == 1:
        v6 = method66(v3)
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
        v21 = method66(v20)
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
        v34 = method64(v33)
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
def method68(v0 : cp.ndarray) -> i32:
    v2 = v0[36:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method67(v0 : cp.ndarray) -> Tuple[US5, bool, static_array, i32, static_array, i32, US1]:
    v1 = method64(v0)
    v3 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method62(v3)
        v8 = US5_0()
    elif v1 == 1:
        v6 = method66(v3)
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
        v21 = method66(v20)
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
        v34 = method64(v33)
        del v33
        v26[v27] = v34
        del v34
        v27 += 1 
    del v27
    v36 = v0[32:].view(cp.int32)
    v37 = v36[0].item()
    del v36
    v38 = method68(v0)
    v40 = v0[40:].view(cp.uint8)
    del v0
    if v38 == 0:
        method62(v40)
        v45 = US1_0()
    elif v38 == 1:
        method62(v40)
        v45 = US1_1()
    elif v38 == 2:
        method62(v40)
        v45 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v38, v40
    return v8, v11, v13, v24, v26, v37, v45
def method63(v0 : cp.ndarray) -> US4:
    v1 = method64(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5, v6, v7, v8, v9, v10 = method65(v3)
        del v3
        return US4_0(v5, v6, v7, v8, v9, v10)
    elif v1 == 1:
        del v1
        method62(v3)
        del v3
        return US4_1()
    elif v1 == 2:
        del v1
        v13, v14, v15, v16, v17, v18 = method65(v3)
        del v3
        return US4_2(v13, v14, v15, v16, v17, v18)
    elif v1 == 3:
        del v1
        v20, v21, v22, v23, v24, v25, v26 = method67(v3)
        del v3
        return US4_3(v20, v21, v22, v23, v24, v25, v26)
    elif v1 == 4:
        del v1
        v28, v29, v30, v31, v32, v33 = method65(v3)
        del v3
        return US4_4(v28, v29, v30, v31, v32, v33)
    elif v1 == 5:
        del v1
        v35, v36, v37, v38, v39, v40 = method65(v3)
        del v3
        return US4_5(v35, v36, v37, v38, v39, v40)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method69(v0 : cp.ndarray) -> i32:
    v2 = v0[80:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method71(v0 : cp.ndarray) -> Tuple[i32, US1]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method61(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method62(v6)
        v11 = US1_0()
    elif v4 == 1:
        method62(v6)
        v11 = US1_1()
    elif v4 == 2:
        method62(v6)
        v11 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v11
def method72(v0 : cp.ndarray) -> Tuple[i32, US6]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method61(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method62(v6)
        v11 = US6_0()
    elif v4 == 1:
        method62(v6)
        v11 = US6_1()
    elif v4 == 2:
        method62(v6)
        v11 = US6_2()
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v11
def method73(v0 : cp.ndarray) -> Tuple[static_array, i32, i32]:
    v2 = static_array(2)
    v3 = 0
    while method44(v3):
        v5 = u64(v3)
        v6 = v5 * 4
        del v5
        v8 = v0[v6:].view(cp.uint8)
        del v6
        v9 = method66(v8)
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
def method70(v0 : cp.ndarray) -> US8:
    v1 = method64(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5 = method66(v3)
        del v3
        return US8_0(v5)
    elif v1 == 1:
        del v1
        v7, v8 = method71(v3)
        del v3
        return US8_1(v7, v8)
    elif v1 == 2:
        del v1
        v10, v11 = method72(v3)
        del v3
        return US8_2(v10, v11)
    elif v1 == 3:
        del v1
        v13, v14, v15 = method73(v3)
        del v3
        return US8_3(v13, v14, v15)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method74(v0 : cp.ndarray) -> US2:
    v1 = method64(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method62(v3)
        del v3
        return US2_0()
    elif v1 == 1:
        del v1
        method62(v3)
        del v3
        return US2_1()
    elif v1 == 2:
        del v1
        method62(v3)
        del v3
        return US2_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method75(v0 : cp.ndarray) -> i32:
    v2 = v0[1128:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method59(v0 : cp.ndarray) -> Tuple[u32, US3, static_array_list, static_array, US7]:
    v1 = method60(v0)
    v2 = method61(v0)
    v4 = v0[16:].view(cp.uint8)
    if v2 == 0:
        method62(v4)
        v9 = US3_0()
    elif v2 == 1:
        v7 = method63(v4)
        v9 = US3_1(v7)
    else:
        raise Exception("Invalid tag.")
    del v2, v4
    v11 = static_array_list(32)
    v12 = method69(v0)
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
        v21 = method70(v20)
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
        v31 = method74(v30)
        del v30
        v23[v24] = v31
        del v31
        v24 += 1 
    del v24
    v32 = method75(v0)
    v34 = v0[1136:].view(cp.uint8)
    del v0
    if v32 == 0:
        method62(v34)
        v51 = US7_0()
    elif v32 == 1:
        v37, v38, v39, v40, v41, v42 = method65(v34)
        v51 = US7_1(v37, v38, v39, v40, v41, v42)
    elif v32 == 2:
        v44, v45, v46, v47, v48, v49 = method65(v34)
        v51 = US7_2(v44, v45, v46, v47, v48, v49)
    else:
        raise Exception("Invalid tag.")
    del v32, v34
    return v1, v9, v11, v23, v51
def method82(v0 : u32) -> object:
    v1 = v0
    del v0
    return v1
def method81(v0 : u32) -> object:
    return method82(v0)
def method84() -> object:
    v0 = []
    return v0
def method88(v0 : US6) -> object:
    match v0:
        case US6_0(): # Jack
            del v0
            v1 = method84()
            v2 = "Jack"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US6_1(): # King
            del v0
            v4 = method84()
            v5 = "King"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US6_2(): # Queen
            del v0
            v7 = method84()
            v8 = "Queen"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method87(v0 : US5) -> object:
    match v0:
        case US5_0(): # None
            del v0
            v1 = method84()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US5_1(v4): # Some
            del v0
            v5 = method88(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method89(v0 : bool) -> object:
    v1 = v0
    del v0
    return v1
def method90(v0 : static_array) -> object:
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
        v11 = method88(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method91(v0 : i32) -> object:
    v1 = v0
    del v0
    return v1
def method92(v0 : static_array) -> object:
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
        v11 = method91(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method86(v0 : US5, v1 : bool, v2 : static_array, v3 : i32, v4 : static_array, v5 : i32) -> object:
    v6 = method87(v0)
    del v0
    v7 = method89(v1)
    del v1
    v8 = method90(v2)
    del v2
    v9 = method91(v3)
    del v3
    v10 = method92(v4)
    del v4
    v11 = method91(v5)
    del v5
    v12 = {'community_card': v6, 'is_button_s_first_move': v7, 'pl_card': v8, 'player_turn': v9, 'pot': v10, 'raises_left': v11}
    del v6, v7, v8, v9, v10, v11
    return v12
def method94(v0 : US1) -> object:
    match v0:
        case US1_0(): # Call
            del v0
            v1 = method84()
            v2 = "Call"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US1_1(): # Fold
            del v0
            v4 = method84()
            v5 = "Fold"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US1_2(): # Raise
            del v0
            v7 = method84()
            v8 = "Raise"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method93(v0 : US5, v1 : bool, v2 : static_array, v3 : i32, v4 : static_array, v5 : i32, v6 : US1) -> object:
    v7 = []
    v8 = method86(v0, v1, v2, v3, v4, v5)
    del v0, v1, v2, v3, v4, v5
    v7.append(v8)
    del v8
    v9 = method94(v6)
    del v6
    v7.append(v9)
    del v9
    v10 = v7
    del v7
    return v10
def method85(v0 : US4) -> object:
    match v0:
        case US4_0(v1, v2, v3, v4, v5, v6): # ChanceCommunityCard
            del v0
            v7 = method86(v1, v2, v3, v4, v5, v6)
            del v1, v2, v3, v4, v5, v6
            v8 = "ChanceCommunityCard"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US4_1(): # ChanceInit
            del v0
            v10 = method84()
            v11 = "ChanceInit"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US4_2(v13, v14, v15, v16, v17, v18): # Round
            del v0
            v19 = method86(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "Round"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case US4_3(v22, v23, v24, v25, v26, v27, v28): # RoundWithAction
            del v0
            v29 = method93(v22, v23, v24, v25, v26, v27, v28)
            del v22, v23, v24, v25, v26, v27, v28
            v30 = "RoundWithAction"
            v31 = [v30,v29]
            del v29, v30
            return v31
        case US4_4(v32, v33, v34, v35, v36, v37): # TerminalCall
            del v0
            v38 = method86(v32, v33, v34, v35, v36, v37)
            del v32, v33, v34, v35, v36, v37
            v39 = "TerminalCall"
            v40 = [v39,v38]
            del v38, v39
            return v40
        case US4_5(v41, v42, v43, v44, v45, v46): # TerminalFold
            del v0
            v47 = method86(v41, v42, v43, v44, v45, v46)
            del v41, v42, v43, v44, v45, v46
            v48 = "TerminalFold"
            v49 = [v48,v47]
            del v47, v48
            return v49
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method83(v0 : US3) -> object:
    match v0:
        case US3_0(): # None
            del v0
            v1 = method84()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US3_1(v4): # Some
            del v0
            v5 = method85(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method80(v0 : u32, v1 : US3) -> object:
    v2 = method81(v0)
    del v0
    v3 = method83(v1)
    del v1
    v4 = {'deck': v2, 'game': v3}
    del v2, v3
    return v4
def method98(v0 : i32, v1 : US1) -> object:
    v2 = []
    v3 = method91(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method94(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method99(v0 : i32, v1 : US6) -> object:
    v2 = []
    v3 = method91(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method88(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method100(v0 : static_array, v1 : i32, v2 : i32) -> object:
    v3 = method90(v0)
    del v0
    v4 = method91(v1)
    del v1
    v5 = method91(v2)
    del v2
    v6 = {'cards_shown': v3, 'chips_won': v4, 'winner_id': v5}
    del v3, v4, v5
    return v6
def method97(v0 : US8) -> object:
    match v0:
        case US8_0(v1): # CommunityCardIs
            del v0
            v2 = method88(v1)
            del v1
            v3 = "CommunityCardIs"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US8_1(v5, v6): # PlayerAction
            del v0
            v7 = method98(v5, v6)
            del v5, v6
            v8 = "PlayerAction"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US8_2(v10, v11): # PlayerGotCard
            del v0
            v12 = method99(v10, v11)
            del v10, v11
            v13 = "PlayerGotCard"
            v14 = [v13,v12]
            del v12, v13
            return v14
        case US8_3(v15, v16, v17): # Showdown
            del v0
            v18 = method100(v15, v16, v17)
            del v15, v16, v17
            v19 = "Showdown"
            v20 = [v19,v18]
            del v18, v19
            return v20
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method96(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method5(v2, v3):
        v6 = v0[v3]
        v7 = method97(v6)
        del v6
        v1.append(v7)
        del v7
        v3 += 1 
    del v0, v2, v3
    return v1
def method102(v0 : US2) -> object:
    match v0:
        case US2_0(): # Computer
            del v0
            v1 = method84()
            v2 = "Computer"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US2_1(): # Human
            del v0
            v4 = method84()
            v5 = "Human"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US2_2(): # Random
            del v0
            v7 = method84()
            v8 = "Random"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method101(v0 : static_array) -> object:
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
        v11 = method102(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method103(v0 : US7) -> object:
    match v0:
        case US7_0(): # GameNotStarted
            del v0
            v1 = method84()
            v2 = "GameNotStarted"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US7_1(v4, v5, v6, v7, v8, v9): # GameOver
            del v0
            v10 = method86(v4, v5, v6, v7, v8, v9)
            del v4, v5, v6, v7, v8, v9
            v11 = "GameOver"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US7_2(v13, v14, v15, v16, v17, v18): # WaitingForActionFromPlayerId
            del v0
            v19 = method86(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "WaitingForActionFromPlayerId"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method95(v0 : static_array_list, v1 : static_array, v2 : US7) -> object:
    v3 = method96(v0)
    del v0
    v4 = method101(v1)
    del v1
    v5 = method103(v2)
    del v2
    v6 = {'messages': v3, 'pl_type': v4, 'ui_game_state': v5}
    del v3, v4, v5
    return v6
def method79(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7) -> object:
    v5 = method80(v0, v1)
    del v0, v1
    v6 = method95(v2, v3, v4)
    del v2, v3, v4
    v7 = {'private': v5, 'public': v6}
    del v5, v6
    return v7
def method109(v0 : cp.ndarray) -> object:
    v1 = v0
    del v0
    return v1
def method108(v0 : cp.ndarray) -> object:
    return method109(v0)
def method107(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    v3 = []
    v4 = method108(v0)
    del v0
    v3.append(v4)
    del v4
    v5 = method108(v1)
    del v1
    v3.append(v5)
    del v5
    v6 = method108(v2)
    del v2
    v3.append(v6)
    del v6
    v7 = method84()
    v3.append(v7)
    del v7
    v8 = v3
    del v3
    return v8
def method106(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    return method107(v0, v1, v2)
def method105(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    return method106(v0, v1, v2)
def method104(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    v3 = method105(v0, v1, v2)
    del v0, v1, v2
    v4 = {'model_ptrs': v3}
    del v3
    return v4
def method78(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray) -> object:
    v8 = method79(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    v9 = method104(v5, v6, v7)
    del v5, v6, v7
    v10 = {'game': v8, 'neural': v9}
    del v8, v9
    return v10
def method114(v0 : f32) -> object:
    v1 = v0
    del v0
    return v1
def method113(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method5(v2, v3):
        v5 = v0[v3]
        v6 = method114(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
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
def method111(v0 : US9) -> object:
    match v0:
        case US9_0(v1): # AddRewardsRando
            del v0
            v2 = method112(v1)
            del v1
            v3 = "AddRewardsRando"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US9_1(v5): # AddRewardsSelf
            del v0
            v6 = method112(v5)
            del v5
            v7 = "AddRewardsSelf"
            v8 = [v7,v6]
            del v6, v7
            return v8
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method110(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method5(v2, v3):
        v5 = v0[v3]
        v6 = method111(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method77(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : list) -> object:
    v9 = []
    v10 = method78(v0, v1, v2, v3, v4, v5, v6, v7)
    del v0, v1, v2, v3, v4, v5, v6, v7
    v9.append(v10)
    del v10
    v11 = method110(v8)
    del v8
    v9.append(v11)
    del v11
    v12 = v9
    del v9
    return v12
def method76(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : list) -> object:
    v9 = method77(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8
    return v9
def method115(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray) -> object:
    v8 = method78(v0, v1, v2, v3, v4, v5, v6, v7)
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
