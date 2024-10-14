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
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
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
__device__ void block_map_24(float * v0, float * v1);
__device__ void block_matmul_25(float * v0, float * v1, int v2, float * v3);
__device__ void block_row_map_26(float * v0, int v1, float * v2);
struct Tuple8;
struct Tuple9;
struct Tuple10;
struct Union12;
struct Union13;
__device__ int tag_28(Union6 v0);
__device__ bool is_pair_29(int v0, int v1);
__device__ Tuple7 order_30(int v0, int v1);
__device__ Union13 compare_hands_27(Union5 v0, bool v1, static_array<Union6,2> v2, int v3, static_array<int,2> v4, int v5);
__device__ void f_32(unsigned char * v0, unsigned int v1);
__device__ void f_33(unsigned char * v0, int v1);
__device__ void f_34(unsigned char * v0);
__device__ void f_36(unsigned char * v0, int v1);
__device__ void f_38(unsigned char * v0, Union6 v1);
__device__ void f_37(unsigned char * v0, Union5 v1, bool v2, static_array<Union6,2> v3, int v4, static_array<int,2> v5, int v6);
__device__ void f_40(unsigned char * v0, int v1);
__device__ void f_39(unsigned char * v0, Union5 v1, bool v2, static_array<Union6,2> v3, int v4, static_array<int,2> v5, int v6, Union1 v7);
__device__ void f_35(unsigned char * v0, Union4 v1);
__device__ void f_41(unsigned char * v0, int v1);
__device__ void f_43(unsigned char * v0, int v1, Union1 v2);
__device__ void f_44(unsigned char * v0, int v1, Union6 v2);
__device__ void f_45(unsigned char * v0, static_array<Union6,2> v1, int v2, int v3);
__device__ void f_42(unsigned char * v0, Union7 v1);
__device__ void f_46(unsigned char * v0, Union2 v1);
__device__ void f_47(unsigned char * v0, int v1);
__device__ void f_31(unsigned char * v0, unsigned int v1, Union3 v2, static_array_list<Union7,32> v3, static_array<Union2,2> v4, Union8 v5);
struct StackMut1;
struct Union14;
__device__ void method_48(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut1 & v3, int v4, Union4 v5);
struct Tuple11;
struct Tuple12;
struct Tuple13;
__device__ void method_49(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut1 & v3, Union4 v4);
__device__ void method_50(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut1 & v3, Union4 v4);
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
struct Closure2 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple8 {
    float v0;
    bool v1;
    __device__ Tuple8() = default;
    __device__ Tuple8(float t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ Tuple8 operator()(Tuple8 tup0, Tuple8 tup1){
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
                return Tuple8{v5, true};
            } else {
                return Tuple8{v0, v1};
            }
        } else {
            if (v3){
                return Tuple8{v2, v3};
            } else {
                return Tuple8{v0, v1};
            }
        }
    }
};
struct Tuple9 {
    float v0;
    int v1;
    __device__ Tuple9() = default;
    __device__ Tuple9(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
    __device__ Tuple9 operator()(Tuple9 tup0, Tuple9 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple9{v0, v1};
        } else {
            return Tuple9{v2, v3};
        }
    }
};
struct Tuple10 {
    int v0;
    bool v1;
    __device__ Tuple10() = default;
    __device__ Tuple10(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure5 {
    __device__ Tuple10 operator()(Tuple10 tup0, Tuple10 tup1){
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
struct Closure6 {
    int v0;
    __device__ Tuple9 operator()(Tuple9 tup0, Tuple9 tup1){
        int & v0 = this->v0;
        float v1 = tup0.v0; int v2 = tup0.v1; float v3 = tup1.v0; int v4 = tup1.v1;
        bool v5;
        v5 = v2 == v0;
        if (v5){
            return Tuple9{v1, v2};
        } else {
            bool v6;
            v6 = v4 == v0;
            if (v6){
                return Tuple9{v3, v4};
            } else {
                return Tuple9{v1, v2};
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
struct Tuple11 {
    double v1;
    int v0;
    __device__ Tuple11() = default;
    __device__ Tuple11(int t0, double t1) : v0(t0), v1(t1) {}
};
struct Tuple12 {
    double v2;
    float v1;
    int v0;
    __device__ Tuple12() = default;
    __device__ Tuple12(int t0, float t1, double t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple13 {
    int v0;
    float v1;
    __device__ Tuple13() = default;
    __device__ Tuple13(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Closure7 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        bool v2;
        v2 = v0 >= v1;
        if (v2){
            return v0;
        } else {
            return v1;
        }
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
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 32;
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
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 1;
    return v1;
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 8;
    return v1;
}
__device__ inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 4;
    return v1;
}
__device__ inline bool while_method_8(int v0){
    bool v1;
    v1 = v0 < 4;
    return v1;
}
__device__ inline bool while_method_9(int v0){
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
    while (while_method_5(v64)){
        int v66;
        v66 = 0;
        while (while_method_5(v66)){
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
            while (while_method_6(v74)){
                int v76;
                v76 = 0;
                #pragma unroll
                while (while_method_5(v76)){
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
            while (while_method_7(v80)){
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
                        while (while_method_5(v129)){
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
                while (while_method_6(v155)){
                    int v157;
                    v157 = 0;
                    #pragma unroll
                    while (while_method_5(v157)){
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
                while (while_method_5(v168)){
                    int v170;
                    v170 = 0;
                    #pragma unroll
                    while (while_method_8(v170)){
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
                            while (while_method_5(v224)){
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
                while (while_method_6(v232)){
                    int v234;
                    v234 = 0;
                    #pragma unroll
                    while (while_method_8(v234)){
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
                        while (while_method_5(v261)){
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
            while (while_method_6(v268)){
                int v270;
                v270 = 0;
                #pragma unroll
                while (while_method_5(v270)){
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
            while (while_method_9(v297)){
                int v299;
                v299 = 0;
                #pragma unroll
                while (while_method_5(v299)){
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
__device__ inline bool while_method_10(int v0){
    bool v1;
    v1 = v0 < 4096;
    return v1;
}
__device__ void block_map_24(float * v0, float * v1){
    int v2;
    v2 = blockIdx.x;
    assert("Tensor range check" && 0 <= v2 && v2 < 24);
    int v3;
    v3 = 16384 * v2;
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 24);
    int v5;
    v5 = 16384 * v4;
    int v6;
    v6 = threadIdx.x;
    int v7;
    v7 = v6;
    while (while_method_10(v7)){
        bool v9;
        v9 = 0 <= v7;
        bool v10;
        v10 = v9 == false;
        if (v10){
            assert("The index needs to be zero or positive." && v9);
        } else {
        }
        int v12;
        v12 = v7 % 16;
        int v13;
        v13 = v7 / 16;
        bool v14;
        v14 = v13 < 256;
        bool v15;
        v15 = v14 == false;
        if (v15){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v14);
        } else {
        }
        assert("Tensor range check" && 0 <= v13 && v13 < 256);
        assert("Tensor range check" && 0 <= v12 && v12 < 16);
        int v17;
        v17 = 4 * v12;
        int v18;
        v18 = v17 + v3;
        int v19;
        v19 = 64 * v13;
        int v20;
        v20 = v19 + v18;
        assert("Tensor range check" && 0 <= v13 && v13 < 256);
        assert("Tensor range check" && 0 <= v12 && v12 < 16);
        int v21;
        v21 = v17 + v5;
        int v22;
        v22 = v19 + v21;
        float v23[4];
        float v24[4];
        int4* v25;
        v25 = reinterpret_cast<int4*>(v1 + v20);
        int4* v26;
        v26 = reinterpret_cast<int4*>(v23 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v25) % 16 == 0 && reinterpret_cast<unsigned long long>(v26) % 16 == 0);
        *v26 = *v25;
        // Pushing the loop unrolling to: 0
        int v27;
        v27 = 0;
        #pragma unroll
        while (while_method_8(v27)){
            assert("Tensor range check" && 0 <= v27 && v27 < 4);
            float v29;
            v29 = v23[v27];
            bool v30;
            v30 = 0.0f >= v29;
            float v31;
            if (v30){
                v31 = 0.0f;
            } else {
                v31 = v29;
            }
            assert("Tensor range check" && 0 <= v27 && v27 < 4);
            v24[v27] = v31;
            v27 += 1 ;
        }
        // Poping the loop unrolling to: 0
        int4* v32;
        v32 = reinterpret_cast<int4*>(v24 + 0);
        int4* v33;
        v33 = reinterpret_cast<int4*>(v0 + v22);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v32) % 16 == 0 && reinterpret_cast<unsigned long long>(v33) % 16 == 0);
        *v33 = *v32;
        v7 += 256 ;
    }
    __syncthreads();
    return ;
}
__device__ inline bool while_method_11(int v0){
    bool v1;
    v1 = v0 < 2;
    return v1;
}
__device__ void block_matmul_25(float * v0, float * v1, int v2, float * v3){
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 24);
    int v5;
    v5 = 16384 * v4;
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
    while (while_method_5(v64)){
        int v66;
        v66 = 0;
        while (while_method_5(v66)){
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
            while (while_method_6(v74)){
                int v76;
                v76 = 0;
                #pragma unroll
                while (while_method_5(v76)){
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
            while (while_method_11(v80)){
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
                v88 = v80 < 2;
                bool v89;
                v89 = v88 == false;
                if (v89){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v88);
                } else {
                }
                bool v91;
                v91 = v82 < 2;
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
                v98 = v70 + v5;
                assert("Tensor range check" && 0 <= v80 && v80 < 2);
                int v99;
                v99 = 32 * v80;
                int v100;
                v100 = v99 + v98;
                float * v101;
                v101 = v3+v100;
                assert("Tensor range check" && 0 <= v66 && v66 < 1);
                int v103;
                v103 = 4096 * v66;
                int v104;
                v104 = v103 + v2;
                if (v83){
                    assert("Tensor range check" && 0 <= v80 && v80 < 2);
                    int v105;
                    v105 = v99 + v104;
                    float * v106;
                    v106 = v1+v105;
                    // Pushing the loop unrolling to: 0
                    v8.producer_acquire();
                    int v108;
                    v108 = threadIdx.x;
                    bool v109;
                    v109 = 0 <= v108;
                    bool v110;
                    v110 = v109 == false;
                    if (v110){
                        assert("The index needs to be zero or positive." && v109);
                    } else {
                    }
                    int v112;
                    v112 = v108 % 8;
                    int v113;
                    v113 = v108 / 8;
                    bool v114;
                    v114 = v113 < 32;
                    bool v115;
                    v115 = v114 == false;
                    if (v115){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v114);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v113 && v113 < 32);
                    assert("Tensor range check" && 0 <= v112 && v112 < 8);
                    int v117;
                    v117 = 4 * v112;
                    int v118;
                    v118 = 36 * v113;
                    int v119;
                    v119 = v118 + v117;
                    int v120;
                    v120 = 64 * v113;
                    int v121;
                    v121 = v120 + v117;
                    float * v122;
                    v122 = v12+v119;
                    float * v124;
                    v124 = v106+v121;
                    int v126;
                    v126 = 0;
                    #pragma unroll
                    while (while_method_0(v126)){
                        int v128;
                        v128 = 0;
                        #pragma unroll
                        while (while_method_5(v128)){
                            assert("Tensor range check" && 0 <= v126 && v126 < 2);
                            assert("Tensor range check" && 0 <= v128 && v128 < 1);
                            int v130;
                            v130 = 32 * v128;
                            int v131;
                            v131 = 1152 * v126;
                            int v132;
                            v132 = v131 + v130;
                            int v133;
                            v133 = 2048 * v126;
                            int v134;
                            v134 = v133 + v130;
                            constexpr int v135 = sizeof(float) * 4;
                            assert("Pointer alignment check" && (unsigned long long)(v124 + v134) % v135 == 0 && (unsigned long long)(v122 + v132) % v135 == 0);
                            cuda::memcpy_async(v122 + v132, v124 + v134, cuda::aligned_size_t<v135>(v135), v8);
                            v128 += 1 ;
                        }
                        v126 += 1 ;
                    }
                    v8.producer_commit();
                    // Poping the loop unrolling to: 0
                } else {
                }
                // Pushing the loop unrolling to: 0
                int v136;
                v136 = threadIdx.x;
                bool v137;
                v137 = 0 <= v136;
                bool v138;
                v138 = v137 == false;
                if (v138){
                    assert("The index needs to be zero or positive." && v137);
                } else {
                }
                int v140;
                v140 = v136 % 8;
                int v141;
                v141 = v136 / 8;
                bool v142;
                v142 = v141 < 32;
                bool v143;
                v143 = v142 == false;
                if (v143){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v142);
                } else {
                }
                assert("Tensor range check" && 0 <= v141 && v141 < 32);
                assert("Tensor range check" && 0 <= v140 && v140 < 8);
                int v145;
                v145 = 4 * v140;
                int v146;
                v146 = 36 * v141;
                int v147;
                v147 = v146 + v145;
                int v148;
                v148 = 64 * v141;
                int v149;
                v149 = v148 + v145;
                float * v150;
                v150 = v10+v147;
                float * v152;
                v152 = v101+v149;
                int v154;
                v154 = 0;
                #pragma unroll
                while (while_method_6(v154)){
                    int v156;
                    v156 = 0;
                    #pragma unroll
                    while (while_method_5(v156)){
                        assert("Tensor range check" && 0 <= v154 && v154 < 8);
                        assert("Tensor range check" && 0 <= v156 && v156 < 1);
                        int v158;
                        v158 = 32 * v156;
                        int v159;
                        v159 = 1152 * v154;
                        int v160;
                        v160 = v159 + v158;
                        int v161;
                        v161 = 2048 * v154;
                        int v162;
                        v162 = v161 + v158;
                        int4* v163;
                        v163 = reinterpret_cast<int4*>(v152 + v162);
                        int4* v164;
                        v164 = reinterpret_cast<int4*>(v150 + v160);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v163) % 16 == 0 && reinterpret_cast<unsigned long long>(v164) % 16 == 0);
                        *v164 = *v163;
                        v156 += 1 ;
                    }
                    v154 += 1 ;
                }
                // Poping the loop unrolling to: 0
                wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v165[1];
                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v166[4];
                cuda::pipeline_consumer_wait_prior<0>(v8);;
                __syncthreads();
                // Pushing the loop unrolling to: 0
                int v167;
                v167 = 0;
                #pragma unroll
                while (while_method_5(v167)){
                    int v169;
                    v169 = 0;
                    #pragma unroll
                    while (while_method_8(v169)){
                        assert("Tensor range check" && 0 <= v167 && v167 < 1);
                        assert("Tensor range check" && 0 <= v169 && v169 < 4);
                        int v171;
                        v171 = 4 * v167;
                        int v172;
                        v172 = v171 + v169;
                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v173 = v166[v172];
                        assert("Tensor range check" && 0 <= v167 && v167 < 1);
                        int v174;
                        v174 = 576 * v167;
                        assert("Tensor range check" && 0 <= v169 && v169 < 4);
                        int v175;
                        v175 = 8 * v169;
                        int v176;
                        v176 = v175 + v174;
                        int v177;
                        v177 = 0;
                        #pragma unroll
                        while (while_method_0(v177)){
                            int v179;
                            v179 = 0;
                            #pragma unroll
                            while (while_method_0(v179)){
                                assert("Tensor range check" && 0 <= v177 && v177 < 2);
                                assert("Tensor range check" && 0 <= v179 && v179 < 2);
                                int v181;
                                v181 = 4 * v179;
                                int v182;
                                v182 = v181 + v176;
                                int v183;
                                v183 = 288 * v177;
                                int v184;
                                v184 = v183 + v182;
                                float v185;
                                v185 = v61[v184];
                                bool v186;
                                v186 = 0 <= v179;
                                bool v188;
                                if (v186){
                                    bool v187;
                                    v187 = v179 < 2;
                                    v188 = v187;
                                } else {
                                    v188 = false;
                                }
                                bool v189;
                                v189 = v188 == false;
                                if (v189){
                                    assert("The indices should be inside the range of the dimension." && v188);
                                } else {
                                }
                                bool v191;
                                v191 = 0 <= v177;
                                bool v193;
                                if (v191){
                                    bool v192;
                                    v192 = v177 < 2;
                                    v193 = v192;
                                } else {
                                    v193 = false;
                                }
                                bool v194;
                                v194 = v193 == false;
                                if (v194){
                                    assert("The indices should be inside the range of the dimension." && v193);
                                } else {
                                }
                                int v196;
                                v196 = v177 * 2;
                                int v197;
                                v197 = v179 + v196;
                                v173.x[v197] = wmma::__float_to_tf32(v185);
                                v179 += 1 ;
                            }
                            v177 += 1 ;
                        }
                        v169 += 1 ;
                    }
                    v167 += 1 ;
                }
                // Poping the loop unrolling to: 0
                v8.consumer_release();
                switch (v97.tag) {
                    case 0: { // None
                        break;
                    }
                    case 1: { // Some
                        int v198 = v97.case1.v0;
                        assert("Tensor range check" && 0 <= v198 && v198 < 2);
                        int v199;
                        v199 = 32 * v198;
                        int v200;
                        v200 = v199 + v104;
                        float * v201;
                        v201 = v1+v200;
                        __syncthreads();
                        // Pushing the loop unrolling to: 0
                        v8.producer_acquire();
                        int v203;
                        v203 = threadIdx.x;
                        bool v204;
                        v204 = 0 <= v203;
                        bool v205;
                        v205 = v204 == false;
                        if (v205){
                            assert("The index needs to be zero or positive." && v204);
                        } else {
                        }
                        int v207;
                        v207 = v203 % 8;
                        int v208;
                        v208 = v203 / 8;
                        bool v209;
                        v209 = v208 < 32;
                        bool v210;
                        v210 = v209 == false;
                        if (v210){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v209);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v208 && v208 < 32);
                        assert("Tensor range check" && 0 <= v207 && v207 < 8);
                        int v212;
                        v212 = 4 * v207;
                        int v213;
                        v213 = 36 * v208;
                        int v214;
                        v214 = v213 + v212;
                        int v215;
                        v215 = 64 * v208;
                        int v216;
                        v216 = v215 + v212;
                        float * v217;
                        v217 = v12+v214;
                        float * v219;
                        v219 = v201+v216;
                        int v221;
                        v221 = 0;
                        #pragma unroll
                        while (while_method_0(v221)){
                            int v223;
                            v223 = 0;
                            #pragma unroll
                            while (while_method_5(v223)){
                                assert("Tensor range check" && 0 <= v221 && v221 < 2);
                                assert("Tensor range check" && 0 <= v223 && v223 < 1);
                                int v225;
                                v225 = 32 * v223;
                                int v226;
                                v226 = 1152 * v221;
                                int v227;
                                v227 = v226 + v225;
                                int v228;
                                v228 = 2048 * v221;
                                int v229;
                                v229 = v228 + v225;
                                constexpr int v230 = sizeof(float) * 4;
                                assert("Pointer alignment check" && (unsigned long long)(v219 + v229) % v230 == 0 && (unsigned long long)(v217 + v227) % v230 == 0);
                                cuda::memcpy_async(v217 + v227, v219 + v229, cuda::aligned_size_t<v230>(v230), v8);
                                v223 += 1 ;
                            }
                            v221 += 1 ;
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
                int v231;
                v231 = 0;
                #pragma unroll
                while (while_method_6(v231)){
                    int v233;
                    v233 = 0;
                    #pragma unroll
                    while (while_method_8(v233)){
                        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v235 = v165[0];
                        assert("Tensor range check" && 0 <= v231 && v231 < 8);
                        int v236;
                        v236 = 576 * v231;
                        assert("Tensor range check" && 0 <= v233 && v233 < 4);
                        int v237;
                        v237 = 8 * v233;
                        int v238;
                        v238 = v237 + v236;
                        int v239;
                        v239 = 0;
                        #pragma unroll
                        while (while_method_0(v239)){
                            int v241;
                            v241 = 0;
                            #pragma unroll
                            while (while_method_0(v241)){
                                assert("Tensor range check" && 0 <= v239 && v239 < 2);
                                assert("Tensor range check" && 0 <= v241 && v241 < 2);
                                int v243;
                                v243 = 288 * v241;
                                int v244;
                                v244 = v243 + v238;
                                int v245;
                                v245 = 4 * v239;
                                int v246;
                                v246 = v245 + v244;
                                float v247;
                                v247 = v45[v246];
                                bool v248;
                                v248 = 0 <= v241;
                                bool v250;
                                if (v248){
                                    bool v249;
                                    v249 = v241 < 2;
                                    v250 = v249;
                                } else {
                                    v250 = false;
                                }
                                bool v251;
                                v251 = v250 == false;
                                if (v251){
                                    assert("The indices should be inside the range of the dimension." && v250);
                                } else {
                                }
                                bool v253;
                                v253 = 0 <= v239;
                                bool v255;
                                if (v253){
                                    bool v254;
                                    v254 = v239 < 2;
                                    v255 = v254;
                                } else {
                                    v255 = false;
                                }
                                bool v256;
                                v256 = v255 == false;
                                if (v256){
                                    assert("The indices should be inside the range of the dimension." && v255);
                                } else {
                                }
                                int v258;
                                v258 = v239 * 2;
                                int v259;
                                v259 = v241 + v258;
                                v235.x[v259] = wmma::__float_to_tf32(v247);
                                v241 += 1 ;
                            }
                            v239 += 1 ;
                        }
                        int v260;
                        v260 = 0;
                        #pragma unroll
                        while (while_method_5(v260)){
                            assert("Tensor range check" && 0 <= v231 && v231 < 8);
                            assert("Tensor range check" && 0 <= v260 && v260 < 1);
                            int v262;
                            v262 = v231 + v260;
                            wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v263 = v63[v262];
                            assert("Tensor range check" && 0 <= v260 && v260 < 1);
                            assert("Tensor range check" && 0 <= v233 && v233 < 4);
                            int v264;
                            v264 = 4 * v260;
                            int v265;
                            v265 = v264 + v233;
                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v266 = v166[v265];
                            wmma::mma_sync(v263, v235, v266, v263);
                            v260 += 1 ;
                        }
                        v233 += 1 ;
                    }
                    v231 += 1 ;
                }
                // Poping the loop unrolling to: 0
                __syncthreads();
                v80 = v82;
            }
            // Pushing the loop unrolling to: 0
            int v267;
            v267 = 0;
            #pragma unroll
            while (while_method_6(v267)){
                int v269;
                v269 = 0;
                #pragma unroll
                while (while_method_5(v269)){
                    assert("Tensor range check" && 0 <= v267 && v267 < 8);
                    assert("Tensor range check" && 0 <= v269 && v269 < 1);
                    int v271;
                    v271 = v267 + v269;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v272 = v63[v271];
                    assert("Tensor range check" && 0 <= v267 && v267 < 8);
                    assert("Tensor range check" && 0 <= v269 && v269 < 1);
                    int v273;
                    v273 = 16 * v269;
                    int v274;
                    v274 = 1152 * v267;
                    int v275;
                    v275 = v274 + v273;
                    float * v276;
                    v276 = v29+v275;
                    wmma::store_matrix_sync(v276, v272, 72, wmma::mem_row_major);
                    v269 += 1 ;
                }
                v267 += 1 ;
            }
            // Poping the loop unrolling to: 0
            __syncthreads();
            // Pushing the loop unrolling to: 0
            int v278;
            v278 = threadIdx.x;
            bool v279;
            v279 = 0 <= v278;
            bool v280;
            v280 = v279 == false;
            if (v280){
                assert("The index needs to be zero or positive." && v279);
            } else {
            }
            int v282;
            v282 = v278 % 16;
            int v283;
            v283 = v278 / 16;
            bool v284;
            v284 = v283 < 16;
            bool v285;
            v285 = v284 == false;
            if (v285){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v284);
            } else {
            }
            assert("Tensor range check" && 0 <= v283 && v283 < 16);
            assert("Tensor range check" && 0 <= v282 && v282 < 16);
            int v287;
            v287 = 4 * v282;
            int v288;
            v288 = 64 * v283;
            int v289;
            v289 = v288 + v287;
            int v290;
            v290 = 72 * v283;
            int v291;
            v291 = v290 + v287;
            float * v292;
            v292 = v72+v289;
            float * v294;
            v294 = v14+v291;
            int v296;
            v296 = 0;
            #pragma unroll
            while (while_method_9(v296)){
                int v298;
                v298 = 0;
                #pragma unroll
                while (while_method_5(v298)){
                    assert("Tensor range check" && 0 <= v296 && v296 < 16);
                    assert("Tensor range check" && 0 <= v298 && v298 < 1);
                    int v300;
                    v300 = 64 * v298;
                    int v301;
                    v301 = 1024 * v296;
                    int v302;
                    v302 = v301 + v300;
                    int v303;
                    v303 = 1152 * v296;
                    int v304;
                    v304 = v303 + v300;
                    int4* v305;
                    v305 = reinterpret_cast<int4*>(v294 + v304);
                    int4* v306;
                    v306 = reinterpret_cast<int4*>(v292 + v302);
                    assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v305) % 16 == 0 && reinterpret_cast<unsigned long long>(v306) % 16 == 0);
                    *v306 = *v305;
                    v298 += 1 ;
                }
                v296 += 1 ;
            }
            // Poping the loop unrolling to: 0
            __syncthreads();
            v66 += 1 ;
        }
        v64 += 1 ;
    }
    return ;
}
__device__ void block_row_map_26(float * v0, int v1, float * v2){
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
    while (while_method_9(v23)){
        assert("Tensor range check" && 0 <= v23 && v23 < 16);
        int v25;
        v25 = 1024 * v23;
        int v26;
        v26 = v25 + v20;
        float v27[4];
        int v28[4];
        int v29;
        v29 = 0;
        while (while_method_5(v29)){
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
        while (while_method_5(v36)){
            int v38;
            v38 = 0;
            while (while_method_8(v38)){
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
        while (while_method_5(v73)){
            int v75;
            v75 = 0;
            while (while_method_8(v75)){
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
        while (while_method_5(v83)){
            int v85;
            v85 = 0;
            while (while_method_8(v85)){
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
                bool v91;
                v91 = isnan(v89);
                bool v92;
                v92 = v91 == false;
                bool v93;
                v93 = v92 == false;
                if (v93){
                    assert("What comes into regret matching must not be a nan." && v92);
                } else {
                }
                float v97;
                if (v90){
                    bool v95;
                    v95 = 0.0f >= v89;
                    if (v95){
                        v97 = 0.0f;
                    } else {
                        v97 = v89;
                    }
                } else {
                    v97 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v83 && v83 < 1);
                assert("Tensor range check" && 0 <= v85 && v85 < 4);
                v82[v88] = v97;
                v85 += 1 ;
            }
            v83 += 1 ;
        }
        float v98;
        v98 = 0.0f;
        int v99;
        v99 = 0;
        while (while_method_5(v99)){
            int v101;
            v101 = 0;
            while (while_method_8(v101)){
                assert("Tensor range check" && 0 <= v99 && v99 < 1);
                assert("Tensor range check" && 0 <= v101 && v101 < 4);
                int v103;
                v103 = 4 * v99;
                int v104;
                v104 = v103 + v101;
                float v105;
                v105 = v82[v104];
                float v106;
                v106 = v98 + v105;
                v98 = v106;
                v101 += 1 ;
            }
            v99 += 1 ;
        }
        auto v107 = cooperative_groups::coalesced_threads();
        int v108;
        v108 = threadIdx.x;
        int v109;
        v109 = v108 / 16;
        auto v110 = cooperative_groups::labeled_partition(v107,v109);
        Closure0 v111{};
        float v112;
        v112 = cooperative_groups::reduce(v110, v98, v111);
        int v113[4];
        int v114;
        v114 = 0;
        while (while_method_5(v114)){
            int v116;
            v116 = 0;
            while (while_method_8(v116)){
                assert("Tensor range check" && 0 <= v114 && v114 < 1);
                assert("Tensor range check" && 0 <= v116 && v116 < 4);
                int v118;
                v118 = 4 * v114;
                int v119;
                v119 = v118 + v116;
                bool v120;
                v120 = v72[v119];
                int v121;
                if (v120){
                    v121 = 1;
                } else {
                    v121 = 0;
                }
                assert("Tensor range check" && 0 <= v114 && v114 < 1);
                assert("Tensor range check" && 0 <= v116 && v116 < 4);
                v113[v119] = v121;
                v116 += 1 ;
            }
            v114 += 1 ;
        }
        int v122;
        v122 = 0;
        int v123;
        v123 = 0;
        while (while_method_5(v123)){
            int v125;
            v125 = 0;
            while (while_method_8(v125)){
                assert("Tensor range check" && 0 <= v123 && v123 < 1);
                assert("Tensor range check" && 0 <= v125 && v125 < 4);
                int v127;
                v127 = 4 * v123;
                int v128;
                v128 = v127 + v125;
                int v129;
                v129 = v113[v128];
                int v130;
                v130 = v122 + v129;
                v122 = v130;
                v125 += 1 ;
            }
            v123 += 1 ;
        }
        auto v131 = cooperative_groups::coalesced_threads();
        int v132;
        v132 = threadIdx.x;
        int v133;
        v133 = v132 / 16;
        auto v134 = cooperative_groups::labeled_partition(v131,v133);
        Closure1 v135{};
        int v136;
        v136 = cooperative_groups::reduce(v134, v122, v135);
        float v137;
        v137 = (float)v136;
        float v138;
        v138 = 1.0f / v137;
        bool v139;
        v139 = isnan(v138);
        bool v140;
        v140 = v139 == false;
        bool v141;
        v141 = v140 == false;
        if (v141){
            assert("Inverse length in regret matching must not be nan." && v140);
        } else {
        }
        float v143[4];
        int v144;
        v144 = 0;
        while (while_method_5(v144)){
            int v146;
            v146 = 0;
            while (while_method_8(v146)){
                assert("Tensor range check" && 0 <= v144 && v144 < 1);
                assert("Tensor range check" && 0 <= v146 && v146 < 4);
                int v148;
                v148 = 4 * v144;
                int v149;
                v149 = v148 + v146;
                float v150;
                v150 = v82[v149];
                bool v151;
                v151 = v72[v149];
                bool v152;
                v152 = v151 == false;
                float v157;
                if (v152){
                    v157 = 0.0f;
                } else {
                    bool v153;
                    v153 = v112 == 0.0f;
                    bool v154;
                    v154 = v153 != true;
                    if (v154){
                        float v155;
                        v155 = v150 / v112;
                        v157 = v155;
                    } else {
                        v157 = v138;
                    }
                }
                bool v158;
                v158 = isnan(v157);
                bool v159;
                v159 = v158 == false;
                bool v160;
                v160 = v159 == false;
                if (v160){
                    assert("What comes out of regret matching must not be a nan." && v159);
                } else {
                }
                bool v162;
                v162 = v157 >= 0.0f;
                bool v163;
                v163 = v162 == false;
                if (v163){
                    assert("What comes out of regret matching must be >= 0." && v162);
                } else {
                }
                bool v165;
                v165 = v157 <= 1.0f;
                bool v166;
                v166 = v165 == false;
                if (v166){
                    assert("What comes out of regret matching must be <= 1." && v165);
                } else {
                }
                assert("Tensor range check" && 0 <= v144 && v144 < 1);
                assert("Tensor range check" && 0 <= v146 && v146 < 4);
                v143[v149] = v157;
                v146 += 1 ;
            }
            v144 += 1 ;
        }
        assert("Tensor range check" && 0 <= v23 && v23 < 16);
        int v168;
        v168 = v25 + v22;
        int v169;
        v169 = 0;
        while (while_method_5(v169)){
            assert("Tensor range check" && 0 <= v169 && v169 < 1);
            int v171;
            v171 = 64 * v169;
            int v172;
            v172 = v171 + v168;
            assert("Tensor range check" && 0 <= v169 && v169 < 1);
            int v173;
            v173 = 4 * v169;
            int4* v174;
            v174 = reinterpret_cast<int4*>(v143 + v173);
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
__device__ int tag_28(Union6 v0){
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
__device__ bool is_pair_29(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
__device__ Tuple7 order_30(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple7{v1, v0};
    } else {
        return Tuple7{v0, v1};
    }
}
__device__ Union13 compare_hands_27(Union5 v0, bool v1, static_array<Union6,2> v2, int v3, static_array<int,2> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            __trap();
            break;
        }
        case 1: { // Some
            Union6 v7 = v0.case1.v0;
            int v8;
            v8 = tag_28(v7);
            Union6 v9;
            v9 = v2[0];
            int v11;
            v11 = tag_28(v9);
            Union6 v12;
            v12 = v2[1];
            int v14;
            v14 = tag_28(v12);
            bool v15;
            v15 = is_pair_29(v8, v11);
            bool v16;
            v16 = is_pair_29(v8, v14);
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
                    Tuple7 tmp33 = order_30(v8, v11);
                    v27 = tmp33.v0; v28 = tmp33.v1;
                    int v29; int v30;
                    Tuple7 tmp34 = order_30(v8, v14);
                    v29 = tmp34.v0; v30 = tmp34.v1;
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
__device__ void f_32(unsigned char * v0, unsigned int v1){
    unsigned int * v2;
    v2 = (unsigned int *)(v0+0ull);
    v2[0] = v1;
    return ;
}
__device__ void f_33(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+4ull);
    v2[0] = v1;
    return ;
}
__device__ void f_34(unsigned char * v0){
    return ;
}
__device__ void f_36(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+0ull);
    v2[0] = v1;
    return ;
}
__device__ void f_38(unsigned char * v0, Union6 v1){
    int v2;
    v2 = v1.tag;
    f_36(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // Jack
            return f_34(v3);
            break;
        }
        case 1: { // King
            return f_34(v3);
            break;
        }
        case 2: { // Queen
            return f_34(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_37(unsigned char * v0, Union5 v1, bool v2, static_array<Union6,2> v3, int v4, static_array<int,2> v5, int v6){
    int v7;
    v7 = v1.tag;
    f_36(v0, v7);
    unsigned char * v8;
    v8 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // None
            f_34(v8);
            break;
        }
        case 1: { // Some
            Union6 v10 = v1.case1.v0;
            f_38(v8, v10);
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
        f_38(v18, v25);
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
        f_36(v34, v41);
        v29 += 1 ;
    }
    int * v43;
    v43 = (int *)(v0+32ull);
    v43[0] = v6;
    return ;
}
__device__ void f_40(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+36ull);
    v2[0] = v1;
    return ;
}
__device__ void f_39(unsigned char * v0, Union5 v1, bool v2, static_array<Union6,2> v3, int v4, static_array<int,2> v5, int v6, Union1 v7){
    int v8;
    v8 = v1.tag;
    f_36(v0, v8);
    unsigned char * v9;
    v9 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // None
            f_34(v9);
            break;
        }
        case 1: { // Some
            Union6 v11 = v1.case1.v0;
            f_38(v9, v11);
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
        f_38(v19, v26);
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
        f_36(v35, v42);
        v30 += 1 ;
    }
    int * v44;
    v44 = (int *)(v0+32ull);
    v44[0] = v6;
    int v46;
    v46 = v7.tag;
    f_40(v0, v46);
    unsigned char * v47;
    v47 = (unsigned char *)(v0+40ull);
    switch (v7.tag) {
        case 0: { // Call
            return f_34(v47);
            break;
        }
        case 1: { // Fold
            return f_34(v47);
            break;
        }
        case 2: { // Raise
            return f_34(v47);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_35(unsigned char * v0, Union4 v1){
    int v2;
    v2 = v1.tag;
    f_36(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // ChanceCommunityCard
            Union5 v5 = v1.case0.v0; bool v6 = v1.case0.v1; static_array<Union6,2> v7 = v1.case0.v2; int v8 = v1.case0.v3; static_array<int,2> v9 = v1.case0.v4; int v10 = v1.case0.v5;
            return f_37(v3, v5, v6, v7, v8, v9, v10);
            break;
        }
        case 1: { // ChanceInit
            return f_34(v3);
            break;
        }
        case 2: { // Round
            Union5 v11 = v1.case2.v0; bool v12 = v1.case2.v1; static_array<Union6,2> v13 = v1.case2.v2; int v14 = v1.case2.v3; static_array<int,2> v15 = v1.case2.v4; int v16 = v1.case2.v5;
            return f_37(v3, v11, v12, v13, v14, v15, v16);
            break;
        }
        case 3: { // RoundWithAction
            Union5 v17 = v1.case3.v0; bool v18 = v1.case3.v1; static_array<Union6,2> v19 = v1.case3.v2; int v20 = v1.case3.v3; static_array<int,2> v21 = v1.case3.v4; int v22 = v1.case3.v5; Union1 v23 = v1.case3.v6;
            return f_39(v3, v17, v18, v19, v20, v21, v22, v23);
            break;
        }
        case 4: { // TerminalCall
            Union5 v24 = v1.case4.v0; bool v25 = v1.case4.v1; static_array<Union6,2> v26 = v1.case4.v2; int v27 = v1.case4.v3; static_array<int,2> v28 = v1.case4.v4; int v29 = v1.case4.v5;
            return f_37(v3, v24, v25, v26, v27, v28, v29);
            break;
        }
        case 5: { // TerminalFold
            Union5 v30 = v1.case5.v0; bool v31 = v1.case5.v1; static_array<Union6,2> v32 = v1.case5.v2; int v33 = v1.case5.v3; static_array<int,2> v34 = v1.case5.v4; int v35 = v1.case5.v5;
            return f_37(v3, v30, v31, v32, v33, v34, v35);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_41(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+80ull);
    v2[0] = v1;
    return ;
}
__device__ void f_43(unsigned char * v0, int v1, Union1 v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0] = v1;
    int v5;
    v5 = v2.tag;
    f_33(v0, v5);
    unsigned char * v6;
    v6 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // Call
            return f_34(v6);
            break;
        }
        case 1: { // Fold
            return f_34(v6);
            break;
        }
        case 2: { // Raise
            return f_34(v6);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_44(unsigned char * v0, int v1, Union6 v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0] = v1;
    int v5;
    v5 = v2.tag;
    f_33(v0, v5);
    unsigned char * v6;
    v6 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // Jack
            return f_34(v6);
            break;
        }
        case 1: { // King
            return f_34(v6);
            break;
        }
        case 2: { // Queen
            return f_34(v6);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_45(unsigned char * v0, static_array<Union6,2> v1, int v2, int v3){
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
        f_38(v8, v15);
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
__device__ void f_42(unsigned char * v0, Union7 v1){
    int v2;
    v2 = v1.tag;
    f_36(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // CommunityCardIs
            Union6 v5 = v1.case0.v0;
            return f_38(v3, v5);
            break;
        }
        case 1: { // PlayerAction
            int v6 = v1.case1.v0; Union1 v7 = v1.case1.v1;
            return f_43(v3, v6, v7);
            break;
        }
        case 2: { // PlayerGotCard
            int v8 = v1.case2.v0; Union6 v9 = v1.case2.v1;
            return f_44(v3, v8, v9);
            break;
        }
        case 3: { // Showdown
            static_array<Union6,2> v10 = v1.case3.v0; int v11 = v1.case3.v1; int v12 = v1.case3.v2;
            return f_45(v3, v10, v11, v12);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_46(unsigned char * v0, Union2 v1){
    int v2;
    v2 = v1.tag;
    f_36(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // Computer
            return f_34(v3);
            break;
        }
        case 1: { // Human
            return f_34(v3);
            break;
        }
        case 2: { // Random
            return f_34(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_47(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+1128ull);
    v2[0] = v1;
    return ;
}
__device__ void f_31(unsigned char * v0, unsigned int v1, Union3 v2, static_array_list<Union7,32> v3, static_array<Union2,2> v4, Union8 v5){
    f_32(v0, v1);
    int v6;
    v6 = v2.tag;
    f_33(v0, v6);
    unsigned char * v7;
    v7 = (unsigned char *)(v0+16ull);
    switch (v2.tag) {
        case 0: { // None
            f_34(v7);
            break;
        }
        case 1: { // Some
            Union4 v9 = v2.case1.v0;
            f_35(v7, v9);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v10;
    v10 = v3.length;
    f_41(v0, v10);
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
        f_42(v17, v19);
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
        f_46(v26, v33);
        v21 += 1 ;
    }
    int v35;
    v35 = v5.tag;
    f_47(v0, v35);
    unsigned char * v36;
    v36 = (unsigned char *)(v0+1136ull);
    switch (v5.tag) {
        case 0: { // GameNotStarted
            return f_34(v36);
            break;
        }
        case 1: { // GameOver
            Union5 v38 = v5.case1.v0; bool v39 = v5.case1.v1; static_array<Union6,2> v40 = v5.case1.v2; int v41 = v5.case1.v3; static_array<int,2> v42 = v5.case1.v4; int v43 = v5.case1.v5;
            return f_37(v36, v38, v39, v40, v41, v42, v43);
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            Union5 v44 = v5.case2.v0; bool v45 = v5.case2.v1; static_array<Union6,2> v46 = v5.case2.v2; int v47 = v5.case2.v3; static_array<int,2> v48 = v5.case2.v4; int v49 = v5.case2.v5;
            return f_37(v36, v44, v45, v46, v47, v48, v49);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ inline bool while_method_12(int v0){
    bool v1;
    v1 = v0 < 256;
    return v1;
}
__device__ inline bool while_method_13(Union3 v0){
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
__device__ void method_48(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut1 & v3, int v4, Union4 v5){
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
    while (while_method_13(v20)){
        Union3 v933;
        switch (v20.tag) {
            case 0: { // None
                v933 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v22 = v20.case1.v0;
                Union14 v773;
                switch (v22.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v742 = v22.case0.v0; bool v743 = v22.case0.v1; static_array<Union6,2> v744 = v22.case0.v2; int v745 = v22.case0.v3; static_array<int,2> v746 = v22.case0.v4; int v747 = v22.case0.v5;
                        curandStatePhilox4_32_10_t & v748 = v3.v5;
                        curandStatePhilox4_32_10_t & v749 = v748;
                        unsigned int & v750 = v3.v0;
                        Union6 v751; unsigned int v752;
                        Tuple6 tmp35 = draw_card_20(v749, v750);
                        v751 = tmp35.v0; v752 = tmp35.v1;
                        v3.v0 = v752;
                        Union7 v753;
                        v753 = Union7{Union7_0{v751}};
                        v18.push(v753);
                        v773 = Union14{Union14_0{v742, v743, v744, v745, v746, v747, v751}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v755 = v3.v5;
                        curandStatePhilox4_32_10_t & v756 = v755;
                        unsigned int & v757 = v3.v0;
                        Union6 v758; unsigned int v759;
                        Tuple6 tmp36 = draw_card_20(v756, v757);
                        v758 = tmp36.v0; v759 = tmp36.v1;
                        v3.v0 = v759;
                        curandStatePhilox4_32_10_t & v760 = v3.v5;
                        curandStatePhilox4_32_10_t & v761 = v760;
                        unsigned int & v762 = v3.v0;
                        Union6 v763; unsigned int v764;
                        Tuple6 tmp37 = draw_card_20(v761, v762);
                        v763 = tmp37.v0; v764 = tmp37.v1;
                        v3.v0 = v764;
                        Union7 v765;
                        v765 = Union7{Union7_2{0, v758}};
                        v18.push(v765);
                        Union7 v766;
                        v766 = Union7{Union7_2{1, v763}};
                        v18.push(v766);
                        v773 = Union14{Union14_1{v758, v763}};
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
                        Union1 v730;
                        switch (v84.tag) {
                            case 0: { // Computer
                                static_array_list<Union7,32> & v87 = v3.v2;
                                curandStatePhilox4_32_10_t & v88 = v3.v5;
                                curandStatePhilox4_32_10_t & v89 = v88;
                                float * v90;
                                v90 = reinterpret_cast<float *>(&v1[7864320ull]);
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
                                while (while_method_4(v159)){
                                    float * v161;
                                    v161 = reinterpret_cast<float *>(&v1[7864320ull]);
                                    assert("Tensor range check" && 0 <= v159 && v159 < 32);
                                    int v163;
                                    v163 = 393216 * v159;
                                    float * v164;
                                    v164 = reinterpret_cast<float *>(&v1[4718592ull]);
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
                                    block_map_24(v164, v173);
                                    float * v175;
                                    v175 = reinterpret_cast<float *>(&v0[1048576ull]);
                                    float * v177;
                                    v177 = reinterpret_cast<float *>(&v2[1048576ull]);
                                    assert("Tensor range check" && 0 <= v159 && v159 < 32);
                                    int v179;
                                    v179 = 4096 * v159;
                                    float * v180;
                                    v180 = reinterpret_cast<float *>(&v1[6291456ull]);
                                    block_matmul_25(v180, v175, v179, v164);
                                    block_row_map_26(v161, v163, v180);
                                    int * v182;
                                    v182 = reinterpret_cast<int *>(&v0[1572864ull]);
                                    bool * v184;
                                    v184 = reinterpret_cast<bool *>(&v0[1572880ull]);
                                    float * v186;
                                    v186 = reinterpret_cast<float *>(&v0[1572912ull]);
                                    float * v188;
                                    v188 = reinterpret_cast<float *>(&v0[1573040ull]);
                                    double * v190;
                                    v190 = reinterpret_cast<double *>(&v1[58195968ull]);
                                    double * v192;
                                    v192 = reinterpret_cast<double *>(&v1[61341696ull]);
                                    v159 += 1 ;
                                }
                                __syncthreads();
                                int * v194;
                                v194 = reinterpret_cast<int *>(&v0[1572864ull]);
                                bool * v196;
                                v196 = reinterpret_cast<bool *>(&v0[1572880ull]);
                                float * v198;
                                v198 = reinterpret_cast<float *>(&v0[1572912ull]);
                                float * v200;
                                v200 = reinterpret_cast<float *>(&v0[1573040ull]);
                                int v202;
                                v202 = v194[0];
                                float * v203;
                                v203 = reinterpret_cast<float *>(&v1[7864320ull]);
                                assert("Tensor range check" && 0 <= v202 && v202 < 32);
                                int v205;
                                v205 = 393216 * v202;
                                int v206;
                                v206 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v206 && v206 < 24);
                                int v207;
                                v207 = 16384 * v206;
                                int v208;
                                v208 = v207 + v205;
                                int v209;
                                v209 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v209 && v209 < 256);
                                int v210;
                                v210 = 64 * v209;
                                int v211;
                                v211 = v210 + v208;
                                float * v212;
                                v212 = v203+v211;
                                int v214;
                                v214 = sizeof(float *);
                                unsigned long long v215;
                                v215 = (unsigned long long)v214;
                                unsigned long long v216;
                                v216 = 256ull * v215;
                                unsigned long long v217;
                                v217 = v216 + 16ull;
                                unsigned long long v218;
                                v218 = v217 - 1ull;
                                unsigned long long v219;
                                v219 = v218 % 16ull;
                                unsigned long long v220;
                                v220 = v218 - v219;
                                unsigned long long v221;
                                v221 = v220 + 1024ull;
                                unsigned long long v222;
                                v222 = v221 + 16ull;
                                unsigned long long v223;
                                v223 = v222 - 1ull;
                                unsigned long long v224;
                                v224 = v223 % 16ull;
                                unsigned long long v225;
                                v225 = v223 - v224;
                                unsigned long long v226;
                                v226 = v225 + 1024ull;
                                bool v227;
                                v227 = v226 <= 98304ull;
                                bool v228;
                                v228 = v227 == false;
                                if (v228){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v227);
                                } else {
                                }
                                extern __shared__ unsigned char v230[];
                                bool v231;
                                v231 = v226 <= v226;
                                bool v232;
                                v232 = v231 == false;
                                if (v232){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v231);
                                } else {
                                }
                                float * * v234;
                                v234 = reinterpret_cast<float * *>(&v230[0ull]);
                                float * v236;
                                v236 = reinterpret_cast<float *>(&v230[v220]);
                                int * v238;
                                v238 = reinterpret_cast<int *>(&v230[v225]);
                                int v240;
                                v240 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v240 && v240 < 256);
                                v234[v240] = v212;
                                __syncthreads();
                                bool v241;
                                v241 = 0 <= v240;
                                bool v242;
                                v242 = v241 == false;
                                if (v242){
                                    assert("The index needs to be zero or positive." && v241);
                                } else {
                                }
                                int v244;
                                v244 = v240 % 16;
                                int v245;
                                v245 = v240 / 16;
                                bool v246;
                                v246 = v245 < 16;
                                bool v247;
                                v247 = v246 == false;
                                if (v247){
                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v246);
                                } else {
                                }
                                assert("Tensor range check" && 0 <= v245 && v245 < 16);
                                int v249;
                                v249 = 0;
                                while (while_method_9(v249)){
                                    bool v251;
                                    v251 = 0 <= v245;
                                    bool v252;
                                    v252 = v251 && v246;
                                    bool v253;
                                    v253 = v252 == false;
                                    if (v253){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v252);
                                    } else {
                                    }
                                    bool v255;
                                    v255 = 0 <= v249;
                                    bool v257;
                                    if (v255){
                                        bool v256;
                                        v256 = v249 < 16;
                                        v257 = v256;
                                    } else {
                                        v257 = false;
                                    }
                                    bool v258;
                                    v258 = v257 == false;
                                    if (v258){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v257);
                                    } else {
                                    }
                                    int v260;
                                    v260 = v249 * 16;
                                    int v261;
                                    v261 = v260 + v245;
                                    assert("Tensor range check" && 0 <= v249 && v249 < 16);
                                    int v262;
                                    v262 = 16 * v249;
                                    int v263;
                                    v263 = v262 + v245;
                                    float * v264;
                                    v264 = v234[v263];
                                    int v265;
                                    v265 = blockIdx.x;
                                    int v266;
                                    v266 = v265 * 256;
                                    int v267;
                                    v267 = v266 + v261;
                                    assert("Tensor range check" && 0 <= v244 && v244 < 16);
                                    int v268;
                                    v268 = 4 * v244;
                                    float v269[4];
                                    int v270[4];
                                    int v271;
                                    v271 = 0;
                                    while (while_method_5(v271)){
                                        assert("Tensor range check" && 0 <= v271 && v271 < 1);
                                        int v273;
                                        v273 = 4 * v271;
                                        assert("Tensor range check" && 0 <= v271 && v271 < 1);
                                        int v274;
                                        v274 = 64 * v271;
                                        int v275;
                                        v275 = v274 + v268;
                                        int4* v276;
                                        v276 = reinterpret_cast<int4*>(v264 + v275);
                                        int4* v277;
                                        v277 = reinterpret_cast<int4*>(v269 + v273);
                                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v276) % 16 == 0 && reinterpret_cast<unsigned long long>(v277) % 16 == 0);
                                        *v277 = *v276;
                                        v271 += 1 ;
                                    }
                                    int v278;
                                    v278 = 0;
                                    while (while_method_5(v278)){
                                        int v280;
                                        v280 = 0;
                                        while (while_method_8(v280)){
                                            bool v282;
                                            v282 = 0 <= v280;
                                            bool v284;
                                            if (v282){
                                                bool v283;
                                                v283 = v280 < 4;
                                                v284 = v283;
                                            } else {
                                                v284 = false;
                                            }
                                            bool v285;
                                            v285 = v284 == false;
                                            if (v285){
                                                assert("The indices should be inside the range of the dimension." && v284);
                                            } else {
                                            }
                                            bool v287;
                                            v287 = 0 <= v244;
                                            bool v289;
                                            if (v287){
                                                bool v288;
                                                v288 = v244 < 16;
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
                                            v292 = v244 * 4;
                                            int v293;
                                            v293 = v280 + v292;
                                            bool v294;
                                            v294 = 0 <= v278;
                                            bool v296;
                                            if (v294){
                                                bool v295;
                                                v295 = v278 < 1;
                                                v296 = v295;
                                            } else {
                                                v296 = false;
                                            }
                                            bool v297;
                                            v297 = v296 == false;
                                            if (v297){
                                                assert("The indices should be inside the range of the dimension." && v296);
                                            } else {
                                            }
                                            int v299;
                                            v299 = v278 * 64;
                                            int v300;
                                            v300 = v293 + v299;
                                            assert("Tensor range check" && 0 <= v278 && v278 < 1);
                                            assert("Tensor range check" && 0 <= v280 && v280 < 4);
                                            int v301;
                                            v301 = 4 * v278;
                                            int v302;
                                            v302 = v301 + v280;
                                            v270[v302] = v300;
                                            v280 += 1 ;
                                        }
                                        v278 += 1 ;
                                    }
                                    float v303[4];
                                    float v304;
                                    v304 = 0.0f;
                                    int v305;
                                    v305 = 0;
                                    while (while_method_5(v305)){
                                        assert("Tensor range check" && 0 <= v305 && v305 < 1);
                                        int v307;
                                        v307 = 4 * v305;
                                        assert("Tensor range check" && 0 <= v305 && v305 < 1);
                                        float v308;
                                        v308 = 0.0f;
                                        int v309;
                                        v309 = 0;
                                        while (while_method_8(v309)){
                                            assert("Tensor range check" && 0 <= v309 && v309 < 4);
                                            int v311;
                                            v311 = v309 + v307;
                                            float v312;
                                            v312 = v269[v311];
                                            float v313;
                                            v313 = v308 + v312;
                                            v308 = v313;
                                            v309 += 1 ;
                                        }
                                        auto v314 = cooperative_groups::coalesced_threads();
                                        int v315;
                                        v315 = threadIdx.x;
                                        int v316;
                                        v316 = v315 / 16;
                                        auto v317 = cooperative_groups::labeled_partition(v314,v316);
                                        Closure2 v318{};
                                        float v319;
                                        v319 = cooperative_groups::inclusive_scan(v317, v308, v318);
                                        float v320;
                                        v320 = v317.shfl_up(v319,1);
                                        bool v321;
                                        v321 = v317.thread_rank() == 0;
                                        float v322;
                                        if (v321){
                                            v322 = 0.0f;
                                        } else {
                                            v322 = v320;
                                        }
                                        float v323;
                                        v323 = v317.shfl(v319,v317.num_threads()-1);
                                        float v324;
                                        v324 = v304 + v322;
                                        float v325;
                                        v325 = v324;
                                        int v326;
                                        v326 = 0;
                                        while (while_method_8(v326)){
                                            assert("Tensor range check" && 0 <= v326 && v326 < 4);
                                            int v328;
                                            v328 = v326 + v307;
                                            float v329;
                                            v329 = v269[v328];
                                            float v330;
                                            v330 = v325 + v329;
                                            assert("Tensor range check" && 0 <= v326 && v326 < 4);
                                            v303[v328] = v330;
                                            v325 = v330;
                                            v326 += 1 ;
                                        }
                                        float v331;
                                        v331 = v304 + v323;
                                        v304 = v331;
                                        v305 += 1 ;
                                    }
                                    float v332[4];
                                    bool v333[4];
                                    int v334;
                                    v334 = 0;
                                    while (while_method_5(v334)){
                                        int v336;
                                        v336 = 0;
                                        while (while_method_8(v336)){
                                            assert("Tensor range check" && 0 <= v334 && v334 < 1);
                                            assert("Tensor range check" && 0 <= v336 && v336 < 4);
                                            int v338;
                                            v338 = 4 * v334;
                                            int v339;
                                            v339 = v338 + v336;
                                            float v340;
                                            v340 = v303[v339];
                                            float v341;
                                            v341 = v269[v339];
                                            bool v342;
                                            v342 = v341 > 0.0f;
                                            assert("Tensor range check" && 0 <= v334 && v334 < 1);
                                            assert("Tensor range check" && 0 <= v336 && v336 < 4);
                                            v332[v339] = v340;
                                            v333[v339] = v342;
                                            v336 += 1 ;
                                        }
                                        v334 += 1 ;
                                    }
                                    float v343; bool v344;
                                    Tuple8 tmp38 = Tuple8{-1.0f / 0.0f, false};
                                    v343 = tmp38.v0; v344 = tmp38.v1;
                                    int v345;
                                    v345 = 0;
                                    while (while_method_5(v345)){
                                        int v347;
                                        v347 = 0;
                                        while (while_method_8(v347)){
                                            assert("Tensor range check" && 0 <= v345 && v345 < 1);
                                            assert("Tensor range check" && 0 <= v347 && v347 < 4);
                                            int v349;
                                            v349 = 4 * v345;
                                            int v350;
                                            v350 = v349 + v347;
                                            float v351;
                                            v351 = v332[v350];
                                            bool v352;
                                            v352 = v333[v350];
                                            float v359; bool v360;
                                            if (v344){
                                                if (v352){
                                                    bool v353;
                                                    v353 = v343 >= v351;
                                                    float v354;
                                                    if (v353){
                                                        v354 = v343;
                                                    } else {
                                                        v354 = v351;
                                                    }
                                                    v359 = v354; v360 = true;
                                                } else {
                                                    v359 = v343; v360 = v344;
                                                }
                                            } else {
                                                if (v352){
                                                    v359 = v351; v360 = v352;
                                                } else {
                                                    v359 = v343; v360 = v344;
                                                }
                                            }
                                            v343 = v359;
                                            v344 = v360;
                                            v347 += 1 ;
                                        }
                                        v345 += 1 ;
                                    }
                                    auto v361 = cooperative_groups::coalesced_threads();
                                    int v362;
                                    v362 = threadIdx.x;
                                    int v363;
                                    v363 = v362 / 16;
                                    auto v364 = cooperative_groups::labeled_partition(v361,v363);
                                    Closure3 v365{};
                                    float v366; bool v367;
                                    Tuple8 tmp39 = cooperative_groups::reduce(v364, Tuple8{v343, v344}, v365);
                                    v366 = tmp39.v0; v367 = tmp39.v1;
                                    bool v368;
                                    v368 = v367 == false;
                                    if (v368){
                                        int v369;
                                        v369 = threadIdx.x;
                                        int v370;
                                        v370 = blockIdx.x;
                                        int v371;
                                        v371 = v370 * 256;
                                        int v372;
                                        v372 = v369 + v371;
                                        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v373 = console_lock;
                                        auto v374 = cooperative_groups::coalesced_threads();
                                        v373.acquire();
                                        int v375;
                                        v375 = 0;
                                        printf("{%s = %d; %s = %c","tid", v372, "x'", '[');
                                        int v376;
                                        v376 = 0;
                                        while (while_method_5(v376)){
                                            int v378;
                                            v378 = v375;
                                            bool v379;
                                            v379 = v378 >= 100;
                                            if (v379){
                                                printf("%s"," ...");
                                                break;
                                            } else {
                                            }
                                            bool v380;
                                            v380 = v376 == 0;
                                            bool v381;
                                            v381 = v380 != true;
                                            if (v381){
                                                printf("%s","; ");
                                            } else {
                                            }
                                            printf("%c",'[');
                                            int v382;
                                            v382 = 0;
                                            while (while_method_8(v382)){
                                                int v384;
                                                v384 = v375;
                                                bool v385;
                                                v385 = v384 >= 100;
                                                if (v385){
                                                    printf("%s"," ...");
                                                    break;
                                                } else {
                                                }
                                                bool v386;
                                                v386 = v382 == 0;
                                                bool v387;
                                                v387 = v386 != true;
                                                if (v387){
                                                    printf("%s","; ");
                                                } else {
                                                }
                                                int v388;
                                                v388 = v375 + 1;
                                                v375 = v388;
                                                int v389;
                                                v389 = v376 * 4;
                                                int v390;
                                                v390 = v389 + v382;
                                                float v391;
                                                v391 = v332[v390];
                                                bool v392;
                                                v392 = v333[v390];
                                                const char * v395;
                                                if (v392){
                                                    const char * v393;
                                                    v393 = "true";
                                                    v395 = v393;
                                                } else {
                                                    const char * v394;
                                                    v394 = "false";
                                                    v395 = v394;
                                                }
                                                printf("%f, %s",v391, v395);
                                                v382 += 1 ;
                                            }
                                            printf("%c",']');
                                            v376 += 1 ;
                                        }
                                        printf("%c",']');
                                        printf("}\n");
                                        v373.release();
                                        v374.sync() ;
                                    } else {
                                    }
                                    if (v368){
                                        assert("The local reduce must be true." && v367);
                                    } else {
                                    }
                                    float v431[4];
                                    int v432[4];
                                    int v433;
                                    v433 = 0;
                                    while (while_method_5(v433)){
                                        int v435;
                                        v435 = 0;
                                        while (while_method_8(v435)){
                                            assert("Tensor range check" && 0 <= v433 && v433 < 1);
                                            assert("Tensor range check" && 0 <= v435 && v435 < 4);
                                            int v437;
                                            v437 = 4 * v433;
                                            int v438;
                                            v438 = v437 + v435;
                                            int v439;
                                            v439 = v270[v438];
                                            float v440;
                                            v440 = curand_uniform(&v89);
                                            assert("Tensor range check" && 0 <= v433 && v433 < 1);
                                            assert("Tensor range check" && 0 <= v435 && v435 < 4);
                                            v431[v438] = v440;
                                            v432[v438] = v439;
                                            v435 += 1 ;
                                        }
                                        v433 += 1 ;
                                    }
                                    float v441; int v442;
                                    Tuple9 tmp40 = Tuple9{0.0f, 2147483647};
                                    v441 = tmp40.v0; v442 = tmp40.v1;
                                    int v443;
                                    v443 = 0;
                                    while (while_method_5(v443)){
                                        int v445;
                                        v445 = 0;
                                        while (while_method_8(v445)){
                                            assert("Tensor range check" && 0 <= v443 && v443 < 1);
                                            assert("Tensor range check" && 0 <= v445 && v445 < 4);
                                            int v447;
                                            v447 = 4 * v443;
                                            int v448;
                                            v448 = v447 + v445;
                                            float v449;
                                            v449 = v431[v448];
                                            int v450;
                                            v450 = v432[v448];
                                            bool v451;
                                            v451 = v442 < v450;
                                            float v452; int v453;
                                            if (v451){
                                                v452 = v441; v453 = v442;
                                            } else {
                                                v452 = v449; v453 = v450;
                                            }
                                            v441 = v452;
                                            v442 = v453;
                                            v445 += 1 ;
                                        }
                                        v443 += 1 ;
                                    }
                                    auto v454 = cooperative_groups::coalesced_threads();
                                    int v455;
                                    v455 = threadIdx.x;
                                    int v456;
                                    v456 = v455 / 16;
                                    auto v457 = cooperative_groups::labeled_partition(v454,v456);
                                    Closure4 v458{};
                                    float v459; int v460;
                                    Tuple9 tmp41 = cooperative_groups::reduce(v457, Tuple9{v441, v442}, v458);
                                    v459 = tmp41.v0; v460 = tmp41.v1;
                                    float v461;
                                    v461 = v366 * v459;
                                    int v462[4];
                                    bool v463[4];
                                    int v464;
                                    v464 = 0;
                                    while (while_method_5(v464)){
                                        int v466;
                                        v466 = 0;
                                        while (while_method_8(v466)){
                                            assert("Tensor range check" && 0 <= v464 && v464 < 1);
                                            assert("Tensor range check" && 0 <= v466 && v466 < 4);
                                            int v468;
                                            v468 = 4 * v464;
                                            int v469;
                                            v469 = v468 + v466;
                                            float v470;
                                            v470 = v332[v469];
                                            bool v471;
                                            v471 = v333[v469];
                                            int v472;
                                            v472 = v270[v469];
                                            int v475; bool v476;
                                            if (v471){
                                                float v473;
                                                v473 = v470 - v461;
                                                bool v474;
                                                v474 = v473 >= 0.0f;
                                                v475 = v472; v476 = v474;
                                            } else {
                                                v475 = 2147483647; v476 = false;
                                            }
                                            assert("Tensor range check" && 0 <= v464 && v464 < 1);
                                            assert("Tensor range check" && 0 <= v466 && v466 < 4);
                                            v462[v469] = v475;
                                            v463[v469] = v476;
                                            v466 += 1 ;
                                        }
                                        v464 += 1 ;
                                    }
                                    int v477; bool v478;
                                    Tuple10 tmp42 = Tuple10{2147483647, false};
                                    v477 = tmp42.v0; v478 = tmp42.v1;
                                    int v479;
                                    v479 = 0;
                                    while (while_method_5(v479)){
                                        int v481;
                                        v481 = 0;
                                        while (while_method_8(v481)){
                                            assert("Tensor range check" && 0 <= v479 && v479 < 1);
                                            assert("Tensor range check" && 0 <= v481 && v481 < 4);
                                            int v483;
                                            v483 = 4 * v479;
                                            int v484;
                                            v484 = v483 + v481;
                                            int v485;
                                            v485 = v462[v484];
                                            bool v486;
                                            v486 = v463[v484];
                                            int v493; bool v494;
                                            if (v478){
                                                if (v486){
                                                    bool v487;
                                                    v487 = v477 < v485;
                                                    int v488;
                                                    if (v487){
                                                        v488 = v477;
                                                    } else {
                                                        v488 = v485;
                                                    }
                                                    v493 = v488; v494 = true;
                                                } else {
                                                    v493 = v477; v494 = v478;
                                                }
                                            } else {
                                                if (v486){
                                                    v493 = v485; v494 = v486;
                                                } else {
                                                    v493 = v477; v494 = v478;
                                                }
                                            }
                                            v477 = v493;
                                            v478 = v494;
                                            v481 += 1 ;
                                        }
                                        v479 += 1 ;
                                    }
                                    auto v495 = cooperative_groups::coalesced_threads();
                                    int v496;
                                    v496 = threadIdx.x;
                                    int v497;
                                    v497 = v496 / 16;
                                    auto v498 = cooperative_groups::labeled_partition(v495,v497);
                                    Closure5 v499{};
                                    int v500; bool v501;
                                    Tuple10 tmp43 = cooperative_groups::reduce(v498, Tuple10{v477, v478}, v499);
                                    v500 = tmp43.v0; v501 = tmp43.v1;
                                    bool v502;
                                    v502 = v501 == false;
                                    if (v502){
                                        int v503;
                                        v503 = threadIdx.x;
                                        int v504;
                                        v504 = blockIdx.x;
                                        int v505;
                                        v505 = v504 * 256;
                                        int v506;
                                        v506 = v503 + v505;
                                        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v507 = console_lock;
                                        auto v508 = cooperative_groups::coalesced_threads();
                                        v507.acquire();
                                        int v509;
                                        v509 = 0;
                                        printf("{%s = %d; %s = %c","tid", v506, "x'", '[');
                                        int v510;
                                        v510 = 0;
                                        while (while_method_5(v510)){
                                            int v512;
                                            v512 = v509;
                                            bool v513;
                                            v513 = v512 >= 100;
                                            if (v513){
                                                printf("%s"," ...");
                                                break;
                                            } else {
                                            }
                                            bool v514;
                                            v514 = v510 == 0;
                                            bool v515;
                                            v515 = v514 != true;
                                            if (v515){
                                                printf("%s","; ");
                                            } else {
                                            }
                                            printf("%c",'[');
                                            int v516;
                                            v516 = 0;
                                            while (while_method_8(v516)){
                                                int v518;
                                                v518 = v509;
                                                bool v519;
                                                v519 = v518 >= 100;
                                                if (v519){
                                                    printf("%s"," ...");
                                                    break;
                                                } else {
                                                }
                                                bool v520;
                                                v520 = v516 == 0;
                                                bool v521;
                                                v521 = v520 != true;
                                                if (v521){
                                                    printf("%s","; ");
                                                } else {
                                                }
                                                int v522;
                                                v522 = v509 + 1;
                                                v509 = v522;
                                                int v523;
                                                v523 = v510 * 4;
                                                int v524;
                                                v524 = v523 + v516;
                                                int v525;
                                                v525 = v462[v524];
                                                bool v526;
                                                v526 = v463[v524];
                                                const char * v529;
                                                if (v526){
                                                    const char * v527;
                                                    v527 = "true";
                                                    v529 = v527;
                                                } else {
                                                    const char * v528;
                                                    v528 = "false";
                                                    v529 = v528;
                                                }
                                                printf("%d, %s",v525, v529);
                                                v516 += 1 ;
                                            }
                                            printf("%c",']');
                                            v510 += 1 ;
                                        }
                                        printf("%c",']');
                                        printf("}\n");
                                        v507.release();
                                        v508.sync() ;
                                    } else {
                                    }
                                    if (v502){
                                        assert("The local reduce must be true." && v501);
                                    } else {
                                    }
                                    float v565; int v566;
                                    Tuple9 tmp44 = Tuple9{0.0f, 2147483647};
                                    v565 = tmp44.v0; v566 = tmp44.v1;
                                    int v567;
                                    v567 = 0;
                                    while (while_method_5(v567)){
                                        int v569;
                                        v569 = 0;
                                        while (while_method_8(v569)){
                                            assert("Tensor range check" && 0 <= v567 && v567 < 1);
                                            assert("Tensor range check" && 0 <= v569 && v569 < 4);
                                            int v571;
                                            v571 = 4 * v567;
                                            int v572;
                                            v572 = v571 + v569;
                                            float v573;
                                            v573 = v269[v572];
                                            int v574;
                                            v574 = v270[v572];
                                            bool v575;
                                            v575 = v566 == v500;
                                            float v579; int v580;
                                            if (v575){
                                                v579 = v565; v580 = v566;
                                            } else {
                                                bool v576;
                                                v576 = v574 == v500;
                                                if (v576){
                                                    v579 = v573; v580 = v574;
                                                } else {
                                                    v579 = v565; v580 = v566;
                                                }
                                            }
                                            v565 = v579;
                                            v566 = v580;
                                            v569 += 1 ;
                                        }
                                        v567 += 1 ;
                                    }
                                    auto v581 = cooperative_groups::coalesced_threads();
                                    int v582;
                                    v582 = threadIdx.x;
                                    int v583;
                                    v583 = v582 / 16;
                                    auto v584 = cooperative_groups::labeled_partition(v581,v583);
                                    Closure6 v585{v500};
                                    float v586; int v587;
                                    Tuple9 tmp45 = cooperative_groups::reduce(v584, Tuple9{v565, v566}, v585);
                                    v586 = tmp45.v0; v587 = tmp45.v1;
                                    bool v588;
                                    v588 = v587 == 2147483647;
                                    bool v589;
                                    v589 = v588 != true;
                                    bool v590;
                                    v590 = v589 == false;
                                    if (v590){
                                        assert("Expected a valid action id in get_prob." && v589);
                                    } else {
                                    }
                                    int v592;
                                    v592 = 0;
                                    while (while_method_5(v592)){
                                        assert("Tensor range check" && 0 <= v592 && v592 < 1);
                                        assert("Tensor range check" && 0 <= v592 && v592 < 1);
                                        v592 += 1 ;
                                    }
                                    assert("Tensor range check" && 0 <= v261 && v261 < 256);
                                    v236[v261] = v586;
                                    v238[v261] = v500;
                                    v249 += 1 ;
                                }
                                __syncthreads();
                                assert("Tensor range check" && 0 <= v240 && v240 < 256);
                                float v594;
                                v594 = v236[v240];
                                int v595;
                                v595 = v238[v240];
                                __syncthreads();
                                extern __shared__ unsigned char v596[];
                                float * v597;
                                v597 = reinterpret_cast<float *>(&v596[0ull]);
                                int * v599;
                                v599 = reinterpret_cast<int *>(&v596[16ull]);
                                int v601;
                                v601 = threadIdx.x;
                                bool v602;
                                v602 = v601 == 0;
                                if (v602){
                                    v597[0] = v594;
                                    v599[0] = v595;
                                } else {
                                }
                                __syncthreads();
                                float v603;
                                v603 = v597[0];
                                int v604;
                                v604 = v599[0];
                                __syncthreads();
                                double * v605;
                                v605 = reinterpret_cast<double *>(&v1[58195968ull]);
                                double * v607;
                                v607 = reinterpret_cast<double *>(&v1[61341696ull]);
                                int v609;
                                v609 = threadIdx.x;
                                int v610;
                                v610 = blockIdx.x;
                                int v611;
                                v611 = v610 * 256;
                                int v612;
                                v612 = v609 + v611;
                                int v613;
                                v613 = 0;
                                while (while_method_4(v613)){
                                    float * v615;
                                    v615 = reinterpret_cast<float *>(&v1[7864320ull]);
                                    int v617;
                                    v617 = blockIdx.x;
                                    int v618;
                                    v618 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v613 && v613 < 32);
                                    assert("Tensor range check" && 0 <= v617 && v617 < 24);
                                    assert("Tensor range check" && 0 <= v618 && v618 < 256);
                                    assert("Tensor range check" && 0 <= v604 && v604 < 64);
                                    int v619;
                                    v619 = 64 * v618;
                                    int v620;
                                    v620 = v619 + v604;
                                    int v621;
                                    v621 = 16384 * v617;
                                    int v622;
                                    v622 = v621 + v620;
                                    int v623;
                                    v623 = 393216 * v613;
                                    int v624;
                                    v624 = v623 + v622;
                                    float v625;
                                    v625 = v615[v624];
                                    double v626;
                                    v626 = (double)v603;
                                    double v627;
                                    v627 = log(v626);
                                    double v628;
                                    v628 = (double)v625;
                                    double v629;
                                    v629 = log(v628);
                                    assert("Tensor range check" && 0 <= v613 && v613 < 32);
                                    assert("Tensor range check" && 0 <= v612 && v612 < 6144);
                                    assert("Tensor range check" && 0 <= v75 && v75 < 2);
                                    int v630;
                                    v630 = 2 * v612;
                                    int v631;
                                    v631 = v630 + v75;
                                    int v632;
                                    v632 = 12288 * v613;
                                    int v633;
                                    v633 = v632 + v631;
                                    double v634;
                                    v634 = v605[v633];
                                    double v635;
                                    v635 = v607[v633];
                                    double v636;
                                    v636 = v629 + v634;
                                    double v637;
                                    v637 = v627 + v635;
                                    bool v638;
                                    v638 = isnan(v637);
                                    bool v639;
                                    v639 = v638 == false;
                                    bool v640;
                                    v640 = v639 == false;
                                    if (v640){
                                        assert("The sampling log probability shouldn't be nan." && v639);
                                    } else {
                                    }
                                    bool v642;
                                    v642 = isnan(v636);
                                    bool v643;
                                    v643 = v642 == false;
                                    bool v644;
                                    v644 = v643 == false;
                                    if (v644){
                                        assert("The policy log probability shouldn't be nan." && v643);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v613 && v613 < 32);
                                    assert("Tensor range check" && 0 <= v612 && v612 < 6144);
                                    assert("Tensor range check" && 0 <= v75 && v75 < 2);
                                    v605[v633] = v636;
                                    v607[v633] = v637;
                                    v613 += 1 ;
                                }
                                bool v646;
                                v646 = 0 == v604;
                                Union12 v655;
                                if (v646){
                                    v655 = Union12{Union12_1{}};
                                } else {
                                    bool v648;
                                    v648 = 1 == v604;
                                    if (v648){
                                        v655 = Union12{Union12_0{}};
                                    } else {
                                        bool v650;
                                        v650 = 2 == v604;
                                        if (v650){
                                            v655 = Union12{Union12_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v655.tag) {
                                    case 0: { // AA_Call
                                        v730 = Union1{Union1_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v656;
                                        v656 = v76[0];
                                        int v658; int v659;
                                        Tuple7 tmp46 = Tuple7{1, v656};
                                        v658 = tmp46.v0; v659 = tmp46.v1;
                                        while (while_method_0(v658)){
                                            bool v661;
                                            v661 = 0 <= v658;
                                            bool v663;
                                            if (v661){
                                                bool v662;
                                                v662 = v658 < 2;
                                                v663 = v662;
                                            } else {
                                                v663 = false;
                                            }
                                            bool v664;
                                            v664 = v663 == false;
                                            if (v664){
                                                assert("Index must be in range." && v663);
                                            } else {
                                            }
                                            int v666;
                                            v666 = v76[v658];
                                            bool v668;
                                            v668 = v659 >= v666;
                                            int v669;
                                            if (v668){
                                                v669 = v659;
                                            } else {
                                                v669 = v666;
                                            }
                                            v659 = v669;
                                            v658 += 1 ;
                                        }
                                        bool v671;
                                        if (v79){
                                            bool v670;
                                            v670 = v75 < 2;
                                            v671 = v670;
                                        } else {
                                            v671 = false;
                                        }
                                        bool v672;
                                        v672 = v671 == false;
                                        if (v672){
                                            assert("Index must be in range." && v671);
                                        } else {
                                        }
                                        int v674;
                                        v674 = v76[v75];
                                        bool v676;
                                        v676 = v674 == v659;
                                        if (v676){
                                            v730 = Union1{Union1_0{}};
                                        } else {
                                            v730 = Union1{Union1_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v681;
                                        v681 = v77 > 0;
                                        if (v681){
                                            v730 = Union1{Union1_2{}};
                                        } else {
                                            v730 = Union1{Union1_0{}};
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
                                curandStatePhilox4_32_10_t & v688 = v3.v5;
                                curandStatePhilox4_32_10_t & v689 = v688;
                                static_array_list<Union1,3> v690;
                                v690 = static_array_list<Union1,3>{};
                                v690.unsafe_set_length(1);
                                Union1 v692;
                                v692 = Union1{Union1_0{}};
                                v690[0] = v692;
                                int v694;
                                v694 = v76[0];
                                int v696;
                                v696 = v76[1];
                                bool v698;
                                v698 = v694 == v696;
                                bool v699;
                                v699 = v698 != true;
                                if (v699){
                                    Union1 v700;
                                    v700 = Union1{Union1_1{}};
                                    v690.push(v700);
                                } else {
                                }
                                bool v701;
                                v701 = v77 > 0;
                                if (v701){
                                    Union1 v702;
                                    v702 = Union1{Union1_2{}};
                                    v690.push(v702);
                                } else {
                                }
                                int v703;
                                v703 = v690.length;
                                int v704;
                                v704 = v703 - 1;
                                int v705;
                                v705 = 0;
                                while (while_method_1(v704, v705)){
                                    int v707;
                                    v707 = v690.length;
                                    int v708;
                                    v708 = int_range_22(v707, v705, v689);
                                    Union1 v709;
                                    v709 = v690[v705];
                                    Union1 v711;
                                    v711 = v690[v708];
                                    v690[v705] = v711;
                                    v690[v708] = v709;
                                    v705 += 1 ;
                                }
                                Union1 v713;
                                v713 = v690.pop();
                                int v714;
                                v714 = sizeof(Union1);
                                unsigned long long v715;
                                v715 = (unsigned long long)v714;
                                bool v716;
                                v716 = v715 <= 98304ull;
                                bool v717;
                                v717 = v716 == false;
                                if (v717){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v716);
                                } else {
                                }
                                extern __shared__ unsigned char v719[];
                                bool v720;
                                v720 = v715 <= v715;
                                bool v721;
                                v721 = v720 == false;
                                if (v721){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v720);
                                } else {
                                }
                                Union1 * v723;
                                v723 = reinterpret_cast<Union1 *>(&v719[0ull]);
                                int v725;
                                v725 = threadIdx.x;
                                bool v726;
                                v726 = v725 == 0;
                                if (v726){
                                    v723[0] = v713;
                                } else {
                                }
                                __syncthreads();
                                Union1 v727;
                                v727 = v723[0];
                                __syncthreads();
                                v730 = v727;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union7 v731;
                        v731 = Union7{Union7_1{v75, v730}};
                        v18.push(v731);
                        v773 = Union14{Union14_2{v72, v73, v74, v75, v76, v77, v730}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v733 = v22.case3.v0; bool v734 = v22.case3.v1; static_array<Union6,2> v735 = v22.case3.v2; int v736 = v22.case3.v3; static_array<int,2> v737 = v22.case3.v4; int v738 = v22.case3.v5; Union1 v739 = v22.case3.v6;
                        Union7 v740;
                        v740 = Union7{Union7_1{v736, v739}};
                        v18.push(v740);
                        v773 = Union14{Union14_2{v733, v734, v735, v736, v737, v738, v739}};
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
                        v56 = compare_hands_27(v43, v44, v45, v46, v47, v48);
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
                        v773 = Union14{Union14_3{}};
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
                        v773 = Union14{Union14_3{}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                switch (v773.tag) {
                    case 0: { // T_game_chance_community_card
                        Union5 v775 = v773.case0.v0; bool v776 = v773.case0.v1; static_array<Union6,2> v777 = v773.case0.v2; int v778 = v773.case0.v3; static_array<int,2> v779 = v773.case0.v4; int v780 = v773.case0.v5; Union6 v781 = v773.case0.v6;
                        int v782;
                        v782 = 2;
                        int v783; int v784;
                        Tuple7 tmp47 = Tuple7{0, 0};
                        v783 = tmp47.v0; v784 = tmp47.v1;
                        while (while_method_0(v783)){
                            bool v786;
                            v786 = 0 <= v783;
                            bool v788;
                            if (v786){
                                bool v787;
                                v787 = v783 < 2;
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
                            v791 = v779[v783];
                            bool v793;
                            v793 = v784 >= v791;
                            int v794;
                            if (v793){
                                v794 = v784;
                            } else {
                                v794 = v791;
                            }
                            v784 = v794;
                            v783 += 1 ;
                        }
                        static_array<int,2> v795;
                        int v797;
                        v797 = 0;
                        while (while_method_0(v797)){
                            v795[v797] = v784;
                            v797 += 1 ;
                        }
                        Union5 v799;
                        v799 = Union5{Union5_1{v781}};
                        Union4 v800;
                        v800 = Union4{Union4_2{v799, true, v777, 0, v795, v782}};
                        v933 = Union3{Union3_1{v800}};
                        break;
                    }
                    case 1: { // T_game_chance_init
                        Union6 v802 = v773.case1.v0; Union6 v803 = v773.case1.v1;
                        int v804;
                        v804 = 2;
                        static_array<int,2> v805;
                        v805[0] = 1;
                        v805[1] = 1;
                        static_array<Union6,2> v807;
                        v807[0] = v802;
                        v807[1] = v803;
                        Union5 v809;
                        v809 = Union5{Union5_0{}};
                        Union4 v810;
                        v810 = Union4{Union4_2{v809, true, v807, 0, v805, v804}};
                        v933 = Union3{Union3_1{v810}};
                        break;
                    }
                    case 2: { // T_game_round
                        Union5 v812 = v773.case2.v0; bool v813 = v773.case2.v1; static_array<Union6,2> v814 = v773.case2.v2; int v815 = v773.case2.v3; static_array<int,2> v816 = v773.case2.v4; int v817 = v773.case2.v5; Union1 v818 = v773.case2.v6;
                        Union4 v925;
                        switch (v812.tag) {
                            case 0: { // None
                                switch (v818.tag) {
                                    case 0: { // Call
                                        if (v813){
                                            int v881;
                                            v881 = v815 ^ 1;
                                            v925 = Union4{Union4_2{v812, false, v814, v881, v816, v817}};
                                        } else {
                                            v925 = Union4{Union4_0{v812, v813, v814, v815, v816, v817}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v925 = Union4{Union4_5{v812, v813, v814, v815, v816, v817}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v885;
                                        v885 = v817 > 0;
                                        if (v885){
                                            int v886;
                                            v886 = v815 ^ 1;
                                            int v887;
                                            v887 = -1 + v817;
                                            int v888; int v889;
                                            Tuple7 tmp48 = Tuple7{0, 0};
                                            v888 = tmp48.v0; v889 = tmp48.v1;
                                            while (while_method_0(v888)){
                                                bool v891;
                                                v891 = 0 <= v888;
                                                bool v893;
                                                if (v891){
                                                    bool v892;
                                                    v892 = v888 < 2;
                                                    v893 = v892;
                                                } else {
                                                    v893 = false;
                                                }
                                                bool v894;
                                                v894 = v893 == false;
                                                if (v894){
                                                    assert("Index must be in range." && v893);
                                                } else {
                                                }
                                                int v896;
                                                v896 = v816[v888];
                                                bool v898;
                                                v898 = v889 >= v896;
                                                int v899;
                                                if (v898){
                                                    v899 = v889;
                                                } else {
                                                    v899 = v896;
                                                }
                                                v889 = v899;
                                                v888 += 1 ;
                                            }
                                            static_array<int,2> v900;
                                            int v902;
                                            v902 = 0;
                                            while (while_method_0(v902)){
                                                v900[v902] = v889;
                                                v902 += 1 ;
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
                                                v913 = v900[v906];
                                                bool v915;
                                                v915 = v906 == v815;
                                                int v917;
                                                if (v915){
                                                    int v916;
                                                    v916 = v913 + 2;
                                                    v917 = v916;
                                                } else {
                                                    v917 = v913;
                                                }
                                                v904[v906] = v917;
                                                v906 += 1 ;
                                            }
                                            v925 = Union4{Union4_2{v812, false, v814, v886, v904, v887}};
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
                                Union6 v819 = v812.case1.v0;
                                switch (v818.tag) {
                                    case 0: { // Call
                                        if (v813){
                                            int v821;
                                            v821 = v815 ^ 1;
                                            v925 = Union4{Union4_2{v812, false, v814, v821, v816, v817}};
                                        } else {
                                            int v823; int v824;
                                            Tuple7 tmp49 = Tuple7{0, 0};
                                            v823 = tmp49.v0; v824 = tmp49.v1;
                                            while (while_method_0(v823)){
                                                bool v826;
                                                v826 = 0 <= v823;
                                                bool v828;
                                                if (v826){
                                                    bool v827;
                                                    v827 = v823 < 2;
                                                    v828 = v827;
                                                } else {
                                                    v828 = false;
                                                }
                                                bool v829;
                                                v829 = v828 == false;
                                                if (v829){
                                                    assert("Index must be in range." && v828);
                                                } else {
                                                }
                                                int v831;
                                                v831 = v816[v823];
                                                bool v833;
                                                v833 = v824 >= v831;
                                                int v834;
                                                if (v833){
                                                    v834 = v824;
                                                } else {
                                                    v834 = v831;
                                                }
                                                v824 = v834;
                                                v823 += 1 ;
                                            }
                                            static_array<int,2> v835;
                                            int v837;
                                            v837 = 0;
                                            while (while_method_0(v837)){
                                                v835[v837] = v824;
                                                v837 += 1 ;
                                            }
                                            v925 = Union4{Union4_4{v812, v813, v814, v815, v835, v817}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v925 = Union4{Union4_5{v812, v813, v814, v815, v816, v817}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v841;
                                        v841 = v817 > 0;
                                        if (v841){
                                            int v842;
                                            v842 = v815 ^ 1;
                                            int v843;
                                            v843 = -1 + v817;
                                            int v844; int v845;
                                            Tuple7 tmp50 = Tuple7{0, 0};
                                            v844 = tmp50.v0; v845 = tmp50.v1;
                                            while (while_method_0(v844)){
                                                bool v847;
                                                v847 = 0 <= v844;
                                                bool v849;
                                                if (v847){
                                                    bool v848;
                                                    v848 = v844 < 2;
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
                                                v852 = v816[v844];
                                                bool v854;
                                                v854 = v845 >= v852;
                                                int v855;
                                                if (v854){
                                                    v855 = v845;
                                                } else {
                                                    v855 = v852;
                                                }
                                                v845 = v855;
                                                v844 += 1 ;
                                            }
                                            static_array<int,2> v856;
                                            int v858;
                                            v858 = 0;
                                            while (while_method_0(v858)){
                                                v856[v858] = v845;
                                                v858 += 1 ;
                                            }
                                            static_array<int,2> v860;
                                            int v862;
                                            v862 = 0;
                                            while (while_method_0(v862)){
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
                                                v869 = v856[v862];
                                                bool v871;
                                                v871 = v862 == v815;
                                                int v873;
                                                if (v871){
                                                    int v872;
                                                    v872 = v869 + 4;
                                                    v873 = v872;
                                                } else {
                                                    v873 = v869;
                                                }
                                                v860[v862] = v873;
                                                v862 += 1 ;
                                            }
                                            v925 = Union4{Union4_2{v812, false, v814, v842, v860, v843}};
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
                        v933 = Union3{Union3_1{v925}};
                        break;
                    }
                    case 3: { // T_none
                        v933 = Union3{Union3_0{}};
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
        v20 = v933;
    }
    return ;
}
__device__ inline bool while_method_14(int v0){
    bool v1;
    v1 = v0 < 16384;
    return v1;
}
__device__ inline bool while_method_15(int v0){
    bool v1;
    v1 = v0 < 128;
    return v1;
}
__device__ inline bool while_method_16(int v0){
    bool v1;
    v1 = v0 < 8192;
    return v1;
}
__device__ inline bool while_method_17(int v0){
    bool v1;
    v1 = v0 < 2048;
    return v1;
}
__device__ inline bool while_method_18(int v0){
    bool v1;
    v1 = v0 < 24;
    return v1;
}
__device__ inline bool while_method_19(int v0){
    bool v1;
    v1 = v0 < 1024;
    return v1;
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
    while (while_method_13(v16)){
        Union3 v929;
        switch (v16.tag) {
            case 0: { // None
                v929 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v18 = v16.case1.v0;
                Union14 v769;
                switch (v18.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v738 = v18.case0.v0; bool v739 = v18.case0.v1; static_array<Union6,2> v740 = v18.case0.v2; int v741 = v18.case0.v3; static_array<int,2> v742 = v18.case0.v4; int v743 = v18.case0.v5;
                        curandStatePhilox4_32_10_t & v744 = v3.v5;
                        curandStatePhilox4_32_10_t & v745 = v744;
                        unsigned int & v746 = v3.v0;
                        Union6 v747; unsigned int v748;
                        Tuple6 tmp56 = draw_card_20(v745, v746);
                        v747 = tmp56.v0; v748 = tmp56.v1;
                        v3.v0 = v748;
                        Union7 v749;
                        v749 = Union7{Union7_0{v747}};
                        v14.push(v749);
                        v769 = Union14{Union14_0{v738, v739, v740, v741, v742, v743, v747}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v751 = v3.v5;
                        curandStatePhilox4_32_10_t & v752 = v751;
                        unsigned int & v753 = v3.v0;
                        Union6 v754; unsigned int v755;
                        Tuple6 tmp57 = draw_card_20(v752, v753);
                        v754 = tmp57.v0; v755 = tmp57.v1;
                        v3.v0 = v755;
                        curandStatePhilox4_32_10_t & v756 = v3.v5;
                        curandStatePhilox4_32_10_t & v757 = v756;
                        unsigned int & v758 = v3.v0;
                        Union6 v759; unsigned int v760;
                        Tuple6 tmp58 = draw_card_20(v757, v758);
                        v759 = tmp58.v0; v760 = tmp58.v1;
                        v3.v0 = v760;
                        Union7 v761;
                        v761 = Union7{Union7_2{0, v754}};
                        v14.push(v761);
                        Union7 v762;
                        v762 = Union7{Union7_2{1, v759}};
                        v14.push(v762);
                        v769 = Union14{Union14_1{v754, v759}};
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
                        Union1 v726;
                        switch (v80.tag) {
                            case 0: { // Computer
                                static_array_list<Union7,32> & v83 = v3.v2;
                                curandStatePhilox4_32_10_t & v84 = v3.v5;
                                curandStatePhilox4_32_10_t & v85 = v84;
                                float * v86;
                                v86 = reinterpret_cast<float *>(&v1[7864320ull]);
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
                                while (while_method_4(v155)){
                                    float * v157;
                                    v157 = reinterpret_cast<float *>(&v1[7864320ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 32);
                                    int v159;
                                    v159 = 393216 * v155;
                                    float * v160;
                                    v160 = reinterpret_cast<float *>(&v1[4718592ull]);
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
                                    block_map_24(v160, v169);
                                    float * v171;
                                    v171 = reinterpret_cast<float *>(&v0[1048576ull]);
                                    float * v173;
                                    v173 = reinterpret_cast<float *>(&v2[1048576ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 32);
                                    int v175;
                                    v175 = 4096 * v155;
                                    float * v176;
                                    v176 = reinterpret_cast<float *>(&v1[6291456ull]);
                                    block_matmul_25(v176, v171, v175, v160);
                                    block_row_map_26(v157, v159, v176);
                                    int * v178;
                                    v178 = reinterpret_cast<int *>(&v0[1572864ull]);
                                    bool * v180;
                                    v180 = reinterpret_cast<bool *>(&v0[1572880ull]);
                                    float * v182;
                                    v182 = reinterpret_cast<float *>(&v0[1572912ull]);
                                    float * v184;
                                    v184 = reinterpret_cast<float *>(&v0[1573040ull]);
                                    double * v186;
                                    v186 = reinterpret_cast<double *>(&v1[58195968ull]);
                                    double * v188;
                                    v188 = reinterpret_cast<double *>(&v1[61341696ull]);
                                    v155 += 1 ;
                                }
                                __syncthreads();
                                int * v190;
                                v190 = reinterpret_cast<int *>(&v0[1572864ull]);
                                bool * v192;
                                v192 = reinterpret_cast<bool *>(&v0[1572880ull]);
                                float * v194;
                                v194 = reinterpret_cast<float *>(&v0[1572912ull]);
                                float * v196;
                                v196 = reinterpret_cast<float *>(&v0[1573040ull]);
                                int v198;
                                v198 = v190[0];
                                float * v199;
                                v199 = reinterpret_cast<float *>(&v1[7864320ull]);
                                assert("Tensor range check" && 0 <= v198 && v198 < 32);
                                int v201;
                                v201 = 393216 * v198;
                                int v202;
                                v202 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v202 && v202 < 24);
                                int v203;
                                v203 = 16384 * v202;
                                int v204;
                                v204 = v203 + v201;
                                int v205;
                                v205 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v205 && v205 < 256);
                                int v206;
                                v206 = 64 * v205;
                                int v207;
                                v207 = v206 + v204;
                                float * v208;
                                v208 = v199+v207;
                                int v210;
                                v210 = sizeof(float *);
                                unsigned long long v211;
                                v211 = (unsigned long long)v210;
                                unsigned long long v212;
                                v212 = 256ull * v211;
                                unsigned long long v213;
                                v213 = v212 + 16ull;
                                unsigned long long v214;
                                v214 = v213 - 1ull;
                                unsigned long long v215;
                                v215 = v214 % 16ull;
                                unsigned long long v216;
                                v216 = v214 - v215;
                                unsigned long long v217;
                                v217 = v216 + 1024ull;
                                unsigned long long v218;
                                v218 = v217 + 16ull;
                                unsigned long long v219;
                                v219 = v218 - 1ull;
                                unsigned long long v220;
                                v220 = v219 % 16ull;
                                unsigned long long v221;
                                v221 = v219 - v220;
                                unsigned long long v222;
                                v222 = v221 + 1024ull;
                                bool v223;
                                v223 = v222 <= 98304ull;
                                bool v224;
                                v224 = v223 == false;
                                if (v224){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v223);
                                } else {
                                }
                                extern __shared__ unsigned char v226[];
                                bool v227;
                                v227 = v222 <= v222;
                                bool v228;
                                v228 = v227 == false;
                                if (v228){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v227);
                                } else {
                                }
                                float * * v230;
                                v230 = reinterpret_cast<float * *>(&v226[0ull]);
                                float * v232;
                                v232 = reinterpret_cast<float *>(&v226[v216]);
                                int * v234;
                                v234 = reinterpret_cast<int *>(&v226[v221]);
                                int v236;
                                v236 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v236 && v236 < 256);
                                v230[v236] = v208;
                                __syncthreads();
                                bool v237;
                                v237 = 0 <= v236;
                                bool v238;
                                v238 = v237 == false;
                                if (v238){
                                    assert("The index needs to be zero or positive." && v237);
                                } else {
                                }
                                int v240;
                                v240 = v236 % 16;
                                int v241;
                                v241 = v236 / 16;
                                bool v242;
                                v242 = v241 < 16;
                                bool v243;
                                v243 = v242 == false;
                                if (v243){
                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v242);
                                } else {
                                }
                                assert("Tensor range check" && 0 <= v241 && v241 < 16);
                                int v245;
                                v245 = 0;
                                while (while_method_9(v245)){
                                    bool v247;
                                    v247 = 0 <= v241;
                                    bool v248;
                                    v248 = v247 && v242;
                                    bool v249;
                                    v249 = v248 == false;
                                    if (v249){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v248);
                                    } else {
                                    }
                                    bool v251;
                                    v251 = 0 <= v245;
                                    bool v253;
                                    if (v251){
                                        bool v252;
                                        v252 = v245 < 16;
                                        v253 = v252;
                                    } else {
                                        v253 = false;
                                    }
                                    bool v254;
                                    v254 = v253 == false;
                                    if (v254){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v253);
                                    } else {
                                    }
                                    int v256;
                                    v256 = v245 * 16;
                                    int v257;
                                    v257 = v256 + v241;
                                    assert("Tensor range check" && 0 <= v245 && v245 < 16);
                                    int v258;
                                    v258 = 16 * v245;
                                    int v259;
                                    v259 = v258 + v241;
                                    float * v260;
                                    v260 = v230[v259];
                                    int v261;
                                    v261 = blockIdx.x;
                                    int v262;
                                    v262 = v261 * 256;
                                    int v263;
                                    v263 = v262 + v257;
                                    assert("Tensor range check" && 0 <= v240 && v240 < 16);
                                    int v264;
                                    v264 = 4 * v240;
                                    float v265[4];
                                    int v266[4];
                                    int v267;
                                    v267 = 0;
                                    while (while_method_5(v267)){
                                        assert("Tensor range check" && 0 <= v267 && v267 < 1);
                                        int v269;
                                        v269 = 4 * v267;
                                        assert("Tensor range check" && 0 <= v267 && v267 < 1);
                                        int v270;
                                        v270 = 64 * v267;
                                        int v271;
                                        v271 = v270 + v264;
                                        int4* v272;
                                        v272 = reinterpret_cast<int4*>(v260 + v271);
                                        int4* v273;
                                        v273 = reinterpret_cast<int4*>(v265 + v269);
                                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v272) % 16 == 0 && reinterpret_cast<unsigned long long>(v273) % 16 == 0);
                                        *v273 = *v272;
                                        v267 += 1 ;
                                    }
                                    int v274;
                                    v274 = 0;
                                    while (while_method_5(v274)){
                                        int v276;
                                        v276 = 0;
                                        while (while_method_8(v276)){
                                            bool v278;
                                            v278 = 0 <= v276;
                                            bool v280;
                                            if (v278){
                                                bool v279;
                                                v279 = v276 < 4;
                                                v280 = v279;
                                            } else {
                                                v280 = false;
                                            }
                                            bool v281;
                                            v281 = v280 == false;
                                            if (v281){
                                                assert("The indices should be inside the range of the dimension." && v280);
                                            } else {
                                            }
                                            bool v283;
                                            v283 = 0 <= v240;
                                            bool v285;
                                            if (v283){
                                                bool v284;
                                                v284 = v240 < 16;
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
                                            v288 = v240 * 4;
                                            int v289;
                                            v289 = v276 + v288;
                                            bool v290;
                                            v290 = 0 <= v274;
                                            bool v292;
                                            if (v290){
                                                bool v291;
                                                v291 = v274 < 1;
                                                v292 = v291;
                                            } else {
                                                v292 = false;
                                            }
                                            bool v293;
                                            v293 = v292 == false;
                                            if (v293){
                                                assert("The indices should be inside the range of the dimension." && v292);
                                            } else {
                                            }
                                            int v295;
                                            v295 = v274 * 64;
                                            int v296;
                                            v296 = v289 + v295;
                                            assert("Tensor range check" && 0 <= v274 && v274 < 1);
                                            assert("Tensor range check" && 0 <= v276 && v276 < 4);
                                            int v297;
                                            v297 = 4 * v274;
                                            int v298;
                                            v298 = v297 + v276;
                                            v266[v298] = v296;
                                            v276 += 1 ;
                                        }
                                        v274 += 1 ;
                                    }
                                    float v299[4];
                                    float v300;
                                    v300 = 0.0f;
                                    int v301;
                                    v301 = 0;
                                    while (while_method_5(v301)){
                                        assert("Tensor range check" && 0 <= v301 && v301 < 1);
                                        int v303;
                                        v303 = 4 * v301;
                                        assert("Tensor range check" && 0 <= v301 && v301 < 1);
                                        float v304;
                                        v304 = 0.0f;
                                        int v305;
                                        v305 = 0;
                                        while (while_method_8(v305)){
                                            assert("Tensor range check" && 0 <= v305 && v305 < 4);
                                            int v307;
                                            v307 = v305 + v303;
                                            float v308;
                                            v308 = v265[v307];
                                            float v309;
                                            v309 = v304 + v308;
                                            v304 = v309;
                                            v305 += 1 ;
                                        }
                                        auto v310 = cooperative_groups::coalesced_threads();
                                        int v311;
                                        v311 = threadIdx.x;
                                        int v312;
                                        v312 = v311 / 16;
                                        auto v313 = cooperative_groups::labeled_partition(v310,v312);
                                        Closure2 v314{};
                                        float v315;
                                        v315 = cooperative_groups::inclusive_scan(v313, v304, v314);
                                        float v316;
                                        v316 = v313.shfl_up(v315,1);
                                        bool v317;
                                        v317 = v313.thread_rank() == 0;
                                        float v318;
                                        if (v317){
                                            v318 = 0.0f;
                                        } else {
                                            v318 = v316;
                                        }
                                        float v319;
                                        v319 = v313.shfl(v315,v313.num_threads()-1);
                                        float v320;
                                        v320 = v300 + v318;
                                        float v321;
                                        v321 = v320;
                                        int v322;
                                        v322 = 0;
                                        while (while_method_8(v322)){
                                            assert("Tensor range check" && 0 <= v322 && v322 < 4);
                                            int v324;
                                            v324 = v322 + v303;
                                            float v325;
                                            v325 = v265[v324];
                                            float v326;
                                            v326 = v321 + v325;
                                            assert("Tensor range check" && 0 <= v322 && v322 < 4);
                                            v299[v324] = v326;
                                            v321 = v326;
                                            v322 += 1 ;
                                        }
                                        float v327;
                                        v327 = v300 + v319;
                                        v300 = v327;
                                        v301 += 1 ;
                                    }
                                    float v328[4];
                                    bool v329[4];
                                    int v330;
                                    v330 = 0;
                                    while (while_method_5(v330)){
                                        int v332;
                                        v332 = 0;
                                        while (while_method_8(v332)){
                                            assert("Tensor range check" && 0 <= v330 && v330 < 1);
                                            assert("Tensor range check" && 0 <= v332 && v332 < 4);
                                            int v334;
                                            v334 = 4 * v330;
                                            int v335;
                                            v335 = v334 + v332;
                                            float v336;
                                            v336 = v299[v335];
                                            float v337;
                                            v337 = v265[v335];
                                            bool v338;
                                            v338 = v337 > 0.0f;
                                            assert("Tensor range check" && 0 <= v330 && v330 < 1);
                                            assert("Tensor range check" && 0 <= v332 && v332 < 4);
                                            v328[v335] = v336;
                                            v329[v335] = v338;
                                            v332 += 1 ;
                                        }
                                        v330 += 1 ;
                                    }
                                    float v339; bool v340;
                                    Tuple8 tmp59 = Tuple8{-1.0f / 0.0f, false};
                                    v339 = tmp59.v0; v340 = tmp59.v1;
                                    int v341;
                                    v341 = 0;
                                    while (while_method_5(v341)){
                                        int v343;
                                        v343 = 0;
                                        while (while_method_8(v343)){
                                            assert("Tensor range check" && 0 <= v341 && v341 < 1);
                                            assert("Tensor range check" && 0 <= v343 && v343 < 4);
                                            int v345;
                                            v345 = 4 * v341;
                                            int v346;
                                            v346 = v345 + v343;
                                            float v347;
                                            v347 = v328[v346];
                                            bool v348;
                                            v348 = v329[v346];
                                            float v355; bool v356;
                                            if (v340){
                                                if (v348){
                                                    bool v349;
                                                    v349 = v339 >= v347;
                                                    float v350;
                                                    if (v349){
                                                        v350 = v339;
                                                    } else {
                                                        v350 = v347;
                                                    }
                                                    v355 = v350; v356 = true;
                                                } else {
                                                    v355 = v339; v356 = v340;
                                                }
                                            } else {
                                                if (v348){
                                                    v355 = v347; v356 = v348;
                                                } else {
                                                    v355 = v339; v356 = v340;
                                                }
                                            }
                                            v339 = v355;
                                            v340 = v356;
                                            v343 += 1 ;
                                        }
                                        v341 += 1 ;
                                    }
                                    auto v357 = cooperative_groups::coalesced_threads();
                                    int v358;
                                    v358 = threadIdx.x;
                                    int v359;
                                    v359 = v358 / 16;
                                    auto v360 = cooperative_groups::labeled_partition(v357,v359);
                                    Closure3 v361{};
                                    float v362; bool v363;
                                    Tuple8 tmp60 = cooperative_groups::reduce(v360, Tuple8{v339, v340}, v361);
                                    v362 = tmp60.v0; v363 = tmp60.v1;
                                    bool v364;
                                    v364 = v363 == false;
                                    if (v364){
                                        int v365;
                                        v365 = threadIdx.x;
                                        int v366;
                                        v366 = blockIdx.x;
                                        int v367;
                                        v367 = v366 * 256;
                                        int v368;
                                        v368 = v365 + v367;
                                        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v369 = console_lock;
                                        auto v370 = cooperative_groups::coalesced_threads();
                                        v369.acquire();
                                        int v371;
                                        v371 = 0;
                                        printf("{%s = %d; %s = %c","tid", v368, "x'", '[');
                                        int v372;
                                        v372 = 0;
                                        while (while_method_5(v372)){
                                            int v374;
                                            v374 = v371;
                                            bool v375;
                                            v375 = v374 >= 100;
                                            if (v375){
                                                printf("%s"," ...");
                                                break;
                                            } else {
                                            }
                                            bool v376;
                                            v376 = v372 == 0;
                                            bool v377;
                                            v377 = v376 != true;
                                            if (v377){
                                                printf("%s","; ");
                                            } else {
                                            }
                                            printf("%c",'[');
                                            int v378;
                                            v378 = 0;
                                            while (while_method_8(v378)){
                                                int v380;
                                                v380 = v371;
                                                bool v381;
                                                v381 = v380 >= 100;
                                                if (v381){
                                                    printf("%s"," ...");
                                                    break;
                                                } else {
                                                }
                                                bool v382;
                                                v382 = v378 == 0;
                                                bool v383;
                                                v383 = v382 != true;
                                                if (v383){
                                                    printf("%s","; ");
                                                } else {
                                                }
                                                int v384;
                                                v384 = v371 + 1;
                                                v371 = v384;
                                                int v385;
                                                v385 = v372 * 4;
                                                int v386;
                                                v386 = v385 + v378;
                                                float v387;
                                                v387 = v328[v386];
                                                bool v388;
                                                v388 = v329[v386];
                                                const char * v391;
                                                if (v388){
                                                    const char * v389;
                                                    v389 = "true";
                                                    v391 = v389;
                                                } else {
                                                    const char * v390;
                                                    v390 = "false";
                                                    v391 = v390;
                                                }
                                                printf("%f, %s",v387, v391);
                                                v378 += 1 ;
                                            }
                                            printf("%c",']');
                                            v372 += 1 ;
                                        }
                                        printf("%c",']');
                                        printf("}\n");
                                        v369.release();
                                        v370.sync() ;
                                    } else {
                                    }
                                    if (v364){
                                        assert("The local reduce must be true." && v363);
                                    } else {
                                    }
                                    float v427[4];
                                    int v428[4];
                                    int v429;
                                    v429 = 0;
                                    while (while_method_5(v429)){
                                        int v431;
                                        v431 = 0;
                                        while (while_method_8(v431)){
                                            assert("Tensor range check" && 0 <= v429 && v429 < 1);
                                            assert("Tensor range check" && 0 <= v431 && v431 < 4);
                                            int v433;
                                            v433 = 4 * v429;
                                            int v434;
                                            v434 = v433 + v431;
                                            int v435;
                                            v435 = v266[v434];
                                            float v436;
                                            v436 = curand_uniform(&v85);
                                            assert("Tensor range check" && 0 <= v429 && v429 < 1);
                                            assert("Tensor range check" && 0 <= v431 && v431 < 4);
                                            v427[v434] = v436;
                                            v428[v434] = v435;
                                            v431 += 1 ;
                                        }
                                        v429 += 1 ;
                                    }
                                    float v437; int v438;
                                    Tuple9 tmp61 = Tuple9{0.0f, 2147483647};
                                    v437 = tmp61.v0; v438 = tmp61.v1;
                                    int v439;
                                    v439 = 0;
                                    while (while_method_5(v439)){
                                        int v441;
                                        v441 = 0;
                                        while (while_method_8(v441)){
                                            assert("Tensor range check" && 0 <= v439 && v439 < 1);
                                            assert("Tensor range check" && 0 <= v441 && v441 < 4);
                                            int v443;
                                            v443 = 4 * v439;
                                            int v444;
                                            v444 = v443 + v441;
                                            float v445;
                                            v445 = v427[v444];
                                            int v446;
                                            v446 = v428[v444];
                                            bool v447;
                                            v447 = v438 < v446;
                                            float v448; int v449;
                                            if (v447){
                                                v448 = v437; v449 = v438;
                                            } else {
                                                v448 = v445; v449 = v446;
                                            }
                                            v437 = v448;
                                            v438 = v449;
                                            v441 += 1 ;
                                        }
                                        v439 += 1 ;
                                    }
                                    auto v450 = cooperative_groups::coalesced_threads();
                                    int v451;
                                    v451 = threadIdx.x;
                                    int v452;
                                    v452 = v451 / 16;
                                    auto v453 = cooperative_groups::labeled_partition(v450,v452);
                                    Closure4 v454{};
                                    float v455; int v456;
                                    Tuple9 tmp62 = cooperative_groups::reduce(v453, Tuple9{v437, v438}, v454);
                                    v455 = tmp62.v0; v456 = tmp62.v1;
                                    float v457;
                                    v457 = v362 * v455;
                                    int v458[4];
                                    bool v459[4];
                                    int v460;
                                    v460 = 0;
                                    while (while_method_5(v460)){
                                        int v462;
                                        v462 = 0;
                                        while (while_method_8(v462)){
                                            assert("Tensor range check" && 0 <= v460 && v460 < 1);
                                            assert("Tensor range check" && 0 <= v462 && v462 < 4);
                                            int v464;
                                            v464 = 4 * v460;
                                            int v465;
                                            v465 = v464 + v462;
                                            float v466;
                                            v466 = v328[v465];
                                            bool v467;
                                            v467 = v329[v465];
                                            int v468;
                                            v468 = v266[v465];
                                            int v471; bool v472;
                                            if (v467){
                                                float v469;
                                                v469 = v466 - v457;
                                                bool v470;
                                                v470 = v469 >= 0.0f;
                                                v471 = v468; v472 = v470;
                                            } else {
                                                v471 = 2147483647; v472 = false;
                                            }
                                            assert("Tensor range check" && 0 <= v460 && v460 < 1);
                                            assert("Tensor range check" && 0 <= v462 && v462 < 4);
                                            v458[v465] = v471;
                                            v459[v465] = v472;
                                            v462 += 1 ;
                                        }
                                        v460 += 1 ;
                                    }
                                    int v473; bool v474;
                                    Tuple10 tmp63 = Tuple10{2147483647, false};
                                    v473 = tmp63.v0; v474 = tmp63.v1;
                                    int v475;
                                    v475 = 0;
                                    while (while_method_5(v475)){
                                        int v477;
                                        v477 = 0;
                                        while (while_method_8(v477)){
                                            assert("Tensor range check" && 0 <= v475 && v475 < 1);
                                            assert("Tensor range check" && 0 <= v477 && v477 < 4);
                                            int v479;
                                            v479 = 4 * v475;
                                            int v480;
                                            v480 = v479 + v477;
                                            int v481;
                                            v481 = v458[v480];
                                            bool v482;
                                            v482 = v459[v480];
                                            int v489; bool v490;
                                            if (v474){
                                                if (v482){
                                                    bool v483;
                                                    v483 = v473 < v481;
                                                    int v484;
                                                    if (v483){
                                                        v484 = v473;
                                                    } else {
                                                        v484 = v481;
                                                    }
                                                    v489 = v484; v490 = true;
                                                } else {
                                                    v489 = v473; v490 = v474;
                                                }
                                            } else {
                                                if (v482){
                                                    v489 = v481; v490 = v482;
                                                } else {
                                                    v489 = v473; v490 = v474;
                                                }
                                            }
                                            v473 = v489;
                                            v474 = v490;
                                            v477 += 1 ;
                                        }
                                        v475 += 1 ;
                                    }
                                    auto v491 = cooperative_groups::coalesced_threads();
                                    int v492;
                                    v492 = threadIdx.x;
                                    int v493;
                                    v493 = v492 / 16;
                                    auto v494 = cooperative_groups::labeled_partition(v491,v493);
                                    Closure5 v495{};
                                    int v496; bool v497;
                                    Tuple10 tmp64 = cooperative_groups::reduce(v494, Tuple10{v473, v474}, v495);
                                    v496 = tmp64.v0; v497 = tmp64.v1;
                                    bool v498;
                                    v498 = v497 == false;
                                    if (v498){
                                        int v499;
                                        v499 = threadIdx.x;
                                        int v500;
                                        v500 = blockIdx.x;
                                        int v501;
                                        v501 = v500 * 256;
                                        int v502;
                                        v502 = v499 + v501;
                                        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v503 = console_lock;
                                        auto v504 = cooperative_groups::coalesced_threads();
                                        v503.acquire();
                                        int v505;
                                        v505 = 0;
                                        printf("{%s = %d; %s = %c","tid", v502, "x'", '[');
                                        int v506;
                                        v506 = 0;
                                        while (while_method_5(v506)){
                                            int v508;
                                            v508 = v505;
                                            bool v509;
                                            v509 = v508 >= 100;
                                            if (v509){
                                                printf("%s"," ...");
                                                break;
                                            } else {
                                            }
                                            bool v510;
                                            v510 = v506 == 0;
                                            bool v511;
                                            v511 = v510 != true;
                                            if (v511){
                                                printf("%s","; ");
                                            } else {
                                            }
                                            printf("%c",'[');
                                            int v512;
                                            v512 = 0;
                                            while (while_method_8(v512)){
                                                int v514;
                                                v514 = v505;
                                                bool v515;
                                                v515 = v514 >= 100;
                                                if (v515){
                                                    printf("%s"," ...");
                                                    break;
                                                } else {
                                                }
                                                bool v516;
                                                v516 = v512 == 0;
                                                bool v517;
                                                v517 = v516 != true;
                                                if (v517){
                                                    printf("%s","; ");
                                                } else {
                                                }
                                                int v518;
                                                v518 = v505 + 1;
                                                v505 = v518;
                                                int v519;
                                                v519 = v506 * 4;
                                                int v520;
                                                v520 = v519 + v512;
                                                int v521;
                                                v521 = v458[v520];
                                                bool v522;
                                                v522 = v459[v520];
                                                const char * v525;
                                                if (v522){
                                                    const char * v523;
                                                    v523 = "true";
                                                    v525 = v523;
                                                } else {
                                                    const char * v524;
                                                    v524 = "false";
                                                    v525 = v524;
                                                }
                                                printf("%d, %s",v521, v525);
                                                v512 += 1 ;
                                            }
                                            printf("%c",']');
                                            v506 += 1 ;
                                        }
                                        printf("%c",']');
                                        printf("}\n");
                                        v503.release();
                                        v504.sync() ;
                                    } else {
                                    }
                                    if (v498){
                                        assert("The local reduce must be true." && v497);
                                    } else {
                                    }
                                    float v561; int v562;
                                    Tuple9 tmp65 = Tuple9{0.0f, 2147483647};
                                    v561 = tmp65.v0; v562 = tmp65.v1;
                                    int v563;
                                    v563 = 0;
                                    while (while_method_5(v563)){
                                        int v565;
                                        v565 = 0;
                                        while (while_method_8(v565)){
                                            assert("Tensor range check" && 0 <= v563 && v563 < 1);
                                            assert("Tensor range check" && 0 <= v565 && v565 < 4);
                                            int v567;
                                            v567 = 4 * v563;
                                            int v568;
                                            v568 = v567 + v565;
                                            float v569;
                                            v569 = v265[v568];
                                            int v570;
                                            v570 = v266[v568];
                                            bool v571;
                                            v571 = v562 == v496;
                                            float v575; int v576;
                                            if (v571){
                                                v575 = v561; v576 = v562;
                                            } else {
                                                bool v572;
                                                v572 = v570 == v496;
                                                if (v572){
                                                    v575 = v569; v576 = v570;
                                                } else {
                                                    v575 = v561; v576 = v562;
                                                }
                                            }
                                            v561 = v575;
                                            v562 = v576;
                                            v565 += 1 ;
                                        }
                                        v563 += 1 ;
                                    }
                                    auto v577 = cooperative_groups::coalesced_threads();
                                    int v578;
                                    v578 = threadIdx.x;
                                    int v579;
                                    v579 = v578 / 16;
                                    auto v580 = cooperative_groups::labeled_partition(v577,v579);
                                    Closure6 v581{v496};
                                    float v582; int v583;
                                    Tuple9 tmp66 = cooperative_groups::reduce(v580, Tuple9{v561, v562}, v581);
                                    v582 = tmp66.v0; v583 = tmp66.v1;
                                    bool v584;
                                    v584 = v583 == 2147483647;
                                    bool v585;
                                    v585 = v584 != true;
                                    bool v586;
                                    v586 = v585 == false;
                                    if (v586){
                                        assert("Expected a valid action id in get_prob." && v585);
                                    } else {
                                    }
                                    int v588;
                                    v588 = 0;
                                    while (while_method_5(v588)){
                                        assert("Tensor range check" && 0 <= v588 && v588 < 1);
                                        assert("Tensor range check" && 0 <= v588 && v588 < 1);
                                        v588 += 1 ;
                                    }
                                    assert("Tensor range check" && 0 <= v257 && v257 < 256);
                                    v232[v257] = v582;
                                    v234[v257] = v496;
                                    v245 += 1 ;
                                }
                                __syncthreads();
                                assert("Tensor range check" && 0 <= v236 && v236 < 256);
                                float v590;
                                v590 = v232[v236];
                                int v591;
                                v591 = v234[v236];
                                __syncthreads();
                                extern __shared__ unsigned char v592[];
                                float * v593;
                                v593 = reinterpret_cast<float *>(&v592[0ull]);
                                int * v595;
                                v595 = reinterpret_cast<int *>(&v592[16ull]);
                                int v597;
                                v597 = threadIdx.x;
                                bool v598;
                                v598 = v597 == 0;
                                if (v598){
                                    v593[0] = v590;
                                    v595[0] = v591;
                                } else {
                                }
                                __syncthreads();
                                float v599;
                                v599 = v593[0];
                                int v600;
                                v600 = v595[0];
                                __syncthreads();
                                double * v601;
                                v601 = reinterpret_cast<double *>(&v1[58195968ull]);
                                double * v603;
                                v603 = reinterpret_cast<double *>(&v1[61341696ull]);
                                int v605;
                                v605 = threadIdx.x;
                                int v606;
                                v606 = blockIdx.x;
                                int v607;
                                v607 = v606 * 256;
                                int v608;
                                v608 = v605 + v607;
                                int v609;
                                v609 = 0;
                                while (while_method_4(v609)){
                                    float * v611;
                                    v611 = reinterpret_cast<float *>(&v1[7864320ull]);
                                    int v613;
                                    v613 = blockIdx.x;
                                    int v614;
                                    v614 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v609 && v609 < 32);
                                    assert("Tensor range check" && 0 <= v613 && v613 < 24);
                                    assert("Tensor range check" && 0 <= v614 && v614 < 256);
                                    assert("Tensor range check" && 0 <= v600 && v600 < 64);
                                    int v615;
                                    v615 = 64 * v614;
                                    int v616;
                                    v616 = v615 + v600;
                                    int v617;
                                    v617 = 16384 * v613;
                                    int v618;
                                    v618 = v617 + v616;
                                    int v619;
                                    v619 = 393216 * v609;
                                    int v620;
                                    v620 = v619 + v618;
                                    float v621;
                                    v621 = v611[v620];
                                    double v622;
                                    v622 = (double)v599;
                                    double v623;
                                    v623 = log(v622);
                                    double v624;
                                    v624 = (double)v621;
                                    double v625;
                                    v625 = log(v624);
                                    assert("Tensor range check" && 0 <= v609 && v609 < 32);
                                    assert("Tensor range check" && 0 <= v608 && v608 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    int v626;
                                    v626 = 2 * v608;
                                    int v627;
                                    v627 = v626 + v71;
                                    int v628;
                                    v628 = 12288 * v609;
                                    int v629;
                                    v629 = v628 + v627;
                                    double v630;
                                    v630 = v601[v629];
                                    double v631;
                                    v631 = v603[v629];
                                    double v632;
                                    v632 = v625 + v630;
                                    double v633;
                                    v633 = v623 + v631;
                                    bool v634;
                                    v634 = isnan(v633);
                                    bool v635;
                                    v635 = v634 == false;
                                    bool v636;
                                    v636 = v635 == false;
                                    if (v636){
                                        assert("The sampling log probability shouldn't be nan." && v635);
                                    } else {
                                    }
                                    bool v638;
                                    v638 = isnan(v632);
                                    bool v639;
                                    v639 = v638 == false;
                                    bool v640;
                                    v640 = v639 == false;
                                    if (v640){
                                        assert("The policy log probability shouldn't be nan." && v639);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v609 && v609 < 32);
                                    assert("Tensor range check" && 0 <= v608 && v608 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    v601[v629] = v632;
                                    v603[v629] = v633;
                                    v609 += 1 ;
                                }
                                bool v642;
                                v642 = 0 == v600;
                                Union12 v651;
                                if (v642){
                                    v651 = Union12{Union12_1{}};
                                } else {
                                    bool v644;
                                    v644 = 1 == v600;
                                    if (v644){
                                        v651 = Union12{Union12_0{}};
                                    } else {
                                        bool v646;
                                        v646 = 2 == v600;
                                        if (v646){
                                            v651 = Union12{Union12_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v651.tag) {
                                    case 0: { // AA_Call
                                        v726 = Union1{Union1_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v652;
                                        v652 = v72[0];
                                        int v654; int v655;
                                        Tuple7 tmp67 = Tuple7{1, v652};
                                        v654 = tmp67.v0; v655 = tmp67.v1;
                                        while (while_method_0(v654)){
                                            bool v657;
                                            v657 = 0 <= v654;
                                            bool v659;
                                            if (v657){
                                                bool v658;
                                                v658 = v654 < 2;
                                                v659 = v658;
                                            } else {
                                                v659 = false;
                                            }
                                            bool v660;
                                            v660 = v659 == false;
                                            if (v660){
                                                assert("Index must be in range." && v659);
                                            } else {
                                            }
                                            int v662;
                                            v662 = v72[v654];
                                            bool v664;
                                            v664 = v655 >= v662;
                                            int v665;
                                            if (v664){
                                                v665 = v655;
                                            } else {
                                                v665 = v662;
                                            }
                                            v655 = v665;
                                            v654 += 1 ;
                                        }
                                        bool v667;
                                        if (v75){
                                            bool v666;
                                            v666 = v71 < 2;
                                            v667 = v666;
                                        } else {
                                            v667 = false;
                                        }
                                        bool v668;
                                        v668 = v667 == false;
                                        if (v668){
                                            assert("Index must be in range." && v667);
                                        } else {
                                        }
                                        int v670;
                                        v670 = v72[v71];
                                        bool v672;
                                        v672 = v670 == v655;
                                        if (v672){
                                            v726 = Union1{Union1_0{}};
                                        } else {
                                            v726 = Union1{Union1_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v677;
                                        v677 = v73 > 0;
                                        if (v677){
                                            v726 = Union1{Union1_2{}};
                                        } else {
                                            v726 = Union1{Union1_0{}};
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
                                curandStatePhilox4_32_10_t & v684 = v3.v5;
                                curandStatePhilox4_32_10_t & v685 = v684;
                                static_array_list<Union1,3> v686;
                                v686 = static_array_list<Union1,3>{};
                                v686.unsafe_set_length(1);
                                Union1 v688;
                                v688 = Union1{Union1_0{}};
                                v686[0] = v688;
                                int v690;
                                v690 = v72[0];
                                int v692;
                                v692 = v72[1];
                                bool v694;
                                v694 = v690 == v692;
                                bool v695;
                                v695 = v694 != true;
                                if (v695){
                                    Union1 v696;
                                    v696 = Union1{Union1_1{}};
                                    v686.push(v696);
                                } else {
                                }
                                bool v697;
                                v697 = v73 > 0;
                                if (v697){
                                    Union1 v698;
                                    v698 = Union1{Union1_2{}};
                                    v686.push(v698);
                                } else {
                                }
                                int v699;
                                v699 = v686.length;
                                int v700;
                                v700 = v699 - 1;
                                int v701;
                                v701 = 0;
                                while (while_method_1(v700, v701)){
                                    int v703;
                                    v703 = v686.length;
                                    int v704;
                                    v704 = int_range_22(v703, v701, v685);
                                    Union1 v705;
                                    v705 = v686[v701];
                                    Union1 v707;
                                    v707 = v686[v704];
                                    v686[v701] = v707;
                                    v686[v704] = v705;
                                    v701 += 1 ;
                                }
                                Union1 v709;
                                v709 = v686.pop();
                                int v710;
                                v710 = sizeof(Union1);
                                unsigned long long v711;
                                v711 = (unsigned long long)v710;
                                bool v712;
                                v712 = v711 <= 98304ull;
                                bool v713;
                                v713 = v712 == false;
                                if (v713){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v712);
                                } else {
                                }
                                extern __shared__ unsigned char v715[];
                                bool v716;
                                v716 = v711 <= v711;
                                bool v717;
                                v717 = v716 == false;
                                if (v717){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v716);
                                } else {
                                }
                                Union1 * v719;
                                v719 = reinterpret_cast<Union1 *>(&v715[0ull]);
                                int v721;
                                v721 = threadIdx.x;
                                bool v722;
                                v722 = v721 == 0;
                                if (v722){
                                    v719[0] = v709;
                                } else {
                                }
                                __syncthreads();
                                Union1 v723;
                                v723 = v719[0];
                                __syncthreads();
                                v726 = v723;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union7 v727;
                        v727 = Union7{Union7_1{v71, v726}};
                        v14.push(v727);
                        v769 = Union14{Union14_2{v68, v69, v70, v71, v72, v73, v726}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v729 = v18.case3.v0; bool v730 = v18.case3.v1; static_array<Union6,2> v731 = v18.case3.v2; int v732 = v18.case3.v3; static_array<int,2> v733 = v18.case3.v4; int v734 = v18.case3.v5; Union1 v735 = v18.case3.v6;
                        Union7 v736;
                        v736 = Union7{Union7_1{v732, v735}};
                        v14.push(v736);
                        v769 = Union14{Union14_2{v729, v730, v731, v732, v733, v734, v735}};
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
                        v52 = compare_hands_27(v39, v40, v41, v42, v43, v44);
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
                        v769 = Union14{Union14_3{}};
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
                        v769 = Union14{Union14_3{}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                switch (v769.tag) {
                    case 0: { // T_game_chance_community_card
                        Union5 v771 = v769.case0.v0; bool v772 = v769.case0.v1; static_array<Union6,2> v773 = v769.case0.v2; int v774 = v769.case0.v3; static_array<int,2> v775 = v769.case0.v4; int v776 = v769.case0.v5; Union6 v777 = v769.case0.v6;
                        int v778;
                        v778 = 2;
                        int v779; int v780;
                        Tuple7 tmp68 = Tuple7{0, 0};
                        v779 = tmp68.v0; v780 = tmp68.v1;
                        while (while_method_0(v779)){
                            bool v782;
                            v782 = 0 <= v779;
                            bool v784;
                            if (v782){
                                bool v783;
                                v783 = v779 < 2;
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
                            v787 = v775[v779];
                            bool v789;
                            v789 = v780 >= v787;
                            int v790;
                            if (v789){
                                v790 = v780;
                            } else {
                                v790 = v787;
                            }
                            v780 = v790;
                            v779 += 1 ;
                        }
                        static_array<int,2> v791;
                        int v793;
                        v793 = 0;
                        while (while_method_0(v793)){
                            v791[v793] = v780;
                            v793 += 1 ;
                        }
                        Union5 v795;
                        v795 = Union5{Union5_1{v777}};
                        Union4 v796;
                        v796 = Union4{Union4_2{v795, true, v773, 0, v791, v778}};
                        v929 = Union3{Union3_1{v796}};
                        break;
                    }
                    case 1: { // T_game_chance_init
                        Union6 v798 = v769.case1.v0; Union6 v799 = v769.case1.v1;
                        int v800;
                        v800 = 2;
                        static_array<int,2> v801;
                        v801[0] = 1;
                        v801[1] = 1;
                        static_array<Union6,2> v803;
                        v803[0] = v798;
                        v803[1] = v799;
                        Union5 v805;
                        v805 = Union5{Union5_0{}};
                        Union4 v806;
                        v806 = Union4{Union4_2{v805, true, v803, 0, v801, v800}};
                        v929 = Union3{Union3_1{v806}};
                        break;
                    }
                    case 2: { // T_game_round
                        Union5 v808 = v769.case2.v0; bool v809 = v769.case2.v1; static_array<Union6,2> v810 = v769.case2.v2; int v811 = v769.case2.v3; static_array<int,2> v812 = v769.case2.v4; int v813 = v769.case2.v5; Union1 v814 = v769.case2.v6;
                        Union4 v921;
                        switch (v808.tag) {
                            case 0: { // None
                                switch (v814.tag) {
                                    case 0: { // Call
                                        if (v809){
                                            int v877;
                                            v877 = v811 ^ 1;
                                            v921 = Union4{Union4_2{v808, false, v810, v877, v812, v813}};
                                        } else {
                                            v921 = Union4{Union4_0{v808, v809, v810, v811, v812, v813}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v921 = Union4{Union4_5{v808, v809, v810, v811, v812, v813}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v881;
                                        v881 = v813 > 0;
                                        if (v881){
                                            int v882;
                                            v882 = v811 ^ 1;
                                            int v883;
                                            v883 = -1 + v813;
                                            int v884; int v885;
                                            Tuple7 tmp69 = Tuple7{0, 0};
                                            v884 = tmp69.v0; v885 = tmp69.v1;
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
                                                v892 = v812[v884];
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
                                            static_array<int,2> v896;
                                            int v898;
                                            v898 = 0;
                                            while (while_method_0(v898)){
                                                v896[v898] = v885;
                                                v898 += 1 ;
                                            }
                                            static_array<int,2> v900;
                                            int v902;
                                            v902 = 0;
                                            while (while_method_0(v902)){
                                                bool v904;
                                                v904 = 0 <= v902;
                                                bool v906;
                                                if (v904){
                                                    bool v905;
                                                    v905 = v902 < 2;
                                                    v906 = v905;
                                                } else {
                                                    v906 = false;
                                                }
                                                bool v907;
                                                v907 = v906 == false;
                                                if (v907){
                                                    assert("Index must be in range." && v906);
                                                } else {
                                                }
                                                int v909;
                                                v909 = v896[v902];
                                                bool v911;
                                                v911 = v902 == v811;
                                                int v913;
                                                if (v911){
                                                    int v912;
                                                    v912 = v909 + 2;
                                                    v913 = v912;
                                                } else {
                                                    v913 = v909;
                                                }
                                                v900[v902] = v913;
                                                v902 += 1 ;
                                            }
                                            v921 = Union4{Union4_2{v808, false, v810, v882, v900, v883}};
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
                                Union6 v815 = v808.case1.v0;
                                switch (v814.tag) {
                                    case 0: { // Call
                                        if (v809){
                                            int v817;
                                            v817 = v811 ^ 1;
                                            v921 = Union4{Union4_2{v808, false, v810, v817, v812, v813}};
                                        } else {
                                            int v819; int v820;
                                            Tuple7 tmp70 = Tuple7{0, 0};
                                            v819 = tmp70.v0; v820 = tmp70.v1;
                                            while (while_method_0(v819)){
                                                bool v822;
                                                v822 = 0 <= v819;
                                                bool v824;
                                                if (v822){
                                                    bool v823;
                                                    v823 = v819 < 2;
                                                    v824 = v823;
                                                } else {
                                                    v824 = false;
                                                }
                                                bool v825;
                                                v825 = v824 == false;
                                                if (v825){
                                                    assert("Index must be in range." && v824);
                                                } else {
                                                }
                                                int v827;
                                                v827 = v812[v819];
                                                bool v829;
                                                v829 = v820 >= v827;
                                                int v830;
                                                if (v829){
                                                    v830 = v820;
                                                } else {
                                                    v830 = v827;
                                                }
                                                v820 = v830;
                                                v819 += 1 ;
                                            }
                                            static_array<int,2> v831;
                                            int v833;
                                            v833 = 0;
                                            while (while_method_0(v833)){
                                                v831[v833] = v820;
                                                v833 += 1 ;
                                            }
                                            v921 = Union4{Union4_4{v808, v809, v810, v811, v831, v813}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v921 = Union4{Union4_5{v808, v809, v810, v811, v812, v813}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v837;
                                        v837 = v813 > 0;
                                        if (v837){
                                            int v838;
                                            v838 = v811 ^ 1;
                                            int v839;
                                            v839 = -1 + v813;
                                            int v840; int v841;
                                            Tuple7 tmp71 = Tuple7{0, 0};
                                            v840 = tmp71.v0; v841 = tmp71.v1;
                                            while (while_method_0(v840)){
                                                bool v843;
                                                v843 = 0 <= v840;
                                                bool v845;
                                                if (v843){
                                                    bool v844;
                                                    v844 = v840 < 2;
                                                    v845 = v844;
                                                } else {
                                                    v845 = false;
                                                }
                                                bool v846;
                                                v846 = v845 == false;
                                                if (v846){
                                                    assert("Index must be in range." && v845);
                                                } else {
                                                }
                                                int v848;
                                                v848 = v812[v840];
                                                bool v850;
                                                v850 = v841 >= v848;
                                                int v851;
                                                if (v850){
                                                    v851 = v841;
                                                } else {
                                                    v851 = v848;
                                                }
                                                v841 = v851;
                                                v840 += 1 ;
                                            }
                                            static_array<int,2> v852;
                                            int v854;
                                            v854 = 0;
                                            while (while_method_0(v854)){
                                                v852[v854] = v841;
                                                v854 += 1 ;
                                            }
                                            static_array<int,2> v856;
                                            int v858;
                                            v858 = 0;
                                            while (while_method_0(v858)){
                                                bool v860;
                                                v860 = 0 <= v858;
                                                bool v862;
                                                if (v860){
                                                    bool v861;
                                                    v861 = v858 < 2;
                                                    v862 = v861;
                                                } else {
                                                    v862 = false;
                                                }
                                                bool v863;
                                                v863 = v862 == false;
                                                if (v863){
                                                    assert("Index must be in range." && v862);
                                                } else {
                                                }
                                                int v865;
                                                v865 = v852[v858];
                                                bool v867;
                                                v867 = v858 == v811;
                                                int v869;
                                                if (v867){
                                                    int v868;
                                                    v868 = v865 + 4;
                                                    v869 = v868;
                                                } else {
                                                    v869 = v865;
                                                }
                                                v856[v858] = v869;
                                                v858 += 1 ;
                                            }
                                            v921 = Union4{Union4_2{v808, false, v810, v838, v856, v839}};
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
                        v929 = Union3{Union3_1{v921}};
                        break;
                    }
                    case 3: { // T_none
                        v929 = Union3{Union3_0{}};
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
        v16 = v929;
    }
    return ;
}
__device__ void method_50(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut1 & v3, Union4 v4){
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
    while (while_method_13(v16)){
        Union3 v937;
        switch (v16.tag) {
            case 0: { // None
                v937 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v18 = v16.case1.v0;
                Union14 v777;
                switch (v18.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v746 = v18.case0.v0; bool v747 = v18.case0.v1; static_array<Union6,2> v748 = v18.case0.v2; int v749 = v18.case0.v3; static_array<int,2> v750 = v18.case0.v4; int v751 = v18.case0.v5;
                        curandStatePhilox4_32_10_t & v752 = v3.v5;
                        curandStatePhilox4_32_10_t & v753 = v752;
                        unsigned int & v754 = v3.v0;
                        Union6 v755; unsigned int v756;
                        Tuple6 tmp74 = draw_card_20(v753, v754);
                        v755 = tmp74.v0; v756 = tmp74.v1;
                        v3.v0 = v756;
                        Union7 v757;
                        v757 = Union7{Union7_0{v755}};
                        v14.push(v757);
                        v777 = Union14{Union14_0{v746, v747, v748, v749, v750, v751, v755}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v759 = v3.v5;
                        curandStatePhilox4_32_10_t & v760 = v759;
                        unsigned int & v761 = v3.v0;
                        Union6 v762; unsigned int v763;
                        Tuple6 tmp75 = draw_card_20(v760, v761);
                        v762 = tmp75.v0; v763 = tmp75.v1;
                        v3.v0 = v763;
                        curandStatePhilox4_32_10_t & v764 = v3.v5;
                        curandStatePhilox4_32_10_t & v765 = v764;
                        unsigned int & v766 = v3.v0;
                        Union6 v767; unsigned int v768;
                        Tuple6 tmp76 = draw_card_20(v765, v766);
                        v767 = tmp76.v0; v768 = tmp76.v1;
                        v3.v0 = v768;
                        Union7 v769;
                        v769 = Union7{Union7_2{0, v762}};
                        v14.push(v769);
                        Union7 v770;
                        v770 = Union7{Union7_2{1, v767}};
                        v14.push(v770);
                        v777 = Union14{Union14_1{v762, v767}};
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
                        Union1 v734;
                        switch (v80.tag) {
                            case 0: { // Computer
                                static_array_list<Union7,32> & v83 = v3.v2;
                                curandStatePhilox4_32_10_t & v84 = v3.v5;
                                curandStatePhilox4_32_10_t & v85 = v84;
                                float * v86;
                                v86 = reinterpret_cast<float *>(&v1[7864320ull]);
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
                                while (while_method_4(v155)){
                                    float * v157;
                                    v157 = reinterpret_cast<float *>(&v1[7864320ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 32);
                                    int v159;
                                    v159 = 393216 * v155;
                                    float * v160;
                                    v160 = reinterpret_cast<float *>(&v1[4718592ull]);
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
                                    block_map_24(v160, v169);
                                    float * v171;
                                    v171 = reinterpret_cast<float *>(&v0[1048576ull]);
                                    float * v173;
                                    v173 = reinterpret_cast<float *>(&v2[1048576ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 32);
                                    int v175;
                                    v175 = 4096 * v155;
                                    float * v176;
                                    v176 = reinterpret_cast<float *>(&v1[6291456ull]);
                                    block_matmul_25(v176, v171, v175, v160);
                                    block_row_map_26(v157, v159, v176);
                                    int * v178;
                                    v178 = reinterpret_cast<int *>(&v0[1572864ull]);
                                    bool * v180;
                                    v180 = reinterpret_cast<bool *>(&v0[1572880ull]);
                                    float * v182;
                                    v182 = reinterpret_cast<float *>(&v0[1572912ull]);
                                    float * v184;
                                    v184 = reinterpret_cast<float *>(&v0[1573040ull]);
                                    double * v186;
                                    v186 = reinterpret_cast<double *>(&v1[58195968ull]);
                                    double * v188;
                                    v188 = reinterpret_cast<double *>(&v1[61341696ull]);
                                    v155 += 1 ;
                                }
                                __syncthreads();
                                int * v190;
                                v190 = reinterpret_cast<int *>(&v0[1572864ull]);
                                bool * v192;
                                v192 = reinterpret_cast<bool *>(&v0[1572880ull]);
                                float * v194;
                                v194 = reinterpret_cast<float *>(&v0[1572912ull]);
                                float * v196;
                                v196 = reinterpret_cast<float *>(&v0[1573040ull]);
                                int v198;
                                v198 = 0;
                                int v199;
                                v199 = 32;
                                int v200;
                                v200 = int_range_22(v199, v198, v85);
                                extern __shared__ unsigned char v201[];
                                int * v202;
                                v202 = reinterpret_cast<int *>(&v201[0ull]);
                                int v204;
                                v204 = threadIdx.x;
                                bool v205;
                                v205 = v204 == 0;
                                if (v205){
                                    v202[0] = v200;
                                } else {
                                }
                                __syncthreads();
                                int v206;
                                v206 = v202[0];
                                __syncthreads();
                                float * v207;
                                v207 = reinterpret_cast<float *>(&v1[7864320ull]);
                                assert("Tensor range check" && 0 <= v206 && v206 < 32);
                                int v209;
                                v209 = 393216 * v206;
                                int v210;
                                v210 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v210 && v210 < 24);
                                int v211;
                                v211 = 16384 * v210;
                                int v212;
                                v212 = v211 + v209;
                                int v213;
                                v213 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v213 && v213 < 256);
                                int v214;
                                v214 = 64 * v213;
                                int v215;
                                v215 = v214 + v212;
                                float * v216;
                                v216 = v207+v215;
                                int v218;
                                v218 = sizeof(float *);
                                unsigned long long v219;
                                v219 = (unsigned long long)v218;
                                unsigned long long v220;
                                v220 = 256ull * v219;
                                unsigned long long v221;
                                v221 = v220 + 16ull;
                                unsigned long long v222;
                                v222 = v221 - 1ull;
                                unsigned long long v223;
                                v223 = v222 % 16ull;
                                unsigned long long v224;
                                v224 = v222 - v223;
                                unsigned long long v225;
                                v225 = v224 + 1024ull;
                                unsigned long long v226;
                                v226 = v225 + 16ull;
                                unsigned long long v227;
                                v227 = v226 - 1ull;
                                unsigned long long v228;
                                v228 = v227 % 16ull;
                                unsigned long long v229;
                                v229 = v227 - v228;
                                unsigned long long v230;
                                v230 = v229 + 1024ull;
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
                                float * * v238;
                                v238 = reinterpret_cast<float * *>(&v234[0ull]);
                                float * v240;
                                v240 = reinterpret_cast<float *>(&v234[v224]);
                                int * v242;
                                v242 = reinterpret_cast<int *>(&v234[v229]);
                                int v244;
                                v244 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v244 && v244 < 256);
                                v238[v244] = v216;
                                __syncthreads();
                                bool v245;
                                v245 = 0 <= v244;
                                bool v246;
                                v246 = v245 == false;
                                if (v246){
                                    assert("The index needs to be zero or positive." && v245);
                                } else {
                                }
                                int v248;
                                v248 = v244 % 16;
                                int v249;
                                v249 = v244 / 16;
                                bool v250;
                                v250 = v249 < 16;
                                bool v251;
                                v251 = v250 == false;
                                if (v251){
                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v250);
                                } else {
                                }
                                assert("Tensor range check" && 0 <= v249 && v249 < 16);
                                int v253;
                                v253 = 0;
                                while (while_method_9(v253)){
                                    bool v255;
                                    v255 = 0 <= v249;
                                    bool v256;
                                    v256 = v255 && v250;
                                    bool v257;
                                    v257 = v256 == false;
                                    if (v257){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v256);
                                    } else {
                                    }
                                    bool v259;
                                    v259 = 0 <= v253;
                                    bool v261;
                                    if (v259){
                                        bool v260;
                                        v260 = v253 < 16;
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
                                    v264 = v253 * 16;
                                    int v265;
                                    v265 = v264 + v249;
                                    assert("Tensor range check" && 0 <= v253 && v253 < 16);
                                    int v266;
                                    v266 = 16 * v253;
                                    int v267;
                                    v267 = v266 + v249;
                                    float * v268;
                                    v268 = v238[v267];
                                    int v269;
                                    v269 = blockIdx.x;
                                    int v270;
                                    v270 = v269 * 256;
                                    int v271;
                                    v271 = v270 + v265;
                                    assert("Tensor range check" && 0 <= v248 && v248 < 16);
                                    int v272;
                                    v272 = 4 * v248;
                                    float v273[4];
                                    int v274[4];
                                    int v275;
                                    v275 = 0;
                                    while (while_method_5(v275)){
                                        assert("Tensor range check" && 0 <= v275 && v275 < 1);
                                        int v277;
                                        v277 = 4 * v275;
                                        assert("Tensor range check" && 0 <= v275 && v275 < 1);
                                        int v278;
                                        v278 = 64 * v275;
                                        int v279;
                                        v279 = v278 + v272;
                                        int4* v280;
                                        v280 = reinterpret_cast<int4*>(v268 + v279);
                                        int4* v281;
                                        v281 = reinterpret_cast<int4*>(v273 + v277);
                                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v280) % 16 == 0 && reinterpret_cast<unsigned long long>(v281) % 16 == 0);
                                        *v281 = *v280;
                                        v275 += 1 ;
                                    }
                                    int v282;
                                    v282 = 0;
                                    while (while_method_5(v282)){
                                        int v284;
                                        v284 = 0;
                                        while (while_method_8(v284)){
                                            bool v286;
                                            v286 = 0 <= v284;
                                            bool v288;
                                            if (v286){
                                                bool v287;
                                                v287 = v284 < 4;
                                                v288 = v287;
                                            } else {
                                                v288 = false;
                                            }
                                            bool v289;
                                            v289 = v288 == false;
                                            if (v289){
                                                assert("The indices should be inside the range of the dimension." && v288);
                                            } else {
                                            }
                                            bool v291;
                                            v291 = 0 <= v248;
                                            bool v293;
                                            if (v291){
                                                bool v292;
                                                v292 = v248 < 16;
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
                                            int v296;
                                            v296 = v248 * 4;
                                            int v297;
                                            v297 = v284 + v296;
                                            bool v298;
                                            v298 = 0 <= v282;
                                            bool v300;
                                            if (v298){
                                                bool v299;
                                                v299 = v282 < 1;
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
                                            int v303;
                                            v303 = v282 * 64;
                                            int v304;
                                            v304 = v297 + v303;
                                            assert("Tensor range check" && 0 <= v282 && v282 < 1);
                                            assert("Tensor range check" && 0 <= v284 && v284 < 4);
                                            int v305;
                                            v305 = 4 * v282;
                                            int v306;
                                            v306 = v305 + v284;
                                            v274[v306] = v304;
                                            v284 += 1 ;
                                        }
                                        v282 += 1 ;
                                    }
                                    float v307[4];
                                    float v308;
                                    v308 = 0.0f;
                                    int v309;
                                    v309 = 0;
                                    while (while_method_5(v309)){
                                        assert("Tensor range check" && 0 <= v309 && v309 < 1);
                                        int v311;
                                        v311 = 4 * v309;
                                        assert("Tensor range check" && 0 <= v309 && v309 < 1);
                                        float v312;
                                        v312 = 0.0f;
                                        int v313;
                                        v313 = 0;
                                        while (while_method_8(v313)){
                                            assert("Tensor range check" && 0 <= v313 && v313 < 4);
                                            int v315;
                                            v315 = v313 + v311;
                                            float v316;
                                            v316 = v273[v315];
                                            float v317;
                                            v317 = v312 + v316;
                                            v312 = v317;
                                            v313 += 1 ;
                                        }
                                        auto v318 = cooperative_groups::coalesced_threads();
                                        int v319;
                                        v319 = threadIdx.x;
                                        int v320;
                                        v320 = v319 / 16;
                                        auto v321 = cooperative_groups::labeled_partition(v318,v320);
                                        Closure2 v322{};
                                        float v323;
                                        v323 = cooperative_groups::inclusive_scan(v321, v312, v322);
                                        float v324;
                                        v324 = v321.shfl_up(v323,1);
                                        bool v325;
                                        v325 = v321.thread_rank() == 0;
                                        float v326;
                                        if (v325){
                                            v326 = 0.0f;
                                        } else {
                                            v326 = v324;
                                        }
                                        float v327;
                                        v327 = v321.shfl(v323,v321.num_threads()-1);
                                        float v328;
                                        v328 = v308 + v326;
                                        float v329;
                                        v329 = v328;
                                        int v330;
                                        v330 = 0;
                                        while (while_method_8(v330)){
                                            assert("Tensor range check" && 0 <= v330 && v330 < 4);
                                            int v332;
                                            v332 = v330 + v311;
                                            float v333;
                                            v333 = v273[v332];
                                            float v334;
                                            v334 = v329 + v333;
                                            assert("Tensor range check" && 0 <= v330 && v330 < 4);
                                            v307[v332] = v334;
                                            v329 = v334;
                                            v330 += 1 ;
                                        }
                                        float v335;
                                        v335 = v308 + v327;
                                        v308 = v335;
                                        v309 += 1 ;
                                    }
                                    float v336[4];
                                    bool v337[4];
                                    int v338;
                                    v338 = 0;
                                    while (while_method_5(v338)){
                                        int v340;
                                        v340 = 0;
                                        while (while_method_8(v340)){
                                            assert("Tensor range check" && 0 <= v338 && v338 < 1);
                                            assert("Tensor range check" && 0 <= v340 && v340 < 4);
                                            int v342;
                                            v342 = 4 * v338;
                                            int v343;
                                            v343 = v342 + v340;
                                            float v344;
                                            v344 = v307[v343];
                                            float v345;
                                            v345 = v273[v343];
                                            bool v346;
                                            v346 = v345 > 0.0f;
                                            assert("Tensor range check" && 0 <= v338 && v338 < 1);
                                            assert("Tensor range check" && 0 <= v340 && v340 < 4);
                                            v336[v343] = v344;
                                            v337[v343] = v346;
                                            v340 += 1 ;
                                        }
                                        v338 += 1 ;
                                    }
                                    float v347; bool v348;
                                    Tuple8 tmp77 = Tuple8{-1.0f / 0.0f, false};
                                    v347 = tmp77.v0; v348 = tmp77.v1;
                                    int v349;
                                    v349 = 0;
                                    while (while_method_5(v349)){
                                        int v351;
                                        v351 = 0;
                                        while (while_method_8(v351)){
                                            assert("Tensor range check" && 0 <= v349 && v349 < 1);
                                            assert("Tensor range check" && 0 <= v351 && v351 < 4);
                                            int v353;
                                            v353 = 4 * v349;
                                            int v354;
                                            v354 = v353 + v351;
                                            float v355;
                                            v355 = v336[v354];
                                            bool v356;
                                            v356 = v337[v354];
                                            float v363; bool v364;
                                            if (v348){
                                                if (v356){
                                                    bool v357;
                                                    v357 = v347 >= v355;
                                                    float v358;
                                                    if (v357){
                                                        v358 = v347;
                                                    } else {
                                                        v358 = v355;
                                                    }
                                                    v363 = v358; v364 = true;
                                                } else {
                                                    v363 = v347; v364 = v348;
                                                }
                                            } else {
                                                if (v356){
                                                    v363 = v355; v364 = v356;
                                                } else {
                                                    v363 = v347; v364 = v348;
                                                }
                                            }
                                            v347 = v363;
                                            v348 = v364;
                                            v351 += 1 ;
                                        }
                                        v349 += 1 ;
                                    }
                                    auto v365 = cooperative_groups::coalesced_threads();
                                    int v366;
                                    v366 = threadIdx.x;
                                    int v367;
                                    v367 = v366 / 16;
                                    auto v368 = cooperative_groups::labeled_partition(v365,v367);
                                    Closure3 v369{};
                                    float v370; bool v371;
                                    Tuple8 tmp78 = cooperative_groups::reduce(v368, Tuple8{v347, v348}, v369);
                                    v370 = tmp78.v0; v371 = tmp78.v1;
                                    bool v372;
                                    v372 = v371 == false;
                                    if (v372){
                                        int v373;
                                        v373 = threadIdx.x;
                                        int v374;
                                        v374 = blockIdx.x;
                                        int v375;
                                        v375 = v374 * 256;
                                        int v376;
                                        v376 = v373 + v375;
                                        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v377 = console_lock;
                                        auto v378 = cooperative_groups::coalesced_threads();
                                        v377.acquire();
                                        int v379;
                                        v379 = 0;
                                        printf("{%s = %d; %s = %c","tid", v376, "x'", '[');
                                        int v380;
                                        v380 = 0;
                                        while (while_method_5(v380)){
                                            int v382;
                                            v382 = v379;
                                            bool v383;
                                            v383 = v382 >= 100;
                                            if (v383){
                                                printf("%s"," ...");
                                                break;
                                            } else {
                                            }
                                            bool v384;
                                            v384 = v380 == 0;
                                            bool v385;
                                            v385 = v384 != true;
                                            if (v385){
                                                printf("%s","; ");
                                            } else {
                                            }
                                            printf("%c",'[');
                                            int v386;
                                            v386 = 0;
                                            while (while_method_8(v386)){
                                                int v388;
                                                v388 = v379;
                                                bool v389;
                                                v389 = v388 >= 100;
                                                if (v389){
                                                    printf("%s"," ...");
                                                    break;
                                                } else {
                                                }
                                                bool v390;
                                                v390 = v386 == 0;
                                                bool v391;
                                                v391 = v390 != true;
                                                if (v391){
                                                    printf("%s","; ");
                                                } else {
                                                }
                                                int v392;
                                                v392 = v379 + 1;
                                                v379 = v392;
                                                int v393;
                                                v393 = v380 * 4;
                                                int v394;
                                                v394 = v393 + v386;
                                                float v395;
                                                v395 = v336[v394];
                                                bool v396;
                                                v396 = v337[v394];
                                                const char * v399;
                                                if (v396){
                                                    const char * v397;
                                                    v397 = "true";
                                                    v399 = v397;
                                                } else {
                                                    const char * v398;
                                                    v398 = "false";
                                                    v399 = v398;
                                                }
                                                printf("%f, %s",v395, v399);
                                                v386 += 1 ;
                                            }
                                            printf("%c",']');
                                            v380 += 1 ;
                                        }
                                        printf("%c",']');
                                        printf("}\n");
                                        v377.release();
                                        v378.sync() ;
                                    } else {
                                    }
                                    if (v372){
                                        assert("The local reduce must be true." && v371);
                                    } else {
                                    }
                                    float v435[4];
                                    int v436[4];
                                    int v437;
                                    v437 = 0;
                                    while (while_method_5(v437)){
                                        int v439;
                                        v439 = 0;
                                        while (while_method_8(v439)){
                                            assert("Tensor range check" && 0 <= v437 && v437 < 1);
                                            assert("Tensor range check" && 0 <= v439 && v439 < 4);
                                            int v441;
                                            v441 = 4 * v437;
                                            int v442;
                                            v442 = v441 + v439;
                                            int v443;
                                            v443 = v274[v442];
                                            float v444;
                                            v444 = curand_uniform(&v85);
                                            assert("Tensor range check" && 0 <= v437 && v437 < 1);
                                            assert("Tensor range check" && 0 <= v439 && v439 < 4);
                                            v435[v442] = v444;
                                            v436[v442] = v443;
                                            v439 += 1 ;
                                        }
                                        v437 += 1 ;
                                    }
                                    float v445; int v446;
                                    Tuple9 tmp79 = Tuple9{0.0f, 2147483647};
                                    v445 = tmp79.v0; v446 = tmp79.v1;
                                    int v447;
                                    v447 = 0;
                                    while (while_method_5(v447)){
                                        int v449;
                                        v449 = 0;
                                        while (while_method_8(v449)){
                                            assert("Tensor range check" && 0 <= v447 && v447 < 1);
                                            assert("Tensor range check" && 0 <= v449 && v449 < 4);
                                            int v451;
                                            v451 = 4 * v447;
                                            int v452;
                                            v452 = v451 + v449;
                                            float v453;
                                            v453 = v435[v452];
                                            int v454;
                                            v454 = v436[v452];
                                            bool v455;
                                            v455 = v446 < v454;
                                            float v456; int v457;
                                            if (v455){
                                                v456 = v445; v457 = v446;
                                            } else {
                                                v456 = v453; v457 = v454;
                                            }
                                            v445 = v456;
                                            v446 = v457;
                                            v449 += 1 ;
                                        }
                                        v447 += 1 ;
                                    }
                                    auto v458 = cooperative_groups::coalesced_threads();
                                    int v459;
                                    v459 = threadIdx.x;
                                    int v460;
                                    v460 = v459 / 16;
                                    auto v461 = cooperative_groups::labeled_partition(v458,v460);
                                    Closure4 v462{};
                                    float v463; int v464;
                                    Tuple9 tmp80 = cooperative_groups::reduce(v461, Tuple9{v445, v446}, v462);
                                    v463 = tmp80.v0; v464 = tmp80.v1;
                                    float v465;
                                    v465 = v370 * v463;
                                    int v466[4];
                                    bool v467[4];
                                    int v468;
                                    v468 = 0;
                                    while (while_method_5(v468)){
                                        int v470;
                                        v470 = 0;
                                        while (while_method_8(v470)){
                                            assert("Tensor range check" && 0 <= v468 && v468 < 1);
                                            assert("Tensor range check" && 0 <= v470 && v470 < 4);
                                            int v472;
                                            v472 = 4 * v468;
                                            int v473;
                                            v473 = v472 + v470;
                                            float v474;
                                            v474 = v336[v473];
                                            bool v475;
                                            v475 = v337[v473];
                                            int v476;
                                            v476 = v274[v473];
                                            int v479; bool v480;
                                            if (v475){
                                                float v477;
                                                v477 = v474 - v465;
                                                bool v478;
                                                v478 = v477 >= 0.0f;
                                                v479 = v476; v480 = v478;
                                            } else {
                                                v479 = 2147483647; v480 = false;
                                            }
                                            assert("Tensor range check" && 0 <= v468 && v468 < 1);
                                            assert("Tensor range check" && 0 <= v470 && v470 < 4);
                                            v466[v473] = v479;
                                            v467[v473] = v480;
                                            v470 += 1 ;
                                        }
                                        v468 += 1 ;
                                    }
                                    int v481; bool v482;
                                    Tuple10 tmp81 = Tuple10{2147483647, false};
                                    v481 = tmp81.v0; v482 = tmp81.v1;
                                    int v483;
                                    v483 = 0;
                                    while (while_method_5(v483)){
                                        int v485;
                                        v485 = 0;
                                        while (while_method_8(v485)){
                                            assert("Tensor range check" && 0 <= v483 && v483 < 1);
                                            assert("Tensor range check" && 0 <= v485 && v485 < 4);
                                            int v487;
                                            v487 = 4 * v483;
                                            int v488;
                                            v488 = v487 + v485;
                                            int v489;
                                            v489 = v466[v488];
                                            bool v490;
                                            v490 = v467[v488];
                                            int v497; bool v498;
                                            if (v482){
                                                if (v490){
                                                    bool v491;
                                                    v491 = v481 < v489;
                                                    int v492;
                                                    if (v491){
                                                        v492 = v481;
                                                    } else {
                                                        v492 = v489;
                                                    }
                                                    v497 = v492; v498 = true;
                                                } else {
                                                    v497 = v481; v498 = v482;
                                                }
                                            } else {
                                                if (v490){
                                                    v497 = v489; v498 = v490;
                                                } else {
                                                    v497 = v481; v498 = v482;
                                                }
                                            }
                                            v481 = v497;
                                            v482 = v498;
                                            v485 += 1 ;
                                        }
                                        v483 += 1 ;
                                    }
                                    auto v499 = cooperative_groups::coalesced_threads();
                                    int v500;
                                    v500 = threadIdx.x;
                                    int v501;
                                    v501 = v500 / 16;
                                    auto v502 = cooperative_groups::labeled_partition(v499,v501);
                                    Closure5 v503{};
                                    int v504; bool v505;
                                    Tuple10 tmp82 = cooperative_groups::reduce(v502, Tuple10{v481, v482}, v503);
                                    v504 = tmp82.v0; v505 = tmp82.v1;
                                    bool v506;
                                    v506 = v505 == false;
                                    if (v506){
                                        int v507;
                                        v507 = threadIdx.x;
                                        int v508;
                                        v508 = blockIdx.x;
                                        int v509;
                                        v509 = v508 * 256;
                                        int v510;
                                        v510 = v507 + v509;
                                        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v511 = console_lock;
                                        auto v512 = cooperative_groups::coalesced_threads();
                                        v511.acquire();
                                        int v513;
                                        v513 = 0;
                                        printf("{%s = %d; %s = %c","tid", v510, "x'", '[');
                                        int v514;
                                        v514 = 0;
                                        while (while_method_5(v514)){
                                            int v516;
                                            v516 = v513;
                                            bool v517;
                                            v517 = v516 >= 100;
                                            if (v517){
                                                printf("%s"," ...");
                                                break;
                                            } else {
                                            }
                                            bool v518;
                                            v518 = v514 == 0;
                                            bool v519;
                                            v519 = v518 != true;
                                            if (v519){
                                                printf("%s","; ");
                                            } else {
                                            }
                                            printf("%c",'[');
                                            int v520;
                                            v520 = 0;
                                            while (while_method_8(v520)){
                                                int v522;
                                                v522 = v513;
                                                bool v523;
                                                v523 = v522 >= 100;
                                                if (v523){
                                                    printf("%s"," ...");
                                                    break;
                                                } else {
                                                }
                                                bool v524;
                                                v524 = v520 == 0;
                                                bool v525;
                                                v525 = v524 != true;
                                                if (v525){
                                                    printf("%s","; ");
                                                } else {
                                                }
                                                int v526;
                                                v526 = v513 + 1;
                                                v513 = v526;
                                                int v527;
                                                v527 = v514 * 4;
                                                int v528;
                                                v528 = v527 + v520;
                                                int v529;
                                                v529 = v466[v528];
                                                bool v530;
                                                v530 = v467[v528];
                                                const char * v533;
                                                if (v530){
                                                    const char * v531;
                                                    v531 = "true";
                                                    v533 = v531;
                                                } else {
                                                    const char * v532;
                                                    v532 = "false";
                                                    v533 = v532;
                                                }
                                                printf("%d, %s",v529, v533);
                                                v520 += 1 ;
                                            }
                                            printf("%c",']');
                                            v514 += 1 ;
                                        }
                                        printf("%c",']');
                                        printf("}\n");
                                        v511.release();
                                        v512.sync() ;
                                    } else {
                                    }
                                    if (v506){
                                        assert("The local reduce must be true." && v505);
                                    } else {
                                    }
                                    float v569; int v570;
                                    Tuple9 tmp83 = Tuple9{0.0f, 2147483647};
                                    v569 = tmp83.v0; v570 = tmp83.v1;
                                    int v571;
                                    v571 = 0;
                                    while (while_method_5(v571)){
                                        int v573;
                                        v573 = 0;
                                        while (while_method_8(v573)){
                                            assert("Tensor range check" && 0 <= v571 && v571 < 1);
                                            assert("Tensor range check" && 0 <= v573 && v573 < 4);
                                            int v575;
                                            v575 = 4 * v571;
                                            int v576;
                                            v576 = v575 + v573;
                                            float v577;
                                            v577 = v273[v576];
                                            int v578;
                                            v578 = v274[v576];
                                            bool v579;
                                            v579 = v570 == v504;
                                            float v583; int v584;
                                            if (v579){
                                                v583 = v569; v584 = v570;
                                            } else {
                                                bool v580;
                                                v580 = v578 == v504;
                                                if (v580){
                                                    v583 = v577; v584 = v578;
                                                } else {
                                                    v583 = v569; v584 = v570;
                                                }
                                            }
                                            v569 = v583;
                                            v570 = v584;
                                            v573 += 1 ;
                                        }
                                        v571 += 1 ;
                                    }
                                    auto v585 = cooperative_groups::coalesced_threads();
                                    int v586;
                                    v586 = threadIdx.x;
                                    int v587;
                                    v587 = v586 / 16;
                                    auto v588 = cooperative_groups::labeled_partition(v585,v587);
                                    Closure6 v589{v504};
                                    float v590; int v591;
                                    Tuple9 tmp84 = cooperative_groups::reduce(v588, Tuple9{v569, v570}, v589);
                                    v590 = tmp84.v0; v591 = tmp84.v1;
                                    bool v592;
                                    v592 = v591 == 2147483647;
                                    bool v593;
                                    v593 = v592 != true;
                                    bool v594;
                                    v594 = v593 == false;
                                    if (v594){
                                        assert("Expected a valid action id in get_prob." && v593);
                                    } else {
                                    }
                                    int v596;
                                    v596 = 0;
                                    while (while_method_5(v596)){
                                        assert("Tensor range check" && 0 <= v596 && v596 < 1);
                                        assert("Tensor range check" && 0 <= v596 && v596 < 1);
                                        v596 += 1 ;
                                    }
                                    assert("Tensor range check" && 0 <= v265 && v265 < 256);
                                    v240[v265] = v590;
                                    v242[v265] = v504;
                                    v253 += 1 ;
                                }
                                __syncthreads();
                                assert("Tensor range check" && 0 <= v244 && v244 < 256);
                                float v598;
                                v598 = v240[v244];
                                int v599;
                                v599 = v242[v244];
                                __syncthreads();
                                extern __shared__ unsigned char v600[];
                                float * v601;
                                v601 = reinterpret_cast<float *>(&v600[0ull]);
                                int * v603;
                                v603 = reinterpret_cast<int *>(&v600[16ull]);
                                int v605;
                                v605 = threadIdx.x;
                                bool v606;
                                v606 = v605 == 0;
                                if (v606){
                                    v601[0] = v598;
                                    v603[0] = v599;
                                } else {
                                }
                                __syncthreads();
                                float v607;
                                v607 = v601[0];
                                int v608;
                                v608 = v603[0];
                                __syncthreads();
                                double * v609;
                                v609 = reinterpret_cast<double *>(&v1[58195968ull]);
                                double * v611;
                                v611 = reinterpret_cast<double *>(&v1[61341696ull]);
                                int v613;
                                v613 = threadIdx.x;
                                int v614;
                                v614 = blockIdx.x;
                                int v615;
                                v615 = v614 * 256;
                                int v616;
                                v616 = v613 + v615;
                                int v617;
                                v617 = 0;
                                while (while_method_4(v617)){
                                    float * v619;
                                    v619 = reinterpret_cast<float *>(&v1[7864320ull]);
                                    int v621;
                                    v621 = blockIdx.x;
                                    int v622;
                                    v622 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v617 && v617 < 32);
                                    assert("Tensor range check" && 0 <= v621 && v621 < 24);
                                    assert("Tensor range check" && 0 <= v622 && v622 < 256);
                                    assert("Tensor range check" && 0 <= v608 && v608 < 64);
                                    int v623;
                                    v623 = 64 * v622;
                                    int v624;
                                    v624 = v623 + v608;
                                    int v625;
                                    v625 = 16384 * v621;
                                    int v626;
                                    v626 = v625 + v624;
                                    int v627;
                                    v627 = 393216 * v617;
                                    int v628;
                                    v628 = v627 + v626;
                                    float v629;
                                    v629 = v619[v628];
                                    double v630;
                                    v630 = (double)v607;
                                    double v631;
                                    v631 = log(v630);
                                    double v632;
                                    v632 = (double)v629;
                                    double v633;
                                    v633 = log(v632);
                                    assert("Tensor range check" && 0 <= v617 && v617 < 32);
                                    assert("Tensor range check" && 0 <= v616 && v616 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    int v634;
                                    v634 = 2 * v616;
                                    int v635;
                                    v635 = v634 + v71;
                                    int v636;
                                    v636 = 12288 * v617;
                                    int v637;
                                    v637 = v636 + v635;
                                    double v638;
                                    v638 = v609[v637];
                                    double v639;
                                    v639 = v611[v637];
                                    double v640;
                                    v640 = v633 + v638;
                                    double v641;
                                    v641 = v631 + v639;
                                    bool v642;
                                    v642 = isnan(v641);
                                    bool v643;
                                    v643 = v642 == false;
                                    bool v644;
                                    v644 = v643 == false;
                                    if (v644){
                                        assert("The sampling log probability shouldn't be nan." && v643);
                                    } else {
                                    }
                                    bool v646;
                                    v646 = isnan(v640);
                                    bool v647;
                                    v647 = v646 == false;
                                    bool v648;
                                    v648 = v647 == false;
                                    if (v648){
                                        assert("The policy log probability shouldn't be nan." && v647);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v617 && v617 < 32);
                                    assert("Tensor range check" && 0 <= v616 && v616 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    v609[v637] = v640;
                                    v611[v637] = v641;
                                    v617 += 1 ;
                                }
                                bool v650;
                                v650 = 0 == v608;
                                Union12 v659;
                                if (v650){
                                    v659 = Union12{Union12_1{}};
                                } else {
                                    bool v652;
                                    v652 = 1 == v608;
                                    if (v652){
                                        v659 = Union12{Union12_0{}};
                                    } else {
                                        bool v654;
                                        v654 = 2 == v608;
                                        if (v654){
                                            v659 = Union12{Union12_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v659.tag) {
                                    case 0: { // AA_Call
                                        v734 = Union1{Union1_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v660;
                                        v660 = v72[0];
                                        int v662; int v663;
                                        Tuple7 tmp85 = Tuple7{1, v660};
                                        v662 = tmp85.v0; v663 = tmp85.v1;
                                        while (while_method_0(v662)){
                                            bool v665;
                                            v665 = 0 <= v662;
                                            bool v667;
                                            if (v665){
                                                bool v666;
                                                v666 = v662 < 2;
                                                v667 = v666;
                                            } else {
                                                v667 = false;
                                            }
                                            bool v668;
                                            v668 = v667 == false;
                                            if (v668){
                                                assert("Index must be in range." && v667);
                                            } else {
                                            }
                                            int v670;
                                            v670 = v72[v662];
                                            bool v672;
                                            v672 = v663 >= v670;
                                            int v673;
                                            if (v672){
                                                v673 = v663;
                                            } else {
                                                v673 = v670;
                                            }
                                            v663 = v673;
                                            v662 += 1 ;
                                        }
                                        bool v675;
                                        if (v75){
                                            bool v674;
                                            v674 = v71 < 2;
                                            v675 = v674;
                                        } else {
                                            v675 = false;
                                        }
                                        bool v676;
                                        v676 = v675 == false;
                                        if (v676){
                                            assert("Index must be in range." && v675);
                                        } else {
                                        }
                                        int v678;
                                        v678 = v72[v71];
                                        bool v680;
                                        v680 = v678 == v663;
                                        if (v680){
                                            v734 = Union1{Union1_0{}};
                                        } else {
                                            v734 = Union1{Union1_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v685;
                                        v685 = v73 > 0;
                                        if (v685){
                                            v734 = Union1{Union1_2{}};
                                        } else {
                                            v734 = Union1{Union1_0{}};
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
                                curandStatePhilox4_32_10_t & v692 = v3.v5;
                                curandStatePhilox4_32_10_t & v693 = v692;
                                static_array_list<Union1,3> v694;
                                v694 = static_array_list<Union1,3>{};
                                v694.unsafe_set_length(1);
                                Union1 v696;
                                v696 = Union1{Union1_0{}};
                                v694[0] = v696;
                                int v698;
                                v698 = v72[0];
                                int v700;
                                v700 = v72[1];
                                bool v702;
                                v702 = v698 == v700;
                                bool v703;
                                v703 = v702 != true;
                                if (v703){
                                    Union1 v704;
                                    v704 = Union1{Union1_1{}};
                                    v694.push(v704);
                                } else {
                                }
                                bool v705;
                                v705 = v73 > 0;
                                if (v705){
                                    Union1 v706;
                                    v706 = Union1{Union1_2{}};
                                    v694.push(v706);
                                } else {
                                }
                                int v707;
                                v707 = v694.length;
                                int v708;
                                v708 = v707 - 1;
                                int v709;
                                v709 = 0;
                                while (while_method_1(v708, v709)){
                                    int v711;
                                    v711 = v694.length;
                                    int v712;
                                    v712 = int_range_22(v711, v709, v693);
                                    Union1 v713;
                                    v713 = v694[v709];
                                    Union1 v715;
                                    v715 = v694[v712];
                                    v694[v709] = v715;
                                    v694[v712] = v713;
                                    v709 += 1 ;
                                }
                                Union1 v717;
                                v717 = v694.pop();
                                int v718;
                                v718 = sizeof(Union1);
                                unsigned long long v719;
                                v719 = (unsigned long long)v718;
                                bool v720;
                                v720 = v719 <= 98304ull;
                                bool v721;
                                v721 = v720 == false;
                                if (v721){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v720);
                                } else {
                                }
                                extern __shared__ unsigned char v723[];
                                bool v724;
                                v724 = v719 <= v719;
                                bool v725;
                                v725 = v724 == false;
                                if (v725){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v724);
                                } else {
                                }
                                Union1 * v727;
                                v727 = reinterpret_cast<Union1 *>(&v723[0ull]);
                                int v729;
                                v729 = threadIdx.x;
                                bool v730;
                                v730 = v729 == 0;
                                if (v730){
                                    v727[0] = v717;
                                } else {
                                }
                                __syncthreads();
                                Union1 v731;
                                v731 = v727[0];
                                __syncthreads();
                                v734 = v731;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union7 v735;
                        v735 = Union7{Union7_1{v71, v734}};
                        v14.push(v735);
                        v777 = Union14{Union14_2{v68, v69, v70, v71, v72, v73, v734}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v737 = v18.case3.v0; bool v738 = v18.case3.v1; static_array<Union6,2> v739 = v18.case3.v2; int v740 = v18.case3.v3; static_array<int,2> v741 = v18.case3.v4; int v742 = v18.case3.v5; Union1 v743 = v18.case3.v6;
                        Union7 v744;
                        v744 = Union7{Union7_1{v740, v743}};
                        v14.push(v744);
                        v777 = Union14{Union14_2{v737, v738, v739, v740, v741, v742, v743}};
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
                        v52 = compare_hands_27(v39, v40, v41, v42, v43, v44);
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
                        v777 = Union14{Union14_3{}};
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
                        v777 = Union14{Union14_3{}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                switch (v777.tag) {
                    case 0: { // T_game_chance_community_card
                        Union5 v779 = v777.case0.v0; bool v780 = v777.case0.v1; static_array<Union6,2> v781 = v777.case0.v2; int v782 = v777.case0.v3; static_array<int,2> v783 = v777.case0.v4; int v784 = v777.case0.v5; Union6 v785 = v777.case0.v6;
                        int v786;
                        v786 = 2;
                        int v787; int v788;
                        Tuple7 tmp86 = Tuple7{0, 0};
                        v787 = tmp86.v0; v788 = tmp86.v1;
                        while (while_method_0(v787)){
                            bool v790;
                            v790 = 0 <= v787;
                            bool v792;
                            if (v790){
                                bool v791;
                                v791 = v787 < 2;
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
                            v795 = v783[v787];
                            bool v797;
                            v797 = v788 >= v795;
                            int v798;
                            if (v797){
                                v798 = v788;
                            } else {
                                v798 = v795;
                            }
                            v788 = v798;
                            v787 += 1 ;
                        }
                        static_array<int,2> v799;
                        int v801;
                        v801 = 0;
                        while (while_method_0(v801)){
                            v799[v801] = v788;
                            v801 += 1 ;
                        }
                        Union5 v803;
                        v803 = Union5{Union5_1{v785}};
                        Union4 v804;
                        v804 = Union4{Union4_2{v803, true, v781, 0, v799, v786}};
                        v937 = Union3{Union3_1{v804}};
                        break;
                    }
                    case 1: { // T_game_chance_init
                        Union6 v806 = v777.case1.v0; Union6 v807 = v777.case1.v1;
                        int v808;
                        v808 = 2;
                        static_array<int,2> v809;
                        v809[0] = 1;
                        v809[1] = 1;
                        static_array<Union6,2> v811;
                        v811[0] = v806;
                        v811[1] = v807;
                        Union5 v813;
                        v813 = Union5{Union5_0{}};
                        Union4 v814;
                        v814 = Union4{Union4_2{v813, true, v811, 0, v809, v808}};
                        v937 = Union3{Union3_1{v814}};
                        break;
                    }
                    case 2: { // T_game_round
                        Union5 v816 = v777.case2.v0; bool v817 = v777.case2.v1; static_array<Union6,2> v818 = v777.case2.v2; int v819 = v777.case2.v3; static_array<int,2> v820 = v777.case2.v4; int v821 = v777.case2.v5; Union1 v822 = v777.case2.v6;
                        Union4 v929;
                        switch (v816.tag) {
                            case 0: { // None
                                switch (v822.tag) {
                                    case 0: { // Call
                                        if (v817){
                                            int v885;
                                            v885 = v819 ^ 1;
                                            v929 = Union4{Union4_2{v816, false, v818, v885, v820, v821}};
                                        } else {
                                            v929 = Union4{Union4_0{v816, v817, v818, v819, v820, v821}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v929 = Union4{Union4_5{v816, v817, v818, v819, v820, v821}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v889;
                                        v889 = v821 > 0;
                                        if (v889){
                                            int v890;
                                            v890 = v819 ^ 1;
                                            int v891;
                                            v891 = -1 + v821;
                                            int v892; int v893;
                                            Tuple7 tmp87 = Tuple7{0, 0};
                                            v892 = tmp87.v0; v893 = tmp87.v1;
                                            while (while_method_0(v892)){
                                                bool v895;
                                                v895 = 0 <= v892;
                                                bool v897;
                                                if (v895){
                                                    bool v896;
                                                    v896 = v892 < 2;
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
                                                v900 = v820[v892];
                                                bool v902;
                                                v902 = v893 >= v900;
                                                int v903;
                                                if (v902){
                                                    v903 = v893;
                                                } else {
                                                    v903 = v900;
                                                }
                                                v893 = v903;
                                                v892 += 1 ;
                                            }
                                            static_array<int,2> v904;
                                            int v906;
                                            v906 = 0;
                                            while (while_method_0(v906)){
                                                v904[v906] = v893;
                                                v906 += 1 ;
                                            }
                                            static_array<int,2> v908;
                                            int v910;
                                            v910 = 0;
                                            while (while_method_0(v910)){
                                                bool v912;
                                                v912 = 0 <= v910;
                                                bool v914;
                                                if (v912){
                                                    bool v913;
                                                    v913 = v910 < 2;
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
                                                int v917;
                                                v917 = v904[v910];
                                                bool v919;
                                                v919 = v910 == v819;
                                                int v921;
                                                if (v919){
                                                    int v920;
                                                    v920 = v917 + 2;
                                                    v921 = v920;
                                                } else {
                                                    v921 = v917;
                                                }
                                                v908[v910] = v921;
                                                v910 += 1 ;
                                            }
                                            v929 = Union4{Union4_2{v816, false, v818, v890, v908, v891}};
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
                                Union6 v823 = v816.case1.v0;
                                switch (v822.tag) {
                                    case 0: { // Call
                                        if (v817){
                                            int v825;
                                            v825 = v819 ^ 1;
                                            v929 = Union4{Union4_2{v816, false, v818, v825, v820, v821}};
                                        } else {
                                            int v827; int v828;
                                            Tuple7 tmp88 = Tuple7{0, 0};
                                            v827 = tmp88.v0; v828 = tmp88.v1;
                                            while (while_method_0(v827)){
                                                bool v830;
                                                v830 = 0 <= v827;
                                                bool v832;
                                                if (v830){
                                                    bool v831;
                                                    v831 = v827 < 2;
                                                    v832 = v831;
                                                } else {
                                                    v832 = false;
                                                }
                                                bool v833;
                                                v833 = v832 == false;
                                                if (v833){
                                                    assert("Index must be in range." && v832);
                                                } else {
                                                }
                                                int v835;
                                                v835 = v820[v827];
                                                bool v837;
                                                v837 = v828 >= v835;
                                                int v838;
                                                if (v837){
                                                    v838 = v828;
                                                } else {
                                                    v838 = v835;
                                                }
                                                v828 = v838;
                                                v827 += 1 ;
                                            }
                                            static_array<int,2> v839;
                                            int v841;
                                            v841 = 0;
                                            while (while_method_0(v841)){
                                                v839[v841] = v828;
                                                v841 += 1 ;
                                            }
                                            v929 = Union4{Union4_4{v816, v817, v818, v819, v839, v821}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v929 = Union4{Union4_5{v816, v817, v818, v819, v820, v821}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v845;
                                        v845 = v821 > 0;
                                        if (v845){
                                            int v846;
                                            v846 = v819 ^ 1;
                                            int v847;
                                            v847 = -1 + v821;
                                            int v848; int v849;
                                            Tuple7 tmp89 = Tuple7{0, 0};
                                            v848 = tmp89.v0; v849 = tmp89.v1;
                                            while (while_method_0(v848)){
                                                bool v851;
                                                v851 = 0 <= v848;
                                                bool v853;
                                                if (v851){
                                                    bool v852;
                                                    v852 = v848 < 2;
                                                    v853 = v852;
                                                } else {
                                                    v853 = false;
                                                }
                                                bool v854;
                                                v854 = v853 == false;
                                                if (v854){
                                                    assert("Index must be in range." && v853);
                                                } else {
                                                }
                                                int v856;
                                                v856 = v820[v848];
                                                bool v858;
                                                v858 = v849 >= v856;
                                                int v859;
                                                if (v858){
                                                    v859 = v849;
                                                } else {
                                                    v859 = v856;
                                                }
                                                v849 = v859;
                                                v848 += 1 ;
                                            }
                                            static_array<int,2> v860;
                                            int v862;
                                            v862 = 0;
                                            while (while_method_0(v862)){
                                                v860[v862] = v849;
                                                v862 += 1 ;
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
                                                v873 = v860[v866];
                                                bool v875;
                                                v875 = v866 == v819;
                                                int v877;
                                                if (v875){
                                                    int v876;
                                                    v876 = v873 + 4;
                                                    v877 = v876;
                                                } else {
                                                    v877 = v873;
                                                }
                                                v864[v866] = v877;
                                                v866 += 1 ;
                                            }
                                            v929 = Union4{Union4_2{v816, false, v818, v846, v864, v847}};
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
                        v937 = Union3{Union3_1{v929}};
                        break;
                    }
                    case 3: { // T_none
                        v937 = Union3{Union3_0{}};
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
        v16 = v937;
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
                Union3 v1138;
                switch (v60.tag) {
                    case 0: { // None
                        v1138 = Union3{Union3_0{}};
                        break;
                    }
                    case 1: { // Some
                        Union4 v62 = v60.case1.v0;
                        switch (v62.tag) {
                            case 0: { // ChanceCommunityCard
                                Union5 v1078 = v62.case0.v0; bool v1079 = v62.case0.v1; static_array<Union6,2> v1080 = v62.case0.v2; int v1081 = v62.case0.v3; static_array<int,2> v1082 = v62.case0.v4; int v1083 = v62.case0.v5;
                                curandStatePhilox4_32_10_t & v1084 = v19.v4;
                                curandStatePhilox4_32_10_t & v1085 = v1084;
                                unsigned int & v1086 = v19.v0;
                                Union6 v1087; unsigned int v1088;
                                Tuple6 tmp11 = draw_card_20(v1085, v1086);
                                v1087 = tmp11.v0; v1088 = tmp11.v1;
                                v19.v0 = v1088;
                                Union7 v1089;
                                v1089 = Union7{Union7_0{v1087}};
                                v58.push(v1089);
                                int v1090;
                                v1090 = 2;
                                int v1091; int v1092;
                                Tuple7 tmp12 = Tuple7{0, 0};
                                v1091 = tmp12.v0; v1092 = tmp12.v1;
                                while (while_method_0(v1091)){
                                    bool v1094;
                                    v1094 = 0 <= v1091;
                                    bool v1096;
                                    if (v1094){
                                        bool v1095;
                                        v1095 = v1091 < 2;
                                        v1096 = v1095;
                                    } else {
                                        v1096 = false;
                                    }
                                    bool v1097;
                                    v1097 = v1096 == false;
                                    if (v1097){
                                        assert("Index must be in range." && v1096);
                                    } else {
                                    }
                                    int v1099;
                                    v1099 = v1082[v1091];
                                    bool v1101;
                                    v1101 = v1092 >= v1099;
                                    int v1102;
                                    if (v1101){
                                        v1102 = v1092;
                                    } else {
                                        v1102 = v1099;
                                    }
                                    v1092 = v1102;
                                    v1091 += 1 ;
                                }
                                static_array<int,2> v1103;
                                int v1105;
                                v1105 = 0;
                                while (while_method_0(v1105)){
                                    v1103[v1105] = v1092;
                                    v1105 += 1 ;
                                }
                                Union5 v1107;
                                v1107 = Union5{Union5_1{v1087}};
                                Union4 v1108;
                                v1108 = Union4{Union4_2{v1107, true, v1080, 0, v1103, v1090}};
                                v1138 = Union3{Union3_1{v1108}};
                                break;
                            }
                            case 1: { // ChanceInit
                                curandStatePhilox4_32_10_t & v1110 = v19.v4;
                                curandStatePhilox4_32_10_t & v1111 = v1110;
                                unsigned int & v1112 = v19.v0;
                                Union6 v1113; unsigned int v1114;
                                Tuple6 tmp13 = draw_card_20(v1111, v1112);
                                v1113 = tmp13.v0; v1114 = tmp13.v1;
                                v19.v0 = v1114;
                                curandStatePhilox4_32_10_t & v1115 = v19.v4;
                                curandStatePhilox4_32_10_t & v1116 = v1115;
                                unsigned int & v1117 = v19.v0;
                                Union6 v1118; unsigned int v1119;
                                Tuple6 tmp14 = draw_card_20(v1116, v1117);
                                v1118 = tmp14.v0; v1119 = tmp14.v1;
                                v19.v0 = v1119;
                                Union7 v1120;
                                v1120 = Union7{Union7_2{0, v1113}};
                                v58.push(v1120);
                                Union7 v1121;
                                v1121 = Union7{Union7_2{1, v1118}};
                                v58.push(v1121);
                                int v1122;
                                v1122 = 2;
                                static_array<int,2> v1123;
                                v1123[0] = 1;
                                v1123[1] = 1;
                                static_array<Union6,2> v1125;
                                v1125[0] = v1113;
                                v1125[1] = v1118;
                                Union5 v1127;
                                v1127 = Union5{Union5_0{}};
                                Union4 v1128;
                                v1128 = Union4{Union4_2{v1127, true, v1125, 0, v1123, v1122}};
                                v1138 = Union3{Union3_1{v1128}};
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
                                        v122 = reinterpret_cast<float *>(&v3[7864320ull]);
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
                                        int * v191;
                                        v191 = reinterpret_cast<int *>(&v2[1572864ull]);
                                        bool * v193;
                                        v193 = reinterpret_cast<bool *>(&v2[1572880ull]);
                                        float * v195;
                                        v195 = reinterpret_cast<float *>(&v2[1572912ull]);
                                        float * v197;
                                        v197 = reinterpret_cast<float *>(&v2[1573040ull]);
                                        int v199;
                                        v199 = 0;
                                        int v200;
                                        v200 = 0;
                                        while (while_method_4(v200)){
                                            assert("Tensor range check" && 0 <= v200 && v200 < 32);
                                            bool v202;
                                            v202 = v193[v200];
                                            int v204;
                                            if (v202){
                                                int v203;
                                                v203 = v199 + 1;
                                                v204 = v203;
                                            } else {
                                                v204 = v199;
                                            }
                                            v199 = v204;
                                            v200 += 1 ;
                                        }
                                        int v205;
                                        v205 = 0;
                                        int v206;
                                        v206 = int_range_22(v199, v205, v121);
                                        int v207;
                                        v207 = 0;
                                        int v208;
                                        v208 = 0;
                                        while (while_method_4(v208)){
                                            assert("Tensor range check" && 0 <= v208 && v208 < 32);
                                            bool v210;
                                            v210 = v193[v208];
                                            if (v210){
                                                bool v211;
                                                v211 = v206 == 0;
                                                if (v211){
                                                    v207 = v208;
                                                } else {
                                                }
                                                int v212;
                                                v212 = v206 - 1;
                                                v206 = v212;
                                            } else {
                                            }
                                            v208 += 1 ;
                                        }
                                        extern __shared__ unsigned char v213[];
                                        int * v214;
                                        v214 = reinterpret_cast<int *>(&v213[0ull]);
                                        int v216;
                                        v216 = threadIdx.x;
                                        bool v217;
                                        v217 = v216 == 0;
                                        if (v217){
                                            v214[0] = v207;
                                        } else {
                                        }
                                        __syncthreads();
                                        int v218;
                                        v218 = v214[0];
                                        __syncthreads();
                                        float * v219;
                                        v219 = reinterpret_cast<float *>(&v3[7864320ull]);
                                        assert("Tensor range check" && 0 <= v218 && v218 < 32);
                                        int v221;
                                        v221 = 393216 * v218;
                                        float * v222;
                                        v222 = reinterpret_cast<float *>(&v3[4718592ull]);
                                        float * v224;
                                        v224 = reinterpret_cast<float *>(&v3[0ull]);
                                        float * v226;
                                        v226 = reinterpret_cast<float *>(&v2[0ull]);
                                        float * v228;
                                        v228 = reinterpret_cast<float *>(&v4[0ull]);
                                        assert("Tensor range check" && 0 <= v218 && v218 < 32);
                                        int v230;
                                        v230 = 8192 * v218;
                                        float * v231;
                                        v231 = reinterpret_cast<float *>(&v3[3145728ull]);
                                        block_matmul_23(v231, v226, v230, v224);
                                        block_map_24(v222, v231);
                                        float * v233;
                                        v233 = reinterpret_cast<float *>(&v2[1048576ull]);
                                        float * v235;
                                        v235 = reinterpret_cast<float *>(&v4[1048576ull]);
                                        assert("Tensor range check" && 0 <= v218 && v218 < 32);
                                        int v237;
                                        v237 = 4096 * v218;
                                        float * v238;
                                        v238 = reinterpret_cast<float *>(&v3[6291456ull]);
                                        block_matmul_25(v238, v233, v237, v222);
                                        block_row_map_26(v219, v221, v238);
                                        int * v240;
                                        v240 = reinterpret_cast<int *>(&v2[1572864ull]);
                                        bool * v242;
                                        v242 = reinterpret_cast<bool *>(&v2[1572880ull]);
                                        float * v244;
                                        v244 = reinterpret_cast<float *>(&v2[1572912ull]);
                                        float * v246;
                                        v246 = reinterpret_cast<float *>(&v2[1573040ull]);
                                        double * v248;
                                        v248 = reinterpret_cast<double *>(&v3[58195968ull]);
                                        double * v250;
                                        v250 = reinterpret_cast<double *>(&v3[61341696ull]);
                                        __syncthreads();
                                        float * v252;
                                        v252 = reinterpret_cast<float *>(&v3[7864320ull]);
                                        assert("Tensor range check" && 0 <= v218 && v218 < 32);
                                        int v254;
                                        v254 = blockIdx.x;
                                        assert("Tensor range check" && 0 <= v254 && v254 < 24);
                                        int v255;
                                        v255 = 16384 * v254;
                                        int v256;
                                        v256 = v255 + v221;
                                        int v257;
                                        v257 = threadIdx.x;
                                        assert("Tensor range check" && 0 <= v257 && v257 < 256);
                                        int v258;
                                        v258 = 64 * v257;
                                        int v259;
                                        v259 = v258 + v256;
                                        float * v260;
                                        v260 = v252+v259;
                                        int v262;
                                        v262 = sizeof(float *);
                                        unsigned long long v263;
                                        v263 = (unsigned long long)v262;
                                        unsigned long long v264;
                                        v264 = 256ull * v263;
                                        unsigned long long v265;
                                        v265 = v264 + 16ull;
                                        unsigned long long v266;
                                        v266 = v265 - 1ull;
                                        unsigned long long v267;
                                        v267 = v266 % 16ull;
                                        unsigned long long v268;
                                        v268 = v266 - v267;
                                        unsigned long long v269;
                                        v269 = v268 + 1024ull;
                                        unsigned long long v270;
                                        v270 = v269 + 16ull;
                                        unsigned long long v271;
                                        v271 = v270 - 1ull;
                                        unsigned long long v272;
                                        v272 = v271 % 16ull;
                                        unsigned long long v273;
                                        v273 = v271 - v272;
                                        unsigned long long v274;
                                        v274 = v273 + 1024ull;
                                        bool v275;
                                        v275 = v274 <= 98304ull;
                                        bool v276;
                                        v276 = v275 == false;
                                        if (v276){
                                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v275);
                                        } else {
                                        }
                                        extern __shared__ unsigned char v278[];
                                        bool v279;
                                        v279 = v274 <= v274;
                                        bool v280;
                                        v280 = v279 == false;
                                        if (v280){
                                            assert("The length of the partition has to be less than or equal to the length of the base array." && v279);
                                        } else {
                                        }
                                        float * * v282;
                                        v282 = reinterpret_cast<float * *>(&v278[0ull]);
                                        float * v284;
                                        v284 = reinterpret_cast<float *>(&v278[v268]);
                                        int * v286;
                                        v286 = reinterpret_cast<int *>(&v278[v273]);
                                        int v288;
                                        v288 = threadIdx.x;
                                        assert("Tensor range check" && 0 <= v288 && v288 < 256);
                                        v282[v288] = v260;
                                        __syncthreads();
                                        bool v289;
                                        v289 = 0 <= v288;
                                        bool v290;
                                        v290 = v289 == false;
                                        if (v290){
                                            assert("The index needs to be zero or positive." && v289);
                                        } else {
                                        }
                                        int v292;
                                        v292 = v288 % 16;
                                        int v293;
                                        v293 = v288 / 16;
                                        bool v294;
                                        v294 = v293 < 16;
                                        bool v295;
                                        v295 = v294 == false;
                                        if (v295){
                                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v294);
                                        } else {
                                        }
                                        assert("Tensor range check" && 0 <= v293 && v293 < 16);
                                        int v297;
                                        v297 = 0;
                                        while (while_method_9(v297)){
                                            bool v299;
                                            v299 = 0 <= v293;
                                            bool v300;
                                            v300 = v299 && v294;
                                            bool v301;
                                            v301 = v300 == false;
                                            if (v301){
                                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v300);
                                            } else {
                                            }
                                            bool v303;
                                            v303 = 0 <= v297;
                                            bool v305;
                                            if (v303){
                                                bool v304;
                                                v304 = v297 < 16;
                                                v305 = v304;
                                            } else {
                                                v305 = false;
                                            }
                                            bool v306;
                                            v306 = v305 == false;
                                            if (v306){
                                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v305);
                                            } else {
                                            }
                                            int v308;
                                            v308 = v297 * 16;
                                            int v309;
                                            v309 = v308 + v293;
                                            assert("Tensor range check" && 0 <= v297 && v297 < 16);
                                            int v310;
                                            v310 = 16 * v297;
                                            int v311;
                                            v311 = v310 + v293;
                                            float * v312;
                                            v312 = v282[v311];
                                            int v313;
                                            v313 = blockIdx.x;
                                            int v314;
                                            v314 = v313 * 256;
                                            int v315;
                                            v315 = v314 + v309;
                                            assert("Tensor range check" && 0 <= v292 && v292 < 16);
                                            int v316;
                                            v316 = 4 * v292;
                                            float v317[4];
                                            int v318[4];
                                            int v319;
                                            v319 = 0;
                                            while (while_method_5(v319)){
                                                assert("Tensor range check" && 0 <= v319 && v319 < 1);
                                                int v321;
                                                v321 = 4 * v319;
                                                assert("Tensor range check" && 0 <= v319 && v319 < 1);
                                                int v322;
                                                v322 = 64 * v319;
                                                int v323;
                                                v323 = v322 + v316;
                                                int4* v324;
                                                v324 = reinterpret_cast<int4*>(v312 + v323);
                                                int4* v325;
                                                v325 = reinterpret_cast<int4*>(v317 + v321);
                                                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v324) % 16 == 0 && reinterpret_cast<unsigned long long>(v325) % 16 == 0);
                                                *v325 = *v324;
                                                v319 += 1 ;
                                            }
                                            int v326;
                                            v326 = 0;
                                            while (while_method_5(v326)){
                                                int v328;
                                                v328 = 0;
                                                while (while_method_8(v328)){
                                                    bool v330;
                                                    v330 = 0 <= v328;
                                                    bool v332;
                                                    if (v330){
                                                        bool v331;
                                                        v331 = v328 < 4;
                                                        v332 = v331;
                                                    } else {
                                                        v332 = false;
                                                    }
                                                    bool v333;
                                                    v333 = v332 == false;
                                                    if (v333){
                                                        assert("The indices should be inside the range of the dimension." && v332);
                                                    } else {
                                                    }
                                                    bool v335;
                                                    v335 = 0 <= v292;
                                                    bool v337;
                                                    if (v335){
                                                        bool v336;
                                                        v336 = v292 < 16;
                                                        v337 = v336;
                                                    } else {
                                                        v337 = false;
                                                    }
                                                    bool v338;
                                                    v338 = v337 == false;
                                                    if (v338){
                                                        assert("The indices should be inside the range of the dimension." && v337);
                                                    } else {
                                                    }
                                                    int v340;
                                                    v340 = v292 * 4;
                                                    int v341;
                                                    v341 = v328 + v340;
                                                    bool v342;
                                                    v342 = 0 <= v326;
                                                    bool v344;
                                                    if (v342){
                                                        bool v343;
                                                        v343 = v326 < 1;
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
                                                    v347 = v326 * 64;
                                                    int v348;
                                                    v348 = v341 + v347;
                                                    assert("Tensor range check" && 0 <= v326 && v326 < 1);
                                                    assert("Tensor range check" && 0 <= v328 && v328 < 4);
                                                    int v349;
                                                    v349 = 4 * v326;
                                                    int v350;
                                                    v350 = v349 + v328;
                                                    v318[v350] = v348;
                                                    v328 += 1 ;
                                                }
                                                v326 += 1 ;
                                            }
                                            float v351[4];
                                            float v352;
                                            v352 = 0.0f;
                                            int v353;
                                            v353 = 0;
                                            while (while_method_5(v353)){
                                                assert("Tensor range check" && 0 <= v353 && v353 < 1);
                                                int v355;
                                                v355 = 4 * v353;
                                                assert("Tensor range check" && 0 <= v353 && v353 < 1);
                                                float v356;
                                                v356 = 0.0f;
                                                int v357;
                                                v357 = 0;
                                                while (while_method_8(v357)){
                                                    assert("Tensor range check" && 0 <= v357 && v357 < 4);
                                                    int v359;
                                                    v359 = v357 + v355;
                                                    float v360;
                                                    v360 = v317[v359];
                                                    float v361;
                                                    v361 = v356 + v360;
                                                    v356 = v361;
                                                    v357 += 1 ;
                                                }
                                                auto v362 = cooperative_groups::coalesced_threads();
                                                int v363;
                                                v363 = threadIdx.x;
                                                int v364;
                                                v364 = v363 / 16;
                                                auto v365 = cooperative_groups::labeled_partition(v362,v364);
                                                Closure2 v366{};
                                                float v367;
                                                v367 = cooperative_groups::inclusive_scan(v365, v356, v366);
                                                float v368;
                                                v368 = v365.shfl_up(v367,1);
                                                bool v369;
                                                v369 = v365.thread_rank() == 0;
                                                float v370;
                                                if (v369){
                                                    v370 = 0.0f;
                                                } else {
                                                    v370 = v368;
                                                }
                                                float v371;
                                                v371 = v365.shfl(v367,v365.num_threads()-1);
                                                float v372;
                                                v372 = v352 + v370;
                                                float v373;
                                                v373 = v372;
                                                int v374;
                                                v374 = 0;
                                                while (while_method_8(v374)){
                                                    assert("Tensor range check" && 0 <= v374 && v374 < 4);
                                                    int v376;
                                                    v376 = v374 + v355;
                                                    float v377;
                                                    v377 = v317[v376];
                                                    float v378;
                                                    v378 = v373 + v377;
                                                    assert("Tensor range check" && 0 <= v374 && v374 < 4);
                                                    v351[v376] = v378;
                                                    v373 = v378;
                                                    v374 += 1 ;
                                                }
                                                float v379;
                                                v379 = v352 + v371;
                                                v352 = v379;
                                                v353 += 1 ;
                                            }
                                            float v380[4];
                                            bool v381[4];
                                            int v382;
                                            v382 = 0;
                                            while (while_method_5(v382)){
                                                int v384;
                                                v384 = 0;
                                                while (while_method_8(v384)){
                                                    assert("Tensor range check" && 0 <= v382 && v382 < 1);
                                                    assert("Tensor range check" && 0 <= v384 && v384 < 4);
                                                    int v386;
                                                    v386 = 4 * v382;
                                                    int v387;
                                                    v387 = v386 + v384;
                                                    float v388;
                                                    v388 = v351[v387];
                                                    float v389;
                                                    v389 = v317[v387];
                                                    bool v390;
                                                    v390 = v389 > 0.0f;
                                                    assert("Tensor range check" && 0 <= v382 && v382 < 1);
                                                    assert("Tensor range check" && 0 <= v384 && v384 < 4);
                                                    v380[v387] = v388;
                                                    v381[v387] = v390;
                                                    v384 += 1 ;
                                                }
                                                v382 += 1 ;
                                            }
                                            float v391; bool v392;
                                            Tuple8 tmp15 = Tuple8{-1.0f / 0.0f, false};
                                            v391 = tmp15.v0; v392 = tmp15.v1;
                                            int v393;
                                            v393 = 0;
                                            while (while_method_5(v393)){
                                                int v395;
                                                v395 = 0;
                                                while (while_method_8(v395)){
                                                    assert("Tensor range check" && 0 <= v393 && v393 < 1);
                                                    assert("Tensor range check" && 0 <= v395 && v395 < 4);
                                                    int v397;
                                                    v397 = 4 * v393;
                                                    int v398;
                                                    v398 = v397 + v395;
                                                    float v399;
                                                    v399 = v380[v398];
                                                    bool v400;
                                                    v400 = v381[v398];
                                                    float v407; bool v408;
                                                    if (v392){
                                                        if (v400){
                                                            bool v401;
                                                            v401 = v391 >= v399;
                                                            float v402;
                                                            if (v401){
                                                                v402 = v391;
                                                            } else {
                                                                v402 = v399;
                                                            }
                                                            v407 = v402; v408 = true;
                                                        } else {
                                                            v407 = v391; v408 = v392;
                                                        }
                                                    } else {
                                                        if (v400){
                                                            v407 = v399; v408 = v400;
                                                        } else {
                                                            v407 = v391; v408 = v392;
                                                        }
                                                    }
                                                    v391 = v407;
                                                    v392 = v408;
                                                    v395 += 1 ;
                                                }
                                                v393 += 1 ;
                                            }
                                            auto v409 = cooperative_groups::coalesced_threads();
                                            int v410;
                                            v410 = threadIdx.x;
                                            int v411;
                                            v411 = v410 / 16;
                                            auto v412 = cooperative_groups::labeled_partition(v409,v411);
                                            Closure3 v413{};
                                            float v414; bool v415;
                                            Tuple8 tmp16 = cooperative_groups::reduce(v412, Tuple8{v391, v392}, v413);
                                            v414 = tmp16.v0; v415 = tmp16.v1;
                                            bool v416;
                                            v416 = v415 == false;
                                            if (v416){
                                                int v417;
                                                v417 = threadIdx.x;
                                                int v418;
                                                v418 = blockIdx.x;
                                                int v419;
                                                v419 = v418 * 256;
                                                int v420;
                                                v420 = v417 + v419;
                                                cuda::counting_semaphore<cuda::thread_scope_system, 1> & v421 = console_lock;
                                                auto v422 = cooperative_groups::coalesced_threads();
                                                v421.acquire();
                                                int v423;
                                                v423 = 0;
                                                printf("{%s = %d; %s = %c","tid", v420, "x'", '[');
                                                int v424;
                                                v424 = 0;
                                                while (while_method_5(v424)){
                                                    int v426;
                                                    v426 = v423;
                                                    bool v427;
                                                    v427 = v426 >= 100;
                                                    if (v427){
                                                        printf("%s"," ...");
                                                        break;
                                                    } else {
                                                    }
                                                    bool v428;
                                                    v428 = v424 == 0;
                                                    bool v429;
                                                    v429 = v428 != true;
                                                    if (v429){
                                                        printf("%s","; ");
                                                    } else {
                                                    }
                                                    printf("%c",'[');
                                                    int v430;
                                                    v430 = 0;
                                                    while (while_method_8(v430)){
                                                        int v432;
                                                        v432 = v423;
                                                        bool v433;
                                                        v433 = v432 >= 100;
                                                        if (v433){
                                                            printf("%s"," ...");
                                                            break;
                                                        } else {
                                                        }
                                                        bool v434;
                                                        v434 = v430 == 0;
                                                        bool v435;
                                                        v435 = v434 != true;
                                                        if (v435){
                                                            printf("%s","; ");
                                                        } else {
                                                        }
                                                        int v436;
                                                        v436 = v423 + 1;
                                                        v423 = v436;
                                                        int v437;
                                                        v437 = v424 * 4;
                                                        int v438;
                                                        v438 = v437 + v430;
                                                        float v439;
                                                        v439 = v380[v438];
                                                        bool v440;
                                                        v440 = v381[v438];
                                                        const char * v443;
                                                        if (v440){
                                                            const char * v441;
                                                            v441 = "true";
                                                            v443 = v441;
                                                        } else {
                                                            const char * v442;
                                                            v442 = "false";
                                                            v443 = v442;
                                                        }
                                                        printf("%f, %s",v439, v443);
                                                        v430 += 1 ;
                                                    }
                                                    printf("%c",']');
                                                    v424 += 1 ;
                                                }
                                                printf("%c",']');
                                                printf("}\n");
                                                v421.release();
                                                v422.sync() ;
                                            } else {
                                            }
                                            if (v416){
                                                assert("The local reduce must be true." && v415);
                                            } else {
                                            }
                                            float v479[4];
                                            int v480[4];
                                            int v481;
                                            v481 = 0;
                                            while (while_method_5(v481)){
                                                int v483;
                                                v483 = 0;
                                                while (while_method_8(v483)){
                                                    assert("Tensor range check" && 0 <= v481 && v481 < 1);
                                                    assert("Tensor range check" && 0 <= v483 && v483 < 4);
                                                    int v485;
                                                    v485 = 4 * v481;
                                                    int v486;
                                                    v486 = v485 + v483;
                                                    int v487;
                                                    v487 = v318[v486];
                                                    float v488;
                                                    v488 = curand_uniform(&v121);
                                                    assert("Tensor range check" && 0 <= v481 && v481 < 1);
                                                    assert("Tensor range check" && 0 <= v483 && v483 < 4);
                                                    v479[v486] = v488;
                                                    v480[v486] = v487;
                                                    v483 += 1 ;
                                                }
                                                v481 += 1 ;
                                            }
                                            float v489; int v490;
                                            Tuple9 tmp17 = Tuple9{0.0f, 2147483647};
                                            v489 = tmp17.v0; v490 = tmp17.v1;
                                            int v491;
                                            v491 = 0;
                                            while (while_method_5(v491)){
                                                int v493;
                                                v493 = 0;
                                                while (while_method_8(v493)){
                                                    assert("Tensor range check" && 0 <= v491 && v491 < 1);
                                                    assert("Tensor range check" && 0 <= v493 && v493 < 4);
                                                    int v495;
                                                    v495 = 4 * v491;
                                                    int v496;
                                                    v496 = v495 + v493;
                                                    float v497;
                                                    v497 = v479[v496];
                                                    int v498;
                                                    v498 = v480[v496];
                                                    bool v499;
                                                    v499 = v490 < v498;
                                                    float v500; int v501;
                                                    if (v499){
                                                        v500 = v489; v501 = v490;
                                                    } else {
                                                        v500 = v497; v501 = v498;
                                                    }
                                                    v489 = v500;
                                                    v490 = v501;
                                                    v493 += 1 ;
                                                }
                                                v491 += 1 ;
                                            }
                                            auto v502 = cooperative_groups::coalesced_threads();
                                            int v503;
                                            v503 = threadIdx.x;
                                            int v504;
                                            v504 = v503 / 16;
                                            auto v505 = cooperative_groups::labeled_partition(v502,v504);
                                            Closure4 v506{};
                                            float v507; int v508;
                                            Tuple9 tmp18 = cooperative_groups::reduce(v505, Tuple9{v489, v490}, v506);
                                            v507 = tmp18.v0; v508 = tmp18.v1;
                                            float v509;
                                            v509 = v414 * v507;
                                            int v510[4];
                                            bool v511[4];
                                            int v512;
                                            v512 = 0;
                                            while (while_method_5(v512)){
                                                int v514;
                                                v514 = 0;
                                                while (while_method_8(v514)){
                                                    assert("Tensor range check" && 0 <= v512 && v512 < 1);
                                                    assert("Tensor range check" && 0 <= v514 && v514 < 4);
                                                    int v516;
                                                    v516 = 4 * v512;
                                                    int v517;
                                                    v517 = v516 + v514;
                                                    float v518;
                                                    v518 = v380[v517];
                                                    bool v519;
                                                    v519 = v381[v517];
                                                    int v520;
                                                    v520 = v318[v517];
                                                    int v523; bool v524;
                                                    if (v519){
                                                        float v521;
                                                        v521 = v518 - v509;
                                                        bool v522;
                                                        v522 = v521 >= 0.0f;
                                                        v523 = v520; v524 = v522;
                                                    } else {
                                                        v523 = 2147483647; v524 = false;
                                                    }
                                                    assert("Tensor range check" && 0 <= v512 && v512 < 1);
                                                    assert("Tensor range check" && 0 <= v514 && v514 < 4);
                                                    v510[v517] = v523;
                                                    v511[v517] = v524;
                                                    v514 += 1 ;
                                                }
                                                v512 += 1 ;
                                            }
                                            int v525; bool v526;
                                            Tuple10 tmp19 = Tuple10{2147483647, false};
                                            v525 = tmp19.v0; v526 = tmp19.v1;
                                            int v527;
                                            v527 = 0;
                                            while (while_method_5(v527)){
                                                int v529;
                                                v529 = 0;
                                                while (while_method_8(v529)){
                                                    assert("Tensor range check" && 0 <= v527 && v527 < 1);
                                                    assert("Tensor range check" && 0 <= v529 && v529 < 4);
                                                    int v531;
                                                    v531 = 4 * v527;
                                                    int v532;
                                                    v532 = v531 + v529;
                                                    int v533;
                                                    v533 = v510[v532];
                                                    bool v534;
                                                    v534 = v511[v532];
                                                    int v541; bool v542;
                                                    if (v526){
                                                        if (v534){
                                                            bool v535;
                                                            v535 = v525 < v533;
                                                            int v536;
                                                            if (v535){
                                                                v536 = v525;
                                                            } else {
                                                                v536 = v533;
                                                            }
                                                            v541 = v536; v542 = true;
                                                        } else {
                                                            v541 = v525; v542 = v526;
                                                        }
                                                    } else {
                                                        if (v534){
                                                            v541 = v533; v542 = v534;
                                                        } else {
                                                            v541 = v525; v542 = v526;
                                                        }
                                                    }
                                                    v525 = v541;
                                                    v526 = v542;
                                                    v529 += 1 ;
                                                }
                                                v527 += 1 ;
                                            }
                                            auto v543 = cooperative_groups::coalesced_threads();
                                            int v544;
                                            v544 = threadIdx.x;
                                            int v545;
                                            v545 = v544 / 16;
                                            auto v546 = cooperative_groups::labeled_partition(v543,v545);
                                            Closure5 v547{};
                                            int v548; bool v549;
                                            Tuple10 tmp20 = cooperative_groups::reduce(v546, Tuple10{v525, v526}, v547);
                                            v548 = tmp20.v0; v549 = tmp20.v1;
                                            bool v550;
                                            v550 = v549 == false;
                                            if (v550){
                                                int v551;
                                                v551 = threadIdx.x;
                                                int v552;
                                                v552 = blockIdx.x;
                                                int v553;
                                                v553 = v552 * 256;
                                                int v554;
                                                v554 = v551 + v553;
                                                cuda::counting_semaphore<cuda::thread_scope_system, 1> & v555 = console_lock;
                                                auto v556 = cooperative_groups::coalesced_threads();
                                                v555.acquire();
                                                int v557;
                                                v557 = 0;
                                                printf("{%s = %d; %s = %c","tid", v554, "x'", '[');
                                                int v558;
                                                v558 = 0;
                                                while (while_method_5(v558)){
                                                    int v560;
                                                    v560 = v557;
                                                    bool v561;
                                                    v561 = v560 >= 100;
                                                    if (v561){
                                                        printf("%s"," ...");
                                                        break;
                                                    } else {
                                                    }
                                                    bool v562;
                                                    v562 = v558 == 0;
                                                    bool v563;
                                                    v563 = v562 != true;
                                                    if (v563){
                                                        printf("%s","; ");
                                                    } else {
                                                    }
                                                    printf("%c",'[');
                                                    int v564;
                                                    v564 = 0;
                                                    while (while_method_8(v564)){
                                                        int v566;
                                                        v566 = v557;
                                                        bool v567;
                                                        v567 = v566 >= 100;
                                                        if (v567){
                                                            printf("%s"," ...");
                                                            break;
                                                        } else {
                                                        }
                                                        bool v568;
                                                        v568 = v564 == 0;
                                                        bool v569;
                                                        v569 = v568 != true;
                                                        if (v569){
                                                            printf("%s","; ");
                                                        } else {
                                                        }
                                                        int v570;
                                                        v570 = v557 + 1;
                                                        v557 = v570;
                                                        int v571;
                                                        v571 = v558 * 4;
                                                        int v572;
                                                        v572 = v571 + v564;
                                                        int v573;
                                                        v573 = v510[v572];
                                                        bool v574;
                                                        v574 = v511[v572];
                                                        const char * v577;
                                                        if (v574){
                                                            const char * v575;
                                                            v575 = "true";
                                                            v577 = v575;
                                                        } else {
                                                            const char * v576;
                                                            v576 = "false";
                                                            v577 = v576;
                                                        }
                                                        printf("%d, %s",v573, v577);
                                                        v564 += 1 ;
                                                    }
                                                    printf("%c",']');
                                                    v558 += 1 ;
                                                }
                                                printf("%c",']');
                                                printf("}\n");
                                                v555.release();
                                                v556.sync() ;
                                            } else {
                                            }
                                            if (v550){
                                                assert("The local reduce must be true." && v549);
                                            } else {
                                            }
                                            float v613; int v614;
                                            Tuple9 tmp21 = Tuple9{0.0f, 2147483647};
                                            v613 = tmp21.v0; v614 = tmp21.v1;
                                            int v615;
                                            v615 = 0;
                                            while (while_method_5(v615)){
                                                int v617;
                                                v617 = 0;
                                                while (while_method_8(v617)){
                                                    assert("Tensor range check" && 0 <= v615 && v615 < 1);
                                                    assert("Tensor range check" && 0 <= v617 && v617 < 4);
                                                    int v619;
                                                    v619 = 4 * v615;
                                                    int v620;
                                                    v620 = v619 + v617;
                                                    float v621;
                                                    v621 = v317[v620];
                                                    int v622;
                                                    v622 = v318[v620];
                                                    bool v623;
                                                    v623 = v614 == v548;
                                                    float v627; int v628;
                                                    if (v623){
                                                        v627 = v613; v628 = v614;
                                                    } else {
                                                        bool v624;
                                                        v624 = v622 == v548;
                                                        if (v624){
                                                            v627 = v621; v628 = v622;
                                                        } else {
                                                            v627 = v613; v628 = v614;
                                                        }
                                                    }
                                                    v613 = v627;
                                                    v614 = v628;
                                                    v617 += 1 ;
                                                }
                                                v615 += 1 ;
                                            }
                                            auto v629 = cooperative_groups::coalesced_threads();
                                            int v630;
                                            v630 = threadIdx.x;
                                            int v631;
                                            v631 = v630 / 16;
                                            auto v632 = cooperative_groups::labeled_partition(v629,v631);
                                            Closure6 v633{v548};
                                            float v634; int v635;
                                            Tuple9 tmp22 = cooperative_groups::reduce(v632, Tuple9{v613, v614}, v633);
                                            v634 = tmp22.v0; v635 = tmp22.v1;
                                            bool v636;
                                            v636 = v635 == 2147483647;
                                            bool v637;
                                            v637 = v636 != true;
                                            bool v638;
                                            v638 = v637 == false;
                                            if (v638){
                                                assert("Expected a valid action id in get_prob." && v637);
                                            } else {
                                            }
                                            int v640;
                                            v640 = 0;
                                            while (while_method_5(v640)){
                                                assert("Tensor range check" && 0 <= v640 && v640 < 1);
                                                assert("Tensor range check" && 0 <= v640 && v640 < 1);
                                                v640 += 1 ;
                                            }
                                            assert("Tensor range check" && 0 <= v309 && v309 < 256);
                                            v284[v309] = v634;
                                            v286[v309] = v548;
                                            v297 += 1 ;
                                        }
                                        __syncthreads();
                                        assert("Tensor range check" && 0 <= v288 && v288 < 256);
                                        float v642;
                                        v642 = v284[v288];
                                        int v643;
                                        v643 = v286[v288];
                                        __syncthreads();
                                        bool v644;
                                        v644 = 0 == v643;
                                        Union12 v653;
                                        if (v644){
                                            v653 = Union12{Union12_1{}};
                                        } else {
                                            bool v646;
                                            v646 = 1 == v643;
                                            if (v646){
                                                v653 = Union12{Union12_0{}};
                                            } else {
                                                bool v648;
                                                v648 = 2 == v643;
                                                if (v648){
                                                    v653 = Union12{Union12_2{}};
                                                } else {
                                                    printf("%s\n", "Invalid output id in the Leduc model.");
                                                    __trap();
                                                }
                                            }
                                        }
                                        Union1 v685;
                                        switch (v653.tag) {
                                            case 0: { // AA_Call
                                                v685 = Union1{Union1_0{}};
                                                break;
                                            }
                                            case 1: { // AA_Fold
                                                int v654;
                                                v654 = v109[0];
                                                int v656; int v657;
                                                Tuple7 tmp23 = Tuple7{1, v654};
                                                v656 = tmp23.v0; v657 = tmp23.v1;
                                                while (while_method_0(v656)){
                                                    bool v659;
                                                    v659 = 0 <= v656;
                                                    bool v661;
                                                    if (v659){
                                                        bool v660;
                                                        v660 = v656 < 2;
                                                        v661 = v660;
                                                    } else {
                                                        v661 = false;
                                                    }
                                                    bool v662;
                                                    v662 = v661 == false;
                                                    if (v662){
                                                        assert("Index must be in range." && v661);
                                                    } else {
                                                    }
                                                    int v664;
                                                    v664 = v109[v656];
                                                    bool v666;
                                                    v666 = v657 >= v664;
                                                    int v667;
                                                    if (v666){
                                                        v667 = v657;
                                                    } else {
                                                        v667 = v664;
                                                    }
                                                    v657 = v667;
                                                    v656 += 1 ;
                                                }
                                                bool v669;
                                                if (v112){
                                                    bool v668;
                                                    v668 = v108 < 2;
                                                    v669 = v668;
                                                } else {
                                                    v669 = false;
                                                }
                                                bool v670;
                                                v670 = v669 == false;
                                                if (v670){
                                                    assert("Index must be in range." && v669);
                                                } else {
                                                }
                                                int v672;
                                                v672 = v109[v108];
                                                bool v674;
                                                v674 = v672 == v657;
                                                if (v674){
                                                    v685 = Union1{Union1_0{}};
                                                } else {
                                                    v685 = Union1{Union1_1{}};
                                                }
                                                break;
                                            }
                                            case 2: { // AA_Raise
                                                bool v679;
                                                v679 = v110 > 0;
                                                if (v679){
                                                    v685 = Union1{Union1_2{}};
                                                } else {
                                                    v685 = Union1{Union1_0{}};
                                                }
                                                break;
                                            }
                                            default: {
                                                assert("Invalid tag." && false); __trap();
                                            }
                                        }
                                        int v686;
                                        v686 = sizeof(Union1);
                                        unsigned long long v687;
                                        v687 = (unsigned long long)v686;
                                        bool v688;
                                        v688 = v687 <= 98304ull;
                                        bool v689;
                                        v689 = v688 == false;
                                        if (v689){
                                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v688);
                                        } else {
                                        }
                                        extern __shared__ unsigned char v691[];
                                        bool v692;
                                        v692 = v687 <= v687;
                                        bool v693;
                                        v693 = v692 == false;
                                        if (v693){
                                            assert("The length of the partition has to be less than or equal to the length of the base array." && v692);
                                        } else {
                                        }
                                        Union1 * v695;
                                        v695 = reinterpret_cast<Union1 *>(&v691[0ull]);
                                        int v697;
                                        v697 = threadIdx.x;
                                        bool v698;
                                        v698 = v697 == 0;
                                        if (v698){
                                            v695[0] = v685;
                                        } else {
                                        }
                                        __syncthreads();
                                        Union1 v699;
                                        v699 = v695[0];
                                        __syncthreads();
                                        Union7 v700;
                                        v700 = Union7{Union7_1{v108, v699}};
                                        v58.push(v700);
                                        Union4 v807;
                                        switch (v105.tag) {
                                            case 0: { // None
                                                switch (v699.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v763;
                                                            v763 = v108 ^ 1;
                                                            v807 = Union4{Union4_2{v105, false, v107, v763, v109, v110}};
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
                                                        bool v767;
                                                        v767 = v110 > 0;
                                                        if (v767){
                                                            int v768;
                                                            v768 = v108 ^ 1;
                                                            int v769;
                                                            v769 = -1 + v110;
                                                            int v770; int v771;
                                                            Tuple7 tmp24 = Tuple7{0, 0};
                                                            v770 = tmp24.v0; v771 = tmp24.v1;
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
                                                Union6 v701 = v105.case1.v0;
                                                switch (v699.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v703;
                                                            v703 = v108 ^ 1;
                                                            v807 = Union4{Union4_2{v105, false, v107, v703, v109, v110}};
                                                        } else {
                                                            int v705; int v706;
                                                            Tuple7 tmp25 = Tuple7{0, 0};
                                                            v705 = tmp25.v0; v706 = tmp25.v1;
                                                            while (while_method_0(v705)){
                                                                bool v708;
                                                                v708 = 0 <= v705;
                                                                bool v710;
                                                                if (v708){
                                                                    bool v709;
                                                                    v709 = v705 < 2;
                                                                    v710 = v709;
                                                                } else {
                                                                    v710 = false;
                                                                }
                                                                bool v711;
                                                                v711 = v710 == false;
                                                                if (v711){
                                                                    assert("Index must be in range." && v710);
                                                                } else {
                                                                }
                                                                int v713;
                                                                v713 = v109[v705];
                                                                bool v715;
                                                                v715 = v706 >= v713;
                                                                int v716;
                                                                if (v715){
                                                                    v716 = v706;
                                                                } else {
                                                                    v716 = v713;
                                                                }
                                                                v706 = v716;
                                                                v705 += 1 ;
                                                            }
                                                            static_array<int,2> v717;
                                                            int v719;
                                                            v719 = 0;
                                                            while (while_method_0(v719)){
                                                                v717[v719] = v706;
                                                                v719 += 1 ;
                                                            }
                                                            v807 = Union4{Union4_4{v105, v106, v107, v108, v717, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v807 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        bool v723;
                                                        v723 = v110 > 0;
                                                        if (v723){
                                                            int v724;
                                                            v724 = v108 ^ 1;
                                                            int v725;
                                                            v725 = -1 + v110;
                                                            int v726; int v727;
                                                            Tuple7 tmp26 = Tuple7{0, 0};
                                                            v726 = tmp26.v0; v727 = tmp26.v1;
                                                            while (while_method_0(v726)){
                                                                bool v729;
                                                                v729 = 0 <= v726;
                                                                bool v731;
                                                                if (v729){
                                                                    bool v730;
                                                                    v730 = v726 < 2;
                                                                    v731 = v730;
                                                                } else {
                                                                    v731 = false;
                                                                }
                                                                bool v732;
                                                                v732 = v731 == false;
                                                                if (v732){
                                                                    assert("Index must be in range." && v731);
                                                                } else {
                                                                }
                                                                int v734;
                                                                v734 = v109[v726];
                                                                bool v736;
                                                                v736 = v727 >= v734;
                                                                int v737;
                                                                if (v736){
                                                                    v737 = v727;
                                                                } else {
                                                                    v737 = v734;
                                                                }
                                                                v727 = v737;
                                                                v726 += 1 ;
                                                            }
                                                            static_array<int,2> v738;
                                                            int v740;
                                                            v740 = 0;
                                                            while (while_method_0(v740)){
                                                                v738[v740] = v727;
                                                                v740 += 1 ;
                                                            }
                                                            static_array<int,2> v742;
                                                            int v744;
                                                            v744 = 0;
                                                            while (while_method_0(v744)){
                                                                bool v746;
                                                                v746 = 0 <= v744;
                                                                bool v748;
                                                                if (v746){
                                                                    bool v747;
                                                                    v747 = v744 < 2;
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
                                                                v751 = v738[v744];
                                                                bool v753;
                                                                v753 = v744 == v108;
                                                                int v755;
                                                                if (v753){
                                                                    int v754;
                                                                    v754 = v751 + 4;
                                                                    v755 = v754;
                                                                } else {
                                                                    v755 = v751;
                                                                }
                                                                v742[v744] = v755;
                                                                v744 += 1 ;
                                                            }
                                                            v807 = Union4{Union4_2{v105, false, v107, v724, v742, v725}};
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
                                        v1138 = Union3{Union3_1{v807}};
                                        break;
                                    }
                                    case 1: { // Human
                                        Union8 v809;
                                        v809 = Union8{Union8_2{v105, v106, v107, v108, v109, v110}};
                                        v19.v5 = v809;
                                        Union3 v810;
                                        v810 = Union3{Union3_1{v62}};
                                        v19.v1 = v810;
                                        v1138 = Union3{Union3_0{}};
                                        break;
                                    }
                                    case 2: { // Random
                                        curandStatePhilox4_32_10_t & v812 = v19.v4;
                                        curandStatePhilox4_32_10_t & v813 = v812;
                                        static_array_list<Union1,3> v814;
                                        v814 = static_array_list<Union1,3>{};
                                        v814.unsafe_set_length(1);
                                        Union1 v816;
                                        v816 = Union1{Union1_0{}};
                                        v814[0] = v816;
                                        int v818;
                                        v818 = v109[0];
                                        int v820;
                                        v820 = v109[1];
                                        bool v822;
                                        v822 = v818 == v820;
                                        bool v823;
                                        v823 = v822 != true;
                                        if (v823){
                                            Union1 v824;
                                            v824 = Union1{Union1_1{}};
                                            v814.push(v824);
                                        } else {
                                        }
                                        bool v825;
                                        v825 = v110 > 0;
                                        if (v825){
                                            Union1 v826;
                                            v826 = Union1{Union1_2{}};
                                            v814.push(v826);
                                        } else {
                                        }
                                        int v827;
                                        v827 = v814.length;
                                        int v828;
                                        v828 = v827 - 1;
                                        int v829;
                                        v829 = 0;
                                        while (while_method_1(v828, v829)){
                                            int v831;
                                            v831 = v814.length;
                                            int v832;
                                            v832 = int_range_22(v831, v829, v813);
                                            Union1 v833;
                                            v833 = v814[v829];
                                            Union1 v835;
                                            v835 = v814[v832];
                                            v814[v829] = v835;
                                            v814[v832] = v833;
                                            v829 += 1 ;
                                        }
                                        Union1 v837;
                                        v837 = v814.pop();
                                        int v838;
                                        v838 = sizeof(Union1);
                                        unsigned long long v839;
                                        v839 = (unsigned long long)v838;
                                        bool v840;
                                        v840 = v839 <= 98304ull;
                                        bool v841;
                                        v841 = v840 == false;
                                        if (v841){
                                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v840);
                                        } else {
                                        }
                                        extern __shared__ unsigned char v843[];
                                        bool v844;
                                        v844 = v839 <= v839;
                                        bool v845;
                                        v845 = v844 == false;
                                        if (v845){
                                            assert("The length of the partition has to be less than or equal to the length of the base array." && v844);
                                        } else {
                                        }
                                        Union1 * v847;
                                        v847 = reinterpret_cast<Union1 *>(&v843[0ull]);
                                        int v849;
                                        v849 = threadIdx.x;
                                        bool v850;
                                        v850 = v849 == 0;
                                        if (v850){
                                            v847[0] = v837;
                                        } else {
                                        }
                                        __syncthreads();
                                        Union1 v851;
                                        v851 = v847[0];
                                        __syncthreads();
                                        Union7 v852;
                                        v852 = Union7{Union7_1{v108, v851}};
                                        v58.push(v852);
                                        Union4 v957;
                                        switch (v105.tag) {
                                            case 0: { // None
                                                switch (v851.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v914;
                                                            v914 = v108 ^ 1;
                                                            v957 = Union4{Union4_2{v105, false, v107, v914, v109, v110}};
                                                        } else {
                                                            v957 = Union4{Union4_0{v105, v106, v107, v108, v109, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v957 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        if (v825){
                                                            int v918;
                                                            v918 = v108 ^ 1;
                                                            int v919;
                                                            v919 = -1 + v110;
                                                            int v920; int v921;
                                                            Tuple7 tmp27 = Tuple7{0, 0};
                                                            v920 = tmp27.v0; v921 = tmp27.v1;
                                                            while (while_method_0(v920)){
                                                                bool v923;
                                                                v923 = 0 <= v920;
                                                                bool v925;
                                                                if (v923){
                                                                    bool v924;
                                                                    v924 = v920 < 2;
                                                                    v925 = v924;
                                                                } else {
                                                                    v925 = false;
                                                                }
                                                                bool v926;
                                                                v926 = v925 == false;
                                                                if (v926){
                                                                    assert("Index must be in range." && v925);
                                                                } else {
                                                                }
                                                                int v928;
                                                                v928 = v109[v920];
                                                                bool v930;
                                                                v930 = v921 >= v928;
                                                                int v931;
                                                                if (v930){
                                                                    v931 = v921;
                                                                } else {
                                                                    v931 = v928;
                                                                }
                                                                v921 = v931;
                                                                v920 += 1 ;
                                                            }
                                                            static_array<int,2> v932;
                                                            int v934;
                                                            v934 = 0;
                                                            while (while_method_0(v934)){
                                                                v932[v934] = v921;
                                                                v934 += 1 ;
                                                            }
                                                            static_array<int,2> v936;
                                                            int v938;
                                                            v938 = 0;
                                                            while (while_method_0(v938)){
                                                                bool v940;
                                                                v940 = 0 <= v938;
                                                                bool v942;
                                                                if (v940){
                                                                    bool v941;
                                                                    v941 = v938 < 2;
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
                                                                v945 = v932[v938];
                                                                bool v947;
                                                                v947 = v938 == v108;
                                                                int v949;
                                                                if (v947){
                                                                    int v948;
                                                                    v948 = v945 + 2;
                                                                    v949 = v948;
                                                                } else {
                                                                    v949 = v945;
                                                                }
                                                                v936[v938] = v949;
                                                                v938 += 1 ;
                                                            }
                                                            v957 = Union4{Union4_2{v105, false, v107, v918, v936, v919}};
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
                                                Union6 v853 = v105.case1.v0;
                                                switch (v851.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v855;
                                                            v855 = v108 ^ 1;
                                                            v957 = Union4{Union4_2{v105, false, v107, v855, v109, v110}};
                                                        } else {
                                                            int v857; int v858;
                                                            Tuple7 tmp28 = Tuple7{0, 0};
                                                            v857 = tmp28.v0; v858 = tmp28.v1;
                                                            while (while_method_0(v857)){
                                                                bool v860;
                                                                v860 = 0 <= v857;
                                                                bool v862;
                                                                if (v860){
                                                                    bool v861;
                                                                    v861 = v857 < 2;
                                                                    v862 = v861;
                                                                } else {
                                                                    v862 = false;
                                                                }
                                                                bool v863;
                                                                v863 = v862 == false;
                                                                if (v863){
                                                                    assert("Index must be in range." && v862);
                                                                } else {
                                                                }
                                                                int v865;
                                                                v865 = v109[v857];
                                                                bool v867;
                                                                v867 = v858 >= v865;
                                                                int v868;
                                                                if (v867){
                                                                    v868 = v858;
                                                                } else {
                                                                    v868 = v865;
                                                                }
                                                                v858 = v868;
                                                                v857 += 1 ;
                                                            }
                                                            static_array<int,2> v869;
                                                            int v871;
                                                            v871 = 0;
                                                            while (while_method_0(v871)){
                                                                v869[v871] = v858;
                                                                v871 += 1 ;
                                                            }
                                                            v957 = Union4{Union4_4{v105, v106, v107, v108, v869, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v957 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        if (v825){
                                                            int v875;
                                                            v875 = v108 ^ 1;
                                                            int v876;
                                                            v876 = -1 + v110;
                                                            int v877; int v878;
                                                            Tuple7 tmp29 = Tuple7{0, 0};
                                                            v877 = tmp29.v0; v878 = tmp29.v1;
                                                            while (while_method_0(v877)){
                                                                bool v880;
                                                                v880 = 0 <= v877;
                                                                bool v882;
                                                                if (v880){
                                                                    bool v881;
                                                                    v881 = v877 < 2;
                                                                    v882 = v881;
                                                                } else {
                                                                    v882 = false;
                                                                }
                                                                bool v883;
                                                                v883 = v882 == false;
                                                                if (v883){
                                                                    assert("Index must be in range." && v882);
                                                                } else {
                                                                }
                                                                int v885;
                                                                v885 = v109[v877];
                                                                bool v887;
                                                                v887 = v878 >= v885;
                                                                int v888;
                                                                if (v887){
                                                                    v888 = v878;
                                                                } else {
                                                                    v888 = v885;
                                                                }
                                                                v878 = v888;
                                                                v877 += 1 ;
                                                            }
                                                            static_array<int,2> v889;
                                                            int v891;
                                                            v891 = 0;
                                                            while (while_method_0(v891)){
                                                                v889[v891] = v878;
                                                                v891 += 1 ;
                                                            }
                                                            static_array<int,2> v893;
                                                            int v895;
                                                            v895 = 0;
                                                            while (while_method_0(v895)){
                                                                bool v897;
                                                                v897 = 0 <= v895;
                                                                bool v899;
                                                                if (v897){
                                                                    bool v898;
                                                                    v898 = v895 < 2;
                                                                    v899 = v898;
                                                                } else {
                                                                    v899 = false;
                                                                }
                                                                bool v900;
                                                                v900 = v899 == false;
                                                                if (v900){
                                                                    assert("Index must be in range." && v899);
                                                                } else {
                                                                }
                                                                int v902;
                                                                v902 = v889[v895];
                                                                bool v904;
                                                                v904 = v895 == v108;
                                                                int v906;
                                                                if (v904){
                                                                    int v905;
                                                                    v905 = v902 + 4;
                                                                    v906 = v905;
                                                                } else {
                                                                    v906 = v902;
                                                                }
                                                                v893[v895] = v906;
                                                                v895 += 1 ;
                                                            }
                                                            v957 = Union4{Union4_2{v105, false, v107, v875, v893, v876}};
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
                                        v1138 = Union3{Union3_1{v957}};
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                break;
                            }
                            case 3: { // RoundWithAction
                                Union5 v962 = v62.case3.v0; bool v963 = v62.case3.v1; static_array<Union6,2> v964 = v62.case3.v2; int v965 = v62.case3.v3; static_array<int,2> v966 = v62.case3.v4; int v967 = v62.case3.v5; Union1 v968 = v62.case3.v6;
                                Union7 v969;
                                v969 = Union7{Union7_1{v965, v968}};
                                v58.push(v969);
                                Union4 v1076;
                                switch (v962.tag) {
                                    case 0: { // None
                                        switch (v968.tag) {
                                            case 0: { // Call
                                                if (v963){
                                                    int v1032;
                                                    v1032 = v965 ^ 1;
                                                    v1076 = Union4{Union4_2{v962, false, v964, v1032, v966, v967}};
                                                } else {
                                                    v1076 = Union4{Union4_0{v962, v963, v964, v965, v966, v967}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v1076 = Union4{Union4_5{v962, v963, v964, v965, v966, v967}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                bool v1036;
                                                v1036 = v967 > 0;
                                                if (v1036){
                                                    int v1037;
                                                    v1037 = v965 ^ 1;
                                                    int v1038;
                                                    v1038 = -1 + v967;
                                                    int v1039; int v1040;
                                                    Tuple7 tmp30 = Tuple7{0, 0};
                                                    v1039 = tmp30.v0; v1040 = tmp30.v1;
                                                    while (while_method_0(v1039)){
                                                        bool v1042;
                                                        v1042 = 0 <= v1039;
                                                        bool v1044;
                                                        if (v1042){
                                                            bool v1043;
                                                            v1043 = v1039 < 2;
                                                            v1044 = v1043;
                                                        } else {
                                                            v1044 = false;
                                                        }
                                                        bool v1045;
                                                        v1045 = v1044 == false;
                                                        if (v1045){
                                                            assert("Index must be in range." && v1044);
                                                        } else {
                                                        }
                                                        int v1047;
                                                        v1047 = v966[v1039];
                                                        bool v1049;
                                                        v1049 = v1040 >= v1047;
                                                        int v1050;
                                                        if (v1049){
                                                            v1050 = v1040;
                                                        } else {
                                                            v1050 = v1047;
                                                        }
                                                        v1040 = v1050;
                                                        v1039 += 1 ;
                                                    }
                                                    static_array<int,2> v1051;
                                                    int v1053;
                                                    v1053 = 0;
                                                    while (while_method_0(v1053)){
                                                        v1051[v1053] = v1040;
                                                        v1053 += 1 ;
                                                    }
                                                    static_array<int,2> v1055;
                                                    int v1057;
                                                    v1057 = 0;
                                                    while (while_method_0(v1057)){
                                                        bool v1059;
                                                        v1059 = 0 <= v1057;
                                                        bool v1061;
                                                        if (v1059){
                                                            bool v1060;
                                                            v1060 = v1057 < 2;
                                                            v1061 = v1060;
                                                        } else {
                                                            v1061 = false;
                                                        }
                                                        bool v1062;
                                                        v1062 = v1061 == false;
                                                        if (v1062){
                                                            assert("Index must be in range." && v1061);
                                                        } else {
                                                        }
                                                        int v1064;
                                                        v1064 = v1051[v1057];
                                                        bool v1066;
                                                        v1066 = v1057 == v965;
                                                        int v1068;
                                                        if (v1066){
                                                            int v1067;
                                                            v1067 = v1064 + 2;
                                                            v1068 = v1067;
                                                        } else {
                                                            v1068 = v1064;
                                                        }
                                                        v1055[v1057] = v1068;
                                                        v1057 += 1 ;
                                                    }
                                                    v1076 = Union4{Union4_2{v962, false, v964, v1037, v1055, v1038}};
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
                                        Union6 v970 = v962.case1.v0;
                                        switch (v968.tag) {
                                            case 0: { // Call
                                                if (v963){
                                                    int v972;
                                                    v972 = v965 ^ 1;
                                                    v1076 = Union4{Union4_2{v962, false, v964, v972, v966, v967}};
                                                } else {
                                                    int v974; int v975;
                                                    Tuple7 tmp31 = Tuple7{0, 0};
                                                    v974 = tmp31.v0; v975 = tmp31.v1;
                                                    while (while_method_0(v974)){
                                                        bool v977;
                                                        v977 = 0 <= v974;
                                                        bool v979;
                                                        if (v977){
                                                            bool v978;
                                                            v978 = v974 < 2;
                                                            v979 = v978;
                                                        } else {
                                                            v979 = false;
                                                        }
                                                        bool v980;
                                                        v980 = v979 == false;
                                                        if (v980){
                                                            assert("Index must be in range." && v979);
                                                        } else {
                                                        }
                                                        int v982;
                                                        v982 = v966[v974];
                                                        bool v984;
                                                        v984 = v975 >= v982;
                                                        int v985;
                                                        if (v984){
                                                            v985 = v975;
                                                        } else {
                                                            v985 = v982;
                                                        }
                                                        v975 = v985;
                                                        v974 += 1 ;
                                                    }
                                                    static_array<int,2> v986;
                                                    int v988;
                                                    v988 = 0;
                                                    while (while_method_0(v988)){
                                                        v986[v988] = v975;
                                                        v988 += 1 ;
                                                    }
                                                    v1076 = Union4{Union4_4{v962, v963, v964, v965, v986, v967}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v1076 = Union4{Union4_5{v962, v963, v964, v965, v966, v967}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                bool v992;
                                                v992 = v967 > 0;
                                                if (v992){
                                                    int v993;
                                                    v993 = v965 ^ 1;
                                                    int v994;
                                                    v994 = -1 + v967;
                                                    int v995; int v996;
                                                    Tuple7 tmp32 = Tuple7{0, 0};
                                                    v995 = tmp32.v0; v996 = tmp32.v1;
                                                    while (while_method_0(v995)){
                                                        bool v998;
                                                        v998 = 0 <= v995;
                                                        bool v1000;
                                                        if (v998){
                                                            bool v999;
                                                            v999 = v995 < 2;
                                                            v1000 = v999;
                                                        } else {
                                                            v1000 = false;
                                                        }
                                                        bool v1001;
                                                        v1001 = v1000 == false;
                                                        if (v1001){
                                                            assert("Index must be in range." && v1000);
                                                        } else {
                                                        }
                                                        int v1003;
                                                        v1003 = v966[v995];
                                                        bool v1005;
                                                        v1005 = v996 >= v1003;
                                                        int v1006;
                                                        if (v1005){
                                                            v1006 = v996;
                                                        } else {
                                                            v1006 = v1003;
                                                        }
                                                        v996 = v1006;
                                                        v995 += 1 ;
                                                    }
                                                    static_array<int,2> v1007;
                                                    int v1009;
                                                    v1009 = 0;
                                                    while (while_method_0(v1009)){
                                                        v1007[v1009] = v996;
                                                        v1009 += 1 ;
                                                    }
                                                    static_array<int,2> v1011;
                                                    int v1013;
                                                    v1013 = 0;
                                                    while (while_method_0(v1013)){
                                                        bool v1015;
                                                        v1015 = 0 <= v1013;
                                                        bool v1017;
                                                        if (v1015){
                                                            bool v1016;
                                                            v1016 = v1013 < 2;
                                                            v1017 = v1016;
                                                        } else {
                                                            v1017 = false;
                                                        }
                                                        bool v1018;
                                                        v1018 = v1017 == false;
                                                        if (v1018){
                                                            assert("Index must be in range." && v1017);
                                                        } else {
                                                        }
                                                        int v1020;
                                                        v1020 = v1007[v1013];
                                                        bool v1022;
                                                        v1022 = v1013 == v965;
                                                        int v1024;
                                                        if (v1022){
                                                            int v1023;
                                                            v1023 = v1020 + 4;
                                                            v1024 = v1023;
                                                        } else {
                                                            v1024 = v1020;
                                                        }
                                                        v1011[v1013] = v1024;
                                                        v1013 += 1 ;
                                                    }
                                                    v1076 = Union4{Union4_2{v962, false, v964, v993, v1011, v994}};
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
                                v1138 = Union3{Union3_1{v1076}};
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
                                v94 = compare_hands_27(v81, v82, v83, v84, v85, v86);
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
                                v1138 = Union3{Union3_0{}};
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
                                v1138 = Union3{Union3_0{}};
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
                v60 = v1138;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v1139;
    v1139 = threadIdx.x;
    int v1140;
    v1140 = blockIdx.x;
    int v1141;
    v1141 = v1140 * 256;
    int v1142;
    v1142 = v1139 + v1141;
    bool v1143;
    v1143 = v1142 == 0;
    if (v1143){
        Union8 & v1144 = v19.v5;
        static_array<Union2,2> & v1145 = v19.v3;
        static_array_list<Union7,32> & v1146 = v19.v2;
        Union3 & v1147 = v19.v1;
        unsigned int & v1148 = v19.v0;
        return f_31(v0, v1148, v1147, v1146, v1145, v1144);
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
    while (while_method_12(v27)){
        int v29;
        v29 = 0;
        while (while_method_6(v29)){
            int v31;
            v31 = 0;
            while (while_method_0(v31)){
                Union4 v33;
                v33 = Union4{Union4_1{}};
                method_48(v0, v1, v2, v26, v31, v33);
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
                v42 = reinterpret_cast<double *>(&v1[58195968ull]);
                double * v44;
                v44 = reinterpret_cast<double *>(&v1[61341696ull]);
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
                int * v51;
                v51 = reinterpret_cast<int *>(&v0[1572864ull]);
                bool * v53;
                v53 = reinterpret_cast<bool *>(&v0[1572880ull]);
                float * v55;
                v55 = reinterpret_cast<float *>(&v0[1572912ull]);
                float * v57;
                v57 = reinterpret_cast<float *>(&v0[1573040ull]);
                double v59[2];
                int v60;
                v60 = 0;
                while (while_method_0(v60)){
                    int v62; double v63;
                    Tuple11 tmp51 = Tuple11{0, 0.0};
                    v62 = tmp51.v0; v63 = tmp51.v1;
                    while (while_method_4(v62)){
                        assert("Tensor range check" && 0 <= v62 && v62 < 32);
                        bool v65;
                        v65 = v53[v62];
                        bool v66;
                        v66 = v65 == false;
                        double v74;
                        if (v66){
                            v74 = 0.0;
                        } else {
                            assert("Tensor range check" && 0 <= v62 && v62 < 32);
                            assert("Tensor range check" && 0 <= v60 && v60 < 2);
                            int v67;
                            v67 = v60 + v50;
                            int v68;
                            v68 = 12288 * v62;
                            int v69;
                            v69 = v68 + v67;
                            double v70;
                            v70 = v42[v69];
                            double v71;
                            v71 = v44[v69];
                            double v72;
                            v72 = v70 - v71;
                            double v73;
                            v73 = exp(v72);
                            v74 = v73;
                        }
                        double v75;
                        v75 = v63 + v74;
                        v63 = v75;
                        v62 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v60 && v60 < 2);
                    v59[v60] = v63;
                    v60 += 1 ;
                }
                double v76;
                v76 = 1.0;
                int v77;
                v77 = 0;
                while (while_method_0(v77)){
                    assert("Tensor range check" && 0 <= v77 && v77 < 2);
                    double v79;
                    v79 = v59[v77];
                    double v80;
                    v80 = v76 * v79;
                    v76 = v80;
                    v77 += 1 ;
                }
                double v81[64];
                int v82;
                v82 = 0;
                while (while_method_4(v82)){
                    int v84;
                    v84 = 0;
                    while (while_method_0(v84)){
                        bool v86;
                        v86 = v76 == 0.0;
                        bool v87;
                        v87 = v86 != true;
                        double v98;
                        if (v87){
                            assert("Tensor range check" && 0 <= v84 && v84 < 2);
                            double v88;
                            v88 = v59[v84];
                            double v89;
                            v89 = v76 / v88;
                            assert("Tensor range check" && 0 <= v82 && v82 < 32);
                            assert("Tensor range check" && 0 <= v84 && v84 < 2);
                            int v90;
                            v90 = v84 + v50;
                            int v91;
                            v91 = 12288 * v82;
                            int v92;
                            v92 = v91 + v90;
                            double v93;
                            v93 = v42[v92];
                            double v94;
                            v94 = v44[v92];
                            double v95;
                            v95 = v93 - v94;
                            double v96;
                            v96 = exp(v95);
                            double v97;
                            v97 = v89 * v96;
                            v98 = v97;
                        } else {
                            v98 = 0.0;
                        }
                        bool v99;
                        v99 = isnan(v98);
                        bool v100;
                        v100 = v99 == false;
                        bool v101;
                        v101 = v100 == false;
                        if (v101){
                            assert("The path probability after integration should not be a nan in integrate_rewards_." && v100);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v82 && v82 < 32);
                        assert("Tensor range check" && 0 <= v84 && v84 < 2);
                        int v103;
                        v103 = 2 * v82;
                        int v104;
                        v104 = v103 + v84;
                        v81[v104] = v98;
                        v84 += 1 ;
                    }
                    v82 += 1 ;
                }
                int v105;
                v105 = 0;
                while (while_method_4(v105)){
                    assert("Tensor range check" && 0 <= v105 && v105 < 32);
                    assert("Tensor range check" && 0 <= v31 && v31 < 2);
                    int v107;
                    v107 = 2 * v105;
                    int v108;
                    v108 = v107 + v31;
                    double v109;
                    v109 = v81[v108];
                    float v110;
                    v110 = (float)v109;
                    float v111;
                    v111 = v40 * v110;
                    assert("Tensor range check" && 0 <= v105 && v105 < 32);
                    assert("Tensor range check" && 0 <= v27 && v27 < 256);
                    int v112;
                    v112 = 256 * v105;
                    int v113;
                    v113 = v112 + v27;
                    float * v114;
                    v114 = v3+v113;
                    float * v116;
                    v116 = v4+v113;
                    float v118;
                    v118 = atomicAdd(v114,v111);
                    float v119;
                    v119 = atomicAdd(v116,v110);
                    v105 += 1 ;
                }
                static_array<float,2> & v120 = v26.v4;
                float * v121;
                v121 = reinterpret_cast<float *>(&v1[7864320ull]);
                int * v123;
                v123 = reinterpret_cast<int *>(&v0[1572864ull]);
                bool * v125;
                v125 = reinterpret_cast<bool *>(&v0[1572880ull]);
                float * v127;
                v127 = reinterpret_cast<float *>(&v0[1572912ull]);
                float * v129;
                v129 = reinterpret_cast<float *>(&v0[1573040ull]);
                double * v131;
                v131 = reinterpret_cast<double *>(&v1[58195968ull]);
                double * v133;
                v133 = reinterpret_cast<double *>(&v1[61341696ull]);
                int v135;
                v135 = threadIdx.x;
                int v136;
                v136 = blockIdx.x;
                int v137;
                v137 = v136 * 256;
                int v138;
                v138 = v135 + v137;
                assert("Tensor range check" && 0 <= v138 && v138 < 6144);
                int v139;
                v139 = 2 * v138;
                double * v140;
                v140 = v131+v139;
                double * v142;
                v142 = v133+v139;
                float v144[2];
                int v145;
                v145 = 0;
                while (while_method_0(v145)){
                    bool v147;
                    v147 = 0 <= v145;
                    bool v149;
                    if (v147){
                        bool v148;
                        v148 = v145 < 2;
                        v149 = v148;
                    } else {
                        v149 = false;
                    }
                    bool v150;
                    v150 = v149 == false;
                    if (v150){
                        assert("Index must be in range." && v149);
                    } else {
                    }
                    float v152;
                    v152 = v120[v145];
                    assert("Tensor range check" && 0 <= v145 && v145 < 2);
                    v144[v145] = v152;
                    v145 += 1 ;
                }
                double v154[2];
                int v155;
                v155 = 0;
                while (while_method_0(v155)){
                    int v157; double v158;
                    Tuple11 tmp52 = Tuple11{0, 0.0};
                    v157 = tmp52.v0; v158 = tmp52.v1;
                    while (while_method_4(v157)){
                        assert("Tensor range check" && 0 <= v157 && v157 < 32);
                        bool v160;
                        v160 = v125[v157];
                        bool v161;
                        v161 = v160 == false;
                        double v168;
                        if (v161){
                            v168 = 0.0;
                        } else {
                            assert("Tensor range check" && 0 <= v157 && v157 < 32);
                            assert("Tensor range check" && 0 <= v155 && v155 < 2);
                            int v162;
                            v162 = 12288 * v157;
                            int v163;
                            v163 = v162 + v155;
                            double v164;
                            v164 = v140[v163];
                            double v165;
                            v165 = v142[v163];
                            double v166;
                            v166 = v164 - v165;
                            double v167;
                            v167 = exp(v166);
                            v168 = v167;
                        }
                        double v169;
                        v169 = v158 + v168;
                        v158 = v169;
                        v157 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v155 && v155 < 2);
                    v154[v155] = v158;
                    v155 += 1 ;
                }
                double v170;
                v170 = 1.0;
                int v171;
                v171 = 0;
                while (while_method_0(v171)){
                    assert("Tensor range check" && 0 <= v171 && v171 < 2);
                    double v173;
                    v173 = v154[v171];
                    double v174;
                    v174 = v170 * v173;
                    v170 = v174;
                    v171 += 1 ;
                }
                double v175[64];
                int v176;
                v176 = 0;
                while (while_method_4(v176)){
                    int v178;
                    v178 = 0;
                    while (while_method_0(v178)){
                        bool v180;
                        v180 = v170 == 0.0;
                        bool v181;
                        v181 = v180 != true;
                        double v191;
                        if (v181){
                            assert("Tensor range check" && 0 <= v178 && v178 < 2);
                            double v182;
                            v182 = v154[v178];
                            double v183;
                            v183 = v170 / v182;
                            assert("Tensor range check" && 0 <= v176 && v176 < 32);
                            assert("Tensor range check" && 0 <= v178 && v178 < 2);
                            int v184;
                            v184 = 12288 * v176;
                            int v185;
                            v185 = v184 + v178;
                            double v186;
                            v186 = v140[v185];
                            double v187;
                            v187 = v142[v185];
                            double v188;
                            v188 = v186 - v187;
                            double v189;
                            v189 = exp(v188);
                            double v190;
                            v190 = v183 * v189;
                            v191 = v190;
                        } else {
                            v191 = 0.0;
                        }
                        bool v192;
                        v192 = isnan(v191);
                        bool v193;
                        v193 = v192 == false;
                        bool v194;
                        v194 = v193 == false;
                        if (v194){
                            assert("The path probability after integration should not be a nan in integrate_rewards_." && v193);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v176 && v176 < 32);
                        assert("Tensor range check" && 0 <= v178 && v178 < 2);
                        int v196;
                        v196 = 2 * v176;
                        int v197;
                        v197 = v196 + v178;
                        v175[v197] = v191;
                        v178 += 1 ;
                    }
                    v176 += 1 ;
                }
                float v198[32];
                float v199[32];
                int v200;
                v200 = 0;
                while (while_method_4(v200)){
                    int v202; float v203; double v204;
                    Tuple12 tmp53 = Tuple12{0, 0.0f, 0.0};
                    v202 = tmp53.v0; v203 = tmp53.v1; v204 = tmp53.v2;
                    while (while_method_0(v202)){
                        assert("Tensor range check" && 0 <= v200 && v200 < 32);
                        assert("Tensor range check" && 0 <= v202 && v202 < 2);
                        int v206;
                        v206 = 2 * v200;
                        int v207;
                        v207 = v206 + v202;
                        double v208;
                        v208 = v175[v207];
                        assert("Tensor range check" && 0 <= v202 && v202 < 2);
                        float v209;
                        v209 = v144[v202];
                        float v210;
                        v210 = (float)v208;
                        float v211;
                        v211 = v210 * v209;
                        float v212;
                        v212 = v203 + v211;
                        double v213;
                        v213 = v204 + v208;
                        v203 = v212;
                        v204 = v213;
                        v202 += 1 ;
                    }
                    float v214;
                    v214 = (float)v204;
                    assert("Tensor range check" && 0 <= v200 && v200 < 32);
                    v198[v200] = v203;
                    v199[v200] = v214;
                    v200 += 1 ;
                }
                int v215;
                v215 = 0;
                while (while_method_4(v215)){
                    assert("Tensor range check" && 0 <= v215 && v215 < 32);
                    float v217;
                    v217 = v198[v215];
                    float v218;
                    v218 = v199[v215];
                    bool v219;
                    v219 = isnan(v218);
                    bool v220;
                    v220 = v219 == false;
                    bool v221;
                    v221 = v220 == false;
                    if (v221){
                        assert("The path probability after integration should not be a nan in calculate updates." && v220);
                    } else {
                    }
                    float v223;
                    v223 = v217 * v218;
                    assert("Tensor range check" && 0 <= v215 && v215 < 32);
                    float * v224;
                    v224 = v127+v215;
                    float * v226;
                    v226 = v129+v215;
                    float v228;
                    v228 = atomicAdd(v224,v223);
                    float v229;
                    v229 = atomicAdd(v226,v218);
                    v215 += 1 ;
                }
                int v230;
                v230 = threadIdx.x;
                int v231;
                v231 = blockIdx.x;
                int v232;
                v232 = v231 * 256;
                int v233;
                v233 = v230 + v232;
                int v234;
                v234 = 0;
                while (while_method_4(v234)){
                    assert("Tensor range check" && 0 <= v234 && v234 < 32);
                    int v236;
                    v236 = 12288 * v234;
                    assert("Tensor range check" && 0 <= v233 && v233 < 6144);
                    int v237;
                    v237 = 2 * v233;
                    int v238;
                    v238 = v237 + v236;
                    double * v239;
                    v239 = v131+v238;
                    double * v241;
                    v241 = v133+v238;
                    double * v243;
                    v243 = v131+v238;
                    double * v245;
                    v245 = v133+v238;
                    int v247;
                    v247 = sizeof(double *);
                    unsigned long long v248;
                    v248 = (unsigned long long)v247;
                    unsigned long long v249;
                    v249 = 256ull * v248;
                    unsigned long long v250;
                    v250 = v249 + 16ull;
                    unsigned long long v251;
                    v251 = v250 - 1ull;
                    unsigned long long v252;
                    v252 = v251 % 16ull;
                    unsigned long long v253;
                    v253 = v251 - v252;
                    unsigned long long v254;
                    v254 = v253 + v249;
                    unsigned long long v255;
                    v255 = v254 + 16ull;
                    unsigned long long v256;
                    v256 = v255 - 1ull;
                    unsigned long long v257;
                    v257 = v256 % 16ull;
                    unsigned long long v258;
                    v258 = v256 - v257;
                    unsigned long long v259;
                    v259 = v258 + v249;
                    unsigned long long v260;
                    v260 = v259 + 16ull;
                    unsigned long long v261;
                    v261 = v260 - 1ull;
                    unsigned long long v262;
                    v262 = v261 % 16ull;
                    unsigned long long v263;
                    v263 = v261 - v262;
                    unsigned long long v264;
                    v264 = v263 + v249;
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
                    double * * v272;
                    v272 = reinterpret_cast<double * *>(&v268[0ull]);
                    double * * v274;
                    v274 = reinterpret_cast<double * *>(&v268[v253]);
                    double * * v276;
                    v276 = reinterpret_cast<double * *>(&v268[v258]);
                    double * * v278;
                    v278 = reinterpret_cast<double * *>(&v268[v263]);
                    int v280;
                    v280 = threadIdx.x;
                    assert("Tensor range check" && 0 <= v280 && v280 < 256);
                    v272[v280] = v239;
                    v274[v280] = v241;
                    v276[v280] = v243;
                    v278[v280] = v245;
                    __syncthreads();
                    bool v281;
                    v281 = 0 <= v280;
                    bool v282;
                    v282 = v281 == false;
                    if (v282){
                        assert("The index needs to be zero or positive." && v281);
                    } else {
                    }
                    int v284;
                    v284 = v280 % 1;
                    bool v285;
                    v285 = v280 < 256;
                    bool v286;
                    v286 = v285 == false;
                    if (v286){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v285);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v280 && v280 < 256);
                    int v288;
                    v288 = 0;
                    while (while_method_5(v288)){
                        bool v290;
                        v290 = v281 && v285;
                        bool v291;
                        v291 = v290 == false;
                        if (v291){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v290);
                        } else {
                        }
                        bool v293;
                        v293 = 0 <= v288;
                        bool v295;
                        if (v293){
                            bool v294;
                            v294 = v288 < 1;
                            v295 = v294;
                        } else {
                            v295 = false;
                        }
                        bool v296;
                        v296 = v295 == false;
                        if (v296){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v295);
                        } else {
                        }
                        int v298;
                        v298 = v288 * 256;
                        int v299;
                        v299 = v298 + v280;
                        assert("Tensor range check" && 0 <= v288 && v288 < 1);
                        int v300;
                        v300 = 256 * v288;
                        int v301;
                        v301 = v300 + v280;
                        double * v302;
                        v302 = v272[v301];
                        double * v303;
                        v303 = v274[v301];
                        double * v304;
                        v304 = v276[v301];
                        double * v305;
                        v305 = v278[v301];
                        int v306;
                        v306 = blockIdx.x;
                        int v307;
                        v307 = v306 * 256;
                        int v308;
                        v308 = v307 + v299;
                        assert("Tensor range check" && 0 <= v284 && v284 < 1);
                        int v309;
                        v309 = 2 * v284;
                        double v310[2];
                        double v311[2];
                        int v312[2];
                        int v313;
                        v313 = 0;
                        while (while_method_5(v313)){
                            assert("Tensor range check" && 0 <= v313 && v313 < 1);
                            int v315;
                            v315 = 2 * v313;
                            assert("Tensor range check" && 0 <= v313 && v313 < 1);
                            int v316;
                            v316 = v315 + v309;
                            int4* v317;
                            v317 = reinterpret_cast<int4*>(v302 + v316);
                            int4* v318;
                            v318 = reinterpret_cast<int4*>(v310 + v315);
                            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v317) % 16 == 0 && reinterpret_cast<unsigned long long>(v318) % 16 == 0);
                            *v318 = *v317;
                            int4* v319;
                            v319 = reinterpret_cast<int4*>(v303 + v316);
                            int4* v320;
                            v320 = reinterpret_cast<int4*>(v311 + v315);
                            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v319) % 16 == 0 && reinterpret_cast<unsigned long long>(v320) % 16 == 0);
                            *v320 = *v319;
                            v313 += 1 ;
                        }
                        int v321;
                        v321 = 0;
                        while (while_method_5(v321)){
                            int v323;
                            v323 = 0;
                            while (while_method_0(v323)){
                                bool v325;
                                v325 = 0 <= v323;
                                bool v327;
                                if (v325){
                                    bool v326;
                                    v326 = v323 < 2;
                                    v327 = v326;
                                } else {
                                    v327 = false;
                                }
                                bool v328;
                                v328 = v327 == false;
                                if (v328){
                                    assert("The indices should be inside the range of the dimension." && v327);
                                } else {
                                }
                                bool v330;
                                v330 = 0 <= v284;
                                bool v332;
                                if (v330){
                                    bool v331;
                                    v331 = v284 < 1;
                                    v332 = v331;
                                } else {
                                    v332 = false;
                                }
                                bool v333;
                                v333 = v332 == false;
                                if (v333){
                                    assert("The indices should be inside the range of the dimension." && v332);
                                } else {
                                }
                                int v335;
                                v335 = v284 * 2;
                                int v336;
                                v336 = v323 + v335;
                                bool v337;
                                v337 = 0 <= v321;
                                bool v339;
                                if (v337){
                                    bool v338;
                                    v338 = v321 < 1;
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
                                int v342;
                                v342 = v321 * 2;
                                int v343;
                                v343 = v336 + v342;
                                assert("Tensor range check" && 0 <= v321 && v321 < 1);
                                assert("Tensor range check" && 0 <= v323 && v323 < 2);
                                int v344;
                                v344 = 2 * v321;
                                int v345;
                                v345 = v344 + v323;
                                v312[v345] = v343;
                                v323 += 1 ;
                            }
                            v321 += 1 ;
                        }
                        double v346[2];
                        double v347[2];
                        int v348;
                        v348 = 0;
                        while (while_method_5(v348)){
                            int v350;
                            v350 = 0;
                            while (while_method_0(v350)){
                                assert("Tensor range check" && 0 <= v348 && v348 < 1);
                                assert("Tensor range check" && 0 <= v350 && v350 < 2);
                                int v352;
                                v352 = 2 * v348;
                                int v353;
                                v353 = v352 + v350;
                                double v354;
                                v354 = v310[v353];
                                double v355;
                                v355 = v311[v353];
                                assert("Tensor range check" && 0 <= v348 && v348 < 1);
                                assert("Tensor range check" && 0 <= v350 && v350 < 2);
                                v346[v353] = 0.0;
                                v347[v353] = 0.0;
                                v350 += 1 ;
                            }
                            v348 += 1 ;
                        }
                        int v356;
                        v356 = 0;
                        while (while_method_5(v356)){
                            assert("Tensor range check" && 0 <= v356 && v356 < 1);
                            int v358;
                            v358 = 2 * v356;
                            int v359;
                            v359 = v358 + v309;
                            assert("Tensor range check" && 0 <= v356 && v356 < 1);
                            int4* v360;
                            v360 = reinterpret_cast<int4*>(v346 + v358);
                            int4* v361;
                            v361 = reinterpret_cast<int4*>(v304 + v359);
                            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v360) % 16 == 0 && reinterpret_cast<unsigned long long>(v361) % 16 == 0);
                            *v361 = *v360;
                            int4* v362;
                            v362 = reinterpret_cast<int4*>(v347 + v358);
                            int4* v363;
                            v363 = reinterpret_cast<int4*>(v305 + v359);
                            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v362) % 16 == 0 && reinterpret_cast<unsigned long long>(v363) % 16 == 0);
                            *v363 = *v362;
                            v356 += 1 ;
                        }
                        assert("Tensor range check" && 0 <= v299 && v299 < 256);
                        v288 += 1 ;
                    }
                    __syncthreads();
                    assert("Tensor range check" && 0 <= v280 && v280 < 256);
                    __syncthreads();
                    v234 += 1 ;
                }
                v31 += 1 ;
            }
            v29 += 1 ;
        }
        cooperative_groups::grid_group & v364 = v26.v1;
        cooperative_groups::grid_group & v365 = v364;
        curandStatePhilox4_32_10_t & v366 = v26.v5;
        curandStatePhilox4_32_10_t & v367 = v366;
        float * v368;
        v368 = reinterpret_cast<float *>(&v0[0ull]);
        float * v370;
        v370 = reinterpret_cast<float *>(&v2[0ull]);
        float * v372;
        v372 = reinterpret_cast<float *>(&v0[1048576ull]);
        float * v374;
        v374 = reinterpret_cast<float *>(&v2[1048576ull]);
        float * v376;
        v376 = reinterpret_cast<float *>(&v1[7864320ull]);
        int * v378;
        v378 = reinterpret_cast<int *>(&v0[1572864ull]);
        bool * v380;
        v380 = reinterpret_cast<bool *>(&v0[1572880ull]);
        float * v382;
        v382 = reinterpret_cast<float *>(&v0[1572912ull]);
        float * v384;
        v384 = reinterpret_cast<float *>(&v0[1573040ull]);
        double * v386;
        v386 = reinterpret_cast<double *>(&v1[58195968ull]);
        double * v388;
        v388 = reinterpret_cast<double *>(&v1[61341696ull]);
        v365.sync() ;
        int v390;
        v390 = threadIdx.x;
        int v391;
        v391 = blockIdx.x;
        int v392;
        v392 = v391 * 256;
        int v393;
        v393 = v390 + v392;
        bool v394;
        v394 = v393 == 0;
        if (v394){
            int v395;
            v395 = 0;
            int v396;
            v396 = 32;
            int v397;
            v397 = int_range_22(v396, v395, v367);
            v378[0] = v397;
        } else {
        }
        __syncwarp();
        float v398[32];
        int v399;
        v399 = 0;
        while (while_method_4(v399)){
            assert("Tensor range check" && 0 <= v399 && v399 < 32);
            float v401;
            v401 = v382[v399];
            float v402;
            v402 = v384[v399];
            bool v403;
            v403 = v402 == 0.0f;
            bool v404;
            v404 = v403 != true;
            float v406;
            if (v404){
                float v405;
                v405 = v401 / v402;
                v406 = v405;
            } else {
                v406 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v399 && v399 < 32);
            v398[v399] = v406;
            v399 += 1 ;
        }
        float v407;
        v407 = 0.0f;
        int v408;
        v408 = 0;
        while (while_method_4(v408)){
            assert("Tensor range check" && 0 <= v408 && v408 < 32);
            float v410;
            v410 = v398[v408];
            float v411;
            v411 = v407 + v410;
            v407 = v411;
            v408 += 1 ;
        }
        float v412;
        v412 = v407 / 32.0f;
        int v413;
        v413 = 0;
        while (while_method_4(v413)){
            assert("Tensor range check" && 0 <= v413 && v413 < 32);
            v382[v413] = 0.0f;
            v384[v413] = 0.0f;
            v413 += 1 ;
        }
        bool v415[32];
        int v416;
        v416 = 0;
        while (while_method_4(v416)){
            assert("Tensor range check" && 0 <= v416 && v416 < 32);
            float v418;
            v418 = v398[v416];
            bool v419;
            v419 = v418 >= v412;
            assert("Tensor range check" && 0 <= v416 && v416 < 32);
            v415[v416] = v419;
            v416 += 1 ;
        }
        int v420;
        v420 = 0;
        while (while_method_4(v420)){
            assert("Tensor range check" && 0 <= v420 && v420 < 32);
            bool v422;
            v422 = v415[v420];
            assert("Tensor range check" && 0 <= v420 && v420 < 32);
            v380[v420] = v422;
            v420 += 1 ;
        }
        extern __shared__ unsigned char v423[];
        float * v424;
        v424 = reinterpret_cast<float *>(&v423[0ull]);
        int v426;
        v426 = blockIdx.x;
        int v427;
        v427 = v426;
        while (while_method_9(v427)){
            bool v429;
            v429 = 0 <= v427;
            bool v430;
            v430 = v429 == false;
            if (v430){
                assert("The index needs to be zero or positive." && v429);
            } else {
            }
            int v432;
            v432 = v427 % 16;
            int v433;
            v433 = v427 / 16;
            bool v434;
            v434 = v433 < 1;
            bool v435;
            v435 = v434 == false;
            if (v435){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v434);
            } else {
            }
            assert("Tensor range check" && 0 <= v433 && v433 < 1);
            assert("Tensor range check" && 0 <= v432 && v432 < 16);
            int v437;
            v437 = 512 * v432;
            int v438;
            v438 = 262144 * v433;
            int v439;
            v439 = v438 + v437;
            int v440;
            v440 = 16384 * v432;
            int v441;
            v441 = 32 * v433;
            int v442;
            v442 = v441 + v440;
            int v443;
            v443 = threadIdx.x;
            int v444;
            v444 = v443;
            while (while_method_14(v444)){
                bool v446;
                v446 = 0 <= v444;
                bool v447;
                v447 = v446 == false;
                if (v447){
                    assert("The index needs to be zero or positive." && v446);
                } else {
                }
                int v449;
                v449 = v444 % 512;
                int v450;
                v450 = v444 / 512;
                bool v451;
                v451 = v450 < 32;
                bool v452;
                v452 = v451 == false;
                if (v452){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v451);
                } else {
                }
                assert("Tensor range check" && 0 <= v450 && v450 < 32);
                assert("Tensor range check" && 0 <= v449 && v449 < 512);
                int v454;
                v454 = v449 + v439;
                int v455;
                v455 = 8192 * v450;
                int v456;
                v456 = v455 + v454;
                float v457;
                v457 = v368[v456];
                assert("Tensor range check" && 0 <= v450 && v450 < 32);
                assert("Tensor range check" && 0 <= v449 && v449 < 512);
                int v458;
                v458 = 513 * v450;
                int v459;
                v459 = v458 + v449;
                v424[v459] = v457;
                v444 += 256 ;
            }
            __syncthreads();
            int v460;
            v460 = threadIdx.x;
            int v461;
            v461 = v460;
            while (while_method_14(v461)){
                bool v463;
                v463 = 0 <= v461;
                bool v464;
                v464 = v463 == false;
                if (v464){
                    assert("The index needs to be zero or positive." && v463);
                } else {
                }
                int v466;
                v466 = v461 % 32;
                int v467;
                v467 = v461 / 32;
                bool v468;
                v468 = v467 < 512;
                bool v469;
                v469 = v468 == false;
                if (v469){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v468);
                } else {
                }
                assert("Tensor range check" && 0 <= v467 && v467 < 512);
                assert("Tensor range check" && 0 <= v466 && v466 < 32);
                int v471;
                v471 = 513 * v466;
                int v472;
                v472 = v467 + v471;
                float v473;
                v473 = v424[v472];
                assert("Tensor range check" && 0 <= v467 && v467 < 512);
                assert("Tensor range check" && 0 <= v466 && v466 < 32);
                int v474;
                v474 = v466 + v442;
                int v475;
                v475 = 32 * v467;
                int v476;
                v476 = v475 + v474;
                v370[v476] = v473;
                v461 += 256 ;
            }
            __syncthreads();
            v427 += 24 ;
        }
        extern __shared__ unsigned char v477[];
        float * v478;
        v478 = reinterpret_cast<float *>(&v477[0ull]);
        int v480;
        v480 = blockIdx.x;
        int v481;
        v481 = v480;
        while (while_method_6(v481)){
            bool v483;
            v483 = 0 <= v481;
            bool v484;
            v484 = v483 == false;
            if (v484){
                assert("The index needs to be zero or positive." && v483);
            } else {
            }
            int v486;
            v486 = v481 % 8;
            int v487;
            v487 = v481 / 8;
            bool v488;
            v488 = v487 < 1;
            bool v489;
            v489 = v488 == false;
            if (v489){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v488);
            } else {
            }
            assert("Tensor range check" && 0 <= v487 && v487 < 1);
            assert("Tensor range check" && 0 <= v486 && v486 < 8);
            int v491;
            v491 = 512 * v486;
            int v492;
            v492 = 131072 * v487;
            int v493;
            v493 = v492 + v491;
            int v494;
            v494 = 16384 * v486;
            int v495;
            v495 = 32 * v487;
            int v496;
            v496 = v495 + v494;
            int v497;
            v497 = threadIdx.x;
            int v498;
            v498 = v497;
            while (while_method_14(v498)){
                bool v500;
                v500 = 0 <= v498;
                bool v501;
                v501 = v500 == false;
                if (v501){
                    assert("The index needs to be zero or positive." && v500);
                } else {
                }
                int v503;
                v503 = v498 % 512;
                int v504;
                v504 = v498 / 512;
                bool v505;
                v505 = v504 < 32;
                bool v506;
                v506 = v505 == false;
                if (v506){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v505);
                } else {
                }
                assert("Tensor range check" && 0 <= v504 && v504 < 32);
                assert("Tensor range check" && 0 <= v503 && v503 < 512);
                int v508;
                v508 = v503 + v493;
                int v509;
                v509 = 4096 * v504;
                int v510;
                v510 = v509 + v508;
                float v511;
                v511 = v372[v510];
                assert("Tensor range check" && 0 <= v504 && v504 < 32);
                assert("Tensor range check" && 0 <= v503 && v503 < 512);
                int v512;
                v512 = 513 * v504;
                int v513;
                v513 = v512 + v503;
                v478[v513] = v511;
                v498 += 256 ;
            }
            __syncthreads();
            int v514;
            v514 = threadIdx.x;
            int v515;
            v515 = v514;
            while (while_method_14(v515)){
                bool v517;
                v517 = 0 <= v515;
                bool v518;
                v518 = v517 == false;
                if (v518){
                    assert("The index needs to be zero or positive." && v517);
                } else {
                }
                int v520;
                v520 = v515 % 32;
                int v521;
                v521 = v515 / 32;
                bool v522;
                v522 = v521 < 512;
                bool v523;
                v523 = v522 == false;
                if (v523){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v522);
                } else {
                }
                assert("Tensor range check" && 0 <= v521 && v521 < 512);
                assert("Tensor range check" && 0 <= v520 && v520 < 32);
                int v525;
                v525 = 513 * v520;
                int v526;
                v526 = v521 + v525;
                float v527;
                v527 = v478[v526];
                assert("Tensor range check" && 0 <= v521 && v521 < 512);
                assert("Tensor range check" && 0 <= v520 && v520 < 32);
                int v528;
                v528 = v520 + v496;
                int v529;
                v529 = 32 * v521;
                int v530;
                v530 = v529 + v528;
                v374[v530] = v527;
                v515 += 256 ;
            }
            __syncthreads();
            v481 += 24 ;
        }
        v365.sync() ;
        int v531;
        v531 = threadIdx.x;
        bool v532;
        v532 = 0 <= v531;
        bool v533;
        v533 = v532 == false;
        if (v533){
            assert("The index needs to be zero or positive." && v532);
        } else {
        }
        int v535;
        v535 = v531 % 8;
        int v536;
        v536 = v531 / 8;
        int v537;
        v537 = v536 % 32;
        int v538;
        v538 = v536 / 32;
        bool v539;
        v539 = v538 < 1;
        bool v540;
        v540 = v539 == false;
        if (v540){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v539);
        } else {
        }
        assert("Tensor range check" && 0 <= v538 && v538 < 1);
        assert("Tensor range check" && 0 <= v537 && v537 < 32);
        assert("Tensor range check" && 0 <= v535 && v535 < 8);
        int v542;
        v542 = 4 * v535;
        int v543;
        v543 = 32 * v537;
        int v544;
        v544 = v543 + v542;
        int v545;
        v545 = 4096 * v538;
        int v546;
        v546 = v545 + v544;
        assert("Tensor range check" && 0 <= v538 && v538 < 1);
        assert("Tensor range check" && 0 <= v537 && v537 < 32);
        assert("Tensor range check" && 0 <= v535 && v535 < 8);
        int v547;
        v547 = blockIdx.x;
        int v548;
        v548 = v547;
        while (while_method_12(v548)){
            bool v550;
            v550 = 0 <= v548;
            bool v551;
            v551 = v550 == false;
            if (v551){
                assert("The index needs to be zero or positive." && v550);
            } else {
            }
            int v553;
            v553 = v548 % 4;
            int v554;
            v554 = v548 / 4;
            bool v555;
            v555 = v554 < 64;
            bool v556;
            v556 = v555 == false;
            if (v556){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v555);
            } else {
            }
            assert("Tensor range check" && 0 <= v554 && v554 < 64);
            assert("Tensor range check" && 0 <= v553 && v553 < 4);
            int v558;
            v558 = 1024 * v553;
            int v559;
            v559 = v558 + v546;
            int v560;
            v560 = 4096 * v554;
            int v561;
            v561 = v560 + v559;
            float v562[4];
            int v563[4];
            int v564;
            v564 = 0;
            while (while_method_5(v564)){
                assert("Tensor range check" && 0 <= v564 && v564 < 1);
                int v566;
                v566 = 4 * v564;
                assert("Tensor range check" && 0 <= v564 && v564 < 1);
                int v567;
                v567 = 32 * v564;
                int v568;
                v568 = v567 + v561;
                int4* v569;
                v569 = reinterpret_cast<int4*>(v370 + v568);
                int4* v570;
                v570 = reinterpret_cast<int4*>(v562 + v566);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v569) % 16 == 0 && reinterpret_cast<unsigned long long>(v570) % 16 == 0);
                *v570 = *v569;
                v564 += 1 ;
            }
            int v571;
            v571 = 0;
            while (while_method_5(v571)){
                int v573;
                v573 = 0;
                while (while_method_8(v573)){
                    bool v575;
                    v575 = 0 <= v573;
                    bool v577;
                    if (v575){
                        bool v576;
                        v576 = v573 < 4;
                        v577 = v576;
                    } else {
                        v577 = false;
                    }
                    bool v578;
                    v578 = v577 == false;
                    if (v578){
                        assert("The indices should be inside the range of the dimension." && v577);
                    } else {
                    }
                    bool v580;
                    v580 = 0 <= v535;
                    bool v582;
                    if (v580){
                        bool v581;
                        v581 = v535 < 8;
                        v582 = v581;
                    } else {
                        v582 = false;
                    }
                    bool v583;
                    v583 = v582 == false;
                    if (v583){
                        assert("The indices should be inside the range of the dimension." && v582);
                    } else {
                    }
                    int v585;
                    v585 = v535 * 4;
                    int v586;
                    v586 = v573 + v585;
                    bool v587;
                    v587 = 0 <= v571;
                    bool v589;
                    if (v587){
                        bool v588;
                        v588 = v571 < 1;
                        v589 = v588;
                    } else {
                        v589 = false;
                    }
                    bool v590;
                    v590 = v589 == false;
                    if (v590){
                        assert("The indices should be inside the range of the dimension." && v589);
                    } else {
                    }
                    int v592;
                    v592 = v571 * 32;
                    int v593;
                    v593 = v586 + v592;
                    assert("Tensor range check" && 0 <= v571 && v571 < 1);
                    assert("Tensor range check" && 0 <= v573 && v573 < 4);
                    int v594;
                    v594 = 4 * v571;
                    int v595;
                    v595 = v594 + v573;
                    v563[v595] = v593;
                    v573 += 1 ;
                }
                v571 += 1 ;
            }
            bool v596;
            v596 = 0 <= v538;
            bool v597;
            v597 = v596 && v539;
            bool v598;
            v598 = v597 == false;
            if (v598){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v597);
            } else {
            }
            bool v600;
            v600 = 0 <= v537;
            bool v602;
            if (v600){
                bool v601;
                v601 = v537 < 32;
                v602 = v601;
            } else {
                v602 = false;
            }
            bool v603;
            v603 = v602 == false;
            if (v603){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v602);
            } else {
            }
            bool v605;
            v605 = 0 <= v554;
            bool v606;
            v606 = v605 && v555;
            bool v607;
            v607 = v606 == false;
            if (v607){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v606);
            } else {
            }
            bool v609;
            v609 = 0 <= v553;
            bool v611;
            if (v609){
                bool v610;
                v610 = v553 < 4;
                v611 = v610;
            } else {
                v611 = false;
            }
            bool v612;
            v612 = v611 == false;
            if (v612){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v611);
            } else {
            }
            int v614;
            v614 = v553 * 32;
            int v615;
            v615 = v554 + v538;
            int v616;
            v616 = v614 + v537;
            bool v617[4];
            int v618;
            v618 = 0;
            while (while_method_5(v618)){
                int v620;
                v620 = 0;
                while (while_method_8(v620)){
                    assert("Tensor range check" && 0 <= v618 && v618 < 1);
                    assert("Tensor range check" && 0 <= v620 && v620 < 4);
                    int v622;
                    v622 = 4 * v618;
                    int v623;
                    v623 = v622 + v620;
                    int v624;
                    v624 = v563[v623];
                    assert("Tensor range check" && 0 <= v624 && v624 < 32);
                    bool v625;
                    v625 = v380[v624];
                    assert("Tensor range check" && 0 <= v618 && v618 < 1);
                    assert("Tensor range check" && 0 <= v620 && v620 < 4);
                    v617[v623] = v625;
                    v620 += 1 ;
                }
                v618 += 1 ;
            }
            int v626[4];
            int v627;
            v627 = 0;
            while (while_method_5(v627)){
                int v629;
                v629 = 0;
                while (while_method_8(v629)){
                    assert("Tensor range check" && 0 <= v627 && v627 < 1);
                    assert("Tensor range check" && 0 <= v629 && v629 < 4);
                    int v631;
                    v631 = 4 * v627;
                    int v632;
                    v632 = v631 + v629;
                    bool v633;
                    v633 = v617[v632];
                    int v634;
                    if (v633){
                        v634 = 1;
                    } else {
                        v634 = 0;
                    }
                    assert("Tensor range check" && 0 <= v627 && v627 < 1);
                    assert("Tensor range check" && 0 <= v629 && v629 < 4);
                    v626[v632] = v634;
                    v629 += 1 ;
                }
                v627 += 1 ;
            }
            int v635;
            v635 = 0;
            int v636;
            v636 = 0;
            while (while_method_5(v636)){
                int v638;
                v638 = 0;
                while (while_method_8(v638)){
                    assert("Tensor range check" && 0 <= v636 && v636 < 1);
                    assert("Tensor range check" && 0 <= v638 && v638 < 4);
                    int v640;
                    v640 = 4 * v636;
                    int v641;
                    v641 = v640 + v638;
                    int v642;
                    v642 = v626[v641];
                    int v643;
                    v643 = v635 + v642;
                    v635 = v643;
                    v638 += 1 ;
                }
                v636 += 1 ;
            }
            auto v644 = cooperative_groups::coalesced_threads();
            int v645;
            v645 = threadIdx.x;
            int v646;
            v646 = v645 / 8;
            auto v647 = cooperative_groups::labeled_partition(v644,v646);
            Closure1 v648{};
            int v649;
            v649 = cooperative_groups::reduce(v647, v635, v648);
            float v650;
            v650 = (float)v649;
            float v651[4];
            int v652;
            v652 = 0;
            while (while_method_5(v652)){
                int v654;
                v654 = 0;
                while (while_method_8(v654)){
                    assert("Tensor range check" && 0 <= v652 && v652 < 1);
                    assert("Tensor range check" && 0 <= v654 && v654 < 4);
                    int v656;
                    v656 = 4 * v652;
                    int v657;
                    v657 = v656 + v654;
                    float v658;
                    v658 = v562[v657];
                    bool v659;
                    v659 = v617[v657];
                    float v660;
                    if (v659){
                        v660 = v658;
                    } else {
                        v660 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v652 && v652 < 1);
                    assert("Tensor range check" && 0 <= v654 && v654 < 4);
                    v651[v657] = v660;
                    v654 += 1 ;
                }
                v652 += 1 ;
            }
            float v661;
            v661 = 0.0f;
            int v662;
            v662 = 0;
            while (while_method_5(v662)){
                int v664;
                v664 = 0;
                while (while_method_8(v664)){
                    assert("Tensor range check" && 0 <= v662 && v662 < 1);
                    assert("Tensor range check" && 0 <= v664 && v664 < 4);
                    int v666;
                    v666 = 4 * v662;
                    int v667;
                    v667 = v666 + v664;
                    float v668;
                    v668 = v651[v667];
                    float v669;
                    v669 = v661 + v668;
                    v661 = v669;
                    v664 += 1 ;
                }
                v662 += 1 ;
            }
            auto v670 = cooperative_groups::coalesced_threads();
            int v671;
            v671 = threadIdx.x;
            int v672;
            v672 = v671 / 8;
            auto v673 = cooperative_groups::labeled_partition(v670,v672);
            Closure0 v674{};
            float v675;
            v675 = cooperative_groups::reduce(v673, v661, v674);
            float v676;
            v676 = v675 / v650;
            float v677[4];
            int v678;
            v678 = 0;
            while (while_method_5(v678)){
                int v680;
                v680 = 0;
                while (while_method_8(v680)){
                    assert("Tensor range check" && 0 <= v678 && v678 < 1);
                    assert("Tensor range check" && 0 <= v680 && v680 < 4);
                    int v682;
                    v682 = 4 * v678;
                    int v683;
                    v683 = v682 + v680;
                    float v684;
                    v684 = v562[v683];
                    float v685;
                    v685 = v684 - v676;
                    float v686;
                    v686 = v685 * v685;
                    assert("Tensor range check" && 0 <= v678 && v678 < 1);
                    assert("Tensor range check" && 0 <= v680 && v680 < 4);
                    v677[v683] = v686;
                    v680 += 1 ;
                }
                v678 += 1 ;
            }
            float v687[4];
            int v688;
            v688 = 0;
            while (while_method_5(v688)){
                int v690;
                v690 = 0;
                while (while_method_8(v690)){
                    assert("Tensor range check" && 0 <= v688 && v688 < 1);
                    assert("Tensor range check" && 0 <= v690 && v690 < 4);
                    int v692;
                    v692 = 4 * v688;
                    int v693;
                    v693 = v692 + v690;
                    float v694;
                    v694 = v677[v693];
                    bool v695;
                    v695 = v617[v693];
                    float v696;
                    if (v695){
                        v696 = v694;
                    } else {
                        v696 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v688 && v688 < 1);
                    assert("Tensor range check" && 0 <= v690 && v690 < 4);
                    v687[v693] = v696;
                    v690 += 1 ;
                }
                v688 += 1 ;
            }
            float v697;
            v697 = 0.0f;
            int v698;
            v698 = 0;
            while (while_method_5(v698)){
                int v700;
                v700 = 0;
                while (while_method_8(v700)){
                    assert("Tensor range check" && 0 <= v698 && v698 < 1);
                    assert("Tensor range check" && 0 <= v700 && v700 < 4);
                    int v702;
                    v702 = 4 * v698;
                    int v703;
                    v703 = v702 + v700;
                    float v704;
                    v704 = v687[v703];
                    float v705;
                    v705 = v697 + v704;
                    v697 = v705;
                    v700 += 1 ;
                }
                v698 += 1 ;
            }
            auto v706 = cooperative_groups::coalesced_threads();
            int v707;
            v707 = threadIdx.x;
            int v708;
            v708 = v707 / 8;
            auto v709 = cooperative_groups::labeled_partition(v706,v708);
            float v710;
            v710 = cooperative_groups::reduce(v709, v697, v674);
            float v711;
            v711 = v710 / v650;
            float v712;
            v712 = sqrt(v711);
            bool v713;
            v713 = v650 > 1.0f;
            float v717;
            if (v713){
                float v714;
                v714 = v712 * v650;
                float v715;
                v715 = v650 - 1.0f;
                float v716;
                v716 = v714 / v715;
                v717 = v716;
            } else {
                v717 = 0.0f;
            }
            float v718[4];
            int v719;
            v719 = 0;
            while (while_method_5(v719)){
                int v721;
                v721 = 0;
                while (while_method_8(v721)){
                    assert("Tensor range check" && 0 <= v719 && v719 < 1);
                    assert("Tensor range check" && 0 <= v721 && v721 < 4);
                    int v723;
                    v723 = 4 * v719;
                    int v724;
                    v724 = v723 + v721;
                    float v725;
                    v725 = v562[v724];
                    bool v726;
                    v726 = v617[v724];
                    float v727;
                    v727 = curand_normal(&v367);
                    bool v728;
                    v728 = v717 >= 0.1f;
                    float v729;
                    if (v728){
                        v729 = v717;
                    } else {
                        v729 = 0.1f;
                    }
                    float v730;
                    v730 = v727 * v729;
                    float v731;
                    v731 = v730 + v676;
                    float v732;
                    if (v726){
                        v732 = v725;
                    } else {
                        v732 = v731;
                    }
                    assert("Tensor range check" && 0 <= v719 && v719 < 1);
                    assert("Tensor range check" && 0 <= v721 && v721 < 4);
                    v718[v724] = v732;
                    v721 += 1 ;
                }
                v719 += 1 ;
            }
            assert("Tensor range check" && 0 <= v554 && v554 < 64);
            assert("Tensor range check" && 0 <= v553 && v553 < 4);
            int v733;
            v733 = 0;
            while (while_method_5(v733)){
                assert("Tensor range check" && 0 <= v733 && v733 < 1);
                int v735;
                v735 = 32 * v733;
                int v736;
                v736 = v735 + v561;
                assert("Tensor range check" && 0 <= v733 && v733 < 1);
                int v737;
                v737 = 4 * v733;
                int4* v738;
                v738 = reinterpret_cast<int4*>(v718 + v737);
                int4* v739;
                v739 = reinterpret_cast<int4*>(v370 + v736);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v738) % 16 == 0 && reinterpret_cast<unsigned long long>(v739) % 16 == 0);
                *v739 = *v738;
                v733 += 1 ;
            }
            v548 += 24 ;
        }
        int v740;
        v740 = threadIdx.x;
        bool v741;
        v741 = 0 <= v740;
        bool v742;
        v742 = v741 == false;
        if (v742){
            assert("The index needs to be zero or positive." && v741);
        } else {
        }
        int v744;
        v744 = v740 % 8;
        int v745;
        v745 = v740 / 8;
        int v746;
        v746 = v745 % 32;
        int v747;
        v747 = v745 / 32;
        bool v748;
        v748 = v747 < 1;
        bool v749;
        v749 = v748 == false;
        if (v749){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v748);
        } else {
        }
        assert("Tensor range check" && 0 <= v747 && v747 < 1);
        assert("Tensor range check" && 0 <= v746 && v746 < 32);
        assert("Tensor range check" && 0 <= v744 && v744 < 8);
        int v751;
        v751 = 4 * v744;
        int v752;
        v752 = 32 * v746;
        int v753;
        v753 = v752 + v751;
        int v754;
        v754 = 2048 * v747;
        int v755;
        v755 = v754 + v753;
        assert("Tensor range check" && 0 <= v747 && v747 < 1);
        assert("Tensor range check" && 0 <= v746 && v746 < 32);
        assert("Tensor range check" && 0 <= v744 && v744 < 8);
        int v756;
        v756 = blockIdx.x;
        int v757;
        v757 = v756;
        while (while_method_15(v757)){
            bool v759;
            v759 = 0 <= v757;
            bool v760;
            v760 = v759 == false;
            if (v760){
                assert("The index needs to be zero or positive." && v759);
            } else {
            }
            int v762;
            v762 = v757 % 2;
            int v763;
            v763 = v757 / 2;
            bool v764;
            v764 = v763 < 64;
            bool v765;
            v765 = v764 == false;
            if (v765){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v764);
            } else {
            }
            assert("Tensor range check" && 0 <= v763 && v763 < 64);
            assert("Tensor range check" && 0 <= v762 && v762 < 2);
            int v767;
            v767 = 1024 * v762;
            int v768;
            v768 = v767 + v755;
            int v769;
            v769 = 2048 * v763;
            int v770;
            v770 = v769 + v768;
            float v771[4];
            int v772[4];
            int v773;
            v773 = 0;
            while (while_method_5(v773)){
                assert("Tensor range check" && 0 <= v773 && v773 < 1);
                int v775;
                v775 = 4 * v773;
                assert("Tensor range check" && 0 <= v773 && v773 < 1);
                int v776;
                v776 = 32 * v773;
                int v777;
                v777 = v776 + v770;
                int4* v778;
                v778 = reinterpret_cast<int4*>(v374 + v777);
                int4* v779;
                v779 = reinterpret_cast<int4*>(v771 + v775);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v778) % 16 == 0 && reinterpret_cast<unsigned long long>(v779) % 16 == 0);
                *v779 = *v778;
                v773 += 1 ;
            }
            int v780;
            v780 = 0;
            while (while_method_5(v780)){
                int v782;
                v782 = 0;
                while (while_method_8(v782)){
                    bool v784;
                    v784 = 0 <= v782;
                    bool v786;
                    if (v784){
                        bool v785;
                        v785 = v782 < 4;
                        v786 = v785;
                    } else {
                        v786 = false;
                    }
                    bool v787;
                    v787 = v786 == false;
                    if (v787){
                        assert("The indices should be inside the range of the dimension." && v786);
                    } else {
                    }
                    bool v789;
                    v789 = 0 <= v744;
                    bool v791;
                    if (v789){
                        bool v790;
                        v790 = v744 < 8;
                        v791 = v790;
                    } else {
                        v791 = false;
                    }
                    bool v792;
                    v792 = v791 == false;
                    if (v792){
                        assert("The indices should be inside the range of the dimension." && v791);
                    } else {
                    }
                    int v794;
                    v794 = v744 * 4;
                    int v795;
                    v795 = v782 + v794;
                    bool v796;
                    v796 = 0 <= v780;
                    bool v798;
                    if (v796){
                        bool v797;
                        v797 = v780 < 1;
                        v798 = v797;
                    } else {
                        v798 = false;
                    }
                    bool v799;
                    v799 = v798 == false;
                    if (v799){
                        assert("The indices should be inside the range of the dimension." && v798);
                    } else {
                    }
                    int v801;
                    v801 = v780 * 32;
                    int v802;
                    v802 = v795 + v801;
                    assert("Tensor range check" && 0 <= v780 && v780 < 1);
                    assert("Tensor range check" && 0 <= v782 && v782 < 4);
                    int v803;
                    v803 = 4 * v780;
                    int v804;
                    v804 = v803 + v782;
                    v772[v804] = v802;
                    v782 += 1 ;
                }
                v780 += 1 ;
            }
            bool v805;
            v805 = 0 <= v747;
            bool v806;
            v806 = v805 && v748;
            bool v807;
            v807 = v806 == false;
            if (v807){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v806);
            } else {
            }
            bool v809;
            v809 = 0 <= v746;
            bool v811;
            if (v809){
                bool v810;
                v810 = v746 < 32;
                v811 = v810;
            } else {
                v811 = false;
            }
            bool v812;
            v812 = v811 == false;
            if (v812){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v811);
            } else {
            }
            bool v814;
            v814 = 0 <= v763;
            bool v815;
            v815 = v814 && v764;
            bool v816;
            v816 = v815 == false;
            if (v816){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v815);
            } else {
            }
            bool v818;
            v818 = 0 <= v762;
            bool v820;
            if (v818){
                bool v819;
                v819 = v762 < 2;
                v820 = v819;
            } else {
                v820 = false;
            }
            bool v821;
            v821 = v820 == false;
            if (v821){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v820);
            } else {
            }
            int v823;
            v823 = v762 * 32;
            int v824;
            v824 = v763 + v747;
            int v825;
            v825 = v823 + v746;
            bool v826[4];
            int v827;
            v827 = 0;
            while (while_method_5(v827)){
                int v829;
                v829 = 0;
                while (while_method_8(v829)){
                    assert("Tensor range check" && 0 <= v827 && v827 < 1);
                    assert("Tensor range check" && 0 <= v829 && v829 < 4);
                    int v831;
                    v831 = 4 * v827;
                    int v832;
                    v832 = v831 + v829;
                    int v833;
                    v833 = v772[v832];
                    assert("Tensor range check" && 0 <= v833 && v833 < 32);
                    bool v834;
                    v834 = v380[v833];
                    assert("Tensor range check" && 0 <= v827 && v827 < 1);
                    assert("Tensor range check" && 0 <= v829 && v829 < 4);
                    v826[v832] = v834;
                    v829 += 1 ;
                }
                v827 += 1 ;
            }
            int v835[4];
            int v836;
            v836 = 0;
            while (while_method_5(v836)){
                int v838;
                v838 = 0;
                while (while_method_8(v838)){
                    assert("Tensor range check" && 0 <= v836 && v836 < 1);
                    assert("Tensor range check" && 0 <= v838 && v838 < 4);
                    int v840;
                    v840 = 4 * v836;
                    int v841;
                    v841 = v840 + v838;
                    bool v842;
                    v842 = v826[v841];
                    int v843;
                    if (v842){
                        v843 = 1;
                    } else {
                        v843 = 0;
                    }
                    assert("Tensor range check" && 0 <= v836 && v836 < 1);
                    assert("Tensor range check" && 0 <= v838 && v838 < 4);
                    v835[v841] = v843;
                    v838 += 1 ;
                }
                v836 += 1 ;
            }
            int v844;
            v844 = 0;
            int v845;
            v845 = 0;
            while (while_method_5(v845)){
                int v847;
                v847 = 0;
                while (while_method_8(v847)){
                    assert("Tensor range check" && 0 <= v845 && v845 < 1);
                    assert("Tensor range check" && 0 <= v847 && v847 < 4);
                    int v849;
                    v849 = 4 * v845;
                    int v850;
                    v850 = v849 + v847;
                    int v851;
                    v851 = v835[v850];
                    int v852;
                    v852 = v844 + v851;
                    v844 = v852;
                    v847 += 1 ;
                }
                v845 += 1 ;
            }
            auto v853 = cooperative_groups::coalesced_threads();
            int v854;
            v854 = threadIdx.x;
            int v855;
            v855 = v854 / 8;
            auto v856 = cooperative_groups::labeled_partition(v853,v855);
            Closure1 v857{};
            int v858;
            v858 = cooperative_groups::reduce(v856, v844, v857);
            float v859;
            v859 = (float)v858;
            float v860[4];
            int v861;
            v861 = 0;
            while (while_method_5(v861)){
                int v863;
                v863 = 0;
                while (while_method_8(v863)){
                    assert("Tensor range check" && 0 <= v861 && v861 < 1);
                    assert("Tensor range check" && 0 <= v863 && v863 < 4);
                    int v865;
                    v865 = 4 * v861;
                    int v866;
                    v866 = v865 + v863;
                    float v867;
                    v867 = v771[v866];
                    bool v868;
                    v868 = v826[v866];
                    float v869;
                    if (v868){
                        v869 = v867;
                    } else {
                        v869 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v861 && v861 < 1);
                    assert("Tensor range check" && 0 <= v863 && v863 < 4);
                    v860[v866] = v869;
                    v863 += 1 ;
                }
                v861 += 1 ;
            }
            float v870;
            v870 = 0.0f;
            int v871;
            v871 = 0;
            while (while_method_5(v871)){
                int v873;
                v873 = 0;
                while (while_method_8(v873)){
                    assert("Tensor range check" && 0 <= v871 && v871 < 1);
                    assert("Tensor range check" && 0 <= v873 && v873 < 4);
                    int v875;
                    v875 = 4 * v871;
                    int v876;
                    v876 = v875 + v873;
                    float v877;
                    v877 = v860[v876];
                    float v878;
                    v878 = v870 + v877;
                    v870 = v878;
                    v873 += 1 ;
                }
                v871 += 1 ;
            }
            auto v879 = cooperative_groups::coalesced_threads();
            int v880;
            v880 = threadIdx.x;
            int v881;
            v881 = v880 / 8;
            auto v882 = cooperative_groups::labeled_partition(v879,v881);
            Closure0 v883{};
            float v884;
            v884 = cooperative_groups::reduce(v882, v870, v883);
            float v885;
            v885 = v884 / v859;
            float v886[4];
            int v887;
            v887 = 0;
            while (while_method_5(v887)){
                int v889;
                v889 = 0;
                while (while_method_8(v889)){
                    assert("Tensor range check" && 0 <= v887 && v887 < 1);
                    assert("Tensor range check" && 0 <= v889 && v889 < 4);
                    int v891;
                    v891 = 4 * v887;
                    int v892;
                    v892 = v891 + v889;
                    float v893;
                    v893 = v771[v892];
                    float v894;
                    v894 = v893 - v885;
                    float v895;
                    v895 = v894 * v894;
                    assert("Tensor range check" && 0 <= v887 && v887 < 1);
                    assert("Tensor range check" && 0 <= v889 && v889 < 4);
                    v886[v892] = v895;
                    v889 += 1 ;
                }
                v887 += 1 ;
            }
            float v896[4];
            int v897;
            v897 = 0;
            while (while_method_5(v897)){
                int v899;
                v899 = 0;
                while (while_method_8(v899)){
                    assert("Tensor range check" && 0 <= v897 && v897 < 1);
                    assert("Tensor range check" && 0 <= v899 && v899 < 4);
                    int v901;
                    v901 = 4 * v897;
                    int v902;
                    v902 = v901 + v899;
                    float v903;
                    v903 = v886[v902];
                    bool v904;
                    v904 = v826[v902];
                    float v905;
                    if (v904){
                        v905 = v903;
                    } else {
                        v905 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v897 && v897 < 1);
                    assert("Tensor range check" && 0 <= v899 && v899 < 4);
                    v896[v902] = v905;
                    v899 += 1 ;
                }
                v897 += 1 ;
            }
            float v906;
            v906 = 0.0f;
            int v907;
            v907 = 0;
            while (while_method_5(v907)){
                int v909;
                v909 = 0;
                while (while_method_8(v909)){
                    assert("Tensor range check" && 0 <= v907 && v907 < 1);
                    assert("Tensor range check" && 0 <= v909 && v909 < 4);
                    int v911;
                    v911 = 4 * v907;
                    int v912;
                    v912 = v911 + v909;
                    float v913;
                    v913 = v896[v912];
                    float v914;
                    v914 = v906 + v913;
                    v906 = v914;
                    v909 += 1 ;
                }
                v907 += 1 ;
            }
            auto v915 = cooperative_groups::coalesced_threads();
            int v916;
            v916 = threadIdx.x;
            int v917;
            v917 = v916 / 8;
            auto v918 = cooperative_groups::labeled_partition(v915,v917);
            float v919;
            v919 = cooperative_groups::reduce(v918, v906, v883);
            float v920;
            v920 = v919 / v859;
            float v921;
            v921 = sqrt(v920);
            bool v922;
            v922 = v859 > 1.0f;
            float v926;
            if (v922){
                float v923;
                v923 = v921 * v859;
                float v924;
                v924 = v859 - 1.0f;
                float v925;
                v925 = v923 / v924;
                v926 = v925;
            } else {
                v926 = 0.0f;
            }
            float v927[4];
            int v928;
            v928 = 0;
            while (while_method_5(v928)){
                int v930;
                v930 = 0;
                while (while_method_8(v930)){
                    assert("Tensor range check" && 0 <= v928 && v928 < 1);
                    assert("Tensor range check" && 0 <= v930 && v930 < 4);
                    int v932;
                    v932 = 4 * v928;
                    int v933;
                    v933 = v932 + v930;
                    float v934;
                    v934 = v771[v933];
                    bool v935;
                    v935 = v826[v933];
                    float v936;
                    v936 = curand_normal(&v367);
                    bool v937;
                    v937 = v926 >= 0.1f;
                    float v938;
                    if (v937){
                        v938 = v926;
                    } else {
                        v938 = 0.1f;
                    }
                    float v939;
                    v939 = v936 * v938;
                    float v940;
                    v940 = v939 + v885;
                    float v941;
                    if (v935){
                        v941 = v934;
                    } else {
                        v941 = v940;
                    }
                    assert("Tensor range check" && 0 <= v928 && v928 < 1);
                    assert("Tensor range check" && 0 <= v930 && v930 < 4);
                    v927[v933] = v941;
                    v930 += 1 ;
                }
                v928 += 1 ;
            }
            assert("Tensor range check" && 0 <= v763 && v763 < 64);
            assert("Tensor range check" && 0 <= v762 && v762 < 2);
            int v942;
            v942 = 0;
            while (while_method_5(v942)){
                assert("Tensor range check" && 0 <= v942 && v942 < 1);
                int v944;
                v944 = 32 * v942;
                int v945;
                v945 = v944 + v770;
                assert("Tensor range check" && 0 <= v942 && v942 < 1);
                int v946;
                v946 = 4 * v942;
                int4* v947;
                v947 = reinterpret_cast<int4*>(v927 + v946);
                int4* v948;
                v948 = reinterpret_cast<int4*>(v374 + v945);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v947) % 16 == 0 && reinterpret_cast<unsigned long long>(v948) % 16 == 0);
                *v948 = *v947;
                v942 += 1 ;
            }
            v757 += 24 ;
        }
        v365.sync() ;
        static float v949[8192];
        int v950;
        v950 = threadIdx.x;
        int v951;
        v951 = blockIdx.x;
        int v952;
        v952 = v951 * 256;
        int v953;
        v953 = v950 + v952;
        int v954;
        v954 = v953 / 32;
        int v955;
        v955 = v954;
        while (while_method_16(v955)){
            bool v957;
            v957 = 0 <= v955;
            bool v958;
            v958 = v957 == false;
            if (v958){
                assert("The index needs to be zero or positive." && v957);
            } else {
            }
            int v960;
            v960 = v955 % 128;
            int v961;
            v961 = v955 / 128;
            bool v962;
            v962 = v961 < 64;
            bool v963;
            v963 = v962 == false;
            if (v963){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v962);
            } else {
            }
            assert("Tensor range check" && 0 <= v961 && v961 < 64);
            assert("Tensor range check" && 0 <= v960 && v960 < 128);
            int v965;
            v965 = 32 * v960;
            int v966;
            v966 = 4096 * v961;
            int v967;
            v967 = v966 + v965;
            float v968;
            v968 = 0.0f;
            int v969;
            v969 = threadIdx.x;
            int v970;
            v970 = v969 % 32;
            int v971;
            v971 = v970;
            while (while_method_6(v971)){
                bool v973;
                v973 = 0 <= v971;
                bool v974;
                v974 = v973 == false;
                if (v974){
                    assert("The index needs to be zero or positive." && v973);
                } else {
                }
                bool v976;
                v976 = v971 < 8;
                bool v977;
                v977 = v976 == false;
                if (v977){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v976);
                } else {
                }
                assert("Tensor range check" && 0 <= v971 && v971 < 8);
                int v979;
                v979 = 4 * v971;
                int v980;
                v980 = v979 + v967;
                float v981[4];
                int4* v982;
                v982 = reinterpret_cast<int4*>(v370 + v980);
                int4* v983;
                v983 = reinterpret_cast<int4*>(v981 + 0);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v982) % 16 == 0 && reinterpret_cast<unsigned long long>(v983) % 16 == 0);
                *v983 = *v982;
                int v984;
                v984 = 0;
                while (while_method_8(v984)){
                    assert("Tensor range check" && 0 <= v984 && v984 < 4);
                    float v986;
                    v986 = v981[v984];
                    float v987;
                    v987 = v986 * v986;
                    float v988;
                    v988 = v968 + v987;
                    v968 = v988;
                    v984 += 1 ;
                }
                v971 += 32 ;
            }
            __syncwarp();
            auto v989 = cooperative_groups::coalesced_threads();
            Closure0 v990{};
            float v991;
            v991 = cooperative_groups::reduce(v989, v968, v990);
            float v992;
            v992 = sqrt(v991);
            assert("Tensor range check" && 0 <= v961 && v961 < 64);
            assert("Tensor range check" && 0 <= v960 && v960 < 128);
            int v993;
            v993 = 128 * v961;
            int v994;
            v994 = v993 + v960;
            v949[v994] = v992;
            v955 += 192 ;
        }
        __syncthreads();
        v365.sync() ;
        float v995;
        v995 = 0.0f;
        int v996;
        v996 = threadIdx.x;
        int v997;
        v997 = blockIdx.x;
        int v998;
        v998 = v997 * 256;
        int v999;
        v999 = v996 + v998;
        int v1000;
        v1000 = v999;
        while (while_method_17(v1000)){
            bool v1002;
            v1002 = 0 <= v1000;
            bool v1003;
            v1003 = v1002 == false;
            if (v1003){
                assert("The index needs to be zero or positive." && v1002);
            } else {
            }
            int v1005;
            v1005 = v1000 % 32;
            int v1006;
            v1006 = v1000 / 32;
            bool v1007;
            v1007 = v1006 < 64;
            bool v1008;
            v1008 = v1007 == false;
            if (v1008){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1007);
            } else {
            }
            assert("Tensor range check" && 0 <= v1006 && v1006 < 64);
            assert("Tensor range check" && 0 <= v1005 && v1005 < 32);
            int v1010;
            v1010 = 4 * v1005;
            int v1011;
            v1011 = 128 * v1006;
            int v1012;
            v1012 = v1011 + v1010;
            float v1013[4];
            int4* v1014;
            v1014 = reinterpret_cast<int4*>(v949 + v1012);
            int4* v1015;
            v1015 = reinterpret_cast<int4*>(v1013 + 0);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1014) % 16 == 0 && reinterpret_cast<unsigned long long>(v1015) % 16 == 0);
            *v1015 = *v1014;
            int v1016; float v1017;
            Tuple13 tmp54 = Tuple13{0, v995};
            v1016 = tmp54.v0; v1017 = tmp54.v1;
            while (while_method_8(v1016)){
                assert("Tensor range check" && 0 <= v1016 && v1016 < 4);
                float v1019;
                v1019 = v1013[v1016];
                bool v1020;
                v1020 = v1017 >= v1019;
                float v1021;
                if (v1020){
                    v1021 = v1017;
                } else {
                    v1021 = v1019;
                }
                v1017 = v1021;
                v1016 += 1 ;
            }
            v995 = v1017;
            v1000 += 6144 ;
        }
        __syncwarp();
        auto v1022 = cooperative_groups::coalesced_threads();
        Closure7 v1023{};
        float v1024;
        v1024 = cooperative_groups::reduce(v1022, v995, v1023);
        int v1025;
        v1025 = threadIdx.x;
        int v1026;
        v1026 = v1025 / 32;
        extern __shared__ unsigned char v1027[];
        float * v1028;
        v1028 = reinterpret_cast<float *>(&v1027[0ull]);
        assert("Tensor range check" && 0 <= v1026 && v1026 < 8);
        v1028[v1026] = v1024;
        __syncthreads();
        int v1030;
        v1030 = threadIdx.x;
        int v1031;
        v1031 = v1030 % 32;
        bool v1032;
        v1032 = v1031 < 8;
        float v1034;
        if (v1032){
            assert("Tensor range check" && 0 <= v1031 && v1031 < 8);
            float v1033;
            v1033 = v1028[v1031];
            v1034 = v1033;
        } else {
            v1034 = 0.0f;
        }
        __syncthreads();
        auto v1035 = cooperative_groups::coalesced_threads();
        float v1036;
        v1036 = cooperative_groups::reduce(v1035, v1034, v1023);
        int v1037;
        v1037 = blockIdx.x;
        static float v1038[24];
        assert("Tensor range check" && 0 <= v1037 && v1037 < 24);
        v1038[v1037] = v1036;
        v365.sync() ;
        float v1039;
        v1039 = 0.0f;
        int v1040;
        v1040 = threadIdx.x;
        int v1041;
        v1041 = v1040 % 32;
        int v1042;
        v1042 = v1041;
        while (while_method_18(v1042)){
            bool v1044;
            v1044 = 0 <= v1042;
            bool v1045;
            v1045 = v1044 == false;
            if (v1045){
                assert("The index needs to be zero or positive." && v1044);
            } else {
            }
            bool v1047;
            v1047 = v1042 < 24;
            bool v1048;
            v1048 = v1047 == false;
            if (v1048){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1047);
            } else {
            }
            assert("Tensor range check" && 0 <= v1042 && v1042 < 24);
            float v1050;
            v1050 = v1038[v1042];
            bool v1051;
            v1051 = v1039 >= v1050;
            float v1052;
            if (v1051){
                v1052 = v1039;
            } else {
                v1052 = v1050;
            }
            v1039 = v1052;
            v1042 += 32 ;
        }
        __syncwarp();
        auto v1053 = cooperative_groups::coalesced_threads();
        float v1054;
        v1054 = cooperative_groups::reduce(v1053, v1039, v1023);
        int v1055;
        v1055 = threadIdx.x;
        int v1056;
        v1056 = blockIdx.x;
        int v1057;
        v1057 = v1056 * 256;
        int v1058;
        v1058 = v1055 + v1057;
        bool v1059;
        v1059 = v1058 == 0;
        if (v1059){
            cuda::counting_semaphore<cuda::thread_scope_system, 1> & v1060 = console_lock;
            auto v1061 = cooperative_groups::coalesced_threads();
            v1060.acquire();
            printf("{%s = %f}\n","max_norm", v1054);
            v1060.release();
            v1061.sync() ;
        } else {
        }
        __syncwarp();
        static float v1064[4096];
        int v1065;
        v1065 = threadIdx.x;
        int v1066;
        v1066 = blockIdx.x;
        int v1067;
        v1067 = v1066 * 256;
        int v1068;
        v1068 = v1065 + v1067;
        int v1069;
        v1069 = v1068 / 32;
        int v1070;
        v1070 = v1069;
        while (while_method_10(v1070)){
            bool v1072;
            v1072 = 0 <= v1070;
            bool v1073;
            v1073 = v1072 == false;
            if (v1073){
                assert("The index needs to be zero or positive." && v1072);
            } else {
            }
            int v1075;
            v1075 = v1070 % 64;
            int v1076;
            v1076 = v1070 / 64;
            bool v1077;
            v1077 = v1076 < 64;
            bool v1078;
            v1078 = v1077 == false;
            if (v1078){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1077);
            } else {
            }
            assert("Tensor range check" && 0 <= v1076 && v1076 < 64);
            assert("Tensor range check" && 0 <= v1075 && v1075 < 64);
            int v1080;
            v1080 = 32 * v1075;
            int v1081;
            v1081 = 2048 * v1076;
            int v1082;
            v1082 = v1081 + v1080;
            float v1083;
            v1083 = 0.0f;
            int v1084;
            v1084 = threadIdx.x;
            int v1085;
            v1085 = v1084 % 32;
            int v1086;
            v1086 = v1085;
            while (while_method_6(v1086)){
                bool v1088;
                v1088 = 0 <= v1086;
                bool v1089;
                v1089 = v1088 == false;
                if (v1089){
                    assert("The index needs to be zero or positive." && v1088);
                } else {
                }
                bool v1091;
                v1091 = v1086 < 8;
                bool v1092;
                v1092 = v1091 == false;
                if (v1092){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1091);
                } else {
                }
                assert("Tensor range check" && 0 <= v1086 && v1086 < 8);
                int v1094;
                v1094 = 4 * v1086;
                int v1095;
                v1095 = v1094 + v1082;
                float v1096[4];
                int4* v1097;
                v1097 = reinterpret_cast<int4*>(v374 + v1095);
                int4* v1098;
                v1098 = reinterpret_cast<int4*>(v1096 + 0);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1097) % 16 == 0 && reinterpret_cast<unsigned long long>(v1098) % 16 == 0);
                *v1098 = *v1097;
                int v1099;
                v1099 = 0;
                while (while_method_8(v1099)){
                    assert("Tensor range check" && 0 <= v1099 && v1099 < 4);
                    float v1101;
                    v1101 = v1096[v1099];
                    float v1102;
                    v1102 = v1101 * v1101;
                    float v1103;
                    v1103 = v1083 + v1102;
                    v1083 = v1103;
                    v1099 += 1 ;
                }
                v1086 += 32 ;
            }
            __syncwarp();
            auto v1104 = cooperative_groups::coalesced_threads();
            Closure0 v1105{};
            float v1106;
            v1106 = cooperative_groups::reduce(v1104, v1083, v1105);
            float v1107;
            v1107 = sqrt(v1106);
            assert("Tensor range check" && 0 <= v1076 && v1076 < 64);
            assert("Tensor range check" && 0 <= v1075 && v1075 < 64);
            int v1108;
            v1108 = 64 * v1076;
            int v1109;
            v1109 = v1108 + v1075;
            v1064[v1109] = v1107;
            v1070 += 192 ;
        }
        __syncthreads();
        v365.sync() ;
        float v1110;
        v1110 = 0.0f;
        int v1111;
        v1111 = threadIdx.x;
        int v1112;
        v1112 = blockIdx.x;
        int v1113;
        v1113 = v1112 * 256;
        int v1114;
        v1114 = v1111 + v1113;
        int v1115;
        v1115 = v1114;
        while (while_method_19(v1115)){
            bool v1117;
            v1117 = 0 <= v1115;
            bool v1118;
            v1118 = v1117 == false;
            if (v1118){
                assert("The index needs to be zero or positive." && v1117);
            } else {
            }
            int v1120;
            v1120 = v1115 % 16;
            int v1121;
            v1121 = v1115 / 16;
            bool v1122;
            v1122 = v1121 < 64;
            bool v1123;
            v1123 = v1122 == false;
            if (v1123){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1122);
            } else {
            }
            assert("Tensor range check" && 0 <= v1121 && v1121 < 64);
            assert("Tensor range check" && 0 <= v1120 && v1120 < 16);
            int v1125;
            v1125 = 4 * v1120;
            int v1126;
            v1126 = 64 * v1121;
            int v1127;
            v1127 = v1126 + v1125;
            float v1128[4];
            int4* v1129;
            v1129 = reinterpret_cast<int4*>(v1064 + v1127);
            int4* v1130;
            v1130 = reinterpret_cast<int4*>(v1128 + 0);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1129) % 16 == 0 && reinterpret_cast<unsigned long long>(v1130) % 16 == 0);
            *v1130 = *v1129;
            int v1131; float v1132;
            Tuple13 tmp55 = Tuple13{0, v1110};
            v1131 = tmp55.v0; v1132 = tmp55.v1;
            while (while_method_8(v1131)){
                assert("Tensor range check" && 0 <= v1131 && v1131 < 4);
                float v1134;
                v1134 = v1128[v1131];
                bool v1135;
                v1135 = v1132 >= v1134;
                float v1136;
                if (v1135){
                    v1136 = v1132;
                } else {
                    v1136 = v1134;
                }
                v1132 = v1136;
                v1131 += 1 ;
            }
            v1110 = v1132;
            v1115 += 6144 ;
        }
        __syncwarp();
        auto v1137 = cooperative_groups::coalesced_threads();
        float v1138;
        v1138 = cooperative_groups::reduce(v1137, v1110, v1023);
        int v1139;
        v1139 = threadIdx.x;
        int v1140;
        v1140 = v1139 / 32;
        extern __shared__ unsigned char v1141[];
        float * v1142;
        v1142 = reinterpret_cast<float *>(&v1141[0ull]);
        assert("Tensor range check" && 0 <= v1140 && v1140 < 8);
        v1142[v1140] = v1138;
        __syncthreads();
        int v1144;
        v1144 = threadIdx.x;
        int v1145;
        v1145 = v1144 % 32;
        bool v1146;
        v1146 = v1145 < 8;
        float v1148;
        if (v1146){
            assert("Tensor range check" && 0 <= v1145 && v1145 < 8);
            float v1147;
            v1147 = v1142[v1145];
            v1148 = v1147;
        } else {
            v1148 = 0.0f;
        }
        __syncthreads();
        auto v1149 = cooperative_groups::coalesced_threads();
        float v1150;
        v1150 = cooperative_groups::reduce(v1149, v1148, v1023);
        int v1151;
        v1151 = blockIdx.x;
        static float v1152[24];
        assert("Tensor range check" && 0 <= v1151 && v1151 < 24);
        v1152[v1151] = v1150;
        v365.sync() ;
        float v1153;
        v1153 = 0.0f;
        int v1154;
        v1154 = threadIdx.x;
        int v1155;
        v1155 = v1154 % 32;
        int v1156;
        v1156 = v1155;
        while (while_method_18(v1156)){
            bool v1158;
            v1158 = 0 <= v1156;
            bool v1159;
            v1159 = v1158 == false;
            if (v1159){
                assert("The index needs to be zero or positive." && v1158);
            } else {
            }
            bool v1161;
            v1161 = v1156 < 24;
            bool v1162;
            v1162 = v1161 == false;
            if (v1162){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1161);
            } else {
            }
            assert("Tensor range check" && 0 <= v1156 && v1156 < 24);
            float v1164;
            v1164 = v1152[v1156];
            bool v1165;
            v1165 = v1153 >= v1164;
            float v1166;
            if (v1165){
                v1166 = v1153;
            } else {
                v1166 = v1164;
            }
            v1153 = v1166;
            v1156 += 32 ;
        }
        __syncwarp();
        auto v1167 = cooperative_groups::coalesced_threads();
        float v1168;
        v1168 = cooperative_groups::reduce(v1167, v1153, v1023);
        int v1169;
        v1169 = threadIdx.x;
        int v1170;
        v1170 = blockIdx.x;
        int v1171;
        v1171 = v1170 * 256;
        int v1172;
        v1172 = v1169 + v1171;
        bool v1173;
        v1173 = v1172 == 0;
        if (v1173){
            cuda::counting_semaphore<cuda::thread_scope_system, 1> & v1174 = console_lock;
            auto v1175 = cooperative_groups::coalesced_threads();
            v1174.acquire();
            printf("{%s = %f}\n","max_norm", v1168);
            v1174.release();
            v1175.sync() ;
        } else {
        }
        __syncwarp();
        extern __shared__ unsigned char v1178[];
        float * v1179;
        v1179 = reinterpret_cast<float *>(&v1178[0ull]);
        int v1181;
        v1181 = blockIdx.x;
        int v1182;
        v1182 = v1181;
        while (while_method_9(v1182)){
            bool v1184;
            v1184 = 0 <= v1182;
            bool v1185;
            v1185 = v1184 == false;
            if (v1185){
                assert("The index needs to be zero or positive." && v1184);
            } else {
            }
            int v1187;
            v1187 = v1182 % 1;
            bool v1188;
            v1188 = v1182 < 16;
            bool v1189;
            v1189 = v1188 == false;
            if (v1189){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1188);
            } else {
            }
            assert("Tensor range check" && 0 <= v1182 && v1182 < 16);
            assert("Tensor range check" && 0 <= v1187 && v1187 < 1);
            int v1191;
            v1191 = 32 * v1187;
            int v1192;
            v1192 = 16384 * v1182;
            int v1193;
            v1193 = v1192 + v1191;
            int v1194;
            v1194 = 262144 * v1187;
            int v1195;
            v1195 = 512 * v1182;
            int v1196;
            v1196 = v1195 + v1194;
            int v1197;
            v1197 = threadIdx.x;
            int v1198;
            v1198 = v1197;
            while (while_method_14(v1198)){
                bool v1200;
                v1200 = 0 <= v1198;
                bool v1201;
                v1201 = v1200 == false;
                if (v1201){
                    assert("The index needs to be zero or positive." && v1200);
                } else {
                }
                int v1203;
                v1203 = v1198 % 32;
                int v1204;
                v1204 = v1198 / 32;
                bool v1205;
                v1205 = v1204 < 512;
                bool v1206;
                v1206 = v1205 == false;
                if (v1206){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1205);
                } else {
                }
                assert("Tensor range check" && 0 <= v1204 && v1204 < 512);
                assert("Tensor range check" && 0 <= v1203 && v1203 < 32);
                int v1208;
                v1208 = v1203 + v1193;
                int v1209;
                v1209 = 32 * v1204;
                int v1210;
                v1210 = v1209 + v1208;
                float v1211;
                v1211 = v370[v1210];
                assert("Tensor range check" && 0 <= v1204 && v1204 < 512);
                assert("Tensor range check" && 0 <= v1203 && v1203 < 32);
                int v1212;
                v1212 = 33 * v1204;
                int v1213;
                v1213 = v1212 + v1203;
                v1179[v1213] = v1211;
                v1198 += 256 ;
            }
            __syncthreads();
            int v1214;
            v1214 = threadIdx.x;
            int v1215;
            v1215 = v1214;
            while (while_method_14(v1215)){
                bool v1217;
                v1217 = 0 <= v1215;
                bool v1218;
                v1218 = v1217 == false;
                if (v1218){
                    assert("The index needs to be zero or positive." && v1217);
                } else {
                }
                int v1220;
                v1220 = v1215 % 512;
                int v1221;
                v1221 = v1215 / 512;
                bool v1222;
                v1222 = v1221 < 32;
                bool v1223;
                v1223 = v1222 == false;
                if (v1223){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1222);
                } else {
                }
                assert("Tensor range check" && 0 <= v1221 && v1221 < 32);
                assert("Tensor range check" && 0 <= v1220 && v1220 < 512);
                int v1225;
                v1225 = 33 * v1220;
                int v1226;
                v1226 = v1221 + v1225;
                float v1227;
                v1227 = v1179[v1226];
                assert("Tensor range check" && 0 <= v1221 && v1221 < 32);
                assert("Tensor range check" && 0 <= v1220 && v1220 < 512);
                int v1228;
                v1228 = v1220 + v1196;
                int v1229;
                v1229 = 8192 * v1221;
                int v1230;
                v1230 = v1229 + v1228;
                v368[v1230] = v1227;
                v1215 += 256 ;
            }
            __syncthreads();
            v1182 += 24 ;
        }
        extern __shared__ unsigned char v1231[];
        float * v1232;
        v1232 = reinterpret_cast<float *>(&v1231[0ull]);
        int v1234;
        v1234 = blockIdx.x;
        int v1235;
        v1235 = v1234;
        while (while_method_6(v1235)){
            bool v1237;
            v1237 = 0 <= v1235;
            bool v1238;
            v1238 = v1237 == false;
            if (v1238){
                assert("The index needs to be zero or positive." && v1237);
            } else {
            }
            int v1240;
            v1240 = v1235 % 1;
            bool v1241;
            v1241 = v1235 < 8;
            bool v1242;
            v1242 = v1241 == false;
            if (v1242){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1241);
            } else {
            }
            assert("Tensor range check" && 0 <= v1235 && v1235 < 8);
            assert("Tensor range check" && 0 <= v1240 && v1240 < 1);
            int v1244;
            v1244 = 32 * v1240;
            int v1245;
            v1245 = 16384 * v1235;
            int v1246;
            v1246 = v1245 + v1244;
            int v1247;
            v1247 = 131072 * v1240;
            int v1248;
            v1248 = 512 * v1235;
            int v1249;
            v1249 = v1248 + v1247;
            int v1250;
            v1250 = threadIdx.x;
            int v1251;
            v1251 = v1250;
            while (while_method_14(v1251)){
                bool v1253;
                v1253 = 0 <= v1251;
                bool v1254;
                v1254 = v1253 == false;
                if (v1254){
                    assert("The index needs to be zero or positive." && v1253);
                } else {
                }
                int v1256;
                v1256 = v1251 % 32;
                int v1257;
                v1257 = v1251 / 32;
                bool v1258;
                v1258 = v1257 < 512;
                bool v1259;
                v1259 = v1258 == false;
                if (v1259){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1258);
                } else {
                }
                assert("Tensor range check" && 0 <= v1257 && v1257 < 512);
                assert("Tensor range check" && 0 <= v1256 && v1256 < 32);
                int v1261;
                v1261 = v1256 + v1246;
                int v1262;
                v1262 = 32 * v1257;
                int v1263;
                v1263 = v1262 + v1261;
                float v1264;
                v1264 = v374[v1263];
                assert("Tensor range check" && 0 <= v1257 && v1257 < 512);
                assert("Tensor range check" && 0 <= v1256 && v1256 < 32);
                int v1265;
                v1265 = 33 * v1257;
                int v1266;
                v1266 = v1265 + v1256;
                v1232[v1266] = v1264;
                v1251 += 256 ;
            }
            __syncthreads();
            int v1267;
            v1267 = threadIdx.x;
            int v1268;
            v1268 = v1267;
            while (while_method_14(v1268)){
                bool v1270;
                v1270 = 0 <= v1268;
                bool v1271;
                v1271 = v1270 == false;
                if (v1271){
                    assert("The index needs to be zero or positive." && v1270);
                } else {
                }
                int v1273;
                v1273 = v1268 % 512;
                int v1274;
                v1274 = v1268 / 512;
                bool v1275;
                v1275 = v1274 < 32;
                bool v1276;
                v1276 = v1275 == false;
                if (v1276){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1275);
                } else {
                }
                assert("Tensor range check" && 0 <= v1274 && v1274 < 32);
                assert("Tensor range check" && 0 <= v1273 && v1273 < 512);
                int v1278;
                v1278 = 33 * v1273;
                int v1279;
                v1279 = v1274 + v1278;
                float v1280;
                v1280 = v1232[v1279];
                assert("Tensor range check" && 0 <= v1274 && v1274 < 32);
                assert("Tensor range check" && 0 <= v1273 && v1273 < 512);
                int v1281;
                v1281 = v1273 + v1249;
                int v1282;
                v1282 = 4096 * v1274;
                int v1283;
                v1283 = v1282 + v1281;
                v372[v1283] = v1280;
                v1268 += 256 ;
            }
            __syncthreads();
            v1235 += 24 ;
        }
        v365.sync() ;
        v27 += 1 ;
    }
    cooperative_groups::grid_group & v1284 = v26.v1;
    cooperative_groups::grid_group & v1285 = v1284;
    int v1286;
    v1286 = threadIdx.x;
    int v1287;
    v1287 = blockIdx.x;
    int v1288;
    v1288 = v1287 * 256;
    int v1289;
    v1289 = v1286 + v1288;
    int v1290;
    v1290 = v1289;
    while (while_method_17(v1290)){
        bool v1292;
        v1292 = 0 <= v1290;
        bool v1293;
        v1293 = v1292 == false;
        if (v1293){
            assert("The index needs to be zero or positive." && v1292);
        } else {
        }
        int v1295;
        v1295 = v1290 % 64;
        int v1296;
        v1296 = v1290 / 64;
        bool v1297;
        v1297 = v1296 < 32;
        bool v1298;
        v1298 = v1297 == false;
        if (v1298){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1297);
        } else {
        }
        assert("Tensor range check" && 0 <= v1296 && v1296 < 32);
        assert("Tensor range check" && 0 <= v1295 && v1295 < 64);
        int v1300;
        v1300 = 4 * v1295;
        int v1301;
        v1301 = 256 * v1296;
        int v1302;
        v1302 = v1301 + v1300;
        assert("Tensor range check" && 0 <= v1296 && v1296 < 32);
        assert("Tensor range check" && 0 <= v1295 && v1295 < 64);
        float v1303[4];
        float v1304[4];
        float v1305[4];
        int4* v1306;
        v1306 = reinterpret_cast<int4*>(v3 + v1302);
        int4* v1307;
        v1307 = reinterpret_cast<int4*>(v1303 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1306) % 16 == 0 && reinterpret_cast<unsigned long long>(v1307) % 16 == 0);
        *v1307 = *v1306;
        int4* v1308;
        v1308 = reinterpret_cast<int4*>(v4 + v1302);
        int4* v1309;
        v1309 = reinterpret_cast<int4*>(v1304 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1308) % 16 == 0 && reinterpret_cast<unsigned long long>(v1309) % 16 == 0);
        *v1309 = *v1308;
        // Pushing the loop unrolling to: 0
        int v1310;
        v1310 = 0;
        #pragma unroll
        while (while_method_8(v1310)){
            assert("Tensor range check" && 0 <= v1310 && v1310 < 4);
            float v1312;
            v1312 = v1303[v1310];
            float v1313;
            v1313 = v1304[v1310];
            bool v1314;
            v1314 = v1313 == 0.0f;
            bool v1315;
            v1315 = v1314 != true;
            float v1317;
            if (v1315){
                float v1316;
                v1316 = v1312 / v1313;
                v1317 = v1316;
            } else {
                v1317 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v1310 && v1310 < 4);
            v1305[v1310] = v1317;
            v1310 += 1 ;
        }
        // Poping the loop unrolling to: 0
        int4* v1318;
        v1318 = reinterpret_cast<int4*>(v1305 + 0);
        int4* v1319;
        v1319 = reinterpret_cast<int4*>(v5 + v1302);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1318) % 16 == 0 && reinterpret_cast<unsigned long long>(v1319) % 16 == 0);
        *v1319 = *v1318;
        v1290 += 6144 ;
    }
    v1285.sync() ;
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
    while (while_method_12(v27)){
        int v29;
        v29 = 0;
        while (while_method_6(v29)){
            Union4 v31;
            v31 = Union4{Union4_1{}};
            method_49(v0, v1, v2, v26, v31);
            static_array<float,2> & v32 = v26.v4;
            float * v33;
            v33 = reinterpret_cast<float *>(&v1[7864320ull]);
            int * v35;
            v35 = reinterpret_cast<int *>(&v0[1572864ull]);
            bool * v37;
            v37 = reinterpret_cast<bool *>(&v0[1572880ull]);
            float * v39;
            v39 = reinterpret_cast<float *>(&v0[1572912ull]);
            float * v41;
            v41 = reinterpret_cast<float *>(&v0[1573040ull]);
            double * v43;
            v43 = reinterpret_cast<double *>(&v1[58195968ull]);
            double * v45;
            v45 = reinterpret_cast<double *>(&v1[61341696ull]);
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
            double * v52;
            v52 = v43+v51;
            double * v54;
            v54 = v45+v51;
            float v56[2];
            int v57;
            v57 = 0;
            while (while_method_0(v57)){
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
                float v64;
                v64 = v32[v57];
                assert("Tensor range check" && 0 <= v57 && v57 < 2);
                v56[v57] = v64;
                v57 += 1 ;
            }
            double v66[2];
            int v67;
            v67 = 0;
            while (while_method_0(v67)){
                int v69; double v70;
                Tuple11 tmp72 = Tuple11{0, 0.0};
                v69 = tmp72.v0; v70 = tmp72.v1;
                while (while_method_4(v69)){
                    assert("Tensor range check" && 0 <= v69 && v69 < 32);
                    bool v72;
                    v72 = v37[v69];
                    bool v73;
                    v73 = v72 == false;
                    double v80;
                    if (v73){
                        v80 = 0.0;
                    } else {
                        assert("Tensor range check" && 0 <= v69 && v69 < 32);
                        assert("Tensor range check" && 0 <= v67 && v67 < 2);
                        int v74;
                        v74 = 12288 * v69;
                        int v75;
                        v75 = v74 + v67;
                        double v76;
                        v76 = v52[v75];
                        double v77;
                        v77 = v54[v75];
                        double v78;
                        v78 = v76 - v77;
                        double v79;
                        v79 = exp(v78);
                        v80 = v79;
                    }
                    double v81;
                    v81 = v70 + v80;
                    v70 = v81;
                    v69 += 1 ;
                }
                assert("Tensor range check" && 0 <= v67 && v67 < 2);
                v66[v67] = v70;
                v67 += 1 ;
            }
            double v82;
            v82 = 1.0;
            int v83;
            v83 = 0;
            while (while_method_0(v83)){
                assert("Tensor range check" && 0 <= v83 && v83 < 2);
                double v85;
                v85 = v66[v83];
                double v86;
                v86 = v82 * v85;
                v82 = v86;
                v83 += 1 ;
            }
            double v87[64];
            int v88;
            v88 = 0;
            while (while_method_4(v88)){
                int v90;
                v90 = 0;
                while (while_method_0(v90)){
                    bool v92;
                    v92 = v82 == 0.0;
                    bool v93;
                    v93 = v92 != true;
                    double v103;
                    if (v93){
                        assert("Tensor range check" && 0 <= v90 && v90 < 2);
                        double v94;
                        v94 = v66[v90];
                        double v95;
                        v95 = v82 / v94;
                        assert("Tensor range check" && 0 <= v88 && v88 < 32);
                        assert("Tensor range check" && 0 <= v90 && v90 < 2);
                        int v96;
                        v96 = 12288 * v88;
                        int v97;
                        v97 = v96 + v90;
                        double v98;
                        v98 = v52[v97];
                        double v99;
                        v99 = v54[v97];
                        double v100;
                        v100 = v98 - v99;
                        double v101;
                        v101 = exp(v100);
                        double v102;
                        v102 = v95 * v101;
                        v103 = v102;
                    } else {
                        v103 = 0.0;
                    }
                    bool v104;
                    v104 = isnan(v103);
                    bool v105;
                    v105 = v104 == false;
                    bool v106;
                    v106 = v105 == false;
                    if (v106){
                        assert("The path probability after integration should not be a nan in integrate_rewards_." && v105);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v88 && v88 < 32);
                    assert("Tensor range check" && 0 <= v90 && v90 < 2);
                    int v108;
                    v108 = 2 * v88;
                    int v109;
                    v109 = v108 + v90;
                    v87[v109] = v103;
                    v90 += 1 ;
                }
                v88 += 1 ;
            }
            float v110[32];
            float v111[32];
            int v112;
            v112 = 0;
            while (while_method_4(v112)){
                int v114; float v115; double v116;
                Tuple12 tmp73 = Tuple12{0, 0.0f, 0.0};
                v114 = tmp73.v0; v115 = tmp73.v1; v116 = tmp73.v2;
                while (while_method_0(v114)){
                    assert("Tensor range check" && 0 <= v112 && v112 < 32);
                    assert("Tensor range check" && 0 <= v114 && v114 < 2);
                    int v118;
                    v118 = 2 * v112;
                    int v119;
                    v119 = v118 + v114;
                    double v120;
                    v120 = v87[v119];
                    assert("Tensor range check" && 0 <= v114 && v114 < 2);
                    float v121;
                    v121 = v56[v114];
                    float v122;
                    v122 = (float)v120;
                    float v123;
                    v123 = v122 * v121;
                    float v124;
                    v124 = v115 + v123;
                    double v125;
                    v125 = v116 + v120;
                    v115 = v124;
                    v116 = v125;
                    v114 += 1 ;
                }
                float v126;
                v126 = (float)v116;
                assert("Tensor range check" && 0 <= v112 && v112 < 32);
                v110[v112] = v115;
                v111[v112] = v126;
                v112 += 1 ;
            }
            int v127;
            v127 = 0;
            while (while_method_4(v127)){
                assert("Tensor range check" && 0 <= v127 && v127 < 32);
                float v129;
                v129 = v110[v127];
                float v130;
                v130 = v111[v127];
                bool v131;
                v131 = isnan(v130);
                bool v132;
                v132 = v131 == false;
                bool v133;
                v133 = v132 == false;
                if (v133){
                    assert("The path probability after integration should not be a nan in calculate updates." && v132);
                } else {
                }
                float v135;
                v135 = v129 * v130;
                assert("Tensor range check" && 0 <= v127 && v127 < 32);
                float * v136;
                v136 = v39+v127;
                float * v138;
                v138 = v41+v127;
                float v140;
                v140 = atomicAdd(v136,v135);
                float v141;
                v141 = atomicAdd(v138,v130);
                v127 += 1 ;
            }
            int v142;
            v142 = threadIdx.x;
            int v143;
            v143 = blockIdx.x;
            int v144;
            v144 = v143 * 256;
            int v145;
            v145 = v142 + v144;
            int v146;
            v146 = 0;
            while (while_method_4(v146)){
                assert("Tensor range check" && 0 <= v146 && v146 < 32);
                int v148;
                v148 = 12288 * v146;
                assert("Tensor range check" && 0 <= v145 && v145 < 6144);
                int v149;
                v149 = 2 * v145;
                int v150;
                v150 = v149 + v148;
                double * v151;
                v151 = v43+v150;
                double * v153;
                v153 = v45+v150;
                double * v155;
                v155 = v43+v150;
                double * v157;
                v157 = v45+v150;
                int v159;
                v159 = sizeof(double *);
                unsigned long long v160;
                v160 = (unsigned long long)v159;
                unsigned long long v161;
                v161 = 256ull * v160;
                unsigned long long v162;
                v162 = v161 + 16ull;
                unsigned long long v163;
                v163 = v162 - 1ull;
                unsigned long long v164;
                v164 = v163 % 16ull;
                unsigned long long v165;
                v165 = v163 - v164;
                unsigned long long v166;
                v166 = v165 + v161;
                unsigned long long v167;
                v167 = v166 + 16ull;
                unsigned long long v168;
                v168 = v167 - 1ull;
                unsigned long long v169;
                v169 = v168 % 16ull;
                unsigned long long v170;
                v170 = v168 - v169;
                unsigned long long v171;
                v171 = v170 + v161;
                unsigned long long v172;
                v172 = v171 + 16ull;
                unsigned long long v173;
                v173 = v172 - 1ull;
                unsigned long long v174;
                v174 = v173 % 16ull;
                unsigned long long v175;
                v175 = v173 - v174;
                unsigned long long v176;
                v176 = v175 + v161;
                bool v177;
                v177 = v176 <= 98304ull;
                bool v178;
                v178 = v177 == false;
                if (v178){
                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v177);
                } else {
                }
                extern __shared__ unsigned char v180[];
                bool v181;
                v181 = v176 <= v176;
                bool v182;
                v182 = v181 == false;
                if (v182){
                    assert("The length of the partition has to be less than or equal to the length of the base array." && v181);
                } else {
                }
                double * * v184;
                v184 = reinterpret_cast<double * *>(&v180[0ull]);
                double * * v186;
                v186 = reinterpret_cast<double * *>(&v180[v165]);
                double * * v188;
                v188 = reinterpret_cast<double * *>(&v180[v170]);
                double * * v190;
                v190 = reinterpret_cast<double * *>(&v180[v175]);
                int v192;
                v192 = threadIdx.x;
                assert("Tensor range check" && 0 <= v192 && v192 < 256);
                v184[v192] = v151;
                v186[v192] = v153;
                v188[v192] = v155;
                v190[v192] = v157;
                __syncthreads();
                bool v193;
                v193 = 0 <= v192;
                bool v194;
                v194 = v193 == false;
                if (v194){
                    assert("The index needs to be zero or positive." && v193);
                } else {
                }
                int v196;
                v196 = v192 % 1;
                bool v197;
                v197 = v192 < 256;
                bool v198;
                v198 = v197 == false;
                if (v198){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v197);
                } else {
                }
                assert("Tensor range check" && 0 <= v192 && v192 < 256);
                int v200;
                v200 = 0;
                while (while_method_5(v200)){
                    bool v202;
                    v202 = v193 && v197;
                    bool v203;
                    v203 = v202 == false;
                    if (v203){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v202);
                    } else {
                    }
                    bool v205;
                    v205 = 0 <= v200;
                    bool v207;
                    if (v205){
                        bool v206;
                        v206 = v200 < 1;
                        v207 = v206;
                    } else {
                        v207 = false;
                    }
                    bool v208;
                    v208 = v207 == false;
                    if (v208){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v207);
                    } else {
                    }
                    int v210;
                    v210 = v200 * 256;
                    int v211;
                    v211 = v210 + v192;
                    assert("Tensor range check" && 0 <= v200 && v200 < 1);
                    int v212;
                    v212 = 256 * v200;
                    int v213;
                    v213 = v212 + v192;
                    double * v214;
                    v214 = v184[v213];
                    double * v215;
                    v215 = v186[v213];
                    double * v216;
                    v216 = v188[v213];
                    double * v217;
                    v217 = v190[v213];
                    int v218;
                    v218 = blockIdx.x;
                    int v219;
                    v219 = v218 * 256;
                    int v220;
                    v220 = v219 + v211;
                    assert("Tensor range check" && 0 <= v196 && v196 < 1);
                    int v221;
                    v221 = 2 * v196;
                    double v222[2];
                    double v223[2];
                    int v224[2];
                    int v225;
                    v225 = 0;
                    while (while_method_5(v225)){
                        assert("Tensor range check" && 0 <= v225 && v225 < 1);
                        int v227;
                        v227 = 2 * v225;
                        assert("Tensor range check" && 0 <= v225 && v225 < 1);
                        int v228;
                        v228 = v227 + v221;
                        int4* v229;
                        v229 = reinterpret_cast<int4*>(v214 + v228);
                        int4* v230;
                        v230 = reinterpret_cast<int4*>(v222 + v227);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v229) % 16 == 0 && reinterpret_cast<unsigned long long>(v230) % 16 == 0);
                        *v230 = *v229;
                        int4* v231;
                        v231 = reinterpret_cast<int4*>(v215 + v228);
                        int4* v232;
                        v232 = reinterpret_cast<int4*>(v223 + v227);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v231) % 16 == 0 && reinterpret_cast<unsigned long long>(v232) % 16 == 0);
                        *v232 = *v231;
                        v225 += 1 ;
                    }
                    int v233;
                    v233 = 0;
                    while (while_method_5(v233)){
                        int v235;
                        v235 = 0;
                        while (while_method_0(v235)){
                            bool v237;
                            v237 = 0 <= v235;
                            bool v239;
                            if (v237){
                                bool v238;
                                v238 = v235 < 2;
                                v239 = v238;
                            } else {
                                v239 = false;
                            }
                            bool v240;
                            v240 = v239 == false;
                            if (v240){
                                assert("The indices should be inside the range of the dimension." && v239);
                            } else {
                            }
                            bool v242;
                            v242 = 0 <= v196;
                            bool v244;
                            if (v242){
                                bool v243;
                                v243 = v196 < 1;
                                v244 = v243;
                            } else {
                                v244 = false;
                            }
                            bool v245;
                            v245 = v244 == false;
                            if (v245){
                                assert("The indices should be inside the range of the dimension." && v244);
                            } else {
                            }
                            int v247;
                            v247 = v196 * 2;
                            int v248;
                            v248 = v235 + v247;
                            bool v249;
                            v249 = 0 <= v233;
                            bool v251;
                            if (v249){
                                bool v250;
                                v250 = v233 < 1;
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
                            int v254;
                            v254 = v233 * 2;
                            int v255;
                            v255 = v248 + v254;
                            assert("Tensor range check" && 0 <= v233 && v233 < 1);
                            assert("Tensor range check" && 0 <= v235 && v235 < 2);
                            int v256;
                            v256 = 2 * v233;
                            int v257;
                            v257 = v256 + v235;
                            v224[v257] = v255;
                            v235 += 1 ;
                        }
                        v233 += 1 ;
                    }
                    double v258[2];
                    double v259[2];
                    int v260;
                    v260 = 0;
                    while (while_method_5(v260)){
                        int v262;
                        v262 = 0;
                        while (while_method_0(v262)){
                            assert("Tensor range check" && 0 <= v260 && v260 < 1);
                            assert("Tensor range check" && 0 <= v262 && v262 < 2);
                            int v264;
                            v264 = 2 * v260;
                            int v265;
                            v265 = v264 + v262;
                            double v266;
                            v266 = v222[v265];
                            double v267;
                            v267 = v223[v265];
                            assert("Tensor range check" && 0 <= v260 && v260 < 1);
                            assert("Tensor range check" && 0 <= v262 && v262 < 2);
                            v258[v265] = 0.0;
                            v259[v265] = 0.0;
                            v262 += 1 ;
                        }
                        v260 += 1 ;
                    }
                    int v268;
                    v268 = 0;
                    while (while_method_5(v268)){
                        assert("Tensor range check" && 0 <= v268 && v268 < 1);
                        int v270;
                        v270 = 2 * v268;
                        int v271;
                        v271 = v270 + v221;
                        assert("Tensor range check" && 0 <= v268 && v268 < 1);
                        int4* v272;
                        v272 = reinterpret_cast<int4*>(v258 + v270);
                        int4* v273;
                        v273 = reinterpret_cast<int4*>(v216 + v271);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v272) % 16 == 0 && reinterpret_cast<unsigned long long>(v273) % 16 == 0);
                        *v273 = *v272;
                        int4* v274;
                        v274 = reinterpret_cast<int4*>(v259 + v270);
                        int4* v275;
                        v275 = reinterpret_cast<int4*>(v217 + v271);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v274) % 16 == 0 && reinterpret_cast<unsigned long long>(v275) % 16 == 0);
                        *v275 = *v274;
                        v268 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v211 && v211 < 256);
                    v200 += 1 ;
                }
                __syncthreads();
                assert("Tensor range check" && 0 <= v192 && v192 < 256);
                __syncthreads();
                v146 += 1 ;
            }
            Union4 v276;
            v276 = Union4{Union4_1{}};
            method_50(v0, v1, v2, v26, v276);
            double * v277;
            v277 = reinterpret_cast<double *>(&v1[58195968ull]);
            double * v279;
            v279 = reinterpret_cast<double *>(&v1[61341696ull]);
            int v281;
            v281 = threadIdx.x;
            int v282;
            v282 = blockIdx.x;
            int v283;
            v283 = v282 * 256;
            int v284;
            v284 = v281 + v283;
            assert("Tensor range check" && 0 <= v284 && v284 < 6144);
            int v285;
            v285 = 2 * v284;
            static_array<float,2> & v286 = v26.v4;
            float v287[2];
            int v288;
            v288 = 0;
            while (while_method_0(v288)){
                bool v290;
                v290 = 0 <= v288;
                bool v292;
                if (v290){
                    bool v291;
                    v291 = v288 < 2;
                    v292 = v291;
                } else {
                    v292 = false;
                }
                bool v293;
                v293 = v292 == false;
                if (v293){
                    assert("Index must be in range." && v292);
                } else {
                }
                float v295;
                v295 = v286[v288];
                assert("Tensor range check" && 0 <= v288 && v288 < 2);
                v287[v288] = v295;
                v288 += 1 ;
            }
            int * v297;
            v297 = reinterpret_cast<int *>(&v0[1572864ull]);
            bool * v299;
            v299 = reinterpret_cast<bool *>(&v0[1572880ull]);
            float * v301;
            v301 = reinterpret_cast<float *>(&v0[1572912ull]);
            float * v303;
            v303 = reinterpret_cast<float *>(&v0[1573040ull]);
            double v305[2];
            int v306;
            v306 = 0;
            while (while_method_0(v306)){
                int v308; double v309;
                Tuple11 tmp90 = Tuple11{0, 0.0};
                v308 = tmp90.v0; v309 = tmp90.v1;
                while (while_method_4(v308)){
                    assert("Tensor range check" && 0 <= v308 && v308 < 32);
                    bool v311;
                    v311 = v299[v308];
                    bool v312;
                    v312 = v311 == false;
                    double v320;
                    if (v312){
                        v320 = 0.0;
                    } else {
                        assert("Tensor range check" && 0 <= v308 && v308 < 32);
                        assert("Tensor range check" && 0 <= v306 && v306 < 2);
                        int v313;
                        v313 = v306 + v285;
                        int v314;
                        v314 = 12288 * v308;
                        int v315;
                        v315 = v314 + v313;
                        double v316;
                        v316 = v277[v315];
                        double v317;
                        v317 = v279[v315];
                        double v318;
                        v318 = v316 - v317;
                        double v319;
                        v319 = exp(v318);
                        v320 = v319;
                    }
                    double v321;
                    v321 = v309 + v320;
                    v309 = v321;
                    v308 += 1 ;
                }
                assert("Tensor range check" && 0 <= v306 && v306 < 2);
                v305[v306] = v309;
                v306 += 1 ;
            }
            double v322;
            v322 = 1.0;
            int v323;
            v323 = 0;
            while (while_method_0(v323)){
                assert("Tensor range check" && 0 <= v323 && v323 < 2);
                double v325;
                v325 = v305[v323];
                double v326;
                v326 = v322 * v325;
                v322 = v326;
                v323 += 1 ;
            }
            double v327[64];
            int v328;
            v328 = 0;
            while (while_method_4(v328)){
                int v330;
                v330 = 0;
                while (while_method_0(v330)){
                    bool v332;
                    v332 = v322 == 0.0;
                    bool v333;
                    v333 = v332 != true;
                    double v344;
                    if (v333){
                        assert("Tensor range check" && 0 <= v330 && v330 < 2);
                        double v334;
                        v334 = v305[v330];
                        double v335;
                        v335 = v322 / v334;
                        assert("Tensor range check" && 0 <= v328 && v328 < 32);
                        assert("Tensor range check" && 0 <= v330 && v330 < 2);
                        int v336;
                        v336 = v330 + v285;
                        int v337;
                        v337 = 12288 * v328;
                        int v338;
                        v338 = v337 + v336;
                        double v339;
                        v339 = v277[v338];
                        double v340;
                        v340 = v279[v338];
                        double v341;
                        v341 = v339 - v340;
                        double v342;
                        v342 = exp(v341);
                        double v343;
                        v343 = v335 * v342;
                        v344 = v343;
                    } else {
                        v344 = 0.0;
                    }
                    bool v345;
                    v345 = isnan(v344);
                    bool v346;
                    v346 = v345 == false;
                    bool v347;
                    v347 = v346 == false;
                    if (v347){
                        assert("The path probability after integration should not be a nan in integrate_rewards_." && v346);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v328 && v328 < 32);
                    assert("Tensor range check" && 0 <= v330 && v330 < 2);
                    int v349;
                    v349 = 2 * v328;
                    int v350;
                    v350 = v349 + v330;
                    v327[v350] = v344;
                    v330 += 1 ;
                }
                v328 += 1 ;
            }
            float v351[32];
            float v352[32];
            int v353;
            v353 = 0;
            while (while_method_4(v353)){
                int v355; float v356; double v357;
                Tuple12 tmp91 = Tuple12{0, 0.0f, 0.0};
                v355 = tmp91.v0; v356 = tmp91.v1; v357 = tmp91.v2;
                while (while_method_0(v355)){
                    assert("Tensor range check" && 0 <= v353 && v353 < 32);
                    assert("Tensor range check" && 0 <= v355 && v355 < 2);
                    int v359;
                    v359 = 2 * v353;
                    int v360;
                    v360 = v359 + v355;
                    double v361;
                    v361 = v327[v360];
                    assert("Tensor range check" && 0 <= v355 && v355 < 2);
                    float v362;
                    v362 = v287[v355];
                    float v363;
                    v363 = (float)v361;
                    float v364;
                    v364 = v363 * v362;
                    float v365;
                    v365 = v356 + v364;
                    double v366;
                    v366 = v357 + v361;
                    v356 = v365;
                    v357 = v366;
                    v355 += 1 ;
                }
                float v367;
                v367 = (float)v357;
                assert("Tensor range check" && 0 <= v353 && v353 < 32);
                v351[v353] = v356;
                v352[v353] = v367;
                v353 += 1 ;
            }
            int v368;
            v368 = 0;
            while (while_method_4(v368)){
                assert("Tensor range check" && 0 <= v368 && v368 < 32);
                float v370;
                v370 = v351[v368];
                float v371;
                v371 = v352[v368];
                assert("Tensor range check" && 0 <= v368 && v368 < 32);
                assert("Tensor range check" && 0 <= v27 && v27 < 256);
                int v372;
                v372 = 256 * v368;
                int v373;
                v373 = v372 + v27;
                float * v374;
                v374 = v3+v373;
                float * v376;
                v376 = v4+v373;
                float v378;
                v378 = atomicAdd(v374,v370);
                float v379;
                v379 = atomicAdd(v376,v371);
                v368 += 1 ;
            }
            double * v380;
            v380 = reinterpret_cast<double *>(&v1[58195968ull]);
            double * v382;
            v382 = reinterpret_cast<double *>(&v1[61341696ull]);
            int v384;
            v384 = threadIdx.x;
            int v385;
            v385 = blockIdx.x;
            int v386;
            v386 = v385 * 256;
            int v387;
            v387 = v384 + v386;
            int v388;
            v388 = 0;
            while (while_method_4(v388)){
                assert("Tensor range check" && 0 <= v388 && v388 < 32);
                int v390;
                v390 = 12288 * v388;
                assert("Tensor range check" && 0 <= v387 && v387 < 6144);
                int v391;
                v391 = 2 * v387;
                int v392;
                v392 = v391 + v390;
                double * v393;
                v393 = v380+v392;
                double * v395;
                v395 = v382+v392;
                double * v397;
                v397 = v380+v392;
                double * v399;
                v399 = v382+v392;
                int v401;
                v401 = sizeof(double *);
                unsigned long long v402;
                v402 = (unsigned long long)v401;
                unsigned long long v403;
                v403 = 256ull * v402;
                unsigned long long v404;
                v404 = v403 + 16ull;
                unsigned long long v405;
                v405 = v404 - 1ull;
                unsigned long long v406;
                v406 = v405 % 16ull;
                unsigned long long v407;
                v407 = v405 - v406;
                unsigned long long v408;
                v408 = v407 + v403;
                unsigned long long v409;
                v409 = v408 + 16ull;
                unsigned long long v410;
                v410 = v409 - 1ull;
                unsigned long long v411;
                v411 = v410 % 16ull;
                unsigned long long v412;
                v412 = v410 - v411;
                unsigned long long v413;
                v413 = v412 + v403;
                unsigned long long v414;
                v414 = v413 + 16ull;
                unsigned long long v415;
                v415 = v414 - 1ull;
                unsigned long long v416;
                v416 = v415 % 16ull;
                unsigned long long v417;
                v417 = v415 - v416;
                unsigned long long v418;
                v418 = v417 + v403;
                bool v419;
                v419 = v418 <= 98304ull;
                bool v420;
                v420 = v419 == false;
                if (v420){
                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v419);
                } else {
                }
                extern __shared__ unsigned char v422[];
                bool v423;
                v423 = v418 <= v418;
                bool v424;
                v424 = v423 == false;
                if (v424){
                    assert("The length of the partition has to be less than or equal to the length of the base array." && v423);
                } else {
                }
                double * * v426;
                v426 = reinterpret_cast<double * *>(&v422[0ull]);
                double * * v428;
                v428 = reinterpret_cast<double * *>(&v422[v407]);
                double * * v430;
                v430 = reinterpret_cast<double * *>(&v422[v412]);
                double * * v432;
                v432 = reinterpret_cast<double * *>(&v422[v417]);
                int v434;
                v434 = threadIdx.x;
                assert("Tensor range check" && 0 <= v434 && v434 < 256);
                v426[v434] = v393;
                v428[v434] = v395;
                v430[v434] = v397;
                v432[v434] = v399;
                __syncthreads();
                bool v435;
                v435 = 0 <= v434;
                bool v436;
                v436 = v435 == false;
                if (v436){
                    assert("The index needs to be zero or positive." && v435);
                } else {
                }
                int v438;
                v438 = v434 % 1;
                bool v439;
                v439 = v434 < 256;
                bool v440;
                v440 = v439 == false;
                if (v440){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v439);
                } else {
                }
                assert("Tensor range check" && 0 <= v434 && v434 < 256);
                int v442;
                v442 = 0;
                while (while_method_5(v442)){
                    bool v444;
                    v444 = v435 && v439;
                    bool v445;
                    v445 = v444 == false;
                    if (v445){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v444);
                    } else {
                    }
                    bool v447;
                    v447 = 0 <= v442;
                    bool v449;
                    if (v447){
                        bool v448;
                        v448 = v442 < 1;
                        v449 = v448;
                    } else {
                        v449 = false;
                    }
                    bool v450;
                    v450 = v449 == false;
                    if (v450){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v449);
                    } else {
                    }
                    int v452;
                    v452 = v442 * 256;
                    int v453;
                    v453 = v452 + v434;
                    assert("Tensor range check" && 0 <= v442 && v442 < 1);
                    int v454;
                    v454 = 256 * v442;
                    int v455;
                    v455 = v454 + v434;
                    double * v456;
                    v456 = v426[v455];
                    double * v457;
                    v457 = v428[v455];
                    double * v458;
                    v458 = v430[v455];
                    double * v459;
                    v459 = v432[v455];
                    int v460;
                    v460 = blockIdx.x;
                    int v461;
                    v461 = v460 * 256;
                    int v462;
                    v462 = v461 + v453;
                    assert("Tensor range check" && 0 <= v438 && v438 < 1);
                    int v463;
                    v463 = 2 * v438;
                    double v464[2];
                    double v465[2];
                    int v466[2];
                    int v467;
                    v467 = 0;
                    while (while_method_5(v467)){
                        assert("Tensor range check" && 0 <= v467 && v467 < 1);
                        int v469;
                        v469 = 2 * v467;
                        assert("Tensor range check" && 0 <= v467 && v467 < 1);
                        int v470;
                        v470 = v469 + v463;
                        int4* v471;
                        v471 = reinterpret_cast<int4*>(v456 + v470);
                        int4* v472;
                        v472 = reinterpret_cast<int4*>(v464 + v469);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v471) % 16 == 0 && reinterpret_cast<unsigned long long>(v472) % 16 == 0);
                        *v472 = *v471;
                        int4* v473;
                        v473 = reinterpret_cast<int4*>(v457 + v470);
                        int4* v474;
                        v474 = reinterpret_cast<int4*>(v465 + v469);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v473) % 16 == 0 && reinterpret_cast<unsigned long long>(v474) % 16 == 0);
                        *v474 = *v473;
                        v467 += 1 ;
                    }
                    int v475;
                    v475 = 0;
                    while (while_method_5(v475)){
                        int v477;
                        v477 = 0;
                        while (while_method_0(v477)){
                            bool v479;
                            v479 = 0 <= v477;
                            bool v481;
                            if (v479){
                                bool v480;
                                v480 = v477 < 2;
                                v481 = v480;
                            } else {
                                v481 = false;
                            }
                            bool v482;
                            v482 = v481 == false;
                            if (v482){
                                assert("The indices should be inside the range of the dimension." && v481);
                            } else {
                            }
                            bool v484;
                            v484 = 0 <= v438;
                            bool v486;
                            if (v484){
                                bool v485;
                                v485 = v438 < 1;
                                v486 = v485;
                            } else {
                                v486 = false;
                            }
                            bool v487;
                            v487 = v486 == false;
                            if (v487){
                                assert("The indices should be inside the range of the dimension." && v486);
                            } else {
                            }
                            int v489;
                            v489 = v438 * 2;
                            int v490;
                            v490 = v477 + v489;
                            bool v491;
                            v491 = 0 <= v475;
                            bool v493;
                            if (v491){
                                bool v492;
                                v492 = v475 < 1;
                                v493 = v492;
                            } else {
                                v493 = false;
                            }
                            bool v494;
                            v494 = v493 == false;
                            if (v494){
                                assert("The indices should be inside the range of the dimension." && v493);
                            } else {
                            }
                            int v496;
                            v496 = v475 * 2;
                            int v497;
                            v497 = v490 + v496;
                            assert("Tensor range check" && 0 <= v475 && v475 < 1);
                            assert("Tensor range check" && 0 <= v477 && v477 < 2);
                            int v498;
                            v498 = 2 * v475;
                            int v499;
                            v499 = v498 + v477;
                            v466[v499] = v497;
                            v477 += 1 ;
                        }
                        v475 += 1 ;
                    }
                    double v500[2];
                    double v501[2];
                    int v502;
                    v502 = 0;
                    while (while_method_5(v502)){
                        int v504;
                        v504 = 0;
                        while (while_method_0(v504)){
                            assert("Tensor range check" && 0 <= v502 && v502 < 1);
                            assert("Tensor range check" && 0 <= v504 && v504 < 2);
                            int v506;
                            v506 = 2 * v502;
                            int v507;
                            v507 = v506 + v504;
                            double v508;
                            v508 = v464[v507];
                            double v509;
                            v509 = v465[v507];
                            assert("Tensor range check" && 0 <= v502 && v502 < 1);
                            assert("Tensor range check" && 0 <= v504 && v504 < 2);
                            v500[v507] = 0.0;
                            v501[v507] = 0.0;
                            v504 += 1 ;
                        }
                        v502 += 1 ;
                    }
                    int v510;
                    v510 = 0;
                    while (while_method_5(v510)){
                        assert("Tensor range check" && 0 <= v510 && v510 < 1);
                        int v512;
                        v512 = 2 * v510;
                        int v513;
                        v513 = v512 + v463;
                        assert("Tensor range check" && 0 <= v510 && v510 < 1);
                        int4* v514;
                        v514 = reinterpret_cast<int4*>(v500 + v512);
                        int4* v515;
                        v515 = reinterpret_cast<int4*>(v458 + v513);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v514) % 16 == 0 && reinterpret_cast<unsigned long long>(v515) % 16 == 0);
                        *v515 = *v514;
                        int4* v516;
                        v516 = reinterpret_cast<int4*>(v501 + v512);
                        int4* v517;
                        v517 = reinterpret_cast<int4*>(v459 + v513);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v516) % 16 == 0 && reinterpret_cast<unsigned long long>(v517) % 16 == 0);
                        *v517 = *v516;
                        v510 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v453 && v453 < 256);
                    v442 += 1 ;
                }
                __syncthreads();
                assert("Tensor range check" && 0 <= v434 && v434 < 256);
                __syncthreads();
                v388 += 1 ;
            }
            v29 += 1 ;
        }
        cooperative_groups::grid_group & v518 = v26.v1;
        cooperative_groups::grid_group & v519 = v518;
        curandStatePhilox4_32_10_t & v520 = v26.v5;
        curandStatePhilox4_32_10_t & v521 = v520;
        float * v522;
        v522 = reinterpret_cast<float *>(&v0[0ull]);
        float * v524;
        v524 = reinterpret_cast<float *>(&v2[0ull]);
        float * v526;
        v526 = reinterpret_cast<float *>(&v0[1048576ull]);
        float * v528;
        v528 = reinterpret_cast<float *>(&v2[1048576ull]);
        float * v530;
        v530 = reinterpret_cast<float *>(&v1[7864320ull]);
        int * v532;
        v532 = reinterpret_cast<int *>(&v0[1572864ull]);
        bool * v534;
        v534 = reinterpret_cast<bool *>(&v0[1572880ull]);
        float * v536;
        v536 = reinterpret_cast<float *>(&v0[1572912ull]);
        float * v538;
        v538 = reinterpret_cast<float *>(&v0[1573040ull]);
        double * v540;
        v540 = reinterpret_cast<double *>(&v1[58195968ull]);
        double * v542;
        v542 = reinterpret_cast<double *>(&v1[61341696ull]);
        v519.sync() ;
        int v544;
        v544 = threadIdx.x;
        int v545;
        v545 = blockIdx.x;
        int v546;
        v546 = v545 * 256;
        int v547;
        v547 = v544 + v546;
        bool v548;
        v548 = v547 == 0;
        if (v548){
            int v549;
            v549 = 0;
            int v550;
            v550 = 32;
            int v551;
            v551 = int_range_22(v550, v549, v521);
            v532[0] = v551;
        } else {
        }
        __syncwarp();
        float v552[32];
        int v553;
        v553 = 0;
        while (while_method_4(v553)){
            assert("Tensor range check" && 0 <= v553 && v553 < 32);
            float v555;
            v555 = v536[v553];
            float v556;
            v556 = v538[v553];
            bool v557;
            v557 = v556 == 0.0f;
            bool v558;
            v558 = v557 != true;
            float v560;
            if (v558){
                float v559;
                v559 = v555 / v556;
                v560 = v559;
            } else {
                v560 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v553 && v553 < 32);
            v552[v553] = v560;
            v553 += 1 ;
        }
        float v561;
        v561 = 0.0f;
        int v562;
        v562 = 0;
        while (while_method_4(v562)){
            assert("Tensor range check" && 0 <= v562 && v562 < 32);
            float v564;
            v564 = v552[v562];
            float v565;
            v565 = v561 + v564;
            v561 = v565;
            v562 += 1 ;
        }
        float v566;
        v566 = v561 / 32.0f;
        int v567;
        v567 = 0;
        while (while_method_4(v567)){
            assert("Tensor range check" && 0 <= v567 && v567 < 32);
            v536[v567] = 0.0f;
            v538[v567] = 0.0f;
            v567 += 1 ;
        }
        bool v569[32];
        int v570;
        v570 = 0;
        while (while_method_4(v570)){
            assert("Tensor range check" && 0 <= v570 && v570 < 32);
            float v572;
            v572 = v552[v570];
            bool v573;
            v573 = v572 >= v566;
            assert("Tensor range check" && 0 <= v570 && v570 < 32);
            v569[v570] = v573;
            v570 += 1 ;
        }
        int v574;
        v574 = 0;
        while (while_method_4(v574)){
            assert("Tensor range check" && 0 <= v574 && v574 < 32);
            bool v576;
            v576 = v569[v574];
            assert("Tensor range check" && 0 <= v574 && v574 < 32);
            v534[v574] = v576;
            v574 += 1 ;
        }
        extern __shared__ unsigned char v577[];
        float * v578;
        v578 = reinterpret_cast<float *>(&v577[0ull]);
        int v580;
        v580 = blockIdx.x;
        int v581;
        v581 = v580;
        while (while_method_9(v581)){
            bool v583;
            v583 = 0 <= v581;
            bool v584;
            v584 = v583 == false;
            if (v584){
                assert("The index needs to be zero or positive." && v583);
            } else {
            }
            int v586;
            v586 = v581 % 16;
            int v587;
            v587 = v581 / 16;
            bool v588;
            v588 = v587 < 1;
            bool v589;
            v589 = v588 == false;
            if (v589){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v588);
            } else {
            }
            assert("Tensor range check" && 0 <= v587 && v587 < 1);
            assert("Tensor range check" && 0 <= v586 && v586 < 16);
            int v591;
            v591 = 512 * v586;
            int v592;
            v592 = 262144 * v587;
            int v593;
            v593 = v592 + v591;
            int v594;
            v594 = 16384 * v586;
            int v595;
            v595 = 32 * v587;
            int v596;
            v596 = v595 + v594;
            int v597;
            v597 = threadIdx.x;
            int v598;
            v598 = v597;
            while (while_method_14(v598)){
                bool v600;
                v600 = 0 <= v598;
                bool v601;
                v601 = v600 == false;
                if (v601){
                    assert("The index needs to be zero or positive." && v600);
                } else {
                }
                int v603;
                v603 = v598 % 512;
                int v604;
                v604 = v598 / 512;
                bool v605;
                v605 = v604 < 32;
                bool v606;
                v606 = v605 == false;
                if (v606){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v605);
                } else {
                }
                assert("Tensor range check" && 0 <= v604 && v604 < 32);
                assert("Tensor range check" && 0 <= v603 && v603 < 512);
                int v608;
                v608 = v603 + v593;
                int v609;
                v609 = 8192 * v604;
                int v610;
                v610 = v609 + v608;
                float v611;
                v611 = v522[v610];
                assert("Tensor range check" && 0 <= v604 && v604 < 32);
                assert("Tensor range check" && 0 <= v603 && v603 < 512);
                int v612;
                v612 = 513 * v604;
                int v613;
                v613 = v612 + v603;
                v578[v613] = v611;
                v598 += 256 ;
            }
            __syncthreads();
            int v614;
            v614 = threadIdx.x;
            int v615;
            v615 = v614;
            while (while_method_14(v615)){
                bool v617;
                v617 = 0 <= v615;
                bool v618;
                v618 = v617 == false;
                if (v618){
                    assert("The index needs to be zero or positive." && v617);
                } else {
                }
                int v620;
                v620 = v615 % 32;
                int v621;
                v621 = v615 / 32;
                bool v622;
                v622 = v621 < 512;
                bool v623;
                v623 = v622 == false;
                if (v623){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v622);
                } else {
                }
                assert("Tensor range check" && 0 <= v621 && v621 < 512);
                assert("Tensor range check" && 0 <= v620 && v620 < 32);
                int v625;
                v625 = 513 * v620;
                int v626;
                v626 = v621 + v625;
                float v627;
                v627 = v578[v626];
                assert("Tensor range check" && 0 <= v621 && v621 < 512);
                assert("Tensor range check" && 0 <= v620 && v620 < 32);
                int v628;
                v628 = v620 + v596;
                int v629;
                v629 = 32 * v621;
                int v630;
                v630 = v629 + v628;
                v524[v630] = v627;
                v615 += 256 ;
            }
            __syncthreads();
            v581 += 24 ;
        }
        extern __shared__ unsigned char v631[];
        float * v632;
        v632 = reinterpret_cast<float *>(&v631[0ull]);
        int v634;
        v634 = blockIdx.x;
        int v635;
        v635 = v634;
        while (while_method_6(v635)){
            bool v637;
            v637 = 0 <= v635;
            bool v638;
            v638 = v637 == false;
            if (v638){
                assert("The index needs to be zero or positive." && v637);
            } else {
            }
            int v640;
            v640 = v635 % 8;
            int v641;
            v641 = v635 / 8;
            bool v642;
            v642 = v641 < 1;
            bool v643;
            v643 = v642 == false;
            if (v643){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v642);
            } else {
            }
            assert("Tensor range check" && 0 <= v641 && v641 < 1);
            assert("Tensor range check" && 0 <= v640 && v640 < 8);
            int v645;
            v645 = 512 * v640;
            int v646;
            v646 = 131072 * v641;
            int v647;
            v647 = v646 + v645;
            int v648;
            v648 = 16384 * v640;
            int v649;
            v649 = 32 * v641;
            int v650;
            v650 = v649 + v648;
            int v651;
            v651 = threadIdx.x;
            int v652;
            v652 = v651;
            while (while_method_14(v652)){
                bool v654;
                v654 = 0 <= v652;
                bool v655;
                v655 = v654 == false;
                if (v655){
                    assert("The index needs to be zero or positive." && v654);
                } else {
                }
                int v657;
                v657 = v652 % 512;
                int v658;
                v658 = v652 / 512;
                bool v659;
                v659 = v658 < 32;
                bool v660;
                v660 = v659 == false;
                if (v660){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v659);
                } else {
                }
                assert("Tensor range check" && 0 <= v658 && v658 < 32);
                assert("Tensor range check" && 0 <= v657 && v657 < 512);
                int v662;
                v662 = v657 + v647;
                int v663;
                v663 = 4096 * v658;
                int v664;
                v664 = v663 + v662;
                float v665;
                v665 = v526[v664];
                assert("Tensor range check" && 0 <= v658 && v658 < 32);
                assert("Tensor range check" && 0 <= v657 && v657 < 512);
                int v666;
                v666 = 513 * v658;
                int v667;
                v667 = v666 + v657;
                v632[v667] = v665;
                v652 += 256 ;
            }
            __syncthreads();
            int v668;
            v668 = threadIdx.x;
            int v669;
            v669 = v668;
            while (while_method_14(v669)){
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
                v679 = 513 * v674;
                int v680;
                v680 = v675 + v679;
                float v681;
                v681 = v632[v680];
                assert("Tensor range check" && 0 <= v675 && v675 < 512);
                assert("Tensor range check" && 0 <= v674 && v674 < 32);
                int v682;
                v682 = v674 + v650;
                int v683;
                v683 = 32 * v675;
                int v684;
                v684 = v683 + v682;
                v528[v684] = v681;
                v669 += 256 ;
            }
            __syncthreads();
            v635 += 24 ;
        }
        v519.sync() ;
        int v685;
        v685 = threadIdx.x;
        bool v686;
        v686 = 0 <= v685;
        bool v687;
        v687 = v686 == false;
        if (v687){
            assert("The index needs to be zero or positive." && v686);
        } else {
        }
        int v689;
        v689 = v685 % 8;
        int v690;
        v690 = v685 / 8;
        int v691;
        v691 = v690 % 32;
        int v692;
        v692 = v690 / 32;
        bool v693;
        v693 = v692 < 1;
        bool v694;
        v694 = v693 == false;
        if (v694){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v693);
        } else {
        }
        assert("Tensor range check" && 0 <= v692 && v692 < 1);
        assert("Tensor range check" && 0 <= v691 && v691 < 32);
        assert("Tensor range check" && 0 <= v689 && v689 < 8);
        int v696;
        v696 = 4 * v689;
        int v697;
        v697 = 32 * v691;
        int v698;
        v698 = v697 + v696;
        int v699;
        v699 = 4096 * v692;
        int v700;
        v700 = v699 + v698;
        assert("Tensor range check" && 0 <= v692 && v692 < 1);
        assert("Tensor range check" && 0 <= v691 && v691 < 32);
        assert("Tensor range check" && 0 <= v689 && v689 < 8);
        int v701;
        v701 = blockIdx.x;
        int v702;
        v702 = v701;
        while (while_method_12(v702)){
            bool v704;
            v704 = 0 <= v702;
            bool v705;
            v705 = v704 == false;
            if (v705){
                assert("The index needs to be zero or positive." && v704);
            } else {
            }
            int v707;
            v707 = v702 % 4;
            int v708;
            v708 = v702 / 4;
            bool v709;
            v709 = v708 < 64;
            bool v710;
            v710 = v709 == false;
            if (v710){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v709);
            } else {
            }
            assert("Tensor range check" && 0 <= v708 && v708 < 64);
            assert("Tensor range check" && 0 <= v707 && v707 < 4);
            int v712;
            v712 = 1024 * v707;
            int v713;
            v713 = v712 + v700;
            int v714;
            v714 = 4096 * v708;
            int v715;
            v715 = v714 + v713;
            float v716[4];
            int v717[4];
            int v718;
            v718 = 0;
            while (while_method_5(v718)){
                assert("Tensor range check" && 0 <= v718 && v718 < 1);
                int v720;
                v720 = 4 * v718;
                assert("Tensor range check" && 0 <= v718 && v718 < 1);
                int v721;
                v721 = 32 * v718;
                int v722;
                v722 = v721 + v715;
                int4* v723;
                v723 = reinterpret_cast<int4*>(v524 + v722);
                int4* v724;
                v724 = reinterpret_cast<int4*>(v716 + v720);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v723) % 16 == 0 && reinterpret_cast<unsigned long long>(v724) % 16 == 0);
                *v724 = *v723;
                v718 += 1 ;
            }
            int v725;
            v725 = 0;
            while (while_method_5(v725)){
                int v727;
                v727 = 0;
                while (while_method_8(v727)){
                    bool v729;
                    v729 = 0 <= v727;
                    bool v731;
                    if (v729){
                        bool v730;
                        v730 = v727 < 4;
                        v731 = v730;
                    } else {
                        v731 = false;
                    }
                    bool v732;
                    v732 = v731 == false;
                    if (v732){
                        assert("The indices should be inside the range of the dimension." && v731);
                    } else {
                    }
                    bool v734;
                    v734 = 0 <= v689;
                    bool v736;
                    if (v734){
                        bool v735;
                        v735 = v689 < 8;
                        v736 = v735;
                    } else {
                        v736 = false;
                    }
                    bool v737;
                    v737 = v736 == false;
                    if (v737){
                        assert("The indices should be inside the range of the dimension." && v736);
                    } else {
                    }
                    int v739;
                    v739 = v689 * 4;
                    int v740;
                    v740 = v727 + v739;
                    bool v741;
                    v741 = 0 <= v725;
                    bool v743;
                    if (v741){
                        bool v742;
                        v742 = v725 < 1;
                        v743 = v742;
                    } else {
                        v743 = false;
                    }
                    bool v744;
                    v744 = v743 == false;
                    if (v744){
                        assert("The indices should be inside the range of the dimension." && v743);
                    } else {
                    }
                    int v746;
                    v746 = v725 * 32;
                    int v747;
                    v747 = v740 + v746;
                    assert("Tensor range check" && 0 <= v725 && v725 < 1);
                    assert("Tensor range check" && 0 <= v727 && v727 < 4);
                    int v748;
                    v748 = 4 * v725;
                    int v749;
                    v749 = v748 + v727;
                    v717[v749] = v747;
                    v727 += 1 ;
                }
                v725 += 1 ;
            }
            bool v750;
            v750 = 0 <= v692;
            bool v751;
            v751 = v750 && v693;
            bool v752;
            v752 = v751 == false;
            if (v752){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v751);
            } else {
            }
            bool v754;
            v754 = 0 <= v691;
            bool v756;
            if (v754){
                bool v755;
                v755 = v691 < 32;
                v756 = v755;
            } else {
                v756 = false;
            }
            bool v757;
            v757 = v756 == false;
            if (v757){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v756);
            } else {
            }
            bool v759;
            v759 = 0 <= v708;
            bool v760;
            v760 = v759 && v709;
            bool v761;
            v761 = v760 == false;
            if (v761){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v760);
            } else {
            }
            bool v763;
            v763 = 0 <= v707;
            bool v765;
            if (v763){
                bool v764;
                v764 = v707 < 4;
                v765 = v764;
            } else {
                v765 = false;
            }
            bool v766;
            v766 = v765 == false;
            if (v766){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v765);
            } else {
            }
            int v768;
            v768 = v707 * 32;
            int v769;
            v769 = v708 + v692;
            int v770;
            v770 = v768 + v691;
            bool v771[4];
            int v772;
            v772 = 0;
            while (while_method_5(v772)){
                int v774;
                v774 = 0;
                while (while_method_8(v774)){
                    assert("Tensor range check" && 0 <= v772 && v772 < 1);
                    assert("Tensor range check" && 0 <= v774 && v774 < 4);
                    int v776;
                    v776 = 4 * v772;
                    int v777;
                    v777 = v776 + v774;
                    int v778;
                    v778 = v717[v777];
                    assert("Tensor range check" && 0 <= v778 && v778 < 32);
                    bool v779;
                    v779 = v534[v778];
                    assert("Tensor range check" && 0 <= v772 && v772 < 1);
                    assert("Tensor range check" && 0 <= v774 && v774 < 4);
                    v771[v777] = v779;
                    v774 += 1 ;
                }
                v772 += 1 ;
            }
            int v780[4];
            int v781;
            v781 = 0;
            while (while_method_5(v781)){
                int v783;
                v783 = 0;
                while (while_method_8(v783)){
                    assert("Tensor range check" && 0 <= v781 && v781 < 1);
                    assert("Tensor range check" && 0 <= v783 && v783 < 4);
                    int v785;
                    v785 = 4 * v781;
                    int v786;
                    v786 = v785 + v783;
                    bool v787;
                    v787 = v771[v786];
                    int v788;
                    if (v787){
                        v788 = 1;
                    } else {
                        v788 = 0;
                    }
                    assert("Tensor range check" && 0 <= v781 && v781 < 1);
                    assert("Tensor range check" && 0 <= v783 && v783 < 4);
                    v780[v786] = v788;
                    v783 += 1 ;
                }
                v781 += 1 ;
            }
            int v789;
            v789 = 0;
            int v790;
            v790 = 0;
            while (while_method_5(v790)){
                int v792;
                v792 = 0;
                while (while_method_8(v792)){
                    assert("Tensor range check" && 0 <= v790 && v790 < 1);
                    assert("Tensor range check" && 0 <= v792 && v792 < 4);
                    int v794;
                    v794 = 4 * v790;
                    int v795;
                    v795 = v794 + v792;
                    int v796;
                    v796 = v780[v795];
                    int v797;
                    v797 = v789 + v796;
                    v789 = v797;
                    v792 += 1 ;
                }
                v790 += 1 ;
            }
            auto v798 = cooperative_groups::coalesced_threads();
            int v799;
            v799 = threadIdx.x;
            int v800;
            v800 = v799 / 8;
            auto v801 = cooperative_groups::labeled_partition(v798,v800);
            Closure1 v802{};
            int v803;
            v803 = cooperative_groups::reduce(v801, v789, v802);
            float v804;
            v804 = (float)v803;
            float v805[4];
            int v806;
            v806 = 0;
            while (while_method_5(v806)){
                int v808;
                v808 = 0;
                while (while_method_8(v808)){
                    assert("Tensor range check" && 0 <= v806 && v806 < 1);
                    assert("Tensor range check" && 0 <= v808 && v808 < 4);
                    int v810;
                    v810 = 4 * v806;
                    int v811;
                    v811 = v810 + v808;
                    float v812;
                    v812 = v716[v811];
                    bool v813;
                    v813 = v771[v811];
                    float v814;
                    if (v813){
                        v814 = v812;
                    } else {
                        v814 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v806 && v806 < 1);
                    assert("Tensor range check" && 0 <= v808 && v808 < 4);
                    v805[v811] = v814;
                    v808 += 1 ;
                }
                v806 += 1 ;
            }
            float v815;
            v815 = 0.0f;
            int v816;
            v816 = 0;
            while (while_method_5(v816)){
                int v818;
                v818 = 0;
                while (while_method_8(v818)){
                    assert("Tensor range check" && 0 <= v816 && v816 < 1);
                    assert("Tensor range check" && 0 <= v818 && v818 < 4);
                    int v820;
                    v820 = 4 * v816;
                    int v821;
                    v821 = v820 + v818;
                    float v822;
                    v822 = v805[v821];
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
            v826 = v825 / 8;
            auto v827 = cooperative_groups::labeled_partition(v824,v826);
            Closure0 v828{};
            float v829;
            v829 = cooperative_groups::reduce(v827, v815, v828);
            float v830;
            v830 = v829 / v804;
            float v831[4];
            int v832;
            v832 = 0;
            while (while_method_5(v832)){
                int v834;
                v834 = 0;
                while (while_method_8(v834)){
                    assert("Tensor range check" && 0 <= v832 && v832 < 1);
                    assert("Tensor range check" && 0 <= v834 && v834 < 4);
                    int v836;
                    v836 = 4 * v832;
                    int v837;
                    v837 = v836 + v834;
                    float v838;
                    v838 = v716[v837];
                    float v839;
                    v839 = v838 - v830;
                    float v840;
                    v840 = v839 * v839;
                    assert("Tensor range check" && 0 <= v832 && v832 < 1);
                    assert("Tensor range check" && 0 <= v834 && v834 < 4);
                    v831[v837] = v840;
                    v834 += 1 ;
                }
                v832 += 1 ;
            }
            float v841[4];
            int v842;
            v842 = 0;
            while (while_method_5(v842)){
                int v844;
                v844 = 0;
                while (while_method_8(v844)){
                    assert("Tensor range check" && 0 <= v842 && v842 < 1);
                    assert("Tensor range check" && 0 <= v844 && v844 < 4);
                    int v846;
                    v846 = 4 * v842;
                    int v847;
                    v847 = v846 + v844;
                    float v848;
                    v848 = v831[v847];
                    bool v849;
                    v849 = v771[v847];
                    float v850;
                    if (v849){
                        v850 = v848;
                    } else {
                        v850 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v842 && v842 < 1);
                    assert("Tensor range check" && 0 <= v844 && v844 < 4);
                    v841[v847] = v850;
                    v844 += 1 ;
                }
                v842 += 1 ;
            }
            float v851;
            v851 = 0.0f;
            int v852;
            v852 = 0;
            while (while_method_5(v852)){
                int v854;
                v854 = 0;
                while (while_method_8(v854)){
                    assert("Tensor range check" && 0 <= v852 && v852 < 1);
                    assert("Tensor range check" && 0 <= v854 && v854 < 4);
                    int v856;
                    v856 = 4 * v852;
                    int v857;
                    v857 = v856 + v854;
                    float v858;
                    v858 = v841[v857];
                    float v859;
                    v859 = v851 + v858;
                    v851 = v859;
                    v854 += 1 ;
                }
                v852 += 1 ;
            }
            auto v860 = cooperative_groups::coalesced_threads();
            int v861;
            v861 = threadIdx.x;
            int v862;
            v862 = v861 / 8;
            auto v863 = cooperative_groups::labeled_partition(v860,v862);
            float v864;
            v864 = cooperative_groups::reduce(v863, v851, v828);
            float v865;
            v865 = v864 / v804;
            float v866;
            v866 = sqrt(v865);
            bool v867;
            v867 = v804 > 1.0f;
            float v871;
            if (v867){
                float v868;
                v868 = v866 * v804;
                float v869;
                v869 = v804 - 1.0f;
                float v870;
                v870 = v868 / v869;
                v871 = v870;
            } else {
                v871 = 0.0f;
            }
            float v872[4];
            int v873;
            v873 = 0;
            while (while_method_5(v873)){
                int v875;
                v875 = 0;
                while (while_method_8(v875)){
                    assert("Tensor range check" && 0 <= v873 && v873 < 1);
                    assert("Tensor range check" && 0 <= v875 && v875 < 4);
                    int v877;
                    v877 = 4 * v873;
                    int v878;
                    v878 = v877 + v875;
                    float v879;
                    v879 = v716[v878];
                    bool v880;
                    v880 = v771[v878];
                    float v881;
                    v881 = curand_normal(&v521);
                    bool v882;
                    v882 = v871 >= 0.1f;
                    float v883;
                    if (v882){
                        v883 = v871;
                    } else {
                        v883 = 0.1f;
                    }
                    float v884;
                    v884 = v881 * v883;
                    float v885;
                    v885 = v884 + v830;
                    float v886;
                    if (v880){
                        v886 = v879;
                    } else {
                        v886 = v885;
                    }
                    assert("Tensor range check" && 0 <= v873 && v873 < 1);
                    assert("Tensor range check" && 0 <= v875 && v875 < 4);
                    v872[v878] = v886;
                    v875 += 1 ;
                }
                v873 += 1 ;
            }
            assert("Tensor range check" && 0 <= v708 && v708 < 64);
            assert("Tensor range check" && 0 <= v707 && v707 < 4);
            int v887;
            v887 = 0;
            while (while_method_5(v887)){
                assert("Tensor range check" && 0 <= v887 && v887 < 1);
                int v889;
                v889 = 32 * v887;
                int v890;
                v890 = v889 + v715;
                assert("Tensor range check" && 0 <= v887 && v887 < 1);
                int v891;
                v891 = 4 * v887;
                int4* v892;
                v892 = reinterpret_cast<int4*>(v872 + v891);
                int4* v893;
                v893 = reinterpret_cast<int4*>(v524 + v890);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v892) % 16 == 0 && reinterpret_cast<unsigned long long>(v893) % 16 == 0);
                *v893 = *v892;
                v887 += 1 ;
            }
            v702 += 24 ;
        }
        int v894;
        v894 = threadIdx.x;
        bool v895;
        v895 = 0 <= v894;
        bool v896;
        v896 = v895 == false;
        if (v896){
            assert("The index needs to be zero or positive." && v895);
        } else {
        }
        int v898;
        v898 = v894 % 8;
        int v899;
        v899 = v894 / 8;
        int v900;
        v900 = v899 % 32;
        int v901;
        v901 = v899 / 32;
        bool v902;
        v902 = v901 < 1;
        bool v903;
        v903 = v902 == false;
        if (v903){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v902);
        } else {
        }
        assert("Tensor range check" && 0 <= v901 && v901 < 1);
        assert("Tensor range check" && 0 <= v900 && v900 < 32);
        assert("Tensor range check" && 0 <= v898 && v898 < 8);
        int v905;
        v905 = 4 * v898;
        int v906;
        v906 = 32 * v900;
        int v907;
        v907 = v906 + v905;
        int v908;
        v908 = 2048 * v901;
        int v909;
        v909 = v908 + v907;
        assert("Tensor range check" && 0 <= v901 && v901 < 1);
        assert("Tensor range check" && 0 <= v900 && v900 < 32);
        assert("Tensor range check" && 0 <= v898 && v898 < 8);
        int v910;
        v910 = blockIdx.x;
        int v911;
        v911 = v910;
        while (while_method_15(v911)){
            bool v913;
            v913 = 0 <= v911;
            bool v914;
            v914 = v913 == false;
            if (v914){
                assert("The index needs to be zero or positive." && v913);
            } else {
            }
            int v916;
            v916 = v911 % 2;
            int v917;
            v917 = v911 / 2;
            bool v918;
            v918 = v917 < 64;
            bool v919;
            v919 = v918 == false;
            if (v919){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v918);
            } else {
            }
            assert("Tensor range check" && 0 <= v917 && v917 < 64);
            assert("Tensor range check" && 0 <= v916 && v916 < 2);
            int v921;
            v921 = 1024 * v916;
            int v922;
            v922 = v921 + v909;
            int v923;
            v923 = 2048 * v917;
            int v924;
            v924 = v923 + v922;
            float v925[4];
            int v926[4];
            int v927;
            v927 = 0;
            while (while_method_5(v927)){
                assert("Tensor range check" && 0 <= v927 && v927 < 1);
                int v929;
                v929 = 4 * v927;
                assert("Tensor range check" && 0 <= v927 && v927 < 1);
                int v930;
                v930 = 32 * v927;
                int v931;
                v931 = v930 + v924;
                int4* v932;
                v932 = reinterpret_cast<int4*>(v528 + v931);
                int4* v933;
                v933 = reinterpret_cast<int4*>(v925 + v929);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v932) % 16 == 0 && reinterpret_cast<unsigned long long>(v933) % 16 == 0);
                *v933 = *v932;
                v927 += 1 ;
            }
            int v934;
            v934 = 0;
            while (while_method_5(v934)){
                int v936;
                v936 = 0;
                while (while_method_8(v936)){
                    bool v938;
                    v938 = 0 <= v936;
                    bool v940;
                    if (v938){
                        bool v939;
                        v939 = v936 < 4;
                        v940 = v939;
                    } else {
                        v940 = false;
                    }
                    bool v941;
                    v941 = v940 == false;
                    if (v941){
                        assert("The indices should be inside the range of the dimension." && v940);
                    } else {
                    }
                    bool v943;
                    v943 = 0 <= v898;
                    bool v945;
                    if (v943){
                        bool v944;
                        v944 = v898 < 8;
                        v945 = v944;
                    } else {
                        v945 = false;
                    }
                    bool v946;
                    v946 = v945 == false;
                    if (v946){
                        assert("The indices should be inside the range of the dimension." && v945);
                    } else {
                    }
                    int v948;
                    v948 = v898 * 4;
                    int v949;
                    v949 = v936 + v948;
                    bool v950;
                    v950 = 0 <= v934;
                    bool v952;
                    if (v950){
                        bool v951;
                        v951 = v934 < 1;
                        v952 = v951;
                    } else {
                        v952 = false;
                    }
                    bool v953;
                    v953 = v952 == false;
                    if (v953){
                        assert("The indices should be inside the range of the dimension." && v952);
                    } else {
                    }
                    int v955;
                    v955 = v934 * 32;
                    int v956;
                    v956 = v949 + v955;
                    assert("Tensor range check" && 0 <= v934 && v934 < 1);
                    assert("Tensor range check" && 0 <= v936 && v936 < 4);
                    int v957;
                    v957 = 4 * v934;
                    int v958;
                    v958 = v957 + v936;
                    v926[v958] = v956;
                    v936 += 1 ;
                }
                v934 += 1 ;
            }
            bool v959;
            v959 = 0 <= v901;
            bool v960;
            v960 = v959 && v902;
            bool v961;
            v961 = v960 == false;
            if (v961){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v960);
            } else {
            }
            bool v963;
            v963 = 0 <= v900;
            bool v965;
            if (v963){
                bool v964;
                v964 = v900 < 32;
                v965 = v964;
            } else {
                v965 = false;
            }
            bool v966;
            v966 = v965 == false;
            if (v966){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v965);
            } else {
            }
            bool v968;
            v968 = 0 <= v917;
            bool v969;
            v969 = v968 && v918;
            bool v970;
            v970 = v969 == false;
            if (v970){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v969);
            } else {
            }
            bool v972;
            v972 = 0 <= v916;
            bool v974;
            if (v972){
                bool v973;
                v973 = v916 < 2;
                v974 = v973;
            } else {
                v974 = false;
            }
            bool v975;
            v975 = v974 == false;
            if (v975){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v974);
            } else {
            }
            int v977;
            v977 = v916 * 32;
            int v978;
            v978 = v917 + v901;
            int v979;
            v979 = v977 + v900;
            bool v980[4];
            int v981;
            v981 = 0;
            while (while_method_5(v981)){
                int v983;
                v983 = 0;
                while (while_method_8(v983)){
                    assert("Tensor range check" && 0 <= v981 && v981 < 1);
                    assert("Tensor range check" && 0 <= v983 && v983 < 4);
                    int v985;
                    v985 = 4 * v981;
                    int v986;
                    v986 = v985 + v983;
                    int v987;
                    v987 = v926[v986];
                    assert("Tensor range check" && 0 <= v987 && v987 < 32);
                    bool v988;
                    v988 = v534[v987];
                    assert("Tensor range check" && 0 <= v981 && v981 < 1);
                    assert("Tensor range check" && 0 <= v983 && v983 < 4);
                    v980[v986] = v988;
                    v983 += 1 ;
                }
                v981 += 1 ;
            }
            int v989[4];
            int v990;
            v990 = 0;
            while (while_method_5(v990)){
                int v992;
                v992 = 0;
                while (while_method_8(v992)){
                    assert("Tensor range check" && 0 <= v990 && v990 < 1);
                    assert("Tensor range check" && 0 <= v992 && v992 < 4);
                    int v994;
                    v994 = 4 * v990;
                    int v995;
                    v995 = v994 + v992;
                    bool v996;
                    v996 = v980[v995];
                    int v997;
                    if (v996){
                        v997 = 1;
                    } else {
                        v997 = 0;
                    }
                    assert("Tensor range check" && 0 <= v990 && v990 < 1);
                    assert("Tensor range check" && 0 <= v992 && v992 < 4);
                    v989[v995] = v997;
                    v992 += 1 ;
                }
                v990 += 1 ;
            }
            int v998;
            v998 = 0;
            int v999;
            v999 = 0;
            while (while_method_5(v999)){
                int v1001;
                v1001 = 0;
                while (while_method_8(v1001)){
                    assert("Tensor range check" && 0 <= v999 && v999 < 1);
                    assert("Tensor range check" && 0 <= v1001 && v1001 < 4);
                    int v1003;
                    v1003 = 4 * v999;
                    int v1004;
                    v1004 = v1003 + v1001;
                    int v1005;
                    v1005 = v989[v1004];
                    int v1006;
                    v1006 = v998 + v1005;
                    v998 = v1006;
                    v1001 += 1 ;
                }
                v999 += 1 ;
            }
            auto v1007 = cooperative_groups::coalesced_threads();
            int v1008;
            v1008 = threadIdx.x;
            int v1009;
            v1009 = v1008 / 8;
            auto v1010 = cooperative_groups::labeled_partition(v1007,v1009);
            Closure1 v1011{};
            int v1012;
            v1012 = cooperative_groups::reduce(v1010, v998, v1011);
            float v1013;
            v1013 = (float)v1012;
            float v1014[4];
            int v1015;
            v1015 = 0;
            while (while_method_5(v1015)){
                int v1017;
                v1017 = 0;
                while (while_method_8(v1017)){
                    assert("Tensor range check" && 0 <= v1015 && v1015 < 1);
                    assert("Tensor range check" && 0 <= v1017 && v1017 < 4);
                    int v1019;
                    v1019 = 4 * v1015;
                    int v1020;
                    v1020 = v1019 + v1017;
                    float v1021;
                    v1021 = v925[v1020];
                    bool v1022;
                    v1022 = v980[v1020];
                    float v1023;
                    if (v1022){
                        v1023 = v1021;
                    } else {
                        v1023 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v1015 && v1015 < 1);
                    assert("Tensor range check" && 0 <= v1017 && v1017 < 4);
                    v1014[v1020] = v1023;
                    v1017 += 1 ;
                }
                v1015 += 1 ;
            }
            float v1024;
            v1024 = 0.0f;
            int v1025;
            v1025 = 0;
            while (while_method_5(v1025)){
                int v1027;
                v1027 = 0;
                while (while_method_8(v1027)){
                    assert("Tensor range check" && 0 <= v1025 && v1025 < 1);
                    assert("Tensor range check" && 0 <= v1027 && v1027 < 4);
                    int v1029;
                    v1029 = 4 * v1025;
                    int v1030;
                    v1030 = v1029 + v1027;
                    float v1031;
                    v1031 = v1014[v1030];
                    float v1032;
                    v1032 = v1024 + v1031;
                    v1024 = v1032;
                    v1027 += 1 ;
                }
                v1025 += 1 ;
            }
            auto v1033 = cooperative_groups::coalesced_threads();
            int v1034;
            v1034 = threadIdx.x;
            int v1035;
            v1035 = v1034 / 8;
            auto v1036 = cooperative_groups::labeled_partition(v1033,v1035);
            Closure0 v1037{};
            float v1038;
            v1038 = cooperative_groups::reduce(v1036, v1024, v1037);
            float v1039;
            v1039 = v1038 / v1013;
            float v1040[4];
            int v1041;
            v1041 = 0;
            while (while_method_5(v1041)){
                int v1043;
                v1043 = 0;
                while (while_method_8(v1043)){
                    assert("Tensor range check" && 0 <= v1041 && v1041 < 1);
                    assert("Tensor range check" && 0 <= v1043 && v1043 < 4);
                    int v1045;
                    v1045 = 4 * v1041;
                    int v1046;
                    v1046 = v1045 + v1043;
                    float v1047;
                    v1047 = v925[v1046];
                    float v1048;
                    v1048 = v1047 - v1039;
                    float v1049;
                    v1049 = v1048 * v1048;
                    assert("Tensor range check" && 0 <= v1041 && v1041 < 1);
                    assert("Tensor range check" && 0 <= v1043 && v1043 < 4);
                    v1040[v1046] = v1049;
                    v1043 += 1 ;
                }
                v1041 += 1 ;
            }
            float v1050[4];
            int v1051;
            v1051 = 0;
            while (while_method_5(v1051)){
                int v1053;
                v1053 = 0;
                while (while_method_8(v1053)){
                    assert("Tensor range check" && 0 <= v1051 && v1051 < 1);
                    assert("Tensor range check" && 0 <= v1053 && v1053 < 4);
                    int v1055;
                    v1055 = 4 * v1051;
                    int v1056;
                    v1056 = v1055 + v1053;
                    float v1057;
                    v1057 = v1040[v1056];
                    bool v1058;
                    v1058 = v980[v1056];
                    float v1059;
                    if (v1058){
                        v1059 = v1057;
                    } else {
                        v1059 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v1051 && v1051 < 1);
                    assert("Tensor range check" && 0 <= v1053 && v1053 < 4);
                    v1050[v1056] = v1059;
                    v1053 += 1 ;
                }
                v1051 += 1 ;
            }
            float v1060;
            v1060 = 0.0f;
            int v1061;
            v1061 = 0;
            while (while_method_5(v1061)){
                int v1063;
                v1063 = 0;
                while (while_method_8(v1063)){
                    assert("Tensor range check" && 0 <= v1061 && v1061 < 1);
                    assert("Tensor range check" && 0 <= v1063 && v1063 < 4);
                    int v1065;
                    v1065 = 4 * v1061;
                    int v1066;
                    v1066 = v1065 + v1063;
                    float v1067;
                    v1067 = v1050[v1066];
                    float v1068;
                    v1068 = v1060 + v1067;
                    v1060 = v1068;
                    v1063 += 1 ;
                }
                v1061 += 1 ;
            }
            auto v1069 = cooperative_groups::coalesced_threads();
            int v1070;
            v1070 = threadIdx.x;
            int v1071;
            v1071 = v1070 / 8;
            auto v1072 = cooperative_groups::labeled_partition(v1069,v1071);
            float v1073;
            v1073 = cooperative_groups::reduce(v1072, v1060, v1037);
            float v1074;
            v1074 = v1073 / v1013;
            float v1075;
            v1075 = sqrt(v1074);
            bool v1076;
            v1076 = v1013 > 1.0f;
            float v1080;
            if (v1076){
                float v1077;
                v1077 = v1075 * v1013;
                float v1078;
                v1078 = v1013 - 1.0f;
                float v1079;
                v1079 = v1077 / v1078;
                v1080 = v1079;
            } else {
                v1080 = 0.0f;
            }
            float v1081[4];
            int v1082;
            v1082 = 0;
            while (while_method_5(v1082)){
                int v1084;
                v1084 = 0;
                while (while_method_8(v1084)){
                    assert("Tensor range check" && 0 <= v1082 && v1082 < 1);
                    assert("Tensor range check" && 0 <= v1084 && v1084 < 4);
                    int v1086;
                    v1086 = 4 * v1082;
                    int v1087;
                    v1087 = v1086 + v1084;
                    float v1088;
                    v1088 = v925[v1087];
                    bool v1089;
                    v1089 = v980[v1087];
                    float v1090;
                    v1090 = curand_normal(&v521);
                    bool v1091;
                    v1091 = v1080 >= 0.1f;
                    float v1092;
                    if (v1091){
                        v1092 = v1080;
                    } else {
                        v1092 = 0.1f;
                    }
                    float v1093;
                    v1093 = v1090 * v1092;
                    float v1094;
                    v1094 = v1093 + v1039;
                    float v1095;
                    if (v1089){
                        v1095 = v1088;
                    } else {
                        v1095 = v1094;
                    }
                    assert("Tensor range check" && 0 <= v1082 && v1082 < 1);
                    assert("Tensor range check" && 0 <= v1084 && v1084 < 4);
                    v1081[v1087] = v1095;
                    v1084 += 1 ;
                }
                v1082 += 1 ;
            }
            assert("Tensor range check" && 0 <= v917 && v917 < 64);
            assert("Tensor range check" && 0 <= v916 && v916 < 2);
            int v1096;
            v1096 = 0;
            while (while_method_5(v1096)){
                assert("Tensor range check" && 0 <= v1096 && v1096 < 1);
                int v1098;
                v1098 = 32 * v1096;
                int v1099;
                v1099 = v1098 + v924;
                assert("Tensor range check" && 0 <= v1096 && v1096 < 1);
                int v1100;
                v1100 = 4 * v1096;
                int4* v1101;
                v1101 = reinterpret_cast<int4*>(v1081 + v1100);
                int4* v1102;
                v1102 = reinterpret_cast<int4*>(v528 + v1099);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1101) % 16 == 0 && reinterpret_cast<unsigned long long>(v1102) % 16 == 0);
                *v1102 = *v1101;
                v1096 += 1 ;
            }
            v911 += 24 ;
        }
        v519.sync() ;
        static float v1103[8192];
        int v1104;
        v1104 = threadIdx.x;
        int v1105;
        v1105 = blockIdx.x;
        int v1106;
        v1106 = v1105 * 256;
        int v1107;
        v1107 = v1104 + v1106;
        int v1108;
        v1108 = v1107 / 32;
        int v1109;
        v1109 = v1108;
        while (while_method_16(v1109)){
            bool v1111;
            v1111 = 0 <= v1109;
            bool v1112;
            v1112 = v1111 == false;
            if (v1112){
                assert("The index needs to be zero or positive." && v1111);
            } else {
            }
            int v1114;
            v1114 = v1109 % 128;
            int v1115;
            v1115 = v1109 / 128;
            bool v1116;
            v1116 = v1115 < 64;
            bool v1117;
            v1117 = v1116 == false;
            if (v1117){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1116);
            } else {
            }
            assert("Tensor range check" && 0 <= v1115 && v1115 < 64);
            assert("Tensor range check" && 0 <= v1114 && v1114 < 128);
            int v1119;
            v1119 = 32 * v1114;
            int v1120;
            v1120 = 4096 * v1115;
            int v1121;
            v1121 = v1120 + v1119;
            float v1122;
            v1122 = 0.0f;
            int v1123;
            v1123 = threadIdx.x;
            int v1124;
            v1124 = v1123 % 32;
            int v1125;
            v1125 = v1124;
            while (while_method_6(v1125)){
                bool v1127;
                v1127 = 0 <= v1125;
                bool v1128;
                v1128 = v1127 == false;
                if (v1128){
                    assert("The index needs to be zero or positive." && v1127);
                } else {
                }
                bool v1130;
                v1130 = v1125 < 8;
                bool v1131;
                v1131 = v1130 == false;
                if (v1131){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1130);
                } else {
                }
                assert("Tensor range check" && 0 <= v1125 && v1125 < 8);
                int v1133;
                v1133 = 4 * v1125;
                int v1134;
                v1134 = v1133 + v1121;
                float v1135[4];
                int4* v1136;
                v1136 = reinterpret_cast<int4*>(v524 + v1134);
                int4* v1137;
                v1137 = reinterpret_cast<int4*>(v1135 + 0);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1136) % 16 == 0 && reinterpret_cast<unsigned long long>(v1137) % 16 == 0);
                *v1137 = *v1136;
                int v1138;
                v1138 = 0;
                while (while_method_8(v1138)){
                    assert("Tensor range check" && 0 <= v1138 && v1138 < 4);
                    float v1140;
                    v1140 = v1135[v1138];
                    float v1141;
                    v1141 = v1140 * v1140;
                    float v1142;
                    v1142 = v1122 + v1141;
                    v1122 = v1142;
                    v1138 += 1 ;
                }
                v1125 += 32 ;
            }
            __syncwarp();
            auto v1143 = cooperative_groups::coalesced_threads();
            Closure0 v1144{};
            float v1145;
            v1145 = cooperative_groups::reduce(v1143, v1122, v1144);
            float v1146;
            v1146 = sqrt(v1145);
            assert("Tensor range check" && 0 <= v1115 && v1115 < 64);
            assert("Tensor range check" && 0 <= v1114 && v1114 < 128);
            int v1147;
            v1147 = 128 * v1115;
            int v1148;
            v1148 = v1147 + v1114;
            v1103[v1148] = v1146;
            v1109 += 192 ;
        }
        __syncthreads();
        v519.sync() ;
        float v1149;
        v1149 = 0.0f;
        int v1150;
        v1150 = threadIdx.x;
        int v1151;
        v1151 = blockIdx.x;
        int v1152;
        v1152 = v1151 * 256;
        int v1153;
        v1153 = v1150 + v1152;
        int v1154;
        v1154 = v1153;
        while (while_method_17(v1154)){
            bool v1156;
            v1156 = 0 <= v1154;
            bool v1157;
            v1157 = v1156 == false;
            if (v1157){
                assert("The index needs to be zero or positive." && v1156);
            } else {
            }
            int v1159;
            v1159 = v1154 % 32;
            int v1160;
            v1160 = v1154 / 32;
            bool v1161;
            v1161 = v1160 < 64;
            bool v1162;
            v1162 = v1161 == false;
            if (v1162){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1161);
            } else {
            }
            assert("Tensor range check" && 0 <= v1160 && v1160 < 64);
            assert("Tensor range check" && 0 <= v1159 && v1159 < 32);
            int v1164;
            v1164 = 4 * v1159;
            int v1165;
            v1165 = 128 * v1160;
            int v1166;
            v1166 = v1165 + v1164;
            float v1167[4];
            int4* v1168;
            v1168 = reinterpret_cast<int4*>(v1103 + v1166);
            int4* v1169;
            v1169 = reinterpret_cast<int4*>(v1167 + 0);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1168) % 16 == 0 && reinterpret_cast<unsigned long long>(v1169) % 16 == 0);
            *v1169 = *v1168;
            int v1170; float v1171;
            Tuple13 tmp92 = Tuple13{0, v1149};
            v1170 = tmp92.v0; v1171 = tmp92.v1;
            while (while_method_8(v1170)){
                assert("Tensor range check" && 0 <= v1170 && v1170 < 4);
                float v1173;
                v1173 = v1167[v1170];
                bool v1174;
                v1174 = v1171 >= v1173;
                float v1175;
                if (v1174){
                    v1175 = v1171;
                } else {
                    v1175 = v1173;
                }
                v1171 = v1175;
                v1170 += 1 ;
            }
            v1149 = v1171;
            v1154 += 6144 ;
        }
        __syncwarp();
        auto v1176 = cooperative_groups::coalesced_threads();
        Closure7 v1177{};
        float v1178;
        v1178 = cooperative_groups::reduce(v1176, v1149, v1177);
        int v1179;
        v1179 = threadIdx.x;
        int v1180;
        v1180 = v1179 / 32;
        extern __shared__ unsigned char v1181[];
        float * v1182;
        v1182 = reinterpret_cast<float *>(&v1181[0ull]);
        assert("Tensor range check" && 0 <= v1180 && v1180 < 8);
        v1182[v1180] = v1178;
        __syncthreads();
        int v1184;
        v1184 = threadIdx.x;
        int v1185;
        v1185 = v1184 % 32;
        bool v1186;
        v1186 = v1185 < 8;
        float v1188;
        if (v1186){
            assert("Tensor range check" && 0 <= v1185 && v1185 < 8);
            float v1187;
            v1187 = v1182[v1185];
            v1188 = v1187;
        } else {
            v1188 = 0.0f;
        }
        __syncthreads();
        auto v1189 = cooperative_groups::coalesced_threads();
        float v1190;
        v1190 = cooperative_groups::reduce(v1189, v1188, v1177);
        int v1191;
        v1191 = blockIdx.x;
        static float v1192[24];
        assert("Tensor range check" && 0 <= v1191 && v1191 < 24);
        v1192[v1191] = v1190;
        v519.sync() ;
        float v1193;
        v1193 = 0.0f;
        int v1194;
        v1194 = threadIdx.x;
        int v1195;
        v1195 = v1194 % 32;
        int v1196;
        v1196 = v1195;
        while (while_method_18(v1196)){
            bool v1198;
            v1198 = 0 <= v1196;
            bool v1199;
            v1199 = v1198 == false;
            if (v1199){
                assert("The index needs to be zero or positive." && v1198);
            } else {
            }
            bool v1201;
            v1201 = v1196 < 24;
            bool v1202;
            v1202 = v1201 == false;
            if (v1202){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1201);
            } else {
            }
            assert("Tensor range check" && 0 <= v1196 && v1196 < 24);
            float v1204;
            v1204 = v1192[v1196];
            bool v1205;
            v1205 = v1193 >= v1204;
            float v1206;
            if (v1205){
                v1206 = v1193;
            } else {
                v1206 = v1204;
            }
            v1193 = v1206;
            v1196 += 32 ;
        }
        __syncwarp();
        auto v1207 = cooperative_groups::coalesced_threads();
        float v1208;
        v1208 = cooperative_groups::reduce(v1207, v1193, v1177);
        int v1209;
        v1209 = threadIdx.x;
        int v1210;
        v1210 = blockIdx.x;
        int v1211;
        v1211 = v1210 * 256;
        int v1212;
        v1212 = v1209 + v1211;
        bool v1213;
        v1213 = v1212 == 0;
        if (v1213){
            cuda::counting_semaphore<cuda::thread_scope_system, 1> & v1214 = console_lock;
            auto v1215 = cooperative_groups::coalesced_threads();
            v1214.acquire();
            printf("{%s = %f}\n","max_norm", v1208);
            v1214.release();
            v1215.sync() ;
        } else {
        }
        __syncwarp();
        static float v1218[4096];
        int v1219;
        v1219 = threadIdx.x;
        int v1220;
        v1220 = blockIdx.x;
        int v1221;
        v1221 = v1220 * 256;
        int v1222;
        v1222 = v1219 + v1221;
        int v1223;
        v1223 = v1222 / 32;
        int v1224;
        v1224 = v1223;
        while (while_method_10(v1224)){
            bool v1226;
            v1226 = 0 <= v1224;
            bool v1227;
            v1227 = v1226 == false;
            if (v1227){
                assert("The index needs to be zero or positive." && v1226);
            } else {
            }
            int v1229;
            v1229 = v1224 % 64;
            int v1230;
            v1230 = v1224 / 64;
            bool v1231;
            v1231 = v1230 < 64;
            bool v1232;
            v1232 = v1231 == false;
            if (v1232){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1231);
            } else {
            }
            assert("Tensor range check" && 0 <= v1230 && v1230 < 64);
            assert("Tensor range check" && 0 <= v1229 && v1229 < 64);
            int v1234;
            v1234 = 32 * v1229;
            int v1235;
            v1235 = 2048 * v1230;
            int v1236;
            v1236 = v1235 + v1234;
            float v1237;
            v1237 = 0.0f;
            int v1238;
            v1238 = threadIdx.x;
            int v1239;
            v1239 = v1238 % 32;
            int v1240;
            v1240 = v1239;
            while (while_method_6(v1240)){
                bool v1242;
                v1242 = 0 <= v1240;
                bool v1243;
                v1243 = v1242 == false;
                if (v1243){
                    assert("The index needs to be zero or positive." && v1242);
                } else {
                }
                bool v1245;
                v1245 = v1240 < 8;
                bool v1246;
                v1246 = v1245 == false;
                if (v1246){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1245);
                } else {
                }
                assert("Tensor range check" && 0 <= v1240 && v1240 < 8);
                int v1248;
                v1248 = 4 * v1240;
                int v1249;
                v1249 = v1248 + v1236;
                float v1250[4];
                int4* v1251;
                v1251 = reinterpret_cast<int4*>(v528 + v1249);
                int4* v1252;
                v1252 = reinterpret_cast<int4*>(v1250 + 0);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1251) % 16 == 0 && reinterpret_cast<unsigned long long>(v1252) % 16 == 0);
                *v1252 = *v1251;
                int v1253;
                v1253 = 0;
                while (while_method_8(v1253)){
                    assert("Tensor range check" && 0 <= v1253 && v1253 < 4);
                    float v1255;
                    v1255 = v1250[v1253];
                    float v1256;
                    v1256 = v1255 * v1255;
                    float v1257;
                    v1257 = v1237 + v1256;
                    v1237 = v1257;
                    v1253 += 1 ;
                }
                v1240 += 32 ;
            }
            __syncwarp();
            auto v1258 = cooperative_groups::coalesced_threads();
            Closure0 v1259{};
            float v1260;
            v1260 = cooperative_groups::reduce(v1258, v1237, v1259);
            float v1261;
            v1261 = sqrt(v1260);
            assert("Tensor range check" && 0 <= v1230 && v1230 < 64);
            assert("Tensor range check" && 0 <= v1229 && v1229 < 64);
            int v1262;
            v1262 = 64 * v1230;
            int v1263;
            v1263 = v1262 + v1229;
            v1218[v1263] = v1261;
            v1224 += 192 ;
        }
        __syncthreads();
        v519.sync() ;
        float v1264;
        v1264 = 0.0f;
        int v1265;
        v1265 = threadIdx.x;
        int v1266;
        v1266 = blockIdx.x;
        int v1267;
        v1267 = v1266 * 256;
        int v1268;
        v1268 = v1265 + v1267;
        int v1269;
        v1269 = v1268;
        while (while_method_19(v1269)){
            bool v1271;
            v1271 = 0 <= v1269;
            bool v1272;
            v1272 = v1271 == false;
            if (v1272){
                assert("The index needs to be zero or positive." && v1271);
            } else {
            }
            int v1274;
            v1274 = v1269 % 16;
            int v1275;
            v1275 = v1269 / 16;
            bool v1276;
            v1276 = v1275 < 64;
            bool v1277;
            v1277 = v1276 == false;
            if (v1277){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1276);
            } else {
            }
            assert("Tensor range check" && 0 <= v1275 && v1275 < 64);
            assert("Tensor range check" && 0 <= v1274 && v1274 < 16);
            int v1279;
            v1279 = 4 * v1274;
            int v1280;
            v1280 = 64 * v1275;
            int v1281;
            v1281 = v1280 + v1279;
            float v1282[4];
            int4* v1283;
            v1283 = reinterpret_cast<int4*>(v1218 + v1281);
            int4* v1284;
            v1284 = reinterpret_cast<int4*>(v1282 + 0);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1283) % 16 == 0 && reinterpret_cast<unsigned long long>(v1284) % 16 == 0);
            *v1284 = *v1283;
            int v1285; float v1286;
            Tuple13 tmp93 = Tuple13{0, v1264};
            v1285 = tmp93.v0; v1286 = tmp93.v1;
            while (while_method_8(v1285)){
                assert("Tensor range check" && 0 <= v1285 && v1285 < 4);
                float v1288;
                v1288 = v1282[v1285];
                bool v1289;
                v1289 = v1286 >= v1288;
                float v1290;
                if (v1289){
                    v1290 = v1286;
                } else {
                    v1290 = v1288;
                }
                v1286 = v1290;
                v1285 += 1 ;
            }
            v1264 = v1286;
            v1269 += 6144 ;
        }
        __syncwarp();
        auto v1291 = cooperative_groups::coalesced_threads();
        float v1292;
        v1292 = cooperative_groups::reduce(v1291, v1264, v1177);
        int v1293;
        v1293 = threadIdx.x;
        int v1294;
        v1294 = v1293 / 32;
        extern __shared__ unsigned char v1295[];
        float * v1296;
        v1296 = reinterpret_cast<float *>(&v1295[0ull]);
        assert("Tensor range check" && 0 <= v1294 && v1294 < 8);
        v1296[v1294] = v1292;
        __syncthreads();
        int v1298;
        v1298 = threadIdx.x;
        int v1299;
        v1299 = v1298 % 32;
        bool v1300;
        v1300 = v1299 < 8;
        float v1302;
        if (v1300){
            assert("Tensor range check" && 0 <= v1299 && v1299 < 8);
            float v1301;
            v1301 = v1296[v1299];
            v1302 = v1301;
        } else {
            v1302 = 0.0f;
        }
        __syncthreads();
        auto v1303 = cooperative_groups::coalesced_threads();
        float v1304;
        v1304 = cooperative_groups::reduce(v1303, v1302, v1177);
        int v1305;
        v1305 = blockIdx.x;
        static float v1306[24];
        assert("Tensor range check" && 0 <= v1305 && v1305 < 24);
        v1306[v1305] = v1304;
        v519.sync() ;
        float v1307;
        v1307 = 0.0f;
        int v1308;
        v1308 = threadIdx.x;
        int v1309;
        v1309 = v1308 % 32;
        int v1310;
        v1310 = v1309;
        while (while_method_18(v1310)){
            bool v1312;
            v1312 = 0 <= v1310;
            bool v1313;
            v1313 = v1312 == false;
            if (v1313){
                assert("The index needs to be zero or positive." && v1312);
            } else {
            }
            bool v1315;
            v1315 = v1310 < 24;
            bool v1316;
            v1316 = v1315 == false;
            if (v1316){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1315);
            } else {
            }
            assert("Tensor range check" && 0 <= v1310 && v1310 < 24);
            float v1318;
            v1318 = v1306[v1310];
            bool v1319;
            v1319 = v1307 >= v1318;
            float v1320;
            if (v1319){
                v1320 = v1307;
            } else {
                v1320 = v1318;
            }
            v1307 = v1320;
            v1310 += 32 ;
        }
        __syncwarp();
        auto v1321 = cooperative_groups::coalesced_threads();
        float v1322;
        v1322 = cooperative_groups::reduce(v1321, v1307, v1177);
        int v1323;
        v1323 = threadIdx.x;
        int v1324;
        v1324 = blockIdx.x;
        int v1325;
        v1325 = v1324 * 256;
        int v1326;
        v1326 = v1323 + v1325;
        bool v1327;
        v1327 = v1326 == 0;
        if (v1327){
            cuda::counting_semaphore<cuda::thread_scope_system, 1> & v1328 = console_lock;
            auto v1329 = cooperative_groups::coalesced_threads();
            v1328.acquire();
            printf("{%s = %f}\n","max_norm", v1322);
            v1328.release();
            v1329.sync() ;
        } else {
        }
        __syncwarp();
        extern __shared__ unsigned char v1332[];
        float * v1333;
        v1333 = reinterpret_cast<float *>(&v1332[0ull]);
        int v1335;
        v1335 = blockIdx.x;
        int v1336;
        v1336 = v1335;
        while (while_method_9(v1336)){
            bool v1338;
            v1338 = 0 <= v1336;
            bool v1339;
            v1339 = v1338 == false;
            if (v1339){
                assert("The index needs to be zero or positive." && v1338);
            } else {
            }
            int v1341;
            v1341 = v1336 % 1;
            bool v1342;
            v1342 = v1336 < 16;
            bool v1343;
            v1343 = v1342 == false;
            if (v1343){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1342);
            } else {
            }
            assert("Tensor range check" && 0 <= v1336 && v1336 < 16);
            assert("Tensor range check" && 0 <= v1341 && v1341 < 1);
            int v1345;
            v1345 = 32 * v1341;
            int v1346;
            v1346 = 16384 * v1336;
            int v1347;
            v1347 = v1346 + v1345;
            int v1348;
            v1348 = 262144 * v1341;
            int v1349;
            v1349 = 512 * v1336;
            int v1350;
            v1350 = v1349 + v1348;
            int v1351;
            v1351 = threadIdx.x;
            int v1352;
            v1352 = v1351;
            while (while_method_14(v1352)){
                bool v1354;
                v1354 = 0 <= v1352;
                bool v1355;
                v1355 = v1354 == false;
                if (v1355){
                    assert("The index needs to be zero or positive." && v1354);
                } else {
                }
                int v1357;
                v1357 = v1352 % 32;
                int v1358;
                v1358 = v1352 / 32;
                bool v1359;
                v1359 = v1358 < 512;
                bool v1360;
                v1360 = v1359 == false;
                if (v1360){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1359);
                } else {
                }
                assert("Tensor range check" && 0 <= v1358 && v1358 < 512);
                assert("Tensor range check" && 0 <= v1357 && v1357 < 32);
                int v1362;
                v1362 = v1357 + v1347;
                int v1363;
                v1363 = 32 * v1358;
                int v1364;
                v1364 = v1363 + v1362;
                float v1365;
                v1365 = v524[v1364];
                assert("Tensor range check" && 0 <= v1358 && v1358 < 512);
                assert("Tensor range check" && 0 <= v1357 && v1357 < 32);
                int v1366;
                v1366 = 33 * v1358;
                int v1367;
                v1367 = v1366 + v1357;
                v1333[v1367] = v1365;
                v1352 += 256 ;
            }
            __syncthreads();
            int v1368;
            v1368 = threadIdx.x;
            int v1369;
            v1369 = v1368;
            while (while_method_14(v1369)){
                bool v1371;
                v1371 = 0 <= v1369;
                bool v1372;
                v1372 = v1371 == false;
                if (v1372){
                    assert("The index needs to be zero or positive." && v1371);
                } else {
                }
                int v1374;
                v1374 = v1369 % 512;
                int v1375;
                v1375 = v1369 / 512;
                bool v1376;
                v1376 = v1375 < 32;
                bool v1377;
                v1377 = v1376 == false;
                if (v1377){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1376);
                } else {
                }
                assert("Tensor range check" && 0 <= v1375 && v1375 < 32);
                assert("Tensor range check" && 0 <= v1374 && v1374 < 512);
                int v1379;
                v1379 = 33 * v1374;
                int v1380;
                v1380 = v1375 + v1379;
                float v1381;
                v1381 = v1333[v1380];
                assert("Tensor range check" && 0 <= v1375 && v1375 < 32);
                assert("Tensor range check" && 0 <= v1374 && v1374 < 512);
                int v1382;
                v1382 = v1374 + v1350;
                int v1383;
                v1383 = 8192 * v1375;
                int v1384;
                v1384 = v1383 + v1382;
                v522[v1384] = v1381;
                v1369 += 256 ;
            }
            __syncthreads();
            v1336 += 24 ;
        }
        extern __shared__ unsigned char v1385[];
        float * v1386;
        v1386 = reinterpret_cast<float *>(&v1385[0ull]);
        int v1388;
        v1388 = blockIdx.x;
        int v1389;
        v1389 = v1388;
        while (while_method_6(v1389)){
            bool v1391;
            v1391 = 0 <= v1389;
            bool v1392;
            v1392 = v1391 == false;
            if (v1392){
                assert("The index needs to be zero or positive." && v1391);
            } else {
            }
            int v1394;
            v1394 = v1389 % 1;
            bool v1395;
            v1395 = v1389 < 8;
            bool v1396;
            v1396 = v1395 == false;
            if (v1396){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1395);
            } else {
            }
            assert("Tensor range check" && 0 <= v1389 && v1389 < 8);
            assert("Tensor range check" && 0 <= v1394 && v1394 < 1);
            int v1398;
            v1398 = 32 * v1394;
            int v1399;
            v1399 = 16384 * v1389;
            int v1400;
            v1400 = v1399 + v1398;
            int v1401;
            v1401 = 131072 * v1394;
            int v1402;
            v1402 = 512 * v1389;
            int v1403;
            v1403 = v1402 + v1401;
            int v1404;
            v1404 = threadIdx.x;
            int v1405;
            v1405 = v1404;
            while (while_method_14(v1405)){
                bool v1407;
                v1407 = 0 <= v1405;
                bool v1408;
                v1408 = v1407 == false;
                if (v1408){
                    assert("The index needs to be zero or positive." && v1407);
                } else {
                }
                int v1410;
                v1410 = v1405 % 32;
                int v1411;
                v1411 = v1405 / 32;
                bool v1412;
                v1412 = v1411 < 512;
                bool v1413;
                v1413 = v1412 == false;
                if (v1413){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1412);
                } else {
                }
                assert("Tensor range check" && 0 <= v1411 && v1411 < 512);
                assert("Tensor range check" && 0 <= v1410 && v1410 < 32);
                int v1415;
                v1415 = v1410 + v1400;
                int v1416;
                v1416 = 32 * v1411;
                int v1417;
                v1417 = v1416 + v1415;
                float v1418;
                v1418 = v528[v1417];
                assert("Tensor range check" && 0 <= v1411 && v1411 < 512);
                assert("Tensor range check" && 0 <= v1410 && v1410 < 32);
                int v1419;
                v1419 = 33 * v1411;
                int v1420;
                v1420 = v1419 + v1410;
                v1386[v1420] = v1418;
                v1405 += 256 ;
            }
            __syncthreads();
            int v1421;
            v1421 = threadIdx.x;
            int v1422;
            v1422 = v1421;
            while (while_method_14(v1422)){
                bool v1424;
                v1424 = 0 <= v1422;
                bool v1425;
                v1425 = v1424 == false;
                if (v1425){
                    assert("The index needs to be zero or positive." && v1424);
                } else {
                }
                int v1427;
                v1427 = v1422 % 512;
                int v1428;
                v1428 = v1422 / 512;
                bool v1429;
                v1429 = v1428 < 32;
                bool v1430;
                v1430 = v1429 == false;
                if (v1430){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1429);
                } else {
                }
                assert("Tensor range check" && 0 <= v1428 && v1428 < 32);
                assert("Tensor range check" && 0 <= v1427 && v1427 < 512);
                int v1432;
                v1432 = 33 * v1427;
                int v1433;
                v1433 = v1428 + v1432;
                float v1434;
                v1434 = v1386[v1433];
                assert("Tensor range check" && 0 <= v1428 && v1428 < 32);
                assert("Tensor range check" && 0 <= v1427 && v1427 < 512);
                int v1435;
                v1435 = v1427 + v1403;
                int v1436;
                v1436 = 4096 * v1428;
                int v1437;
                v1437 = v1436 + v1435;
                v526[v1437] = v1434;
                v1422 += 256 ;
            }
            __syncthreads();
            v1389 += 24 ;
        }
        v519.sync() ;
        v27 += 1 ;
    }
    cooperative_groups::grid_group & v1438 = v26.v1;
    cooperative_groups::grid_group & v1439 = v1438;
    int v1440;
    v1440 = threadIdx.x;
    int v1441;
    v1441 = blockIdx.x;
    int v1442;
    v1442 = v1441 * 256;
    int v1443;
    v1443 = v1440 + v1442;
    int v1444;
    v1444 = v1443;
    while (while_method_17(v1444)){
        bool v1446;
        v1446 = 0 <= v1444;
        bool v1447;
        v1447 = v1446 == false;
        if (v1447){
            assert("The index needs to be zero or positive." && v1446);
        } else {
        }
        int v1449;
        v1449 = v1444 % 64;
        int v1450;
        v1450 = v1444 / 64;
        bool v1451;
        v1451 = v1450 < 32;
        bool v1452;
        v1452 = v1451 == false;
        if (v1452){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1451);
        } else {
        }
        assert("Tensor range check" && 0 <= v1450 && v1450 < 32);
        assert("Tensor range check" && 0 <= v1449 && v1449 < 64);
        int v1454;
        v1454 = 4 * v1449;
        int v1455;
        v1455 = 256 * v1450;
        int v1456;
        v1456 = v1455 + v1454;
        assert("Tensor range check" && 0 <= v1450 && v1450 < 32);
        assert("Tensor range check" && 0 <= v1449 && v1449 < 64);
        float v1457[4];
        float v1458[4];
        float v1459[4];
        int4* v1460;
        v1460 = reinterpret_cast<int4*>(v3 + v1456);
        int4* v1461;
        v1461 = reinterpret_cast<int4*>(v1457 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1460) % 16 == 0 && reinterpret_cast<unsigned long long>(v1461) % 16 == 0);
        *v1461 = *v1460;
        int4* v1462;
        v1462 = reinterpret_cast<int4*>(v4 + v1456);
        int4* v1463;
        v1463 = reinterpret_cast<int4*>(v1458 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1462) % 16 == 0 && reinterpret_cast<unsigned long long>(v1463) % 16 == 0);
        *v1463 = *v1462;
        // Pushing the loop unrolling to: 0
        int v1464;
        v1464 = 0;
        #pragma unroll
        while (while_method_8(v1464)){
            assert("Tensor range check" && 0 <= v1464 && v1464 < 4);
            float v1466;
            v1466 = v1457[v1464];
            float v1467;
            v1467 = v1458[v1464];
            bool v1468;
            v1468 = v1467 == 0.0f;
            bool v1469;
            v1469 = v1468 != true;
            float v1471;
            if (v1469){
                float v1470;
                v1470 = v1466 / v1467;
                v1471 = v1470;
            } else {
                v1471 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v1464 && v1464 < 4);
            v1459[v1464] = v1471;
            v1464 += 1 ;
        }
        // Poping the loop unrolling to: 0
        int4* v1472;
        v1472 = reinterpret_cast<int4*>(v1459 + 0);
        int4* v1473;
        v1473 = reinterpret_cast<int4*>(v5 + v1456);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1472) % 16 == 0 && reinterpret_cast<unsigned long long>(v1473) % 16 == 0);
        *v1473 = *v1472;
        v1444 += 6144 ;
    }
    v1439.sync() ;
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
                method54(v11, v2)
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
                method54(v11, v2)
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
                v39 = cp.zeros(8192,dtype=cp.float32) # type: ignore
                v40 = cp.zeros(8192,dtype=cp.float32) # type: ignore
                v41 = cp.empty(8192,dtype=cp.float32)
                v42 = cp.cuda.Device().attributes['MultiProcessorCount']
                v43 = v42 == 24
                del v42
                v44 = v43 == False
                if v44:
                    v45 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v43, v45
                    del v45
                else:
                    pass
                del v43, v44
                v46 = 1
                v47 = raw_module.get_function(f"entry{v46}")
                del v46
                v47.max_dynamic_shared_size_bytes = 98304 
                print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
                v47((24,),(256,),(v8, v9, v10, v39, v40, v41),shared_mem=98304)
                del v39, v40, v47
                v48 = []
                v50 = v41[0:]
                del v41
                v51 = v50.get()
                del v50
                v52 = 0
                while method57(v52):
                    v54 = []
                    v55 = 0
                    while method58(v55):
                        assert 0 <= v52 < 32, 'Tensor range check'
                        assert 0 <= v55 < 256, 'Tensor range check'
                        v57 = 256 * v52
                        v58 = v57 + v55
                        del v57
                        v59 = v51[v58].item()
                        del v58
                        v54.append(v59)
                        del v59
                        v55 += 1 
                    del v55
                    v48.append(v54)
                    del v54
                    v52 += 1 
                del v51, v52
                v60 = US9_0(v48)
                del v48
                v18.append(v60)
                del v60
            case US0_4(): # StartTrainingVsSelf
                v61 = cp.zeros(8192,dtype=cp.float32) # type: ignore
                v62 = cp.zeros(8192,dtype=cp.float32) # type: ignore
                v63 = cp.empty(8192,dtype=cp.float32)
                v64 = cp.cuda.Device().attributes['MultiProcessorCount']
                v65 = v64 == 24
                del v64
                v66 = v65 == False
                if v66:
                    v67 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v65, v67
                    del v67
                else:
                    pass
                del v65, v66
                v68 = 2
                v69 = raw_module.get_function(f"entry{v68}")
                del v68
                v69.max_dynamic_shared_size_bytes = 98304 
                print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
                v69((24,),(256,),(v8, v9, v10, v61, v62, v63),shared_mem=98304)
                del v61, v62, v69
                v70 = []
                v72 = v63[0:]
                del v63
                v73 = v72.get()
                del v72
                v74 = 0
                while method57(v74):
                    v76 = []
                    v77 = 0
                    while method58(v77):
                        assert 0 <= v74 < 32, 'Tensor range check'
                        assert 0 <= v77 < 256, 'Tensor range check'
                        v79 = 256 * v74
                        v80 = v79 + v77
                        del v79
                        v81 = v73[v80].item()
                        del v80
                        v76.append(v81)
                        del v81
                        v77 += 1 
                    del v77
                    v70.append(v76)
                    del v76
                    v74 += 1 
                del v73, v74
                v82 = US9_1(v70)
                del v70
                v18.append(v82)
                del v82
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
        v0 = cp.empty(1572864,dtype=cp.uint8)
        v1 = cp.empty(64487424,dtype=cp.uint8)
        v2 = cp.empty(1573168,dtype=cp.uint8)
        v4 = v1[0:0+4*786432].view(cp.float32)
        del v4
        v6 = v2[0:0+4*262144].view(cp.float32)
        v8 = v0[0:0+4*262144].view(cp.float32)
        del v8
        v10 = v2[1048576:1048576+4*131072].view(cp.float32)
        v12 = v0[1048576:1048576+4*131072].view(cp.float32)
        del v12
        v14 = v1[7864320:7864320+4*12582912].view(cp.float32)
        del v14
        v16 = v2[1572864:1572864+4*1].view(cp.int32)
        v18 = v2[1572880:1572880+1*32].view(cp.bool_)
        v20 = v2[1572912:1572912+4*32].view(cp.float32)
        v22 = v2[1573040:1573040+4*32].view(cp.float32)
        v24 = v1[58195968:58195968+8*393216].view(cp.float64)
        v26 = v1[61341696:61341696+8*393216].view(cp.float64)
        v27 = cp.random.normal(0.0,0.1,262144,dtype=cp.float32) # type: ignore
        cp.copyto(v6[0:0+262144],v27[0:0+262144])
        del v6, v27
        v28 = cp.random.normal(0.0,0.1,131072,dtype=cp.float32) # type: ignore
        cp.copyto(v10[0:0+131072],v28[0:0+131072])
        del v10, v28
        v16[:] = 0
        del v16
        v20[:] = 0
        del v20
        v22[:] = 0
        del v22
        v18[:] = 1
        del v18
        v24[:] = 0
        del v24
        v26[:] = 0
        del v26
        v30 = static_array(2)
        v32 = US2_0()
        v30[0] = v32
        del v32
        v34 = US2_1()
        v30[1] = v34
        del v34
        v36 = static_array_list(32)
        v37 = 63
        v38 = US3_0()
        v39 = US7_0()
        return method115(v37, v38, v36, v30, v39, v2, v1, v0)
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
def method58(v0 : i32) -> bool:
    v1 = v0 < 256
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
