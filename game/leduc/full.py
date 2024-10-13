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
__device__ void block_row_map_24(float * v0, int v1, float * v2);
struct Tuple8;
struct Tuple9;
struct Tuple10;
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
struct Tuple11;
struct Tuple12;
struct Tuple13;
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
                    Tuple7 tmp33 = order_28(v8, v11);
                    v27 = tmp33.v0; v28 = tmp33.v1;
                    int v29; int v30;
                    Tuple7 tmp34 = order_28(v8, v14);
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
__device__ inline bool while_method_10(int v0){
    bool v1;
    v1 = v0 < 256;
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
    while (while_method_11(v20)){
        Union3 v924;
        switch (v20.tag) {
            case 0: { // None
                v924 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v22 = v20.case1.v0;
                Union14 v764;
                switch (v22.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v733 = v22.case0.v0; bool v734 = v22.case0.v1; static_array<Union6,2> v735 = v22.case0.v2; int v736 = v22.case0.v3; static_array<int,2> v737 = v22.case0.v4; int v738 = v22.case0.v5;
                        curandStatePhilox4_32_10_t & v739 = v3.v5;
                        curandStatePhilox4_32_10_t & v740 = v739;
                        unsigned int & v741 = v3.v0;
                        Union6 v742; unsigned int v743;
                        Tuple6 tmp35 = draw_card_20(v740, v741);
                        v742 = tmp35.v0; v743 = tmp35.v1;
                        v3.v0 = v743;
                        Union7 v744;
                        v744 = Union7{Union7_0{v742}};
                        v18.push(v744);
                        v764 = Union14{Union14_0{v733, v734, v735, v736, v737, v738, v742}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v746 = v3.v5;
                        curandStatePhilox4_32_10_t & v747 = v746;
                        unsigned int & v748 = v3.v0;
                        Union6 v749; unsigned int v750;
                        Tuple6 tmp36 = draw_card_20(v747, v748);
                        v749 = tmp36.v0; v750 = tmp36.v1;
                        v3.v0 = v750;
                        curandStatePhilox4_32_10_t & v751 = v3.v5;
                        curandStatePhilox4_32_10_t & v752 = v751;
                        unsigned int & v753 = v3.v0;
                        Union6 v754; unsigned int v755;
                        Tuple6 tmp37 = draw_card_20(v752, v753);
                        v754 = tmp37.v0; v755 = tmp37.v1;
                        v3.v0 = v755;
                        Union7 v756;
                        v756 = Union7{Union7_2{0, v749}};
                        v18.push(v756);
                        Union7 v757;
                        v757 = Union7{Union7_2{1, v754}};
                        v18.push(v757);
                        v764 = Union14{Union14_1{v749, v754}};
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
                        Union1 v721;
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
                                while (while_method_4(v159)){
                                    float * v161;
                                    v161 = reinterpret_cast<float *>(&v1[4718592ull]);
                                    assert("Tensor range check" && 0 <= v159 && v159 < 32);
                                    int v163;
                                    v163 = 393216 * v159;
                                    float * v164;
                                    v164 = reinterpret_cast<float *>(&v1[0ull]);
                                    float * v166;
                                    v166 = reinterpret_cast<float *>(&v0[0ull]);
                                    float * v168;
                                    v168 = reinterpret_cast<float *>(&v2[0ull]);
                                    assert("Tensor range check" && 0 <= v159 && v159 < 32);
                                    int v170;
                                    v170 = 8192 * v159;
                                    float * v171;
                                    v171 = reinterpret_cast<float *>(&v1[3145728ull]);
                                    block_matmul_23(v171, v166, v170, v164);
                                    block_row_map_24(v161, v163, v171);
                                    int * v173;
                                    v173 = reinterpret_cast<int *>(&v0[1048576ull]);
                                    bool * v175;
                                    v175 = reinterpret_cast<bool *>(&v0[1048592ull]);
                                    float * v177;
                                    v177 = reinterpret_cast<float *>(&v0[1048624ull]);
                                    float * v179;
                                    v179 = reinterpret_cast<float *>(&v0[1048752ull]);
                                    double * v181;
                                    v181 = reinterpret_cast<double *>(&v1[55050240ull]);
                                    double * v183;
                                    v183 = reinterpret_cast<double *>(&v1[58195968ull]);
                                    v159 += 1 ;
                                }
                                __syncthreads();
                                int * v185;
                                v185 = reinterpret_cast<int *>(&v0[1048576ull]);
                                bool * v187;
                                v187 = reinterpret_cast<bool *>(&v0[1048592ull]);
                                float * v189;
                                v189 = reinterpret_cast<float *>(&v0[1048624ull]);
                                float * v191;
                                v191 = reinterpret_cast<float *>(&v0[1048752ull]);
                                int v193;
                                v193 = v185[0];
                                float * v194;
                                v194 = reinterpret_cast<float *>(&v1[4718592ull]);
                                assert("Tensor range check" && 0 <= v193 && v193 < 32);
                                int v196;
                                v196 = 393216 * v193;
                                int v197;
                                v197 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v197 && v197 < 24);
                                int v198;
                                v198 = 16384 * v197;
                                int v199;
                                v199 = v198 + v196;
                                int v200;
                                v200 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v200 && v200 < 256);
                                int v201;
                                v201 = 64 * v200;
                                int v202;
                                v202 = v201 + v199;
                                float * v203;
                                v203 = v194+v202;
                                int v205;
                                v205 = sizeof(float *);
                                unsigned long long v206;
                                v206 = (unsigned long long)v205;
                                unsigned long long v207;
                                v207 = 256ull * v206;
                                unsigned long long v208;
                                v208 = v207 + 16ull;
                                unsigned long long v209;
                                v209 = v208 - 1ull;
                                unsigned long long v210;
                                v210 = v209 % 16ull;
                                unsigned long long v211;
                                v211 = v209 - v210;
                                unsigned long long v212;
                                v212 = v211 + 1024ull;
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
                                bool v218;
                                v218 = v217 <= 98304ull;
                                bool v219;
                                v219 = v218 == false;
                                if (v219){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v218);
                                } else {
                                }
                                extern __shared__ unsigned char v221[];
                                bool v222;
                                v222 = v217 <= v217;
                                bool v223;
                                v223 = v222 == false;
                                if (v223){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v222);
                                } else {
                                }
                                float * * v225;
                                v225 = reinterpret_cast<float * *>(&v221[0ull]);
                                float * v227;
                                v227 = reinterpret_cast<float *>(&v221[v211]);
                                int * v229;
                                v229 = reinterpret_cast<int *>(&v221[v216]);
                                int v231;
                                v231 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v231 && v231 < 256);
                                v225[v231] = v203;
                                __syncthreads();
                                bool v232;
                                v232 = 0 <= v231;
                                bool v233;
                                v233 = v232 == false;
                                if (v233){
                                    assert("The index needs to be zero or positive." && v232);
                                } else {
                                }
                                int v235;
                                v235 = v231 % 16;
                                int v236;
                                v236 = v231 / 16;
                                bool v237;
                                v237 = v236 < 16;
                                bool v238;
                                v238 = v237 == false;
                                if (v238){
                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v237);
                                } else {
                                }
                                assert("Tensor range check" && 0 <= v236 && v236 < 16);
                                int v240;
                                v240 = 0;
                                while (while_method_9(v240)){
                                    bool v242;
                                    v242 = 0 <= v236;
                                    bool v243;
                                    v243 = v242 && v237;
                                    bool v244;
                                    v244 = v243 == false;
                                    if (v244){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v243);
                                    } else {
                                    }
                                    bool v246;
                                    v246 = 0 <= v240;
                                    bool v248;
                                    if (v246){
                                        bool v247;
                                        v247 = v240 < 16;
                                        v248 = v247;
                                    } else {
                                        v248 = false;
                                    }
                                    bool v249;
                                    v249 = v248 == false;
                                    if (v249){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v248);
                                    } else {
                                    }
                                    int v251;
                                    v251 = v240 * 16;
                                    int v252;
                                    v252 = v251 + v236;
                                    assert("Tensor range check" && 0 <= v240 && v240 < 16);
                                    int v253;
                                    v253 = 16 * v240;
                                    int v254;
                                    v254 = v253 + v236;
                                    float * v255;
                                    v255 = v225[v254];
                                    int v256;
                                    v256 = blockIdx.x;
                                    int v257;
                                    v257 = v256 * 256;
                                    int v258;
                                    v258 = v257 + v252;
                                    assert("Tensor range check" && 0 <= v235 && v235 < 16);
                                    int v259;
                                    v259 = 4 * v235;
                                    float v260[4];
                                    int v261[4];
                                    int v262;
                                    v262 = 0;
                                    while (while_method_5(v262)){
                                        assert("Tensor range check" && 0 <= v262 && v262 < 1);
                                        int v264;
                                        v264 = 4 * v262;
                                        assert("Tensor range check" && 0 <= v262 && v262 < 1);
                                        int v265;
                                        v265 = 64 * v262;
                                        int v266;
                                        v266 = v265 + v259;
                                        int4* v267;
                                        v267 = reinterpret_cast<int4*>(v255 + v266);
                                        int4* v268;
                                        v268 = reinterpret_cast<int4*>(v260 + v264);
                                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v267) % 16 == 0 && reinterpret_cast<unsigned long long>(v268) % 16 == 0);
                                        *v268 = *v267;
                                        v262 += 1 ;
                                    }
                                    int v269;
                                    v269 = 0;
                                    while (while_method_5(v269)){
                                        int v271;
                                        v271 = 0;
                                        while (while_method_8(v271)){
                                            bool v273;
                                            v273 = 0 <= v271;
                                            bool v275;
                                            if (v273){
                                                bool v274;
                                                v274 = v271 < 4;
                                                v275 = v274;
                                            } else {
                                                v275 = false;
                                            }
                                            bool v276;
                                            v276 = v275 == false;
                                            if (v276){
                                                assert("The indices should be inside the range of the dimension." && v275);
                                            } else {
                                            }
                                            bool v278;
                                            v278 = 0 <= v235;
                                            bool v280;
                                            if (v278){
                                                bool v279;
                                                v279 = v235 < 16;
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
                                            int v283;
                                            v283 = v235 * 4;
                                            int v284;
                                            v284 = v271 + v283;
                                            bool v285;
                                            v285 = 0 <= v269;
                                            bool v287;
                                            if (v285){
                                                bool v286;
                                                v286 = v269 < 1;
                                                v287 = v286;
                                            } else {
                                                v287 = false;
                                            }
                                            bool v288;
                                            v288 = v287 == false;
                                            if (v288){
                                                assert("The indices should be inside the range of the dimension." && v287);
                                            } else {
                                            }
                                            int v290;
                                            v290 = v269 * 64;
                                            int v291;
                                            v291 = v284 + v290;
                                            assert("Tensor range check" && 0 <= v269 && v269 < 1);
                                            assert("Tensor range check" && 0 <= v271 && v271 < 4);
                                            int v292;
                                            v292 = 4 * v269;
                                            int v293;
                                            v293 = v292 + v271;
                                            v261[v293] = v291;
                                            v271 += 1 ;
                                        }
                                        v269 += 1 ;
                                    }
                                    float v294[4];
                                    float v295;
                                    v295 = 0.0f;
                                    int v296;
                                    v296 = 0;
                                    while (while_method_5(v296)){
                                        assert("Tensor range check" && 0 <= v296 && v296 < 1);
                                        int v298;
                                        v298 = 4 * v296;
                                        assert("Tensor range check" && 0 <= v296 && v296 < 1);
                                        float v299;
                                        v299 = 0.0f;
                                        int v300;
                                        v300 = 0;
                                        while (while_method_8(v300)){
                                            assert("Tensor range check" && 0 <= v300 && v300 < 4);
                                            int v302;
                                            v302 = v300 + v298;
                                            float v303;
                                            v303 = v260[v302];
                                            float v304;
                                            v304 = v299 + v303;
                                            v299 = v304;
                                            v300 += 1 ;
                                        }
                                        auto v305 = cooperative_groups::coalesced_threads();
                                        int v306;
                                        v306 = threadIdx.x;
                                        int v307;
                                        v307 = v306 / 16;
                                        auto v308 = cooperative_groups::labeled_partition(v305,v307);
                                        Closure2 v309{};
                                        float v310;
                                        v310 = cooperative_groups::inclusive_scan(v308, v299, v309);
                                        float v311;
                                        v311 = v308.shfl_up(v310,1);
                                        bool v312;
                                        v312 = v308.thread_rank() == 0;
                                        float v313;
                                        if (v312){
                                            v313 = 0.0f;
                                        } else {
                                            v313 = v311;
                                        }
                                        float v314;
                                        v314 = v308.shfl(v310,v308.num_threads()-1);
                                        float v315;
                                        v315 = v295 + v313;
                                        float v316;
                                        v316 = v315;
                                        int v317;
                                        v317 = 0;
                                        while (while_method_8(v317)){
                                            assert("Tensor range check" && 0 <= v317 && v317 < 4);
                                            int v319;
                                            v319 = v317 + v298;
                                            float v320;
                                            v320 = v260[v319];
                                            float v321;
                                            v321 = v316 + v320;
                                            assert("Tensor range check" && 0 <= v317 && v317 < 4);
                                            v294[v319] = v321;
                                            v316 = v321;
                                            v317 += 1 ;
                                        }
                                        float v322;
                                        v322 = v295 + v314;
                                        v295 = v322;
                                        v296 += 1 ;
                                    }
                                    float v323[4];
                                    bool v324[4];
                                    int v325;
                                    v325 = 0;
                                    while (while_method_5(v325)){
                                        int v327;
                                        v327 = 0;
                                        while (while_method_8(v327)){
                                            assert("Tensor range check" && 0 <= v325 && v325 < 1);
                                            assert("Tensor range check" && 0 <= v327 && v327 < 4);
                                            int v329;
                                            v329 = 4 * v325;
                                            int v330;
                                            v330 = v329 + v327;
                                            float v331;
                                            v331 = v294[v330];
                                            float v332;
                                            v332 = v260[v330];
                                            bool v333;
                                            v333 = v332 > 0.0f;
                                            assert("Tensor range check" && 0 <= v325 && v325 < 1);
                                            assert("Tensor range check" && 0 <= v327 && v327 < 4);
                                            v323[v330] = v331;
                                            v324[v330] = v333;
                                            v327 += 1 ;
                                        }
                                        v325 += 1 ;
                                    }
                                    float v334; bool v335;
                                    Tuple8 tmp38 = Tuple8{-1.0f / 0.0f, false};
                                    v334 = tmp38.v0; v335 = tmp38.v1;
                                    int v336;
                                    v336 = 0;
                                    while (while_method_5(v336)){
                                        int v338;
                                        v338 = 0;
                                        while (while_method_8(v338)){
                                            assert("Tensor range check" && 0 <= v336 && v336 < 1);
                                            assert("Tensor range check" && 0 <= v338 && v338 < 4);
                                            int v340;
                                            v340 = 4 * v336;
                                            int v341;
                                            v341 = v340 + v338;
                                            float v342;
                                            v342 = v323[v341];
                                            bool v343;
                                            v343 = v324[v341];
                                            float v350; bool v351;
                                            if (v335){
                                                if (v343){
                                                    bool v344;
                                                    v344 = v334 >= v342;
                                                    float v345;
                                                    if (v344){
                                                        v345 = v334;
                                                    } else {
                                                        v345 = v342;
                                                    }
                                                    v350 = v345; v351 = true;
                                                } else {
                                                    v350 = v334; v351 = v335;
                                                }
                                            } else {
                                                if (v343){
                                                    v350 = v342; v351 = v343;
                                                } else {
                                                    v350 = v334; v351 = v335;
                                                }
                                            }
                                            v334 = v350;
                                            v335 = v351;
                                            v338 += 1 ;
                                        }
                                        v336 += 1 ;
                                    }
                                    auto v352 = cooperative_groups::coalesced_threads();
                                    int v353;
                                    v353 = threadIdx.x;
                                    int v354;
                                    v354 = v353 / 16;
                                    auto v355 = cooperative_groups::labeled_partition(v352,v354);
                                    Closure3 v356{};
                                    float v357; bool v358;
                                    Tuple8 tmp39 = cooperative_groups::reduce(v355, Tuple8{v334, v335}, v356);
                                    v357 = tmp39.v0; v358 = tmp39.v1;
                                    bool v359;
                                    v359 = v358 == false;
                                    if (v359){
                                        int v360;
                                        v360 = threadIdx.x;
                                        int v361;
                                        v361 = blockIdx.x;
                                        int v362;
                                        v362 = v361 * 256;
                                        int v363;
                                        v363 = v360 + v362;
                                        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v364 = console_lock;
                                        auto v365 = cooperative_groups::coalesced_threads();
                                        v364.acquire();
                                        int v366;
                                        v366 = 0;
                                        printf("{%s = %d; %s = %c","tid", v363, "x'", '[');
                                        int v367;
                                        v367 = 0;
                                        while (while_method_5(v367)){
                                            int v369;
                                            v369 = v366;
                                            bool v370;
                                            v370 = v369 >= 100;
                                            if (v370){
                                                printf("%s"," ...");
                                                break;
                                            } else {
                                            }
                                            bool v371;
                                            v371 = v367 == 0;
                                            bool v372;
                                            v372 = v371 != true;
                                            if (v372){
                                                printf("%s","; ");
                                            } else {
                                            }
                                            printf("%c",'[');
                                            int v373;
                                            v373 = 0;
                                            while (while_method_8(v373)){
                                                int v375;
                                                v375 = v366;
                                                bool v376;
                                                v376 = v375 >= 100;
                                                if (v376){
                                                    printf("%s"," ...");
                                                    break;
                                                } else {
                                                }
                                                bool v377;
                                                v377 = v373 == 0;
                                                bool v378;
                                                v378 = v377 != true;
                                                if (v378){
                                                    printf("%s","; ");
                                                } else {
                                                }
                                                int v379;
                                                v379 = v366 + 1;
                                                v366 = v379;
                                                int v380;
                                                v380 = v367 * 4;
                                                int v381;
                                                v381 = v380 + v373;
                                                float v382;
                                                v382 = v323[v381];
                                                bool v383;
                                                v383 = v324[v381];
                                                const char * v386;
                                                if (v383){
                                                    const char * v384;
                                                    v384 = "true";
                                                    v386 = v384;
                                                } else {
                                                    const char * v385;
                                                    v385 = "false";
                                                    v386 = v385;
                                                }
                                                printf("%f, %s",v382, v386);
                                                v373 += 1 ;
                                            }
                                            printf("%c",']');
                                            v367 += 1 ;
                                        }
                                        printf("%c",']');
                                        printf("}\n");
                                        v364.release();
                                        v365.sync() ;
                                    } else {
                                    }
                                    if (v359){
                                        assert("The local reduce must be true." && v358);
                                    } else {
                                    }
                                    float v422[4];
                                    int v423[4];
                                    int v424;
                                    v424 = 0;
                                    while (while_method_5(v424)){
                                        int v426;
                                        v426 = 0;
                                        while (while_method_8(v426)){
                                            assert("Tensor range check" && 0 <= v424 && v424 < 1);
                                            assert("Tensor range check" && 0 <= v426 && v426 < 4);
                                            int v428;
                                            v428 = 4 * v424;
                                            int v429;
                                            v429 = v428 + v426;
                                            int v430;
                                            v430 = v261[v429];
                                            float v431;
                                            v431 = curand_uniform(&v89);
                                            assert("Tensor range check" && 0 <= v424 && v424 < 1);
                                            assert("Tensor range check" && 0 <= v426 && v426 < 4);
                                            v422[v429] = v431;
                                            v423[v429] = v430;
                                            v426 += 1 ;
                                        }
                                        v424 += 1 ;
                                    }
                                    float v432; int v433;
                                    Tuple9 tmp40 = Tuple9{0.0f, 2147483647};
                                    v432 = tmp40.v0; v433 = tmp40.v1;
                                    int v434;
                                    v434 = 0;
                                    while (while_method_5(v434)){
                                        int v436;
                                        v436 = 0;
                                        while (while_method_8(v436)){
                                            assert("Tensor range check" && 0 <= v434 && v434 < 1);
                                            assert("Tensor range check" && 0 <= v436 && v436 < 4);
                                            int v438;
                                            v438 = 4 * v434;
                                            int v439;
                                            v439 = v438 + v436;
                                            float v440;
                                            v440 = v422[v439];
                                            int v441;
                                            v441 = v423[v439];
                                            bool v442;
                                            v442 = v433 < v441;
                                            float v443; int v444;
                                            if (v442){
                                                v443 = v432; v444 = v433;
                                            } else {
                                                v443 = v440; v444 = v441;
                                            }
                                            v432 = v443;
                                            v433 = v444;
                                            v436 += 1 ;
                                        }
                                        v434 += 1 ;
                                    }
                                    auto v445 = cooperative_groups::coalesced_threads();
                                    int v446;
                                    v446 = threadIdx.x;
                                    int v447;
                                    v447 = v446 / 16;
                                    auto v448 = cooperative_groups::labeled_partition(v445,v447);
                                    Closure4 v449{};
                                    float v450; int v451;
                                    Tuple9 tmp41 = cooperative_groups::reduce(v448, Tuple9{v432, v433}, v449);
                                    v450 = tmp41.v0; v451 = tmp41.v1;
                                    float v452;
                                    v452 = v357 * v450;
                                    int v453[4];
                                    bool v454[4];
                                    int v455;
                                    v455 = 0;
                                    while (while_method_5(v455)){
                                        int v457;
                                        v457 = 0;
                                        while (while_method_8(v457)){
                                            assert("Tensor range check" && 0 <= v455 && v455 < 1);
                                            assert("Tensor range check" && 0 <= v457 && v457 < 4);
                                            int v459;
                                            v459 = 4 * v455;
                                            int v460;
                                            v460 = v459 + v457;
                                            float v461;
                                            v461 = v323[v460];
                                            bool v462;
                                            v462 = v324[v460];
                                            int v463;
                                            v463 = v261[v460];
                                            int v466; bool v467;
                                            if (v462){
                                                float v464;
                                                v464 = v461 - v452;
                                                bool v465;
                                                v465 = v464 >= 0.0f;
                                                v466 = v463; v467 = v465;
                                            } else {
                                                v466 = 2147483647; v467 = false;
                                            }
                                            assert("Tensor range check" && 0 <= v455 && v455 < 1);
                                            assert("Tensor range check" && 0 <= v457 && v457 < 4);
                                            v453[v460] = v466;
                                            v454[v460] = v467;
                                            v457 += 1 ;
                                        }
                                        v455 += 1 ;
                                    }
                                    int v468; bool v469;
                                    Tuple10 tmp42 = Tuple10{2147483647, false};
                                    v468 = tmp42.v0; v469 = tmp42.v1;
                                    int v470;
                                    v470 = 0;
                                    while (while_method_5(v470)){
                                        int v472;
                                        v472 = 0;
                                        while (while_method_8(v472)){
                                            assert("Tensor range check" && 0 <= v470 && v470 < 1);
                                            assert("Tensor range check" && 0 <= v472 && v472 < 4);
                                            int v474;
                                            v474 = 4 * v470;
                                            int v475;
                                            v475 = v474 + v472;
                                            int v476;
                                            v476 = v453[v475];
                                            bool v477;
                                            v477 = v454[v475];
                                            int v484; bool v485;
                                            if (v469){
                                                if (v477){
                                                    bool v478;
                                                    v478 = v468 < v476;
                                                    int v479;
                                                    if (v478){
                                                        v479 = v468;
                                                    } else {
                                                        v479 = v476;
                                                    }
                                                    v484 = v479; v485 = true;
                                                } else {
                                                    v484 = v468; v485 = v469;
                                                }
                                            } else {
                                                if (v477){
                                                    v484 = v476; v485 = v477;
                                                } else {
                                                    v484 = v468; v485 = v469;
                                                }
                                            }
                                            v468 = v484;
                                            v469 = v485;
                                            v472 += 1 ;
                                        }
                                        v470 += 1 ;
                                    }
                                    auto v486 = cooperative_groups::coalesced_threads();
                                    int v487;
                                    v487 = threadIdx.x;
                                    int v488;
                                    v488 = v487 / 16;
                                    auto v489 = cooperative_groups::labeled_partition(v486,v488);
                                    Closure5 v490{};
                                    int v491; bool v492;
                                    Tuple10 tmp43 = cooperative_groups::reduce(v489, Tuple10{v468, v469}, v490);
                                    v491 = tmp43.v0; v492 = tmp43.v1;
                                    bool v493;
                                    v493 = v492 == false;
                                    if (v493){
                                        int v494;
                                        v494 = threadIdx.x;
                                        int v495;
                                        v495 = blockIdx.x;
                                        int v496;
                                        v496 = v495 * 256;
                                        int v497;
                                        v497 = v494 + v496;
                                        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v498 = console_lock;
                                        auto v499 = cooperative_groups::coalesced_threads();
                                        v498.acquire();
                                        int v500;
                                        v500 = 0;
                                        printf("{%s = %d; %s = %c","tid", v497, "x'", '[');
                                        int v501;
                                        v501 = 0;
                                        while (while_method_5(v501)){
                                            int v503;
                                            v503 = v500;
                                            bool v504;
                                            v504 = v503 >= 100;
                                            if (v504){
                                                printf("%s"," ...");
                                                break;
                                            } else {
                                            }
                                            bool v505;
                                            v505 = v501 == 0;
                                            bool v506;
                                            v506 = v505 != true;
                                            if (v506){
                                                printf("%s","; ");
                                            } else {
                                            }
                                            printf("%c",'[');
                                            int v507;
                                            v507 = 0;
                                            while (while_method_8(v507)){
                                                int v509;
                                                v509 = v500;
                                                bool v510;
                                                v510 = v509 >= 100;
                                                if (v510){
                                                    printf("%s"," ...");
                                                    break;
                                                } else {
                                                }
                                                bool v511;
                                                v511 = v507 == 0;
                                                bool v512;
                                                v512 = v511 != true;
                                                if (v512){
                                                    printf("%s","; ");
                                                } else {
                                                }
                                                int v513;
                                                v513 = v500 + 1;
                                                v500 = v513;
                                                int v514;
                                                v514 = v501 * 4;
                                                int v515;
                                                v515 = v514 + v507;
                                                int v516;
                                                v516 = v453[v515];
                                                bool v517;
                                                v517 = v454[v515];
                                                const char * v520;
                                                if (v517){
                                                    const char * v518;
                                                    v518 = "true";
                                                    v520 = v518;
                                                } else {
                                                    const char * v519;
                                                    v519 = "false";
                                                    v520 = v519;
                                                }
                                                printf("%d, %s",v516, v520);
                                                v507 += 1 ;
                                            }
                                            printf("%c",']');
                                            v501 += 1 ;
                                        }
                                        printf("%c",']');
                                        printf("}\n");
                                        v498.release();
                                        v499.sync() ;
                                    } else {
                                    }
                                    if (v493){
                                        assert("The local reduce must be true." && v492);
                                    } else {
                                    }
                                    float v556; int v557;
                                    Tuple9 tmp44 = Tuple9{0.0f, 2147483647};
                                    v556 = tmp44.v0; v557 = tmp44.v1;
                                    int v558;
                                    v558 = 0;
                                    while (while_method_5(v558)){
                                        int v560;
                                        v560 = 0;
                                        while (while_method_8(v560)){
                                            assert("Tensor range check" && 0 <= v558 && v558 < 1);
                                            assert("Tensor range check" && 0 <= v560 && v560 < 4);
                                            int v562;
                                            v562 = 4 * v558;
                                            int v563;
                                            v563 = v562 + v560;
                                            float v564;
                                            v564 = v260[v563];
                                            int v565;
                                            v565 = v261[v563];
                                            bool v566;
                                            v566 = v557 == v491;
                                            float v570; int v571;
                                            if (v566){
                                                v570 = v556; v571 = v557;
                                            } else {
                                                bool v567;
                                                v567 = v565 == v491;
                                                if (v567){
                                                    v570 = v564; v571 = v565;
                                                } else {
                                                    v570 = v556; v571 = v557;
                                                }
                                            }
                                            v556 = v570;
                                            v557 = v571;
                                            v560 += 1 ;
                                        }
                                        v558 += 1 ;
                                    }
                                    auto v572 = cooperative_groups::coalesced_threads();
                                    int v573;
                                    v573 = threadIdx.x;
                                    int v574;
                                    v574 = v573 / 16;
                                    auto v575 = cooperative_groups::labeled_partition(v572,v574);
                                    Closure6 v576{v491};
                                    float v577; int v578;
                                    Tuple9 tmp45 = cooperative_groups::reduce(v575, Tuple9{v556, v557}, v576);
                                    v577 = tmp45.v0; v578 = tmp45.v1;
                                    bool v579;
                                    v579 = v578 == 2147483647;
                                    bool v580;
                                    v580 = v579 != true;
                                    bool v581;
                                    v581 = v580 == false;
                                    if (v581){
                                        assert("Expected a valid action id in get_prob." && v580);
                                    } else {
                                    }
                                    int v583;
                                    v583 = 0;
                                    while (while_method_5(v583)){
                                        assert("Tensor range check" && 0 <= v583 && v583 < 1);
                                        assert("Tensor range check" && 0 <= v583 && v583 < 1);
                                        v583 += 1 ;
                                    }
                                    assert("Tensor range check" && 0 <= v252 && v252 < 256);
                                    v227[v252] = v577;
                                    v229[v252] = v491;
                                    v240 += 1 ;
                                }
                                __syncthreads();
                                assert("Tensor range check" && 0 <= v231 && v231 < 256);
                                float v585;
                                v585 = v227[v231];
                                int v586;
                                v586 = v229[v231];
                                __syncthreads();
                                extern __shared__ unsigned char v587[];
                                float * v588;
                                v588 = reinterpret_cast<float *>(&v587[0ull]);
                                int * v590;
                                v590 = reinterpret_cast<int *>(&v587[16ull]);
                                int v592;
                                v592 = threadIdx.x;
                                bool v593;
                                v593 = v592 == 0;
                                if (v593){
                                    v588[0] = v585;
                                    v590[0] = v586;
                                } else {
                                }
                                __syncthreads();
                                float v594;
                                v594 = v588[0];
                                int v595;
                                v595 = v590[0];
                                __syncthreads();
                                double * v596;
                                v596 = reinterpret_cast<double *>(&v1[55050240ull]);
                                double * v598;
                                v598 = reinterpret_cast<double *>(&v1[58195968ull]);
                                int v600;
                                v600 = threadIdx.x;
                                int v601;
                                v601 = blockIdx.x;
                                int v602;
                                v602 = v601 * 256;
                                int v603;
                                v603 = v600 + v602;
                                int v604;
                                v604 = 0;
                                while (while_method_4(v604)){
                                    float * v606;
                                    v606 = reinterpret_cast<float *>(&v1[4718592ull]);
                                    int v608;
                                    v608 = blockIdx.x;
                                    int v609;
                                    v609 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v604 && v604 < 32);
                                    assert("Tensor range check" && 0 <= v608 && v608 < 24);
                                    assert("Tensor range check" && 0 <= v609 && v609 < 256);
                                    assert("Tensor range check" && 0 <= v595 && v595 < 64);
                                    int v610;
                                    v610 = 64 * v609;
                                    int v611;
                                    v611 = v610 + v595;
                                    int v612;
                                    v612 = 16384 * v608;
                                    int v613;
                                    v613 = v612 + v611;
                                    int v614;
                                    v614 = 393216 * v604;
                                    int v615;
                                    v615 = v614 + v613;
                                    float v616;
                                    v616 = v606[v615];
                                    double v617;
                                    v617 = (double)v594;
                                    double v618;
                                    v618 = log(v617);
                                    double v619;
                                    v619 = (double)v616;
                                    double v620;
                                    v620 = log(v619);
                                    assert("Tensor range check" && 0 <= v604 && v604 < 32);
                                    assert("Tensor range check" && 0 <= v603 && v603 < 6144);
                                    assert("Tensor range check" && 0 <= v75 && v75 < 2);
                                    int v621;
                                    v621 = 2 * v603;
                                    int v622;
                                    v622 = v621 + v75;
                                    int v623;
                                    v623 = 12288 * v604;
                                    int v624;
                                    v624 = v623 + v622;
                                    double v625;
                                    v625 = v596[v624];
                                    double v626;
                                    v626 = v598[v624];
                                    double v627;
                                    v627 = v620 + v625;
                                    double v628;
                                    v628 = v618 + v626;
                                    bool v629;
                                    v629 = isnan(v628);
                                    bool v630;
                                    v630 = v629 == false;
                                    bool v631;
                                    v631 = v630 == false;
                                    if (v631){
                                        assert("The sampling log probability shouldn't be nan." && v630);
                                    } else {
                                    }
                                    bool v633;
                                    v633 = isnan(v627);
                                    bool v634;
                                    v634 = v633 == false;
                                    bool v635;
                                    v635 = v634 == false;
                                    if (v635){
                                        assert("The policy log probability shouldn't be nan." && v634);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v604 && v604 < 32);
                                    assert("Tensor range check" && 0 <= v603 && v603 < 6144);
                                    assert("Tensor range check" && 0 <= v75 && v75 < 2);
                                    v596[v624] = v627;
                                    v598[v624] = v628;
                                    v604 += 1 ;
                                }
                                bool v637;
                                v637 = 0 == v595;
                                Union12 v646;
                                if (v637){
                                    v646 = Union12{Union12_1{}};
                                } else {
                                    bool v639;
                                    v639 = 1 == v595;
                                    if (v639){
                                        v646 = Union12{Union12_0{}};
                                    } else {
                                        bool v641;
                                        v641 = 2 == v595;
                                        if (v641){
                                            v646 = Union12{Union12_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v646.tag) {
                                    case 0: { // AA_Call
                                        v721 = Union1{Union1_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v647;
                                        v647 = v76[0];
                                        int v649; int v650;
                                        Tuple7 tmp46 = Tuple7{1, v647};
                                        v649 = tmp46.v0; v650 = tmp46.v1;
                                        while (while_method_0(v649)){
                                            bool v652;
                                            v652 = 0 <= v649;
                                            bool v654;
                                            if (v652){
                                                bool v653;
                                                v653 = v649 < 2;
                                                v654 = v653;
                                            } else {
                                                v654 = false;
                                            }
                                            bool v655;
                                            v655 = v654 == false;
                                            if (v655){
                                                assert("Index must be in range." && v654);
                                            } else {
                                            }
                                            int v657;
                                            v657 = v76[v649];
                                            bool v659;
                                            v659 = v650 >= v657;
                                            int v660;
                                            if (v659){
                                                v660 = v650;
                                            } else {
                                                v660 = v657;
                                            }
                                            v650 = v660;
                                            v649 += 1 ;
                                        }
                                        bool v662;
                                        if (v79){
                                            bool v661;
                                            v661 = v75 < 2;
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
                                        v665 = v76[v75];
                                        bool v667;
                                        v667 = v665 == v650;
                                        if (v667){
                                            v721 = Union1{Union1_0{}};
                                        } else {
                                            v721 = Union1{Union1_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v672;
                                        v672 = v77 > 0;
                                        if (v672){
                                            v721 = Union1{Union1_2{}};
                                        } else {
                                            v721 = Union1{Union1_0{}};
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
                                curandStatePhilox4_32_10_t & v679 = v3.v5;
                                curandStatePhilox4_32_10_t & v680 = v679;
                                static_array_list<Union1,3> v681;
                                v681 = static_array_list<Union1,3>{};
                                v681.unsafe_set_length(1);
                                Union1 v683;
                                v683 = Union1{Union1_0{}};
                                v681[0] = v683;
                                int v685;
                                v685 = v76[0];
                                int v687;
                                v687 = v76[1];
                                bool v689;
                                v689 = v685 == v687;
                                bool v690;
                                v690 = v689 != true;
                                if (v690){
                                    Union1 v691;
                                    v691 = Union1{Union1_1{}};
                                    v681.push(v691);
                                } else {
                                }
                                bool v692;
                                v692 = v77 > 0;
                                if (v692){
                                    Union1 v693;
                                    v693 = Union1{Union1_2{}};
                                    v681.push(v693);
                                } else {
                                }
                                int v694;
                                v694 = v681.length;
                                int v695;
                                v695 = v694 - 1;
                                int v696;
                                v696 = 0;
                                while (while_method_1(v695, v696)){
                                    int v698;
                                    v698 = v681.length;
                                    int v699;
                                    v699 = int_range_22(v698, v696, v680);
                                    Union1 v700;
                                    v700 = v681[v696];
                                    Union1 v702;
                                    v702 = v681[v699];
                                    v681[v696] = v702;
                                    v681[v699] = v700;
                                    v696 += 1 ;
                                }
                                Union1 v704;
                                v704 = v681.pop();
                                int v705;
                                v705 = sizeof(Union1);
                                unsigned long long v706;
                                v706 = (unsigned long long)v705;
                                bool v707;
                                v707 = v706 <= 98304ull;
                                bool v708;
                                v708 = v707 == false;
                                if (v708){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v707);
                                } else {
                                }
                                extern __shared__ unsigned char v710[];
                                bool v711;
                                v711 = v706 <= v706;
                                bool v712;
                                v712 = v711 == false;
                                if (v712){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v711);
                                } else {
                                }
                                Union1 * v714;
                                v714 = reinterpret_cast<Union1 *>(&v710[0ull]);
                                int v716;
                                v716 = threadIdx.x;
                                bool v717;
                                v717 = v716 == 0;
                                if (v717){
                                    v714[0] = v704;
                                } else {
                                }
                                __syncthreads();
                                Union1 v718;
                                v718 = v714[0];
                                __syncthreads();
                                v721 = v718;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union7 v722;
                        v722 = Union7{Union7_1{v75, v721}};
                        v18.push(v722);
                        v764 = Union14{Union14_2{v72, v73, v74, v75, v76, v77, v721}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v724 = v22.case3.v0; bool v725 = v22.case3.v1; static_array<Union6,2> v726 = v22.case3.v2; int v727 = v22.case3.v3; static_array<int,2> v728 = v22.case3.v4; int v729 = v22.case3.v5; Union1 v730 = v22.case3.v6;
                        Union7 v731;
                        v731 = Union7{Union7_1{v727, v730}};
                        v18.push(v731);
                        v764 = Union14{Union14_2{v724, v725, v726, v727, v728, v729, v730}};
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
                        v764 = Union14{Union14_3{}};
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
                        v764 = Union14{Union14_3{}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                switch (v764.tag) {
                    case 0: { // T_game_chance_community_card
                        Union5 v766 = v764.case0.v0; bool v767 = v764.case0.v1; static_array<Union6,2> v768 = v764.case0.v2; int v769 = v764.case0.v3; static_array<int,2> v770 = v764.case0.v4; int v771 = v764.case0.v5; Union6 v772 = v764.case0.v6;
                        int v773;
                        v773 = 2;
                        int v774; int v775;
                        Tuple7 tmp47 = Tuple7{0, 0};
                        v774 = tmp47.v0; v775 = tmp47.v1;
                        while (while_method_0(v774)){
                            bool v777;
                            v777 = 0 <= v774;
                            bool v779;
                            if (v777){
                                bool v778;
                                v778 = v774 < 2;
                                v779 = v778;
                            } else {
                                v779 = false;
                            }
                            bool v780;
                            v780 = v779 == false;
                            if (v780){
                                assert("Index must be in range." && v779);
                            } else {
                            }
                            int v782;
                            v782 = v770[v774];
                            bool v784;
                            v784 = v775 >= v782;
                            int v785;
                            if (v784){
                                v785 = v775;
                            } else {
                                v785 = v782;
                            }
                            v775 = v785;
                            v774 += 1 ;
                        }
                        static_array<int,2> v786;
                        int v788;
                        v788 = 0;
                        while (while_method_0(v788)){
                            v786[v788] = v775;
                            v788 += 1 ;
                        }
                        Union5 v790;
                        v790 = Union5{Union5_1{v772}};
                        Union4 v791;
                        v791 = Union4{Union4_2{v790, true, v768, 0, v786, v773}};
                        v924 = Union3{Union3_1{v791}};
                        break;
                    }
                    case 1: { // T_game_chance_init
                        Union6 v793 = v764.case1.v0; Union6 v794 = v764.case1.v1;
                        int v795;
                        v795 = 2;
                        static_array<int,2> v796;
                        v796[0] = 1;
                        v796[1] = 1;
                        static_array<Union6,2> v798;
                        v798[0] = v793;
                        v798[1] = v794;
                        Union5 v800;
                        v800 = Union5{Union5_0{}};
                        Union4 v801;
                        v801 = Union4{Union4_2{v800, true, v798, 0, v796, v795}};
                        v924 = Union3{Union3_1{v801}};
                        break;
                    }
                    case 2: { // T_game_round
                        Union5 v803 = v764.case2.v0; bool v804 = v764.case2.v1; static_array<Union6,2> v805 = v764.case2.v2; int v806 = v764.case2.v3; static_array<int,2> v807 = v764.case2.v4; int v808 = v764.case2.v5; Union1 v809 = v764.case2.v6;
                        Union4 v916;
                        switch (v803.tag) {
                            case 0: { // None
                                switch (v809.tag) {
                                    case 0: { // Call
                                        if (v804){
                                            int v872;
                                            v872 = v806 ^ 1;
                                            v916 = Union4{Union4_2{v803, false, v805, v872, v807, v808}};
                                        } else {
                                            v916 = Union4{Union4_0{v803, v804, v805, v806, v807, v808}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v916 = Union4{Union4_5{v803, v804, v805, v806, v807, v808}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v876;
                                        v876 = v808 > 0;
                                        if (v876){
                                            int v877;
                                            v877 = v806 ^ 1;
                                            int v878;
                                            v878 = -1 + v808;
                                            int v879; int v880;
                                            Tuple7 tmp48 = Tuple7{0, 0};
                                            v879 = tmp48.v0; v880 = tmp48.v1;
                                            while (while_method_0(v879)){
                                                bool v882;
                                                v882 = 0 <= v879;
                                                bool v884;
                                                if (v882){
                                                    bool v883;
                                                    v883 = v879 < 2;
                                                    v884 = v883;
                                                } else {
                                                    v884 = false;
                                                }
                                                bool v885;
                                                v885 = v884 == false;
                                                if (v885){
                                                    assert("Index must be in range." && v884);
                                                } else {
                                                }
                                                int v887;
                                                v887 = v807[v879];
                                                bool v889;
                                                v889 = v880 >= v887;
                                                int v890;
                                                if (v889){
                                                    v890 = v880;
                                                } else {
                                                    v890 = v887;
                                                }
                                                v880 = v890;
                                                v879 += 1 ;
                                            }
                                            static_array<int,2> v891;
                                            int v893;
                                            v893 = 0;
                                            while (while_method_0(v893)){
                                                v891[v893] = v880;
                                                v893 += 1 ;
                                            }
                                            static_array<int,2> v895;
                                            int v897;
                                            v897 = 0;
                                            while (while_method_0(v897)){
                                                bool v899;
                                                v899 = 0 <= v897;
                                                bool v901;
                                                if (v899){
                                                    bool v900;
                                                    v900 = v897 < 2;
                                                    v901 = v900;
                                                } else {
                                                    v901 = false;
                                                }
                                                bool v902;
                                                v902 = v901 == false;
                                                if (v902){
                                                    assert("Index must be in range." && v901);
                                                } else {
                                                }
                                                int v904;
                                                v904 = v891[v897];
                                                bool v906;
                                                v906 = v897 == v806;
                                                int v908;
                                                if (v906){
                                                    int v907;
                                                    v907 = v904 + 2;
                                                    v908 = v907;
                                                } else {
                                                    v908 = v904;
                                                }
                                                v895[v897] = v908;
                                                v897 += 1 ;
                                            }
                                            v916 = Union4{Union4_2{v803, false, v805, v877, v895, v878}};
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
                                Union6 v810 = v803.case1.v0;
                                switch (v809.tag) {
                                    case 0: { // Call
                                        if (v804){
                                            int v812;
                                            v812 = v806 ^ 1;
                                            v916 = Union4{Union4_2{v803, false, v805, v812, v807, v808}};
                                        } else {
                                            int v814; int v815;
                                            Tuple7 tmp49 = Tuple7{0, 0};
                                            v814 = tmp49.v0; v815 = tmp49.v1;
                                            while (while_method_0(v814)){
                                                bool v817;
                                                v817 = 0 <= v814;
                                                bool v819;
                                                if (v817){
                                                    bool v818;
                                                    v818 = v814 < 2;
                                                    v819 = v818;
                                                } else {
                                                    v819 = false;
                                                }
                                                bool v820;
                                                v820 = v819 == false;
                                                if (v820){
                                                    assert("Index must be in range." && v819);
                                                } else {
                                                }
                                                int v822;
                                                v822 = v807[v814];
                                                bool v824;
                                                v824 = v815 >= v822;
                                                int v825;
                                                if (v824){
                                                    v825 = v815;
                                                } else {
                                                    v825 = v822;
                                                }
                                                v815 = v825;
                                                v814 += 1 ;
                                            }
                                            static_array<int,2> v826;
                                            int v828;
                                            v828 = 0;
                                            while (while_method_0(v828)){
                                                v826[v828] = v815;
                                                v828 += 1 ;
                                            }
                                            v916 = Union4{Union4_4{v803, v804, v805, v806, v826, v808}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v916 = Union4{Union4_5{v803, v804, v805, v806, v807, v808}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v832;
                                        v832 = v808 > 0;
                                        if (v832){
                                            int v833;
                                            v833 = v806 ^ 1;
                                            int v834;
                                            v834 = -1 + v808;
                                            int v835; int v836;
                                            Tuple7 tmp50 = Tuple7{0, 0};
                                            v835 = tmp50.v0; v836 = tmp50.v1;
                                            while (while_method_0(v835)){
                                                bool v838;
                                                v838 = 0 <= v835;
                                                bool v840;
                                                if (v838){
                                                    bool v839;
                                                    v839 = v835 < 2;
                                                    v840 = v839;
                                                } else {
                                                    v840 = false;
                                                }
                                                bool v841;
                                                v841 = v840 == false;
                                                if (v841){
                                                    assert("Index must be in range." && v840);
                                                } else {
                                                }
                                                int v843;
                                                v843 = v807[v835];
                                                bool v845;
                                                v845 = v836 >= v843;
                                                int v846;
                                                if (v845){
                                                    v846 = v836;
                                                } else {
                                                    v846 = v843;
                                                }
                                                v836 = v846;
                                                v835 += 1 ;
                                            }
                                            static_array<int,2> v847;
                                            int v849;
                                            v849 = 0;
                                            while (while_method_0(v849)){
                                                v847[v849] = v836;
                                                v849 += 1 ;
                                            }
                                            static_array<int,2> v851;
                                            int v853;
                                            v853 = 0;
                                            while (while_method_0(v853)){
                                                bool v855;
                                                v855 = 0 <= v853;
                                                bool v857;
                                                if (v855){
                                                    bool v856;
                                                    v856 = v853 < 2;
                                                    v857 = v856;
                                                } else {
                                                    v857 = false;
                                                }
                                                bool v858;
                                                v858 = v857 == false;
                                                if (v858){
                                                    assert("Index must be in range." && v857);
                                                } else {
                                                }
                                                int v860;
                                                v860 = v847[v853];
                                                bool v862;
                                                v862 = v853 == v806;
                                                int v864;
                                                if (v862){
                                                    int v863;
                                                    v863 = v860 + 4;
                                                    v864 = v863;
                                                } else {
                                                    v864 = v860;
                                                }
                                                v851[v853] = v864;
                                                v853 += 1 ;
                                            }
                                            v916 = Union4{Union4_2{v803, false, v805, v833, v851, v834}};
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
                        v924 = Union3{Union3_1{v916}};
                        break;
                    }
                    case 3: { // T_none
                        v924 = Union3{Union3_0{}};
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
        v20 = v924;
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
    v1 = v0 < 8192;
    return v1;
}
__device__ inline bool while_method_14(int v0){
    bool v1;
    v1 = v0 < 2048;
    return v1;
}
__device__ inline bool while_method_15(int v0){
    bool v1;
    v1 = v0 < 24;
    return v1;
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
    while (while_method_11(v16)){
        Union3 v920;
        switch (v16.tag) {
            case 0: { // None
                v920 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v18 = v16.case1.v0;
                Union14 v760;
                switch (v18.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v729 = v18.case0.v0; bool v730 = v18.case0.v1; static_array<Union6,2> v731 = v18.case0.v2; int v732 = v18.case0.v3; static_array<int,2> v733 = v18.case0.v4; int v734 = v18.case0.v5;
                        curandStatePhilox4_32_10_t & v735 = v3.v5;
                        curandStatePhilox4_32_10_t & v736 = v735;
                        unsigned int & v737 = v3.v0;
                        Union6 v738; unsigned int v739;
                        Tuple6 tmp55 = draw_card_20(v736, v737);
                        v738 = tmp55.v0; v739 = tmp55.v1;
                        v3.v0 = v739;
                        Union7 v740;
                        v740 = Union7{Union7_0{v738}};
                        v14.push(v740);
                        v760 = Union14{Union14_0{v729, v730, v731, v732, v733, v734, v738}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v742 = v3.v5;
                        curandStatePhilox4_32_10_t & v743 = v742;
                        unsigned int & v744 = v3.v0;
                        Union6 v745; unsigned int v746;
                        Tuple6 tmp56 = draw_card_20(v743, v744);
                        v745 = tmp56.v0; v746 = tmp56.v1;
                        v3.v0 = v746;
                        curandStatePhilox4_32_10_t & v747 = v3.v5;
                        curandStatePhilox4_32_10_t & v748 = v747;
                        unsigned int & v749 = v3.v0;
                        Union6 v750; unsigned int v751;
                        Tuple6 tmp57 = draw_card_20(v748, v749);
                        v750 = tmp57.v0; v751 = tmp57.v1;
                        v3.v0 = v751;
                        Union7 v752;
                        v752 = Union7{Union7_2{0, v745}};
                        v14.push(v752);
                        Union7 v753;
                        v753 = Union7{Union7_2{1, v750}};
                        v14.push(v753);
                        v760 = Union14{Union14_1{v745, v750}};
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
                        Union1 v717;
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
                                while (while_method_4(v155)){
                                    float * v157;
                                    v157 = reinterpret_cast<float *>(&v1[4718592ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 32);
                                    int v159;
                                    v159 = 393216 * v155;
                                    float * v160;
                                    v160 = reinterpret_cast<float *>(&v1[0ull]);
                                    float * v162;
                                    v162 = reinterpret_cast<float *>(&v0[0ull]);
                                    float * v164;
                                    v164 = reinterpret_cast<float *>(&v2[0ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 32);
                                    int v166;
                                    v166 = 8192 * v155;
                                    float * v167;
                                    v167 = reinterpret_cast<float *>(&v1[3145728ull]);
                                    block_matmul_23(v167, v162, v166, v160);
                                    block_row_map_24(v157, v159, v167);
                                    int * v169;
                                    v169 = reinterpret_cast<int *>(&v0[1048576ull]);
                                    bool * v171;
                                    v171 = reinterpret_cast<bool *>(&v0[1048592ull]);
                                    float * v173;
                                    v173 = reinterpret_cast<float *>(&v0[1048624ull]);
                                    float * v175;
                                    v175 = reinterpret_cast<float *>(&v0[1048752ull]);
                                    double * v177;
                                    v177 = reinterpret_cast<double *>(&v1[55050240ull]);
                                    double * v179;
                                    v179 = reinterpret_cast<double *>(&v1[58195968ull]);
                                    v155 += 1 ;
                                }
                                __syncthreads();
                                int * v181;
                                v181 = reinterpret_cast<int *>(&v0[1048576ull]);
                                bool * v183;
                                v183 = reinterpret_cast<bool *>(&v0[1048592ull]);
                                float * v185;
                                v185 = reinterpret_cast<float *>(&v0[1048624ull]);
                                float * v187;
                                v187 = reinterpret_cast<float *>(&v0[1048752ull]);
                                int v189;
                                v189 = v181[0];
                                float * v190;
                                v190 = reinterpret_cast<float *>(&v1[4718592ull]);
                                assert("Tensor range check" && 0 <= v189 && v189 < 32);
                                int v192;
                                v192 = 393216 * v189;
                                int v193;
                                v193 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v193 && v193 < 24);
                                int v194;
                                v194 = 16384 * v193;
                                int v195;
                                v195 = v194 + v192;
                                int v196;
                                v196 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v196 && v196 < 256);
                                int v197;
                                v197 = 64 * v196;
                                int v198;
                                v198 = v197 + v195;
                                float * v199;
                                v199 = v190+v198;
                                int v201;
                                v201 = sizeof(float *);
                                unsigned long long v202;
                                v202 = (unsigned long long)v201;
                                unsigned long long v203;
                                v203 = 256ull * v202;
                                unsigned long long v204;
                                v204 = v203 + 16ull;
                                unsigned long long v205;
                                v205 = v204 - 1ull;
                                unsigned long long v206;
                                v206 = v205 % 16ull;
                                unsigned long long v207;
                                v207 = v205 - v206;
                                unsigned long long v208;
                                v208 = v207 + 1024ull;
                                unsigned long long v209;
                                v209 = v208 + 16ull;
                                unsigned long long v210;
                                v210 = v209 - 1ull;
                                unsigned long long v211;
                                v211 = v210 % 16ull;
                                unsigned long long v212;
                                v212 = v210 - v211;
                                unsigned long long v213;
                                v213 = v212 + 1024ull;
                                bool v214;
                                v214 = v213 <= 98304ull;
                                bool v215;
                                v215 = v214 == false;
                                if (v215){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v214);
                                } else {
                                }
                                extern __shared__ unsigned char v217[];
                                bool v218;
                                v218 = v213 <= v213;
                                bool v219;
                                v219 = v218 == false;
                                if (v219){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v218);
                                } else {
                                }
                                float * * v221;
                                v221 = reinterpret_cast<float * *>(&v217[0ull]);
                                float * v223;
                                v223 = reinterpret_cast<float *>(&v217[v207]);
                                int * v225;
                                v225 = reinterpret_cast<int *>(&v217[v212]);
                                int v227;
                                v227 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v227 && v227 < 256);
                                v221[v227] = v199;
                                __syncthreads();
                                bool v228;
                                v228 = 0 <= v227;
                                bool v229;
                                v229 = v228 == false;
                                if (v229){
                                    assert("The index needs to be zero or positive." && v228);
                                } else {
                                }
                                int v231;
                                v231 = v227 % 16;
                                int v232;
                                v232 = v227 / 16;
                                bool v233;
                                v233 = v232 < 16;
                                bool v234;
                                v234 = v233 == false;
                                if (v234){
                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v233);
                                } else {
                                }
                                assert("Tensor range check" && 0 <= v232 && v232 < 16);
                                int v236;
                                v236 = 0;
                                while (while_method_9(v236)){
                                    bool v238;
                                    v238 = 0 <= v232;
                                    bool v239;
                                    v239 = v238 && v233;
                                    bool v240;
                                    v240 = v239 == false;
                                    if (v240){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v239);
                                    } else {
                                    }
                                    bool v242;
                                    v242 = 0 <= v236;
                                    bool v244;
                                    if (v242){
                                        bool v243;
                                        v243 = v236 < 16;
                                        v244 = v243;
                                    } else {
                                        v244 = false;
                                    }
                                    bool v245;
                                    v245 = v244 == false;
                                    if (v245){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v244);
                                    } else {
                                    }
                                    int v247;
                                    v247 = v236 * 16;
                                    int v248;
                                    v248 = v247 + v232;
                                    assert("Tensor range check" && 0 <= v236 && v236 < 16);
                                    int v249;
                                    v249 = 16 * v236;
                                    int v250;
                                    v250 = v249 + v232;
                                    float * v251;
                                    v251 = v221[v250];
                                    int v252;
                                    v252 = blockIdx.x;
                                    int v253;
                                    v253 = v252 * 256;
                                    int v254;
                                    v254 = v253 + v248;
                                    assert("Tensor range check" && 0 <= v231 && v231 < 16);
                                    int v255;
                                    v255 = 4 * v231;
                                    float v256[4];
                                    int v257[4];
                                    int v258;
                                    v258 = 0;
                                    while (while_method_5(v258)){
                                        assert("Tensor range check" && 0 <= v258 && v258 < 1);
                                        int v260;
                                        v260 = 4 * v258;
                                        assert("Tensor range check" && 0 <= v258 && v258 < 1);
                                        int v261;
                                        v261 = 64 * v258;
                                        int v262;
                                        v262 = v261 + v255;
                                        int4* v263;
                                        v263 = reinterpret_cast<int4*>(v251 + v262);
                                        int4* v264;
                                        v264 = reinterpret_cast<int4*>(v256 + v260);
                                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v263) % 16 == 0 && reinterpret_cast<unsigned long long>(v264) % 16 == 0);
                                        *v264 = *v263;
                                        v258 += 1 ;
                                    }
                                    int v265;
                                    v265 = 0;
                                    while (while_method_5(v265)){
                                        int v267;
                                        v267 = 0;
                                        while (while_method_8(v267)){
                                            bool v269;
                                            v269 = 0 <= v267;
                                            bool v271;
                                            if (v269){
                                                bool v270;
                                                v270 = v267 < 4;
                                                v271 = v270;
                                            } else {
                                                v271 = false;
                                            }
                                            bool v272;
                                            v272 = v271 == false;
                                            if (v272){
                                                assert("The indices should be inside the range of the dimension." && v271);
                                            } else {
                                            }
                                            bool v274;
                                            v274 = 0 <= v231;
                                            bool v276;
                                            if (v274){
                                                bool v275;
                                                v275 = v231 < 16;
                                                v276 = v275;
                                            } else {
                                                v276 = false;
                                            }
                                            bool v277;
                                            v277 = v276 == false;
                                            if (v277){
                                                assert("The indices should be inside the range of the dimension." && v276);
                                            } else {
                                            }
                                            int v279;
                                            v279 = v231 * 4;
                                            int v280;
                                            v280 = v267 + v279;
                                            bool v281;
                                            v281 = 0 <= v265;
                                            bool v283;
                                            if (v281){
                                                bool v282;
                                                v282 = v265 < 1;
                                                v283 = v282;
                                            } else {
                                                v283 = false;
                                            }
                                            bool v284;
                                            v284 = v283 == false;
                                            if (v284){
                                                assert("The indices should be inside the range of the dimension." && v283);
                                            } else {
                                            }
                                            int v286;
                                            v286 = v265 * 64;
                                            int v287;
                                            v287 = v280 + v286;
                                            assert("Tensor range check" && 0 <= v265 && v265 < 1);
                                            assert("Tensor range check" && 0 <= v267 && v267 < 4);
                                            int v288;
                                            v288 = 4 * v265;
                                            int v289;
                                            v289 = v288 + v267;
                                            v257[v289] = v287;
                                            v267 += 1 ;
                                        }
                                        v265 += 1 ;
                                    }
                                    float v290[4];
                                    float v291;
                                    v291 = 0.0f;
                                    int v292;
                                    v292 = 0;
                                    while (while_method_5(v292)){
                                        assert("Tensor range check" && 0 <= v292 && v292 < 1);
                                        int v294;
                                        v294 = 4 * v292;
                                        assert("Tensor range check" && 0 <= v292 && v292 < 1);
                                        float v295;
                                        v295 = 0.0f;
                                        int v296;
                                        v296 = 0;
                                        while (while_method_8(v296)){
                                            assert("Tensor range check" && 0 <= v296 && v296 < 4);
                                            int v298;
                                            v298 = v296 + v294;
                                            float v299;
                                            v299 = v256[v298];
                                            float v300;
                                            v300 = v295 + v299;
                                            v295 = v300;
                                            v296 += 1 ;
                                        }
                                        auto v301 = cooperative_groups::coalesced_threads();
                                        int v302;
                                        v302 = threadIdx.x;
                                        int v303;
                                        v303 = v302 / 16;
                                        auto v304 = cooperative_groups::labeled_partition(v301,v303);
                                        Closure2 v305{};
                                        float v306;
                                        v306 = cooperative_groups::inclusive_scan(v304, v295, v305);
                                        float v307;
                                        v307 = v304.shfl_up(v306,1);
                                        bool v308;
                                        v308 = v304.thread_rank() == 0;
                                        float v309;
                                        if (v308){
                                            v309 = 0.0f;
                                        } else {
                                            v309 = v307;
                                        }
                                        float v310;
                                        v310 = v304.shfl(v306,v304.num_threads()-1);
                                        float v311;
                                        v311 = v291 + v309;
                                        float v312;
                                        v312 = v311;
                                        int v313;
                                        v313 = 0;
                                        while (while_method_8(v313)){
                                            assert("Tensor range check" && 0 <= v313 && v313 < 4);
                                            int v315;
                                            v315 = v313 + v294;
                                            float v316;
                                            v316 = v256[v315];
                                            float v317;
                                            v317 = v312 + v316;
                                            assert("Tensor range check" && 0 <= v313 && v313 < 4);
                                            v290[v315] = v317;
                                            v312 = v317;
                                            v313 += 1 ;
                                        }
                                        float v318;
                                        v318 = v291 + v310;
                                        v291 = v318;
                                        v292 += 1 ;
                                    }
                                    float v319[4];
                                    bool v320[4];
                                    int v321;
                                    v321 = 0;
                                    while (while_method_5(v321)){
                                        int v323;
                                        v323 = 0;
                                        while (while_method_8(v323)){
                                            assert("Tensor range check" && 0 <= v321 && v321 < 1);
                                            assert("Tensor range check" && 0 <= v323 && v323 < 4);
                                            int v325;
                                            v325 = 4 * v321;
                                            int v326;
                                            v326 = v325 + v323;
                                            float v327;
                                            v327 = v290[v326];
                                            float v328;
                                            v328 = v256[v326];
                                            bool v329;
                                            v329 = v328 > 0.0f;
                                            assert("Tensor range check" && 0 <= v321 && v321 < 1);
                                            assert("Tensor range check" && 0 <= v323 && v323 < 4);
                                            v319[v326] = v327;
                                            v320[v326] = v329;
                                            v323 += 1 ;
                                        }
                                        v321 += 1 ;
                                    }
                                    float v330; bool v331;
                                    Tuple8 tmp58 = Tuple8{-1.0f / 0.0f, false};
                                    v330 = tmp58.v0; v331 = tmp58.v1;
                                    int v332;
                                    v332 = 0;
                                    while (while_method_5(v332)){
                                        int v334;
                                        v334 = 0;
                                        while (while_method_8(v334)){
                                            assert("Tensor range check" && 0 <= v332 && v332 < 1);
                                            assert("Tensor range check" && 0 <= v334 && v334 < 4);
                                            int v336;
                                            v336 = 4 * v332;
                                            int v337;
                                            v337 = v336 + v334;
                                            float v338;
                                            v338 = v319[v337];
                                            bool v339;
                                            v339 = v320[v337];
                                            float v346; bool v347;
                                            if (v331){
                                                if (v339){
                                                    bool v340;
                                                    v340 = v330 >= v338;
                                                    float v341;
                                                    if (v340){
                                                        v341 = v330;
                                                    } else {
                                                        v341 = v338;
                                                    }
                                                    v346 = v341; v347 = true;
                                                } else {
                                                    v346 = v330; v347 = v331;
                                                }
                                            } else {
                                                if (v339){
                                                    v346 = v338; v347 = v339;
                                                } else {
                                                    v346 = v330; v347 = v331;
                                                }
                                            }
                                            v330 = v346;
                                            v331 = v347;
                                            v334 += 1 ;
                                        }
                                        v332 += 1 ;
                                    }
                                    auto v348 = cooperative_groups::coalesced_threads();
                                    int v349;
                                    v349 = threadIdx.x;
                                    int v350;
                                    v350 = v349 / 16;
                                    auto v351 = cooperative_groups::labeled_partition(v348,v350);
                                    Closure3 v352{};
                                    float v353; bool v354;
                                    Tuple8 tmp59 = cooperative_groups::reduce(v351, Tuple8{v330, v331}, v352);
                                    v353 = tmp59.v0; v354 = tmp59.v1;
                                    bool v355;
                                    v355 = v354 == false;
                                    if (v355){
                                        int v356;
                                        v356 = threadIdx.x;
                                        int v357;
                                        v357 = blockIdx.x;
                                        int v358;
                                        v358 = v357 * 256;
                                        int v359;
                                        v359 = v356 + v358;
                                        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v360 = console_lock;
                                        auto v361 = cooperative_groups::coalesced_threads();
                                        v360.acquire();
                                        int v362;
                                        v362 = 0;
                                        printf("{%s = %d; %s = %c","tid", v359, "x'", '[');
                                        int v363;
                                        v363 = 0;
                                        while (while_method_5(v363)){
                                            int v365;
                                            v365 = v362;
                                            bool v366;
                                            v366 = v365 >= 100;
                                            if (v366){
                                                printf("%s"," ...");
                                                break;
                                            } else {
                                            }
                                            bool v367;
                                            v367 = v363 == 0;
                                            bool v368;
                                            v368 = v367 != true;
                                            if (v368){
                                                printf("%s","; ");
                                            } else {
                                            }
                                            printf("%c",'[');
                                            int v369;
                                            v369 = 0;
                                            while (while_method_8(v369)){
                                                int v371;
                                                v371 = v362;
                                                bool v372;
                                                v372 = v371 >= 100;
                                                if (v372){
                                                    printf("%s"," ...");
                                                    break;
                                                } else {
                                                }
                                                bool v373;
                                                v373 = v369 == 0;
                                                bool v374;
                                                v374 = v373 != true;
                                                if (v374){
                                                    printf("%s","; ");
                                                } else {
                                                }
                                                int v375;
                                                v375 = v362 + 1;
                                                v362 = v375;
                                                int v376;
                                                v376 = v363 * 4;
                                                int v377;
                                                v377 = v376 + v369;
                                                float v378;
                                                v378 = v319[v377];
                                                bool v379;
                                                v379 = v320[v377];
                                                const char * v382;
                                                if (v379){
                                                    const char * v380;
                                                    v380 = "true";
                                                    v382 = v380;
                                                } else {
                                                    const char * v381;
                                                    v381 = "false";
                                                    v382 = v381;
                                                }
                                                printf("%f, %s",v378, v382);
                                                v369 += 1 ;
                                            }
                                            printf("%c",']');
                                            v363 += 1 ;
                                        }
                                        printf("%c",']');
                                        printf("}\n");
                                        v360.release();
                                        v361.sync() ;
                                    } else {
                                    }
                                    if (v355){
                                        assert("The local reduce must be true." && v354);
                                    } else {
                                    }
                                    float v418[4];
                                    int v419[4];
                                    int v420;
                                    v420 = 0;
                                    while (while_method_5(v420)){
                                        int v422;
                                        v422 = 0;
                                        while (while_method_8(v422)){
                                            assert("Tensor range check" && 0 <= v420 && v420 < 1);
                                            assert("Tensor range check" && 0 <= v422 && v422 < 4);
                                            int v424;
                                            v424 = 4 * v420;
                                            int v425;
                                            v425 = v424 + v422;
                                            int v426;
                                            v426 = v257[v425];
                                            float v427;
                                            v427 = curand_uniform(&v85);
                                            assert("Tensor range check" && 0 <= v420 && v420 < 1);
                                            assert("Tensor range check" && 0 <= v422 && v422 < 4);
                                            v418[v425] = v427;
                                            v419[v425] = v426;
                                            v422 += 1 ;
                                        }
                                        v420 += 1 ;
                                    }
                                    float v428; int v429;
                                    Tuple9 tmp60 = Tuple9{0.0f, 2147483647};
                                    v428 = tmp60.v0; v429 = tmp60.v1;
                                    int v430;
                                    v430 = 0;
                                    while (while_method_5(v430)){
                                        int v432;
                                        v432 = 0;
                                        while (while_method_8(v432)){
                                            assert("Tensor range check" && 0 <= v430 && v430 < 1);
                                            assert("Tensor range check" && 0 <= v432 && v432 < 4);
                                            int v434;
                                            v434 = 4 * v430;
                                            int v435;
                                            v435 = v434 + v432;
                                            float v436;
                                            v436 = v418[v435];
                                            int v437;
                                            v437 = v419[v435];
                                            bool v438;
                                            v438 = v429 < v437;
                                            float v439; int v440;
                                            if (v438){
                                                v439 = v428; v440 = v429;
                                            } else {
                                                v439 = v436; v440 = v437;
                                            }
                                            v428 = v439;
                                            v429 = v440;
                                            v432 += 1 ;
                                        }
                                        v430 += 1 ;
                                    }
                                    auto v441 = cooperative_groups::coalesced_threads();
                                    int v442;
                                    v442 = threadIdx.x;
                                    int v443;
                                    v443 = v442 / 16;
                                    auto v444 = cooperative_groups::labeled_partition(v441,v443);
                                    Closure4 v445{};
                                    float v446; int v447;
                                    Tuple9 tmp61 = cooperative_groups::reduce(v444, Tuple9{v428, v429}, v445);
                                    v446 = tmp61.v0; v447 = tmp61.v1;
                                    float v448;
                                    v448 = v353 * v446;
                                    int v449[4];
                                    bool v450[4];
                                    int v451;
                                    v451 = 0;
                                    while (while_method_5(v451)){
                                        int v453;
                                        v453 = 0;
                                        while (while_method_8(v453)){
                                            assert("Tensor range check" && 0 <= v451 && v451 < 1);
                                            assert("Tensor range check" && 0 <= v453 && v453 < 4);
                                            int v455;
                                            v455 = 4 * v451;
                                            int v456;
                                            v456 = v455 + v453;
                                            float v457;
                                            v457 = v319[v456];
                                            bool v458;
                                            v458 = v320[v456];
                                            int v459;
                                            v459 = v257[v456];
                                            int v462; bool v463;
                                            if (v458){
                                                float v460;
                                                v460 = v457 - v448;
                                                bool v461;
                                                v461 = v460 >= 0.0f;
                                                v462 = v459; v463 = v461;
                                            } else {
                                                v462 = 2147483647; v463 = false;
                                            }
                                            assert("Tensor range check" && 0 <= v451 && v451 < 1);
                                            assert("Tensor range check" && 0 <= v453 && v453 < 4);
                                            v449[v456] = v462;
                                            v450[v456] = v463;
                                            v453 += 1 ;
                                        }
                                        v451 += 1 ;
                                    }
                                    int v464; bool v465;
                                    Tuple10 tmp62 = Tuple10{2147483647, false};
                                    v464 = tmp62.v0; v465 = tmp62.v1;
                                    int v466;
                                    v466 = 0;
                                    while (while_method_5(v466)){
                                        int v468;
                                        v468 = 0;
                                        while (while_method_8(v468)){
                                            assert("Tensor range check" && 0 <= v466 && v466 < 1);
                                            assert("Tensor range check" && 0 <= v468 && v468 < 4);
                                            int v470;
                                            v470 = 4 * v466;
                                            int v471;
                                            v471 = v470 + v468;
                                            int v472;
                                            v472 = v449[v471];
                                            bool v473;
                                            v473 = v450[v471];
                                            int v480; bool v481;
                                            if (v465){
                                                if (v473){
                                                    bool v474;
                                                    v474 = v464 < v472;
                                                    int v475;
                                                    if (v474){
                                                        v475 = v464;
                                                    } else {
                                                        v475 = v472;
                                                    }
                                                    v480 = v475; v481 = true;
                                                } else {
                                                    v480 = v464; v481 = v465;
                                                }
                                            } else {
                                                if (v473){
                                                    v480 = v472; v481 = v473;
                                                } else {
                                                    v480 = v464; v481 = v465;
                                                }
                                            }
                                            v464 = v480;
                                            v465 = v481;
                                            v468 += 1 ;
                                        }
                                        v466 += 1 ;
                                    }
                                    auto v482 = cooperative_groups::coalesced_threads();
                                    int v483;
                                    v483 = threadIdx.x;
                                    int v484;
                                    v484 = v483 / 16;
                                    auto v485 = cooperative_groups::labeled_partition(v482,v484);
                                    Closure5 v486{};
                                    int v487; bool v488;
                                    Tuple10 tmp63 = cooperative_groups::reduce(v485, Tuple10{v464, v465}, v486);
                                    v487 = tmp63.v0; v488 = tmp63.v1;
                                    bool v489;
                                    v489 = v488 == false;
                                    if (v489){
                                        int v490;
                                        v490 = threadIdx.x;
                                        int v491;
                                        v491 = blockIdx.x;
                                        int v492;
                                        v492 = v491 * 256;
                                        int v493;
                                        v493 = v490 + v492;
                                        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v494 = console_lock;
                                        auto v495 = cooperative_groups::coalesced_threads();
                                        v494.acquire();
                                        int v496;
                                        v496 = 0;
                                        printf("{%s = %d; %s = %c","tid", v493, "x'", '[');
                                        int v497;
                                        v497 = 0;
                                        while (while_method_5(v497)){
                                            int v499;
                                            v499 = v496;
                                            bool v500;
                                            v500 = v499 >= 100;
                                            if (v500){
                                                printf("%s"," ...");
                                                break;
                                            } else {
                                            }
                                            bool v501;
                                            v501 = v497 == 0;
                                            bool v502;
                                            v502 = v501 != true;
                                            if (v502){
                                                printf("%s","; ");
                                            } else {
                                            }
                                            printf("%c",'[');
                                            int v503;
                                            v503 = 0;
                                            while (while_method_8(v503)){
                                                int v505;
                                                v505 = v496;
                                                bool v506;
                                                v506 = v505 >= 100;
                                                if (v506){
                                                    printf("%s"," ...");
                                                    break;
                                                } else {
                                                }
                                                bool v507;
                                                v507 = v503 == 0;
                                                bool v508;
                                                v508 = v507 != true;
                                                if (v508){
                                                    printf("%s","; ");
                                                } else {
                                                }
                                                int v509;
                                                v509 = v496 + 1;
                                                v496 = v509;
                                                int v510;
                                                v510 = v497 * 4;
                                                int v511;
                                                v511 = v510 + v503;
                                                int v512;
                                                v512 = v449[v511];
                                                bool v513;
                                                v513 = v450[v511];
                                                const char * v516;
                                                if (v513){
                                                    const char * v514;
                                                    v514 = "true";
                                                    v516 = v514;
                                                } else {
                                                    const char * v515;
                                                    v515 = "false";
                                                    v516 = v515;
                                                }
                                                printf("%d, %s",v512, v516);
                                                v503 += 1 ;
                                            }
                                            printf("%c",']');
                                            v497 += 1 ;
                                        }
                                        printf("%c",']');
                                        printf("}\n");
                                        v494.release();
                                        v495.sync() ;
                                    } else {
                                    }
                                    if (v489){
                                        assert("The local reduce must be true." && v488);
                                    } else {
                                    }
                                    float v552; int v553;
                                    Tuple9 tmp64 = Tuple9{0.0f, 2147483647};
                                    v552 = tmp64.v0; v553 = tmp64.v1;
                                    int v554;
                                    v554 = 0;
                                    while (while_method_5(v554)){
                                        int v556;
                                        v556 = 0;
                                        while (while_method_8(v556)){
                                            assert("Tensor range check" && 0 <= v554 && v554 < 1);
                                            assert("Tensor range check" && 0 <= v556 && v556 < 4);
                                            int v558;
                                            v558 = 4 * v554;
                                            int v559;
                                            v559 = v558 + v556;
                                            float v560;
                                            v560 = v256[v559];
                                            int v561;
                                            v561 = v257[v559];
                                            bool v562;
                                            v562 = v553 == v487;
                                            float v566; int v567;
                                            if (v562){
                                                v566 = v552; v567 = v553;
                                            } else {
                                                bool v563;
                                                v563 = v561 == v487;
                                                if (v563){
                                                    v566 = v560; v567 = v561;
                                                } else {
                                                    v566 = v552; v567 = v553;
                                                }
                                            }
                                            v552 = v566;
                                            v553 = v567;
                                            v556 += 1 ;
                                        }
                                        v554 += 1 ;
                                    }
                                    auto v568 = cooperative_groups::coalesced_threads();
                                    int v569;
                                    v569 = threadIdx.x;
                                    int v570;
                                    v570 = v569 / 16;
                                    auto v571 = cooperative_groups::labeled_partition(v568,v570);
                                    Closure6 v572{v487};
                                    float v573; int v574;
                                    Tuple9 tmp65 = cooperative_groups::reduce(v571, Tuple9{v552, v553}, v572);
                                    v573 = tmp65.v0; v574 = tmp65.v1;
                                    bool v575;
                                    v575 = v574 == 2147483647;
                                    bool v576;
                                    v576 = v575 != true;
                                    bool v577;
                                    v577 = v576 == false;
                                    if (v577){
                                        assert("Expected a valid action id in get_prob." && v576);
                                    } else {
                                    }
                                    int v579;
                                    v579 = 0;
                                    while (while_method_5(v579)){
                                        assert("Tensor range check" && 0 <= v579 && v579 < 1);
                                        assert("Tensor range check" && 0 <= v579 && v579 < 1);
                                        v579 += 1 ;
                                    }
                                    assert("Tensor range check" && 0 <= v248 && v248 < 256);
                                    v223[v248] = v573;
                                    v225[v248] = v487;
                                    v236 += 1 ;
                                }
                                __syncthreads();
                                assert("Tensor range check" && 0 <= v227 && v227 < 256);
                                float v581;
                                v581 = v223[v227];
                                int v582;
                                v582 = v225[v227];
                                __syncthreads();
                                extern __shared__ unsigned char v583[];
                                float * v584;
                                v584 = reinterpret_cast<float *>(&v583[0ull]);
                                int * v586;
                                v586 = reinterpret_cast<int *>(&v583[16ull]);
                                int v588;
                                v588 = threadIdx.x;
                                bool v589;
                                v589 = v588 == 0;
                                if (v589){
                                    v584[0] = v581;
                                    v586[0] = v582;
                                } else {
                                }
                                __syncthreads();
                                float v590;
                                v590 = v584[0];
                                int v591;
                                v591 = v586[0];
                                __syncthreads();
                                double * v592;
                                v592 = reinterpret_cast<double *>(&v1[55050240ull]);
                                double * v594;
                                v594 = reinterpret_cast<double *>(&v1[58195968ull]);
                                int v596;
                                v596 = threadIdx.x;
                                int v597;
                                v597 = blockIdx.x;
                                int v598;
                                v598 = v597 * 256;
                                int v599;
                                v599 = v596 + v598;
                                int v600;
                                v600 = 0;
                                while (while_method_4(v600)){
                                    float * v602;
                                    v602 = reinterpret_cast<float *>(&v1[4718592ull]);
                                    int v604;
                                    v604 = blockIdx.x;
                                    int v605;
                                    v605 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v600 && v600 < 32);
                                    assert("Tensor range check" && 0 <= v604 && v604 < 24);
                                    assert("Tensor range check" && 0 <= v605 && v605 < 256);
                                    assert("Tensor range check" && 0 <= v591 && v591 < 64);
                                    int v606;
                                    v606 = 64 * v605;
                                    int v607;
                                    v607 = v606 + v591;
                                    int v608;
                                    v608 = 16384 * v604;
                                    int v609;
                                    v609 = v608 + v607;
                                    int v610;
                                    v610 = 393216 * v600;
                                    int v611;
                                    v611 = v610 + v609;
                                    float v612;
                                    v612 = v602[v611];
                                    double v613;
                                    v613 = (double)v590;
                                    double v614;
                                    v614 = log(v613);
                                    double v615;
                                    v615 = (double)v612;
                                    double v616;
                                    v616 = log(v615);
                                    assert("Tensor range check" && 0 <= v600 && v600 < 32);
                                    assert("Tensor range check" && 0 <= v599 && v599 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    int v617;
                                    v617 = 2 * v599;
                                    int v618;
                                    v618 = v617 + v71;
                                    int v619;
                                    v619 = 12288 * v600;
                                    int v620;
                                    v620 = v619 + v618;
                                    double v621;
                                    v621 = v592[v620];
                                    double v622;
                                    v622 = v594[v620];
                                    double v623;
                                    v623 = v616 + v621;
                                    double v624;
                                    v624 = v614 + v622;
                                    bool v625;
                                    v625 = isnan(v624);
                                    bool v626;
                                    v626 = v625 == false;
                                    bool v627;
                                    v627 = v626 == false;
                                    if (v627){
                                        assert("The sampling log probability shouldn't be nan." && v626);
                                    } else {
                                    }
                                    bool v629;
                                    v629 = isnan(v623);
                                    bool v630;
                                    v630 = v629 == false;
                                    bool v631;
                                    v631 = v630 == false;
                                    if (v631){
                                        assert("The policy log probability shouldn't be nan." && v630);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v600 && v600 < 32);
                                    assert("Tensor range check" && 0 <= v599 && v599 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    v592[v620] = v623;
                                    v594[v620] = v624;
                                    v600 += 1 ;
                                }
                                bool v633;
                                v633 = 0 == v591;
                                Union12 v642;
                                if (v633){
                                    v642 = Union12{Union12_1{}};
                                } else {
                                    bool v635;
                                    v635 = 1 == v591;
                                    if (v635){
                                        v642 = Union12{Union12_0{}};
                                    } else {
                                        bool v637;
                                        v637 = 2 == v591;
                                        if (v637){
                                            v642 = Union12{Union12_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v642.tag) {
                                    case 0: { // AA_Call
                                        v717 = Union1{Union1_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v643;
                                        v643 = v72[0];
                                        int v645; int v646;
                                        Tuple7 tmp66 = Tuple7{1, v643};
                                        v645 = tmp66.v0; v646 = tmp66.v1;
                                        while (while_method_0(v645)){
                                            bool v648;
                                            v648 = 0 <= v645;
                                            bool v650;
                                            if (v648){
                                                bool v649;
                                                v649 = v645 < 2;
                                                v650 = v649;
                                            } else {
                                                v650 = false;
                                            }
                                            bool v651;
                                            v651 = v650 == false;
                                            if (v651){
                                                assert("Index must be in range." && v650);
                                            } else {
                                            }
                                            int v653;
                                            v653 = v72[v645];
                                            bool v655;
                                            v655 = v646 >= v653;
                                            int v656;
                                            if (v655){
                                                v656 = v646;
                                            } else {
                                                v656 = v653;
                                            }
                                            v646 = v656;
                                            v645 += 1 ;
                                        }
                                        bool v658;
                                        if (v75){
                                            bool v657;
                                            v657 = v71 < 2;
                                            v658 = v657;
                                        } else {
                                            v658 = false;
                                        }
                                        bool v659;
                                        v659 = v658 == false;
                                        if (v659){
                                            assert("Index must be in range." && v658);
                                        } else {
                                        }
                                        int v661;
                                        v661 = v72[v71];
                                        bool v663;
                                        v663 = v661 == v646;
                                        if (v663){
                                            v717 = Union1{Union1_0{}};
                                        } else {
                                            v717 = Union1{Union1_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v668;
                                        v668 = v73 > 0;
                                        if (v668){
                                            v717 = Union1{Union1_2{}};
                                        } else {
                                            v717 = Union1{Union1_0{}};
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
                                curandStatePhilox4_32_10_t & v675 = v3.v5;
                                curandStatePhilox4_32_10_t & v676 = v675;
                                static_array_list<Union1,3> v677;
                                v677 = static_array_list<Union1,3>{};
                                v677.unsafe_set_length(1);
                                Union1 v679;
                                v679 = Union1{Union1_0{}};
                                v677[0] = v679;
                                int v681;
                                v681 = v72[0];
                                int v683;
                                v683 = v72[1];
                                bool v685;
                                v685 = v681 == v683;
                                bool v686;
                                v686 = v685 != true;
                                if (v686){
                                    Union1 v687;
                                    v687 = Union1{Union1_1{}};
                                    v677.push(v687);
                                } else {
                                }
                                bool v688;
                                v688 = v73 > 0;
                                if (v688){
                                    Union1 v689;
                                    v689 = Union1{Union1_2{}};
                                    v677.push(v689);
                                } else {
                                }
                                int v690;
                                v690 = v677.length;
                                int v691;
                                v691 = v690 - 1;
                                int v692;
                                v692 = 0;
                                while (while_method_1(v691, v692)){
                                    int v694;
                                    v694 = v677.length;
                                    int v695;
                                    v695 = int_range_22(v694, v692, v676);
                                    Union1 v696;
                                    v696 = v677[v692];
                                    Union1 v698;
                                    v698 = v677[v695];
                                    v677[v692] = v698;
                                    v677[v695] = v696;
                                    v692 += 1 ;
                                }
                                Union1 v700;
                                v700 = v677.pop();
                                int v701;
                                v701 = sizeof(Union1);
                                unsigned long long v702;
                                v702 = (unsigned long long)v701;
                                bool v703;
                                v703 = v702 <= 98304ull;
                                bool v704;
                                v704 = v703 == false;
                                if (v704){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v703);
                                } else {
                                }
                                extern __shared__ unsigned char v706[];
                                bool v707;
                                v707 = v702 <= v702;
                                bool v708;
                                v708 = v707 == false;
                                if (v708){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v707);
                                } else {
                                }
                                Union1 * v710;
                                v710 = reinterpret_cast<Union1 *>(&v706[0ull]);
                                int v712;
                                v712 = threadIdx.x;
                                bool v713;
                                v713 = v712 == 0;
                                if (v713){
                                    v710[0] = v700;
                                } else {
                                }
                                __syncthreads();
                                Union1 v714;
                                v714 = v710[0];
                                __syncthreads();
                                v717 = v714;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union7 v718;
                        v718 = Union7{Union7_1{v71, v717}};
                        v14.push(v718);
                        v760 = Union14{Union14_2{v68, v69, v70, v71, v72, v73, v717}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v720 = v18.case3.v0; bool v721 = v18.case3.v1; static_array<Union6,2> v722 = v18.case3.v2; int v723 = v18.case3.v3; static_array<int,2> v724 = v18.case3.v4; int v725 = v18.case3.v5; Union1 v726 = v18.case3.v6;
                        Union7 v727;
                        v727 = Union7{Union7_1{v723, v726}};
                        v14.push(v727);
                        v760 = Union14{Union14_2{v720, v721, v722, v723, v724, v725, v726}};
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
                        v760 = Union14{Union14_3{}};
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
                        v760 = Union14{Union14_3{}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                switch (v760.tag) {
                    case 0: { // T_game_chance_community_card
                        Union5 v762 = v760.case0.v0; bool v763 = v760.case0.v1; static_array<Union6,2> v764 = v760.case0.v2; int v765 = v760.case0.v3; static_array<int,2> v766 = v760.case0.v4; int v767 = v760.case0.v5; Union6 v768 = v760.case0.v6;
                        int v769;
                        v769 = 2;
                        int v770; int v771;
                        Tuple7 tmp67 = Tuple7{0, 0};
                        v770 = tmp67.v0; v771 = tmp67.v1;
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
                            v778 = v766[v770];
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
                        Union5 v786;
                        v786 = Union5{Union5_1{v768}};
                        Union4 v787;
                        v787 = Union4{Union4_2{v786, true, v764, 0, v782, v769}};
                        v920 = Union3{Union3_1{v787}};
                        break;
                    }
                    case 1: { // T_game_chance_init
                        Union6 v789 = v760.case1.v0; Union6 v790 = v760.case1.v1;
                        int v791;
                        v791 = 2;
                        static_array<int,2> v792;
                        v792[0] = 1;
                        v792[1] = 1;
                        static_array<Union6,2> v794;
                        v794[0] = v789;
                        v794[1] = v790;
                        Union5 v796;
                        v796 = Union5{Union5_0{}};
                        Union4 v797;
                        v797 = Union4{Union4_2{v796, true, v794, 0, v792, v791}};
                        v920 = Union3{Union3_1{v797}};
                        break;
                    }
                    case 2: { // T_game_round
                        Union5 v799 = v760.case2.v0; bool v800 = v760.case2.v1; static_array<Union6,2> v801 = v760.case2.v2; int v802 = v760.case2.v3; static_array<int,2> v803 = v760.case2.v4; int v804 = v760.case2.v5; Union1 v805 = v760.case2.v6;
                        Union4 v912;
                        switch (v799.tag) {
                            case 0: { // None
                                switch (v805.tag) {
                                    case 0: { // Call
                                        if (v800){
                                            int v868;
                                            v868 = v802 ^ 1;
                                            v912 = Union4{Union4_2{v799, false, v801, v868, v803, v804}};
                                        } else {
                                            v912 = Union4{Union4_0{v799, v800, v801, v802, v803, v804}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v912 = Union4{Union4_5{v799, v800, v801, v802, v803, v804}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v872;
                                        v872 = v804 > 0;
                                        if (v872){
                                            int v873;
                                            v873 = v802 ^ 1;
                                            int v874;
                                            v874 = -1 + v804;
                                            int v875; int v876;
                                            Tuple7 tmp68 = Tuple7{0, 0};
                                            v875 = tmp68.v0; v876 = tmp68.v1;
                                            while (while_method_0(v875)){
                                                bool v878;
                                                v878 = 0 <= v875;
                                                bool v880;
                                                if (v878){
                                                    bool v879;
                                                    v879 = v875 < 2;
                                                    v880 = v879;
                                                } else {
                                                    v880 = false;
                                                }
                                                bool v881;
                                                v881 = v880 == false;
                                                if (v881){
                                                    assert("Index must be in range." && v880);
                                                } else {
                                                }
                                                int v883;
                                                v883 = v803[v875];
                                                bool v885;
                                                v885 = v876 >= v883;
                                                int v886;
                                                if (v885){
                                                    v886 = v876;
                                                } else {
                                                    v886 = v883;
                                                }
                                                v876 = v886;
                                                v875 += 1 ;
                                            }
                                            static_array<int,2> v887;
                                            int v889;
                                            v889 = 0;
                                            while (while_method_0(v889)){
                                                v887[v889] = v876;
                                                v889 += 1 ;
                                            }
                                            static_array<int,2> v891;
                                            int v893;
                                            v893 = 0;
                                            while (while_method_0(v893)){
                                                bool v895;
                                                v895 = 0 <= v893;
                                                bool v897;
                                                if (v895){
                                                    bool v896;
                                                    v896 = v893 < 2;
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
                                                v900 = v887[v893];
                                                bool v902;
                                                v902 = v893 == v802;
                                                int v904;
                                                if (v902){
                                                    int v903;
                                                    v903 = v900 + 2;
                                                    v904 = v903;
                                                } else {
                                                    v904 = v900;
                                                }
                                                v891[v893] = v904;
                                                v893 += 1 ;
                                            }
                                            v912 = Union4{Union4_2{v799, false, v801, v873, v891, v874}};
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
                                Union6 v806 = v799.case1.v0;
                                switch (v805.tag) {
                                    case 0: { // Call
                                        if (v800){
                                            int v808;
                                            v808 = v802 ^ 1;
                                            v912 = Union4{Union4_2{v799, false, v801, v808, v803, v804}};
                                        } else {
                                            int v810; int v811;
                                            Tuple7 tmp69 = Tuple7{0, 0};
                                            v810 = tmp69.v0; v811 = tmp69.v1;
                                            while (while_method_0(v810)){
                                                bool v813;
                                                v813 = 0 <= v810;
                                                bool v815;
                                                if (v813){
                                                    bool v814;
                                                    v814 = v810 < 2;
                                                    v815 = v814;
                                                } else {
                                                    v815 = false;
                                                }
                                                bool v816;
                                                v816 = v815 == false;
                                                if (v816){
                                                    assert("Index must be in range." && v815);
                                                } else {
                                                }
                                                int v818;
                                                v818 = v803[v810];
                                                bool v820;
                                                v820 = v811 >= v818;
                                                int v821;
                                                if (v820){
                                                    v821 = v811;
                                                } else {
                                                    v821 = v818;
                                                }
                                                v811 = v821;
                                                v810 += 1 ;
                                            }
                                            static_array<int,2> v822;
                                            int v824;
                                            v824 = 0;
                                            while (while_method_0(v824)){
                                                v822[v824] = v811;
                                                v824 += 1 ;
                                            }
                                            v912 = Union4{Union4_4{v799, v800, v801, v802, v822, v804}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v912 = Union4{Union4_5{v799, v800, v801, v802, v803, v804}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v828;
                                        v828 = v804 > 0;
                                        if (v828){
                                            int v829;
                                            v829 = v802 ^ 1;
                                            int v830;
                                            v830 = -1 + v804;
                                            int v831; int v832;
                                            Tuple7 tmp70 = Tuple7{0, 0};
                                            v831 = tmp70.v0; v832 = tmp70.v1;
                                            while (while_method_0(v831)){
                                                bool v834;
                                                v834 = 0 <= v831;
                                                bool v836;
                                                if (v834){
                                                    bool v835;
                                                    v835 = v831 < 2;
                                                    v836 = v835;
                                                } else {
                                                    v836 = false;
                                                }
                                                bool v837;
                                                v837 = v836 == false;
                                                if (v837){
                                                    assert("Index must be in range." && v836);
                                                } else {
                                                }
                                                int v839;
                                                v839 = v803[v831];
                                                bool v841;
                                                v841 = v832 >= v839;
                                                int v842;
                                                if (v841){
                                                    v842 = v832;
                                                } else {
                                                    v842 = v839;
                                                }
                                                v832 = v842;
                                                v831 += 1 ;
                                            }
                                            static_array<int,2> v843;
                                            int v845;
                                            v845 = 0;
                                            while (while_method_0(v845)){
                                                v843[v845] = v832;
                                                v845 += 1 ;
                                            }
                                            static_array<int,2> v847;
                                            int v849;
                                            v849 = 0;
                                            while (while_method_0(v849)){
                                                bool v851;
                                                v851 = 0 <= v849;
                                                bool v853;
                                                if (v851){
                                                    bool v852;
                                                    v852 = v849 < 2;
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
                                                v856 = v843[v849];
                                                bool v858;
                                                v858 = v849 == v802;
                                                int v860;
                                                if (v858){
                                                    int v859;
                                                    v859 = v856 + 4;
                                                    v860 = v859;
                                                } else {
                                                    v860 = v856;
                                                }
                                                v847[v849] = v860;
                                                v849 += 1 ;
                                            }
                                            v912 = Union4{Union4_2{v799, false, v801, v829, v847, v830}};
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
                        v920 = Union3{Union3_1{v912}};
                        break;
                    }
                    case 3: { // T_none
                        v920 = Union3{Union3_0{}};
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
        v16 = v920;
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
    while (while_method_11(v16)){
        Union3 v928;
        switch (v16.tag) {
            case 0: { // None
                v928 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v18 = v16.case1.v0;
                Union14 v768;
                switch (v18.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v737 = v18.case0.v0; bool v738 = v18.case0.v1; static_array<Union6,2> v739 = v18.case0.v2; int v740 = v18.case0.v3; static_array<int,2> v741 = v18.case0.v4; int v742 = v18.case0.v5;
                        curandStatePhilox4_32_10_t & v743 = v3.v5;
                        curandStatePhilox4_32_10_t & v744 = v743;
                        unsigned int & v745 = v3.v0;
                        Union6 v746; unsigned int v747;
                        Tuple6 tmp73 = draw_card_20(v744, v745);
                        v746 = tmp73.v0; v747 = tmp73.v1;
                        v3.v0 = v747;
                        Union7 v748;
                        v748 = Union7{Union7_0{v746}};
                        v14.push(v748);
                        v768 = Union14{Union14_0{v737, v738, v739, v740, v741, v742, v746}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v750 = v3.v5;
                        curandStatePhilox4_32_10_t & v751 = v750;
                        unsigned int & v752 = v3.v0;
                        Union6 v753; unsigned int v754;
                        Tuple6 tmp74 = draw_card_20(v751, v752);
                        v753 = tmp74.v0; v754 = tmp74.v1;
                        v3.v0 = v754;
                        curandStatePhilox4_32_10_t & v755 = v3.v5;
                        curandStatePhilox4_32_10_t & v756 = v755;
                        unsigned int & v757 = v3.v0;
                        Union6 v758; unsigned int v759;
                        Tuple6 tmp75 = draw_card_20(v756, v757);
                        v758 = tmp75.v0; v759 = tmp75.v1;
                        v3.v0 = v759;
                        Union7 v760;
                        v760 = Union7{Union7_2{0, v753}};
                        v14.push(v760);
                        Union7 v761;
                        v761 = Union7{Union7_2{1, v758}};
                        v14.push(v761);
                        v768 = Union14{Union14_1{v753, v758}};
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
                        Union1 v725;
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
                                while (while_method_4(v155)){
                                    float * v157;
                                    v157 = reinterpret_cast<float *>(&v1[4718592ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 32);
                                    int v159;
                                    v159 = 393216 * v155;
                                    float * v160;
                                    v160 = reinterpret_cast<float *>(&v1[0ull]);
                                    float * v162;
                                    v162 = reinterpret_cast<float *>(&v0[0ull]);
                                    float * v164;
                                    v164 = reinterpret_cast<float *>(&v2[0ull]);
                                    assert("Tensor range check" && 0 <= v155 && v155 < 32);
                                    int v166;
                                    v166 = 8192 * v155;
                                    float * v167;
                                    v167 = reinterpret_cast<float *>(&v1[3145728ull]);
                                    block_matmul_23(v167, v162, v166, v160);
                                    block_row_map_24(v157, v159, v167);
                                    int * v169;
                                    v169 = reinterpret_cast<int *>(&v0[1048576ull]);
                                    bool * v171;
                                    v171 = reinterpret_cast<bool *>(&v0[1048592ull]);
                                    float * v173;
                                    v173 = reinterpret_cast<float *>(&v0[1048624ull]);
                                    float * v175;
                                    v175 = reinterpret_cast<float *>(&v0[1048752ull]);
                                    double * v177;
                                    v177 = reinterpret_cast<double *>(&v1[55050240ull]);
                                    double * v179;
                                    v179 = reinterpret_cast<double *>(&v1[58195968ull]);
                                    v155 += 1 ;
                                }
                                __syncthreads();
                                int * v181;
                                v181 = reinterpret_cast<int *>(&v0[1048576ull]);
                                bool * v183;
                                v183 = reinterpret_cast<bool *>(&v0[1048592ull]);
                                float * v185;
                                v185 = reinterpret_cast<float *>(&v0[1048624ull]);
                                float * v187;
                                v187 = reinterpret_cast<float *>(&v0[1048752ull]);
                                int v189;
                                v189 = 0;
                                int v190;
                                v190 = 32;
                                int v191;
                                v191 = int_range_22(v190, v189, v85);
                                extern __shared__ unsigned char v192[];
                                int * v193;
                                v193 = reinterpret_cast<int *>(&v192[0ull]);
                                int v195;
                                v195 = threadIdx.x;
                                bool v196;
                                v196 = v195 == 0;
                                if (v196){
                                    v193[0] = v191;
                                } else {
                                }
                                __syncthreads();
                                int v197;
                                v197 = v193[0];
                                __syncthreads();
                                float * v198;
                                v198 = reinterpret_cast<float *>(&v1[4718592ull]);
                                assert("Tensor range check" && 0 <= v197 && v197 < 32);
                                int v200;
                                v200 = 393216 * v197;
                                int v201;
                                v201 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v201 && v201 < 24);
                                int v202;
                                v202 = 16384 * v201;
                                int v203;
                                v203 = v202 + v200;
                                int v204;
                                v204 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v204 && v204 < 256);
                                int v205;
                                v205 = 64 * v204;
                                int v206;
                                v206 = v205 + v203;
                                float * v207;
                                v207 = v198+v206;
                                int v209;
                                v209 = sizeof(float *);
                                unsigned long long v210;
                                v210 = (unsigned long long)v209;
                                unsigned long long v211;
                                v211 = 256ull * v210;
                                unsigned long long v212;
                                v212 = v211 + 16ull;
                                unsigned long long v213;
                                v213 = v212 - 1ull;
                                unsigned long long v214;
                                v214 = v213 % 16ull;
                                unsigned long long v215;
                                v215 = v213 - v214;
                                unsigned long long v216;
                                v216 = v215 + 1024ull;
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
                                bool v222;
                                v222 = v221 <= 98304ull;
                                bool v223;
                                v223 = v222 == false;
                                if (v223){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v222);
                                } else {
                                }
                                extern __shared__ unsigned char v225[];
                                bool v226;
                                v226 = v221 <= v221;
                                bool v227;
                                v227 = v226 == false;
                                if (v227){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v226);
                                } else {
                                }
                                float * * v229;
                                v229 = reinterpret_cast<float * *>(&v225[0ull]);
                                float * v231;
                                v231 = reinterpret_cast<float *>(&v225[v215]);
                                int * v233;
                                v233 = reinterpret_cast<int *>(&v225[v220]);
                                int v235;
                                v235 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v235 && v235 < 256);
                                v229[v235] = v207;
                                __syncthreads();
                                bool v236;
                                v236 = 0 <= v235;
                                bool v237;
                                v237 = v236 == false;
                                if (v237){
                                    assert("The index needs to be zero or positive." && v236);
                                } else {
                                }
                                int v239;
                                v239 = v235 % 16;
                                int v240;
                                v240 = v235 / 16;
                                bool v241;
                                v241 = v240 < 16;
                                bool v242;
                                v242 = v241 == false;
                                if (v242){
                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v241);
                                } else {
                                }
                                assert("Tensor range check" && 0 <= v240 && v240 < 16);
                                int v244;
                                v244 = 0;
                                while (while_method_9(v244)){
                                    bool v246;
                                    v246 = 0 <= v240;
                                    bool v247;
                                    v247 = v246 && v241;
                                    bool v248;
                                    v248 = v247 == false;
                                    if (v248){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v247);
                                    } else {
                                    }
                                    bool v250;
                                    v250 = 0 <= v244;
                                    bool v252;
                                    if (v250){
                                        bool v251;
                                        v251 = v244 < 16;
                                        v252 = v251;
                                    } else {
                                        v252 = false;
                                    }
                                    bool v253;
                                    v253 = v252 == false;
                                    if (v253){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v252);
                                    } else {
                                    }
                                    int v255;
                                    v255 = v244 * 16;
                                    int v256;
                                    v256 = v255 + v240;
                                    assert("Tensor range check" && 0 <= v244 && v244 < 16);
                                    int v257;
                                    v257 = 16 * v244;
                                    int v258;
                                    v258 = v257 + v240;
                                    float * v259;
                                    v259 = v229[v258];
                                    int v260;
                                    v260 = blockIdx.x;
                                    int v261;
                                    v261 = v260 * 256;
                                    int v262;
                                    v262 = v261 + v256;
                                    assert("Tensor range check" && 0 <= v239 && v239 < 16);
                                    int v263;
                                    v263 = 4 * v239;
                                    float v264[4];
                                    int v265[4];
                                    int v266;
                                    v266 = 0;
                                    while (while_method_5(v266)){
                                        assert("Tensor range check" && 0 <= v266 && v266 < 1);
                                        int v268;
                                        v268 = 4 * v266;
                                        assert("Tensor range check" && 0 <= v266 && v266 < 1);
                                        int v269;
                                        v269 = 64 * v266;
                                        int v270;
                                        v270 = v269 + v263;
                                        int4* v271;
                                        v271 = reinterpret_cast<int4*>(v259 + v270);
                                        int4* v272;
                                        v272 = reinterpret_cast<int4*>(v264 + v268);
                                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v271) % 16 == 0 && reinterpret_cast<unsigned long long>(v272) % 16 == 0);
                                        *v272 = *v271;
                                        v266 += 1 ;
                                    }
                                    int v273;
                                    v273 = 0;
                                    while (while_method_5(v273)){
                                        int v275;
                                        v275 = 0;
                                        while (while_method_8(v275)){
                                            bool v277;
                                            v277 = 0 <= v275;
                                            bool v279;
                                            if (v277){
                                                bool v278;
                                                v278 = v275 < 4;
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
                                            bool v282;
                                            v282 = 0 <= v239;
                                            bool v284;
                                            if (v282){
                                                bool v283;
                                                v283 = v239 < 16;
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
                                            int v287;
                                            v287 = v239 * 4;
                                            int v288;
                                            v288 = v275 + v287;
                                            bool v289;
                                            v289 = 0 <= v273;
                                            bool v291;
                                            if (v289){
                                                bool v290;
                                                v290 = v273 < 1;
                                                v291 = v290;
                                            } else {
                                                v291 = false;
                                            }
                                            bool v292;
                                            v292 = v291 == false;
                                            if (v292){
                                                assert("The indices should be inside the range of the dimension." && v291);
                                            } else {
                                            }
                                            int v294;
                                            v294 = v273 * 64;
                                            int v295;
                                            v295 = v288 + v294;
                                            assert("Tensor range check" && 0 <= v273 && v273 < 1);
                                            assert("Tensor range check" && 0 <= v275 && v275 < 4);
                                            int v296;
                                            v296 = 4 * v273;
                                            int v297;
                                            v297 = v296 + v275;
                                            v265[v297] = v295;
                                            v275 += 1 ;
                                        }
                                        v273 += 1 ;
                                    }
                                    float v298[4];
                                    float v299;
                                    v299 = 0.0f;
                                    int v300;
                                    v300 = 0;
                                    while (while_method_5(v300)){
                                        assert("Tensor range check" && 0 <= v300 && v300 < 1);
                                        int v302;
                                        v302 = 4 * v300;
                                        assert("Tensor range check" && 0 <= v300 && v300 < 1);
                                        float v303;
                                        v303 = 0.0f;
                                        int v304;
                                        v304 = 0;
                                        while (while_method_8(v304)){
                                            assert("Tensor range check" && 0 <= v304 && v304 < 4);
                                            int v306;
                                            v306 = v304 + v302;
                                            float v307;
                                            v307 = v264[v306];
                                            float v308;
                                            v308 = v303 + v307;
                                            v303 = v308;
                                            v304 += 1 ;
                                        }
                                        auto v309 = cooperative_groups::coalesced_threads();
                                        int v310;
                                        v310 = threadIdx.x;
                                        int v311;
                                        v311 = v310 / 16;
                                        auto v312 = cooperative_groups::labeled_partition(v309,v311);
                                        Closure2 v313{};
                                        float v314;
                                        v314 = cooperative_groups::inclusive_scan(v312, v303, v313);
                                        float v315;
                                        v315 = v312.shfl_up(v314,1);
                                        bool v316;
                                        v316 = v312.thread_rank() == 0;
                                        float v317;
                                        if (v316){
                                            v317 = 0.0f;
                                        } else {
                                            v317 = v315;
                                        }
                                        float v318;
                                        v318 = v312.shfl(v314,v312.num_threads()-1);
                                        float v319;
                                        v319 = v299 + v317;
                                        float v320;
                                        v320 = v319;
                                        int v321;
                                        v321 = 0;
                                        while (while_method_8(v321)){
                                            assert("Tensor range check" && 0 <= v321 && v321 < 4);
                                            int v323;
                                            v323 = v321 + v302;
                                            float v324;
                                            v324 = v264[v323];
                                            float v325;
                                            v325 = v320 + v324;
                                            assert("Tensor range check" && 0 <= v321 && v321 < 4);
                                            v298[v323] = v325;
                                            v320 = v325;
                                            v321 += 1 ;
                                        }
                                        float v326;
                                        v326 = v299 + v318;
                                        v299 = v326;
                                        v300 += 1 ;
                                    }
                                    float v327[4];
                                    bool v328[4];
                                    int v329;
                                    v329 = 0;
                                    while (while_method_5(v329)){
                                        int v331;
                                        v331 = 0;
                                        while (while_method_8(v331)){
                                            assert("Tensor range check" && 0 <= v329 && v329 < 1);
                                            assert("Tensor range check" && 0 <= v331 && v331 < 4);
                                            int v333;
                                            v333 = 4 * v329;
                                            int v334;
                                            v334 = v333 + v331;
                                            float v335;
                                            v335 = v298[v334];
                                            float v336;
                                            v336 = v264[v334];
                                            bool v337;
                                            v337 = v336 > 0.0f;
                                            assert("Tensor range check" && 0 <= v329 && v329 < 1);
                                            assert("Tensor range check" && 0 <= v331 && v331 < 4);
                                            v327[v334] = v335;
                                            v328[v334] = v337;
                                            v331 += 1 ;
                                        }
                                        v329 += 1 ;
                                    }
                                    float v338; bool v339;
                                    Tuple8 tmp76 = Tuple8{-1.0f / 0.0f, false};
                                    v338 = tmp76.v0; v339 = tmp76.v1;
                                    int v340;
                                    v340 = 0;
                                    while (while_method_5(v340)){
                                        int v342;
                                        v342 = 0;
                                        while (while_method_8(v342)){
                                            assert("Tensor range check" && 0 <= v340 && v340 < 1);
                                            assert("Tensor range check" && 0 <= v342 && v342 < 4);
                                            int v344;
                                            v344 = 4 * v340;
                                            int v345;
                                            v345 = v344 + v342;
                                            float v346;
                                            v346 = v327[v345];
                                            bool v347;
                                            v347 = v328[v345];
                                            float v354; bool v355;
                                            if (v339){
                                                if (v347){
                                                    bool v348;
                                                    v348 = v338 >= v346;
                                                    float v349;
                                                    if (v348){
                                                        v349 = v338;
                                                    } else {
                                                        v349 = v346;
                                                    }
                                                    v354 = v349; v355 = true;
                                                } else {
                                                    v354 = v338; v355 = v339;
                                                }
                                            } else {
                                                if (v347){
                                                    v354 = v346; v355 = v347;
                                                } else {
                                                    v354 = v338; v355 = v339;
                                                }
                                            }
                                            v338 = v354;
                                            v339 = v355;
                                            v342 += 1 ;
                                        }
                                        v340 += 1 ;
                                    }
                                    auto v356 = cooperative_groups::coalesced_threads();
                                    int v357;
                                    v357 = threadIdx.x;
                                    int v358;
                                    v358 = v357 / 16;
                                    auto v359 = cooperative_groups::labeled_partition(v356,v358);
                                    Closure3 v360{};
                                    float v361; bool v362;
                                    Tuple8 tmp77 = cooperative_groups::reduce(v359, Tuple8{v338, v339}, v360);
                                    v361 = tmp77.v0; v362 = tmp77.v1;
                                    bool v363;
                                    v363 = v362 == false;
                                    if (v363){
                                        int v364;
                                        v364 = threadIdx.x;
                                        int v365;
                                        v365 = blockIdx.x;
                                        int v366;
                                        v366 = v365 * 256;
                                        int v367;
                                        v367 = v364 + v366;
                                        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v368 = console_lock;
                                        auto v369 = cooperative_groups::coalesced_threads();
                                        v368.acquire();
                                        int v370;
                                        v370 = 0;
                                        printf("{%s = %d; %s = %c","tid", v367, "x'", '[');
                                        int v371;
                                        v371 = 0;
                                        while (while_method_5(v371)){
                                            int v373;
                                            v373 = v370;
                                            bool v374;
                                            v374 = v373 >= 100;
                                            if (v374){
                                                printf("%s"," ...");
                                                break;
                                            } else {
                                            }
                                            bool v375;
                                            v375 = v371 == 0;
                                            bool v376;
                                            v376 = v375 != true;
                                            if (v376){
                                                printf("%s","; ");
                                            } else {
                                            }
                                            printf("%c",'[');
                                            int v377;
                                            v377 = 0;
                                            while (while_method_8(v377)){
                                                int v379;
                                                v379 = v370;
                                                bool v380;
                                                v380 = v379 >= 100;
                                                if (v380){
                                                    printf("%s"," ...");
                                                    break;
                                                } else {
                                                }
                                                bool v381;
                                                v381 = v377 == 0;
                                                bool v382;
                                                v382 = v381 != true;
                                                if (v382){
                                                    printf("%s","; ");
                                                } else {
                                                }
                                                int v383;
                                                v383 = v370 + 1;
                                                v370 = v383;
                                                int v384;
                                                v384 = v371 * 4;
                                                int v385;
                                                v385 = v384 + v377;
                                                float v386;
                                                v386 = v327[v385];
                                                bool v387;
                                                v387 = v328[v385];
                                                const char * v390;
                                                if (v387){
                                                    const char * v388;
                                                    v388 = "true";
                                                    v390 = v388;
                                                } else {
                                                    const char * v389;
                                                    v389 = "false";
                                                    v390 = v389;
                                                }
                                                printf("%f, %s",v386, v390);
                                                v377 += 1 ;
                                            }
                                            printf("%c",']');
                                            v371 += 1 ;
                                        }
                                        printf("%c",']');
                                        printf("}\n");
                                        v368.release();
                                        v369.sync() ;
                                    } else {
                                    }
                                    if (v363){
                                        assert("The local reduce must be true." && v362);
                                    } else {
                                    }
                                    float v426[4];
                                    int v427[4];
                                    int v428;
                                    v428 = 0;
                                    while (while_method_5(v428)){
                                        int v430;
                                        v430 = 0;
                                        while (while_method_8(v430)){
                                            assert("Tensor range check" && 0 <= v428 && v428 < 1);
                                            assert("Tensor range check" && 0 <= v430 && v430 < 4);
                                            int v432;
                                            v432 = 4 * v428;
                                            int v433;
                                            v433 = v432 + v430;
                                            int v434;
                                            v434 = v265[v433];
                                            float v435;
                                            v435 = curand_uniform(&v85);
                                            assert("Tensor range check" && 0 <= v428 && v428 < 1);
                                            assert("Tensor range check" && 0 <= v430 && v430 < 4);
                                            v426[v433] = v435;
                                            v427[v433] = v434;
                                            v430 += 1 ;
                                        }
                                        v428 += 1 ;
                                    }
                                    float v436; int v437;
                                    Tuple9 tmp78 = Tuple9{0.0f, 2147483647};
                                    v436 = tmp78.v0; v437 = tmp78.v1;
                                    int v438;
                                    v438 = 0;
                                    while (while_method_5(v438)){
                                        int v440;
                                        v440 = 0;
                                        while (while_method_8(v440)){
                                            assert("Tensor range check" && 0 <= v438 && v438 < 1);
                                            assert("Tensor range check" && 0 <= v440 && v440 < 4);
                                            int v442;
                                            v442 = 4 * v438;
                                            int v443;
                                            v443 = v442 + v440;
                                            float v444;
                                            v444 = v426[v443];
                                            int v445;
                                            v445 = v427[v443];
                                            bool v446;
                                            v446 = v437 < v445;
                                            float v447; int v448;
                                            if (v446){
                                                v447 = v436; v448 = v437;
                                            } else {
                                                v447 = v444; v448 = v445;
                                            }
                                            v436 = v447;
                                            v437 = v448;
                                            v440 += 1 ;
                                        }
                                        v438 += 1 ;
                                    }
                                    auto v449 = cooperative_groups::coalesced_threads();
                                    int v450;
                                    v450 = threadIdx.x;
                                    int v451;
                                    v451 = v450 / 16;
                                    auto v452 = cooperative_groups::labeled_partition(v449,v451);
                                    Closure4 v453{};
                                    float v454; int v455;
                                    Tuple9 tmp79 = cooperative_groups::reduce(v452, Tuple9{v436, v437}, v453);
                                    v454 = tmp79.v0; v455 = tmp79.v1;
                                    float v456;
                                    v456 = v361 * v454;
                                    int v457[4];
                                    bool v458[4];
                                    int v459;
                                    v459 = 0;
                                    while (while_method_5(v459)){
                                        int v461;
                                        v461 = 0;
                                        while (while_method_8(v461)){
                                            assert("Tensor range check" && 0 <= v459 && v459 < 1);
                                            assert("Tensor range check" && 0 <= v461 && v461 < 4);
                                            int v463;
                                            v463 = 4 * v459;
                                            int v464;
                                            v464 = v463 + v461;
                                            float v465;
                                            v465 = v327[v464];
                                            bool v466;
                                            v466 = v328[v464];
                                            int v467;
                                            v467 = v265[v464];
                                            int v470; bool v471;
                                            if (v466){
                                                float v468;
                                                v468 = v465 - v456;
                                                bool v469;
                                                v469 = v468 >= 0.0f;
                                                v470 = v467; v471 = v469;
                                            } else {
                                                v470 = 2147483647; v471 = false;
                                            }
                                            assert("Tensor range check" && 0 <= v459 && v459 < 1);
                                            assert("Tensor range check" && 0 <= v461 && v461 < 4);
                                            v457[v464] = v470;
                                            v458[v464] = v471;
                                            v461 += 1 ;
                                        }
                                        v459 += 1 ;
                                    }
                                    int v472; bool v473;
                                    Tuple10 tmp80 = Tuple10{2147483647, false};
                                    v472 = tmp80.v0; v473 = tmp80.v1;
                                    int v474;
                                    v474 = 0;
                                    while (while_method_5(v474)){
                                        int v476;
                                        v476 = 0;
                                        while (while_method_8(v476)){
                                            assert("Tensor range check" && 0 <= v474 && v474 < 1);
                                            assert("Tensor range check" && 0 <= v476 && v476 < 4);
                                            int v478;
                                            v478 = 4 * v474;
                                            int v479;
                                            v479 = v478 + v476;
                                            int v480;
                                            v480 = v457[v479];
                                            bool v481;
                                            v481 = v458[v479];
                                            int v488; bool v489;
                                            if (v473){
                                                if (v481){
                                                    bool v482;
                                                    v482 = v472 < v480;
                                                    int v483;
                                                    if (v482){
                                                        v483 = v472;
                                                    } else {
                                                        v483 = v480;
                                                    }
                                                    v488 = v483; v489 = true;
                                                } else {
                                                    v488 = v472; v489 = v473;
                                                }
                                            } else {
                                                if (v481){
                                                    v488 = v480; v489 = v481;
                                                } else {
                                                    v488 = v472; v489 = v473;
                                                }
                                            }
                                            v472 = v488;
                                            v473 = v489;
                                            v476 += 1 ;
                                        }
                                        v474 += 1 ;
                                    }
                                    auto v490 = cooperative_groups::coalesced_threads();
                                    int v491;
                                    v491 = threadIdx.x;
                                    int v492;
                                    v492 = v491 / 16;
                                    auto v493 = cooperative_groups::labeled_partition(v490,v492);
                                    Closure5 v494{};
                                    int v495; bool v496;
                                    Tuple10 tmp81 = cooperative_groups::reduce(v493, Tuple10{v472, v473}, v494);
                                    v495 = tmp81.v0; v496 = tmp81.v1;
                                    bool v497;
                                    v497 = v496 == false;
                                    if (v497){
                                        int v498;
                                        v498 = threadIdx.x;
                                        int v499;
                                        v499 = blockIdx.x;
                                        int v500;
                                        v500 = v499 * 256;
                                        int v501;
                                        v501 = v498 + v500;
                                        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v502 = console_lock;
                                        auto v503 = cooperative_groups::coalesced_threads();
                                        v502.acquire();
                                        int v504;
                                        v504 = 0;
                                        printf("{%s = %d; %s = %c","tid", v501, "x'", '[');
                                        int v505;
                                        v505 = 0;
                                        while (while_method_5(v505)){
                                            int v507;
                                            v507 = v504;
                                            bool v508;
                                            v508 = v507 >= 100;
                                            if (v508){
                                                printf("%s"," ...");
                                                break;
                                            } else {
                                            }
                                            bool v509;
                                            v509 = v505 == 0;
                                            bool v510;
                                            v510 = v509 != true;
                                            if (v510){
                                                printf("%s","; ");
                                            } else {
                                            }
                                            printf("%c",'[');
                                            int v511;
                                            v511 = 0;
                                            while (while_method_8(v511)){
                                                int v513;
                                                v513 = v504;
                                                bool v514;
                                                v514 = v513 >= 100;
                                                if (v514){
                                                    printf("%s"," ...");
                                                    break;
                                                } else {
                                                }
                                                bool v515;
                                                v515 = v511 == 0;
                                                bool v516;
                                                v516 = v515 != true;
                                                if (v516){
                                                    printf("%s","; ");
                                                } else {
                                                }
                                                int v517;
                                                v517 = v504 + 1;
                                                v504 = v517;
                                                int v518;
                                                v518 = v505 * 4;
                                                int v519;
                                                v519 = v518 + v511;
                                                int v520;
                                                v520 = v457[v519];
                                                bool v521;
                                                v521 = v458[v519];
                                                const char * v524;
                                                if (v521){
                                                    const char * v522;
                                                    v522 = "true";
                                                    v524 = v522;
                                                } else {
                                                    const char * v523;
                                                    v523 = "false";
                                                    v524 = v523;
                                                }
                                                printf("%d, %s",v520, v524);
                                                v511 += 1 ;
                                            }
                                            printf("%c",']');
                                            v505 += 1 ;
                                        }
                                        printf("%c",']');
                                        printf("}\n");
                                        v502.release();
                                        v503.sync() ;
                                    } else {
                                    }
                                    if (v497){
                                        assert("The local reduce must be true." && v496);
                                    } else {
                                    }
                                    float v560; int v561;
                                    Tuple9 tmp82 = Tuple9{0.0f, 2147483647};
                                    v560 = tmp82.v0; v561 = tmp82.v1;
                                    int v562;
                                    v562 = 0;
                                    while (while_method_5(v562)){
                                        int v564;
                                        v564 = 0;
                                        while (while_method_8(v564)){
                                            assert("Tensor range check" && 0 <= v562 && v562 < 1);
                                            assert("Tensor range check" && 0 <= v564 && v564 < 4);
                                            int v566;
                                            v566 = 4 * v562;
                                            int v567;
                                            v567 = v566 + v564;
                                            float v568;
                                            v568 = v264[v567];
                                            int v569;
                                            v569 = v265[v567];
                                            bool v570;
                                            v570 = v561 == v495;
                                            float v574; int v575;
                                            if (v570){
                                                v574 = v560; v575 = v561;
                                            } else {
                                                bool v571;
                                                v571 = v569 == v495;
                                                if (v571){
                                                    v574 = v568; v575 = v569;
                                                } else {
                                                    v574 = v560; v575 = v561;
                                                }
                                            }
                                            v560 = v574;
                                            v561 = v575;
                                            v564 += 1 ;
                                        }
                                        v562 += 1 ;
                                    }
                                    auto v576 = cooperative_groups::coalesced_threads();
                                    int v577;
                                    v577 = threadIdx.x;
                                    int v578;
                                    v578 = v577 / 16;
                                    auto v579 = cooperative_groups::labeled_partition(v576,v578);
                                    Closure6 v580{v495};
                                    float v581; int v582;
                                    Tuple9 tmp83 = cooperative_groups::reduce(v579, Tuple9{v560, v561}, v580);
                                    v581 = tmp83.v0; v582 = tmp83.v1;
                                    bool v583;
                                    v583 = v582 == 2147483647;
                                    bool v584;
                                    v584 = v583 != true;
                                    bool v585;
                                    v585 = v584 == false;
                                    if (v585){
                                        assert("Expected a valid action id in get_prob." && v584);
                                    } else {
                                    }
                                    int v587;
                                    v587 = 0;
                                    while (while_method_5(v587)){
                                        assert("Tensor range check" && 0 <= v587 && v587 < 1);
                                        assert("Tensor range check" && 0 <= v587 && v587 < 1);
                                        v587 += 1 ;
                                    }
                                    assert("Tensor range check" && 0 <= v256 && v256 < 256);
                                    v231[v256] = v581;
                                    v233[v256] = v495;
                                    v244 += 1 ;
                                }
                                __syncthreads();
                                assert("Tensor range check" && 0 <= v235 && v235 < 256);
                                float v589;
                                v589 = v231[v235];
                                int v590;
                                v590 = v233[v235];
                                __syncthreads();
                                extern __shared__ unsigned char v591[];
                                float * v592;
                                v592 = reinterpret_cast<float *>(&v591[0ull]);
                                int * v594;
                                v594 = reinterpret_cast<int *>(&v591[16ull]);
                                int v596;
                                v596 = threadIdx.x;
                                bool v597;
                                v597 = v596 == 0;
                                if (v597){
                                    v592[0] = v589;
                                    v594[0] = v590;
                                } else {
                                }
                                __syncthreads();
                                float v598;
                                v598 = v592[0];
                                int v599;
                                v599 = v594[0];
                                __syncthreads();
                                double * v600;
                                v600 = reinterpret_cast<double *>(&v1[55050240ull]);
                                double * v602;
                                v602 = reinterpret_cast<double *>(&v1[58195968ull]);
                                int v604;
                                v604 = threadIdx.x;
                                int v605;
                                v605 = blockIdx.x;
                                int v606;
                                v606 = v605 * 256;
                                int v607;
                                v607 = v604 + v606;
                                int v608;
                                v608 = 0;
                                while (while_method_4(v608)){
                                    float * v610;
                                    v610 = reinterpret_cast<float *>(&v1[4718592ull]);
                                    int v612;
                                    v612 = blockIdx.x;
                                    int v613;
                                    v613 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v608 && v608 < 32);
                                    assert("Tensor range check" && 0 <= v612 && v612 < 24);
                                    assert("Tensor range check" && 0 <= v613 && v613 < 256);
                                    assert("Tensor range check" && 0 <= v599 && v599 < 64);
                                    int v614;
                                    v614 = 64 * v613;
                                    int v615;
                                    v615 = v614 + v599;
                                    int v616;
                                    v616 = 16384 * v612;
                                    int v617;
                                    v617 = v616 + v615;
                                    int v618;
                                    v618 = 393216 * v608;
                                    int v619;
                                    v619 = v618 + v617;
                                    float v620;
                                    v620 = v610[v619];
                                    double v621;
                                    v621 = (double)v598;
                                    double v622;
                                    v622 = log(v621);
                                    double v623;
                                    v623 = (double)v620;
                                    double v624;
                                    v624 = log(v623);
                                    assert("Tensor range check" && 0 <= v608 && v608 < 32);
                                    assert("Tensor range check" && 0 <= v607 && v607 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    int v625;
                                    v625 = 2 * v607;
                                    int v626;
                                    v626 = v625 + v71;
                                    int v627;
                                    v627 = 12288 * v608;
                                    int v628;
                                    v628 = v627 + v626;
                                    double v629;
                                    v629 = v600[v628];
                                    double v630;
                                    v630 = v602[v628];
                                    double v631;
                                    v631 = v624 + v629;
                                    double v632;
                                    v632 = v622 + v630;
                                    bool v633;
                                    v633 = isnan(v632);
                                    bool v634;
                                    v634 = v633 == false;
                                    bool v635;
                                    v635 = v634 == false;
                                    if (v635){
                                        assert("The sampling log probability shouldn't be nan." && v634);
                                    } else {
                                    }
                                    bool v637;
                                    v637 = isnan(v631);
                                    bool v638;
                                    v638 = v637 == false;
                                    bool v639;
                                    v639 = v638 == false;
                                    if (v639){
                                        assert("The policy log probability shouldn't be nan." && v638);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v608 && v608 < 32);
                                    assert("Tensor range check" && 0 <= v607 && v607 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    v600[v628] = v631;
                                    v602[v628] = v632;
                                    v608 += 1 ;
                                }
                                bool v641;
                                v641 = 0 == v599;
                                Union12 v650;
                                if (v641){
                                    v650 = Union12{Union12_1{}};
                                } else {
                                    bool v643;
                                    v643 = 1 == v599;
                                    if (v643){
                                        v650 = Union12{Union12_0{}};
                                    } else {
                                        bool v645;
                                        v645 = 2 == v599;
                                        if (v645){
                                            v650 = Union12{Union12_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v650.tag) {
                                    case 0: { // AA_Call
                                        v725 = Union1{Union1_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v651;
                                        v651 = v72[0];
                                        int v653; int v654;
                                        Tuple7 tmp84 = Tuple7{1, v651};
                                        v653 = tmp84.v0; v654 = tmp84.v1;
                                        while (while_method_0(v653)){
                                            bool v656;
                                            v656 = 0 <= v653;
                                            bool v658;
                                            if (v656){
                                                bool v657;
                                                v657 = v653 < 2;
                                                v658 = v657;
                                            } else {
                                                v658 = false;
                                            }
                                            bool v659;
                                            v659 = v658 == false;
                                            if (v659){
                                                assert("Index must be in range." && v658);
                                            } else {
                                            }
                                            int v661;
                                            v661 = v72[v653];
                                            bool v663;
                                            v663 = v654 >= v661;
                                            int v664;
                                            if (v663){
                                                v664 = v654;
                                            } else {
                                                v664 = v661;
                                            }
                                            v654 = v664;
                                            v653 += 1 ;
                                        }
                                        bool v666;
                                        if (v75){
                                            bool v665;
                                            v665 = v71 < 2;
                                            v666 = v665;
                                        } else {
                                            v666 = false;
                                        }
                                        bool v667;
                                        v667 = v666 == false;
                                        if (v667){
                                            assert("Index must be in range." && v666);
                                        } else {
                                        }
                                        int v669;
                                        v669 = v72[v71];
                                        bool v671;
                                        v671 = v669 == v654;
                                        if (v671){
                                            v725 = Union1{Union1_0{}};
                                        } else {
                                            v725 = Union1{Union1_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v676;
                                        v676 = v73 > 0;
                                        if (v676){
                                            v725 = Union1{Union1_2{}};
                                        } else {
                                            v725 = Union1{Union1_0{}};
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
                                curandStatePhilox4_32_10_t & v683 = v3.v5;
                                curandStatePhilox4_32_10_t & v684 = v683;
                                static_array_list<Union1,3> v685;
                                v685 = static_array_list<Union1,3>{};
                                v685.unsafe_set_length(1);
                                Union1 v687;
                                v687 = Union1{Union1_0{}};
                                v685[0] = v687;
                                int v689;
                                v689 = v72[0];
                                int v691;
                                v691 = v72[1];
                                bool v693;
                                v693 = v689 == v691;
                                bool v694;
                                v694 = v693 != true;
                                if (v694){
                                    Union1 v695;
                                    v695 = Union1{Union1_1{}};
                                    v685.push(v695);
                                } else {
                                }
                                bool v696;
                                v696 = v73 > 0;
                                if (v696){
                                    Union1 v697;
                                    v697 = Union1{Union1_2{}};
                                    v685.push(v697);
                                } else {
                                }
                                int v698;
                                v698 = v685.length;
                                int v699;
                                v699 = v698 - 1;
                                int v700;
                                v700 = 0;
                                while (while_method_1(v699, v700)){
                                    int v702;
                                    v702 = v685.length;
                                    int v703;
                                    v703 = int_range_22(v702, v700, v684);
                                    Union1 v704;
                                    v704 = v685[v700];
                                    Union1 v706;
                                    v706 = v685[v703];
                                    v685[v700] = v706;
                                    v685[v703] = v704;
                                    v700 += 1 ;
                                }
                                Union1 v708;
                                v708 = v685.pop();
                                int v709;
                                v709 = sizeof(Union1);
                                unsigned long long v710;
                                v710 = (unsigned long long)v709;
                                bool v711;
                                v711 = v710 <= 98304ull;
                                bool v712;
                                v712 = v711 == false;
                                if (v712){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v711);
                                } else {
                                }
                                extern __shared__ unsigned char v714[];
                                bool v715;
                                v715 = v710 <= v710;
                                bool v716;
                                v716 = v715 == false;
                                if (v716){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v715);
                                } else {
                                }
                                Union1 * v718;
                                v718 = reinterpret_cast<Union1 *>(&v714[0ull]);
                                int v720;
                                v720 = threadIdx.x;
                                bool v721;
                                v721 = v720 == 0;
                                if (v721){
                                    v718[0] = v708;
                                } else {
                                }
                                __syncthreads();
                                Union1 v722;
                                v722 = v718[0];
                                __syncthreads();
                                v725 = v722;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union7 v726;
                        v726 = Union7{Union7_1{v71, v725}};
                        v14.push(v726);
                        v768 = Union14{Union14_2{v68, v69, v70, v71, v72, v73, v725}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v728 = v18.case3.v0; bool v729 = v18.case3.v1; static_array<Union6,2> v730 = v18.case3.v2; int v731 = v18.case3.v3; static_array<int,2> v732 = v18.case3.v4; int v733 = v18.case3.v5; Union1 v734 = v18.case3.v6;
                        Union7 v735;
                        v735 = Union7{Union7_1{v731, v734}};
                        v14.push(v735);
                        v768 = Union14{Union14_2{v728, v729, v730, v731, v732, v733, v734}};
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
                        v768 = Union14{Union14_3{}};
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
                        v768 = Union14{Union14_3{}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                switch (v768.tag) {
                    case 0: { // T_game_chance_community_card
                        Union5 v770 = v768.case0.v0; bool v771 = v768.case0.v1; static_array<Union6,2> v772 = v768.case0.v2; int v773 = v768.case0.v3; static_array<int,2> v774 = v768.case0.v4; int v775 = v768.case0.v5; Union6 v776 = v768.case0.v6;
                        int v777;
                        v777 = 2;
                        int v778; int v779;
                        Tuple7 tmp85 = Tuple7{0, 0};
                        v778 = tmp85.v0; v779 = tmp85.v1;
                        while (while_method_0(v778)){
                            bool v781;
                            v781 = 0 <= v778;
                            bool v783;
                            if (v781){
                                bool v782;
                                v782 = v778 < 2;
                                v783 = v782;
                            } else {
                                v783 = false;
                            }
                            bool v784;
                            v784 = v783 == false;
                            if (v784){
                                assert("Index must be in range." && v783);
                            } else {
                            }
                            int v786;
                            v786 = v774[v778];
                            bool v788;
                            v788 = v779 >= v786;
                            int v789;
                            if (v788){
                                v789 = v779;
                            } else {
                                v789 = v786;
                            }
                            v779 = v789;
                            v778 += 1 ;
                        }
                        static_array<int,2> v790;
                        int v792;
                        v792 = 0;
                        while (while_method_0(v792)){
                            v790[v792] = v779;
                            v792 += 1 ;
                        }
                        Union5 v794;
                        v794 = Union5{Union5_1{v776}};
                        Union4 v795;
                        v795 = Union4{Union4_2{v794, true, v772, 0, v790, v777}};
                        v928 = Union3{Union3_1{v795}};
                        break;
                    }
                    case 1: { // T_game_chance_init
                        Union6 v797 = v768.case1.v0; Union6 v798 = v768.case1.v1;
                        int v799;
                        v799 = 2;
                        static_array<int,2> v800;
                        v800[0] = 1;
                        v800[1] = 1;
                        static_array<Union6,2> v802;
                        v802[0] = v797;
                        v802[1] = v798;
                        Union5 v804;
                        v804 = Union5{Union5_0{}};
                        Union4 v805;
                        v805 = Union4{Union4_2{v804, true, v802, 0, v800, v799}};
                        v928 = Union3{Union3_1{v805}};
                        break;
                    }
                    case 2: { // T_game_round
                        Union5 v807 = v768.case2.v0; bool v808 = v768.case2.v1; static_array<Union6,2> v809 = v768.case2.v2; int v810 = v768.case2.v3; static_array<int,2> v811 = v768.case2.v4; int v812 = v768.case2.v5; Union1 v813 = v768.case2.v6;
                        Union4 v920;
                        switch (v807.tag) {
                            case 0: { // None
                                switch (v813.tag) {
                                    case 0: { // Call
                                        if (v808){
                                            int v876;
                                            v876 = v810 ^ 1;
                                            v920 = Union4{Union4_2{v807, false, v809, v876, v811, v812}};
                                        } else {
                                            v920 = Union4{Union4_0{v807, v808, v809, v810, v811, v812}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v920 = Union4{Union4_5{v807, v808, v809, v810, v811, v812}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v880;
                                        v880 = v812 > 0;
                                        if (v880){
                                            int v881;
                                            v881 = v810 ^ 1;
                                            int v882;
                                            v882 = -1 + v812;
                                            int v883; int v884;
                                            Tuple7 tmp86 = Tuple7{0, 0};
                                            v883 = tmp86.v0; v884 = tmp86.v1;
                                            while (while_method_0(v883)){
                                                bool v886;
                                                v886 = 0 <= v883;
                                                bool v888;
                                                if (v886){
                                                    bool v887;
                                                    v887 = v883 < 2;
                                                    v888 = v887;
                                                } else {
                                                    v888 = false;
                                                }
                                                bool v889;
                                                v889 = v888 == false;
                                                if (v889){
                                                    assert("Index must be in range." && v888);
                                                } else {
                                                }
                                                int v891;
                                                v891 = v811[v883];
                                                bool v893;
                                                v893 = v884 >= v891;
                                                int v894;
                                                if (v893){
                                                    v894 = v884;
                                                } else {
                                                    v894 = v891;
                                                }
                                                v884 = v894;
                                                v883 += 1 ;
                                            }
                                            static_array<int,2> v895;
                                            int v897;
                                            v897 = 0;
                                            while (while_method_0(v897)){
                                                v895[v897] = v884;
                                                v897 += 1 ;
                                            }
                                            static_array<int,2> v899;
                                            int v901;
                                            v901 = 0;
                                            while (while_method_0(v901)){
                                                bool v903;
                                                v903 = 0 <= v901;
                                                bool v905;
                                                if (v903){
                                                    bool v904;
                                                    v904 = v901 < 2;
                                                    v905 = v904;
                                                } else {
                                                    v905 = false;
                                                }
                                                bool v906;
                                                v906 = v905 == false;
                                                if (v906){
                                                    assert("Index must be in range." && v905);
                                                } else {
                                                }
                                                int v908;
                                                v908 = v895[v901];
                                                bool v910;
                                                v910 = v901 == v810;
                                                int v912;
                                                if (v910){
                                                    int v911;
                                                    v911 = v908 + 2;
                                                    v912 = v911;
                                                } else {
                                                    v912 = v908;
                                                }
                                                v899[v901] = v912;
                                                v901 += 1 ;
                                            }
                                            v920 = Union4{Union4_2{v807, false, v809, v881, v899, v882}};
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
                                Union6 v814 = v807.case1.v0;
                                switch (v813.tag) {
                                    case 0: { // Call
                                        if (v808){
                                            int v816;
                                            v816 = v810 ^ 1;
                                            v920 = Union4{Union4_2{v807, false, v809, v816, v811, v812}};
                                        } else {
                                            int v818; int v819;
                                            Tuple7 tmp87 = Tuple7{0, 0};
                                            v818 = tmp87.v0; v819 = tmp87.v1;
                                            while (while_method_0(v818)){
                                                bool v821;
                                                v821 = 0 <= v818;
                                                bool v823;
                                                if (v821){
                                                    bool v822;
                                                    v822 = v818 < 2;
                                                    v823 = v822;
                                                } else {
                                                    v823 = false;
                                                }
                                                bool v824;
                                                v824 = v823 == false;
                                                if (v824){
                                                    assert("Index must be in range." && v823);
                                                } else {
                                                }
                                                int v826;
                                                v826 = v811[v818];
                                                bool v828;
                                                v828 = v819 >= v826;
                                                int v829;
                                                if (v828){
                                                    v829 = v819;
                                                } else {
                                                    v829 = v826;
                                                }
                                                v819 = v829;
                                                v818 += 1 ;
                                            }
                                            static_array<int,2> v830;
                                            int v832;
                                            v832 = 0;
                                            while (while_method_0(v832)){
                                                v830[v832] = v819;
                                                v832 += 1 ;
                                            }
                                            v920 = Union4{Union4_4{v807, v808, v809, v810, v830, v812}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v920 = Union4{Union4_5{v807, v808, v809, v810, v811, v812}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v836;
                                        v836 = v812 > 0;
                                        if (v836){
                                            int v837;
                                            v837 = v810 ^ 1;
                                            int v838;
                                            v838 = -1 + v812;
                                            int v839; int v840;
                                            Tuple7 tmp88 = Tuple7{0, 0};
                                            v839 = tmp88.v0; v840 = tmp88.v1;
                                            while (while_method_0(v839)){
                                                bool v842;
                                                v842 = 0 <= v839;
                                                bool v844;
                                                if (v842){
                                                    bool v843;
                                                    v843 = v839 < 2;
                                                    v844 = v843;
                                                } else {
                                                    v844 = false;
                                                }
                                                bool v845;
                                                v845 = v844 == false;
                                                if (v845){
                                                    assert("Index must be in range." && v844);
                                                } else {
                                                }
                                                int v847;
                                                v847 = v811[v839];
                                                bool v849;
                                                v849 = v840 >= v847;
                                                int v850;
                                                if (v849){
                                                    v850 = v840;
                                                } else {
                                                    v850 = v847;
                                                }
                                                v840 = v850;
                                                v839 += 1 ;
                                            }
                                            static_array<int,2> v851;
                                            int v853;
                                            v853 = 0;
                                            while (while_method_0(v853)){
                                                v851[v853] = v840;
                                                v853 += 1 ;
                                            }
                                            static_array<int,2> v855;
                                            int v857;
                                            v857 = 0;
                                            while (while_method_0(v857)){
                                                bool v859;
                                                v859 = 0 <= v857;
                                                bool v861;
                                                if (v859){
                                                    bool v860;
                                                    v860 = v857 < 2;
                                                    v861 = v860;
                                                } else {
                                                    v861 = false;
                                                }
                                                bool v862;
                                                v862 = v861 == false;
                                                if (v862){
                                                    assert("Index must be in range." && v861);
                                                } else {
                                                }
                                                int v864;
                                                v864 = v851[v857];
                                                bool v866;
                                                v866 = v857 == v810;
                                                int v868;
                                                if (v866){
                                                    int v867;
                                                    v867 = v864 + 4;
                                                    v868 = v867;
                                                } else {
                                                    v868 = v864;
                                                }
                                                v855[v857] = v868;
                                                v857 += 1 ;
                                            }
                                            v920 = Union4{Union4_2{v807, false, v809, v837, v855, v838}};
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
                        v928 = Union3{Union3_1{v920}};
                        break;
                    }
                    case 3: { // T_none
                        v928 = Union3{Union3_0{}};
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
        v16 = v928;
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
                Union3 v1129;
                switch (v60.tag) {
                    case 0: { // None
                        v1129 = Union3{Union3_0{}};
                        break;
                    }
                    case 1: { // Some
                        Union4 v62 = v60.case1.v0;
                        switch (v62.tag) {
                            case 0: { // ChanceCommunityCard
                                Union5 v1069 = v62.case0.v0; bool v1070 = v62.case0.v1; static_array<Union6,2> v1071 = v62.case0.v2; int v1072 = v62.case0.v3; static_array<int,2> v1073 = v62.case0.v4; int v1074 = v62.case0.v5;
                                curandStatePhilox4_32_10_t & v1075 = v19.v4;
                                curandStatePhilox4_32_10_t & v1076 = v1075;
                                unsigned int & v1077 = v19.v0;
                                Union6 v1078; unsigned int v1079;
                                Tuple6 tmp11 = draw_card_20(v1076, v1077);
                                v1078 = tmp11.v0; v1079 = tmp11.v1;
                                v19.v0 = v1079;
                                Union7 v1080;
                                v1080 = Union7{Union7_0{v1078}};
                                v58.push(v1080);
                                int v1081;
                                v1081 = 2;
                                int v1082; int v1083;
                                Tuple7 tmp12 = Tuple7{0, 0};
                                v1082 = tmp12.v0; v1083 = tmp12.v1;
                                while (while_method_0(v1082)){
                                    bool v1085;
                                    v1085 = 0 <= v1082;
                                    bool v1087;
                                    if (v1085){
                                        bool v1086;
                                        v1086 = v1082 < 2;
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
                                    v1090 = v1073[v1082];
                                    bool v1092;
                                    v1092 = v1083 >= v1090;
                                    int v1093;
                                    if (v1092){
                                        v1093 = v1083;
                                    } else {
                                        v1093 = v1090;
                                    }
                                    v1083 = v1093;
                                    v1082 += 1 ;
                                }
                                static_array<int,2> v1094;
                                int v1096;
                                v1096 = 0;
                                while (while_method_0(v1096)){
                                    v1094[v1096] = v1083;
                                    v1096 += 1 ;
                                }
                                Union5 v1098;
                                v1098 = Union5{Union5_1{v1078}};
                                Union4 v1099;
                                v1099 = Union4{Union4_2{v1098, true, v1071, 0, v1094, v1081}};
                                v1129 = Union3{Union3_1{v1099}};
                                break;
                            }
                            case 1: { // ChanceInit
                                curandStatePhilox4_32_10_t & v1101 = v19.v4;
                                curandStatePhilox4_32_10_t & v1102 = v1101;
                                unsigned int & v1103 = v19.v0;
                                Union6 v1104; unsigned int v1105;
                                Tuple6 tmp13 = draw_card_20(v1102, v1103);
                                v1104 = tmp13.v0; v1105 = tmp13.v1;
                                v19.v0 = v1105;
                                curandStatePhilox4_32_10_t & v1106 = v19.v4;
                                curandStatePhilox4_32_10_t & v1107 = v1106;
                                unsigned int & v1108 = v19.v0;
                                Union6 v1109; unsigned int v1110;
                                Tuple6 tmp14 = draw_card_20(v1107, v1108);
                                v1109 = tmp14.v0; v1110 = tmp14.v1;
                                v19.v0 = v1110;
                                Union7 v1111;
                                v1111 = Union7{Union7_2{0, v1104}};
                                v58.push(v1111);
                                Union7 v1112;
                                v1112 = Union7{Union7_2{1, v1109}};
                                v58.push(v1112);
                                int v1113;
                                v1113 = 2;
                                static_array<int,2> v1114;
                                v1114[0] = 1;
                                v1114[1] = 1;
                                static_array<Union6,2> v1116;
                                v1116[0] = v1104;
                                v1116[1] = v1109;
                                Union5 v1118;
                                v1118 = Union5{Union5_0{}};
                                Union4 v1119;
                                v1119 = Union4{Union4_2{v1118, true, v1116, 0, v1114, v1113}};
                                v1129 = Union3{Union3_1{v1119}};
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
                                        int * v191;
                                        v191 = reinterpret_cast<int *>(&v2[1048576ull]);
                                        bool * v193;
                                        v193 = reinterpret_cast<bool *>(&v2[1048592ull]);
                                        float * v195;
                                        v195 = reinterpret_cast<float *>(&v2[1048624ull]);
                                        float * v197;
                                        v197 = reinterpret_cast<float *>(&v2[1048752ull]);
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
                                        v219 = reinterpret_cast<float *>(&v3[4718592ull]);
                                        assert("Tensor range check" && 0 <= v218 && v218 < 32);
                                        int v221;
                                        v221 = 393216 * v218;
                                        float * v222;
                                        v222 = reinterpret_cast<float *>(&v3[0ull]);
                                        float * v224;
                                        v224 = reinterpret_cast<float *>(&v2[0ull]);
                                        float * v226;
                                        v226 = reinterpret_cast<float *>(&v4[0ull]);
                                        assert("Tensor range check" && 0 <= v218 && v218 < 32);
                                        int v228;
                                        v228 = 8192 * v218;
                                        float * v229;
                                        v229 = reinterpret_cast<float *>(&v3[3145728ull]);
                                        block_matmul_23(v229, v224, v228, v222);
                                        block_row_map_24(v219, v221, v229);
                                        int * v231;
                                        v231 = reinterpret_cast<int *>(&v2[1048576ull]);
                                        bool * v233;
                                        v233 = reinterpret_cast<bool *>(&v2[1048592ull]);
                                        float * v235;
                                        v235 = reinterpret_cast<float *>(&v2[1048624ull]);
                                        float * v237;
                                        v237 = reinterpret_cast<float *>(&v2[1048752ull]);
                                        double * v239;
                                        v239 = reinterpret_cast<double *>(&v3[55050240ull]);
                                        double * v241;
                                        v241 = reinterpret_cast<double *>(&v3[58195968ull]);
                                        __syncthreads();
                                        float * v243;
                                        v243 = reinterpret_cast<float *>(&v3[4718592ull]);
                                        assert("Tensor range check" && 0 <= v218 && v218 < 32);
                                        int v245;
                                        v245 = blockIdx.x;
                                        assert("Tensor range check" && 0 <= v245 && v245 < 24);
                                        int v246;
                                        v246 = 16384 * v245;
                                        int v247;
                                        v247 = v246 + v221;
                                        int v248;
                                        v248 = threadIdx.x;
                                        assert("Tensor range check" && 0 <= v248 && v248 < 256);
                                        int v249;
                                        v249 = 64 * v248;
                                        int v250;
                                        v250 = v249 + v247;
                                        float * v251;
                                        v251 = v243+v250;
                                        int v253;
                                        v253 = sizeof(float *);
                                        unsigned long long v254;
                                        v254 = (unsigned long long)v253;
                                        unsigned long long v255;
                                        v255 = 256ull * v254;
                                        unsigned long long v256;
                                        v256 = v255 + 16ull;
                                        unsigned long long v257;
                                        v257 = v256 - 1ull;
                                        unsigned long long v258;
                                        v258 = v257 % 16ull;
                                        unsigned long long v259;
                                        v259 = v257 - v258;
                                        unsigned long long v260;
                                        v260 = v259 + 1024ull;
                                        unsigned long long v261;
                                        v261 = v260 + 16ull;
                                        unsigned long long v262;
                                        v262 = v261 - 1ull;
                                        unsigned long long v263;
                                        v263 = v262 % 16ull;
                                        unsigned long long v264;
                                        v264 = v262 - v263;
                                        unsigned long long v265;
                                        v265 = v264 + 1024ull;
                                        bool v266;
                                        v266 = v265 <= 98304ull;
                                        bool v267;
                                        v267 = v266 == false;
                                        if (v267){
                                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v266);
                                        } else {
                                        }
                                        extern __shared__ unsigned char v269[];
                                        bool v270;
                                        v270 = v265 <= v265;
                                        bool v271;
                                        v271 = v270 == false;
                                        if (v271){
                                            assert("The length of the partition has to be less than or equal to the length of the base array." && v270);
                                        } else {
                                        }
                                        float * * v273;
                                        v273 = reinterpret_cast<float * *>(&v269[0ull]);
                                        float * v275;
                                        v275 = reinterpret_cast<float *>(&v269[v259]);
                                        int * v277;
                                        v277 = reinterpret_cast<int *>(&v269[v264]);
                                        int v279;
                                        v279 = threadIdx.x;
                                        assert("Tensor range check" && 0 <= v279 && v279 < 256);
                                        v273[v279] = v251;
                                        __syncthreads();
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
                                        int v288;
                                        v288 = 0;
                                        while (while_method_9(v288)){
                                            bool v290;
                                            v290 = 0 <= v284;
                                            bool v291;
                                            v291 = v290 && v285;
                                            bool v292;
                                            v292 = v291 == false;
                                            if (v292){
                                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v291);
                                            } else {
                                            }
                                            bool v294;
                                            v294 = 0 <= v288;
                                            bool v296;
                                            if (v294){
                                                bool v295;
                                                v295 = v288 < 16;
                                                v296 = v295;
                                            } else {
                                                v296 = false;
                                            }
                                            bool v297;
                                            v297 = v296 == false;
                                            if (v297){
                                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v296);
                                            } else {
                                            }
                                            int v299;
                                            v299 = v288 * 16;
                                            int v300;
                                            v300 = v299 + v284;
                                            assert("Tensor range check" && 0 <= v288 && v288 < 16);
                                            int v301;
                                            v301 = 16 * v288;
                                            int v302;
                                            v302 = v301 + v284;
                                            float * v303;
                                            v303 = v273[v302];
                                            int v304;
                                            v304 = blockIdx.x;
                                            int v305;
                                            v305 = v304 * 256;
                                            int v306;
                                            v306 = v305 + v300;
                                            assert("Tensor range check" && 0 <= v283 && v283 < 16);
                                            int v307;
                                            v307 = 4 * v283;
                                            float v308[4];
                                            int v309[4];
                                            int v310;
                                            v310 = 0;
                                            while (while_method_5(v310)){
                                                assert("Tensor range check" && 0 <= v310 && v310 < 1);
                                                int v312;
                                                v312 = 4 * v310;
                                                assert("Tensor range check" && 0 <= v310 && v310 < 1);
                                                int v313;
                                                v313 = 64 * v310;
                                                int v314;
                                                v314 = v313 + v307;
                                                int4* v315;
                                                v315 = reinterpret_cast<int4*>(v303 + v314);
                                                int4* v316;
                                                v316 = reinterpret_cast<int4*>(v308 + v312);
                                                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v315) % 16 == 0 && reinterpret_cast<unsigned long long>(v316) % 16 == 0);
                                                *v316 = *v315;
                                                v310 += 1 ;
                                            }
                                            int v317;
                                            v317 = 0;
                                            while (while_method_5(v317)){
                                                int v319;
                                                v319 = 0;
                                                while (while_method_8(v319)){
                                                    bool v321;
                                                    v321 = 0 <= v319;
                                                    bool v323;
                                                    if (v321){
                                                        bool v322;
                                                        v322 = v319 < 4;
                                                        v323 = v322;
                                                    } else {
                                                        v323 = false;
                                                    }
                                                    bool v324;
                                                    v324 = v323 == false;
                                                    if (v324){
                                                        assert("The indices should be inside the range of the dimension." && v323);
                                                    } else {
                                                    }
                                                    bool v326;
                                                    v326 = 0 <= v283;
                                                    bool v328;
                                                    if (v326){
                                                        bool v327;
                                                        v327 = v283 < 16;
                                                        v328 = v327;
                                                    } else {
                                                        v328 = false;
                                                    }
                                                    bool v329;
                                                    v329 = v328 == false;
                                                    if (v329){
                                                        assert("The indices should be inside the range of the dimension." && v328);
                                                    } else {
                                                    }
                                                    int v331;
                                                    v331 = v283 * 4;
                                                    int v332;
                                                    v332 = v319 + v331;
                                                    bool v333;
                                                    v333 = 0 <= v317;
                                                    bool v335;
                                                    if (v333){
                                                        bool v334;
                                                        v334 = v317 < 1;
                                                        v335 = v334;
                                                    } else {
                                                        v335 = false;
                                                    }
                                                    bool v336;
                                                    v336 = v335 == false;
                                                    if (v336){
                                                        assert("The indices should be inside the range of the dimension." && v335);
                                                    } else {
                                                    }
                                                    int v338;
                                                    v338 = v317 * 64;
                                                    int v339;
                                                    v339 = v332 + v338;
                                                    assert("Tensor range check" && 0 <= v317 && v317 < 1);
                                                    assert("Tensor range check" && 0 <= v319 && v319 < 4);
                                                    int v340;
                                                    v340 = 4 * v317;
                                                    int v341;
                                                    v341 = v340 + v319;
                                                    v309[v341] = v339;
                                                    v319 += 1 ;
                                                }
                                                v317 += 1 ;
                                            }
                                            float v342[4];
                                            float v343;
                                            v343 = 0.0f;
                                            int v344;
                                            v344 = 0;
                                            while (while_method_5(v344)){
                                                assert("Tensor range check" && 0 <= v344 && v344 < 1);
                                                int v346;
                                                v346 = 4 * v344;
                                                assert("Tensor range check" && 0 <= v344 && v344 < 1);
                                                float v347;
                                                v347 = 0.0f;
                                                int v348;
                                                v348 = 0;
                                                while (while_method_8(v348)){
                                                    assert("Tensor range check" && 0 <= v348 && v348 < 4);
                                                    int v350;
                                                    v350 = v348 + v346;
                                                    float v351;
                                                    v351 = v308[v350];
                                                    float v352;
                                                    v352 = v347 + v351;
                                                    v347 = v352;
                                                    v348 += 1 ;
                                                }
                                                auto v353 = cooperative_groups::coalesced_threads();
                                                int v354;
                                                v354 = threadIdx.x;
                                                int v355;
                                                v355 = v354 / 16;
                                                auto v356 = cooperative_groups::labeled_partition(v353,v355);
                                                Closure2 v357{};
                                                float v358;
                                                v358 = cooperative_groups::inclusive_scan(v356, v347, v357);
                                                float v359;
                                                v359 = v356.shfl_up(v358,1);
                                                bool v360;
                                                v360 = v356.thread_rank() == 0;
                                                float v361;
                                                if (v360){
                                                    v361 = 0.0f;
                                                } else {
                                                    v361 = v359;
                                                }
                                                float v362;
                                                v362 = v356.shfl(v358,v356.num_threads()-1);
                                                float v363;
                                                v363 = v343 + v361;
                                                float v364;
                                                v364 = v363;
                                                int v365;
                                                v365 = 0;
                                                while (while_method_8(v365)){
                                                    assert("Tensor range check" && 0 <= v365 && v365 < 4);
                                                    int v367;
                                                    v367 = v365 + v346;
                                                    float v368;
                                                    v368 = v308[v367];
                                                    float v369;
                                                    v369 = v364 + v368;
                                                    assert("Tensor range check" && 0 <= v365 && v365 < 4);
                                                    v342[v367] = v369;
                                                    v364 = v369;
                                                    v365 += 1 ;
                                                }
                                                float v370;
                                                v370 = v343 + v362;
                                                v343 = v370;
                                                v344 += 1 ;
                                            }
                                            float v371[4];
                                            bool v372[4];
                                            int v373;
                                            v373 = 0;
                                            while (while_method_5(v373)){
                                                int v375;
                                                v375 = 0;
                                                while (while_method_8(v375)){
                                                    assert("Tensor range check" && 0 <= v373 && v373 < 1);
                                                    assert("Tensor range check" && 0 <= v375 && v375 < 4);
                                                    int v377;
                                                    v377 = 4 * v373;
                                                    int v378;
                                                    v378 = v377 + v375;
                                                    float v379;
                                                    v379 = v342[v378];
                                                    float v380;
                                                    v380 = v308[v378];
                                                    bool v381;
                                                    v381 = v380 > 0.0f;
                                                    assert("Tensor range check" && 0 <= v373 && v373 < 1);
                                                    assert("Tensor range check" && 0 <= v375 && v375 < 4);
                                                    v371[v378] = v379;
                                                    v372[v378] = v381;
                                                    v375 += 1 ;
                                                }
                                                v373 += 1 ;
                                            }
                                            float v382; bool v383;
                                            Tuple8 tmp15 = Tuple8{-1.0f / 0.0f, false};
                                            v382 = tmp15.v0; v383 = tmp15.v1;
                                            int v384;
                                            v384 = 0;
                                            while (while_method_5(v384)){
                                                int v386;
                                                v386 = 0;
                                                while (while_method_8(v386)){
                                                    assert("Tensor range check" && 0 <= v384 && v384 < 1);
                                                    assert("Tensor range check" && 0 <= v386 && v386 < 4);
                                                    int v388;
                                                    v388 = 4 * v384;
                                                    int v389;
                                                    v389 = v388 + v386;
                                                    float v390;
                                                    v390 = v371[v389];
                                                    bool v391;
                                                    v391 = v372[v389];
                                                    float v398; bool v399;
                                                    if (v383){
                                                        if (v391){
                                                            bool v392;
                                                            v392 = v382 >= v390;
                                                            float v393;
                                                            if (v392){
                                                                v393 = v382;
                                                            } else {
                                                                v393 = v390;
                                                            }
                                                            v398 = v393; v399 = true;
                                                        } else {
                                                            v398 = v382; v399 = v383;
                                                        }
                                                    } else {
                                                        if (v391){
                                                            v398 = v390; v399 = v391;
                                                        } else {
                                                            v398 = v382; v399 = v383;
                                                        }
                                                    }
                                                    v382 = v398;
                                                    v383 = v399;
                                                    v386 += 1 ;
                                                }
                                                v384 += 1 ;
                                            }
                                            auto v400 = cooperative_groups::coalesced_threads();
                                            int v401;
                                            v401 = threadIdx.x;
                                            int v402;
                                            v402 = v401 / 16;
                                            auto v403 = cooperative_groups::labeled_partition(v400,v402);
                                            Closure3 v404{};
                                            float v405; bool v406;
                                            Tuple8 tmp16 = cooperative_groups::reduce(v403, Tuple8{v382, v383}, v404);
                                            v405 = tmp16.v0; v406 = tmp16.v1;
                                            bool v407;
                                            v407 = v406 == false;
                                            if (v407){
                                                int v408;
                                                v408 = threadIdx.x;
                                                int v409;
                                                v409 = blockIdx.x;
                                                int v410;
                                                v410 = v409 * 256;
                                                int v411;
                                                v411 = v408 + v410;
                                                cuda::counting_semaphore<cuda::thread_scope_system, 1> & v412 = console_lock;
                                                auto v413 = cooperative_groups::coalesced_threads();
                                                v412.acquire();
                                                int v414;
                                                v414 = 0;
                                                printf("{%s = %d; %s = %c","tid", v411, "x'", '[');
                                                int v415;
                                                v415 = 0;
                                                while (while_method_5(v415)){
                                                    int v417;
                                                    v417 = v414;
                                                    bool v418;
                                                    v418 = v417 >= 100;
                                                    if (v418){
                                                        printf("%s"," ...");
                                                        break;
                                                    } else {
                                                    }
                                                    bool v419;
                                                    v419 = v415 == 0;
                                                    bool v420;
                                                    v420 = v419 != true;
                                                    if (v420){
                                                        printf("%s","; ");
                                                    } else {
                                                    }
                                                    printf("%c",'[');
                                                    int v421;
                                                    v421 = 0;
                                                    while (while_method_8(v421)){
                                                        int v423;
                                                        v423 = v414;
                                                        bool v424;
                                                        v424 = v423 >= 100;
                                                        if (v424){
                                                            printf("%s"," ...");
                                                            break;
                                                        } else {
                                                        }
                                                        bool v425;
                                                        v425 = v421 == 0;
                                                        bool v426;
                                                        v426 = v425 != true;
                                                        if (v426){
                                                            printf("%s","; ");
                                                        } else {
                                                        }
                                                        int v427;
                                                        v427 = v414 + 1;
                                                        v414 = v427;
                                                        int v428;
                                                        v428 = v415 * 4;
                                                        int v429;
                                                        v429 = v428 + v421;
                                                        float v430;
                                                        v430 = v371[v429];
                                                        bool v431;
                                                        v431 = v372[v429];
                                                        const char * v434;
                                                        if (v431){
                                                            const char * v432;
                                                            v432 = "true";
                                                            v434 = v432;
                                                        } else {
                                                            const char * v433;
                                                            v433 = "false";
                                                            v434 = v433;
                                                        }
                                                        printf("%f, %s",v430, v434);
                                                        v421 += 1 ;
                                                    }
                                                    printf("%c",']');
                                                    v415 += 1 ;
                                                }
                                                printf("%c",']');
                                                printf("}\n");
                                                v412.release();
                                                v413.sync() ;
                                            } else {
                                            }
                                            if (v407){
                                                assert("The local reduce must be true." && v406);
                                            } else {
                                            }
                                            float v470[4];
                                            int v471[4];
                                            int v472;
                                            v472 = 0;
                                            while (while_method_5(v472)){
                                                int v474;
                                                v474 = 0;
                                                while (while_method_8(v474)){
                                                    assert("Tensor range check" && 0 <= v472 && v472 < 1);
                                                    assert("Tensor range check" && 0 <= v474 && v474 < 4);
                                                    int v476;
                                                    v476 = 4 * v472;
                                                    int v477;
                                                    v477 = v476 + v474;
                                                    int v478;
                                                    v478 = v309[v477];
                                                    float v479;
                                                    v479 = curand_uniform(&v121);
                                                    assert("Tensor range check" && 0 <= v472 && v472 < 1);
                                                    assert("Tensor range check" && 0 <= v474 && v474 < 4);
                                                    v470[v477] = v479;
                                                    v471[v477] = v478;
                                                    v474 += 1 ;
                                                }
                                                v472 += 1 ;
                                            }
                                            float v480; int v481;
                                            Tuple9 tmp17 = Tuple9{0.0f, 2147483647};
                                            v480 = tmp17.v0; v481 = tmp17.v1;
                                            int v482;
                                            v482 = 0;
                                            while (while_method_5(v482)){
                                                int v484;
                                                v484 = 0;
                                                while (while_method_8(v484)){
                                                    assert("Tensor range check" && 0 <= v482 && v482 < 1);
                                                    assert("Tensor range check" && 0 <= v484 && v484 < 4);
                                                    int v486;
                                                    v486 = 4 * v482;
                                                    int v487;
                                                    v487 = v486 + v484;
                                                    float v488;
                                                    v488 = v470[v487];
                                                    int v489;
                                                    v489 = v471[v487];
                                                    bool v490;
                                                    v490 = v481 < v489;
                                                    float v491; int v492;
                                                    if (v490){
                                                        v491 = v480; v492 = v481;
                                                    } else {
                                                        v491 = v488; v492 = v489;
                                                    }
                                                    v480 = v491;
                                                    v481 = v492;
                                                    v484 += 1 ;
                                                }
                                                v482 += 1 ;
                                            }
                                            auto v493 = cooperative_groups::coalesced_threads();
                                            int v494;
                                            v494 = threadIdx.x;
                                            int v495;
                                            v495 = v494 / 16;
                                            auto v496 = cooperative_groups::labeled_partition(v493,v495);
                                            Closure4 v497{};
                                            float v498; int v499;
                                            Tuple9 tmp18 = cooperative_groups::reduce(v496, Tuple9{v480, v481}, v497);
                                            v498 = tmp18.v0; v499 = tmp18.v1;
                                            float v500;
                                            v500 = v405 * v498;
                                            int v501[4];
                                            bool v502[4];
                                            int v503;
                                            v503 = 0;
                                            while (while_method_5(v503)){
                                                int v505;
                                                v505 = 0;
                                                while (while_method_8(v505)){
                                                    assert("Tensor range check" && 0 <= v503 && v503 < 1);
                                                    assert("Tensor range check" && 0 <= v505 && v505 < 4);
                                                    int v507;
                                                    v507 = 4 * v503;
                                                    int v508;
                                                    v508 = v507 + v505;
                                                    float v509;
                                                    v509 = v371[v508];
                                                    bool v510;
                                                    v510 = v372[v508];
                                                    int v511;
                                                    v511 = v309[v508];
                                                    int v514; bool v515;
                                                    if (v510){
                                                        float v512;
                                                        v512 = v509 - v500;
                                                        bool v513;
                                                        v513 = v512 >= 0.0f;
                                                        v514 = v511; v515 = v513;
                                                    } else {
                                                        v514 = 2147483647; v515 = false;
                                                    }
                                                    assert("Tensor range check" && 0 <= v503 && v503 < 1);
                                                    assert("Tensor range check" && 0 <= v505 && v505 < 4);
                                                    v501[v508] = v514;
                                                    v502[v508] = v515;
                                                    v505 += 1 ;
                                                }
                                                v503 += 1 ;
                                            }
                                            int v516; bool v517;
                                            Tuple10 tmp19 = Tuple10{2147483647, false};
                                            v516 = tmp19.v0; v517 = tmp19.v1;
                                            int v518;
                                            v518 = 0;
                                            while (while_method_5(v518)){
                                                int v520;
                                                v520 = 0;
                                                while (while_method_8(v520)){
                                                    assert("Tensor range check" && 0 <= v518 && v518 < 1);
                                                    assert("Tensor range check" && 0 <= v520 && v520 < 4);
                                                    int v522;
                                                    v522 = 4 * v518;
                                                    int v523;
                                                    v523 = v522 + v520;
                                                    int v524;
                                                    v524 = v501[v523];
                                                    bool v525;
                                                    v525 = v502[v523];
                                                    int v532; bool v533;
                                                    if (v517){
                                                        if (v525){
                                                            bool v526;
                                                            v526 = v516 < v524;
                                                            int v527;
                                                            if (v526){
                                                                v527 = v516;
                                                            } else {
                                                                v527 = v524;
                                                            }
                                                            v532 = v527; v533 = true;
                                                        } else {
                                                            v532 = v516; v533 = v517;
                                                        }
                                                    } else {
                                                        if (v525){
                                                            v532 = v524; v533 = v525;
                                                        } else {
                                                            v532 = v516; v533 = v517;
                                                        }
                                                    }
                                                    v516 = v532;
                                                    v517 = v533;
                                                    v520 += 1 ;
                                                }
                                                v518 += 1 ;
                                            }
                                            auto v534 = cooperative_groups::coalesced_threads();
                                            int v535;
                                            v535 = threadIdx.x;
                                            int v536;
                                            v536 = v535 / 16;
                                            auto v537 = cooperative_groups::labeled_partition(v534,v536);
                                            Closure5 v538{};
                                            int v539; bool v540;
                                            Tuple10 tmp20 = cooperative_groups::reduce(v537, Tuple10{v516, v517}, v538);
                                            v539 = tmp20.v0; v540 = tmp20.v1;
                                            bool v541;
                                            v541 = v540 == false;
                                            if (v541){
                                                int v542;
                                                v542 = threadIdx.x;
                                                int v543;
                                                v543 = blockIdx.x;
                                                int v544;
                                                v544 = v543 * 256;
                                                int v545;
                                                v545 = v542 + v544;
                                                cuda::counting_semaphore<cuda::thread_scope_system, 1> & v546 = console_lock;
                                                auto v547 = cooperative_groups::coalesced_threads();
                                                v546.acquire();
                                                int v548;
                                                v548 = 0;
                                                printf("{%s = %d; %s = %c","tid", v545, "x'", '[');
                                                int v549;
                                                v549 = 0;
                                                while (while_method_5(v549)){
                                                    int v551;
                                                    v551 = v548;
                                                    bool v552;
                                                    v552 = v551 >= 100;
                                                    if (v552){
                                                        printf("%s"," ...");
                                                        break;
                                                    } else {
                                                    }
                                                    bool v553;
                                                    v553 = v549 == 0;
                                                    bool v554;
                                                    v554 = v553 != true;
                                                    if (v554){
                                                        printf("%s","; ");
                                                    } else {
                                                    }
                                                    printf("%c",'[');
                                                    int v555;
                                                    v555 = 0;
                                                    while (while_method_8(v555)){
                                                        int v557;
                                                        v557 = v548;
                                                        bool v558;
                                                        v558 = v557 >= 100;
                                                        if (v558){
                                                            printf("%s"," ...");
                                                            break;
                                                        } else {
                                                        }
                                                        bool v559;
                                                        v559 = v555 == 0;
                                                        bool v560;
                                                        v560 = v559 != true;
                                                        if (v560){
                                                            printf("%s","; ");
                                                        } else {
                                                        }
                                                        int v561;
                                                        v561 = v548 + 1;
                                                        v548 = v561;
                                                        int v562;
                                                        v562 = v549 * 4;
                                                        int v563;
                                                        v563 = v562 + v555;
                                                        int v564;
                                                        v564 = v501[v563];
                                                        bool v565;
                                                        v565 = v502[v563];
                                                        const char * v568;
                                                        if (v565){
                                                            const char * v566;
                                                            v566 = "true";
                                                            v568 = v566;
                                                        } else {
                                                            const char * v567;
                                                            v567 = "false";
                                                            v568 = v567;
                                                        }
                                                        printf("%d, %s",v564, v568);
                                                        v555 += 1 ;
                                                    }
                                                    printf("%c",']');
                                                    v549 += 1 ;
                                                }
                                                printf("%c",']');
                                                printf("}\n");
                                                v546.release();
                                                v547.sync() ;
                                            } else {
                                            }
                                            if (v541){
                                                assert("The local reduce must be true." && v540);
                                            } else {
                                            }
                                            float v604; int v605;
                                            Tuple9 tmp21 = Tuple9{0.0f, 2147483647};
                                            v604 = tmp21.v0; v605 = tmp21.v1;
                                            int v606;
                                            v606 = 0;
                                            while (while_method_5(v606)){
                                                int v608;
                                                v608 = 0;
                                                while (while_method_8(v608)){
                                                    assert("Tensor range check" && 0 <= v606 && v606 < 1);
                                                    assert("Tensor range check" && 0 <= v608 && v608 < 4);
                                                    int v610;
                                                    v610 = 4 * v606;
                                                    int v611;
                                                    v611 = v610 + v608;
                                                    float v612;
                                                    v612 = v308[v611];
                                                    int v613;
                                                    v613 = v309[v611];
                                                    bool v614;
                                                    v614 = v605 == v539;
                                                    float v618; int v619;
                                                    if (v614){
                                                        v618 = v604; v619 = v605;
                                                    } else {
                                                        bool v615;
                                                        v615 = v613 == v539;
                                                        if (v615){
                                                            v618 = v612; v619 = v613;
                                                        } else {
                                                            v618 = v604; v619 = v605;
                                                        }
                                                    }
                                                    v604 = v618;
                                                    v605 = v619;
                                                    v608 += 1 ;
                                                }
                                                v606 += 1 ;
                                            }
                                            auto v620 = cooperative_groups::coalesced_threads();
                                            int v621;
                                            v621 = threadIdx.x;
                                            int v622;
                                            v622 = v621 / 16;
                                            auto v623 = cooperative_groups::labeled_partition(v620,v622);
                                            Closure6 v624{v539};
                                            float v625; int v626;
                                            Tuple9 tmp22 = cooperative_groups::reduce(v623, Tuple9{v604, v605}, v624);
                                            v625 = tmp22.v0; v626 = tmp22.v1;
                                            bool v627;
                                            v627 = v626 == 2147483647;
                                            bool v628;
                                            v628 = v627 != true;
                                            bool v629;
                                            v629 = v628 == false;
                                            if (v629){
                                                assert("Expected a valid action id in get_prob." && v628);
                                            } else {
                                            }
                                            int v631;
                                            v631 = 0;
                                            while (while_method_5(v631)){
                                                assert("Tensor range check" && 0 <= v631 && v631 < 1);
                                                assert("Tensor range check" && 0 <= v631 && v631 < 1);
                                                v631 += 1 ;
                                            }
                                            assert("Tensor range check" && 0 <= v300 && v300 < 256);
                                            v275[v300] = v625;
                                            v277[v300] = v539;
                                            v288 += 1 ;
                                        }
                                        __syncthreads();
                                        assert("Tensor range check" && 0 <= v279 && v279 < 256);
                                        float v633;
                                        v633 = v275[v279];
                                        int v634;
                                        v634 = v277[v279];
                                        __syncthreads();
                                        bool v635;
                                        v635 = 0 == v634;
                                        Union12 v644;
                                        if (v635){
                                            v644 = Union12{Union12_1{}};
                                        } else {
                                            bool v637;
                                            v637 = 1 == v634;
                                            if (v637){
                                                v644 = Union12{Union12_0{}};
                                            } else {
                                                bool v639;
                                                v639 = 2 == v634;
                                                if (v639){
                                                    v644 = Union12{Union12_2{}};
                                                } else {
                                                    printf("%s\n", "Invalid output id in the Leduc model.");
                                                    __trap();
                                                }
                                            }
                                        }
                                        Union1 v676;
                                        switch (v644.tag) {
                                            case 0: { // AA_Call
                                                v676 = Union1{Union1_0{}};
                                                break;
                                            }
                                            case 1: { // AA_Fold
                                                int v645;
                                                v645 = v109[0];
                                                int v647; int v648;
                                                Tuple7 tmp23 = Tuple7{1, v645};
                                                v647 = tmp23.v0; v648 = tmp23.v1;
                                                while (while_method_0(v647)){
                                                    bool v650;
                                                    v650 = 0 <= v647;
                                                    bool v652;
                                                    if (v650){
                                                        bool v651;
                                                        v651 = v647 < 2;
                                                        v652 = v651;
                                                    } else {
                                                        v652 = false;
                                                    }
                                                    bool v653;
                                                    v653 = v652 == false;
                                                    if (v653){
                                                        assert("Index must be in range." && v652);
                                                    } else {
                                                    }
                                                    int v655;
                                                    v655 = v109[v647];
                                                    bool v657;
                                                    v657 = v648 >= v655;
                                                    int v658;
                                                    if (v657){
                                                        v658 = v648;
                                                    } else {
                                                        v658 = v655;
                                                    }
                                                    v648 = v658;
                                                    v647 += 1 ;
                                                }
                                                bool v660;
                                                if (v112){
                                                    bool v659;
                                                    v659 = v108 < 2;
                                                    v660 = v659;
                                                } else {
                                                    v660 = false;
                                                }
                                                bool v661;
                                                v661 = v660 == false;
                                                if (v661){
                                                    assert("Index must be in range." && v660);
                                                } else {
                                                }
                                                int v663;
                                                v663 = v109[v108];
                                                bool v665;
                                                v665 = v663 == v648;
                                                if (v665){
                                                    v676 = Union1{Union1_0{}};
                                                } else {
                                                    v676 = Union1{Union1_1{}};
                                                }
                                                break;
                                            }
                                            case 2: { // AA_Raise
                                                bool v670;
                                                v670 = v110 > 0;
                                                if (v670){
                                                    v676 = Union1{Union1_2{}};
                                                } else {
                                                    v676 = Union1{Union1_0{}};
                                                }
                                                break;
                                            }
                                            default: {
                                                assert("Invalid tag." && false); __trap();
                                            }
                                        }
                                        int v677;
                                        v677 = sizeof(Union1);
                                        unsigned long long v678;
                                        v678 = (unsigned long long)v677;
                                        bool v679;
                                        v679 = v678 <= 98304ull;
                                        bool v680;
                                        v680 = v679 == false;
                                        if (v680){
                                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v679);
                                        } else {
                                        }
                                        extern __shared__ unsigned char v682[];
                                        bool v683;
                                        v683 = v678 <= v678;
                                        bool v684;
                                        v684 = v683 == false;
                                        if (v684){
                                            assert("The length of the partition has to be less than or equal to the length of the base array." && v683);
                                        } else {
                                        }
                                        Union1 * v686;
                                        v686 = reinterpret_cast<Union1 *>(&v682[0ull]);
                                        int v688;
                                        v688 = threadIdx.x;
                                        bool v689;
                                        v689 = v688 == 0;
                                        if (v689){
                                            v686[0] = v676;
                                        } else {
                                        }
                                        __syncthreads();
                                        Union1 v690;
                                        v690 = v686[0];
                                        __syncthreads();
                                        Union7 v691;
                                        v691 = Union7{Union7_1{v108, v690}};
                                        v58.push(v691);
                                        Union4 v798;
                                        switch (v105.tag) {
                                            case 0: { // None
                                                switch (v690.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v754;
                                                            v754 = v108 ^ 1;
                                                            v798 = Union4{Union4_2{v105, false, v107, v754, v109, v110}};
                                                        } else {
                                                            v798 = Union4{Union4_0{v105, v106, v107, v108, v109, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v798 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        bool v758;
                                                        v758 = v110 > 0;
                                                        if (v758){
                                                            int v759;
                                                            v759 = v108 ^ 1;
                                                            int v760;
                                                            v760 = -1 + v110;
                                                            int v761; int v762;
                                                            Tuple7 tmp24 = Tuple7{0, 0};
                                                            v761 = tmp24.v0; v762 = tmp24.v1;
                                                            while (while_method_0(v761)){
                                                                bool v764;
                                                                v764 = 0 <= v761;
                                                                bool v766;
                                                                if (v764){
                                                                    bool v765;
                                                                    v765 = v761 < 2;
                                                                    v766 = v765;
                                                                } else {
                                                                    v766 = false;
                                                                }
                                                                bool v767;
                                                                v767 = v766 == false;
                                                                if (v767){
                                                                    assert("Index must be in range." && v766);
                                                                } else {
                                                                }
                                                                int v769;
                                                                v769 = v109[v761];
                                                                bool v771;
                                                                v771 = v762 >= v769;
                                                                int v772;
                                                                if (v771){
                                                                    v772 = v762;
                                                                } else {
                                                                    v772 = v769;
                                                                }
                                                                v762 = v772;
                                                                v761 += 1 ;
                                                            }
                                                            static_array<int,2> v773;
                                                            int v775;
                                                            v775 = 0;
                                                            while (while_method_0(v775)){
                                                                v773[v775] = v762;
                                                                v775 += 1 ;
                                                            }
                                                            static_array<int,2> v777;
                                                            int v779;
                                                            v779 = 0;
                                                            while (while_method_0(v779)){
                                                                bool v781;
                                                                v781 = 0 <= v779;
                                                                bool v783;
                                                                if (v781){
                                                                    bool v782;
                                                                    v782 = v779 < 2;
                                                                    v783 = v782;
                                                                } else {
                                                                    v783 = false;
                                                                }
                                                                bool v784;
                                                                v784 = v783 == false;
                                                                if (v784){
                                                                    assert("Index must be in range." && v783);
                                                                } else {
                                                                }
                                                                int v786;
                                                                v786 = v773[v779];
                                                                bool v788;
                                                                v788 = v779 == v108;
                                                                int v790;
                                                                if (v788){
                                                                    int v789;
                                                                    v789 = v786 + 2;
                                                                    v790 = v789;
                                                                } else {
                                                                    v790 = v786;
                                                                }
                                                                v777[v779] = v790;
                                                                v779 += 1 ;
                                                            }
                                                            v798 = Union4{Union4_2{v105, false, v107, v759, v777, v760}};
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
                                                Union6 v692 = v105.case1.v0;
                                                switch (v690.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v694;
                                                            v694 = v108 ^ 1;
                                                            v798 = Union4{Union4_2{v105, false, v107, v694, v109, v110}};
                                                        } else {
                                                            int v696; int v697;
                                                            Tuple7 tmp25 = Tuple7{0, 0};
                                                            v696 = tmp25.v0; v697 = tmp25.v1;
                                                            while (while_method_0(v696)){
                                                                bool v699;
                                                                v699 = 0 <= v696;
                                                                bool v701;
                                                                if (v699){
                                                                    bool v700;
                                                                    v700 = v696 < 2;
                                                                    v701 = v700;
                                                                } else {
                                                                    v701 = false;
                                                                }
                                                                bool v702;
                                                                v702 = v701 == false;
                                                                if (v702){
                                                                    assert("Index must be in range." && v701);
                                                                } else {
                                                                }
                                                                int v704;
                                                                v704 = v109[v696];
                                                                bool v706;
                                                                v706 = v697 >= v704;
                                                                int v707;
                                                                if (v706){
                                                                    v707 = v697;
                                                                } else {
                                                                    v707 = v704;
                                                                }
                                                                v697 = v707;
                                                                v696 += 1 ;
                                                            }
                                                            static_array<int,2> v708;
                                                            int v710;
                                                            v710 = 0;
                                                            while (while_method_0(v710)){
                                                                v708[v710] = v697;
                                                                v710 += 1 ;
                                                            }
                                                            v798 = Union4{Union4_4{v105, v106, v107, v108, v708, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v798 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        bool v714;
                                                        v714 = v110 > 0;
                                                        if (v714){
                                                            int v715;
                                                            v715 = v108 ^ 1;
                                                            int v716;
                                                            v716 = -1 + v110;
                                                            int v717; int v718;
                                                            Tuple7 tmp26 = Tuple7{0, 0};
                                                            v717 = tmp26.v0; v718 = tmp26.v1;
                                                            while (while_method_0(v717)){
                                                                bool v720;
                                                                v720 = 0 <= v717;
                                                                bool v722;
                                                                if (v720){
                                                                    bool v721;
                                                                    v721 = v717 < 2;
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
                                                                v725 = v109[v717];
                                                                bool v727;
                                                                v727 = v718 >= v725;
                                                                int v728;
                                                                if (v727){
                                                                    v728 = v718;
                                                                } else {
                                                                    v728 = v725;
                                                                }
                                                                v718 = v728;
                                                                v717 += 1 ;
                                                            }
                                                            static_array<int,2> v729;
                                                            int v731;
                                                            v731 = 0;
                                                            while (while_method_0(v731)){
                                                                v729[v731] = v718;
                                                                v731 += 1 ;
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
                                                                v742 = v729[v735];
                                                                bool v744;
                                                                v744 = v735 == v108;
                                                                int v746;
                                                                if (v744){
                                                                    int v745;
                                                                    v745 = v742 + 4;
                                                                    v746 = v745;
                                                                } else {
                                                                    v746 = v742;
                                                                }
                                                                v733[v735] = v746;
                                                                v735 += 1 ;
                                                            }
                                                            v798 = Union4{Union4_2{v105, false, v107, v715, v733, v716}};
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
                                        v1129 = Union3{Union3_1{v798}};
                                        break;
                                    }
                                    case 1: { // Human
                                        Union8 v800;
                                        v800 = Union8{Union8_2{v105, v106, v107, v108, v109, v110}};
                                        v19.v5 = v800;
                                        Union3 v801;
                                        v801 = Union3{Union3_1{v62}};
                                        v19.v1 = v801;
                                        v1129 = Union3{Union3_0{}};
                                        break;
                                    }
                                    case 2: { // Random
                                        curandStatePhilox4_32_10_t & v803 = v19.v4;
                                        curandStatePhilox4_32_10_t & v804 = v803;
                                        static_array_list<Union1,3> v805;
                                        v805 = static_array_list<Union1,3>{};
                                        v805.unsafe_set_length(1);
                                        Union1 v807;
                                        v807 = Union1{Union1_0{}};
                                        v805[0] = v807;
                                        int v809;
                                        v809 = v109[0];
                                        int v811;
                                        v811 = v109[1];
                                        bool v813;
                                        v813 = v809 == v811;
                                        bool v814;
                                        v814 = v813 != true;
                                        if (v814){
                                            Union1 v815;
                                            v815 = Union1{Union1_1{}};
                                            v805.push(v815);
                                        } else {
                                        }
                                        bool v816;
                                        v816 = v110 > 0;
                                        if (v816){
                                            Union1 v817;
                                            v817 = Union1{Union1_2{}};
                                            v805.push(v817);
                                        } else {
                                        }
                                        int v818;
                                        v818 = v805.length;
                                        int v819;
                                        v819 = v818 - 1;
                                        int v820;
                                        v820 = 0;
                                        while (while_method_1(v819, v820)){
                                            int v822;
                                            v822 = v805.length;
                                            int v823;
                                            v823 = int_range_22(v822, v820, v804);
                                            Union1 v824;
                                            v824 = v805[v820];
                                            Union1 v826;
                                            v826 = v805[v823];
                                            v805[v820] = v826;
                                            v805[v823] = v824;
                                            v820 += 1 ;
                                        }
                                        Union1 v828;
                                        v828 = v805.pop();
                                        int v829;
                                        v829 = sizeof(Union1);
                                        unsigned long long v830;
                                        v830 = (unsigned long long)v829;
                                        bool v831;
                                        v831 = v830 <= 98304ull;
                                        bool v832;
                                        v832 = v831 == false;
                                        if (v832){
                                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v831);
                                        } else {
                                        }
                                        extern __shared__ unsigned char v834[];
                                        bool v835;
                                        v835 = v830 <= v830;
                                        bool v836;
                                        v836 = v835 == false;
                                        if (v836){
                                            assert("The length of the partition has to be less than or equal to the length of the base array." && v835);
                                        } else {
                                        }
                                        Union1 * v838;
                                        v838 = reinterpret_cast<Union1 *>(&v834[0ull]);
                                        int v840;
                                        v840 = threadIdx.x;
                                        bool v841;
                                        v841 = v840 == 0;
                                        if (v841){
                                            v838[0] = v828;
                                        } else {
                                        }
                                        __syncthreads();
                                        Union1 v842;
                                        v842 = v838[0];
                                        __syncthreads();
                                        Union7 v843;
                                        v843 = Union7{Union7_1{v108, v842}};
                                        v58.push(v843);
                                        Union4 v948;
                                        switch (v105.tag) {
                                            case 0: { // None
                                                switch (v842.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v905;
                                                            v905 = v108 ^ 1;
                                                            v948 = Union4{Union4_2{v105, false, v107, v905, v109, v110}};
                                                        } else {
                                                            v948 = Union4{Union4_0{v105, v106, v107, v108, v109, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v948 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        if (v816){
                                                            int v909;
                                                            v909 = v108 ^ 1;
                                                            int v910;
                                                            v910 = -1 + v110;
                                                            int v911; int v912;
                                                            Tuple7 tmp27 = Tuple7{0, 0};
                                                            v911 = tmp27.v0; v912 = tmp27.v1;
                                                            while (while_method_0(v911)){
                                                                bool v914;
                                                                v914 = 0 <= v911;
                                                                bool v916;
                                                                if (v914){
                                                                    bool v915;
                                                                    v915 = v911 < 2;
                                                                    v916 = v915;
                                                                } else {
                                                                    v916 = false;
                                                                }
                                                                bool v917;
                                                                v917 = v916 == false;
                                                                if (v917){
                                                                    assert("Index must be in range." && v916);
                                                                } else {
                                                                }
                                                                int v919;
                                                                v919 = v109[v911];
                                                                bool v921;
                                                                v921 = v912 >= v919;
                                                                int v922;
                                                                if (v921){
                                                                    v922 = v912;
                                                                } else {
                                                                    v922 = v919;
                                                                }
                                                                v912 = v922;
                                                                v911 += 1 ;
                                                            }
                                                            static_array<int,2> v923;
                                                            int v925;
                                                            v925 = 0;
                                                            while (while_method_0(v925)){
                                                                v923[v925] = v912;
                                                                v925 += 1 ;
                                                            }
                                                            static_array<int,2> v927;
                                                            int v929;
                                                            v929 = 0;
                                                            while (while_method_0(v929)){
                                                                bool v931;
                                                                v931 = 0 <= v929;
                                                                bool v933;
                                                                if (v931){
                                                                    bool v932;
                                                                    v932 = v929 < 2;
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
                                                                v936 = v923[v929];
                                                                bool v938;
                                                                v938 = v929 == v108;
                                                                int v940;
                                                                if (v938){
                                                                    int v939;
                                                                    v939 = v936 + 2;
                                                                    v940 = v939;
                                                                } else {
                                                                    v940 = v936;
                                                                }
                                                                v927[v929] = v940;
                                                                v929 += 1 ;
                                                            }
                                                            v948 = Union4{Union4_2{v105, false, v107, v909, v927, v910}};
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
                                                Union6 v844 = v105.case1.v0;
                                                switch (v842.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v846;
                                                            v846 = v108 ^ 1;
                                                            v948 = Union4{Union4_2{v105, false, v107, v846, v109, v110}};
                                                        } else {
                                                            int v848; int v849;
                                                            Tuple7 tmp28 = Tuple7{0, 0};
                                                            v848 = tmp28.v0; v849 = tmp28.v1;
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
                                                                v856 = v109[v848];
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
                                                            v948 = Union4{Union4_4{v105, v106, v107, v108, v860, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v948 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        if (v816){
                                                            int v866;
                                                            v866 = v108 ^ 1;
                                                            int v867;
                                                            v867 = -1 + v110;
                                                            int v868; int v869;
                                                            Tuple7 tmp29 = Tuple7{0, 0};
                                                            v868 = tmp29.v0; v869 = tmp29.v1;
                                                            while (while_method_0(v868)){
                                                                bool v871;
                                                                v871 = 0 <= v868;
                                                                bool v873;
                                                                if (v871){
                                                                    bool v872;
                                                                    v872 = v868 < 2;
                                                                    v873 = v872;
                                                                } else {
                                                                    v873 = false;
                                                                }
                                                                bool v874;
                                                                v874 = v873 == false;
                                                                if (v874){
                                                                    assert("Index must be in range." && v873);
                                                                } else {
                                                                }
                                                                int v876;
                                                                v876 = v109[v868];
                                                                bool v878;
                                                                v878 = v869 >= v876;
                                                                int v879;
                                                                if (v878){
                                                                    v879 = v869;
                                                                } else {
                                                                    v879 = v876;
                                                                }
                                                                v869 = v879;
                                                                v868 += 1 ;
                                                            }
                                                            static_array<int,2> v880;
                                                            int v882;
                                                            v882 = 0;
                                                            while (while_method_0(v882)){
                                                                v880[v882] = v869;
                                                                v882 += 1 ;
                                                            }
                                                            static_array<int,2> v884;
                                                            int v886;
                                                            v886 = 0;
                                                            while (while_method_0(v886)){
                                                                bool v888;
                                                                v888 = 0 <= v886;
                                                                bool v890;
                                                                if (v888){
                                                                    bool v889;
                                                                    v889 = v886 < 2;
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
                                                                v893 = v880[v886];
                                                                bool v895;
                                                                v895 = v886 == v108;
                                                                int v897;
                                                                if (v895){
                                                                    int v896;
                                                                    v896 = v893 + 4;
                                                                    v897 = v896;
                                                                } else {
                                                                    v897 = v893;
                                                                }
                                                                v884[v886] = v897;
                                                                v886 += 1 ;
                                                            }
                                                            v948 = Union4{Union4_2{v105, false, v107, v866, v884, v867}};
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
                                        v1129 = Union3{Union3_1{v948}};
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                break;
                            }
                            case 3: { // RoundWithAction
                                Union5 v953 = v62.case3.v0; bool v954 = v62.case3.v1; static_array<Union6,2> v955 = v62.case3.v2; int v956 = v62.case3.v3; static_array<int,2> v957 = v62.case3.v4; int v958 = v62.case3.v5; Union1 v959 = v62.case3.v6;
                                Union7 v960;
                                v960 = Union7{Union7_1{v956, v959}};
                                v58.push(v960);
                                Union4 v1067;
                                switch (v953.tag) {
                                    case 0: { // None
                                        switch (v959.tag) {
                                            case 0: { // Call
                                                if (v954){
                                                    int v1023;
                                                    v1023 = v956 ^ 1;
                                                    v1067 = Union4{Union4_2{v953, false, v955, v1023, v957, v958}};
                                                } else {
                                                    v1067 = Union4{Union4_0{v953, v954, v955, v956, v957, v958}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v1067 = Union4{Union4_5{v953, v954, v955, v956, v957, v958}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                bool v1027;
                                                v1027 = v958 > 0;
                                                if (v1027){
                                                    int v1028;
                                                    v1028 = v956 ^ 1;
                                                    int v1029;
                                                    v1029 = -1 + v958;
                                                    int v1030; int v1031;
                                                    Tuple7 tmp30 = Tuple7{0, 0};
                                                    v1030 = tmp30.v0; v1031 = tmp30.v1;
                                                    while (while_method_0(v1030)){
                                                        bool v1033;
                                                        v1033 = 0 <= v1030;
                                                        bool v1035;
                                                        if (v1033){
                                                            bool v1034;
                                                            v1034 = v1030 < 2;
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
                                                        v1038 = v957[v1030];
                                                        bool v1040;
                                                        v1040 = v1031 >= v1038;
                                                        int v1041;
                                                        if (v1040){
                                                            v1041 = v1031;
                                                        } else {
                                                            v1041 = v1038;
                                                        }
                                                        v1031 = v1041;
                                                        v1030 += 1 ;
                                                    }
                                                    static_array<int,2> v1042;
                                                    int v1044;
                                                    v1044 = 0;
                                                    while (while_method_0(v1044)){
                                                        v1042[v1044] = v1031;
                                                        v1044 += 1 ;
                                                    }
                                                    static_array<int,2> v1046;
                                                    int v1048;
                                                    v1048 = 0;
                                                    while (while_method_0(v1048)){
                                                        bool v1050;
                                                        v1050 = 0 <= v1048;
                                                        bool v1052;
                                                        if (v1050){
                                                            bool v1051;
                                                            v1051 = v1048 < 2;
                                                            v1052 = v1051;
                                                        } else {
                                                            v1052 = false;
                                                        }
                                                        bool v1053;
                                                        v1053 = v1052 == false;
                                                        if (v1053){
                                                            assert("Index must be in range." && v1052);
                                                        } else {
                                                        }
                                                        int v1055;
                                                        v1055 = v1042[v1048];
                                                        bool v1057;
                                                        v1057 = v1048 == v956;
                                                        int v1059;
                                                        if (v1057){
                                                            int v1058;
                                                            v1058 = v1055 + 2;
                                                            v1059 = v1058;
                                                        } else {
                                                            v1059 = v1055;
                                                        }
                                                        v1046[v1048] = v1059;
                                                        v1048 += 1 ;
                                                    }
                                                    v1067 = Union4{Union4_2{v953, false, v955, v1028, v1046, v1029}};
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
                                        Union6 v961 = v953.case1.v0;
                                        switch (v959.tag) {
                                            case 0: { // Call
                                                if (v954){
                                                    int v963;
                                                    v963 = v956 ^ 1;
                                                    v1067 = Union4{Union4_2{v953, false, v955, v963, v957, v958}};
                                                } else {
                                                    int v965; int v966;
                                                    Tuple7 tmp31 = Tuple7{0, 0};
                                                    v965 = tmp31.v0; v966 = tmp31.v1;
                                                    while (while_method_0(v965)){
                                                        bool v968;
                                                        v968 = 0 <= v965;
                                                        bool v970;
                                                        if (v968){
                                                            bool v969;
                                                            v969 = v965 < 2;
                                                            v970 = v969;
                                                        } else {
                                                            v970 = false;
                                                        }
                                                        bool v971;
                                                        v971 = v970 == false;
                                                        if (v971){
                                                            assert("Index must be in range." && v970);
                                                        } else {
                                                        }
                                                        int v973;
                                                        v973 = v957[v965];
                                                        bool v975;
                                                        v975 = v966 >= v973;
                                                        int v976;
                                                        if (v975){
                                                            v976 = v966;
                                                        } else {
                                                            v976 = v973;
                                                        }
                                                        v966 = v976;
                                                        v965 += 1 ;
                                                    }
                                                    static_array<int,2> v977;
                                                    int v979;
                                                    v979 = 0;
                                                    while (while_method_0(v979)){
                                                        v977[v979] = v966;
                                                        v979 += 1 ;
                                                    }
                                                    v1067 = Union4{Union4_4{v953, v954, v955, v956, v977, v958}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v1067 = Union4{Union4_5{v953, v954, v955, v956, v957, v958}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                bool v983;
                                                v983 = v958 > 0;
                                                if (v983){
                                                    int v984;
                                                    v984 = v956 ^ 1;
                                                    int v985;
                                                    v985 = -1 + v958;
                                                    int v986; int v987;
                                                    Tuple7 tmp32 = Tuple7{0, 0};
                                                    v986 = tmp32.v0; v987 = tmp32.v1;
                                                    while (while_method_0(v986)){
                                                        bool v989;
                                                        v989 = 0 <= v986;
                                                        bool v991;
                                                        if (v989){
                                                            bool v990;
                                                            v990 = v986 < 2;
                                                            v991 = v990;
                                                        } else {
                                                            v991 = false;
                                                        }
                                                        bool v992;
                                                        v992 = v991 == false;
                                                        if (v992){
                                                            assert("Index must be in range." && v991);
                                                        } else {
                                                        }
                                                        int v994;
                                                        v994 = v957[v986];
                                                        bool v996;
                                                        v996 = v987 >= v994;
                                                        int v997;
                                                        if (v996){
                                                            v997 = v987;
                                                        } else {
                                                            v997 = v994;
                                                        }
                                                        v987 = v997;
                                                        v986 += 1 ;
                                                    }
                                                    static_array<int,2> v998;
                                                    int v1000;
                                                    v1000 = 0;
                                                    while (while_method_0(v1000)){
                                                        v998[v1000] = v987;
                                                        v1000 += 1 ;
                                                    }
                                                    static_array<int,2> v1002;
                                                    int v1004;
                                                    v1004 = 0;
                                                    while (while_method_0(v1004)){
                                                        bool v1006;
                                                        v1006 = 0 <= v1004;
                                                        bool v1008;
                                                        if (v1006){
                                                            bool v1007;
                                                            v1007 = v1004 < 2;
                                                            v1008 = v1007;
                                                        } else {
                                                            v1008 = false;
                                                        }
                                                        bool v1009;
                                                        v1009 = v1008 == false;
                                                        if (v1009){
                                                            assert("Index must be in range." && v1008);
                                                        } else {
                                                        }
                                                        int v1011;
                                                        v1011 = v998[v1004];
                                                        bool v1013;
                                                        v1013 = v1004 == v956;
                                                        int v1015;
                                                        if (v1013){
                                                            int v1014;
                                                            v1014 = v1011 + 4;
                                                            v1015 = v1014;
                                                        } else {
                                                            v1015 = v1011;
                                                        }
                                                        v1002[v1004] = v1015;
                                                        v1004 += 1 ;
                                                    }
                                                    v1067 = Union4{Union4_2{v953, false, v955, v984, v1002, v985}};
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
                                v1129 = Union3{Union3_1{v1067}};
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
                                v1129 = Union3{Union3_0{}};
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
                                v1129 = Union3{Union3_0{}};
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
                v60 = v1129;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v1130;
    v1130 = threadIdx.x;
    int v1131;
    v1131 = blockIdx.x;
    int v1132;
    v1132 = v1131 * 256;
    int v1133;
    v1133 = v1130 + v1132;
    bool v1134;
    v1134 = v1133 == 0;
    if (v1134){
        Union8 & v1135 = v19.v5;
        static_array<Union2,2> & v1136 = v19.v3;
        static_array_list<Union7,32> & v1137 = v19.v2;
        Union3 & v1138 = v19.v1;
        unsigned int & v1139 = v19.v0;
        return f_29(v0, v1139, v1138, v1137, v1136, v1135);
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
        while (while_method_6(v29)){
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
                v42 = reinterpret_cast<double *>(&v1[55050240ull]);
                double * v44;
                v44 = reinterpret_cast<double *>(&v1[58195968ull]);
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
                v51 = reinterpret_cast<int *>(&v0[1048576ull]);
                bool * v53;
                v53 = reinterpret_cast<bool *>(&v0[1048592ull]);
                float * v55;
                v55 = reinterpret_cast<float *>(&v0[1048624ull]);
                float * v57;
                v57 = reinterpret_cast<float *>(&v0[1048752ull]);
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
                v121 = reinterpret_cast<float *>(&v1[4718592ull]);
                int * v123;
                v123 = reinterpret_cast<int *>(&v0[1048576ull]);
                bool * v125;
                v125 = reinterpret_cast<bool *>(&v0[1048592ull]);
                float * v127;
                v127 = reinterpret_cast<float *>(&v0[1048624ull]);
                float * v129;
                v129 = reinterpret_cast<float *>(&v0[1048752ull]);
                double * v131;
                v131 = reinterpret_cast<double *>(&v1[55050240ull]);
                double * v133;
                v133 = reinterpret_cast<double *>(&v1[58195968ull]);
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
        v372 = reinterpret_cast<float *>(&v1[4718592ull]);
        int * v374;
        v374 = reinterpret_cast<int *>(&v0[1048576ull]);
        bool * v376;
        v376 = reinterpret_cast<bool *>(&v0[1048592ull]);
        float * v378;
        v378 = reinterpret_cast<float *>(&v0[1048624ull]);
        float * v380;
        v380 = reinterpret_cast<float *>(&v0[1048752ull]);
        double * v382;
        v382 = reinterpret_cast<double *>(&v1[55050240ull]);
        double * v384;
        v384 = reinterpret_cast<double *>(&v1[58195968ull]);
        v365.sync() ;
        int v386;
        v386 = threadIdx.x;
        int v387;
        v387 = blockIdx.x;
        int v388;
        v388 = v387 * 256;
        int v389;
        v389 = v386 + v388;
        bool v390;
        v390 = v389 == 0;
        if (v390){
            int v391;
            v391 = 0;
            int v392;
            v392 = 32;
            int v393;
            v393 = int_range_22(v392, v391, v367);
            v374[0] = v393;
        } else {
        }
        __syncwarp();
        float v394[32];
        int v395;
        v395 = 0;
        while (while_method_4(v395)){
            assert("Tensor range check" && 0 <= v395 && v395 < 32);
            float v397;
            v397 = v378[v395];
            float v398;
            v398 = v380[v395];
            bool v399;
            v399 = v398 == 0.0f;
            bool v400;
            v400 = v399 != true;
            float v402;
            if (v400){
                float v401;
                v401 = v397 / v398;
                v402 = v401;
            } else {
                v402 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v395 && v395 < 32);
            v394[v395] = v402;
            v395 += 1 ;
        }
        float v403;
        v403 = 0.0f;
        int v404;
        v404 = 0;
        while (while_method_4(v404)){
            assert("Tensor range check" && 0 <= v404 && v404 < 32);
            float v406;
            v406 = v394[v404];
            float v407;
            v407 = v403 + v406;
            v403 = v407;
            v404 += 1 ;
        }
        float v408;
        v408 = v403 / 32.0f;
        int v409;
        v409 = 0;
        while (while_method_4(v409)){
            assert("Tensor range check" && 0 <= v409 && v409 < 32);
            v378[v409] = 0.0f;
            v380[v409] = 0.0f;
            v409 += 1 ;
        }
        bool v411[32];
        int v412;
        v412 = 0;
        while (while_method_4(v412)){
            assert("Tensor range check" && 0 <= v412 && v412 < 32);
            float v414;
            v414 = v394[v412];
            bool v415;
            v415 = v414 >= v408;
            assert("Tensor range check" && 0 <= v412 && v412 < 32);
            v411[v412] = v415;
            v412 += 1 ;
        }
        int v416;
        v416 = 0;
        while (while_method_4(v416)){
            assert("Tensor range check" && 0 <= v416 && v416 < 32);
            bool v418;
            v418 = v411[v416];
            assert("Tensor range check" && 0 <= v416 && v416 < 32);
            v376[v416] = v418;
            v416 += 1 ;
        }
        extern __shared__ unsigned char v419[];
        float * v420;
        v420 = reinterpret_cast<float *>(&v419[0ull]);
        int v422;
        v422 = blockIdx.x;
        int v423;
        v423 = v422;
        while (while_method_9(v423)){
            bool v425;
            v425 = 0 <= v423;
            bool v426;
            v426 = v425 == false;
            if (v426){
                assert("The index needs to be zero or positive." && v425);
            } else {
            }
            int v428;
            v428 = v423 % 16;
            int v429;
            v429 = v423 / 16;
            bool v430;
            v430 = v429 < 1;
            bool v431;
            v431 = v430 == false;
            if (v431){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v430);
            } else {
            }
            assert("Tensor range check" && 0 <= v429 && v429 < 1);
            assert("Tensor range check" && 0 <= v428 && v428 < 16);
            int v433;
            v433 = 512 * v428;
            int v434;
            v434 = 262144 * v429;
            int v435;
            v435 = v434 + v433;
            int v436;
            v436 = 16384 * v428;
            int v437;
            v437 = 32 * v429;
            int v438;
            v438 = v437 + v436;
            int v439;
            v439 = threadIdx.x;
            int v440;
            v440 = v439;
            while (while_method_12(v440)){
                bool v442;
                v442 = 0 <= v440;
                bool v443;
                v443 = v442 == false;
                if (v443){
                    assert("The index needs to be zero or positive." && v442);
                } else {
                }
                int v445;
                v445 = v440 % 512;
                int v446;
                v446 = v440 / 512;
                bool v447;
                v447 = v446 < 32;
                bool v448;
                v448 = v447 == false;
                if (v448){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v447);
                } else {
                }
                assert("Tensor range check" && 0 <= v446 && v446 < 32);
                assert("Tensor range check" && 0 <= v445 && v445 < 512);
                int v450;
                v450 = v445 + v435;
                int v451;
                v451 = 8192 * v446;
                int v452;
                v452 = v451 + v450;
                float v453;
                v453 = v368[v452];
                assert("Tensor range check" && 0 <= v446 && v446 < 32);
                assert("Tensor range check" && 0 <= v445 && v445 < 512);
                int v454;
                v454 = 513 * v446;
                int v455;
                v455 = v454 + v445;
                v420[v455] = v453;
                v440 += 256 ;
            }
            __syncthreads();
            int v456;
            v456 = threadIdx.x;
            int v457;
            v457 = v456;
            while (while_method_12(v457)){
                bool v459;
                v459 = 0 <= v457;
                bool v460;
                v460 = v459 == false;
                if (v460){
                    assert("The index needs to be zero or positive." && v459);
                } else {
                }
                int v462;
                v462 = v457 % 32;
                int v463;
                v463 = v457 / 32;
                bool v464;
                v464 = v463 < 512;
                bool v465;
                v465 = v464 == false;
                if (v465){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v464);
                } else {
                }
                assert("Tensor range check" && 0 <= v463 && v463 < 512);
                assert("Tensor range check" && 0 <= v462 && v462 < 32);
                int v467;
                v467 = 513 * v462;
                int v468;
                v468 = v463 + v467;
                float v469;
                v469 = v420[v468];
                assert("Tensor range check" && 0 <= v463 && v463 < 512);
                assert("Tensor range check" && 0 <= v462 && v462 < 32);
                int v470;
                v470 = v462 + v438;
                int v471;
                v471 = 32 * v463;
                int v472;
                v472 = v471 + v470;
                v370[v472] = v469;
                v457 += 256 ;
            }
            __syncthreads();
            v423 += 24 ;
        }
        v365.sync() ;
        int v473;
        v473 = threadIdx.x;
        bool v474;
        v474 = 0 <= v473;
        bool v475;
        v475 = v474 == false;
        if (v475){
            assert("The index needs to be zero or positive." && v474);
        } else {
        }
        int v477;
        v477 = v473 % 8;
        int v478;
        v478 = v473 / 8;
        int v479;
        v479 = v478 % 32;
        int v480;
        v480 = v478 / 32;
        bool v481;
        v481 = v480 < 1;
        bool v482;
        v482 = v481 == false;
        if (v482){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v481);
        } else {
        }
        assert("Tensor range check" && 0 <= v480 && v480 < 1);
        assert("Tensor range check" && 0 <= v479 && v479 < 32);
        assert("Tensor range check" && 0 <= v477 && v477 < 8);
        int v484;
        v484 = 4 * v477;
        int v485;
        v485 = 32 * v479;
        int v486;
        v486 = v485 + v484;
        int v487;
        v487 = 4096 * v480;
        int v488;
        v488 = v487 + v486;
        assert("Tensor range check" && 0 <= v480 && v480 < 1);
        assert("Tensor range check" && 0 <= v479 && v479 < 32);
        assert("Tensor range check" && 0 <= v477 && v477 < 8);
        int v489;
        v489 = blockIdx.x;
        int v490;
        v490 = v489;
        while (while_method_10(v490)){
            bool v492;
            v492 = 0 <= v490;
            bool v493;
            v493 = v492 == false;
            if (v493){
                assert("The index needs to be zero or positive." && v492);
            } else {
            }
            int v495;
            v495 = v490 % 4;
            int v496;
            v496 = v490 / 4;
            bool v497;
            v497 = v496 < 64;
            bool v498;
            v498 = v497 == false;
            if (v498){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v497);
            } else {
            }
            assert("Tensor range check" && 0 <= v496 && v496 < 64);
            assert("Tensor range check" && 0 <= v495 && v495 < 4);
            int v500;
            v500 = 1024 * v495;
            int v501;
            v501 = v500 + v488;
            int v502;
            v502 = 4096 * v496;
            int v503;
            v503 = v502 + v501;
            float v504[4];
            int v505[4];
            int v506;
            v506 = 0;
            while (while_method_5(v506)){
                assert("Tensor range check" && 0 <= v506 && v506 < 1);
                int v508;
                v508 = 4 * v506;
                assert("Tensor range check" && 0 <= v506 && v506 < 1);
                int v509;
                v509 = 32 * v506;
                int v510;
                v510 = v509 + v503;
                int4* v511;
                v511 = reinterpret_cast<int4*>(v370 + v510);
                int4* v512;
                v512 = reinterpret_cast<int4*>(v504 + v508);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v511) % 16 == 0 && reinterpret_cast<unsigned long long>(v512) % 16 == 0);
                *v512 = *v511;
                v506 += 1 ;
            }
            int v513;
            v513 = 0;
            while (while_method_5(v513)){
                int v515;
                v515 = 0;
                while (while_method_8(v515)){
                    bool v517;
                    v517 = 0 <= v515;
                    bool v519;
                    if (v517){
                        bool v518;
                        v518 = v515 < 4;
                        v519 = v518;
                    } else {
                        v519 = false;
                    }
                    bool v520;
                    v520 = v519 == false;
                    if (v520){
                        assert("The indices should be inside the range of the dimension." && v519);
                    } else {
                    }
                    bool v522;
                    v522 = 0 <= v477;
                    bool v524;
                    if (v522){
                        bool v523;
                        v523 = v477 < 8;
                        v524 = v523;
                    } else {
                        v524 = false;
                    }
                    bool v525;
                    v525 = v524 == false;
                    if (v525){
                        assert("The indices should be inside the range of the dimension." && v524);
                    } else {
                    }
                    int v527;
                    v527 = v477 * 4;
                    int v528;
                    v528 = v515 + v527;
                    bool v529;
                    v529 = 0 <= v513;
                    bool v531;
                    if (v529){
                        bool v530;
                        v530 = v513 < 1;
                        v531 = v530;
                    } else {
                        v531 = false;
                    }
                    bool v532;
                    v532 = v531 == false;
                    if (v532){
                        assert("The indices should be inside the range of the dimension." && v531);
                    } else {
                    }
                    int v534;
                    v534 = v513 * 32;
                    int v535;
                    v535 = v528 + v534;
                    assert("Tensor range check" && 0 <= v513 && v513 < 1);
                    assert("Tensor range check" && 0 <= v515 && v515 < 4);
                    int v536;
                    v536 = 4 * v513;
                    int v537;
                    v537 = v536 + v515;
                    v505[v537] = v535;
                    v515 += 1 ;
                }
                v513 += 1 ;
            }
            bool v538;
            v538 = 0 <= v480;
            bool v539;
            v539 = v538 && v481;
            bool v540;
            v540 = v539 == false;
            if (v540){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v539);
            } else {
            }
            bool v542;
            v542 = 0 <= v479;
            bool v544;
            if (v542){
                bool v543;
                v543 = v479 < 32;
                v544 = v543;
            } else {
                v544 = false;
            }
            bool v545;
            v545 = v544 == false;
            if (v545){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v544);
            } else {
            }
            bool v547;
            v547 = 0 <= v496;
            bool v548;
            v548 = v547 && v497;
            bool v549;
            v549 = v548 == false;
            if (v549){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v548);
            } else {
            }
            bool v551;
            v551 = 0 <= v495;
            bool v553;
            if (v551){
                bool v552;
                v552 = v495 < 4;
                v553 = v552;
            } else {
                v553 = false;
            }
            bool v554;
            v554 = v553 == false;
            if (v554){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v553);
            } else {
            }
            int v556;
            v556 = v495 * 32;
            int v557;
            v557 = v496 + v480;
            int v558;
            v558 = v556 + v479;
            bool v559[4];
            int v560;
            v560 = 0;
            while (while_method_5(v560)){
                int v562;
                v562 = 0;
                while (while_method_8(v562)){
                    assert("Tensor range check" && 0 <= v560 && v560 < 1);
                    assert("Tensor range check" && 0 <= v562 && v562 < 4);
                    int v564;
                    v564 = 4 * v560;
                    int v565;
                    v565 = v564 + v562;
                    int v566;
                    v566 = v505[v565];
                    assert("Tensor range check" && 0 <= v566 && v566 < 32);
                    bool v567;
                    v567 = v376[v566];
                    assert("Tensor range check" && 0 <= v560 && v560 < 1);
                    assert("Tensor range check" && 0 <= v562 && v562 < 4);
                    v559[v565] = v567;
                    v562 += 1 ;
                }
                v560 += 1 ;
            }
            int v568[4];
            int v569;
            v569 = 0;
            while (while_method_5(v569)){
                int v571;
                v571 = 0;
                while (while_method_8(v571)){
                    assert("Tensor range check" && 0 <= v569 && v569 < 1);
                    assert("Tensor range check" && 0 <= v571 && v571 < 4);
                    int v573;
                    v573 = 4 * v569;
                    int v574;
                    v574 = v573 + v571;
                    bool v575;
                    v575 = v559[v574];
                    int v576;
                    if (v575){
                        v576 = 1;
                    } else {
                        v576 = 0;
                    }
                    assert("Tensor range check" && 0 <= v569 && v569 < 1);
                    assert("Tensor range check" && 0 <= v571 && v571 < 4);
                    v568[v574] = v576;
                    v571 += 1 ;
                }
                v569 += 1 ;
            }
            int v577;
            v577 = 0;
            int v578;
            v578 = 0;
            while (while_method_5(v578)){
                int v580;
                v580 = 0;
                while (while_method_8(v580)){
                    assert("Tensor range check" && 0 <= v578 && v578 < 1);
                    assert("Tensor range check" && 0 <= v580 && v580 < 4);
                    int v582;
                    v582 = 4 * v578;
                    int v583;
                    v583 = v582 + v580;
                    int v584;
                    v584 = v568[v583];
                    int v585;
                    v585 = v577 + v584;
                    v577 = v585;
                    v580 += 1 ;
                }
                v578 += 1 ;
            }
            auto v586 = cooperative_groups::coalesced_threads();
            int v587;
            v587 = threadIdx.x;
            int v588;
            v588 = v587 / 8;
            auto v589 = cooperative_groups::labeled_partition(v586,v588);
            Closure1 v590{};
            int v591;
            v591 = cooperative_groups::reduce(v589, v577, v590);
            float v592;
            v592 = (float)v591;
            float v593[4];
            int v594;
            v594 = 0;
            while (while_method_5(v594)){
                int v596;
                v596 = 0;
                while (while_method_8(v596)){
                    assert("Tensor range check" && 0 <= v594 && v594 < 1);
                    assert("Tensor range check" && 0 <= v596 && v596 < 4);
                    int v598;
                    v598 = 4 * v594;
                    int v599;
                    v599 = v598 + v596;
                    float v600;
                    v600 = v504[v599];
                    bool v601;
                    v601 = v559[v599];
                    float v602;
                    if (v601){
                        v602 = v600;
                    } else {
                        v602 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v594 && v594 < 1);
                    assert("Tensor range check" && 0 <= v596 && v596 < 4);
                    v593[v599] = v602;
                    v596 += 1 ;
                }
                v594 += 1 ;
            }
            float v603;
            v603 = 0.0f;
            int v604;
            v604 = 0;
            while (while_method_5(v604)){
                int v606;
                v606 = 0;
                while (while_method_8(v606)){
                    assert("Tensor range check" && 0 <= v604 && v604 < 1);
                    assert("Tensor range check" && 0 <= v606 && v606 < 4);
                    int v608;
                    v608 = 4 * v604;
                    int v609;
                    v609 = v608 + v606;
                    float v610;
                    v610 = v593[v609];
                    float v611;
                    v611 = v603 + v610;
                    v603 = v611;
                    v606 += 1 ;
                }
                v604 += 1 ;
            }
            auto v612 = cooperative_groups::coalesced_threads();
            int v613;
            v613 = threadIdx.x;
            int v614;
            v614 = v613 / 8;
            auto v615 = cooperative_groups::labeled_partition(v612,v614);
            Closure0 v616{};
            float v617;
            v617 = cooperative_groups::reduce(v615, v603, v616);
            float v618;
            v618 = v617 / v592;
            float v619[4];
            int v620;
            v620 = 0;
            while (while_method_5(v620)){
                int v622;
                v622 = 0;
                while (while_method_8(v622)){
                    assert("Tensor range check" && 0 <= v620 && v620 < 1);
                    assert("Tensor range check" && 0 <= v622 && v622 < 4);
                    int v624;
                    v624 = 4 * v620;
                    int v625;
                    v625 = v624 + v622;
                    float v626;
                    v626 = v504[v625];
                    float v627;
                    v627 = v626 - v618;
                    float v628;
                    v628 = v627 * v627;
                    assert("Tensor range check" && 0 <= v620 && v620 < 1);
                    assert("Tensor range check" && 0 <= v622 && v622 < 4);
                    v619[v625] = v628;
                    v622 += 1 ;
                }
                v620 += 1 ;
            }
            float v629[4];
            int v630;
            v630 = 0;
            while (while_method_5(v630)){
                int v632;
                v632 = 0;
                while (while_method_8(v632)){
                    assert("Tensor range check" && 0 <= v630 && v630 < 1);
                    assert("Tensor range check" && 0 <= v632 && v632 < 4);
                    int v634;
                    v634 = 4 * v630;
                    int v635;
                    v635 = v634 + v632;
                    float v636;
                    v636 = v619[v635];
                    bool v637;
                    v637 = v559[v635];
                    float v638;
                    if (v637){
                        v638 = v636;
                    } else {
                        v638 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v630 && v630 < 1);
                    assert("Tensor range check" && 0 <= v632 && v632 < 4);
                    v629[v635] = v638;
                    v632 += 1 ;
                }
                v630 += 1 ;
            }
            float v639;
            v639 = 0.0f;
            int v640;
            v640 = 0;
            while (while_method_5(v640)){
                int v642;
                v642 = 0;
                while (while_method_8(v642)){
                    assert("Tensor range check" && 0 <= v640 && v640 < 1);
                    assert("Tensor range check" && 0 <= v642 && v642 < 4);
                    int v644;
                    v644 = 4 * v640;
                    int v645;
                    v645 = v644 + v642;
                    float v646;
                    v646 = v629[v645];
                    float v647;
                    v647 = v639 + v646;
                    v639 = v647;
                    v642 += 1 ;
                }
                v640 += 1 ;
            }
            auto v648 = cooperative_groups::coalesced_threads();
            int v649;
            v649 = threadIdx.x;
            int v650;
            v650 = v649 / 8;
            auto v651 = cooperative_groups::labeled_partition(v648,v650);
            float v652;
            v652 = cooperative_groups::reduce(v651, v639, v616);
            float v653;
            v653 = v652 / v592;
            float v654;
            v654 = sqrt(v653);
            bool v655;
            v655 = v592 > 1.0f;
            float v659;
            if (v655){
                float v656;
                v656 = v654 * v592;
                float v657;
                v657 = v592 - 1.0f;
                float v658;
                v658 = v656 / v657;
                v659 = v658;
            } else {
                v659 = 0.0f;
            }
            float v660[4];
            int v661;
            v661 = 0;
            while (while_method_5(v661)){
                int v663;
                v663 = 0;
                while (while_method_8(v663)){
                    assert("Tensor range check" && 0 <= v661 && v661 < 1);
                    assert("Tensor range check" && 0 <= v663 && v663 < 4);
                    int v665;
                    v665 = 4 * v661;
                    int v666;
                    v666 = v665 + v663;
                    float v667;
                    v667 = v504[v666];
                    bool v668;
                    v668 = v559[v666];
                    float v669;
                    v669 = curand_normal(&v367);
                    bool v670;
                    v670 = v659 >= 0.1f;
                    float v671;
                    if (v670){
                        v671 = v659;
                    } else {
                        v671 = 0.1f;
                    }
                    float v672;
                    v672 = v669 * v671;
                    float v673;
                    v673 = v672 + v618;
                    float v674;
                    if (v668){
                        v674 = v667;
                    } else {
                        v674 = v673;
                    }
                    assert("Tensor range check" && 0 <= v661 && v661 < 1);
                    assert("Tensor range check" && 0 <= v663 && v663 < 4);
                    v660[v666] = v674;
                    v663 += 1 ;
                }
                v661 += 1 ;
            }
            assert("Tensor range check" && 0 <= v496 && v496 < 64);
            assert("Tensor range check" && 0 <= v495 && v495 < 4);
            int v675;
            v675 = 0;
            while (while_method_5(v675)){
                assert("Tensor range check" && 0 <= v675 && v675 < 1);
                int v677;
                v677 = 32 * v675;
                int v678;
                v678 = v677 + v503;
                assert("Tensor range check" && 0 <= v675 && v675 < 1);
                int v679;
                v679 = 4 * v675;
                int4* v680;
                v680 = reinterpret_cast<int4*>(v660 + v679);
                int4* v681;
                v681 = reinterpret_cast<int4*>(v370 + v678);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v680) % 16 == 0 && reinterpret_cast<unsigned long long>(v681) % 16 == 0);
                *v681 = *v680;
                v675 += 1 ;
            }
            v490 += 24 ;
        }
        v365.sync() ;
        static float v682[8192];
        int v683;
        v683 = threadIdx.x;
        int v684;
        v684 = blockIdx.x;
        int v685;
        v685 = v684 * 256;
        int v686;
        v686 = v683 + v685;
        int v687;
        v687 = v686 / 32;
        int v688;
        v688 = v687;
        while (while_method_13(v688)){
            bool v690;
            v690 = 0 <= v688;
            bool v691;
            v691 = v690 == false;
            if (v691){
                assert("The index needs to be zero or positive." && v690);
            } else {
            }
            int v693;
            v693 = v688 % 128;
            int v694;
            v694 = v688 / 128;
            bool v695;
            v695 = v694 < 64;
            bool v696;
            v696 = v695 == false;
            if (v696){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v695);
            } else {
            }
            assert("Tensor range check" && 0 <= v694 && v694 < 64);
            assert("Tensor range check" && 0 <= v693 && v693 < 128);
            int v698;
            v698 = 32 * v693;
            int v699;
            v699 = 4096 * v694;
            int v700;
            v700 = v699 + v698;
            float v701;
            v701 = 0.0f;
            int v702;
            v702 = threadIdx.x;
            int v703;
            v703 = v702 % 32;
            int v704;
            v704 = v703;
            while (while_method_6(v704)){
                bool v706;
                v706 = 0 <= v704;
                bool v707;
                v707 = v706 == false;
                if (v707){
                    assert("The index needs to be zero or positive." && v706);
                } else {
                }
                bool v709;
                v709 = v704 < 8;
                bool v710;
                v710 = v709 == false;
                if (v710){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v709);
                } else {
                }
                assert("Tensor range check" && 0 <= v704 && v704 < 8);
                int v712;
                v712 = 4 * v704;
                int v713;
                v713 = v712 + v700;
                float v714[4];
                int4* v715;
                v715 = reinterpret_cast<int4*>(v370 + v713);
                int4* v716;
                v716 = reinterpret_cast<int4*>(v714 + 0);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v715) % 16 == 0 && reinterpret_cast<unsigned long long>(v716) % 16 == 0);
                *v716 = *v715;
                int v717;
                v717 = 0;
                while (while_method_8(v717)){
                    assert("Tensor range check" && 0 <= v717 && v717 < 4);
                    float v719;
                    v719 = v714[v717];
                    float v720;
                    v720 = v719 * v719;
                    float v721;
                    v721 = v701 + v720;
                    v701 = v721;
                    v717 += 1 ;
                }
                v704 += 32 ;
            }
            __syncwarp();
            auto v722 = cooperative_groups::coalesced_threads();
            Closure0 v723{};
            float v724;
            v724 = cooperative_groups::reduce(v722, v701, v723);
            float v725;
            v725 = sqrt(v724);
            assert("Tensor range check" && 0 <= v694 && v694 < 64);
            assert("Tensor range check" && 0 <= v693 && v693 < 128);
            int v726;
            v726 = 128 * v694;
            int v727;
            v727 = v726 + v693;
            v682[v727] = v725;
            v688 += 192 ;
        }
        __syncthreads();
        v365.sync() ;
        float v728;
        v728 = 0.0f;
        int v729;
        v729 = threadIdx.x;
        int v730;
        v730 = blockIdx.x;
        int v731;
        v731 = v730 * 256;
        int v732;
        v732 = v729 + v731;
        int v733;
        v733 = v732;
        while (while_method_14(v733)){
            bool v735;
            v735 = 0 <= v733;
            bool v736;
            v736 = v735 == false;
            if (v736){
                assert("The index needs to be zero or positive." && v735);
            } else {
            }
            int v738;
            v738 = v733 % 32;
            int v739;
            v739 = v733 / 32;
            bool v740;
            v740 = v739 < 64;
            bool v741;
            v741 = v740 == false;
            if (v741){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v740);
            } else {
            }
            assert("Tensor range check" && 0 <= v739 && v739 < 64);
            assert("Tensor range check" && 0 <= v738 && v738 < 32);
            int v743;
            v743 = 4 * v738;
            int v744;
            v744 = 128 * v739;
            int v745;
            v745 = v744 + v743;
            float v746[4];
            int4* v747;
            v747 = reinterpret_cast<int4*>(v682 + v745);
            int4* v748;
            v748 = reinterpret_cast<int4*>(v746 + 0);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v747) % 16 == 0 && reinterpret_cast<unsigned long long>(v748) % 16 == 0);
            *v748 = *v747;
            int v749; float v750;
            Tuple13 tmp54 = Tuple13{0, v728};
            v749 = tmp54.v0; v750 = tmp54.v1;
            while (while_method_8(v749)){
                assert("Tensor range check" && 0 <= v749 && v749 < 4);
                float v752;
                v752 = v746[v749];
                bool v753;
                v753 = v750 >= v752;
                float v754;
                if (v753){
                    v754 = v750;
                } else {
                    v754 = v752;
                }
                v750 = v754;
                v749 += 1 ;
            }
            v728 = v750;
            v733 += 6144 ;
        }
        __syncwarp();
        auto v755 = cooperative_groups::coalesced_threads();
        Closure7 v756{};
        float v757;
        v757 = cooperative_groups::reduce(v755, v728, v756);
        int v758;
        v758 = threadIdx.x;
        int v759;
        v759 = v758 / 32;
        extern __shared__ unsigned char v760[];
        float * v761;
        v761 = reinterpret_cast<float *>(&v760[0ull]);
        assert("Tensor range check" && 0 <= v759 && v759 < 8);
        v761[v759] = v757;
        __syncthreads();
        int v763;
        v763 = threadIdx.x;
        int v764;
        v764 = v763 % 32;
        bool v765;
        v765 = v764 < 8;
        float v767;
        if (v765){
            assert("Tensor range check" && 0 <= v764 && v764 < 8);
            float v766;
            v766 = v761[v764];
            v767 = v766;
        } else {
            v767 = 0.0f;
        }
        __syncthreads();
        auto v768 = cooperative_groups::coalesced_threads();
        float v769;
        v769 = cooperative_groups::reduce(v768, v767, v756);
        int v770;
        v770 = blockIdx.x;
        static float v771[24];
        assert("Tensor range check" && 0 <= v770 && v770 < 24);
        v771[v770] = v769;
        v365.sync() ;
        float v772;
        v772 = 0.0f;
        int v773;
        v773 = threadIdx.x;
        int v774;
        v774 = v773 % 32;
        int v775;
        v775 = v774;
        while (while_method_15(v775)){
            bool v777;
            v777 = 0 <= v775;
            bool v778;
            v778 = v777 == false;
            if (v778){
                assert("The index needs to be zero or positive." && v777);
            } else {
            }
            bool v780;
            v780 = v775 < 24;
            bool v781;
            v781 = v780 == false;
            if (v781){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v780);
            } else {
            }
            assert("Tensor range check" && 0 <= v775 && v775 < 24);
            float v783;
            v783 = v771[v775];
            bool v784;
            v784 = v772 >= v783;
            float v785;
            if (v784){
                v785 = v772;
            } else {
                v785 = v783;
            }
            v772 = v785;
            v775 += 32 ;
        }
        __syncwarp();
        auto v786 = cooperative_groups::coalesced_threads();
        float v787;
        v787 = cooperative_groups::reduce(v786, v772, v756);
        int v788;
        v788 = threadIdx.x;
        int v789;
        v789 = blockIdx.x;
        int v790;
        v790 = v789 * 256;
        int v791;
        v791 = v788 + v790;
        bool v792;
        v792 = v791 == 0;
        if (v792){
            cuda::counting_semaphore<cuda::thread_scope_system, 1> & v793 = console_lock;
            auto v794 = cooperative_groups::coalesced_threads();
            v793.acquire();
            printf("{%s = %f}\n","max_norm", v787);
            v793.release();
            v794.sync() ;
        } else {
        }
        __syncwarp();
        extern __shared__ unsigned char v797[];
        float * v798;
        v798 = reinterpret_cast<float *>(&v797[0ull]);
        int v800;
        v800 = blockIdx.x;
        int v801;
        v801 = v800;
        while (while_method_9(v801)){
            bool v803;
            v803 = 0 <= v801;
            bool v804;
            v804 = v803 == false;
            if (v804){
                assert("The index needs to be zero or positive." && v803);
            } else {
            }
            int v806;
            v806 = v801 % 1;
            bool v807;
            v807 = v801 < 16;
            bool v808;
            v808 = v807 == false;
            if (v808){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v807);
            } else {
            }
            assert("Tensor range check" && 0 <= v801 && v801 < 16);
            assert("Tensor range check" && 0 <= v806 && v806 < 1);
            int v810;
            v810 = 32 * v806;
            int v811;
            v811 = 16384 * v801;
            int v812;
            v812 = v811 + v810;
            int v813;
            v813 = 262144 * v806;
            int v814;
            v814 = 512 * v801;
            int v815;
            v815 = v814 + v813;
            int v816;
            v816 = threadIdx.x;
            int v817;
            v817 = v816;
            while (while_method_12(v817)){
                bool v819;
                v819 = 0 <= v817;
                bool v820;
                v820 = v819 == false;
                if (v820){
                    assert("The index needs to be zero or positive." && v819);
                } else {
                }
                int v822;
                v822 = v817 % 32;
                int v823;
                v823 = v817 / 32;
                bool v824;
                v824 = v823 < 512;
                bool v825;
                v825 = v824 == false;
                if (v825){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v824);
                } else {
                }
                assert("Tensor range check" && 0 <= v823 && v823 < 512);
                assert("Tensor range check" && 0 <= v822 && v822 < 32);
                int v827;
                v827 = v822 + v812;
                int v828;
                v828 = 32 * v823;
                int v829;
                v829 = v828 + v827;
                float v830;
                v830 = v370[v829];
                assert("Tensor range check" && 0 <= v823 && v823 < 512);
                assert("Tensor range check" && 0 <= v822 && v822 < 32);
                int v831;
                v831 = 33 * v823;
                int v832;
                v832 = v831 + v822;
                v798[v832] = v830;
                v817 += 256 ;
            }
            __syncthreads();
            int v833;
            v833 = threadIdx.x;
            int v834;
            v834 = v833;
            while (while_method_12(v834)){
                bool v836;
                v836 = 0 <= v834;
                bool v837;
                v837 = v836 == false;
                if (v837){
                    assert("The index needs to be zero or positive." && v836);
                } else {
                }
                int v839;
                v839 = v834 % 512;
                int v840;
                v840 = v834 / 512;
                bool v841;
                v841 = v840 < 32;
                bool v842;
                v842 = v841 == false;
                if (v842){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v841);
                } else {
                }
                assert("Tensor range check" && 0 <= v840 && v840 < 32);
                assert("Tensor range check" && 0 <= v839 && v839 < 512);
                int v844;
                v844 = 33 * v839;
                int v845;
                v845 = v840 + v844;
                float v846;
                v846 = v798[v845];
                assert("Tensor range check" && 0 <= v840 && v840 < 32);
                assert("Tensor range check" && 0 <= v839 && v839 < 512);
                int v847;
                v847 = v839 + v815;
                int v848;
                v848 = 8192 * v840;
                int v849;
                v849 = v848 + v847;
                v368[v849] = v846;
                v834 += 256 ;
            }
            __syncthreads();
            v801 += 24 ;
        }
        v365.sync() ;
        v27 += 1 ;
    }
    cooperative_groups::grid_group & v850 = v26.v1;
    cooperative_groups::grid_group & v851 = v850;
    int v852;
    v852 = threadIdx.x;
    int v853;
    v853 = blockIdx.x;
    int v854;
    v854 = v853 * 256;
    int v855;
    v855 = v852 + v854;
    int v856;
    v856 = v855;
    while (while_method_14(v856)){
        bool v858;
        v858 = 0 <= v856;
        bool v859;
        v859 = v858 == false;
        if (v859){
            assert("The index needs to be zero or positive." && v858);
        } else {
        }
        int v861;
        v861 = v856 % 64;
        int v862;
        v862 = v856 / 64;
        bool v863;
        v863 = v862 < 32;
        bool v864;
        v864 = v863 == false;
        if (v864){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v863);
        } else {
        }
        assert("Tensor range check" && 0 <= v862 && v862 < 32);
        assert("Tensor range check" && 0 <= v861 && v861 < 64);
        int v866;
        v866 = 4 * v861;
        int v867;
        v867 = 256 * v862;
        int v868;
        v868 = v867 + v866;
        assert("Tensor range check" && 0 <= v862 && v862 < 32);
        assert("Tensor range check" && 0 <= v861 && v861 < 64);
        float v869[4];
        float v870[4];
        float v871[4];
        int4* v872;
        v872 = reinterpret_cast<int4*>(v3 + v868);
        int4* v873;
        v873 = reinterpret_cast<int4*>(v869 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v872) % 16 == 0 && reinterpret_cast<unsigned long long>(v873) % 16 == 0);
        *v873 = *v872;
        int4* v874;
        v874 = reinterpret_cast<int4*>(v4 + v868);
        int4* v875;
        v875 = reinterpret_cast<int4*>(v870 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v874) % 16 == 0 && reinterpret_cast<unsigned long long>(v875) % 16 == 0);
        *v875 = *v874;
        // Pushing the loop unrolling to: 0
        int v876;
        v876 = 0;
        #pragma unroll
        while (while_method_8(v876)){
            assert("Tensor range check" && 0 <= v876 && v876 < 4);
            float v878;
            v878 = v869[v876];
            float v879;
            v879 = v870[v876];
            bool v880;
            v880 = v879 == 0.0f;
            bool v881;
            v881 = v880 != true;
            float v883;
            if (v881){
                float v882;
                v882 = v878 / v879;
                v883 = v882;
            } else {
                v883 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v876 && v876 < 4);
            v871[v876] = v883;
            v876 += 1 ;
        }
        // Poping the loop unrolling to: 0
        int4* v884;
        v884 = reinterpret_cast<int4*>(v871 + 0);
        int4* v885;
        v885 = reinterpret_cast<int4*>(v5 + v868);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v884) % 16 == 0 && reinterpret_cast<unsigned long long>(v885) % 16 == 0);
        *v885 = *v884;
        v856 += 6144 ;
    }
    v851.sync() ;
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
        while (while_method_6(v29)){
            Union4 v31;
            v31 = Union4{Union4_1{}};
            method_47(v0, v1, v2, v26, v31);
            static_array<float,2> & v32 = v26.v4;
            float * v33;
            v33 = reinterpret_cast<float *>(&v1[4718592ull]);
            int * v35;
            v35 = reinterpret_cast<int *>(&v0[1048576ull]);
            bool * v37;
            v37 = reinterpret_cast<bool *>(&v0[1048592ull]);
            float * v39;
            v39 = reinterpret_cast<float *>(&v0[1048624ull]);
            float * v41;
            v41 = reinterpret_cast<float *>(&v0[1048752ull]);
            double * v43;
            v43 = reinterpret_cast<double *>(&v1[55050240ull]);
            double * v45;
            v45 = reinterpret_cast<double *>(&v1[58195968ull]);
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
                Tuple11 tmp71 = Tuple11{0, 0.0};
                v69 = tmp71.v0; v70 = tmp71.v1;
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
                Tuple12 tmp72 = Tuple12{0, 0.0f, 0.0};
                v114 = tmp72.v0; v115 = tmp72.v1; v116 = tmp72.v2;
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
            method_48(v0, v1, v2, v26, v276);
            double * v277;
            v277 = reinterpret_cast<double *>(&v1[55050240ull]);
            double * v279;
            v279 = reinterpret_cast<double *>(&v1[58195968ull]);
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
            v297 = reinterpret_cast<int *>(&v0[1048576ull]);
            bool * v299;
            v299 = reinterpret_cast<bool *>(&v0[1048592ull]);
            float * v301;
            v301 = reinterpret_cast<float *>(&v0[1048624ull]);
            float * v303;
            v303 = reinterpret_cast<float *>(&v0[1048752ull]);
            double v305[2];
            int v306;
            v306 = 0;
            while (while_method_0(v306)){
                int v308; double v309;
                Tuple11 tmp89 = Tuple11{0, 0.0};
                v308 = tmp89.v0; v309 = tmp89.v1;
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
                Tuple12 tmp90 = Tuple12{0, 0.0f, 0.0};
                v355 = tmp90.v0; v356 = tmp90.v1; v357 = tmp90.v2;
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
            v380 = reinterpret_cast<double *>(&v1[55050240ull]);
            double * v382;
            v382 = reinterpret_cast<double *>(&v1[58195968ull]);
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
        v526 = reinterpret_cast<float *>(&v1[4718592ull]);
        int * v528;
        v528 = reinterpret_cast<int *>(&v0[1048576ull]);
        bool * v530;
        v530 = reinterpret_cast<bool *>(&v0[1048592ull]);
        float * v532;
        v532 = reinterpret_cast<float *>(&v0[1048624ull]);
        float * v534;
        v534 = reinterpret_cast<float *>(&v0[1048752ull]);
        double * v536;
        v536 = reinterpret_cast<double *>(&v1[55050240ull]);
        double * v538;
        v538 = reinterpret_cast<double *>(&v1[58195968ull]);
        v519.sync() ;
        int v540;
        v540 = threadIdx.x;
        int v541;
        v541 = blockIdx.x;
        int v542;
        v542 = v541 * 256;
        int v543;
        v543 = v540 + v542;
        bool v544;
        v544 = v543 == 0;
        if (v544){
            int v545;
            v545 = 0;
            int v546;
            v546 = 32;
            int v547;
            v547 = int_range_22(v546, v545, v521);
            v528[0] = v547;
        } else {
        }
        __syncwarp();
        float v548[32];
        int v549;
        v549 = 0;
        while (while_method_4(v549)){
            assert("Tensor range check" && 0 <= v549 && v549 < 32);
            float v551;
            v551 = v532[v549];
            float v552;
            v552 = v534[v549];
            bool v553;
            v553 = v552 == 0.0f;
            bool v554;
            v554 = v553 != true;
            float v556;
            if (v554){
                float v555;
                v555 = v551 / v552;
                v556 = v555;
            } else {
                v556 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v549 && v549 < 32);
            v548[v549] = v556;
            v549 += 1 ;
        }
        float v557;
        v557 = 0.0f;
        int v558;
        v558 = 0;
        while (while_method_4(v558)){
            assert("Tensor range check" && 0 <= v558 && v558 < 32);
            float v560;
            v560 = v548[v558];
            float v561;
            v561 = v557 + v560;
            v557 = v561;
            v558 += 1 ;
        }
        float v562;
        v562 = v557 / 32.0f;
        int v563;
        v563 = 0;
        while (while_method_4(v563)){
            assert("Tensor range check" && 0 <= v563 && v563 < 32);
            v532[v563] = 0.0f;
            v534[v563] = 0.0f;
            v563 += 1 ;
        }
        bool v565[32];
        int v566;
        v566 = 0;
        while (while_method_4(v566)){
            assert("Tensor range check" && 0 <= v566 && v566 < 32);
            float v568;
            v568 = v548[v566];
            bool v569;
            v569 = v568 >= v562;
            assert("Tensor range check" && 0 <= v566 && v566 < 32);
            v565[v566] = v569;
            v566 += 1 ;
        }
        int v570;
        v570 = 0;
        while (while_method_4(v570)){
            assert("Tensor range check" && 0 <= v570 && v570 < 32);
            bool v572;
            v572 = v565[v570];
            assert("Tensor range check" && 0 <= v570 && v570 < 32);
            v530[v570] = v572;
            v570 += 1 ;
        }
        extern __shared__ unsigned char v573[];
        float * v574;
        v574 = reinterpret_cast<float *>(&v573[0ull]);
        int v576;
        v576 = blockIdx.x;
        int v577;
        v577 = v576;
        while (while_method_9(v577)){
            bool v579;
            v579 = 0 <= v577;
            bool v580;
            v580 = v579 == false;
            if (v580){
                assert("The index needs to be zero or positive." && v579);
            } else {
            }
            int v582;
            v582 = v577 % 16;
            int v583;
            v583 = v577 / 16;
            bool v584;
            v584 = v583 < 1;
            bool v585;
            v585 = v584 == false;
            if (v585){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v584);
            } else {
            }
            assert("Tensor range check" && 0 <= v583 && v583 < 1);
            assert("Tensor range check" && 0 <= v582 && v582 < 16);
            int v587;
            v587 = 512 * v582;
            int v588;
            v588 = 262144 * v583;
            int v589;
            v589 = v588 + v587;
            int v590;
            v590 = 16384 * v582;
            int v591;
            v591 = 32 * v583;
            int v592;
            v592 = v591 + v590;
            int v593;
            v593 = threadIdx.x;
            int v594;
            v594 = v593;
            while (while_method_12(v594)){
                bool v596;
                v596 = 0 <= v594;
                bool v597;
                v597 = v596 == false;
                if (v597){
                    assert("The index needs to be zero or positive." && v596);
                } else {
                }
                int v599;
                v599 = v594 % 512;
                int v600;
                v600 = v594 / 512;
                bool v601;
                v601 = v600 < 32;
                bool v602;
                v602 = v601 == false;
                if (v602){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v601);
                } else {
                }
                assert("Tensor range check" && 0 <= v600 && v600 < 32);
                assert("Tensor range check" && 0 <= v599 && v599 < 512);
                int v604;
                v604 = v599 + v589;
                int v605;
                v605 = 8192 * v600;
                int v606;
                v606 = v605 + v604;
                float v607;
                v607 = v522[v606];
                assert("Tensor range check" && 0 <= v600 && v600 < 32);
                assert("Tensor range check" && 0 <= v599 && v599 < 512);
                int v608;
                v608 = 513 * v600;
                int v609;
                v609 = v608 + v599;
                v574[v609] = v607;
                v594 += 256 ;
            }
            __syncthreads();
            int v610;
            v610 = threadIdx.x;
            int v611;
            v611 = v610;
            while (while_method_12(v611)){
                bool v613;
                v613 = 0 <= v611;
                bool v614;
                v614 = v613 == false;
                if (v614){
                    assert("The index needs to be zero or positive." && v613);
                } else {
                }
                int v616;
                v616 = v611 % 32;
                int v617;
                v617 = v611 / 32;
                bool v618;
                v618 = v617 < 512;
                bool v619;
                v619 = v618 == false;
                if (v619){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v618);
                } else {
                }
                assert("Tensor range check" && 0 <= v617 && v617 < 512);
                assert("Tensor range check" && 0 <= v616 && v616 < 32);
                int v621;
                v621 = 513 * v616;
                int v622;
                v622 = v617 + v621;
                float v623;
                v623 = v574[v622];
                assert("Tensor range check" && 0 <= v617 && v617 < 512);
                assert("Tensor range check" && 0 <= v616 && v616 < 32);
                int v624;
                v624 = v616 + v592;
                int v625;
                v625 = 32 * v617;
                int v626;
                v626 = v625 + v624;
                v524[v626] = v623;
                v611 += 256 ;
            }
            __syncthreads();
            v577 += 24 ;
        }
        v519.sync() ;
        int v627;
        v627 = threadIdx.x;
        bool v628;
        v628 = 0 <= v627;
        bool v629;
        v629 = v628 == false;
        if (v629){
            assert("The index needs to be zero or positive." && v628);
        } else {
        }
        int v631;
        v631 = v627 % 8;
        int v632;
        v632 = v627 / 8;
        int v633;
        v633 = v632 % 32;
        int v634;
        v634 = v632 / 32;
        bool v635;
        v635 = v634 < 1;
        bool v636;
        v636 = v635 == false;
        if (v636){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v635);
        } else {
        }
        assert("Tensor range check" && 0 <= v634 && v634 < 1);
        assert("Tensor range check" && 0 <= v633 && v633 < 32);
        assert("Tensor range check" && 0 <= v631 && v631 < 8);
        int v638;
        v638 = 4 * v631;
        int v639;
        v639 = 32 * v633;
        int v640;
        v640 = v639 + v638;
        int v641;
        v641 = 4096 * v634;
        int v642;
        v642 = v641 + v640;
        assert("Tensor range check" && 0 <= v634 && v634 < 1);
        assert("Tensor range check" && 0 <= v633 && v633 < 32);
        assert("Tensor range check" && 0 <= v631 && v631 < 8);
        int v643;
        v643 = blockIdx.x;
        int v644;
        v644 = v643;
        while (while_method_10(v644)){
            bool v646;
            v646 = 0 <= v644;
            bool v647;
            v647 = v646 == false;
            if (v647){
                assert("The index needs to be zero or positive." && v646);
            } else {
            }
            int v649;
            v649 = v644 % 4;
            int v650;
            v650 = v644 / 4;
            bool v651;
            v651 = v650 < 64;
            bool v652;
            v652 = v651 == false;
            if (v652){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v651);
            } else {
            }
            assert("Tensor range check" && 0 <= v650 && v650 < 64);
            assert("Tensor range check" && 0 <= v649 && v649 < 4);
            int v654;
            v654 = 1024 * v649;
            int v655;
            v655 = v654 + v642;
            int v656;
            v656 = 4096 * v650;
            int v657;
            v657 = v656 + v655;
            float v658[4];
            int v659[4];
            int v660;
            v660 = 0;
            while (while_method_5(v660)){
                assert("Tensor range check" && 0 <= v660 && v660 < 1);
                int v662;
                v662 = 4 * v660;
                assert("Tensor range check" && 0 <= v660 && v660 < 1);
                int v663;
                v663 = 32 * v660;
                int v664;
                v664 = v663 + v657;
                int4* v665;
                v665 = reinterpret_cast<int4*>(v524 + v664);
                int4* v666;
                v666 = reinterpret_cast<int4*>(v658 + v662);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v665) % 16 == 0 && reinterpret_cast<unsigned long long>(v666) % 16 == 0);
                *v666 = *v665;
                v660 += 1 ;
            }
            int v667;
            v667 = 0;
            while (while_method_5(v667)){
                int v669;
                v669 = 0;
                while (while_method_8(v669)){
                    bool v671;
                    v671 = 0 <= v669;
                    bool v673;
                    if (v671){
                        bool v672;
                        v672 = v669 < 4;
                        v673 = v672;
                    } else {
                        v673 = false;
                    }
                    bool v674;
                    v674 = v673 == false;
                    if (v674){
                        assert("The indices should be inside the range of the dimension." && v673);
                    } else {
                    }
                    bool v676;
                    v676 = 0 <= v631;
                    bool v678;
                    if (v676){
                        bool v677;
                        v677 = v631 < 8;
                        v678 = v677;
                    } else {
                        v678 = false;
                    }
                    bool v679;
                    v679 = v678 == false;
                    if (v679){
                        assert("The indices should be inside the range of the dimension." && v678);
                    } else {
                    }
                    int v681;
                    v681 = v631 * 4;
                    int v682;
                    v682 = v669 + v681;
                    bool v683;
                    v683 = 0 <= v667;
                    bool v685;
                    if (v683){
                        bool v684;
                        v684 = v667 < 1;
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
                    v688 = v667 * 32;
                    int v689;
                    v689 = v682 + v688;
                    assert("Tensor range check" && 0 <= v667 && v667 < 1);
                    assert("Tensor range check" && 0 <= v669 && v669 < 4);
                    int v690;
                    v690 = 4 * v667;
                    int v691;
                    v691 = v690 + v669;
                    v659[v691] = v689;
                    v669 += 1 ;
                }
                v667 += 1 ;
            }
            bool v692;
            v692 = 0 <= v634;
            bool v693;
            v693 = v692 && v635;
            bool v694;
            v694 = v693 == false;
            if (v694){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v693);
            } else {
            }
            bool v696;
            v696 = 0 <= v633;
            bool v698;
            if (v696){
                bool v697;
                v697 = v633 < 32;
                v698 = v697;
            } else {
                v698 = false;
            }
            bool v699;
            v699 = v698 == false;
            if (v699){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v698);
            } else {
            }
            bool v701;
            v701 = 0 <= v650;
            bool v702;
            v702 = v701 && v651;
            bool v703;
            v703 = v702 == false;
            if (v703){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v702);
            } else {
            }
            bool v705;
            v705 = 0 <= v649;
            bool v707;
            if (v705){
                bool v706;
                v706 = v649 < 4;
                v707 = v706;
            } else {
                v707 = false;
            }
            bool v708;
            v708 = v707 == false;
            if (v708){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v707);
            } else {
            }
            int v710;
            v710 = v649 * 32;
            int v711;
            v711 = v650 + v634;
            int v712;
            v712 = v710 + v633;
            bool v713[4];
            int v714;
            v714 = 0;
            while (while_method_5(v714)){
                int v716;
                v716 = 0;
                while (while_method_8(v716)){
                    assert("Tensor range check" && 0 <= v714 && v714 < 1);
                    assert("Tensor range check" && 0 <= v716 && v716 < 4);
                    int v718;
                    v718 = 4 * v714;
                    int v719;
                    v719 = v718 + v716;
                    int v720;
                    v720 = v659[v719];
                    assert("Tensor range check" && 0 <= v720 && v720 < 32);
                    bool v721;
                    v721 = v530[v720];
                    assert("Tensor range check" && 0 <= v714 && v714 < 1);
                    assert("Tensor range check" && 0 <= v716 && v716 < 4);
                    v713[v719] = v721;
                    v716 += 1 ;
                }
                v714 += 1 ;
            }
            int v722[4];
            int v723;
            v723 = 0;
            while (while_method_5(v723)){
                int v725;
                v725 = 0;
                while (while_method_8(v725)){
                    assert("Tensor range check" && 0 <= v723 && v723 < 1);
                    assert("Tensor range check" && 0 <= v725 && v725 < 4);
                    int v727;
                    v727 = 4 * v723;
                    int v728;
                    v728 = v727 + v725;
                    bool v729;
                    v729 = v713[v728];
                    int v730;
                    if (v729){
                        v730 = 1;
                    } else {
                        v730 = 0;
                    }
                    assert("Tensor range check" && 0 <= v723 && v723 < 1);
                    assert("Tensor range check" && 0 <= v725 && v725 < 4);
                    v722[v728] = v730;
                    v725 += 1 ;
                }
                v723 += 1 ;
            }
            int v731;
            v731 = 0;
            int v732;
            v732 = 0;
            while (while_method_5(v732)){
                int v734;
                v734 = 0;
                while (while_method_8(v734)){
                    assert("Tensor range check" && 0 <= v732 && v732 < 1);
                    assert("Tensor range check" && 0 <= v734 && v734 < 4);
                    int v736;
                    v736 = 4 * v732;
                    int v737;
                    v737 = v736 + v734;
                    int v738;
                    v738 = v722[v737];
                    int v739;
                    v739 = v731 + v738;
                    v731 = v739;
                    v734 += 1 ;
                }
                v732 += 1 ;
            }
            auto v740 = cooperative_groups::coalesced_threads();
            int v741;
            v741 = threadIdx.x;
            int v742;
            v742 = v741 / 8;
            auto v743 = cooperative_groups::labeled_partition(v740,v742);
            Closure1 v744{};
            int v745;
            v745 = cooperative_groups::reduce(v743, v731, v744);
            float v746;
            v746 = (float)v745;
            float v747[4];
            int v748;
            v748 = 0;
            while (while_method_5(v748)){
                int v750;
                v750 = 0;
                while (while_method_8(v750)){
                    assert("Tensor range check" && 0 <= v748 && v748 < 1);
                    assert("Tensor range check" && 0 <= v750 && v750 < 4);
                    int v752;
                    v752 = 4 * v748;
                    int v753;
                    v753 = v752 + v750;
                    float v754;
                    v754 = v658[v753];
                    bool v755;
                    v755 = v713[v753];
                    float v756;
                    if (v755){
                        v756 = v754;
                    } else {
                        v756 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v748 && v748 < 1);
                    assert("Tensor range check" && 0 <= v750 && v750 < 4);
                    v747[v753] = v756;
                    v750 += 1 ;
                }
                v748 += 1 ;
            }
            float v757;
            v757 = 0.0f;
            int v758;
            v758 = 0;
            while (while_method_5(v758)){
                int v760;
                v760 = 0;
                while (while_method_8(v760)){
                    assert("Tensor range check" && 0 <= v758 && v758 < 1);
                    assert("Tensor range check" && 0 <= v760 && v760 < 4);
                    int v762;
                    v762 = 4 * v758;
                    int v763;
                    v763 = v762 + v760;
                    float v764;
                    v764 = v747[v763];
                    float v765;
                    v765 = v757 + v764;
                    v757 = v765;
                    v760 += 1 ;
                }
                v758 += 1 ;
            }
            auto v766 = cooperative_groups::coalesced_threads();
            int v767;
            v767 = threadIdx.x;
            int v768;
            v768 = v767 / 8;
            auto v769 = cooperative_groups::labeled_partition(v766,v768);
            Closure0 v770{};
            float v771;
            v771 = cooperative_groups::reduce(v769, v757, v770);
            float v772;
            v772 = v771 / v746;
            float v773[4];
            int v774;
            v774 = 0;
            while (while_method_5(v774)){
                int v776;
                v776 = 0;
                while (while_method_8(v776)){
                    assert("Tensor range check" && 0 <= v774 && v774 < 1);
                    assert("Tensor range check" && 0 <= v776 && v776 < 4);
                    int v778;
                    v778 = 4 * v774;
                    int v779;
                    v779 = v778 + v776;
                    float v780;
                    v780 = v658[v779];
                    float v781;
                    v781 = v780 - v772;
                    float v782;
                    v782 = v781 * v781;
                    assert("Tensor range check" && 0 <= v774 && v774 < 1);
                    assert("Tensor range check" && 0 <= v776 && v776 < 4);
                    v773[v779] = v782;
                    v776 += 1 ;
                }
                v774 += 1 ;
            }
            float v783[4];
            int v784;
            v784 = 0;
            while (while_method_5(v784)){
                int v786;
                v786 = 0;
                while (while_method_8(v786)){
                    assert("Tensor range check" && 0 <= v784 && v784 < 1);
                    assert("Tensor range check" && 0 <= v786 && v786 < 4);
                    int v788;
                    v788 = 4 * v784;
                    int v789;
                    v789 = v788 + v786;
                    float v790;
                    v790 = v773[v789];
                    bool v791;
                    v791 = v713[v789];
                    float v792;
                    if (v791){
                        v792 = v790;
                    } else {
                        v792 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v784 && v784 < 1);
                    assert("Tensor range check" && 0 <= v786 && v786 < 4);
                    v783[v789] = v792;
                    v786 += 1 ;
                }
                v784 += 1 ;
            }
            float v793;
            v793 = 0.0f;
            int v794;
            v794 = 0;
            while (while_method_5(v794)){
                int v796;
                v796 = 0;
                while (while_method_8(v796)){
                    assert("Tensor range check" && 0 <= v794 && v794 < 1);
                    assert("Tensor range check" && 0 <= v796 && v796 < 4);
                    int v798;
                    v798 = 4 * v794;
                    int v799;
                    v799 = v798 + v796;
                    float v800;
                    v800 = v783[v799];
                    float v801;
                    v801 = v793 + v800;
                    v793 = v801;
                    v796 += 1 ;
                }
                v794 += 1 ;
            }
            auto v802 = cooperative_groups::coalesced_threads();
            int v803;
            v803 = threadIdx.x;
            int v804;
            v804 = v803 / 8;
            auto v805 = cooperative_groups::labeled_partition(v802,v804);
            float v806;
            v806 = cooperative_groups::reduce(v805, v793, v770);
            float v807;
            v807 = v806 / v746;
            float v808;
            v808 = sqrt(v807);
            bool v809;
            v809 = v746 > 1.0f;
            float v813;
            if (v809){
                float v810;
                v810 = v808 * v746;
                float v811;
                v811 = v746 - 1.0f;
                float v812;
                v812 = v810 / v811;
                v813 = v812;
            } else {
                v813 = 0.0f;
            }
            float v814[4];
            int v815;
            v815 = 0;
            while (while_method_5(v815)){
                int v817;
                v817 = 0;
                while (while_method_8(v817)){
                    assert("Tensor range check" && 0 <= v815 && v815 < 1);
                    assert("Tensor range check" && 0 <= v817 && v817 < 4);
                    int v819;
                    v819 = 4 * v815;
                    int v820;
                    v820 = v819 + v817;
                    float v821;
                    v821 = v658[v820];
                    bool v822;
                    v822 = v713[v820];
                    float v823;
                    v823 = curand_normal(&v521);
                    bool v824;
                    v824 = v813 >= 0.1f;
                    float v825;
                    if (v824){
                        v825 = v813;
                    } else {
                        v825 = 0.1f;
                    }
                    float v826;
                    v826 = v823 * v825;
                    float v827;
                    v827 = v826 + v772;
                    float v828;
                    if (v822){
                        v828 = v821;
                    } else {
                        v828 = v827;
                    }
                    assert("Tensor range check" && 0 <= v815 && v815 < 1);
                    assert("Tensor range check" && 0 <= v817 && v817 < 4);
                    v814[v820] = v828;
                    v817 += 1 ;
                }
                v815 += 1 ;
            }
            assert("Tensor range check" && 0 <= v650 && v650 < 64);
            assert("Tensor range check" && 0 <= v649 && v649 < 4);
            int v829;
            v829 = 0;
            while (while_method_5(v829)){
                assert("Tensor range check" && 0 <= v829 && v829 < 1);
                int v831;
                v831 = 32 * v829;
                int v832;
                v832 = v831 + v657;
                assert("Tensor range check" && 0 <= v829 && v829 < 1);
                int v833;
                v833 = 4 * v829;
                int4* v834;
                v834 = reinterpret_cast<int4*>(v814 + v833);
                int4* v835;
                v835 = reinterpret_cast<int4*>(v524 + v832);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v834) % 16 == 0 && reinterpret_cast<unsigned long long>(v835) % 16 == 0);
                *v835 = *v834;
                v829 += 1 ;
            }
            v644 += 24 ;
        }
        v519.sync() ;
        static float v836[8192];
        int v837;
        v837 = threadIdx.x;
        int v838;
        v838 = blockIdx.x;
        int v839;
        v839 = v838 * 256;
        int v840;
        v840 = v837 + v839;
        int v841;
        v841 = v840 / 32;
        int v842;
        v842 = v841;
        while (while_method_13(v842)){
            bool v844;
            v844 = 0 <= v842;
            bool v845;
            v845 = v844 == false;
            if (v845){
                assert("The index needs to be zero or positive." && v844);
            } else {
            }
            int v847;
            v847 = v842 % 128;
            int v848;
            v848 = v842 / 128;
            bool v849;
            v849 = v848 < 64;
            bool v850;
            v850 = v849 == false;
            if (v850){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v849);
            } else {
            }
            assert("Tensor range check" && 0 <= v848 && v848 < 64);
            assert("Tensor range check" && 0 <= v847 && v847 < 128);
            int v852;
            v852 = 32 * v847;
            int v853;
            v853 = 4096 * v848;
            int v854;
            v854 = v853 + v852;
            float v855;
            v855 = 0.0f;
            int v856;
            v856 = threadIdx.x;
            int v857;
            v857 = v856 % 32;
            int v858;
            v858 = v857;
            while (while_method_6(v858)){
                bool v860;
                v860 = 0 <= v858;
                bool v861;
                v861 = v860 == false;
                if (v861){
                    assert("The index needs to be zero or positive." && v860);
                } else {
                }
                bool v863;
                v863 = v858 < 8;
                bool v864;
                v864 = v863 == false;
                if (v864){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v863);
                } else {
                }
                assert("Tensor range check" && 0 <= v858 && v858 < 8);
                int v866;
                v866 = 4 * v858;
                int v867;
                v867 = v866 + v854;
                float v868[4];
                int4* v869;
                v869 = reinterpret_cast<int4*>(v524 + v867);
                int4* v870;
                v870 = reinterpret_cast<int4*>(v868 + 0);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v869) % 16 == 0 && reinterpret_cast<unsigned long long>(v870) % 16 == 0);
                *v870 = *v869;
                int v871;
                v871 = 0;
                while (while_method_8(v871)){
                    assert("Tensor range check" && 0 <= v871 && v871 < 4);
                    float v873;
                    v873 = v868[v871];
                    float v874;
                    v874 = v873 * v873;
                    float v875;
                    v875 = v855 + v874;
                    v855 = v875;
                    v871 += 1 ;
                }
                v858 += 32 ;
            }
            __syncwarp();
            auto v876 = cooperative_groups::coalesced_threads();
            Closure0 v877{};
            float v878;
            v878 = cooperative_groups::reduce(v876, v855, v877);
            float v879;
            v879 = sqrt(v878);
            assert("Tensor range check" && 0 <= v848 && v848 < 64);
            assert("Tensor range check" && 0 <= v847 && v847 < 128);
            int v880;
            v880 = 128 * v848;
            int v881;
            v881 = v880 + v847;
            v836[v881] = v879;
            v842 += 192 ;
        }
        __syncthreads();
        v519.sync() ;
        float v882;
        v882 = 0.0f;
        int v883;
        v883 = threadIdx.x;
        int v884;
        v884 = blockIdx.x;
        int v885;
        v885 = v884 * 256;
        int v886;
        v886 = v883 + v885;
        int v887;
        v887 = v886;
        while (while_method_14(v887)){
            bool v889;
            v889 = 0 <= v887;
            bool v890;
            v890 = v889 == false;
            if (v890){
                assert("The index needs to be zero or positive." && v889);
            } else {
            }
            int v892;
            v892 = v887 % 32;
            int v893;
            v893 = v887 / 32;
            bool v894;
            v894 = v893 < 64;
            bool v895;
            v895 = v894 == false;
            if (v895){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v894);
            } else {
            }
            assert("Tensor range check" && 0 <= v893 && v893 < 64);
            assert("Tensor range check" && 0 <= v892 && v892 < 32);
            int v897;
            v897 = 4 * v892;
            int v898;
            v898 = 128 * v893;
            int v899;
            v899 = v898 + v897;
            float v900[4];
            int4* v901;
            v901 = reinterpret_cast<int4*>(v836 + v899);
            int4* v902;
            v902 = reinterpret_cast<int4*>(v900 + 0);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v901) % 16 == 0 && reinterpret_cast<unsigned long long>(v902) % 16 == 0);
            *v902 = *v901;
            int v903; float v904;
            Tuple13 tmp91 = Tuple13{0, v882};
            v903 = tmp91.v0; v904 = tmp91.v1;
            while (while_method_8(v903)){
                assert("Tensor range check" && 0 <= v903 && v903 < 4);
                float v906;
                v906 = v900[v903];
                bool v907;
                v907 = v904 >= v906;
                float v908;
                if (v907){
                    v908 = v904;
                } else {
                    v908 = v906;
                }
                v904 = v908;
                v903 += 1 ;
            }
            v882 = v904;
            v887 += 6144 ;
        }
        __syncwarp();
        auto v909 = cooperative_groups::coalesced_threads();
        Closure7 v910{};
        float v911;
        v911 = cooperative_groups::reduce(v909, v882, v910);
        int v912;
        v912 = threadIdx.x;
        int v913;
        v913 = v912 / 32;
        extern __shared__ unsigned char v914[];
        float * v915;
        v915 = reinterpret_cast<float *>(&v914[0ull]);
        assert("Tensor range check" && 0 <= v913 && v913 < 8);
        v915[v913] = v911;
        __syncthreads();
        int v917;
        v917 = threadIdx.x;
        int v918;
        v918 = v917 % 32;
        bool v919;
        v919 = v918 < 8;
        float v921;
        if (v919){
            assert("Tensor range check" && 0 <= v918 && v918 < 8);
            float v920;
            v920 = v915[v918];
            v921 = v920;
        } else {
            v921 = 0.0f;
        }
        __syncthreads();
        auto v922 = cooperative_groups::coalesced_threads();
        float v923;
        v923 = cooperative_groups::reduce(v922, v921, v910);
        int v924;
        v924 = blockIdx.x;
        static float v925[24];
        assert("Tensor range check" && 0 <= v924 && v924 < 24);
        v925[v924] = v923;
        v519.sync() ;
        float v926;
        v926 = 0.0f;
        int v927;
        v927 = threadIdx.x;
        int v928;
        v928 = v927 % 32;
        int v929;
        v929 = v928;
        while (while_method_15(v929)){
            bool v931;
            v931 = 0 <= v929;
            bool v932;
            v932 = v931 == false;
            if (v932){
                assert("The index needs to be zero or positive." && v931);
            } else {
            }
            bool v934;
            v934 = v929 < 24;
            bool v935;
            v935 = v934 == false;
            if (v935){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v934);
            } else {
            }
            assert("Tensor range check" && 0 <= v929 && v929 < 24);
            float v937;
            v937 = v925[v929];
            bool v938;
            v938 = v926 >= v937;
            float v939;
            if (v938){
                v939 = v926;
            } else {
                v939 = v937;
            }
            v926 = v939;
            v929 += 32 ;
        }
        __syncwarp();
        auto v940 = cooperative_groups::coalesced_threads();
        float v941;
        v941 = cooperative_groups::reduce(v940, v926, v910);
        int v942;
        v942 = threadIdx.x;
        int v943;
        v943 = blockIdx.x;
        int v944;
        v944 = v943 * 256;
        int v945;
        v945 = v942 + v944;
        bool v946;
        v946 = v945 == 0;
        if (v946){
            cuda::counting_semaphore<cuda::thread_scope_system, 1> & v947 = console_lock;
            auto v948 = cooperative_groups::coalesced_threads();
            v947.acquire();
            printf("{%s = %f}\n","max_norm", v941);
            v947.release();
            v948.sync() ;
        } else {
        }
        __syncwarp();
        extern __shared__ unsigned char v951[];
        float * v952;
        v952 = reinterpret_cast<float *>(&v951[0ull]);
        int v954;
        v954 = blockIdx.x;
        int v955;
        v955 = v954;
        while (while_method_9(v955)){
            bool v957;
            v957 = 0 <= v955;
            bool v958;
            v958 = v957 == false;
            if (v958){
                assert("The index needs to be zero or positive." && v957);
            } else {
            }
            int v960;
            v960 = v955 % 1;
            bool v961;
            v961 = v955 < 16;
            bool v962;
            v962 = v961 == false;
            if (v962){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v961);
            } else {
            }
            assert("Tensor range check" && 0 <= v955 && v955 < 16);
            assert("Tensor range check" && 0 <= v960 && v960 < 1);
            int v964;
            v964 = 32 * v960;
            int v965;
            v965 = 16384 * v955;
            int v966;
            v966 = v965 + v964;
            int v967;
            v967 = 262144 * v960;
            int v968;
            v968 = 512 * v955;
            int v969;
            v969 = v968 + v967;
            int v970;
            v970 = threadIdx.x;
            int v971;
            v971 = v970;
            while (while_method_12(v971)){
                bool v973;
                v973 = 0 <= v971;
                bool v974;
                v974 = v973 == false;
                if (v974){
                    assert("The index needs to be zero or positive." && v973);
                } else {
                }
                int v976;
                v976 = v971 % 32;
                int v977;
                v977 = v971 / 32;
                bool v978;
                v978 = v977 < 512;
                bool v979;
                v979 = v978 == false;
                if (v979){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v978);
                } else {
                }
                assert("Tensor range check" && 0 <= v977 && v977 < 512);
                assert("Tensor range check" && 0 <= v976 && v976 < 32);
                int v981;
                v981 = v976 + v966;
                int v982;
                v982 = 32 * v977;
                int v983;
                v983 = v982 + v981;
                float v984;
                v984 = v524[v983];
                assert("Tensor range check" && 0 <= v977 && v977 < 512);
                assert("Tensor range check" && 0 <= v976 && v976 < 32);
                int v985;
                v985 = 33 * v977;
                int v986;
                v986 = v985 + v976;
                v952[v986] = v984;
                v971 += 256 ;
            }
            __syncthreads();
            int v987;
            v987 = threadIdx.x;
            int v988;
            v988 = v987;
            while (while_method_12(v988)){
                bool v990;
                v990 = 0 <= v988;
                bool v991;
                v991 = v990 == false;
                if (v991){
                    assert("The index needs to be zero or positive." && v990);
                } else {
                }
                int v993;
                v993 = v988 % 512;
                int v994;
                v994 = v988 / 512;
                bool v995;
                v995 = v994 < 32;
                bool v996;
                v996 = v995 == false;
                if (v996){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v995);
                } else {
                }
                assert("Tensor range check" && 0 <= v994 && v994 < 32);
                assert("Tensor range check" && 0 <= v993 && v993 < 512);
                int v998;
                v998 = 33 * v993;
                int v999;
                v999 = v994 + v998;
                float v1000;
                v1000 = v952[v999];
                assert("Tensor range check" && 0 <= v994 && v994 < 32);
                assert("Tensor range check" && 0 <= v993 && v993 < 512);
                int v1001;
                v1001 = v993 + v969;
                int v1002;
                v1002 = 8192 * v994;
                int v1003;
                v1003 = v1002 + v1001;
                v522[v1003] = v1000;
                v988 += 256 ;
            }
            __syncthreads();
            v955 += 24 ;
        }
        v519.sync() ;
        v27 += 1 ;
    }
    cooperative_groups::grid_group & v1004 = v26.v1;
    cooperative_groups::grid_group & v1005 = v1004;
    int v1006;
    v1006 = threadIdx.x;
    int v1007;
    v1007 = blockIdx.x;
    int v1008;
    v1008 = v1007 * 256;
    int v1009;
    v1009 = v1006 + v1008;
    int v1010;
    v1010 = v1009;
    while (while_method_14(v1010)){
        bool v1012;
        v1012 = 0 <= v1010;
        bool v1013;
        v1013 = v1012 == false;
        if (v1013){
            assert("The index needs to be zero or positive." && v1012);
        } else {
        }
        int v1015;
        v1015 = v1010 % 64;
        int v1016;
        v1016 = v1010 / 64;
        bool v1017;
        v1017 = v1016 < 32;
        bool v1018;
        v1018 = v1017 == false;
        if (v1018){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1017);
        } else {
        }
        assert("Tensor range check" && 0 <= v1016 && v1016 < 32);
        assert("Tensor range check" && 0 <= v1015 && v1015 < 64);
        int v1020;
        v1020 = 4 * v1015;
        int v1021;
        v1021 = 256 * v1016;
        int v1022;
        v1022 = v1021 + v1020;
        assert("Tensor range check" && 0 <= v1016 && v1016 < 32);
        assert("Tensor range check" && 0 <= v1015 && v1015 < 64);
        float v1023[4];
        float v1024[4];
        float v1025[4];
        int4* v1026;
        v1026 = reinterpret_cast<int4*>(v3 + v1022);
        int4* v1027;
        v1027 = reinterpret_cast<int4*>(v1023 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1026) % 16 == 0 && reinterpret_cast<unsigned long long>(v1027) % 16 == 0);
        *v1027 = *v1026;
        int4* v1028;
        v1028 = reinterpret_cast<int4*>(v4 + v1022);
        int4* v1029;
        v1029 = reinterpret_cast<int4*>(v1024 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1028) % 16 == 0 && reinterpret_cast<unsigned long long>(v1029) % 16 == 0);
        *v1029 = *v1028;
        // Pushing the loop unrolling to: 0
        int v1030;
        v1030 = 0;
        #pragma unroll
        while (while_method_8(v1030)){
            assert("Tensor range check" && 0 <= v1030 && v1030 < 4);
            float v1032;
            v1032 = v1023[v1030];
            float v1033;
            v1033 = v1024[v1030];
            bool v1034;
            v1034 = v1033 == 0.0f;
            bool v1035;
            v1035 = v1034 != true;
            float v1037;
            if (v1035){
                float v1036;
                v1036 = v1032 / v1033;
                v1037 = v1036;
            } else {
                v1037 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v1030 && v1030 < 4);
            v1025[v1030] = v1037;
            v1030 += 1 ;
        }
        // Poping the loop unrolling to: 0
        int4* v1038;
        v1038 = reinterpret_cast<int4*>(v1025 + 0);
        int4* v1039;
        v1039 = reinterpret_cast<int4*>(v5 + v1022);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1038) % 16 == 0 && reinterpret_cast<unsigned long long>(v1039) % 16 == 0);
        *v1039 = *v1038;
        v1010 += 6144 ;
    }
    v1005.sync() ;
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
        v0 = cp.empty(1048576,dtype=cp.uint8)
        v1 = cp.empty(61341696,dtype=cp.uint8)
        v2 = cp.empty(1048880,dtype=cp.uint8)
        v4 = v1[0:0+4*786432].view(cp.float32)
        del v4
        v6 = v2[0:0+4*262144].view(cp.float32)
        v8 = v0[0:0+4*262144].view(cp.float32)
        del v8
        v10 = v1[4718592:4718592+4*12582912].view(cp.float32)
        del v10
        v12 = v2[1048576:1048576+4*1].view(cp.int32)
        v14 = v2[1048592:1048592+1*32].view(cp.bool_)
        v16 = v2[1048624:1048624+4*32].view(cp.float32)
        v18 = v2[1048752:1048752+4*32].view(cp.float32)
        v20 = v1[55050240:55050240+8*393216].view(cp.float64)
        v22 = v1[58195968:58195968+8*393216].view(cp.float64)
        v23 = cp.random.normal(0.0,0.1,262144,dtype=cp.float32) # type: ignore
        cp.copyto(v6[0:0+262144],v23[0:0+262144])
        del v6, v23
        v12[:] = 0
        del v12
        v16[:] = 0
        del v16
        v18[:] = 0
        del v18
        v14[:] = 1
        del v14
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
        return method115(v32, v33, v31, v25, v34, v2, v1, v0)
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
