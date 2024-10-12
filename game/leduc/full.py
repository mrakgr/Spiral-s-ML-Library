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
        while (while_method_4(v99)){
            int v101;
            v101 = 0;
            while (while_method_7(v101)){
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
        while (while_method_4(v114)){
            int v116;
            v116 = 0;
            while (while_method_7(v116)){
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
        while (while_method_4(v123)){
            int v125;
            v125 = 0;
            while (while_method_7(v125)){
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
        while (while_method_4(v144)){
            int v146;
            v146 = 0;
            while (while_method_7(v146)){
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
__device__ inline bool while_method_9(int v0){
    bool v1;
    v1 = v0 < 256;
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
__device__ inline bool while_method_11(int v0){
    bool v1;
    v1 = v0 < 32;
    return v1;
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
        Union3 v920;
        switch (v20.tag) {
            case 0: { // None
                v920 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v22 = v20.case1.v0;
                Union14 v760;
                switch (v22.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v729 = v22.case0.v0; bool v730 = v22.case0.v1; static_array<Union6,2> v731 = v22.case0.v2; int v732 = v22.case0.v3; static_array<int,2> v733 = v22.case0.v4; int v734 = v22.case0.v5;
                        curandStatePhilox4_32_10_t & v735 = v3.v5;
                        curandStatePhilox4_32_10_t & v736 = v735;
                        unsigned int & v737 = v3.v0;
                        Union6 v738; unsigned int v739;
                        Tuple6 tmp35 = draw_card_20(v736, v737);
                        v738 = tmp35.v0; v739 = tmp35.v1;
                        v3.v0 = v739;
                        Union7 v740;
                        v740 = Union7{Union7_0{v738}};
                        v18.push(v740);
                        v760 = Union14{Union14_0{v729, v730, v731, v732, v733, v734, v738}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v742 = v3.v5;
                        curandStatePhilox4_32_10_t & v743 = v742;
                        unsigned int & v744 = v3.v0;
                        Union6 v745; unsigned int v746;
                        Tuple6 tmp36 = draw_card_20(v743, v744);
                        v745 = tmp36.v0; v746 = tmp36.v1;
                        v3.v0 = v746;
                        curandStatePhilox4_32_10_t & v747 = v3.v5;
                        curandStatePhilox4_32_10_t & v748 = v747;
                        unsigned int & v749 = v3.v0;
                        Union6 v750; unsigned int v751;
                        Tuple6 tmp37 = draw_card_20(v748, v749);
                        v750 = tmp37.v0; v751 = tmp37.v1;
                        v3.v0 = v751;
                        Union7 v752;
                        v752 = Union7{Union7_2{0, v745}};
                        v18.push(v752);
                        Union7 v753;
                        v753 = Union7{Union7_2{1, v750}};
                        v18.push(v753);
                        v760 = Union14{Union14_1{v745, v750}};
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
                        Union1 v717;
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
                                while (while_method_11(v159)){
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
                                    float * v175;
                                    v175 = reinterpret_cast<float *>(&v0[1048592ull]);
                                    float * v177;
                                    v177 = reinterpret_cast<float *>(&v0[1048720ull]);
                                    double * v179;
                                    v179 = reinterpret_cast<double *>(&v1[55050240ull]);
                                    double * v181;
                                    v181 = reinterpret_cast<double *>(&v1[58195968ull]);
                                    v159 += 1 ;
                                }
                                __syncthreads();
                                int * v183;
                                v183 = reinterpret_cast<int *>(&v0[1048576ull]);
                                float * v185;
                                v185 = reinterpret_cast<float *>(&v0[1048592ull]);
                                float * v187;
                                v187 = reinterpret_cast<float *>(&v0[1048720ull]);
                                int v189;
                                v189 = v183[0];
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
                                while (while_method_8(v236)){
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
                                    while (while_method_4(v258)){
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
                                    while (while_method_4(v265)){
                                        int v267;
                                        v267 = 0;
                                        while (while_method_7(v267)){
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
                                    while (while_method_4(v292)){
                                        assert("Tensor range check" && 0 <= v292 && v292 < 1);
                                        int v294;
                                        v294 = 4 * v292;
                                        assert("Tensor range check" && 0 <= v292 && v292 < 1);
                                        float v295;
                                        v295 = 0.0f;
                                        int v296;
                                        v296 = 0;
                                        while (while_method_7(v296)){
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
                                        while (while_method_7(v313)){
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
                                    while (while_method_4(v321)){
                                        int v323;
                                        v323 = 0;
                                        while (while_method_7(v323)){
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
                                    Tuple8 tmp38 = Tuple8{-1.0f / 0.0f, false};
                                    v330 = tmp38.v0; v331 = tmp38.v1;
                                    int v332;
                                    v332 = 0;
                                    while (while_method_4(v332)){
                                        int v334;
                                        v334 = 0;
                                        while (while_method_7(v334)){
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
                                    Tuple8 tmp39 = cooperative_groups::reduce(v351, Tuple8{v330, v331}, v352);
                                    v353 = tmp39.v0; v354 = tmp39.v1;
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
                                        while (while_method_4(v363)){
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
                                            while (while_method_7(v369)){
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
                                    while (while_method_4(v420)){
                                        int v422;
                                        v422 = 0;
                                        while (while_method_7(v422)){
                                            assert("Tensor range check" && 0 <= v420 && v420 < 1);
                                            assert("Tensor range check" && 0 <= v422 && v422 < 4);
                                            int v424;
                                            v424 = 4 * v420;
                                            int v425;
                                            v425 = v424 + v422;
                                            int v426;
                                            v426 = v257[v425];
                                            float v427;
                                            v427 = curand_uniform(&v89);
                                            assert("Tensor range check" && 0 <= v420 && v420 < 1);
                                            assert("Tensor range check" && 0 <= v422 && v422 < 4);
                                            v418[v425] = v427;
                                            v419[v425] = v426;
                                            v422 += 1 ;
                                        }
                                        v420 += 1 ;
                                    }
                                    float v428; int v429;
                                    Tuple9 tmp40 = Tuple9{0.0f, 2147483647};
                                    v428 = tmp40.v0; v429 = tmp40.v1;
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
                                    Tuple9 tmp41 = cooperative_groups::reduce(v444, Tuple9{v428, v429}, v445);
                                    v446 = tmp41.v0; v447 = tmp41.v1;
                                    float v448;
                                    v448 = v353 * v446;
                                    int v449[4];
                                    bool v450[4];
                                    int v451;
                                    v451 = 0;
                                    while (while_method_4(v451)){
                                        int v453;
                                        v453 = 0;
                                        while (while_method_7(v453)){
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
                                    Tuple10 tmp42 = Tuple10{2147483647, false};
                                    v464 = tmp42.v0; v465 = tmp42.v1;
                                    int v466;
                                    v466 = 0;
                                    while (while_method_4(v466)){
                                        int v468;
                                        v468 = 0;
                                        while (while_method_7(v468)){
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
                                    Tuple10 tmp43 = cooperative_groups::reduce(v485, Tuple10{v464, v465}, v486);
                                    v487 = tmp43.v0; v488 = tmp43.v1;
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
                                        while (while_method_4(v497)){
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
                                            while (while_method_7(v503)){
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
                                    Tuple9 tmp44 = Tuple9{0.0f, 2147483647};
                                    v552 = tmp44.v0; v553 = tmp44.v1;
                                    int v554;
                                    v554 = 0;
                                    while (while_method_4(v554)){
                                        int v556;
                                        v556 = 0;
                                        while (while_method_7(v556)){
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
                                    Tuple9 tmp45 = cooperative_groups::reduce(v571, Tuple9{v552, v553}, v572);
                                    v573 = tmp45.v0; v574 = tmp45.v1;
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
                                    while (while_method_4(v579)){
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
                                while (while_method_11(v600)){
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
                                    assert("Tensor range check" && 0 <= v75 && v75 < 2);
                                    int v617;
                                    v617 = 2 * v599;
                                    int v618;
                                    v618 = v617 + v75;
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
                                    assert("Tensor range check" && 0 <= v75 && v75 < 2);
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
                                        v643 = v76[0];
                                        int v645; int v646;
                                        Tuple7 tmp46 = Tuple7{1, v643};
                                        v645 = tmp46.v0; v646 = tmp46.v1;
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
                                            v653 = v76[v645];
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
                                        if (v79){
                                            bool v657;
                                            v657 = v75 < 2;
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
                                        v661 = v76[v75];
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
                                        v668 = v77 > 0;
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
                                v681 = v76[0];
                                int v683;
                                v683 = v76[1];
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
                                v688 = v77 > 0;
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
                        v718 = Union7{Union7_1{v75, v717}};
                        v18.push(v718);
                        v760 = Union14{Union14_2{v72, v73, v74, v75, v76, v77, v717}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v720 = v22.case3.v0; bool v721 = v22.case3.v1; static_array<Union6,2> v722 = v22.case3.v2; int v723 = v22.case3.v3; static_array<int,2> v724 = v22.case3.v4; int v725 = v22.case3.v5; Union1 v726 = v22.case3.v6;
                        Union7 v727;
                        v727 = Union7{Union7_1{v723, v726}};
                        v18.push(v727);
                        v760 = Union14{Union14_2{v720, v721, v722, v723, v724, v725, v726}};
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
                        v760 = Union14{Union14_3{}};
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
                        Tuple7 tmp47 = Tuple7{0, 0};
                        v770 = tmp47.v0; v771 = tmp47.v1;
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
                                            Tuple7 tmp48 = Tuple7{0, 0};
                                            v875 = tmp48.v0; v876 = tmp48.v1;
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
                                            Tuple7 tmp49 = Tuple7{0, 0};
                                            v810 = tmp49.v0; v811 = tmp49.v1;
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
                                            Tuple7 tmp50 = Tuple7{0, 0};
                                            v831 = tmp50.v0; v832 = tmp50.v1;
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
        v20 = v920;
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
    while (while_method_10(v16)){
        Union3 v916;
        switch (v16.tag) {
            case 0: { // None
                v916 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v18 = v16.case1.v0;
                Union14 v756;
                switch (v18.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v725 = v18.case0.v0; bool v726 = v18.case0.v1; static_array<Union6,2> v727 = v18.case0.v2; int v728 = v18.case0.v3; static_array<int,2> v729 = v18.case0.v4; int v730 = v18.case0.v5;
                        curandStatePhilox4_32_10_t & v731 = v3.v5;
                        curandStatePhilox4_32_10_t & v732 = v731;
                        unsigned int & v733 = v3.v0;
                        Union6 v734; unsigned int v735;
                        Tuple6 tmp55 = draw_card_20(v732, v733);
                        v734 = tmp55.v0; v735 = tmp55.v1;
                        v3.v0 = v735;
                        Union7 v736;
                        v736 = Union7{Union7_0{v734}};
                        v14.push(v736);
                        v756 = Union14{Union14_0{v725, v726, v727, v728, v729, v730, v734}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v738 = v3.v5;
                        curandStatePhilox4_32_10_t & v739 = v738;
                        unsigned int & v740 = v3.v0;
                        Union6 v741; unsigned int v742;
                        Tuple6 tmp56 = draw_card_20(v739, v740);
                        v741 = tmp56.v0; v742 = tmp56.v1;
                        v3.v0 = v742;
                        curandStatePhilox4_32_10_t & v743 = v3.v5;
                        curandStatePhilox4_32_10_t & v744 = v743;
                        unsigned int & v745 = v3.v0;
                        Union6 v746; unsigned int v747;
                        Tuple6 tmp57 = draw_card_20(v744, v745);
                        v746 = tmp57.v0; v747 = tmp57.v1;
                        v3.v0 = v747;
                        Union7 v748;
                        v748 = Union7{Union7_2{0, v741}};
                        v14.push(v748);
                        Union7 v749;
                        v749 = Union7{Union7_2{1, v746}};
                        v14.push(v749);
                        v756 = Union14{Union14_1{v741, v746}};
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
                        Union1 v713;
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
                                while (while_method_11(v155)){
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
                                    float * v171;
                                    v171 = reinterpret_cast<float *>(&v0[1048592ull]);
                                    float * v173;
                                    v173 = reinterpret_cast<float *>(&v0[1048720ull]);
                                    double * v175;
                                    v175 = reinterpret_cast<double *>(&v1[55050240ull]);
                                    double * v177;
                                    v177 = reinterpret_cast<double *>(&v1[58195968ull]);
                                    v155 += 1 ;
                                }
                                __syncthreads();
                                int * v179;
                                v179 = reinterpret_cast<int *>(&v0[1048576ull]);
                                float * v181;
                                v181 = reinterpret_cast<float *>(&v0[1048592ull]);
                                float * v183;
                                v183 = reinterpret_cast<float *>(&v0[1048720ull]);
                                int v185;
                                v185 = v179[0];
                                float * v186;
                                v186 = reinterpret_cast<float *>(&v1[4718592ull]);
                                assert("Tensor range check" && 0 <= v185 && v185 < 32);
                                int v188;
                                v188 = 393216 * v185;
                                int v189;
                                v189 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v189 && v189 < 24);
                                int v190;
                                v190 = 16384 * v189;
                                int v191;
                                v191 = v190 + v188;
                                int v192;
                                v192 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v192 && v192 < 256);
                                int v193;
                                v193 = 64 * v192;
                                int v194;
                                v194 = v193 + v191;
                                float * v195;
                                v195 = v186+v194;
                                int v197;
                                v197 = sizeof(float *);
                                unsigned long long v198;
                                v198 = (unsigned long long)v197;
                                unsigned long long v199;
                                v199 = 256ull * v198;
                                unsigned long long v200;
                                v200 = v199 + 16ull;
                                unsigned long long v201;
                                v201 = v200 - 1ull;
                                unsigned long long v202;
                                v202 = v201 % 16ull;
                                unsigned long long v203;
                                v203 = v201 - v202;
                                unsigned long long v204;
                                v204 = v203 + 1024ull;
                                unsigned long long v205;
                                v205 = v204 + 16ull;
                                unsigned long long v206;
                                v206 = v205 - 1ull;
                                unsigned long long v207;
                                v207 = v206 % 16ull;
                                unsigned long long v208;
                                v208 = v206 - v207;
                                unsigned long long v209;
                                v209 = v208 + 1024ull;
                                bool v210;
                                v210 = v209 <= 98304ull;
                                bool v211;
                                v211 = v210 == false;
                                if (v211){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v210);
                                } else {
                                }
                                extern __shared__ unsigned char v213[];
                                bool v214;
                                v214 = v209 <= v209;
                                bool v215;
                                v215 = v214 == false;
                                if (v215){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v214);
                                } else {
                                }
                                float * * v217;
                                v217 = reinterpret_cast<float * *>(&v213[0ull]);
                                float * v219;
                                v219 = reinterpret_cast<float *>(&v213[v203]);
                                int * v221;
                                v221 = reinterpret_cast<int *>(&v213[v208]);
                                int v223;
                                v223 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v223 && v223 < 256);
                                v217[v223] = v195;
                                __syncthreads();
                                bool v224;
                                v224 = 0 <= v223;
                                bool v225;
                                v225 = v224 == false;
                                if (v225){
                                    assert("The index needs to be zero or positive." && v224);
                                } else {
                                }
                                int v227;
                                v227 = v223 % 16;
                                int v228;
                                v228 = v223 / 16;
                                bool v229;
                                v229 = v228 < 16;
                                bool v230;
                                v230 = v229 == false;
                                if (v230){
                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v229);
                                } else {
                                }
                                assert("Tensor range check" && 0 <= v228 && v228 < 16);
                                int v232;
                                v232 = 0;
                                while (while_method_8(v232)){
                                    bool v234;
                                    v234 = 0 <= v228;
                                    bool v235;
                                    v235 = v234 && v229;
                                    bool v236;
                                    v236 = v235 == false;
                                    if (v236){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v235);
                                    } else {
                                    }
                                    bool v238;
                                    v238 = 0 <= v232;
                                    bool v240;
                                    if (v238){
                                        bool v239;
                                        v239 = v232 < 16;
                                        v240 = v239;
                                    } else {
                                        v240 = false;
                                    }
                                    bool v241;
                                    v241 = v240 == false;
                                    if (v241){
                                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v240);
                                    } else {
                                    }
                                    int v243;
                                    v243 = v232 * 16;
                                    int v244;
                                    v244 = v243 + v228;
                                    assert("Tensor range check" && 0 <= v232 && v232 < 16);
                                    int v245;
                                    v245 = 16 * v232;
                                    int v246;
                                    v246 = v245 + v228;
                                    float * v247;
                                    v247 = v217[v246];
                                    int v248;
                                    v248 = blockIdx.x;
                                    int v249;
                                    v249 = v248 * 256;
                                    int v250;
                                    v250 = v249 + v244;
                                    assert("Tensor range check" && 0 <= v227 && v227 < 16);
                                    int v251;
                                    v251 = 4 * v227;
                                    float v252[4];
                                    int v253[4];
                                    int v254;
                                    v254 = 0;
                                    while (while_method_4(v254)){
                                        assert("Tensor range check" && 0 <= v254 && v254 < 1);
                                        int v256;
                                        v256 = 4 * v254;
                                        assert("Tensor range check" && 0 <= v254 && v254 < 1);
                                        int v257;
                                        v257 = 64 * v254;
                                        int v258;
                                        v258 = v257 + v251;
                                        int4* v259;
                                        v259 = reinterpret_cast<int4*>(v247 + v258);
                                        int4* v260;
                                        v260 = reinterpret_cast<int4*>(v252 + v256);
                                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v259) % 16 == 0 && reinterpret_cast<unsigned long long>(v260) % 16 == 0);
                                        *v260 = *v259;
                                        v254 += 1 ;
                                    }
                                    int v261;
                                    v261 = 0;
                                    while (while_method_4(v261)){
                                        int v263;
                                        v263 = 0;
                                        while (while_method_7(v263)){
                                            bool v265;
                                            v265 = 0 <= v263;
                                            bool v267;
                                            if (v265){
                                                bool v266;
                                                v266 = v263 < 4;
                                                v267 = v266;
                                            } else {
                                                v267 = false;
                                            }
                                            bool v268;
                                            v268 = v267 == false;
                                            if (v268){
                                                assert("The indices should be inside the range of the dimension." && v267);
                                            } else {
                                            }
                                            bool v270;
                                            v270 = 0 <= v227;
                                            bool v272;
                                            if (v270){
                                                bool v271;
                                                v271 = v227 < 16;
                                                v272 = v271;
                                            } else {
                                                v272 = false;
                                            }
                                            bool v273;
                                            v273 = v272 == false;
                                            if (v273){
                                                assert("The indices should be inside the range of the dimension." && v272);
                                            } else {
                                            }
                                            int v275;
                                            v275 = v227 * 4;
                                            int v276;
                                            v276 = v263 + v275;
                                            bool v277;
                                            v277 = 0 <= v261;
                                            bool v279;
                                            if (v277){
                                                bool v278;
                                                v278 = v261 < 1;
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
                                            int v282;
                                            v282 = v261 * 64;
                                            int v283;
                                            v283 = v276 + v282;
                                            assert("Tensor range check" && 0 <= v261 && v261 < 1);
                                            assert("Tensor range check" && 0 <= v263 && v263 < 4);
                                            int v284;
                                            v284 = 4 * v261;
                                            int v285;
                                            v285 = v284 + v263;
                                            v253[v285] = v283;
                                            v263 += 1 ;
                                        }
                                        v261 += 1 ;
                                    }
                                    float v286[4];
                                    float v287;
                                    v287 = 0.0f;
                                    int v288;
                                    v288 = 0;
                                    while (while_method_4(v288)){
                                        assert("Tensor range check" && 0 <= v288 && v288 < 1);
                                        int v290;
                                        v290 = 4 * v288;
                                        assert("Tensor range check" && 0 <= v288 && v288 < 1);
                                        float v291;
                                        v291 = 0.0f;
                                        int v292;
                                        v292 = 0;
                                        while (while_method_7(v292)){
                                            assert("Tensor range check" && 0 <= v292 && v292 < 4);
                                            int v294;
                                            v294 = v292 + v290;
                                            float v295;
                                            v295 = v252[v294];
                                            float v296;
                                            v296 = v291 + v295;
                                            v291 = v296;
                                            v292 += 1 ;
                                        }
                                        auto v297 = cooperative_groups::coalesced_threads();
                                        int v298;
                                        v298 = threadIdx.x;
                                        int v299;
                                        v299 = v298 / 16;
                                        auto v300 = cooperative_groups::labeled_partition(v297,v299);
                                        Closure2 v301{};
                                        float v302;
                                        v302 = cooperative_groups::inclusive_scan(v300, v291, v301);
                                        float v303;
                                        v303 = v300.shfl_up(v302,1);
                                        bool v304;
                                        v304 = v300.thread_rank() == 0;
                                        float v305;
                                        if (v304){
                                            v305 = 0.0f;
                                        } else {
                                            v305 = v303;
                                        }
                                        float v306;
                                        v306 = v300.shfl(v302,v300.num_threads()-1);
                                        float v307;
                                        v307 = v287 + v305;
                                        float v308;
                                        v308 = v307;
                                        int v309;
                                        v309 = 0;
                                        while (while_method_7(v309)){
                                            assert("Tensor range check" && 0 <= v309 && v309 < 4);
                                            int v311;
                                            v311 = v309 + v290;
                                            float v312;
                                            v312 = v252[v311];
                                            float v313;
                                            v313 = v308 + v312;
                                            assert("Tensor range check" && 0 <= v309 && v309 < 4);
                                            v286[v311] = v313;
                                            v308 = v313;
                                            v309 += 1 ;
                                        }
                                        float v314;
                                        v314 = v287 + v306;
                                        v287 = v314;
                                        v288 += 1 ;
                                    }
                                    float v315[4];
                                    bool v316[4];
                                    int v317;
                                    v317 = 0;
                                    while (while_method_4(v317)){
                                        int v319;
                                        v319 = 0;
                                        while (while_method_7(v319)){
                                            assert("Tensor range check" && 0 <= v317 && v317 < 1);
                                            assert("Tensor range check" && 0 <= v319 && v319 < 4);
                                            int v321;
                                            v321 = 4 * v317;
                                            int v322;
                                            v322 = v321 + v319;
                                            float v323;
                                            v323 = v286[v322];
                                            float v324;
                                            v324 = v252[v322];
                                            bool v325;
                                            v325 = v324 > 0.0f;
                                            assert("Tensor range check" && 0 <= v317 && v317 < 1);
                                            assert("Tensor range check" && 0 <= v319 && v319 < 4);
                                            v315[v322] = v323;
                                            v316[v322] = v325;
                                            v319 += 1 ;
                                        }
                                        v317 += 1 ;
                                    }
                                    float v326; bool v327;
                                    Tuple8 tmp58 = Tuple8{-1.0f / 0.0f, false};
                                    v326 = tmp58.v0; v327 = tmp58.v1;
                                    int v328;
                                    v328 = 0;
                                    while (while_method_4(v328)){
                                        int v330;
                                        v330 = 0;
                                        while (while_method_7(v330)){
                                            assert("Tensor range check" && 0 <= v328 && v328 < 1);
                                            assert("Tensor range check" && 0 <= v330 && v330 < 4);
                                            int v332;
                                            v332 = 4 * v328;
                                            int v333;
                                            v333 = v332 + v330;
                                            float v334;
                                            v334 = v315[v333];
                                            bool v335;
                                            v335 = v316[v333];
                                            float v342; bool v343;
                                            if (v327){
                                                if (v335){
                                                    bool v336;
                                                    v336 = v326 >= v334;
                                                    float v337;
                                                    if (v336){
                                                        v337 = v326;
                                                    } else {
                                                        v337 = v334;
                                                    }
                                                    v342 = v337; v343 = true;
                                                } else {
                                                    v342 = v326; v343 = v327;
                                                }
                                            } else {
                                                if (v335){
                                                    v342 = v334; v343 = v335;
                                                } else {
                                                    v342 = v326; v343 = v327;
                                                }
                                            }
                                            v326 = v342;
                                            v327 = v343;
                                            v330 += 1 ;
                                        }
                                        v328 += 1 ;
                                    }
                                    auto v344 = cooperative_groups::coalesced_threads();
                                    int v345;
                                    v345 = threadIdx.x;
                                    int v346;
                                    v346 = v345 / 16;
                                    auto v347 = cooperative_groups::labeled_partition(v344,v346);
                                    Closure3 v348{};
                                    float v349; bool v350;
                                    Tuple8 tmp59 = cooperative_groups::reduce(v347, Tuple8{v326, v327}, v348);
                                    v349 = tmp59.v0; v350 = tmp59.v1;
                                    bool v351;
                                    v351 = v350 == false;
                                    if (v351){
                                        int v352;
                                        v352 = threadIdx.x;
                                        int v353;
                                        v353 = blockIdx.x;
                                        int v354;
                                        v354 = v353 * 256;
                                        int v355;
                                        v355 = v352 + v354;
                                        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v356 = console_lock;
                                        auto v357 = cooperative_groups::coalesced_threads();
                                        v356.acquire();
                                        int v358;
                                        v358 = 0;
                                        printf("{%s = %d; %s = %c","tid", v355, "x'", '[');
                                        int v359;
                                        v359 = 0;
                                        while (while_method_4(v359)){
                                            int v361;
                                            v361 = v358;
                                            bool v362;
                                            v362 = v361 >= 100;
                                            if (v362){
                                                printf("%s"," ...");
                                                break;
                                            } else {
                                            }
                                            bool v363;
                                            v363 = v359 == 0;
                                            bool v364;
                                            v364 = v363 != true;
                                            if (v364){
                                                printf("%s","; ");
                                            } else {
                                            }
                                            printf("%c",'[');
                                            int v365;
                                            v365 = 0;
                                            while (while_method_7(v365)){
                                                int v367;
                                                v367 = v358;
                                                bool v368;
                                                v368 = v367 >= 100;
                                                if (v368){
                                                    printf("%s"," ...");
                                                    break;
                                                } else {
                                                }
                                                bool v369;
                                                v369 = v365 == 0;
                                                bool v370;
                                                v370 = v369 != true;
                                                if (v370){
                                                    printf("%s","; ");
                                                } else {
                                                }
                                                int v371;
                                                v371 = v358 + 1;
                                                v358 = v371;
                                                int v372;
                                                v372 = v359 * 4;
                                                int v373;
                                                v373 = v372 + v365;
                                                float v374;
                                                v374 = v315[v373];
                                                bool v375;
                                                v375 = v316[v373];
                                                const char * v378;
                                                if (v375){
                                                    const char * v376;
                                                    v376 = "true";
                                                    v378 = v376;
                                                } else {
                                                    const char * v377;
                                                    v377 = "false";
                                                    v378 = v377;
                                                }
                                                printf("%f, %s",v374, v378);
                                                v365 += 1 ;
                                            }
                                            printf("%c",']');
                                            v359 += 1 ;
                                        }
                                        printf("%c",']');
                                        printf("}\n");
                                        v356.release();
                                        v357.sync() ;
                                    } else {
                                    }
                                    if (v351){
                                        assert("The local reduce must be true." && v350);
                                    } else {
                                    }
                                    float v414[4];
                                    int v415[4];
                                    int v416;
                                    v416 = 0;
                                    while (while_method_4(v416)){
                                        int v418;
                                        v418 = 0;
                                        while (while_method_7(v418)){
                                            assert("Tensor range check" && 0 <= v416 && v416 < 1);
                                            assert("Tensor range check" && 0 <= v418 && v418 < 4);
                                            int v420;
                                            v420 = 4 * v416;
                                            int v421;
                                            v421 = v420 + v418;
                                            int v422;
                                            v422 = v253[v421];
                                            float v423;
                                            v423 = curand_uniform(&v85);
                                            assert("Tensor range check" && 0 <= v416 && v416 < 1);
                                            assert("Tensor range check" && 0 <= v418 && v418 < 4);
                                            v414[v421] = v423;
                                            v415[v421] = v422;
                                            v418 += 1 ;
                                        }
                                        v416 += 1 ;
                                    }
                                    float v424; int v425;
                                    Tuple9 tmp60 = Tuple9{0.0f, 2147483647};
                                    v424 = tmp60.v0; v425 = tmp60.v1;
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
                                            v432 = v414[v431];
                                            int v433;
                                            v433 = v415[v431];
                                            bool v434;
                                            v434 = v425 < v433;
                                            float v435; int v436;
                                            if (v434){
                                                v435 = v424; v436 = v425;
                                            } else {
                                                v435 = v432; v436 = v433;
                                            }
                                            v424 = v435;
                                            v425 = v436;
                                            v428 += 1 ;
                                        }
                                        v426 += 1 ;
                                    }
                                    auto v437 = cooperative_groups::coalesced_threads();
                                    int v438;
                                    v438 = threadIdx.x;
                                    int v439;
                                    v439 = v438 / 16;
                                    auto v440 = cooperative_groups::labeled_partition(v437,v439);
                                    Closure4 v441{};
                                    float v442; int v443;
                                    Tuple9 tmp61 = cooperative_groups::reduce(v440, Tuple9{v424, v425}, v441);
                                    v442 = tmp61.v0; v443 = tmp61.v1;
                                    float v444;
                                    v444 = v349 * v442;
                                    int v445[4];
                                    bool v446[4];
                                    int v447;
                                    v447 = 0;
                                    while (while_method_4(v447)){
                                        int v449;
                                        v449 = 0;
                                        while (while_method_7(v449)){
                                            assert("Tensor range check" && 0 <= v447 && v447 < 1);
                                            assert("Tensor range check" && 0 <= v449 && v449 < 4);
                                            int v451;
                                            v451 = 4 * v447;
                                            int v452;
                                            v452 = v451 + v449;
                                            float v453;
                                            v453 = v315[v452];
                                            bool v454;
                                            v454 = v316[v452];
                                            int v455;
                                            v455 = v253[v452];
                                            int v458; bool v459;
                                            if (v454){
                                                float v456;
                                                v456 = v453 - v444;
                                                bool v457;
                                                v457 = v456 >= 0.0f;
                                                v458 = v455; v459 = v457;
                                            } else {
                                                v458 = 2147483647; v459 = false;
                                            }
                                            assert("Tensor range check" && 0 <= v447 && v447 < 1);
                                            assert("Tensor range check" && 0 <= v449 && v449 < 4);
                                            v445[v452] = v458;
                                            v446[v452] = v459;
                                            v449 += 1 ;
                                        }
                                        v447 += 1 ;
                                    }
                                    int v460; bool v461;
                                    Tuple10 tmp62 = Tuple10{2147483647, false};
                                    v460 = tmp62.v0; v461 = tmp62.v1;
                                    int v462;
                                    v462 = 0;
                                    while (while_method_4(v462)){
                                        int v464;
                                        v464 = 0;
                                        while (while_method_7(v464)){
                                            assert("Tensor range check" && 0 <= v462 && v462 < 1);
                                            assert("Tensor range check" && 0 <= v464 && v464 < 4);
                                            int v466;
                                            v466 = 4 * v462;
                                            int v467;
                                            v467 = v466 + v464;
                                            int v468;
                                            v468 = v445[v467];
                                            bool v469;
                                            v469 = v446[v467];
                                            int v476; bool v477;
                                            if (v461){
                                                if (v469){
                                                    bool v470;
                                                    v470 = v460 < v468;
                                                    int v471;
                                                    if (v470){
                                                        v471 = v460;
                                                    } else {
                                                        v471 = v468;
                                                    }
                                                    v476 = v471; v477 = true;
                                                } else {
                                                    v476 = v460; v477 = v461;
                                                }
                                            } else {
                                                if (v469){
                                                    v476 = v468; v477 = v469;
                                                } else {
                                                    v476 = v460; v477 = v461;
                                                }
                                            }
                                            v460 = v476;
                                            v461 = v477;
                                            v464 += 1 ;
                                        }
                                        v462 += 1 ;
                                    }
                                    auto v478 = cooperative_groups::coalesced_threads();
                                    int v479;
                                    v479 = threadIdx.x;
                                    int v480;
                                    v480 = v479 / 16;
                                    auto v481 = cooperative_groups::labeled_partition(v478,v480);
                                    Closure5 v482{};
                                    int v483; bool v484;
                                    Tuple10 tmp63 = cooperative_groups::reduce(v481, Tuple10{v460, v461}, v482);
                                    v483 = tmp63.v0; v484 = tmp63.v1;
                                    bool v485;
                                    v485 = v484 == false;
                                    if (v485){
                                        int v486;
                                        v486 = threadIdx.x;
                                        int v487;
                                        v487 = blockIdx.x;
                                        int v488;
                                        v488 = v487 * 256;
                                        int v489;
                                        v489 = v486 + v488;
                                        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v490 = console_lock;
                                        auto v491 = cooperative_groups::coalesced_threads();
                                        v490.acquire();
                                        int v492;
                                        v492 = 0;
                                        printf("{%s = %d; %s = %c","tid", v489, "x'", '[');
                                        int v493;
                                        v493 = 0;
                                        while (while_method_4(v493)){
                                            int v495;
                                            v495 = v492;
                                            bool v496;
                                            v496 = v495 >= 100;
                                            if (v496){
                                                printf("%s"," ...");
                                                break;
                                            } else {
                                            }
                                            bool v497;
                                            v497 = v493 == 0;
                                            bool v498;
                                            v498 = v497 != true;
                                            if (v498){
                                                printf("%s","; ");
                                            } else {
                                            }
                                            printf("%c",'[');
                                            int v499;
                                            v499 = 0;
                                            while (while_method_7(v499)){
                                                int v501;
                                                v501 = v492;
                                                bool v502;
                                                v502 = v501 >= 100;
                                                if (v502){
                                                    printf("%s"," ...");
                                                    break;
                                                } else {
                                                }
                                                bool v503;
                                                v503 = v499 == 0;
                                                bool v504;
                                                v504 = v503 != true;
                                                if (v504){
                                                    printf("%s","; ");
                                                } else {
                                                }
                                                int v505;
                                                v505 = v492 + 1;
                                                v492 = v505;
                                                int v506;
                                                v506 = v493 * 4;
                                                int v507;
                                                v507 = v506 + v499;
                                                int v508;
                                                v508 = v445[v507];
                                                bool v509;
                                                v509 = v446[v507];
                                                const char * v512;
                                                if (v509){
                                                    const char * v510;
                                                    v510 = "true";
                                                    v512 = v510;
                                                } else {
                                                    const char * v511;
                                                    v511 = "false";
                                                    v512 = v511;
                                                }
                                                printf("%d, %s",v508, v512);
                                                v499 += 1 ;
                                            }
                                            printf("%c",']');
                                            v493 += 1 ;
                                        }
                                        printf("%c",']');
                                        printf("}\n");
                                        v490.release();
                                        v491.sync() ;
                                    } else {
                                    }
                                    if (v485){
                                        assert("The local reduce must be true." && v484);
                                    } else {
                                    }
                                    float v548; int v549;
                                    Tuple9 tmp64 = Tuple9{0.0f, 2147483647};
                                    v548 = tmp64.v0; v549 = tmp64.v1;
                                    int v550;
                                    v550 = 0;
                                    while (while_method_4(v550)){
                                        int v552;
                                        v552 = 0;
                                        while (while_method_7(v552)){
                                            assert("Tensor range check" && 0 <= v550 && v550 < 1);
                                            assert("Tensor range check" && 0 <= v552 && v552 < 4);
                                            int v554;
                                            v554 = 4 * v550;
                                            int v555;
                                            v555 = v554 + v552;
                                            float v556;
                                            v556 = v252[v555];
                                            int v557;
                                            v557 = v253[v555];
                                            bool v558;
                                            v558 = v549 == v483;
                                            float v562; int v563;
                                            if (v558){
                                                v562 = v548; v563 = v549;
                                            } else {
                                                bool v559;
                                                v559 = v557 == v483;
                                                if (v559){
                                                    v562 = v556; v563 = v557;
                                                } else {
                                                    v562 = v548; v563 = v549;
                                                }
                                            }
                                            v548 = v562;
                                            v549 = v563;
                                            v552 += 1 ;
                                        }
                                        v550 += 1 ;
                                    }
                                    auto v564 = cooperative_groups::coalesced_threads();
                                    int v565;
                                    v565 = threadIdx.x;
                                    int v566;
                                    v566 = v565 / 16;
                                    auto v567 = cooperative_groups::labeled_partition(v564,v566);
                                    Closure6 v568{v483};
                                    float v569; int v570;
                                    Tuple9 tmp65 = cooperative_groups::reduce(v567, Tuple9{v548, v549}, v568);
                                    v569 = tmp65.v0; v570 = tmp65.v1;
                                    bool v571;
                                    v571 = v570 == 2147483647;
                                    bool v572;
                                    v572 = v571 != true;
                                    bool v573;
                                    v573 = v572 == false;
                                    if (v573){
                                        assert("Expected a valid action id in get_prob." && v572);
                                    } else {
                                    }
                                    int v575;
                                    v575 = 0;
                                    while (while_method_4(v575)){
                                        assert("Tensor range check" && 0 <= v575 && v575 < 1);
                                        assert("Tensor range check" && 0 <= v575 && v575 < 1);
                                        v575 += 1 ;
                                    }
                                    assert("Tensor range check" && 0 <= v244 && v244 < 256);
                                    v219[v244] = v569;
                                    v221[v244] = v483;
                                    v232 += 1 ;
                                }
                                __syncthreads();
                                assert("Tensor range check" && 0 <= v223 && v223 < 256);
                                float v577;
                                v577 = v219[v223];
                                int v578;
                                v578 = v221[v223];
                                __syncthreads();
                                extern __shared__ unsigned char v579[];
                                float * v580;
                                v580 = reinterpret_cast<float *>(&v579[0ull]);
                                int * v582;
                                v582 = reinterpret_cast<int *>(&v579[16ull]);
                                int v584;
                                v584 = threadIdx.x;
                                bool v585;
                                v585 = v584 == 0;
                                if (v585){
                                    v580[0] = v577;
                                    v582[0] = v578;
                                } else {
                                }
                                __syncthreads();
                                float v586;
                                v586 = v580[0];
                                int v587;
                                v587 = v582[0];
                                __syncthreads();
                                double * v588;
                                v588 = reinterpret_cast<double *>(&v1[55050240ull]);
                                double * v590;
                                v590 = reinterpret_cast<double *>(&v1[58195968ull]);
                                int v592;
                                v592 = threadIdx.x;
                                int v593;
                                v593 = blockIdx.x;
                                int v594;
                                v594 = v593 * 256;
                                int v595;
                                v595 = v592 + v594;
                                int v596;
                                v596 = 0;
                                while (while_method_11(v596)){
                                    float * v598;
                                    v598 = reinterpret_cast<float *>(&v1[4718592ull]);
                                    int v600;
                                    v600 = blockIdx.x;
                                    int v601;
                                    v601 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v596 && v596 < 32);
                                    assert("Tensor range check" && 0 <= v600 && v600 < 24);
                                    assert("Tensor range check" && 0 <= v601 && v601 < 256);
                                    assert("Tensor range check" && 0 <= v587 && v587 < 64);
                                    int v602;
                                    v602 = 64 * v601;
                                    int v603;
                                    v603 = v602 + v587;
                                    int v604;
                                    v604 = 16384 * v600;
                                    int v605;
                                    v605 = v604 + v603;
                                    int v606;
                                    v606 = 393216 * v596;
                                    int v607;
                                    v607 = v606 + v605;
                                    float v608;
                                    v608 = v598[v607];
                                    double v609;
                                    v609 = (double)v586;
                                    double v610;
                                    v610 = log(v609);
                                    double v611;
                                    v611 = (double)v608;
                                    double v612;
                                    v612 = log(v611);
                                    assert("Tensor range check" && 0 <= v596 && v596 < 32);
                                    assert("Tensor range check" && 0 <= v595 && v595 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    int v613;
                                    v613 = 2 * v595;
                                    int v614;
                                    v614 = v613 + v71;
                                    int v615;
                                    v615 = 12288 * v596;
                                    int v616;
                                    v616 = v615 + v614;
                                    double v617;
                                    v617 = v588[v616];
                                    double v618;
                                    v618 = v590[v616];
                                    double v619;
                                    v619 = v612 + v617;
                                    double v620;
                                    v620 = v610 + v618;
                                    bool v621;
                                    v621 = isnan(v620);
                                    bool v622;
                                    v622 = v621 == false;
                                    bool v623;
                                    v623 = v622 == false;
                                    if (v623){
                                        assert("The sampling log probability shouldn't be nan." && v622);
                                    } else {
                                    }
                                    bool v625;
                                    v625 = isnan(v619);
                                    bool v626;
                                    v626 = v625 == false;
                                    bool v627;
                                    v627 = v626 == false;
                                    if (v627){
                                        assert("The policy log probability shouldn't be nan." && v626);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v596 && v596 < 32);
                                    assert("Tensor range check" && 0 <= v595 && v595 < 6144);
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    v588[v616] = v619;
                                    v590[v616] = v620;
                                    v596 += 1 ;
                                }
                                bool v629;
                                v629 = 0 == v587;
                                Union12 v638;
                                if (v629){
                                    v638 = Union12{Union12_1{}};
                                } else {
                                    bool v631;
                                    v631 = 1 == v587;
                                    if (v631){
                                        v638 = Union12{Union12_0{}};
                                    } else {
                                        bool v633;
                                        v633 = 2 == v587;
                                        if (v633){
                                            v638 = Union12{Union12_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v638.tag) {
                                    case 0: { // AA_Call
                                        v713 = Union1{Union1_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v639;
                                        v639 = v72[0];
                                        int v641; int v642;
                                        Tuple7 tmp66 = Tuple7{1, v639};
                                        v641 = tmp66.v0; v642 = tmp66.v1;
                                        while (while_method_0(v641)){
                                            bool v644;
                                            v644 = 0 <= v641;
                                            bool v646;
                                            if (v644){
                                                bool v645;
                                                v645 = v641 < 2;
                                                v646 = v645;
                                            } else {
                                                v646 = false;
                                            }
                                            bool v647;
                                            v647 = v646 == false;
                                            if (v647){
                                                assert("Index must be in range." && v646);
                                            } else {
                                            }
                                            int v649;
                                            v649 = v72[v641];
                                            bool v651;
                                            v651 = v642 >= v649;
                                            int v652;
                                            if (v651){
                                                v652 = v642;
                                            } else {
                                                v652 = v649;
                                            }
                                            v642 = v652;
                                            v641 += 1 ;
                                        }
                                        bool v654;
                                        if (v75){
                                            bool v653;
                                            v653 = v71 < 2;
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
                                        v657 = v72[v71];
                                        bool v659;
                                        v659 = v657 == v642;
                                        if (v659){
                                            v713 = Union1{Union1_0{}};
                                        } else {
                                            v713 = Union1{Union1_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v664;
                                        v664 = v73 > 0;
                                        if (v664){
                                            v713 = Union1{Union1_2{}};
                                        } else {
                                            v713 = Union1{Union1_0{}};
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
                                curandStatePhilox4_32_10_t & v671 = v3.v5;
                                curandStatePhilox4_32_10_t & v672 = v671;
                                static_array_list<Union1,3> v673;
                                v673 = static_array_list<Union1,3>{};
                                v673.unsafe_set_length(1);
                                Union1 v675;
                                v675 = Union1{Union1_0{}};
                                v673[0] = v675;
                                int v677;
                                v677 = v72[0];
                                int v679;
                                v679 = v72[1];
                                bool v681;
                                v681 = v677 == v679;
                                bool v682;
                                v682 = v681 != true;
                                if (v682){
                                    Union1 v683;
                                    v683 = Union1{Union1_1{}};
                                    v673.push(v683);
                                } else {
                                }
                                bool v684;
                                v684 = v73 > 0;
                                if (v684){
                                    Union1 v685;
                                    v685 = Union1{Union1_2{}};
                                    v673.push(v685);
                                } else {
                                }
                                int v686;
                                v686 = v673.length;
                                int v687;
                                v687 = v686 - 1;
                                int v688;
                                v688 = 0;
                                while (while_method_1(v687, v688)){
                                    int v690;
                                    v690 = v673.length;
                                    int v691;
                                    v691 = int_range_22(v690, v688, v672);
                                    Union1 v692;
                                    v692 = v673[v688];
                                    Union1 v694;
                                    v694 = v673[v691];
                                    v673[v688] = v694;
                                    v673[v691] = v692;
                                    v688 += 1 ;
                                }
                                Union1 v696;
                                v696 = v673.pop();
                                int v697;
                                v697 = sizeof(Union1);
                                unsigned long long v698;
                                v698 = (unsigned long long)v697;
                                bool v699;
                                v699 = v698 <= 98304ull;
                                bool v700;
                                v700 = v699 == false;
                                if (v700){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v699);
                                } else {
                                }
                                extern __shared__ unsigned char v702[];
                                bool v703;
                                v703 = v698 <= v698;
                                bool v704;
                                v704 = v703 == false;
                                if (v704){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v703);
                                } else {
                                }
                                Union1 * v706;
                                v706 = reinterpret_cast<Union1 *>(&v702[0ull]);
                                int v708;
                                v708 = threadIdx.x;
                                bool v709;
                                v709 = v708 == 0;
                                if (v709){
                                    v706[0] = v696;
                                } else {
                                }
                                __syncthreads();
                                Union1 v710;
                                v710 = v706[0];
                                __syncthreads();
                                v713 = v710;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union7 v714;
                        v714 = Union7{Union7_1{v71, v713}};
                        v14.push(v714);
                        v756 = Union14{Union14_2{v68, v69, v70, v71, v72, v73, v713}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v716 = v18.case3.v0; bool v717 = v18.case3.v1; static_array<Union6,2> v718 = v18.case3.v2; int v719 = v18.case3.v3; static_array<int,2> v720 = v18.case3.v4; int v721 = v18.case3.v5; Union1 v722 = v18.case3.v6;
                        Union7 v723;
                        v723 = Union7{Union7_1{v719, v722}};
                        v14.push(v723);
                        v756 = Union14{Union14_2{v716, v717, v718, v719, v720, v721, v722}};
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
                        v756 = Union14{Union14_3{}};
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
                        v756 = Union14{Union14_3{}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                switch (v756.tag) {
                    case 0: { // T_game_chance_community_card
                        Union5 v758 = v756.case0.v0; bool v759 = v756.case0.v1; static_array<Union6,2> v760 = v756.case0.v2; int v761 = v756.case0.v3; static_array<int,2> v762 = v756.case0.v4; int v763 = v756.case0.v5; Union6 v764 = v756.case0.v6;
                        int v765;
                        v765 = 2;
                        int v766; int v767;
                        Tuple7 tmp67 = Tuple7{0, 0};
                        v766 = tmp67.v0; v767 = tmp67.v1;
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
                            v774 = v762[v766];
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
                        Union5 v782;
                        v782 = Union5{Union5_1{v764}};
                        Union4 v783;
                        v783 = Union4{Union4_2{v782, true, v760, 0, v778, v765}};
                        v916 = Union3{Union3_1{v783}};
                        break;
                    }
                    case 1: { // T_game_chance_init
                        Union6 v785 = v756.case1.v0; Union6 v786 = v756.case1.v1;
                        int v787;
                        v787 = 2;
                        static_array<int,2> v788;
                        v788[0] = 1;
                        v788[1] = 1;
                        static_array<Union6,2> v790;
                        v790[0] = v785;
                        v790[1] = v786;
                        Union5 v792;
                        v792 = Union5{Union5_0{}};
                        Union4 v793;
                        v793 = Union4{Union4_2{v792, true, v790, 0, v788, v787}};
                        v916 = Union3{Union3_1{v793}};
                        break;
                    }
                    case 2: { // T_game_round
                        Union5 v795 = v756.case2.v0; bool v796 = v756.case2.v1; static_array<Union6,2> v797 = v756.case2.v2; int v798 = v756.case2.v3; static_array<int,2> v799 = v756.case2.v4; int v800 = v756.case2.v5; Union1 v801 = v756.case2.v6;
                        Union4 v908;
                        switch (v795.tag) {
                            case 0: { // None
                                switch (v801.tag) {
                                    case 0: { // Call
                                        if (v796){
                                            int v864;
                                            v864 = v798 ^ 1;
                                            v908 = Union4{Union4_2{v795, false, v797, v864, v799, v800}};
                                        } else {
                                            v908 = Union4{Union4_0{v795, v796, v797, v798, v799, v800}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v908 = Union4{Union4_5{v795, v796, v797, v798, v799, v800}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v868;
                                        v868 = v800 > 0;
                                        if (v868){
                                            int v869;
                                            v869 = v798 ^ 1;
                                            int v870;
                                            v870 = -1 + v800;
                                            int v871; int v872;
                                            Tuple7 tmp68 = Tuple7{0, 0};
                                            v871 = tmp68.v0; v872 = tmp68.v1;
                                            while (while_method_0(v871)){
                                                bool v874;
                                                v874 = 0 <= v871;
                                                bool v876;
                                                if (v874){
                                                    bool v875;
                                                    v875 = v871 < 2;
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
                                                v879 = v799[v871];
                                                bool v881;
                                                v881 = v872 >= v879;
                                                int v882;
                                                if (v881){
                                                    v882 = v872;
                                                } else {
                                                    v882 = v879;
                                                }
                                                v872 = v882;
                                                v871 += 1 ;
                                            }
                                            static_array<int,2> v883;
                                            int v885;
                                            v885 = 0;
                                            while (while_method_0(v885)){
                                                v883[v885] = v872;
                                                v885 += 1 ;
                                            }
                                            static_array<int,2> v887;
                                            int v889;
                                            v889 = 0;
                                            while (while_method_0(v889)){
                                                bool v891;
                                                v891 = 0 <= v889;
                                                bool v893;
                                                if (v891){
                                                    bool v892;
                                                    v892 = v889 < 2;
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
                                                v896 = v883[v889];
                                                bool v898;
                                                v898 = v889 == v798;
                                                int v900;
                                                if (v898){
                                                    int v899;
                                                    v899 = v896 + 2;
                                                    v900 = v899;
                                                } else {
                                                    v900 = v896;
                                                }
                                                v887[v889] = v900;
                                                v889 += 1 ;
                                            }
                                            v908 = Union4{Union4_2{v795, false, v797, v869, v887, v870}};
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
                                Union6 v802 = v795.case1.v0;
                                switch (v801.tag) {
                                    case 0: { // Call
                                        if (v796){
                                            int v804;
                                            v804 = v798 ^ 1;
                                            v908 = Union4{Union4_2{v795, false, v797, v804, v799, v800}};
                                        } else {
                                            int v806; int v807;
                                            Tuple7 tmp69 = Tuple7{0, 0};
                                            v806 = tmp69.v0; v807 = tmp69.v1;
                                            while (while_method_0(v806)){
                                                bool v809;
                                                v809 = 0 <= v806;
                                                bool v811;
                                                if (v809){
                                                    bool v810;
                                                    v810 = v806 < 2;
                                                    v811 = v810;
                                                } else {
                                                    v811 = false;
                                                }
                                                bool v812;
                                                v812 = v811 == false;
                                                if (v812){
                                                    assert("Index must be in range." && v811);
                                                } else {
                                                }
                                                int v814;
                                                v814 = v799[v806];
                                                bool v816;
                                                v816 = v807 >= v814;
                                                int v817;
                                                if (v816){
                                                    v817 = v807;
                                                } else {
                                                    v817 = v814;
                                                }
                                                v807 = v817;
                                                v806 += 1 ;
                                            }
                                            static_array<int,2> v818;
                                            int v820;
                                            v820 = 0;
                                            while (while_method_0(v820)){
                                                v818[v820] = v807;
                                                v820 += 1 ;
                                            }
                                            v908 = Union4{Union4_4{v795, v796, v797, v798, v818, v800}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v908 = Union4{Union4_5{v795, v796, v797, v798, v799, v800}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v824;
                                        v824 = v800 > 0;
                                        if (v824){
                                            int v825;
                                            v825 = v798 ^ 1;
                                            int v826;
                                            v826 = -1 + v800;
                                            int v827; int v828;
                                            Tuple7 tmp70 = Tuple7{0, 0};
                                            v827 = tmp70.v0; v828 = tmp70.v1;
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
                                                v835 = v799[v827];
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
                                            static_array<int,2> v843;
                                            int v845;
                                            v845 = 0;
                                            while (while_method_0(v845)){
                                                bool v847;
                                                v847 = 0 <= v845;
                                                bool v849;
                                                if (v847){
                                                    bool v848;
                                                    v848 = v845 < 2;
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
                                                v852 = v839[v845];
                                                bool v854;
                                                v854 = v845 == v798;
                                                int v856;
                                                if (v854){
                                                    int v855;
                                                    v855 = v852 + 4;
                                                    v856 = v855;
                                                } else {
                                                    v856 = v852;
                                                }
                                                v843[v845] = v856;
                                                v845 += 1 ;
                                            }
                                            v908 = Union4{Union4_2{v795, false, v797, v825, v843, v826}};
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
                        v916 = Union3{Union3_1{v908}};
                        break;
                    }
                    case 3: { // T_none
                        v916 = Union3{Union3_0{}};
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
        v16 = v916;
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
        Union3 v924;
        switch (v16.tag) {
            case 0: { // None
                v924 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v18 = v16.case1.v0;
                Union14 v764;
                switch (v18.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v733 = v18.case0.v0; bool v734 = v18.case0.v1; static_array<Union6,2> v735 = v18.case0.v2; int v736 = v18.case0.v3; static_array<int,2> v737 = v18.case0.v4; int v738 = v18.case0.v5;
                        curandStatePhilox4_32_10_t & v739 = v3.v5;
                        curandStatePhilox4_32_10_t & v740 = v739;
                        unsigned int & v741 = v3.v0;
                        Union6 v742; unsigned int v743;
                        Tuple6 tmp73 = draw_card_20(v740, v741);
                        v742 = tmp73.v0; v743 = tmp73.v1;
                        v3.v0 = v743;
                        Union7 v744;
                        v744 = Union7{Union7_0{v742}};
                        v14.push(v744);
                        v764 = Union14{Union14_0{v733, v734, v735, v736, v737, v738, v742}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v746 = v3.v5;
                        curandStatePhilox4_32_10_t & v747 = v746;
                        unsigned int & v748 = v3.v0;
                        Union6 v749; unsigned int v750;
                        Tuple6 tmp74 = draw_card_20(v747, v748);
                        v749 = tmp74.v0; v750 = tmp74.v1;
                        v3.v0 = v750;
                        curandStatePhilox4_32_10_t & v751 = v3.v5;
                        curandStatePhilox4_32_10_t & v752 = v751;
                        unsigned int & v753 = v3.v0;
                        Union6 v754; unsigned int v755;
                        Tuple6 tmp75 = draw_card_20(v752, v753);
                        v754 = tmp75.v0; v755 = tmp75.v1;
                        v3.v0 = v755;
                        Union7 v756;
                        v756 = Union7{Union7_2{0, v749}};
                        v14.push(v756);
                        Union7 v757;
                        v757 = Union7{Union7_2{1, v754}};
                        v14.push(v757);
                        v764 = Union14{Union14_1{v749, v754}};
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
                        Union1 v721;
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
                                while (while_method_11(v155)){
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
                                    float * v171;
                                    v171 = reinterpret_cast<float *>(&v0[1048592ull]);
                                    float * v173;
                                    v173 = reinterpret_cast<float *>(&v0[1048720ull]);
                                    double * v175;
                                    v175 = reinterpret_cast<double *>(&v1[55050240ull]);
                                    double * v177;
                                    v177 = reinterpret_cast<double *>(&v1[58195968ull]);
                                    v155 += 1 ;
                                }
                                __syncthreads();
                                int * v179;
                                v179 = reinterpret_cast<int *>(&v0[1048576ull]);
                                float * v181;
                                v181 = reinterpret_cast<float *>(&v0[1048592ull]);
                                float * v183;
                                v183 = reinterpret_cast<float *>(&v0[1048720ull]);
                                int v185;
                                v185 = 0;
                                int v186;
                                v186 = 32;
                                int v187;
                                v187 = int_range_22(v186, v185, v85);
                                extern __shared__ unsigned char v188[];
                                int * v189;
                                v189 = reinterpret_cast<int *>(&v188[0ull]);
                                int v191;
                                v191 = threadIdx.x;
                                bool v192;
                                v192 = v191 == 0;
                                if (v192){
                                    v189[0] = v187;
                                } else {
                                }
                                __syncthreads();
                                int v193;
                                v193 = v189[0];
                                __syncthreads();
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
                                while (while_method_8(v240)){
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
                                    while (while_method_4(v262)){
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
                                    while (while_method_4(v269)){
                                        int v271;
                                        v271 = 0;
                                        while (while_method_7(v271)){
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
                                    while (while_method_4(v296)){
                                        assert("Tensor range check" && 0 <= v296 && v296 < 1);
                                        int v298;
                                        v298 = 4 * v296;
                                        assert("Tensor range check" && 0 <= v296 && v296 < 1);
                                        float v299;
                                        v299 = 0.0f;
                                        int v300;
                                        v300 = 0;
                                        while (while_method_7(v300)){
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
                                        while (while_method_7(v317)){
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
                                    while (while_method_4(v325)){
                                        int v327;
                                        v327 = 0;
                                        while (while_method_7(v327)){
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
                                    Tuple8 tmp76 = Tuple8{-1.0f / 0.0f, false};
                                    v334 = tmp76.v0; v335 = tmp76.v1;
                                    int v336;
                                    v336 = 0;
                                    while (while_method_4(v336)){
                                        int v338;
                                        v338 = 0;
                                        while (while_method_7(v338)){
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
                                    Tuple8 tmp77 = cooperative_groups::reduce(v355, Tuple8{v334, v335}, v356);
                                    v357 = tmp77.v0; v358 = tmp77.v1;
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
                                        while (while_method_4(v367)){
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
                                            while (while_method_7(v373)){
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
                                    while (while_method_4(v424)){
                                        int v426;
                                        v426 = 0;
                                        while (while_method_7(v426)){
                                            assert("Tensor range check" && 0 <= v424 && v424 < 1);
                                            assert("Tensor range check" && 0 <= v426 && v426 < 4);
                                            int v428;
                                            v428 = 4 * v424;
                                            int v429;
                                            v429 = v428 + v426;
                                            int v430;
                                            v430 = v261[v429];
                                            float v431;
                                            v431 = curand_uniform(&v85);
                                            assert("Tensor range check" && 0 <= v424 && v424 < 1);
                                            assert("Tensor range check" && 0 <= v426 && v426 < 4);
                                            v422[v429] = v431;
                                            v423[v429] = v430;
                                            v426 += 1 ;
                                        }
                                        v424 += 1 ;
                                    }
                                    float v432; int v433;
                                    Tuple9 tmp78 = Tuple9{0.0f, 2147483647};
                                    v432 = tmp78.v0; v433 = tmp78.v1;
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
                                    Tuple9 tmp79 = cooperative_groups::reduce(v448, Tuple9{v432, v433}, v449);
                                    v450 = tmp79.v0; v451 = tmp79.v1;
                                    float v452;
                                    v452 = v357 * v450;
                                    int v453[4];
                                    bool v454[4];
                                    int v455;
                                    v455 = 0;
                                    while (while_method_4(v455)){
                                        int v457;
                                        v457 = 0;
                                        while (while_method_7(v457)){
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
                                    Tuple10 tmp80 = Tuple10{2147483647, false};
                                    v468 = tmp80.v0; v469 = tmp80.v1;
                                    int v470;
                                    v470 = 0;
                                    while (while_method_4(v470)){
                                        int v472;
                                        v472 = 0;
                                        while (while_method_7(v472)){
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
                                    Tuple10 tmp81 = cooperative_groups::reduce(v489, Tuple10{v468, v469}, v490);
                                    v491 = tmp81.v0; v492 = tmp81.v1;
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
                                        while (while_method_4(v501)){
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
                                            while (while_method_7(v507)){
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
                                    Tuple9 tmp82 = Tuple9{0.0f, 2147483647};
                                    v556 = tmp82.v0; v557 = tmp82.v1;
                                    int v558;
                                    v558 = 0;
                                    while (while_method_4(v558)){
                                        int v560;
                                        v560 = 0;
                                        while (while_method_7(v560)){
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
                                    Tuple9 tmp83 = cooperative_groups::reduce(v575, Tuple9{v556, v557}, v576);
                                    v577 = tmp83.v0; v578 = tmp83.v1;
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
                                    while (while_method_4(v583)){
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
                                while (while_method_11(v604)){
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
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
                                    int v621;
                                    v621 = 2 * v603;
                                    int v622;
                                    v622 = v621 + v71;
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
                                    assert("Tensor range check" && 0 <= v71 && v71 < 2);
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
                                        v647 = v72[0];
                                        int v649; int v650;
                                        Tuple7 tmp84 = Tuple7{1, v647};
                                        v649 = tmp84.v0; v650 = tmp84.v1;
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
                                            v657 = v72[v649];
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
                                        if (v75){
                                            bool v661;
                                            v661 = v71 < 2;
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
                                        v665 = v72[v71];
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
                                        v672 = v73 > 0;
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
                                v685 = v72[0];
                                int v687;
                                v687 = v72[1];
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
                                v692 = v73 > 0;
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
                        v722 = Union7{Union7_1{v71, v721}};
                        v14.push(v722);
                        v764 = Union14{Union14_2{v68, v69, v70, v71, v72, v73, v721}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v724 = v18.case3.v0; bool v725 = v18.case3.v1; static_array<Union6,2> v726 = v18.case3.v2; int v727 = v18.case3.v3; static_array<int,2> v728 = v18.case3.v4; int v729 = v18.case3.v5; Union1 v730 = v18.case3.v6;
                        Union7 v731;
                        v731 = Union7{Union7_1{v727, v730}};
                        v14.push(v731);
                        v764 = Union14{Union14_2{v724, v725, v726, v727, v728, v729, v730}};
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
                        v764 = Union14{Union14_3{}};
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
                        Tuple7 tmp85 = Tuple7{0, 0};
                        v774 = tmp85.v0; v775 = tmp85.v1;
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
                                            Tuple7 tmp86 = Tuple7{0, 0};
                                            v879 = tmp86.v0; v880 = tmp86.v1;
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
                                            Tuple7 tmp87 = Tuple7{0, 0};
                                            v814 = tmp87.v0; v815 = tmp87.v1;
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
                                            Tuple7 tmp88 = Tuple7{0, 0};
                                            v835 = tmp88.v0; v836 = tmp88.v1;
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
        v16 = v924;
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
                Union3 v1108;
                switch (v60.tag) {
                    case 0: { // None
                        v1108 = Union3{Union3_0{}};
                        break;
                    }
                    case 1: { // Some
                        Union4 v62 = v60.case1.v0;
                        switch (v62.tag) {
                            case 0: { // ChanceCommunityCard
                                Union5 v1048 = v62.case0.v0; bool v1049 = v62.case0.v1; static_array<Union6,2> v1050 = v62.case0.v2; int v1051 = v62.case0.v3; static_array<int,2> v1052 = v62.case0.v4; int v1053 = v62.case0.v5;
                                curandStatePhilox4_32_10_t & v1054 = v19.v4;
                                curandStatePhilox4_32_10_t & v1055 = v1054;
                                unsigned int & v1056 = v19.v0;
                                Union6 v1057; unsigned int v1058;
                                Tuple6 tmp11 = draw_card_20(v1055, v1056);
                                v1057 = tmp11.v0; v1058 = tmp11.v1;
                                v19.v0 = v1058;
                                Union7 v1059;
                                v1059 = Union7{Union7_0{v1057}};
                                v58.push(v1059);
                                int v1060;
                                v1060 = 2;
                                int v1061; int v1062;
                                Tuple7 tmp12 = Tuple7{0, 0};
                                v1061 = tmp12.v0; v1062 = tmp12.v1;
                                while (while_method_0(v1061)){
                                    bool v1064;
                                    v1064 = 0 <= v1061;
                                    bool v1066;
                                    if (v1064){
                                        bool v1065;
                                        v1065 = v1061 < 2;
                                        v1066 = v1065;
                                    } else {
                                        v1066 = false;
                                    }
                                    bool v1067;
                                    v1067 = v1066 == false;
                                    if (v1067){
                                        assert("Index must be in range." && v1066);
                                    } else {
                                    }
                                    int v1069;
                                    v1069 = v1052[v1061];
                                    bool v1071;
                                    v1071 = v1062 >= v1069;
                                    int v1072;
                                    if (v1071){
                                        v1072 = v1062;
                                    } else {
                                        v1072 = v1069;
                                    }
                                    v1062 = v1072;
                                    v1061 += 1 ;
                                }
                                static_array<int,2> v1073;
                                int v1075;
                                v1075 = 0;
                                while (while_method_0(v1075)){
                                    v1073[v1075] = v1062;
                                    v1075 += 1 ;
                                }
                                Union5 v1077;
                                v1077 = Union5{Union5_1{v1057}};
                                Union4 v1078;
                                v1078 = Union4{Union4_2{v1077, true, v1050, 0, v1073, v1060}};
                                v1108 = Union3{Union3_1{v1078}};
                                break;
                            }
                            case 1: { // ChanceInit
                                curandStatePhilox4_32_10_t & v1080 = v19.v4;
                                curandStatePhilox4_32_10_t & v1081 = v1080;
                                unsigned int & v1082 = v19.v0;
                                Union6 v1083; unsigned int v1084;
                                Tuple6 tmp13 = draw_card_20(v1081, v1082);
                                v1083 = tmp13.v0; v1084 = tmp13.v1;
                                v19.v0 = v1084;
                                curandStatePhilox4_32_10_t & v1085 = v19.v4;
                                curandStatePhilox4_32_10_t & v1086 = v1085;
                                unsigned int & v1087 = v19.v0;
                                Union6 v1088; unsigned int v1089;
                                Tuple6 tmp14 = draw_card_20(v1086, v1087);
                                v1088 = tmp14.v0; v1089 = tmp14.v1;
                                v19.v0 = v1089;
                                Union7 v1090;
                                v1090 = Union7{Union7_2{0, v1083}};
                                v58.push(v1090);
                                Union7 v1091;
                                v1091 = Union7{Union7_2{1, v1088}};
                                v58.push(v1091);
                                int v1092;
                                v1092 = 2;
                                static_array<int,2> v1093;
                                v1093[0] = 1;
                                v1093[1] = 1;
                                static_array<Union6,2> v1095;
                                v1095[0] = v1083;
                                v1095[1] = v1088;
                                Union5 v1097;
                                v1097 = Union5{Union5_0{}};
                                Union4 v1098;
                                v1098 = Union4{Union4_2{v1097, true, v1095, 0, v1093, v1092}};
                                v1108 = Union3{Union3_1{v1098}};
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
                                        v200 = reinterpret_cast<float *>(&v3[4718592ull]);
                                        assert("Tensor range check" && 0 <= v199 && v199 < 32);
                                        int v202;
                                        v202 = 393216 * v199;
                                        float * v203;
                                        v203 = reinterpret_cast<float *>(&v3[0ull]);
                                        float * v205;
                                        v205 = reinterpret_cast<float *>(&v2[0ull]);
                                        float * v207;
                                        v207 = reinterpret_cast<float *>(&v4[0ull]);
                                        assert("Tensor range check" && 0 <= v199 && v199 < 32);
                                        int v209;
                                        v209 = 8192 * v199;
                                        float * v210;
                                        v210 = reinterpret_cast<float *>(&v3[3145728ull]);
                                        block_matmul_23(v210, v205, v209, v203);
                                        block_row_map_24(v200, v202, v210);
                                        int * v212;
                                        v212 = reinterpret_cast<int *>(&v2[1048576ull]);
                                        float * v214;
                                        v214 = reinterpret_cast<float *>(&v2[1048592ull]);
                                        float * v216;
                                        v216 = reinterpret_cast<float *>(&v2[1048720ull]);
                                        double * v218;
                                        v218 = reinterpret_cast<double *>(&v3[55050240ull]);
                                        double * v220;
                                        v220 = reinterpret_cast<double *>(&v3[58195968ull]);
                                        __syncthreads();
                                        float * v222;
                                        v222 = reinterpret_cast<float *>(&v3[4718592ull]);
                                        assert("Tensor range check" && 0 <= v199 && v199 < 32);
                                        int v224;
                                        v224 = blockIdx.x;
                                        assert("Tensor range check" && 0 <= v224 && v224 < 24);
                                        int v225;
                                        v225 = 16384 * v224;
                                        int v226;
                                        v226 = v225 + v202;
                                        int v227;
                                        v227 = threadIdx.x;
                                        assert("Tensor range check" && 0 <= v227 && v227 < 256);
                                        int v228;
                                        v228 = 64 * v227;
                                        int v229;
                                        v229 = v228 + v226;
                                        float * v230;
                                        v230 = v222+v229;
                                        int v232;
                                        v232 = sizeof(float *);
                                        unsigned long long v233;
                                        v233 = (unsigned long long)v232;
                                        unsigned long long v234;
                                        v234 = 256ull * v233;
                                        unsigned long long v235;
                                        v235 = v234 + 16ull;
                                        unsigned long long v236;
                                        v236 = v235 - 1ull;
                                        unsigned long long v237;
                                        v237 = v236 % 16ull;
                                        unsigned long long v238;
                                        v238 = v236 - v237;
                                        unsigned long long v239;
                                        v239 = v238 + 1024ull;
                                        unsigned long long v240;
                                        v240 = v239 + 16ull;
                                        unsigned long long v241;
                                        v241 = v240 - 1ull;
                                        unsigned long long v242;
                                        v242 = v241 % 16ull;
                                        unsigned long long v243;
                                        v243 = v241 - v242;
                                        unsigned long long v244;
                                        v244 = v243 + 1024ull;
                                        bool v245;
                                        v245 = v244 <= 98304ull;
                                        bool v246;
                                        v246 = v245 == false;
                                        if (v246){
                                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v245);
                                        } else {
                                        }
                                        extern __shared__ unsigned char v248[];
                                        bool v249;
                                        v249 = v244 <= v244;
                                        bool v250;
                                        v250 = v249 == false;
                                        if (v250){
                                            assert("The length of the partition has to be less than or equal to the length of the base array." && v249);
                                        } else {
                                        }
                                        float * * v252;
                                        v252 = reinterpret_cast<float * *>(&v248[0ull]);
                                        float * v254;
                                        v254 = reinterpret_cast<float *>(&v248[v238]);
                                        int * v256;
                                        v256 = reinterpret_cast<int *>(&v248[v243]);
                                        int v258;
                                        v258 = threadIdx.x;
                                        assert("Tensor range check" && 0 <= v258 && v258 < 256);
                                        v252[v258] = v230;
                                        __syncthreads();
                                        bool v259;
                                        v259 = 0 <= v258;
                                        bool v260;
                                        v260 = v259 == false;
                                        if (v260){
                                            assert("The index needs to be zero or positive." && v259);
                                        } else {
                                        }
                                        int v262;
                                        v262 = v258 % 16;
                                        int v263;
                                        v263 = v258 / 16;
                                        bool v264;
                                        v264 = v263 < 16;
                                        bool v265;
                                        v265 = v264 == false;
                                        if (v265){
                                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v264);
                                        } else {
                                        }
                                        assert("Tensor range check" && 0 <= v263 && v263 < 16);
                                        int v267;
                                        v267 = 0;
                                        while (while_method_8(v267)){
                                            bool v269;
                                            v269 = 0 <= v263;
                                            bool v270;
                                            v270 = v269 && v264;
                                            bool v271;
                                            v271 = v270 == false;
                                            if (v271){
                                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v270);
                                            } else {
                                            }
                                            bool v273;
                                            v273 = 0 <= v267;
                                            bool v275;
                                            if (v273){
                                                bool v274;
                                                v274 = v267 < 16;
                                                v275 = v274;
                                            } else {
                                                v275 = false;
                                            }
                                            bool v276;
                                            v276 = v275 == false;
                                            if (v276){
                                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v275);
                                            } else {
                                            }
                                            int v278;
                                            v278 = v267 * 16;
                                            int v279;
                                            v279 = v278 + v263;
                                            assert("Tensor range check" && 0 <= v267 && v267 < 16);
                                            int v280;
                                            v280 = 16 * v267;
                                            int v281;
                                            v281 = v280 + v263;
                                            float * v282;
                                            v282 = v252[v281];
                                            int v283;
                                            v283 = blockIdx.x;
                                            int v284;
                                            v284 = v283 * 256;
                                            int v285;
                                            v285 = v284 + v279;
                                            assert("Tensor range check" && 0 <= v262 && v262 < 16);
                                            int v286;
                                            v286 = 4 * v262;
                                            float v287[4];
                                            int v288[4];
                                            int v289;
                                            v289 = 0;
                                            while (while_method_4(v289)){
                                                assert("Tensor range check" && 0 <= v289 && v289 < 1);
                                                int v291;
                                                v291 = 4 * v289;
                                                assert("Tensor range check" && 0 <= v289 && v289 < 1);
                                                int v292;
                                                v292 = 64 * v289;
                                                int v293;
                                                v293 = v292 + v286;
                                                int4* v294;
                                                v294 = reinterpret_cast<int4*>(v282 + v293);
                                                int4* v295;
                                                v295 = reinterpret_cast<int4*>(v287 + v291);
                                                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v294) % 16 == 0 && reinterpret_cast<unsigned long long>(v295) % 16 == 0);
                                                *v295 = *v294;
                                                v289 += 1 ;
                                            }
                                            int v296;
                                            v296 = 0;
                                            while (while_method_4(v296)){
                                                int v298;
                                                v298 = 0;
                                                while (while_method_7(v298)){
                                                    bool v300;
                                                    v300 = 0 <= v298;
                                                    bool v302;
                                                    if (v300){
                                                        bool v301;
                                                        v301 = v298 < 4;
                                                        v302 = v301;
                                                    } else {
                                                        v302 = false;
                                                    }
                                                    bool v303;
                                                    v303 = v302 == false;
                                                    if (v303){
                                                        assert("The indices should be inside the range of the dimension." && v302);
                                                    } else {
                                                    }
                                                    bool v305;
                                                    v305 = 0 <= v262;
                                                    bool v307;
                                                    if (v305){
                                                        bool v306;
                                                        v306 = v262 < 16;
                                                        v307 = v306;
                                                    } else {
                                                        v307 = false;
                                                    }
                                                    bool v308;
                                                    v308 = v307 == false;
                                                    if (v308){
                                                        assert("The indices should be inside the range of the dimension." && v307);
                                                    } else {
                                                    }
                                                    int v310;
                                                    v310 = v262 * 4;
                                                    int v311;
                                                    v311 = v298 + v310;
                                                    bool v312;
                                                    v312 = 0 <= v296;
                                                    bool v314;
                                                    if (v312){
                                                        bool v313;
                                                        v313 = v296 < 1;
                                                        v314 = v313;
                                                    } else {
                                                        v314 = false;
                                                    }
                                                    bool v315;
                                                    v315 = v314 == false;
                                                    if (v315){
                                                        assert("The indices should be inside the range of the dimension." && v314);
                                                    } else {
                                                    }
                                                    int v317;
                                                    v317 = v296 * 64;
                                                    int v318;
                                                    v318 = v311 + v317;
                                                    assert("Tensor range check" && 0 <= v296 && v296 < 1);
                                                    assert("Tensor range check" && 0 <= v298 && v298 < 4);
                                                    int v319;
                                                    v319 = 4 * v296;
                                                    int v320;
                                                    v320 = v319 + v298;
                                                    v288[v320] = v318;
                                                    v298 += 1 ;
                                                }
                                                v296 += 1 ;
                                            }
                                            float v321[4];
                                            float v322;
                                            v322 = 0.0f;
                                            int v323;
                                            v323 = 0;
                                            while (while_method_4(v323)){
                                                assert("Tensor range check" && 0 <= v323 && v323 < 1);
                                                int v325;
                                                v325 = 4 * v323;
                                                assert("Tensor range check" && 0 <= v323 && v323 < 1);
                                                float v326;
                                                v326 = 0.0f;
                                                int v327;
                                                v327 = 0;
                                                while (while_method_7(v327)){
                                                    assert("Tensor range check" && 0 <= v327 && v327 < 4);
                                                    int v329;
                                                    v329 = v327 + v325;
                                                    float v330;
                                                    v330 = v287[v329];
                                                    float v331;
                                                    v331 = v326 + v330;
                                                    v326 = v331;
                                                    v327 += 1 ;
                                                }
                                                auto v332 = cooperative_groups::coalesced_threads();
                                                int v333;
                                                v333 = threadIdx.x;
                                                int v334;
                                                v334 = v333 / 16;
                                                auto v335 = cooperative_groups::labeled_partition(v332,v334);
                                                Closure2 v336{};
                                                float v337;
                                                v337 = cooperative_groups::inclusive_scan(v335, v326, v336);
                                                float v338;
                                                v338 = v335.shfl_up(v337,1);
                                                bool v339;
                                                v339 = v335.thread_rank() == 0;
                                                float v340;
                                                if (v339){
                                                    v340 = 0.0f;
                                                } else {
                                                    v340 = v338;
                                                }
                                                float v341;
                                                v341 = v335.shfl(v337,v335.num_threads()-1);
                                                float v342;
                                                v342 = v322 + v340;
                                                float v343;
                                                v343 = v342;
                                                int v344;
                                                v344 = 0;
                                                while (while_method_7(v344)){
                                                    assert("Tensor range check" && 0 <= v344 && v344 < 4);
                                                    int v346;
                                                    v346 = v344 + v325;
                                                    float v347;
                                                    v347 = v287[v346];
                                                    float v348;
                                                    v348 = v343 + v347;
                                                    assert("Tensor range check" && 0 <= v344 && v344 < 4);
                                                    v321[v346] = v348;
                                                    v343 = v348;
                                                    v344 += 1 ;
                                                }
                                                float v349;
                                                v349 = v322 + v341;
                                                v322 = v349;
                                                v323 += 1 ;
                                            }
                                            float v350[4];
                                            bool v351[4];
                                            int v352;
                                            v352 = 0;
                                            while (while_method_4(v352)){
                                                int v354;
                                                v354 = 0;
                                                while (while_method_7(v354)){
                                                    assert("Tensor range check" && 0 <= v352 && v352 < 1);
                                                    assert("Tensor range check" && 0 <= v354 && v354 < 4);
                                                    int v356;
                                                    v356 = 4 * v352;
                                                    int v357;
                                                    v357 = v356 + v354;
                                                    float v358;
                                                    v358 = v321[v357];
                                                    float v359;
                                                    v359 = v287[v357];
                                                    bool v360;
                                                    v360 = v359 > 0.0f;
                                                    assert("Tensor range check" && 0 <= v352 && v352 < 1);
                                                    assert("Tensor range check" && 0 <= v354 && v354 < 4);
                                                    v350[v357] = v358;
                                                    v351[v357] = v360;
                                                    v354 += 1 ;
                                                }
                                                v352 += 1 ;
                                            }
                                            float v361; bool v362;
                                            Tuple8 tmp15 = Tuple8{-1.0f / 0.0f, false};
                                            v361 = tmp15.v0; v362 = tmp15.v1;
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
                                                    v369 = v350[v368];
                                                    bool v370;
                                                    v370 = v351[v368];
                                                    float v377; bool v378;
                                                    if (v362){
                                                        if (v370){
                                                            bool v371;
                                                            v371 = v361 >= v369;
                                                            float v372;
                                                            if (v371){
                                                                v372 = v361;
                                                            } else {
                                                                v372 = v369;
                                                            }
                                                            v377 = v372; v378 = true;
                                                        } else {
                                                            v377 = v361; v378 = v362;
                                                        }
                                                    } else {
                                                        if (v370){
                                                            v377 = v369; v378 = v370;
                                                        } else {
                                                            v377 = v361; v378 = v362;
                                                        }
                                                    }
                                                    v361 = v377;
                                                    v362 = v378;
                                                    v365 += 1 ;
                                                }
                                                v363 += 1 ;
                                            }
                                            auto v379 = cooperative_groups::coalesced_threads();
                                            int v380;
                                            v380 = threadIdx.x;
                                            int v381;
                                            v381 = v380 / 16;
                                            auto v382 = cooperative_groups::labeled_partition(v379,v381);
                                            Closure3 v383{};
                                            float v384; bool v385;
                                            Tuple8 tmp16 = cooperative_groups::reduce(v382, Tuple8{v361, v362}, v383);
                                            v384 = tmp16.v0; v385 = tmp16.v1;
                                            bool v386;
                                            v386 = v385 == false;
                                            if (v386){
                                                int v387;
                                                v387 = threadIdx.x;
                                                int v388;
                                                v388 = blockIdx.x;
                                                int v389;
                                                v389 = v388 * 256;
                                                int v390;
                                                v390 = v387 + v389;
                                                cuda::counting_semaphore<cuda::thread_scope_system, 1> & v391 = console_lock;
                                                auto v392 = cooperative_groups::coalesced_threads();
                                                v391.acquire();
                                                int v393;
                                                v393 = 0;
                                                printf("{%s = %d; %s = %c","tid", v390, "x'", '[');
                                                int v394;
                                                v394 = 0;
                                                while (while_method_4(v394)){
                                                    int v396;
                                                    v396 = v393;
                                                    bool v397;
                                                    v397 = v396 >= 100;
                                                    if (v397){
                                                        printf("%s"," ...");
                                                        break;
                                                    } else {
                                                    }
                                                    bool v398;
                                                    v398 = v394 == 0;
                                                    bool v399;
                                                    v399 = v398 != true;
                                                    if (v399){
                                                        printf("%s","; ");
                                                    } else {
                                                    }
                                                    printf("%c",'[');
                                                    int v400;
                                                    v400 = 0;
                                                    while (while_method_7(v400)){
                                                        int v402;
                                                        v402 = v393;
                                                        bool v403;
                                                        v403 = v402 >= 100;
                                                        if (v403){
                                                            printf("%s"," ...");
                                                            break;
                                                        } else {
                                                        }
                                                        bool v404;
                                                        v404 = v400 == 0;
                                                        bool v405;
                                                        v405 = v404 != true;
                                                        if (v405){
                                                            printf("%s","; ");
                                                        } else {
                                                        }
                                                        int v406;
                                                        v406 = v393 + 1;
                                                        v393 = v406;
                                                        int v407;
                                                        v407 = v394 * 4;
                                                        int v408;
                                                        v408 = v407 + v400;
                                                        float v409;
                                                        v409 = v350[v408];
                                                        bool v410;
                                                        v410 = v351[v408];
                                                        const char * v413;
                                                        if (v410){
                                                            const char * v411;
                                                            v411 = "true";
                                                            v413 = v411;
                                                        } else {
                                                            const char * v412;
                                                            v412 = "false";
                                                            v413 = v412;
                                                        }
                                                        printf("%f, %s",v409, v413);
                                                        v400 += 1 ;
                                                    }
                                                    printf("%c",']');
                                                    v394 += 1 ;
                                                }
                                                printf("%c",']');
                                                printf("}\n");
                                                v391.release();
                                                v392.sync() ;
                                            } else {
                                            }
                                            if (v386){
                                                assert("The local reduce must be true." && v385);
                                            } else {
                                            }
                                            float v449[4];
                                            int v450[4];
                                            int v451;
                                            v451 = 0;
                                            while (while_method_4(v451)){
                                                int v453;
                                                v453 = 0;
                                                while (while_method_7(v453)){
                                                    assert("Tensor range check" && 0 <= v451 && v451 < 1);
                                                    assert("Tensor range check" && 0 <= v453 && v453 < 4);
                                                    int v455;
                                                    v455 = 4 * v451;
                                                    int v456;
                                                    v456 = v455 + v453;
                                                    int v457;
                                                    v457 = v288[v456];
                                                    float v458;
                                                    v458 = curand_uniform(&v121);
                                                    assert("Tensor range check" && 0 <= v451 && v451 < 1);
                                                    assert("Tensor range check" && 0 <= v453 && v453 < 4);
                                                    v449[v456] = v458;
                                                    v450[v456] = v457;
                                                    v453 += 1 ;
                                                }
                                                v451 += 1 ;
                                            }
                                            float v459; int v460;
                                            Tuple9 tmp17 = Tuple9{0.0f, 2147483647};
                                            v459 = tmp17.v0; v460 = tmp17.v1;
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
                                                    v467 = v449[v466];
                                                    int v468;
                                                    v468 = v450[v466];
                                                    bool v469;
                                                    v469 = v460 < v468;
                                                    float v470; int v471;
                                                    if (v469){
                                                        v470 = v459; v471 = v460;
                                                    } else {
                                                        v470 = v467; v471 = v468;
                                                    }
                                                    v459 = v470;
                                                    v460 = v471;
                                                    v463 += 1 ;
                                                }
                                                v461 += 1 ;
                                            }
                                            auto v472 = cooperative_groups::coalesced_threads();
                                            int v473;
                                            v473 = threadIdx.x;
                                            int v474;
                                            v474 = v473 / 16;
                                            auto v475 = cooperative_groups::labeled_partition(v472,v474);
                                            Closure4 v476{};
                                            float v477; int v478;
                                            Tuple9 tmp18 = cooperative_groups::reduce(v475, Tuple9{v459, v460}, v476);
                                            v477 = tmp18.v0; v478 = tmp18.v1;
                                            float v479;
                                            v479 = v384 * v477;
                                            int v480[4];
                                            bool v481[4];
                                            int v482;
                                            v482 = 0;
                                            while (while_method_4(v482)){
                                                int v484;
                                                v484 = 0;
                                                while (while_method_7(v484)){
                                                    assert("Tensor range check" && 0 <= v482 && v482 < 1);
                                                    assert("Tensor range check" && 0 <= v484 && v484 < 4);
                                                    int v486;
                                                    v486 = 4 * v482;
                                                    int v487;
                                                    v487 = v486 + v484;
                                                    float v488;
                                                    v488 = v350[v487];
                                                    bool v489;
                                                    v489 = v351[v487];
                                                    int v490;
                                                    v490 = v288[v487];
                                                    int v493; bool v494;
                                                    if (v489){
                                                        float v491;
                                                        v491 = v488 - v479;
                                                        bool v492;
                                                        v492 = v491 >= 0.0f;
                                                        v493 = v490; v494 = v492;
                                                    } else {
                                                        v493 = 2147483647; v494 = false;
                                                    }
                                                    assert("Tensor range check" && 0 <= v482 && v482 < 1);
                                                    assert("Tensor range check" && 0 <= v484 && v484 < 4);
                                                    v480[v487] = v493;
                                                    v481[v487] = v494;
                                                    v484 += 1 ;
                                                }
                                                v482 += 1 ;
                                            }
                                            int v495; bool v496;
                                            Tuple10 tmp19 = Tuple10{2147483647, false};
                                            v495 = tmp19.v0; v496 = tmp19.v1;
                                            int v497;
                                            v497 = 0;
                                            while (while_method_4(v497)){
                                                int v499;
                                                v499 = 0;
                                                while (while_method_7(v499)){
                                                    assert("Tensor range check" && 0 <= v497 && v497 < 1);
                                                    assert("Tensor range check" && 0 <= v499 && v499 < 4);
                                                    int v501;
                                                    v501 = 4 * v497;
                                                    int v502;
                                                    v502 = v501 + v499;
                                                    int v503;
                                                    v503 = v480[v502];
                                                    bool v504;
                                                    v504 = v481[v502];
                                                    int v511; bool v512;
                                                    if (v496){
                                                        if (v504){
                                                            bool v505;
                                                            v505 = v495 < v503;
                                                            int v506;
                                                            if (v505){
                                                                v506 = v495;
                                                            } else {
                                                                v506 = v503;
                                                            }
                                                            v511 = v506; v512 = true;
                                                        } else {
                                                            v511 = v495; v512 = v496;
                                                        }
                                                    } else {
                                                        if (v504){
                                                            v511 = v503; v512 = v504;
                                                        } else {
                                                            v511 = v495; v512 = v496;
                                                        }
                                                    }
                                                    v495 = v511;
                                                    v496 = v512;
                                                    v499 += 1 ;
                                                }
                                                v497 += 1 ;
                                            }
                                            auto v513 = cooperative_groups::coalesced_threads();
                                            int v514;
                                            v514 = threadIdx.x;
                                            int v515;
                                            v515 = v514 / 16;
                                            auto v516 = cooperative_groups::labeled_partition(v513,v515);
                                            Closure5 v517{};
                                            int v518; bool v519;
                                            Tuple10 tmp20 = cooperative_groups::reduce(v516, Tuple10{v495, v496}, v517);
                                            v518 = tmp20.v0; v519 = tmp20.v1;
                                            bool v520;
                                            v520 = v519 == false;
                                            if (v520){
                                                int v521;
                                                v521 = threadIdx.x;
                                                int v522;
                                                v522 = blockIdx.x;
                                                int v523;
                                                v523 = v522 * 256;
                                                int v524;
                                                v524 = v521 + v523;
                                                cuda::counting_semaphore<cuda::thread_scope_system, 1> & v525 = console_lock;
                                                auto v526 = cooperative_groups::coalesced_threads();
                                                v525.acquire();
                                                int v527;
                                                v527 = 0;
                                                printf("{%s = %d; %s = %c","tid", v524, "x'", '[');
                                                int v528;
                                                v528 = 0;
                                                while (while_method_4(v528)){
                                                    int v530;
                                                    v530 = v527;
                                                    bool v531;
                                                    v531 = v530 >= 100;
                                                    if (v531){
                                                        printf("%s"," ...");
                                                        break;
                                                    } else {
                                                    }
                                                    bool v532;
                                                    v532 = v528 == 0;
                                                    bool v533;
                                                    v533 = v532 != true;
                                                    if (v533){
                                                        printf("%s","; ");
                                                    } else {
                                                    }
                                                    printf("%c",'[');
                                                    int v534;
                                                    v534 = 0;
                                                    while (while_method_7(v534)){
                                                        int v536;
                                                        v536 = v527;
                                                        bool v537;
                                                        v537 = v536 >= 100;
                                                        if (v537){
                                                            printf("%s"," ...");
                                                            break;
                                                        } else {
                                                        }
                                                        bool v538;
                                                        v538 = v534 == 0;
                                                        bool v539;
                                                        v539 = v538 != true;
                                                        if (v539){
                                                            printf("%s","; ");
                                                        } else {
                                                        }
                                                        int v540;
                                                        v540 = v527 + 1;
                                                        v527 = v540;
                                                        int v541;
                                                        v541 = v528 * 4;
                                                        int v542;
                                                        v542 = v541 + v534;
                                                        int v543;
                                                        v543 = v480[v542];
                                                        bool v544;
                                                        v544 = v481[v542];
                                                        const char * v547;
                                                        if (v544){
                                                            const char * v545;
                                                            v545 = "true";
                                                            v547 = v545;
                                                        } else {
                                                            const char * v546;
                                                            v546 = "false";
                                                            v547 = v546;
                                                        }
                                                        printf("%d, %s",v543, v547);
                                                        v534 += 1 ;
                                                    }
                                                    printf("%c",']');
                                                    v528 += 1 ;
                                                }
                                                printf("%c",']');
                                                printf("}\n");
                                                v525.release();
                                                v526.sync() ;
                                            } else {
                                            }
                                            if (v520){
                                                assert("The local reduce must be true." && v519);
                                            } else {
                                            }
                                            float v583; int v584;
                                            Tuple9 tmp21 = Tuple9{0.0f, 2147483647};
                                            v583 = tmp21.v0; v584 = tmp21.v1;
                                            int v585;
                                            v585 = 0;
                                            while (while_method_4(v585)){
                                                int v587;
                                                v587 = 0;
                                                while (while_method_7(v587)){
                                                    assert("Tensor range check" && 0 <= v585 && v585 < 1);
                                                    assert("Tensor range check" && 0 <= v587 && v587 < 4);
                                                    int v589;
                                                    v589 = 4 * v585;
                                                    int v590;
                                                    v590 = v589 + v587;
                                                    float v591;
                                                    v591 = v287[v590];
                                                    int v592;
                                                    v592 = v288[v590];
                                                    bool v593;
                                                    v593 = v584 == v518;
                                                    float v597; int v598;
                                                    if (v593){
                                                        v597 = v583; v598 = v584;
                                                    } else {
                                                        bool v594;
                                                        v594 = v592 == v518;
                                                        if (v594){
                                                            v597 = v591; v598 = v592;
                                                        } else {
                                                            v597 = v583; v598 = v584;
                                                        }
                                                    }
                                                    v583 = v597;
                                                    v584 = v598;
                                                    v587 += 1 ;
                                                }
                                                v585 += 1 ;
                                            }
                                            auto v599 = cooperative_groups::coalesced_threads();
                                            int v600;
                                            v600 = threadIdx.x;
                                            int v601;
                                            v601 = v600 / 16;
                                            auto v602 = cooperative_groups::labeled_partition(v599,v601);
                                            Closure6 v603{v518};
                                            float v604; int v605;
                                            Tuple9 tmp22 = cooperative_groups::reduce(v602, Tuple9{v583, v584}, v603);
                                            v604 = tmp22.v0; v605 = tmp22.v1;
                                            bool v606;
                                            v606 = v605 == 2147483647;
                                            bool v607;
                                            v607 = v606 != true;
                                            bool v608;
                                            v608 = v607 == false;
                                            if (v608){
                                                assert("Expected a valid action id in get_prob." && v607);
                                            } else {
                                            }
                                            int v610;
                                            v610 = 0;
                                            while (while_method_4(v610)){
                                                assert("Tensor range check" && 0 <= v610 && v610 < 1);
                                                assert("Tensor range check" && 0 <= v610 && v610 < 1);
                                                v610 += 1 ;
                                            }
                                            assert("Tensor range check" && 0 <= v279 && v279 < 256);
                                            v254[v279] = v604;
                                            v256[v279] = v518;
                                            v267 += 1 ;
                                        }
                                        __syncthreads();
                                        assert("Tensor range check" && 0 <= v258 && v258 < 256);
                                        float v612;
                                        v612 = v254[v258];
                                        int v613;
                                        v613 = v256[v258];
                                        __syncthreads();
                                        bool v614;
                                        v614 = 0 == v613;
                                        Union12 v623;
                                        if (v614){
                                            v623 = Union12{Union12_1{}};
                                        } else {
                                            bool v616;
                                            v616 = 1 == v613;
                                            if (v616){
                                                v623 = Union12{Union12_0{}};
                                            } else {
                                                bool v618;
                                                v618 = 2 == v613;
                                                if (v618){
                                                    v623 = Union12{Union12_2{}};
                                                } else {
                                                    printf("%s\n", "Invalid output id in the Leduc model.");
                                                    __trap();
                                                }
                                            }
                                        }
                                        Union1 v655;
                                        switch (v623.tag) {
                                            case 0: { // AA_Call
                                                v655 = Union1{Union1_0{}};
                                                break;
                                            }
                                            case 1: { // AA_Fold
                                                int v624;
                                                v624 = v109[0];
                                                int v626; int v627;
                                                Tuple7 tmp23 = Tuple7{1, v624};
                                                v626 = tmp23.v0; v627 = tmp23.v1;
                                                while (while_method_0(v626)){
                                                    bool v629;
                                                    v629 = 0 <= v626;
                                                    bool v631;
                                                    if (v629){
                                                        bool v630;
                                                        v630 = v626 < 2;
                                                        v631 = v630;
                                                    } else {
                                                        v631 = false;
                                                    }
                                                    bool v632;
                                                    v632 = v631 == false;
                                                    if (v632){
                                                        assert("Index must be in range." && v631);
                                                    } else {
                                                    }
                                                    int v634;
                                                    v634 = v109[v626];
                                                    bool v636;
                                                    v636 = v627 >= v634;
                                                    int v637;
                                                    if (v636){
                                                        v637 = v627;
                                                    } else {
                                                        v637 = v634;
                                                    }
                                                    v627 = v637;
                                                    v626 += 1 ;
                                                }
                                                bool v639;
                                                if (v112){
                                                    bool v638;
                                                    v638 = v108 < 2;
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
                                                v642 = v109[v108];
                                                bool v644;
                                                v644 = v642 == v627;
                                                if (v644){
                                                    v655 = Union1{Union1_0{}};
                                                } else {
                                                    v655 = Union1{Union1_1{}};
                                                }
                                                break;
                                            }
                                            case 2: { // AA_Raise
                                                bool v649;
                                                v649 = v110 > 0;
                                                if (v649){
                                                    v655 = Union1{Union1_2{}};
                                                } else {
                                                    v655 = Union1{Union1_0{}};
                                                }
                                                break;
                                            }
                                            default: {
                                                assert("Invalid tag." && false); __trap();
                                            }
                                        }
                                        int v656;
                                        v656 = sizeof(Union1);
                                        unsigned long long v657;
                                        v657 = (unsigned long long)v656;
                                        bool v658;
                                        v658 = v657 <= 98304ull;
                                        bool v659;
                                        v659 = v658 == false;
                                        if (v659){
                                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v658);
                                        } else {
                                        }
                                        extern __shared__ unsigned char v661[];
                                        bool v662;
                                        v662 = v657 <= v657;
                                        bool v663;
                                        v663 = v662 == false;
                                        if (v663){
                                            assert("The length of the partition has to be less than or equal to the length of the base array." && v662);
                                        } else {
                                        }
                                        Union1 * v665;
                                        v665 = reinterpret_cast<Union1 *>(&v661[0ull]);
                                        int v667;
                                        v667 = threadIdx.x;
                                        bool v668;
                                        v668 = v667 == 0;
                                        if (v668){
                                            v665[0] = v655;
                                        } else {
                                        }
                                        __syncthreads();
                                        Union1 v669;
                                        v669 = v665[0];
                                        __syncthreads();
                                        Union7 v670;
                                        v670 = Union7{Union7_1{v108, v669}};
                                        v58.push(v670);
                                        Union4 v777;
                                        switch (v105.tag) {
                                            case 0: { // None
                                                switch (v669.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v733;
                                                            v733 = v108 ^ 1;
                                                            v777 = Union4{Union4_2{v105, false, v107, v733, v109, v110}};
                                                        } else {
                                                            v777 = Union4{Union4_0{v105, v106, v107, v108, v109, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v777 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        bool v737;
                                                        v737 = v110 > 0;
                                                        if (v737){
                                                            int v738;
                                                            v738 = v108 ^ 1;
                                                            int v739;
                                                            v739 = -1 + v110;
                                                            int v740; int v741;
                                                            Tuple7 tmp24 = Tuple7{0, 0};
                                                            v740 = tmp24.v0; v741 = tmp24.v1;
                                                            while (while_method_0(v740)){
                                                                bool v743;
                                                                v743 = 0 <= v740;
                                                                bool v745;
                                                                if (v743){
                                                                    bool v744;
                                                                    v744 = v740 < 2;
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
                                                                v748 = v109[v740];
                                                                bool v750;
                                                                v750 = v741 >= v748;
                                                                int v751;
                                                                if (v750){
                                                                    v751 = v741;
                                                                } else {
                                                                    v751 = v748;
                                                                }
                                                                v741 = v751;
                                                                v740 += 1 ;
                                                            }
                                                            static_array<int,2> v752;
                                                            int v754;
                                                            v754 = 0;
                                                            while (while_method_0(v754)){
                                                                v752[v754] = v741;
                                                                v754 += 1 ;
                                                            }
                                                            static_array<int,2> v756;
                                                            int v758;
                                                            v758 = 0;
                                                            while (while_method_0(v758)){
                                                                bool v760;
                                                                v760 = 0 <= v758;
                                                                bool v762;
                                                                if (v760){
                                                                    bool v761;
                                                                    v761 = v758 < 2;
                                                                    v762 = v761;
                                                                } else {
                                                                    v762 = false;
                                                                }
                                                                bool v763;
                                                                v763 = v762 == false;
                                                                if (v763){
                                                                    assert("Index must be in range." && v762);
                                                                } else {
                                                                }
                                                                int v765;
                                                                v765 = v752[v758];
                                                                bool v767;
                                                                v767 = v758 == v108;
                                                                int v769;
                                                                if (v767){
                                                                    int v768;
                                                                    v768 = v765 + 2;
                                                                    v769 = v768;
                                                                } else {
                                                                    v769 = v765;
                                                                }
                                                                v756[v758] = v769;
                                                                v758 += 1 ;
                                                            }
                                                            v777 = Union4{Union4_2{v105, false, v107, v738, v756, v739}};
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
                                                Union6 v671 = v105.case1.v0;
                                                switch (v669.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v673;
                                                            v673 = v108 ^ 1;
                                                            v777 = Union4{Union4_2{v105, false, v107, v673, v109, v110}};
                                                        } else {
                                                            int v675; int v676;
                                                            Tuple7 tmp25 = Tuple7{0, 0};
                                                            v675 = tmp25.v0; v676 = tmp25.v1;
                                                            while (while_method_0(v675)){
                                                                bool v678;
                                                                v678 = 0 <= v675;
                                                                bool v680;
                                                                if (v678){
                                                                    bool v679;
                                                                    v679 = v675 < 2;
                                                                    v680 = v679;
                                                                } else {
                                                                    v680 = false;
                                                                }
                                                                bool v681;
                                                                v681 = v680 == false;
                                                                if (v681){
                                                                    assert("Index must be in range." && v680);
                                                                } else {
                                                                }
                                                                int v683;
                                                                v683 = v109[v675];
                                                                bool v685;
                                                                v685 = v676 >= v683;
                                                                int v686;
                                                                if (v685){
                                                                    v686 = v676;
                                                                } else {
                                                                    v686 = v683;
                                                                }
                                                                v676 = v686;
                                                                v675 += 1 ;
                                                            }
                                                            static_array<int,2> v687;
                                                            int v689;
                                                            v689 = 0;
                                                            while (while_method_0(v689)){
                                                                v687[v689] = v676;
                                                                v689 += 1 ;
                                                            }
                                                            v777 = Union4{Union4_4{v105, v106, v107, v108, v687, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v777 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        bool v693;
                                                        v693 = v110 > 0;
                                                        if (v693){
                                                            int v694;
                                                            v694 = v108 ^ 1;
                                                            int v695;
                                                            v695 = -1 + v110;
                                                            int v696; int v697;
                                                            Tuple7 tmp26 = Tuple7{0, 0};
                                                            v696 = tmp26.v0; v697 = tmp26.v1;
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
                                                            static_array<int,2> v712;
                                                            int v714;
                                                            v714 = 0;
                                                            while (while_method_0(v714)){
                                                                bool v716;
                                                                v716 = 0 <= v714;
                                                                bool v718;
                                                                if (v716){
                                                                    bool v717;
                                                                    v717 = v714 < 2;
                                                                    v718 = v717;
                                                                } else {
                                                                    v718 = false;
                                                                }
                                                                bool v719;
                                                                v719 = v718 == false;
                                                                if (v719){
                                                                    assert("Index must be in range." && v718);
                                                                } else {
                                                                }
                                                                int v721;
                                                                v721 = v708[v714];
                                                                bool v723;
                                                                v723 = v714 == v108;
                                                                int v725;
                                                                if (v723){
                                                                    int v724;
                                                                    v724 = v721 + 4;
                                                                    v725 = v724;
                                                                } else {
                                                                    v725 = v721;
                                                                }
                                                                v712[v714] = v725;
                                                                v714 += 1 ;
                                                            }
                                                            v777 = Union4{Union4_2{v105, false, v107, v694, v712, v695}};
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
                                        v1108 = Union3{Union3_1{v777}};
                                        break;
                                    }
                                    case 1: { // Human
                                        Union8 v779;
                                        v779 = Union8{Union8_2{v105, v106, v107, v108, v109, v110}};
                                        v19.v5 = v779;
                                        Union3 v780;
                                        v780 = Union3{Union3_1{v62}};
                                        v19.v1 = v780;
                                        v1108 = Union3{Union3_0{}};
                                        break;
                                    }
                                    case 2: { // Random
                                        curandStatePhilox4_32_10_t & v782 = v19.v4;
                                        curandStatePhilox4_32_10_t & v783 = v782;
                                        static_array_list<Union1,3> v784;
                                        v784 = static_array_list<Union1,3>{};
                                        v784.unsafe_set_length(1);
                                        Union1 v786;
                                        v786 = Union1{Union1_0{}};
                                        v784[0] = v786;
                                        int v788;
                                        v788 = v109[0];
                                        int v790;
                                        v790 = v109[1];
                                        bool v792;
                                        v792 = v788 == v790;
                                        bool v793;
                                        v793 = v792 != true;
                                        if (v793){
                                            Union1 v794;
                                            v794 = Union1{Union1_1{}};
                                            v784.push(v794);
                                        } else {
                                        }
                                        bool v795;
                                        v795 = v110 > 0;
                                        if (v795){
                                            Union1 v796;
                                            v796 = Union1{Union1_2{}};
                                            v784.push(v796);
                                        } else {
                                        }
                                        int v797;
                                        v797 = v784.length;
                                        int v798;
                                        v798 = v797 - 1;
                                        int v799;
                                        v799 = 0;
                                        while (while_method_1(v798, v799)){
                                            int v801;
                                            v801 = v784.length;
                                            int v802;
                                            v802 = int_range_22(v801, v799, v783);
                                            Union1 v803;
                                            v803 = v784[v799];
                                            Union1 v805;
                                            v805 = v784[v802];
                                            v784[v799] = v805;
                                            v784[v802] = v803;
                                            v799 += 1 ;
                                        }
                                        Union1 v807;
                                        v807 = v784.pop();
                                        int v808;
                                        v808 = sizeof(Union1);
                                        unsigned long long v809;
                                        v809 = (unsigned long long)v808;
                                        bool v810;
                                        v810 = v809 <= 98304ull;
                                        bool v811;
                                        v811 = v810 == false;
                                        if (v811){
                                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v810);
                                        } else {
                                        }
                                        extern __shared__ unsigned char v813[];
                                        bool v814;
                                        v814 = v809 <= v809;
                                        bool v815;
                                        v815 = v814 == false;
                                        if (v815){
                                            assert("The length of the partition has to be less than or equal to the length of the base array." && v814);
                                        } else {
                                        }
                                        Union1 * v817;
                                        v817 = reinterpret_cast<Union1 *>(&v813[0ull]);
                                        int v819;
                                        v819 = threadIdx.x;
                                        bool v820;
                                        v820 = v819 == 0;
                                        if (v820){
                                            v817[0] = v807;
                                        } else {
                                        }
                                        __syncthreads();
                                        Union1 v821;
                                        v821 = v817[0];
                                        __syncthreads();
                                        Union7 v822;
                                        v822 = Union7{Union7_1{v108, v821}};
                                        v58.push(v822);
                                        Union4 v927;
                                        switch (v105.tag) {
                                            case 0: { // None
                                                switch (v821.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v884;
                                                            v884 = v108 ^ 1;
                                                            v927 = Union4{Union4_2{v105, false, v107, v884, v109, v110}};
                                                        } else {
                                                            v927 = Union4{Union4_0{v105, v106, v107, v108, v109, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v927 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        if (v795){
                                                            int v888;
                                                            v888 = v108 ^ 1;
                                                            int v889;
                                                            v889 = -1 + v110;
                                                            int v890; int v891;
                                                            Tuple7 tmp27 = Tuple7{0, 0};
                                                            v890 = tmp27.v0; v891 = tmp27.v1;
                                                            while (while_method_0(v890)){
                                                                bool v893;
                                                                v893 = 0 <= v890;
                                                                bool v895;
                                                                if (v893){
                                                                    bool v894;
                                                                    v894 = v890 < 2;
                                                                    v895 = v894;
                                                                } else {
                                                                    v895 = false;
                                                                }
                                                                bool v896;
                                                                v896 = v895 == false;
                                                                if (v896){
                                                                    assert("Index must be in range." && v895);
                                                                } else {
                                                                }
                                                                int v898;
                                                                v898 = v109[v890];
                                                                bool v900;
                                                                v900 = v891 >= v898;
                                                                int v901;
                                                                if (v900){
                                                                    v901 = v891;
                                                                } else {
                                                                    v901 = v898;
                                                                }
                                                                v891 = v901;
                                                                v890 += 1 ;
                                                            }
                                                            static_array<int,2> v902;
                                                            int v904;
                                                            v904 = 0;
                                                            while (while_method_0(v904)){
                                                                v902[v904] = v891;
                                                                v904 += 1 ;
                                                            }
                                                            static_array<int,2> v906;
                                                            int v908;
                                                            v908 = 0;
                                                            while (while_method_0(v908)){
                                                                bool v910;
                                                                v910 = 0 <= v908;
                                                                bool v912;
                                                                if (v910){
                                                                    bool v911;
                                                                    v911 = v908 < 2;
                                                                    v912 = v911;
                                                                } else {
                                                                    v912 = false;
                                                                }
                                                                bool v913;
                                                                v913 = v912 == false;
                                                                if (v913){
                                                                    assert("Index must be in range." && v912);
                                                                } else {
                                                                }
                                                                int v915;
                                                                v915 = v902[v908];
                                                                bool v917;
                                                                v917 = v908 == v108;
                                                                int v919;
                                                                if (v917){
                                                                    int v918;
                                                                    v918 = v915 + 2;
                                                                    v919 = v918;
                                                                } else {
                                                                    v919 = v915;
                                                                }
                                                                v906[v908] = v919;
                                                                v908 += 1 ;
                                                            }
                                                            v927 = Union4{Union4_2{v105, false, v107, v888, v906, v889}};
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
                                                Union6 v823 = v105.case1.v0;
                                                switch (v821.tag) {
                                                    case 0: { // Call
                                                        if (v106){
                                                            int v825;
                                                            v825 = v108 ^ 1;
                                                            v927 = Union4{Union4_2{v105, false, v107, v825, v109, v110}};
                                                        } else {
                                                            int v827; int v828;
                                                            Tuple7 tmp28 = Tuple7{0, 0};
                                                            v827 = tmp28.v0; v828 = tmp28.v1;
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
                                                                v835 = v109[v827];
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
                                                            v927 = Union4{Union4_4{v105, v106, v107, v108, v839, v110}};
                                                        }
                                                        break;
                                                    }
                                                    case 1: { // Fold
                                                        v927 = Union4{Union4_5{v105, v106, v107, v108, v109, v110}};
                                                        break;
                                                    }
                                                    case 2: { // Raise
                                                        if (v795){
                                                            int v845;
                                                            v845 = v108 ^ 1;
                                                            int v846;
                                                            v846 = -1 + v110;
                                                            int v847; int v848;
                                                            Tuple7 tmp29 = Tuple7{0, 0};
                                                            v847 = tmp29.v0; v848 = tmp29.v1;
                                                            while (while_method_0(v847)){
                                                                bool v850;
                                                                v850 = 0 <= v847;
                                                                bool v852;
                                                                if (v850){
                                                                    bool v851;
                                                                    v851 = v847 < 2;
                                                                    v852 = v851;
                                                                } else {
                                                                    v852 = false;
                                                                }
                                                                bool v853;
                                                                v853 = v852 == false;
                                                                if (v853){
                                                                    assert("Index must be in range." && v852);
                                                                } else {
                                                                }
                                                                int v855;
                                                                v855 = v109[v847];
                                                                bool v857;
                                                                v857 = v848 >= v855;
                                                                int v858;
                                                                if (v857){
                                                                    v858 = v848;
                                                                } else {
                                                                    v858 = v855;
                                                                }
                                                                v848 = v858;
                                                                v847 += 1 ;
                                                            }
                                                            static_array<int,2> v859;
                                                            int v861;
                                                            v861 = 0;
                                                            while (while_method_0(v861)){
                                                                v859[v861] = v848;
                                                                v861 += 1 ;
                                                            }
                                                            static_array<int,2> v863;
                                                            int v865;
                                                            v865 = 0;
                                                            while (while_method_0(v865)){
                                                                bool v867;
                                                                v867 = 0 <= v865;
                                                                bool v869;
                                                                if (v867){
                                                                    bool v868;
                                                                    v868 = v865 < 2;
                                                                    v869 = v868;
                                                                } else {
                                                                    v869 = false;
                                                                }
                                                                bool v870;
                                                                v870 = v869 == false;
                                                                if (v870){
                                                                    assert("Index must be in range." && v869);
                                                                } else {
                                                                }
                                                                int v872;
                                                                v872 = v859[v865];
                                                                bool v874;
                                                                v874 = v865 == v108;
                                                                int v876;
                                                                if (v874){
                                                                    int v875;
                                                                    v875 = v872 + 4;
                                                                    v876 = v875;
                                                                } else {
                                                                    v876 = v872;
                                                                }
                                                                v863[v865] = v876;
                                                                v865 += 1 ;
                                                            }
                                                            v927 = Union4{Union4_2{v105, false, v107, v845, v863, v846}};
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
                                        v1108 = Union3{Union3_1{v927}};
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                break;
                            }
                            case 3: { // RoundWithAction
                                Union5 v932 = v62.case3.v0; bool v933 = v62.case3.v1; static_array<Union6,2> v934 = v62.case3.v2; int v935 = v62.case3.v3; static_array<int,2> v936 = v62.case3.v4; int v937 = v62.case3.v5; Union1 v938 = v62.case3.v6;
                                Union7 v939;
                                v939 = Union7{Union7_1{v935, v938}};
                                v58.push(v939);
                                Union4 v1046;
                                switch (v932.tag) {
                                    case 0: { // None
                                        switch (v938.tag) {
                                            case 0: { // Call
                                                if (v933){
                                                    int v1002;
                                                    v1002 = v935 ^ 1;
                                                    v1046 = Union4{Union4_2{v932, false, v934, v1002, v936, v937}};
                                                } else {
                                                    v1046 = Union4{Union4_0{v932, v933, v934, v935, v936, v937}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v1046 = Union4{Union4_5{v932, v933, v934, v935, v936, v937}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                bool v1006;
                                                v1006 = v937 > 0;
                                                if (v1006){
                                                    int v1007;
                                                    v1007 = v935 ^ 1;
                                                    int v1008;
                                                    v1008 = -1 + v937;
                                                    int v1009; int v1010;
                                                    Tuple7 tmp30 = Tuple7{0, 0};
                                                    v1009 = tmp30.v0; v1010 = tmp30.v1;
                                                    while (while_method_0(v1009)){
                                                        bool v1012;
                                                        v1012 = 0 <= v1009;
                                                        bool v1014;
                                                        if (v1012){
                                                            bool v1013;
                                                            v1013 = v1009 < 2;
                                                            v1014 = v1013;
                                                        } else {
                                                            v1014 = false;
                                                        }
                                                        bool v1015;
                                                        v1015 = v1014 == false;
                                                        if (v1015){
                                                            assert("Index must be in range." && v1014);
                                                        } else {
                                                        }
                                                        int v1017;
                                                        v1017 = v936[v1009];
                                                        bool v1019;
                                                        v1019 = v1010 >= v1017;
                                                        int v1020;
                                                        if (v1019){
                                                            v1020 = v1010;
                                                        } else {
                                                            v1020 = v1017;
                                                        }
                                                        v1010 = v1020;
                                                        v1009 += 1 ;
                                                    }
                                                    static_array<int,2> v1021;
                                                    int v1023;
                                                    v1023 = 0;
                                                    while (while_method_0(v1023)){
                                                        v1021[v1023] = v1010;
                                                        v1023 += 1 ;
                                                    }
                                                    static_array<int,2> v1025;
                                                    int v1027;
                                                    v1027 = 0;
                                                    while (while_method_0(v1027)){
                                                        bool v1029;
                                                        v1029 = 0 <= v1027;
                                                        bool v1031;
                                                        if (v1029){
                                                            bool v1030;
                                                            v1030 = v1027 < 2;
                                                            v1031 = v1030;
                                                        } else {
                                                            v1031 = false;
                                                        }
                                                        bool v1032;
                                                        v1032 = v1031 == false;
                                                        if (v1032){
                                                            assert("Index must be in range." && v1031);
                                                        } else {
                                                        }
                                                        int v1034;
                                                        v1034 = v1021[v1027];
                                                        bool v1036;
                                                        v1036 = v1027 == v935;
                                                        int v1038;
                                                        if (v1036){
                                                            int v1037;
                                                            v1037 = v1034 + 2;
                                                            v1038 = v1037;
                                                        } else {
                                                            v1038 = v1034;
                                                        }
                                                        v1025[v1027] = v1038;
                                                        v1027 += 1 ;
                                                    }
                                                    v1046 = Union4{Union4_2{v932, false, v934, v1007, v1025, v1008}};
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
                                        Union6 v940 = v932.case1.v0;
                                        switch (v938.tag) {
                                            case 0: { // Call
                                                if (v933){
                                                    int v942;
                                                    v942 = v935 ^ 1;
                                                    v1046 = Union4{Union4_2{v932, false, v934, v942, v936, v937}};
                                                } else {
                                                    int v944; int v945;
                                                    Tuple7 tmp31 = Tuple7{0, 0};
                                                    v944 = tmp31.v0; v945 = tmp31.v1;
                                                    while (while_method_0(v944)){
                                                        bool v947;
                                                        v947 = 0 <= v944;
                                                        bool v949;
                                                        if (v947){
                                                            bool v948;
                                                            v948 = v944 < 2;
                                                            v949 = v948;
                                                        } else {
                                                            v949 = false;
                                                        }
                                                        bool v950;
                                                        v950 = v949 == false;
                                                        if (v950){
                                                            assert("Index must be in range." && v949);
                                                        } else {
                                                        }
                                                        int v952;
                                                        v952 = v936[v944];
                                                        bool v954;
                                                        v954 = v945 >= v952;
                                                        int v955;
                                                        if (v954){
                                                            v955 = v945;
                                                        } else {
                                                            v955 = v952;
                                                        }
                                                        v945 = v955;
                                                        v944 += 1 ;
                                                    }
                                                    static_array<int,2> v956;
                                                    int v958;
                                                    v958 = 0;
                                                    while (while_method_0(v958)){
                                                        v956[v958] = v945;
                                                        v958 += 1 ;
                                                    }
                                                    v1046 = Union4{Union4_4{v932, v933, v934, v935, v956, v937}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v1046 = Union4{Union4_5{v932, v933, v934, v935, v936, v937}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                bool v962;
                                                v962 = v937 > 0;
                                                if (v962){
                                                    int v963;
                                                    v963 = v935 ^ 1;
                                                    int v964;
                                                    v964 = -1 + v937;
                                                    int v965; int v966;
                                                    Tuple7 tmp32 = Tuple7{0, 0};
                                                    v965 = tmp32.v0; v966 = tmp32.v1;
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
                                                        v973 = v936[v965];
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
                                                    static_array<int,2> v981;
                                                    int v983;
                                                    v983 = 0;
                                                    while (while_method_0(v983)){
                                                        bool v985;
                                                        v985 = 0 <= v983;
                                                        bool v987;
                                                        if (v985){
                                                            bool v986;
                                                            v986 = v983 < 2;
                                                            v987 = v986;
                                                        } else {
                                                            v987 = false;
                                                        }
                                                        bool v988;
                                                        v988 = v987 == false;
                                                        if (v988){
                                                            assert("Index must be in range." && v987);
                                                        } else {
                                                        }
                                                        int v990;
                                                        v990 = v977[v983];
                                                        bool v992;
                                                        v992 = v983 == v935;
                                                        int v994;
                                                        if (v992){
                                                            int v993;
                                                            v993 = v990 + 4;
                                                            v994 = v993;
                                                        } else {
                                                            v994 = v990;
                                                        }
                                                        v981[v983] = v994;
                                                        v983 += 1 ;
                                                    }
                                                    v1046 = Union4{Union4_2{v932, false, v934, v963, v981, v964}};
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
                                v1108 = Union3{Union3_1{v1046}};
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
                                v1108 = Union3{Union3_0{}};
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
                                v1108 = Union3{Union3_0{}};
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
                v60 = v1108;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v1109;
    v1109 = threadIdx.x;
    int v1110;
    v1110 = blockIdx.x;
    int v1111;
    v1111 = v1110 * 256;
    int v1112;
    v1112 = v1109 + v1111;
    bool v1113;
    v1113 = v1112 == 0;
    if (v1113){
        Union8 & v1114 = v19.v5;
        static_array<Union2,2> & v1115 = v19.v3;
        static_array_list<Union7,32> & v1116 = v19.v2;
        Union3 & v1117 = v19.v1;
        unsigned int & v1118 = v19.v0;
        return f_29(v0, v1118, v1117, v1116, v1115, v1114);
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
                double v51[2];
                int v52;
                v52 = 0;
                while (while_method_0(v52)){
                    int v54; double v55;
                    Tuple11 tmp51 = Tuple11{0, 0.0};
                    v54 = tmp51.v0; v55 = tmp51.v1;
                    while (while_method_11(v54)){
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
                while (while_method_11(v71)){
                    int v73;
                    v73 = 0;
                    while (while_method_0(v73)){
                        bool v75;
                        v75 = v65 == 0.0;
                        bool v76;
                        v76 = v75 != true;
                        double v87;
                        if (v76){
                            assert("Tensor range check" && 0 <= v73 && v73 < 2);
                            double v77;
                            v77 = v51[v73];
                            double v78;
                            v78 = v65 / v77;
                            assert("Tensor range check" && 0 <= v71 && v71 < 32);
                            assert("Tensor range check" && 0 <= v73 && v73 < 2);
                            int v79;
                            v79 = v73 + v50;
                            int v80;
                            v80 = 12288 * v71;
                            int v81;
                            v81 = v80 + v79;
                            double v82;
                            v82 = v42[v81];
                            double v83;
                            v83 = v44[v81];
                            double v84;
                            v84 = v82 - v83;
                            double v85;
                            v85 = exp(v84);
                            double v86;
                            v86 = v78 * v85;
                            v87 = v86;
                        } else {
                            v87 = 0.0;
                        }
                        bool v88;
                        v88 = isnan(v87);
                        bool v89;
                        v89 = v88 == false;
                        bool v90;
                        v90 = v89 == false;
                        if (v90){
                            assert("The path probability after integration should not be a nan in integrate_rewards_." && v89);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v71 && v71 < 32);
                        assert("Tensor range check" && 0 <= v73 && v73 < 2);
                        int v92;
                        v92 = 2 * v71;
                        int v93;
                        v93 = v92 + v73;
                        v70[v93] = v87;
                        v73 += 1 ;
                    }
                    v71 += 1 ;
                }
                int v94;
                v94 = 0;
                while (while_method_11(v94)){
                    assert("Tensor range check" && 0 <= v94 && v94 < 32);
                    assert("Tensor range check" && 0 <= v31 && v31 < 2);
                    int v96;
                    v96 = 2 * v94;
                    int v97;
                    v97 = v96 + v31;
                    double v98;
                    v98 = v70[v97];
                    float v99;
                    v99 = (float)v98;
                    float v100;
                    v100 = v40 * v99;
                    assert("Tensor range check" && 0 <= v94 && v94 < 32);
                    assert("Tensor range check" && 0 <= v27 && v27 < 256);
                    int v101;
                    v101 = 256 * v94;
                    int v102;
                    v102 = v101 + v27;
                    float * v103;
                    v103 = v3+v102;
                    float * v105;
                    v105 = v4+v102;
                    float v107;
                    v107 = atomicAdd(v103,v100);
                    float v108;
                    v108 = atomicAdd(v105,v99);
                    v94 += 1 ;
                }
                static_array<float,2> & v109 = v26.v4;
                float * v110;
                v110 = reinterpret_cast<float *>(&v1[4718592ull]);
                int * v112;
                v112 = reinterpret_cast<int *>(&v0[1048576ull]);
                float * v114;
                v114 = reinterpret_cast<float *>(&v0[1048592ull]);
                float * v116;
                v116 = reinterpret_cast<float *>(&v0[1048720ull]);
                double * v118;
                v118 = reinterpret_cast<double *>(&v1[55050240ull]);
                double * v120;
                v120 = reinterpret_cast<double *>(&v1[58195968ull]);
                int v122;
                v122 = threadIdx.x;
                int v123;
                v123 = blockIdx.x;
                int v124;
                v124 = v123 * 256;
                int v125;
                v125 = v122 + v124;
                assert("Tensor range check" && 0 <= v125 && v125 < 6144);
                int v126;
                v126 = 2 * v125;
                double * v127;
                v127 = v118+v126;
                double * v129;
                v129 = v120+v126;
                float v131[2];
                int v132;
                v132 = 0;
                while (while_method_0(v132)){
                    bool v134;
                    v134 = 0 <= v132;
                    bool v136;
                    if (v134){
                        bool v135;
                        v135 = v132 < 2;
                        v136 = v135;
                    } else {
                        v136 = false;
                    }
                    bool v137;
                    v137 = v136 == false;
                    if (v137){
                        assert("Index must be in range." && v136);
                    } else {
                    }
                    float v139;
                    v139 = v109[v132];
                    assert("Tensor range check" && 0 <= v132 && v132 < 2);
                    v131[v132] = v139;
                    v132 += 1 ;
                }
                double v141[2];
                int v142;
                v142 = 0;
                while (while_method_0(v142)){
                    int v144; double v145;
                    Tuple11 tmp52 = Tuple11{0, 0.0};
                    v144 = tmp52.v0; v145 = tmp52.v1;
                    while (while_method_11(v144)){
                        assert("Tensor range check" && 0 <= v144 && v144 < 32);
                        assert("Tensor range check" && 0 <= v142 && v142 < 2);
                        int v147;
                        v147 = 12288 * v144;
                        int v148;
                        v148 = v147 + v142;
                        double v149;
                        v149 = v127[v148];
                        double v150;
                        v150 = v129[v148];
                        double v151;
                        v151 = v149 - v150;
                        double v152;
                        v152 = exp(v151);
                        double v153;
                        v153 = v145 + v152;
                        v145 = v153;
                        v144 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v142 && v142 < 2);
                    v141[v142] = v145;
                    v142 += 1 ;
                }
                double v154;
                v154 = 1.0;
                int v155;
                v155 = 0;
                while (while_method_0(v155)){
                    assert("Tensor range check" && 0 <= v155 && v155 < 2);
                    double v157;
                    v157 = v141[v155];
                    double v158;
                    v158 = v154 * v157;
                    v154 = v158;
                    v155 += 1 ;
                }
                double v159[64];
                int v160;
                v160 = 0;
                while (while_method_11(v160)){
                    int v162;
                    v162 = 0;
                    while (while_method_0(v162)){
                        bool v164;
                        v164 = v154 == 0.0;
                        bool v165;
                        v165 = v164 != true;
                        double v175;
                        if (v165){
                            assert("Tensor range check" && 0 <= v162 && v162 < 2);
                            double v166;
                            v166 = v141[v162];
                            double v167;
                            v167 = v154 / v166;
                            assert("Tensor range check" && 0 <= v160 && v160 < 32);
                            assert("Tensor range check" && 0 <= v162 && v162 < 2);
                            int v168;
                            v168 = 12288 * v160;
                            int v169;
                            v169 = v168 + v162;
                            double v170;
                            v170 = v127[v169];
                            double v171;
                            v171 = v129[v169];
                            double v172;
                            v172 = v170 - v171;
                            double v173;
                            v173 = exp(v172);
                            double v174;
                            v174 = v167 * v173;
                            v175 = v174;
                        } else {
                            v175 = 0.0;
                        }
                        bool v176;
                        v176 = isnan(v175);
                        bool v177;
                        v177 = v176 == false;
                        bool v178;
                        v178 = v177 == false;
                        if (v178){
                            assert("The path probability after integration should not be a nan in integrate_rewards_." && v177);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v160 && v160 < 32);
                        assert("Tensor range check" && 0 <= v162 && v162 < 2);
                        int v180;
                        v180 = 2 * v160;
                        int v181;
                        v181 = v180 + v162;
                        v159[v181] = v175;
                        v162 += 1 ;
                    }
                    v160 += 1 ;
                }
                float v182[32];
                float v183[32];
                int v184;
                v184 = 0;
                while (while_method_11(v184)){
                    int v186; float v187; double v188;
                    Tuple12 tmp53 = Tuple12{0, 0.0f, 0.0};
                    v186 = tmp53.v0; v187 = tmp53.v1; v188 = tmp53.v2;
                    while (while_method_0(v186)){
                        assert("Tensor range check" && 0 <= v184 && v184 < 32);
                        assert("Tensor range check" && 0 <= v186 && v186 < 2);
                        int v190;
                        v190 = 2 * v184;
                        int v191;
                        v191 = v190 + v186;
                        double v192;
                        v192 = v159[v191];
                        assert("Tensor range check" && 0 <= v186 && v186 < 2);
                        float v193;
                        v193 = v131[v186];
                        float v194;
                        v194 = (float)v192;
                        float v195;
                        v195 = v194 * v193;
                        float v196;
                        v196 = v187 + v195;
                        double v197;
                        v197 = v188 + v192;
                        v187 = v196;
                        v188 = v197;
                        v186 += 1 ;
                    }
                    float v198;
                    v198 = (float)v188;
                    assert("Tensor range check" && 0 <= v184 && v184 < 32);
                    v182[v184] = v187;
                    v183[v184] = v198;
                    v184 += 1 ;
                }
                int v199;
                v199 = 0;
                while (while_method_11(v199)){
                    assert("Tensor range check" && 0 <= v199 && v199 < 32);
                    float v201;
                    v201 = v182[v199];
                    float v202;
                    v202 = v183[v199];
                    bool v203;
                    v203 = isnan(v202);
                    bool v204;
                    v204 = v203 == false;
                    bool v205;
                    v205 = v204 == false;
                    if (v205){
                        assert("The path probability after integration should not be a nan in calculate updates." && v204);
                    } else {
                    }
                    float v207;
                    v207 = v201 * v202;
                    assert("Tensor range check" && 0 <= v199 && v199 < 32);
                    float * v208;
                    v208 = v114+v199;
                    float * v210;
                    v210 = v116+v199;
                    float v212;
                    v212 = atomicAdd(v208,v207);
                    float v213;
                    v213 = atomicAdd(v210,v202);
                    v199 += 1 ;
                }
                int v214;
                v214 = threadIdx.x;
                int v215;
                v215 = blockIdx.x;
                int v216;
                v216 = v215 * 256;
                int v217;
                v217 = v214 + v216;
                int v218;
                v218 = 0;
                while (while_method_11(v218)){
                    assert("Tensor range check" && 0 <= v218 && v218 < 32);
                    int v220;
                    v220 = 12288 * v218;
                    assert("Tensor range check" && 0 <= v217 && v217 < 6144);
                    int v221;
                    v221 = 2 * v217;
                    int v222;
                    v222 = v221 + v220;
                    double * v223;
                    v223 = v118+v222;
                    double * v225;
                    v225 = v120+v222;
                    double * v227;
                    v227 = v118+v222;
                    double * v229;
                    v229 = v120+v222;
                    int v231;
                    v231 = sizeof(double *);
                    unsigned long long v232;
                    v232 = (unsigned long long)v231;
                    unsigned long long v233;
                    v233 = 256ull * v232;
                    unsigned long long v234;
                    v234 = v233 + 16ull;
                    unsigned long long v235;
                    v235 = v234 - 1ull;
                    unsigned long long v236;
                    v236 = v235 % 16ull;
                    unsigned long long v237;
                    v237 = v235 - v236;
                    unsigned long long v238;
                    v238 = v237 + v233;
                    unsigned long long v239;
                    v239 = v238 + 16ull;
                    unsigned long long v240;
                    v240 = v239 - 1ull;
                    unsigned long long v241;
                    v241 = v240 % 16ull;
                    unsigned long long v242;
                    v242 = v240 - v241;
                    unsigned long long v243;
                    v243 = v242 + v233;
                    unsigned long long v244;
                    v244 = v243 + 16ull;
                    unsigned long long v245;
                    v245 = v244 - 1ull;
                    unsigned long long v246;
                    v246 = v245 % 16ull;
                    unsigned long long v247;
                    v247 = v245 - v246;
                    unsigned long long v248;
                    v248 = v247 + v233;
                    bool v249;
                    v249 = v248 <= 98304ull;
                    bool v250;
                    v250 = v249 == false;
                    if (v250){
                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v249);
                    } else {
                    }
                    extern __shared__ unsigned char v252[];
                    bool v253;
                    v253 = v248 <= v248;
                    bool v254;
                    v254 = v253 == false;
                    if (v254){
                        assert("The length of the partition has to be less than or equal to the length of the base array." && v253);
                    } else {
                    }
                    double * * v256;
                    v256 = reinterpret_cast<double * *>(&v252[0ull]);
                    double * * v258;
                    v258 = reinterpret_cast<double * *>(&v252[v237]);
                    double * * v260;
                    v260 = reinterpret_cast<double * *>(&v252[v242]);
                    double * * v262;
                    v262 = reinterpret_cast<double * *>(&v252[v247]);
                    int v264;
                    v264 = threadIdx.x;
                    assert("Tensor range check" && 0 <= v264 && v264 < 256);
                    v256[v264] = v223;
                    v258[v264] = v225;
                    v260[v264] = v227;
                    v262[v264] = v229;
                    __syncthreads();
                    bool v265;
                    v265 = 0 <= v264;
                    bool v266;
                    v266 = v265 == false;
                    if (v266){
                        assert("The index needs to be zero or positive." && v265);
                    } else {
                    }
                    int v268;
                    v268 = v264 % 1;
                    bool v269;
                    v269 = v264 < 256;
                    bool v270;
                    v270 = v269 == false;
                    if (v270){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v269);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v264 && v264 < 256);
                    int v272;
                    v272 = 0;
                    while (while_method_4(v272)){
                        bool v274;
                        v274 = v265 && v269;
                        bool v275;
                        v275 = v274 == false;
                        if (v275){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v274);
                        } else {
                        }
                        bool v277;
                        v277 = 0 <= v272;
                        bool v279;
                        if (v277){
                            bool v278;
                            v278 = v272 < 1;
                            v279 = v278;
                        } else {
                            v279 = false;
                        }
                        bool v280;
                        v280 = v279 == false;
                        if (v280){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v279);
                        } else {
                        }
                        int v282;
                        v282 = v272 * 256;
                        int v283;
                        v283 = v282 + v264;
                        assert("Tensor range check" && 0 <= v272 && v272 < 1);
                        int v284;
                        v284 = 256 * v272;
                        int v285;
                        v285 = v284 + v264;
                        double * v286;
                        v286 = v256[v285];
                        double * v287;
                        v287 = v258[v285];
                        double * v288;
                        v288 = v260[v285];
                        double * v289;
                        v289 = v262[v285];
                        int v290;
                        v290 = blockIdx.x;
                        int v291;
                        v291 = v290 * 256;
                        int v292;
                        v292 = v291 + v283;
                        assert("Tensor range check" && 0 <= v268 && v268 < 1);
                        int v293;
                        v293 = 2 * v268;
                        double v294[2];
                        double v295[2];
                        int v296[2];
                        int v297;
                        v297 = 0;
                        while (while_method_4(v297)){
                            assert("Tensor range check" && 0 <= v297 && v297 < 1);
                            int v299;
                            v299 = 2 * v297;
                            assert("Tensor range check" && 0 <= v297 && v297 < 1);
                            int v300;
                            v300 = v299 + v293;
                            int4* v301;
                            v301 = reinterpret_cast<int4*>(v286 + v300);
                            int4* v302;
                            v302 = reinterpret_cast<int4*>(v294 + v299);
                            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v301) % 16 == 0 && reinterpret_cast<unsigned long long>(v302) % 16 == 0);
                            *v302 = *v301;
                            int4* v303;
                            v303 = reinterpret_cast<int4*>(v287 + v300);
                            int4* v304;
                            v304 = reinterpret_cast<int4*>(v295 + v299);
                            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v303) % 16 == 0 && reinterpret_cast<unsigned long long>(v304) % 16 == 0);
                            *v304 = *v303;
                            v297 += 1 ;
                        }
                        int v305;
                        v305 = 0;
                        while (while_method_4(v305)){
                            int v307;
                            v307 = 0;
                            while (while_method_0(v307)){
                                bool v309;
                                v309 = 0 <= v307;
                                bool v311;
                                if (v309){
                                    bool v310;
                                    v310 = v307 < 2;
                                    v311 = v310;
                                } else {
                                    v311 = false;
                                }
                                bool v312;
                                v312 = v311 == false;
                                if (v312){
                                    assert("The indices should be inside the range of the dimension." && v311);
                                } else {
                                }
                                bool v314;
                                v314 = 0 <= v268;
                                bool v316;
                                if (v314){
                                    bool v315;
                                    v315 = v268 < 1;
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
                                v319 = v268 * 2;
                                int v320;
                                v320 = v307 + v319;
                                bool v321;
                                v321 = 0 <= v305;
                                bool v323;
                                if (v321){
                                    bool v322;
                                    v322 = v305 < 1;
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
                                int v326;
                                v326 = v305 * 2;
                                int v327;
                                v327 = v320 + v326;
                                assert("Tensor range check" && 0 <= v305 && v305 < 1);
                                assert("Tensor range check" && 0 <= v307 && v307 < 2);
                                int v328;
                                v328 = 2 * v305;
                                int v329;
                                v329 = v328 + v307;
                                v296[v329] = v327;
                                v307 += 1 ;
                            }
                            v305 += 1 ;
                        }
                        double v330[2];
                        double v331[2];
                        int v332;
                        v332 = 0;
                        while (while_method_4(v332)){
                            int v334;
                            v334 = 0;
                            while (while_method_0(v334)){
                                assert("Tensor range check" && 0 <= v332 && v332 < 1);
                                assert("Tensor range check" && 0 <= v334 && v334 < 2);
                                int v336;
                                v336 = 2 * v332;
                                int v337;
                                v337 = v336 + v334;
                                double v338;
                                v338 = v294[v337];
                                double v339;
                                v339 = v295[v337];
                                assert("Tensor range check" && 0 <= v332 && v332 < 1);
                                assert("Tensor range check" && 0 <= v334 && v334 < 2);
                                v330[v337] = 0.0;
                                v331[v337] = 0.0;
                                v334 += 1 ;
                            }
                            v332 += 1 ;
                        }
                        int v340;
                        v340 = 0;
                        while (while_method_4(v340)){
                            assert("Tensor range check" && 0 <= v340 && v340 < 1);
                            int v342;
                            v342 = 2 * v340;
                            int v343;
                            v343 = v342 + v293;
                            assert("Tensor range check" && 0 <= v340 && v340 < 1);
                            int4* v344;
                            v344 = reinterpret_cast<int4*>(v330 + v342);
                            int4* v345;
                            v345 = reinterpret_cast<int4*>(v288 + v343);
                            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v344) % 16 == 0 && reinterpret_cast<unsigned long long>(v345) % 16 == 0);
                            *v345 = *v344;
                            int4* v346;
                            v346 = reinterpret_cast<int4*>(v331 + v342);
                            int4* v347;
                            v347 = reinterpret_cast<int4*>(v289 + v343);
                            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v346) % 16 == 0 && reinterpret_cast<unsigned long long>(v347) % 16 == 0);
                            *v347 = *v346;
                            v340 += 1 ;
                        }
                        assert("Tensor range check" && 0 <= v283 && v283 < 256);
                        v272 += 1 ;
                    }
                    __syncthreads();
                    assert("Tensor range check" && 0 <= v264 && v264 < 256);
                    __syncthreads();
                    v218 += 1 ;
                }
                v31 += 1 ;
            }
            v29 += 1 ;
        }
        cooperative_groups::grid_group & v348 = v26.v1;
        cooperative_groups::grid_group & v349 = v348;
        curandStatePhilox4_32_10_t & v350 = v26.v5;
        curandStatePhilox4_32_10_t & v351 = v350;
        float * v352;
        v352 = reinterpret_cast<float *>(&v0[0ull]);
        float * v354;
        v354 = reinterpret_cast<float *>(&v2[0ull]);
        float * v356;
        v356 = reinterpret_cast<float *>(&v1[4718592ull]);
        int * v358;
        v358 = reinterpret_cast<int *>(&v0[1048576ull]);
        float * v360;
        v360 = reinterpret_cast<float *>(&v0[1048592ull]);
        float * v362;
        v362 = reinterpret_cast<float *>(&v0[1048720ull]);
        double * v364;
        v364 = reinterpret_cast<double *>(&v1[55050240ull]);
        double * v366;
        v366 = reinterpret_cast<double *>(&v1[58195968ull]);
        v349.sync() ;
        int v368;
        v368 = threadIdx.x;
        int v369;
        v369 = blockIdx.x;
        int v370;
        v370 = v369 * 256;
        int v371;
        v371 = v368 + v370;
        bool v372;
        v372 = v371 == 0;
        if (v372){
            int v373;
            v373 = 0;
            int v374;
            v374 = 32;
            int v375;
            v375 = int_range_22(v374, v373, v351);
            v358[0] = v375;
        } else {
        }
        __syncwarp();
        extern __shared__ unsigned char v376[];
        float * v377;
        v377 = reinterpret_cast<float *>(&v376[0ull]);
        int v379;
        v379 = blockIdx.x;
        int v380;
        v380 = v379;
        while (while_method_8(v380)){
            bool v382;
            v382 = 0 <= v380;
            bool v383;
            v383 = v382 == false;
            if (v383){
                assert("The index needs to be zero or positive." && v382);
            } else {
            }
            int v385;
            v385 = v380 % 16;
            int v386;
            v386 = v380 / 16;
            bool v387;
            v387 = v386 < 1;
            bool v388;
            v388 = v387 == false;
            if (v388){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v387);
            } else {
            }
            assert("Tensor range check" && 0 <= v386 && v386 < 1);
            assert("Tensor range check" && 0 <= v385 && v385 < 16);
            int v390;
            v390 = 512 * v385;
            int v391;
            v391 = 262144 * v386;
            int v392;
            v392 = v391 + v390;
            int v393;
            v393 = 16384 * v385;
            int v394;
            v394 = 32 * v386;
            int v395;
            v395 = v394 + v393;
            int v396;
            v396 = threadIdx.x;
            int v397;
            v397 = v396;
            while (while_method_12(v397)){
                bool v399;
                v399 = 0 <= v397;
                bool v400;
                v400 = v399 == false;
                if (v400){
                    assert("The index needs to be zero or positive." && v399);
                } else {
                }
                int v402;
                v402 = v397 % 512;
                int v403;
                v403 = v397 / 512;
                bool v404;
                v404 = v403 < 32;
                bool v405;
                v405 = v404 == false;
                if (v405){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v404);
                } else {
                }
                assert("Tensor range check" && 0 <= v403 && v403 < 32);
                assert("Tensor range check" && 0 <= v402 && v402 < 512);
                int v407;
                v407 = v402 + v392;
                int v408;
                v408 = 8192 * v403;
                int v409;
                v409 = v408 + v407;
                float v410;
                v410 = v352[v409];
                assert("Tensor range check" && 0 <= v403 && v403 < 32);
                assert("Tensor range check" && 0 <= v402 && v402 < 512);
                int v411;
                v411 = 513 * v403;
                int v412;
                v412 = v411 + v402;
                v377[v412] = v410;
                v397 += 256 ;
            }
            __syncthreads();
            int v413;
            v413 = threadIdx.x;
            int v414;
            v414 = v413;
            while (while_method_12(v414)){
                bool v416;
                v416 = 0 <= v414;
                bool v417;
                v417 = v416 == false;
                if (v417){
                    assert("The index needs to be zero or positive." && v416);
                } else {
                }
                int v419;
                v419 = v414 % 32;
                int v420;
                v420 = v414 / 32;
                bool v421;
                v421 = v420 < 512;
                bool v422;
                v422 = v421 == false;
                if (v422){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v421);
                } else {
                }
                assert("Tensor range check" && 0 <= v420 && v420 < 512);
                assert("Tensor range check" && 0 <= v419 && v419 < 32);
                int v424;
                v424 = 513 * v419;
                int v425;
                v425 = v420 + v424;
                float v426;
                v426 = v377[v425];
                assert("Tensor range check" && 0 <= v420 && v420 < 512);
                assert("Tensor range check" && 0 <= v419 && v419 < 32);
                int v427;
                v427 = v419 + v395;
                int v428;
                v428 = 32 * v420;
                int v429;
                v429 = v428 + v427;
                v354[v429] = v426;
                v414 += 256 ;
            }
            __syncthreads();
            v380 += 24 ;
        }
        v349.sync() ;
        int v430;
        v430 = threadIdx.x;
        bool v431;
        v431 = 0 <= v430;
        bool v432;
        v432 = v431 == false;
        if (v432){
            assert("The index needs to be zero or positive." && v431);
        } else {
        }
        int v434;
        v434 = v430 % 8;
        int v435;
        v435 = v430 / 8;
        int v436;
        v436 = v435 % 32;
        int v437;
        v437 = v435 / 32;
        bool v438;
        v438 = v437 < 1;
        bool v439;
        v439 = v438 == false;
        if (v439){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v438);
        } else {
        }
        assert("Tensor range check" && 0 <= v437 && v437 < 1);
        assert("Tensor range check" && 0 <= v436 && v436 < 32);
        assert("Tensor range check" && 0 <= v434 && v434 < 8);
        int v441;
        v441 = 4 * v434;
        int v442;
        v442 = 32 * v436;
        int v443;
        v443 = v442 + v441;
        int v444;
        v444 = 4096 * v437;
        int v445;
        v445 = v444 + v443;
        assert("Tensor range check" && 0 <= v437 && v437 < 1);
        assert("Tensor range check" && 0 <= v436 && v436 < 32);
        assert("Tensor range check" && 0 <= v434 && v434 < 8);
        int v446;
        v446 = blockIdx.x;
        int v447;
        v447 = v446;
        while (while_method_9(v447)){
            bool v449;
            v449 = 0 <= v447;
            bool v450;
            v450 = v449 == false;
            if (v450){
                assert("The index needs to be zero or positive." && v449);
            } else {
            }
            int v452;
            v452 = v447 % 4;
            int v453;
            v453 = v447 / 4;
            bool v454;
            v454 = v453 < 64;
            bool v455;
            v455 = v454 == false;
            if (v455){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v454);
            } else {
            }
            assert("Tensor range check" && 0 <= v453 && v453 < 64);
            assert("Tensor range check" && 0 <= v452 && v452 < 4);
            int v457;
            v457 = 1024 * v452;
            int v458;
            v458 = v457 + v445;
            int v459;
            v459 = 4096 * v453;
            int v460;
            v460 = v459 + v458;
            float v461[4];
            int v462[4];
            int v463;
            v463 = 0;
            while (while_method_4(v463)){
                assert("Tensor range check" && 0 <= v463 && v463 < 1);
                int v465;
                v465 = 4 * v463;
                assert("Tensor range check" && 0 <= v463 && v463 < 1);
                int v466;
                v466 = 32 * v463;
                int v467;
                v467 = v466 + v460;
                int4* v468;
                v468 = reinterpret_cast<int4*>(v354 + v467);
                int4* v469;
                v469 = reinterpret_cast<int4*>(v461 + v465);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v468) % 16 == 0 && reinterpret_cast<unsigned long long>(v469) % 16 == 0);
                *v469 = *v468;
                v463 += 1 ;
            }
            int v470;
            v470 = 0;
            while (while_method_4(v470)){
                int v472;
                v472 = 0;
                while (while_method_7(v472)){
                    bool v474;
                    v474 = 0 <= v472;
                    bool v476;
                    if (v474){
                        bool v475;
                        v475 = v472 < 4;
                        v476 = v475;
                    } else {
                        v476 = false;
                    }
                    bool v477;
                    v477 = v476 == false;
                    if (v477){
                        assert("The indices should be inside the range of the dimension." && v476);
                    } else {
                    }
                    bool v479;
                    v479 = 0 <= v434;
                    bool v481;
                    if (v479){
                        bool v480;
                        v480 = v434 < 8;
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
                    int v484;
                    v484 = v434 * 4;
                    int v485;
                    v485 = v472 + v484;
                    bool v486;
                    v486 = 0 <= v470;
                    bool v488;
                    if (v486){
                        bool v487;
                        v487 = v470 < 1;
                        v488 = v487;
                    } else {
                        v488 = false;
                    }
                    bool v489;
                    v489 = v488 == false;
                    if (v489){
                        assert("The indices should be inside the range of the dimension." && v488);
                    } else {
                    }
                    int v491;
                    v491 = v470 * 32;
                    int v492;
                    v492 = v485 + v491;
                    assert("Tensor range check" && 0 <= v470 && v470 < 1);
                    assert("Tensor range check" && 0 <= v472 && v472 < 4);
                    int v493;
                    v493 = 4 * v470;
                    int v494;
                    v494 = v493 + v472;
                    v462[v494] = v492;
                    v472 += 1 ;
                }
                v470 += 1 ;
            }
            bool v495;
            v495 = 0 <= v437;
            bool v496;
            v496 = v495 && v438;
            bool v497;
            v497 = v496 == false;
            if (v497){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v496);
            } else {
            }
            bool v499;
            v499 = 0 <= v436;
            bool v501;
            if (v499){
                bool v500;
                v500 = v436 < 32;
                v501 = v500;
            } else {
                v501 = false;
            }
            bool v502;
            v502 = v501 == false;
            if (v502){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v501);
            } else {
            }
            bool v504;
            v504 = 0 <= v453;
            bool v505;
            v505 = v504 && v454;
            bool v506;
            v506 = v505 == false;
            if (v506){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v505);
            } else {
            }
            bool v508;
            v508 = 0 <= v452;
            bool v510;
            if (v508){
                bool v509;
                v509 = v452 < 4;
                v510 = v509;
            } else {
                v510 = false;
            }
            bool v511;
            v511 = v510 == false;
            if (v511){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v510);
            } else {
            }
            int v513;
            v513 = v452 * 32;
            int v514;
            v514 = v453 + v437;
            int v515;
            v515 = v513 + v436;
            float v516[4];
            int v517;
            v517 = 0;
            while (while_method_4(v517)){
                int v519;
                v519 = 0;
                while (while_method_7(v519)){
                    assert("Tensor range check" && 0 <= v517 && v517 < 1);
                    assert("Tensor range check" && 0 <= v519 && v519 < 4);
                    int v521;
                    v521 = 4 * v517;
                    int v522;
                    v522 = v521 + v519;
                    int v523;
                    v523 = v462[v522];
                    assert("Tensor range check" && 0 <= v523 && v523 < 32);
                    float v524;
                    v524 = v360[v523];
                    float v525;
                    v525 = v362[v523];
                    bool v526;
                    v526 = isnan(v524);
                    bool v527;
                    v527 = v526 == false;
                    bool v528;
                    v528 = v527 == false;
                    if (v528){
                        assert("The reward numerator must not be a nan." && v527);
                    } else {
                    }
                    bool v530;
                    v530 = isnan(v525);
                    bool v531;
                    v531 = v530 == false;
                    bool v532;
                    v532 = v531 == false;
                    if (v532){
                        assert("The reward count must not be a nan." && v531);
                    } else {
                    }
                    bool v534;
                    v534 = v525 == 0.0f;
                    bool v535;
                    v535 = v534 != true;
                    float v537;
                    if (v535){
                        float v536;
                        v536 = v524 / v525;
                        v537 = v536;
                    } else {
                        v537 = 0.0f;
                    }
                    bool v538;
                    v538 = isnan(v537);
                    bool v539;
                    v539 = v538 == false;
                    bool v540;
                    v540 = v539 == false;
                    if (v540){
                        assert("The reward must not be a nan." && v539);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v517 && v517 < 1);
                    assert("Tensor range check" && 0 <= v519 && v519 < 4);
                    v516[v522] = v537;
                    v519 += 1 ;
                }
                v517 += 1 ;
            }
            float v542;
            v542 = 0.0f;
            int v543;
            v543 = 0;
            while (while_method_4(v543)){
                int v545;
                v545 = 0;
                while (while_method_7(v545)){
                    assert("Tensor range check" && 0 <= v543 && v543 < 1);
                    assert("Tensor range check" && 0 <= v545 && v545 < 4);
                    int v547;
                    v547 = 4 * v543;
                    int v548;
                    v548 = v547 + v545;
                    float v549;
                    v549 = v516[v548];
                    float v550;
                    v550 = v542 + v549;
                    v542 = v550;
                    v545 += 1 ;
                }
                v543 += 1 ;
            }
            auto v551 = cooperative_groups::coalesced_threads();
            int v552;
            v552 = threadIdx.x;
            int v553;
            v553 = v552 / 8;
            auto v554 = cooperative_groups::labeled_partition(v551,v553);
            Closure0 v555{};
            float v556;
            v556 = cooperative_groups::reduce(v554, v542, v555);
            float v557;
            v557 = v556 / 32.0f;
            bool v558[4];
            int v559;
            v559 = 0;
            while (while_method_4(v559)){
                int v561;
                v561 = 0;
                while (while_method_7(v561)){
                    assert("Tensor range check" && 0 <= v559 && v559 < 1);
                    assert("Tensor range check" && 0 <= v561 && v561 < 4);
                    int v563;
                    v563 = 4 * v559;
                    int v564;
                    v564 = v563 + v561;
                    float v565;
                    v565 = v516[v564];
                    bool v566;
                    v566 = v565 >= v557;
                    assert("Tensor range check" && 0 <= v559 && v559 < 1);
                    assert("Tensor range check" && 0 <= v561 && v561 < 4);
                    v558[v564] = v566;
                    v561 += 1 ;
                }
                v559 += 1 ;
            }
            int v567[4];
            int v568;
            v568 = 0;
            while (while_method_4(v568)){
                int v570;
                v570 = 0;
                while (while_method_7(v570)){
                    assert("Tensor range check" && 0 <= v568 && v568 < 1);
                    assert("Tensor range check" && 0 <= v570 && v570 < 4);
                    int v572;
                    v572 = 4 * v568;
                    int v573;
                    v573 = v572 + v570;
                    bool v574;
                    v574 = v558[v573];
                    int v575;
                    if (v574){
                        v575 = 1;
                    } else {
                        v575 = 0;
                    }
                    assert("Tensor range check" && 0 <= v568 && v568 < 1);
                    assert("Tensor range check" && 0 <= v570 && v570 < 4);
                    v567[v573] = v575;
                    v570 += 1 ;
                }
                v568 += 1 ;
            }
            int v576;
            v576 = 0;
            int v577;
            v577 = 0;
            while (while_method_4(v577)){
                int v579;
                v579 = 0;
                while (while_method_7(v579)){
                    assert("Tensor range check" && 0 <= v577 && v577 < 1);
                    assert("Tensor range check" && 0 <= v579 && v579 < 4);
                    int v581;
                    v581 = 4 * v577;
                    int v582;
                    v582 = v581 + v579;
                    int v583;
                    v583 = v567[v582];
                    int v584;
                    v584 = v576 + v583;
                    v576 = v584;
                    v579 += 1 ;
                }
                v577 += 1 ;
            }
            auto v585 = cooperative_groups::coalesced_threads();
            int v586;
            v586 = threadIdx.x;
            int v587;
            v587 = v586 / 8;
            auto v588 = cooperative_groups::labeled_partition(v585,v587);
            Closure1 v589{};
            int v590;
            v590 = cooperative_groups::reduce(v588, v576, v589);
            float v591;
            v591 = (float)v590;
            float v592[4];
            int v593;
            v593 = 0;
            while (while_method_4(v593)){
                int v595;
                v595 = 0;
                while (while_method_7(v595)){
                    assert("Tensor range check" && 0 <= v593 && v593 < 1);
                    assert("Tensor range check" && 0 <= v595 && v595 < 4);
                    int v597;
                    v597 = 4 * v593;
                    int v598;
                    v598 = v597 + v595;
                    float v599;
                    v599 = v461[v598];
                    bool v600;
                    v600 = v558[v598];
                    float v601;
                    if (v600){
                        v601 = v599;
                    } else {
                        v601 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v593 && v593 < 1);
                    assert("Tensor range check" && 0 <= v595 && v595 < 4);
                    v592[v598] = v601;
                    v595 += 1 ;
                }
                v593 += 1 ;
            }
            float v602;
            v602 = 0.0f;
            int v603;
            v603 = 0;
            while (while_method_4(v603)){
                int v605;
                v605 = 0;
                while (while_method_7(v605)){
                    assert("Tensor range check" && 0 <= v603 && v603 < 1);
                    assert("Tensor range check" && 0 <= v605 && v605 < 4);
                    int v607;
                    v607 = 4 * v603;
                    int v608;
                    v608 = v607 + v605;
                    float v609;
                    v609 = v592[v608];
                    float v610;
                    v610 = v602 + v609;
                    v602 = v610;
                    v605 += 1 ;
                }
                v603 += 1 ;
            }
            auto v611 = cooperative_groups::coalesced_threads();
            int v612;
            v612 = threadIdx.x;
            int v613;
            v613 = v612 / 8;
            auto v614 = cooperative_groups::labeled_partition(v611,v613);
            float v615;
            v615 = cooperative_groups::reduce(v614, v602, v555);
            float v616;
            v616 = v615 / v591;
            float v617[4];
            int v618;
            v618 = 0;
            while (while_method_4(v618)){
                int v620;
                v620 = 0;
                while (while_method_7(v620)){
                    assert("Tensor range check" && 0 <= v618 && v618 < 1);
                    assert("Tensor range check" && 0 <= v620 && v620 < 4);
                    int v622;
                    v622 = 4 * v618;
                    int v623;
                    v623 = v622 + v620;
                    float v624;
                    v624 = v461[v623];
                    float v625;
                    v625 = v624 - v616;
                    float v626;
                    v626 = v625 * v625;
                    assert("Tensor range check" && 0 <= v618 && v618 < 1);
                    assert("Tensor range check" && 0 <= v620 && v620 < 4);
                    v617[v623] = v626;
                    v620 += 1 ;
                }
                v618 += 1 ;
            }
            float v627[4];
            int v628;
            v628 = 0;
            while (while_method_4(v628)){
                int v630;
                v630 = 0;
                while (while_method_7(v630)){
                    assert("Tensor range check" && 0 <= v628 && v628 < 1);
                    assert("Tensor range check" && 0 <= v630 && v630 < 4);
                    int v632;
                    v632 = 4 * v628;
                    int v633;
                    v633 = v632 + v630;
                    float v634;
                    v634 = v617[v633];
                    bool v635;
                    v635 = v558[v633];
                    float v636;
                    if (v635){
                        v636 = v634;
                    } else {
                        v636 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v628 && v628 < 1);
                    assert("Tensor range check" && 0 <= v630 && v630 < 4);
                    v627[v633] = v636;
                    v630 += 1 ;
                }
                v628 += 1 ;
            }
            float v637;
            v637 = 0.0f;
            int v638;
            v638 = 0;
            while (while_method_4(v638)){
                int v640;
                v640 = 0;
                while (while_method_7(v640)){
                    assert("Tensor range check" && 0 <= v638 && v638 < 1);
                    assert("Tensor range check" && 0 <= v640 && v640 < 4);
                    int v642;
                    v642 = 4 * v638;
                    int v643;
                    v643 = v642 + v640;
                    float v644;
                    v644 = v627[v643];
                    float v645;
                    v645 = v637 + v644;
                    v637 = v645;
                    v640 += 1 ;
                }
                v638 += 1 ;
            }
            auto v646 = cooperative_groups::coalesced_threads();
            int v647;
            v647 = threadIdx.x;
            int v648;
            v648 = v647 / 8;
            auto v649 = cooperative_groups::labeled_partition(v646,v648);
            float v650;
            v650 = cooperative_groups::reduce(v649, v637, v555);
            float v651;
            v651 = v650 / v591;
            float v652;
            v652 = sqrt(v651);
            bool v653;
            v653 = v591 > 1.0f;
            float v657;
            if (v653){
                float v654;
                v654 = v652 * v591;
                float v655;
                v655 = v591 - 1.0f;
                float v656;
                v656 = v654 / v655;
                v657 = v656;
            } else {
                v657 = 0.0f;
            }
            float v658[4];
            int v659;
            v659 = 0;
            while (while_method_4(v659)){
                int v661;
                v661 = 0;
                while (while_method_7(v661)){
                    assert("Tensor range check" && 0 <= v659 && v659 < 1);
                    assert("Tensor range check" && 0 <= v661 && v661 < 4);
                    int v663;
                    v663 = 4 * v659;
                    int v664;
                    v664 = v663 + v661;
                    float v665;
                    v665 = v461[v664];
                    bool v666;
                    v666 = v558[v664];
                    float v667;
                    v667 = curand_normal(&v351);
                    bool v668;
                    v668 = v657 >= 0.1f;
                    float v669;
                    if (v668){
                        v669 = v657;
                    } else {
                        v669 = 0.1f;
                    }
                    float v670;
                    v670 = v667 * v669;
                    float v671;
                    v671 = v670 + v616;
                    float v672;
                    if (v666){
                        v672 = v665;
                    } else {
                        v672 = v671;
                    }
                    assert("Tensor range check" && 0 <= v659 && v659 < 1);
                    assert("Tensor range check" && 0 <= v661 && v661 < 4);
                    v658[v664] = v672;
                    v661 += 1 ;
                }
                v659 += 1 ;
            }
            assert("Tensor range check" && 0 <= v453 && v453 < 64);
            assert("Tensor range check" && 0 <= v452 && v452 < 4);
            int v673;
            v673 = 0;
            while (while_method_4(v673)){
                assert("Tensor range check" && 0 <= v673 && v673 < 1);
                int v675;
                v675 = 32 * v673;
                int v676;
                v676 = v675 + v460;
                assert("Tensor range check" && 0 <= v673 && v673 < 1);
                int v677;
                v677 = 4 * v673;
                int4* v678;
                v678 = reinterpret_cast<int4*>(v658 + v677);
                int4* v679;
                v679 = reinterpret_cast<int4*>(v354 + v676);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v678) % 16 == 0 && reinterpret_cast<unsigned long long>(v679) % 16 == 0);
                *v679 = *v678;
                v673 += 1 ;
            }
            v447 += 24 ;
        }
        v349.sync() ;
        static float v680[8192];
        int v681;
        v681 = threadIdx.x;
        int v682;
        v682 = blockIdx.x;
        int v683;
        v683 = v682 * 256;
        int v684;
        v684 = v681 + v683;
        int v685;
        v685 = v684 / 32;
        int v686;
        v686 = v685;
        while (while_method_13(v686)){
            bool v688;
            v688 = 0 <= v686;
            bool v689;
            v689 = v688 == false;
            if (v689){
                assert("The index needs to be zero or positive." && v688);
            } else {
            }
            int v691;
            v691 = v686 % 128;
            int v692;
            v692 = v686 / 128;
            bool v693;
            v693 = v692 < 64;
            bool v694;
            v694 = v693 == false;
            if (v694){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v693);
            } else {
            }
            assert("Tensor range check" && 0 <= v692 && v692 < 64);
            assert("Tensor range check" && 0 <= v691 && v691 < 128);
            int v696;
            v696 = 32 * v691;
            int v697;
            v697 = 4096 * v692;
            int v698;
            v698 = v697 + v696;
            float v699;
            v699 = 0.0f;
            int v700;
            v700 = threadIdx.x;
            int v701;
            v701 = v700 % 32;
            int v702;
            v702 = v701;
            while (while_method_5(v702)){
                bool v704;
                v704 = 0 <= v702;
                bool v705;
                v705 = v704 == false;
                if (v705){
                    assert("The index needs to be zero or positive." && v704);
                } else {
                }
                bool v707;
                v707 = v702 < 8;
                bool v708;
                v708 = v707 == false;
                if (v708){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v707);
                } else {
                }
                assert("Tensor range check" && 0 <= v702 && v702 < 8);
                int v710;
                v710 = 4 * v702;
                int v711;
                v711 = v710 + v698;
                float v712[4];
                int4* v713;
                v713 = reinterpret_cast<int4*>(v354 + v711);
                int4* v714;
                v714 = reinterpret_cast<int4*>(v712 + 0);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v713) % 16 == 0 && reinterpret_cast<unsigned long long>(v714) % 16 == 0);
                *v714 = *v713;
                int v715;
                v715 = 0;
                while (while_method_7(v715)){
                    assert("Tensor range check" && 0 <= v715 && v715 < 4);
                    float v717;
                    v717 = v712[v715];
                    float v718;
                    v718 = v717 * v717;
                    float v719;
                    v719 = v699 + v718;
                    v699 = v719;
                    v715 += 1 ;
                }
                v702 += 32 ;
            }
            __syncwarp();
            auto v720 = cooperative_groups::coalesced_threads();
            Closure0 v721{};
            float v722;
            v722 = cooperative_groups::reduce(v720, v699, v721);
            float v723;
            v723 = sqrt(v722);
            assert("Tensor range check" && 0 <= v692 && v692 < 64);
            assert("Tensor range check" && 0 <= v691 && v691 < 128);
            int v724;
            v724 = 128 * v692;
            int v725;
            v725 = v724 + v691;
            v680[v725] = v723;
            v686 += 192 ;
        }
        __syncthreads();
        v349.sync() ;
        float v726;
        v726 = 0.0f;
        int v727;
        v727 = threadIdx.x;
        int v728;
        v728 = blockIdx.x;
        int v729;
        v729 = v728 * 256;
        int v730;
        v730 = v727 + v729;
        int v731;
        v731 = v730;
        while (while_method_14(v731)){
            bool v733;
            v733 = 0 <= v731;
            bool v734;
            v734 = v733 == false;
            if (v734){
                assert("The index needs to be zero or positive." && v733);
            } else {
            }
            int v736;
            v736 = v731 % 32;
            int v737;
            v737 = v731 / 32;
            bool v738;
            v738 = v737 < 64;
            bool v739;
            v739 = v738 == false;
            if (v739){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v738);
            } else {
            }
            assert("Tensor range check" && 0 <= v737 && v737 < 64);
            assert("Tensor range check" && 0 <= v736 && v736 < 32);
            int v741;
            v741 = 4 * v736;
            int v742;
            v742 = 128 * v737;
            int v743;
            v743 = v742 + v741;
            float v744[4];
            int4* v745;
            v745 = reinterpret_cast<int4*>(v680 + v743);
            int4* v746;
            v746 = reinterpret_cast<int4*>(v744 + 0);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v745) % 16 == 0 && reinterpret_cast<unsigned long long>(v746) % 16 == 0);
            *v746 = *v745;
            int v747; float v748;
            Tuple13 tmp54 = Tuple13{0, v726};
            v747 = tmp54.v0; v748 = tmp54.v1;
            while (while_method_7(v747)){
                assert("Tensor range check" && 0 <= v747 && v747 < 4);
                float v750;
                v750 = v744[v747];
                bool v751;
                v751 = v748 >= v750;
                float v752;
                if (v751){
                    v752 = v748;
                } else {
                    v752 = v750;
                }
                v748 = v752;
                v747 += 1 ;
            }
            v726 = v748;
            v731 += 6144 ;
        }
        __syncwarp();
        auto v753 = cooperative_groups::coalesced_threads();
        Closure7 v754{};
        float v755;
        v755 = cooperative_groups::reduce(v753, v726, v754);
        int v756;
        v756 = threadIdx.x;
        int v757;
        v757 = v756 / 32;
        extern __shared__ unsigned char v758[];
        float * v759;
        v759 = reinterpret_cast<float *>(&v758[0ull]);
        assert("Tensor range check" && 0 <= v757 && v757 < 8);
        v759[v757] = v755;
        __syncthreads();
        int v761;
        v761 = threadIdx.x;
        int v762;
        v762 = v761 % 32;
        bool v763;
        v763 = v762 < 8;
        float v765;
        if (v763){
            assert("Tensor range check" && 0 <= v762 && v762 < 8);
            float v764;
            v764 = v759[v762];
            v765 = v764;
        } else {
            v765 = 0.0f;
        }
        __syncthreads();
        auto v766 = cooperative_groups::coalesced_threads();
        float v767;
        v767 = cooperative_groups::reduce(v766, v765, v754);
        int v768;
        v768 = blockIdx.x;
        static float v769[24];
        assert("Tensor range check" && 0 <= v768 && v768 < 24);
        v769[v768] = v767;
        v349.sync() ;
        float v770;
        v770 = 0.0f;
        int v771;
        v771 = threadIdx.x;
        int v772;
        v772 = v771 % 32;
        int v773;
        v773 = v772;
        while (while_method_15(v773)){
            bool v775;
            v775 = 0 <= v773;
            bool v776;
            v776 = v775 == false;
            if (v776){
                assert("The index needs to be zero or positive." && v775);
            } else {
            }
            bool v778;
            v778 = v773 < 24;
            bool v779;
            v779 = v778 == false;
            if (v779){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v778);
            } else {
            }
            assert("Tensor range check" && 0 <= v773 && v773 < 24);
            float v781;
            v781 = v769[v773];
            bool v782;
            v782 = v770 >= v781;
            float v783;
            if (v782){
                v783 = v770;
            } else {
                v783 = v781;
            }
            v770 = v783;
            v773 += 32 ;
        }
        __syncwarp();
        auto v784 = cooperative_groups::coalesced_threads();
        float v785;
        v785 = cooperative_groups::reduce(v784, v770, v754);
        int v786;
        v786 = threadIdx.x;
        int v787;
        v787 = blockIdx.x;
        int v788;
        v788 = v787 * 256;
        int v789;
        v789 = v786 + v788;
        bool v790;
        v790 = v789 == 0;
        if (v790){
            cuda::counting_semaphore<cuda::thread_scope_system, 1> & v791 = console_lock;
            auto v792 = cooperative_groups::coalesced_threads();
            v791.acquire();
            printf("{%s = %f}\n","max_norm", v785);
            v791.release();
            v792.sync() ;
        } else {
        }
        __syncwarp();
        extern __shared__ unsigned char v795[];
        float * v796;
        v796 = reinterpret_cast<float *>(&v795[0ull]);
        int v798;
        v798 = blockIdx.x;
        int v799;
        v799 = v798;
        while (while_method_8(v799)){
            bool v801;
            v801 = 0 <= v799;
            bool v802;
            v802 = v801 == false;
            if (v802){
                assert("The index needs to be zero or positive." && v801);
            } else {
            }
            int v804;
            v804 = v799 % 1;
            bool v805;
            v805 = v799 < 16;
            bool v806;
            v806 = v805 == false;
            if (v806){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v805);
            } else {
            }
            assert("Tensor range check" && 0 <= v799 && v799 < 16);
            assert("Tensor range check" && 0 <= v804 && v804 < 1);
            int v808;
            v808 = 32 * v804;
            int v809;
            v809 = 16384 * v799;
            int v810;
            v810 = v809 + v808;
            int v811;
            v811 = 262144 * v804;
            int v812;
            v812 = 512 * v799;
            int v813;
            v813 = v812 + v811;
            int v814;
            v814 = threadIdx.x;
            int v815;
            v815 = v814;
            while (while_method_12(v815)){
                bool v817;
                v817 = 0 <= v815;
                bool v818;
                v818 = v817 == false;
                if (v818){
                    assert("The index needs to be zero or positive." && v817);
                } else {
                }
                int v820;
                v820 = v815 % 32;
                int v821;
                v821 = v815 / 32;
                bool v822;
                v822 = v821 < 512;
                bool v823;
                v823 = v822 == false;
                if (v823){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v822);
                } else {
                }
                assert("Tensor range check" && 0 <= v821 && v821 < 512);
                assert("Tensor range check" && 0 <= v820 && v820 < 32);
                int v825;
                v825 = v820 + v810;
                int v826;
                v826 = 32 * v821;
                int v827;
                v827 = v826 + v825;
                float v828;
                v828 = v354[v827];
                assert("Tensor range check" && 0 <= v821 && v821 < 512);
                assert("Tensor range check" && 0 <= v820 && v820 < 32);
                int v829;
                v829 = 33 * v821;
                int v830;
                v830 = v829 + v820;
                v796[v830] = v828;
                v815 += 256 ;
            }
            __syncthreads();
            int v831;
            v831 = threadIdx.x;
            int v832;
            v832 = v831;
            while (while_method_12(v832)){
                bool v834;
                v834 = 0 <= v832;
                bool v835;
                v835 = v834 == false;
                if (v835){
                    assert("The index needs to be zero or positive." && v834);
                } else {
                }
                int v837;
                v837 = v832 % 512;
                int v838;
                v838 = v832 / 512;
                bool v839;
                v839 = v838 < 32;
                bool v840;
                v840 = v839 == false;
                if (v840){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v839);
                } else {
                }
                assert("Tensor range check" && 0 <= v838 && v838 < 32);
                assert("Tensor range check" && 0 <= v837 && v837 < 512);
                int v842;
                v842 = 33 * v837;
                int v843;
                v843 = v838 + v842;
                float v844;
                v844 = v796[v843];
                assert("Tensor range check" && 0 <= v838 && v838 < 32);
                assert("Tensor range check" && 0 <= v837 && v837 < 512);
                int v845;
                v845 = v837 + v813;
                int v846;
                v846 = 8192 * v838;
                int v847;
                v847 = v846 + v845;
                v352[v847] = v844;
                v832 += 256 ;
            }
            __syncthreads();
            v799 += 24 ;
        }
        v349.sync() ;
        int v848;
        v848 = threadIdx.x;
        int v849;
        v849 = blockIdx.x;
        int v850;
        v850 = v849 * 256;
        int v851;
        v851 = v848 + v850;
        int v852;
        v852 = v851;
        while (while_method_5(v852)){
            bool v854;
            v854 = 0 <= v852;
            bool v855;
            v855 = v854 == false;
            if (v855){
                assert("The index needs to be zero or positive." && v854);
            } else {
            }
            bool v857;
            v857 = v852 < 8;
            bool v858;
            v858 = v857 == false;
            if (v858){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v857);
            } else {
            }
            assert("Tensor range check" && 0 <= v852 && v852 < 8);
            int v860;
            v860 = 4 * v852;
            assert("Tensor range check" && 0 <= v852 && v852 < 8);
            float v861[4];
            float v862[4];
            float v863[4];
            float v864[4];
            int4* v865;
            v865 = reinterpret_cast<int4*>(v360 + v860);
            int4* v866;
            v866 = reinterpret_cast<int4*>(v861 + 0);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v865) % 16 == 0 && reinterpret_cast<unsigned long long>(v866) % 16 == 0);
            *v866 = *v865;
            int4* v867;
            v867 = reinterpret_cast<int4*>(v362 + v860);
            int4* v868;
            v868 = reinterpret_cast<int4*>(v862 + 0);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v867) % 16 == 0 && reinterpret_cast<unsigned long long>(v868) % 16 == 0);
            *v868 = *v867;
            // Pushing the loop unrolling to: 0
            int v869;
            v869 = 0;
            #pragma unroll
            while (while_method_7(v869)){
                assert("Tensor range check" && 0 <= v869 && v869 < 4);
                float v871;
                v871 = v861[v869];
                float v872;
                v872 = v862[v869];
                assert("Tensor range check" && 0 <= v869 && v869 < 4);
                v863[v869] = 0.0f;
                v864[v869] = 0.0f;
                v869 += 1 ;
            }
            // Poping the loop unrolling to: 0
            int4* v873;
            v873 = reinterpret_cast<int4*>(v863 + 0);
            int4* v874;
            v874 = reinterpret_cast<int4*>(v360 + v860);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v873) % 16 == 0 && reinterpret_cast<unsigned long long>(v874) % 16 == 0);
            *v874 = *v873;
            int4* v875;
            v875 = reinterpret_cast<int4*>(v864 + 0);
            int4* v876;
            v876 = reinterpret_cast<int4*>(v362 + v860);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v875) % 16 == 0 && reinterpret_cast<unsigned long long>(v876) % 16 == 0);
            *v876 = *v875;
            v852 += 6144 ;
        }
        v349.sync() ;
        v27 += 1 ;
    }
    cooperative_groups::grid_group & v877 = v26.v1;
    cooperative_groups::grid_group & v878 = v877;
    int v879;
    v879 = threadIdx.x;
    int v880;
    v880 = blockIdx.x;
    int v881;
    v881 = v880 * 256;
    int v882;
    v882 = v879 + v881;
    int v883;
    v883 = v882;
    while (while_method_14(v883)){
        bool v885;
        v885 = 0 <= v883;
        bool v886;
        v886 = v885 == false;
        if (v886){
            assert("The index needs to be zero or positive." && v885);
        } else {
        }
        int v888;
        v888 = v883 % 64;
        int v889;
        v889 = v883 / 64;
        bool v890;
        v890 = v889 < 32;
        bool v891;
        v891 = v890 == false;
        if (v891){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v890);
        } else {
        }
        assert("Tensor range check" && 0 <= v889 && v889 < 32);
        assert("Tensor range check" && 0 <= v888 && v888 < 64);
        int v893;
        v893 = 4 * v888;
        int v894;
        v894 = 256 * v889;
        int v895;
        v895 = v894 + v893;
        assert("Tensor range check" && 0 <= v889 && v889 < 32);
        assert("Tensor range check" && 0 <= v888 && v888 < 64);
        float v896[4];
        float v897[4];
        float v898[4];
        int4* v899;
        v899 = reinterpret_cast<int4*>(v3 + v895);
        int4* v900;
        v900 = reinterpret_cast<int4*>(v896 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v899) % 16 == 0 && reinterpret_cast<unsigned long long>(v900) % 16 == 0);
        *v900 = *v899;
        int4* v901;
        v901 = reinterpret_cast<int4*>(v4 + v895);
        int4* v902;
        v902 = reinterpret_cast<int4*>(v897 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v901) % 16 == 0 && reinterpret_cast<unsigned long long>(v902) % 16 == 0);
        *v902 = *v901;
        // Pushing the loop unrolling to: 0
        int v903;
        v903 = 0;
        #pragma unroll
        while (while_method_7(v903)){
            assert("Tensor range check" && 0 <= v903 && v903 < 4);
            float v905;
            v905 = v896[v903];
            float v906;
            v906 = v897[v903];
            bool v907;
            v907 = v906 == 0.0f;
            bool v908;
            v908 = v907 != true;
            float v910;
            if (v908){
                float v909;
                v909 = v905 / v906;
                v910 = v909;
            } else {
                v910 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v903 && v903 < 4);
            v898[v903] = v910;
            v903 += 1 ;
        }
        // Poping the loop unrolling to: 0
        int4* v911;
        v911 = reinterpret_cast<int4*>(v898 + 0);
        int4* v912;
        v912 = reinterpret_cast<int4*>(v5 + v895);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v911) % 16 == 0 && reinterpret_cast<unsigned long long>(v912) % 16 == 0);
        *v912 = *v911;
        v883 += 6144 ;
    }
    v878.sync() ;
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
            v35 = reinterpret_cast<int *>(&v0[1048576ull]);
            float * v37;
            v37 = reinterpret_cast<float *>(&v0[1048592ull]);
            float * v39;
            v39 = reinterpret_cast<float *>(&v0[1048720ull]);
            double * v41;
            v41 = reinterpret_cast<double *>(&v1[55050240ull]);
            double * v43;
            v43 = reinterpret_cast<double *>(&v1[58195968ull]);
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
                Tuple11 tmp71 = Tuple11{0, 0.0};
                v67 = tmp71.v0; v68 = tmp71.v1;
                while (while_method_11(v67)){
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
            while (while_method_11(v83)){
                int v85;
                v85 = 0;
                while (while_method_0(v85)){
                    bool v87;
                    v87 = v77 == 0.0;
                    bool v88;
                    v88 = v87 != true;
                    double v98;
                    if (v88){
                        assert("Tensor range check" && 0 <= v85 && v85 < 2);
                        double v89;
                        v89 = v64[v85];
                        double v90;
                        v90 = v77 / v89;
                        assert("Tensor range check" && 0 <= v83 && v83 < 32);
                        assert("Tensor range check" && 0 <= v85 && v85 < 2);
                        int v91;
                        v91 = 12288 * v83;
                        int v92;
                        v92 = v91 + v85;
                        double v93;
                        v93 = v50[v92];
                        double v94;
                        v94 = v52[v92];
                        double v95;
                        v95 = v93 - v94;
                        double v96;
                        v96 = exp(v95);
                        double v97;
                        v97 = v90 * v96;
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
                    assert("Tensor range check" && 0 <= v83 && v83 < 32);
                    assert("Tensor range check" && 0 <= v85 && v85 < 2);
                    int v103;
                    v103 = 2 * v83;
                    int v104;
                    v104 = v103 + v85;
                    v82[v104] = v98;
                    v85 += 1 ;
                }
                v83 += 1 ;
            }
            float v105[32];
            float v106[32];
            int v107;
            v107 = 0;
            while (while_method_11(v107)){
                int v109; float v110; double v111;
                Tuple12 tmp72 = Tuple12{0, 0.0f, 0.0};
                v109 = tmp72.v0; v110 = tmp72.v1; v111 = tmp72.v2;
                while (while_method_0(v109)){
                    assert("Tensor range check" && 0 <= v107 && v107 < 32);
                    assert("Tensor range check" && 0 <= v109 && v109 < 2);
                    int v113;
                    v113 = 2 * v107;
                    int v114;
                    v114 = v113 + v109;
                    double v115;
                    v115 = v82[v114];
                    assert("Tensor range check" && 0 <= v109 && v109 < 2);
                    float v116;
                    v116 = v54[v109];
                    float v117;
                    v117 = (float)v115;
                    float v118;
                    v118 = v117 * v116;
                    float v119;
                    v119 = v110 + v118;
                    double v120;
                    v120 = v111 + v115;
                    v110 = v119;
                    v111 = v120;
                    v109 += 1 ;
                }
                float v121;
                v121 = (float)v111;
                assert("Tensor range check" && 0 <= v107 && v107 < 32);
                v105[v107] = v110;
                v106[v107] = v121;
                v107 += 1 ;
            }
            int v122;
            v122 = 0;
            while (while_method_11(v122)){
                assert("Tensor range check" && 0 <= v122 && v122 < 32);
                float v124;
                v124 = v105[v122];
                float v125;
                v125 = v106[v122];
                bool v126;
                v126 = isnan(v125);
                bool v127;
                v127 = v126 == false;
                bool v128;
                v128 = v127 == false;
                if (v128){
                    assert("The path probability after integration should not be a nan in calculate updates." && v127);
                } else {
                }
                float v130;
                v130 = v124 * v125;
                assert("Tensor range check" && 0 <= v122 && v122 < 32);
                float * v131;
                v131 = v37+v122;
                float * v133;
                v133 = v39+v122;
                float v135;
                v135 = atomicAdd(v131,v130);
                float v136;
                v136 = atomicAdd(v133,v125);
                v122 += 1 ;
            }
            int v137;
            v137 = threadIdx.x;
            int v138;
            v138 = blockIdx.x;
            int v139;
            v139 = v138 * 256;
            int v140;
            v140 = v137 + v139;
            int v141;
            v141 = 0;
            while (while_method_11(v141)){
                assert("Tensor range check" && 0 <= v141 && v141 < 32);
                int v143;
                v143 = 12288 * v141;
                assert("Tensor range check" && 0 <= v140 && v140 < 6144);
                int v144;
                v144 = 2 * v140;
                int v145;
                v145 = v144 + v143;
                double * v146;
                v146 = v41+v145;
                double * v148;
                v148 = v43+v145;
                double * v150;
                v150 = v41+v145;
                double * v152;
                v152 = v43+v145;
                int v154;
                v154 = sizeof(double *);
                unsigned long long v155;
                v155 = (unsigned long long)v154;
                unsigned long long v156;
                v156 = 256ull * v155;
                unsigned long long v157;
                v157 = v156 + 16ull;
                unsigned long long v158;
                v158 = v157 - 1ull;
                unsigned long long v159;
                v159 = v158 % 16ull;
                unsigned long long v160;
                v160 = v158 - v159;
                unsigned long long v161;
                v161 = v160 + v156;
                unsigned long long v162;
                v162 = v161 + 16ull;
                unsigned long long v163;
                v163 = v162 - 1ull;
                unsigned long long v164;
                v164 = v163 % 16ull;
                unsigned long long v165;
                v165 = v163 - v164;
                unsigned long long v166;
                v166 = v165 + v156;
                unsigned long long v167;
                v167 = v166 + 16ull;
                unsigned long long v168;
                v168 = v167 - 1ull;
                unsigned long long v169;
                v169 = v168 % 16ull;
                unsigned long long v170;
                v170 = v168 - v169;
                unsigned long long v171;
                v171 = v170 + v156;
                bool v172;
                v172 = v171 <= 98304ull;
                bool v173;
                v173 = v172 == false;
                if (v173){
                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v172);
                } else {
                }
                extern __shared__ unsigned char v175[];
                bool v176;
                v176 = v171 <= v171;
                bool v177;
                v177 = v176 == false;
                if (v177){
                    assert("The length of the partition has to be less than or equal to the length of the base array." && v176);
                } else {
                }
                double * * v179;
                v179 = reinterpret_cast<double * *>(&v175[0ull]);
                double * * v181;
                v181 = reinterpret_cast<double * *>(&v175[v160]);
                double * * v183;
                v183 = reinterpret_cast<double * *>(&v175[v165]);
                double * * v185;
                v185 = reinterpret_cast<double * *>(&v175[v170]);
                int v187;
                v187 = threadIdx.x;
                assert("Tensor range check" && 0 <= v187 && v187 < 256);
                v179[v187] = v146;
                v181[v187] = v148;
                v183[v187] = v150;
                v185[v187] = v152;
                __syncthreads();
                bool v188;
                v188 = 0 <= v187;
                bool v189;
                v189 = v188 == false;
                if (v189){
                    assert("The index needs to be zero or positive." && v188);
                } else {
                }
                int v191;
                v191 = v187 % 1;
                bool v192;
                v192 = v187 < 256;
                bool v193;
                v193 = v192 == false;
                if (v193){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v192);
                } else {
                }
                assert("Tensor range check" && 0 <= v187 && v187 < 256);
                int v195;
                v195 = 0;
                while (while_method_4(v195)){
                    bool v197;
                    v197 = v188 && v192;
                    bool v198;
                    v198 = v197 == false;
                    if (v198){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v197);
                    } else {
                    }
                    bool v200;
                    v200 = 0 <= v195;
                    bool v202;
                    if (v200){
                        bool v201;
                        v201 = v195 < 1;
                        v202 = v201;
                    } else {
                        v202 = false;
                    }
                    bool v203;
                    v203 = v202 == false;
                    if (v203){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v202);
                    } else {
                    }
                    int v205;
                    v205 = v195 * 256;
                    int v206;
                    v206 = v205 + v187;
                    assert("Tensor range check" && 0 <= v195 && v195 < 1);
                    int v207;
                    v207 = 256 * v195;
                    int v208;
                    v208 = v207 + v187;
                    double * v209;
                    v209 = v179[v208];
                    double * v210;
                    v210 = v181[v208];
                    double * v211;
                    v211 = v183[v208];
                    double * v212;
                    v212 = v185[v208];
                    int v213;
                    v213 = blockIdx.x;
                    int v214;
                    v214 = v213 * 256;
                    int v215;
                    v215 = v214 + v206;
                    assert("Tensor range check" && 0 <= v191 && v191 < 1);
                    int v216;
                    v216 = 2 * v191;
                    double v217[2];
                    double v218[2];
                    int v219[2];
                    int v220;
                    v220 = 0;
                    while (while_method_4(v220)){
                        assert("Tensor range check" && 0 <= v220 && v220 < 1);
                        int v222;
                        v222 = 2 * v220;
                        assert("Tensor range check" && 0 <= v220 && v220 < 1);
                        int v223;
                        v223 = v222 + v216;
                        int4* v224;
                        v224 = reinterpret_cast<int4*>(v209 + v223);
                        int4* v225;
                        v225 = reinterpret_cast<int4*>(v217 + v222);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v224) % 16 == 0 && reinterpret_cast<unsigned long long>(v225) % 16 == 0);
                        *v225 = *v224;
                        int4* v226;
                        v226 = reinterpret_cast<int4*>(v210 + v223);
                        int4* v227;
                        v227 = reinterpret_cast<int4*>(v218 + v222);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v226) % 16 == 0 && reinterpret_cast<unsigned long long>(v227) % 16 == 0);
                        *v227 = *v226;
                        v220 += 1 ;
                    }
                    int v228;
                    v228 = 0;
                    while (while_method_4(v228)){
                        int v230;
                        v230 = 0;
                        while (while_method_0(v230)){
                            bool v232;
                            v232 = 0 <= v230;
                            bool v234;
                            if (v232){
                                bool v233;
                                v233 = v230 < 2;
                                v234 = v233;
                            } else {
                                v234 = false;
                            }
                            bool v235;
                            v235 = v234 == false;
                            if (v235){
                                assert("The indices should be inside the range of the dimension." && v234);
                            } else {
                            }
                            bool v237;
                            v237 = 0 <= v191;
                            bool v239;
                            if (v237){
                                bool v238;
                                v238 = v191 < 1;
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
                            int v242;
                            v242 = v191 * 2;
                            int v243;
                            v243 = v230 + v242;
                            bool v244;
                            v244 = 0 <= v228;
                            bool v246;
                            if (v244){
                                bool v245;
                                v245 = v228 < 1;
                                v246 = v245;
                            } else {
                                v246 = false;
                            }
                            bool v247;
                            v247 = v246 == false;
                            if (v247){
                                assert("The indices should be inside the range of the dimension." && v246);
                            } else {
                            }
                            int v249;
                            v249 = v228 * 2;
                            int v250;
                            v250 = v243 + v249;
                            assert("Tensor range check" && 0 <= v228 && v228 < 1);
                            assert("Tensor range check" && 0 <= v230 && v230 < 2);
                            int v251;
                            v251 = 2 * v228;
                            int v252;
                            v252 = v251 + v230;
                            v219[v252] = v250;
                            v230 += 1 ;
                        }
                        v228 += 1 ;
                    }
                    double v253[2];
                    double v254[2];
                    int v255;
                    v255 = 0;
                    while (while_method_4(v255)){
                        int v257;
                        v257 = 0;
                        while (while_method_0(v257)){
                            assert("Tensor range check" && 0 <= v255 && v255 < 1);
                            assert("Tensor range check" && 0 <= v257 && v257 < 2);
                            int v259;
                            v259 = 2 * v255;
                            int v260;
                            v260 = v259 + v257;
                            double v261;
                            v261 = v217[v260];
                            double v262;
                            v262 = v218[v260];
                            assert("Tensor range check" && 0 <= v255 && v255 < 1);
                            assert("Tensor range check" && 0 <= v257 && v257 < 2);
                            v253[v260] = 0.0;
                            v254[v260] = 0.0;
                            v257 += 1 ;
                        }
                        v255 += 1 ;
                    }
                    int v263;
                    v263 = 0;
                    while (while_method_4(v263)){
                        assert("Tensor range check" && 0 <= v263 && v263 < 1);
                        int v265;
                        v265 = 2 * v263;
                        int v266;
                        v266 = v265 + v216;
                        assert("Tensor range check" && 0 <= v263 && v263 < 1);
                        int4* v267;
                        v267 = reinterpret_cast<int4*>(v253 + v265);
                        int4* v268;
                        v268 = reinterpret_cast<int4*>(v211 + v266);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v267) % 16 == 0 && reinterpret_cast<unsigned long long>(v268) % 16 == 0);
                        *v268 = *v267;
                        int4* v269;
                        v269 = reinterpret_cast<int4*>(v254 + v265);
                        int4* v270;
                        v270 = reinterpret_cast<int4*>(v212 + v266);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v269) % 16 == 0 && reinterpret_cast<unsigned long long>(v270) % 16 == 0);
                        *v270 = *v269;
                        v263 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v206 && v206 < 256);
                    v195 += 1 ;
                }
                __syncthreads();
                assert("Tensor range check" && 0 <= v187 && v187 < 256);
                __syncthreads();
                v141 += 1 ;
            }
            Union4 v271;
            v271 = Union4{Union4_1{}};
            method_48(v0, v1, v2, v26, v271);
            double * v272;
            v272 = reinterpret_cast<double *>(&v1[55050240ull]);
            double * v274;
            v274 = reinterpret_cast<double *>(&v1[58195968ull]);
            int v276;
            v276 = threadIdx.x;
            int v277;
            v277 = blockIdx.x;
            int v278;
            v278 = v277 * 256;
            int v279;
            v279 = v276 + v278;
            assert("Tensor range check" && 0 <= v279 && v279 < 6144);
            int v280;
            v280 = 2 * v279;
            static_array<float,2> & v281 = v26.v4;
            float v282[2];
            int v283;
            v283 = 0;
            while (while_method_0(v283)){
                bool v285;
                v285 = 0 <= v283;
                bool v287;
                if (v285){
                    bool v286;
                    v286 = v283 < 2;
                    v287 = v286;
                } else {
                    v287 = false;
                }
                bool v288;
                v288 = v287 == false;
                if (v288){
                    assert("Index must be in range." && v287);
                } else {
                }
                float v290;
                v290 = v281[v283];
                assert("Tensor range check" && 0 <= v283 && v283 < 2);
                v282[v283] = v290;
                v283 += 1 ;
            }
            double v292[2];
            int v293;
            v293 = 0;
            while (while_method_0(v293)){
                int v295; double v296;
                Tuple11 tmp89 = Tuple11{0, 0.0};
                v295 = tmp89.v0; v296 = tmp89.v1;
                while (while_method_11(v295)){
                    assert("Tensor range check" && 0 <= v295 && v295 < 32);
                    assert("Tensor range check" && 0 <= v293 && v293 < 2);
                    int v298;
                    v298 = v293 + v280;
                    int v299;
                    v299 = 12288 * v295;
                    int v300;
                    v300 = v299 + v298;
                    double v301;
                    v301 = v272[v300];
                    double v302;
                    v302 = v274[v300];
                    double v303;
                    v303 = v301 - v302;
                    double v304;
                    v304 = exp(v303);
                    double v305;
                    v305 = v296 + v304;
                    v296 = v305;
                    v295 += 1 ;
                }
                assert("Tensor range check" && 0 <= v293 && v293 < 2);
                v292[v293] = v296;
                v293 += 1 ;
            }
            double v306;
            v306 = 1.0;
            int v307;
            v307 = 0;
            while (while_method_0(v307)){
                assert("Tensor range check" && 0 <= v307 && v307 < 2);
                double v309;
                v309 = v292[v307];
                double v310;
                v310 = v306 * v309;
                v306 = v310;
                v307 += 1 ;
            }
            double v311[64];
            int v312;
            v312 = 0;
            while (while_method_11(v312)){
                int v314;
                v314 = 0;
                while (while_method_0(v314)){
                    bool v316;
                    v316 = v306 == 0.0;
                    bool v317;
                    v317 = v316 != true;
                    double v328;
                    if (v317){
                        assert("Tensor range check" && 0 <= v314 && v314 < 2);
                        double v318;
                        v318 = v292[v314];
                        double v319;
                        v319 = v306 / v318;
                        assert("Tensor range check" && 0 <= v312 && v312 < 32);
                        assert("Tensor range check" && 0 <= v314 && v314 < 2);
                        int v320;
                        v320 = v314 + v280;
                        int v321;
                        v321 = 12288 * v312;
                        int v322;
                        v322 = v321 + v320;
                        double v323;
                        v323 = v272[v322];
                        double v324;
                        v324 = v274[v322];
                        double v325;
                        v325 = v323 - v324;
                        double v326;
                        v326 = exp(v325);
                        double v327;
                        v327 = v319 * v326;
                        v328 = v327;
                    } else {
                        v328 = 0.0;
                    }
                    bool v329;
                    v329 = isnan(v328);
                    bool v330;
                    v330 = v329 == false;
                    bool v331;
                    v331 = v330 == false;
                    if (v331){
                        assert("The path probability after integration should not be a nan in integrate_rewards_." && v330);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v312 && v312 < 32);
                    assert("Tensor range check" && 0 <= v314 && v314 < 2);
                    int v333;
                    v333 = 2 * v312;
                    int v334;
                    v334 = v333 + v314;
                    v311[v334] = v328;
                    v314 += 1 ;
                }
                v312 += 1 ;
            }
            float v335[32];
            float v336[32];
            int v337;
            v337 = 0;
            while (while_method_11(v337)){
                int v339; float v340; double v341;
                Tuple12 tmp90 = Tuple12{0, 0.0f, 0.0};
                v339 = tmp90.v0; v340 = tmp90.v1; v341 = tmp90.v2;
                while (while_method_0(v339)){
                    assert("Tensor range check" && 0 <= v337 && v337 < 32);
                    assert("Tensor range check" && 0 <= v339 && v339 < 2);
                    int v343;
                    v343 = 2 * v337;
                    int v344;
                    v344 = v343 + v339;
                    double v345;
                    v345 = v311[v344];
                    assert("Tensor range check" && 0 <= v339 && v339 < 2);
                    float v346;
                    v346 = v282[v339];
                    float v347;
                    v347 = (float)v345;
                    float v348;
                    v348 = v347 * v346;
                    float v349;
                    v349 = v340 + v348;
                    double v350;
                    v350 = v341 + v345;
                    v340 = v349;
                    v341 = v350;
                    v339 += 1 ;
                }
                float v351;
                v351 = (float)v341;
                assert("Tensor range check" && 0 <= v337 && v337 < 32);
                v335[v337] = v340;
                v336[v337] = v351;
                v337 += 1 ;
            }
            int v352;
            v352 = 0;
            while (while_method_11(v352)){
                assert("Tensor range check" && 0 <= v352 && v352 < 32);
                float v354;
                v354 = v335[v352];
                float v355;
                v355 = v336[v352];
                assert("Tensor range check" && 0 <= v352 && v352 < 32);
                assert("Tensor range check" && 0 <= v27 && v27 < 256);
                int v356;
                v356 = 256 * v352;
                int v357;
                v357 = v356 + v27;
                float * v358;
                v358 = v3+v357;
                float * v360;
                v360 = v4+v357;
                float v362;
                v362 = atomicAdd(v358,v354);
                float v363;
                v363 = atomicAdd(v360,v355);
                v352 += 1 ;
            }
            double * v364;
            v364 = reinterpret_cast<double *>(&v1[55050240ull]);
            double * v366;
            v366 = reinterpret_cast<double *>(&v1[58195968ull]);
            int v368;
            v368 = threadIdx.x;
            int v369;
            v369 = blockIdx.x;
            int v370;
            v370 = v369 * 256;
            int v371;
            v371 = v368 + v370;
            int v372;
            v372 = 0;
            while (while_method_11(v372)){
                assert("Tensor range check" && 0 <= v372 && v372 < 32);
                int v374;
                v374 = 12288 * v372;
                assert("Tensor range check" && 0 <= v371 && v371 < 6144);
                int v375;
                v375 = 2 * v371;
                int v376;
                v376 = v375 + v374;
                double * v377;
                v377 = v364+v376;
                double * v379;
                v379 = v366+v376;
                double * v381;
                v381 = v364+v376;
                double * v383;
                v383 = v366+v376;
                int v385;
                v385 = sizeof(double *);
                unsigned long long v386;
                v386 = (unsigned long long)v385;
                unsigned long long v387;
                v387 = 256ull * v386;
                unsigned long long v388;
                v388 = v387 + 16ull;
                unsigned long long v389;
                v389 = v388 - 1ull;
                unsigned long long v390;
                v390 = v389 % 16ull;
                unsigned long long v391;
                v391 = v389 - v390;
                unsigned long long v392;
                v392 = v391 + v387;
                unsigned long long v393;
                v393 = v392 + 16ull;
                unsigned long long v394;
                v394 = v393 - 1ull;
                unsigned long long v395;
                v395 = v394 % 16ull;
                unsigned long long v396;
                v396 = v394 - v395;
                unsigned long long v397;
                v397 = v396 + v387;
                unsigned long long v398;
                v398 = v397 + 16ull;
                unsigned long long v399;
                v399 = v398 - 1ull;
                unsigned long long v400;
                v400 = v399 % 16ull;
                unsigned long long v401;
                v401 = v399 - v400;
                unsigned long long v402;
                v402 = v401 + v387;
                bool v403;
                v403 = v402 <= 98304ull;
                bool v404;
                v404 = v403 == false;
                if (v404){
                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v403);
                } else {
                }
                extern __shared__ unsigned char v406[];
                bool v407;
                v407 = v402 <= v402;
                bool v408;
                v408 = v407 == false;
                if (v408){
                    assert("The length of the partition has to be less than or equal to the length of the base array." && v407);
                } else {
                }
                double * * v410;
                v410 = reinterpret_cast<double * *>(&v406[0ull]);
                double * * v412;
                v412 = reinterpret_cast<double * *>(&v406[v391]);
                double * * v414;
                v414 = reinterpret_cast<double * *>(&v406[v396]);
                double * * v416;
                v416 = reinterpret_cast<double * *>(&v406[v401]);
                int v418;
                v418 = threadIdx.x;
                assert("Tensor range check" && 0 <= v418 && v418 < 256);
                v410[v418] = v377;
                v412[v418] = v379;
                v414[v418] = v381;
                v416[v418] = v383;
                __syncthreads();
                bool v419;
                v419 = 0 <= v418;
                bool v420;
                v420 = v419 == false;
                if (v420){
                    assert("The index needs to be zero or positive." && v419);
                } else {
                }
                int v422;
                v422 = v418 % 1;
                bool v423;
                v423 = v418 < 256;
                bool v424;
                v424 = v423 == false;
                if (v424){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v423);
                } else {
                }
                assert("Tensor range check" && 0 <= v418 && v418 < 256);
                int v426;
                v426 = 0;
                while (while_method_4(v426)){
                    bool v428;
                    v428 = v419 && v423;
                    bool v429;
                    v429 = v428 == false;
                    if (v429){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v428);
                    } else {
                    }
                    bool v431;
                    v431 = 0 <= v426;
                    bool v433;
                    if (v431){
                        bool v432;
                        v432 = v426 < 1;
                        v433 = v432;
                    } else {
                        v433 = false;
                    }
                    bool v434;
                    v434 = v433 == false;
                    if (v434){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v433);
                    } else {
                    }
                    int v436;
                    v436 = v426 * 256;
                    int v437;
                    v437 = v436 + v418;
                    assert("Tensor range check" && 0 <= v426 && v426 < 1);
                    int v438;
                    v438 = 256 * v426;
                    int v439;
                    v439 = v438 + v418;
                    double * v440;
                    v440 = v410[v439];
                    double * v441;
                    v441 = v412[v439];
                    double * v442;
                    v442 = v414[v439];
                    double * v443;
                    v443 = v416[v439];
                    int v444;
                    v444 = blockIdx.x;
                    int v445;
                    v445 = v444 * 256;
                    int v446;
                    v446 = v445 + v437;
                    assert("Tensor range check" && 0 <= v422 && v422 < 1);
                    int v447;
                    v447 = 2 * v422;
                    double v448[2];
                    double v449[2];
                    int v450[2];
                    int v451;
                    v451 = 0;
                    while (while_method_4(v451)){
                        assert("Tensor range check" && 0 <= v451 && v451 < 1);
                        int v453;
                        v453 = 2 * v451;
                        assert("Tensor range check" && 0 <= v451 && v451 < 1);
                        int v454;
                        v454 = v453 + v447;
                        int4* v455;
                        v455 = reinterpret_cast<int4*>(v440 + v454);
                        int4* v456;
                        v456 = reinterpret_cast<int4*>(v448 + v453);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v455) % 16 == 0 && reinterpret_cast<unsigned long long>(v456) % 16 == 0);
                        *v456 = *v455;
                        int4* v457;
                        v457 = reinterpret_cast<int4*>(v441 + v454);
                        int4* v458;
                        v458 = reinterpret_cast<int4*>(v449 + v453);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v457) % 16 == 0 && reinterpret_cast<unsigned long long>(v458) % 16 == 0);
                        *v458 = *v457;
                        v451 += 1 ;
                    }
                    int v459;
                    v459 = 0;
                    while (while_method_4(v459)){
                        int v461;
                        v461 = 0;
                        while (while_method_0(v461)){
                            bool v463;
                            v463 = 0 <= v461;
                            bool v465;
                            if (v463){
                                bool v464;
                                v464 = v461 < 2;
                                v465 = v464;
                            } else {
                                v465 = false;
                            }
                            bool v466;
                            v466 = v465 == false;
                            if (v466){
                                assert("The indices should be inside the range of the dimension." && v465);
                            } else {
                            }
                            bool v468;
                            v468 = 0 <= v422;
                            bool v470;
                            if (v468){
                                bool v469;
                                v469 = v422 < 1;
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
                            v473 = v422 * 2;
                            int v474;
                            v474 = v461 + v473;
                            bool v475;
                            v475 = 0 <= v459;
                            bool v477;
                            if (v475){
                                bool v476;
                                v476 = v459 < 1;
                                v477 = v476;
                            } else {
                                v477 = false;
                            }
                            bool v478;
                            v478 = v477 == false;
                            if (v478){
                                assert("The indices should be inside the range of the dimension." && v477);
                            } else {
                            }
                            int v480;
                            v480 = v459 * 2;
                            int v481;
                            v481 = v474 + v480;
                            assert("Tensor range check" && 0 <= v459 && v459 < 1);
                            assert("Tensor range check" && 0 <= v461 && v461 < 2);
                            int v482;
                            v482 = 2 * v459;
                            int v483;
                            v483 = v482 + v461;
                            v450[v483] = v481;
                            v461 += 1 ;
                        }
                        v459 += 1 ;
                    }
                    double v484[2];
                    double v485[2];
                    int v486;
                    v486 = 0;
                    while (while_method_4(v486)){
                        int v488;
                        v488 = 0;
                        while (while_method_0(v488)){
                            assert("Tensor range check" && 0 <= v486 && v486 < 1);
                            assert("Tensor range check" && 0 <= v488 && v488 < 2);
                            int v490;
                            v490 = 2 * v486;
                            int v491;
                            v491 = v490 + v488;
                            double v492;
                            v492 = v448[v491];
                            double v493;
                            v493 = v449[v491];
                            assert("Tensor range check" && 0 <= v486 && v486 < 1);
                            assert("Tensor range check" && 0 <= v488 && v488 < 2);
                            v484[v491] = 0.0;
                            v485[v491] = 0.0;
                            v488 += 1 ;
                        }
                        v486 += 1 ;
                    }
                    int v494;
                    v494 = 0;
                    while (while_method_4(v494)){
                        assert("Tensor range check" && 0 <= v494 && v494 < 1);
                        int v496;
                        v496 = 2 * v494;
                        int v497;
                        v497 = v496 + v447;
                        assert("Tensor range check" && 0 <= v494 && v494 < 1);
                        int4* v498;
                        v498 = reinterpret_cast<int4*>(v484 + v496);
                        int4* v499;
                        v499 = reinterpret_cast<int4*>(v442 + v497);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v498) % 16 == 0 && reinterpret_cast<unsigned long long>(v499) % 16 == 0);
                        *v499 = *v498;
                        int4* v500;
                        v500 = reinterpret_cast<int4*>(v485 + v496);
                        int4* v501;
                        v501 = reinterpret_cast<int4*>(v443 + v497);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v500) % 16 == 0 && reinterpret_cast<unsigned long long>(v501) % 16 == 0);
                        *v501 = *v500;
                        v494 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v437 && v437 < 256);
                    v426 += 1 ;
                }
                __syncthreads();
                assert("Tensor range check" && 0 <= v418 && v418 < 256);
                __syncthreads();
                v372 += 1 ;
            }
            v29 += 1 ;
        }
        cooperative_groups::grid_group & v502 = v26.v1;
        cooperative_groups::grid_group & v503 = v502;
        curandStatePhilox4_32_10_t & v504 = v26.v5;
        curandStatePhilox4_32_10_t & v505 = v504;
        float * v506;
        v506 = reinterpret_cast<float *>(&v0[0ull]);
        float * v508;
        v508 = reinterpret_cast<float *>(&v2[0ull]);
        float * v510;
        v510 = reinterpret_cast<float *>(&v1[4718592ull]);
        int * v512;
        v512 = reinterpret_cast<int *>(&v0[1048576ull]);
        float * v514;
        v514 = reinterpret_cast<float *>(&v0[1048592ull]);
        float * v516;
        v516 = reinterpret_cast<float *>(&v0[1048720ull]);
        double * v518;
        v518 = reinterpret_cast<double *>(&v1[55050240ull]);
        double * v520;
        v520 = reinterpret_cast<double *>(&v1[58195968ull]);
        v503.sync() ;
        int v522;
        v522 = threadIdx.x;
        int v523;
        v523 = blockIdx.x;
        int v524;
        v524 = v523 * 256;
        int v525;
        v525 = v522 + v524;
        bool v526;
        v526 = v525 == 0;
        if (v526){
            int v527;
            v527 = 0;
            int v528;
            v528 = 32;
            int v529;
            v529 = int_range_22(v528, v527, v505);
            v512[0] = v529;
        } else {
        }
        __syncwarp();
        extern __shared__ unsigned char v530[];
        float * v531;
        v531 = reinterpret_cast<float *>(&v530[0ull]);
        int v533;
        v533 = blockIdx.x;
        int v534;
        v534 = v533;
        while (while_method_8(v534)){
            bool v536;
            v536 = 0 <= v534;
            bool v537;
            v537 = v536 == false;
            if (v537){
                assert("The index needs to be zero or positive." && v536);
            } else {
            }
            int v539;
            v539 = v534 % 16;
            int v540;
            v540 = v534 / 16;
            bool v541;
            v541 = v540 < 1;
            bool v542;
            v542 = v541 == false;
            if (v542){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v541);
            } else {
            }
            assert("Tensor range check" && 0 <= v540 && v540 < 1);
            assert("Tensor range check" && 0 <= v539 && v539 < 16);
            int v544;
            v544 = 512 * v539;
            int v545;
            v545 = 262144 * v540;
            int v546;
            v546 = v545 + v544;
            int v547;
            v547 = 16384 * v539;
            int v548;
            v548 = 32 * v540;
            int v549;
            v549 = v548 + v547;
            int v550;
            v550 = threadIdx.x;
            int v551;
            v551 = v550;
            while (while_method_12(v551)){
                bool v553;
                v553 = 0 <= v551;
                bool v554;
                v554 = v553 == false;
                if (v554){
                    assert("The index needs to be zero or positive." && v553);
                } else {
                }
                int v556;
                v556 = v551 % 512;
                int v557;
                v557 = v551 / 512;
                bool v558;
                v558 = v557 < 32;
                bool v559;
                v559 = v558 == false;
                if (v559){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v558);
                } else {
                }
                assert("Tensor range check" && 0 <= v557 && v557 < 32);
                assert("Tensor range check" && 0 <= v556 && v556 < 512);
                int v561;
                v561 = v556 + v546;
                int v562;
                v562 = 8192 * v557;
                int v563;
                v563 = v562 + v561;
                float v564;
                v564 = v506[v563];
                assert("Tensor range check" && 0 <= v557 && v557 < 32);
                assert("Tensor range check" && 0 <= v556 && v556 < 512);
                int v565;
                v565 = 513 * v557;
                int v566;
                v566 = v565 + v556;
                v531[v566] = v564;
                v551 += 256 ;
            }
            __syncthreads();
            int v567;
            v567 = threadIdx.x;
            int v568;
            v568 = v567;
            while (while_method_12(v568)){
                bool v570;
                v570 = 0 <= v568;
                bool v571;
                v571 = v570 == false;
                if (v571){
                    assert("The index needs to be zero or positive." && v570);
                } else {
                }
                int v573;
                v573 = v568 % 32;
                int v574;
                v574 = v568 / 32;
                bool v575;
                v575 = v574 < 512;
                bool v576;
                v576 = v575 == false;
                if (v576){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v575);
                } else {
                }
                assert("Tensor range check" && 0 <= v574 && v574 < 512);
                assert("Tensor range check" && 0 <= v573 && v573 < 32);
                int v578;
                v578 = 513 * v573;
                int v579;
                v579 = v574 + v578;
                float v580;
                v580 = v531[v579];
                assert("Tensor range check" && 0 <= v574 && v574 < 512);
                assert("Tensor range check" && 0 <= v573 && v573 < 32);
                int v581;
                v581 = v573 + v549;
                int v582;
                v582 = 32 * v574;
                int v583;
                v583 = v582 + v581;
                v508[v583] = v580;
                v568 += 256 ;
            }
            __syncthreads();
            v534 += 24 ;
        }
        v503.sync() ;
        int v584;
        v584 = threadIdx.x;
        bool v585;
        v585 = 0 <= v584;
        bool v586;
        v586 = v585 == false;
        if (v586){
            assert("The index needs to be zero or positive." && v585);
        } else {
        }
        int v588;
        v588 = v584 % 8;
        int v589;
        v589 = v584 / 8;
        int v590;
        v590 = v589 % 32;
        int v591;
        v591 = v589 / 32;
        bool v592;
        v592 = v591 < 1;
        bool v593;
        v593 = v592 == false;
        if (v593){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v592);
        } else {
        }
        assert("Tensor range check" && 0 <= v591 && v591 < 1);
        assert("Tensor range check" && 0 <= v590 && v590 < 32);
        assert("Tensor range check" && 0 <= v588 && v588 < 8);
        int v595;
        v595 = 4 * v588;
        int v596;
        v596 = 32 * v590;
        int v597;
        v597 = v596 + v595;
        int v598;
        v598 = 4096 * v591;
        int v599;
        v599 = v598 + v597;
        assert("Tensor range check" && 0 <= v591 && v591 < 1);
        assert("Tensor range check" && 0 <= v590 && v590 < 32);
        assert("Tensor range check" && 0 <= v588 && v588 < 8);
        int v600;
        v600 = blockIdx.x;
        int v601;
        v601 = v600;
        while (while_method_9(v601)){
            bool v603;
            v603 = 0 <= v601;
            bool v604;
            v604 = v603 == false;
            if (v604){
                assert("The index needs to be zero or positive." && v603);
            } else {
            }
            int v606;
            v606 = v601 % 4;
            int v607;
            v607 = v601 / 4;
            bool v608;
            v608 = v607 < 64;
            bool v609;
            v609 = v608 == false;
            if (v609){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v608);
            } else {
            }
            assert("Tensor range check" && 0 <= v607 && v607 < 64);
            assert("Tensor range check" && 0 <= v606 && v606 < 4);
            int v611;
            v611 = 1024 * v606;
            int v612;
            v612 = v611 + v599;
            int v613;
            v613 = 4096 * v607;
            int v614;
            v614 = v613 + v612;
            float v615[4];
            int v616[4];
            int v617;
            v617 = 0;
            while (while_method_4(v617)){
                assert("Tensor range check" && 0 <= v617 && v617 < 1);
                int v619;
                v619 = 4 * v617;
                assert("Tensor range check" && 0 <= v617 && v617 < 1);
                int v620;
                v620 = 32 * v617;
                int v621;
                v621 = v620 + v614;
                int4* v622;
                v622 = reinterpret_cast<int4*>(v508 + v621);
                int4* v623;
                v623 = reinterpret_cast<int4*>(v615 + v619);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v622) % 16 == 0 && reinterpret_cast<unsigned long long>(v623) % 16 == 0);
                *v623 = *v622;
                v617 += 1 ;
            }
            int v624;
            v624 = 0;
            while (while_method_4(v624)){
                int v626;
                v626 = 0;
                while (while_method_7(v626)){
                    bool v628;
                    v628 = 0 <= v626;
                    bool v630;
                    if (v628){
                        bool v629;
                        v629 = v626 < 4;
                        v630 = v629;
                    } else {
                        v630 = false;
                    }
                    bool v631;
                    v631 = v630 == false;
                    if (v631){
                        assert("The indices should be inside the range of the dimension." && v630);
                    } else {
                    }
                    bool v633;
                    v633 = 0 <= v588;
                    bool v635;
                    if (v633){
                        bool v634;
                        v634 = v588 < 8;
                        v635 = v634;
                    } else {
                        v635 = false;
                    }
                    bool v636;
                    v636 = v635 == false;
                    if (v636){
                        assert("The indices should be inside the range of the dimension." && v635);
                    } else {
                    }
                    int v638;
                    v638 = v588 * 4;
                    int v639;
                    v639 = v626 + v638;
                    bool v640;
                    v640 = 0 <= v624;
                    bool v642;
                    if (v640){
                        bool v641;
                        v641 = v624 < 1;
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
                    int v645;
                    v645 = v624 * 32;
                    int v646;
                    v646 = v639 + v645;
                    assert("Tensor range check" && 0 <= v624 && v624 < 1);
                    assert("Tensor range check" && 0 <= v626 && v626 < 4);
                    int v647;
                    v647 = 4 * v624;
                    int v648;
                    v648 = v647 + v626;
                    v616[v648] = v646;
                    v626 += 1 ;
                }
                v624 += 1 ;
            }
            bool v649;
            v649 = 0 <= v591;
            bool v650;
            v650 = v649 && v592;
            bool v651;
            v651 = v650 == false;
            if (v651){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v650);
            } else {
            }
            bool v653;
            v653 = 0 <= v590;
            bool v655;
            if (v653){
                bool v654;
                v654 = v590 < 32;
                v655 = v654;
            } else {
                v655 = false;
            }
            bool v656;
            v656 = v655 == false;
            if (v656){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v655);
            } else {
            }
            bool v658;
            v658 = 0 <= v607;
            bool v659;
            v659 = v658 && v608;
            bool v660;
            v660 = v659 == false;
            if (v660){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v659);
            } else {
            }
            bool v662;
            v662 = 0 <= v606;
            bool v664;
            if (v662){
                bool v663;
                v663 = v606 < 4;
                v664 = v663;
            } else {
                v664 = false;
            }
            bool v665;
            v665 = v664 == false;
            if (v665){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v664);
            } else {
            }
            int v667;
            v667 = v606 * 32;
            int v668;
            v668 = v607 + v591;
            int v669;
            v669 = v667 + v590;
            float v670[4];
            int v671;
            v671 = 0;
            while (while_method_4(v671)){
                int v673;
                v673 = 0;
                while (while_method_7(v673)){
                    assert("Tensor range check" && 0 <= v671 && v671 < 1);
                    assert("Tensor range check" && 0 <= v673 && v673 < 4);
                    int v675;
                    v675 = 4 * v671;
                    int v676;
                    v676 = v675 + v673;
                    int v677;
                    v677 = v616[v676];
                    assert("Tensor range check" && 0 <= v677 && v677 < 32);
                    float v678;
                    v678 = v514[v677];
                    float v679;
                    v679 = v516[v677];
                    bool v680;
                    v680 = isnan(v678);
                    bool v681;
                    v681 = v680 == false;
                    bool v682;
                    v682 = v681 == false;
                    if (v682){
                        assert("The reward numerator must not be a nan." && v681);
                    } else {
                    }
                    bool v684;
                    v684 = isnan(v679);
                    bool v685;
                    v685 = v684 == false;
                    bool v686;
                    v686 = v685 == false;
                    if (v686){
                        assert("The reward count must not be a nan." && v685);
                    } else {
                    }
                    bool v688;
                    v688 = v679 == 0.0f;
                    bool v689;
                    v689 = v688 != true;
                    float v691;
                    if (v689){
                        float v690;
                        v690 = v678 / v679;
                        v691 = v690;
                    } else {
                        v691 = 0.0f;
                    }
                    bool v692;
                    v692 = isnan(v691);
                    bool v693;
                    v693 = v692 == false;
                    bool v694;
                    v694 = v693 == false;
                    if (v694){
                        assert("The reward must not be a nan." && v693);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v671 && v671 < 1);
                    assert("Tensor range check" && 0 <= v673 && v673 < 4);
                    v670[v676] = v691;
                    v673 += 1 ;
                }
                v671 += 1 ;
            }
            float v696;
            v696 = 0.0f;
            int v697;
            v697 = 0;
            while (while_method_4(v697)){
                int v699;
                v699 = 0;
                while (while_method_7(v699)){
                    assert("Tensor range check" && 0 <= v697 && v697 < 1);
                    assert("Tensor range check" && 0 <= v699 && v699 < 4);
                    int v701;
                    v701 = 4 * v697;
                    int v702;
                    v702 = v701 + v699;
                    float v703;
                    v703 = v670[v702];
                    float v704;
                    v704 = v696 + v703;
                    v696 = v704;
                    v699 += 1 ;
                }
                v697 += 1 ;
            }
            auto v705 = cooperative_groups::coalesced_threads();
            int v706;
            v706 = threadIdx.x;
            int v707;
            v707 = v706 / 8;
            auto v708 = cooperative_groups::labeled_partition(v705,v707);
            Closure0 v709{};
            float v710;
            v710 = cooperative_groups::reduce(v708, v696, v709);
            float v711;
            v711 = v710 / 32.0f;
            bool v712[4];
            int v713;
            v713 = 0;
            while (while_method_4(v713)){
                int v715;
                v715 = 0;
                while (while_method_7(v715)){
                    assert("Tensor range check" && 0 <= v713 && v713 < 1);
                    assert("Tensor range check" && 0 <= v715 && v715 < 4);
                    int v717;
                    v717 = 4 * v713;
                    int v718;
                    v718 = v717 + v715;
                    float v719;
                    v719 = v670[v718];
                    bool v720;
                    v720 = v719 >= v711;
                    assert("Tensor range check" && 0 <= v713 && v713 < 1);
                    assert("Tensor range check" && 0 <= v715 && v715 < 4);
                    v712[v718] = v720;
                    v715 += 1 ;
                }
                v713 += 1 ;
            }
            int v721[4];
            int v722;
            v722 = 0;
            while (while_method_4(v722)){
                int v724;
                v724 = 0;
                while (while_method_7(v724)){
                    assert("Tensor range check" && 0 <= v722 && v722 < 1);
                    assert("Tensor range check" && 0 <= v724 && v724 < 4);
                    int v726;
                    v726 = 4 * v722;
                    int v727;
                    v727 = v726 + v724;
                    bool v728;
                    v728 = v712[v727];
                    int v729;
                    if (v728){
                        v729 = 1;
                    } else {
                        v729 = 0;
                    }
                    assert("Tensor range check" && 0 <= v722 && v722 < 1);
                    assert("Tensor range check" && 0 <= v724 && v724 < 4);
                    v721[v727] = v729;
                    v724 += 1 ;
                }
                v722 += 1 ;
            }
            int v730;
            v730 = 0;
            int v731;
            v731 = 0;
            while (while_method_4(v731)){
                int v733;
                v733 = 0;
                while (while_method_7(v733)){
                    assert("Tensor range check" && 0 <= v731 && v731 < 1);
                    assert("Tensor range check" && 0 <= v733 && v733 < 4);
                    int v735;
                    v735 = 4 * v731;
                    int v736;
                    v736 = v735 + v733;
                    int v737;
                    v737 = v721[v736];
                    int v738;
                    v738 = v730 + v737;
                    v730 = v738;
                    v733 += 1 ;
                }
                v731 += 1 ;
            }
            auto v739 = cooperative_groups::coalesced_threads();
            int v740;
            v740 = threadIdx.x;
            int v741;
            v741 = v740 / 8;
            auto v742 = cooperative_groups::labeled_partition(v739,v741);
            Closure1 v743{};
            int v744;
            v744 = cooperative_groups::reduce(v742, v730, v743);
            float v745;
            v745 = (float)v744;
            float v746[4];
            int v747;
            v747 = 0;
            while (while_method_4(v747)){
                int v749;
                v749 = 0;
                while (while_method_7(v749)){
                    assert("Tensor range check" && 0 <= v747 && v747 < 1);
                    assert("Tensor range check" && 0 <= v749 && v749 < 4);
                    int v751;
                    v751 = 4 * v747;
                    int v752;
                    v752 = v751 + v749;
                    float v753;
                    v753 = v615[v752];
                    bool v754;
                    v754 = v712[v752];
                    float v755;
                    if (v754){
                        v755 = v753;
                    } else {
                        v755 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v747 && v747 < 1);
                    assert("Tensor range check" && 0 <= v749 && v749 < 4);
                    v746[v752] = v755;
                    v749 += 1 ;
                }
                v747 += 1 ;
            }
            float v756;
            v756 = 0.0f;
            int v757;
            v757 = 0;
            while (while_method_4(v757)){
                int v759;
                v759 = 0;
                while (while_method_7(v759)){
                    assert("Tensor range check" && 0 <= v757 && v757 < 1);
                    assert("Tensor range check" && 0 <= v759 && v759 < 4);
                    int v761;
                    v761 = 4 * v757;
                    int v762;
                    v762 = v761 + v759;
                    float v763;
                    v763 = v746[v762];
                    float v764;
                    v764 = v756 + v763;
                    v756 = v764;
                    v759 += 1 ;
                }
                v757 += 1 ;
            }
            auto v765 = cooperative_groups::coalesced_threads();
            int v766;
            v766 = threadIdx.x;
            int v767;
            v767 = v766 / 8;
            auto v768 = cooperative_groups::labeled_partition(v765,v767);
            float v769;
            v769 = cooperative_groups::reduce(v768, v756, v709);
            float v770;
            v770 = v769 / v745;
            float v771[4];
            int v772;
            v772 = 0;
            while (while_method_4(v772)){
                int v774;
                v774 = 0;
                while (while_method_7(v774)){
                    assert("Tensor range check" && 0 <= v772 && v772 < 1);
                    assert("Tensor range check" && 0 <= v774 && v774 < 4);
                    int v776;
                    v776 = 4 * v772;
                    int v777;
                    v777 = v776 + v774;
                    float v778;
                    v778 = v615[v777];
                    float v779;
                    v779 = v778 - v770;
                    float v780;
                    v780 = v779 * v779;
                    assert("Tensor range check" && 0 <= v772 && v772 < 1);
                    assert("Tensor range check" && 0 <= v774 && v774 < 4);
                    v771[v777] = v780;
                    v774 += 1 ;
                }
                v772 += 1 ;
            }
            float v781[4];
            int v782;
            v782 = 0;
            while (while_method_4(v782)){
                int v784;
                v784 = 0;
                while (while_method_7(v784)){
                    assert("Tensor range check" && 0 <= v782 && v782 < 1);
                    assert("Tensor range check" && 0 <= v784 && v784 < 4);
                    int v786;
                    v786 = 4 * v782;
                    int v787;
                    v787 = v786 + v784;
                    float v788;
                    v788 = v771[v787];
                    bool v789;
                    v789 = v712[v787];
                    float v790;
                    if (v789){
                        v790 = v788;
                    } else {
                        v790 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v782 && v782 < 1);
                    assert("Tensor range check" && 0 <= v784 && v784 < 4);
                    v781[v787] = v790;
                    v784 += 1 ;
                }
                v782 += 1 ;
            }
            float v791;
            v791 = 0.0f;
            int v792;
            v792 = 0;
            while (while_method_4(v792)){
                int v794;
                v794 = 0;
                while (while_method_7(v794)){
                    assert("Tensor range check" && 0 <= v792 && v792 < 1);
                    assert("Tensor range check" && 0 <= v794 && v794 < 4);
                    int v796;
                    v796 = 4 * v792;
                    int v797;
                    v797 = v796 + v794;
                    float v798;
                    v798 = v781[v797];
                    float v799;
                    v799 = v791 + v798;
                    v791 = v799;
                    v794 += 1 ;
                }
                v792 += 1 ;
            }
            auto v800 = cooperative_groups::coalesced_threads();
            int v801;
            v801 = threadIdx.x;
            int v802;
            v802 = v801 / 8;
            auto v803 = cooperative_groups::labeled_partition(v800,v802);
            float v804;
            v804 = cooperative_groups::reduce(v803, v791, v709);
            float v805;
            v805 = v804 / v745;
            float v806;
            v806 = sqrt(v805);
            bool v807;
            v807 = v745 > 1.0f;
            float v811;
            if (v807){
                float v808;
                v808 = v806 * v745;
                float v809;
                v809 = v745 - 1.0f;
                float v810;
                v810 = v808 / v809;
                v811 = v810;
            } else {
                v811 = 0.0f;
            }
            float v812[4];
            int v813;
            v813 = 0;
            while (while_method_4(v813)){
                int v815;
                v815 = 0;
                while (while_method_7(v815)){
                    assert("Tensor range check" && 0 <= v813 && v813 < 1);
                    assert("Tensor range check" && 0 <= v815 && v815 < 4);
                    int v817;
                    v817 = 4 * v813;
                    int v818;
                    v818 = v817 + v815;
                    float v819;
                    v819 = v615[v818];
                    bool v820;
                    v820 = v712[v818];
                    float v821;
                    v821 = curand_normal(&v505);
                    bool v822;
                    v822 = v811 >= 0.1f;
                    float v823;
                    if (v822){
                        v823 = v811;
                    } else {
                        v823 = 0.1f;
                    }
                    float v824;
                    v824 = v821 * v823;
                    float v825;
                    v825 = v824 + v770;
                    float v826;
                    if (v820){
                        v826 = v819;
                    } else {
                        v826 = v825;
                    }
                    assert("Tensor range check" && 0 <= v813 && v813 < 1);
                    assert("Tensor range check" && 0 <= v815 && v815 < 4);
                    v812[v818] = v826;
                    v815 += 1 ;
                }
                v813 += 1 ;
            }
            assert("Tensor range check" && 0 <= v607 && v607 < 64);
            assert("Tensor range check" && 0 <= v606 && v606 < 4);
            int v827;
            v827 = 0;
            while (while_method_4(v827)){
                assert("Tensor range check" && 0 <= v827 && v827 < 1);
                int v829;
                v829 = 32 * v827;
                int v830;
                v830 = v829 + v614;
                assert("Tensor range check" && 0 <= v827 && v827 < 1);
                int v831;
                v831 = 4 * v827;
                int4* v832;
                v832 = reinterpret_cast<int4*>(v812 + v831);
                int4* v833;
                v833 = reinterpret_cast<int4*>(v508 + v830);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v832) % 16 == 0 && reinterpret_cast<unsigned long long>(v833) % 16 == 0);
                *v833 = *v832;
                v827 += 1 ;
            }
            v601 += 24 ;
        }
        v503.sync() ;
        static float v834[8192];
        int v835;
        v835 = threadIdx.x;
        int v836;
        v836 = blockIdx.x;
        int v837;
        v837 = v836 * 256;
        int v838;
        v838 = v835 + v837;
        int v839;
        v839 = v838 / 32;
        int v840;
        v840 = v839;
        while (while_method_13(v840)){
            bool v842;
            v842 = 0 <= v840;
            bool v843;
            v843 = v842 == false;
            if (v843){
                assert("The index needs to be zero or positive." && v842);
            } else {
            }
            int v845;
            v845 = v840 % 128;
            int v846;
            v846 = v840 / 128;
            bool v847;
            v847 = v846 < 64;
            bool v848;
            v848 = v847 == false;
            if (v848){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v847);
            } else {
            }
            assert("Tensor range check" && 0 <= v846 && v846 < 64);
            assert("Tensor range check" && 0 <= v845 && v845 < 128);
            int v850;
            v850 = 32 * v845;
            int v851;
            v851 = 4096 * v846;
            int v852;
            v852 = v851 + v850;
            float v853;
            v853 = 0.0f;
            int v854;
            v854 = threadIdx.x;
            int v855;
            v855 = v854 % 32;
            int v856;
            v856 = v855;
            while (while_method_5(v856)){
                bool v858;
                v858 = 0 <= v856;
                bool v859;
                v859 = v858 == false;
                if (v859){
                    assert("The index needs to be zero or positive." && v858);
                } else {
                }
                bool v861;
                v861 = v856 < 8;
                bool v862;
                v862 = v861 == false;
                if (v862){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v861);
                } else {
                }
                assert("Tensor range check" && 0 <= v856 && v856 < 8);
                int v864;
                v864 = 4 * v856;
                int v865;
                v865 = v864 + v852;
                float v866[4];
                int4* v867;
                v867 = reinterpret_cast<int4*>(v508 + v865);
                int4* v868;
                v868 = reinterpret_cast<int4*>(v866 + 0);
                assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v867) % 16 == 0 && reinterpret_cast<unsigned long long>(v868) % 16 == 0);
                *v868 = *v867;
                int v869;
                v869 = 0;
                while (while_method_7(v869)){
                    assert("Tensor range check" && 0 <= v869 && v869 < 4);
                    float v871;
                    v871 = v866[v869];
                    float v872;
                    v872 = v871 * v871;
                    float v873;
                    v873 = v853 + v872;
                    v853 = v873;
                    v869 += 1 ;
                }
                v856 += 32 ;
            }
            __syncwarp();
            auto v874 = cooperative_groups::coalesced_threads();
            Closure0 v875{};
            float v876;
            v876 = cooperative_groups::reduce(v874, v853, v875);
            float v877;
            v877 = sqrt(v876);
            assert("Tensor range check" && 0 <= v846 && v846 < 64);
            assert("Tensor range check" && 0 <= v845 && v845 < 128);
            int v878;
            v878 = 128 * v846;
            int v879;
            v879 = v878 + v845;
            v834[v879] = v877;
            v840 += 192 ;
        }
        __syncthreads();
        v503.sync() ;
        float v880;
        v880 = 0.0f;
        int v881;
        v881 = threadIdx.x;
        int v882;
        v882 = blockIdx.x;
        int v883;
        v883 = v882 * 256;
        int v884;
        v884 = v881 + v883;
        int v885;
        v885 = v884;
        while (while_method_14(v885)){
            bool v887;
            v887 = 0 <= v885;
            bool v888;
            v888 = v887 == false;
            if (v888){
                assert("The index needs to be zero or positive." && v887);
            } else {
            }
            int v890;
            v890 = v885 % 32;
            int v891;
            v891 = v885 / 32;
            bool v892;
            v892 = v891 < 64;
            bool v893;
            v893 = v892 == false;
            if (v893){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v892);
            } else {
            }
            assert("Tensor range check" && 0 <= v891 && v891 < 64);
            assert("Tensor range check" && 0 <= v890 && v890 < 32);
            int v895;
            v895 = 4 * v890;
            int v896;
            v896 = 128 * v891;
            int v897;
            v897 = v896 + v895;
            float v898[4];
            int4* v899;
            v899 = reinterpret_cast<int4*>(v834 + v897);
            int4* v900;
            v900 = reinterpret_cast<int4*>(v898 + 0);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v899) % 16 == 0 && reinterpret_cast<unsigned long long>(v900) % 16 == 0);
            *v900 = *v899;
            int v901; float v902;
            Tuple13 tmp91 = Tuple13{0, v880};
            v901 = tmp91.v0; v902 = tmp91.v1;
            while (while_method_7(v901)){
                assert("Tensor range check" && 0 <= v901 && v901 < 4);
                float v904;
                v904 = v898[v901];
                bool v905;
                v905 = v902 >= v904;
                float v906;
                if (v905){
                    v906 = v902;
                } else {
                    v906 = v904;
                }
                v902 = v906;
                v901 += 1 ;
            }
            v880 = v902;
            v885 += 6144 ;
        }
        __syncwarp();
        auto v907 = cooperative_groups::coalesced_threads();
        Closure7 v908{};
        float v909;
        v909 = cooperative_groups::reduce(v907, v880, v908);
        int v910;
        v910 = threadIdx.x;
        int v911;
        v911 = v910 / 32;
        extern __shared__ unsigned char v912[];
        float * v913;
        v913 = reinterpret_cast<float *>(&v912[0ull]);
        assert("Tensor range check" && 0 <= v911 && v911 < 8);
        v913[v911] = v909;
        __syncthreads();
        int v915;
        v915 = threadIdx.x;
        int v916;
        v916 = v915 % 32;
        bool v917;
        v917 = v916 < 8;
        float v919;
        if (v917){
            assert("Tensor range check" && 0 <= v916 && v916 < 8);
            float v918;
            v918 = v913[v916];
            v919 = v918;
        } else {
            v919 = 0.0f;
        }
        __syncthreads();
        auto v920 = cooperative_groups::coalesced_threads();
        float v921;
        v921 = cooperative_groups::reduce(v920, v919, v908);
        int v922;
        v922 = blockIdx.x;
        static float v923[24];
        assert("Tensor range check" && 0 <= v922 && v922 < 24);
        v923[v922] = v921;
        v503.sync() ;
        float v924;
        v924 = 0.0f;
        int v925;
        v925 = threadIdx.x;
        int v926;
        v926 = v925 % 32;
        int v927;
        v927 = v926;
        while (while_method_15(v927)){
            bool v929;
            v929 = 0 <= v927;
            bool v930;
            v930 = v929 == false;
            if (v930){
                assert("The index needs to be zero or positive." && v929);
            } else {
            }
            bool v932;
            v932 = v927 < 24;
            bool v933;
            v933 = v932 == false;
            if (v933){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v932);
            } else {
            }
            assert("Tensor range check" && 0 <= v927 && v927 < 24);
            float v935;
            v935 = v923[v927];
            bool v936;
            v936 = v924 >= v935;
            float v937;
            if (v936){
                v937 = v924;
            } else {
                v937 = v935;
            }
            v924 = v937;
            v927 += 32 ;
        }
        __syncwarp();
        auto v938 = cooperative_groups::coalesced_threads();
        float v939;
        v939 = cooperative_groups::reduce(v938, v924, v908);
        int v940;
        v940 = threadIdx.x;
        int v941;
        v941 = blockIdx.x;
        int v942;
        v942 = v941 * 256;
        int v943;
        v943 = v940 + v942;
        bool v944;
        v944 = v943 == 0;
        if (v944){
            cuda::counting_semaphore<cuda::thread_scope_system, 1> & v945 = console_lock;
            auto v946 = cooperative_groups::coalesced_threads();
            v945.acquire();
            printf("{%s = %f}\n","max_norm", v939);
            v945.release();
            v946.sync() ;
        } else {
        }
        __syncwarp();
        extern __shared__ unsigned char v949[];
        float * v950;
        v950 = reinterpret_cast<float *>(&v949[0ull]);
        int v952;
        v952 = blockIdx.x;
        int v953;
        v953 = v952;
        while (while_method_8(v953)){
            bool v955;
            v955 = 0 <= v953;
            bool v956;
            v956 = v955 == false;
            if (v956){
                assert("The index needs to be zero or positive." && v955);
            } else {
            }
            int v958;
            v958 = v953 % 1;
            bool v959;
            v959 = v953 < 16;
            bool v960;
            v960 = v959 == false;
            if (v960){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v959);
            } else {
            }
            assert("Tensor range check" && 0 <= v953 && v953 < 16);
            assert("Tensor range check" && 0 <= v958 && v958 < 1);
            int v962;
            v962 = 32 * v958;
            int v963;
            v963 = 16384 * v953;
            int v964;
            v964 = v963 + v962;
            int v965;
            v965 = 262144 * v958;
            int v966;
            v966 = 512 * v953;
            int v967;
            v967 = v966 + v965;
            int v968;
            v968 = threadIdx.x;
            int v969;
            v969 = v968;
            while (while_method_12(v969)){
                bool v971;
                v971 = 0 <= v969;
                bool v972;
                v972 = v971 == false;
                if (v972){
                    assert("The index needs to be zero or positive." && v971);
                } else {
                }
                int v974;
                v974 = v969 % 32;
                int v975;
                v975 = v969 / 32;
                bool v976;
                v976 = v975 < 512;
                bool v977;
                v977 = v976 == false;
                if (v977){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v976);
                } else {
                }
                assert("Tensor range check" && 0 <= v975 && v975 < 512);
                assert("Tensor range check" && 0 <= v974 && v974 < 32);
                int v979;
                v979 = v974 + v964;
                int v980;
                v980 = 32 * v975;
                int v981;
                v981 = v980 + v979;
                float v982;
                v982 = v508[v981];
                assert("Tensor range check" && 0 <= v975 && v975 < 512);
                assert("Tensor range check" && 0 <= v974 && v974 < 32);
                int v983;
                v983 = 33 * v975;
                int v984;
                v984 = v983 + v974;
                v950[v984] = v982;
                v969 += 256 ;
            }
            __syncthreads();
            int v985;
            v985 = threadIdx.x;
            int v986;
            v986 = v985;
            while (while_method_12(v986)){
                bool v988;
                v988 = 0 <= v986;
                bool v989;
                v989 = v988 == false;
                if (v989){
                    assert("The index needs to be zero or positive." && v988);
                } else {
                }
                int v991;
                v991 = v986 % 512;
                int v992;
                v992 = v986 / 512;
                bool v993;
                v993 = v992 < 32;
                bool v994;
                v994 = v993 == false;
                if (v994){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v993);
                } else {
                }
                assert("Tensor range check" && 0 <= v992 && v992 < 32);
                assert("Tensor range check" && 0 <= v991 && v991 < 512);
                int v996;
                v996 = 33 * v991;
                int v997;
                v997 = v992 + v996;
                float v998;
                v998 = v950[v997];
                assert("Tensor range check" && 0 <= v992 && v992 < 32);
                assert("Tensor range check" && 0 <= v991 && v991 < 512);
                int v999;
                v999 = v991 + v967;
                int v1000;
                v1000 = 8192 * v992;
                int v1001;
                v1001 = v1000 + v999;
                v506[v1001] = v998;
                v986 += 256 ;
            }
            __syncthreads();
            v953 += 24 ;
        }
        v503.sync() ;
        int v1002;
        v1002 = threadIdx.x;
        int v1003;
        v1003 = blockIdx.x;
        int v1004;
        v1004 = v1003 * 256;
        int v1005;
        v1005 = v1002 + v1004;
        int v1006;
        v1006 = v1005;
        while (while_method_5(v1006)){
            bool v1008;
            v1008 = 0 <= v1006;
            bool v1009;
            v1009 = v1008 == false;
            if (v1009){
                assert("The index needs to be zero or positive." && v1008);
            } else {
            }
            bool v1011;
            v1011 = v1006 < 8;
            bool v1012;
            v1012 = v1011 == false;
            if (v1012){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1011);
            } else {
            }
            assert("Tensor range check" && 0 <= v1006 && v1006 < 8);
            int v1014;
            v1014 = 4 * v1006;
            assert("Tensor range check" && 0 <= v1006 && v1006 < 8);
            float v1015[4];
            float v1016[4];
            float v1017[4];
            float v1018[4];
            int4* v1019;
            v1019 = reinterpret_cast<int4*>(v514 + v1014);
            int4* v1020;
            v1020 = reinterpret_cast<int4*>(v1015 + 0);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1019) % 16 == 0 && reinterpret_cast<unsigned long long>(v1020) % 16 == 0);
            *v1020 = *v1019;
            int4* v1021;
            v1021 = reinterpret_cast<int4*>(v516 + v1014);
            int4* v1022;
            v1022 = reinterpret_cast<int4*>(v1016 + 0);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1021) % 16 == 0 && reinterpret_cast<unsigned long long>(v1022) % 16 == 0);
            *v1022 = *v1021;
            // Pushing the loop unrolling to: 0
            int v1023;
            v1023 = 0;
            #pragma unroll
            while (while_method_7(v1023)){
                assert("Tensor range check" && 0 <= v1023 && v1023 < 4);
                float v1025;
                v1025 = v1015[v1023];
                float v1026;
                v1026 = v1016[v1023];
                assert("Tensor range check" && 0 <= v1023 && v1023 < 4);
                v1017[v1023] = 0.0f;
                v1018[v1023] = 0.0f;
                v1023 += 1 ;
            }
            // Poping the loop unrolling to: 0
            int4* v1027;
            v1027 = reinterpret_cast<int4*>(v1017 + 0);
            int4* v1028;
            v1028 = reinterpret_cast<int4*>(v514 + v1014);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1027) % 16 == 0 && reinterpret_cast<unsigned long long>(v1028) % 16 == 0);
            *v1028 = *v1027;
            int4* v1029;
            v1029 = reinterpret_cast<int4*>(v1018 + 0);
            int4* v1030;
            v1030 = reinterpret_cast<int4*>(v516 + v1014);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1029) % 16 == 0 && reinterpret_cast<unsigned long long>(v1030) % 16 == 0);
            *v1030 = *v1029;
            v1006 += 6144 ;
        }
        v503.sync() ;
        v27 += 1 ;
    }
    cooperative_groups::grid_group & v1031 = v26.v1;
    cooperative_groups::grid_group & v1032 = v1031;
    int v1033;
    v1033 = threadIdx.x;
    int v1034;
    v1034 = blockIdx.x;
    int v1035;
    v1035 = v1034 * 256;
    int v1036;
    v1036 = v1033 + v1035;
    int v1037;
    v1037 = v1036;
    while (while_method_14(v1037)){
        bool v1039;
        v1039 = 0 <= v1037;
        bool v1040;
        v1040 = v1039 == false;
        if (v1040){
            assert("The index needs to be zero or positive." && v1039);
        } else {
        }
        int v1042;
        v1042 = v1037 % 64;
        int v1043;
        v1043 = v1037 / 64;
        bool v1044;
        v1044 = v1043 < 32;
        bool v1045;
        v1045 = v1044 == false;
        if (v1045){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1044);
        } else {
        }
        assert("Tensor range check" && 0 <= v1043 && v1043 < 32);
        assert("Tensor range check" && 0 <= v1042 && v1042 < 64);
        int v1047;
        v1047 = 4 * v1042;
        int v1048;
        v1048 = 256 * v1043;
        int v1049;
        v1049 = v1048 + v1047;
        assert("Tensor range check" && 0 <= v1043 && v1043 < 32);
        assert("Tensor range check" && 0 <= v1042 && v1042 < 64);
        float v1050[4];
        float v1051[4];
        float v1052[4];
        int4* v1053;
        v1053 = reinterpret_cast<int4*>(v3 + v1049);
        int4* v1054;
        v1054 = reinterpret_cast<int4*>(v1050 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1053) % 16 == 0 && reinterpret_cast<unsigned long long>(v1054) % 16 == 0);
        *v1054 = *v1053;
        int4* v1055;
        v1055 = reinterpret_cast<int4*>(v4 + v1049);
        int4* v1056;
        v1056 = reinterpret_cast<int4*>(v1051 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1055) % 16 == 0 && reinterpret_cast<unsigned long long>(v1056) % 16 == 0);
        *v1056 = *v1055;
        // Pushing the loop unrolling to: 0
        int v1057;
        v1057 = 0;
        #pragma unroll
        while (while_method_7(v1057)){
            assert("Tensor range check" && 0 <= v1057 && v1057 < 4);
            float v1059;
            v1059 = v1050[v1057];
            float v1060;
            v1060 = v1051[v1057];
            bool v1061;
            v1061 = v1060 == 0.0f;
            bool v1062;
            v1062 = v1061 != true;
            float v1064;
            if (v1062){
                float v1063;
                v1063 = v1059 / v1060;
                v1064 = v1063;
            } else {
                v1064 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v1057 && v1057 < 4);
            v1052[v1057] = v1064;
            v1057 += 1 ;
        }
        // Poping the loop unrolling to: 0
        int4* v1065;
        v1065 = reinterpret_cast<int4*>(v1052 + 0);
        int4* v1066;
        v1066 = reinterpret_cast<int4*>(v5 + v1049);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1065) % 16 == 0 && reinterpret_cast<unsigned long long>(v1066) % 16 == 0);
        *v1066 = *v1065;
        v1037 += 6144 ;
    }
    v1032.sync() ;
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
        v2 = cp.empty(1048848,dtype=cp.uint8)
        v4 = v1[0:0+4*786432].view(cp.float32)
        del v4
        v6 = v2[0:0+4*262144].view(cp.float32)
        v8 = v0[0:0+4*262144].view(cp.float32)
        del v8
        v10 = v1[4718592:4718592+4*12582912].view(cp.float32)
        del v10
        v12 = v2[1048576:1048576+4*1].view(cp.int32)
        v14 = v2[1048592:1048592+4*32].view(cp.float32)
        v16 = v2[1048720:1048720+4*32].view(cp.float32)
        v18 = v1[55050240:55050240+8*393216].view(cp.float64)
        v20 = v1[58195968:58195968+8*393216].view(cp.float64)
        v21 = cp.random.normal(0.0,0.1,262144,dtype=cp.float32) # type: ignore
        cp.copyto(v6[0:0+262144],v21[0:0+262144])
        del v6, v21
        v12[:] = 0
        del v12
        v14[:] = 0
        del v14
        v16[:] = 0
        del v16
        v18[:] = 0
        del v18
        v20[:] = 0
        del v20
        v23 = static_array(2)
        v25 = US2_0()
        v23[0] = v25
        del v25
        v27 = US2_1()
        v23[1] = v27
        del v27
        v29 = static_array_list(32)
        v30 = 63
        v31 = US3_0()
        v32 = US7_0()
        return method115(v30, v31, v29, v23, v32, v2, v1, v0)
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
