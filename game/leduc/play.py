kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
#include <curand_kernel.h>
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
struct Union3;
struct Union6;
struct Union5;
struct Union4;
struct Union7;
struct Union8;
struct Tuple0;
__device__ Union3 f_7(unsigned char * v0);
__device__ int f_8(unsigned char * v0);
struct Tuple1;
__device__ Tuple1 f_10(unsigned char * v0);
struct Tuple2;
__device__ int f_12(unsigned char * v0);
__device__ Tuple2 f_11(unsigned char * v0);
__device__ Union5 f_9(unsigned char * v0);
__device__ int f_13(unsigned char * v0);
struct Tuple3;
__device__ int f_16(unsigned char * v0);
__device__ Tuple3 f_15(unsigned char * v0);
struct Tuple4;
__device__ Tuple4 f_17(unsigned char * v0);
struct Tuple5;
__device__ Tuple5 f_18(unsigned char * v0);
__device__ Union7 f_14(unsigned char * v0);
__device__ int f_19(unsigned char * v0);
__device__ Tuple0 f_6(unsigned char * v0);
struct Tuple6;
struct Union9;
struct Union10;
struct Tuple7;
struct Union11;
struct Union12;
__device__ void method_22(float * v0, int v1, float * v2, int v3, float * v4, int v5);
struct Tuple8;
struct Tuple9;
struct Tuple10;
struct Tuple11;
__device__ void method_23(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5);
__device__ unsigned int loop_25(unsigned int v0, curandStatePhilox4_32_10_t & v1);
__device__ int int_range_24(int v0, int v1, curandStatePhilox4_32_10_t & v2);
__device__ Tuple7 noinline_run_21(unsigned char * v0, unsigned char * v1, Union9 v2);
__device__ void method_26(Union1 v0);
struct Union13;
__device__ int tag_28(Union3 v0);
__device__ bool is_pair_29(int v0, int v1);
__device__ Tuple6 order_30(int v0, int v1);
__device__ Union13 compare_hands_27(Union6 v0, bool v1, static_array<Union3,2l> v2, int v3, static_array<int,2l> v4, int v5);
__device__ void play_loop_20(unsigned char * v0, unsigned long long v1, unsigned char * v2, unsigned long long v3, static_array_list<Union3,6l> & v4, Union4 & v5, static_array_list<Union7,32l> & v6, static_array<Union2,2l> & v7, Union8 & v8, Union5 v9);
__device__ void f_32(unsigned char * v0, int v1);
__device__ void f_34(unsigned char * v0);
__device__ void f_33(unsigned char * v0, Union3 v1);
__device__ void f_35(unsigned char * v0, int v1);
__device__ void f_37(unsigned char * v0, Union6 v1, bool v2, static_array<Union3,2l> v3, int v4, static_array<int,2l> v5, int v6);
__device__ void f_39(unsigned char * v0, int v1);
__device__ void f_38(unsigned char * v0, Union6 v1, bool v2, static_array<Union3,2l> v3, int v4, static_array<int,2l> v5, int v6, Union1 v7);
__device__ void f_36(unsigned char * v0, Union5 v1);
__device__ void f_40(unsigned char * v0, int v1);
__device__ void f_43(unsigned char * v0, int v1);
__device__ void f_42(unsigned char * v0, int v1, Union1 v2);
__device__ void f_44(unsigned char * v0, int v1, Union3 v2);
__device__ void f_45(unsigned char * v0, static_array<Union3,2l> v1, int v2, int v3);
__device__ void f_41(unsigned char * v0, Union7 v1);
__device__ void f_46(unsigned char * v0, Union2 v1);
__device__ void f_47(unsigned char * v0, int v1);
__device__ void f_31(unsigned char * v0, static_array_list<Union3,6l> v1, Union4 v2, static_array_list<Union7,32l> v3, static_array<Union2,2l> v4, Union8 v5);
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
struct Union3_0 { // Jack
};
struct Union3_1 { // King
};
struct Union3_2 { // Queen
};
struct Union3 {
    union {
        Union3_0 case0; // Jack
        Union3_1 case1; // King
        Union3_2 case2; // Queen
    };
    unsigned char tag{255};
    __device__ Union3() {}
    __device__ Union3(Union3_0 t) : tag(0), case0(t) {} // Jack
    __device__ Union3(Union3_1 t) : tag(1), case1(t) {} // King
    __device__ Union3(Union3_2 t) : tag(2), case2(t) {} // Queen
    __device__ Union3(Union3 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(x.case0); break; // Jack
            case 1: new (&this->case1) Union3_1(x.case1); break; // King
            case 2: new (&this->case2) Union3_2(x.case2); break; // Queen
        }
    }
    __device__ Union3(Union3 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(std::move(x.case0)); break; // Jack
            case 1: new (&this->case1) Union3_1(std::move(x.case1)); break; // King
            case 2: new (&this->case2) Union3_2(std::move(x.case2)); break; // Queen
        }
    }
    __device__ Union3 & operator=(Union3 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Jack
                case 1: this->case1 = x.case1; break; // King
                case 2: this->case2 = x.case2; break; // Queen
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
                case 0: this->case0 = std::move(x.case0); break; // Jack
                case 1: this->case1 = std::move(x.case1); break; // King
                case 2: this->case2 = std::move(x.case2); break; // Queen
            }
        } else {
            this->~Union3();
            new (this) Union3{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union3() {
        switch(this->tag){
            case 0: this->case0.~Union3_0(); break; // Jack
            case 1: this->case1.~Union3_1(); break; // King
            case 2: this->case2.~Union3_2(); break; // Queen
        }
        this->tag = 255;
    }
};
struct Union6_0 { // None
};
struct Union6_1 { // Some
    Union3 v0;
    __device__ Union6_1(Union3 t0) : v0(t0) {}
    __device__ Union6_1() = delete;
};
struct Union6 {
    union {
        Union6_0 case0; // None
        Union6_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union6() {}
    __device__ Union6(Union6_0 t) : tag(0), case0(t) {} // None
    __device__ Union6(Union6_1 t) : tag(1), case1(t) {} // Some
    __device__ Union6(Union6 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(x.case0); break; // None
            case 1: new (&this->case1) Union6_1(x.case1); break; // Some
        }
    }
    __device__ Union6(Union6 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union6_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union6 & operator=(Union6 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union6();
            new (this) Union6{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union6() {
        switch(this->tag){
            case 0: this->case0.~Union6_0(); break; // None
            case 1: this->case1.~Union6_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union5_0 { // ChanceCommunityCard
    Union6 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union5_0(Union6 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union5_0() = delete;
};
struct Union5_1 { // ChanceInit
};
struct Union5_2 { // Round
    Union6 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union5_2(Union6 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union5_2() = delete;
};
struct Union5_3 { // RoundWithAction
    Union6 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    Union1 v6;
    int v3;
    int v5;
    bool v1;
    __device__ Union5_3(Union6 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union5_3() = delete;
};
struct Union5_4 { // TerminalCall
    Union6 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union5_4(Union6 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union5_4() = delete;
};
struct Union5_5 { // TerminalFold
    Union6 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union5_5(Union6 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union5_5() = delete;
};
struct Union5 {
    union {
        Union5_0 case0; // ChanceCommunityCard
        Union5_1 case1; // ChanceInit
        Union5_2 case2; // Round
        Union5_3 case3; // RoundWithAction
        Union5_4 case4; // TerminalCall
        Union5_5 case5; // TerminalFold
    };
    unsigned char tag{255};
    __device__ Union5() {}
    __device__ Union5(Union5_0 t) : tag(0), case0(t) {} // ChanceCommunityCard
    __device__ Union5(Union5_1 t) : tag(1), case1(t) {} // ChanceInit
    __device__ Union5(Union5_2 t) : tag(2), case2(t) {} // Round
    __device__ Union5(Union5_3 t) : tag(3), case3(t) {} // RoundWithAction
    __device__ Union5(Union5_4 t) : tag(4), case4(t) {} // TerminalCall
    __device__ Union5(Union5_5 t) : tag(5), case5(t) {} // TerminalFold
    __device__ Union5(Union5 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(x.case0); break; // ChanceCommunityCard
            case 1: new (&this->case1) Union5_1(x.case1); break; // ChanceInit
            case 2: new (&this->case2) Union5_2(x.case2); break; // Round
            case 3: new (&this->case3) Union5_3(x.case3); break; // RoundWithAction
            case 4: new (&this->case4) Union5_4(x.case4); break; // TerminalCall
            case 5: new (&this->case5) Union5_5(x.case5); break; // TerminalFold
        }
    }
    __device__ Union5(Union5 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(std::move(x.case0)); break; // ChanceCommunityCard
            case 1: new (&this->case1) Union5_1(std::move(x.case1)); break; // ChanceInit
            case 2: new (&this->case2) Union5_2(std::move(x.case2)); break; // Round
            case 3: new (&this->case3) Union5_3(std::move(x.case3)); break; // RoundWithAction
            case 4: new (&this->case4) Union5_4(std::move(x.case4)); break; // TerminalCall
            case 5: new (&this->case5) Union5_5(std::move(x.case5)); break; // TerminalFold
        }
    }
    __device__ Union5 & operator=(Union5 & x) {
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
            this->~Union5();
            new (this) Union5{x};
        }
        return *this;
    }
    __device__ Union5 & operator=(Union5 && x) {
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
            this->~Union5();
            new (this) Union5{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union5() {
        switch(this->tag){
            case 0: this->case0.~Union5_0(); break; // ChanceCommunityCard
            case 1: this->case1.~Union5_1(); break; // ChanceInit
            case 2: this->case2.~Union5_2(); break; // Round
            case 3: this->case3.~Union5_3(); break; // RoundWithAction
            case 4: this->case4.~Union5_4(); break; // TerminalCall
            case 5: this->case5.~Union5_5(); break; // TerminalFold
        }
        this->tag = 255;
    }
};
struct Union4_0 { // None
};
struct Union4_1 { // Some
    Union5 v0;
    __device__ Union4_1(Union5 t0) : v0(t0) {}
    __device__ Union4_1() = delete;
};
struct Union4 {
    union {
        Union4_0 case0; // None
        Union4_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union4() {}
    __device__ Union4(Union4_0 t) : tag(0), case0(t) {} // None
    __device__ Union4(Union4_1 t) : tag(1), case1(t) {} // Some
    __device__ Union4(Union4 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(x.case0); break; // None
            case 1: new (&this->case1) Union4_1(x.case1); break; // Some
        }
    }
    __device__ Union4(Union4 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union4_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union4 & operator=(Union4 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union4();
            new (this) Union4{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union4() {
        switch(this->tag){
            case 0: this->case0.~Union4_0(); break; // None
            case 1: this->case1.~Union4_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union7_0 { // CommunityCardIs
    Union3 v0;
    __device__ Union7_0(Union3 t0) : v0(t0) {}
    __device__ Union7_0() = delete;
};
struct Union7_1 { // PlayerAction
    Union1 v1;
    int v0;
    __device__ Union7_1(int t0, Union1 t1) : v0(t0), v1(t1) {}
    __device__ Union7_1() = delete;
};
struct Union7_2 { // PlayerGotCard
    Union3 v1;
    int v0;
    __device__ Union7_2(int t0, Union3 t1) : v0(t0), v1(t1) {}
    __device__ Union7_2() = delete;
};
struct Union7_3 { // Showdown
    static_array<Union3,2l> v0;
    int v1;
    int v2;
    __device__ Union7_3(static_array<Union3,2l> t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
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
    Union6 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union8_1(Union6 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union8_1() = delete;
};
struct Union8_2 { // WaitingForActionFromPlayerId
    Union6 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union8_2(Union6 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
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
    static_array_list<Union3,6l> v0;
    Union4 v1;
    static_array_list<Union7,32l> v2;
    static_array<Union2,2l> v3;
    Union8 v4;
    __device__ Tuple0() = default;
    __device__ Tuple0(static_array_list<Union3,6l> t0, Union4 t1, static_array_list<Union7,32l> t2, static_array<Union2,2l> t3, Union8 t4) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4) {}
};
struct Tuple1 {
    Union6 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(Union6 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Tuple2 {
    Union6 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    Union1 v6;
    int v3;
    int v5;
    bool v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(Union6 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
};
struct Tuple3 {
    Union1 v1;
    int v0;
    __device__ Tuple3() = default;
    __device__ Tuple3(int t0, Union1 t1) : v0(t0), v1(t1) {}
};
struct Tuple4 {
    Union3 v1;
    int v0;
    __device__ Tuple4() = default;
    __device__ Tuple4(int t0, Union3 t1) : v0(t0), v1(t1) {}
};
struct Tuple5 {
    static_array<Union3,2l> v0;
    int v1;
    int v2;
    __device__ Tuple5() = default;
    __device__ Tuple5(static_array<Union3,2l> t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple6 {
    int v0;
    int v1;
    __device__ Tuple6() = default;
    __device__ Tuple6(int t0, int t1) : v0(t0), v1(t1) {}
};
struct Union9_0 { // None
};
struct Union9_1 { // Some
    Union6 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    static_array_list<Union7,32l> v6;
    int v3;
    int v5;
    bool v1;
    __device__ Union9_1(Union6 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5, static_array_list<Union7,32l> t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
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
struct Union10_0 { // AA_Call
};
struct Union10_1 { // AA_Fold
};
struct Union10_2 { // AA_Raise
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
struct Tuple7 {
    Union10 v0;
    float * v1;
    int v2;
    int v3;
    int v4;
    float v5;
    __device__ Tuple7() = default;
    __device__ Tuple7(Union10 t0, float * t1, int t2, int t3, int t4, float t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Union11_0 { // C1of2
    Union1 v0;
    __device__ Union11_0(Union1 t0) : v0(t0) {}
    __device__ Union11_0() = delete;
};
struct Union11_1 { // C2of2
    Union3 v0;
    __device__ Union11_1(Union3 t0) : v0(t0) {}
    __device__ Union11_1() = delete;
};
struct Union11 {
    union {
        Union11_0 case0; // C1of2
        Union11_1 case1; // C2of2
    };
    unsigned char tag{255};
    __device__ Union11() {}
    __device__ Union11(Union11_0 t) : tag(0), case0(t) {} // C1of2
    __device__ Union11(Union11_1 t) : tag(1), case1(t) {} // C2of2
    __device__ Union11(Union11 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union11_0(x.case0); break; // C1of2
            case 1: new (&this->case1) Union11_1(x.case1); break; // C2of2
        }
    }
    __device__ Union11(Union11 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union11_0(std::move(x.case0)); break; // C1of2
            case 1: new (&this->case1) Union11_1(std::move(x.case1)); break; // C2of2
        }
    }
    __device__ Union11 & operator=(Union11 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // C1of2
                case 1: this->case1 = x.case1; break; // C2of2
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
                case 0: this->case0 = std::move(x.case0); break; // C1of2
                case 1: this->case1 = std::move(x.case1); break; // C2of2
            }
        } else {
            this->~Union11();
            new (this) Union11{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union11() {
        switch(this->tag){
            case 0: this->case0.~Union11_0(); break; // C1of2
            case 1: this->case1.~Union11_1(); break; // C2of2
        }
        this->tag = 255;
    }
};
struct Union12_0 { // None
};
struct Union12_1 { // Some
    Union11 v0;
    __device__ Union12_1(Union11 t0) : v0(t0) {}
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
__device__ inline bool while_method_1(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ Union3 f_7(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    switch (v1) {
        case 0: {
            f_3(v2);
            return Union3{Union3_0{}};
            break;
        }
        case 1: {
            f_3(v2);
            return Union3{Union3_1{}};
            break;
        }
        case 2: {
            f_3(v2);
            return Union3{Union3_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ int f_8(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+28ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ Tuple1 f_10(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    Union6 v8;
    switch (v1) {
        case 0: {
            f_3(v2);
            v8 = Union6{Union6_0{}};
            break;
        }
        case 1: {
            Union3 v6;
            v6 = f_7(v2);
            v8 = Union6{Union6_1{v6}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    bool * v9;
    v9 = (bool *)(v0+8ull);
    bool v11;
    v11 = v9[0l];
    static_array<Union3,2l> v12;
    int v14;
    v14 = 0l;
    while (while_method_0(v14)){
        unsigned long long v16;
        v16 = (unsigned long long)v14;
        unsigned long long v17;
        v17 = v16 * 4ull;
        unsigned long long v18;
        v18 = 12ull + v17;
        unsigned char * v19;
        v19 = (unsigned char *)(v0+v18);
        Union3 v21;
        v21 = f_7(v19);
        v12[v14] = v21;
        v14 += 1l ;
    }
    int * v22;
    v22 = (int *)(v0+20ull);
    int v24;
    v24 = v22[0l];
    static_array<int,2l> v25;
    int v27;
    v27 = 0l;
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
        v27 += 1l ;
    }
    int * v35;
    v35 = (int *)(v0+32ull);
    int v37;
    v37 = v35[0l];
    return Tuple1{v8, v11, v12, v24, v25, v37};
}
__device__ int f_12(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+36ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ Tuple2 f_11(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    Union6 v8;
    switch (v1) {
        case 0: {
            f_3(v2);
            v8 = Union6{Union6_0{}};
            break;
        }
        case 1: {
            Union3 v6;
            v6 = f_7(v2);
            v8 = Union6{Union6_1{v6}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    bool * v9;
    v9 = (bool *)(v0+8ull);
    bool v11;
    v11 = v9[0l];
    static_array<Union3,2l> v12;
    int v14;
    v14 = 0l;
    while (while_method_0(v14)){
        unsigned long long v16;
        v16 = (unsigned long long)v14;
        unsigned long long v17;
        v17 = v16 * 4ull;
        unsigned long long v18;
        v18 = 12ull + v17;
        unsigned char * v19;
        v19 = (unsigned char *)(v0+v18);
        Union3 v21;
        v21 = f_7(v19);
        v12[v14] = v21;
        v14 += 1l ;
    }
    int * v22;
    v22 = (int *)(v0+20ull);
    int v24;
    v24 = v22[0l];
    static_array<int,2l> v25;
    int v27;
    v27 = 0l;
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
        v27 += 1l ;
    }
    int * v35;
    v35 = (int *)(v0+32ull);
    int v37;
    v37 = v35[0l];
    int v38;
    v38 = f_12(v0);
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
            asm("exit;");
        }
    }
    return Tuple2{v8, v11, v12, v24, v25, v37, v45};
}
__device__ Union5 f_9(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            Union6 v5; bool v6; static_array<Union3,2l> v7; int v8; static_array<int,2l> v9; int v10;
            Tuple1 tmp0 = f_10(v2);
            v5 = tmp0.v0; v6 = tmp0.v1; v7 = tmp0.v2; v8 = tmp0.v3; v9 = tmp0.v4; v10 = tmp0.v5;
            return Union5{Union5_0{v5, v6, v7, v8, v9, v10}};
            break;
        }
        case 1: {
            f_3(v2);
            return Union5{Union5_1{}};
            break;
        }
        case 2: {
            Union6 v13; bool v14; static_array<Union3,2l> v15; int v16; static_array<int,2l> v17; int v18;
            Tuple1 tmp1 = f_10(v2);
            v13 = tmp1.v0; v14 = tmp1.v1; v15 = tmp1.v2; v16 = tmp1.v3; v17 = tmp1.v4; v18 = tmp1.v5;
            return Union5{Union5_2{v13, v14, v15, v16, v17, v18}};
            break;
        }
        case 3: {
            Union6 v20; bool v21; static_array<Union3,2l> v22; int v23; static_array<int,2l> v24; int v25; Union1 v26;
            Tuple2 tmp2 = f_11(v2);
            v20 = tmp2.v0; v21 = tmp2.v1; v22 = tmp2.v2; v23 = tmp2.v3; v24 = tmp2.v4; v25 = tmp2.v5; v26 = tmp2.v6;
            return Union5{Union5_3{v20, v21, v22, v23, v24, v25, v26}};
            break;
        }
        case 4: {
            Union6 v28; bool v29; static_array<Union3,2l> v30; int v31; static_array<int,2l> v32; int v33;
            Tuple1 tmp3 = f_10(v2);
            v28 = tmp3.v0; v29 = tmp3.v1; v30 = tmp3.v2; v31 = tmp3.v3; v32 = tmp3.v4; v33 = tmp3.v5;
            return Union5{Union5_4{v28, v29, v30, v31, v32, v33}};
            break;
        }
        case 5: {
            Union6 v35; bool v36; static_array<Union3,2l> v37; int v38; static_array<int,2l> v39; int v40;
            Tuple1 tmp4 = f_10(v2);
            v35 = tmp4.v0; v36 = tmp4.v1; v37 = tmp4.v2; v38 = tmp4.v3; v39 = tmp4.v4; v40 = tmp4.v5;
            return Union5{Union5_5{v35, v36, v37, v38, v39, v40}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ int f_13(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+96ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ int f_16(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+4ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ Tuple3 f_15(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    int v4;
    v4 = f_16(v0);
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
            asm("exit;");
        }
    }
    return Tuple3{v3, v11};
}
__device__ Tuple4 f_17(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    int v4;
    v4 = f_16(v0);
    unsigned char * v5;
    v5 = (unsigned char *)(v0+8ull);
    Union3 v11;
    switch (v4) {
        case 0: {
            f_3(v5);
            v11 = Union3{Union3_0{}};
            break;
        }
        case 1: {
            f_3(v5);
            v11 = Union3{Union3_1{}};
            break;
        }
        case 2: {
            f_3(v5);
            v11 = Union3{Union3_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple4{v3, v11};
}
__device__ Tuple5 f_18(unsigned char * v0){
    static_array<Union3,2l> v1;
    int v3;
    v3 = 0l;
    while (while_method_0(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned long long v6;
        v6 = v5 * 4ull;
        unsigned char * v7;
        v7 = (unsigned char *)(v0+v6);
        Union3 v9;
        v9 = f_7(v7);
        v1[v3] = v9;
        v3 += 1l ;
    }
    int * v10;
    v10 = (int *)(v0+8ull);
    int v12;
    v12 = v10[0l];
    int * v13;
    v13 = (int *)(v0+12ull);
    int v15;
    v15 = v13[0l];
    return Tuple5{v1, v12, v15};
}
__device__ Union7 f_14(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            Union3 v5;
            v5 = f_7(v2);
            return Union7{Union7_0{v5}};
            break;
        }
        case 1: {
            int v7; Union1 v8;
            Tuple3 tmp5 = f_15(v2);
            v7 = tmp5.v0; v8 = tmp5.v1;
            return Union7{Union7_1{v7, v8}};
            break;
        }
        case 2: {
            int v10; Union3 v11;
            Tuple4 tmp6 = f_17(v2);
            v10 = tmp6.v0; v11 = tmp6.v1;
            return Union7{Union7_2{v10, v11}};
            break;
        }
        case 3: {
            static_array<Union3,2l> v13; int v14; int v15;
            Tuple5 tmp7 = f_18(v2);
            v13 = tmp7.v0; v14 = tmp7.v1; v15 = tmp7.v2;
            return Union7{Union7_3{v13, v14, v15}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ int f_19(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+1144ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ Tuple0 f_6(unsigned char * v0){
    static_array_list<Union3,6l> v1;
    v1 = static_array_list<Union3,6l>{};
    int v3;
    v3 = f_1(v0);
    v1.unsafe_set_length(v3);
    int v4;
    v4 = v1.length;
    int v5;
    v5 = 0l;
    while (while_method_1(v4, v5)){
        unsigned long long v7;
        v7 = (unsigned long long)v5;
        unsigned long long v8;
        v8 = v7 * 4ull;
        unsigned long long v9;
        v9 = 4ull + v8;
        unsigned char * v10;
        v10 = (unsigned char *)(v0+v9);
        Union3 v12;
        v12 = f_7(v10);
        v1[v5] = v12;
        v5 += 1l ;
    }
    int v13;
    v13 = f_8(v0);
    unsigned char * v14;
    v14 = (unsigned char *)(v0+32ull);
    Union4 v20;
    switch (v13) {
        case 0: {
            f_3(v14);
            v20 = Union4{Union4_0{}};
            break;
        }
        case 1: {
            Union5 v18;
            v18 = f_9(v14);
            v20 = Union4{Union4_1{v18}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    static_array_list<Union7,32l> v21;
    v21 = static_array_list<Union7,32l>{};
    int v23;
    v23 = f_13(v0);
    v21.unsafe_set_length(v23);
    int v24;
    v24 = v21.length;
    int v25;
    v25 = 0l;
    while (while_method_1(v24, v25)){
        unsigned long long v27;
        v27 = (unsigned long long)v25;
        unsigned long long v28;
        v28 = v27 * 32ull;
        unsigned long long v29;
        v29 = 112ull + v28;
        unsigned char * v30;
        v30 = (unsigned char *)(v0+v29);
        Union7 v32;
        v32 = f_14(v30);
        v21[v25] = v32;
        v25 += 1l ;
    }
    static_array<Union2,2l> v33;
    int v35;
    v35 = 0l;
    while (while_method_0(v35)){
        unsigned long long v37;
        v37 = (unsigned long long)v35;
        unsigned long long v38;
        v38 = v37 * 4ull;
        unsigned long long v39;
        v39 = 1136ull + v38;
        unsigned char * v40;
        v40 = (unsigned char *)(v0+v39);
        Union2 v42;
        v42 = f_5(v40);
        v33[v35] = v42;
        v35 += 1l ;
    }
    int v43;
    v43 = f_19(v0);
    unsigned char * v44;
    v44 = (unsigned char *)(v0+1152ull);
    Union8 v62;
    switch (v43) {
        case 0: {
            f_3(v44);
            v62 = Union8{Union8_0{}};
            break;
        }
        case 1: {
            Union6 v48; bool v49; static_array<Union3,2l> v50; int v51; static_array<int,2l> v52; int v53;
            Tuple1 tmp8 = f_10(v44);
            v48 = tmp8.v0; v49 = tmp8.v1; v50 = tmp8.v2; v51 = tmp8.v3; v52 = tmp8.v4; v53 = tmp8.v5;
            v62 = Union8{Union8_1{v48, v49, v50, v51, v52, v53}};
            break;
        }
        case 2: {
            Union6 v55; bool v56; static_array<Union3,2l> v57; int v58; static_array<int,2l> v59; int v60;
            Tuple1 tmp9 = f_10(v44);
            v55 = tmp9.v0; v56 = tmp9.v1; v57 = tmp9.v2; v58 = tmp9.v3; v59 = tmp9.v4; v60 = tmp9.v5;
            v62 = Union8{Union8_2{v55, v56, v57, v58, v59, v60}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple0{v1, v20, v21, v33, v62};
}
__device__ inline bool while_method_2(Union4 v0){
    switch (v0.tag) {
        case 0: { // None
            return false;
            break;
        }
        case 1: { // Some
            Union5 v1 = v0.case1.v0;
            return true;
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 4096l;
    return v1;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 8l;
    return v1;
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 16l;
    return v1;
}
__device__ void method_22(float * v0, int v1, float * v2, int v3, float * v4, int v5){
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
        while (while_method_5(v67)){
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
            while (while_method_7(v81)){
                assert("Tensor range check" && 0 <= v65 && v65 < 2l);
                int v83;
                v83 = v71 + v5;
                assert("Tensor range check" && 0 <= v81 && v81 < 16l);
                int v84;
                v84 = 8l * v81;
                int v85;
                v85 = v84 + v83;
                float * v86;
                v86 = v4+v85;
                assert("Tensor range check" && 0 <= v67 && v67 < 8l);
                int v88;
                v88 = 2048l * v67;
                int v89;
                v89 = v88 + v1;
                assert("Tensor range check" && 0 <= v81 && v81 < 16l);
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
                v105 = 128l * v98;
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
                        v118 = 2048l * v111;
                        int v119;
                        v119 = v118 + v115;
                        float v120[4l];
                        int v121;
                        v121 = 0l;
                        #pragma unroll
                        while (while_method_4(v121)){
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
                v140 = 128l * v133;
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
                        v153 = 2048l * v146;
                        int v154;
                        v154 = v153 + v150;
                        float v155[4l];
                        int v156;
                        v156 = 0l;
                        #pragma unroll
                        while (while_method_4(v156)){
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
            v258 = 128l * v253;
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
                    v271 = 1024l * v266;
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
__device__ inline bool while_method_8(int v0){
    bool v1;
    v1 = v0 < 32l;
    return v1;
}
__device__ void method_23(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5){
    int v6;
    v6 = blockIdx.x;
    assert("Tensor range check" && 0 <= v6 && v6 < 1l);
    int v7;
    v7 = 4096l * v6;
    int v8;
    v8 = blockIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 1l);
    int v9;
    v9 = 4096l * v8;
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
    v25 = 128l * v19;
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
    while (while_method_8(v30)){
        assert("Tensor range check" && 0 <= v30 && v30 < 32l);
        int v32;
        v32 = 128l * v30;
        int v33;
        v33 = v32 + v26;
        float v34[4l];
        int v35[4l];
        int v36;
        v36 = 0l;
        while (while_method_6(v36)){
            assert("Tensor range check" && 0 <= v36 && v36 < 1l);
            int v38;
            v38 = 4l * v36;
            assert("Tensor range check" && 0 <= v36 && v36 < 1l);
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
        while (while_method_6(v43)){
            int v45;
            v45 = 0l;
            while (while_method_4(v45)){
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
                    v60 = v43 < 1l;
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
                assert("Tensor range check" && 0 <= v43 && v43 < 1l);
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
        bool v78[4l];
        int v79;
        v79 = 0l;
        while (while_method_6(v79)){
            int v81;
            v81 = 0l;
            while (while_method_4(v81)){
                assert("Tensor range check" && 0 <= v79 && v79 < 1l);
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
                v87 = v86 < 3l;
                assert("Tensor range check" && 0 <= v79 && v79 < 1l);
                assert("Tensor range check" && 0 <= v81 && v81 < 4l);
                v78[v84] = v87;
                v81 += 1l ;
            }
            v79 += 1l ;
        }
        int v88[4l];
        int v89;
        v89 = 0l;
        while (while_method_6(v89)){
            int v91;
            v91 = 0l;
            while (while_method_4(v91)){
                assert("Tensor range check" && 0 <= v89 && v89 < 1l);
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
                assert("Tensor range check" && 0 <= v89 && v89 < 1l);
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
        while (while_method_6(v98)){
            int v100;
            v100 = 0l;
            while (while_method_4(v100)){
                assert("Tensor range check" && 0 <= v98 && v98 < 1l);
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
        float v112[4l];
        int v113;
        v113 = 0l;
        while (while_method_6(v113)){
            int v115;
            v115 = 0l;
            while (while_method_4(v115)){
                assert("Tensor range check" && 0 <= v113 && v113 < 1l);
                assert("Tensor range check" && 0 <= v115 && v115 < 4l);
                int v117;
                v117 = 4l * v113;
                int v118;
                v118 = v117 + v115;
                float v119;
                v119 = v34[v118];
                bool v120;
                v120 = v78[v118];
                float v121;
                if (v120){
                    v121 = v119;
                } else {
                    v121 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v113 && v113 < 1l);
                assert("Tensor range check" && 0 <= v115 && v115 < 4l);
                v112[v118] = v121;
                v115 += 1l ;
            }
            v113 += 1l ;
        }
        float v122;
        v122 = 0.0f;
        int v123;
        v123 = 0l;
        while (while_method_6(v123)){
            int v125;
            v125 = 0l;
            while (while_method_4(v125)){
                assert("Tensor range check" && 0 <= v123 && v123 < 1l);
                assert("Tensor range check" && 0 <= v125 && v125 < 4l);
                int v127;
                v127 = 4l * v123;
                int v128;
                v128 = v127 + v125;
                float v129;
                v129 = v112[v128];
                float v130;
                v130 = v122 + v129;
                v122 = v130;
                v125 += 1l ;
            }
            v123 += 1l ;
        }
        auto v131 = cooperative_groups::coalesced_threads();
        int v132;
        v132 = threadIdx.x;
        int v133;
        v133 = v132 / 32l;
        auto v134 = cooperative_groups::labeled_partition(v131,v133);
        Closure1 v135{};
        float v136;
        v136 = cooperative_groups::reduce(v134, v122, v135);
        float v137;
        v137 = (float)v111;
        float v138;
        v138 = v136 / v137;
        float v139[4l];
        int v140;
        v140 = 0l;
        while (while_method_6(v140)){
            int v142;
            v142 = 0l;
            while (while_method_4(v142)){
                assert("Tensor range check" && 0 <= v140 && v140 < 1l);
                assert("Tensor range check" && 0 <= v142 && v142 < 4l);
                int v144;
                v144 = 4l * v140;
                int v145;
                v145 = v144 + v142;
                float v146;
                v146 = v34[v145];
                bool v147;
                v147 = v78[v145];
                float v148;
                if (v147){
                    v148 = v146;
                } else {
                    v148 = -1.0f / 0.0f;
                }
                float v149;
                v149 = v148 - v138;
                float v150;
                v150 = exp(v149);
                assert("Tensor range check" && 0 <= v140 && v140 < 1l);
                assert("Tensor range check" && 0 <= v142 && v142 < 4l);
                v139[v145] = v150;
                v142 += 1l ;
            }
            v140 += 1l ;
        }
        float v151;
        v151 = 0.0f;
        int v152;
        v152 = 0l;
        while (while_method_6(v152)){
            int v154;
            v154 = 0l;
            while (while_method_4(v154)){
                assert("Tensor range check" && 0 <= v152 && v152 < 1l);
                assert("Tensor range check" && 0 <= v154 && v154 < 4l);
                int v156;
                v156 = 4l * v152;
                int v157;
                v157 = v156 + v154;
                float v158;
                v158 = v139[v157];
                float v159;
                v159 = v151 + v158;
                v151 = v159;
                v154 += 1l ;
            }
            v152 += 1l ;
        }
        auto v160 = cooperative_groups::coalesced_threads();
        int v161;
        v161 = threadIdx.x;
        int v162;
        v162 = v161 / 32l;
        auto v163 = cooperative_groups::labeled_partition(v160,v162);
        float v164;
        v164 = cooperative_groups::reduce(v163, v151, v135);
        float v165[4l];
        int v166;
        v166 = 0l;
        while (while_method_6(v166)){
            int v168;
            v168 = 0l;
            while (while_method_4(v168)){
                assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                assert("Tensor range check" && 0 <= v168 && v168 < 4l);
                int v170;
                v170 = 4l * v166;
                int v171;
                v171 = v170 + v168;
                float v172;
                v172 = v139[v171];
                bool v173;
                v173 = v164 == 0.0f;
                bool v174;
                v174 = v173 != true;
                float v176;
                if (v174){
                    float v175;
                    v175 = v172 / v164;
                    v176 = v175;
                } else {
                    v176 = 0.0078125f;
                }
                assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                assert("Tensor range check" && 0 <= v168 && v168 < 4l);
                v165[v171] = v176;
                v168 += 1l ;
            }
            v166 += 1l ;
        }
        float v177[4l];
        float v178;
        v178 = 0.0f;
        int v179;
        v179 = 0l;
        while (while_method_6(v179)){
            assert("Tensor range check" && 0 <= v179 && v179 < 1l);
            int v181;
            v181 = 4l * v179;
            assert("Tensor range check" && 0 <= v179 && v179 < 1l);
            int v182; float v183;
            Tuple8 tmp12 = Tuple8{0l, 0.0f};
            v182 = tmp12.v0; v183 = tmp12.v1;
            while (while_method_4(v182)){
                assert("Tensor range check" && 0 <= v182 && v182 < 4l);
                int v185;
                v185 = v182 + v181;
                float v186;
                v186 = v165[v185];
                float v187;
                v187 = v183 + v186;
                v183 = v187;
                v182 += 1l ;
            }
            auto v188 = cooperative_groups::coalesced_threads();
            int v189;
            v189 = threadIdx.x;
            int v190;
            v190 = v189 / 32l;
            auto v191 = cooperative_groups::labeled_partition(v188,v190);
            Closure2 v192{};
            float v193;
            v193 = cooperative_groups::inclusive_scan(v191, v183, v192);
            float v194;
            v194 = v191.shfl_up(v193,1);
            bool v195;
            v195 = v191.thread_rank() == 0;
            float v196;
            if (v195){
                v196 = 0.0f;
            } else {
                v196 = v194;
            }
            float v197;
            v197 = v191.shfl(v193,v191.num_threads()-1);
            float v198;
            v198 = v178 + v196;
            int v199; float v200;
            Tuple8 tmp13 = Tuple8{0l, v198};
            v199 = tmp13.v0; v200 = tmp13.v1;
            while (while_method_4(v199)){
                assert("Tensor range check" && 0 <= v199 && v199 < 4l);
                int v202;
                v202 = v199 + v181;
                float v203;
                v203 = v165[v202];
                float v204;
                v204 = v200 + v203;
                assert("Tensor range check" && 0 <= v199 && v199 < 4l);
                v177[v202] = v204;
                v200 = v204;
                v199 += 1l ;
            }
            float v205;
            v205 = v178 + v197;
            v178 = v205;
            v179 += 1l ;
        }
        float v206[4l];
        bool v207[4l];
        int v208;
        v208 = 0l;
        while (while_method_6(v208)){
            int v210;
            v210 = 0l;
            while (while_method_4(v210)){
                assert("Tensor range check" && 0 <= v208 && v208 < 1l);
                assert("Tensor range check" && 0 <= v210 && v210 < 4l);
                int v212;
                v212 = 4l * v208;
                int v213;
                v213 = v212 + v210;
                float v214;
                v214 = v177[v213];
                float v215;
                v215 = v165[v213];
                bool v216;
                v216 = v215 > 0.0f;
                assert("Tensor range check" && 0 <= v208 && v208 < 1l);
                assert("Tensor range check" && 0 <= v210 && v210 < 4l);
                v206[v213] = v214;
                v207[v213] = v216;
                v210 += 1l ;
            }
            v208 += 1l ;
        }
        float v217; bool v218;
        Tuple9 tmp14 = Tuple9{-1.0f / 0.0f, false};
        v217 = tmp14.v0; v218 = tmp14.v1;
        int v219;
        v219 = 0l;
        while (while_method_6(v219)){
            int v221;
            v221 = 0l;
            while (while_method_4(v221)){
                assert("Tensor range check" && 0 <= v219 && v219 < 1l);
                assert("Tensor range check" && 0 <= v221 && v221 < 4l);
                int v223;
                v223 = 4l * v219;
                int v224;
                v224 = v223 + v221;
                float v225;
                v225 = v206[v224];
                bool v226;
                v226 = v207[v224];
                float v233; bool v234;
                if (v218){
                    if (v226){
                        bool v227;
                        v227 = v217 >= v225;
                        float v228;
                        if (v227){
                            v228 = v217;
                        } else {
                            v228 = v225;
                        }
                        v233 = v228; v234 = true;
                    } else {
                        v233 = v217; v234 = v218;
                    }
                } else {
                    if (v226){
                        v233 = v225; v234 = v226;
                    } else {
                        v233 = v217; v234 = v218;
                    }
                }
                v217 = v233;
                v218 = v234;
                v221 += 1l ;
            }
            v219 += 1l ;
        }
        auto v235 = cooperative_groups::coalesced_threads();
        int v236;
        v236 = threadIdx.x;
        int v237;
        v237 = v236 / 32l;
        auto v238 = cooperative_groups::labeled_partition(v235,v237);
        Closure3 v239{};
        float v240; bool v241;
        Tuple9 tmp15 = cooperative_groups::reduce(v238, Tuple9{v217, v218}, v239);
        v240 = tmp15.v0; v241 = tmp15.v1;
        bool v242;
        v242 = v241 == false;
        if (v242){
            assert("The local reduce must be true." && v241);
        } else {
        }
        float v244[4l];
        int v245[4l];
        int v246;
        v246 = 0l;
        while (while_method_6(v246)){
            int v248;
            v248 = 0l;
            while (while_method_4(v248)){
                assert("Tensor range check" && 0 <= v246 && v246 < 1l);
                assert("Tensor range check" && 0 <= v248 && v248 < 4l);
                int v250;
                v250 = 4l * v246;
                int v251;
                v251 = v250 + v248;
                int v252;
                v252 = v35[v251];
                float v253;
                v253 = curand_uniform(&v5);
                assert("Tensor range check" && 0 <= v246 && v246 < 1l);
                assert("Tensor range check" && 0 <= v248 && v248 < 4l);
                v244[v251] = v253;
                v245[v251] = v252;
                v248 += 1l ;
            }
            v246 += 1l ;
        }
        float v254; int v255;
        Tuple10 tmp16 = Tuple10{0.0f, 2147483647l};
        v254 = tmp16.v0; v255 = tmp16.v1;
        int v256;
        v256 = 0l;
        while (while_method_6(v256)){
            int v258;
            v258 = 0l;
            while (while_method_4(v258)){
                assert("Tensor range check" && 0 <= v256 && v256 < 1l);
                assert("Tensor range check" && 0 <= v258 && v258 < 4l);
                int v260;
                v260 = 4l * v256;
                int v261;
                v261 = v260 + v258;
                float v262;
                v262 = v244[v261];
                int v263;
                v263 = v245[v261];
                bool v264;
                v264 = v255 < v263;
                float v265; int v266;
                if (v264){
                    v265 = v254; v266 = v255;
                } else {
                    v265 = v262; v266 = v263;
                }
                v254 = v265;
                v255 = v266;
                v258 += 1l ;
            }
            v256 += 1l ;
        }
        auto v267 = cooperative_groups::coalesced_threads();
        int v268;
        v268 = threadIdx.x;
        int v269;
        v269 = v268 / 32l;
        auto v270 = cooperative_groups::labeled_partition(v267,v269);
        Closure4 v271{};
        float v272; int v273;
        Tuple10 tmp17 = cooperative_groups::reduce(v270, Tuple10{v254, v255}, v271);
        v272 = tmp17.v0; v273 = tmp17.v1;
        float v274;
        v274 = v240 * v272;
        int v275[4l];
        bool v276[4l];
        int v277;
        v277 = 0l;
        while (while_method_6(v277)){
            int v279;
            v279 = 0l;
            while (while_method_4(v279)){
                assert("Tensor range check" && 0 <= v277 && v277 < 1l);
                assert("Tensor range check" && 0 <= v279 && v279 < 4l);
                int v281;
                v281 = 4l * v277;
                int v282;
                v282 = v281 + v279;
                float v283;
                v283 = v206[v282];
                bool v284;
                v284 = v207[v282];
                int v285;
                v285 = v35[v282];
                int v288; bool v289;
                if (v284){
                    float v286;
                    v286 = v283 - v274;
                    bool v287;
                    v287 = v286 >= 0.0f;
                    v288 = v285; v289 = v287;
                } else {
                    v288 = 2147483647l; v289 = false;
                }
                assert("Tensor range check" && 0 <= v277 && v277 < 1l);
                assert("Tensor range check" && 0 <= v279 && v279 < 4l);
                v275[v282] = v288;
                v276[v282] = v289;
                v279 += 1l ;
            }
            v277 += 1l ;
        }
        int v290; bool v291;
        Tuple11 tmp18 = Tuple11{2147483647l, false};
        v290 = tmp18.v0; v291 = tmp18.v1;
        int v292;
        v292 = 0l;
        while (while_method_6(v292)){
            int v294;
            v294 = 0l;
            while (while_method_4(v294)){
                assert("Tensor range check" && 0 <= v292 && v292 < 1l);
                assert("Tensor range check" && 0 <= v294 && v294 < 4l);
                int v296;
                v296 = 4l * v292;
                int v297;
                v297 = v296 + v294;
                int v298;
                v298 = v275[v297];
                bool v299;
                v299 = v276[v297];
                int v306; bool v307;
                if (v291){
                    if (v299){
                        bool v300;
                        v300 = v290 < v298;
                        int v301;
                        if (v300){
                            v301 = v290;
                        } else {
                            v301 = v298;
                        }
                        v306 = v301; v307 = true;
                    } else {
                        v306 = v290; v307 = v291;
                    }
                } else {
                    if (v299){
                        v306 = v298; v307 = v299;
                    } else {
                        v306 = v290; v307 = v291;
                    }
                }
                v290 = v306;
                v291 = v307;
                v294 += 1l ;
            }
            v292 += 1l ;
        }
        auto v308 = cooperative_groups::coalesced_threads();
        int v309;
        v309 = threadIdx.x;
        int v310;
        v310 = v309 / 32l;
        auto v311 = cooperative_groups::labeled_partition(v308,v310);
        Closure5 v312{};
        int v313; bool v314;
        Tuple11 tmp19 = cooperative_groups::reduce(v311, Tuple11{v290, v291}, v312);
        v313 = tmp19.v0; v314 = tmp19.v1;
        bool v315;
        v315 = v314 == false;
        if (v315){
            assert("The local reduce must be true." && v314);
        } else {
        }
        bool v317;
        v317 = v313 < 3l;
        bool v318;
        v318 = v317 == false;
        if (v318){
            assert("The masking requirement is violated in masked_softmax_and_discrete_sample_." && v317);
        } else {
        }
        assert("Tensor range check" && 0 <= v30 && v30 < 32l);
        int v320;
        v320 = v32 + v28;
        int v321;
        v321 = 0l;
        while (while_method_6(v321)){
            assert("Tensor range check" && 0 <= v321 && v321 < 1l);
            int v323;
            v323 = 128l * v321;
            int v324;
            v324 = v323 + v320;
            assert("Tensor range check" && 0 <= v321 && v321 < 1l);
            int v325;
            v325 = 4l * v321;
            int4* v326;
            v326 = reinterpret_cast<int4*>(v165 + v325);
            int4* v327;
            v327 = reinterpret_cast<int4*>(v2 + v324);
            assert("Pointer alignment check" && (unsigned long long)(v326) % 4l == 0 && (unsigned long long)(v327) % 4l == 0);
            *v327 = *v326;
            v321 += 1l ;
        }
        assert("Tensor range check" && 0 <= v30 && v30 < 32l);
        int v328;
        v328 = v30 + v29;
        v0[v328] = v313;
        v30 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return ;
}
__device__ unsigned int loop_25(unsigned int v0, curandStatePhilox4_32_10_t & v1){
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
        return loop_25(v0, v1);
    }
}
__device__ int int_range_24(int v0, int v1, curandStatePhilox4_32_10_t & v2){
    int v3;
    v3 = v0 - v1;
    unsigned int v4;
    v4 = (unsigned int)v3;
    unsigned int v5;
    v5 = loop_25(v4, v2);
    unsigned int v6;
    v6 = (unsigned int)v1;
    unsigned int v7;
    v7 = v5 + v6;
    int v8;
    v8 = (int)v7;
    return v8;
}
__device__ __noinline__ Tuple7 noinline_run_21(unsigned char * v0, unsigned char * v1, Union9 v2){
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v3;
    v3 = reinterpret_cast<float *>(&v0[32768ull]);
    float * v5;
    v5 = reinterpret_cast<float *>(&v0[0ull]);
    int * v7;
    v7 = reinterpret_cast<int *>(&v0[98304ull]);
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
    v19 = 4096l * v18;
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
    while (while_method_3(v28)){
        bool v30;
        v30 = 0l <= v28;
        bool v31;
        v31 = v30 == false;
        if (v31){
            assert("The index needs to be zero or positive." && v30);
        } else {
        }
        int v33;
        v33 = v28 % 128l;
        int v34;
        v34 = v28 / 128l;
        bool v35;
        v35 = v34 < 32l;
        bool v36;
        v36 = v35 == false;
        if (v36){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v35);
        } else {
        }
        assert("Tensor range check" && 0 <= v34 && v34 < 32l);
        assert("Tensor range check" && 0 <= v33 && v33 < 128l);
        int v38;
        v38 = v33 + v19;
        int v39;
        v39 = 128l * v34;
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
            Union6 v41 = v2.case1.v0; bool v42 = v2.case1.v1; static_array<Union3,2l> v43 = v2.case1.v2; int v44 = v2.case1.v3; static_array<int,2l> v45 = v2.case1.v4; int v46 = v2.case1.v5; static_array_list<Union7,32l> v47 = v2.case1.v6;
            int v48;
            v48 = threadIdx.x;
            assert("Tensor range check" && 0 <= v48 && v48 < 32l);
            int v49;
            v49 = 128l * v48;
            int v50;
            v50 = v49 + v19;
            static_array_list<Union11,10l> v51;
            v51 = static_array_list<Union11,10l>{};
            int v53;
            v53 = v47.length;
            int v54;
            v54 = 0l;
            while (while_method_1(v53, v54)){
                Union7 v56;
                v56 = v47[v54];
                Union12 v75;
                switch (v56.tag) {
                    case 0: { // CommunityCardIs
                        Union3 v65 = v56.case0.v0;
                        Union11 v66;
                        v66 = Union11{Union11_1{v65}};
                        v75 = Union12{Union12_1{v66}};
                        break;
                    }
                    case 1: { // PlayerAction
                        int v68 = v56.case1.v0; Union1 v69 = v56.case1.v1;
                        Union11 v70;
                        v70 = Union11{Union11_0{v69}};
                        v75 = Union12{Union12_1{v70}};
                        break;
                    }
                    case 2: { // PlayerGotCard
                        int v58 = v56.case2.v0; Union3 v59 = v56.case2.v1;
                        bool v60;
                        v60 = v58 == v44;
                        if (v60){
                            Union11 v61;
                            v61 = Union11{Union11_1{v59}};
                            v75 = Union12{Union12_1{v61}};
                        } else {
                            v75 = Union12{Union12_0{}};
                        }
                        break;
                    }
                    default: {
                        v75 = Union12{Union12_0{}};
                    }
                }
                switch (v75.tag) {
                    case 0: { // None
                        break;
                    }
                    case 1: { // Some
                        Union11 v76 = v75.case1.v0;
                        v51.push(v76);
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                v54 += 1l ;
            }
            float * v77;
            v77 = v16+v50;
            int v79;
            v79 = v51.length;
            bool v80;
            v80 = v79 == 0l;
            if (v80){
                v77[0l] = 1.0f;
            } else {
            }
            int v81;
            v81 = v51.length;
            int v82;
            v82 = 0l;
            while (while_method_1(v81, v82)){
                Union11 v84;
                v84 = v51[v82];
                int v86;
                v86 = v82 * 6l;
                int v87;
                v87 = 1l + v86;
                switch (v84.tag) {
                    case 0: { // C1of2
                        Union1 v88 = v84.case0.v0;
                        switch (v88.tag) {
                            case 0: { // Call
                                v77[v87] = 1.0f;
                                break;
                            }
                            case 1: { // Fold
                                int v89;
                                v89 = v87 + 1l;
                                v77[v89] = 1.0f;
                                break;
                            }
                            case 2: { // Raise
                                int v90;
                                v90 = v87 + 2l;
                                v77[v90] = 1.0f;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                            }
                        }
                        break;
                    }
                    case 1: { // C2of2
                        Union3 v91 = v84.case1.v0;
                        int v92;
                        v92 = v87 + 3l;
                        switch (v91.tag) {
                            case 0: { // Jack
                                v77[v92] = 1.0f;
                                break;
                            }
                            case 1: { // King
                                int v93;
                                v93 = v92 + 1l;
                                v77[v93] = 1.0f;
                                break;
                            }
                            case 2: { // Queen
                                int v94;
                                v94 = v92 + 2l;
                                v77[v94] = 1.0f;
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
                v82 += 1l ;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v95;
    v95 = 0l;
    while (while_method_4(v95)){
        float * v97;
        v97 = reinterpret_cast<float *>(&v0[0ull]);
        float * v99;
        v99 = reinterpret_cast<float *>(&v1[0ull]);
        assert("Tensor range check" && 0 <= v95 && v95 < 4l);
        int v101;
        v101 = 16384l * v95;
        float * v102;
        v102 = reinterpret_cast<float *>(&v0[16384ull]);
        int v104;
        v104 = blockIdx.x;
        assert("Tensor range check" && 0 <= v104 && v104 < 1l);
        int v105;
        v105 = 4096l * v104;
        int v106;
        v106 = blockIdx.x;
        assert("Tensor range check" && 0 <= v106 && v106 < 1l);
        int v107;
        v107 = 4096l * v106;
        method_22(v99, v101, v102, v107, v97, v105);
        float * v108;
        v108 = reinterpret_cast<float *>(&v0[32768ull]);
        assert("Tensor range check" && 0 <= v95 && v95 < 4l);
        int v110;
        v110 = 4096l * v95;
        int * v111;
        v111 = reinterpret_cast<int *>(&v0[98304ull]);
        assert("Tensor range check" && 0 <= v95 && v95 < 4l);
        int v113;
        v113 = 32l * v95;
        method_23(v111, v113, v108, v110, v102, v26);
        v95 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v114;
    v114 = 0l;
    int v115;
    v115 = 4l;
    int v116;
    v116 = int_range_24(v115, v114, v26);
    float * v117;
    v117 = reinterpret_cast<float *>(&v0[32768ull]);
    int * v119;
    v119 = reinterpret_cast<int *>(&v0[98304ull]);
    assert("Tensor range check" && 0 <= v116 && v116 < 4l);
    int v121;
    v121 = 32l * v116;
    int v122;
    v122 = blockIdx.x;
    assert("Tensor range check" && 0 <= v122 && v122 < 1l);
    int v123;
    v123 = 32l * v122;
    int v124;
    v124 = v123 + v121;
    int v125;
    v125 = threadIdx.x;
    assert("Tensor range check" && 0 <= v125 && v125 < 32l);
    int v126;
    v126 = v125 + v124;
    int v127;
    v127 = v119[v126];
    bool v128;
    v128 = 0l == v127;
    Union10 v137;
    if (v128){
        v137 = Union10{Union10_1{}};
    } else {
        bool v130;
        v130 = 1l == v127;
        if (v130){
            v137 = Union10{Union10_0{}};
        } else {
            bool v132;
            v132 = 2l == v127;
            if (v132){
                v137 = Union10{Union10_2{}};
            } else {
                printf("%s\n", "Invalid output id in the Leduc model.");
                asm("exit;");
            }
        }
    }
    int v138;
    v138 = blockIdx.x;
    int v139;
    v139 = threadIdx.x;
    assert("Tensor range check" && 0 <= v116 && v116 < 4l);
    assert("Tensor range check" && 0 <= v138 && v138 < 1l);
    assert("Tensor range check" && 0 <= v139 && v139 < 32l);
    assert("Tensor range check" && 0 <= v127 && v127 < 128l);
    int v140;
    v140 = 128l * v139;
    int v141;
    v141 = v140 + v127;
    int v142;
    v142 = 4096l * v138;
    int v143;
    v143 = v142 + v141;
    int v144;
    v144 = 4096l * v116;
    int v145;
    v145 = v144 + v143;
    float v146;
    v146 = v117[v145];
    int v147;
    v147 = blockIdx.x;
    assert("Tensor range check" && 0 <= v147 && v147 < 1l);
    int v148;
    v148 = 4096l * v147;
    int v149;
    v149 = threadIdx.x;
    assert("Tensor range check" && 0 <= v149 && v149 < 32l);
    int v150;
    v150 = 128l * v149;
    int v151;
    v151 = v150 + v148;
    assert("Tensor range check" && 0 <= v127 && v127 < 128l);
    int v152;
    v152 = v127 + v151;
    return Tuple7{v137, v117, v152, 4096l, 4l, v146};
}
__device__ void method_26(Union1 v0){
    switch (v0.tag) {
        case 0: { // Call
            printf("%s","Call");
            return ;
            break;
        }
        case 1: { // Fold
            printf("%s","Fold");
            return ;
            break;
        }
        case 2: { // Raise
            printf("%s","Raise");
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ int tag_28(Union3 v0){
    switch (v0.tag) {
        case 0: { // Jack
            return 0l;
            break;
        }
        case 1: { // King
            return 2l;
            break;
        }
        case 2: { // Queen
            return 1l;
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ bool is_pair_29(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
__device__ Tuple6 order_30(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple6{v1, v0};
    } else {
        return Tuple6{v0, v1};
    }
}
__device__ Union13 compare_hands_27(Union6 v0, bool v1, static_array<Union3,2l> v2, int v3, static_array<int,2l> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            asm("exit;");
            break;
        }
        case 1: { // Some
            Union3 v7 = v0.case1.v0;
            int v8;
            v8 = tag_28(v7);
            Union3 v9;
            v9 = v2[0l];
            int v11;
            v11 = tag_28(v9);
            Union3 v12;
            v12 = v2[1l];
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
                    Tuple6 tmp31 = order_30(v8, v11);
                    v27 = tmp31.v0; v28 = tmp31.v1;
                    int v29; int v30;
                    Tuple6 tmp32 = order_30(v8, v14);
                    v29 = tmp32.v0; v30 = tmp32.v1;
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
            assert("Invalid tag." && false);
        }
    }
}
__device__ void play_loop_20(unsigned char * v0, unsigned long long v1, unsigned char * v2, unsigned long long v3, static_array_list<Union3,6l> & v4, Union4 & v5, static_array_list<Union7,32l> & v6, static_array<Union2,2l> & v7, Union8 & v8, Union5 v9){
    static_array_list<Union7,32l> & v10 = v6;
    static_array_list<Union3,6l> & v11 = v4;
    Union4 v12;
    v12 = Union4{Union4_1{v9}};
    Union4 v13;
    v13 = v12;
    while (while_method_2(v13)){
        Union4 v461;
        switch (v13.tag) {
            case 0: { // None
                v461 = Union4{Union4_0{}};
                break;
            }
            case 1: { // Some
                Union5 v15 = v13.case1.v0;
                switch (v15.tag) {
                    case 0: { // ChanceCommunityCard
                        Union6 v418 = v15.case0.v0; bool v419 = v15.case0.v1; static_array<Union3,2l> v420 = v15.case0.v2; int v421 = v15.case0.v3; static_array<int,2l> v422 = v15.case0.v4; int v423 = v15.case0.v5;
                        Union3 v424;
                        v424 = v11.pop();
                        Union7 v425;
                        v425 = Union7{Union7_0{v424}};
                        v10.push(v425);
                        int v426;
                        v426 = 2l;
                        int v427; int v428;
                        Tuple6 tmp11 = Tuple6{0l, 0l};
                        v427 = tmp11.v0; v428 = tmp11.v1;
                        while (while_method_0(v427)){
                            int v430;
                            v430 = v422[v427];
                            bool v432;
                            v432 = v428 >= v430;
                            int v433;
                            if (v432){
                                v433 = v428;
                            } else {
                                v433 = v430;
                            }
                            v428 = v433;
                            v427 += 1l ;
                        }
                        static_array<int,2l> v434;
                        int v436;
                        v436 = 0l;
                        while (while_method_0(v436)){
                            v434[v436] = v428;
                            v436 += 1l ;
                        }
                        Union6 v438;
                        v438 = Union6{Union6_1{v424}};
                        Union5 v439;
                        v439 = Union5{Union5_2{v438, true, v420, 0l, v434, v426}};
                        v461 = Union4{Union4_1{v439}};
                        break;
                    }
                    case 1: { // ChanceInit
                        Union3 v441;
                        v441 = v11.pop();
                        Union3 v442;
                        v442 = v11.pop();
                        Union7 v443;
                        v443 = Union7{Union7_2{0l, v441}};
                        v10.push(v443);
                        Union7 v444;
                        v444 = Union7{Union7_2{1l, v442}};
                        v10.push(v444);
                        int v445;
                        v445 = 2l;
                        static_array<int,2l> v446;
                        v446[0l] = 1l;
                        v446[1l] = 1l;
                        static_array<Union3,2l> v448;
                        v448[0l] = v441;
                        v448[1l] = v442;
                        Union6 v450;
                        v450 = Union6{Union6_0{}};
                        Union5 v451;
                        v451 = Union5{Union5_2{v450, true, v448, 0l, v446, v445}};
                        v461 = Union4{Union4_1{v451}};
                        break;
                    }
                    case 2: { // Round
                        Union6 v49 = v15.case2.v0; bool v50 = v15.case2.v1; static_array<Union3,2l> v51 = v15.case2.v2; int v52 = v15.case2.v3; static_array<int,2l> v53 = v15.case2.v4; int v54 = v15.case2.v5;
                        static_array<Union2,2l> v55 = v7;
                        Union2 v56;
                        v56 = v55[v52];
                        switch (v56.tag) {
                            case 0: { // Computer
                                bool v58;
                                v58 = 262144ull == v3;
                                bool v59;
                                v59 = v58 == false;
                                if (v59){
                                    assert("The params needs to have matching offsets." && v58);
                                } else {
                                }
                                bool v61;
                                v61 = 98816ull == v1;
                                bool v62;
                                v62 = v61 == false;
                                if (v62){
                                    assert("The outputs needs to have matching offsets." && v61);
                                } else {
                                }
                                static_array_list<Union7,32l> & v64 = v6;
                                cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v65 = console_lock;
                                auto v66 = cooperative_groups::coalesced_threads();
                                v65.acquire();
                                printf("%s\n","Running the GPU model.");
                                v65.release();
                                v66.sync() ;
                                Union9 v69;
                                v69 = Union9{Union9_1{v49, v50, v51, v52, v53, v54, v64}};
                                Union10 v70; float * v71; int v72; int v73; int v74; float v75;
                                Tuple7 tmp20 = noinline_run_21(v0, v2, v69);
                                v70 = tmp20.v0; v71 = tmp20.v1; v72 = tmp20.v2; v73 = tmp20.v3; v74 = tmp20.v4; v75 = tmp20.v5;
                                Union1 v98;
                                switch (v70.tag) {
                                    case 0: { // AA_Call
                                        v98 = Union1{Union1_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v76;
                                        v76 = v53[0l];
                                        int v78; int v79;
                                        Tuple6 tmp21 = Tuple6{1l, v76};
                                        v78 = tmp21.v0; v79 = tmp21.v1;
                                        while (while_method_0(v78)){
                                            int v81;
                                            v81 = v53[v78];
                                            bool v83;
                                            v83 = v79 >= v81;
                                            int v84;
                                            if (v83){
                                                v84 = v79;
                                            } else {
                                                v84 = v81;
                                            }
                                            v79 = v84;
                                            v78 += 1l ;
                                        }
                                        int v85;
                                        v85 = v53[v52];
                                        bool v87;
                                        v87 = v85 == v79;
                                        if (v87){
                                            v98 = Union1{Union1_0{}};
                                        } else {
                                            v98 = Union1{Union1_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v92;
                                        v92 = v54 > 0l;
                                        if (v92){
                                            v98 = Union1{Union1_2{}};
                                        } else {
                                            v98 = Union1{Union1_0{}};
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                    }
                                }
                                cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v99 = console_lock;
                                auto v100 = cooperative_groups::coalesced_threads();
                                v99.acquire();
                                printf("%s","The action is: ");
                                v99.release();
                                v100.sync() ;
                                cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v103 = console_lock;
                                auto v104 = cooperative_groups::coalesced_threads();
                                v103.acquire();
                                printf("");
                                method_26(v98);
                                printf("\n");
                                v103.release();
                                v104.sync() ;
                                Union7 v107;
                                v107 = Union7{Union7_1{v52, v98}};
                                v10.push(v107);
                                Union5 v193;
                                switch (v49.tag) {
                                    case 0: { // None
                                        switch (v98.tag) {
                                            case 0: { // Call
                                                if (v50){
                                                    bool v157;
                                                    v157 = v52 == 0l;
                                                    int v158;
                                                    if (v157){
                                                        v158 = 1l;
                                                    } else {
                                                        v158 = 0l;
                                                    }
                                                    v193 = Union5{Union5_2{v49, false, v51, v158, v53, v54}};
                                                } else {
                                                    v193 = Union5{Union5_0{v49, v50, v51, v52, v53, v54}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v193 = Union5{Union5_5{v49, v50, v51, v52, v53, v54}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                bool v162;
                                                v162 = v54 > 0l;
                                                if (v162){
                                                    bool v163;
                                                    v163 = v52 == 0l;
                                                    int v164;
                                                    if (v163){
                                                        v164 = 1l;
                                                    } else {
                                                        v164 = 0l;
                                                    }
                                                    int v165;
                                                    v165 = -1l + v54;
                                                    int v166; int v167;
                                                    Tuple6 tmp22 = Tuple6{0l, 0l};
                                                    v166 = tmp22.v0; v167 = tmp22.v1;
                                                    while (while_method_0(v166)){
                                                        int v169;
                                                        v169 = v53[v166];
                                                        bool v171;
                                                        v171 = v167 >= v169;
                                                        int v172;
                                                        if (v171){
                                                            v172 = v167;
                                                        } else {
                                                            v172 = v169;
                                                        }
                                                        v167 = v172;
                                                        v166 += 1l ;
                                                    }
                                                    static_array<int,2l> v173;
                                                    int v175;
                                                    v175 = 0l;
                                                    while (while_method_0(v175)){
                                                        v173[v175] = v167;
                                                        v175 += 1l ;
                                                    }
                                                    static_array<int,2l> v177;
                                                    int v179;
                                                    v179 = 0l;
                                                    while (while_method_0(v179)){
                                                        int v181;
                                                        v181 = v173[v179];
                                                        bool v183;
                                                        v183 = v179 == v52;
                                                        int v185;
                                                        if (v183){
                                                            int v184;
                                                            v184 = v181 + 2l;
                                                            v185 = v184;
                                                        } else {
                                                            v185 = v181;
                                                        }
                                                        v177[v179] = v185;
                                                        v179 += 1l ;
                                                    }
                                                    v193 = Union5{Union5_2{v49, false, v51, v164, v177, v165}};
                                                } else {
                                                    printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                                    asm("exit;");
                                                }
                                                break;
                                            }
                                            default: {
                                                assert("Invalid tag." && false);
                                            }
                                        }
                                        break;
                                    }
                                    case 1: { // Some
                                        Union3 v108 = v49.case1.v0;
                                        switch (v98.tag) {
                                            case 0: { // Call
                                                if (v50){
                                                    bool v110;
                                                    v110 = v52 == 0l;
                                                    int v111;
                                                    if (v110){
                                                        v111 = 1l;
                                                    } else {
                                                        v111 = 0l;
                                                    }
                                                    v193 = Union5{Union5_2{v49, false, v51, v111, v53, v54}};
                                                } else {
                                                    int v113; int v114;
                                                    Tuple6 tmp23 = Tuple6{0l, 0l};
                                                    v113 = tmp23.v0; v114 = tmp23.v1;
                                                    while (while_method_0(v113)){
                                                        int v116;
                                                        v116 = v53[v113];
                                                        bool v118;
                                                        v118 = v114 >= v116;
                                                        int v119;
                                                        if (v118){
                                                            v119 = v114;
                                                        } else {
                                                            v119 = v116;
                                                        }
                                                        v114 = v119;
                                                        v113 += 1l ;
                                                    }
                                                    static_array<int,2l> v120;
                                                    int v122;
                                                    v122 = 0l;
                                                    while (while_method_0(v122)){
                                                        v120[v122] = v114;
                                                        v122 += 1l ;
                                                    }
                                                    v193 = Union5{Union5_4{v49, v50, v51, v52, v120, v54}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v193 = Union5{Union5_5{v49, v50, v51, v52, v53, v54}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                bool v126;
                                                v126 = v54 > 0l;
                                                if (v126){
                                                    bool v127;
                                                    v127 = v52 == 0l;
                                                    int v128;
                                                    if (v127){
                                                        v128 = 1l;
                                                    } else {
                                                        v128 = 0l;
                                                    }
                                                    int v129;
                                                    v129 = -1l + v54;
                                                    int v130; int v131;
                                                    Tuple6 tmp24 = Tuple6{0l, 0l};
                                                    v130 = tmp24.v0; v131 = tmp24.v1;
                                                    while (while_method_0(v130)){
                                                        int v133;
                                                        v133 = v53[v130];
                                                        bool v135;
                                                        v135 = v131 >= v133;
                                                        int v136;
                                                        if (v135){
                                                            v136 = v131;
                                                        } else {
                                                            v136 = v133;
                                                        }
                                                        v131 = v136;
                                                        v130 += 1l ;
                                                    }
                                                    static_array<int,2l> v137;
                                                    int v139;
                                                    v139 = 0l;
                                                    while (while_method_0(v139)){
                                                        v137[v139] = v131;
                                                        v139 += 1l ;
                                                    }
                                                    static_array<int,2l> v141;
                                                    int v143;
                                                    v143 = 0l;
                                                    while (while_method_0(v143)){
                                                        int v145;
                                                        v145 = v137[v143];
                                                        bool v147;
                                                        v147 = v143 == v52;
                                                        int v149;
                                                        if (v147){
                                                            int v148;
                                                            v148 = v145 + 4l;
                                                            v149 = v148;
                                                        } else {
                                                            v149 = v145;
                                                        }
                                                        v141[v143] = v149;
                                                        v143 += 1l ;
                                                    }
                                                    v193 = Union5{Union5_2{v49, false, v51, v128, v141, v129}};
                                                } else {
                                                    printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                                    asm("exit;");
                                                }
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
                                v461 = Union4{Union4_1{v193}};
                                break;
                            }
                            case 1: { // Human
                                Union8 v195;
                                v195 = Union8{Union8_2{v49, v50, v51, v52, v53, v54}};
                                v8 = v195;
                                Union4 v196;
                                v196 = Union4{Union4_1{v15}};
                                v5 = v196;
                                v461 = Union4{Union4_0{}};
                                break;
                            }
                            case 2: { // Random
                                static_array_list<Union1,3l> v198;
                                v198 = static_array_list<Union1,3l>{};
                                v198.unsafe_set_length(1l);
                                Union1 v200;
                                v200 = Union1{Union1_0{}};
                                v198[0l] = v200;
                                int v202;
                                v202 = v53[0l];
                                int v204;
                                v204 = v53[1l];
                                bool v206;
                                v206 = v202 == v204;
                                bool v207;
                                v207 = v206 != true;
                                if (v207){
                                    Union1 v208;
                                    v208 = Union1{Union1_1{}};
                                    v198.push(v208);
                                } else {
                                }
                                bool v209;
                                v209 = v54 > 0l;
                                if (v209){
                                    Union1 v210;
                                    v210 = Union1{Union1_2{}};
                                    v198.push(v210);
                                } else {
                                }
                                unsigned long long v211;
                                v211 = clock64();
                                curandStatePhilox4_32_10_t v212;
                                curand_init(v211,0ull,0ull,&v212);
                                int v213;
                                v213 = v198.length;
                                int v214;
                                v214 = v213 - 1l;
                                int v215;
                                v215 = 0l;
                                while (while_method_1(v214, v215)){
                                    int v217;
                                    v217 = v198.length;
                                    int v218;
                                    v218 = int_range_24(v217, v215, v212);
                                    Union1 v219;
                                    v219 = v198[v215];
                                    Union1 v221;
                                    v221 = v198[v218];
                                    v198[v215] = v221;
                                    v198[v218] = v219;
                                    v215 += 1l ;
                                }
                                Union1 v233;
                                v233 = v198.pop();
                                Union7 v234;
                                v234 = Union7{Union7_1{v52, v233}};
                                v10.push(v234);
                                Union5 v318;
                                switch (v49.tag) {
                                    case 0: { // None
                                        switch (v233.tag) {
                                            case 0: { // Call
                                                if (v50){
                                                    bool v283;
                                                    v283 = v52 == 0l;
                                                    int v284;
                                                    if (v283){
                                                        v284 = 1l;
                                                    } else {
                                                        v284 = 0l;
                                                    }
                                                    v318 = Union5{Union5_2{v49, false, v51, v284, v53, v54}};
                                                } else {
                                                    v318 = Union5{Union5_0{v49, v50, v51, v52, v53, v54}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v318 = Union5{Union5_5{v49, v50, v51, v52, v53, v54}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                if (v209){
                                                    bool v288;
                                                    v288 = v52 == 0l;
                                                    int v289;
                                                    if (v288){
                                                        v289 = 1l;
                                                    } else {
                                                        v289 = 0l;
                                                    }
                                                    int v290;
                                                    v290 = -1l + v54;
                                                    int v291; int v292;
                                                    Tuple6 tmp25 = Tuple6{0l, 0l};
                                                    v291 = tmp25.v0; v292 = tmp25.v1;
                                                    while (while_method_0(v291)){
                                                        int v294;
                                                        v294 = v53[v291];
                                                        bool v296;
                                                        v296 = v292 >= v294;
                                                        int v297;
                                                        if (v296){
                                                            v297 = v292;
                                                        } else {
                                                            v297 = v294;
                                                        }
                                                        v292 = v297;
                                                        v291 += 1l ;
                                                    }
                                                    static_array<int,2l> v298;
                                                    int v300;
                                                    v300 = 0l;
                                                    while (while_method_0(v300)){
                                                        v298[v300] = v292;
                                                        v300 += 1l ;
                                                    }
                                                    static_array<int,2l> v302;
                                                    int v304;
                                                    v304 = 0l;
                                                    while (while_method_0(v304)){
                                                        int v306;
                                                        v306 = v298[v304];
                                                        bool v308;
                                                        v308 = v304 == v52;
                                                        int v310;
                                                        if (v308){
                                                            int v309;
                                                            v309 = v306 + 2l;
                                                            v310 = v309;
                                                        } else {
                                                            v310 = v306;
                                                        }
                                                        v302[v304] = v310;
                                                        v304 += 1l ;
                                                    }
                                                    v318 = Union5{Union5_2{v49, false, v51, v289, v302, v290}};
                                                } else {
                                                    printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                                    asm("exit;");
                                                }
                                                break;
                                            }
                                            default: {
                                                assert("Invalid tag." && false);
                                            }
                                        }
                                        break;
                                    }
                                    case 1: { // Some
                                        Union3 v235 = v49.case1.v0;
                                        switch (v233.tag) {
                                            case 0: { // Call
                                                if (v50){
                                                    bool v237;
                                                    v237 = v52 == 0l;
                                                    int v238;
                                                    if (v237){
                                                        v238 = 1l;
                                                    } else {
                                                        v238 = 0l;
                                                    }
                                                    v318 = Union5{Union5_2{v49, false, v51, v238, v53, v54}};
                                                } else {
                                                    int v240; int v241;
                                                    Tuple6 tmp26 = Tuple6{0l, 0l};
                                                    v240 = tmp26.v0; v241 = tmp26.v1;
                                                    while (while_method_0(v240)){
                                                        int v243;
                                                        v243 = v53[v240];
                                                        bool v245;
                                                        v245 = v241 >= v243;
                                                        int v246;
                                                        if (v245){
                                                            v246 = v241;
                                                        } else {
                                                            v246 = v243;
                                                        }
                                                        v241 = v246;
                                                        v240 += 1l ;
                                                    }
                                                    static_array<int,2l> v247;
                                                    int v249;
                                                    v249 = 0l;
                                                    while (while_method_0(v249)){
                                                        v247[v249] = v241;
                                                        v249 += 1l ;
                                                    }
                                                    v318 = Union5{Union5_4{v49, v50, v51, v52, v247, v54}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v318 = Union5{Union5_5{v49, v50, v51, v52, v53, v54}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                if (v209){
                                                    bool v253;
                                                    v253 = v52 == 0l;
                                                    int v254;
                                                    if (v253){
                                                        v254 = 1l;
                                                    } else {
                                                        v254 = 0l;
                                                    }
                                                    int v255;
                                                    v255 = -1l + v54;
                                                    int v256; int v257;
                                                    Tuple6 tmp27 = Tuple6{0l, 0l};
                                                    v256 = tmp27.v0; v257 = tmp27.v1;
                                                    while (while_method_0(v256)){
                                                        int v259;
                                                        v259 = v53[v256];
                                                        bool v261;
                                                        v261 = v257 >= v259;
                                                        int v262;
                                                        if (v261){
                                                            v262 = v257;
                                                        } else {
                                                            v262 = v259;
                                                        }
                                                        v257 = v262;
                                                        v256 += 1l ;
                                                    }
                                                    static_array<int,2l> v263;
                                                    int v265;
                                                    v265 = 0l;
                                                    while (while_method_0(v265)){
                                                        v263[v265] = v257;
                                                        v265 += 1l ;
                                                    }
                                                    static_array<int,2l> v267;
                                                    int v269;
                                                    v269 = 0l;
                                                    while (while_method_0(v269)){
                                                        int v271;
                                                        v271 = v263[v269];
                                                        bool v273;
                                                        v273 = v269 == v52;
                                                        int v275;
                                                        if (v273){
                                                            int v274;
                                                            v274 = v271 + 4l;
                                                            v275 = v274;
                                                        } else {
                                                            v275 = v271;
                                                        }
                                                        v267[v269] = v275;
                                                        v269 += 1l ;
                                                    }
                                                    v318 = Union5{Union5_2{v49, false, v51, v254, v267, v255}};
                                                } else {
                                                    printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                                    asm("exit;");
                                                }
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
                                v461 = Union4{Union4_1{v318}};
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                            }
                        }
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union6 v323 = v15.case3.v0; bool v324 = v15.case3.v1; static_array<Union3,2l> v325 = v15.case3.v2; int v326 = v15.case3.v3; static_array<int,2l> v327 = v15.case3.v4; int v328 = v15.case3.v5; Union1 v329 = v15.case3.v6;
                        Union7 v330;
                        v330 = Union7{Union7_1{v326, v329}};
                        v10.push(v330);
                        Union5 v416;
                        switch (v323.tag) {
                            case 0: { // None
                                switch (v329.tag) {
                                    case 0: { // Call
                                        if (v324){
                                            bool v380;
                                            v380 = v326 == 0l;
                                            int v381;
                                            if (v380){
                                                v381 = 1l;
                                            } else {
                                                v381 = 0l;
                                            }
                                            v416 = Union5{Union5_2{v323, false, v325, v381, v327, v328}};
                                        } else {
                                            v416 = Union5{Union5_0{v323, v324, v325, v326, v327, v328}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v416 = Union5{Union5_5{v323, v324, v325, v326, v327, v328}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v385;
                                        v385 = v328 > 0l;
                                        if (v385){
                                            bool v386;
                                            v386 = v326 == 0l;
                                            int v387;
                                            if (v386){
                                                v387 = 1l;
                                            } else {
                                                v387 = 0l;
                                            }
                                            int v388;
                                            v388 = -1l + v328;
                                            int v389; int v390;
                                            Tuple6 tmp28 = Tuple6{0l, 0l};
                                            v389 = tmp28.v0; v390 = tmp28.v1;
                                            while (while_method_0(v389)){
                                                int v392;
                                                v392 = v327[v389];
                                                bool v394;
                                                v394 = v390 >= v392;
                                                int v395;
                                                if (v394){
                                                    v395 = v390;
                                                } else {
                                                    v395 = v392;
                                                }
                                                v390 = v395;
                                                v389 += 1l ;
                                            }
                                            static_array<int,2l> v396;
                                            int v398;
                                            v398 = 0l;
                                            while (while_method_0(v398)){
                                                v396[v398] = v390;
                                                v398 += 1l ;
                                            }
                                            static_array<int,2l> v400;
                                            int v402;
                                            v402 = 0l;
                                            while (while_method_0(v402)){
                                                int v404;
                                                v404 = v396[v402];
                                                bool v406;
                                                v406 = v402 == v326;
                                                int v408;
                                                if (v406){
                                                    int v407;
                                                    v407 = v404 + 2l;
                                                    v408 = v407;
                                                } else {
                                                    v408 = v404;
                                                }
                                                v400[v402] = v408;
                                                v402 += 1l ;
                                            }
                                            v416 = Union5{Union5_2{v323, false, v325, v387, v400, v388}};
                                        } else {
                                            printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                            asm("exit;");
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                    }
                                }
                                break;
                            }
                            case 1: { // Some
                                Union3 v331 = v323.case1.v0;
                                switch (v329.tag) {
                                    case 0: { // Call
                                        if (v324){
                                            bool v333;
                                            v333 = v326 == 0l;
                                            int v334;
                                            if (v333){
                                                v334 = 1l;
                                            } else {
                                                v334 = 0l;
                                            }
                                            v416 = Union5{Union5_2{v323, false, v325, v334, v327, v328}};
                                        } else {
                                            int v336; int v337;
                                            Tuple6 tmp29 = Tuple6{0l, 0l};
                                            v336 = tmp29.v0; v337 = tmp29.v1;
                                            while (while_method_0(v336)){
                                                int v339;
                                                v339 = v327[v336];
                                                bool v341;
                                                v341 = v337 >= v339;
                                                int v342;
                                                if (v341){
                                                    v342 = v337;
                                                } else {
                                                    v342 = v339;
                                                }
                                                v337 = v342;
                                                v336 += 1l ;
                                            }
                                            static_array<int,2l> v343;
                                            int v345;
                                            v345 = 0l;
                                            while (while_method_0(v345)){
                                                v343[v345] = v337;
                                                v345 += 1l ;
                                            }
                                            v416 = Union5{Union5_4{v323, v324, v325, v326, v343, v328}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v416 = Union5{Union5_5{v323, v324, v325, v326, v327, v328}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v349;
                                        v349 = v328 > 0l;
                                        if (v349){
                                            bool v350;
                                            v350 = v326 == 0l;
                                            int v351;
                                            if (v350){
                                                v351 = 1l;
                                            } else {
                                                v351 = 0l;
                                            }
                                            int v352;
                                            v352 = -1l + v328;
                                            int v353; int v354;
                                            Tuple6 tmp30 = Tuple6{0l, 0l};
                                            v353 = tmp30.v0; v354 = tmp30.v1;
                                            while (while_method_0(v353)){
                                                int v356;
                                                v356 = v327[v353];
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
                                            static_array<int,2l> v360;
                                            int v362;
                                            v362 = 0l;
                                            while (while_method_0(v362)){
                                                v360[v362] = v354;
                                                v362 += 1l ;
                                            }
                                            static_array<int,2l> v364;
                                            int v366;
                                            v366 = 0l;
                                            while (while_method_0(v366)){
                                                int v368;
                                                v368 = v360[v366];
                                                bool v370;
                                                v370 = v366 == v326;
                                                int v372;
                                                if (v370){
                                                    int v371;
                                                    v371 = v368 + 4l;
                                                    v372 = v371;
                                                } else {
                                                    v372 = v368;
                                                }
                                                v364[v366] = v372;
                                                v366 += 1l ;
                                            }
                                            v416 = Union5{Union5_2{v323, false, v325, v351, v364, v352}};
                                        } else {
                                            printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                            asm("exit;");
                                        }
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
                        v461 = Union4{Union4_1{v416}};
                        break;
                    }
                    case 4: { // TerminalCall
                        Union6 v30 = v15.case4.v0; bool v31 = v15.case4.v1; static_array<Union3,2l> v32 = v15.case4.v2; int v33 = v15.case4.v3; static_array<int,2l> v34 = v15.case4.v4; int v35 = v15.case4.v5;
                        int v36;
                        v36 = v34[v33];
                        Union13 v38;
                        v38 = compare_hands_27(v30, v31, v32, v33, v34, v35);
                        int v43; int v44;
                        switch (v38.tag) {
                            case 0: { // Eq
                                v43 = 0l; v44 = -1l;
                                break;
                            }
                            case 1: { // Gt
                                v43 = v36; v44 = 0l;
                                break;
                            }
                            case 2: { // Lt
                                v43 = v36; v44 = 1l;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                            }
                        }
                        Union7 v45;
                        v45 = Union7{Union7_3{v32, v43, v44}};
                        v10.push(v45);
                        Union8 v46;
                        v46 = Union8{Union8_1{v30, v31, v32, v33, v34, v35}};
                        v8 = v46;
                        Union4 v47;
                        v47 = Union4{Union4_0{}};
                        v5 = v47;
                        v461 = Union4{Union4_0{}};
                        break;
                    }
                    case 5: { // TerminalFold
                        Union6 v16 = v15.case5.v0; bool v17 = v15.case5.v1; static_array<Union3,2l> v18 = v15.case5.v2; int v19 = v15.case5.v3; static_array<int,2l> v20 = v15.case5.v4; int v21 = v15.case5.v5;
                        int v22;
                        v22 = v20[v19];
                        bool v24;
                        v24 = v19 == 0l;
                        int v25;
                        if (v24){
                            v25 = 1l;
                        } else {
                            v25 = 0l;
                        }
                        Union7 v26;
                        v26 = Union7{Union7_3{v18, v22, v25}};
                        v10.push(v26);
                        Union8 v27;
                        v27 = Union8{Union8_1{v16, v17, v18, v19, v20, v21}};
                        v8 = v27;
                        Union4 v28;
                        v28 = Union4{Union4_0{}};
                        v5 = v28;
                        v461 = Union4{Union4_0{}};
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
        v13 = v461;
    }
    return ;
}
__device__ void f_32(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+0ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_34(unsigned char * v0){
    return ;
}
__device__ void f_33(unsigned char * v0, Union3 v1){
    int v2;
    v2 = v1.tag;
    f_32(v0, v2);
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
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_35(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+28ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_37(unsigned char * v0, Union6 v1, bool v2, static_array<Union3,2l> v3, int v4, static_array<int,2l> v5, int v6){
    int v7;
    v7 = v1.tag;
    f_32(v0, v7);
    unsigned char * v8;
    v8 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // None
            f_34(v8);
            break;
        }
        case 1: { // Some
            Union3 v10 = v1.case1.v0;
            f_33(v8, v10);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    bool * v11;
    v11 = (bool *)(v0+8ull);
    v11[0l] = v2;
    int v13;
    v13 = 0l;
    while (while_method_0(v13)){
        unsigned long long v15;
        v15 = (unsigned long long)v13;
        unsigned long long v16;
        v16 = v15 * 4ull;
        unsigned long long v17;
        v17 = 12ull + v16;
        unsigned char * v18;
        v18 = (unsigned char *)(v0+v17);
        Union3 v20;
        v20 = v3[v13];
        f_33(v18, v20);
        v13 += 1l ;
    }
    int * v22;
    v22 = (int *)(v0+20ull);
    v22[0l] = v4;
    int v24;
    v24 = 0l;
    while (while_method_0(v24)){
        unsigned long long v26;
        v26 = (unsigned long long)v24;
        unsigned long long v27;
        v27 = v26 * 4ull;
        unsigned long long v28;
        v28 = 24ull + v27;
        unsigned char * v29;
        v29 = (unsigned char *)(v0+v28);
        int v31;
        v31 = v5[v24];
        f_32(v29, v31);
        v24 += 1l ;
    }
    int * v33;
    v33 = (int *)(v0+32ull);
    v33[0l] = v6;
    return ;
}
__device__ void f_39(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+36ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_38(unsigned char * v0, Union6 v1, bool v2, static_array<Union3,2l> v3, int v4, static_array<int,2l> v5, int v6, Union1 v7){
    int v8;
    v8 = v1.tag;
    f_32(v0, v8);
    unsigned char * v9;
    v9 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // None
            f_34(v9);
            break;
        }
        case 1: { // Some
            Union3 v11 = v1.case1.v0;
            f_33(v9, v11);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    bool * v12;
    v12 = (bool *)(v0+8ull);
    v12[0l] = v2;
    int v14;
    v14 = 0l;
    while (while_method_0(v14)){
        unsigned long long v16;
        v16 = (unsigned long long)v14;
        unsigned long long v17;
        v17 = v16 * 4ull;
        unsigned long long v18;
        v18 = 12ull + v17;
        unsigned char * v19;
        v19 = (unsigned char *)(v0+v18);
        Union3 v21;
        v21 = v3[v14];
        f_33(v19, v21);
        v14 += 1l ;
    }
    int * v23;
    v23 = (int *)(v0+20ull);
    v23[0l] = v4;
    int v25;
    v25 = 0l;
    while (while_method_0(v25)){
        unsigned long long v27;
        v27 = (unsigned long long)v25;
        unsigned long long v28;
        v28 = v27 * 4ull;
        unsigned long long v29;
        v29 = 24ull + v28;
        unsigned char * v30;
        v30 = (unsigned char *)(v0+v29);
        int v32;
        v32 = v5[v25];
        f_32(v30, v32);
        v25 += 1l ;
    }
    int * v34;
    v34 = (int *)(v0+32ull);
    v34[0l] = v6;
    int v36;
    v36 = v7.tag;
    f_39(v0, v36);
    unsigned char * v37;
    v37 = (unsigned char *)(v0+40ull);
    switch (v7.tag) {
        case 0: { // Call
            return f_34(v37);
            break;
        }
        case 1: { // Fold
            return f_34(v37);
            break;
        }
        case 2: { // Raise
            return f_34(v37);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_36(unsigned char * v0, Union5 v1){
    int v2;
    v2 = v1.tag;
    f_32(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // ChanceCommunityCard
            Union6 v5 = v1.case0.v0; bool v6 = v1.case0.v1; static_array<Union3,2l> v7 = v1.case0.v2; int v8 = v1.case0.v3; static_array<int,2l> v9 = v1.case0.v4; int v10 = v1.case0.v5;
            return f_37(v3, v5, v6, v7, v8, v9, v10);
            break;
        }
        case 1: { // ChanceInit
            return f_34(v3);
            break;
        }
        case 2: { // Round
            Union6 v11 = v1.case2.v0; bool v12 = v1.case2.v1; static_array<Union3,2l> v13 = v1.case2.v2; int v14 = v1.case2.v3; static_array<int,2l> v15 = v1.case2.v4; int v16 = v1.case2.v5;
            return f_37(v3, v11, v12, v13, v14, v15, v16);
            break;
        }
        case 3: { // RoundWithAction
            Union6 v17 = v1.case3.v0; bool v18 = v1.case3.v1; static_array<Union3,2l> v19 = v1.case3.v2; int v20 = v1.case3.v3; static_array<int,2l> v21 = v1.case3.v4; int v22 = v1.case3.v5; Union1 v23 = v1.case3.v6;
            return f_38(v3, v17, v18, v19, v20, v21, v22, v23);
            break;
        }
        case 4: { // TerminalCall
            Union6 v24 = v1.case4.v0; bool v25 = v1.case4.v1; static_array<Union3,2l> v26 = v1.case4.v2; int v27 = v1.case4.v3; static_array<int,2l> v28 = v1.case4.v4; int v29 = v1.case4.v5;
            return f_37(v3, v24, v25, v26, v27, v28, v29);
            break;
        }
        case 5: { // TerminalFold
            Union6 v30 = v1.case5.v0; bool v31 = v1.case5.v1; static_array<Union3,2l> v32 = v1.case5.v2; int v33 = v1.case5.v3; static_array<int,2l> v34 = v1.case5.v4; int v35 = v1.case5.v5;
            return f_37(v3, v30, v31, v32, v33, v34, v35);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_40(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+96ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_43(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+4ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_42(unsigned char * v0, int v1, Union1 v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0l] = v1;
    int v5;
    v5 = v2.tag;
    f_43(v0, v5);
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
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_44(unsigned char * v0, int v1, Union3 v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0l] = v1;
    int v5;
    v5 = v2.tag;
    f_43(v0, v5);
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
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_45(unsigned char * v0, static_array<Union3,2l> v1, int v2, int v3){
    int v4;
    v4 = 0l;
    while (while_method_0(v4)){
        unsigned long long v6;
        v6 = (unsigned long long)v4;
        unsigned long long v7;
        v7 = v6 * 4ull;
        unsigned char * v8;
        v8 = (unsigned char *)(v0+v7);
        Union3 v10;
        v10 = v1[v4];
        f_33(v8, v10);
        v4 += 1l ;
    }
    int * v12;
    v12 = (int *)(v0+8ull);
    v12[0l] = v2;
    int * v14;
    v14 = (int *)(v0+12ull);
    v14[0l] = v3;
    return ;
}
__device__ void f_41(unsigned char * v0, Union7 v1){
    int v2;
    v2 = v1.tag;
    f_32(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // CommunityCardIs
            Union3 v5 = v1.case0.v0;
            return f_33(v3, v5);
            break;
        }
        case 1: { // PlayerAction
            int v6 = v1.case1.v0; Union1 v7 = v1.case1.v1;
            return f_42(v3, v6, v7);
            break;
        }
        case 2: { // PlayerGotCard
            int v8 = v1.case2.v0; Union3 v9 = v1.case2.v1;
            return f_44(v3, v8, v9);
            break;
        }
        case 3: { // Showdown
            static_array<Union3,2l> v10 = v1.case3.v0; int v11 = v1.case3.v1; int v12 = v1.case3.v2;
            return f_45(v3, v10, v11, v12);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_46(unsigned char * v0, Union2 v1){
    int v2;
    v2 = v1.tag;
    f_32(v0, v2);
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
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_47(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+1144ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_31(unsigned char * v0, static_array_list<Union3,6l> v1, Union4 v2, static_array_list<Union7,32l> v3, static_array<Union2,2l> v4, Union8 v5){
    int v6;
    v6 = v1.length;
    f_32(v0, v6);
    int v7;
    v7 = v1.length;
    int v8;
    v8 = 0l;
    while (while_method_1(v7, v8)){
        unsigned long long v10;
        v10 = (unsigned long long)v8;
        unsigned long long v11;
        v11 = v10 * 4ull;
        unsigned long long v12;
        v12 = 4ull + v11;
        unsigned char * v13;
        v13 = (unsigned char *)(v0+v12);
        Union3 v15;
        v15 = v1[v8];
        f_33(v13, v15);
        v8 += 1l ;
    }
    int v17;
    v17 = v2.tag;
    f_35(v0, v17);
    unsigned char * v18;
    v18 = (unsigned char *)(v0+32ull);
    switch (v2.tag) {
        case 0: { // None
            f_34(v18);
            break;
        }
        case 1: { // Some
            Union5 v20 = v2.case1.v0;
            f_36(v18, v20);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    int v21;
    v21 = v3.length;
    f_40(v0, v21);
    int v22;
    v22 = v3.length;
    int v23;
    v23 = 0l;
    while (while_method_1(v22, v23)){
        unsigned long long v25;
        v25 = (unsigned long long)v23;
        unsigned long long v26;
        v26 = v25 * 32ull;
        unsigned long long v27;
        v27 = 112ull + v26;
        unsigned char * v28;
        v28 = (unsigned char *)(v0+v27);
        Union7 v30;
        v30 = v3[v23];
        f_41(v28, v30);
        v23 += 1l ;
    }
    int v32;
    v32 = 0l;
    while (while_method_0(v32)){
        unsigned long long v34;
        v34 = (unsigned long long)v32;
        unsigned long long v35;
        v35 = v34 * 4ull;
        unsigned long long v36;
        v36 = 1136ull + v35;
        unsigned char * v37;
        v37 = (unsigned char *)(v0+v36);
        Union2 v39;
        v39 = v4[v32];
        f_46(v37, v39);
        v32 += 1l ;
    }
    int v41;
    v41 = v5.tag;
    f_47(v0, v41);
    unsigned char * v42;
    v42 = (unsigned char *)(v0+1152ull);
    switch (v5.tag) {
        case 0: { // GameNotStarted
            return f_34(v42);
            break;
        }
        case 1: { // GameOver
            Union6 v44 = v5.case1.v0; bool v45 = v5.case1.v1; static_array<Union3,2l> v46 = v5.case1.v2; int v47 = v5.case1.v3; static_array<int,2l> v48 = v5.case1.v4; int v49 = v5.case1.v5;
            return f_37(v42, v44, v45, v46, v47, v48, v49);
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            Union6 v50 = v5.case2.v0; bool v51 = v5.case2.v1; static_array<Union3,2l> v52 = v5.case2.v2; int v53 = v5.case2.v3; static_array<int,2l> v54 = v5.case2.v4; int v55 = v5.case2.v5;
            return f_37(v42, v50, v51, v52, v53, v54, v55);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ inline bool while_method_9(){
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
        static_array_list<Union3,6l> v12; Union4 v13; static_array_list<Union7,32l> v14; static_array<Union2,2l> v15; Union8 v16;
        Tuple0 tmp10 = f_6(v0);
        v12 = tmp10.v0; v13 = tmp10.v1; v14 = tmp10.v2; v15 = tmp10.v3; v16 = tmp10.v4;
        Union8 & v17 = v16;
        static_array<Union2,2l> & v18 = v15;
        Union4 & v19 = v13;
        static_array_list<Union3,6l> & v20 = v12;
        static_array_list<Union7,32l> & v21 = v14;
        switch (v11.tag) {
            case 0: { // ActionSelected
                Union1 v70 = v11.case0.v0;
                Union4 v71 = v19;
                switch (v71.tag) {
                    case 0: { // None
                        printf("%s\n", "The game hasn't been started in ActionSelected.");
                        asm("exit;");
                        break;
                    }
                    case 1: { // Some
                        Union5 v72 = v71.case1.v0;
                        switch (v72.tag) {
                            case 2: { // Round
                                Union6 v73 = v72.case2.v0; bool v74 = v72.case2.v1; static_array<Union3,2l> v75 = v72.case2.v2; int v76 = v72.case2.v3; static_array<int,2l> v77 = v72.case2.v4; int v78 = v72.case2.v5;
                                Union5 v79;
                                v79 = Union5{Union5_3{v73, v74, v75, v76, v77, v78, v70}};
                                play_loop_20(v2, v3, v4, v5, v20, v19, v21, v18, v17, v79);
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
                static_array<Union2,2l> v69 = v11.case1.v0;
                v18 = v69;
                break;
            }
            case 2: { // StartGame
                static_array<Union2,2l> v22;
                Union2 v24;
                v24 = Union2{Union2_0{}};
                v22[0l] = v24;
                Union2 v26;
                v26 = Union2{Union2_1{}};
                v22[1l] = v26;
                static_array_list<Union7,32l> v28;
                v28 = static_array_list<Union7,32l>{};
                static_array_list<Union3,6l> v30;
                v30 = static_array_list<Union3,6l>{};
                v30.unsafe_set_length(6l);
                Union3 v32;
                v32 = Union3{Union3_1{}};
                v30[0l] = v32;
                Union3 v34;
                v34 = Union3{Union3_1{}};
                v30[1l] = v34;
                Union3 v36;
                v36 = Union3{Union3_2{}};
                v30[2l] = v36;
                Union3 v38;
                v38 = Union3{Union3_2{}};
                v30[3l] = v38;
                Union3 v40;
                v40 = Union3{Union3_0{}};
                v30[4l] = v40;
                Union3 v42;
                v42 = Union3{Union3_0{}};
                v30[5l] = v42;
                unsigned long long v44;
                v44 = clock64();
                curandStatePhilox4_32_10_t v45;
                curand_init(v44,0ull,0ull,&v45);
                int v46;
                v46 = v30.length;
                int v47;
                v47 = v46 - 1l;
                int v48;
                v48 = 0l;
                while (while_method_1(v47, v48)){
                    int v50;
                    v50 = v30.length;
                    int v51;
                    v51 = int_range_24(v50, v48, v45);
                    Union3 v52;
                    v52 = v30[v48];
                    Union3 v54;
                    v54 = v30[v51];
                    v30[v48] = v54;
                    v30[v51] = v52;
                    v48 += 1l ;
                }
                Union8 v66;
                v66 = Union8{Union8_0{}};
                v17 = v66;
                v18 = v22;
                Union4 v67;
                v67 = Union4{Union4_0{}};
                v19 = v67;
                v20 = v30;
                v21 = v28;
                Union5 v68;
                v68 = Union5{Union5_1{}};
                play_loop_20(v2, v3, v4, v5, v20, v19, v21, v18, v17, v68);
                break;
            }
            default: {
                assert("Invalid tag." && false);
            }
        }
        f_31(v0, v12, v13, v14, v15, v16);
    } else {
    }
    atomicAdd(&v6,-1l);
    while (while_method_9()){
        bool v81;
        v81 = 262144ull == v5;
        bool v82;
        v82 = v81 == false;
        if (v82){
            assert("The params needs to have matching offsets." && v81);
        } else {
        }
        bool v84;
        v84 = 98816ull == v3;
        bool v85;
        v85 = v84 == false;
        if (v85){
            assert("The outputs needs to have matching offsets." && v84);
        } else {
        }
        Union9 v87;
        v87 = Union9{Union9_0{}};
        Union10 v88; float * v89; int v90; int v91; int v92; float v93;
        Tuple7 tmp33 = noinline_run_21(v2, v4, v87);
        v88 = tmp33.v0; v89 = tmp33.v1; v90 = tmp33.v2; v91 = tmp33.v3; v92 = tmp33.v4; v93 = tmp33.v5;
        int v94 = v6;
        bool v95;
        v95 = v94 == 0l;
        if (v95){
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
import random
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
US0 = Union[US0_0, US0_1, US0_2]
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
def Closure0():
    def inner(v0 : object, v1 : object) -> object:
        v2 = cp.empty(16,dtype=cp.uint8)
        v3 = cp.empty(1200,dtype=cp.uint8)
        v4 = method0(v0)
        method7(v2, v4)
        del v4
        v5, v6, v7, v8, v9, v10, v11, v12, v13 = method14(v1)
        method43(v3, v5, v6, v7, v8, v9)
        del v5, v6, v7, v8, v9
        v16 = "{}\n"
        v17 = "Going to run the Leduc game kernel."
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
        v30, v31, v32, v33, v34 = method57(v3)
        del v3
        return method74(v30, v31, v32, v33, v34, v10, v11, v12, v13)
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
        v7 = static_array_list(32)
        v9 = static_array_list(6)
        v9.unsafe_set_length(6)
        v11 = US6_1()
        v9[0] = v11
        del v11
        v13 = US6_1()
        v9[1] = v13
        del v13
        v15 = US6_2()
        v9[2] = v15
        del v15
        v17 = US6_2()
        v9[3] = v17
        del v17
        v19 = US6_0()
        v9[4] = v19
        del v19
        v21 = US6_0()
        v9[5] = v21
        del v21
        v34 = v9.length
        v35 = v34 - 1
        del v34
        v36 = 0
        while method5(v35, v36):
            v38 = v9.length
            v39 = random.randrange(v36, v38)
            del v38
            v41 = v9[v36]
            v43 = v9[v39]
            v9[v36] = v43
            del v43
            v9[v39] = v41
            del v39, v41
            v36 += 1 
        del v35, v36
        v44 = cp.empty(262144,dtype=cp.uint8)
        v45 = cp.empty(98816,dtype=cp.uint8)
        v47 = v44[0:0+4*65536].view(cp.float32)
        v48 = cp.random.normal(0.0,1.0,65536,dtype=cp.float32) # type: ignore
        cp.copyto(v47[0:0+65536],v48[0:0+65536])
        del v47, v48
        v49 = US3_0()
        v50 = US7_0()
        v51 = 98816
        v52 = 262144
        return method74(v9, v49, v7, v1, v50, v45, v51, v44, v52)
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
                del v2, v9
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method0(v0 : object) -> US0:
    return method1(v0)
def method8(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[0:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method10(v0 : cp.ndarray) -> None:
    del v0
    return 
def method9(v0 : cp.ndarray, v1 : US1) -> None:
    v2 = v1.tag
    method8(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US1_0(): # Call
            del v1
            return method10(v4)
        case US1_1(): # Fold
            del v1
            return method10(v4)
        case US1_2(): # Raise
            del v1
            return method10(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method12(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method13(v0 : cp.ndarray, v1 : US2) -> None:
    v2 = v1.tag
    method8(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US2_0(): # Computer
            del v1
            return method10(v4)
        case US2_1(): # Human
            del v1
            return method10(v4)
        case US2_2(): # Random
            del v1
            return method10(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method11(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method12(v2):
        v4 = u64(v2)
        v5 = v4 * 4
        del v4
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v9 = v1[v2]
        method13(v7, v9)
        del v7, v9
        v2 += 1 
    del v0, v1, v2
    return 
def method7(v0 : cp.ndarray, v1 : US0) -> None:
    v2 = v1.tag
    method8(v0, v2)
    del v2
    v4 = v0[8:].view(cp.uint8)
    del v0
    match v1:
        case US0_0(v5): # ActionSelected
            del v1
            return method9(v4, v5)
        case US0_1(v6): # PlayerChanged
            del v1
            return method11(v4, v6)
        case US0_2(): # StartGame
            del v1
            return method10(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method19(v0 : object) -> US6:
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
def method18(v0 : object) -> static_array_list:
    v1 = len(v0) # type: ignore
    assert (6 >= v1), f'The length of the original object has to be greater than or equal to the static array dimension.\nExpected: 6\nGot: {v1} '
    del v1
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v2 = len(v0) # type: ignore
    v3 = 6 >= v2
    v4 = v3 == False
    if v4:
        v5 = "The type level dimension has to equal the value passed at runtime into create."
        assert v3, v5
        del v5
    else:
        pass
    del v3, v4
    v7 = static_array_list(6)
    v7.unsafe_set_length(v2)
    v8 = 0
    while method5(v2, v8):
        v10 = v0[v8]
        v11 = method19(v10)
        del v10
        v7[v8] = v11
        del v11
        v8 += 1 
    del v0, v2, v8
    return v7
def method23(v0 : object) -> US5:
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
            v6 = method19(v2)
            del v2
            return US5_1(v6)
        else:
            del v2, v5
            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
            del v1
            raise Exception("Error")
def method24(v0 : object) -> bool:
    assert isinstance(v0,bool), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
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
    while method5(v1, v7):
        v9 = v0[v7]
        v10 = method19(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method26(v0 : object) -> i32:
    assert isinstance(v0,i32), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method27(v0 : object) -> static_array:
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
        v10 = method26(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method22(v0 : object) -> Tuple[US5, bool, static_array, i32, static_array, i32]:
    v1 = v0["community_card"] # type: ignore
    v2 = method23(v1)
    del v1
    v3 = v0["is_button_s_first_move"] # type: ignore
    v4 = method24(v3)
    del v3
    v5 = v0["pl_card"] # type: ignore
    v6 = method25(v5)
    del v5
    v7 = v0["player_turn"] # type: ignore
    v8 = method26(v7)
    del v7
    v9 = v0["pot"] # type: ignore
    v10 = method27(v9)
    del v9
    v11 = v0["raises_left"] # type: ignore
    del v0
    v12 = method26(v11)
    del v11
    return v2, v4, v6, v8, v10, v12
def method28(v0 : object) -> Tuple[US5, bool, static_array, i32, static_array, i32, US1]:
    v1 = v0[0] # type: ignore
    v2, v3, v4, v5, v6, v7 = method22(v1)
    del v1
    v8 = v0[1] # type: ignore
    del v0
    v9 = method2(v8)
    del v8
    return v2, v3, v4, v5, v6, v7, v9
def method21(v0 : object) -> US4:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "ChanceCommunityCard" == v1
    if v3:
        del v1, v3
        v4, v5, v6, v7, v8, v9 = method22(v2)
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
                v14, v15, v16, v17, v18, v19 = method22(v2)
                del v2
                return US4_2(v14, v15, v16, v17, v18, v19)
            else:
                del v13
                v21 = "RoundWithAction" == v1
                if v21:
                    del v1, v21
                    v22, v23, v24, v25, v26, v27, v28 = method28(v2)
                    del v2
                    return US4_3(v22, v23, v24, v25, v26, v27, v28)
                else:
                    del v21
                    v30 = "TerminalCall" == v1
                    if v30:
                        del v1, v30
                        v31, v32, v33, v34, v35, v36 = method22(v2)
                        del v2
                        return US4_4(v31, v32, v33, v34, v35, v36)
                    else:
                        del v30
                        v38 = "TerminalFold" == v1
                        if v38:
                            del v1, v38
                            v39, v40, v41, v42, v43, v44 = method22(v2)
                            del v2
                            return US4_5(v39, v40, v41, v42, v43, v44)
                        else:
                            del v2, v38
                            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                            del v1
                            raise Exception("Error")
def method20(v0 : object) -> US3:
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
            v6 = method21(v2)
            del v2
            return US3_1(v6)
        else:
            del v2, v5
            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
            del v1
            raise Exception("Error")
def method17(v0 : object) -> Tuple[static_array_list, US3]:
    v1 = v0["deck"] # type: ignore
    v2 = method18(v1)
    del v1
    v3 = v0["game"] # type: ignore
    del v0
    v4 = method20(v3)
    del v3
    return v2, v4
def method32(v0 : object) -> Tuple[i32, US1]:
    v1 = v0[0] # type: ignore
    v2 = method26(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method2(v3)
    del v3
    return v2, v4
def method33(v0 : object) -> Tuple[i32, US6]:
    v1 = v0[0] # type: ignore
    v2 = method26(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method19(v3)
    del v3
    return v2, v4
def method34(v0 : object) -> Tuple[static_array, i32, i32]:
    v1 = v0["cards_shown"] # type: ignore
    v2 = method25(v1)
    del v1
    v3 = v0["chips_won"] # type: ignore
    v4 = method26(v3)
    del v3
    v5 = v0["winner_id"] # type: ignore
    del v0
    v6 = method26(v5)
    del v5
    return v2, v4, v6
def method31(v0 : object) -> US8:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "CommunityCardIs" == v1
    if v3:
        del v1, v3
        v4 = method19(v2)
        del v2
        return US8_0(v4)
    else:
        del v3
        v6 = "PlayerAction" == v1
        if v6:
            del v1, v6
            v7, v8 = method32(v2)
            del v2
            return US8_1(v7, v8)
        else:
            del v6
            v10 = "PlayerGotCard" == v1
            if v10:
                del v1, v10
                v11, v12 = method33(v2)
                del v2
                return US8_2(v11, v12)
            else:
                del v10
                v14 = "Showdown" == v1
                if v14:
                    del v1, v14
                    v15, v16, v17 = method34(v2)
                    del v2
                    return US8_3(v15, v16, v17)
                else:
                    del v2, v14
                    raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                    del v1
                    raise Exception("Error")
def method30(v0 : object) -> static_array_list:
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
        v11 = method31(v10)
        del v10
        v7[v8] = v11
        del v11
        v8 += 1 
    del v0, v2, v8
    return v7
def method35(v0 : object) -> US7:
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
            v6, v7, v8, v9, v10, v11 = method22(v2)
            del v2
            return US7_1(v6, v7, v8, v9, v10, v11)
        else:
            del v5
            v13 = "WaitingForActionFromPlayerId" == v1
            if v13:
                del v1, v13
                v14, v15, v16, v17, v18, v19 = method22(v2)
                del v2
                return US7_2(v14, v15, v16, v17, v18, v19)
            else:
                del v2, v13
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method29(v0 : object) -> Tuple[static_array_list, static_array, US7]:
    v1 = v0["messages"] # type: ignore
    v2 = method30(v1)
    del v1
    v3 = v0["pl_type"] # type: ignore
    v4 = method4(v3)
    del v3
    v5 = v0["ui_game_state"] # type: ignore
    del v0
    v6 = method35(v5)
    del v5
    return v2, v4, v6
def method16(v0 : object) -> Tuple[static_array_list, US3, static_array_list, static_array, US7]:
    v1 = v0["private"] # type: ignore
    v2, v3 = method17(v1)
    del v1
    v4 = v0["public"] # type: ignore
    del v0
    v5, v6, v7 = method29(v4)
    del v4
    return v2, v3, v5, v6, v7
def method41(v0 : object) -> cp.ndarray:
    assert isinstance(v0,cp.ndarray), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method40(v0 : object) -> cp.ndarray:
    v1 = method41(v0)
    del v0
    return v1
def method42(v0 : object) -> u64:
    assert isinstance(v0,u64), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method39(v0 : object) -> Tuple[cp.ndarray, u64]:
    v1 = v0[0] # type: ignore
    v2 = method40(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method42(v3)
    del v3
    return v2, v4
def method38(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["output"] # type: ignore
    v2, v3 = method39(v1)
    del v1
    v4 = v0["param"] # type: ignore
    del v0
    v5, v6 = method39(v4)
    del v4
    return v2, v3, v5, v6
def method37(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1, v2, v3, v4 = method38(v0)
    del v0
    return v1, v2, v3, v4
def method36(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["model_data"] # type: ignore
    del v0
    v2, v3, v4, v5 = method37(v1)
    del v1
    return v2, v3, v4, v5
def method15(v0 : object) -> Tuple[static_array_list, US3, static_array_list, static_array, US7, cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["game"] # type: ignore
    v2, v3, v4, v5, v6 = method16(v1)
    del v1
    v7 = v0["neural"] # type: ignore
    del v0
    v8, v9, v10, v11 = method36(v7)
    del v7
    return v2, v3, v4, v5, v6, v8, v9, v10, v11
def method14(v0 : object) -> Tuple[static_array_list, US3, static_array_list, static_array, US7, cp.ndarray, u64, cp.ndarray, u64]:
    return method15(v0)
def method44(v0 : cp.ndarray, v1 : US6) -> None:
    v2 = v1.tag
    method8(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US6_0(): # Jack
            del v1
            return method10(v4)
        case US6_1(): # King
            del v1
            return method10(v4)
        case US6_2(): # Queen
            del v1
            return method10(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method45(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[28:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method47(v0 : cp.ndarray, v1 : US5, v2 : bool, v3 : static_array, v4 : i32, v5 : static_array, v6 : i32) -> None:
    v7 = v1.tag
    method8(v0, v7)
    del v7
    v9 = v0[4:].view(cp.uint8)
    match v1:
        case US5_0(): # None
            method10(v9)
        case US5_1(v10): # Some
            method44(v9, v10)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v1, v9
    v12 = v0[8:].view(cp.bool_)
    v12[0] = v2
    del v2, v12
    v13 = 0
    while method12(v13):
        v15 = u64(v13)
        v16 = v15 * 4
        del v15
        v17 = 12 + v16
        del v16
        v19 = v0[v17:].view(cp.uint8)
        del v17
        v21 = v3[v13]
        method44(v19, v21)
        del v19, v21
        v13 += 1 
    del v3, v13
    v23 = v0[20:].view(cp.int32)
    v23[0] = v4
    del v4, v23
    v24 = 0
    while method12(v24):
        v26 = u64(v24)
        v27 = v26 * 4
        del v26
        v28 = 24 + v27
        del v27
        v30 = v0[v28:].view(cp.uint8)
        del v28
        v32 = v5[v24]
        method8(v30, v32)
        del v30, v32
        v24 += 1 
    del v5, v24
    v34 = v0[32:].view(cp.int32)
    del v0
    v34[0] = v6
    del v6, v34
    return 
def method49(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[36:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method48(v0 : cp.ndarray, v1 : US5, v2 : bool, v3 : static_array, v4 : i32, v5 : static_array, v6 : i32, v7 : US1) -> None:
    v8 = v1.tag
    method8(v0, v8)
    del v8
    v10 = v0[4:].view(cp.uint8)
    match v1:
        case US5_0(): # None
            method10(v10)
        case US5_1(v11): # Some
            method44(v10, v11)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v1, v10
    v13 = v0[8:].view(cp.bool_)
    v13[0] = v2
    del v2, v13
    v14 = 0
    while method12(v14):
        v16 = u64(v14)
        v17 = v16 * 4
        del v16
        v18 = 12 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v22 = v3[v14]
        method44(v20, v22)
        del v20, v22
        v14 += 1 
    del v3, v14
    v24 = v0[20:].view(cp.int32)
    v24[0] = v4
    del v4, v24
    v25 = 0
    while method12(v25):
        v27 = u64(v25)
        v28 = v27 * 4
        del v27
        v29 = 24 + v28
        del v28
        v31 = v0[v29:].view(cp.uint8)
        del v29
        v33 = v5[v25]
        method8(v31, v33)
        del v31, v33
        v25 += 1 
    del v5, v25
    v35 = v0[32:].view(cp.int32)
    v35[0] = v6
    del v6, v35
    v36 = v7.tag
    method49(v0, v36)
    del v36
    v38 = v0[40:].view(cp.uint8)
    del v0
    match v7:
        case US1_0(): # Call
            del v7
            return method10(v38)
        case US1_1(): # Fold
            del v7
            return method10(v38)
        case US1_2(): # Raise
            del v7
            return method10(v38)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method46(v0 : cp.ndarray, v1 : US4) -> None:
    v2 = v1.tag
    method8(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US4_0(v5, v6, v7, v8, v9, v10): # ChanceCommunityCard
            del v1
            return method47(v4, v5, v6, v7, v8, v9, v10)
        case US4_1(): # ChanceInit
            del v1
            return method10(v4)
        case US4_2(v11, v12, v13, v14, v15, v16): # Round
            del v1
            return method47(v4, v11, v12, v13, v14, v15, v16)
        case US4_3(v17, v18, v19, v20, v21, v22, v23): # RoundWithAction
            del v1
            return method48(v4, v17, v18, v19, v20, v21, v22, v23)
        case US4_4(v24, v25, v26, v27, v28, v29): # TerminalCall
            del v1
            return method47(v4, v24, v25, v26, v27, v28, v29)
        case US4_5(v30, v31, v32, v33, v34, v35): # TerminalFold
            del v1
            return method47(v4, v30, v31, v32, v33, v34, v35)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method50(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[96:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method53(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[4:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method52(v0 : cp.ndarray, v1 : i32, v2 : US1) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = v2.tag
    method53(v0, v5)
    del v5
    v7 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US1_0(): # Call
            del v2
            return method10(v7)
        case US1_1(): # Fold
            del v2
            return method10(v7)
        case US1_2(): # Raise
            del v2
            return method10(v7)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method54(v0 : cp.ndarray, v1 : i32, v2 : US6) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = v2.tag
    method53(v0, v5)
    del v5
    v7 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US6_0(): # Jack
            del v2
            return method10(v7)
        case US6_1(): # King
            del v2
            return method10(v7)
        case US6_2(): # Queen
            del v2
            return method10(v7)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method55(v0 : cp.ndarray, v1 : static_array, v2 : i32, v3 : i32) -> None:
    v4 = 0
    while method12(v4):
        v6 = u64(v4)
        v7 = v6 * 4
        del v6
        v9 = v0[v7:].view(cp.uint8)
        del v7
        v11 = v1[v4]
        method44(v9, v11)
        del v9, v11
        v4 += 1 
    del v1, v4
    v13 = v0[8:].view(cp.int32)
    v13[0] = v2
    del v2, v13
    v15 = v0[12:].view(cp.int32)
    del v0
    v15[0] = v3
    del v3, v15
    return 
def method51(v0 : cp.ndarray, v1 : US8) -> None:
    v2 = v1.tag
    method8(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US8_0(v5): # CommunityCardIs
            del v1
            return method44(v4, v5)
        case US8_1(v6, v7): # PlayerAction
            del v1
            return method52(v4, v6, v7)
        case US8_2(v8, v9): # PlayerGotCard
            del v1
            return method54(v4, v8, v9)
        case US8_3(v10, v11, v12): # Showdown
            del v1
            return method55(v4, v10, v11, v12)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method56(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[1144:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method43(v0 : cp.ndarray, v1 : static_array_list, v2 : US3, v3 : static_array_list, v4 : static_array, v5 : US7) -> None:
    v6 = v1.length
    method8(v0, v6)
    del v6
    v7 = v1.length
    v8 = 0
    while method5(v7, v8):
        v10 = u64(v8)
        v11 = v10 * 4
        del v10
        v12 = 4 + v11
        del v11
        v14 = v0[v12:].view(cp.uint8)
        del v12
        v16 = v1[v8]
        method44(v14, v16)
        del v14, v16
        v8 += 1 
    del v1, v7, v8
    v17 = v2.tag
    method45(v0, v17)
    del v17
    v19 = v0[32:].view(cp.uint8)
    match v2:
        case US3_0(): # None
            method10(v19)
        case US3_1(v20): # Some
            method46(v19, v20)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v2, v19
    v21 = v3.length
    method50(v0, v21)
    del v21
    v22 = v3.length
    v23 = 0
    while method5(v22, v23):
        v25 = u64(v23)
        v26 = v25 * 32
        del v25
        v27 = 112 + v26
        del v26
        v29 = v0[v27:].view(cp.uint8)
        del v27
        v31 = v3[v23]
        method51(v29, v31)
        del v29, v31
        v23 += 1 
    del v3, v22, v23
    v32 = 0
    while method12(v32):
        v34 = u64(v32)
        v35 = v34 * 4
        del v34
        v36 = 1136 + v35
        del v35
        v38 = v0[v36:].view(cp.uint8)
        del v36
        v40 = v4[v32]
        method13(v38, v40)
        del v38, v40
        v32 += 1 
    del v4, v32
    v41 = v5.tag
    method56(v0, v41)
    del v41
    v43 = v0[1152:].view(cp.uint8)
    del v0
    match v5:
        case US7_0(): # GameNotStarted
            del v5
            return method10(v43)
        case US7_1(v44, v45, v46, v47, v48, v49): # GameOver
            del v5
            return method47(v43, v44, v45, v46, v47, v48, v49)
        case US7_2(v50, v51, v52, v53, v54, v55): # WaitingForActionFromPlayerId
            del v5
            return method47(v43, v50, v51, v52, v53, v54, v55)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method58(v0 : cp.ndarray) -> i32:
    v2 = v0[0:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method60(v0 : cp.ndarray) -> None:
    del v0
    return 
def method59(v0 : cp.ndarray) -> US6:
    v1 = method58(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method60(v3)
        del v3
        return US6_0()
    elif v1 == 1:
        del v1
        method60(v3)
        del v3
        return US6_1()
    elif v1 == 2:
        del v1
        method60(v3)
        del v3
        return US6_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method61(v0 : cp.ndarray) -> i32:
    v2 = v0[28:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method63(v0 : cp.ndarray) -> Tuple[US5, bool, static_array, i32, static_array, i32]:
    v1 = method58(v0)
    v3 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method60(v3)
        v8 = US5_0()
    elif v1 == 1:
        v6 = method59(v3)
        v8 = US5_1(v6)
    else:
        raise Exception("Invalid tag.")
    del v1, v3
    v10 = v0[8:].view(cp.bool_)
    v11 = v10[0].item()
    del v10
    v13 = static_array(2)
    v14 = 0
    while method12(v14):
        v16 = u64(v14)
        v17 = v16 * 4
        del v16
        v18 = 12 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v21 = method59(v20)
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
    while method12(v27):
        v29 = u64(v27)
        v30 = v29 * 4
        del v29
        v31 = 24 + v30
        del v30
        v33 = v0[v31:].view(cp.uint8)
        del v31
        v34 = method58(v33)
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
def method65(v0 : cp.ndarray) -> i32:
    v2 = v0[36:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method64(v0 : cp.ndarray) -> Tuple[US5, bool, static_array, i32, static_array, i32, US1]:
    v1 = method58(v0)
    v3 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method60(v3)
        v8 = US5_0()
    elif v1 == 1:
        v6 = method59(v3)
        v8 = US5_1(v6)
    else:
        raise Exception("Invalid tag.")
    del v1, v3
    v10 = v0[8:].view(cp.bool_)
    v11 = v10[0].item()
    del v10
    v13 = static_array(2)
    v14 = 0
    while method12(v14):
        v16 = u64(v14)
        v17 = v16 * 4
        del v16
        v18 = 12 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v21 = method59(v20)
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
    while method12(v27):
        v29 = u64(v27)
        v30 = v29 * 4
        del v29
        v31 = 24 + v30
        del v30
        v33 = v0[v31:].view(cp.uint8)
        del v31
        v34 = method58(v33)
        del v33
        v26[v27] = v34
        del v34
        v27 += 1 
    del v27
    v36 = v0[32:].view(cp.int32)
    v37 = v36[0].item()
    del v36
    v38 = method65(v0)
    v40 = v0[40:].view(cp.uint8)
    del v0
    if v38 == 0:
        method60(v40)
        v45 = US1_0()
    elif v38 == 1:
        method60(v40)
        v45 = US1_1()
    elif v38 == 2:
        method60(v40)
        v45 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v38, v40
    return v8, v11, v13, v24, v26, v37, v45
def method62(v0 : cp.ndarray) -> US4:
    v1 = method58(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5, v6, v7, v8, v9, v10 = method63(v3)
        del v3
        return US4_0(v5, v6, v7, v8, v9, v10)
    elif v1 == 1:
        del v1
        method60(v3)
        del v3
        return US4_1()
    elif v1 == 2:
        del v1
        v13, v14, v15, v16, v17, v18 = method63(v3)
        del v3
        return US4_2(v13, v14, v15, v16, v17, v18)
    elif v1 == 3:
        del v1
        v20, v21, v22, v23, v24, v25, v26 = method64(v3)
        del v3
        return US4_3(v20, v21, v22, v23, v24, v25, v26)
    elif v1 == 4:
        del v1
        v28, v29, v30, v31, v32, v33 = method63(v3)
        del v3
        return US4_4(v28, v29, v30, v31, v32, v33)
    elif v1 == 5:
        del v1
        v35, v36, v37, v38, v39, v40 = method63(v3)
        del v3
        return US4_5(v35, v36, v37, v38, v39, v40)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method66(v0 : cp.ndarray) -> i32:
    v2 = v0[96:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method69(v0 : cp.ndarray) -> i32:
    v2 = v0[4:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method68(v0 : cp.ndarray) -> Tuple[i32, US1]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method69(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method60(v6)
        v11 = US1_0()
    elif v4 == 1:
        method60(v6)
        v11 = US1_1()
    elif v4 == 2:
        method60(v6)
        v11 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v11
def method70(v0 : cp.ndarray) -> Tuple[i32, US6]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method69(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method60(v6)
        v11 = US6_0()
    elif v4 == 1:
        method60(v6)
        v11 = US6_1()
    elif v4 == 2:
        method60(v6)
        v11 = US6_2()
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v11
def method71(v0 : cp.ndarray) -> Tuple[static_array, i32, i32]:
    v2 = static_array(2)
    v3 = 0
    while method12(v3):
        v5 = u64(v3)
        v6 = v5 * 4
        del v5
        v8 = v0[v6:].view(cp.uint8)
        del v6
        v9 = method59(v8)
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
def method67(v0 : cp.ndarray) -> US8:
    v1 = method58(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5 = method59(v3)
        del v3
        return US8_0(v5)
    elif v1 == 1:
        del v1
        v7, v8 = method68(v3)
        del v3
        return US8_1(v7, v8)
    elif v1 == 2:
        del v1
        v10, v11 = method70(v3)
        del v3
        return US8_2(v10, v11)
    elif v1 == 3:
        del v1
        v13, v14, v15 = method71(v3)
        del v3
        return US8_3(v13, v14, v15)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method72(v0 : cp.ndarray) -> US2:
    v1 = method58(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method60(v3)
        del v3
        return US2_0()
    elif v1 == 1:
        del v1
        method60(v3)
        del v3
        return US2_1()
    elif v1 == 2:
        del v1
        method60(v3)
        del v3
        return US2_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method73(v0 : cp.ndarray) -> i32:
    v2 = v0[1144:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method57(v0 : cp.ndarray) -> Tuple[static_array_list, US3, static_array_list, static_array, US7]:
    v2 = static_array_list(6)
    v3 = method58(v0)
    v2.unsafe_set_length(v3)
    del v3
    v4 = v2.length
    v5 = 0
    while method5(v4, v5):
        v7 = u64(v5)
        v8 = v7 * 4
        del v7
        v9 = 4 + v8
        del v8
        v11 = v0[v9:].view(cp.uint8)
        del v9
        v12 = method59(v11)
        del v11
        v2[v5] = v12
        del v12
        v5 += 1 
    del v4, v5
    v13 = method61(v0)
    v15 = v0[32:].view(cp.uint8)
    if v13 == 0:
        method60(v15)
        v20 = US3_0()
    elif v13 == 1:
        v18 = method62(v15)
        v20 = US3_1(v18)
    else:
        raise Exception("Invalid tag.")
    del v13, v15
    v22 = static_array_list(32)
    v23 = method66(v0)
    v22.unsafe_set_length(v23)
    del v23
    v24 = v22.length
    v25 = 0
    while method5(v24, v25):
        v27 = u64(v25)
        v28 = v27 * 32
        del v27
        v29 = 112 + v28
        del v28
        v31 = v0[v29:].view(cp.uint8)
        del v29
        v32 = method67(v31)
        del v31
        v22[v25] = v32
        del v32
        v25 += 1 
    del v24, v25
    v34 = static_array(2)
    v35 = 0
    while method12(v35):
        v37 = u64(v35)
        v38 = v37 * 4
        del v37
        v39 = 1136 + v38
        del v38
        v41 = v0[v39:].view(cp.uint8)
        del v39
        v42 = method72(v41)
        del v41
        v34[v35] = v42
        del v42
        v35 += 1 
    del v35
    v43 = method73(v0)
    v45 = v0[1152:].view(cp.uint8)
    del v0
    if v43 == 0:
        method60(v45)
        v62 = US7_0()
    elif v43 == 1:
        v48, v49, v50, v51, v52, v53 = method63(v45)
        v62 = US7_1(v48, v49, v50, v51, v52, v53)
    elif v43 == 2:
        v55, v56, v57, v58, v59, v60 = method63(v45)
        v62 = US7_2(v55, v56, v57, v58, v59, v60)
    else:
        raise Exception("Invalid tag.")
    del v43, v45
    return v2, v20, v22, v34, v62
def method80() -> object:
    v0 = []
    return v0
def method79(v0 : US6) -> object:
    match v0:
        case US6_0(): # Jack
            del v0
            v1 = method80()
            v2 = "Jack"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US6_1(): # King
            del v0
            v4 = method80()
            v5 = "King"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US6_2(): # Queen
            del v0
            v7 = method80()
            v8 = "Queen"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method78(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method5(v2, v3):
        v6 = v0[v3]
        v7 = method79(v6)
        del v6
        v1.append(v7)
        del v7
        v3 += 1 
    del v0, v2, v3
    return v1
def method84(v0 : US5) -> object:
    match v0:
        case US5_0(): # None
            del v0
            v1 = method80()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US5_1(v4): # Some
            del v0
            v5 = method79(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method85(v0 : bool) -> object:
    v1 = v0
    del v0
    return v1
def method86(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method12(v2):
        v5 = v0[v2]
        v6 = method79(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method87(v0 : i32) -> object:
    v1 = v0
    del v0
    return v1
def method88(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method12(v2):
        v5 = v0[v2]
        v6 = method87(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method83(v0 : US5, v1 : bool, v2 : static_array, v3 : i32, v4 : static_array, v5 : i32) -> object:
    v6 = method84(v0)
    del v0
    v7 = method85(v1)
    del v1
    v8 = method86(v2)
    del v2
    v9 = method87(v3)
    del v3
    v10 = method88(v4)
    del v4
    v11 = method87(v5)
    del v5
    v12 = {'community_card': v6, 'is_button_s_first_move': v7, 'pl_card': v8, 'player_turn': v9, 'pot': v10, 'raises_left': v11}
    del v6, v7, v8, v9, v10, v11
    return v12
def method90(v0 : US1) -> object:
    match v0:
        case US1_0(): # Call
            del v0
            v1 = method80()
            v2 = "Call"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US1_1(): # Fold
            del v0
            v4 = method80()
            v5 = "Fold"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US1_2(): # Raise
            del v0
            v7 = method80()
            v8 = "Raise"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method89(v0 : US5, v1 : bool, v2 : static_array, v3 : i32, v4 : static_array, v5 : i32, v6 : US1) -> object:
    v7 = []
    v8 = method83(v0, v1, v2, v3, v4, v5)
    del v0, v1, v2, v3, v4, v5
    v7.append(v8)
    del v8
    v9 = method90(v6)
    del v6
    v7.append(v9)
    del v9
    v10 = v7
    del v7
    return v10
def method82(v0 : US4) -> object:
    match v0:
        case US4_0(v1, v2, v3, v4, v5, v6): # ChanceCommunityCard
            del v0
            v7 = method83(v1, v2, v3, v4, v5, v6)
            del v1, v2, v3, v4, v5, v6
            v8 = "ChanceCommunityCard"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US4_1(): # ChanceInit
            del v0
            v10 = method80()
            v11 = "ChanceInit"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US4_2(v13, v14, v15, v16, v17, v18): # Round
            del v0
            v19 = method83(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "Round"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case US4_3(v22, v23, v24, v25, v26, v27, v28): # RoundWithAction
            del v0
            v29 = method89(v22, v23, v24, v25, v26, v27, v28)
            del v22, v23, v24, v25, v26, v27, v28
            v30 = "RoundWithAction"
            v31 = [v30,v29]
            del v29, v30
            return v31
        case US4_4(v32, v33, v34, v35, v36, v37): # TerminalCall
            del v0
            v38 = method83(v32, v33, v34, v35, v36, v37)
            del v32, v33, v34, v35, v36, v37
            v39 = "TerminalCall"
            v40 = [v39,v38]
            del v38, v39
            return v40
        case US4_5(v41, v42, v43, v44, v45, v46): # TerminalFold
            del v0
            v47 = method83(v41, v42, v43, v44, v45, v46)
            del v41, v42, v43, v44, v45, v46
            v48 = "TerminalFold"
            v49 = [v48,v47]
            del v47, v48
            return v49
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method81(v0 : US3) -> object:
    match v0:
        case US3_0(): # None
            del v0
            v1 = method80()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US3_1(v4): # Some
            del v0
            v5 = method82(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method77(v0 : static_array_list, v1 : US3) -> object:
    v2 = method78(v0)
    del v0
    v3 = method81(v1)
    del v1
    v4 = {'deck': v2, 'game': v3}
    del v2, v3
    return v4
def method94(v0 : i32, v1 : US1) -> object:
    v2 = []
    v3 = method87(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method90(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method95(v0 : i32, v1 : US6) -> object:
    v2 = []
    v3 = method87(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method79(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method96(v0 : static_array, v1 : i32, v2 : i32) -> object:
    v3 = method86(v0)
    del v0
    v4 = method87(v1)
    del v1
    v5 = method87(v2)
    del v2
    v6 = {'cards_shown': v3, 'chips_won': v4, 'winner_id': v5}
    del v3, v4, v5
    return v6
def method93(v0 : US8) -> object:
    match v0:
        case US8_0(v1): # CommunityCardIs
            del v0
            v2 = method79(v1)
            del v1
            v3 = "CommunityCardIs"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US8_1(v5, v6): # PlayerAction
            del v0
            v7 = method94(v5, v6)
            del v5, v6
            v8 = "PlayerAction"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US8_2(v10, v11): # PlayerGotCard
            del v0
            v12 = method95(v10, v11)
            del v10, v11
            v13 = "PlayerGotCard"
            v14 = [v13,v12]
            del v12, v13
            return v14
        case US8_3(v15, v16, v17): # Showdown
            del v0
            v18 = method96(v15, v16, v17)
            del v15, v16, v17
            v19 = "Showdown"
            v20 = [v19,v18]
            del v18, v19
            return v20
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method92(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method5(v2, v3):
        v6 = v0[v3]
        v7 = method93(v6)
        del v6
        v1.append(v7)
        del v7
        v3 += 1 
    del v0, v2, v3
    return v1
def method98(v0 : US2) -> object:
    match v0:
        case US2_0(): # Computer
            del v0
            v1 = method80()
            v2 = "Computer"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US2_1(): # Human
            del v0
            v4 = method80()
            v5 = "Human"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US2_2(): # Random
            del v0
            v7 = method80()
            v8 = "Random"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method97(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method12(v2):
        v5 = v0[v2]
        v6 = method98(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method99(v0 : US7) -> object:
    match v0:
        case US7_0(): # GameNotStarted
            del v0
            v1 = method80()
            v2 = "GameNotStarted"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US7_1(v4, v5, v6, v7, v8, v9): # GameOver
            del v0
            v10 = method83(v4, v5, v6, v7, v8, v9)
            del v4, v5, v6, v7, v8, v9
            v11 = "GameOver"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US7_2(v13, v14, v15, v16, v17, v18): # WaitingForActionFromPlayerId
            del v0
            v19 = method83(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "WaitingForActionFromPlayerId"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method91(v0 : static_array_list, v1 : static_array, v2 : US7) -> object:
    v3 = method92(v0)
    del v0
    v4 = method97(v1)
    del v1
    v5 = method99(v2)
    del v2
    v6 = {'messages': v3, 'pl_type': v4, 'ui_game_state': v5}
    del v3, v4, v5
    return v6
def method76(v0 : static_array_list, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7) -> object:
    v5 = method77(v0, v1)
    del v0, v1
    v6 = method91(v2, v3, v4)
    del v2, v3, v4
    v7 = {'private': v5, 'public': v6}
    del v5, v6
    return v7
def method105(v0 : cp.ndarray) -> object:
    v1 = v0
    del v0
    return v1
def method104(v0 : cp.ndarray) -> object:
    return method105(v0)
def method106(v0 : u64) -> object:
    v1 = v0
    del v0
    return v1
def method103(v0 : cp.ndarray, v1 : u64) -> object:
    v2 = []
    v3 = method104(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method106(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method102(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method103(v0, v1)
    del v0, v1
    v5 = method103(v2, v3)
    del v2, v3
    v6 = {'output': v4, 'param': v5}
    del v4, v5
    return v6
def method101(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    return method102(v0, v1, v2, v3)
def method100(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method101(v0, v1, v2, v3)
    del v0, v1, v2, v3
    v5 = {'model_data': v4}
    del v4
    return v5
def method75(v0 : static_array_list, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64) -> object:
    v9 = method76(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    v10 = method100(v5, v6, v7, v8)
    del v5, v6, v7, v8
    v11 = {'game': v9, 'neural': v10}
    del v9, v10
    return v11
def method74(v0 : static_array_list, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64) -> object:
    v9 = method75(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8
    return v9
def main():
    v0 = Closure0()
    v1 = Closure1()
    v2 = collections.namedtuple("Leduc_Game",['event_loop_gpu', 'init'])(v0, v1)
    del v0, v1
    return v2

if __name__ == '__main__': print(main())
