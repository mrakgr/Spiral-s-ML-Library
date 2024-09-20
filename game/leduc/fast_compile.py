kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <mma.h>
using namespace nvcuda;
#include <cuda/pipeline>
#include <cooperative_groups/memcpy_async.h>
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

struct Union0;
struct Union2;
struct Union3;
struct Union1;
struct StackMut0;
struct Union5;
struct Union4;
struct Union6;
struct Tuple0;
__device__ unsigned int loop_2(unsigned int v0, curandStatePhilox4_32_10_t & v1);
__device__ Tuple0 draw_card_1(curandStatePhilox4_32_10_t & v0, unsigned int v1);
struct Tuple1;
struct Union7;
struct Union8;
struct Union9;
__device__ void method_3(unsigned int * v0, int v1, float * v2);
struct Tuple2;
struct Tuple3;
struct Tuple4;
struct Tuple5;
__device__ Tuple2 method_4(curandStatePhilox4_32_10_t & v0, int * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int v9, int v10);
__device__ float method_5(int * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int v8, int v9, int v10);
struct Union10;
__device__ int int_range_6(int v0, int v1, curandStatePhilox4_32_10_t & v2);
struct Union11;
__device__ int tag_8(Union2 v0);
__device__ bool is_pair_9(int v0, int v1);
__device__ Tuple1 order_10(int v0, int v1);
__device__ Union11 compare_hands_7(Union5 v0, bool v1, static_array<Union2,2> v2, int v3, static_array<int,2> v4, int v5);
__device__ void method_0(unsigned char * v0, unsigned char * v1, StackMut0 & v2, int v3, Union4 v4);
struct Union0_0 { // Computer
};
struct Union0_1 { // Human
};
struct Union0_2 { // Random
};
struct Union0 {
    union {
        Union0_0 case0; // Computer
        Union0_1 case1; // Human
        Union0_2 case2; // Random
    };
    unsigned char tag{255};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // Computer
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // Human
    __device__ Union0(Union0_2 t) : tag(2), case2(t) {} // Random
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // Computer
            case 1: new (&this->case1) Union0_1(x.case1); break; // Human
            case 2: new (&this->case2) Union0_2(x.case2); break; // Random
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // Computer
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // Human
            case 2: new (&this->case2) Union0_2(std::move(x.case2)); break; // Random
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Computer
                case 1: this->case1 = x.case1; break; // Human
                case 2: this->case2 = x.case2; break; // Random
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
                case 0: this->case0 = std::move(x.case0); break; // Computer
                case 1: this->case1 = std::move(x.case1); break; // Human
                case 2: this->case2 = std::move(x.case2); break; // Random
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // Computer
            case 1: this->case1.~Union0_1(); break; // Human
            case 2: this->case2.~Union0_2(); break; // Random
        }
        this->tag = 255;
    }
};
struct Union2_0 { // Jack
};
struct Union2_1 { // King
};
struct Union2_2 { // Queen
};
struct Union2 {
    union {
        Union2_0 case0; // Jack
        Union2_1 case1; // King
        Union2_2 case2; // Queen
    };
    unsigned char tag{255};
    __device__ Union2() {}
    __device__ Union2(Union2_0 t) : tag(0), case0(t) {} // Jack
    __device__ Union2(Union2_1 t) : tag(1), case1(t) {} // King
    __device__ Union2(Union2_2 t) : tag(2), case2(t) {} // Queen
    __device__ Union2(Union2 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(x.case0); break; // Jack
            case 1: new (&this->case1) Union2_1(x.case1); break; // King
            case 2: new (&this->case2) Union2_2(x.case2); break; // Queen
        }
    }
    __device__ Union2(Union2 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(std::move(x.case0)); break; // Jack
            case 1: new (&this->case1) Union2_1(std::move(x.case1)); break; // King
            case 2: new (&this->case2) Union2_2(std::move(x.case2)); break; // Queen
        }
    }
    __device__ Union2 & operator=(Union2 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Jack
                case 1: this->case1 = x.case1; break; // King
                case 2: this->case2 = x.case2; break; // Queen
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
                case 0: this->case0 = std::move(x.case0); break; // Jack
                case 1: this->case1 = std::move(x.case1); break; // King
                case 2: this->case2 = std::move(x.case2); break; // Queen
            }
        } else {
            this->~Union2();
            new (this) Union2{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union2() {
        switch(this->tag){
            case 0: this->case0.~Union2_0(); break; // Jack
            case 1: this->case1.~Union2_1(); break; // King
            case 2: this->case2.~Union2_2(); break; // Queen
        }
        this->tag = 255;
    }
};
struct Union3_0 { // Call
};
struct Union3_1 { // Fold
};
struct Union3_2 { // Raise
};
struct Union3 {
    union {
        Union3_0 case0; // Call
        Union3_1 case1; // Fold
        Union3_2 case2; // Raise
    };
    unsigned char tag{255};
    __device__ Union3() {}
    __device__ Union3(Union3_0 t) : tag(0), case0(t) {} // Call
    __device__ Union3(Union3_1 t) : tag(1), case1(t) {} // Fold
    __device__ Union3(Union3_2 t) : tag(2), case2(t) {} // Raise
    __device__ Union3(Union3 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(x.case0); break; // Call
            case 1: new (&this->case1) Union3_1(x.case1); break; // Fold
            case 2: new (&this->case2) Union3_2(x.case2); break; // Raise
        }
    }
    __device__ Union3(Union3 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(std::move(x.case0)); break; // Call
            case 1: new (&this->case1) Union3_1(std::move(x.case1)); break; // Fold
            case 2: new (&this->case2) Union3_2(std::move(x.case2)); break; // Raise
        }
    }
    __device__ Union3 & operator=(Union3 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Call
                case 1: this->case1 = x.case1; break; // Fold
                case 2: this->case2 = x.case2; break; // Raise
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
                case 0: this->case0 = std::move(x.case0); break; // Call
                case 1: this->case1 = std::move(x.case1); break; // Fold
                case 2: this->case2 = std::move(x.case2); break; // Raise
            }
        } else {
            this->~Union3();
            new (this) Union3{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union3() {
        switch(this->tag){
            case 0: this->case0.~Union3_0(); break; // Call
            case 1: this->case1.~Union3_1(); break; // Fold
            case 2: this->case2.~Union3_2(); break; // Raise
        }
        this->tag = 255;
    }
};
struct Union1_0 { // CommunityCardIs
    Union2 v0;
    __device__ Union1_0(Union2 t0) : v0(t0) {}
    __device__ Union1_0() = delete;
};
struct Union1_1 { // PlayerAction
    Union3 v1;
    int v0;
    __device__ Union1_1(int t0, Union3 t1) : v0(t0), v1(t1) {}
    __device__ Union1_1() = delete;
};
struct Union1_2 { // PlayerGotCard
    Union2 v1;
    int v0;
    __device__ Union1_2(int t0, Union2 t1) : v0(t0), v1(t1) {}
    __device__ Union1_2() = delete;
};
struct Union1_3 { // Showdown
    static_array<Union2,2> v0;
    int v1;
    int v2;
    __device__ Union1_3(static_array<Union2,2> t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
    __device__ Union1_3() = delete;
};
struct Union1 {
    union {
        Union1_0 case0; // CommunityCardIs
        Union1_1 case1; // PlayerAction
        Union1_2 case2; // PlayerGotCard
        Union1_3 case3; // Showdown
    };
    unsigned char tag{255};
    __device__ Union1() {}
    __device__ Union1(Union1_0 t) : tag(0), case0(t) {} // CommunityCardIs
    __device__ Union1(Union1_1 t) : tag(1), case1(t) {} // PlayerAction
    __device__ Union1(Union1_2 t) : tag(2), case2(t) {} // PlayerGotCard
    __device__ Union1(Union1_3 t) : tag(3), case3(t) {} // Showdown
    __device__ Union1(Union1 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(x.case0); break; // CommunityCardIs
            case 1: new (&this->case1) Union1_1(x.case1); break; // PlayerAction
            case 2: new (&this->case2) Union1_2(x.case2); break; // PlayerGotCard
            case 3: new (&this->case3) Union1_3(x.case3); break; // Showdown
        }
    }
    __device__ Union1(Union1 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // CommunityCardIs
            case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // PlayerAction
            case 2: new (&this->case2) Union1_2(std::move(x.case2)); break; // PlayerGotCard
            case 3: new (&this->case3) Union1_3(std::move(x.case3)); break; // Showdown
        }
    }
    __device__ Union1 & operator=(Union1 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // CommunityCardIs
                case 1: this->case1 = x.case1; break; // PlayerAction
                case 2: this->case2 = x.case2; break; // PlayerGotCard
                case 3: this->case3 = x.case3; break; // Showdown
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
                case 0: this->case0 = std::move(x.case0); break; // CommunityCardIs
                case 1: this->case1 = std::move(x.case1); break; // PlayerAction
                case 2: this->case2 = std::move(x.case2); break; // PlayerGotCard
                case 3: this->case3 = std::move(x.case3); break; // Showdown
            }
        } else {
            this->~Union1();
            new (this) Union1{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union1() {
        switch(this->tag){
            case 0: this->case0.~Union1_0(); break; // CommunityCardIs
            case 1: this->case1.~Union1_1(); break; // PlayerAction
            case 2: this->case2.~Union1_2(); break; // PlayerGotCard
            case 3: this->case3.~Union1_3(); break; // Showdown
        }
        this->tag = 255;
    }
};
struct StackMut0 {
    cooperative_groups::grid_group v1;
    static_array_list<Union1,32> v2;
    static_array<Union0,2> v3;
    static_array<float,2> v4;
    curandStatePhilox4_32_10_t v5;
    unsigned int v0;
    __device__ StackMut0() = default;
    __device__ StackMut0(unsigned int t0, cooperative_groups::grid_group t1, static_array_list<Union1,32> t2, static_array<Union0,2> t3, static_array<float,2> t4, curandStatePhilox4_32_10_t t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Union5_0 { // None
};
struct Union5_1 { // Some
    Union2 v0;
    __device__ Union5_1(Union2 t0) : v0(t0) {}
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
    static_array<Union2,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_0(Union5 t0, bool t1, static_array<Union2,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_0() = delete;
};
struct Union4_1 { // ChanceInit
};
struct Union4_2 { // Round
    Union5 v0;
    static_array<Union2,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_2(Union5 t0, bool t1, static_array<Union2,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_2() = delete;
};
struct Union4_3 { // RoundWithAction
    Union5 v0;
    static_array<Union2,2> v2;
    static_array<int,2> v4;
    Union3 v6;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_3(Union5 t0, bool t1, static_array<Union2,2> t2, int t3, static_array<int,2> t4, int t5, Union3 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union4_3() = delete;
};
struct Union4_4 { // TerminalCall
    Union5 v0;
    static_array<Union2,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_4(Union5 t0, bool t1, static_array<Union2,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_4() = delete;
};
struct Union4_5 { // TerminalFold
    Union5 v0;
    static_array<Union2,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_5(Union5 t0, bool t1, static_array<Union2,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
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
struct Union6_0 { // None
};
struct Union6_1 { // Some
    Union4 v0;
    __device__ Union6_1(Union4 t0) : v0(t0) {}
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
struct Tuple0 {
    Union2 v0;
    unsigned int v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(Union2 t0, unsigned int t1) : v0(t0), v1(t1) {}
};
struct Tuple1 {
    int v0;
    int v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(int t0, int t1) : v0(t0), v1(t1) {}
};
struct Union7_0 { // C1of2
    Union3 v0;
    __device__ Union7_0(Union3 t0) : v0(t0) {}
    __device__ Union7_0() = delete;
};
struct Union7_1 { // C2of2
    Union2 v0;
    __device__ Union7_1(Union2 t0) : v0(t0) {}
    __device__ Union7_1() = delete;
};
struct Union7 {
    union {
        Union7_0 case0; // C1of2
        Union7_1 case1; // C2of2
    };
    unsigned char tag{255};
    __device__ Union7() {}
    __device__ Union7(Union7_0 t) : tag(0), case0(t) {} // C1of2
    __device__ Union7(Union7_1 t) : tag(1), case1(t) {} // C2of2
    __device__ Union7(Union7 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(x.case0); break; // C1of2
            case 1: new (&this->case1) Union7_1(x.case1); break; // C2of2
        }
    }
    __device__ Union7(Union7 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(std::move(x.case0)); break; // C1of2
            case 1: new (&this->case1) Union7_1(std::move(x.case1)); break; // C2of2
        }
    }
    __device__ Union7 & operator=(Union7 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // C1of2
                case 1: this->case1 = x.case1; break; // C2of2
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
                case 0: this->case0 = std::move(x.case0); break; // C1of2
                case 1: this->case1 = std::move(x.case1); break; // C2of2
            }
        } else {
            this->~Union7();
            new (this) Union7{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union7() {
        switch(this->tag){
            case 0: this->case0.~Union7_0(); break; // C1of2
            case 1: this->case1.~Union7_1(); break; // C2of2
        }
        this->tag = 255;
    }
};
struct Union8_0 { // None
};
struct Union8_1 { // Some
    Union7 v0;
    __device__ Union8_1(Union7 t0) : v0(t0) {}
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
struct Tuple2 {
    float v0;
    int v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(float t0, int t1) : v0(t0), v1(t1) {}
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
struct Tuple3 {
    int v0;
    float v1;
    __device__ Tuple3() = default;
    __device__ Tuple3(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple4 {
    float v0;
    bool v1;
    __device__ Tuple4() = default;
    __device__ Tuple4(float t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
    __device__ Tuple4 operator()(Tuple4 tup0, Tuple4 tup1){
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
                return Tuple4{v5, true};
            } else {
                return Tuple4{v0, v1};
            }
        } else {
            if (v3){
                return Tuple4{v2, v3};
            } else {
                return Tuple4{v0, v1};
            }
        }
    }
};
struct Closure5 {
    __device__ Tuple2 operator()(Tuple2 tup0, Tuple2 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple2{v0, v1};
        } else {
            return Tuple2{v2, v3};
        }
    }
};
struct Tuple5 {
    int v0;
    bool v1;
    __device__ Tuple5() = default;
    __device__ Tuple5(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure6 {
    __device__ Tuple5 operator()(Tuple5 tup0, Tuple5 tup1){
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
                return Tuple5{v5, true};
            } else {
                return Tuple5{v0, v1};
            }
        } else {
            if (v3){
                return Tuple5{v2, v3};
            } else {
                return Tuple5{v0, v1};
            }
        }
    }
};
struct Closure7 {
    int v0;
    __device__ Tuple2 operator()(Tuple2 tup0, Tuple2 tup1){
        int & v0 = this->v0;
        float v1 = tup0.v0; int v2 = tup0.v1; float v3 = tup1.v0; int v4 = tup1.v1;
        bool v5;
        v5 = v2 == v0;
        if (v5){
            return Tuple2{v1, v2};
        } else {
            bool v6;
            v6 = v4 == v0;
            if (v6){
                return Tuple2{v3, v4};
            } else {
                return Tuple2{v1, v2};
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
struct Closure8 {
    __device__ bool operator()(bool tup0, bool tup1){
        bool v0 = tup0; bool v1 = tup1;
        bool v2;
        v2 = v0 || v1;
        return v2;
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 4;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 8;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 2;
    return v1;
}
__device__ inline bool while_method_3(Union6 v0){
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
__device__ unsigned int loop_2(unsigned int v0, curandStatePhilox4_32_10_t & v1){
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
        return loop_2(v0, v1);
    }
}
__device__ Tuple0 draw_card_1(curandStatePhilox4_32_10_t & v0, unsigned int v1){
    int v2;
    v2 = __popc(v1);
    unsigned int v3;
    v3 = (unsigned int)v2;
    unsigned int v4;
    v4 = loop_2(v3, v0);
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
    Union2 v31;
    if (v13){
        v31 = Union2{Union2_1{}};
    } else {
        bool v15;
        v15 = 1u == v12;
        if (v15){
            v31 = Union2{Union2_1{}};
        } else {
            bool v17;
            v17 = 2u == v12;
            if (v17){
                v31 = Union2{Union2_2{}};
            } else {
                bool v19;
                v19 = 3u == v12;
                if (v19){
                    v31 = Union2{Union2_2{}};
                } else {
                    bool v21;
                    v21 = 4u == v12;
                    if (v21){
                        v31 = Union2{Union2_0{}};
                    } else {
                        bool v23;
                        v23 = 5u == v12;
                        if (v23){
                            v31 = Union2{Union2_0{}};
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
    return Tuple0{v31, v34};
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 32768;
    return v1;
}
__device__ inline bool while_method_5(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 1;
    return v1;
}
__device__ inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 2;
    return v1;
}
__device__ inline bool while_method_8(int v0){
    bool v1;
    v1 = v0 < 16;
    return v1;
}
__device__ inline bool while_method_9(int v0){
    bool v1;
    v1 = v0 < 32;
    return v1;
}
__device__ void method_3(unsigned int * v0, int v1, float * v2){
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
    while (while_method_9(v22)){
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
            while (while_method_0(v37)){
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
            while (while_method_0(v74)){
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
            while (while_method_0(v86)){
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
__device__ Tuple2 method_4(curandStatePhilox4_32_10_t & v0, int * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int v9, int v10){
    assert("Tensor range check" && 0 <= v10 && v10 < 4);
    int v11;
    v11 = 16384 * v10;
    assert("Tensor range check" && 0 <= v9 && v9 < 4096);
    int v12;
    v12 = 4 * v9;
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
    v55 = v51 % 1;
    bool v56;
    v56 = v51 < 256;
    bool v57;
    v57 = v56 == false;
    if (v57){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v56);
    } else {
    }
    assert("Tensor range check" && 0 <= v51 && v51 < 256);
    int v59;
    v59 = 0;
    while (while_method_6(v59)){
        bool v61;
        v61 = v52 && v56;
        bool v62;
        v62 = v61 == false;
        if (v62){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v61);
        } else {
        }
        bool v64;
        v64 = 0 <= v59;
        bool v66;
        if (v64){
            bool v65;
            v65 = v59 < 1;
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
        v69 = v59 * 256;
        int v70;
        v70 = v69 + v51;
        assert("Tensor range check" && 0 <= v59 && v59 < 1);
        int v71;
        v71 = 256 * v59;
        int v72;
        v72 = v71 + v51;
        float * v73;
        v73 = v43[v72];
        float * v74;
        v74 = v45[v72];
        int v75;
        v75 = blockIdx.x;
        int v76;
        v76 = v75 * 256;
        int v77;
        v77 = v76 + v70;
        assert("Tensor range check" && 0 <= v55 && v55 < 1);
        int v78;
        v78 = 4 * v55;
        float v79[4];
        float v80[4];
        int v81[4];
        int v82;
        v82 = 0;
        while (while_method_6(v82)){
            assert("Tensor range check" && 0 <= v82 && v82 < 1);
            int v84;
            v84 = 4 * v82;
            assert("Tensor range check" && 0 <= v82 && v82 < 1);
            int v85;
            v85 = v84 + v78;
            int4* v86;
            v86 = reinterpret_cast<int4*>(v73 + v85);
            int4* v87;
            v87 = reinterpret_cast<int4*>(v79 + v84);
            assert("Pointer alignment check" && (unsigned long long)(v86) % 4 == 0 && (unsigned long long)(v87) % 4 == 0);
            *v87 = *v86;
            int4* v88;
            v88 = reinterpret_cast<int4*>(v74 + v85);
            int4* v89;
            v89 = reinterpret_cast<int4*>(v80 + v84);
            assert("Pointer alignment check" && (unsigned long long)(v88) % 4 == 0 && (unsigned long long)(v89) % 4 == 0);
            *v89 = *v88;
            v82 += 1 ;
        }
        int v90;
        v90 = 0;
        while (while_method_6(v90)){
            int v92;
            v92 = 0;
            while (while_method_0(v92)){
                bool v94;
                v94 = 0 <= v92;
                bool v96;
                if (v94){
                    bool v95;
                    v95 = v92 < 4;
                    v96 = v95;
                } else {
                    v96 = false;
                }
                bool v97;
                v97 = v96 == false;
                if (v97){
                    assert("The indices should be inside the range of the dimension." && v96);
                } else {
                }
                bool v99;
                v99 = 0 <= v55;
                bool v101;
                if (v99){
                    bool v100;
                    v100 = v55 < 1;
                    v101 = v100;
                } else {
                    v101 = false;
                }
                bool v102;
                v102 = v101 == false;
                if (v102){
                    assert("The indices should be inside the range of the dimension." && v101);
                } else {
                }
                int v104;
                v104 = v55 * 4;
                int v105;
                v105 = v92 + v104;
                bool v106;
                v106 = 0 <= v90;
                bool v108;
                if (v106){
                    bool v107;
                    v107 = v90 < 1;
                    v108 = v107;
                } else {
                    v108 = false;
                }
                bool v109;
                v109 = v108 == false;
                if (v109){
                    assert("The indices should be inside the range of the dimension." && v108);
                } else {
                }
                int v111;
                v111 = v90 * 4;
                int v112;
                v112 = v105 + v111;
                assert("Tensor range check" && 0 <= v90 && v90 < 1);
                assert("Tensor range check" && 0 <= v92 && v92 < 4);
                int v113;
                v113 = 4 * v90;
                int v114;
                v114 = v113 + v92;
                v81[v114] = v112;
                v92 += 1 ;
            }
            v90 += 1 ;
        }
        bool v115[4];
        int v116;
        v116 = 0;
        while (while_method_6(v116)){
            int v118;
            v118 = 0;
            while (while_method_0(v118)){
                assert("Tensor range check" && 0 <= v116 && v116 < 1);
                assert("Tensor range check" && 0 <= v118 && v118 < 4);
                int v120;
                v120 = 4 * v116;
                int v121;
                v121 = v120 + v118;
                float v122;
                v122 = v79[v121];
                int v123;
                v123 = v81[v121];
                bool v124;
                v124 = v123 < 3;
                assert("Tensor range check" && 0 <= v116 && v116 < 1);
                assert("Tensor range check" && 0 <= v118 && v118 < 4);
                v115[v121] = v124;
                v118 += 1 ;
            }
            v116 += 1 ;
        }
        float v125[4];
        int v126;
        v126 = 0;
        while (while_method_6(v126)){
            int v128;
            v128 = 0;
            while (while_method_0(v128)){
                assert("Tensor range check" && 0 <= v126 && v126 < 1);
                assert("Tensor range check" && 0 <= v128 && v128 < 4);
                int v130;
                v130 = 4 * v126;
                int v131;
                v131 = v130 + v128;
                float v132;
                v132 = v79[v131];
                bool v133;
                v133 = v115[v131];
                float v136;
                if (v133){
                    bool v134;
                    v134 = 0.0f >= v132;
                    if (v134){
                        v136 = 0.0f;
                    } else {
                        v136 = v132;
                    }
                } else {
                    v136 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v126 && v126 < 1);
                assert("Tensor range check" && 0 <= v128 && v128 < 4);
                v125[v131] = v136;
                v128 += 1 ;
            }
            v126 += 1 ;
        }
        float v137;
        v137 = 0.0f;
        int v138;
        v138 = 0;
        while (while_method_6(v138)){
            int v140;
            v140 = 0;
            while (while_method_0(v140)){
                assert("Tensor range check" && 0 <= v138 && v138 < 1);
                assert("Tensor range check" && 0 <= v140 && v140 < 4);
                int v142;
                v142 = 4 * v138;
                int v143;
                v143 = v142 + v140;
                float v144;
                v144 = v125[v143];
                float v145;
                v145 = v137 + v144;
                v137 = v145;
                v140 += 1 ;
            }
            v138 += 1 ;
        }
        auto v146 = cooperative_groups::coalesced_threads();
        int v147;
        v147 = threadIdx.x;
        auto v148 = cooperative_groups::labeled_partition(v146,v147);
        Closure1 v149{};
        float v150;
        v150 = cooperative_groups::reduce(v148, v137, v149);
        int v151[4];
        int v152;
        v152 = 0;
        while (while_method_6(v152)){
            int v154;
            v154 = 0;
            while (while_method_0(v154)){
                assert("Tensor range check" && 0 <= v152 && v152 < 1);
                assert("Tensor range check" && 0 <= v154 && v154 < 4);
                int v156;
                v156 = 4 * v152;
                int v157;
                v157 = v156 + v154;
                bool v158;
                v158 = v115[v157];
                int v159;
                if (v158){
                    v159 = 1;
                } else {
                    v159 = 0;
                }
                assert("Tensor range check" && 0 <= v152 && v152 < 1);
                assert("Tensor range check" && 0 <= v154 && v154 < 4);
                v151[v157] = v159;
                v154 += 1 ;
            }
            v152 += 1 ;
        }
        int v160;
        v160 = 0;
        int v161;
        v161 = 0;
        while (while_method_6(v161)){
            int v163;
            v163 = 0;
            while (while_method_0(v163)){
                assert("Tensor range check" && 0 <= v161 && v161 < 1);
                assert("Tensor range check" && 0 <= v163 && v163 < 4);
                int v165;
                v165 = 4 * v161;
                int v166;
                v166 = v165 + v163;
                int v167;
                v167 = v151[v166];
                int v168;
                v168 = v160 + v167;
                v160 = v168;
                v163 += 1 ;
            }
            v161 += 1 ;
        }
        auto v169 = cooperative_groups::coalesced_threads();
        int v170;
        v170 = threadIdx.x;
        auto v171 = cooperative_groups::labeled_partition(v169,v170);
        Closure2 v172{};
        int v173;
        v173 = cooperative_groups::reduce(v171, v160, v172);
        float v174;
        v174 = (float)v173;
        float v175;
        v175 = 1.0f / v174;
        float v176[4];
        int v177;
        v177 = 0;
        while (while_method_6(v177)){
            int v179;
            v179 = 0;
            while (while_method_0(v179)){
                assert("Tensor range check" && 0 <= v177 && v177 < 1);
                assert("Tensor range check" && 0 <= v179 && v179 < 4);
                int v181;
                v181 = 4 * v177;
                int v182;
                v182 = v181 + v179;
                float v183;
                v183 = v125[v182];
                bool v184;
                v184 = v115[v182];
                bool v185;
                v185 = v184 == false;
                float v190;
                if (v185){
                    v190 = 0.0f;
                } else {
                    bool v186;
                    v186 = v150 == 0.0f;
                    bool v187;
                    v187 = v186 != true;
                    if (v187){
                        float v188;
                        v188 = v183 / v150;
                        v190 = v188;
                    } else {
                        v190 = v175;
                    }
                }
                assert("Tensor range check" && 0 <= v177 && v177 < 1);
                assert("Tensor range check" && 0 <= v179 && v179 < 4);
                v176[v182] = v190;
                v179 += 1 ;
            }
            v177 += 1 ;
        }
        float v191[4];
        float v192;
        v192 = 0.0f;
        int v193;
        v193 = 0;
        while (while_method_6(v193)){
            assert("Tensor range check" && 0 <= v193 && v193 < 1);
            int v195;
            v195 = 4 * v193;
            assert("Tensor range check" && 0 <= v193 && v193 < 1);
            int v196; float v197;
            Tuple3 tmp4 = Tuple3{0, 0.0f};
            v196 = tmp4.v0; v197 = tmp4.v1;
            while (while_method_0(v196)){
                assert("Tensor range check" && 0 <= v196 && v196 < 4);
                int v199;
                v199 = v196 + v195;
                float v200;
                v200 = v176[v199];
                float v201;
                v201 = v197 + v200;
                v197 = v201;
                v196 += 1 ;
            }
            auto v202 = cooperative_groups::coalesced_threads();
            int v203;
            v203 = threadIdx.x;
            auto v204 = cooperative_groups::labeled_partition(v202,v203);
            Closure3 v205{};
            float v206;
            v206 = cooperative_groups::inclusive_scan(v204, v197, v205);
            float v207;
            v207 = v204.shfl_up(v206,1);
            bool v208;
            v208 = v204.thread_rank() == 0;
            float v209;
            if (v208){
                v209 = 0.0f;
            } else {
                v209 = v207;
            }
            float v210;
            v210 = v204.shfl(v206,v204.num_threads()-1);
            float v211;
            v211 = v192 + v209;
            int v212; float v213;
            Tuple3 tmp5 = Tuple3{0, v211};
            v212 = tmp5.v0; v213 = tmp5.v1;
            while (while_method_0(v212)){
                assert("Tensor range check" && 0 <= v212 && v212 < 4);
                int v215;
                v215 = v212 + v195;
                float v216;
                v216 = v176[v215];
                float v217;
                v217 = v213 + v216;
                assert("Tensor range check" && 0 <= v212 && v212 < 4);
                v191[v215] = v217;
                v213 = v217;
                v212 += 1 ;
            }
            float v218;
            v218 = v192 + v210;
            v192 = v218;
            v193 += 1 ;
        }
        float v219[4];
        bool v220[4];
        int v221;
        v221 = 0;
        while (while_method_6(v221)){
            int v223;
            v223 = 0;
            while (while_method_0(v223)){
                assert("Tensor range check" && 0 <= v221 && v221 < 1);
                assert("Tensor range check" && 0 <= v223 && v223 < 4);
                int v225;
                v225 = 4 * v221;
                int v226;
                v226 = v225 + v223;
                float v227;
                v227 = v191[v226];
                float v228;
                v228 = v176[v226];
                bool v229;
                v229 = v228 > 0.0f;
                assert("Tensor range check" && 0 <= v221 && v221 < 1);
                assert("Tensor range check" && 0 <= v223 && v223 < 4);
                v219[v226] = v227;
                v220[v226] = v229;
                v223 += 1 ;
            }
            v221 += 1 ;
        }
        float v230; bool v231;
        Tuple4 tmp6 = Tuple4{-1.0f / 0.0f, false};
        v230 = tmp6.v0; v231 = tmp6.v1;
        int v232;
        v232 = 0;
        while (while_method_6(v232)){
            int v234;
            v234 = 0;
            while (while_method_0(v234)){
                assert("Tensor range check" && 0 <= v232 && v232 < 1);
                assert("Tensor range check" && 0 <= v234 && v234 < 4);
                int v236;
                v236 = 4 * v232;
                int v237;
                v237 = v236 + v234;
                float v238;
                v238 = v219[v237];
                bool v239;
                v239 = v220[v237];
                float v246; bool v247;
                if (v231){
                    if (v239){
                        bool v240;
                        v240 = v230 >= v238;
                        float v241;
                        if (v240){
                            v241 = v230;
                        } else {
                            v241 = v238;
                        }
                        v246 = v241; v247 = true;
                    } else {
                        v246 = v230; v247 = v231;
                    }
                } else {
                    if (v239){
                        v246 = v238; v247 = v239;
                    } else {
                        v246 = v230; v247 = v231;
                    }
                }
                v230 = v246;
                v231 = v247;
                v234 += 1 ;
            }
            v232 += 1 ;
        }
        auto v248 = cooperative_groups::coalesced_threads();
        int v249;
        v249 = threadIdx.x;
        auto v250 = cooperative_groups::labeled_partition(v248,v249);
        Closure4 v251{};
        float v252; bool v253;
        Tuple4 tmp7 = cooperative_groups::reduce(v250, Tuple4{v230, v231}, v251);
        v252 = tmp7.v0; v253 = tmp7.v1;
        bool v254;
        v254 = v253 == false;
        if (v254){
            assert("The local reduce must be true." && v253);
        } else {
        }
        float v256[4];
        int v257[4];
        int v258;
        v258 = 0;
        while (while_method_6(v258)){
            int v260;
            v260 = 0;
            while (while_method_0(v260)){
                assert("Tensor range check" && 0 <= v258 && v258 < 1);
                assert("Tensor range check" && 0 <= v260 && v260 < 4);
                int v262;
                v262 = 4 * v258;
                int v263;
                v263 = v262 + v260;
                int v264;
                v264 = v81[v263];
                float v265;
                v265 = curand_uniform(&v0);
                assert("Tensor range check" && 0 <= v258 && v258 < 1);
                assert("Tensor range check" && 0 <= v260 && v260 < 4);
                v256[v263] = v265;
                v257[v263] = v264;
                v260 += 1 ;
            }
            v258 += 1 ;
        }
        float v266; int v267;
        Tuple2 tmp8 = Tuple2{0.0f, 2147483647};
        v266 = tmp8.v0; v267 = tmp8.v1;
        int v268;
        v268 = 0;
        while (while_method_6(v268)){
            int v270;
            v270 = 0;
            while (while_method_0(v270)){
                assert("Tensor range check" && 0 <= v268 && v268 < 1);
                assert("Tensor range check" && 0 <= v270 && v270 < 4);
                int v272;
                v272 = 4 * v268;
                int v273;
                v273 = v272 + v270;
                float v274;
                v274 = v256[v273];
                int v275;
                v275 = v257[v273];
                bool v276;
                v276 = v267 < v275;
                float v277; int v278;
                if (v276){
                    v277 = v266; v278 = v267;
                } else {
                    v277 = v274; v278 = v275;
                }
                v266 = v277;
                v267 = v278;
                v270 += 1 ;
            }
            v268 += 1 ;
        }
        auto v279 = cooperative_groups::coalesced_threads();
        int v280;
        v280 = threadIdx.x;
        auto v281 = cooperative_groups::labeled_partition(v279,v280);
        Closure5 v282{};
        float v283; int v284;
        Tuple2 tmp9 = cooperative_groups::reduce(v281, Tuple2{v266, v267}, v282);
        v283 = tmp9.v0; v284 = tmp9.v1;
        float v285;
        v285 = v252 * v283;
        int v286[4];
        bool v287[4];
        int v288;
        v288 = 0;
        while (while_method_6(v288)){
            int v290;
            v290 = 0;
            while (while_method_0(v290)){
                assert("Tensor range check" && 0 <= v288 && v288 < 1);
                assert("Tensor range check" && 0 <= v290 && v290 < 4);
                int v292;
                v292 = 4 * v288;
                int v293;
                v293 = v292 + v290;
                float v294;
                v294 = v219[v293];
                bool v295;
                v295 = v220[v293];
                int v296;
                v296 = v81[v293];
                int v299; bool v300;
                if (v295){
                    float v297;
                    v297 = v294 - v285;
                    bool v298;
                    v298 = v297 >= 0.0f;
                    v299 = v296; v300 = v298;
                } else {
                    v299 = 2147483647; v300 = false;
                }
                assert("Tensor range check" && 0 <= v288 && v288 < 1);
                assert("Tensor range check" && 0 <= v290 && v290 < 4);
                v286[v293] = v299;
                v287[v293] = v300;
                v290 += 1 ;
            }
            v288 += 1 ;
        }
        int v301; bool v302;
        Tuple5 tmp10 = Tuple5{2147483647, false};
        v301 = tmp10.v0; v302 = tmp10.v1;
        int v303;
        v303 = 0;
        while (while_method_6(v303)){
            int v305;
            v305 = 0;
            while (while_method_0(v305)){
                assert("Tensor range check" && 0 <= v303 && v303 < 1);
                assert("Tensor range check" && 0 <= v305 && v305 < 4);
                int v307;
                v307 = 4 * v303;
                int v308;
                v308 = v307 + v305;
                int v309;
                v309 = v286[v308];
                bool v310;
                v310 = v287[v308];
                int v317; bool v318;
                if (v302){
                    if (v310){
                        bool v311;
                        v311 = v301 < v309;
                        int v312;
                        if (v311){
                            v312 = v301;
                        } else {
                            v312 = v309;
                        }
                        v317 = v312; v318 = true;
                    } else {
                        v317 = v301; v318 = v302;
                    }
                } else {
                    if (v310){
                        v317 = v309; v318 = v310;
                    } else {
                        v317 = v301; v318 = v302;
                    }
                }
                v301 = v317;
                v302 = v318;
                v305 += 1 ;
            }
            v303 += 1 ;
        }
        auto v319 = cooperative_groups::coalesced_threads();
        int v320;
        v320 = threadIdx.x;
        auto v321 = cooperative_groups::labeled_partition(v319,v320);
        Closure6 v322{};
        int v323; bool v324;
        Tuple5 tmp11 = cooperative_groups::reduce(v321, Tuple5{v301, v302}, v322);
        v323 = tmp11.v0; v324 = tmp11.v1;
        bool v325;
        v325 = v324 == false;
        if (v325){
            assert("The local reduce must be true." && v324);
        } else {
        }
        bool v327[4];
        int v328;
        v328 = 0;
        while (while_method_6(v328)){
            int v330;
            v330 = 0;
            while (while_method_0(v330)){
                assert("Tensor range check" && 0 <= v328 && v328 < 1);
                assert("Tensor range check" && 0 <= v330 && v330 < 4);
                int v332;
                v332 = 4 * v328;
                int v333;
                v333 = v332 + v330;
                float v334;
                v334 = v80[v333];
                int v335;
                v335 = v81[v333];
                bool v336;
                v336 = v335 < 3;
                assert("Tensor range check" && 0 <= v328 && v328 < 1);
                assert("Tensor range check" && 0 <= v330 && v330 < 4);
                v327[v333] = v336;
                v330 += 1 ;
            }
            v328 += 1 ;
        }
        float v337[4];
        int v338;
        v338 = 0;
        while (while_method_6(v338)){
            int v340;
            v340 = 0;
            while (while_method_0(v340)){
                assert("Tensor range check" && 0 <= v338 && v338 < 1);
                assert("Tensor range check" && 0 <= v340 && v340 < 4);
                int v342;
                v342 = 4 * v338;
                int v343;
                v343 = v342 + v340;
                float v344;
                v344 = v80[v343];
                bool v345;
                v345 = v327[v343];
                float v348;
                if (v345){
                    bool v346;
                    v346 = 0.0f >= v344;
                    if (v346){
                        v348 = 0.0f;
                    } else {
                        v348 = v344;
                    }
                } else {
                    v348 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v338 && v338 < 1);
                assert("Tensor range check" && 0 <= v340 && v340 < 4);
                v337[v343] = v348;
                v340 += 1 ;
            }
            v338 += 1 ;
        }
        float v349;
        v349 = 0.0f;
        int v350;
        v350 = 0;
        while (while_method_6(v350)){
            int v352;
            v352 = 0;
            while (while_method_0(v352)){
                assert("Tensor range check" && 0 <= v350 && v350 < 1);
                assert("Tensor range check" && 0 <= v352 && v352 < 4);
                int v354;
                v354 = 4 * v350;
                int v355;
                v355 = v354 + v352;
                float v356;
                v356 = v337[v355];
                float v357;
                v357 = v349 + v356;
                v349 = v357;
                v352 += 1 ;
            }
            v350 += 1 ;
        }
        auto v358 = cooperative_groups::coalesced_threads();
        int v359;
        v359 = threadIdx.x;
        auto v360 = cooperative_groups::labeled_partition(v358,v359);
        float v361;
        v361 = cooperative_groups::reduce(v360, v349, v149);
        int v362[4];
        int v363;
        v363 = 0;
        while (while_method_6(v363)){
            int v365;
            v365 = 0;
            while (while_method_0(v365)){
                assert("Tensor range check" && 0 <= v363 && v363 < 1);
                assert("Tensor range check" && 0 <= v365 && v365 < 4);
                int v367;
                v367 = 4 * v363;
                int v368;
                v368 = v367 + v365;
                bool v369;
                v369 = v327[v368];
                int v370;
                if (v369){
                    v370 = 1;
                } else {
                    v370 = 0;
                }
                assert("Tensor range check" && 0 <= v363 && v363 < 1);
                assert("Tensor range check" && 0 <= v365 && v365 < 4);
                v362[v368] = v370;
                v365 += 1 ;
            }
            v363 += 1 ;
        }
        int v371;
        v371 = 0;
        int v372;
        v372 = 0;
        while (while_method_6(v372)){
            int v374;
            v374 = 0;
            while (while_method_0(v374)){
                assert("Tensor range check" && 0 <= v372 && v372 < 1);
                assert("Tensor range check" && 0 <= v374 && v374 < 4);
                int v376;
                v376 = 4 * v372;
                int v377;
                v377 = v376 + v374;
                int v378;
                v378 = v362[v377];
                int v379;
                v379 = v371 + v378;
                v371 = v379;
                v374 += 1 ;
            }
            v372 += 1 ;
        }
        auto v380 = cooperative_groups::coalesced_threads();
        int v381;
        v381 = threadIdx.x;
        auto v382 = cooperative_groups::labeled_partition(v380,v381);
        int v383;
        v383 = cooperative_groups::reduce(v382, v371, v172);
        float v384;
        v384 = (float)v383;
        float v385;
        v385 = 1.0f / v384;
        float v386[4];
        int v387;
        v387 = 0;
        while (while_method_6(v387)){
            int v389;
            v389 = 0;
            while (while_method_0(v389)){
                assert("Tensor range check" && 0 <= v387 && v387 < 1);
                assert("Tensor range check" && 0 <= v389 && v389 < 4);
                int v391;
                v391 = 4 * v387;
                int v392;
                v392 = v391 + v389;
                float v393;
                v393 = v337[v392];
                bool v394;
                v394 = v327[v392];
                bool v395;
                v395 = v394 == false;
                float v400;
                if (v395){
                    v400 = 0.0f;
                } else {
                    bool v396;
                    v396 = v361 == 0.0f;
                    bool v397;
                    v397 = v396 != true;
                    if (v397){
                        float v398;
                        v398 = v393 / v361;
                        v400 = v398;
                    } else {
                        v400 = v385;
                    }
                }
                assert("Tensor range check" && 0 <= v387 && v387 < 1);
                assert("Tensor range check" && 0 <= v389 && v389 < 4);
                v386[v392] = v400;
                v389 += 1 ;
            }
            v387 += 1 ;
        }
        float v401; int v402;
        Tuple2 tmp12 = Tuple2{0.0f, 2147483647};
        v401 = tmp12.v0; v402 = tmp12.v1;
        int v403;
        v403 = 0;
        while (while_method_6(v403)){
            int v405;
            v405 = 0;
            while (while_method_0(v405)){
                assert("Tensor range check" && 0 <= v403 && v403 < 1);
                assert("Tensor range check" && 0 <= v405 && v405 < 4);
                int v407;
                v407 = 4 * v403;
                int v408;
                v408 = v407 + v405;
                float v409;
                v409 = v176[v408];
                int v410;
                v410 = v81[v408];
                bool v411;
                v411 = v402 == v323;
                float v415; int v416;
                if (v411){
                    v415 = v401; v416 = v402;
                } else {
                    bool v412;
                    v412 = v410 == v323;
                    if (v412){
                        v415 = v409; v416 = v410;
                    } else {
                        v415 = v401; v416 = v402;
                    }
                }
                v401 = v415;
                v402 = v416;
                v405 += 1 ;
            }
            v403 += 1 ;
        }
        auto v417 = cooperative_groups::coalesced_threads();
        int v418;
        v418 = threadIdx.x;
        auto v419 = cooperative_groups::labeled_partition(v417,v418);
        Closure7 v420{v323};
        float v421; int v422;
        Tuple2 tmp13 = cooperative_groups::reduce(v419, Tuple2{v401, v402}, v420);
        v421 = tmp13.v0; v422 = tmp13.v1;
        bool v423;
        v423 = v422 == 2147483647;
        bool v424;
        v424 = v423 != true;
        bool v425;
        v425 = v424 == false;
        if (v425){
            assert("Expected a valid action id in get_action." && v424);
        } else {
        }
        int v427;
        v427 = 0;
        while (while_method_6(v427)){
            assert("Tensor range check" && 0 <= v427 && v427 < 1);
            assert("Tensor range check" && 0 <= v427 && v427 < 1);
            v427 += 1 ;
        }
        assert("Tensor range check" && 0 <= v70 && v70 < 256);
        v47[v70] = v421;
        v49[v70] = v323;
        v59 += 1 ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0));
    assert("Tensor range check" && 0 <= v51 && v51 < 256);
    float v429;
    v429 = v47[v51];
    int v430;
    v430 = v49[v51];
    asm("barrier.cta.sync %0;" :: "r"(0));
    return Tuple2{v429, v430};
}
__device__ float method_5(int * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int v8, int v9, int v10){
    assert("Tensor range check" && 0 <= v9 && v9 < 4);
    int v11;
    v11 = 16384 * v9;
    assert("Tensor range check" && 0 <= v8 && v8 < 4096);
    int v12;
    v12 = 4 * v8;
    int v13;
    v13 = v12 + v11;
    float * v14;
    v14 = v2+v13;
    int v16;
    v16 = sizeof(float *);
    unsigned long long v17;
    v17 = (unsigned long long)v16;
    unsigned long long v18;
    v18 = 256ull * v17;
    unsigned long long v19;
    v19 = 1024ull + v18;
    unsigned long long v20;
    v20 = v19 + 16ull;
    unsigned long long v21;
    v21 = v20 - 1ull;
    unsigned long long v22;
    v22 = v21 % 16ull;
    unsigned long long v23;
    v23 = v21 - v22;
    unsigned long long v24;
    v24 = v23 + 1024ull;
    bool v25;
    v25 = v24 <= 98304ull;
    bool v26;
    v26 = v25 == false;
    if (v26){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v25);
    } else {
    }
    extern __shared__ unsigned char v28[];
    bool v29;
    v29 = v24 <= v24;
    bool v30;
    v30 = v29 == false;
    if (v30){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v29);
    } else {
    }
    int * v32;
    v32 = reinterpret_cast<int *>(&v28[0ull]);
    float * * v34;
    v34 = reinterpret_cast<float * *>(&v28[1024ull]);
    float * v36;
    v36 = reinterpret_cast<float *>(&v28[v23]);
    int v38;
    v38 = threadIdx.x;
    assert("Tensor range check" && 0 <= v38 && v38 < 256);
    v32[v38] = v10;
    v34[v38] = v14;
    asm("barrier.cta.sync %0;" :: "r"(0));
    bool v39;
    v39 = 0 <= v38;
    bool v40;
    v40 = v39 == false;
    if (v40){
        assert("The index needs to be zero or positive." && v39);
    } else {
    }
    int v42;
    v42 = v38 % 1;
    bool v43;
    v43 = v38 < 256;
    bool v44;
    v44 = v43 == false;
    if (v44){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v43);
    } else {
    }
    assert("Tensor range check" && 0 <= v38 && v38 < 256);
    int v46;
    v46 = 0;
    while (while_method_6(v46)){
        bool v48;
        v48 = v39 && v43;
        bool v49;
        v49 = v48 == false;
        if (v49){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v48);
        } else {
        }
        bool v51;
        v51 = 0 <= v46;
        bool v53;
        if (v51){
            bool v52;
            v52 = v46 < 1;
            v53 = v52;
        } else {
            v53 = false;
        }
        bool v54;
        v54 = v53 == false;
        if (v54){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v53);
        } else {
        }
        int v56;
        v56 = v46 * 256;
        int v57;
        v57 = v56 + v38;
        assert("Tensor range check" && 0 <= v46 && v46 < 1);
        int v58;
        v58 = 256 * v46;
        int v59;
        v59 = v58 + v38;
        int v60;
        v60 = v32[v59];
        float * v61;
        v61 = v34[v59];
        int v62;
        v62 = blockIdx.x;
        int v63;
        v63 = v62 * 256;
        int v64;
        v64 = v63 + v57;
        assert("Tensor range check" && 0 <= v42 && v42 < 1);
        int v65;
        v65 = 4 * v42;
        float v66[4];
        int v67[4];
        int v68;
        v68 = 0;
        while (while_method_6(v68)){
            assert("Tensor range check" && 0 <= v68 && v68 < 1);
            int v70;
            v70 = 4 * v68;
            assert("Tensor range check" && 0 <= v68 && v68 < 1);
            int v71;
            v71 = v70 + v65;
            int4* v72;
            v72 = reinterpret_cast<int4*>(v61 + v71);
            int4* v73;
            v73 = reinterpret_cast<int4*>(v66 + v70);
            assert("Pointer alignment check" && (unsigned long long)(v72) % 4 == 0 && (unsigned long long)(v73) % 4 == 0);
            *v73 = *v72;
            v68 += 1 ;
        }
        int v74;
        v74 = 0;
        while (while_method_6(v74)){
            int v76;
            v76 = 0;
            while (while_method_0(v76)){
                bool v78;
                v78 = 0 <= v76;
                bool v80;
                if (v78){
                    bool v79;
                    v79 = v76 < 4;
                    v80 = v79;
                } else {
                    v80 = false;
                }
                bool v81;
                v81 = v80 == false;
                if (v81){
                    assert("The indices should be inside the range of the dimension." && v80);
                } else {
                }
                bool v83;
                v83 = 0 <= v42;
                bool v85;
                if (v83){
                    bool v84;
                    v84 = v42 < 1;
                    v85 = v84;
                } else {
                    v85 = false;
                }
                bool v86;
                v86 = v85 == false;
                if (v86){
                    assert("The indices should be inside the range of the dimension." && v85);
                } else {
                }
                int v88;
                v88 = v42 * 4;
                int v89;
                v89 = v76 + v88;
                bool v90;
                v90 = 0 <= v74;
                bool v92;
                if (v90){
                    bool v91;
                    v91 = v74 < 1;
                    v92 = v91;
                } else {
                    v92 = false;
                }
                bool v93;
                v93 = v92 == false;
                if (v93){
                    assert("The indices should be inside the range of the dimension." && v92);
                } else {
                }
                int v95;
                v95 = v74 * 4;
                int v96;
                v96 = v89 + v95;
                assert("Tensor range check" && 0 <= v74 && v74 < 1);
                assert("Tensor range check" && 0 <= v76 && v76 < 4);
                int v97;
                v97 = 4 * v74;
                int v98;
                v98 = v97 + v76;
                v67[v98] = v96;
                v76 += 1 ;
            }
            v74 += 1 ;
        }
        bool v99[4];
        int v100;
        v100 = 0;
        while (while_method_6(v100)){
            int v102;
            v102 = 0;
            while (while_method_0(v102)){
                assert("Tensor range check" && 0 <= v100 && v100 < 1);
                assert("Tensor range check" && 0 <= v102 && v102 < 4);
                int v104;
                v104 = 4 * v100;
                int v105;
                v105 = v104 + v102;
                float v106;
                v106 = v66[v105];
                int v107;
                v107 = v67[v105];
                bool v108;
                v108 = v107 < 3;
                assert("Tensor range check" && 0 <= v100 && v100 < 1);
                assert("Tensor range check" && 0 <= v102 && v102 < 4);
                v99[v105] = v108;
                v102 += 1 ;
            }
            v100 += 1 ;
        }
        float v109[4];
        int v110;
        v110 = 0;
        while (while_method_6(v110)){
            int v112;
            v112 = 0;
            while (while_method_0(v112)){
                assert("Tensor range check" && 0 <= v110 && v110 < 1);
                assert("Tensor range check" && 0 <= v112 && v112 < 4);
                int v114;
                v114 = 4 * v110;
                int v115;
                v115 = v114 + v112;
                float v116;
                v116 = v66[v115];
                bool v117;
                v117 = v99[v115];
                float v120;
                if (v117){
                    bool v118;
                    v118 = 0.0f >= v116;
                    if (v118){
                        v120 = 0.0f;
                    } else {
                        v120 = v116;
                    }
                } else {
                    v120 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v110 && v110 < 1);
                assert("Tensor range check" && 0 <= v112 && v112 < 4);
                v109[v115] = v120;
                v112 += 1 ;
            }
            v110 += 1 ;
        }
        float v121;
        v121 = 0.0f;
        int v122;
        v122 = 0;
        while (while_method_6(v122)){
            int v124;
            v124 = 0;
            while (while_method_0(v124)){
                assert("Tensor range check" && 0 <= v122 && v122 < 1);
                assert("Tensor range check" && 0 <= v124 && v124 < 4);
                int v126;
                v126 = 4 * v122;
                int v127;
                v127 = v126 + v124;
                float v128;
                v128 = v109[v127];
                float v129;
                v129 = v121 + v128;
                v121 = v129;
                v124 += 1 ;
            }
            v122 += 1 ;
        }
        auto v130 = cooperative_groups::coalesced_threads();
        int v131;
        v131 = threadIdx.x;
        auto v132 = cooperative_groups::labeled_partition(v130,v131);
        Closure1 v133{};
        float v134;
        v134 = cooperative_groups::reduce(v132, v121, v133);
        int v135[4];
        int v136;
        v136 = 0;
        while (while_method_6(v136)){
            int v138;
            v138 = 0;
            while (while_method_0(v138)){
                assert("Tensor range check" && 0 <= v136 && v136 < 1);
                assert("Tensor range check" && 0 <= v138 && v138 < 4);
                int v140;
                v140 = 4 * v136;
                int v141;
                v141 = v140 + v138;
                bool v142;
                v142 = v99[v141];
                int v143;
                if (v142){
                    v143 = 1;
                } else {
                    v143 = 0;
                }
                assert("Tensor range check" && 0 <= v136 && v136 < 1);
                assert("Tensor range check" && 0 <= v138 && v138 < 4);
                v135[v141] = v143;
                v138 += 1 ;
            }
            v136 += 1 ;
        }
        int v144;
        v144 = 0;
        int v145;
        v145 = 0;
        while (while_method_6(v145)){
            int v147;
            v147 = 0;
            while (while_method_0(v147)){
                assert("Tensor range check" && 0 <= v145 && v145 < 1);
                assert("Tensor range check" && 0 <= v147 && v147 < 4);
                int v149;
                v149 = 4 * v145;
                int v150;
                v150 = v149 + v147;
                int v151;
                v151 = v135[v150];
                int v152;
                v152 = v144 + v151;
                v144 = v152;
                v147 += 1 ;
            }
            v145 += 1 ;
        }
        auto v153 = cooperative_groups::coalesced_threads();
        int v154;
        v154 = threadIdx.x;
        auto v155 = cooperative_groups::labeled_partition(v153,v154);
        Closure2 v156{};
        int v157;
        v157 = cooperative_groups::reduce(v155, v144, v156);
        float v158;
        v158 = (float)v157;
        float v159;
        v159 = 1.0f / v158;
        float v160[4];
        int v161;
        v161 = 0;
        while (while_method_6(v161)){
            int v163;
            v163 = 0;
            while (while_method_0(v163)){
                assert("Tensor range check" && 0 <= v161 && v161 < 1);
                assert("Tensor range check" && 0 <= v163 && v163 < 4);
                int v165;
                v165 = 4 * v161;
                int v166;
                v166 = v165 + v163;
                float v167;
                v167 = v109[v166];
                bool v168;
                v168 = v99[v166];
                bool v169;
                v169 = v168 == false;
                float v174;
                if (v169){
                    v174 = 0.0f;
                } else {
                    bool v170;
                    v170 = v134 == 0.0f;
                    bool v171;
                    v171 = v170 != true;
                    if (v171){
                        float v172;
                        v172 = v167 / v134;
                        v174 = v172;
                    } else {
                        v174 = v159;
                    }
                }
                assert("Tensor range check" && 0 <= v161 && v161 < 1);
                assert("Tensor range check" && 0 <= v163 && v163 < 4);
                v160[v166] = v174;
                v163 += 1 ;
            }
            v161 += 1 ;
        }
        float v175; int v176;
        Tuple2 tmp15 = Tuple2{0.0f, 2147483647};
        v175 = tmp15.v0; v176 = tmp15.v1;
        int v177;
        v177 = 0;
        while (while_method_6(v177)){
            int v179;
            v179 = 0;
            while (while_method_0(v179)){
                assert("Tensor range check" && 0 <= v177 && v177 < 1);
                assert("Tensor range check" && 0 <= v179 && v179 < 4);
                int v181;
                v181 = 4 * v177;
                int v182;
                v182 = v181 + v179;
                float v183;
                v183 = v160[v182];
                int v184;
                v184 = v67[v182];
                bool v185;
                v185 = v176 == v60;
                float v189; int v190;
                if (v185){
                    v189 = v175; v190 = v176;
                } else {
                    bool v186;
                    v186 = v184 == v60;
                    if (v186){
                        v189 = v183; v190 = v184;
                    } else {
                        v189 = v175; v190 = v176;
                    }
                }
                v175 = v189;
                v176 = v190;
                v179 += 1 ;
            }
            v177 += 1 ;
        }
        auto v191 = cooperative_groups::coalesced_threads();
        int v192;
        v192 = threadIdx.x;
        auto v193 = cooperative_groups::labeled_partition(v191,v192);
        Closure7 v194{v60};
        float v195; int v196;
        Tuple2 tmp16 = cooperative_groups::reduce(v193, Tuple2{v175, v176}, v194);
        v195 = tmp16.v0; v196 = tmp16.v1;
        bool v197;
        v197 = v196 == 2147483647;
        bool v198;
        v198 = v197 != true;
        bool v199;
        v199 = v198 == false;
        if (v199){
            assert("Expected a valid action id in get_action." && v198);
        } else {
        }
        int v201;
        v201 = 0;
        while (while_method_6(v201)){
            assert("Tensor range check" && 0 <= v201 && v201 < 1);
            assert("Tensor range check" && 0 <= v201 && v201 < 1);
            v201 += 1 ;
        }
        assert("Tensor range check" && 0 <= v57 && v57 < 256);
        v36[v57] = v195;
        v46 += 1 ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0));
    assert("Tensor range check" && 0 <= v38 && v38 < 256);
    float v203;
    v203 = v36[v38];
    asm("barrier.cta.sync %0;" :: "r"(0));
    return v203;
}
__device__ int int_range_6(int v0, int v1, curandStatePhilox4_32_10_t & v2){
    int v3;
    v3 = v0 - v1;
    unsigned int v4;
    v4 = (unsigned int)v3;
    unsigned int v5;
    v5 = loop_2(v4, v2);
    unsigned int v6;
    v6 = (unsigned int)v1;
    unsigned int v7;
    v7 = v5 + v6;
    int v8;
    v8 = (int)v7;
    return v8;
}
__device__ int tag_8(Union2 v0){
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
__device__ bool is_pair_9(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
__device__ Tuple1 order_10(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple1{v1, v0};
    } else {
        return Tuple1{v0, v1};
    }
}
__device__ Union11 compare_hands_7(Union5 v0, bool v1, static_array<Union2,2> v2, int v3, static_array<int,2> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            __trap();
            break;
        }
        case 1: { // Some
            Union2 v7 = v0.case1.v0;
            int v8;
            v8 = tag_8(v7);
            Union2 v9;
            v9 = v2[0];
            int v11;
            v11 = tag_8(v9);
            Union2 v12;
            v12 = v2[1];
            int v14;
            v14 = tag_8(v12);
            bool v15;
            v15 = is_pair_9(v8, v11);
            bool v16;
            v16 = is_pair_9(v8, v14);
            if (v15){
                if (v16){
                    bool v17;
                    v17 = v11 < v14;
                    if (v17){
                        return Union11{Union11_2{}};
                    } else {
                        bool v19;
                        v19 = v11 > v14;
                        if (v19){
                            return Union11{Union11_1{}};
                        } else {
                            return Union11{Union11_0{}};
                        }
                    }
                } else {
                    return Union11{Union11_1{}};
                }
            } else {
                if (v16){
                    return Union11{Union11_2{}};
                } else {
                    int v27; int v28;
                    Tuple1 tmp24 = order_10(v8, v11);
                    v27 = tmp24.v0; v28 = tmp24.v1;
                    int v29; int v30;
                    Tuple1 tmp25 = order_10(v8, v14);
                    v29 = tmp25.v0; v30 = tmp25.v1;
                    bool v31;
                    v31 = v27 < v29;
                    Union11 v37;
                    if (v31){
                        v37 = Union11{Union11_2{}};
                    } else {
                        bool v33;
                        v33 = v27 > v29;
                        if (v33){
                            v37 = Union11{Union11_1{}};
                        } else {
                            v37 = Union11{Union11_0{}};
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
                            return Union11{Union11_2{}};
                        } else {
                            bool v41;
                            v41 = v28 > v30;
                            if (v41){
                                return Union11{Union11_1{}};
                            } else {
                                return Union11{Union11_0{}};
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
__device__ void method_0(unsigned char * v0, unsigned char * v1, StackMut0 & v2, int v3, Union4 v4){
    v2.v0 = 63u;
    static_array<float,2> v5;
    v5[0] = 0.0f;
    v5[1] = 0.0f;
    v2.v4 = v5;
    static_array_list<Union1,32> & v7 = v2.v2;
    v7.unsafe_set_length(0);
    static_array<Union0,2> v8;
    Union0 v10;
    v10 = Union0{Union0_0{}};
    v8[0] = v10;
    Union0 v12;
    v12 = Union0{Union0_0{}};
    v8[1] = v12;
    int v14;
    v14 = v3 ^ 1;
    Union0 v15;
    v15 = Union0{Union0_2{}};
    v8[v14] = v15;
    v2.v3 = v8;
    static_array_list<Union1,32> & v17 = v2.v2;
    Union6 v18;
    v18 = Union6{Union6_1{v4}};
    Union6 v19;
    v19 = v18;
    while (while_method_3(v19)){
        Union6 v1033;
        switch (v19.tag) {
            case 0: { // None
                v1033 = Union6{Union6_0{}};
                break;
            }
            case 1: { // Some
                Union4 v21 = v19.case1.v0;
                switch (v21.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v978 = v21.case0.v0; bool v979 = v21.case0.v1; static_array<Union2,2> v980 = v21.case0.v2; int v981 = v21.case0.v3; static_array<int,2> v982 = v21.case0.v4; int v983 = v21.case0.v5;
                        curandStatePhilox4_32_10_t & v984 = v2.v5;
                        curandStatePhilox4_32_10_t & v985 = v984;
                        unsigned int & v986 = v2.v0;
                        Union2 v987; unsigned int v988;
                        Tuple0 tmp0 = draw_card_1(v985, v986);
                        v987 = tmp0.v0; v988 = tmp0.v1;
                        v2.v0 = v988;
                        Union1 v989;
                        v989 = Union1{Union1_0{v987}};
                        v17.push(v989);
                        int v990;
                        v990 = 2;
                        int v991; int v992;
                        Tuple1 tmp1 = Tuple1{0, 0};
                        v991 = tmp1.v0; v992 = tmp1.v1;
                        while (while_method_2(v991)){
                            int v994;
                            v994 = v982[v991];
                            bool v996;
                            v996 = v992 >= v994;
                            int v997;
                            if (v996){
                                v997 = v992;
                            } else {
                                v997 = v994;
                            }
                            v992 = v997;
                            v991 += 1 ;
                        }
                        static_array<int,2> v998;
                        int v1000;
                        v1000 = 0;
                        while (while_method_2(v1000)){
                            v998[v1000] = v992;
                            v1000 += 1 ;
                        }
                        Union5 v1002;
                        v1002 = Union5{Union5_1{v987}};
                        Union4 v1003;
                        v1003 = Union4{Union4_2{v1002, true, v980, 0, v998, v990}};
                        v1033 = Union6{Union6_1{v1003}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v1005 = v2.v5;
                        curandStatePhilox4_32_10_t & v1006 = v1005;
                        unsigned int & v1007 = v2.v0;
                        Union2 v1008; unsigned int v1009;
                        Tuple0 tmp2 = draw_card_1(v1006, v1007);
                        v1008 = tmp2.v0; v1009 = tmp2.v1;
                        v2.v0 = v1009;
                        curandStatePhilox4_32_10_t & v1010 = v2.v5;
                        curandStatePhilox4_32_10_t & v1011 = v1010;
                        unsigned int & v1012 = v2.v0;
                        Union2 v1013; unsigned int v1014;
                        Tuple0 tmp3 = draw_card_1(v1011, v1012);
                        v1013 = tmp3.v0; v1014 = tmp3.v1;
                        v2.v0 = v1014;
                        Union1 v1015;
                        v1015 = Union1{Union1_2{0, v1008}};
                        v17.push(v1015);
                        Union1 v1016;
                        v1016 = Union1{Union1_2{1, v1013}};
                        v17.push(v1016);
                        int v1017;
                        v1017 = 2;
                        static_array<int,2> v1018;
                        v1018[0] = 1;
                        v1018[1] = 1;
                        static_array<Union2,2> v1020;
                        v1020[0] = v1008;
                        v1020[1] = v1013;
                        Union5 v1022;
                        v1022 = Union5{Union5_0{}};
                        Union4 v1023;
                        v1023 = Union4{Union4_2{v1022, true, v1020, 0, v1018, v1017}};
                        v1033 = Union6{Union6_1{v1023}};
                        break;
                    }
                    case 2: { // Round
                        Union5 v64 = v21.case2.v0; bool v65 = v21.case2.v1; static_array<Union2,2> v66 = v21.case2.v2; int v67 = v21.case2.v3; static_array<int,2> v68 = v21.case2.v4; int v69 = v21.case2.v5;
                        static_array<Union0,2> & v70 = v2.v3;
                        Union0 v71;
                        v71 = v70[v67];
                        Union3 v794;
                        switch (v71.tag) {
                            case 0: { // Computer
                                static_array_list<Union1,32> & v74 = v2.v2;
                                curandStatePhilox4_32_10_t & v75 = v2.v5;
                                curandStatePhilox4_32_10_t & v76 = v75;
                                unsigned int * v77;
                                v77 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
                                float * v79;
                                v79 = reinterpret_cast<float *>(&v0[0ull]);
                                int v81;
                                v81 = threadIdx.x;
                                int v82;
                                v82 = blockIdx.x;
                                int v83;
                                v83 = v82 * 256;
                                int v84;
                                v84 = v81 + v83;
                                unsigned long long v85;
                                v85 = (unsigned long long)v84;
                                curandStatePhilox4_32_10_t v86;
                                curand_init(12344321ull,v85,0ull,&v86);
                                float * v87;
                                v87 = reinterpret_cast<float *>(&v0[0ull]);
                                int v89;
                                v89 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v89 && v89 < 24);
                                int v90;
                                v90 = 32768 * v89;
                                int v91;
                                v91 = threadIdx.x;
                                int v92;
                                v92 = blockIdx.x;
                                int v93;
                                v93 = v92 * 256;
                                int v94;
                                v94 = v91 + v93;
                                unsigned long long v95;
                                v95 = (unsigned long long)v94;
                                curandStatePhilox4_32_10_t v96;
                                curand_init(12344321ull,v95,0ull,&v96);
                                int v97;
                                v97 = threadIdx.x;
                                int v98;
                                v98 = v97;
                                while (while_method_4(v98)){
                                    bool v100;
                                    v100 = 0 <= v98;
                                    bool v101;
                                    v101 = v100 == false;
                                    if (v101){
                                        assert("The index needs to be zero or positive." && v100);
                                    } else {
                                    }
                                    int v103;
                                    v103 = v98 % 128;
                                    int v104;
                                    v104 = v98 / 128;
                                    bool v105;
                                    v105 = v104 < 256;
                                    bool v106;
                                    v106 = v105 == false;
                                    if (v106){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v105);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v104 && v104 < 256);
                                    assert("Tensor range check" && 0 <= v103 && v103 < 128);
                                    int v108;
                                    v108 = v103 + v90;
                                    int v109;
                                    v109 = 128 * v104;
                                    int v110;
                                    v110 = v109 + v108;
                                    v87[v110] = 0.0f;
                                    v98 += 256 ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                int v111;
                                v111 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v111 && v111 < 256);
                                int v112;
                                v112 = 128 * v111;
                                int v113;
                                v113 = v112 + v90;
                                static_array_list<Union7,10> v114;
                                v114 = static_array_list<Union7,10>{};
                                int v116;
                                v116 = v74.length;
                                int v117;
                                v117 = 0;
                                while (while_method_5(v116, v117)){
                                    Union1 v119;
                                    v119 = v74[v117];
                                    Union8 v138;
                                    switch (v119.tag) {
                                        case 0: { // CommunityCardIs
                                            Union2 v128 = v119.case0.v0;
                                            Union7 v129;
                                            v129 = Union7{Union7_1{v128}};
                                            v138 = Union8{Union8_1{v129}};
                                            break;
                                        }
                                        case 1: { // PlayerAction
                                            int v131 = v119.case1.v0; Union3 v132 = v119.case1.v1;
                                            Union7 v133;
                                            v133 = Union7{Union7_0{v132}};
                                            v138 = Union8{Union8_1{v133}};
                                            break;
                                        }
                                        case 2: { // PlayerGotCard
                                            int v121 = v119.case2.v0; Union2 v122 = v119.case2.v1;
                                            bool v123;
                                            v123 = v121 == v67;
                                            if (v123){
                                                Union7 v124;
                                                v124 = Union7{Union7_1{v122}};
                                                v138 = Union8{Union8_1{v124}};
                                            } else {
                                                v138 = Union8{Union8_0{}};
                                            }
                                            break;
                                        }
                                        default: {
                                            v138 = Union8{Union8_0{}};
                                        }
                                    }
                                    switch (v138.tag) {
                                        case 0: { // None
                                            break;
                                        }
                                        case 1: { // Some
                                            Union7 v139 = v138.case1.v0;
                                            v114.push(v139);
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false); __trap();
                                        }
                                    }
                                    v117 += 1 ;
                                }
                                float * v140;
                                v140 = v87+v113;
                                int v142;
                                v142 = v114.length;
                                bool v143;
                                v143 = v142 == 0;
                                if (v143){
                                    v140[0] = 1.0f;
                                } else {
                                }
                                int v144;
                                v144 = v114.length;
                                int v145;
                                v145 = 0;
                                while (while_method_5(v144, v145)){
                                    Union7 v147;
                                    v147 = v114[v145];
                                    int v149;
                                    v149 = v145 * 6;
                                    int v150;
                                    v150 = 1 + v149;
                                    switch (v147.tag) {
                                        case 0: { // C1of2
                                            Union3 v151 = v147.case0.v0;
                                            switch (v151.tag) {
                                                case 0: { // Call
                                                    v140[v150] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // Fold
                                                    int v152;
                                                    v152 = v150 + 1;
                                                    v140[v152] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // Raise
                                                    int v153;
                                                    v153 = v150 + 2;
                                                    v140[v153] = 1.0f;
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false); __trap();
                                                }
                                            }
                                            break;
                                        }
                                        case 1: { // C2of2
                                            Union2 v154 = v147.case1.v0;
                                            int v155;
                                            v155 = v150 + 3;
                                            switch (v154.tag) {
                                                case 0: { // Jack
                                                    v140[v155] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // King
                                                    int v156;
                                                    v156 = v155 + 1;
                                                    v140[v156] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // Queen
                                                    int v157;
                                                    v157 = v155 + 2;
                                                    v140[v157] = 1.0f;
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
                                    v145 += 1 ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                int v158;
                                v158 = 0;
                                while (while_method_0(v158)){
                                    float * v160;
                                    v160 = reinterpret_cast<float *>(&v0[0ull]);
                                    float * v162;
                                    v162 = reinterpret_cast<float *>(&v1[0ull]);
                                    assert("Tensor range check" && 0 <= v158 && v158 < 4);
                                    int v164;
                                    v164 = 16384 * v158;
                                    float * v165;
                                    v165 = reinterpret_cast<float *>(&v0[3145728ull]);
                                    int v167;
                                    v167 = blockIdx.x;
                                    assert("Tensor range check" && 0 <= v167 && v167 < 24);
                                    int v168;
                                    v168 = 32768 * v167;
                                    int v169;
                                    v169 = blockIdx.x;
                                    assert("Tensor range check" && 0 <= v169 && v169 < 24);
                                    int v170;
                                    v170 = 32768 * v169;
                                    cuda::pipeline<cuda::thread_scope_thread> v171 = cuda::make_pipeline();
                                    extern __shared__ unsigned char v172[];
                                    float * v173;
                                    v173 = reinterpret_cast<float *>(&v172[0ull]);
                                    float * v175;
                                    v175 = reinterpret_cast<float *>(&v172[34816ull]);
                                    float * v177;
                                    v177 = reinterpret_cast<float *>(&v172[0ull]);
                                    int v179;
                                    v179 = threadIdx.x;
                                    int v180;
                                    v180 = v179 / 32;
                                    bool v181;
                                    v181 = 0 <= v180;
                                    bool v182;
                                    v182 = v181 == false;
                                    if (v182){
                                        assert("The index needs to be zero or positive." && v181);
                                    } else {
                                    }
                                    int v184;
                                    v184 = v180 % 8;
                                    int v185;
                                    v185 = v180 / 8;
                                    bool v186;
                                    v186 = v185 < 1;
                                    bool v187;
                                    v187 = v186 == false;
                                    if (v187){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v186);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v185 && v185 < 1);
                                    assert("Tensor range check" && 0 <= v184 && v184 < 8);
                                    int v189;
                                    v189 = 16 * v184;
                                    int v190;
                                    v190 = 17408 * v185;
                                    int v191;
                                    v191 = v190 + v189;
                                    float * v192;
                                    v192 = v177+v191;
                                    assert("Tensor range check" && 0 <= v185 && v185 < 1);
                                    int v194;
                                    v194 = 8704 * v185;
                                    int v195;
                                    v195 = threadIdx.x;
                                    int v196;
                                    v196 = v195 % 32;
                                    bool v197;
                                    v197 = 0 <= v196;
                                    bool v198;
                                    v198 = v197 == false;
                                    if (v198){
                                        assert("The index needs to be zero or positive." && v197);
                                    } else {
                                    }
                                    int v200;
                                    v200 = v196 % 4;
                                    int v201;
                                    v201 = v196 / 4;
                                    bool v202;
                                    v202 = v201 < 8;
                                    bool v203;
                                    v203 = v202 == false;
                                    if (v203){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v202);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v201 && v201 < 8);
                                    assert("Tensor range check" && 0 <= v200 && v200 < 4);
                                    int v205;
                                    v205 = v200 + v194;
                                    int v206;
                                    v206 = 68 * v201;
                                    int v207;
                                    v207 = v206 + v205;
                                    float * v208;
                                    v208 = v173+v207;
                                    assert("Tensor range check" && 0 <= v184 && v184 < 8);
                                    int v210;
                                    v210 = 1088 * v184;
                                    int v211;
                                    v211 = threadIdx.x;
                                    int v212;
                                    v212 = v211 % 32;
                                    bool v213;
                                    v213 = 0 <= v212;
                                    bool v214;
                                    v214 = v213 == false;
                                    if (v214){
                                        assert("The index needs to be zero or positive." && v213);
                                    } else {
                                    }
                                    int v216;
                                    v216 = v212 % 4;
                                    int v217;
                                    v217 = v212 / 4;
                                    bool v218;
                                    v218 = v217 < 8;
                                    bool v219;
                                    v219 = v218 == false;
                                    if (v219){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v218);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v217 && v217 < 8);
                                    assert("Tensor range check" && 0 <= v216 && v216 < 4);
                                    int v221;
                                    v221 = v216 + v210;
                                    int v222;
                                    v222 = 68 * v217;
                                    int v223;
                                    v223 = v222 + v221;
                                    float * v224;
                                    v224 = v175+v223;
                                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> v226[8];
                                    int v227;
                                    v227 = 0;
                                    while (while_method_2(v227)){
                                        int v229;
                                        v229 = 0;
                                        while (while_method_6(v229)){
                                            assert("Tensor range check" && 0 <= v227 && v227 < 2);
                                            assert("Tensor range check" && 0 <= v229 && v229 < 1);
                                            int v231;
                                            v231 = 128 * v229;
                                            int v232;
                                            v232 = v231 + v170;
                                            int v233;
                                            v233 = 16384 * v227;
                                            int v234;
                                            v234 = v233 + v232;
                                            float * v235;
                                            v235 = v165+v234;
                                            // Pushing the loop unrolling to: 0
                                            int v237;
                                            v237 = 0;
                                            #pragma unroll
                                            while (while_method_1(v237)){
                                                int v239;
                                                v239 = 0;
                                                #pragma unroll
                                                while (while_method_6(v239)){
                                                    assert("Tensor range check" && 0 <= v237 && v237 < 8);
                                                    assert("Tensor range check" && 0 <= v239 && v239 < 1);
                                                    int v241;
                                                    v241 = v237 + v239;
                                                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v242 = v226[v241];
                                                    wmma::fill_fragment(v242, 0.0f);
                                                    v239 += 1 ;
                                                }
                                                v237 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            int v243;
                                            v243 = 0;
                                            while (while_method_7(v243)){
                                                int v245;
                                                v245 = v243 + 1;
                                                bool v246;
                                                v246 = v243 == 0;
                                                int v247;
                                                v247 = v243 % 2;
                                                bool v248;
                                                v248 = 0 <= v243;
                                                bool v249;
                                                v249 = v248 == false;
                                                if (v249){
                                                    assert("The index needs to be zero or positive." && v248);
                                                } else {
                                                }
                                                bool v251;
                                                v251 = v243 < 2;
                                                bool v252;
                                                v252 = v251 == false;
                                                if (v252){
                                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v251);
                                                } else {
                                                }
                                                bool v254;
                                                v254 = v245 < 2;
                                                Union9 v260;
                                                if (v254){
                                                    bool v255;
                                                    v255 = 0 <= v245;
                                                    bool v256;
                                                    v256 = v255 == false;
                                                    if (v256){
                                                        assert("The index needs to be zero or positive." && v255);
                                                    } else {
                                                    }
                                                    v260 = Union9{Union9_1{v245}};
                                                } else {
                                                    v260 = Union9{Union9_0{}};
                                                }
                                                assert("Tensor range check" && 0 <= v227 && v227 < 2);
                                                int v261;
                                                v261 = v233 + v168;
                                                assert("Tensor range check" && 0 <= v243 && v243 < 2);
                                                int v262;
                                                v262 = 64 * v243;
                                                int v263;
                                                v263 = v262 + v261;
                                                float * v264;
                                                v264 = v160+v263;
                                                assert("Tensor range check" && 0 <= v229 && v229 < 1);
                                                int v266;
                                                v266 = 16384 * v229;
                                                int v267;
                                                v267 = v266 + v164;
                                                if (v246){
                                                    assert("Tensor range check" && 0 <= v243 && v243 < 2);
                                                    int v268;
                                                    v268 = v262 + v267;
                                                    float * v269;
                                                    v269 = v162+v268;
                                                    // Pushing the loop unrolling to: 0
                                                    v171.producer_acquire();
                                                    int v271;
                                                    v271 = threadIdx.x;
                                                    bool v272;
                                                    v272 = 0 <= v271;
                                                    bool v273;
                                                    v273 = v272 == false;
                                                    if (v273){
                                                        assert("The index needs to be zero or positive." && v272);
                                                    } else {
                                                    }
                                                    int v275;
                                                    v275 = v271 % 16;
                                                    int v276;
                                                    v276 = v271 / 16;
                                                    bool v277;
                                                    v277 = v276 < 16;
                                                    bool v278;
                                                    v278 = v277 == false;
                                                    if (v278){
                                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v277);
                                                    } else {
                                                    }
                                                    assert("Tensor range check" && 0 <= v276 && v276 < 16);
                                                    assert("Tensor range check" && 0 <= v275 && v275 < 16);
                                                    int v280;
                                                    v280 = 4 * v275;
                                                    int v281;
                                                    v281 = 68 * v276;
                                                    int v282;
                                                    v282 = v281 + v280;
                                                    int v283;
                                                    v283 = 128 * v276;
                                                    int v284;
                                                    v284 = v283 + v280;
                                                    float * v285;
                                                    v285 = v175+v282;
                                                    float * v287;
                                                    v287 = v269+v284;
                                                    int v289;
                                                    v289 = 0;
                                                    #pragma unroll
                                                    while (while_method_1(v289)){
                                                        int v291;
                                                        v291 = 0;
                                                        #pragma unroll
                                                        while (while_method_6(v291)){
                                                            assert("Tensor range check" && 0 <= v289 && v289 < 8);
                                                            assert("Tensor range check" && 0 <= v291 && v291 < 1);
                                                            int v293;
                                                            v293 = 64 * v291;
                                                            int v294;
                                                            v294 = 1088 * v289;
                                                            int v295;
                                                            v295 = v294 + v293;
                                                            int v296;
                                                            v296 = 2048 * v289;
                                                            int v297;
                                                            v297 = v296 + v293;
                                                            constexpr int v298 = sizeof(float) * 4;
                                                            assert("Pointer alignment check" && (unsigned long long)(v287 + v297) % v298 == 0 && (unsigned long long)(v285 + v295) % v298 == 0);
                                                            cuda::memcpy_async(v285 + v295, v287 + v297, cuda::aligned_size_t<v298>(v298), v171);
                                                            v291 += 1 ;
                                                        }
                                                        v289 += 1 ;
                                                    }
                                                    v171.producer_commit();
                                                    // Poping the loop unrolling to: 0
                                                } else {
                                                }
                                                // Pushing the loop unrolling to: 0
                                                int v299;
                                                v299 = threadIdx.x;
                                                bool v300;
                                                v300 = 0 <= v299;
                                                bool v301;
                                                v301 = v300 == false;
                                                if (v301){
                                                    assert("The index needs to be zero or positive." && v300);
                                                } else {
                                                }
                                                int v303;
                                                v303 = v299 % 16;
                                                int v304;
                                                v304 = v299 / 16;
                                                bool v305;
                                                v305 = v304 < 16;
                                                bool v306;
                                                v306 = v305 == false;
                                                if (v306){
                                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v305);
                                                } else {
                                                }
                                                assert("Tensor range check" && 0 <= v304 && v304 < 16);
                                                assert("Tensor range check" && 0 <= v303 && v303 < 16);
                                                int v308;
                                                v308 = 4 * v303;
                                                int v309;
                                                v309 = 68 * v304;
                                                int v310;
                                                v310 = v309 + v308;
                                                int v311;
                                                v311 = 128 * v304;
                                                int v312;
                                                v312 = v311 + v308;
                                                float * v313;
                                                v313 = v173+v310;
                                                float * v315;
                                                v315 = v264+v312;
                                                int v317;
                                                v317 = 0;
                                                #pragma unroll
                                                while (while_method_1(v317)){
                                                    int v319;
                                                    v319 = 0;
                                                    #pragma unroll
                                                    while (while_method_6(v319)){
                                                        assert("Tensor range check" && 0 <= v317 && v317 < 8);
                                                        assert("Tensor range check" && 0 <= v319 && v319 < 1);
                                                        int v321;
                                                        v321 = 64 * v319;
                                                        int v322;
                                                        v322 = 1088 * v317;
                                                        int v323;
                                                        v323 = v322 + v321;
                                                        int v324;
                                                        v324 = 2048 * v317;
                                                        int v325;
                                                        v325 = v324 + v321;
                                                        int4* v326;
                                                        v326 = reinterpret_cast<int4*>(v315 + v325);
                                                        int4* v327;
                                                        v327 = reinterpret_cast<int4*>(v313 + v323);
                                                        assert("Pointer alignment check" && (unsigned long long)(v326) % 4 == 0 && (unsigned long long)(v327) % 4 == 0);
                                                        *v327 = *v326;
                                                        v319 += 1 ;
                                                    }
                                                    v317 += 1 ;
                                                }
                                                // Poping the loop unrolling to: 0
                                                wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v328[1];
                                                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v329[8];
                                                cuda::pipeline_consumer_wait_prior<0>(v171);;
                                                asm("barrier.cta.sync %0;" :: "r"(0));
                                                // Pushing the loop unrolling to: 0
                                                int v330;
                                                v330 = 0;
                                                #pragma unroll
                                                while (while_method_6(v330)){
                                                    int v332;
                                                    v332 = 0;
                                                    #pragma unroll
                                                    while (while_method_1(v332)){
                                                        assert("Tensor range check" && 0 <= v330 && v330 < 1);
                                                        assert("Tensor range check" && 0 <= v332 && v332 < 8);
                                                        int v334;
                                                        v334 = 8 * v330;
                                                        int v335;
                                                        v335 = v334 + v332;
                                                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v336 = v329[v335];
                                                        assert("Tensor range check" && 0 <= v330 && v330 < 1);
                                                        int v337;
                                                        v337 = 1088 * v330;
                                                        assert("Tensor range check" && 0 <= v332 && v332 < 8);
                                                        int v338;
                                                        v338 = 8 * v332;
                                                        int v339;
                                                        v339 = v338 + v337;
                                                        int v340;
                                                        v340 = 0;
                                                        #pragma unroll
                                                        while (while_method_2(v340)){
                                                            int v342;
                                                            v342 = 0;
                                                            #pragma unroll
                                                            while (while_method_2(v342)){
                                                                assert("Tensor range check" && 0 <= v340 && v340 < 2);
                                                                assert("Tensor range check" && 0 <= v342 && v342 < 2);
                                                                int v344;
                                                                v344 = 4 * v342;
                                                                int v345;
                                                                v345 = v344 + v339;
                                                                int v346;
                                                                v346 = 544 * v340;
                                                                int v347;
                                                                v347 = v346 + v345;
                                                                float v348;
                                                                v348 = v224[v347];
                                                                bool v349;
                                                                v349 = 0 <= v342;
                                                                bool v351;
                                                                if (v349){
                                                                    bool v350;
                                                                    v350 = v342 < 2;
                                                                    v351 = v350;
                                                                } else {
                                                                    v351 = false;
                                                                }
                                                                bool v352;
                                                                v352 = v351 == false;
                                                                if (v352){
                                                                    assert("The indices should be inside the range of the dimension." && v351);
                                                                } else {
                                                                }
                                                                bool v354;
                                                                v354 = 0 <= v340;
                                                                bool v356;
                                                                if (v354){
                                                                    bool v355;
                                                                    v355 = v340 < 2;
                                                                    v356 = v355;
                                                                } else {
                                                                    v356 = false;
                                                                }
                                                                bool v357;
                                                                v357 = v356 == false;
                                                                if (v357){
                                                                    assert("The indices should be inside the range of the dimension." && v356);
                                                                } else {
                                                                }
                                                                int v359;
                                                                v359 = v340 * 2;
                                                                int v360;
                                                                v360 = v342 + v359;
                                                                v336.x[v360] = wmma::__float_to_tf32(v348);
                                                                v342 += 1 ;
                                                            }
                                                            v340 += 1 ;
                                                        }
                                                        v332 += 1 ;
                                                    }
                                                    v330 += 1 ;
                                                }
                                                // Poping the loop unrolling to: 0
                                                v171.consumer_release();
                                                switch (v260.tag) {
                                                    case 0: { // None
                                                        break;
                                                    }
                                                    case 1: { // Some
                                                        int v361 = v260.case1.v0;
                                                        assert("Tensor range check" && 0 <= v361 && v361 < 2);
                                                        int v362;
                                                        v362 = 64 * v361;
                                                        int v363;
                                                        v363 = v362 + v267;
                                                        float * v364;
                                                        v364 = v162+v363;
                                                        asm("barrier.cta.sync %0;" :: "r"(0));
                                                        // Pushing the loop unrolling to: 0
                                                        v171.producer_acquire();
                                                        int v366;
                                                        v366 = threadIdx.x;
                                                        bool v367;
                                                        v367 = 0 <= v366;
                                                        bool v368;
                                                        v368 = v367 == false;
                                                        if (v368){
                                                            assert("The index needs to be zero or positive." && v367);
                                                        } else {
                                                        }
                                                        int v370;
                                                        v370 = v366 % 16;
                                                        int v371;
                                                        v371 = v366 / 16;
                                                        bool v372;
                                                        v372 = v371 < 16;
                                                        bool v373;
                                                        v373 = v372 == false;
                                                        if (v373){
                                                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v372);
                                                        } else {
                                                        }
                                                        assert("Tensor range check" && 0 <= v371 && v371 < 16);
                                                        assert("Tensor range check" && 0 <= v370 && v370 < 16);
                                                        int v375;
                                                        v375 = 4 * v370;
                                                        int v376;
                                                        v376 = 68 * v371;
                                                        int v377;
                                                        v377 = v376 + v375;
                                                        int v378;
                                                        v378 = 128 * v371;
                                                        int v379;
                                                        v379 = v378 + v375;
                                                        float * v380;
                                                        v380 = v175+v377;
                                                        float * v382;
                                                        v382 = v364+v379;
                                                        int v384;
                                                        v384 = 0;
                                                        #pragma unroll
                                                        while (while_method_1(v384)){
                                                            int v386;
                                                            v386 = 0;
                                                            #pragma unroll
                                                            while (while_method_6(v386)){
                                                                assert("Tensor range check" && 0 <= v384 && v384 < 8);
                                                                assert("Tensor range check" && 0 <= v386 && v386 < 1);
                                                                int v388;
                                                                v388 = 64 * v386;
                                                                int v389;
                                                                v389 = 1088 * v384;
                                                                int v390;
                                                                v390 = v389 + v388;
                                                                int v391;
                                                                v391 = 2048 * v384;
                                                                int v392;
                                                                v392 = v391 + v388;
                                                                constexpr int v393 = sizeof(float) * 4;
                                                                assert("Pointer alignment check" && (unsigned long long)(v382 + v392) % v393 == 0 && (unsigned long long)(v380 + v390) % v393 == 0);
                                                                cuda::memcpy_async(v380 + v390, v382 + v392, cuda::aligned_size_t<v393>(v393), v171);
                                                                v386 += 1 ;
                                                            }
                                                            v384 += 1 ;
                                                        }
                                                        v171.producer_commit();
                                                        // Poping the loop unrolling to: 0
                                                        break;
                                                    }
                                                    default: {
                                                        assert("Invalid tag." && false); __trap();
                                                    }
                                                }
                                                // Pushing the loop unrolling to: 0
                                                int v394;
                                                v394 = 0;
                                                #pragma unroll
                                                while (while_method_1(v394)){
                                                    int v396;
                                                    v396 = 0;
                                                    #pragma unroll
                                                    while (while_method_1(v396)){
                                                        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v398 = v328[0];
                                                        assert("Tensor range check" && 0 <= v394 && v394 < 8);
                                                        int v399;
                                                        v399 = 1088 * v394;
                                                        assert("Tensor range check" && 0 <= v396 && v396 < 8);
                                                        int v400;
                                                        v400 = 8 * v396;
                                                        int v401;
                                                        v401 = v400 + v399;
                                                        int v402;
                                                        v402 = 0;
                                                        #pragma unroll
                                                        while (while_method_2(v402)){
                                                            int v404;
                                                            v404 = 0;
                                                            #pragma unroll
                                                            while (while_method_2(v404)){
                                                                assert("Tensor range check" && 0 <= v402 && v402 < 2);
                                                                assert("Tensor range check" && 0 <= v404 && v404 < 2);
                                                                int v406;
                                                                v406 = 544 * v404;
                                                                int v407;
                                                                v407 = v406 + v401;
                                                                int v408;
                                                                v408 = 4 * v402;
                                                                int v409;
                                                                v409 = v408 + v407;
                                                                float v410;
                                                                v410 = v208[v409];
                                                                bool v411;
                                                                v411 = 0 <= v404;
                                                                bool v413;
                                                                if (v411){
                                                                    bool v412;
                                                                    v412 = v404 < 2;
                                                                    v413 = v412;
                                                                } else {
                                                                    v413 = false;
                                                                }
                                                                bool v414;
                                                                v414 = v413 == false;
                                                                if (v414){
                                                                    assert("The indices should be inside the range of the dimension." && v413);
                                                                } else {
                                                                }
                                                                bool v416;
                                                                v416 = 0 <= v402;
                                                                bool v418;
                                                                if (v416){
                                                                    bool v417;
                                                                    v417 = v402 < 2;
                                                                    v418 = v417;
                                                                } else {
                                                                    v418 = false;
                                                                }
                                                                bool v419;
                                                                v419 = v418 == false;
                                                                if (v419){
                                                                    assert("The indices should be inside the range of the dimension." && v418);
                                                                } else {
                                                                }
                                                                int v421;
                                                                v421 = v402 * 2;
                                                                int v422;
                                                                v422 = v404 + v421;
                                                                v398.x[v422] = wmma::__float_to_tf32(v410);
                                                                v404 += 1 ;
                                                            }
                                                            v402 += 1 ;
                                                        }
                                                        int v423;
                                                        v423 = 0;
                                                        #pragma unroll
                                                        while (while_method_6(v423)){
                                                            assert("Tensor range check" && 0 <= v394 && v394 < 8);
                                                            assert("Tensor range check" && 0 <= v423 && v423 < 1);
                                                            int v425;
                                                            v425 = v394 + v423;
                                                            wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v426 = v226[v425];
                                                            assert("Tensor range check" && 0 <= v423 && v423 < 1);
                                                            assert("Tensor range check" && 0 <= v396 && v396 < 8);
                                                            int v427;
                                                            v427 = 8 * v423;
                                                            int v428;
                                                            v428 = v427 + v396;
                                                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v429 = v329[v428];
                                                            wmma::mma_sync(v426, v398, v429, v426);
                                                            v423 += 1 ;
                                                        }
                                                        v396 += 1 ;
                                                    }
                                                    v394 += 1 ;
                                                }
                                                // Poping the loop unrolling to: 0
                                                asm("barrier.cta.sync %0;" :: "r"(0));
                                                v243 = v245;
                                            }
                                            // Pushing the loop unrolling to: 0
                                            int v430;
                                            v430 = 0;
                                            #pragma unroll
                                            while (while_method_1(v430)){
                                                int v432;
                                                v432 = 0;
                                                #pragma unroll
                                                while (while_method_6(v432)){
                                                    assert("Tensor range check" && 0 <= v430 && v430 < 8);
                                                    assert("Tensor range check" && 0 <= v432 && v432 < 1);
                                                    int v434;
                                                    v434 = v430 + v432;
                                                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v435 = v226[v434];
                                                    assert("Tensor range check" && 0 <= v430 && v430 < 8);
                                                    assert("Tensor range check" && 0 <= v432 && v432 < 1);
                                                    int v436;
                                                    v436 = 16 * v432;
                                                    int v437;
                                                    v437 = 2176 * v430;
                                                    int v438;
                                                    v438 = v437 + v436;
                                                    float * v439;
                                                    v439 = v192+v438;
                                                    wmma::store_matrix_sync(v439, v435, 136, wmma::mem_row_major);
                                                    v432 += 1 ;
                                                }
                                                v430 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            asm("barrier.cta.sync %0;" :: "r"(0));
                                            // Pushing the loop unrolling to: 0
                                            int v441;
                                            v441 = threadIdx.x;
                                            bool v442;
                                            v442 = 0 <= v441;
                                            bool v443;
                                            v443 = v442 == false;
                                            if (v443){
                                                assert("The index needs to be zero or positive." && v442);
                                            } else {
                                            }
                                            int v445;
                                            v445 = v441 % 32;
                                            int v446;
                                            v446 = v441 / 32;
                                            bool v447;
                                            v447 = v446 < 8;
                                            bool v448;
                                            v448 = v447 == false;
                                            if (v448){
                                                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v447);
                                            } else {
                                            }
                                            assert("Tensor range check" && 0 <= v446 && v446 < 8);
                                            assert("Tensor range check" && 0 <= v445 && v445 < 32);
                                            int v450;
                                            v450 = 4 * v445;
                                            int v451;
                                            v451 = 128 * v446;
                                            int v452;
                                            v452 = v451 + v450;
                                            int v453;
                                            v453 = 136 * v446;
                                            int v454;
                                            v454 = v453 + v450;
                                            float * v455;
                                            v455 = v235+v452;
                                            float * v457;
                                            v457 = v177+v454;
                                            int v459;
                                            v459 = 0;
                                            #pragma unroll
                                            while (while_method_8(v459)){
                                                int v461;
                                                v461 = 0;
                                                #pragma unroll
                                                while (while_method_6(v461)){
                                                    assert("Tensor range check" && 0 <= v459 && v459 < 16);
                                                    assert("Tensor range check" && 0 <= v461 && v461 < 1);
                                                    int v463;
                                                    v463 = 128 * v461;
                                                    int v464;
                                                    v464 = 1024 * v459;
                                                    int v465;
                                                    v465 = v464 + v463;
                                                    int v466;
                                                    v466 = 1088 * v459;
                                                    int v467;
                                                    v467 = v466 + v463;
                                                    int4* v468;
                                                    v468 = reinterpret_cast<int4*>(v457 + v467);
                                                    int4* v469;
                                                    v469 = reinterpret_cast<int4*>(v455 + v465);
                                                    assert("Pointer alignment check" && (unsigned long long)(v468) % 4 == 0 && (unsigned long long)(v469) % 4 == 0);
                                                    *v469 = *v468;
                                                    v461 += 1 ;
                                                }
                                                v459 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            asm("barrier.cta.sync %0;" :: "r"(0));
                                            v229 += 1 ;
                                        }
                                        v227 += 1 ;
                                    }
                                    unsigned int * v470;
                                    v470 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
                                    assert("Tensor range check" && 0 <= v158 && v158 < 4);
                                    int v472;
                                    v472 = 6144 * v158;
                                    method_3(v470, v472, v165);
                                    int * v473;
                                    v473 = reinterpret_cast<int *>(&v1[262144ull]);
                                    float * v475;
                                    v475 = reinterpret_cast<float *>(&v1[262160ull]);
                                    float * v477;
                                    v477 = reinterpret_cast<float *>(&v1[524304ull]);
                                    float * v479;
                                    v479 = reinterpret_cast<float *>(&v1[786448ull]);
                                    float * v481;
                                    v481 = reinterpret_cast<float *>(&v1[1048592ull]);
                                    float * v483;
                                    v483 = reinterpret_cast<float *>(&v1[1310736ull]);
                                    float * v485;
                                    v485 = reinterpret_cast<float *>(&v1[1572880ull]);
                                    float * v487;
                                    v487 = reinterpret_cast<float *>(&v1[1835024ull]);
                                    int * v489;
                                    v489 = reinterpret_cast<int *>(&v0[6389760ull]);
                                    float * v491;
                                    v491 = reinterpret_cast<float *>(&v0[7962624ull]);
                                    int * v493;
                                    v493 = reinterpret_cast<int *>(&v0[9535488ull]);
                                    int * v495;
                                    v495 = reinterpret_cast<int *>(&v0[11108352ull]);
                                    double * v497;
                                    v497 = reinterpret_cast<double *>(&v0[12681216ull]);
                                    double * v499;
                                    v499 = reinterpret_cast<double *>(&v0[18972672ull]);
                                    double * v501;
                                    v501 = reinterpret_cast<double *>(&v1[2097168ull]);
                                    double * v503;
                                    v503 = reinterpret_cast<double *>(&v1[2490384ull]);
                                    int * v505;
                                    v505 = reinterpret_cast<int *>(&v1[2883600ull]);
                                    v158 += 1 ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                int * v507;
                                v507 = reinterpret_cast<int *>(&v1[262144ull]);
                                float * v509;
                                v509 = reinterpret_cast<float *>(&v1[262160ull]);
                                float * v511;
                                v511 = reinterpret_cast<float *>(&v1[524304ull]);
                                float * v513;
                                v513 = reinterpret_cast<float *>(&v1[786448ull]);
                                float * v515;
                                v515 = reinterpret_cast<float *>(&v1[1048592ull]);
                                float * v517;
                                v517 = reinterpret_cast<float *>(&v1[1310736ull]);
                                float * v519;
                                v519 = reinterpret_cast<float *>(&v1[1572880ull]);
                                float * v521;
                                v521 = reinterpret_cast<float *>(&v1[1835024ull]);
                                int v523;
                                v523 = v507[0];
                                unsigned int * v524;
                                v524 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
                                int v526;
                                v526 = blockIdx.x;
                                int v527;
                                v527 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v523 && v523 < 4);
                                assert("Tensor range check" && 0 <= v526 && v526 < 24);
                                assert("Tensor range check" && 0 <= v527 && v527 < 256);
                                int v528;
                                v528 = 256 * v526;
                                int v529;
                                v529 = v528 + v527;
                                int v530;
                                v530 = 6144 * v523;
                                int v531;
                                v531 = v530 + v529;
                                unsigned int v532;
                                v532 = v524[v531];
                                int v533;
                                v533 = (int)v532;
                                float v534; int v535;
                                Tuple2 tmp14 = method_4(v76, v507, v509, v511, v513, v515, v517, v519, v521, v533, v523);
                                v534 = tmp14.v0; v535 = tmp14.v1;
                                extern __shared__ unsigned char v536[];
                                float * v537;
                                v537 = reinterpret_cast<float *>(&v536[0ull]);
                                int * v539;
                                v539 = reinterpret_cast<int *>(&v536[16ull]);
                                int v541;
                                v541 = threadIdx.x;
                                bool v542;
                                v542 = v541 == 0;
                                if (v542){
                                    v537[0] = v534;
                                    v539[0] = v535;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                float v543;
                                v543 = v537[0];
                                int v544;
                                v544 = v539[0];
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                double * v545;
                                v545 = reinterpret_cast<double *>(&v1[2097168ull]);
                                double * v547;
                                v547 = reinterpret_cast<double *>(&v1[2490384ull]);
                                int * v549;
                                v549 = reinterpret_cast<int *>(&v1[2883600ull]);
                                int * v551;
                                v551 = reinterpret_cast<int *>(&v0[6389760ull]);
                                float * v553;
                                v553 = reinterpret_cast<float *>(&v0[7962624ull]);
                                int * v555;
                                v555 = reinterpret_cast<int *>(&v0[9535488ull]);
                                int * v557;
                                v557 = reinterpret_cast<int *>(&v0[11108352ull]);
                                double * v559;
                                v559 = reinterpret_cast<double *>(&v0[12681216ull]);
                                double * v561;
                                v561 = reinterpret_cast<double *>(&v0[18972672ull]);
                                int v563;
                                v563 = threadIdx.x;
                                int v564;
                                v564 = blockIdx.x;
                                int v565;
                                v565 = v564 * 256;
                                int v566;
                                v566 = v563 + v565;
                                int v567;
                                v567 = 0;
                                while (while_method_0(v567)){
                                    unsigned int * v569;
                                    v569 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
                                    int v571;
                                    v571 = blockIdx.x;
                                    int v572;
                                    v572 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v567 && v567 < 4);
                                    assert("Tensor range check" && 0 <= v571 && v571 < 24);
                                    assert("Tensor range check" && 0 <= v572 && v572 < 256);
                                    int v573;
                                    v573 = 256 * v571;
                                    int v574;
                                    v574 = v573 + v572;
                                    int v575;
                                    v575 = 6144 * v567;
                                    int v576;
                                    v576 = v575 + v574;
                                    unsigned int v577;
                                    v577 = v569[v576];
                                    int v578;
                                    v578 = (int)v577;
                                    float v579;
                                    v579 = method_5(v507, v509, v511, v513, v515, v517, v519, v521, v578, v567, v544);
                                    assert("Tensor range check" && 0 <= v567 && v567 < 4);
                                    assert("Tensor range check" && 0 <= v566 && v566 < 6144);
                                    int v580;
                                    v580 = v575 + v566;
                                    int v581;
                                    v581 = v549[v580];
                                    int v582;
                                    v582 = v581 + 1;
                                    assert("Tensor range check" && 0 <= v567 && v567 < 4);
                                    assert("Tensor range check" && 0 <= v566 && v566 < 6144);
                                    v549[v580] = v582;
                                    assert("Tensor range check" && 0 <= v567 && v567 < 4);
                                    assert("Tensor range check" && 0 <= v581 && v581 < 16);
                                    assert("Tensor range check" && 0 <= v566 && v566 < 6144);
                                    int v583;
                                    v583 = 6144 * v581;
                                    int v584;
                                    v584 = v583 + v566;
                                    int v585;
                                    v585 = 98304 * v567;
                                    int v586;
                                    v586 = v585 + v584;
                                    v551[v586] = v544;
                                    v553[v586] = v543;
                                    v555[v586] = v67;
                                    v557[v586] = v578;
                                    assert("Tensor range check" && 0 <= v567 && v567 < 4);
                                    int v587;
                                    v587 = 12288 * v567;
                                    assert("Tensor range check" && 0 <= v566 && v566 < 6144);
                                    int v588;
                                    v588 = 2 * v566;
                                    int v589;
                                    v589 = v588 + v587;
                                    assert("Tensor range check" && 0 <= v567 && v567 < 4);
                                    int v590;
                                    v590 = 196608 * v567;
                                    assert("Tensor range check" && 0 <= v581 && v581 < 16);
                                    int v591;
                                    v591 = 12288 * v581;
                                    int v592;
                                    v592 = v591 + v590;
                                    assert("Tensor range check" && 0 <= v566 && v566 < 6144);
                                    int v593;
                                    v593 = v588 + v592;
                                    double * v594;
                                    v594 = v545+v589;
                                    double * v596;
                                    v596 = v547+v589;
                                    double * v598;
                                    v598 = v559+v593;
                                    double * v600;
                                    v600 = v561+v593;
                                    int v602;
                                    v602 = sizeof(double *);
                                    unsigned long long v603;
                                    v603 = (unsigned long long)v602;
                                    unsigned long long v604;
                                    v604 = 256ull * v603;
                                    unsigned long long v605;
                                    v605 = v604 + 16ull;
                                    unsigned long long v606;
                                    v606 = v605 - 1ull;
                                    unsigned long long v607;
                                    v607 = v606 % 16ull;
                                    unsigned long long v608;
                                    v608 = v606 - v607;
                                    unsigned long long v609;
                                    v609 = v608 + v604;
                                    unsigned long long v610;
                                    v610 = v609 + 16ull;
                                    unsigned long long v611;
                                    v611 = v610 - 1ull;
                                    unsigned long long v612;
                                    v612 = v611 % 16ull;
                                    unsigned long long v613;
                                    v613 = v611 - v612;
                                    unsigned long long v614;
                                    v614 = v613 + v604;
                                    unsigned long long v615;
                                    v615 = v614 + 16ull;
                                    unsigned long long v616;
                                    v616 = v615 - 1ull;
                                    unsigned long long v617;
                                    v617 = v616 % 16ull;
                                    unsigned long long v618;
                                    v618 = v616 - v617;
                                    unsigned long long v619;
                                    v619 = v618 + v604;
                                    bool v620;
                                    v620 = v619 <= 98304ull;
                                    bool v621;
                                    v621 = v620 == false;
                                    if (v621){
                                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v620);
                                    } else {
                                    }
                                    extern __shared__ unsigned char v623[];
                                    bool v624;
                                    v624 = v619 <= v619;
                                    bool v625;
                                    v625 = v624 == false;
                                    if (v625){
                                        assert("The length of the partition has to be less than or equal to the length of the base array." && v624);
                                    } else {
                                    }
                                    double * * v627;
                                    v627 = reinterpret_cast<double * *>(&v623[0ull]);
                                    double * * v629;
                                    v629 = reinterpret_cast<double * *>(&v623[v608]);
                                    double * * v631;
                                    v631 = reinterpret_cast<double * *>(&v623[v613]);
                                    double * * v633;
                                    v633 = reinterpret_cast<double * *>(&v623[v618]);
                                    int v635;
                                    v635 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v635 && v635 < 256);
                                    v627[v635] = v594;
                                    v629[v635] = v596;
                                    v631[v635] = v598;
                                    v633[v635] = v600;
                                    asm("barrier.cta.sync %0;" :: "r"(0));
                                    bool v636;
                                    v636 = 0 <= v635;
                                    bool v637;
                                    v637 = v636 == false;
                                    if (v637){
                                        assert("The index needs to be zero or positive." && v636);
                                    } else {
                                    }
                                    int v639;
                                    v639 = v635 % 1;
                                    bool v640;
                                    v640 = v635 < 256;
                                    bool v641;
                                    v641 = v640 == false;
                                    if (v641){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v640);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v635 && v635 < 256);
                                    int v643;
                                    v643 = 0;
                                    while (while_method_6(v643)){
                                        bool v645;
                                        v645 = v636 && v640;
                                        bool v646;
                                        v646 = v645 == false;
                                        if (v646){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v645);
                                        } else {
                                        }
                                        bool v648;
                                        v648 = 0 <= v643;
                                        bool v650;
                                        if (v648){
                                            bool v649;
                                            v649 = v643 < 1;
                                            v650 = v649;
                                        } else {
                                            v650 = false;
                                        }
                                        bool v651;
                                        v651 = v650 == false;
                                        if (v651){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v650);
                                        } else {
                                        }
                                        int v653;
                                        v653 = v643 * 256;
                                        int v654;
                                        v654 = v653 + v635;
                                        assert("Tensor range check" && 0 <= v643 && v643 < 1);
                                        int v655;
                                        v655 = 256 * v643;
                                        int v656;
                                        v656 = v655 + v635;
                                        double * v657;
                                        v657 = v627[v656];
                                        double * v658;
                                        v658 = v629[v656];
                                        double * v659;
                                        v659 = v631[v656];
                                        double * v660;
                                        v660 = v633[v656];
                                        int v661;
                                        v661 = blockIdx.x;
                                        int v662;
                                        v662 = v661 * 256;
                                        int v663;
                                        v663 = v662 + v654;
                                        assert("Tensor range check" && 0 <= v639 && v639 < 1);
                                        int v664;
                                        v664 = 2 * v639;
                                        double v665[2];
                                        double v666[2];
                                        int v667[2];
                                        int v668;
                                        v668 = 0;
                                        while (while_method_6(v668)){
                                            assert("Tensor range check" && 0 <= v668 && v668 < 1);
                                            int v670;
                                            v670 = 2 * v668;
                                            assert("Tensor range check" && 0 <= v668 && v668 < 1);
                                            int v671;
                                            v671 = v670 + v664;
                                            int4* v672;
                                            v672 = reinterpret_cast<int4*>(v657 + v671);
                                            int4* v673;
                                            v673 = reinterpret_cast<int4*>(v665 + v670);
                                            assert("Pointer alignment check" && (unsigned long long)(v672) % 2 == 0 && (unsigned long long)(v673) % 2 == 0);
                                            *v673 = *v672;
                                            int4* v674;
                                            v674 = reinterpret_cast<int4*>(v658 + v671);
                                            int4* v675;
                                            v675 = reinterpret_cast<int4*>(v666 + v670);
                                            assert("Pointer alignment check" && (unsigned long long)(v674) % 2 == 0 && (unsigned long long)(v675) % 2 == 0);
                                            *v675 = *v674;
                                            v668 += 1 ;
                                        }
                                        int v676;
                                        v676 = 0;
                                        while (while_method_6(v676)){
                                            int v678;
                                            v678 = 0;
                                            while (while_method_2(v678)){
                                                bool v680;
                                                v680 = 0 <= v678;
                                                bool v682;
                                                if (v680){
                                                    bool v681;
                                                    v681 = v678 < 2;
                                                    v682 = v681;
                                                } else {
                                                    v682 = false;
                                                }
                                                bool v683;
                                                v683 = v682 == false;
                                                if (v683){
                                                    assert("The indices should be inside the range of the dimension." && v682);
                                                } else {
                                                }
                                                bool v685;
                                                v685 = 0 <= v639;
                                                bool v687;
                                                if (v685){
                                                    bool v686;
                                                    v686 = v639 < 1;
                                                    v687 = v686;
                                                } else {
                                                    v687 = false;
                                                }
                                                bool v688;
                                                v688 = v687 == false;
                                                if (v688){
                                                    assert("The indices should be inside the range of the dimension." && v687);
                                                } else {
                                                }
                                                int v690;
                                                v690 = v639 * 2;
                                                int v691;
                                                v691 = v678 + v690;
                                                bool v692;
                                                v692 = 0 <= v676;
                                                bool v694;
                                                if (v692){
                                                    bool v693;
                                                    v693 = v676 < 1;
                                                    v694 = v693;
                                                } else {
                                                    v694 = false;
                                                }
                                                bool v695;
                                                v695 = v694 == false;
                                                if (v695){
                                                    assert("The indices should be inside the range of the dimension." && v694);
                                                } else {
                                                }
                                                int v697;
                                                v697 = v676 * 2;
                                                int v698;
                                                v698 = v691 + v697;
                                                assert("Tensor range check" && 0 <= v676 && v676 < 1);
                                                assert("Tensor range check" && 0 <= v678 && v678 < 2);
                                                int v699;
                                                v699 = 2 * v676;
                                                int v700;
                                                v700 = v699 + v678;
                                                v667[v700] = v698;
                                                v678 += 1 ;
                                            }
                                            v676 += 1 ;
                                        }
                                        int v701;
                                        v701 = 0;
                                        while (while_method_6(v701)){
                                            assert("Tensor range check" && 0 <= v701 && v701 < 1);
                                            int v703;
                                            v703 = 2 * v701;
                                            int v704;
                                            v704 = v703 + v664;
                                            assert("Tensor range check" && 0 <= v701 && v701 < 1);
                                            int4* v705;
                                            v705 = reinterpret_cast<int4*>(v665 + v703);
                                            int4* v706;
                                            v706 = reinterpret_cast<int4*>(v659 + v704);
                                            assert("Pointer alignment check" && (unsigned long long)(v705) % 2 == 0 && (unsigned long long)(v706) % 2 == 0);
                                            *v706 = *v705;
                                            int4* v707;
                                            v707 = reinterpret_cast<int4*>(v666 + v703);
                                            int4* v708;
                                            v708 = reinterpret_cast<int4*>(v660 + v704);
                                            assert("Pointer alignment check" && (unsigned long long)(v707) % 2 == 0 && (unsigned long long)(v708) % 2 == 0);
                                            *v708 = *v707;
                                            v701 += 1 ;
                                        }
                                        assert("Tensor range check" && 0 <= v654 && v654 < 256);
                                        v643 += 1 ;
                                    }
                                    asm("barrier.cta.sync %0;" :: "r"(0));
                                    assert("Tensor range check" && 0 <= v635 && v635 < 256);
                                    asm("barrier.cta.sync %0;" :: "r"(0));
                                    double v709;
                                    v709 = (double)v543;
                                    double v710;
                                    v710 = log(v709);
                                    double v711;
                                    v711 = (double)v579;
                                    double v712;
                                    v712 = log(v711);
                                    assert("Tensor range check" && 0 <= v567 && v567 < 4);
                                    assert("Tensor range check" && 0 <= v566 && v566 < 6144);
                                    assert("Tensor range check" && 0 <= v67 && v67 < 2);
                                    int v713;
                                    v713 = v588 + v67;
                                    int v714;
                                    v714 = v587 + v713;
                                    double v715;
                                    v715 = v545[v714];
                                    double v716;
                                    v716 = v547[v714];
                                    double v717;
                                    v717 = v712 + v715;
                                    double v718;
                                    v718 = v710 + v716;
                                    assert("Tensor range check" && 0 <= v567 && v567 < 4);
                                    assert("Tensor range check" && 0 <= v566 && v566 < 6144);
                                    assert("Tensor range check" && 0 <= v67 && v67 < 2);
                                    v545[v714] = v717;
                                    v547[v714] = v718;
                                    v567 += 1 ;
                                }
                                bool v719;
                                v719 = 0 == v544;
                                Union10 v728;
                                if (v719){
                                    v728 = Union10{Union10_1{}};
                                } else {
                                    bool v721;
                                    v721 = 1 == v544;
                                    if (v721){
                                        v728 = Union10{Union10_0{}};
                                    } else {
                                        bool v723;
                                        v723 = 2 == v544;
                                        if (v723){
                                            v728 = Union10{Union10_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v728.tag) {
                                    case 0: { // AA_Call
                                        v794 = Union3{Union3_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v729;
                                        v729 = v68[0];
                                        int v731; int v732;
                                        Tuple1 tmp17 = Tuple1{1, v729};
                                        v731 = tmp17.v0; v732 = tmp17.v1;
                                        while (while_method_2(v731)){
                                            int v734;
                                            v734 = v68[v731];
                                            bool v736;
                                            v736 = v732 >= v734;
                                            int v737;
                                            if (v736){
                                                v737 = v732;
                                            } else {
                                                v737 = v734;
                                            }
                                            v732 = v737;
                                            v731 += 1 ;
                                        }
                                        int v738;
                                        v738 = v68[v67];
                                        bool v740;
                                        v740 = v738 == v732;
                                        if (v740){
                                            v794 = Union3{Union3_0{}};
                                        } else {
                                            v794 = Union3{Union3_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v745;
                                        v745 = v69 > 0;
                                        if (v745){
                                            v794 = Union3{Union3_2{}};
                                        } else {
                                            v794 = Union3{Union3_0{}};
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
                                printf("%s\n", "Humans not allowed during training.");
                                __trap();
                                break;
                            }
                            case 2: { // Random
                                curandStatePhilox4_32_10_t & v752 = v2.v5;
                                curandStatePhilox4_32_10_t & v753 = v752;
                                static_array_list<Union3,3> v754;
                                v754 = static_array_list<Union3,3>{};
                                v754.unsafe_set_length(1);
                                Union3 v756;
                                v756 = Union3{Union3_0{}};
                                v754[0] = v756;
                                int v758;
                                v758 = v68[0];
                                int v760;
                                v760 = v68[1];
                                bool v762;
                                v762 = v758 == v760;
                                bool v763;
                                v763 = v762 != true;
                                if (v763){
                                    Union3 v764;
                                    v764 = Union3{Union3_1{}};
                                    v754.push(v764);
                                } else {
                                }
                                bool v765;
                                v765 = v69 > 0;
                                if (v765){
                                    Union3 v766;
                                    v766 = Union3{Union3_2{}};
                                    v754.push(v766);
                                } else {
                                }
                                int v767;
                                v767 = v754.length;
                                int v768;
                                v768 = v767 - 1;
                                int v769;
                                v769 = 0;
                                while (while_method_5(v768, v769)){
                                    int v771;
                                    v771 = v754.length;
                                    int v772;
                                    v772 = int_range_6(v771, v769, v753);
                                    Union3 v773;
                                    v773 = v754[v769];
                                    Union3 v775;
                                    v775 = v754[v772];
                                    v754[v769] = v775;
                                    v754[v772] = v773;
                                    v769 += 1 ;
                                }
                                Union3 v777;
                                v777 = v754.pop();
                                int v778;
                                v778 = sizeof(Union3);
                                unsigned long long v779;
                                v779 = (unsigned long long)v778;
                                bool v780;
                                v780 = v779 <= 98304ull;
                                bool v781;
                                v781 = v780 == false;
                                if (v781){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v780);
                                } else {
                                }
                                extern __shared__ unsigned char v783[];
                                bool v784;
                                v784 = v779 <= v779;
                                bool v785;
                                v785 = v784 == false;
                                if (v785){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v784);
                                } else {
                                }
                                Union3 * v787;
                                v787 = reinterpret_cast<Union3 *>(&v783[0ull]);
                                int v789;
                                v789 = threadIdx.x;
                                bool v790;
                                v790 = v789 == 0;
                                if (v790){
                                    v787[0] = v777;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                Union3 v791;
                                v791 = v787[0];
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                v794 = v791;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union1 v795;
                        v795 = Union1{Union1_1{v67, v794}};
                        v17.push(v795);
                        Union4 v881;
                        switch (v64.tag) {
                            case 0: { // None
                                switch (v794.tag) {
                                    case 0: { // Call
                                        if (v65){
                                            bool v845;
                                            v845 = v67 == 0;
                                            int v846;
                                            if (v845){
                                                v846 = 1;
                                            } else {
                                                v846 = 0;
                                            }
                                            v881 = Union4{Union4_2{v64, false, v66, v846, v68, v69}};
                                        } else {
                                            v881 = Union4{Union4_0{v64, v65, v66, v67, v68, v69}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v881 = Union4{Union4_5{v64, v65, v66, v67, v68, v69}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v850;
                                        v850 = v69 > 0;
                                        if (v850){
                                            bool v851;
                                            v851 = v67 == 0;
                                            int v852;
                                            if (v851){
                                                v852 = 1;
                                            } else {
                                                v852 = 0;
                                            }
                                            int v853;
                                            v853 = -1 + v69;
                                            int v854; int v855;
                                            Tuple1 tmp18 = Tuple1{0, 0};
                                            v854 = tmp18.v0; v855 = tmp18.v1;
                                            while (while_method_2(v854)){
                                                int v857;
                                                v857 = v68[v854];
                                                bool v859;
                                                v859 = v855 >= v857;
                                                int v860;
                                                if (v859){
                                                    v860 = v855;
                                                } else {
                                                    v860 = v857;
                                                }
                                                v855 = v860;
                                                v854 += 1 ;
                                            }
                                            static_array<int,2> v861;
                                            int v863;
                                            v863 = 0;
                                            while (while_method_2(v863)){
                                                v861[v863] = v855;
                                                v863 += 1 ;
                                            }
                                            static_array<int,2> v865;
                                            int v867;
                                            v867 = 0;
                                            while (while_method_2(v867)){
                                                int v869;
                                                v869 = v861[v867];
                                                bool v871;
                                                v871 = v867 == v67;
                                                int v873;
                                                if (v871){
                                                    int v872;
                                                    v872 = v869 + 2;
                                                    v873 = v872;
                                                } else {
                                                    v873 = v869;
                                                }
                                                v865[v867] = v873;
                                                v867 += 1 ;
                                            }
                                            v881 = Union4{Union4_2{v64, false, v66, v852, v865, v853}};
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
                                Union2 v796 = v64.case1.v0;
                                switch (v794.tag) {
                                    case 0: { // Call
                                        if (v65){
                                            bool v798;
                                            v798 = v67 == 0;
                                            int v799;
                                            if (v798){
                                                v799 = 1;
                                            } else {
                                                v799 = 0;
                                            }
                                            v881 = Union4{Union4_2{v64, false, v66, v799, v68, v69}};
                                        } else {
                                            int v801; int v802;
                                            Tuple1 tmp19 = Tuple1{0, 0};
                                            v801 = tmp19.v0; v802 = tmp19.v1;
                                            while (while_method_2(v801)){
                                                int v804;
                                                v804 = v68[v801];
                                                bool v806;
                                                v806 = v802 >= v804;
                                                int v807;
                                                if (v806){
                                                    v807 = v802;
                                                } else {
                                                    v807 = v804;
                                                }
                                                v802 = v807;
                                                v801 += 1 ;
                                            }
                                            static_array<int,2> v808;
                                            int v810;
                                            v810 = 0;
                                            while (while_method_2(v810)){
                                                v808[v810] = v802;
                                                v810 += 1 ;
                                            }
                                            v881 = Union4{Union4_4{v64, v65, v66, v67, v808, v69}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v881 = Union4{Union4_5{v64, v65, v66, v67, v68, v69}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v814;
                                        v814 = v69 > 0;
                                        if (v814){
                                            bool v815;
                                            v815 = v67 == 0;
                                            int v816;
                                            if (v815){
                                                v816 = 1;
                                            } else {
                                                v816 = 0;
                                            }
                                            int v817;
                                            v817 = -1 + v69;
                                            int v818; int v819;
                                            Tuple1 tmp20 = Tuple1{0, 0};
                                            v818 = tmp20.v0; v819 = tmp20.v1;
                                            while (while_method_2(v818)){
                                                int v821;
                                                v821 = v68[v818];
                                                bool v823;
                                                v823 = v819 >= v821;
                                                int v824;
                                                if (v823){
                                                    v824 = v819;
                                                } else {
                                                    v824 = v821;
                                                }
                                                v819 = v824;
                                                v818 += 1 ;
                                            }
                                            static_array<int,2> v825;
                                            int v827;
                                            v827 = 0;
                                            while (while_method_2(v827)){
                                                v825[v827] = v819;
                                                v827 += 1 ;
                                            }
                                            static_array<int,2> v829;
                                            int v831;
                                            v831 = 0;
                                            while (while_method_2(v831)){
                                                int v833;
                                                v833 = v825[v831];
                                                bool v835;
                                                v835 = v831 == v67;
                                                int v837;
                                                if (v835){
                                                    int v836;
                                                    v836 = v833 + 4;
                                                    v837 = v836;
                                                } else {
                                                    v837 = v833;
                                                }
                                                v829[v831] = v837;
                                                v831 += 1 ;
                                            }
                                            v881 = Union4{Union4_2{v64, false, v66, v816, v829, v817}};
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
                        v1033 = Union6{Union6_1{v881}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v883 = v21.case3.v0; bool v884 = v21.case3.v1; static_array<Union2,2> v885 = v21.case3.v2; int v886 = v21.case3.v3; static_array<int,2> v887 = v21.case3.v4; int v888 = v21.case3.v5; Union3 v889 = v21.case3.v6;
                        Union1 v890;
                        v890 = Union1{Union1_1{v886, v889}};
                        v17.push(v890);
                        Union4 v976;
                        switch (v883.tag) {
                            case 0: { // None
                                switch (v889.tag) {
                                    case 0: { // Call
                                        if (v884){
                                            bool v940;
                                            v940 = v886 == 0;
                                            int v941;
                                            if (v940){
                                                v941 = 1;
                                            } else {
                                                v941 = 0;
                                            }
                                            v976 = Union4{Union4_2{v883, false, v885, v941, v887, v888}};
                                        } else {
                                            v976 = Union4{Union4_0{v883, v884, v885, v886, v887, v888}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v976 = Union4{Union4_5{v883, v884, v885, v886, v887, v888}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v945;
                                        v945 = v888 > 0;
                                        if (v945){
                                            bool v946;
                                            v946 = v886 == 0;
                                            int v947;
                                            if (v946){
                                                v947 = 1;
                                            } else {
                                                v947 = 0;
                                            }
                                            int v948;
                                            v948 = -1 + v888;
                                            int v949; int v950;
                                            Tuple1 tmp21 = Tuple1{0, 0};
                                            v949 = tmp21.v0; v950 = tmp21.v1;
                                            while (while_method_2(v949)){
                                                int v952;
                                                v952 = v887[v949];
                                                bool v954;
                                                v954 = v950 >= v952;
                                                int v955;
                                                if (v954){
                                                    v955 = v950;
                                                } else {
                                                    v955 = v952;
                                                }
                                                v950 = v955;
                                                v949 += 1 ;
                                            }
                                            static_array<int,2> v956;
                                            int v958;
                                            v958 = 0;
                                            while (while_method_2(v958)){
                                                v956[v958] = v950;
                                                v958 += 1 ;
                                            }
                                            static_array<int,2> v960;
                                            int v962;
                                            v962 = 0;
                                            while (while_method_2(v962)){
                                                int v964;
                                                v964 = v956[v962];
                                                bool v966;
                                                v966 = v962 == v886;
                                                int v968;
                                                if (v966){
                                                    int v967;
                                                    v967 = v964 + 2;
                                                    v968 = v967;
                                                } else {
                                                    v968 = v964;
                                                }
                                                v960[v962] = v968;
                                                v962 += 1 ;
                                            }
                                            v976 = Union4{Union4_2{v883, false, v885, v947, v960, v948}};
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
                                Union2 v891 = v883.case1.v0;
                                switch (v889.tag) {
                                    case 0: { // Call
                                        if (v884){
                                            bool v893;
                                            v893 = v886 == 0;
                                            int v894;
                                            if (v893){
                                                v894 = 1;
                                            } else {
                                                v894 = 0;
                                            }
                                            v976 = Union4{Union4_2{v883, false, v885, v894, v887, v888}};
                                        } else {
                                            int v896; int v897;
                                            Tuple1 tmp22 = Tuple1{0, 0};
                                            v896 = tmp22.v0; v897 = tmp22.v1;
                                            while (while_method_2(v896)){
                                                int v899;
                                                v899 = v887[v896];
                                                bool v901;
                                                v901 = v897 >= v899;
                                                int v902;
                                                if (v901){
                                                    v902 = v897;
                                                } else {
                                                    v902 = v899;
                                                }
                                                v897 = v902;
                                                v896 += 1 ;
                                            }
                                            static_array<int,2> v903;
                                            int v905;
                                            v905 = 0;
                                            while (while_method_2(v905)){
                                                v903[v905] = v897;
                                                v905 += 1 ;
                                            }
                                            v976 = Union4{Union4_4{v883, v884, v885, v886, v903, v888}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v976 = Union4{Union4_5{v883, v884, v885, v886, v887, v888}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v909;
                                        v909 = v888 > 0;
                                        if (v909){
                                            bool v910;
                                            v910 = v886 == 0;
                                            int v911;
                                            if (v910){
                                                v911 = 1;
                                            } else {
                                                v911 = 0;
                                            }
                                            int v912;
                                            v912 = -1 + v888;
                                            int v913; int v914;
                                            Tuple1 tmp23 = Tuple1{0, 0};
                                            v913 = tmp23.v0; v914 = tmp23.v1;
                                            while (while_method_2(v913)){
                                                int v916;
                                                v916 = v887[v913];
                                                bool v918;
                                                v918 = v914 >= v916;
                                                int v919;
                                                if (v918){
                                                    v919 = v914;
                                                } else {
                                                    v919 = v916;
                                                }
                                                v914 = v919;
                                                v913 += 1 ;
                                            }
                                            static_array<int,2> v920;
                                            int v922;
                                            v922 = 0;
                                            while (while_method_2(v922)){
                                                v920[v922] = v914;
                                                v922 += 1 ;
                                            }
                                            static_array<int,2> v924;
                                            int v926;
                                            v926 = 0;
                                            while (while_method_2(v926)){
                                                int v928;
                                                v928 = v920[v926];
                                                bool v930;
                                                v930 = v926 == v886;
                                                int v932;
                                                if (v930){
                                                    int v931;
                                                    v931 = v928 + 4;
                                                    v932 = v931;
                                                } else {
                                                    v932 = v928;
                                                }
                                                v924[v926] = v932;
                                                v926 += 1 ;
                                            }
                                            v976 = Union4{Union4_2{v883, false, v885, v911, v924, v912}};
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
                        v1033 = Union6{Union6_1{v976}};
                        break;
                    }
                    case 4: { // TerminalCall
                        Union5 v39 = v21.case4.v0; bool v40 = v21.case4.v1; static_array<Union2,2> v41 = v21.case4.v2; int v42 = v21.case4.v3; static_array<int,2> v43 = v21.case4.v4; int v44 = v21.case4.v5;
                        int v45;
                        v45 = v43[v42];
                        Union11 v47;
                        v47 = compare_hands_7(v39, v40, v41, v42, v43, v44);
                        int v52; int v53;
                        switch (v47.tag) {
                            case 0: { // Eq
                                v52 = 0; v53 = -1;
                                break;
                            }
                            case 1: { // Gt
                                v52 = v45; v53 = 0;
                                break;
                            }
                            case 2: { // Lt
                                v52 = v45; v53 = 1;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        int v54;
                        v54 = -v53;
                        bool v55;
                        v55 = v53 >= v54;
                        int v56;
                        if (v55){
                            v56 = v53;
                        } else {
                            v56 = v54;
                        }
                        float v57;
                        v57 = (float)v52;
                        static_array<float,2> & v58 = v2.v4;
                        v58[v56] = v57;
                        bool v59;
                        v59 = v56 == 0;
                        int v60;
                        if (v59){
                            v60 = 1;
                        } else {
                            v60 = 0;
                        }
                        float v61;
                        v61 = -v57;
                        v58[v60] = v61;
                        Union1 v62;
                        v62 = Union1{Union1_3{v41, v52, v53}};
                        v17.push(v62);
                        v1033 = Union6{Union6_0{}};
                        break;
                    }
                    case 5: { // TerminalFold
                        Union5 v22 = v21.case5.v0; bool v23 = v21.case5.v1; static_array<Union2,2> v24 = v21.case5.v2; int v25 = v21.case5.v3; static_array<int,2> v26 = v21.case5.v4; int v27 = v21.case5.v5;
                        int v28;
                        v28 = v26[v25];
                        int v30;
                        v30 = -v28;
                        float v31;
                        v31 = (float)v30;
                        static_array<float,2> & v32 = v2.v4;
                        v32[v25] = v31;
                        bool v33;
                        v33 = v25 == 0;
                        int v34;
                        if (v33){
                            v34 = 1;
                        } else {
                            v34 = 0;
                        }
                        float v35;
                        v35 = -v31;
                        v32[v34] = v35;
                        int v36;
                        if (v33){
                            v36 = 1;
                        } else {
                            v36 = 0;
                        }
                        Union1 v37;
                        v37 = Union1{Union1_3{v24, v28, v36}};
                        v17.push(v37);
                        v1033 = Union6{Union6_0{}};
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
        v19 = v1033;
    }
    return ;
}
__device__ inline bool while_method_10(int v0){
    bool v1;
    v1 = v0 > 0;
    return v1;
}
__device__ inline bool while_method_11(int v0){
    bool v1;
    v1 = v0 < 64;
    return v1;
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned long long v1, unsigned char * v2, unsigned long long v3, float * v4, float * v5, float * v6) {
    auto v7 = cooperative_groups::this_grid();
    unsigned long long v8;
    v8 = clock64();
    int v9;
    v9 = threadIdx.x;
    int v10;
    v10 = blockIdx.x;
    int v11;
    v11 = v10 * 256;
    int v12;
    v12 = v9 + v11;
    unsigned long long v13;
    v13 = (unsigned long long)v12;
    curandStatePhilox4_32_10_t v14;
    curand_init(v8,v13,0ull,&v14);
    static_array<Union0,2> v15;
    Union0 v17;
    v17 = Union0{Union0_2{}};
    v15[0] = v17;
    Union0 v19;
    v19 = Union0{Union0_2{}};
    v15[1] = v19;
    static_array_list<Union1,32> v21;
    v21 = static_array_list<Union1,32>{};
    static_array<float,2> v23;
    v23[0] = 0.0f;
    v23[1] = 0.0f;
    cooperative_groups::grid_group & v25 = v7;
    curandStatePhilox4_32_10_t & v26 = v14;
    StackMut0 v27{63u, v25, v21, v15, v23, v26};
    bool v28;
    v28 = 2981904ull == v3;
    bool v29;
    v29 = v28 == false;
    if (v29){
        assert("The params needs to have matching offsets." && v28);
    } else {
    }
    bool v31;
    v31 = 25264128ull == v1;
    bool v32;
    v32 = v31 == false;
    if (v32){
        assert("The outputs needs to have matching offsets." && v31);
    } else {
    }
    int v34;
    v34 = 0;
    while (while_method_0(v34)){
        int v36;
        v36 = 0;
        while (while_method_1(v36)){
            int v38;
            v38 = 0;
            while (while_method_2(v38)){
                Union4 v40;
                v40 = Union4{Union4_1{}};
                method_0(v0, v2, v27, v38, v40);
                static_array<float,2> & v41 = v27.v4;
                float v42;
                v42 = v41[v38];
                int v44;
                v44 = 0;
                while (while_method_0(v44)){
                    double * v46;
                    v46 = reinterpret_cast<double *>(&v2[2097168ull]);
                    double * v48;
                    v48 = reinterpret_cast<double *>(&v2[2490384ull]);
                    int * v50;
                    v50 = reinterpret_cast<int *>(&v2[2883600ull]);
                    assert("Tensor range check" && 0 <= v44 && v44 < 4);
                    int v52;
                    v52 = 12288 * v44;
                    int v53;
                    v53 = threadIdx.x;
                    int v54;
                    v54 = blockIdx.x;
                    int v55;
                    v55 = v54 * 256;
                    int v56;
                    v56 = v53 + v55;
                    assert("Tensor range check" && 0 <= v56 && v56 < 6144);
                    int v57;
                    v57 = 2 * v56;
                    int v58;
                    v58 = v57 + v52;
                    double v59;
                    v59 = 0.0;
                    int v60;
                    v60 = 0;
                    while (while_method_2(v60)){
                        assert("Tensor range check" && 0 <= v60 && v60 < 2);
                        int v62;
                        v62 = v60 + v58;
                        double v63;
                        v63 = v46[v62];
                        double v64;
                        v64 = v59 + v63;
                        v59 = v64;
                        v60 += 1 ;
                    }
                    double v65;
                    v65 = 0.0;
                    int v66;
                    v66 = 0;
                    while (while_method_2(v66)){
                        assert("Tensor range check" && 0 <= v66 && v66 < 2);
                        int v68;
                        v68 = v66 + v58;
                        double v69;
                        v69 = v48[v68];
                        double v70;
                        v70 = v65 + v69;
                        v65 = v70;
                        v66 += 1 ;
                    }
                    double v71;
                    v71 = v59 - v65;
                    double v72;
                    v72 = exp(v71);
                    float v73;
                    v73 = (float)v72;
                    float v74;
                    v74 = v42 * v73;
                    assert("Tensor range check" && 0 <= v44 && v44 < 4);
                    assert("Tensor range check" && 0 <= v34 && v34 < 4);
                    int v75;
                    v75 = 4 * v44;
                    int v76;
                    v76 = v75 + v34;
                    float * v77;
                    v77 = v4+v76;
                    float * v79;
                    v79 = v5+v76;
                    float v81;
                    v81 = atomicAdd(v77,v74);
                    float v82;
                    v82 = atomicAdd(v79,v73);
                    v44 += 1 ;
                }
                static_array<float,2> & v83 = v27.v4;
                unsigned int * v84;
                v84 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
                int * v86;
                v86 = reinterpret_cast<int *>(&v2[262144ull]);
                float * v88;
                v88 = reinterpret_cast<float *>(&v2[262160ull]);
                float * v90;
                v90 = reinterpret_cast<float *>(&v2[524304ull]);
                float * v92;
                v92 = reinterpret_cast<float *>(&v2[786448ull]);
                float * v94;
                v94 = reinterpret_cast<float *>(&v2[1048592ull]);
                float * v96;
                v96 = reinterpret_cast<float *>(&v2[1310736ull]);
                float * v98;
                v98 = reinterpret_cast<float *>(&v2[1572880ull]);
                float * v100;
                v100 = reinterpret_cast<float *>(&v2[1835024ull]);
                int * v102;
                v102 = reinterpret_cast<int *>(&v0[6389760ull]);
                float * v104;
                v104 = reinterpret_cast<float *>(&v0[7962624ull]);
                int * v106;
                v106 = reinterpret_cast<int *>(&v0[9535488ull]);
                int * v108;
                v108 = reinterpret_cast<int *>(&v0[11108352ull]);
                double * v110;
                v110 = reinterpret_cast<double *>(&v0[12681216ull]);
                double * v112;
                v112 = reinterpret_cast<double *>(&v0[18972672ull]);
                double * v114;
                v114 = reinterpret_cast<double *>(&v2[2097168ull]);
                double * v116;
                v116 = reinterpret_cast<double *>(&v2[2490384ull]);
                int * v118;
                v118 = reinterpret_cast<int *>(&v2[2883600ull]);
                int v120;
                v120 = 0;
                while (while_method_0(v120)){
                    int v122;
                    v122 = threadIdx.x;
                    int v123;
                    v123 = blockIdx.x;
                    int v124;
                    v124 = v123 * 256;
                    int v125;
                    v125 = v122 + v124;
                    float v126[2];
                    int v127;
                    v127 = 0;
                    while (while_method_2(v127)){
                        float v129;
                        v129 = v83[v127];
                        v126[v127] = v129;
                        v127 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v120 && v120 < 4);
                    assert("Tensor range check" && 0 <= v125 && v125 < 6144);
                    int v131;
                    v131 = 6144 * v120;
                    int v132;
                    v132 = v131 + v125;
                    int v133;
                    v133 = v118[v132];
                    int v134;
                    v134 = v133;
                    while (while_method_10(v134)){
                        v134 -= 1 ;
                        assert("Tensor range check" && 0 <= v120 && v120 < 4);
                        assert("Tensor range check" && 0 <= v134 && v134 < 16);
                        assert("Tensor range check" && 0 <= v125 && v125 < 6144);
                        int v136;
                        v136 = 6144 * v134;
                        int v137;
                        v137 = v136 + v125;
                        int v138;
                        v138 = 98304 * v120;
                        int v139;
                        v139 = v138 + v137;
                        int v140;
                        v140 = v102[v139];
                        float v141;
                        v141 = v104[v139];
                        int v142;
                        v142 = v106[v139];
                        int v143;
                        v143 = v108[v139];
                        assert("Tensor range check" && 0 <= v142 && v142 < 2);
                        float v144;
                        v144 = v126[v142];
                        assert("Tensor range check" && 0 <= v120 && v120 < 4);
                        int v145;
                        v145 = 16384 * v120;
                        assert("Tensor range check" && 0 <= v143 && v143 < 4096);
                        int v146;
                        v146 = 4 * v143;
                        int v147;
                        v147 = v146 + v145;
                        float * v148;
                        v148 = v88+v147;
                        float * v150;
                        v150 = v90+v147;
                        float * v152;
                        v152 = v92+v147;
                        float * v154;
                        v154 = v94+v147;
                        float * v156;
                        v156 = v96+v147;
                        float * v158;
                        v158 = v98+v147;
                        float * v160;
                        v160 = v100+v147;
                        assert("Tensor range check" && 0 <= v120 && v120 < 4);
                        int v162;
                        v162 = 196608 * v120;
                        assert("Tensor range check" && 0 <= v134 && v134 < 16);
                        int v163;
                        v163 = 12288 * v134;
                        int v164;
                        v164 = v163 + v162;
                        assert("Tensor range check" && 0 <= v125 && v125 < 6144);
                        int v165;
                        v165 = 2 * v125;
                        int v166;
                        v166 = v165 + v164;
                        double v167[2];
                        int v168;
                        v168 = 0;
                        while (while_method_2(v168)){
                            assert("Tensor range check" && 0 <= v168 && v168 < 2);
                            int v170;
                            v170 = v168 + v166;
                            double v171;
                            v171 = v110[v170];
                            bool v172;
                            v172 = v142 == v168;
                            double v173;
                            if (v172){
                                v173 = 0.0;
                            } else {
                                v173 = v171;
                            }
                            assert("Tensor range check" && 0 <= v168 && v168 < 2);
                            v167[v168] = v173;
                            v168 += 1 ;
                        }
                        double v174;
                        v174 = 0.0;
                        int v175;
                        v175 = 0;
                        while (while_method_2(v175)){
                            assert("Tensor range check" && 0 <= v175 && v175 < 2);
                            double v177;
                            v177 = v167[v175];
                            double v178;
                            v178 = v174 + v177;
                            v174 = v178;
                            v175 += 1 ;
                        }
                        double v179;
                        v179 = 0.0;
                        int v180;
                        v180 = 0;
                        while (while_method_2(v180)){
                            assert("Tensor range check" && 0 <= v180 && v180 < 2);
                            int v182;
                            v182 = v180 + v166;
                            double v183;
                            v183 = v112[v182];
                            double v184;
                            v184 = v179 + v183;
                            v179 = v184;
                            v180 += 1 ;
                        }
                        double v185;
                        v185 = v174 - v179;
                        double v186;
                        v186 = exp(v185);
                        float v187;
                        v187 = (float)v186;
                        float v188;
                        v188 = v144 * v187;
                        assert("Tensor range check" && 0 <= v140 && v140 < 4);
                        float * v189;
                        v189 = v158+v140;
                        float * v191;
                        v191 = v160+v140;
                        float v193;
                        v193 = atomicAdd(v189,v188);
                        float v194;
                        v194 = atomicAdd(v191,v187);
                        float * v195;
                        v195 = v150+0;
                        float * v197;
                        v197 = v154+0;
                        float * v199;
                        v199 = v156+0;
                        int v201;
                        v201 = sizeof(float *);
                        unsigned long long v202;
                        v202 = (unsigned long long)v201;
                        unsigned long long v203;
                        v203 = 256ull * v202;
                        unsigned long long v204;
                        v204 = 4096ull + v203;
                        unsigned long long v205;
                        v205 = v204 + 16ull;
                        unsigned long long v206;
                        v206 = v205 - 1ull;
                        unsigned long long v207;
                        v207 = v206 % 16ull;
                        unsigned long long v208;
                        v208 = v206 - v207;
                        unsigned long long v209;
                        v209 = v208 + v203;
                        unsigned long long v210;
                        v210 = v209 + 16ull;
                        unsigned long long v211;
                        v211 = v210 - 1ull;
                        unsigned long long v212;
                        v212 = v211 % 16ull;
                        unsigned long long v213;
                        v213 = v211 - v212;
                        unsigned long long v214;
                        v214 = v213 + v203;
                        unsigned long long v215;
                        v215 = v214 + 16ull;
                        unsigned long long v216;
                        v216 = v215 - 1ull;
                        unsigned long long v217;
                        v217 = v216 % 16ull;
                        unsigned long long v218;
                        v218 = v216 - v217;
                        unsigned long long v219;
                        v219 = v218 + v203;
                        unsigned long long v220;
                        v220 = v219 + 16ull;
                        unsigned long long v221;
                        v221 = v220 - 1ull;
                        unsigned long long v222;
                        v222 = v221 % 16ull;
                        unsigned long long v223;
                        v223 = v221 - v222;
                        unsigned long long v224;
                        v224 = v223 + 1024ull;
                        bool v225;
                        v225 = v224 <= 98304ull;
                        bool v226;
                        v226 = v225 == false;
                        if (v226){
                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v225);
                        } else {
                        }
                        extern __shared__ unsigned char v228[];
                        bool v229;
                        v229 = v224 <= v224;
                        bool v230;
                        v230 = v229 == false;
                        if (v230){
                            assert("The length of the partition has to be less than or equal to the length of the base array." && v229);
                        } else {
                        }
                        float * v232;
                        v232 = reinterpret_cast<float *>(&v228[0ull]);
                        int * v234;
                        v234 = reinterpret_cast<int *>(&v228[1024ull]);
                        float * v236;
                        v236 = reinterpret_cast<float *>(&v228[2048ull]);
                        float * v238;
                        v238 = reinterpret_cast<float *>(&v228[3072ull]);
                        float * * v240;
                        v240 = reinterpret_cast<float * *>(&v228[4096ull]);
                        float * * v242;
                        v242 = reinterpret_cast<float * *>(&v228[v208]);
                        float * * v244;
                        v244 = reinterpret_cast<float * *>(&v228[v213]);
                        float * * v246;
                        v246 = reinterpret_cast<float * *>(&v228[v218]);
                        float * v248;
                        v248 = reinterpret_cast<float *>(&v228[v223]);
                        int v250;
                        v250 = threadIdx.x;
                        assert("Tensor range check" && 0 <= v250 && v250 < 256);
                        v232[v250] = v141;
                        v234[v250] = v140;
                        v236[v250] = v144;
                        v238[v250] = v187;
                        v240[v250] = v152;
                        v242[v250] = v195;
                        v244[v250] = v197;
                        v246[v250] = v199;
                        asm("barrier.cta.sync %0;" :: "r"(0));
                        bool v251;
                        v251 = 0 <= v250;
                        bool v252;
                        v252 = v251 == false;
                        if (v252){
                            assert("The index needs to be zero or positive." && v251);
                        } else {
                        }
                        int v254;
                        v254 = v250 % 1;
                        bool v255;
                        v255 = v250 < 256;
                        bool v256;
                        v256 = v255 == false;
                        if (v256){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v255);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v250 && v250 < 256);
                        int v258;
                        v258 = 0;
                        while (while_method_6(v258)){
                            bool v260;
                            v260 = v251 && v255;
                            bool v261;
                            v261 = v260 == false;
                            if (v261){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v260);
                            } else {
                            }
                            bool v263;
                            v263 = 0 <= v258;
                            bool v265;
                            if (v263){
                                bool v264;
                                v264 = v258 < 1;
                                v265 = v264;
                            } else {
                                v265 = false;
                            }
                            bool v266;
                            v266 = v265 == false;
                            if (v266){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v265);
                            } else {
                            }
                            int v268;
                            v268 = v258 * 256;
                            int v269;
                            v269 = v268 + v250;
                            assert("Tensor range check" && 0 <= v258 && v258 < 1);
                            int v270;
                            v270 = 256 * v258;
                            int v271;
                            v271 = v270 + v250;
                            float v272;
                            v272 = v232[v271];
                            int v273;
                            v273 = v234[v271];
                            float v274;
                            v274 = v236[v271];
                            float v275;
                            v275 = v238[v271];
                            float * v276;
                            v276 = v240[v271];
                            float * v277;
                            v277 = v242[v271];
                            float * v278;
                            v278 = v244[v271];
                            float * v279;
                            v279 = v246[v271];
                            int v280;
                            v280 = blockIdx.x;
                            int v281;
                            v281 = v280 * 256;
                            int v282;
                            v282 = v281 + v269;
                            assert("Tensor range check" && 0 <= v254 && v254 < 1);
                            int v283;
                            v283 = 4 * v254;
                            float v284[4];
                            float v285[4];
                            float v286[4];
                            int v287[4];
                            int v288;
                            v288 = 0;
                            while (while_method_6(v288)){
                                assert("Tensor range check" && 0 <= v288 && v288 < 1);
                                int v290;
                                v290 = 4 * v288;
                                assert("Tensor range check" && 0 <= v288 && v288 < 1);
                                int v291;
                                v291 = v290 + v283;
                                int4* v292;
                                v292 = reinterpret_cast<int4*>(v277 + v291);
                                int4* v293;
                                v293 = reinterpret_cast<int4*>(v284 + v290);
                                assert("Pointer alignment check" && (unsigned long long)(v292) % 4 == 0 && (unsigned long long)(v293) % 4 == 0);
                                *v293 = *v292;
                                int4* v294;
                                v294 = reinterpret_cast<int4*>(v278 + v291);
                                int4* v295;
                                v295 = reinterpret_cast<int4*>(v285 + v290);
                                assert("Pointer alignment check" && (unsigned long long)(v294) % 4 == 0 && (unsigned long long)(v295) % 4 == 0);
                                *v295 = *v294;
                                int4* v296;
                                v296 = reinterpret_cast<int4*>(v279 + v291);
                                int4* v297;
                                v297 = reinterpret_cast<int4*>(v286 + v290);
                                assert("Pointer alignment check" && (unsigned long long)(v296) % 4 == 0 && (unsigned long long)(v297) % 4 == 0);
                                *v297 = *v296;
                                v288 += 1 ;
                            }
                            int v298;
                            v298 = 0;
                            while (while_method_6(v298)){
                                int v300;
                                v300 = 0;
                                while (while_method_0(v300)){
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
                                    v307 = 0 <= v254;
                                    bool v309;
                                    if (v307){
                                        bool v308;
                                        v308 = v254 < 1;
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
                                    v312 = v254 * 4;
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
                                    v319 = v298 * 4;
                                    int v320;
                                    v320 = v313 + v319;
                                    assert("Tensor range check" && 0 <= v298 && v298 < 1);
                                    assert("Tensor range check" && 0 <= v300 && v300 < 4);
                                    int v321;
                                    v321 = 4 * v298;
                                    int v322;
                                    v322 = v321 + v300;
                                    v287[v322] = v320;
                                    v300 += 1 ;
                                }
                                v298 += 1 ;
                            }
                            float v323[4];
                            int v324;
                            v324 = 0;
                            while (while_method_6(v324)){
                                int v326;
                                v326 = 0;
                                while (while_method_0(v326)){
                                    assert("Tensor range check" && 0 <= v324 && v324 < 1);
                                    assert("Tensor range check" && 0 <= v326 && v326 < 4);
                                    int v328;
                                    v328 = 4 * v324;
                                    int v329;
                                    v329 = v328 + v326;
                                    float v330;
                                    v330 = v285[v329];
                                    float v331;
                                    v331 = v286[v329];
                                    bool v332;
                                    v332 = v331 == 0.0f;
                                    bool v333;
                                    v333 = v332 != true;
                                    float v335;
                                    if (v333){
                                        float v334;
                                        v334 = v330 / v331;
                                        v335 = v334;
                                    } else {
                                        v335 = 0.0f;
                                    }
                                    assert("Tensor range check" && 0 <= v324 && v324 < 1);
                                    assert("Tensor range check" && 0 <= v326 && v326 < 4);
                                    v323[v329] = v335;
                                    v326 += 1 ;
                                }
                                v324 += 1 ;
                            }
                            bool v336[4];
                            int v337;
                            v337 = 0;
                            while (while_method_6(v337)){
                                int v339;
                                v339 = 0;
                                while (while_method_0(v339)){
                                    assert("Tensor range check" && 0 <= v337 && v337 < 1);
                                    assert("Tensor range check" && 0 <= v339 && v339 < 4);
                                    int v341;
                                    v341 = 4 * v337;
                                    int v342;
                                    v342 = v341 + v339;
                                    float v343;
                                    v343 = v284[v342];
                                    int v344;
                                    v344 = v287[v342];
                                    bool v345;
                                    v345 = v344 < 3;
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
                                while (while_method_0(v349)){
                                    assert("Tensor range check" && 0 <= v347 && v347 < 1);
                                    assert("Tensor range check" && 0 <= v349 && v349 < 4);
                                    int v351;
                                    v351 = 4 * v347;
                                    int v352;
                                    v352 = v351 + v349;
                                    float v353;
                                    v353 = v284[v352];
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
                                while (while_method_0(v361)){
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
                            auto v369 = cooperative_groups::labeled_partition(v367,v368);
                            Closure1 v370{};
                            float v371;
                            v371 = cooperative_groups::reduce(v369, v358, v370);
                            int v372[4];
                            int v373;
                            v373 = 0;
                            while (while_method_6(v373)){
                                int v375;
                                v375 = 0;
                                while (while_method_0(v375)){
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
                                while (while_method_0(v384)){
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
                            auto v392 = cooperative_groups::labeled_partition(v390,v391);
                            Closure2 v393{};
                            int v394;
                            v394 = cooperative_groups::reduce(v392, v381, v393);
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
                                while (while_method_0(v400)){
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
                            float v412[4];
                            int v413;
                            v413 = 0;
                            while (while_method_6(v413)){
                                int v415;
                                v415 = 0;
                                while (while_method_0(v415)){
                                    assert("Tensor range check" && 0 <= v413 && v413 < 1);
                                    assert("Tensor range check" && 0 <= v415 && v415 < 4);
                                    int v417;
                                    v417 = 4 * v413;
                                    int v418;
                                    v418 = v417 + v415;
                                    float v419;
                                    v419 = v323[v418];
                                    int v420;
                                    v420 = v287[v418];
                                    bool v421;
                                    v421 = v273 == v420;
                                    float v424;
                                    if (v421){
                                        float v422;
                                        v422 = v274 - v419;
                                        float v423;
                                        v423 = v422 / v272;
                                        v424 = v423;
                                    } else {
                                        v424 = 0.0f;
                                    }
                                    float v425;
                                    v425 = v424 + v419;
                                    assert("Tensor range check" && 0 <= v413 && v413 < 1);
                                    assert("Tensor range check" && 0 <= v415 && v415 < 4);
                                    v412[v418] = v425;
                                    v415 += 1 ;
                                }
                                v413 += 1 ;
                            }
                            float v426[4];
                            int v427;
                            v427 = 0;
                            while (while_method_6(v427)){
                                int v429;
                                v429 = 0;
                                while (while_method_0(v429)){
                                    assert("Tensor range check" && 0 <= v427 && v427 < 1);
                                    assert("Tensor range check" && 0 <= v429 && v429 < 4);
                                    int v431;
                                    v431 = 4 * v427;
                                    int v432;
                                    v432 = v431 + v429;
                                    float v433;
                                    v433 = v397[v432];
                                    float v434;
                                    v434 = v412[v432];
                                    float v435;
                                    v435 = v433 * v434;
                                    assert("Tensor range check" && 0 <= v427 && v427 < 1);
                                    assert("Tensor range check" && 0 <= v429 && v429 < 4);
                                    v426[v432] = v435;
                                    v429 += 1 ;
                                }
                                v427 += 1 ;
                            }
                            float v436;
                            v436 = 0.0f;
                            int v437;
                            v437 = 0;
                            while (while_method_6(v437)){
                                int v439;
                                v439 = 0;
                                while (while_method_0(v439)){
                                    assert("Tensor range check" && 0 <= v437 && v437 < 1);
                                    assert("Tensor range check" && 0 <= v439 && v439 < 4);
                                    int v441;
                                    v441 = 4 * v437;
                                    int v442;
                                    v442 = v441 + v439;
                                    float v443;
                                    v443 = v426[v442];
                                    float v444;
                                    v444 = v436 + v443;
                                    v436 = v444;
                                    v439 += 1 ;
                                }
                                v437 += 1 ;
                            }
                            auto v445 = cooperative_groups::coalesced_threads();
                            int v446;
                            v446 = threadIdx.x;
                            auto v447 = cooperative_groups::labeled_partition(v445,v446);
                            float v448;
                            v448 = cooperative_groups::reduce(v447, v436, v370);
                            int v449;
                            v449 = 0;
                            while (while_method_6(v449)){
                                int v451;
                                v451 = 0;
                                while (while_method_0(v451)){
                                    assert("Tensor range check" && 0 <= v449 && v449 < 1);
                                    assert("Tensor range check" && 0 <= v451 && v451 < 4);
                                    int v453;
                                    v453 = 4 * v449;
                                    int v454;
                                    v454 = v453 + v451;
                                    float v455;
                                    v455 = v412[v454];
                                    int v456;
                                    v456 = v287[v454];
                                    float v457;
                                    v457 = v455 - v448;
                                    float v458;
                                    v458 = v275 * v457;
                                    assert("Tensor range check" && 0 <= v456 && v456 < 4);
                                    float * v459;
                                    v459 = v276+v456;
                                    float v461;
                                    v461 = atomicAdd(v459,v458);
                                    v451 += 1 ;
                                }
                                v449 += 1 ;
                            }
                            int v462;
                            v462 = 0;
                            while (while_method_6(v462)){
                                assert("Tensor range check" && 0 <= v462 && v462 < 1);
                                assert("Tensor range check" && 0 <= v462 && v462 < 1);
                                v462 += 1 ;
                            }
                            assert("Tensor range check" && 0 <= v269 && v269 < 256);
                            v248[v269] = v448;
                            v258 += 1 ;
                        }
                        asm("barrier.cta.sync %0;" :: "r"(0));
                        assert("Tensor range check" && 0 <= v250 && v250 < 256);
                        float v464;
                        v464 = v248[v250];
                        asm("barrier.cta.sync %0;" :: "r"(0));
                        assert("Tensor range check" && 0 <= v142 && v142 < 2);
                        v126[v142] = v464;
                    }
                    int v465;
                    v465 = threadIdx.x;
                    int v466;
                    v466 = blockIdx.x;
                    int v467;
                    v467 = v466 * 256;
                    int v468;
                    v468 = v465 + v467;
                    assert("Tensor range check" && 0 <= v120 && v120 < 4);
                    int v469;
                    v469 = 12288 * v120;
                    assert("Tensor range check" && 0 <= v468 && v468 < 6144);
                    int v470;
                    v470 = 2 * v468;
                    int v471;
                    v471 = v470 + v469;
                    double * v472;
                    v472 = v114+v471;
                    double * v474;
                    v474 = v116+v471;
                    double * v476;
                    v476 = v472+0;
                    double * v478;
                    v478 = v474+0;
                    double * v480;
                    v480 = v472+0;
                    double * v482;
                    v482 = v474+0;
                    int v484;
                    v484 = sizeof(double *);
                    unsigned long long v485;
                    v485 = (unsigned long long)v484;
                    unsigned long long v486;
                    v486 = 256ull * v485;
                    unsigned long long v487;
                    v487 = v486 + 16ull;
                    unsigned long long v488;
                    v488 = v487 - 1ull;
                    unsigned long long v489;
                    v489 = v488 % 16ull;
                    unsigned long long v490;
                    v490 = v488 - v489;
                    unsigned long long v491;
                    v491 = v490 + v486;
                    unsigned long long v492;
                    v492 = v491 + 16ull;
                    unsigned long long v493;
                    v493 = v492 - 1ull;
                    unsigned long long v494;
                    v494 = v493 % 16ull;
                    unsigned long long v495;
                    v495 = v493 - v494;
                    unsigned long long v496;
                    v496 = v495 + v486;
                    unsigned long long v497;
                    v497 = v496 + 16ull;
                    unsigned long long v498;
                    v498 = v497 - 1ull;
                    unsigned long long v499;
                    v499 = v498 % 16ull;
                    unsigned long long v500;
                    v500 = v498 - v499;
                    unsigned long long v501;
                    v501 = v500 + v486;
                    bool v502;
                    v502 = v501 <= 98304ull;
                    bool v503;
                    v503 = v502 == false;
                    if (v503){
                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v502);
                    } else {
                    }
                    extern __shared__ unsigned char v505[];
                    bool v506;
                    v506 = v501 <= v501;
                    bool v507;
                    v507 = v506 == false;
                    if (v507){
                        assert("The length of the partition has to be less than or equal to the length of the base array." && v506);
                    } else {
                    }
                    double * * v509;
                    v509 = reinterpret_cast<double * *>(&v505[0ull]);
                    double * * v511;
                    v511 = reinterpret_cast<double * *>(&v505[v490]);
                    double * * v513;
                    v513 = reinterpret_cast<double * *>(&v505[v495]);
                    double * * v515;
                    v515 = reinterpret_cast<double * *>(&v505[v500]);
                    int v517;
                    v517 = threadIdx.x;
                    assert("Tensor range check" && 0 <= v517 && v517 < 256);
                    v509[v517] = v476;
                    v511[v517] = v478;
                    v513[v517] = v480;
                    v515[v517] = v482;
                    asm("barrier.cta.sync %0;" :: "r"(0));
                    bool v518;
                    v518 = 0 <= v517;
                    bool v519;
                    v519 = v518 == false;
                    if (v519){
                        assert("The index needs to be zero or positive." && v518);
                    } else {
                    }
                    int v521;
                    v521 = v517 % 1;
                    bool v522;
                    v522 = v517 < 256;
                    bool v523;
                    v523 = v522 == false;
                    if (v523){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v522);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v517 && v517 < 256);
                    int v525;
                    v525 = 0;
                    while (while_method_6(v525)){
                        bool v527;
                        v527 = v518 && v522;
                        bool v528;
                        v528 = v527 == false;
                        if (v528){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v527);
                        } else {
                        }
                        bool v530;
                        v530 = 0 <= v525;
                        bool v532;
                        if (v530){
                            bool v531;
                            v531 = v525 < 1;
                            v532 = v531;
                        } else {
                            v532 = false;
                        }
                        bool v533;
                        v533 = v532 == false;
                        if (v533){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v532);
                        } else {
                        }
                        int v535;
                        v535 = v525 * 256;
                        int v536;
                        v536 = v535 + v517;
                        assert("Tensor range check" && 0 <= v525 && v525 < 1);
                        int v537;
                        v537 = 256 * v525;
                        int v538;
                        v538 = v537 + v517;
                        double * v539;
                        v539 = v509[v538];
                        double * v540;
                        v540 = v511[v538];
                        double * v541;
                        v541 = v513[v538];
                        double * v542;
                        v542 = v515[v538];
                        int v543;
                        v543 = blockIdx.x;
                        int v544;
                        v544 = v543 * 256;
                        int v545;
                        v545 = v544 + v536;
                        assert("Tensor range check" && 0 <= v521 && v521 < 1);
                        int v546;
                        v546 = 2 * v521;
                        double v547[2];
                        double v548[2];
                        int v549[2];
                        int v550;
                        v550 = 0;
                        while (while_method_6(v550)){
                            assert("Tensor range check" && 0 <= v550 && v550 < 1);
                            int v552;
                            v552 = 2 * v550;
                            assert("Tensor range check" && 0 <= v550 && v550 < 1);
                            int v553;
                            v553 = v552 + v546;
                            int4* v554;
                            v554 = reinterpret_cast<int4*>(v539 + v553);
                            int4* v555;
                            v555 = reinterpret_cast<int4*>(v547 + v552);
                            assert("Pointer alignment check" && (unsigned long long)(v554) % 2 == 0 && (unsigned long long)(v555) % 2 == 0);
                            *v555 = *v554;
                            int4* v556;
                            v556 = reinterpret_cast<int4*>(v540 + v553);
                            int4* v557;
                            v557 = reinterpret_cast<int4*>(v548 + v552);
                            assert("Pointer alignment check" && (unsigned long long)(v556) % 2 == 0 && (unsigned long long)(v557) % 2 == 0);
                            *v557 = *v556;
                            v550 += 1 ;
                        }
                        int v558;
                        v558 = 0;
                        while (while_method_6(v558)){
                            int v560;
                            v560 = 0;
                            while (while_method_2(v560)){
                                bool v562;
                                v562 = 0 <= v560;
                                bool v564;
                                if (v562){
                                    bool v563;
                                    v563 = v560 < 2;
                                    v564 = v563;
                                } else {
                                    v564 = false;
                                }
                                bool v565;
                                v565 = v564 == false;
                                if (v565){
                                    assert("The indices should be inside the range of the dimension." && v564);
                                } else {
                                }
                                bool v567;
                                v567 = 0 <= v521;
                                bool v569;
                                if (v567){
                                    bool v568;
                                    v568 = v521 < 1;
                                    v569 = v568;
                                } else {
                                    v569 = false;
                                }
                                bool v570;
                                v570 = v569 == false;
                                if (v570){
                                    assert("The indices should be inside the range of the dimension." && v569);
                                } else {
                                }
                                int v572;
                                v572 = v521 * 2;
                                int v573;
                                v573 = v560 + v572;
                                bool v574;
                                v574 = 0 <= v558;
                                bool v576;
                                if (v574){
                                    bool v575;
                                    v575 = v558 < 1;
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
                                v579 = v558 * 2;
                                int v580;
                                v580 = v573 + v579;
                                assert("Tensor range check" && 0 <= v558 && v558 < 1);
                                assert("Tensor range check" && 0 <= v560 && v560 < 2);
                                int v581;
                                v581 = 2 * v558;
                                int v582;
                                v582 = v581 + v560;
                                v549[v582] = v580;
                                v560 += 1 ;
                            }
                            v558 += 1 ;
                        }
                        double v583[2];
                        double v584[2];
                        int v585;
                        v585 = 0;
                        while (while_method_6(v585)){
                            int v587;
                            v587 = 0;
                            while (while_method_2(v587)){
                                assert("Tensor range check" && 0 <= v585 && v585 < 1);
                                assert("Tensor range check" && 0 <= v587 && v587 < 2);
                                int v589;
                                v589 = 2 * v585;
                                int v590;
                                v590 = v589 + v587;
                                double v591;
                                v591 = v547[v590];
                                double v592;
                                v592 = v548[v590];
                                assert("Tensor range check" && 0 <= v585 && v585 < 1);
                                assert("Tensor range check" && 0 <= v587 && v587 < 2);
                                v583[v590] = 0.0;
                                v584[v590] = 0.0;
                                v587 += 1 ;
                            }
                            v585 += 1 ;
                        }
                        int v593;
                        v593 = 0;
                        while (while_method_6(v593)){
                            assert("Tensor range check" && 0 <= v593 && v593 < 1);
                            int v595;
                            v595 = 2 * v593;
                            int v596;
                            v596 = v595 + v546;
                            assert("Tensor range check" && 0 <= v593 && v593 < 1);
                            int4* v597;
                            v597 = reinterpret_cast<int4*>(v583 + v595);
                            int4* v598;
                            v598 = reinterpret_cast<int4*>(v541 + v596);
                            assert("Pointer alignment check" && (unsigned long long)(v597) % 2 == 0 && (unsigned long long)(v598) % 2 == 0);
                            *v598 = *v597;
                            int4* v599;
                            v599 = reinterpret_cast<int4*>(v584 + v595);
                            int4* v600;
                            v600 = reinterpret_cast<int4*>(v542 + v596);
                            assert("Pointer alignment check" && (unsigned long long)(v599) % 2 == 0 && (unsigned long long)(v600) % 2 == 0);
                            *v600 = *v599;
                            v593 += 1 ;
                        }
                        assert("Tensor range check" && 0 <= v536 && v536 < 256);
                        v525 += 1 ;
                    }
                    asm("barrier.cta.sync %0;" :: "r"(0));
                    assert("Tensor range check" && 0 <= v517 && v517 < 256);
                    asm("barrier.cta.sync %0;" :: "r"(0));
                    assert("Tensor range check" && 0 <= v120 && v120 < 4);
                    assert("Tensor range check" && 0 <= v468 && v468 < 6144);
                    int v601;
                    v601 = v131 + v468;
                    v118[v601] = 0;
                    v120 += 1 ;
                }
                v38 += 1 ;
            }
            v36 += 1 ;
        }
        cooperative_groups::grid_group & v602 = v27.v1;
        cooperative_groups::grid_group & v603 = v602;
        curandStatePhilox4_32_10_t & v604 = v27.v5;
        curandStatePhilox4_32_10_t & v605 = v604;
        unsigned int * v606;
        v606 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
        int * v608;
        v608 = reinterpret_cast<int *>(&v2[262144ull]);
        float * v610;
        v610 = reinterpret_cast<float *>(&v2[262160ull]);
        float * v612;
        v612 = reinterpret_cast<float *>(&v2[524304ull]);
        float * v614;
        v614 = reinterpret_cast<float *>(&v2[786448ull]);
        float * v616;
        v616 = reinterpret_cast<float *>(&v2[1048592ull]);
        float * v618;
        v618 = reinterpret_cast<float *>(&v2[1310736ull]);
        float * v620;
        v620 = reinterpret_cast<float *>(&v2[1572880ull]);
        float * v622;
        v622 = reinterpret_cast<float *>(&v2[1835024ull]);
        int * v624;
        v624 = reinterpret_cast<int *>(&v0[6389760ull]);
        float * v626;
        v626 = reinterpret_cast<float *>(&v0[7962624ull]);
        int * v628;
        v628 = reinterpret_cast<int *>(&v0[9535488ull]);
        int * v630;
        v630 = reinterpret_cast<int *>(&v0[11108352ull]);
        double * v632;
        v632 = reinterpret_cast<double *>(&v0[12681216ull]);
        double * v634;
        v634 = reinterpret_cast<double *>(&v0[18972672ull]);
        double * v636;
        v636 = reinterpret_cast<double *>(&v2[2097168ull]);
        double * v638;
        v638 = reinterpret_cast<double *>(&v2[2490384ull]);
        int * v640;
        v640 = reinterpret_cast<int *>(&v2[2883600ull]);
        v603.sync() ;
        int v642;
        v642 = threadIdx.x;
        int v643;
        v643 = blockIdx.x;
        int v644;
        v644 = v643 * 256;
        int v645;
        v645 = v642 + v644;
        bool v646;
        v646 = v645 == 0;
        if (v646){
            int v647;
            v647 = 0;
            int v648;
            v648 = 4;
            int v649;
            v649 = int_range_6(v648, v647, v605);
            v608[0] = v649;
        } else {
        }
        __syncwarp();
        int v650;
        v650 = threadIdx.x;
        bool v651;
        v651 = 0 <= v650;
        bool v652;
        v652 = v651 == false;
        if (v652){
            assert("The index needs to be zero or positive." && v651);
        } else {
        }
        int v654;
        v654 = v650 % 1;
        int v655;
        v655 = v650 % 256;
        int v656;
        v656 = v650 / 256;
        bool v657;
        v657 = v656 < 1;
        bool v658;
        v658 = v657 == false;
        if (v658){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v657);
        } else {
        }
        assert("Tensor range check" && 0 <= v656 && v656 < 1);
        assert("Tensor range check" && 0 <= v655 && v655 < 256);
        assert("Tensor range check" && 0 <= v654 && v654 < 1);
        int v660;
        v660 = 4 * v654;
        int v661;
        v661 = 4 * v655;
        int v662;
        v662 = v661 + v660;
        int v663;
        v663 = 16384 * v656;
        int v664;
        v664 = v663 + v662;
        assert("Tensor range check" && 0 <= v656 && v656 < 1);
        assert("Tensor range check" && 0 <= v655 && v655 < 256);
        assert("Tensor range check" && 0 <= v654 && v654 < 1);
        int v665;
        v665 = blockIdx.x;
        int v666;
        v666 = v665;
        while (while_method_11(v666)){
            bool v668;
            v668 = 0 <= v666;
            bool v669;
            v669 = v668 == false;
            if (v669){
                assert("The index needs to be zero or positive." && v668);
            } else {
            }
            int v671;
            v671 = v666 % 16;
            int v672;
            v672 = v666 / 16;
            bool v673;
            v673 = v672 < 4;
            bool v674;
            v674 = v673 == false;
            if (v674){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v673);
            } else {
            }
            assert("Tensor range check" && 0 <= v672 && v672 < 4);
            assert("Tensor range check" && 0 <= v671 && v671 < 16);
            int v676;
            v676 = 1024 * v671;
            int v677;
            v677 = v676 + v664;
            int v678;
            v678 = 16384 * v672;
            int v679;
            v679 = v678 + v677;
            float v680[4];
            float v681[4];
            float v682[4];
            float v683[4];
            float v684[4];
            float v685[4];
            float v686[4];
            int v687[4];
            int v688;
            v688 = 0;
            while (while_method_6(v688)){
                assert("Tensor range check" && 0 <= v688 && v688 < 1);
                int v690;
                v690 = 4 * v688;
                assert("Tensor range check" && 0 <= v688 && v688 < 1);
                int v691;
                v691 = v690 + v679;
                int4* v692;
                v692 = reinterpret_cast<int4*>(v610 + v691);
                int4* v693;
                v693 = reinterpret_cast<int4*>(v680 + v690);
                assert("Pointer alignment check" && (unsigned long long)(v692) % 4 == 0 && (unsigned long long)(v693) % 4 == 0);
                *v693 = *v692;
                int4* v694;
                v694 = reinterpret_cast<int4*>(v612 + v691);
                int4* v695;
                v695 = reinterpret_cast<int4*>(v681 + v690);
                assert("Pointer alignment check" && (unsigned long long)(v694) % 4 == 0 && (unsigned long long)(v695) % 4 == 0);
                *v695 = *v694;
                int4* v696;
                v696 = reinterpret_cast<int4*>(v614 + v691);
                int4* v697;
                v697 = reinterpret_cast<int4*>(v682 + v690);
                assert("Pointer alignment check" && (unsigned long long)(v696) % 4 == 0 && (unsigned long long)(v697) % 4 == 0);
                *v697 = *v696;
                int4* v698;
                v698 = reinterpret_cast<int4*>(v616 + v691);
                int4* v699;
                v699 = reinterpret_cast<int4*>(v683 + v690);
                assert("Pointer alignment check" && (unsigned long long)(v698) % 4 == 0 && (unsigned long long)(v699) % 4 == 0);
                *v699 = *v698;
                int4* v700;
                v700 = reinterpret_cast<int4*>(v618 + v691);
                int4* v701;
                v701 = reinterpret_cast<int4*>(v684 + v690);
                assert("Pointer alignment check" && (unsigned long long)(v700) % 4 == 0 && (unsigned long long)(v701) % 4 == 0);
                *v701 = *v700;
                int4* v702;
                v702 = reinterpret_cast<int4*>(v620 + v691);
                int4* v703;
                v703 = reinterpret_cast<int4*>(v685 + v690);
                assert("Pointer alignment check" && (unsigned long long)(v702) % 4 == 0 && (unsigned long long)(v703) % 4 == 0);
                *v703 = *v702;
                int4* v704;
                v704 = reinterpret_cast<int4*>(v622 + v691);
                int4* v705;
                v705 = reinterpret_cast<int4*>(v686 + v690);
                assert("Pointer alignment check" && (unsigned long long)(v704) % 4 == 0 && (unsigned long long)(v705) % 4 == 0);
                *v705 = *v704;
                v688 += 1 ;
            }
            int v706;
            v706 = 0;
            while (while_method_6(v706)){
                int v708;
                v708 = 0;
                while (while_method_0(v708)){
                    bool v710;
                    v710 = 0 <= v708;
                    bool v712;
                    if (v710){
                        bool v711;
                        v711 = v708 < 4;
                        v712 = v711;
                    } else {
                        v712 = false;
                    }
                    bool v713;
                    v713 = v712 == false;
                    if (v713){
                        assert("The indices should be inside the range of the dimension." && v712);
                    } else {
                    }
                    bool v715;
                    v715 = 0 <= v654;
                    bool v717;
                    if (v715){
                        bool v716;
                        v716 = v654 < 1;
                        v717 = v716;
                    } else {
                        v717 = false;
                    }
                    bool v718;
                    v718 = v717 == false;
                    if (v718){
                        assert("The indices should be inside the range of the dimension." && v717);
                    } else {
                    }
                    int v720;
                    v720 = v654 * 4;
                    int v721;
                    v721 = v708 + v720;
                    bool v722;
                    v722 = 0 <= v706;
                    bool v724;
                    if (v722){
                        bool v723;
                        v723 = v706 < 1;
                        v724 = v723;
                    } else {
                        v724 = false;
                    }
                    bool v725;
                    v725 = v724 == false;
                    if (v725){
                        assert("The indices should be inside the range of the dimension." && v724);
                    } else {
                    }
                    int v727;
                    v727 = v706 * 4;
                    int v728;
                    v728 = v721 + v727;
                    assert("Tensor range check" && 0 <= v706 && v706 < 1);
                    assert("Tensor range check" && 0 <= v708 && v708 < 4);
                    int v729;
                    v729 = 4 * v706;
                    int v730;
                    v730 = v729 + v708;
                    v687[v730] = v728;
                    v708 += 1 ;
                }
                v706 += 1 ;
            }
            bool v731;
            v731 = 0 <= v656;
            bool v732;
            v732 = v731 && v657;
            bool v733;
            v733 = v732 == false;
            if (v733){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v732);
            } else {
            }
            bool v735;
            v735 = 0 <= v655;
            bool v737;
            if (v735){
                bool v736;
                v736 = v655 < 256;
                v737 = v736;
            } else {
                v737 = false;
            }
            bool v738;
            v738 = v737 == false;
            if (v738){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v737);
            } else {
            }
            bool v740;
            v740 = 0 <= v672;
            bool v741;
            v741 = v740 && v673;
            bool v742;
            v742 = v741 == false;
            if (v742){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v741);
            } else {
            }
            bool v744;
            v744 = 0 <= v671;
            bool v746;
            if (v744){
                bool v745;
                v745 = v671 < 16;
                v746 = v745;
            } else {
                v746 = false;
            }
            bool v747;
            v747 = v746 == false;
            if (v747){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v746);
            } else {
            }
            int v749;
            v749 = v671 * 256;
            int v750;
            v750 = v672 + v656;
            int v751;
            v751 = v749 + v655;
            bool v752[4];
            int v753;
            v753 = 0;
            while (while_method_6(v753)){
                int v755;
                v755 = 0;
                while (while_method_0(v755)){
                    assert("Tensor range check" && 0 <= v753 && v753 < 1);
                    assert("Tensor range check" && 0 <= v755 && v755 < 4);
                    int v757;
                    v757 = 4 * v753;
                    int v758;
                    v758 = v757 + v755;
                    float v759;
                    v759 = v682[v758];
                    bool v760;
                    v760 = v759 == 0.0f;
                    bool v761;
                    v761 = v760 != true;
                    assert("Tensor range check" && 0 <= v753 && v753 < 1);
                    assert("Tensor range check" && 0 <= v755 && v755 < 4);
                    v752[v758] = v761;
                    v755 += 1 ;
                }
                v753 += 1 ;
            }
            bool v762;
            v762 = false;
            int v763;
            v763 = 0;
            while (while_method_6(v763)){
                int v765;
                v765 = 0;
                while (while_method_0(v765)){
                    assert("Tensor range check" && 0 <= v763 && v763 < 1);
                    assert("Tensor range check" && 0 <= v765 && v765 < 4);
                    int v767;
                    v767 = 4 * v763;
                    int v768;
                    v768 = v767 + v765;
                    bool v769;
                    v769 = v752[v768];
                    bool v770;
                    v770 = v762 || v769;
                    v762 = v770;
                    v765 += 1 ;
                }
                v763 += 1 ;
            }
            auto v771 = cooperative_groups::coalesced_threads();
            int v772;
            v772 = threadIdx.x;
            auto v773 = cooperative_groups::labeled_partition(v771,v772);
            Closure8 v774{};
            bool v775;
            v775 = cooperative_groups::reduce(v773, v762, v774);
            if (v775){
                float v776[4];
                int v777;
                v777 = 0;
                while (while_method_6(v777)){
                    int v779;
                    v779 = 0;
                    while (while_method_0(v779)){
                        assert("Tensor range check" && 0 <= v777 && v777 < 1);
                        assert("Tensor range check" && 0 <= v779 && v779 < 4);
                        int v781;
                        v781 = 4 * v777;
                        int v782;
                        v782 = v781 + v779;
                        float v783;
                        v783 = v681[v782];
                        float v784;
                        v784 = v682[v782];
                        float v785;
                        v785 = v783 + v784;
                        bool v786;
                        v786 = 0.0f >= v785;
                        float v787;
                        if (v786){
                            v787 = 0.0f;
                        } else {
                            v787 = v785;
                        }
                        assert("Tensor range check" && 0 <= v777 && v777 < 1);
                        assert("Tensor range check" && 0 <= v779 && v779 < 4);
                        v776[v782] = v787;
                        v779 += 1 ;
                    }
                    v777 += 1 ;
                }
                float v788[4];
                int v789;
                v789 = 0;
                while (while_method_6(v789)){
                    int v791;
                    v791 = 0;
                    while (while_method_0(v791)){
                        assert("Tensor range check" && 0 <= v789 && v789 < 1);
                        assert("Tensor range check" && 0 <= v791 && v791 < 4);
                        int v793;
                        v793 = 4 * v789;
                        int v794;
                        v794 = v793 + v791;
                        float v795;
                        v795 = v776[v794];
                        bool v796;
                        v796 = 0.0f >= v795;
                        float v797;
                        if (v796){
                            v797 = 0.0f;
                        } else {
                            v797 = v795;
                        }
                        assert("Tensor range check" && 0 <= v789 && v789 < 1);
                        assert("Tensor range check" && 0 <= v791 && v791 < 4);
                        v788[v794] = v797;
                        v791 += 1 ;
                    }
                    v789 += 1 ;
                }
                float v798;
                v798 = 0.0f;
                int v799;
                v799 = 0;
                while (while_method_6(v799)){
                    int v801;
                    v801 = 0;
                    while (while_method_0(v801)){
                        assert("Tensor range check" && 0 <= v799 && v799 < 1);
                        assert("Tensor range check" && 0 <= v801 && v801 < 4);
                        int v803;
                        v803 = 4 * v799;
                        int v804;
                        v804 = v803 + v801;
                        float v805;
                        v805 = v788[v804];
                        float v806;
                        v806 = v798 + v805;
                        v798 = v806;
                        v801 += 1 ;
                    }
                    v799 += 1 ;
                }
                auto v807 = cooperative_groups::coalesced_threads();
                int v808;
                v808 = threadIdx.x;
                auto v809 = cooperative_groups::labeled_partition(v807,v808);
                Closure1 v810{};
                float v811;
                v811 = cooperative_groups::reduce(v809, v798, v810);
                float v812[4];
                int v813;
                v813 = 0;
                while (while_method_6(v813)){
                    int v815;
                    v815 = 0;
                    while (while_method_0(v815)){
                        assert("Tensor range check" && 0 <= v813 && v813 < 1);
                        assert("Tensor range check" && 0 <= v815 && v815 < 4);
                        int v817;
                        v817 = 4 * v813;
                        int v818;
                        v818 = v817 + v815;
                        float v819;
                        v819 = v788[v818];
                        bool v820;
                        v820 = v811 == 0.0f;
                        bool v821;
                        v821 = v820 != true;
                        float v823;
                        if (v821){
                            float v822;
                            v822 = v819 / v811;
                            v823 = v822;
                        } else {
                            v823 = 0.25f;
                        }
                        assert("Tensor range check" && 0 <= v813 && v813 < 1);
                        assert("Tensor range check" && 0 <= v815 && v815 < 4);
                        v812[v818] = v823;
                        v815 += 1 ;
                    }
                    v813 += 1 ;
                }
                float v824[4];
                int v825;
                v825 = 0;
                while (while_method_6(v825)){
                    int v827;
                    v827 = 0;
                    while (while_method_0(v827)){
                        assert("Tensor range check" && 0 <= v825 && v825 < 1);
                        assert("Tensor range check" && 0 <= v827 && v827 < 4);
                        int v829;
                        v829 = 4 * v825;
                        int v830;
                        v830 = v829 + v827;
                        float v831;
                        v831 = v680[v830];
                        float v832;
                        v832 = v812[v830];
                        float v833;
                        v833 = v831 + v832;
                        assert("Tensor range check" && 0 <= v825 && v825 < 1);
                        assert("Tensor range check" && 0 <= v827 && v827 < 4);
                        v824[v830] = v833;
                        v827 += 1 ;
                    }
                    v825 += 1 ;
                }
                float v834[4];
                int v835;
                v835 = 0;
                while (while_method_6(v835)){
                    int v837;
                    v837 = 0;
                    while (while_method_0(v837)){
                        assert("Tensor range check" && 0 <= v835 && v835 < 1);
                        assert("Tensor range check" && 0 <= v837 && v837 < 4);
                        int v839;
                        v839 = 4 * v835;
                        int v840;
                        v840 = v839 + v837;
                        float v841;
                        v841 = v824[v840];
                        float v842;
                        v842 = -v841;
                        bool v843;
                        v843 = v841 >= v842;
                        float v844;
                        if (v843){
                            v844 = v841;
                        } else {
                            v844 = v842;
                        }
                        assert("Tensor range check" && 0 <= v835 && v835 < 1);
                        assert("Tensor range check" && 0 <= v837 && v837 < 4);
                        v834[v840] = v844;
                        v837 += 1 ;
                    }
                    v835 += 1 ;
                }
                float v845;
                v845 = 0.0f;
                int v846;
                v846 = 0;
                while (while_method_6(v846)){
                    int v848;
                    v848 = 0;
                    while (while_method_0(v848)){
                        assert("Tensor range check" && 0 <= v846 && v846 < 1);
                        assert("Tensor range check" && 0 <= v848 && v848 < 4);
                        int v850;
                        v850 = 4 * v846;
                        int v851;
                        v851 = v850 + v848;
                        float v852;
                        v852 = v834[v851];
                        float v853;
                        v853 = v845 + v852;
                        v845 = v853;
                        v848 += 1 ;
                    }
                    v846 += 1 ;
                }
                auto v854 = cooperative_groups::coalesced_threads();
                int v855;
                v855 = threadIdx.x;
                auto v856 = cooperative_groups::labeled_partition(v854,v855);
                float v857;
                v857 = cooperative_groups::reduce(v856, v845, v810);
                bool v858;
                v858 = v857 > 100.0f;
                float v860;
                if (v858){
                    float v859;
                    v859 = 100.0f / v857;
                    v860 = v859;
                } else {
                    v860 = 1.0f;
                }
                float v861[4];
                int v862;
                v862 = 0;
                while (while_method_6(v862)){
                    int v864;
                    v864 = 0;
                    while (while_method_0(v864)){
                        assert("Tensor range check" && 0 <= v862 && v862 < 1);
                        assert("Tensor range check" && 0 <= v864 && v864 < 4);
                        int v866;
                        v866 = 4 * v862;
                        int v867;
                        v867 = v866 + v864;
                        float v868;
                        v868 = v834[v867];
                        float v869;
                        v869 = v860 * v868;
                        assert("Tensor range check" && 0 <= v862 && v862 < 1);
                        assert("Tensor range check" && 0 <= v864 && v864 < 4);
                        v861[v867] = v869;
                        v864 += 1 ;
                    }
                    v862 += 1 ;
                }
                float v870[4];
                float v871[4];
                int v872;
                v872 = 0;
                while (while_method_6(v872)){
                    int v874;
                    v874 = 0;
                    while (while_method_0(v874)){
                        assert("Tensor range check" && 0 <= v872 && v872 < 1);
                        assert("Tensor range check" && 0 <= v874 && v874 < 4);
                        int v876;
                        v876 = 4 * v872;
                        int v877;
                        v877 = v876 + v874;
                        float v878;
                        v878 = v680[v877];
                        float v879;
                        v879 = v681[v877];
                        float v880;
                        v880 = v682[v877];
                        float v881;
                        v881 = v683[v877];
                        float v882;
                        v882 = v684[v877];
                        float v883;
                        v883 = v685[v877];
                        float v884;
                        v884 = v686[v877];
                        float v885;
                        v885 = v881 + v883;
                        float v886;
                        v886 = v882 + v884;
                        assert("Tensor range check" && 0 <= v872 && v872 < 1);
                        assert("Tensor range check" && 0 <= v874 && v874 < 4);
                        v870[v877] = v885;
                        v871[v877] = v886;
                        v874 += 1 ;
                    }
                    v872 += 1 ;
                }
                int v887;
                v887 = 0;
                while (while_method_6(v887)){
                    int v889;
                    v889 = 0;
                    while (while_method_0(v889)){
                        assert("Tensor range check" && 0 <= v887 && v887 < 1);
                        assert("Tensor range check" && 0 <= v889 && v889 < 4);
                        int v891;
                        v891 = 4 * v887;
                        int v892;
                        v892 = v891 + v889;
                        float v893;
                        v893 = v861[v892];
                        float v894;
                        v894 = v776[v892];
                        float v895;
                        v895 = v870[v892];
                        float v896;
                        v896 = v871[v892];
                        assert("Tensor range check" && 0 <= v887 && v887 < 1);
                        assert("Tensor range check" && 0 <= v889 && v889 < 4);
                        v680[v892] = v893;
                        v681[v892] = v894;
                        v682[v892] = 0.0f;
                        v683[v892] = v895;
                        v684[v892] = v896;
                        v685[v892] = 0.0f;
                        v686[v892] = 0.0f;
                        v889 += 1 ;
                    }
                    v887 += 1 ;
                }
            } else {
            }
            assert("Tensor range check" && 0 <= v672 && v672 < 4);
            assert("Tensor range check" && 0 <= v671 && v671 < 16);
            int v897;
            v897 = 0;
            while (while_method_6(v897)){
                assert("Tensor range check" && 0 <= v897 && v897 < 1);
                int v899;
                v899 = 4 * v897;
                int v900;
                v900 = v899 + v679;
                assert("Tensor range check" && 0 <= v897 && v897 < 1);
                int4* v901;
                v901 = reinterpret_cast<int4*>(v680 + v899);
                int4* v902;
                v902 = reinterpret_cast<int4*>(v610 + v900);
                assert("Pointer alignment check" && (unsigned long long)(v901) % 4 == 0 && (unsigned long long)(v902) % 4 == 0);
                *v902 = *v901;
                int4* v903;
                v903 = reinterpret_cast<int4*>(v681 + v899);
                int4* v904;
                v904 = reinterpret_cast<int4*>(v612 + v900);
                assert("Pointer alignment check" && (unsigned long long)(v903) % 4 == 0 && (unsigned long long)(v904) % 4 == 0);
                *v904 = *v903;
                int4* v905;
                v905 = reinterpret_cast<int4*>(v682 + v899);
                int4* v906;
                v906 = reinterpret_cast<int4*>(v614 + v900);
                assert("Pointer alignment check" && (unsigned long long)(v905) % 4 == 0 && (unsigned long long)(v906) % 4 == 0);
                *v906 = *v905;
                int4* v907;
                v907 = reinterpret_cast<int4*>(v683 + v899);
                int4* v908;
                v908 = reinterpret_cast<int4*>(v616 + v900);
                assert("Pointer alignment check" && (unsigned long long)(v907) % 4 == 0 && (unsigned long long)(v908) % 4 == 0);
                *v908 = *v907;
                int4* v909;
                v909 = reinterpret_cast<int4*>(v684 + v899);
                int4* v910;
                v910 = reinterpret_cast<int4*>(v618 + v900);
                assert("Pointer alignment check" && (unsigned long long)(v909) % 4 == 0 && (unsigned long long)(v910) % 4 == 0);
                *v910 = *v909;
                int4* v911;
                v911 = reinterpret_cast<int4*>(v685 + v899);
                int4* v912;
                v912 = reinterpret_cast<int4*>(v620 + v900);
                assert("Pointer alignment check" && (unsigned long long)(v911) % 4 == 0 && (unsigned long long)(v912) % 4 == 0);
                *v912 = *v911;
                int4* v913;
                v913 = reinterpret_cast<int4*>(v686 + v899);
                int4* v914;
                v914 = reinterpret_cast<int4*>(v622 + v900);
                assert("Pointer alignment check" && (unsigned long long)(v913) % 4 == 0 && (unsigned long long)(v914) % 4 == 0);
                *v914 = *v913;
                v897 += 1 ;
            }
            v666 += 24 ;
        }
        v603.sync() ;
        v34 += 1 ;
    }
    cooperative_groups::grid_group & v915 = v27.v1;
    cooperative_groups::grid_group & v916 = v915;
    int v917;
    v917 = threadIdx.x;
    int v918;
    v918 = blockIdx.x;
    int v919;
    v919 = v918 * 256;
    int v920;
    v920 = v917 + v919;
    int v921;
    v921 = v920;
    while (while_method_0(v921)){
        bool v923;
        v923 = 0 <= v921;
        bool v924;
        v924 = v923 == false;
        if (v924){
            assert("The index needs to be zero or positive." && v923);
        } else {
        }
        int v926;
        v926 = v921 % 1;
        bool v927;
        v927 = v921 < 4;
        bool v928;
        v928 = v927 == false;
        if (v928){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v927);
        } else {
        }
        assert("Tensor range check" && 0 <= v921 && v921 < 4);
        assert("Tensor range check" && 0 <= v926 && v926 < 1);
        int v930;
        v930 = 4 * v926;
        int v931;
        v931 = 4 * v921;
        int v932;
        v932 = v931 + v930;
        assert("Tensor range check" && 0 <= v921 && v921 < 4);
        assert("Tensor range check" && 0 <= v926 && v926 < 1);
        float v933[4];
        float v934[4];
        float v935[4];
        int4* v936;
        v936 = reinterpret_cast<int4*>(v4 + v932);
        int4* v937;
        v937 = reinterpret_cast<int4*>(v933 + 0);
        assert("Pointer alignment check" && (unsigned long long)(v936) % 4 == 0 && (unsigned long long)(v937) % 4 == 0);
        *v937 = *v936;
        int4* v938;
        v938 = reinterpret_cast<int4*>(v5 + v932);
        int4* v939;
        v939 = reinterpret_cast<int4*>(v934 + 0);
        assert("Pointer alignment check" && (unsigned long long)(v938) % 4 == 0 && (unsigned long long)(v939) % 4 == 0);
        *v939 = *v938;
        // Pushing the loop unrolling to: 0
        int v940;
        v940 = 0;
        #pragma unroll
        while (while_method_0(v940)){
            assert("Tensor range check" && 0 <= v940 && v940 < 4);
            float v942;
            v942 = v933[v940];
            float v943;
            v943 = v934[v940];
            bool v944;
            v944 = v943 == 0.0f;
            bool v945;
            v945 = v944 != true;
            float v947;
            if (v945){
                float v946;
                v946 = v942 / v943;
                v947 = v946;
            } else {
                v947 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v940 && v940 < 4);
            v935[v940] = v947;
            v940 += 1 ;
        }
        // Poping the loop unrolling to: 0
        int4* v948;
        v948 = reinterpret_cast<int4*>(v935 + 0);
        int4* v949;
        v949 = reinterpret_cast<int4*>(v6 + v932);
        assert("Pointer alignment check" && (unsigned long long)(v948) % 4 == 0 && (unsigned long long)(v949) % 4 == 0);
        *v949 = *v948;
        v921 += 6144 ;
    }
    v916.sync() ;
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
def method3() -> object:
    v0 = []
    return v0
def method2(v0 : US1) -> object:
    match v0:
        case US1_0(): # Call
            del v0
            v1 = method3()
            v2 = "Call"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US1_1(): # Fold
            del v0
            v4 = method3()
            v5 = "Fold"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US1_2(): # Raise
            del v0
            v7 = method3()
            v8 = "Raise"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method5(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method6(v0 : US2) -> object:
    match v0:
        case US2_0(): # Computer
            del v0
            v1 = method3()
            v2 = "Computer"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US2_1(): # Human
            del v0
            v4 = method3()
            v5 = "Human"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US2_2(): # Random
            del v0
            v7 = method3()
            v8 = "Random"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method4(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method5(v2):
        v5 = v0[v2]
        v6 = method6(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method1(v0 : US0) -> object:
    match v0:
        case US0_0(v1): # ActionSelected
            del v0
            v2 = method2(v1)
            del v1
            v3 = "ActionSelected"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US0_1(v5): # PlayerChanged
            del v0
            v6 = method4(v5)
            del v5
            v7 = "PlayerChanged"
            v8 = [v7,v6]
            del v6, v7
            return v8
        case US0_2(): # StartGame
            del v0
            v9 = method3()
            v10 = "StartGame"
            v11 = [v10,v9]
            del v9, v10
            return v11
        case US0_3(): # StartTrainingVsRando
            del v0
            v12 = method3()
            v13 = "StartTrainingVsRando"
            v14 = [v13,v12]
            del v12, v13
            return v14
        case US0_4(): # StartTrainingVsSelf
            del v0
            v15 = method3()
            v16 = "StartTrainingVsSelf"
            v17 = [v16,v15]
            del v15, v16
            return v17
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method0(v0 : US0) -> object:
    v1 = method1(v0)
    del v0
    return v1
def method12(v0 : u32) -> object:
    v1 = v0
    del v0
    return v1
def method11(v0 : u32) -> object:
    return method12(v0)
def method17(v0 : US6) -> object:
    match v0:
        case US6_0(): # Jack
            del v0
            v1 = method3()
            v2 = "Jack"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US6_1(): # King
            del v0
            v4 = method3()
            v5 = "King"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US6_2(): # Queen
            del v0
            v7 = method3()
            v8 = "Queen"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method16(v0 : US5) -> object:
    match v0:
        case US5_0(): # None
            del v0
            v1 = method3()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US5_1(v4): # Some
            del v0
            v5 = method17(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method18(v0 : bool) -> object:
    v1 = v0
    del v0
    return v1
def method19(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method5(v2):
        v5 = v0[v2]
        v6 = method17(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method20(v0 : i32) -> object:
    v1 = v0
    del v0
    return v1
def method21(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method5(v2):
        v5 = v0[v2]
        v6 = method20(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method15(v0 : US5, v1 : bool, v2 : static_array, v3 : i32, v4 : static_array, v5 : i32) -> object:
    v6 = method16(v0)
    del v0
    v7 = method18(v1)
    del v1
    v8 = method19(v2)
    del v2
    v9 = method20(v3)
    del v3
    v10 = method21(v4)
    del v4
    v11 = method20(v5)
    del v5
    v12 = {'community_card': v6, 'is_button_s_first_move': v7, 'pl_card': v8, 'player_turn': v9, 'pot': v10, 'raises_left': v11}
    del v6, v7, v8, v9, v10, v11
    return v12
def method22(v0 : US5, v1 : bool, v2 : static_array, v3 : i32, v4 : static_array, v5 : i32, v6 : US1) -> object:
    v7 = []
    v8 = method15(v0, v1, v2, v3, v4, v5)
    del v0, v1, v2, v3, v4, v5
    v7.append(v8)
    del v8
    v9 = method2(v6)
    del v6
    v7.append(v9)
    del v9
    v10 = v7
    del v7
    return v10
def method14(v0 : US4) -> object:
    match v0:
        case US4_0(v1, v2, v3, v4, v5, v6): # ChanceCommunityCard
            del v0
            v7 = method15(v1, v2, v3, v4, v5, v6)
            del v1, v2, v3, v4, v5, v6
            v8 = "ChanceCommunityCard"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US4_1(): # ChanceInit
            del v0
            v10 = method3()
            v11 = "ChanceInit"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US4_2(v13, v14, v15, v16, v17, v18): # Round
            del v0
            v19 = method15(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "Round"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case US4_3(v22, v23, v24, v25, v26, v27, v28): # RoundWithAction
            del v0
            v29 = method22(v22, v23, v24, v25, v26, v27, v28)
            del v22, v23, v24, v25, v26, v27, v28
            v30 = "RoundWithAction"
            v31 = [v30,v29]
            del v29, v30
            return v31
        case US4_4(v32, v33, v34, v35, v36, v37): # TerminalCall
            del v0
            v38 = method15(v32, v33, v34, v35, v36, v37)
            del v32, v33, v34, v35, v36, v37
            v39 = "TerminalCall"
            v40 = [v39,v38]
            del v38, v39
            return v40
        case US4_5(v41, v42, v43, v44, v45, v46): # TerminalFold
            del v0
            v47 = method15(v41, v42, v43, v44, v45, v46)
            del v41, v42, v43, v44, v45, v46
            v48 = "TerminalFold"
            v49 = [v48,v47]
            del v47, v48
            return v49
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method13(v0 : US3) -> object:
    match v0:
        case US3_0(): # None
            del v0
            v1 = method3()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US3_1(v4): # Some
            del v0
            v5 = method14(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method10(v0 : u32, v1 : US3) -> object:
    v2 = method11(v0)
    del v0
    v3 = method13(v1)
    del v1
    v4 = {'deck': v2, 'game': v3}
    del v2, v3
    return v4
def method25(v0 : i32, v1 : i32) -> bool:
    v2 = v1 < v0
    del v0, v1
    return v2
def method27(v0 : i32, v1 : US1) -> object:
    v2 = []
    v3 = method20(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method2(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method28(v0 : i32, v1 : US6) -> object:
    v2 = []
    v3 = method20(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method17(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method29(v0 : static_array, v1 : i32, v2 : i32) -> object:
    v3 = method19(v0)
    del v0
    v4 = method20(v1)
    del v1
    v5 = method20(v2)
    del v2
    v6 = {'cards_shown': v3, 'chips_won': v4, 'winner_id': v5}
    del v3, v4, v5
    return v6
def method26(v0 : US8) -> object:
    match v0:
        case US8_0(v1): # CommunityCardIs
            del v0
            v2 = method17(v1)
            del v1
            v3 = "CommunityCardIs"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US8_1(v5, v6): # PlayerAction
            del v0
            v7 = method27(v5, v6)
            del v5, v6
            v8 = "PlayerAction"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US8_2(v10, v11): # PlayerGotCard
            del v0
            v12 = method28(v10, v11)
            del v10, v11
            v13 = "PlayerGotCard"
            v14 = [v13,v12]
            del v12, v13
            return v14
        case US8_3(v15, v16, v17): # Showdown
            del v0
            v18 = method29(v15, v16, v17)
            del v15, v16, v17
            v19 = "Showdown"
            v20 = [v19,v18]
            del v18, v19
            return v20
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method24(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method25(v2, v3):
        v6 = v0[v3]
        v7 = method26(v6)
        del v6
        v1.append(v7)
        del v7
        v3 += 1 
    del v0, v2, v3
    return v1
def method30(v0 : US7) -> object:
    match v0:
        case US7_0(): # GameNotStarted
            del v0
            v1 = method3()
            v2 = "GameNotStarted"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US7_1(v4, v5, v6, v7, v8, v9): # GameOver
            del v0
            v10 = method15(v4, v5, v6, v7, v8, v9)
            del v4, v5, v6, v7, v8, v9
            v11 = "GameOver"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US7_2(v13, v14, v15, v16, v17, v18): # WaitingForActionFromPlayerId
            del v0
            v19 = method15(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "WaitingForActionFromPlayerId"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method23(v0 : static_array_list, v1 : static_array, v2 : US7) -> object:
    v3 = method24(v0)
    del v0
    v4 = method4(v1)
    del v1
    v5 = method30(v2)
    del v2
    v6 = {'messages': v3, 'pl_type': v4, 'ui_game_state': v5}
    del v3, v4, v5
    return v6
def method9(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7) -> object:
    v5 = method10(v0, v1)
    del v0, v1
    v6 = method23(v2, v3, v4)
    del v2, v3, v4
    v7 = {'private': v5, 'public': v6}
    del v5, v6
    return v7
def method36(v0 : cp.ndarray) -> object:
    v1 = v0
    del v0
    return v1
def method35(v0 : cp.ndarray) -> object:
    return method36(v0)
def method37(v0 : u64) -> object:
    v1 = v0
    del v0
    return v1
def method34(v0 : cp.ndarray, v1 : u64) -> object:
    v2 = []
    v3 = method35(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method37(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method33(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method34(v0, v1)
    del v0, v1
    v5 = method34(v2, v3)
    del v2, v3
    v6 = {'output': v4, 'param': v5}
    del v4, v5
    return v6
def method32(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    return method33(v0, v1, v2, v3)
def method31(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method32(v0, v1, v2, v3)
    del v0, v1, v2, v3
    v5 = {'model_data': v4}
    del v4
    return v5
def method8(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64) -> object:
    v9 = method9(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    v10 = method31(v5, v6, v7, v8)
    del v5, v6, v7, v8
    v11 = {'game': v9, 'neural': v10}
    del v9, v10
    return v11
def method7(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64) -> object:
    v9 = method8(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8
    return v9
def method41(v0 : object) -> None:
    assert v0 == [], f'Expected an unit type. Got: {v0}'
    del v0
    return 
def method40(v0 : object) -> US1:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "Call" == v1
    if v3:
        del v1, v3
        method41(v2)
        del v2
        return US1_0()
    else:
        del v3
        v5 = "Fold" == v1
        if v5:
            del v1, v5
            method41(v2)
            del v2
            return US1_1()
        else:
            del v5
            v7 = "Raise" == v1
            if v7:
                del v1, v7
                method41(v2)
                del v2
                return US1_2()
            else:
                del v2, v7
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method43(v0 : object) -> US2:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "Computer" == v1
    if v3:
        del v1, v3
        method41(v2)
        del v2
        return US2_0()
    else:
        del v3
        v5 = "Human" == v1
        if v5:
            del v1, v5
            method41(v2)
            del v2
            return US2_1()
        else:
            del v5
            v7 = "Random" == v1
            if v7:
                del v1, v7
                method41(v2)
                del v2
                return US2_2()
            else:
                del v2, v7
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
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
    while method25(v1, v7):
        v9 = v0[v7]
        v10 = method43(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method39(v0 : object) -> US0:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "ActionSelected" == v1
    if v3:
        del v1, v3
        v4 = method40(v2)
        del v2
        return US0_0(v4)
    else:
        del v3
        v6 = "PlayerChanged" == v1
        if v6:
            del v1, v6
            v7 = method42(v2)
            del v2
            return US0_1(v7)
        else:
            del v6
            v9 = "StartGame" == v1
            if v9:
                del v1, v9
                method41(v2)
                del v2
                return US0_2()
            else:
                del v9
                v11 = "StartTrainingVsRando" == v1
                if v11:
                    del v1, v11
                    method41(v2)
                    del v2
                    return US0_3()
                else:
                    del v11
                    v13 = "StartTrainingVsSelf" == v1
                    if v13:
                        del v1, v13
                        method41(v2)
                        del v2
                        return US0_4()
                    else:
                        del v2, v13
                        raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                        del v1
                        raise Exception("Error")
def method38(v0 : object) -> US0:
    return method39(v0)
def method49(v0 : object) -> u32:
    assert isinstance(v0,u32), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method48(v0 : object) -> u32:
    v1 = method49(v0)
    del v0
    return v1
def method54(v0 : object) -> US6:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "Jack" == v1
    if v3:
        del v1, v3
        method41(v2)
        del v2
        return US6_0()
    else:
        del v3
        v5 = "King" == v1
        if v5:
            del v1, v5
            method41(v2)
            del v2
            return US6_1()
        else:
            del v5
            v7 = "Queen" == v1
            if v7:
                del v1, v7
                method41(v2)
                del v2
                return US6_2()
            else:
                del v2, v7
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method53(v0 : object) -> US5:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "None" == v1
    if v3:
        del v1, v3
        method41(v2)
        del v2
        return US5_0()
    else:
        del v3
        v5 = "Some" == v1
        if v5:
            del v1, v5
            v6 = method54(v2)
            del v2
            return US5_1(v6)
        else:
            del v2, v5
            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
            del v1
            raise Exception("Error")
def method55(v0 : object) -> bool:
    assert isinstance(v0,bool), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method56(v0 : object) -> static_array:
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
    while method25(v1, v7):
        v9 = v0[v7]
        v10 = method54(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method57(v0 : object) -> i32:
    assert isinstance(v0,i32), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method58(v0 : object) -> static_array:
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
    while method25(v1, v7):
        v9 = v0[v7]
        v10 = method57(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method52(v0 : object) -> Tuple[US5, bool, static_array, i32, static_array, i32]:
    v1 = v0["community_card"] # type: ignore
    v2 = method53(v1)
    del v1
    v3 = v0["is_button_s_first_move"] # type: ignore
    v4 = method55(v3)
    del v3
    v5 = v0["pl_card"] # type: ignore
    v6 = method56(v5)
    del v5
    v7 = v0["player_turn"] # type: ignore
    v8 = method57(v7)
    del v7
    v9 = v0["pot"] # type: ignore
    v10 = method58(v9)
    del v9
    v11 = v0["raises_left"] # type: ignore
    del v0
    v12 = method57(v11)
    del v11
    return v2, v4, v6, v8, v10, v12
def method59(v0 : object) -> Tuple[US5, bool, static_array, i32, static_array, i32, US1]:
    v1 = v0[0] # type: ignore
    v2, v3, v4, v5, v6, v7 = method52(v1)
    del v1
    v8 = v0[1] # type: ignore
    del v0
    v9 = method40(v8)
    del v8
    return v2, v3, v4, v5, v6, v7, v9
def method51(v0 : object) -> US4:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "ChanceCommunityCard" == v1
    if v3:
        del v1, v3
        v4, v5, v6, v7, v8, v9 = method52(v2)
        del v2
        return US4_0(v4, v5, v6, v7, v8, v9)
    else:
        del v3
        v11 = "ChanceInit" == v1
        if v11:
            del v1, v11
            method41(v2)
            del v2
            return US4_1()
        else:
            del v11
            v13 = "Round" == v1
            if v13:
                del v1, v13
                v14, v15, v16, v17, v18, v19 = method52(v2)
                del v2
                return US4_2(v14, v15, v16, v17, v18, v19)
            else:
                del v13
                v21 = "RoundWithAction" == v1
                if v21:
                    del v1, v21
                    v22, v23, v24, v25, v26, v27, v28 = method59(v2)
                    del v2
                    return US4_3(v22, v23, v24, v25, v26, v27, v28)
                else:
                    del v21
                    v30 = "TerminalCall" == v1
                    if v30:
                        del v1, v30
                        v31, v32, v33, v34, v35, v36 = method52(v2)
                        del v2
                        return US4_4(v31, v32, v33, v34, v35, v36)
                    else:
                        del v30
                        v38 = "TerminalFold" == v1
                        if v38:
                            del v1, v38
                            v39, v40, v41, v42, v43, v44 = method52(v2)
                            del v2
                            return US4_5(v39, v40, v41, v42, v43, v44)
                        else:
                            del v2, v38
                            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                            del v1
                            raise Exception("Error")
def method50(v0 : object) -> US3:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "None" == v1
    if v3:
        del v1, v3
        method41(v2)
        del v2
        return US3_0()
    else:
        del v3
        v5 = "Some" == v1
        if v5:
            del v1, v5
            v6 = method51(v2)
            del v2
            return US3_1(v6)
        else:
            del v2, v5
            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
            del v1
            raise Exception("Error")
def method47(v0 : object) -> Tuple[u32, US3]:
    v1 = v0["deck"] # type: ignore
    v2 = method48(v1)
    del v1
    v3 = v0["game"] # type: ignore
    del v0
    v4 = method50(v3)
    del v3
    return v2, v4
def method63(v0 : object) -> Tuple[i32, US1]:
    v1 = v0[0] # type: ignore
    v2 = method57(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method40(v3)
    del v3
    return v2, v4
def method64(v0 : object) -> Tuple[i32, US6]:
    v1 = v0[0] # type: ignore
    v2 = method57(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method54(v3)
    del v3
    return v2, v4
def method65(v0 : object) -> Tuple[static_array, i32, i32]:
    v1 = v0["cards_shown"] # type: ignore
    v2 = method56(v1)
    del v1
    v3 = v0["chips_won"] # type: ignore
    v4 = method57(v3)
    del v3
    v5 = v0["winner_id"] # type: ignore
    del v0
    v6 = method57(v5)
    del v5
    return v2, v4, v6
def method62(v0 : object) -> US8:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "CommunityCardIs" == v1
    if v3:
        del v1, v3
        v4 = method54(v2)
        del v2
        return US8_0(v4)
    else:
        del v3
        v6 = "PlayerAction" == v1
        if v6:
            del v1, v6
            v7, v8 = method63(v2)
            del v2
            return US8_1(v7, v8)
        else:
            del v6
            v10 = "PlayerGotCard" == v1
            if v10:
                del v1, v10
                v11, v12 = method64(v2)
                del v2
                return US8_2(v11, v12)
            else:
                del v10
                v14 = "Showdown" == v1
                if v14:
                    del v1, v14
                    v15, v16, v17 = method65(v2)
                    del v2
                    return US8_3(v15, v16, v17)
                else:
                    del v2, v14
                    raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                    del v1
                    raise Exception("Error")
def method61(v0 : object) -> static_array_list:
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
    while method25(v2, v8):
        v10 = v0[v8]
        v11 = method62(v10)
        del v10
        v7[v8] = v11
        del v11
        v8 += 1 
    del v0, v2, v8
    return v7
def method66(v0 : object) -> US7:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "GameNotStarted" == v1
    if v3:
        del v1, v3
        method41(v2)
        del v2
        return US7_0()
    else:
        del v3
        v5 = "GameOver" == v1
        if v5:
            del v1, v5
            v6, v7, v8, v9, v10, v11 = method52(v2)
            del v2
            return US7_1(v6, v7, v8, v9, v10, v11)
        else:
            del v5
            v13 = "WaitingForActionFromPlayerId" == v1
            if v13:
                del v1, v13
                v14, v15, v16, v17, v18, v19 = method52(v2)
                del v2
                return US7_2(v14, v15, v16, v17, v18, v19)
            else:
                del v2, v13
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method60(v0 : object) -> Tuple[static_array_list, static_array, US7]:
    v1 = v0["messages"] # type: ignore
    v2 = method61(v1)
    del v1
    v3 = v0["pl_type"] # type: ignore
    v4 = method42(v3)
    del v3
    v5 = v0["ui_game_state"] # type: ignore
    del v0
    v6 = method66(v5)
    del v5
    return v2, v4, v6
def method46(v0 : object) -> Tuple[u32, US3, static_array_list, static_array, US7]:
    v1 = v0["private"] # type: ignore
    v2, v3 = method47(v1)
    del v1
    v4 = v0["public"] # type: ignore
    del v0
    v5, v6, v7 = method60(v4)
    del v4
    return v2, v3, v5, v6, v7
def method72(v0 : object) -> cp.ndarray:
    assert isinstance(v0,cp.ndarray), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method71(v0 : object) -> cp.ndarray:
    v1 = method72(v0)
    del v0
    return v1
def method73(v0 : object) -> u64:
    assert isinstance(v0,u64), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method70(v0 : object) -> Tuple[cp.ndarray, u64]:
    v1 = v0[0] # type: ignore
    v2 = method71(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method73(v3)
    del v3
    return v2, v4
def method69(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["output"] # type: ignore
    v2, v3 = method70(v1)
    del v1
    v4 = v0["param"] # type: ignore
    del v0
    v5, v6 = method70(v4)
    del v4
    return v2, v3, v5, v6
def method68(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1, v2, v3, v4 = method69(v0)
    del v0
    return v1, v2, v3, v4
def method67(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["model_data"] # type: ignore
    del v0
    v2, v3, v4, v5 = method68(v1)
    del v1
    return v2, v3, v4, v5
def method45(v0 : object) -> Tuple[u32, US3, static_array_list, static_array, US7, cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["game"] # type: ignore
    v2, v3, v4, v5, v6 = method46(v1)
    del v1
    v7 = v0["neural"] # type: ignore
    del v0
    v8, v9, v10, v11 = method67(v7)
    del v7
    return v2, v3, v4, v5, v6, v8, v9, v10, v11
def method44(v0 : object) -> Tuple[u32, US3, static_array_list, static_array, US7, cp.ndarray, u64, cp.ndarray, u64]:
    return method45(v0)
def method75(v0 : cp.ndarray, v1 : u32) -> None:
    v3 = v0[0:].view(cp.uint32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method76(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[4:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method77(v0 : cp.ndarray) -> None:
    del v0
    return 
def method79(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[0:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method81(v0 : cp.ndarray, v1 : US6) -> None:
    v2 = v1.tag
    method79(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US6_0(): # Jack
            del v1
            return method77(v4)
        case US6_1(): # King
            del v1
            return method77(v4)
        case US6_2(): # Queen
            del v1
            return method77(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method80(v0 : cp.ndarray, v1 : US5, v2 : bool, v3 : static_array, v4 : i32, v5 : static_array, v6 : i32) -> None:
    v7 = v1.tag
    method79(v0, v7)
    del v7
    v9 = v0[4:].view(cp.uint8)
    match v1:
        case US5_0(): # None
            method77(v9)
        case US5_1(v10): # Some
            method81(v9, v10)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v1, v9
    v12 = v0[8:].view(cp.bool_)
    v12[0] = v2
    del v2, v12
    v13 = 0
    while method5(v13):
        v15 = u64(v13)
        v16 = v15 * 4
        del v15
        v17 = 12 + v16
        del v16
        v19 = v0[v17:].view(cp.uint8)
        del v17
        v21 = v3[v13]
        method81(v19, v21)
        del v19, v21
        v13 += 1 
    del v3, v13
    v23 = v0[20:].view(cp.int32)
    v23[0] = v4
    del v4, v23
    v24 = 0
    while method5(v24):
        v26 = u64(v24)
        v27 = v26 * 4
        del v26
        v28 = 24 + v27
        del v27
        v30 = v0[v28:].view(cp.uint8)
        del v28
        v32 = v5[v24]
        method79(v30, v32)
        del v30, v32
        v24 += 1 
    del v5, v24
    v34 = v0[32:].view(cp.int32)
    del v0
    v34[0] = v6
    del v6, v34
    return 
def method83(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[36:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method82(v0 : cp.ndarray, v1 : US5, v2 : bool, v3 : static_array, v4 : i32, v5 : static_array, v6 : i32, v7 : US1) -> None:
    v8 = v1.tag
    method79(v0, v8)
    del v8
    v10 = v0[4:].view(cp.uint8)
    match v1:
        case US5_0(): # None
            method77(v10)
        case US5_1(v11): # Some
            method81(v10, v11)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v1, v10
    v13 = v0[8:].view(cp.bool_)
    v13[0] = v2
    del v2, v13
    v14 = 0
    while method5(v14):
        v16 = u64(v14)
        v17 = v16 * 4
        del v16
        v18 = 12 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v22 = v3[v14]
        method81(v20, v22)
        del v20, v22
        v14 += 1 
    del v3, v14
    v24 = v0[20:].view(cp.int32)
    v24[0] = v4
    del v4, v24
    v25 = 0
    while method5(v25):
        v27 = u64(v25)
        v28 = v27 * 4
        del v27
        v29 = 24 + v28
        del v28
        v31 = v0[v29:].view(cp.uint8)
        del v29
        v33 = v5[v25]
        method79(v31, v33)
        del v31, v33
        v25 += 1 
    del v5, v25
    v35 = v0[32:].view(cp.int32)
    v35[0] = v6
    del v6, v35
    v36 = v7.tag
    method83(v0, v36)
    del v36
    v38 = v0[40:].view(cp.uint8)
    del v0
    match v7:
        case US1_0(): # Call
            del v7
            return method77(v38)
        case US1_1(): # Fold
            del v7
            return method77(v38)
        case US1_2(): # Raise
            del v7
            return method77(v38)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method78(v0 : cp.ndarray, v1 : US4) -> None:
    v2 = v1.tag
    method79(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US4_0(v5, v6, v7, v8, v9, v10): # ChanceCommunityCard
            del v1
            return method80(v4, v5, v6, v7, v8, v9, v10)
        case US4_1(): # ChanceInit
            del v1
            return method77(v4)
        case US4_2(v11, v12, v13, v14, v15, v16): # Round
            del v1
            return method80(v4, v11, v12, v13, v14, v15, v16)
        case US4_3(v17, v18, v19, v20, v21, v22, v23): # RoundWithAction
            del v1
            return method82(v4, v17, v18, v19, v20, v21, v22, v23)
        case US4_4(v24, v25, v26, v27, v28, v29): # TerminalCall
            del v1
            return method80(v4, v24, v25, v26, v27, v28, v29)
        case US4_5(v30, v31, v32, v33, v34, v35): # TerminalFold
            del v1
            return method80(v4, v30, v31, v32, v33, v34, v35)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method84(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[80:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method86(v0 : cp.ndarray, v1 : i32, v2 : US1) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = v2.tag
    method76(v0, v5)
    del v5
    v7 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US1_0(): # Call
            del v2
            return method77(v7)
        case US1_1(): # Fold
            del v2
            return method77(v7)
        case US1_2(): # Raise
            del v2
            return method77(v7)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method87(v0 : cp.ndarray, v1 : i32, v2 : US6) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = v2.tag
    method76(v0, v5)
    del v5
    v7 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US6_0(): # Jack
            del v2
            return method77(v7)
        case US6_1(): # King
            del v2
            return method77(v7)
        case US6_2(): # Queen
            del v2
            return method77(v7)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method88(v0 : cp.ndarray, v1 : static_array, v2 : i32, v3 : i32) -> None:
    v4 = 0
    while method5(v4):
        v6 = u64(v4)
        v7 = v6 * 4
        del v6
        v9 = v0[v7:].view(cp.uint8)
        del v7
        v11 = v1[v4]
        method81(v9, v11)
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
def method85(v0 : cp.ndarray, v1 : US8) -> None:
    v2 = v1.tag
    method79(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US8_0(v5): # CommunityCardIs
            del v1
            return method81(v4, v5)
        case US8_1(v6, v7): # PlayerAction
            del v1
            return method86(v4, v6, v7)
        case US8_2(v8, v9): # PlayerGotCard
            del v1
            return method87(v4, v8, v9)
        case US8_3(v10, v11, v12): # Showdown
            del v1
            return method88(v4, v10, v11, v12)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method89(v0 : cp.ndarray, v1 : US2) -> None:
    v2 = v1.tag
    method79(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US2_0(): # Computer
            del v1
            return method77(v4)
        case US2_1(): # Human
            del v1
            return method77(v4)
        case US2_2(): # Random
            del v1
            return method77(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method90(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[1128:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method74(v0 : cp.ndarray, v1 : u32, v2 : US3, v3 : static_array_list, v4 : static_array, v5 : US7) -> None:
    method75(v0, v1)
    del v1
    v6 = v2.tag
    method76(v0, v6)
    del v6
    v8 = v0[16:].view(cp.uint8)
    match v2:
        case US3_0(): # None
            method77(v8)
        case US3_1(v9): # Some
            method78(v8, v9)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v2, v8
    v10 = v3.length
    method84(v0, v10)
    del v10
    v11 = v3.length
    v12 = 0
    while method25(v11, v12):
        v14 = u64(v12)
        v15 = v14 * 32
        del v14
        v16 = 96 + v15
        del v15
        v18 = v0[v16:].view(cp.uint8)
        del v16
        v20 = v3[v12]
        method85(v18, v20)
        del v18, v20
        v12 += 1 
    del v3, v11, v12
    v21 = 0
    while method5(v21):
        v23 = u64(v21)
        v24 = v23 * 4
        del v23
        v25 = 1120 + v24
        del v24
        v27 = v0[v25:].view(cp.uint8)
        del v25
        v29 = v4[v21]
        method89(v27, v29)
        del v27, v29
        v21 += 1 
    del v4, v21
    v30 = v5.tag
    method90(v0, v30)
    del v30
    v32 = v0[1136:].view(cp.uint8)
    del v0
    match v5:
        case US7_0(): # GameNotStarted
            del v5
            return method77(v32)
        case US7_1(v33, v34, v35, v36, v37, v38): # GameOver
            del v5
            return method80(v32, v33, v34, v35, v36, v37, v38)
        case US7_2(v39, v40, v41, v42, v43, v44): # WaitingForActionFromPlayerId
            del v5
            return method80(v32, v39, v40, v41, v42, v43, v44)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method91(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method93(v0 : cp.ndarray) -> u32:
    v2 = v0[0:].view(cp.uint32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method94(v0 : cp.ndarray) -> i32:
    v2 = v0[4:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method95(v0 : cp.ndarray) -> None:
    del v0
    return 
def method97(v0 : cp.ndarray) -> i32:
    v2 = v0[0:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method99(v0 : cp.ndarray) -> US6:
    v1 = method97(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method95(v3)
        del v3
        return US6_0()
    elif v1 == 1:
        del v1
        method95(v3)
        del v3
        return US6_1()
    elif v1 == 2:
        del v1
        method95(v3)
        del v3
        return US6_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method98(v0 : cp.ndarray) -> Tuple[US5, bool, static_array, i32, static_array, i32]:
    v1 = method97(v0)
    v3 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method95(v3)
        v8 = US5_0()
    elif v1 == 1:
        v6 = method99(v3)
        v8 = US5_1(v6)
    else:
        raise Exception("Invalid tag.")
    del v1, v3
    v10 = v0[8:].view(cp.bool_)
    v11 = v10[0].item()
    del v10
    v13 = static_array(2)
    v14 = 0
    while method5(v14):
        v16 = u64(v14)
        v17 = v16 * 4
        del v16
        v18 = 12 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v21 = method99(v20)
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
    while method5(v27):
        v29 = u64(v27)
        v30 = v29 * 4
        del v29
        v31 = 24 + v30
        del v30
        v33 = v0[v31:].view(cp.uint8)
        del v31
        v34 = method97(v33)
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
def method101(v0 : cp.ndarray) -> i32:
    v2 = v0[36:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method100(v0 : cp.ndarray) -> Tuple[US5, bool, static_array, i32, static_array, i32, US1]:
    v1 = method97(v0)
    v3 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method95(v3)
        v8 = US5_0()
    elif v1 == 1:
        v6 = method99(v3)
        v8 = US5_1(v6)
    else:
        raise Exception("Invalid tag.")
    del v1, v3
    v10 = v0[8:].view(cp.bool_)
    v11 = v10[0].item()
    del v10
    v13 = static_array(2)
    v14 = 0
    while method5(v14):
        v16 = u64(v14)
        v17 = v16 * 4
        del v16
        v18 = 12 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v21 = method99(v20)
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
    while method5(v27):
        v29 = u64(v27)
        v30 = v29 * 4
        del v29
        v31 = 24 + v30
        del v30
        v33 = v0[v31:].view(cp.uint8)
        del v31
        v34 = method97(v33)
        del v33
        v26[v27] = v34
        del v34
        v27 += 1 
    del v27
    v36 = v0[32:].view(cp.int32)
    v37 = v36[0].item()
    del v36
    v38 = method101(v0)
    v40 = v0[40:].view(cp.uint8)
    del v0
    if v38 == 0:
        method95(v40)
        v45 = US1_0()
    elif v38 == 1:
        method95(v40)
        v45 = US1_1()
    elif v38 == 2:
        method95(v40)
        v45 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v38, v40
    return v8, v11, v13, v24, v26, v37, v45
def method96(v0 : cp.ndarray) -> US4:
    v1 = method97(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5, v6, v7, v8, v9, v10 = method98(v3)
        del v3
        return US4_0(v5, v6, v7, v8, v9, v10)
    elif v1 == 1:
        del v1
        method95(v3)
        del v3
        return US4_1()
    elif v1 == 2:
        del v1
        v13, v14, v15, v16, v17, v18 = method98(v3)
        del v3
        return US4_2(v13, v14, v15, v16, v17, v18)
    elif v1 == 3:
        del v1
        v20, v21, v22, v23, v24, v25, v26 = method100(v3)
        del v3
        return US4_3(v20, v21, v22, v23, v24, v25, v26)
    elif v1 == 4:
        del v1
        v28, v29, v30, v31, v32, v33 = method98(v3)
        del v3
        return US4_4(v28, v29, v30, v31, v32, v33)
    elif v1 == 5:
        del v1
        v35, v36, v37, v38, v39, v40 = method98(v3)
        del v3
        return US4_5(v35, v36, v37, v38, v39, v40)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method102(v0 : cp.ndarray) -> i32:
    v2 = v0[80:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method104(v0 : cp.ndarray) -> Tuple[i32, US1]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method94(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method95(v6)
        v11 = US1_0()
    elif v4 == 1:
        method95(v6)
        v11 = US1_1()
    elif v4 == 2:
        method95(v6)
        v11 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v11
def method105(v0 : cp.ndarray) -> Tuple[i32, US6]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method94(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method95(v6)
        v11 = US6_0()
    elif v4 == 1:
        method95(v6)
        v11 = US6_1()
    elif v4 == 2:
        method95(v6)
        v11 = US6_2()
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v11
def method106(v0 : cp.ndarray) -> Tuple[static_array, i32, i32]:
    v2 = static_array(2)
    v3 = 0
    while method5(v3):
        v5 = u64(v3)
        v6 = v5 * 4
        del v5
        v8 = v0[v6:].view(cp.uint8)
        del v6
        v9 = method99(v8)
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
def method103(v0 : cp.ndarray) -> US8:
    v1 = method97(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5 = method99(v3)
        del v3
        return US8_0(v5)
    elif v1 == 1:
        del v1
        v7, v8 = method104(v3)
        del v3
        return US8_1(v7, v8)
    elif v1 == 2:
        del v1
        v10, v11 = method105(v3)
        del v3
        return US8_2(v10, v11)
    elif v1 == 3:
        del v1
        v13, v14, v15 = method106(v3)
        del v3
        return US8_3(v13, v14, v15)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method107(v0 : cp.ndarray) -> US2:
    v1 = method97(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method95(v3)
        del v3
        return US2_0()
    elif v1 == 1:
        del v1
        method95(v3)
        del v3
        return US2_1()
    elif v1 == 2:
        del v1
        method95(v3)
        del v3
        return US2_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method108(v0 : cp.ndarray) -> i32:
    v2 = v0[1128:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method92(v0 : cp.ndarray) -> Tuple[u32, US3, static_array_list, static_array, US7]:
    v1 = method93(v0)
    v2 = method94(v0)
    v4 = v0[16:].view(cp.uint8)
    if v2 == 0:
        method95(v4)
        v9 = US3_0()
    elif v2 == 1:
        v7 = method96(v4)
        v9 = US3_1(v7)
    else:
        raise Exception("Invalid tag.")
    del v2, v4
    v11 = static_array_list(32)
    v12 = method102(v0)
    v11.unsafe_set_length(v12)
    del v12
    v13 = v11.length
    v14 = 0
    while method25(v13, v14):
        v16 = u64(v14)
        v17 = v16 * 32
        del v16
        v18 = 96 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v21 = method103(v20)
        del v20
        v11[v14] = v21
        del v21
        v14 += 1 
    del v13, v14
    v23 = static_array(2)
    v24 = 0
    while method5(v24):
        v26 = u64(v24)
        v27 = v26 * 4
        del v26
        v28 = 1120 + v27
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
    v34 = v0[1136:].view(cp.uint8)
    del v0
    if v32 == 0:
        method95(v34)
        v51 = US7_0()
    elif v32 == 1:
        v37, v38, v39, v40, v41, v42 = method98(v34)
        v51 = US7_1(v37, v38, v39, v40, v41, v42)
    elif v32 == 2:
        v44, v45, v46, v47, v48, v49 = method98(v34)
        v51 = US7_2(v44, v45, v46, v47, v48, v49)
    else:
        raise Exception("Invalid tag.")
    del v32, v34
    return v1, v9, v11, v23, v51
def method115(v0 : f32) -> object:
    v1 = v0
    del v0
    return v1
def method114(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method25(v2, v3):
        v5 = v0[v3]
        v6 = method115(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method113(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method25(v2, v3):
        v5 = v0[v3]
        v6 = method114(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method112(v0 : US9) -> object:
    match v0:
        case US9_0(v1): # AddRewardsRando
            del v0
            v2 = method113(v1)
            del v1
            v3 = "AddRewardsRando"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US9_1(v5): # AddRewardsSelf
            del v0
            v6 = method113(v5)
            del v5
            v7 = "AddRewardsSelf"
            v8 = [v7,v6]
            del v6, v7
            return v8
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method111(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method25(v2, v3):
        v5 = v0[v3]
        v6 = method112(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method110(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64, v9 : list) -> object:
    v10 = []
    v11 = method8(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8
    v10.append(v11)
    del v11
    v12 = method111(v9)
    del v9
    v10.append(v12)
    del v12
    v13 = v10
    del v10
    return v13
def method109(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64, v9 : list) -> object:
    v10 = method110(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9
    return v10
def main_body():
    v0 = US0_3()
    v1 = method0(v0)
    del v0
    v3 = static_array(2)
    v5 = US2_0()
    v3[0] = v5
    del v5
    v7 = US2_1()
    v3[1] = v7
    del v7
    v9 = static_array_list(32)
    v10 = cp.empty(2981904,dtype=cp.uint8)
    v11 = cp.empty(25264128,dtype=cp.uint8)
    v13 = v10[0:0+4*65536].view(cp.float32)
    v14 = cp.random.normal(0.0,0.00390625,65536,dtype=cp.float32) # type: ignore
    cp.copyto(v13[0:0+65536],v14[0:0+65536])
    del v13, v14
    v16 = v10[262144:262144+4*1].view(cp.int32)
    v18 = v10[262160:262160+4*65536].view(cp.float32)
    v20 = v10[524304:524304+4*65536].view(cp.float32)
    v22 = v10[786448:786448+4*65536].view(cp.float32)
    v24 = v10[1048592:1048592+4*65536].view(cp.float32)
    v26 = v10[1310736:1310736+4*65536].view(cp.float32)
    v28 = v10[1572880:1572880+4*65536].view(cp.float32)
    v30 = v10[1835024:1835024+4*65536].view(cp.float32)
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
    v32 = v10[2097168:2097168+8*49152].view(cp.float64)
    v34 = v10[2490384:2490384+8*49152].view(cp.float64)
    v36 = v10[2883600:2883600+4*24576].view(cp.int32)
    v32[:] = 0
    del v32
    v34[:] = 0
    del v34
    v36[:] = 0
    del v36
    v37 = 63
    v38 = US3_0()
    v39 = US7_0()
    v40 = 25264128
    v41 = 2981904
    v42 = method7(v37, v38, v9, v3, v39, v11, v40, v10, v41)
    del v3, v9, v10, v11, v37, v38, v39, v40, v41
    v43 = method38(v1)
    del v1
    v44, v45, v46, v47, v48, v49, v50, v51, v52 = method44(v42)
    del v42
    v53 = cp.empty(16,dtype=cp.uint8)
    del v53
    v54 = cp.empty(1184,dtype=cp.uint8)
    method74(v54, v44, v45, v46, v47, v48)
    del v44, v45, v46, v47, v48
    v57 = "{}\n"
    v58 = "Going to run the Leduc full kernel."
    print(v57.format(v58),end="")
    del v57, v58
    v59 = time.perf_counter()
    v60 = []
    match v43:
        case US0_3(): # StartTrainingVsRando
            v61 = cp.zeros(16,dtype=cp.float32) # type: ignore
            v62 = cp.zeros(16,dtype=cp.float32) # type: ignore
            v63 = cp.empty(16,dtype=cp.float32)
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
            v68 = 0
            v69 = raw_module.get_function(f"entry{v68}")
            del v68
            v69.max_dynamic_shared_size_bytes = 98304 
            print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
            v69((24,),(256,),(v49, v50, v51, v52, v61, v62, v63),shared_mem=98304)
            del v61, v62, v69
            v70 = []
            v72 = v63[0:]
            del v63
            v73 = v72.get()
            del v72
            v74 = 0
            while method91(v74):
                v76 = []
                v77 = 0
                while method91(v77):
                    assert 0 <= v74 < 4, 'Tensor range check'
                    assert 0 <= v77 < 4, 'Tensor range check'
                    v79 = 4 * v74
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
            v82 = US9_0(v70)
            del v70
            v60.append(v82)
            del v82
        case t:
            raise Exception("Temporarily disabled for the fast_compile bug report.")
    del v43
    cp.cuda.get_current_stream().synchronize()
    v83 = time.perf_counter()
    v86 = "{}"
    v87 = "The time it took to run the kernel (in seconds) is: "
    print(v86.format(v87),end="")
    del v86, v87
    v88 = v83 - v59
    del v59, v83
    v91 = "{:.6f}\n"
    print(v91.format(v88),end="")
    del v88, v91
    v92, v93, v94, v95, v96 = method92(v54)
    del v54
    return method109(v92, v93, v94, v95, v96, v49, v50, v51, v52, v60)

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
