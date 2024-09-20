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
__device__ void method_0(unsigned char * v0, unsigned char * v1, StackMut0 & v2, Union4 v3);
struct Union0_0 { // T_Computer
};
struct Union0_1 { // T_Random
};
struct Union0 {
    union {
        Union0_0 case0; // T_Computer
        Union0_1 case1; // T_Random
    };
    unsigned char tag{255};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // T_Computer
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // T_Random
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // T_Computer
            case 1: new (&this->case1) Union0_1(x.case1); break; // T_Random
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // T_Computer
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // T_Random
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // T_Computer
                case 1: this->case1 = x.case1; break; // T_Random
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
                case 0: this->case0 = std::move(x.case0); break; // T_Computer
                case 1: this->case1 = std::move(x.case1); break; // T_Random
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // T_Computer
            case 1: this->case1.~Union0_1(); break; // T_Random
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
    v1 = v0 < 16;
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
    v1 = v0 < 8;
    return v1;
}
__device__ inline bool while_method_8(int v0){
    bool v1;
    v1 = v0 < 2;
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
__device__ void method_0(unsigned char * v0, unsigned char * v1, StackMut0 & v2, Union4 v3){
    static_array_list<Union1,32> & v4 = v2.v2;
    Union6 v5;
    v5 = Union6{Union6_1{v3}};
    Union6 v6;
    v6 = v5;
    while (while_method_3(v6)){
        Union6 v1018;
        switch (v6.tag) {
            case 0: { // None
                v1018 = Union6{Union6_0{}};
                break;
            }
            case 1: { // Some
                Union4 v8 = v6.case1.v0;
                switch (v8.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v963 = v8.case0.v0; bool v964 = v8.case0.v1; static_array<Union2,2> v965 = v8.case0.v2; int v966 = v8.case0.v3; static_array<int,2> v967 = v8.case0.v4; int v968 = v8.case0.v5;
                        curandStatePhilox4_32_10_t & v969 = v2.v5;
                        curandStatePhilox4_32_10_t & v970 = v969;
                        unsigned int & v971 = v2.v0;
                        Union2 v972; unsigned int v973;
                        Tuple0 tmp0 = draw_card_1(v970, v971);
                        v972 = tmp0.v0; v973 = tmp0.v1;
                        v2.v0 = v973;
                        Union1 v974;
                        v974 = Union1{Union1_0{v972}};
                        v4.push(v974);
                        int v975;
                        v975 = 2;
                        int v976; int v977;
                        Tuple1 tmp1 = Tuple1{0, 0};
                        v976 = tmp1.v0; v977 = tmp1.v1;
                        while (while_method_2(v976)){
                            int v979;
                            v979 = v967[v976];
                            bool v981;
                            v981 = v977 >= v979;
                            int v982;
                            if (v981){
                                v982 = v977;
                            } else {
                                v982 = v979;
                            }
                            v977 = v982;
                            v976 += 1 ;
                        }
                        static_array<int,2> v983;
                        int v985;
                        v985 = 0;
                        while (while_method_2(v985)){
                            v983[v985] = v977;
                            v985 += 1 ;
                        }
                        Union5 v987;
                        v987 = Union5{Union5_1{v972}};
                        Union4 v988;
                        v988 = Union4{Union4_2{v987, true, v965, 0, v983, v975}};
                        v1018 = Union6{Union6_1{v988}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v990 = v2.v5;
                        curandStatePhilox4_32_10_t & v991 = v990;
                        unsigned int & v992 = v2.v0;
                        Union2 v993; unsigned int v994;
                        Tuple0 tmp2 = draw_card_1(v991, v992);
                        v993 = tmp2.v0; v994 = tmp2.v1;
                        v2.v0 = v994;
                        curandStatePhilox4_32_10_t & v995 = v2.v5;
                        curandStatePhilox4_32_10_t & v996 = v995;
                        unsigned int & v997 = v2.v0;
                        Union2 v998; unsigned int v999;
                        Tuple0 tmp3 = draw_card_1(v996, v997);
                        v998 = tmp3.v0; v999 = tmp3.v1;
                        v2.v0 = v999;
                        Union1 v1000;
                        v1000 = Union1{Union1_2{0, v993}};
                        v4.push(v1000);
                        Union1 v1001;
                        v1001 = Union1{Union1_2{1, v998}};
                        v4.push(v1001);
                        int v1002;
                        v1002 = 2;
                        static_array<int,2> v1003;
                        v1003[0] = 1;
                        v1003[1] = 1;
                        static_array<Union2,2> v1005;
                        v1005[0] = v993;
                        v1005[1] = v998;
                        Union5 v1007;
                        v1007 = Union5{Union5_0{}};
                        Union4 v1008;
                        v1008 = Union4{Union4_2{v1007, true, v1005, 0, v1003, v1002}};
                        v1018 = Union6{Union6_1{v1008}};
                        break;
                    }
                    case 2: { // Round
                        Union5 v51 = v8.case2.v0; bool v52 = v8.case2.v1; static_array<Union2,2> v53 = v8.case2.v2; int v54 = v8.case2.v3; static_array<int,2> v55 = v8.case2.v4; int v56 = v8.case2.v5;
                        static_array<Union0,2> & v57 = v2.v3;
                        Union0 v58;
                        v58 = v57[v54];
                        Union3 v779;
                        switch (v58.tag) {
                            case 0: { // T_Computer
                                static_array_list<Union1,32> & v60 = v2.v2;
                                curandStatePhilox4_32_10_t & v61 = v2.v5;
                                curandStatePhilox4_32_10_t & v62 = v61;
                                unsigned int * v63;
                                v63 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
                                float * v65;
                                v65 = reinterpret_cast<float *>(&v0[0ull]);
                                int v67;
                                v67 = threadIdx.x;
                                int v68;
                                v68 = blockIdx.x;
                                int v69;
                                v69 = v68 * 256;
                                int v70;
                                v70 = v67 + v69;
                                unsigned long long v71;
                                v71 = (unsigned long long)v70;
                                curandStatePhilox4_32_10_t v72;
                                curand_init(12344321ull,v71,0ull,&v72);
                                float * v73;
                                v73 = reinterpret_cast<float *>(&v0[0ull]);
                                int v75;
                                v75 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v75 && v75 < 24);
                                int v76;
                                v76 = 32768 * v75;
                                int v77;
                                v77 = threadIdx.x;
                                int v78;
                                v78 = blockIdx.x;
                                int v79;
                                v79 = v78 * 256;
                                int v80;
                                v80 = v77 + v79;
                                unsigned long long v81;
                                v81 = (unsigned long long)v80;
                                curandStatePhilox4_32_10_t v82;
                                curand_init(12344321ull,v81,0ull,&v82);
                                int v83;
                                v83 = threadIdx.x;
                                int v84;
                                v84 = v83;
                                while (while_method_4(v84)){
                                    bool v86;
                                    v86 = 0 <= v84;
                                    bool v87;
                                    v87 = v86 == false;
                                    if (v87){
                                        assert("The index needs to be zero or positive." && v86);
                                    } else {
                                    }
                                    int v89;
                                    v89 = v84 % 128;
                                    int v90;
                                    v90 = v84 / 128;
                                    bool v91;
                                    v91 = v90 < 256;
                                    bool v92;
                                    v92 = v91 == false;
                                    if (v92){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v91);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v90 && v90 < 256);
                                    assert("Tensor range check" && 0 <= v89 && v89 < 128);
                                    int v94;
                                    v94 = v89 + v76;
                                    int v95;
                                    v95 = 128 * v90;
                                    int v96;
                                    v96 = v95 + v94;
                                    v73[v96] = 0.0f;
                                    v84 += 256 ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                int v97;
                                v97 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v97 && v97 < 256);
                                int v98;
                                v98 = 128 * v97;
                                int v99;
                                v99 = v98 + v76;
                                static_array_list<Union7,10> v100;
                                v100 = static_array_list<Union7,10>{};
                                int v102;
                                v102 = v60.length;
                                int v103;
                                v103 = 0;
                                while (while_method_5(v102, v103)){
                                    Union1 v105;
                                    v105 = v60[v103];
                                    Union8 v124;
                                    switch (v105.tag) {
                                        case 0: { // CommunityCardIs
                                            Union2 v114 = v105.case0.v0;
                                            Union7 v115;
                                            v115 = Union7{Union7_1{v114}};
                                            v124 = Union8{Union8_1{v115}};
                                            break;
                                        }
                                        case 1: { // PlayerAction
                                            int v117 = v105.case1.v0; Union3 v118 = v105.case1.v1;
                                            Union7 v119;
                                            v119 = Union7{Union7_0{v118}};
                                            v124 = Union8{Union8_1{v119}};
                                            break;
                                        }
                                        case 2: { // PlayerGotCard
                                            int v107 = v105.case2.v0; Union2 v108 = v105.case2.v1;
                                            bool v109;
                                            v109 = v107 == v54;
                                            if (v109){
                                                Union7 v110;
                                                v110 = Union7{Union7_1{v108}};
                                                v124 = Union8{Union8_1{v110}};
                                            } else {
                                                v124 = Union8{Union8_0{}};
                                            }
                                            break;
                                        }
                                        default: {
                                            v124 = Union8{Union8_0{}};
                                        }
                                    }
                                    switch (v124.tag) {
                                        case 0: { // None
                                            break;
                                        }
                                        case 1: { // Some
                                            Union7 v125 = v124.case1.v0;
                                            v100.push(v125);
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false); __trap();
                                        }
                                    }
                                    v103 += 1 ;
                                }
                                float * v126;
                                v126 = v73+v99;
                                int v128;
                                v128 = v100.length;
                                bool v129;
                                v129 = v128 == 0;
                                if (v129){
                                    v126[0] = 1.0f;
                                } else {
                                }
                                int v130;
                                v130 = v100.length;
                                int v131;
                                v131 = 0;
                                while (while_method_5(v130, v131)){
                                    Union7 v133;
                                    v133 = v100[v131];
                                    int v135;
                                    v135 = v131 * 6;
                                    int v136;
                                    v136 = 1 + v135;
                                    switch (v133.tag) {
                                        case 0: { // C1of2
                                            Union3 v137 = v133.case0.v0;
                                            switch (v137.tag) {
                                                case 0: { // Call
                                                    v126[v136] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // Fold
                                                    int v138;
                                                    v138 = v136 + 1;
                                                    v126[v138] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // Raise
                                                    int v139;
                                                    v139 = v136 + 2;
                                                    v126[v139] = 1.0f;
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false); __trap();
                                                }
                                            }
                                            break;
                                        }
                                        case 1: { // C2of2
                                            Union2 v140 = v133.case1.v0;
                                            int v141;
                                            v141 = v136 + 3;
                                            switch (v140.tag) {
                                                case 0: { // Jack
                                                    v126[v141] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // King
                                                    int v142;
                                                    v142 = v141 + 1;
                                                    v126[v142] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // Queen
                                                    int v143;
                                                    v143 = v141 + 2;
                                                    v126[v143] = 1.0f;
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
                                    v131 += 1 ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                int v144;
                                v144 = 0;
                                while (while_method_0(v144)){
                                    float * v146;
                                    v146 = reinterpret_cast<float *>(&v0[0ull]);
                                    float * v148;
                                    v148 = reinterpret_cast<float *>(&v1[0ull]);
                                    assert("Tensor range check" && 0 <= v144 && v144 < 4);
                                    int v150;
                                    v150 = 16384 * v144;
                                    float * v151;
                                    v151 = reinterpret_cast<float *>(&v0[3145728ull]);
                                    int v153;
                                    v153 = blockIdx.x;
                                    assert("Tensor range check" && 0 <= v153 && v153 < 24);
                                    int v154;
                                    v154 = 32768 * v153;
                                    int v155;
                                    v155 = blockIdx.x;
                                    assert("Tensor range check" && 0 <= v155 && v155 < 24);
                                    int v156;
                                    v156 = 32768 * v155;
                                    cuda::pipeline<cuda::thread_scope_thread> v157 = cuda::make_pipeline();
                                    extern __shared__ unsigned char v158[];
                                    float * v159;
                                    v159 = reinterpret_cast<float *>(&v158[0ull]);
                                    float * v161;
                                    v161 = reinterpret_cast<float *>(&v158[34816ull]);
                                    float * v163;
                                    v163 = reinterpret_cast<float *>(&v158[0ull]);
                                    int v165;
                                    v165 = threadIdx.x;
                                    int v166;
                                    v166 = v165 / 32;
                                    bool v167;
                                    v167 = 0 <= v166;
                                    bool v168;
                                    v168 = v167 == false;
                                    if (v168){
                                        assert("The index needs to be zero or positive." && v167);
                                    } else {
                                    }
                                    int v170;
                                    v170 = v166 % 8;
                                    int v171;
                                    v171 = v166 / 8;
                                    bool v172;
                                    v172 = v171 < 1;
                                    bool v173;
                                    v173 = v172 == false;
                                    if (v173){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v172);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v171 && v171 < 1);
                                    assert("Tensor range check" && 0 <= v170 && v170 < 8);
                                    int v175;
                                    v175 = 16 * v170;
                                    int v176;
                                    v176 = 17408 * v171;
                                    int v177;
                                    v177 = v176 + v175;
                                    float * v178;
                                    v178 = v163+v177;
                                    assert("Tensor range check" && 0 <= v171 && v171 < 1);
                                    int v180;
                                    v180 = 8704 * v171;
                                    int v181;
                                    v181 = threadIdx.x;
                                    int v182;
                                    v182 = v181 % 32;
                                    bool v183;
                                    v183 = 0 <= v182;
                                    bool v184;
                                    v184 = v183 == false;
                                    if (v184){
                                        assert("The index needs to be zero or positive." && v183);
                                    } else {
                                    }
                                    int v186;
                                    v186 = v182 % 4;
                                    int v187;
                                    v187 = v182 / 4;
                                    bool v188;
                                    v188 = v187 < 8;
                                    bool v189;
                                    v189 = v188 == false;
                                    if (v189){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v188);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v187 && v187 < 8);
                                    assert("Tensor range check" && 0 <= v186 && v186 < 4);
                                    int v191;
                                    v191 = v186 + v180;
                                    int v192;
                                    v192 = 68 * v187;
                                    int v193;
                                    v193 = v192 + v191;
                                    float * v194;
                                    v194 = v159+v193;
                                    assert("Tensor range check" && 0 <= v170 && v170 < 8);
                                    int v196;
                                    v196 = 1088 * v170;
                                    int v197;
                                    v197 = threadIdx.x;
                                    int v198;
                                    v198 = v197 % 32;
                                    bool v199;
                                    v199 = 0 <= v198;
                                    bool v200;
                                    v200 = v199 == false;
                                    if (v200){
                                        assert("The index needs to be zero or positive." && v199);
                                    } else {
                                    }
                                    int v202;
                                    v202 = v198 % 4;
                                    int v203;
                                    v203 = v198 / 4;
                                    bool v204;
                                    v204 = v203 < 8;
                                    bool v205;
                                    v205 = v204 == false;
                                    if (v205){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v204);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v203 && v203 < 8);
                                    assert("Tensor range check" && 0 <= v202 && v202 < 4);
                                    int v207;
                                    v207 = v202 + v196;
                                    int v208;
                                    v208 = 68 * v203;
                                    int v209;
                                    v209 = v208 + v207;
                                    float * v210;
                                    v210 = v161+v209;
                                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> v212[8];
                                    int v213;
                                    v213 = 0;
                                    while (while_method_2(v213)){
                                        int v215;
                                        v215 = 0;
                                        while (while_method_6(v215)){
                                            assert("Tensor range check" && 0 <= v213 && v213 < 2);
                                            assert("Tensor range check" && 0 <= v215 && v215 < 1);
                                            int v217;
                                            v217 = 128 * v215;
                                            int v218;
                                            v218 = v217 + v156;
                                            int v219;
                                            v219 = 16384 * v213;
                                            int v220;
                                            v220 = v219 + v218;
                                            float * v221;
                                            v221 = v151+v220;
                                            // Pushing the loop unrolling to: 0
                                            int v223;
                                            v223 = 0;
                                            #pragma unroll
                                            while (while_method_7(v223)){
                                                int v225;
                                                v225 = 0;
                                                #pragma unroll
                                                while (while_method_6(v225)){
                                                    assert("Tensor range check" && 0 <= v223 && v223 < 8);
                                                    assert("Tensor range check" && 0 <= v225 && v225 < 1);
                                                    int v227;
                                                    v227 = v223 + v225;
                                                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v228 = v212[v227];
                                                    wmma::fill_fragment(v228, 0.0f);
                                                    v225 += 1 ;
                                                }
                                                v223 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            int v229;
                                            v229 = 0;
                                            while (while_method_8(v229)){
                                                int v231;
                                                v231 = v229 + 1;
                                                bool v232;
                                                v232 = v229 == 0;
                                                int v233;
                                                v233 = v229 % 2;
                                                bool v234;
                                                v234 = 0 <= v229;
                                                bool v235;
                                                v235 = v234 == false;
                                                if (v235){
                                                    assert("The index needs to be zero or positive." && v234);
                                                } else {
                                                }
                                                bool v237;
                                                v237 = v229 < 2;
                                                bool v238;
                                                v238 = v237 == false;
                                                if (v238){
                                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v237);
                                                } else {
                                                }
                                                bool v240;
                                                v240 = v231 < 2;
                                                Union9 v246;
                                                if (v240){
                                                    bool v241;
                                                    v241 = 0 <= v231;
                                                    bool v242;
                                                    v242 = v241 == false;
                                                    if (v242){
                                                        assert("The index needs to be zero or positive." && v241);
                                                    } else {
                                                    }
                                                    v246 = Union9{Union9_1{v231}};
                                                } else {
                                                    v246 = Union9{Union9_0{}};
                                                }
                                                assert("Tensor range check" && 0 <= v213 && v213 < 2);
                                                int v247;
                                                v247 = v219 + v154;
                                                assert("Tensor range check" && 0 <= v229 && v229 < 2);
                                                int v248;
                                                v248 = 64 * v229;
                                                int v249;
                                                v249 = v248 + v247;
                                                float * v250;
                                                v250 = v146+v249;
                                                assert("Tensor range check" && 0 <= v215 && v215 < 1);
                                                int v252;
                                                v252 = 16384 * v215;
                                                int v253;
                                                v253 = v252 + v150;
                                                if (v232){
                                                    assert("Tensor range check" && 0 <= v229 && v229 < 2);
                                                    int v254;
                                                    v254 = v248 + v253;
                                                    float * v255;
                                                    v255 = v148+v254;
                                                    // Pushing the loop unrolling to: 0
                                                    v157.producer_acquire();
                                                    int v257;
                                                    v257 = threadIdx.x;
                                                    bool v258;
                                                    v258 = 0 <= v257;
                                                    bool v259;
                                                    v259 = v258 == false;
                                                    if (v259){
                                                        assert("The index needs to be zero or positive." && v258);
                                                    } else {
                                                    }
                                                    int v261;
                                                    v261 = v257 % 16;
                                                    int v262;
                                                    v262 = v257 / 16;
                                                    bool v263;
                                                    v263 = v262 < 16;
                                                    bool v264;
                                                    v264 = v263 == false;
                                                    if (v264){
                                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v263);
                                                    } else {
                                                    }
                                                    assert("Tensor range check" && 0 <= v262 && v262 < 16);
                                                    assert("Tensor range check" && 0 <= v261 && v261 < 16);
                                                    int v266;
                                                    v266 = 4 * v261;
                                                    int v267;
                                                    v267 = 68 * v262;
                                                    int v268;
                                                    v268 = v267 + v266;
                                                    int v269;
                                                    v269 = 128 * v262;
                                                    int v270;
                                                    v270 = v269 + v266;
                                                    float * v271;
                                                    v271 = v161+v268;
                                                    float * v273;
                                                    v273 = v255+v270;
                                                    int v275;
                                                    v275 = 0;
                                                    #pragma unroll
                                                    while (while_method_7(v275)){
                                                        int v277;
                                                        v277 = 0;
                                                        #pragma unroll
                                                        while (while_method_6(v277)){
                                                            assert("Tensor range check" && 0 <= v275 && v275 < 8);
                                                            assert("Tensor range check" && 0 <= v277 && v277 < 1);
                                                            int v279;
                                                            v279 = 64 * v277;
                                                            int v280;
                                                            v280 = 1088 * v275;
                                                            int v281;
                                                            v281 = v280 + v279;
                                                            int v282;
                                                            v282 = 2048 * v275;
                                                            int v283;
                                                            v283 = v282 + v279;
                                                            constexpr int v284 = sizeof(float) * 4;
                                                            assert("Pointer alignment check" && (unsigned long long)(v273 + v283) % v284 == 0 && (unsigned long long)(v271 + v281) % v284 == 0);
                                                            cuda::memcpy_async(v271 + v281, v273 + v283, cuda::aligned_size_t<v284>(v284), v157);
                                                            v277 += 1 ;
                                                        }
                                                        v275 += 1 ;
                                                    }
                                                    v157.producer_commit();
                                                    // Poping the loop unrolling to: 0
                                                } else {
                                                }
                                                // Pushing the loop unrolling to: 0
                                                int v285;
                                                v285 = threadIdx.x;
                                                bool v286;
                                                v286 = 0 <= v285;
                                                bool v287;
                                                v287 = v286 == false;
                                                if (v287){
                                                    assert("The index needs to be zero or positive." && v286);
                                                } else {
                                                }
                                                int v289;
                                                v289 = v285 % 16;
                                                int v290;
                                                v290 = v285 / 16;
                                                bool v291;
                                                v291 = v290 < 16;
                                                bool v292;
                                                v292 = v291 == false;
                                                if (v292){
                                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v291);
                                                } else {
                                                }
                                                assert("Tensor range check" && 0 <= v290 && v290 < 16);
                                                assert("Tensor range check" && 0 <= v289 && v289 < 16);
                                                int v294;
                                                v294 = 4 * v289;
                                                int v295;
                                                v295 = 68 * v290;
                                                int v296;
                                                v296 = v295 + v294;
                                                int v297;
                                                v297 = 128 * v290;
                                                int v298;
                                                v298 = v297 + v294;
                                                float * v299;
                                                v299 = v159+v296;
                                                float * v301;
                                                v301 = v250+v298;
                                                int v303;
                                                v303 = 0;
                                                #pragma unroll
                                                while (while_method_7(v303)){
                                                    int v305;
                                                    v305 = 0;
                                                    #pragma unroll
                                                    while (while_method_6(v305)){
                                                        assert("Tensor range check" && 0 <= v303 && v303 < 8);
                                                        assert("Tensor range check" && 0 <= v305 && v305 < 1);
                                                        int v307;
                                                        v307 = 64 * v305;
                                                        int v308;
                                                        v308 = 1088 * v303;
                                                        int v309;
                                                        v309 = v308 + v307;
                                                        int v310;
                                                        v310 = 2048 * v303;
                                                        int v311;
                                                        v311 = v310 + v307;
                                                        int4* v312;
                                                        v312 = reinterpret_cast<int4*>(v301 + v311);
                                                        int4* v313;
                                                        v313 = reinterpret_cast<int4*>(v299 + v309);
                                                        assert("Pointer alignment check" && (unsigned long long)(v312) % 4 == 0 && (unsigned long long)(v313) % 4 == 0);
                                                        *v313 = *v312;
                                                        v305 += 1 ;
                                                    }
                                                    v303 += 1 ;
                                                }
                                                // Poping the loop unrolling to: 0
                                                wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v314[1];
                                                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v315[8];
                                                cuda::pipeline_consumer_wait_prior<0>(v157);;
                                                asm("barrier.cta.sync %0;" :: "r"(0));
                                                // Pushing the loop unrolling to: 0
                                                int v316;
                                                v316 = 0;
                                                #pragma unroll
                                                while (while_method_6(v316)){
                                                    int v318;
                                                    v318 = 0;
                                                    #pragma unroll
                                                    while (while_method_7(v318)){
                                                        assert("Tensor range check" && 0 <= v316 && v316 < 1);
                                                        assert("Tensor range check" && 0 <= v318 && v318 < 8);
                                                        int v320;
                                                        v320 = 8 * v316;
                                                        int v321;
                                                        v321 = v320 + v318;
                                                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v322 = v315[v321];
                                                        assert("Tensor range check" && 0 <= v316 && v316 < 1);
                                                        int v323;
                                                        v323 = 1088 * v316;
                                                        assert("Tensor range check" && 0 <= v318 && v318 < 8);
                                                        int v324;
                                                        v324 = 8 * v318;
                                                        int v325;
                                                        v325 = v324 + v323;
                                                        int v326;
                                                        v326 = 0;
                                                        #pragma unroll
                                                        while (while_method_2(v326)){
                                                            int v328;
                                                            v328 = 0;
                                                            #pragma unroll
                                                            while (while_method_2(v328)){
                                                                assert("Tensor range check" && 0 <= v326 && v326 < 2);
                                                                assert("Tensor range check" && 0 <= v328 && v328 < 2);
                                                                int v330;
                                                                v330 = 4 * v328;
                                                                int v331;
                                                                v331 = v330 + v325;
                                                                int v332;
                                                                v332 = 544 * v326;
                                                                int v333;
                                                                v333 = v332 + v331;
                                                                float v334;
                                                                v334 = v210[v333];
                                                                bool v335;
                                                                v335 = 0 <= v328;
                                                                bool v337;
                                                                if (v335){
                                                                    bool v336;
                                                                    v336 = v328 < 2;
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
                                                                bool v340;
                                                                v340 = 0 <= v326;
                                                                bool v342;
                                                                if (v340){
                                                                    bool v341;
                                                                    v341 = v326 < 2;
                                                                    v342 = v341;
                                                                } else {
                                                                    v342 = false;
                                                                }
                                                                bool v343;
                                                                v343 = v342 == false;
                                                                if (v343){
                                                                    assert("The indices should be inside the range of the dimension." && v342);
                                                                } else {
                                                                }
                                                                int v345;
                                                                v345 = v326 * 2;
                                                                int v346;
                                                                v346 = v328 + v345;
                                                                v322.x[v346] = wmma::__float_to_tf32(v334);
                                                                v328 += 1 ;
                                                            }
                                                            v326 += 1 ;
                                                        }
                                                        v318 += 1 ;
                                                    }
                                                    v316 += 1 ;
                                                }
                                                // Poping the loop unrolling to: 0
                                                v157.consumer_release();
                                                switch (v246.tag) {
                                                    case 0: { // None
                                                        break;
                                                    }
                                                    case 1: { // Some
                                                        int v347 = v246.case1.v0;
                                                        assert("Tensor range check" && 0 <= v347 && v347 < 2);
                                                        int v348;
                                                        v348 = 64 * v347;
                                                        int v349;
                                                        v349 = v348 + v253;
                                                        float * v350;
                                                        v350 = v148+v349;
                                                        asm("barrier.cta.sync %0;" :: "r"(0));
                                                        // Pushing the loop unrolling to: 0
                                                        v157.producer_acquire();
                                                        int v352;
                                                        v352 = threadIdx.x;
                                                        bool v353;
                                                        v353 = 0 <= v352;
                                                        bool v354;
                                                        v354 = v353 == false;
                                                        if (v354){
                                                            assert("The index needs to be zero or positive." && v353);
                                                        } else {
                                                        }
                                                        int v356;
                                                        v356 = v352 % 16;
                                                        int v357;
                                                        v357 = v352 / 16;
                                                        bool v358;
                                                        v358 = v357 < 16;
                                                        bool v359;
                                                        v359 = v358 == false;
                                                        if (v359){
                                                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v358);
                                                        } else {
                                                        }
                                                        assert("Tensor range check" && 0 <= v357 && v357 < 16);
                                                        assert("Tensor range check" && 0 <= v356 && v356 < 16);
                                                        int v361;
                                                        v361 = 4 * v356;
                                                        int v362;
                                                        v362 = 68 * v357;
                                                        int v363;
                                                        v363 = v362 + v361;
                                                        int v364;
                                                        v364 = 128 * v357;
                                                        int v365;
                                                        v365 = v364 + v361;
                                                        float * v366;
                                                        v366 = v161+v363;
                                                        float * v368;
                                                        v368 = v350+v365;
                                                        int v370;
                                                        v370 = 0;
                                                        #pragma unroll
                                                        while (while_method_7(v370)){
                                                            int v372;
                                                            v372 = 0;
                                                            #pragma unroll
                                                            while (while_method_6(v372)){
                                                                assert("Tensor range check" && 0 <= v370 && v370 < 8);
                                                                assert("Tensor range check" && 0 <= v372 && v372 < 1);
                                                                int v374;
                                                                v374 = 64 * v372;
                                                                int v375;
                                                                v375 = 1088 * v370;
                                                                int v376;
                                                                v376 = v375 + v374;
                                                                int v377;
                                                                v377 = 2048 * v370;
                                                                int v378;
                                                                v378 = v377 + v374;
                                                                constexpr int v379 = sizeof(float) * 4;
                                                                assert("Pointer alignment check" && (unsigned long long)(v368 + v378) % v379 == 0 && (unsigned long long)(v366 + v376) % v379 == 0);
                                                                cuda::memcpy_async(v366 + v376, v368 + v378, cuda::aligned_size_t<v379>(v379), v157);
                                                                v372 += 1 ;
                                                            }
                                                            v370 += 1 ;
                                                        }
                                                        v157.producer_commit();
                                                        // Poping the loop unrolling to: 0
                                                        break;
                                                    }
                                                    default: {
                                                        assert("Invalid tag." && false); __trap();
                                                    }
                                                }
                                                // Pushing the loop unrolling to: 0
                                                int v380;
                                                v380 = 0;
                                                #pragma unroll
                                                while (while_method_7(v380)){
                                                    int v382;
                                                    v382 = 0;
                                                    #pragma unroll
                                                    while (while_method_7(v382)){
                                                        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v384 = v314[0];
                                                        assert("Tensor range check" && 0 <= v380 && v380 < 8);
                                                        int v385;
                                                        v385 = 1088 * v380;
                                                        assert("Tensor range check" && 0 <= v382 && v382 < 8);
                                                        int v386;
                                                        v386 = 8 * v382;
                                                        int v387;
                                                        v387 = v386 + v385;
                                                        int v388;
                                                        v388 = 0;
                                                        #pragma unroll
                                                        while (while_method_2(v388)){
                                                            int v390;
                                                            v390 = 0;
                                                            #pragma unroll
                                                            while (while_method_2(v390)){
                                                                assert("Tensor range check" && 0 <= v388 && v388 < 2);
                                                                assert("Tensor range check" && 0 <= v390 && v390 < 2);
                                                                int v392;
                                                                v392 = 544 * v390;
                                                                int v393;
                                                                v393 = v392 + v387;
                                                                int v394;
                                                                v394 = 4 * v388;
                                                                int v395;
                                                                v395 = v394 + v393;
                                                                float v396;
                                                                v396 = v194[v395];
                                                                bool v397;
                                                                v397 = 0 <= v390;
                                                                bool v399;
                                                                if (v397){
                                                                    bool v398;
                                                                    v398 = v390 < 2;
                                                                    v399 = v398;
                                                                } else {
                                                                    v399 = false;
                                                                }
                                                                bool v400;
                                                                v400 = v399 == false;
                                                                if (v400){
                                                                    assert("The indices should be inside the range of the dimension." && v399);
                                                                } else {
                                                                }
                                                                bool v402;
                                                                v402 = 0 <= v388;
                                                                bool v404;
                                                                if (v402){
                                                                    bool v403;
                                                                    v403 = v388 < 2;
                                                                    v404 = v403;
                                                                } else {
                                                                    v404 = false;
                                                                }
                                                                bool v405;
                                                                v405 = v404 == false;
                                                                if (v405){
                                                                    assert("The indices should be inside the range of the dimension." && v404);
                                                                } else {
                                                                }
                                                                int v407;
                                                                v407 = v388 * 2;
                                                                int v408;
                                                                v408 = v390 + v407;
                                                                v384.x[v408] = wmma::__float_to_tf32(v396);
                                                                v390 += 1 ;
                                                            }
                                                            v388 += 1 ;
                                                        }
                                                        int v409;
                                                        v409 = 0;
                                                        #pragma unroll
                                                        while (while_method_6(v409)){
                                                            assert("Tensor range check" && 0 <= v380 && v380 < 8);
                                                            assert("Tensor range check" && 0 <= v409 && v409 < 1);
                                                            int v411;
                                                            v411 = v380 + v409;
                                                            wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v412 = v212[v411];
                                                            assert("Tensor range check" && 0 <= v409 && v409 < 1);
                                                            assert("Tensor range check" && 0 <= v382 && v382 < 8);
                                                            int v413;
                                                            v413 = 8 * v409;
                                                            int v414;
                                                            v414 = v413 + v382;
                                                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v415 = v315[v414];
                                                            wmma::mma_sync(v412, v384, v415, v412);
                                                            v409 += 1 ;
                                                        }
                                                        v382 += 1 ;
                                                    }
                                                    v380 += 1 ;
                                                }
                                                // Poping the loop unrolling to: 0
                                                asm("barrier.cta.sync %0;" :: "r"(0));
                                                v229 = v231;
                                            }
                                            // Pushing the loop unrolling to: 0
                                            int v416;
                                            v416 = 0;
                                            #pragma unroll
                                            while (while_method_7(v416)){
                                                int v418;
                                                v418 = 0;
                                                #pragma unroll
                                                while (while_method_6(v418)){
                                                    assert("Tensor range check" && 0 <= v416 && v416 < 8);
                                                    assert("Tensor range check" && 0 <= v418 && v418 < 1);
                                                    int v420;
                                                    v420 = v416 + v418;
                                                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v421 = v212[v420];
                                                    assert("Tensor range check" && 0 <= v416 && v416 < 8);
                                                    assert("Tensor range check" && 0 <= v418 && v418 < 1);
                                                    int v422;
                                                    v422 = 16 * v418;
                                                    int v423;
                                                    v423 = 2176 * v416;
                                                    int v424;
                                                    v424 = v423 + v422;
                                                    float * v425;
                                                    v425 = v178+v424;
                                                    wmma::store_matrix_sync(v425, v421, 136, wmma::mem_row_major);
                                                    v418 += 1 ;
                                                }
                                                v416 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            asm("barrier.cta.sync %0;" :: "r"(0));
                                            // Pushing the loop unrolling to: 0
                                            int v427;
                                            v427 = threadIdx.x;
                                            bool v428;
                                            v428 = 0 <= v427;
                                            bool v429;
                                            v429 = v428 == false;
                                            if (v429){
                                                assert("The index needs to be zero or positive." && v428);
                                            } else {
                                            }
                                            int v431;
                                            v431 = v427 % 32;
                                            int v432;
                                            v432 = v427 / 32;
                                            bool v433;
                                            v433 = v432 < 8;
                                            bool v434;
                                            v434 = v433 == false;
                                            if (v434){
                                                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v433);
                                            } else {
                                            }
                                            assert("Tensor range check" && 0 <= v432 && v432 < 8);
                                            assert("Tensor range check" && 0 <= v431 && v431 < 32);
                                            int v436;
                                            v436 = 4 * v431;
                                            int v437;
                                            v437 = 128 * v432;
                                            int v438;
                                            v438 = v437 + v436;
                                            int v439;
                                            v439 = 136 * v432;
                                            int v440;
                                            v440 = v439 + v436;
                                            float * v441;
                                            v441 = v221+v438;
                                            float * v443;
                                            v443 = v163+v440;
                                            int v445;
                                            v445 = 0;
                                            #pragma unroll
                                            while (while_method_1(v445)){
                                                int v447;
                                                v447 = 0;
                                                #pragma unroll
                                                while (while_method_6(v447)){
                                                    assert("Tensor range check" && 0 <= v445 && v445 < 16);
                                                    assert("Tensor range check" && 0 <= v447 && v447 < 1);
                                                    int v449;
                                                    v449 = 128 * v447;
                                                    int v450;
                                                    v450 = 1024 * v445;
                                                    int v451;
                                                    v451 = v450 + v449;
                                                    int v452;
                                                    v452 = 1088 * v445;
                                                    int v453;
                                                    v453 = v452 + v449;
                                                    int4* v454;
                                                    v454 = reinterpret_cast<int4*>(v443 + v453);
                                                    int4* v455;
                                                    v455 = reinterpret_cast<int4*>(v441 + v451);
                                                    assert("Pointer alignment check" && (unsigned long long)(v454) % 4 == 0 && (unsigned long long)(v455) % 4 == 0);
                                                    *v455 = *v454;
                                                    v447 += 1 ;
                                                }
                                                v445 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            asm("barrier.cta.sync %0;" :: "r"(0));
                                            v215 += 1 ;
                                        }
                                        v213 += 1 ;
                                    }
                                    unsigned int * v456;
                                    v456 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
                                    assert("Tensor range check" && 0 <= v144 && v144 < 4);
                                    int v458;
                                    v458 = 6144 * v144;
                                    method_3(v456, v458, v151);
                                    int * v459;
                                    v459 = reinterpret_cast<int *>(&v1[262144ull]);
                                    float * v461;
                                    v461 = reinterpret_cast<float *>(&v1[262160ull]);
                                    float * v463;
                                    v463 = reinterpret_cast<float *>(&v1[524304ull]);
                                    float * v465;
                                    v465 = reinterpret_cast<float *>(&v1[786448ull]);
                                    float * v467;
                                    v467 = reinterpret_cast<float *>(&v1[1048592ull]);
                                    float * v469;
                                    v469 = reinterpret_cast<float *>(&v1[1310736ull]);
                                    float * v471;
                                    v471 = reinterpret_cast<float *>(&v1[1572880ull]);
                                    float * v473;
                                    v473 = reinterpret_cast<float *>(&v1[1835024ull]);
                                    int * v475;
                                    v475 = reinterpret_cast<int *>(&v0[6389760ull]);
                                    float * v477;
                                    v477 = reinterpret_cast<float *>(&v0[7962624ull]);
                                    int * v479;
                                    v479 = reinterpret_cast<int *>(&v0[9535488ull]);
                                    int * v481;
                                    v481 = reinterpret_cast<int *>(&v0[11108352ull]);
                                    double * v483;
                                    v483 = reinterpret_cast<double *>(&v0[12681216ull]);
                                    double * v485;
                                    v485 = reinterpret_cast<double *>(&v0[18972672ull]);
                                    double * v487;
                                    v487 = reinterpret_cast<double *>(&v1[2097168ull]);
                                    double * v489;
                                    v489 = reinterpret_cast<double *>(&v1[2490384ull]);
                                    int * v491;
                                    v491 = reinterpret_cast<int *>(&v1[2883600ull]);
                                    v144 += 1 ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                int * v493;
                                v493 = reinterpret_cast<int *>(&v1[262144ull]);
                                float * v495;
                                v495 = reinterpret_cast<float *>(&v1[262160ull]);
                                float * v497;
                                v497 = reinterpret_cast<float *>(&v1[524304ull]);
                                float * v499;
                                v499 = reinterpret_cast<float *>(&v1[786448ull]);
                                float * v501;
                                v501 = reinterpret_cast<float *>(&v1[1048592ull]);
                                float * v503;
                                v503 = reinterpret_cast<float *>(&v1[1310736ull]);
                                float * v505;
                                v505 = reinterpret_cast<float *>(&v1[1572880ull]);
                                float * v507;
                                v507 = reinterpret_cast<float *>(&v1[1835024ull]);
                                int v509;
                                v509 = v493[0];
                                unsigned int * v510;
                                v510 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
                                int v512;
                                v512 = blockIdx.x;
                                int v513;
                                v513 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v509 && v509 < 4);
                                assert("Tensor range check" && 0 <= v512 && v512 < 24);
                                assert("Tensor range check" && 0 <= v513 && v513 < 256);
                                int v514;
                                v514 = 256 * v512;
                                int v515;
                                v515 = v514 + v513;
                                int v516;
                                v516 = 6144 * v509;
                                int v517;
                                v517 = v516 + v515;
                                unsigned int v518;
                                v518 = v510[v517];
                                int v519;
                                v519 = (int)v518;
                                float v520; int v521;
                                Tuple2 tmp14 = method_4(v62, v493, v495, v497, v499, v501, v503, v505, v507, v519, v509);
                                v520 = tmp14.v0; v521 = tmp14.v1;
                                extern __shared__ unsigned char v522[];
                                float * v523;
                                v523 = reinterpret_cast<float *>(&v522[0ull]);
                                int * v525;
                                v525 = reinterpret_cast<int *>(&v522[16ull]);
                                int v527;
                                v527 = threadIdx.x;
                                bool v528;
                                v528 = v527 == 0;
                                if (v528){
                                    v523[0] = v520;
                                    v525[0] = v521;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                float v529;
                                v529 = v523[0];
                                int v530;
                                v530 = v525[0];
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                double * v531;
                                v531 = reinterpret_cast<double *>(&v1[2097168ull]);
                                double * v533;
                                v533 = reinterpret_cast<double *>(&v1[2490384ull]);
                                int * v535;
                                v535 = reinterpret_cast<int *>(&v1[2883600ull]);
                                int * v537;
                                v537 = reinterpret_cast<int *>(&v0[6389760ull]);
                                float * v539;
                                v539 = reinterpret_cast<float *>(&v0[7962624ull]);
                                int * v541;
                                v541 = reinterpret_cast<int *>(&v0[9535488ull]);
                                int * v543;
                                v543 = reinterpret_cast<int *>(&v0[11108352ull]);
                                double * v545;
                                v545 = reinterpret_cast<double *>(&v0[12681216ull]);
                                double * v547;
                                v547 = reinterpret_cast<double *>(&v0[18972672ull]);
                                int v549;
                                v549 = threadIdx.x;
                                int v550;
                                v550 = blockIdx.x;
                                int v551;
                                v551 = v550 * 256;
                                int v552;
                                v552 = v549 + v551;
                                int v553;
                                v553 = 0;
                                while (while_method_0(v553)){
                                    unsigned int * v555;
                                    v555 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
                                    int v557;
                                    v557 = blockIdx.x;
                                    int v558;
                                    v558 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v553 && v553 < 4);
                                    assert("Tensor range check" && 0 <= v557 && v557 < 24);
                                    assert("Tensor range check" && 0 <= v558 && v558 < 256);
                                    int v559;
                                    v559 = 256 * v557;
                                    int v560;
                                    v560 = v559 + v558;
                                    int v561;
                                    v561 = 6144 * v553;
                                    int v562;
                                    v562 = v561 + v560;
                                    unsigned int v563;
                                    v563 = v555[v562];
                                    int v564;
                                    v564 = (int)v563;
                                    float v565;
                                    v565 = method_5(v493, v495, v497, v499, v501, v503, v505, v507, v564, v553, v530);
                                    assert("Tensor range check" && 0 <= v553 && v553 < 4);
                                    assert("Tensor range check" && 0 <= v552 && v552 < 6144);
                                    int v566;
                                    v566 = v561 + v552;
                                    int v567;
                                    v567 = v535[v566];
                                    int v568;
                                    v568 = v567 + 1;
                                    assert("Tensor range check" && 0 <= v553 && v553 < 4);
                                    assert("Tensor range check" && 0 <= v552 && v552 < 6144);
                                    v535[v566] = v568;
                                    assert("Tensor range check" && 0 <= v553 && v553 < 4);
                                    assert("Tensor range check" && 0 <= v567 && v567 < 16);
                                    assert("Tensor range check" && 0 <= v552 && v552 < 6144);
                                    int v569;
                                    v569 = 6144 * v567;
                                    int v570;
                                    v570 = v569 + v552;
                                    int v571;
                                    v571 = 98304 * v553;
                                    int v572;
                                    v572 = v571 + v570;
                                    v537[v572] = v530;
                                    v539[v572] = v529;
                                    v541[v572] = v54;
                                    v543[v572] = v564;
                                    assert("Tensor range check" && 0 <= v553 && v553 < 4);
                                    int v573;
                                    v573 = 12288 * v553;
                                    assert("Tensor range check" && 0 <= v552 && v552 < 6144);
                                    int v574;
                                    v574 = 2 * v552;
                                    int v575;
                                    v575 = v574 + v573;
                                    assert("Tensor range check" && 0 <= v553 && v553 < 4);
                                    int v576;
                                    v576 = 196608 * v553;
                                    assert("Tensor range check" && 0 <= v567 && v567 < 16);
                                    int v577;
                                    v577 = 12288 * v567;
                                    int v578;
                                    v578 = v577 + v576;
                                    assert("Tensor range check" && 0 <= v552 && v552 < 6144);
                                    int v579;
                                    v579 = v574 + v578;
                                    double * v580;
                                    v580 = v531+v575;
                                    double * v582;
                                    v582 = v533+v575;
                                    double * v584;
                                    v584 = v545+v579;
                                    double * v586;
                                    v586 = v547+v579;
                                    int v588;
                                    v588 = sizeof(double *);
                                    unsigned long long v589;
                                    v589 = (unsigned long long)v588;
                                    unsigned long long v590;
                                    v590 = 256ull * v589;
                                    unsigned long long v591;
                                    v591 = v590 + 16ull;
                                    unsigned long long v592;
                                    v592 = v591 - 1ull;
                                    unsigned long long v593;
                                    v593 = v592 % 16ull;
                                    unsigned long long v594;
                                    v594 = v592 - v593;
                                    unsigned long long v595;
                                    v595 = v594 + v590;
                                    unsigned long long v596;
                                    v596 = v595 + 16ull;
                                    unsigned long long v597;
                                    v597 = v596 - 1ull;
                                    unsigned long long v598;
                                    v598 = v597 % 16ull;
                                    unsigned long long v599;
                                    v599 = v597 - v598;
                                    unsigned long long v600;
                                    v600 = v599 + v590;
                                    unsigned long long v601;
                                    v601 = v600 + 16ull;
                                    unsigned long long v602;
                                    v602 = v601 - 1ull;
                                    unsigned long long v603;
                                    v603 = v602 % 16ull;
                                    unsigned long long v604;
                                    v604 = v602 - v603;
                                    unsigned long long v605;
                                    v605 = v604 + v590;
                                    bool v606;
                                    v606 = v605 <= 98304ull;
                                    bool v607;
                                    v607 = v606 == false;
                                    if (v607){
                                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v606);
                                    } else {
                                    }
                                    extern __shared__ unsigned char v609[];
                                    bool v610;
                                    v610 = v605 <= v605;
                                    bool v611;
                                    v611 = v610 == false;
                                    if (v611){
                                        assert("The length of the partition has to be less than or equal to the length of the base array." && v610);
                                    } else {
                                    }
                                    double * * v613;
                                    v613 = reinterpret_cast<double * *>(&v609[0ull]);
                                    double * * v615;
                                    v615 = reinterpret_cast<double * *>(&v609[v594]);
                                    double * * v617;
                                    v617 = reinterpret_cast<double * *>(&v609[v599]);
                                    double * * v619;
                                    v619 = reinterpret_cast<double * *>(&v609[v604]);
                                    int v621;
                                    v621 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v621 && v621 < 256);
                                    v613[v621] = v580;
                                    v615[v621] = v582;
                                    v617[v621] = v584;
                                    v619[v621] = v586;
                                    asm("barrier.cta.sync %0;" :: "r"(0));
                                    bool v622;
                                    v622 = 0 <= v621;
                                    bool v623;
                                    v623 = v622 == false;
                                    if (v623){
                                        assert("The index needs to be zero or positive." && v622);
                                    } else {
                                    }
                                    int v625;
                                    v625 = v621 % 1;
                                    bool v626;
                                    v626 = v621 < 256;
                                    bool v627;
                                    v627 = v626 == false;
                                    if (v627){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v626);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v621 && v621 < 256);
                                    int v629;
                                    v629 = 0;
                                    while (while_method_6(v629)){
                                        bool v631;
                                        v631 = v622 && v626;
                                        bool v632;
                                        v632 = v631 == false;
                                        if (v632){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v631);
                                        } else {
                                        }
                                        bool v634;
                                        v634 = 0 <= v629;
                                        bool v636;
                                        if (v634){
                                            bool v635;
                                            v635 = v629 < 1;
                                            v636 = v635;
                                        } else {
                                            v636 = false;
                                        }
                                        bool v637;
                                        v637 = v636 == false;
                                        if (v637){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v636);
                                        } else {
                                        }
                                        int v639;
                                        v639 = v629 * 256;
                                        int v640;
                                        v640 = v639 + v621;
                                        assert("Tensor range check" && 0 <= v629 && v629 < 1);
                                        int v641;
                                        v641 = 256 * v629;
                                        int v642;
                                        v642 = v641 + v621;
                                        double * v643;
                                        v643 = v613[v642];
                                        double * v644;
                                        v644 = v615[v642];
                                        double * v645;
                                        v645 = v617[v642];
                                        double * v646;
                                        v646 = v619[v642];
                                        int v647;
                                        v647 = blockIdx.x;
                                        int v648;
                                        v648 = v647 * 256;
                                        int v649;
                                        v649 = v648 + v640;
                                        assert("Tensor range check" && 0 <= v625 && v625 < 1);
                                        int v650;
                                        v650 = 2 * v625;
                                        double v651[2];
                                        double v652[2];
                                        int v653[2];
                                        int v654;
                                        v654 = 0;
                                        while (while_method_6(v654)){
                                            assert("Tensor range check" && 0 <= v654 && v654 < 1);
                                            int v656;
                                            v656 = 2 * v654;
                                            assert("Tensor range check" && 0 <= v654 && v654 < 1);
                                            int v657;
                                            v657 = v656 + v650;
                                            int4* v658;
                                            v658 = reinterpret_cast<int4*>(v643 + v657);
                                            int4* v659;
                                            v659 = reinterpret_cast<int4*>(v651 + v656);
                                            assert("Pointer alignment check" && (unsigned long long)(v658) % 2 == 0 && (unsigned long long)(v659) % 2 == 0);
                                            *v659 = *v658;
                                            int4* v660;
                                            v660 = reinterpret_cast<int4*>(v644 + v657);
                                            int4* v661;
                                            v661 = reinterpret_cast<int4*>(v652 + v656);
                                            assert("Pointer alignment check" && (unsigned long long)(v660) % 2 == 0 && (unsigned long long)(v661) % 2 == 0);
                                            *v661 = *v660;
                                            v654 += 1 ;
                                        }
                                        int v662;
                                        v662 = 0;
                                        while (while_method_6(v662)){
                                            int v664;
                                            v664 = 0;
                                            while (while_method_2(v664)){
                                                bool v666;
                                                v666 = 0 <= v664;
                                                bool v668;
                                                if (v666){
                                                    bool v667;
                                                    v667 = v664 < 2;
                                                    v668 = v667;
                                                } else {
                                                    v668 = false;
                                                }
                                                bool v669;
                                                v669 = v668 == false;
                                                if (v669){
                                                    assert("The indices should be inside the range of the dimension." && v668);
                                                } else {
                                                }
                                                bool v671;
                                                v671 = 0 <= v625;
                                                bool v673;
                                                if (v671){
                                                    bool v672;
                                                    v672 = v625 < 1;
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
                                                int v676;
                                                v676 = v625 * 2;
                                                int v677;
                                                v677 = v664 + v676;
                                                bool v678;
                                                v678 = 0 <= v662;
                                                bool v680;
                                                if (v678){
                                                    bool v679;
                                                    v679 = v662 < 1;
                                                    v680 = v679;
                                                } else {
                                                    v680 = false;
                                                }
                                                bool v681;
                                                v681 = v680 == false;
                                                if (v681){
                                                    assert("The indices should be inside the range of the dimension." && v680);
                                                } else {
                                                }
                                                int v683;
                                                v683 = v662 * 2;
                                                int v684;
                                                v684 = v677 + v683;
                                                assert("Tensor range check" && 0 <= v662 && v662 < 1);
                                                assert("Tensor range check" && 0 <= v664 && v664 < 2);
                                                int v685;
                                                v685 = 2 * v662;
                                                int v686;
                                                v686 = v685 + v664;
                                                v653[v686] = v684;
                                                v664 += 1 ;
                                            }
                                            v662 += 1 ;
                                        }
                                        int v687;
                                        v687 = 0;
                                        while (while_method_6(v687)){
                                            assert("Tensor range check" && 0 <= v687 && v687 < 1);
                                            int v689;
                                            v689 = 2 * v687;
                                            int v690;
                                            v690 = v689 + v650;
                                            assert("Tensor range check" && 0 <= v687 && v687 < 1);
                                            int4* v691;
                                            v691 = reinterpret_cast<int4*>(v651 + v689);
                                            int4* v692;
                                            v692 = reinterpret_cast<int4*>(v645 + v690);
                                            assert("Pointer alignment check" && (unsigned long long)(v691) % 2 == 0 && (unsigned long long)(v692) % 2 == 0);
                                            *v692 = *v691;
                                            int4* v693;
                                            v693 = reinterpret_cast<int4*>(v652 + v689);
                                            int4* v694;
                                            v694 = reinterpret_cast<int4*>(v646 + v690);
                                            assert("Pointer alignment check" && (unsigned long long)(v693) % 2 == 0 && (unsigned long long)(v694) % 2 == 0);
                                            *v694 = *v693;
                                            v687 += 1 ;
                                        }
                                        assert("Tensor range check" && 0 <= v640 && v640 < 256);
                                        v629 += 1 ;
                                    }
                                    asm("barrier.cta.sync %0;" :: "r"(0));
                                    assert("Tensor range check" && 0 <= v621 && v621 < 256);
                                    asm("barrier.cta.sync %0;" :: "r"(0));
                                    double v695;
                                    v695 = (double)v529;
                                    double v696;
                                    v696 = log(v695);
                                    double v697;
                                    v697 = (double)v565;
                                    double v698;
                                    v698 = log(v697);
                                    assert("Tensor range check" && 0 <= v553 && v553 < 4);
                                    assert("Tensor range check" && 0 <= v552 && v552 < 6144);
                                    assert("Tensor range check" && 0 <= v54 && v54 < 2);
                                    int v699;
                                    v699 = v574 + v54;
                                    int v700;
                                    v700 = v573 + v699;
                                    double v701;
                                    v701 = v531[v700];
                                    double v702;
                                    v702 = v533[v700];
                                    double v703;
                                    v703 = v698 + v701;
                                    double v704;
                                    v704 = v696 + v702;
                                    assert("Tensor range check" && 0 <= v553 && v553 < 4);
                                    assert("Tensor range check" && 0 <= v552 && v552 < 6144);
                                    assert("Tensor range check" && 0 <= v54 && v54 < 2);
                                    v531[v700] = v703;
                                    v533[v700] = v704;
                                    v553 += 1 ;
                                }
                                bool v705;
                                v705 = 0 == v530;
                                Union10 v714;
                                if (v705){
                                    v714 = Union10{Union10_1{}};
                                } else {
                                    bool v707;
                                    v707 = 1 == v530;
                                    if (v707){
                                        v714 = Union10{Union10_0{}};
                                    } else {
                                        bool v709;
                                        v709 = 2 == v530;
                                        if (v709){
                                            v714 = Union10{Union10_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v714.tag) {
                                    case 0: { // AA_Call
                                        v779 = Union3{Union3_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v715;
                                        v715 = v55[0];
                                        int v717; int v718;
                                        Tuple1 tmp17 = Tuple1{1, v715};
                                        v717 = tmp17.v0; v718 = tmp17.v1;
                                        while (while_method_2(v717)){
                                            int v720;
                                            v720 = v55[v717];
                                            bool v722;
                                            v722 = v718 >= v720;
                                            int v723;
                                            if (v722){
                                                v723 = v718;
                                            } else {
                                                v723 = v720;
                                            }
                                            v718 = v723;
                                            v717 += 1 ;
                                        }
                                        int v724;
                                        v724 = v55[v54];
                                        bool v726;
                                        v726 = v724 == v718;
                                        if (v726){
                                            v779 = Union3{Union3_0{}};
                                        } else {
                                            v779 = Union3{Union3_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v731;
                                        v731 = v56 > 0;
                                        if (v731){
                                            v779 = Union3{Union3_2{}};
                                        } else {
                                            v779 = Union3{Union3_0{}};
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                break;
                            }
                            case 1: { // T_Random
                                curandStatePhilox4_32_10_t & v738 = v2.v5;
                                curandStatePhilox4_32_10_t & v739 = v738;
                                static_array_list<Union3,3> v740;
                                v740 = static_array_list<Union3,3>{};
                                v740.unsafe_set_length(1);
                                Union3 v742;
                                v742 = Union3{Union3_0{}};
                                v740[0] = v742;
                                int v744;
                                v744 = v55[0];
                                int v746;
                                v746 = v55[1];
                                bool v748;
                                v748 = v744 == v746;
                                bool v749;
                                v749 = v748 != true;
                                if (v749){
                                    Union3 v750;
                                    v750 = Union3{Union3_1{}};
                                    v740.push(v750);
                                } else {
                                }
                                bool v751;
                                v751 = v56 > 0;
                                if (v751){
                                    Union3 v752;
                                    v752 = Union3{Union3_2{}};
                                    v740.push(v752);
                                } else {
                                }
                                int v753;
                                v753 = v740.length;
                                int v754;
                                v754 = v753 - 1;
                                int v755;
                                v755 = 0;
                                while (while_method_5(v754, v755)){
                                    int v757;
                                    v757 = v740.length;
                                    int v758;
                                    v758 = int_range_6(v757, v755, v739);
                                    Union3 v759;
                                    v759 = v740[v755];
                                    Union3 v761;
                                    v761 = v740[v758];
                                    v740[v755] = v761;
                                    v740[v758] = v759;
                                    v755 += 1 ;
                                }
                                Union3 v763;
                                v763 = v740.pop();
                                int v764;
                                v764 = sizeof(Union3);
                                unsigned long long v765;
                                v765 = (unsigned long long)v764;
                                bool v766;
                                v766 = v765 <= 98304ull;
                                bool v767;
                                v767 = v766 == false;
                                if (v767){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v766);
                                } else {
                                }
                                extern __shared__ unsigned char v769[];
                                bool v770;
                                v770 = v765 <= v765;
                                bool v771;
                                v771 = v770 == false;
                                if (v771){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v770);
                                } else {
                                }
                                Union3 * v773;
                                v773 = reinterpret_cast<Union3 *>(&v769[0ull]);
                                int v775;
                                v775 = threadIdx.x;
                                bool v776;
                                v776 = v775 == 0;
                                if (v776){
                                    v773[0] = v763;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                Union3 v777;
                                v777 = v773[0];
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                v779 = v777;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union1 v780;
                        v780 = Union1{Union1_1{v54, v779}};
                        v4.push(v780);
                        Union4 v866;
                        switch (v51.tag) {
                            case 0: { // None
                                switch (v779.tag) {
                                    case 0: { // Call
                                        if (v52){
                                            bool v830;
                                            v830 = v54 == 0;
                                            int v831;
                                            if (v830){
                                                v831 = 1;
                                            } else {
                                                v831 = 0;
                                            }
                                            v866 = Union4{Union4_2{v51, false, v53, v831, v55, v56}};
                                        } else {
                                            v866 = Union4{Union4_0{v51, v52, v53, v54, v55, v56}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v866 = Union4{Union4_5{v51, v52, v53, v54, v55, v56}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v835;
                                        v835 = v56 > 0;
                                        if (v835){
                                            bool v836;
                                            v836 = v54 == 0;
                                            int v837;
                                            if (v836){
                                                v837 = 1;
                                            } else {
                                                v837 = 0;
                                            }
                                            int v838;
                                            v838 = -1 + v56;
                                            int v839; int v840;
                                            Tuple1 tmp18 = Tuple1{0, 0};
                                            v839 = tmp18.v0; v840 = tmp18.v1;
                                            while (while_method_2(v839)){
                                                int v842;
                                                v842 = v55[v839];
                                                bool v844;
                                                v844 = v840 >= v842;
                                                int v845;
                                                if (v844){
                                                    v845 = v840;
                                                } else {
                                                    v845 = v842;
                                                }
                                                v840 = v845;
                                                v839 += 1 ;
                                            }
                                            static_array<int,2> v846;
                                            int v848;
                                            v848 = 0;
                                            while (while_method_2(v848)){
                                                v846[v848] = v840;
                                                v848 += 1 ;
                                            }
                                            static_array<int,2> v850;
                                            int v852;
                                            v852 = 0;
                                            while (while_method_2(v852)){
                                                int v854;
                                                v854 = v846[v852];
                                                bool v856;
                                                v856 = v852 == v54;
                                                int v858;
                                                if (v856){
                                                    int v857;
                                                    v857 = v854 + 2;
                                                    v858 = v857;
                                                } else {
                                                    v858 = v854;
                                                }
                                                v850[v852] = v858;
                                                v852 += 1 ;
                                            }
                                            v866 = Union4{Union4_2{v51, false, v53, v837, v850, v838}};
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
                                Union2 v781 = v51.case1.v0;
                                switch (v779.tag) {
                                    case 0: { // Call
                                        if (v52){
                                            bool v783;
                                            v783 = v54 == 0;
                                            int v784;
                                            if (v783){
                                                v784 = 1;
                                            } else {
                                                v784 = 0;
                                            }
                                            v866 = Union4{Union4_2{v51, false, v53, v784, v55, v56}};
                                        } else {
                                            int v786; int v787;
                                            Tuple1 tmp19 = Tuple1{0, 0};
                                            v786 = tmp19.v0; v787 = tmp19.v1;
                                            while (while_method_2(v786)){
                                                int v789;
                                                v789 = v55[v786];
                                                bool v791;
                                                v791 = v787 >= v789;
                                                int v792;
                                                if (v791){
                                                    v792 = v787;
                                                } else {
                                                    v792 = v789;
                                                }
                                                v787 = v792;
                                                v786 += 1 ;
                                            }
                                            static_array<int,2> v793;
                                            int v795;
                                            v795 = 0;
                                            while (while_method_2(v795)){
                                                v793[v795] = v787;
                                                v795 += 1 ;
                                            }
                                            v866 = Union4{Union4_4{v51, v52, v53, v54, v793, v56}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v866 = Union4{Union4_5{v51, v52, v53, v54, v55, v56}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v799;
                                        v799 = v56 > 0;
                                        if (v799){
                                            bool v800;
                                            v800 = v54 == 0;
                                            int v801;
                                            if (v800){
                                                v801 = 1;
                                            } else {
                                                v801 = 0;
                                            }
                                            int v802;
                                            v802 = -1 + v56;
                                            int v803; int v804;
                                            Tuple1 tmp20 = Tuple1{0, 0};
                                            v803 = tmp20.v0; v804 = tmp20.v1;
                                            while (while_method_2(v803)){
                                                int v806;
                                                v806 = v55[v803];
                                                bool v808;
                                                v808 = v804 >= v806;
                                                int v809;
                                                if (v808){
                                                    v809 = v804;
                                                } else {
                                                    v809 = v806;
                                                }
                                                v804 = v809;
                                                v803 += 1 ;
                                            }
                                            static_array<int,2> v810;
                                            int v812;
                                            v812 = 0;
                                            while (while_method_2(v812)){
                                                v810[v812] = v804;
                                                v812 += 1 ;
                                            }
                                            static_array<int,2> v814;
                                            int v816;
                                            v816 = 0;
                                            while (while_method_2(v816)){
                                                int v818;
                                                v818 = v810[v816];
                                                bool v820;
                                                v820 = v816 == v54;
                                                int v822;
                                                if (v820){
                                                    int v821;
                                                    v821 = v818 + 4;
                                                    v822 = v821;
                                                } else {
                                                    v822 = v818;
                                                }
                                                v814[v816] = v822;
                                                v816 += 1 ;
                                            }
                                            v866 = Union4{Union4_2{v51, false, v53, v801, v814, v802}};
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
                        v1018 = Union6{Union6_1{v866}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v868 = v8.case3.v0; bool v869 = v8.case3.v1; static_array<Union2,2> v870 = v8.case3.v2; int v871 = v8.case3.v3; static_array<int,2> v872 = v8.case3.v4; int v873 = v8.case3.v5; Union3 v874 = v8.case3.v6;
                        Union1 v875;
                        v875 = Union1{Union1_1{v871, v874}};
                        v4.push(v875);
                        Union4 v961;
                        switch (v868.tag) {
                            case 0: { // None
                                switch (v874.tag) {
                                    case 0: { // Call
                                        if (v869){
                                            bool v925;
                                            v925 = v871 == 0;
                                            int v926;
                                            if (v925){
                                                v926 = 1;
                                            } else {
                                                v926 = 0;
                                            }
                                            v961 = Union4{Union4_2{v868, false, v870, v926, v872, v873}};
                                        } else {
                                            v961 = Union4{Union4_0{v868, v869, v870, v871, v872, v873}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v961 = Union4{Union4_5{v868, v869, v870, v871, v872, v873}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v930;
                                        v930 = v873 > 0;
                                        if (v930){
                                            bool v931;
                                            v931 = v871 == 0;
                                            int v932;
                                            if (v931){
                                                v932 = 1;
                                            } else {
                                                v932 = 0;
                                            }
                                            int v933;
                                            v933 = -1 + v873;
                                            int v934; int v935;
                                            Tuple1 tmp21 = Tuple1{0, 0};
                                            v934 = tmp21.v0; v935 = tmp21.v1;
                                            while (while_method_2(v934)){
                                                int v937;
                                                v937 = v872[v934];
                                                bool v939;
                                                v939 = v935 >= v937;
                                                int v940;
                                                if (v939){
                                                    v940 = v935;
                                                } else {
                                                    v940 = v937;
                                                }
                                                v935 = v940;
                                                v934 += 1 ;
                                            }
                                            static_array<int,2> v941;
                                            int v943;
                                            v943 = 0;
                                            while (while_method_2(v943)){
                                                v941[v943] = v935;
                                                v943 += 1 ;
                                            }
                                            static_array<int,2> v945;
                                            int v947;
                                            v947 = 0;
                                            while (while_method_2(v947)){
                                                int v949;
                                                v949 = v941[v947];
                                                bool v951;
                                                v951 = v947 == v871;
                                                int v953;
                                                if (v951){
                                                    int v952;
                                                    v952 = v949 + 2;
                                                    v953 = v952;
                                                } else {
                                                    v953 = v949;
                                                }
                                                v945[v947] = v953;
                                                v947 += 1 ;
                                            }
                                            v961 = Union4{Union4_2{v868, false, v870, v932, v945, v933}};
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
                                Union2 v876 = v868.case1.v0;
                                switch (v874.tag) {
                                    case 0: { // Call
                                        if (v869){
                                            bool v878;
                                            v878 = v871 == 0;
                                            int v879;
                                            if (v878){
                                                v879 = 1;
                                            } else {
                                                v879 = 0;
                                            }
                                            v961 = Union4{Union4_2{v868, false, v870, v879, v872, v873}};
                                        } else {
                                            int v881; int v882;
                                            Tuple1 tmp22 = Tuple1{0, 0};
                                            v881 = tmp22.v0; v882 = tmp22.v1;
                                            while (while_method_2(v881)){
                                                int v884;
                                                v884 = v872[v881];
                                                bool v886;
                                                v886 = v882 >= v884;
                                                int v887;
                                                if (v886){
                                                    v887 = v882;
                                                } else {
                                                    v887 = v884;
                                                }
                                                v882 = v887;
                                                v881 += 1 ;
                                            }
                                            static_array<int,2> v888;
                                            int v890;
                                            v890 = 0;
                                            while (while_method_2(v890)){
                                                v888[v890] = v882;
                                                v890 += 1 ;
                                            }
                                            v961 = Union4{Union4_4{v868, v869, v870, v871, v888, v873}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v961 = Union4{Union4_5{v868, v869, v870, v871, v872, v873}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v894;
                                        v894 = v873 > 0;
                                        if (v894){
                                            bool v895;
                                            v895 = v871 == 0;
                                            int v896;
                                            if (v895){
                                                v896 = 1;
                                            } else {
                                                v896 = 0;
                                            }
                                            int v897;
                                            v897 = -1 + v873;
                                            int v898; int v899;
                                            Tuple1 tmp23 = Tuple1{0, 0};
                                            v898 = tmp23.v0; v899 = tmp23.v1;
                                            while (while_method_2(v898)){
                                                int v901;
                                                v901 = v872[v898];
                                                bool v903;
                                                v903 = v899 >= v901;
                                                int v904;
                                                if (v903){
                                                    v904 = v899;
                                                } else {
                                                    v904 = v901;
                                                }
                                                v899 = v904;
                                                v898 += 1 ;
                                            }
                                            static_array<int,2> v905;
                                            int v907;
                                            v907 = 0;
                                            while (while_method_2(v907)){
                                                v905[v907] = v899;
                                                v907 += 1 ;
                                            }
                                            static_array<int,2> v909;
                                            int v911;
                                            v911 = 0;
                                            while (while_method_2(v911)){
                                                int v913;
                                                v913 = v905[v911];
                                                bool v915;
                                                v915 = v911 == v871;
                                                int v917;
                                                if (v915){
                                                    int v916;
                                                    v916 = v913 + 4;
                                                    v917 = v916;
                                                } else {
                                                    v917 = v913;
                                                }
                                                v909[v911] = v917;
                                                v911 += 1 ;
                                            }
                                            v961 = Union4{Union4_2{v868, false, v870, v896, v909, v897}};
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
                        v1018 = Union6{Union6_1{v961}};
                        break;
                    }
                    case 4: { // TerminalCall
                        Union5 v26 = v8.case4.v0; bool v27 = v8.case4.v1; static_array<Union2,2> v28 = v8.case4.v2; int v29 = v8.case4.v3; static_array<int,2> v30 = v8.case4.v4; int v31 = v8.case4.v5;
                        int v32;
                        v32 = v30[v29];
                        Union11 v34;
                        v34 = compare_hands_7(v26, v27, v28, v29, v30, v31);
                        int v39; int v40;
                        switch (v34.tag) {
                            case 0: { // Eq
                                v39 = 0; v40 = -1;
                                break;
                            }
                            case 1: { // Gt
                                v39 = v32; v40 = 0;
                                break;
                            }
                            case 2: { // Lt
                                v39 = v32; v40 = 1;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        int v41;
                        v41 = -v40;
                        bool v42;
                        v42 = v40 >= v41;
                        int v43;
                        if (v42){
                            v43 = v40;
                        } else {
                            v43 = v41;
                        }
                        float v44;
                        v44 = (float)v39;
                        static_array<float,2> & v45 = v2.v4;
                        v45[v43] = v44;
                        bool v46;
                        v46 = v43 == 0;
                        int v47;
                        if (v46){
                            v47 = 1;
                        } else {
                            v47 = 0;
                        }
                        float v48;
                        v48 = -v44;
                        v45[v47] = v48;
                        Union1 v49;
                        v49 = Union1{Union1_3{v28, v39, v40}};
                        v4.push(v49);
                        v1018 = Union6{Union6_0{}};
                        break;
                    }
                    case 5: { // TerminalFold
                        Union5 v9 = v8.case5.v0; bool v10 = v8.case5.v1; static_array<Union2,2> v11 = v8.case5.v2; int v12 = v8.case5.v3; static_array<int,2> v13 = v8.case5.v4; int v14 = v8.case5.v5;
                        int v15;
                        v15 = v13[v12];
                        int v17;
                        v17 = -v15;
                        float v18;
                        v18 = (float)v17;
                        static_array<float,2> & v19 = v2.v4;
                        v19[v12] = v18;
                        bool v20;
                        v20 = v12 == 0;
                        int v21;
                        if (v20){
                            v21 = 1;
                        } else {
                            v21 = 0;
                        }
                        float v22;
                        v22 = -v18;
                        v19[v21] = v22;
                        int v23;
                        if (v20){
                            v23 = 1;
                        } else {
                            v23 = 0;
                        }
                        Union1 v24;
                        v24 = Union1{Union1_3{v11, v15, v23}};
                        v4.push(v24);
                        v1018 = Union6{Union6_0{}};
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
        v6 = v1018;
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
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1) {
    auto v2 = cooperative_groups::this_grid();
    unsigned long long v3;
    v3 = clock64();
    int v4;
    v4 = threadIdx.x;
    int v5;
    v5 = blockIdx.x;
    int v6;
    v6 = v5 * 256;
    int v7;
    v7 = v4 + v6;
    unsigned long long v8;
    v8 = (unsigned long long)v7;
    curandStatePhilox4_32_10_t v9;
    curand_init(v3,v8,0ull,&v9);
    static_array<Union0,2> v10;
    Union0 v12;
    v12 = Union0{Union0_1{}};
    v10[0] = v12;
    Union0 v14;
    v14 = Union0{Union0_1{}};
    v10[1] = v14;
    static_array_list<Union1,32> v16;
    v16 = static_array_list<Union1,32>{};
    static_array<float,2> v18;
    v18[0] = 0.0f;
    v18[1] = 0.0f;
    cooperative_groups::grid_group & v20 = v2;
    curandStatePhilox4_32_10_t & v21 = v9;
    StackMut0 v22{63u, v20, v16, v10, v18, v21};
    int v23;
    v23 = 0;
    while (while_method_0(v23)){
        int v25;
        v25 = 0;
        while (while_method_1(v25)){
            int v27;
            v27 = 0;
            while (while_method_2(v27)){
                v22.v0 = 63u;
                static_array<float,2> v29;
                v29[0] = 0.0f;
                v29[1] = 0.0f;
                v22.v4 = v29;
                static_array_list<Union1,32> & v31 = v22.v2;
                v31.unsafe_set_length(0);
                static_array<Union0,2> v32;
                Union0 v34;
                v34 = Union0{Union0_1{}};
                v32[0] = v34;
                Union0 v36;
                v36 = Union0{Union0_1{}};
                v32[1] = v36;
                Union0 v38;
                v38 = Union0{Union0_0{}};
                v32[v27] = v38;
                v22.v3 = v32;
                Union4 v40;
                v40 = Union4{Union4_1{}};
                method_0(v0, v1, v22, v40);
                static_array<float,2> & v41 = v22.v4;
                static_array<float,2> v42;
                v42 = v41;
                unsigned int * v43;
                v43 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
                int * v45;
                v45 = reinterpret_cast<int *>(&v1[262144ull]);
                float * v47;
                v47 = reinterpret_cast<float *>(&v1[262160ull]);
                float * v49;
                v49 = reinterpret_cast<float *>(&v1[524304ull]);
                float * v51;
                v51 = reinterpret_cast<float *>(&v1[786448ull]);
                float * v53;
                v53 = reinterpret_cast<float *>(&v1[1048592ull]);
                float * v55;
                v55 = reinterpret_cast<float *>(&v1[1310736ull]);
                float * v57;
                v57 = reinterpret_cast<float *>(&v1[1572880ull]);
                float * v59;
                v59 = reinterpret_cast<float *>(&v1[1835024ull]);
                int * v61;
                v61 = reinterpret_cast<int *>(&v0[6389760ull]);
                float * v63;
                v63 = reinterpret_cast<float *>(&v0[7962624ull]);
                int * v65;
                v65 = reinterpret_cast<int *>(&v0[9535488ull]);
                int * v67;
                v67 = reinterpret_cast<int *>(&v0[11108352ull]);
                double * v69;
                v69 = reinterpret_cast<double *>(&v0[12681216ull]);
                double * v71;
                v71 = reinterpret_cast<double *>(&v0[18972672ull]);
                double * v73;
                v73 = reinterpret_cast<double *>(&v1[2097168ull]);
                double * v75;
                v75 = reinterpret_cast<double *>(&v1[2490384ull]);
                int * v77;
                v77 = reinterpret_cast<int *>(&v1[2883600ull]);
                int v79;
                v79 = 0;
                while (while_method_0(v79)){
                    int v81;
                    v81 = threadIdx.x;
                    int v82;
                    v82 = blockIdx.x;
                    int v83;
                    v83 = v82 * 256;
                    int v84;
                    v84 = v81 + v83;
                    float v85[2];
                    int v86;
                    v86 = 0;
                    while (while_method_2(v86)){
                        float v88;
                        v88 = v42[v86];
                        v85[v86] = v88;
                        v86 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v79 && v79 < 4);
                    assert("Tensor range check" && 0 <= v84 && v84 < 6144);
                    int v90;
                    v90 = 6144 * v79;
                    int v91;
                    v91 = v90 + v84;
                    int v92;
                    v92 = v77[v91];
                    int v93;
                    v93 = v92;
                    while (while_method_10(v93)){
                        v93 -= 1 ;
                        assert("Tensor range check" && 0 <= v79 && v79 < 4);
                        assert("Tensor range check" && 0 <= v93 && v93 < 16);
                        assert("Tensor range check" && 0 <= v84 && v84 < 6144);
                        int v95;
                        v95 = 6144 * v93;
                        int v96;
                        v96 = v95 + v84;
                        int v97;
                        v97 = 98304 * v79;
                        int v98;
                        v98 = v97 + v96;
                        int v99;
                        v99 = v61[v98];
                        float v100;
                        v100 = v63[v98];
                        int v101;
                        v101 = v65[v98];
                        int v102;
                        v102 = v67[v98];
                        assert("Tensor range check" && 0 <= v101 && v101 < 2);
                        float v103;
                        v103 = v85[v101];
                        assert("Tensor range check" && 0 <= v79 && v79 < 4);
                        int v104;
                        v104 = 16384 * v79;
                        assert("Tensor range check" && 0 <= v102 && v102 < 4096);
                        int v105;
                        v105 = 4 * v102;
                        int v106;
                        v106 = v105 + v104;
                        float * v107;
                        v107 = v47+v106;
                        float * v109;
                        v109 = v49+v106;
                        float * v111;
                        v111 = v51+v106;
                        float * v113;
                        v113 = v53+v106;
                        float * v115;
                        v115 = v55+v106;
                        float * v117;
                        v117 = v57+v106;
                        float * v119;
                        v119 = v59+v106;
                        assert("Tensor range check" && 0 <= v79 && v79 < 4);
                        int v121;
                        v121 = 196608 * v79;
                        assert("Tensor range check" && 0 <= v93 && v93 < 16);
                        int v122;
                        v122 = 12288 * v93;
                        int v123;
                        v123 = v122 + v121;
                        assert("Tensor range check" && 0 <= v84 && v84 < 6144);
                        int v124;
                        v124 = 2 * v84;
                        int v125;
                        v125 = v124 + v123;
                        double v126[2];
                        int v127;
                        v127 = 0;
                        while (while_method_2(v127)){
                            assert("Tensor range check" && 0 <= v127 && v127 < 2);
                            int v129;
                            v129 = v127 + v125;
                            double v130;
                            v130 = v69[v129];
                            bool v131;
                            v131 = v101 == v127;
                            double v132;
                            if (v131){
                                v132 = 0.0;
                            } else {
                                v132 = v130;
                            }
                            assert("Tensor range check" && 0 <= v127 && v127 < 2);
                            v126[v127] = v132;
                            v127 += 1 ;
                        }
                        double v133;
                        v133 = 0.0;
                        int v134;
                        v134 = 0;
                        while (while_method_2(v134)){
                            assert("Tensor range check" && 0 <= v134 && v134 < 2);
                            double v136;
                            v136 = v126[v134];
                            double v137;
                            v137 = v133 + v136;
                            v133 = v137;
                            v134 += 1 ;
                        }
                        double v138;
                        v138 = 0.0;
                        int v139;
                        v139 = 0;
                        while (while_method_2(v139)){
                            assert("Tensor range check" && 0 <= v139 && v139 < 2);
                            int v141;
                            v141 = v139 + v125;
                            double v142;
                            v142 = v71[v141];
                            double v143;
                            v143 = v138 + v142;
                            v138 = v143;
                            v139 += 1 ;
                        }
                        double v144;
                        v144 = v133 - v138;
                        double v145;
                        v145 = exp(v144);
                        float v146;
                        v146 = (float)v145;
                        float v147;
                        v147 = v103 * v146;
                        assert("Tensor range check" && 0 <= v99 && v99 < 4);
                        float * v148;
                        v148 = v117+v99;
                        float * v150;
                        v150 = v119+v99;
                        float v152;
                        v152 = atomicAdd(v148,v147);
                        float v153;
                        v153 = atomicAdd(v150,v146);
                        float * v154;
                        v154 = v109+0;
                        float * v156;
                        v156 = v113+0;
                        float * v158;
                        v158 = v115+0;
                        int v160;
                        v160 = sizeof(float *);
                        unsigned long long v161;
                        v161 = (unsigned long long)v160;
                        unsigned long long v162;
                        v162 = 256ull * v161;
                        unsigned long long v163;
                        v163 = 4096ull + v162;
                        unsigned long long v164;
                        v164 = v163 + 16ull;
                        unsigned long long v165;
                        v165 = v164 - 1ull;
                        unsigned long long v166;
                        v166 = v165 % 16ull;
                        unsigned long long v167;
                        v167 = v165 - v166;
                        unsigned long long v168;
                        v168 = v167 + v162;
                        unsigned long long v169;
                        v169 = v168 + 16ull;
                        unsigned long long v170;
                        v170 = v169 - 1ull;
                        unsigned long long v171;
                        v171 = v170 % 16ull;
                        unsigned long long v172;
                        v172 = v170 - v171;
                        unsigned long long v173;
                        v173 = v172 + v162;
                        unsigned long long v174;
                        v174 = v173 + 16ull;
                        unsigned long long v175;
                        v175 = v174 - 1ull;
                        unsigned long long v176;
                        v176 = v175 % 16ull;
                        unsigned long long v177;
                        v177 = v175 - v176;
                        unsigned long long v178;
                        v178 = v177 + v162;
                        unsigned long long v179;
                        v179 = v178 + 16ull;
                        unsigned long long v180;
                        v180 = v179 - 1ull;
                        unsigned long long v181;
                        v181 = v180 % 16ull;
                        unsigned long long v182;
                        v182 = v180 - v181;
                        unsigned long long v183;
                        v183 = v182 + 1024ull;
                        bool v184;
                        v184 = v183 <= 98304ull;
                        bool v185;
                        v185 = v184 == false;
                        if (v185){
                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v184);
                        } else {
                        }
                        extern __shared__ unsigned char v187[];
                        bool v188;
                        v188 = v183 <= v183;
                        bool v189;
                        v189 = v188 == false;
                        if (v189){
                            assert("The length of the partition has to be less than or equal to the length of the base array." && v188);
                        } else {
                        }
                        float * v191;
                        v191 = reinterpret_cast<float *>(&v187[0ull]);
                        int * v193;
                        v193 = reinterpret_cast<int *>(&v187[1024ull]);
                        float * v195;
                        v195 = reinterpret_cast<float *>(&v187[2048ull]);
                        float * v197;
                        v197 = reinterpret_cast<float *>(&v187[3072ull]);
                        float * * v199;
                        v199 = reinterpret_cast<float * *>(&v187[4096ull]);
                        float * * v201;
                        v201 = reinterpret_cast<float * *>(&v187[v167]);
                        float * * v203;
                        v203 = reinterpret_cast<float * *>(&v187[v172]);
                        float * * v205;
                        v205 = reinterpret_cast<float * *>(&v187[v177]);
                        float * v207;
                        v207 = reinterpret_cast<float *>(&v187[v182]);
                        int v209;
                        v209 = threadIdx.x;
                        assert("Tensor range check" && 0 <= v209 && v209 < 256);
                        v191[v209] = v100;
                        v193[v209] = v99;
                        v195[v209] = v103;
                        v197[v209] = v146;
                        v199[v209] = v111;
                        v201[v209] = v154;
                        v203[v209] = v156;
                        v205[v209] = v158;
                        asm("barrier.cta.sync %0;" :: "r"(0));
                        bool v210;
                        v210 = 0 <= v209;
                        bool v211;
                        v211 = v210 == false;
                        if (v211){
                            assert("The index needs to be zero or positive." && v210);
                        } else {
                        }
                        int v213;
                        v213 = v209 % 1;
                        bool v214;
                        v214 = v209 < 256;
                        bool v215;
                        v215 = v214 == false;
                        if (v215){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v214);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v209 && v209 < 256);
                        int v217;
                        v217 = 0;
                        while (while_method_6(v217)){
                            bool v219;
                            v219 = v210 && v214;
                            bool v220;
                            v220 = v219 == false;
                            if (v220){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v219);
                            } else {
                            }
                            bool v222;
                            v222 = 0 <= v217;
                            bool v224;
                            if (v222){
                                bool v223;
                                v223 = v217 < 1;
                                v224 = v223;
                            } else {
                                v224 = false;
                            }
                            bool v225;
                            v225 = v224 == false;
                            if (v225){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v224);
                            } else {
                            }
                            int v227;
                            v227 = v217 * 256;
                            int v228;
                            v228 = v227 + v209;
                            assert("Tensor range check" && 0 <= v217 && v217 < 1);
                            int v229;
                            v229 = 256 * v217;
                            int v230;
                            v230 = v229 + v209;
                            float v231;
                            v231 = v191[v230];
                            int v232;
                            v232 = v193[v230];
                            float v233;
                            v233 = v195[v230];
                            float v234;
                            v234 = v197[v230];
                            float * v235;
                            v235 = v199[v230];
                            float * v236;
                            v236 = v201[v230];
                            float * v237;
                            v237 = v203[v230];
                            float * v238;
                            v238 = v205[v230];
                            int v239;
                            v239 = blockIdx.x;
                            int v240;
                            v240 = v239 * 256;
                            int v241;
                            v241 = v240 + v228;
                            assert("Tensor range check" && 0 <= v213 && v213 < 1);
                            int v242;
                            v242 = 4 * v213;
                            float v243[4];
                            float v244[4];
                            float v245[4];
                            int v246[4];
                            int v247;
                            v247 = 0;
                            while (while_method_6(v247)){
                                assert("Tensor range check" && 0 <= v247 && v247 < 1);
                                int v249;
                                v249 = 4 * v247;
                                assert("Tensor range check" && 0 <= v247 && v247 < 1);
                                int v250;
                                v250 = v249 + v242;
                                int4* v251;
                                v251 = reinterpret_cast<int4*>(v236 + v250);
                                int4* v252;
                                v252 = reinterpret_cast<int4*>(v243 + v249);
                                assert("Pointer alignment check" && (unsigned long long)(v251) % 4 == 0 && (unsigned long long)(v252) % 4 == 0);
                                *v252 = *v251;
                                int4* v253;
                                v253 = reinterpret_cast<int4*>(v237 + v250);
                                int4* v254;
                                v254 = reinterpret_cast<int4*>(v244 + v249);
                                assert("Pointer alignment check" && (unsigned long long)(v253) % 4 == 0 && (unsigned long long)(v254) % 4 == 0);
                                *v254 = *v253;
                                int4* v255;
                                v255 = reinterpret_cast<int4*>(v238 + v250);
                                int4* v256;
                                v256 = reinterpret_cast<int4*>(v245 + v249);
                                assert("Pointer alignment check" && (unsigned long long)(v255) % 4 == 0 && (unsigned long long)(v256) % 4 == 0);
                                *v256 = *v255;
                                v247 += 1 ;
                            }
                            int v257;
                            v257 = 0;
                            while (while_method_6(v257)){
                                int v259;
                                v259 = 0;
                                while (while_method_0(v259)){
                                    bool v261;
                                    v261 = 0 <= v259;
                                    bool v263;
                                    if (v261){
                                        bool v262;
                                        v262 = v259 < 4;
                                        v263 = v262;
                                    } else {
                                        v263 = false;
                                    }
                                    bool v264;
                                    v264 = v263 == false;
                                    if (v264){
                                        assert("The indices should be inside the range of the dimension." && v263);
                                    } else {
                                    }
                                    bool v266;
                                    v266 = 0 <= v213;
                                    bool v268;
                                    if (v266){
                                        bool v267;
                                        v267 = v213 < 1;
                                        v268 = v267;
                                    } else {
                                        v268 = false;
                                    }
                                    bool v269;
                                    v269 = v268 == false;
                                    if (v269){
                                        assert("The indices should be inside the range of the dimension." && v268);
                                    } else {
                                    }
                                    int v271;
                                    v271 = v213 * 4;
                                    int v272;
                                    v272 = v259 + v271;
                                    bool v273;
                                    v273 = 0 <= v257;
                                    bool v275;
                                    if (v273){
                                        bool v274;
                                        v274 = v257 < 1;
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
                                    int v278;
                                    v278 = v257 * 4;
                                    int v279;
                                    v279 = v272 + v278;
                                    assert("Tensor range check" && 0 <= v257 && v257 < 1);
                                    assert("Tensor range check" && 0 <= v259 && v259 < 4);
                                    int v280;
                                    v280 = 4 * v257;
                                    int v281;
                                    v281 = v280 + v259;
                                    v246[v281] = v279;
                                    v259 += 1 ;
                                }
                                v257 += 1 ;
                            }
                            float v282[4];
                            int v283;
                            v283 = 0;
                            while (while_method_6(v283)){
                                int v285;
                                v285 = 0;
                                while (while_method_0(v285)){
                                    assert("Tensor range check" && 0 <= v283 && v283 < 1);
                                    assert("Tensor range check" && 0 <= v285 && v285 < 4);
                                    int v287;
                                    v287 = 4 * v283;
                                    int v288;
                                    v288 = v287 + v285;
                                    float v289;
                                    v289 = v244[v288];
                                    float v290;
                                    v290 = v245[v288];
                                    bool v291;
                                    v291 = v290 == 0.0f;
                                    bool v292;
                                    v292 = v291 != true;
                                    float v294;
                                    if (v292){
                                        float v293;
                                        v293 = v289 / v290;
                                        v294 = v293;
                                    } else {
                                        v294 = 0.0f;
                                    }
                                    assert("Tensor range check" && 0 <= v283 && v283 < 1);
                                    assert("Tensor range check" && 0 <= v285 && v285 < 4);
                                    v282[v288] = v294;
                                    v285 += 1 ;
                                }
                                v283 += 1 ;
                            }
                            bool v295[4];
                            int v296;
                            v296 = 0;
                            while (while_method_6(v296)){
                                int v298;
                                v298 = 0;
                                while (while_method_0(v298)){
                                    assert("Tensor range check" && 0 <= v296 && v296 < 1);
                                    assert("Tensor range check" && 0 <= v298 && v298 < 4);
                                    int v300;
                                    v300 = 4 * v296;
                                    int v301;
                                    v301 = v300 + v298;
                                    float v302;
                                    v302 = v243[v301];
                                    int v303;
                                    v303 = v246[v301];
                                    bool v304;
                                    v304 = v303 < 3;
                                    assert("Tensor range check" && 0 <= v296 && v296 < 1);
                                    assert("Tensor range check" && 0 <= v298 && v298 < 4);
                                    v295[v301] = v304;
                                    v298 += 1 ;
                                }
                                v296 += 1 ;
                            }
                            float v305[4];
                            int v306;
                            v306 = 0;
                            while (while_method_6(v306)){
                                int v308;
                                v308 = 0;
                                while (while_method_0(v308)){
                                    assert("Tensor range check" && 0 <= v306 && v306 < 1);
                                    assert("Tensor range check" && 0 <= v308 && v308 < 4);
                                    int v310;
                                    v310 = 4 * v306;
                                    int v311;
                                    v311 = v310 + v308;
                                    float v312;
                                    v312 = v243[v311];
                                    bool v313;
                                    v313 = v295[v311];
                                    float v316;
                                    if (v313){
                                        bool v314;
                                        v314 = 0.0f >= v312;
                                        if (v314){
                                            v316 = 0.0f;
                                        } else {
                                            v316 = v312;
                                        }
                                    } else {
                                        v316 = 0.0f;
                                    }
                                    assert("Tensor range check" && 0 <= v306 && v306 < 1);
                                    assert("Tensor range check" && 0 <= v308 && v308 < 4);
                                    v305[v311] = v316;
                                    v308 += 1 ;
                                }
                                v306 += 1 ;
                            }
                            float v317;
                            v317 = 0.0f;
                            int v318;
                            v318 = 0;
                            while (while_method_6(v318)){
                                int v320;
                                v320 = 0;
                                while (while_method_0(v320)){
                                    assert("Tensor range check" && 0 <= v318 && v318 < 1);
                                    assert("Tensor range check" && 0 <= v320 && v320 < 4);
                                    int v322;
                                    v322 = 4 * v318;
                                    int v323;
                                    v323 = v322 + v320;
                                    float v324;
                                    v324 = v305[v323];
                                    float v325;
                                    v325 = v317 + v324;
                                    v317 = v325;
                                    v320 += 1 ;
                                }
                                v318 += 1 ;
                            }
                            auto v326 = cooperative_groups::coalesced_threads();
                            int v327;
                            v327 = threadIdx.x;
                            auto v328 = cooperative_groups::labeled_partition(v326,v327);
                            Closure1 v329{};
                            float v330;
                            v330 = cooperative_groups::reduce(v328, v317, v329);
                            int v331[4];
                            int v332;
                            v332 = 0;
                            while (while_method_6(v332)){
                                int v334;
                                v334 = 0;
                                while (while_method_0(v334)){
                                    assert("Tensor range check" && 0 <= v332 && v332 < 1);
                                    assert("Tensor range check" && 0 <= v334 && v334 < 4);
                                    int v336;
                                    v336 = 4 * v332;
                                    int v337;
                                    v337 = v336 + v334;
                                    bool v338;
                                    v338 = v295[v337];
                                    int v339;
                                    if (v338){
                                        v339 = 1;
                                    } else {
                                        v339 = 0;
                                    }
                                    assert("Tensor range check" && 0 <= v332 && v332 < 1);
                                    assert("Tensor range check" && 0 <= v334 && v334 < 4);
                                    v331[v337] = v339;
                                    v334 += 1 ;
                                }
                                v332 += 1 ;
                            }
                            int v340;
                            v340 = 0;
                            int v341;
                            v341 = 0;
                            while (while_method_6(v341)){
                                int v343;
                                v343 = 0;
                                while (while_method_0(v343)){
                                    assert("Tensor range check" && 0 <= v341 && v341 < 1);
                                    assert("Tensor range check" && 0 <= v343 && v343 < 4);
                                    int v345;
                                    v345 = 4 * v341;
                                    int v346;
                                    v346 = v345 + v343;
                                    int v347;
                                    v347 = v331[v346];
                                    int v348;
                                    v348 = v340 + v347;
                                    v340 = v348;
                                    v343 += 1 ;
                                }
                                v341 += 1 ;
                            }
                            auto v349 = cooperative_groups::coalesced_threads();
                            int v350;
                            v350 = threadIdx.x;
                            auto v351 = cooperative_groups::labeled_partition(v349,v350);
                            Closure2 v352{};
                            int v353;
                            v353 = cooperative_groups::reduce(v351, v340, v352);
                            float v354;
                            v354 = (float)v353;
                            float v355;
                            v355 = 1.0f / v354;
                            float v356[4];
                            int v357;
                            v357 = 0;
                            while (while_method_6(v357)){
                                int v359;
                                v359 = 0;
                                while (while_method_0(v359)){
                                    assert("Tensor range check" && 0 <= v357 && v357 < 1);
                                    assert("Tensor range check" && 0 <= v359 && v359 < 4);
                                    int v361;
                                    v361 = 4 * v357;
                                    int v362;
                                    v362 = v361 + v359;
                                    float v363;
                                    v363 = v305[v362];
                                    bool v364;
                                    v364 = v295[v362];
                                    bool v365;
                                    v365 = v364 == false;
                                    float v370;
                                    if (v365){
                                        v370 = 0.0f;
                                    } else {
                                        bool v366;
                                        v366 = v330 == 0.0f;
                                        bool v367;
                                        v367 = v366 != true;
                                        if (v367){
                                            float v368;
                                            v368 = v363 / v330;
                                            v370 = v368;
                                        } else {
                                            v370 = v355;
                                        }
                                    }
                                    assert("Tensor range check" && 0 <= v357 && v357 < 1);
                                    assert("Tensor range check" && 0 <= v359 && v359 < 4);
                                    v356[v362] = v370;
                                    v359 += 1 ;
                                }
                                v357 += 1 ;
                            }
                            float v371[4];
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
                                    float v378;
                                    v378 = v282[v377];
                                    int v379;
                                    v379 = v246[v377];
                                    bool v380;
                                    v380 = v232 == v379;
                                    float v383;
                                    if (v380){
                                        float v381;
                                        v381 = v233 - v378;
                                        float v382;
                                        v382 = v381 / v231;
                                        v383 = v382;
                                    } else {
                                        v383 = 0.0f;
                                    }
                                    float v384;
                                    v384 = v383 + v378;
                                    assert("Tensor range check" && 0 <= v372 && v372 < 1);
                                    assert("Tensor range check" && 0 <= v374 && v374 < 4);
                                    v371[v377] = v384;
                                    v374 += 1 ;
                                }
                                v372 += 1 ;
                            }
                            float v385[4];
                            int v386;
                            v386 = 0;
                            while (while_method_6(v386)){
                                int v388;
                                v388 = 0;
                                while (while_method_0(v388)){
                                    assert("Tensor range check" && 0 <= v386 && v386 < 1);
                                    assert("Tensor range check" && 0 <= v388 && v388 < 4);
                                    int v390;
                                    v390 = 4 * v386;
                                    int v391;
                                    v391 = v390 + v388;
                                    float v392;
                                    v392 = v356[v391];
                                    float v393;
                                    v393 = v371[v391];
                                    float v394;
                                    v394 = v392 * v393;
                                    assert("Tensor range check" && 0 <= v386 && v386 < 1);
                                    assert("Tensor range check" && 0 <= v388 && v388 < 4);
                                    v385[v391] = v394;
                                    v388 += 1 ;
                                }
                                v386 += 1 ;
                            }
                            float v395;
                            v395 = 0.0f;
                            int v396;
                            v396 = 0;
                            while (while_method_6(v396)){
                                int v398;
                                v398 = 0;
                                while (while_method_0(v398)){
                                    assert("Tensor range check" && 0 <= v396 && v396 < 1);
                                    assert("Tensor range check" && 0 <= v398 && v398 < 4);
                                    int v400;
                                    v400 = 4 * v396;
                                    int v401;
                                    v401 = v400 + v398;
                                    float v402;
                                    v402 = v385[v401];
                                    float v403;
                                    v403 = v395 + v402;
                                    v395 = v403;
                                    v398 += 1 ;
                                }
                                v396 += 1 ;
                            }
                            auto v404 = cooperative_groups::coalesced_threads();
                            int v405;
                            v405 = threadIdx.x;
                            auto v406 = cooperative_groups::labeled_partition(v404,v405);
                            float v407;
                            v407 = cooperative_groups::reduce(v406, v395, v329);
                            int v408;
                            v408 = 0;
                            while (while_method_6(v408)){
                                int v410;
                                v410 = 0;
                                while (while_method_0(v410)){
                                    assert("Tensor range check" && 0 <= v408 && v408 < 1);
                                    assert("Tensor range check" && 0 <= v410 && v410 < 4);
                                    int v412;
                                    v412 = 4 * v408;
                                    int v413;
                                    v413 = v412 + v410;
                                    float v414;
                                    v414 = v371[v413];
                                    int v415;
                                    v415 = v246[v413];
                                    float v416;
                                    v416 = v414 - v407;
                                    float v417;
                                    v417 = v234 * v416;
                                    assert("Tensor range check" && 0 <= v415 && v415 < 4);
                                    float * v418;
                                    v418 = v235+v415;
                                    float v420;
                                    v420 = atomicAdd(v418,v417);
                                    v410 += 1 ;
                                }
                                v408 += 1 ;
                            }
                            int v421;
                            v421 = 0;
                            while (while_method_6(v421)){
                                assert("Tensor range check" && 0 <= v421 && v421 < 1);
                                assert("Tensor range check" && 0 <= v421 && v421 < 1);
                                v421 += 1 ;
                            }
                            assert("Tensor range check" && 0 <= v228 && v228 < 256);
                            v207[v228] = v407;
                            v217 += 1 ;
                        }
                        asm("barrier.cta.sync %0;" :: "r"(0));
                        assert("Tensor range check" && 0 <= v209 && v209 < 256);
                        float v423;
                        v423 = v207[v209];
                        asm("barrier.cta.sync %0;" :: "r"(0));
                        assert("Tensor range check" && 0 <= v101 && v101 < 2);
                        v85[v101] = v423;
                    }
                    int v424;
                    v424 = threadIdx.x;
                    int v425;
                    v425 = blockIdx.x;
                    int v426;
                    v426 = v425 * 256;
                    int v427;
                    v427 = v424 + v426;
                    assert("Tensor range check" && 0 <= v79 && v79 < 4);
                    int v428;
                    v428 = 12288 * v79;
                    assert("Tensor range check" && 0 <= v427 && v427 < 6144);
                    int v429;
                    v429 = 2 * v427;
                    int v430;
                    v430 = v429 + v428;
                    double * v431;
                    v431 = v73+v430;
                    double * v433;
                    v433 = v75+v430;
                    double * v435;
                    v435 = v431+0;
                    double * v437;
                    v437 = v433+0;
                    double * v439;
                    v439 = v431+0;
                    double * v441;
                    v441 = v433+0;
                    int v443;
                    v443 = sizeof(double *);
                    unsigned long long v444;
                    v444 = (unsigned long long)v443;
                    unsigned long long v445;
                    v445 = 256ull * v444;
                    unsigned long long v446;
                    v446 = v445 + 16ull;
                    unsigned long long v447;
                    v447 = v446 - 1ull;
                    unsigned long long v448;
                    v448 = v447 % 16ull;
                    unsigned long long v449;
                    v449 = v447 - v448;
                    unsigned long long v450;
                    v450 = v449 + v445;
                    unsigned long long v451;
                    v451 = v450 + 16ull;
                    unsigned long long v452;
                    v452 = v451 - 1ull;
                    unsigned long long v453;
                    v453 = v452 % 16ull;
                    unsigned long long v454;
                    v454 = v452 - v453;
                    unsigned long long v455;
                    v455 = v454 + v445;
                    unsigned long long v456;
                    v456 = v455 + 16ull;
                    unsigned long long v457;
                    v457 = v456 - 1ull;
                    unsigned long long v458;
                    v458 = v457 % 16ull;
                    unsigned long long v459;
                    v459 = v457 - v458;
                    unsigned long long v460;
                    v460 = v459 + v445;
                    bool v461;
                    v461 = v460 <= 98304ull;
                    bool v462;
                    v462 = v461 == false;
                    if (v462){
                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v461);
                    } else {
                    }
                    extern __shared__ unsigned char v464[];
                    bool v465;
                    v465 = v460 <= v460;
                    bool v466;
                    v466 = v465 == false;
                    if (v466){
                        assert("The length of the partition has to be less than or equal to the length of the base array." && v465);
                    } else {
                    }
                    double * * v468;
                    v468 = reinterpret_cast<double * *>(&v464[0ull]);
                    double * * v470;
                    v470 = reinterpret_cast<double * *>(&v464[v449]);
                    double * * v472;
                    v472 = reinterpret_cast<double * *>(&v464[v454]);
                    double * * v474;
                    v474 = reinterpret_cast<double * *>(&v464[v459]);
                    int v476;
                    v476 = threadIdx.x;
                    assert("Tensor range check" && 0 <= v476 && v476 < 256);
                    v468[v476] = v435;
                    v470[v476] = v437;
                    v472[v476] = v439;
                    v474[v476] = v441;
                    asm("barrier.cta.sync %0;" :: "r"(0));
                    bool v477;
                    v477 = 0 <= v476;
                    bool v478;
                    v478 = v477 == false;
                    if (v478){
                        assert("The index needs to be zero or positive." && v477);
                    } else {
                    }
                    int v480;
                    v480 = v476 % 1;
                    bool v481;
                    v481 = v476 < 256;
                    bool v482;
                    v482 = v481 == false;
                    if (v482){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v481);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v476 && v476 < 256);
                    int v484;
                    v484 = 0;
                    while (while_method_6(v484)){
                        bool v486;
                        v486 = v477 && v481;
                        bool v487;
                        v487 = v486 == false;
                        if (v487){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v486);
                        } else {
                        }
                        bool v489;
                        v489 = 0 <= v484;
                        bool v491;
                        if (v489){
                            bool v490;
                            v490 = v484 < 1;
                            v491 = v490;
                        } else {
                            v491 = false;
                        }
                        bool v492;
                        v492 = v491 == false;
                        if (v492){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v491);
                        } else {
                        }
                        int v494;
                        v494 = v484 * 256;
                        int v495;
                        v495 = v494 + v476;
                        assert("Tensor range check" && 0 <= v484 && v484 < 1);
                        int v496;
                        v496 = 256 * v484;
                        int v497;
                        v497 = v496 + v476;
                        double * v498;
                        v498 = v468[v497];
                        double * v499;
                        v499 = v470[v497];
                        double * v500;
                        v500 = v472[v497];
                        double * v501;
                        v501 = v474[v497];
                        int v502;
                        v502 = blockIdx.x;
                        int v503;
                        v503 = v502 * 256;
                        int v504;
                        v504 = v503 + v495;
                        assert("Tensor range check" && 0 <= v480 && v480 < 1);
                        int v505;
                        v505 = 2 * v480;
                        double v506[2];
                        double v507[2];
                        int v508[2];
                        int v509;
                        v509 = 0;
                        while (while_method_6(v509)){
                            assert("Tensor range check" && 0 <= v509 && v509 < 1);
                            int v511;
                            v511 = 2 * v509;
                            assert("Tensor range check" && 0 <= v509 && v509 < 1);
                            int v512;
                            v512 = v511 + v505;
                            int4* v513;
                            v513 = reinterpret_cast<int4*>(v498 + v512);
                            int4* v514;
                            v514 = reinterpret_cast<int4*>(v506 + v511);
                            assert("Pointer alignment check" && (unsigned long long)(v513) % 2 == 0 && (unsigned long long)(v514) % 2 == 0);
                            *v514 = *v513;
                            int4* v515;
                            v515 = reinterpret_cast<int4*>(v499 + v512);
                            int4* v516;
                            v516 = reinterpret_cast<int4*>(v507 + v511);
                            assert("Pointer alignment check" && (unsigned long long)(v515) % 2 == 0 && (unsigned long long)(v516) % 2 == 0);
                            *v516 = *v515;
                            v509 += 1 ;
                        }
                        int v517;
                        v517 = 0;
                        while (while_method_6(v517)){
                            int v519;
                            v519 = 0;
                            while (while_method_2(v519)){
                                bool v521;
                                v521 = 0 <= v519;
                                bool v523;
                                if (v521){
                                    bool v522;
                                    v522 = v519 < 2;
                                    v523 = v522;
                                } else {
                                    v523 = false;
                                }
                                bool v524;
                                v524 = v523 == false;
                                if (v524){
                                    assert("The indices should be inside the range of the dimension." && v523);
                                } else {
                                }
                                bool v526;
                                v526 = 0 <= v480;
                                bool v528;
                                if (v526){
                                    bool v527;
                                    v527 = v480 < 1;
                                    v528 = v527;
                                } else {
                                    v528 = false;
                                }
                                bool v529;
                                v529 = v528 == false;
                                if (v529){
                                    assert("The indices should be inside the range of the dimension." && v528);
                                } else {
                                }
                                int v531;
                                v531 = v480 * 2;
                                int v532;
                                v532 = v519 + v531;
                                bool v533;
                                v533 = 0 <= v517;
                                bool v535;
                                if (v533){
                                    bool v534;
                                    v534 = v517 < 1;
                                    v535 = v534;
                                } else {
                                    v535 = false;
                                }
                                bool v536;
                                v536 = v535 == false;
                                if (v536){
                                    assert("The indices should be inside the range of the dimension." && v535);
                                } else {
                                }
                                int v538;
                                v538 = v517 * 2;
                                int v539;
                                v539 = v532 + v538;
                                assert("Tensor range check" && 0 <= v517 && v517 < 1);
                                assert("Tensor range check" && 0 <= v519 && v519 < 2);
                                int v540;
                                v540 = 2 * v517;
                                int v541;
                                v541 = v540 + v519;
                                v508[v541] = v539;
                                v519 += 1 ;
                            }
                            v517 += 1 ;
                        }
                        double v542[2];
                        double v543[2];
                        int v544;
                        v544 = 0;
                        while (while_method_6(v544)){
                            int v546;
                            v546 = 0;
                            while (while_method_2(v546)){
                                assert("Tensor range check" && 0 <= v544 && v544 < 1);
                                assert("Tensor range check" && 0 <= v546 && v546 < 2);
                                int v548;
                                v548 = 2 * v544;
                                int v549;
                                v549 = v548 + v546;
                                double v550;
                                v550 = v506[v549];
                                double v551;
                                v551 = v507[v549];
                                assert("Tensor range check" && 0 <= v544 && v544 < 1);
                                assert("Tensor range check" && 0 <= v546 && v546 < 2);
                                v542[v549] = 0.0;
                                v543[v549] = 0.0;
                                v546 += 1 ;
                            }
                            v544 += 1 ;
                        }
                        int v552;
                        v552 = 0;
                        while (while_method_6(v552)){
                            assert("Tensor range check" && 0 <= v552 && v552 < 1);
                            int v554;
                            v554 = 2 * v552;
                            int v555;
                            v555 = v554 + v505;
                            assert("Tensor range check" && 0 <= v552 && v552 < 1);
                            int4* v556;
                            v556 = reinterpret_cast<int4*>(v542 + v554);
                            int4* v557;
                            v557 = reinterpret_cast<int4*>(v500 + v555);
                            assert("Pointer alignment check" && (unsigned long long)(v556) % 2 == 0 && (unsigned long long)(v557) % 2 == 0);
                            *v557 = *v556;
                            int4* v558;
                            v558 = reinterpret_cast<int4*>(v543 + v554);
                            int4* v559;
                            v559 = reinterpret_cast<int4*>(v501 + v555);
                            assert("Pointer alignment check" && (unsigned long long)(v558) % 2 == 0 && (unsigned long long)(v559) % 2 == 0);
                            *v559 = *v558;
                            v552 += 1 ;
                        }
                        assert("Tensor range check" && 0 <= v495 && v495 < 256);
                        v484 += 1 ;
                    }
                    asm("barrier.cta.sync %0;" :: "r"(0));
                    assert("Tensor range check" && 0 <= v476 && v476 < 256);
                    asm("barrier.cta.sync %0;" :: "r"(0));
                    assert("Tensor range check" && 0 <= v79 && v79 < 4);
                    assert("Tensor range check" && 0 <= v427 && v427 < 6144);
                    int v560;
                    v560 = v90 + v427;
                    v77[v560] = 0;
                    v79 += 1 ;
                }
                v27 += 1 ;
            }
            v25 += 1 ;
        }
        cooperative_groups::grid_group & v561 = v22.v1;
        cooperative_groups::grid_group & v562 = v561;
        curandStatePhilox4_32_10_t & v563 = v22.v5;
        curandStatePhilox4_32_10_t & v564 = v563;
        unsigned int * v565;
        v565 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
        int * v567;
        v567 = reinterpret_cast<int *>(&v1[262144ull]);
        float * v569;
        v569 = reinterpret_cast<float *>(&v1[262160ull]);
        float * v571;
        v571 = reinterpret_cast<float *>(&v1[524304ull]);
        float * v573;
        v573 = reinterpret_cast<float *>(&v1[786448ull]);
        float * v575;
        v575 = reinterpret_cast<float *>(&v1[1048592ull]);
        float * v577;
        v577 = reinterpret_cast<float *>(&v1[1310736ull]);
        float * v579;
        v579 = reinterpret_cast<float *>(&v1[1572880ull]);
        float * v581;
        v581 = reinterpret_cast<float *>(&v1[1835024ull]);
        int * v583;
        v583 = reinterpret_cast<int *>(&v0[6389760ull]);
        float * v585;
        v585 = reinterpret_cast<float *>(&v0[7962624ull]);
        int * v587;
        v587 = reinterpret_cast<int *>(&v0[9535488ull]);
        int * v589;
        v589 = reinterpret_cast<int *>(&v0[11108352ull]);
        double * v591;
        v591 = reinterpret_cast<double *>(&v0[12681216ull]);
        double * v593;
        v593 = reinterpret_cast<double *>(&v0[18972672ull]);
        double * v595;
        v595 = reinterpret_cast<double *>(&v1[2097168ull]);
        double * v597;
        v597 = reinterpret_cast<double *>(&v1[2490384ull]);
        int * v599;
        v599 = reinterpret_cast<int *>(&v1[2883600ull]);
        v562.sync() ;
        int v601;
        v601 = threadIdx.x;
        int v602;
        v602 = blockIdx.x;
        int v603;
        v603 = v602 * 256;
        int v604;
        v604 = v601 + v603;
        bool v605;
        v605 = v604 == 0;
        if (v605){
            int v606;
            v606 = 0;
            int v607;
            v607 = 4;
            int v608;
            v608 = int_range_6(v607, v606, v564);
            v567[0] = v608;
        } else {
        }
        __syncwarp();
        int v609;
        v609 = threadIdx.x;
        bool v610;
        v610 = 0 <= v609;
        bool v611;
        v611 = v610 == false;
        if (v611){
            assert("The index needs to be zero or positive." && v610);
        } else {
        }
        int v613;
        v613 = v609 % 1;
        int v614;
        v614 = v609 % 256;
        int v615;
        v615 = v609 / 256;
        bool v616;
        v616 = v615 < 1;
        bool v617;
        v617 = v616 == false;
        if (v617){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v616);
        } else {
        }
        assert("Tensor range check" && 0 <= v615 && v615 < 1);
        assert("Tensor range check" && 0 <= v614 && v614 < 256);
        assert("Tensor range check" && 0 <= v613 && v613 < 1);
        int v619;
        v619 = 4 * v613;
        int v620;
        v620 = 4 * v614;
        int v621;
        v621 = v620 + v619;
        int v622;
        v622 = 16384 * v615;
        int v623;
        v623 = v622 + v621;
        assert("Tensor range check" && 0 <= v615 && v615 < 1);
        assert("Tensor range check" && 0 <= v614 && v614 < 256);
        assert("Tensor range check" && 0 <= v613 && v613 < 1);
        int v624;
        v624 = blockIdx.x;
        int v625;
        v625 = v624;
        while (while_method_11(v625)){
            bool v627;
            v627 = 0 <= v625;
            bool v628;
            v628 = v627 == false;
            if (v628){
                assert("The index needs to be zero or positive." && v627);
            } else {
            }
            int v630;
            v630 = v625 % 16;
            int v631;
            v631 = v625 / 16;
            bool v632;
            v632 = v631 < 4;
            bool v633;
            v633 = v632 == false;
            if (v633){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v632);
            } else {
            }
            assert("Tensor range check" && 0 <= v631 && v631 < 4);
            assert("Tensor range check" && 0 <= v630 && v630 < 16);
            int v635;
            v635 = 1024 * v630;
            int v636;
            v636 = v635 + v623;
            int v637;
            v637 = 16384 * v631;
            int v638;
            v638 = v637 + v636;
            float v639[4];
            float v640[4];
            float v641[4];
            float v642[4];
            float v643[4];
            float v644[4];
            float v645[4];
            int v646[4];
            int v647;
            v647 = 0;
            while (while_method_6(v647)){
                assert("Tensor range check" && 0 <= v647 && v647 < 1);
                int v649;
                v649 = 4 * v647;
                assert("Tensor range check" && 0 <= v647 && v647 < 1);
                int v650;
                v650 = v649 + v638;
                int4* v651;
                v651 = reinterpret_cast<int4*>(v569 + v650);
                int4* v652;
                v652 = reinterpret_cast<int4*>(v639 + v649);
                assert("Pointer alignment check" && (unsigned long long)(v651) % 4 == 0 && (unsigned long long)(v652) % 4 == 0);
                *v652 = *v651;
                int4* v653;
                v653 = reinterpret_cast<int4*>(v571 + v650);
                int4* v654;
                v654 = reinterpret_cast<int4*>(v640 + v649);
                assert("Pointer alignment check" && (unsigned long long)(v653) % 4 == 0 && (unsigned long long)(v654) % 4 == 0);
                *v654 = *v653;
                int4* v655;
                v655 = reinterpret_cast<int4*>(v573 + v650);
                int4* v656;
                v656 = reinterpret_cast<int4*>(v641 + v649);
                assert("Pointer alignment check" && (unsigned long long)(v655) % 4 == 0 && (unsigned long long)(v656) % 4 == 0);
                *v656 = *v655;
                int4* v657;
                v657 = reinterpret_cast<int4*>(v575 + v650);
                int4* v658;
                v658 = reinterpret_cast<int4*>(v642 + v649);
                assert("Pointer alignment check" && (unsigned long long)(v657) % 4 == 0 && (unsigned long long)(v658) % 4 == 0);
                *v658 = *v657;
                int4* v659;
                v659 = reinterpret_cast<int4*>(v577 + v650);
                int4* v660;
                v660 = reinterpret_cast<int4*>(v643 + v649);
                assert("Pointer alignment check" && (unsigned long long)(v659) % 4 == 0 && (unsigned long long)(v660) % 4 == 0);
                *v660 = *v659;
                int4* v661;
                v661 = reinterpret_cast<int4*>(v579 + v650);
                int4* v662;
                v662 = reinterpret_cast<int4*>(v644 + v649);
                assert("Pointer alignment check" && (unsigned long long)(v661) % 4 == 0 && (unsigned long long)(v662) % 4 == 0);
                *v662 = *v661;
                int4* v663;
                v663 = reinterpret_cast<int4*>(v581 + v650);
                int4* v664;
                v664 = reinterpret_cast<int4*>(v645 + v649);
                assert("Pointer alignment check" && (unsigned long long)(v663) % 4 == 0 && (unsigned long long)(v664) % 4 == 0);
                *v664 = *v663;
                v647 += 1 ;
            }
            int v665;
            v665 = 0;
            while (while_method_6(v665)){
                int v667;
                v667 = 0;
                while (while_method_0(v667)){
                    bool v669;
                    v669 = 0 <= v667;
                    bool v671;
                    if (v669){
                        bool v670;
                        v670 = v667 < 4;
                        v671 = v670;
                    } else {
                        v671 = false;
                    }
                    bool v672;
                    v672 = v671 == false;
                    if (v672){
                        assert("The indices should be inside the range of the dimension." && v671);
                    } else {
                    }
                    bool v674;
                    v674 = 0 <= v613;
                    bool v676;
                    if (v674){
                        bool v675;
                        v675 = v613 < 1;
                        v676 = v675;
                    } else {
                        v676 = false;
                    }
                    bool v677;
                    v677 = v676 == false;
                    if (v677){
                        assert("The indices should be inside the range of the dimension." && v676);
                    } else {
                    }
                    int v679;
                    v679 = v613 * 4;
                    int v680;
                    v680 = v667 + v679;
                    bool v681;
                    v681 = 0 <= v665;
                    bool v683;
                    if (v681){
                        bool v682;
                        v682 = v665 < 1;
                        v683 = v682;
                    } else {
                        v683 = false;
                    }
                    bool v684;
                    v684 = v683 == false;
                    if (v684){
                        assert("The indices should be inside the range of the dimension." && v683);
                    } else {
                    }
                    int v686;
                    v686 = v665 * 4;
                    int v687;
                    v687 = v680 + v686;
                    assert("Tensor range check" && 0 <= v665 && v665 < 1);
                    assert("Tensor range check" && 0 <= v667 && v667 < 4);
                    int v688;
                    v688 = 4 * v665;
                    int v689;
                    v689 = v688 + v667;
                    v646[v689] = v687;
                    v667 += 1 ;
                }
                v665 += 1 ;
            }
            bool v690;
            v690 = 0 <= v615;
            bool v691;
            v691 = v690 && v616;
            bool v692;
            v692 = v691 == false;
            if (v692){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v691);
            } else {
            }
            bool v694;
            v694 = 0 <= v614;
            bool v696;
            if (v694){
                bool v695;
                v695 = v614 < 256;
                v696 = v695;
            } else {
                v696 = false;
            }
            bool v697;
            v697 = v696 == false;
            if (v697){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v696);
            } else {
            }
            bool v699;
            v699 = 0 <= v631;
            bool v700;
            v700 = v699 && v632;
            bool v701;
            v701 = v700 == false;
            if (v701){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v700);
            } else {
            }
            bool v703;
            v703 = 0 <= v630;
            bool v705;
            if (v703){
                bool v704;
                v704 = v630 < 16;
                v705 = v704;
            } else {
                v705 = false;
            }
            bool v706;
            v706 = v705 == false;
            if (v706){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v705);
            } else {
            }
            int v708;
            v708 = v630 * 256;
            int v709;
            v709 = v631 + v615;
            int v710;
            v710 = v708 + v614;
            bool v711[4];
            int v712;
            v712 = 0;
            while (while_method_6(v712)){
                int v714;
                v714 = 0;
                while (while_method_0(v714)){
                    assert("Tensor range check" && 0 <= v712 && v712 < 1);
                    assert("Tensor range check" && 0 <= v714 && v714 < 4);
                    int v716;
                    v716 = 4 * v712;
                    int v717;
                    v717 = v716 + v714;
                    float v718;
                    v718 = v641[v717];
                    bool v719;
                    v719 = v718 == 0.0f;
                    bool v720;
                    v720 = v719 != true;
                    assert("Tensor range check" && 0 <= v712 && v712 < 1);
                    assert("Tensor range check" && 0 <= v714 && v714 < 4);
                    v711[v717] = v720;
                    v714 += 1 ;
                }
                v712 += 1 ;
            }
            bool v721;
            v721 = false;
            int v722;
            v722 = 0;
            while (while_method_6(v722)){
                int v724;
                v724 = 0;
                while (while_method_0(v724)){
                    assert("Tensor range check" && 0 <= v722 && v722 < 1);
                    assert("Tensor range check" && 0 <= v724 && v724 < 4);
                    int v726;
                    v726 = 4 * v722;
                    int v727;
                    v727 = v726 + v724;
                    bool v728;
                    v728 = v711[v727];
                    bool v729;
                    v729 = v721 || v728;
                    v721 = v729;
                    v724 += 1 ;
                }
                v722 += 1 ;
            }
            auto v730 = cooperative_groups::coalesced_threads();
            int v731;
            v731 = threadIdx.x;
            auto v732 = cooperative_groups::labeled_partition(v730,v731);
            Closure8 v733{};
            bool v734;
            v734 = cooperative_groups::reduce(v732, v721, v733);
            if (v734){
                float v735[4];
                int v736;
                v736 = 0;
                while (while_method_6(v736)){
                    int v738;
                    v738 = 0;
                    while (while_method_0(v738)){
                        assert("Tensor range check" && 0 <= v736 && v736 < 1);
                        assert("Tensor range check" && 0 <= v738 && v738 < 4);
                        int v740;
                        v740 = 4 * v736;
                        int v741;
                        v741 = v740 + v738;
                        float v742;
                        v742 = v640[v741];
                        float v743;
                        v743 = v641[v741];
                        float v744;
                        v744 = v742 + v743;
                        bool v745;
                        v745 = 0.0f >= v744;
                        float v746;
                        if (v745){
                            v746 = 0.0f;
                        } else {
                            v746 = v744;
                        }
                        assert("Tensor range check" && 0 <= v736 && v736 < 1);
                        assert("Tensor range check" && 0 <= v738 && v738 < 4);
                        v735[v741] = v746;
                        v738 += 1 ;
                    }
                    v736 += 1 ;
                }
                float v747[4];
                int v748;
                v748 = 0;
                while (while_method_6(v748)){
                    int v750;
                    v750 = 0;
                    while (while_method_0(v750)){
                        assert("Tensor range check" && 0 <= v748 && v748 < 1);
                        assert("Tensor range check" && 0 <= v750 && v750 < 4);
                        int v752;
                        v752 = 4 * v748;
                        int v753;
                        v753 = v752 + v750;
                        float v754;
                        v754 = v735[v753];
                        bool v755;
                        v755 = 0.0f >= v754;
                        float v756;
                        if (v755){
                            v756 = 0.0f;
                        } else {
                            v756 = v754;
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
                while (while_method_6(v758)){
                    int v760;
                    v760 = 0;
                    while (while_method_0(v760)){
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
                auto v768 = cooperative_groups::labeled_partition(v766,v767);
                Closure1 v769{};
                float v770;
                v770 = cooperative_groups::reduce(v768, v757, v769);
                float v771[4];
                int v772;
                v772 = 0;
                while (while_method_6(v772)){
                    int v774;
                    v774 = 0;
                    while (while_method_0(v774)){
                        assert("Tensor range check" && 0 <= v772 && v772 < 1);
                        assert("Tensor range check" && 0 <= v774 && v774 < 4);
                        int v776;
                        v776 = 4 * v772;
                        int v777;
                        v777 = v776 + v774;
                        float v778;
                        v778 = v747[v777];
                        bool v779;
                        v779 = v770 == 0.0f;
                        bool v780;
                        v780 = v779 != true;
                        float v782;
                        if (v780){
                            float v781;
                            v781 = v778 / v770;
                            v782 = v781;
                        } else {
                            v782 = 0.25f;
                        }
                        assert("Tensor range check" && 0 <= v772 && v772 < 1);
                        assert("Tensor range check" && 0 <= v774 && v774 < 4);
                        v771[v777] = v782;
                        v774 += 1 ;
                    }
                    v772 += 1 ;
                }
                float v783[4];
                int v784;
                v784 = 0;
                while (while_method_6(v784)){
                    int v786;
                    v786 = 0;
                    while (while_method_0(v786)){
                        assert("Tensor range check" && 0 <= v784 && v784 < 1);
                        assert("Tensor range check" && 0 <= v786 && v786 < 4);
                        int v788;
                        v788 = 4 * v784;
                        int v789;
                        v789 = v788 + v786;
                        float v790;
                        v790 = v639[v789];
                        float v791;
                        v791 = v771[v789];
                        float v792;
                        v792 = v790 + v791;
                        assert("Tensor range check" && 0 <= v784 && v784 < 1);
                        assert("Tensor range check" && 0 <= v786 && v786 < 4);
                        v783[v789] = v792;
                        v786 += 1 ;
                    }
                    v784 += 1 ;
                }
                float v793[4];
                int v794;
                v794 = 0;
                while (while_method_6(v794)){
                    int v796;
                    v796 = 0;
                    while (while_method_0(v796)){
                        assert("Tensor range check" && 0 <= v794 && v794 < 1);
                        assert("Tensor range check" && 0 <= v796 && v796 < 4);
                        int v798;
                        v798 = 4 * v794;
                        int v799;
                        v799 = v798 + v796;
                        float v800;
                        v800 = v783[v799];
                        float v801;
                        v801 = -v800;
                        bool v802;
                        v802 = v800 >= v801;
                        float v803;
                        if (v802){
                            v803 = v800;
                        } else {
                            v803 = v801;
                        }
                        assert("Tensor range check" && 0 <= v794 && v794 < 1);
                        assert("Tensor range check" && 0 <= v796 && v796 < 4);
                        v793[v799] = v803;
                        v796 += 1 ;
                    }
                    v794 += 1 ;
                }
                float v804;
                v804 = 0.0f;
                int v805;
                v805 = 0;
                while (while_method_6(v805)){
                    int v807;
                    v807 = 0;
                    while (while_method_0(v807)){
                        assert("Tensor range check" && 0 <= v805 && v805 < 1);
                        assert("Tensor range check" && 0 <= v807 && v807 < 4);
                        int v809;
                        v809 = 4 * v805;
                        int v810;
                        v810 = v809 + v807;
                        float v811;
                        v811 = v793[v810];
                        float v812;
                        v812 = v804 + v811;
                        v804 = v812;
                        v807 += 1 ;
                    }
                    v805 += 1 ;
                }
                auto v813 = cooperative_groups::coalesced_threads();
                int v814;
                v814 = threadIdx.x;
                auto v815 = cooperative_groups::labeled_partition(v813,v814);
                float v816;
                v816 = cooperative_groups::reduce(v815, v804, v769);
                bool v817;
                v817 = v816 > 100.0f;
                float v819;
                if (v817){
                    float v818;
                    v818 = 100.0f / v816;
                    v819 = v818;
                } else {
                    v819 = 1.0f;
                }
                float v820[4];
                int v821;
                v821 = 0;
                while (while_method_6(v821)){
                    int v823;
                    v823 = 0;
                    while (while_method_0(v823)){
                        assert("Tensor range check" && 0 <= v821 && v821 < 1);
                        assert("Tensor range check" && 0 <= v823 && v823 < 4);
                        int v825;
                        v825 = 4 * v821;
                        int v826;
                        v826 = v825 + v823;
                        float v827;
                        v827 = v793[v826];
                        float v828;
                        v828 = v819 * v827;
                        assert("Tensor range check" && 0 <= v821 && v821 < 1);
                        assert("Tensor range check" && 0 <= v823 && v823 < 4);
                        v820[v826] = v828;
                        v823 += 1 ;
                    }
                    v821 += 1 ;
                }
                float v829[4];
                float v830[4];
                int v831;
                v831 = 0;
                while (while_method_6(v831)){
                    int v833;
                    v833 = 0;
                    while (while_method_0(v833)){
                        assert("Tensor range check" && 0 <= v831 && v831 < 1);
                        assert("Tensor range check" && 0 <= v833 && v833 < 4);
                        int v835;
                        v835 = 4 * v831;
                        int v836;
                        v836 = v835 + v833;
                        float v837;
                        v837 = v639[v836];
                        float v838;
                        v838 = v640[v836];
                        float v839;
                        v839 = v641[v836];
                        float v840;
                        v840 = v642[v836];
                        float v841;
                        v841 = v643[v836];
                        float v842;
                        v842 = v644[v836];
                        float v843;
                        v843 = v645[v836];
                        float v844;
                        v844 = v840 + v842;
                        float v845;
                        v845 = v841 + v843;
                        assert("Tensor range check" && 0 <= v831 && v831 < 1);
                        assert("Tensor range check" && 0 <= v833 && v833 < 4);
                        v829[v836] = v844;
                        v830[v836] = v845;
                        v833 += 1 ;
                    }
                    v831 += 1 ;
                }
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
                        v852 = v820[v851];
                        float v853;
                        v853 = v735[v851];
                        float v854;
                        v854 = v829[v851];
                        float v855;
                        v855 = v830[v851];
                        assert("Tensor range check" && 0 <= v846 && v846 < 1);
                        assert("Tensor range check" && 0 <= v848 && v848 < 4);
                        v639[v851] = v852;
                        v640[v851] = v853;
                        v641[v851] = 0.0f;
                        v642[v851] = v854;
                        v643[v851] = v855;
                        v644[v851] = 0.0f;
                        v645[v851] = 0.0f;
                        v848 += 1 ;
                    }
                    v846 += 1 ;
                }
            } else {
            }
            assert("Tensor range check" && 0 <= v631 && v631 < 4);
            assert("Tensor range check" && 0 <= v630 && v630 < 16);
            int v856;
            v856 = 0;
            while (while_method_6(v856)){
                assert("Tensor range check" && 0 <= v856 && v856 < 1);
                int v858;
                v858 = 4 * v856;
                int v859;
                v859 = v858 + v638;
                assert("Tensor range check" && 0 <= v856 && v856 < 1);
                int4* v860;
                v860 = reinterpret_cast<int4*>(v639 + v858);
                int4* v861;
                v861 = reinterpret_cast<int4*>(v569 + v859);
                assert("Pointer alignment check" && (unsigned long long)(v860) % 4 == 0 && (unsigned long long)(v861) % 4 == 0);
                *v861 = *v860;
                int4* v862;
                v862 = reinterpret_cast<int4*>(v640 + v858);
                int4* v863;
                v863 = reinterpret_cast<int4*>(v571 + v859);
                assert("Pointer alignment check" && (unsigned long long)(v862) % 4 == 0 && (unsigned long long)(v863) % 4 == 0);
                *v863 = *v862;
                int4* v864;
                v864 = reinterpret_cast<int4*>(v641 + v858);
                int4* v865;
                v865 = reinterpret_cast<int4*>(v573 + v859);
                assert("Pointer alignment check" && (unsigned long long)(v864) % 4 == 0 && (unsigned long long)(v865) % 4 == 0);
                *v865 = *v864;
                int4* v866;
                v866 = reinterpret_cast<int4*>(v642 + v858);
                int4* v867;
                v867 = reinterpret_cast<int4*>(v575 + v859);
                assert("Pointer alignment check" && (unsigned long long)(v866) % 4 == 0 && (unsigned long long)(v867) % 4 == 0);
                *v867 = *v866;
                int4* v868;
                v868 = reinterpret_cast<int4*>(v643 + v858);
                int4* v869;
                v869 = reinterpret_cast<int4*>(v577 + v859);
                assert("Pointer alignment check" && (unsigned long long)(v868) % 4 == 0 && (unsigned long long)(v869) % 4 == 0);
                *v869 = *v868;
                int4* v870;
                v870 = reinterpret_cast<int4*>(v644 + v858);
                int4* v871;
                v871 = reinterpret_cast<int4*>(v579 + v859);
                assert("Pointer alignment check" && (unsigned long long)(v870) % 4 == 0 && (unsigned long long)(v871) % 4 == 0);
                *v871 = *v870;
                int4* v872;
                v872 = reinterpret_cast<int4*>(v645 + v858);
                int4* v873;
                v873 = reinterpret_cast<int4*>(v581 + v859);
                assert("Pointer alignment check" && (unsigned long long)(v872) % 4 == 0 && (unsigned long long)(v873) % 4 == 0);
                *v873 = *v872;
                v856 += 1 ;
            }
            v625 += 24 ;
        }
        v562.sync() ;
        v23 += 1 ;
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
i8 = int; i16 = int; i32 = int; i64 = int; u8 = int; u16 = int; u32 = int; u64 = int; f32 = float; f64 = float; char = str; string = str

import time
options = []
options.append('--define-macro=NDEBUG')
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39,177')
options.append('--restrict')
options.append('--maxrregcount=255')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
class US0_0(NamedTuple): # Computer
    tag = 0
class US0_1(NamedTuple): # Human
    tag = 1
class US0_2(NamedTuple): # Random
    tag = 2
US0 = Union[US0_0, US0_1, US0_2]
class US4_0(NamedTuple): # Jack
    tag = 0
class US4_1(NamedTuple): # King
    tag = 1
class US4_2(NamedTuple): # Queen
    tag = 2
US4 = Union[US4_0, US4_1, US4_2]
class US3_0(NamedTuple): # None
    tag = 0
class US3_1(NamedTuple): # Some
    v0 : US4
    tag = 1
US3 = Union[US3_0, US3_1]
class US5_0(NamedTuple): # Call
    tag = 0
class US5_1(NamedTuple): # Fold
    tag = 1
class US5_2(NamedTuple): # Raise
    tag = 2
US5 = Union[US5_0, US5_1, US5_2]
class US2_0(NamedTuple): # ChanceCommunityCard
    v0 : US3
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 0
class US2_1(NamedTuple): # ChanceInit
    tag = 1
class US2_2(NamedTuple): # Round
    v0 : US3
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 2
class US2_3(NamedTuple): # RoundWithAction
    v0 : US3
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    v6 : US5
    tag = 3
class US2_4(NamedTuple): # TerminalCall
    v0 : US3
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 4
class US2_5(NamedTuple): # TerminalFold
    v0 : US3
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 5
US2 = Union[US2_0, US2_1, US2_2, US2_3, US2_4, US2_5]
class US1_0(NamedTuple): # None
    tag = 0
class US1_1(NamedTuple): # Some
    v0 : US2
    tag = 1
US1 = Union[US1_0, US1_1]
class US6_0(NamedTuple): # GameNotStarted
    tag = 0
class US6_1(NamedTuple): # GameOver
    v0 : US3
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 1
class US6_2(NamedTuple): # WaitingForActionFromPlayerId
    v0 : US3
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 2
US6 = Union[US6_0, US6_1, US6_2]
class US7_0(NamedTuple): # CommunityCardIs
    v0 : US4
    tag = 0
class US7_1(NamedTuple): # PlayerAction
    v0 : i32
    v1 : US5
    tag = 1
class US7_2(NamedTuple): # PlayerGotCard
    v0 : i32
    v1 : US4
    tag = 2
class US7_3(NamedTuple): # Showdown
    v0 : static_array
    v1 : i32
    v2 : i32
    tag = 3
US7 = Union[US7_0, US7_1, US7_2, US7_3]
class US8_0(NamedTuple): # AddRewardsRando
    v0 : list
    tag = 0
class US8_1(NamedTuple): # AddRewardsSelf
    v0 : list
    tag = 1
US8 = Union[US8_0, US8_1]
def method1(v0 : cp.ndarray, v1 : u32) -> None:
    v3 = v0[0:].view(cp.uint32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method2(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[4:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method3(v0 : cp.ndarray) -> None:
    del v0
    return 
def method5(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[0:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method7(v0 : cp.ndarray, v1 : US4) -> None:
    v2 = v1.tag
    method5(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US4_0(): # Jack
            del v1
            return method3(v4)
        case US4_1(): # King
            del v1
            return method3(v4)
        case US4_2(): # Queen
            del v1
            return method3(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method8(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method6(v0 : cp.ndarray, v1 : US3, v2 : bool, v3 : static_array, v4 : i32, v5 : static_array, v6 : i32) -> None:
    v7 = v1.tag
    method5(v0, v7)
    del v7
    v9 = v0[4:].view(cp.uint8)
    match v1:
        case US3_0(): # None
            method3(v9)
        case US3_1(v10): # Some
            method7(v9, v10)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v1, v9
    v12 = v0[8:].view(cp.bool_)
    v12[0] = v2
    del v2, v12
    v13 = 0
    while method8(v13):
        v15 = u64(v13)
        v16 = v15 * 4
        del v15
        v17 = 12 + v16
        del v16
        v19 = v0[v17:].view(cp.uint8)
        del v17
        v21 = v3[v13]
        method7(v19, v21)
        del v19, v21
        v13 += 1 
    del v3, v13
    v23 = v0[20:].view(cp.int32)
    v23[0] = v4
    del v4, v23
    v24 = 0
    while method8(v24):
        v26 = u64(v24)
        v27 = v26 * 4
        del v26
        v28 = 24 + v27
        del v27
        v30 = v0[v28:].view(cp.uint8)
        del v28
        v32 = v5[v24]
        method5(v30, v32)
        del v30, v32
        v24 += 1 
    del v5, v24
    v34 = v0[32:].view(cp.int32)
    del v0
    v34[0] = v6
    del v6, v34
    return 
def method10(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[36:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method9(v0 : cp.ndarray, v1 : US3, v2 : bool, v3 : static_array, v4 : i32, v5 : static_array, v6 : i32, v7 : US5) -> None:
    v8 = v1.tag
    method5(v0, v8)
    del v8
    v10 = v0[4:].view(cp.uint8)
    match v1:
        case US3_0(): # None
            method3(v10)
        case US3_1(v11): # Some
            method7(v10, v11)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v1, v10
    v13 = v0[8:].view(cp.bool_)
    v13[0] = v2
    del v2, v13
    v14 = 0
    while method8(v14):
        v16 = u64(v14)
        v17 = v16 * 4
        del v16
        v18 = 12 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v22 = v3[v14]
        method7(v20, v22)
        del v20, v22
        v14 += 1 
    del v3, v14
    v24 = v0[20:].view(cp.int32)
    v24[0] = v4
    del v4, v24
    v25 = 0
    while method8(v25):
        v27 = u64(v25)
        v28 = v27 * 4
        del v27
        v29 = 24 + v28
        del v28
        v31 = v0[v29:].view(cp.uint8)
        del v29
        v33 = v5[v25]
        method5(v31, v33)
        del v31, v33
        v25 += 1 
    del v5, v25
    v35 = v0[32:].view(cp.int32)
    v35[0] = v6
    del v6, v35
    v36 = v7.tag
    method10(v0, v36)
    del v36
    v38 = v0[40:].view(cp.uint8)
    del v0
    match v7:
        case US5_0(): # Call
            del v7
            return method3(v38)
        case US5_1(): # Fold
            del v7
            return method3(v38)
        case US5_2(): # Raise
            del v7
            return method3(v38)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method4(v0 : cp.ndarray, v1 : US2) -> None:
    v2 = v1.tag
    method5(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US2_0(v5, v6, v7, v8, v9, v10): # ChanceCommunityCard
            del v1
            return method6(v4, v5, v6, v7, v8, v9, v10)
        case US2_1(): # ChanceInit
            del v1
            return method3(v4)
        case US2_2(v11, v12, v13, v14, v15, v16): # Round
            del v1
            return method6(v4, v11, v12, v13, v14, v15, v16)
        case US2_3(v17, v18, v19, v20, v21, v22, v23): # RoundWithAction
            del v1
            return method9(v4, v17, v18, v19, v20, v21, v22, v23)
        case US2_4(v24, v25, v26, v27, v28, v29): # TerminalCall
            del v1
            return method6(v4, v24, v25, v26, v27, v28, v29)
        case US2_5(v30, v31, v32, v33, v34, v35): # TerminalFold
            del v1
            return method6(v4, v30, v31, v32, v33, v34, v35)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method11(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[80:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method12(v0 : i32, v1 : i32) -> bool:
    v2 = v1 < v0
    del v0, v1
    return v2
def method14(v0 : cp.ndarray, v1 : i32, v2 : US5) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = v2.tag
    method2(v0, v5)
    del v5
    v7 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US5_0(): # Call
            del v2
            return method3(v7)
        case US5_1(): # Fold
            del v2
            return method3(v7)
        case US5_2(): # Raise
            del v2
            return method3(v7)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method15(v0 : cp.ndarray, v1 : i32, v2 : US4) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = v2.tag
    method2(v0, v5)
    del v5
    v7 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US4_0(): # Jack
            del v2
            return method3(v7)
        case US4_1(): # King
            del v2
            return method3(v7)
        case US4_2(): # Queen
            del v2
            return method3(v7)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method16(v0 : cp.ndarray, v1 : static_array, v2 : i32, v3 : i32) -> None:
    v4 = 0
    while method8(v4):
        v6 = u64(v4)
        v7 = v6 * 4
        del v6
        v9 = v0[v7:].view(cp.uint8)
        del v7
        v11 = v1[v4]
        method7(v9, v11)
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
def method13(v0 : cp.ndarray, v1 : US7) -> None:
    v2 = v1.tag
    method5(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US7_0(v5): # CommunityCardIs
            del v1
            return method7(v4, v5)
        case US7_1(v6, v7): # PlayerAction
            del v1
            return method14(v4, v6, v7)
        case US7_2(v8, v9): # PlayerGotCard
            del v1
            return method15(v4, v8, v9)
        case US7_3(v10, v11, v12): # Showdown
            del v1
            return method16(v4, v10, v11, v12)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method17(v0 : cp.ndarray, v1 : US0) -> None:
    v2 = v1.tag
    method5(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US0_0(): # Computer
            del v1
            return method3(v4)
        case US0_1(): # Human
            del v1
            return method3(v4)
        case US0_2(): # Random
            del v1
            return method3(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method18(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[1128:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method0(v0 : cp.ndarray, v1 : u32, v2 : US1, v3 : static_array_list, v4 : static_array, v5 : US6) -> None:
    method1(v0, v1)
    del v1
    v6 = v2.tag
    method2(v0, v6)
    del v6
    v8 = v0[16:].view(cp.uint8)
    match v2:
        case US1_0(): # None
            method3(v8)
        case US1_1(v9): # Some
            method4(v8, v9)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v2, v8
    v10 = v3.length
    method11(v0, v10)
    del v10
    v11 = v3.length
    v12 = 0
    while method12(v11, v12):
        v14 = u64(v12)
        v15 = v14 * 32
        del v14
        v16 = 96 + v15
        del v15
        v18 = v0[v16:].view(cp.uint8)
        del v16
        v20 = v3[v12]
        method13(v18, v20)
        del v18, v20
        v12 += 1 
    del v3, v11, v12
    v21 = 0
    while method8(v21):
        v23 = u64(v21)
        v24 = v23 * 4
        del v23
        v25 = 1120 + v24
        del v24
        v27 = v0[v25:].view(cp.uint8)
        del v25
        v29 = v4[v21]
        method17(v27, v29)
        del v27, v29
        v21 += 1 
    del v4, v21
    v30 = v5.tag
    method18(v0, v30)
    del v30
    v32 = v0[1136:].view(cp.uint8)
    del v0
    match v5:
        case US6_0(): # GameNotStarted
            del v5
            return method3(v32)
        case US6_1(v33, v34, v35, v36, v37, v38): # GameOver
            del v5
            return method6(v32, v33, v34, v35, v36, v37, v38)
        case US6_2(v39, v40, v41, v42, v43, v44): # WaitingForActionFromPlayerId
            del v5
            return method6(v32, v39, v40, v41, v42, v43, v44)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method19(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def main_body():
    v1 = static_array(2)
    v3 = US0_0()
    v1[0] = v3
    del v3
    v5 = US0_1()
    v1[1] = v5
    del v5
    v7 = static_array_list(32)
    v8 = cp.empty(2981904,dtype=cp.uint8)
    v9 = cp.empty(25264128,dtype=cp.uint8)
    v11 = v8[0:0+4*65536].view(cp.float32)
    v12 = cp.random.normal(0.0,0.00390625,65536,dtype=cp.float32) # type: ignore
    cp.copyto(v11[0:0+65536],v12[0:0+65536])
    del v11, v12
    v14 = v8[262144:262144+4*1].view(cp.int32)
    v16 = v8[262160:262160+4*65536].view(cp.float32)
    v18 = v8[524304:524304+4*65536].view(cp.float32)
    v20 = v8[786448:786448+4*65536].view(cp.float32)
    v22 = v8[1048592:1048592+4*65536].view(cp.float32)
    v24 = v8[1310736:1310736+4*65536].view(cp.float32)
    v26 = v8[1572880:1572880+4*65536].view(cp.float32)
    v28 = v8[1835024:1835024+4*65536].view(cp.float32)
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
    v30 = v8[2097168:2097168+8*49152].view(cp.float64)
    v32 = v8[2490384:2490384+8*49152].view(cp.float64)
    v34 = v8[2883600:2883600+4*24576].view(cp.int32)
    v30[:] = 0
    del v30
    v32[:] = 0
    del v32
    v34[:] = 0
    del v34
    v35 = cp.empty(1184,dtype=cp.uint8)
    v36 = 63
    v37 = US1_0()
    v38 = US6_0()
    method0(v35, v36, v37, v7, v1, v38)
    del v1, v7, v35, v36, v37, v38
    v41 = "{}\n"
    v42 = "Going to run the Leduc full kernel (performance)."
    print(v41.format(v42),end="")
    del v41, v42
    v43 = time.perf_counter()
    v44 = []
    v45 = cp.zeros(16,dtype=cp.float32) # type: ignore
    del v45
    v46 = cp.zeros(16,dtype=cp.float32) # type: ignore
    del v46
    v47 = cp.empty(16,dtype=cp.float32)
    v48 = cp.cuda.Device().attributes['MultiProcessorCount']
    v49 = v48 == 24
    del v48
    v50 = v49 == False
    if v50:
        v51 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v49, v51
        del v51
    else:
        pass
    del v49, v50
    v52 = 0
    v53 = raw_module.get_function(f"entry{v52}")
    del v52
    v53.max_dynamic_shared_size_bytes = 98304 
    print(f'Threads per block, blocks per grid: {256}, {24}')
    v53((24,),(256,),(v9, v8),shared_mem=98304)
    del v8, v9, v53
    v54 = []
    v56 = v47[0:]
    del v47
    v57 = v56.get()
    del v56
    v58 = 0
    while method19(v58):
        v60 = []
        v61 = 0
        while method19(v61):
            assert 0 <= v58 < 4, 'Tensor range check'
            assert 0 <= v61 < 4, 'Tensor range check'
            v63 = 4 * v58
            v64 = v63 + v61
            del v63
            v65 = v57[v64].item()
            del v64
            v60.append(v65)
            del v65
            v61 += 1 
        del v61
        v54.append(v60)
        del v60
        v58 += 1 
    del v57, v58
    v66 = US8_0(v54)
    del v54
    v44.append(v66)
    del v44, v66
    cp.cuda.get_current_stream().synchronize()
    v69 = "{}"
    v70 = "The time it took to run the kernel (in seconds) is: "
    print(v69.format(v70),end="")
    del v69, v70
    v71 = time.perf_counter()
    v72 = v71 - v43
    del v43, v71
    v75 = "{:.6f}\n"
    print(v75.format(v72),end="")
    del v72, v75
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
