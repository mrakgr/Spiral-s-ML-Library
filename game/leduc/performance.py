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
struct Mut0;
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
__device__ void method_0(unsigned char * v0, unsigned char * v1, sptr<Mut0> v2, Union4 v3);
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
struct Mut0 {
    int refc{0};
    cooperative_groups::grid_group v1;
    static_array_list<Union1,32> v2;
    static_array<Union0,2> v3;
    static_array<float,2> v4;
    curandStatePhilox4_32_10_t v5;
    unsigned int v0;
    __device__ Mut0() = default;
    __device__ Mut0(unsigned int t0, cooperative_groups::grid_group t1, static_array_list<Union1,32> t2, static_array<Union0,2> t3, static_array<float,2> t4, curandStatePhilox4_32_10_t t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
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
__device__ inline bool while_method_2(Union6 v0){
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
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 2;
    return v1;
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
    v1 = v0 < 8;
    return v1;
}
__device__ inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 1;
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
        while (while_method_7(v28)){
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
        while (while_method_7(v35)){
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
        while (while_method_7(v72)){
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
        while (while_method_7(v84)){
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
    while (while_method_7(v59)){
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
        while (while_method_7(v82)){
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
        while (while_method_7(v90)){
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
        while (while_method_7(v116)){
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
        while (while_method_7(v126)){
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
        while (while_method_7(v138)){
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
        while (while_method_7(v152)){
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
        while (while_method_7(v161)){
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
        while (while_method_7(v177)){
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
        while (while_method_7(v193)){
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
        while (while_method_7(v221)){
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
        while (while_method_7(v232)){
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
        while (while_method_7(v258)){
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
        while (while_method_7(v268)){
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
        while (while_method_7(v288)){
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
        while (while_method_7(v303)){
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
        while (while_method_7(v328)){
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
        while (while_method_7(v338)){
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
        while (while_method_7(v350)){
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
        while (while_method_7(v363)){
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
        while (while_method_7(v372)){
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
        while (while_method_7(v387)){
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
        while (while_method_7(v403)){
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
        while (while_method_7(v427)){
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
    while (while_method_7(v46)){
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
        while (while_method_7(v68)){
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
        while (while_method_7(v74)){
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
        while (while_method_7(v100)){
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
        while (while_method_7(v110)){
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
        while (while_method_7(v122)){
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
        while (while_method_7(v136)){
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
        while (while_method_7(v145)){
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
        while (while_method_7(v161)){
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
        while (while_method_7(v177)){
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
        while (while_method_7(v201)){
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
__device__ void method_0(unsigned char * v0, unsigned char * v1, sptr<Mut0> v2, Union4 v3){
    static_array_list<Union1,32> & v4 = v2.base->v2;
    Union6 v5;
    v5 = Union6{Union6_1{v3}};
    Union6 v6;
    v6 = v5;
    while (while_method_2(v6)){
        Union6 v1024;
        switch (v6.tag) {
            case 0: { // None
                v1024 = Union6{Union6_0{}};
                break;
            }
            case 1: { // Some
                Union4 v8 = v6.case1.v0;
                switch (v8.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v969 = v8.case0.v0; bool v970 = v8.case0.v1; static_array<Union2,2> v971 = v8.case0.v2; int v972 = v8.case0.v3; static_array<int,2> v973 = v8.case0.v4; int v974 = v8.case0.v5;
                        curandStatePhilox4_32_10_t & v975 = v2.base->v5;
                        curandStatePhilox4_32_10_t & v976 = v975;
                        unsigned int & v977 = v2.base->v0;
                        Union2 v978; unsigned int v979;
                        Tuple0 tmp0 = draw_card_1(v976, v977);
                        v978 = tmp0.v0; v979 = tmp0.v1;
                        v2.base->v0 = v979;
                        Union1 v980;
                        v980 = Union1{Union1_0{v978}};
                        v4.push(v980);
                        int v981;
                        v981 = 2;
                        int v982; int v983;
                        Tuple1 tmp1 = Tuple1{0, 0};
                        v982 = tmp1.v0; v983 = tmp1.v1;
                        while (while_method_3(v982)){
                            int v985;
                            v985 = v973[v982];
                            bool v987;
                            v987 = v983 >= v985;
                            int v988;
                            if (v987){
                                v988 = v983;
                            } else {
                                v988 = v985;
                            }
                            v983 = v988;
                            v982 += 1 ;
                        }
                        static_array<int,2> v989;
                        int v991;
                        v991 = 0;
                        while (while_method_3(v991)){
                            v989[v991] = v983;
                            v991 += 1 ;
                        }
                        Union5 v993;
                        v993 = Union5{Union5_1{v978}};
                        Union4 v994;
                        v994 = Union4{Union4_2{v993, true, v971, 0, v989, v981}};
                        v1024 = Union6{Union6_1{v994}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v996 = v2.base->v5;
                        curandStatePhilox4_32_10_t & v997 = v996;
                        unsigned int & v998 = v2.base->v0;
                        Union2 v999; unsigned int v1000;
                        Tuple0 tmp2 = draw_card_1(v997, v998);
                        v999 = tmp2.v0; v1000 = tmp2.v1;
                        v2.base->v0 = v1000;
                        curandStatePhilox4_32_10_t & v1001 = v2.base->v5;
                        curandStatePhilox4_32_10_t & v1002 = v1001;
                        unsigned int & v1003 = v2.base->v0;
                        Union2 v1004; unsigned int v1005;
                        Tuple0 tmp3 = draw_card_1(v1002, v1003);
                        v1004 = tmp3.v0; v1005 = tmp3.v1;
                        v2.base->v0 = v1005;
                        Union1 v1006;
                        v1006 = Union1{Union1_2{0, v999}};
                        v4.push(v1006);
                        Union1 v1007;
                        v1007 = Union1{Union1_2{1, v1004}};
                        v4.push(v1007);
                        int v1008;
                        v1008 = 2;
                        static_array<int,2> v1009;
                        v1009[0] = 1;
                        v1009[1] = 1;
                        static_array<Union2,2> v1011;
                        v1011[0] = v999;
                        v1011[1] = v1004;
                        Union5 v1013;
                        v1013 = Union5{Union5_0{}};
                        Union4 v1014;
                        v1014 = Union4{Union4_2{v1013, true, v1011, 0, v1009, v1008}};
                        v1024 = Union6{Union6_1{v1014}};
                        break;
                    }
                    case 2: { // Round
                        Union5 v51 = v8.case2.v0; bool v52 = v8.case2.v1; static_array<Union2,2> v53 = v8.case2.v2; int v54 = v8.case2.v3; static_array<int,2> v55 = v8.case2.v4; int v56 = v8.case2.v5;
                        static_array<Union0,2> & v57 = v2.base->v3;
                        Union0 v58;
                        v58 = v57[v54];
                        Union3 v785;
                        switch (v58.tag) {
                            case 0: { // T_Computer
                                static_array_list<Union1,32> & v60 = v2.base->v2;
                                curandStatePhilox4_32_10_t & v61 = v2.base->v5;
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
                                    v213 = blockIdx.x;
                                    int v214;
                                    v214 = v213;
                                    while (while_method_3(v214)){
                                        bool v216;
                                        v216 = 0 <= v214;
                                        bool v217;
                                        v217 = v216 == false;
                                        if (v217){
                                            assert("The index needs to be zero or positive." && v216);
                                        } else {
                                        }
                                        int v219;
                                        v219 = v214 % 1;
                                        bool v220;
                                        v220 = v214 < 2;
                                        bool v221;
                                        v221 = v220 == false;
                                        if (v221){
                                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v220);
                                        } else {
                                        }
                                        assert("Tensor range check" && 0 <= v214 && v214 < 2);
                                        assert("Tensor range check" && 0 <= v219 && v219 < 1);
                                        int v223;
                                        v223 = 128 * v219;
                                        int v224;
                                        v224 = v223 + v156;
                                        int v225;
                                        v225 = 16384 * v214;
                                        int v226;
                                        v226 = v225 + v224;
                                        float * v227;
                                        v227 = v151+v226;
                                        // Pushing the loop unrolling to: 0
                                        int v229;
                                        v229 = 0;
                                        #pragma unroll
                                        while (while_method_6(v229)){
                                            int v231;
                                            v231 = 0;
                                            #pragma unroll
                                            while (while_method_7(v231)){
                                                assert("Tensor range check" && 0 <= v229 && v229 < 8);
                                                assert("Tensor range check" && 0 <= v231 && v231 < 1);
                                                int v233;
                                                v233 = v229 + v231;
                                                wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v234 = v212[v233];
                                                wmma::fill_fragment(v234, 0.0f);
                                                v231 += 1 ;
                                            }
                                            v229 += 1 ;
                                        }
                                        // Poping the loop unrolling to: 0
                                        int v235;
                                        v235 = 0;
                                        while (while_method_8(v235)){
                                            int v237;
                                            v237 = v235 + 1;
                                            bool v238;
                                            v238 = v235 == 0;
                                            int v239;
                                            v239 = v235 % 2;
                                            bool v240;
                                            v240 = 0 <= v235;
                                            bool v241;
                                            v241 = v240 == false;
                                            if (v241){
                                                assert("The index needs to be zero or positive." && v240);
                                            } else {
                                            }
                                            bool v243;
                                            v243 = v235 < 2;
                                            bool v244;
                                            v244 = v243 == false;
                                            if (v244){
                                                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v243);
                                            } else {
                                            }
                                            bool v246;
                                            v246 = v237 < 2;
                                            Union9 v252;
                                            if (v246){
                                                bool v247;
                                                v247 = 0 <= v237;
                                                bool v248;
                                                v248 = v247 == false;
                                                if (v248){
                                                    assert("The index needs to be zero or positive." && v247);
                                                } else {
                                                }
                                                v252 = Union9{Union9_1{v237}};
                                            } else {
                                                v252 = Union9{Union9_0{}};
                                            }
                                            assert("Tensor range check" && 0 <= v214 && v214 < 2);
                                            int v253;
                                            v253 = v225 + v154;
                                            assert("Tensor range check" && 0 <= v235 && v235 < 2);
                                            int v254;
                                            v254 = 64 * v235;
                                            int v255;
                                            v255 = v254 + v253;
                                            float * v256;
                                            v256 = v146+v255;
                                            assert("Tensor range check" && 0 <= v219 && v219 < 1);
                                            int v258;
                                            v258 = 16384 * v219;
                                            int v259;
                                            v259 = v258 + v150;
                                            if (v238){
                                                assert("Tensor range check" && 0 <= v235 && v235 < 2);
                                                int v260;
                                                v260 = v254 + v259;
                                                float * v261;
                                                v261 = v148+v260;
                                                // Pushing the loop unrolling to: 0
                                                v157.producer_acquire();
                                                int v263;
                                                v263 = threadIdx.x;
                                                bool v264;
                                                v264 = 0 <= v263;
                                                bool v265;
                                                v265 = v264 == false;
                                                if (v265){
                                                    assert("The index needs to be zero or positive." && v264);
                                                } else {
                                                }
                                                int v267;
                                                v267 = v263 % 16;
                                                int v268;
                                                v268 = v263 / 16;
                                                bool v269;
                                                v269 = v268 < 16;
                                                bool v270;
                                                v270 = v269 == false;
                                                if (v270){
                                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v269);
                                                } else {
                                                }
                                                assert("Tensor range check" && 0 <= v268 && v268 < 16);
                                                assert("Tensor range check" && 0 <= v267 && v267 < 16);
                                                int v272;
                                                v272 = 4 * v267;
                                                int v273;
                                                v273 = 68 * v268;
                                                int v274;
                                                v274 = v273 + v272;
                                                int v275;
                                                v275 = 128 * v268;
                                                int v276;
                                                v276 = v275 + v272;
                                                float * v277;
                                                v277 = v161+v274;
                                                float * v279;
                                                v279 = v261+v276;
                                                int v281;
                                                v281 = 0;
                                                #pragma unroll
                                                while (while_method_6(v281)){
                                                    int v283;
                                                    v283 = 0;
                                                    #pragma unroll
                                                    while (while_method_7(v283)){
                                                        assert("Tensor range check" && 0 <= v281 && v281 < 8);
                                                        assert("Tensor range check" && 0 <= v283 && v283 < 1);
                                                        int v285;
                                                        v285 = 64 * v283;
                                                        int v286;
                                                        v286 = 1088 * v281;
                                                        int v287;
                                                        v287 = v286 + v285;
                                                        int v288;
                                                        v288 = 2048 * v281;
                                                        int v289;
                                                        v289 = v288 + v285;
                                                        constexpr int v290 = sizeof(float) * 4;
                                                        assert("Pointer alignment check" && (unsigned long long)(v279 + v289) % v290 == 0 && (unsigned long long)(v277 + v287) % v290 == 0);
                                                        cuda::memcpy_async(v277 + v287, v279 + v289, cuda::aligned_size_t<v290>(v290), v157);
                                                        v283 += 1 ;
                                                    }
                                                    v281 += 1 ;
                                                }
                                                v157.producer_commit();
                                                // Poping the loop unrolling to: 0
                                            } else {
                                            }
                                            // Pushing the loop unrolling to: 0
                                            int v291;
                                            v291 = threadIdx.x;
                                            bool v292;
                                            v292 = 0 <= v291;
                                            bool v293;
                                            v293 = v292 == false;
                                            if (v293){
                                                assert("The index needs to be zero or positive." && v292);
                                            } else {
                                            }
                                            int v295;
                                            v295 = v291 % 16;
                                            int v296;
                                            v296 = v291 / 16;
                                            bool v297;
                                            v297 = v296 < 16;
                                            bool v298;
                                            v298 = v297 == false;
                                            if (v298){
                                                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v297);
                                            } else {
                                            }
                                            assert("Tensor range check" && 0 <= v296 && v296 < 16);
                                            assert("Tensor range check" && 0 <= v295 && v295 < 16);
                                            int v300;
                                            v300 = 4 * v295;
                                            int v301;
                                            v301 = 68 * v296;
                                            int v302;
                                            v302 = v301 + v300;
                                            int v303;
                                            v303 = 128 * v296;
                                            int v304;
                                            v304 = v303 + v300;
                                            float * v305;
                                            v305 = v159+v302;
                                            float * v307;
                                            v307 = v256+v304;
                                            int v309;
                                            v309 = 0;
                                            #pragma unroll
                                            while (while_method_6(v309)){
                                                int v311;
                                                v311 = 0;
                                                #pragma unroll
                                                while (while_method_7(v311)){
                                                    assert("Tensor range check" && 0 <= v309 && v309 < 8);
                                                    assert("Tensor range check" && 0 <= v311 && v311 < 1);
                                                    int v313;
                                                    v313 = 64 * v311;
                                                    int v314;
                                                    v314 = 1088 * v309;
                                                    int v315;
                                                    v315 = v314 + v313;
                                                    int v316;
                                                    v316 = 2048 * v309;
                                                    int v317;
                                                    v317 = v316 + v313;
                                                    int4* v318;
                                                    v318 = reinterpret_cast<int4*>(v307 + v317);
                                                    int4* v319;
                                                    v319 = reinterpret_cast<int4*>(v305 + v315);
                                                    assert("Pointer alignment check" && (unsigned long long)(v318) % 4 == 0 && (unsigned long long)(v319) % 4 == 0);
                                                    *v319 = *v318;
                                                    v311 += 1 ;
                                                }
                                                v309 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v320[1];
                                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v321[8];
                                            cuda::pipeline_consumer_wait_prior<0>(v157);;
                                            asm("barrier.cta.sync %0;" :: "r"(0));
                                            // Pushing the loop unrolling to: 0
                                            int v322;
                                            v322 = 0;
                                            #pragma unroll
                                            while (while_method_7(v322)){
                                                int v324;
                                                v324 = 0;
                                                #pragma unroll
                                                while (while_method_6(v324)){
                                                    assert("Tensor range check" && 0 <= v322 && v322 < 1);
                                                    assert("Tensor range check" && 0 <= v324 && v324 < 8);
                                                    int v326;
                                                    v326 = 8 * v322;
                                                    int v327;
                                                    v327 = v326 + v324;
                                                    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v328 = v321[v327];
                                                    assert("Tensor range check" && 0 <= v322 && v322 < 1);
                                                    int v329;
                                                    v329 = 1088 * v322;
                                                    assert("Tensor range check" && 0 <= v324 && v324 < 8);
                                                    int v330;
                                                    v330 = 8 * v324;
                                                    int v331;
                                                    v331 = v330 + v329;
                                                    int v332;
                                                    v332 = 0;
                                                    #pragma unroll
                                                    while (while_method_3(v332)){
                                                        int v334;
                                                        v334 = 0;
                                                        #pragma unroll
                                                        while (while_method_3(v334)){
                                                            assert("Tensor range check" && 0 <= v332 && v332 < 2);
                                                            assert("Tensor range check" && 0 <= v334 && v334 < 2);
                                                            int v336;
                                                            v336 = 4 * v334;
                                                            int v337;
                                                            v337 = v336 + v331;
                                                            int v338;
                                                            v338 = 544 * v332;
                                                            int v339;
                                                            v339 = v338 + v337;
                                                            float v340;
                                                            v340 = v210[v339];
                                                            bool v341;
                                                            v341 = 0 <= v334;
                                                            bool v343;
                                                            if (v341){
                                                                bool v342;
                                                                v342 = v334 < 2;
                                                                v343 = v342;
                                                            } else {
                                                                v343 = false;
                                                            }
                                                            bool v344;
                                                            v344 = v343 == false;
                                                            if (v344){
                                                                assert("The indices should be inside the range of the dimension." && v343);
                                                            } else {
                                                            }
                                                            bool v346;
                                                            v346 = 0 <= v332;
                                                            bool v348;
                                                            if (v346){
                                                                bool v347;
                                                                v347 = v332 < 2;
                                                                v348 = v347;
                                                            } else {
                                                                v348 = false;
                                                            }
                                                            bool v349;
                                                            v349 = v348 == false;
                                                            if (v349){
                                                                assert("The indices should be inside the range of the dimension." && v348);
                                                            } else {
                                                            }
                                                            int v351;
                                                            v351 = v332 * 2;
                                                            int v352;
                                                            v352 = v334 + v351;
                                                            v328.x[v352] = wmma::__float_to_tf32(v340);
                                                            v334 += 1 ;
                                                        }
                                                        v332 += 1 ;
                                                    }
                                                    v324 += 1 ;
                                                }
                                                v322 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            v157.consumer_release();
                                            switch (v252.tag) {
                                                case 0: { // None
                                                    break;
                                                }
                                                case 1: { // Some
                                                    int v353 = v252.case1.v0;
                                                    assert("Tensor range check" && 0 <= v353 && v353 < 2);
                                                    int v354;
                                                    v354 = 64 * v353;
                                                    int v355;
                                                    v355 = v354 + v259;
                                                    float * v356;
                                                    v356 = v148+v355;
                                                    asm("barrier.cta.sync %0;" :: "r"(0));
                                                    // Pushing the loop unrolling to: 0
                                                    v157.producer_acquire();
                                                    int v358;
                                                    v358 = threadIdx.x;
                                                    bool v359;
                                                    v359 = 0 <= v358;
                                                    bool v360;
                                                    v360 = v359 == false;
                                                    if (v360){
                                                        assert("The index needs to be zero or positive." && v359);
                                                    } else {
                                                    }
                                                    int v362;
                                                    v362 = v358 % 16;
                                                    int v363;
                                                    v363 = v358 / 16;
                                                    bool v364;
                                                    v364 = v363 < 16;
                                                    bool v365;
                                                    v365 = v364 == false;
                                                    if (v365){
                                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v364);
                                                    } else {
                                                    }
                                                    assert("Tensor range check" && 0 <= v363 && v363 < 16);
                                                    assert("Tensor range check" && 0 <= v362 && v362 < 16);
                                                    int v367;
                                                    v367 = 4 * v362;
                                                    int v368;
                                                    v368 = 68 * v363;
                                                    int v369;
                                                    v369 = v368 + v367;
                                                    int v370;
                                                    v370 = 128 * v363;
                                                    int v371;
                                                    v371 = v370 + v367;
                                                    float * v372;
                                                    v372 = v161+v369;
                                                    float * v374;
                                                    v374 = v356+v371;
                                                    int v376;
                                                    v376 = 0;
                                                    #pragma unroll
                                                    while (while_method_6(v376)){
                                                        int v378;
                                                        v378 = 0;
                                                        #pragma unroll
                                                        while (while_method_7(v378)){
                                                            assert("Tensor range check" && 0 <= v376 && v376 < 8);
                                                            assert("Tensor range check" && 0 <= v378 && v378 < 1);
                                                            int v380;
                                                            v380 = 64 * v378;
                                                            int v381;
                                                            v381 = 1088 * v376;
                                                            int v382;
                                                            v382 = v381 + v380;
                                                            int v383;
                                                            v383 = 2048 * v376;
                                                            int v384;
                                                            v384 = v383 + v380;
                                                            constexpr int v385 = sizeof(float) * 4;
                                                            assert("Pointer alignment check" && (unsigned long long)(v374 + v384) % v385 == 0 && (unsigned long long)(v372 + v382) % v385 == 0);
                                                            cuda::memcpy_async(v372 + v382, v374 + v384, cuda::aligned_size_t<v385>(v385), v157);
                                                            v378 += 1 ;
                                                        }
                                                        v376 += 1 ;
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
                                            int v386;
                                            v386 = 0;
                                            #pragma unroll
                                            while (while_method_6(v386)){
                                                int v388;
                                                v388 = 0;
                                                #pragma unroll
                                                while (while_method_6(v388)){
                                                    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v390 = v320[0];
                                                    assert("Tensor range check" && 0 <= v386 && v386 < 8);
                                                    int v391;
                                                    v391 = 1088 * v386;
                                                    assert("Tensor range check" && 0 <= v388 && v388 < 8);
                                                    int v392;
                                                    v392 = 8 * v388;
                                                    int v393;
                                                    v393 = v392 + v391;
                                                    int v394;
                                                    v394 = 0;
                                                    #pragma unroll
                                                    while (while_method_3(v394)){
                                                        int v396;
                                                        v396 = 0;
                                                        #pragma unroll
                                                        while (while_method_3(v396)){
                                                            assert("Tensor range check" && 0 <= v394 && v394 < 2);
                                                            assert("Tensor range check" && 0 <= v396 && v396 < 2);
                                                            int v398;
                                                            v398 = 544 * v396;
                                                            int v399;
                                                            v399 = v398 + v393;
                                                            int v400;
                                                            v400 = 4 * v394;
                                                            int v401;
                                                            v401 = v400 + v399;
                                                            float v402;
                                                            v402 = v194[v401];
                                                            bool v403;
                                                            v403 = 0 <= v396;
                                                            bool v405;
                                                            if (v403){
                                                                bool v404;
                                                                v404 = v396 < 2;
                                                                v405 = v404;
                                                            } else {
                                                                v405 = false;
                                                            }
                                                            bool v406;
                                                            v406 = v405 == false;
                                                            if (v406){
                                                                assert("The indices should be inside the range of the dimension." && v405);
                                                            } else {
                                                            }
                                                            bool v408;
                                                            v408 = 0 <= v394;
                                                            bool v410;
                                                            if (v408){
                                                                bool v409;
                                                                v409 = v394 < 2;
                                                                v410 = v409;
                                                            } else {
                                                                v410 = false;
                                                            }
                                                            bool v411;
                                                            v411 = v410 == false;
                                                            if (v411){
                                                                assert("The indices should be inside the range of the dimension." && v410);
                                                            } else {
                                                            }
                                                            int v413;
                                                            v413 = v394 * 2;
                                                            int v414;
                                                            v414 = v396 + v413;
                                                            v390.x[v414] = wmma::__float_to_tf32(v402);
                                                            v396 += 1 ;
                                                        }
                                                        v394 += 1 ;
                                                    }
                                                    int v415;
                                                    v415 = 0;
                                                    #pragma unroll
                                                    while (while_method_7(v415)){
                                                        assert("Tensor range check" && 0 <= v386 && v386 < 8);
                                                        assert("Tensor range check" && 0 <= v415 && v415 < 1);
                                                        int v417;
                                                        v417 = v386 + v415;
                                                        wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v418 = v212[v417];
                                                        assert("Tensor range check" && 0 <= v415 && v415 < 1);
                                                        assert("Tensor range check" && 0 <= v388 && v388 < 8);
                                                        int v419;
                                                        v419 = 8 * v415;
                                                        int v420;
                                                        v420 = v419 + v388;
                                                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v421 = v321[v420];
                                                        wmma::mma_sync(v418, v390, v421, v418);
                                                        v415 += 1 ;
                                                    }
                                                    v388 += 1 ;
                                                }
                                                v386 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            asm("barrier.cta.sync %0;" :: "r"(0));
                                            v235 = v237;
                                        }
                                        // Pushing the loop unrolling to: 0
                                        int v422;
                                        v422 = 0;
                                        #pragma unroll
                                        while (while_method_6(v422)){
                                            int v424;
                                            v424 = 0;
                                            #pragma unroll
                                            while (while_method_7(v424)){
                                                assert("Tensor range check" && 0 <= v422 && v422 < 8);
                                                assert("Tensor range check" && 0 <= v424 && v424 < 1);
                                                int v426;
                                                v426 = v422 + v424;
                                                wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v427 = v212[v426];
                                                assert("Tensor range check" && 0 <= v422 && v422 < 8);
                                                assert("Tensor range check" && 0 <= v424 && v424 < 1);
                                                int v428;
                                                v428 = 16 * v424;
                                                int v429;
                                                v429 = 2176 * v422;
                                                int v430;
                                                v430 = v429 + v428;
                                                float * v431;
                                                v431 = v178+v430;
                                                wmma::store_matrix_sync(v431, v427, 136, wmma::mem_row_major);
                                                v424 += 1 ;
                                            }
                                            v422 += 1 ;
                                        }
                                        // Poping the loop unrolling to: 0
                                        asm("barrier.cta.sync %0;" :: "r"(0));
                                        // Pushing the loop unrolling to: 0
                                        int v433;
                                        v433 = threadIdx.x;
                                        bool v434;
                                        v434 = 0 <= v433;
                                        bool v435;
                                        v435 = v434 == false;
                                        if (v435){
                                            assert("The index needs to be zero or positive." && v434);
                                        } else {
                                        }
                                        int v437;
                                        v437 = v433 % 32;
                                        int v438;
                                        v438 = v433 / 32;
                                        bool v439;
                                        v439 = v438 < 8;
                                        bool v440;
                                        v440 = v439 == false;
                                        if (v440){
                                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v439);
                                        } else {
                                        }
                                        assert("Tensor range check" && 0 <= v438 && v438 < 8);
                                        assert("Tensor range check" && 0 <= v437 && v437 < 32);
                                        int v442;
                                        v442 = 4 * v437;
                                        int v443;
                                        v443 = 128 * v438;
                                        int v444;
                                        v444 = v443 + v442;
                                        int v445;
                                        v445 = 136 * v438;
                                        int v446;
                                        v446 = v445 + v442;
                                        float * v447;
                                        v447 = v227+v444;
                                        float * v449;
                                        v449 = v163+v446;
                                        int v451;
                                        v451 = 0;
                                        #pragma unroll
                                        while (while_method_1(v451)){
                                            int v453;
                                            v453 = 0;
                                            #pragma unroll
                                            while (while_method_7(v453)){
                                                assert("Tensor range check" && 0 <= v451 && v451 < 16);
                                                assert("Tensor range check" && 0 <= v453 && v453 < 1);
                                                int v455;
                                                v455 = 128 * v453;
                                                int v456;
                                                v456 = 1024 * v451;
                                                int v457;
                                                v457 = v456 + v455;
                                                int v458;
                                                v458 = 1088 * v451;
                                                int v459;
                                                v459 = v458 + v455;
                                                int4* v460;
                                                v460 = reinterpret_cast<int4*>(v449 + v459);
                                                int4* v461;
                                                v461 = reinterpret_cast<int4*>(v447 + v457);
                                                assert("Pointer alignment check" && (unsigned long long)(v460) % 4 == 0 && (unsigned long long)(v461) % 4 == 0);
                                                *v461 = *v460;
                                                v453 += 1 ;
                                            }
                                            v451 += 1 ;
                                        }
                                        // Poping the loop unrolling to: 0
                                        asm("barrier.cta.sync %0;" :: "r"(0));
                                        v214 += 24 ;
                                    }
                                    unsigned int * v462;
                                    v462 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
                                    assert("Tensor range check" && 0 <= v144 && v144 < 4);
                                    int v464;
                                    v464 = 6144 * v144;
                                    method_3(v462, v464, v151);
                                    int * v465;
                                    v465 = reinterpret_cast<int *>(&v1[262144ull]);
                                    float * v467;
                                    v467 = reinterpret_cast<float *>(&v1[262160ull]);
                                    float * v469;
                                    v469 = reinterpret_cast<float *>(&v1[524304ull]);
                                    float * v471;
                                    v471 = reinterpret_cast<float *>(&v1[786448ull]);
                                    float * v473;
                                    v473 = reinterpret_cast<float *>(&v1[1048592ull]);
                                    float * v475;
                                    v475 = reinterpret_cast<float *>(&v1[1310736ull]);
                                    float * v477;
                                    v477 = reinterpret_cast<float *>(&v1[1572880ull]);
                                    float * v479;
                                    v479 = reinterpret_cast<float *>(&v1[1835024ull]);
                                    int * v481;
                                    v481 = reinterpret_cast<int *>(&v0[6389760ull]);
                                    float * v483;
                                    v483 = reinterpret_cast<float *>(&v0[7962624ull]);
                                    int * v485;
                                    v485 = reinterpret_cast<int *>(&v0[9535488ull]);
                                    int * v487;
                                    v487 = reinterpret_cast<int *>(&v0[11108352ull]);
                                    double * v489;
                                    v489 = reinterpret_cast<double *>(&v0[12681216ull]);
                                    double * v491;
                                    v491 = reinterpret_cast<double *>(&v0[18972672ull]);
                                    double * v493;
                                    v493 = reinterpret_cast<double *>(&v1[2097168ull]);
                                    double * v495;
                                    v495 = reinterpret_cast<double *>(&v1[2490384ull]);
                                    int * v497;
                                    v497 = reinterpret_cast<int *>(&v1[2883600ull]);
                                    v144 += 1 ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                int * v499;
                                v499 = reinterpret_cast<int *>(&v1[262144ull]);
                                float * v501;
                                v501 = reinterpret_cast<float *>(&v1[262160ull]);
                                float * v503;
                                v503 = reinterpret_cast<float *>(&v1[524304ull]);
                                float * v505;
                                v505 = reinterpret_cast<float *>(&v1[786448ull]);
                                float * v507;
                                v507 = reinterpret_cast<float *>(&v1[1048592ull]);
                                float * v509;
                                v509 = reinterpret_cast<float *>(&v1[1310736ull]);
                                float * v511;
                                v511 = reinterpret_cast<float *>(&v1[1572880ull]);
                                float * v513;
                                v513 = reinterpret_cast<float *>(&v1[1835024ull]);
                                int v515;
                                v515 = v499[0];
                                unsigned int * v516;
                                v516 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
                                int v518;
                                v518 = blockIdx.x;
                                int v519;
                                v519 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v515 && v515 < 4);
                                assert("Tensor range check" && 0 <= v518 && v518 < 24);
                                assert("Tensor range check" && 0 <= v519 && v519 < 256);
                                int v520;
                                v520 = 256 * v518;
                                int v521;
                                v521 = v520 + v519;
                                int v522;
                                v522 = 6144 * v515;
                                int v523;
                                v523 = v522 + v521;
                                unsigned int v524;
                                v524 = v516[v523];
                                int v525;
                                v525 = (int)v524;
                                float v526; int v527;
                                Tuple2 tmp14 = method_4(v62, v499, v501, v503, v505, v507, v509, v511, v513, v525, v515);
                                v526 = tmp14.v0; v527 = tmp14.v1;
                                extern __shared__ unsigned char v528[];
                                float * v529;
                                v529 = reinterpret_cast<float *>(&v528[0ull]);
                                int * v531;
                                v531 = reinterpret_cast<int *>(&v528[16ull]);
                                int v533;
                                v533 = threadIdx.x;
                                bool v534;
                                v534 = v533 == 0;
                                if (v534){
                                    v529[0] = v526;
                                    v531[0] = v527;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                float v535;
                                v535 = v529[0];
                                int v536;
                                v536 = v531[0];
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                double * v537;
                                v537 = reinterpret_cast<double *>(&v1[2097168ull]);
                                double * v539;
                                v539 = reinterpret_cast<double *>(&v1[2490384ull]);
                                int * v541;
                                v541 = reinterpret_cast<int *>(&v1[2883600ull]);
                                int * v543;
                                v543 = reinterpret_cast<int *>(&v0[6389760ull]);
                                float * v545;
                                v545 = reinterpret_cast<float *>(&v0[7962624ull]);
                                int * v547;
                                v547 = reinterpret_cast<int *>(&v0[9535488ull]);
                                int * v549;
                                v549 = reinterpret_cast<int *>(&v0[11108352ull]);
                                double * v551;
                                v551 = reinterpret_cast<double *>(&v0[12681216ull]);
                                double * v553;
                                v553 = reinterpret_cast<double *>(&v0[18972672ull]);
                                int v555;
                                v555 = threadIdx.x;
                                int v556;
                                v556 = blockIdx.x;
                                int v557;
                                v557 = v556 * 256;
                                int v558;
                                v558 = v555 + v557;
                                int v559;
                                v559 = 0;
                                while (while_method_0(v559)){
                                    unsigned int * v561;
                                    v561 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
                                    int v563;
                                    v563 = blockIdx.x;
                                    int v564;
                                    v564 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v559 && v559 < 4);
                                    assert("Tensor range check" && 0 <= v563 && v563 < 24);
                                    assert("Tensor range check" && 0 <= v564 && v564 < 256);
                                    int v565;
                                    v565 = 256 * v563;
                                    int v566;
                                    v566 = v565 + v564;
                                    int v567;
                                    v567 = 6144 * v559;
                                    int v568;
                                    v568 = v567 + v566;
                                    unsigned int v569;
                                    v569 = v561[v568];
                                    int v570;
                                    v570 = (int)v569;
                                    float v571;
                                    v571 = method_5(v499, v501, v503, v505, v507, v509, v511, v513, v570, v559, v536);
                                    assert("Tensor range check" && 0 <= v559 && v559 < 4);
                                    assert("Tensor range check" && 0 <= v558 && v558 < 6144);
                                    int v572;
                                    v572 = v567 + v558;
                                    int v573;
                                    v573 = v541[v572];
                                    int v574;
                                    v574 = v573 + 1;
                                    assert("Tensor range check" && 0 <= v559 && v559 < 4);
                                    assert("Tensor range check" && 0 <= v558 && v558 < 6144);
                                    v541[v572] = v574;
                                    assert("Tensor range check" && 0 <= v559 && v559 < 4);
                                    assert("Tensor range check" && 0 <= v573 && v573 < 16);
                                    assert("Tensor range check" && 0 <= v558 && v558 < 6144);
                                    int v575;
                                    v575 = 6144 * v573;
                                    int v576;
                                    v576 = v575 + v558;
                                    int v577;
                                    v577 = 98304 * v559;
                                    int v578;
                                    v578 = v577 + v576;
                                    v543[v578] = v536;
                                    v545[v578] = v535;
                                    v547[v578] = v54;
                                    v549[v578] = v570;
                                    assert("Tensor range check" && 0 <= v559 && v559 < 4);
                                    int v579;
                                    v579 = 12288 * v559;
                                    assert("Tensor range check" && 0 <= v558 && v558 < 6144);
                                    int v580;
                                    v580 = 2 * v558;
                                    int v581;
                                    v581 = v580 + v579;
                                    assert("Tensor range check" && 0 <= v559 && v559 < 4);
                                    int v582;
                                    v582 = 196608 * v559;
                                    assert("Tensor range check" && 0 <= v573 && v573 < 16);
                                    int v583;
                                    v583 = 12288 * v573;
                                    int v584;
                                    v584 = v583 + v582;
                                    assert("Tensor range check" && 0 <= v558 && v558 < 6144);
                                    int v585;
                                    v585 = v580 + v584;
                                    double * v586;
                                    v586 = v537+v581;
                                    double * v588;
                                    v588 = v539+v581;
                                    double * v590;
                                    v590 = v551+v585;
                                    double * v592;
                                    v592 = v553+v585;
                                    int v594;
                                    v594 = sizeof(double *);
                                    unsigned long long v595;
                                    v595 = (unsigned long long)v594;
                                    unsigned long long v596;
                                    v596 = 256ull * v595;
                                    unsigned long long v597;
                                    v597 = v596 + 16ull;
                                    unsigned long long v598;
                                    v598 = v597 - 1ull;
                                    unsigned long long v599;
                                    v599 = v598 % 16ull;
                                    unsigned long long v600;
                                    v600 = v598 - v599;
                                    unsigned long long v601;
                                    v601 = v600 + v596;
                                    unsigned long long v602;
                                    v602 = v601 + 16ull;
                                    unsigned long long v603;
                                    v603 = v602 - 1ull;
                                    unsigned long long v604;
                                    v604 = v603 % 16ull;
                                    unsigned long long v605;
                                    v605 = v603 - v604;
                                    unsigned long long v606;
                                    v606 = v605 + v596;
                                    unsigned long long v607;
                                    v607 = v606 + 16ull;
                                    unsigned long long v608;
                                    v608 = v607 - 1ull;
                                    unsigned long long v609;
                                    v609 = v608 % 16ull;
                                    unsigned long long v610;
                                    v610 = v608 - v609;
                                    unsigned long long v611;
                                    v611 = v610 + v596;
                                    bool v612;
                                    v612 = v611 <= 98304ull;
                                    bool v613;
                                    v613 = v612 == false;
                                    if (v613){
                                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v612);
                                    } else {
                                    }
                                    extern __shared__ unsigned char v615[];
                                    bool v616;
                                    v616 = v611 <= v611;
                                    bool v617;
                                    v617 = v616 == false;
                                    if (v617){
                                        assert("The length of the partition has to be less than or equal to the length of the base array." && v616);
                                    } else {
                                    }
                                    double * * v619;
                                    v619 = reinterpret_cast<double * *>(&v615[0ull]);
                                    double * * v621;
                                    v621 = reinterpret_cast<double * *>(&v615[v600]);
                                    double * * v623;
                                    v623 = reinterpret_cast<double * *>(&v615[v605]);
                                    double * * v625;
                                    v625 = reinterpret_cast<double * *>(&v615[v610]);
                                    int v627;
                                    v627 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v627 && v627 < 256);
                                    v619[v627] = v586;
                                    v621[v627] = v588;
                                    v623[v627] = v590;
                                    v625[v627] = v592;
                                    asm("barrier.cta.sync %0;" :: "r"(0));
                                    bool v628;
                                    v628 = 0 <= v627;
                                    bool v629;
                                    v629 = v628 == false;
                                    if (v629){
                                        assert("The index needs to be zero or positive." && v628);
                                    } else {
                                    }
                                    int v631;
                                    v631 = v627 % 1;
                                    bool v632;
                                    v632 = v627 < 256;
                                    bool v633;
                                    v633 = v632 == false;
                                    if (v633){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v632);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v627 && v627 < 256);
                                    int v635;
                                    v635 = 0;
                                    while (while_method_7(v635)){
                                        bool v637;
                                        v637 = v628 && v632;
                                        bool v638;
                                        v638 = v637 == false;
                                        if (v638){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v637);
                                        } else {
                                        }
                                        bool v640;
                                        v640 = 0 <= v635;
                                        bool v642;
                                        if (v640){
                                            bool v641;
                                            v641 = v635 < 1;
                                            v642 = v641;
                                        } else {
                                            v642 = false;
                                        }
                                        bool v643;
                                        v643 = v642 == false;
                                        if (v643){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v642);
                                        } else {
                                        }
                                        int v645;
                                        v645 = v635 * 256;
                                        int v646;
                                        v646 = v645 + v627;
                                        assert("Tensor range check" && 0 <= v635 && v635 < 1);
                                        int v647;
                                        v647 = 256 * v635;
                                        int v648;
                                        v648 = v647 + v627;
                                        double * v649;
                                        v649 = v619[v648];
                                        double * v650;
                                        v650 = v621[v648];
                                        double * v651;
                                        v651 = v623[v648];
                                        double * v652;
                                        v652 = v625[v648];
                                        int v653;
                                        v653 = blockIdx.x;
                                        int v654;
                                        v654 = v653 * 256;
                                        int v655;
                                        v655 = v654 + v646;
                                        assert("Tensor range check" && 0 <= v631 && v631 < 1);
                                        int v656;
                                        v656 = 2 * v631;
                                        double v657[2];
                                        double v658[2];
                                        int v659[2];
                                        int v660;
                                        v660 = 0;
                                        while (while_method_7(v660)){
                                            assert("Tensor range check" && 0 <= v660 && v660 < 1);
                                            int v662;
                                            v662 = 2 * v660;
                                            assert("Tensor range check" && 0 <= v660 && v660 < 1);
                                            int v663;
                                            v663 = v662 + v656;
                                            int4* v664;
                                            v664 = reinterpret_cast<int4*>(v649 + v663);
                                            int4* v665;
                                            v665 = reinterpret_cast<int4*>(v657 + v662);
                                            assert("Pointer alignment check" && (unsigned long long)(v664) % 2 == 0 && (unsigned long long)(v665) % 2 == 0);
                                            *v665 = *v664;
                                            int4* v666;
                                            v666 = reinterpret_cast<int4*>(v650 + v663);
                                            int4* v667;
                                            v667 = reinterpret_cast<int4*>(v658 + v662);
                                            assert("Pointer alignment check" && (unsigned long long)(v666) % 2 == 0 && (unsigned long long)(v667) % 2 == 0);
                                            *v667 = *v666;
                                            v660 += 1 ;
                                        }
                                        int v668;
                                        v668 = 0;
                                        while (while_method_7(v668)){
                                            int v670;
                                            v670 = 0;
                                            while (while_method_3(v670)){
                                                bool v672;
                                                v672 = 0 <= v670;
                                                bool v674;
                                                if (v672){
                                                    bool v673;
                                                    v673 = v670 < 2;
                                                    v674 = v673;
                                                } else {
                                                    v674 = false;
                                                }
                                                bool v675;
                                                v675 = v674 == false;
                                                if (v675){
                                                    assert("The indices should be inside the range of the dimension." && v674);
                                                } else {
                                                }
                                                bool v677;
                                                v677 = 0 <= v631;
                                                bool v679;
                                                if (v677){
                                                    bool v678;
                                                    v678 = v631 < 1;
                                                    v679 = v678;
                                                } else {
                                                    v679 = false;
                                                }
                                                bool v680;
                                                v680 = v679 == false;
                                                if (v680){
                                                    assert("The indices should be inside the range of the dimension." && v679);
                                                } else {
                                                }
                                                int v682;
                                                v682 = v631 * 2;
                                                int v683;
                                                v683 = v670 + v682;
                                                bool v684;
                                                v684 = 0 <= v668;
                                                bool v686;
                                                if (v684){
                                                    bool v685;
                                                    v685 = v668 < 1;
                                                    v686 = v685;
                                                } else {
                                                    v686 = false;
                                                }
                                                bool v687;
                                                v687 = v686 == false;
                                                if (v687){
                                                    assert("The indices should be inside the range of the dimension." && v686);
                                                } else {
                                                }
                                                int v689;
                                                v689 = v668 * 2;
                                                int v690;
                                                v690 = v683 + v689;
                                                assert("Tensor range check" && 0 <= v668 && v668 < 1);
                                                assert("Tensor range check" && 0 <= v670 && v670 < 2);
                                                int v691;
                                                v691 = 2 * v668;
                                                int v692;
                                                v692 = v691 + v670;
                                                v659[v692] = v690;
                                                v670 += 1 ;
                                            }
                                            v668 += 1 ;
                                        }
                                        int v693;
                                        v693 = 0;
                                        while (while_method_7(v693)){
                                            assert("Tensor range check" && 0 <= v693 && v693 < 1);
                                            int v695;
                                            v695 = 2 * v693;
                                            int v696;
                                            v696 = v695 + v656;
                                            assert("Tensor range check" && 0 <= v693 && v693 < 1);
                                            int4* v697;
                                            v697 = reinterpret_cast<int4*>(v657 + v695);
                                            int4* v698;
                                            v698 = reinterpret_cast<int4*>(v651 + v696);
                                            assert("Pointer alignment check" && (unsigned long long)(v697) % 2 == 0 && (unsigned long long)(v698) % 2 == 0);
                                            *v698 = *v697;
                                            int4* v699;
                                            v699 = reinterpret_cast<int4*>(v658 + v695);
                                            int4* v700;
                                            v700 = reinterpret_cast<int4*>(v652 + v696);
                                            assert("Pointer alignment check" && (unsigned long long)(v699) % 2 == 0 && (unsigned long long)(v700) % 2 == 0);
                                            *v700 = *v699;
                                            v693 += 1 ;
                                        }
                                        assert("Tensor range check" && 0 <= v646 && v646 < 256);
                                        v635 += 1 ;
                                    }
                                    asm("barrier.cta.sync %0;" :: "r"(0));
                                    assert("Tensor range check" && 0 <= v627 && v627 < 256);
                                    asm("barrier.cta.sync %0;" :: "r"(0));
                                    double v701;
                                    v701 = (double)v535;
                                    double v702;
                                    v702 = log(v701);
                                    double v703;
                                    v703 = (double)v571;
                                    double v704;
                                    v704 = log(v703);
                                    assert("Tensor range check" && 0 <= v559 && v559 < 4);
                                    assert("Tensor range check" && 0 <= v558 && v558 < 6144);
                                    assert("Tensor range check" && 0 <= v54 && v54 < 2);
                                    int v705;
                                    v705 = v580 + v54;
                                    int v706;
                                    v706 = v579 + v705;
                                    double v707;
                                    v707 = v537[v706];
                                    double v708;
                                    v708 = v539[v706];
                                    double v709;
                                    v709 = v704 + v707;
                                    double v710;
                                    v710 = v702 + v708;
                                    assert("Tensor range check" && 0 <= v559 && v559 < 4);
                                    assert("Tensor range check" && 0 <= v558 && v558 < 6144);
                                    assert("Tensor range check" && 0 <= v54 && v54 < 2);
                                    v537[v706] = v709;
                                    v539[v706] = v710;
                                    v559 += 1 ;
                                }
                                bool v711;
                                v711 = 0 == v536;
                                Union10 v720;
                                if (v711){
                                    v720 = Union10{Union10_1{}};
                                } else {
                                    bool v713;
                                    v713 = 1 == v536;
                                    if (v713){
                                        v720 = Union10{Union10_0{}};
                                    } else {
                                        bool v715;
                                        v715 = 2 == v536;
                                        if (v715){
                                            v720 = Union10{Union10_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v720.tag) {
                                    case 0: { // AA_Call
                                        v785 = Union3{Union3_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v721;
                                        v721 = v55[0];
                                        int v723; int v724;
                                        Tuple1 tmp17 = Tuple1{1, v721};
                                        v723 = tmp17.v0; v724 = tmp17.v1;
                                        while (while_method_3(v723)){
                                            int v726;
                                            v726 = v55[v723];
                                            bool v728;
                                            v728 = v724 >= v726;
                                            int v729;
                                            if (v728){
                                                v729 = v724;
                                            } else {
                                                v729 = v726;
                                            }
                                            v724 = v729;
                                            v723 += 1 ;
                                        }
                                        int v730;
                                        v730 = v55[v54];
                                        bool v732;
                                        v732 = v730 == v724;
                                        if (v732){
                                            v785 = Union3{Union3_0{}};
                                        } else {
                                            v785 = Union3{Union3_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v737;
                                        v737 = v56 > 0;
                                        if (v737){
                                            v785 = Union3{Union3_2{}};
                                        } else {
                                            v785 = Union3{Union3_0{}};
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
                                curandStatePhilox4_32_10_t & v744 = v2.base->v5;
                                curandStatePhilox4_32_10_t & v745 = v744;
                                static_array_list<Union3,3> v746;
                                v746 = static_array_list<Union3,3>{};
                                v746.unsafe_set_length(1);
                                Union3 v748;
                                v748 = Union3{Union3_0{}};
                                v746[0] = v748;
                                int v750;
                                v750 = v55[0];
                                int v752;
                                v752 = v55[1];
                                bool v754;
                                v754 = v750 == v752;
                                bool v755;
                                v755 = v754 != true;
                                if (v755){
                                    Union3 v756;
                                    v756 = Union3{Union3_1{}};
                                    v746.push(v756);
                                } else {
                                }
                                bool v757;
                                v757 = v56 > 0;
                                if (v757){
                                    Union3 v758;
                                    v758 = Union3{Union3_2{}};
                                    v746.push(v758);
                                } else {
                                }
                                int v759;
                                v759 = v746.length;
                                int v760;
                                v760 = v759 - 1;
                                int v761;
                                v761 = 0;
                                while (while_method_5(v760, v761)){
                                    int v763;
                                    v763 = v746.length;
                                    int v764;
                                    v764 = int_range_6(v763, v761, v745);
                                    Union3 v765;
                                    v765 = v746[v761];
                                    Union3 v767;
                                    v767 = v746[v764];
                                    v746[v761] = v767;
                                    v746[v764] = v765;
                                    v761 += 1 ;
                                }
                                Union3 v769;
                                v769 = v746.pop();
                                int v770;
                                v770 = sizeof(Union3);
                                unsigned long long v771;
                                v771 = (unsigned long long)v770;
                                bool v772;
                                v772 = v771 <= 98304ull;
                                bool v773;
                                v773 = v772 == false;
                                if (v773){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v772);
                                } else {
                                }
                                extern __shared__ unsigned char v775[];
                                bool v776;
                                v776 = v771 <= v771;
                                bool v777;
                                v777 = v776 == false;
                                if (v777){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v776);
                                } else {
                                }
                                Union3 * v779;
                                v779 = reinterpret_cast<Union3 *>(&v775[0ull]);
                                int v781;
                                v781 = threadIdx.x;
                                bool v782;
                                v782 = v781 == 0;
                                if (v782){
                                    v779[0] = v769;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                Union3 v783;
                                v783 = v779[0];
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                v785 = v783;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union1 v786;
                        v786 = Union1{Union1_1{v54, v785}};
                        v4.push(v786);
                        Union4 v872;
                        switch (v51.tag) {
                            case 0: { // None
                                switch (v785.tag) {
                                    case 0: { // Call
                                        if (v52){
                                            bool v836;
                                            v836 = v54 == 0;
                                            int v837;
                                            if (v836){
                                                v837 = 1;
                                            } else {
                                                v837 = 0;
                                            }
                                            v872 = Union4{Union4_2{v51, false, v53, v837, v55, v56}};
                                        } else {
                                            v872 = Union4{Union4_0{v51, v52, v53, v54, v55, v56}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v872 = Union4{Union4_5{v51, v52, v53, v54, v55, v56}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v841;
                                        v841 = v56 > 0;
                                        if (v841){
                                            bool v842;
                                            v842 = v54 == 0;
                                            int v843;
                                            if (v842){
                                                v843 = 1;
                                            } else {
                                                v843 = 0;
                                            }
                                            int v844;
                                            v844 = -1 + v56;
                                            int v845; int v846;
                                            Tuple1 tmp18 = Tuple1{0, 0};
                                            v845 = tmp18.v0; v846 = tmp18.v1;
                                            while (while_method_3(v845)){
                                                int v848;
                                                v848 = v55[v845];
                                                bool v850;
                                                v850 = v846 >= v848;
                                                int v851;
                                                if (v850){
                                                    v851 = v846;
                                                } else {
                                                    v851 = v848;
                                                }
                                                v846 = v851;
                                                v845 += 1 ;
                                            }
                                            static_array<int,2> v852;
                                            int v854;
                                            v854 = 0;
                                            while (while_method_3(v854)){
                                                v852[v854] = v846;
                                                v854 += 1 ;
                                            }
                                            static_array<int,2> v856;
                                            int v858;
                                            v858 = 0;
                                            while (while_method_3(v858)){
                                                int v860;
                                                v860 = v852[v858];
                                                bool v862;
                                                v862 = v858 == v54;
                                                int v864;
                                                if (v862){
                                                    int v863;
                                                    v863 = v860 + 2;
                                                    v864 = v863;
                                                } else {
                                                    v864 = v860;
                                                }
                                                v856[v858] = v864;
                                                v858 += 1 ;
                                            }
                                            v872 = Union4{Union4_2{v51, false, v53, v843, v856, v844}};
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
                                Union2 v787 = v51.case1.v0;
                                switch (v785.tag) {
                                    case 0: { // Call
                                        if (v52){
                                            bool v789;
                                            v789 = v54 == 0;
                                            int v790;
                                            if (v789){
                                                v790 = 1;
                                            } else {
                                                v790 = 0;
                                            }
                                            v872 = Union4{Union4_2{v51, false, v53, v790, v55, v56}};
                                        } else {
                                            int v792; int v793;
                                            Tuple1 tmp19 = Tuple1{0, 0};
                                            v792 = tmp19.v0; v793 = tmp19.v1;
                                            while (while_method_3(v792)){
                                                int v795;
                                                v795 = v55[v792];
                                                bool v797;
                                                v797 = v793 >= v795;
                                                int v798;
                                                if (v797){
                                                    v798 = v793;
                                                } else {
                                                    v798 = v795;
                                                }
                                                v793 = v798;
                                                v792 += 1 ;
                                            }
                                            static_array<int,2> v799;
                                            int v801;
                                            v801 = 0;
                                            while (while_method_3(v801)){
                                                v799[v801] = v793;
                                                v801 += 1 ;
                                            }
                                            v872 = Union4{Union4_4{v51, v52, v53, v54, v799, v56}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v872 = Union4{Union4_5{v51, v52, v53, v54, v55, v56}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v805;
                                        v805 = v56 > 0;
                                        if (v805){
                                            bool v806;
                                            v806 = v54 == 0;
                                            int v807;
                                            if (v806){
                                                v807 = 1;
                                            } else {
                                                v807 = 0;
                                            }
                                            int v808;
                                            v808 = -1 + v56;
                                            int v809; int v810;
                                            Tuple1 tmp20 = Tuple1{0, 0};
                                            v809 = tmp20.v0; v810 = tmp20.v1;
                                            while (while_method_3(v809)){
                                                int v812;
                                                v812 = v55[v809];
                                                bool v814;
                                                v814 = v810 >= v812;
                                                int v815;
                                                if (v814){
                                                    v815 = v810;
                                                } else {
                                                    v815 = v812;
                                                }
                                                v810 = v815;
                                                v809 += 1 ;
                                            }
                                            static_array<int,2> v816;
                                            int v818;
                                            v818 = 0;
                                            while (while_method_3(v818)){
                                                v816[v818] = v810;
                                                v818 += 1 ;
                                            }
                                            static_array<int,2> v820;
                                            int v822;
                                            v822 = 0;
                                            while (while_method_3(v822)){
                                                int v824;
                                                v824 = v816[v822];
                                                bool v826;
                                                v826 = v822 == v54;
                                                int v828;
                                                if (v826){
                                                    int v827;
                                                    v827 = v824 + 4;
                                                    v828 = v827;
                                                } else {
                                                    v828 = v824;
                                                }
                                                v820[v822] = v828;
                                                v822 += 1 ;
                                            }
                                            v872 = Union4{Union4_2{v51, false, v53, v807, v820, v808}};
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
                        v1024 = Union6{Union6_1{v872}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v874 = v8.case3.v0; bool v875 = v8.case3.v1; static_array<Union2,2> v876 = v8.case3.v2; int v877 = v8.case3.v3; static_array<int,2> v878 = v8.case3.v4; int v879 = v8.case3.v5; Union3 v880 = v8.case3.v6;
                        Union1 v881;
                        v881 = Union1{Union1_1{v877, v880}};
                        v4.push(v881);
                        Union4 v967;
                        switch (v874.tag) {
                            case 0: { // None
                                switch (v880.tag) {
                                    case 0: { // Call
                                        if (v875){
                                            bool v931;
                                            v931 = v877 == 0;
                                            int v932;
                                            if (v931){
                                                v932 = 1;
                                            } else {
                                                v932 = 0;
                                            }
                                            v967 = Union4{Union4_2{v874, false, v876, v932, v878, v879}};
                                        } else {
                                            v967 = Union4{Union4_0{v874, v875, v876, v877, v878, v879}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v967 = Union4{Union4_5{v874, v875, v876, v877, v878, v879}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v936;
                                        v936 = v879 > 0;
                                        if (v936){
                                            bool v937;
                                            v937 = v877 == 0;
                                            int v938;
                                            if (v937){
                                                v938 = 1;
                                            } else {
                                                v938 = 0;
                                            }
                                            int v939;
                                            v939 = -1 + v879;
                                            int v940; int v941;
                                            Tuple1 tmp21 = Tuple1{0, 0};
                                            v940 = tmp21.v0; v941 = tmp21.v1;
                                            while (while_method_3(v940)){
                                                int v943;
                                                v943 = v878[v940];
                                                bool v945;
                                                v945 = v941 >= v943;
                                                int v946;
                                                if (v945){
                                                    v946 = v941;
                                                } else {
                                                    v946 = v943;
                                                }
                                                v941 = v946;
                                                v940 += 1 ;
                                            }
                                            static_array<int,2> v947;
                                            int v949;
                                            v949 = 0;
                                            while (while_method_3(v949)){
                                                v947[v949] = v941;
                                                v949 += 1 ;
                                            }
                                            static_array<int,2> v951;
                                            int v953;
                                            v953 = 0;
                                            while (while_method_3(v953)){
                                                int v955;
                                                v955 = v947[v953];
                                                bool v957;
                                                v957 = v953 == v877;
                                                int v959;
                                                if (v957){
                                                    int v958;
                                                    v958 = v955 + 2;
                                                    v959 = v958;
                                                } else {
                                                    v959 = v955;
                                                }
                                                v951[v953] = v959;
                                                v953 += 1 ;
                                            }
                                            v967 = Union4{Union4_2{v874, false, v876, v938, v951, v939}};
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
                                Union2 v882 = v874.case1.v0;
                                switch (v880.tag) {
                                    case 0: { // Call
                                        if (v875){
                                            bool v884;
                                            v884 = v877 == 0;
                                            int v885;
                                            if (v884){
                                                v885 = 1;
                                            } else {
                                                v885 = 0;
                                            }
                                            v967 = Union4{Union4_2{v874, false, v876, v885, v878, v879}};
                                        } else {
                                            int v887; int v888;
                                            Tuple1 tmp22 = Tuple1{0, 0};
                                            v887 = tmp22.v0; v888 = tmp22.v1;
                                            while (while_method_3(v887)){
                                                int v890;
                                                v890 = v878[v887];
                                                bool v892;
                                                v892 = v888 >= v890;
                                                int v893;
                                                if (v892){
                                                    v893 = v888;
                                                } else {
                                                    v893 = v890;
                                                }
                                                v888 = v893;
                                                v887 += 1 ;
                                            }
                                            static_array<int,2> v894;
                                            int v896;
                                            v896 = 0;
                                            while (while_method_3(v896)){
                                                v894[v896] = v888;
                                                v896 += 1 ;
                                            }
                                            v967 = Union4{Union4_4{v874, v875, v876, v877, v894, v879}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v967 = Union4{Union4_5{v874, v875, v876, v877, v878, v879}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v900;
                                        v900 = v879 > 0;
                                        if (v900){
                                            bool v901;
                                            v901 = v877 == 0;
                                            int v902;
                                            if (v901){
                                                v902 = 1;
                                            } else {
                                                v902 = 0;
                                            }
                                            int v903;
                                            v903 = -1 + v879;
                                            int v904; int v905;
                                            Tuple1 tmp23 = Tuple1{0, 0};
                                            v904 = tmp23.v0; v905 = tmp23.v1;
                                            while (while_method_3(v904)){
                                                int v907;
                                                v907 = v878[v904];
                                                bool v909;
                                                v909 = v905 >= v907;
                                                int v910;
                                                if (v909){
                                                    v910 = v905;
                                                } else {
                                                    v910 = v907;
                                                }
                                                v905 = v910;
                                                v904 += 1 ;
                                            }
                                            static_array<int,2> v911;
                                            int v913;
                                            v913 = 0;
                                            while (while_method_3(v913)){
                                                v911[v913] = v905;
                                                v913 += 1 ;
                                            }
                                            static_array<int,2> v915;
                                            int v917;
                                            v917 = 0;
                                            while (while_method_3(v917)){
                                                int v919;
                                                v919 = v911[v917];
                                                bool v921;
                                                v921 = v917 == v877;
                                                int v923;
                                                if (v921){
                                                    int v922;
                                                    v922 = v919 + 4;
                                                    v923 = v922;
                                                } else {
                                                    v923 = v919;
                                                }
                                                v915[v917] = v923;
                                                v917 += 1 ;
                                            }
                                            v967 = Union4{Union4_2{v874, false, v876, v902, v915, v903}};
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
                        v1024 = Union6{Union6_1{v967}};
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
                        static_array<float,2> & v45 = v2.base->v4;
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
                        v1024 = Union6{Union6_0{}};
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
                        static_array<float,2> & v19 = v2.base->v4;
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
                        v1024 = Union6{Union6_0{}};
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
        v6 = v1024;
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
    sptr<Mut0> v22;
    v22 = sptr<Mut0>{new Mut0{63u, v20, v16, v10, v18, v21}};
    int v23;
    v23 = 0;
    while (while_method_0(v23)){
        int v25;
        v25 = 0;
        while (while_method_1(v25)){
            v22.base->v0 = 63u;
            static_array<float,2> v27;
            v27[0] = 0.0f;
            v27[1] = 0.0f;
            v22.base->v4 = v27;
            static_array_list<Union1,32> & v29 = v22.base->v2;
            v29.unsafe_set_length(0);
            static_array<Union0,2> v30;
            Union0 v32;
            v32 = Union0{Union0_1{}};
            v30[0] = v32;
            Union0 v34;
            v34 = Union0{Union0_1{}};
            v30[1] = v34;
            Union0 v36;
            v36 = Union0{Union0_0{}};
            v30[0] = v36;
            v22.base->v3 = v30;
            Union4 v38;
            v38 = Union4{Union4_1{}};
            method_0(v0, v1, v22, v38);
            static_array<float,2> & v39 = v22.base->v4;
            static_array<float,2> v40;
            v40 = v39;
            unsigned int * v41;
            v41 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
            int * v43;
            v43 = reinterpret_cast<int *>(&v1[262144ull]);
            float * v45;
            v45 = reinterpret_cast<float *>(&v1[262160ull]);
            float * v47;
            v47 = reinterpret_cast<float *>(&v1[524304ull]);
            float * v49;
            v49 = reinterpret_cast<float *>(&v1[786448ull]);
            float * v51;
            v51 = reinterpret_cast<float *>(&v1[1048592ull]);
            float * v53;
            v53 = reinterpret_cast<float *>(&v1[1310736ull]);
            float * v55;
            v55 = reinterpret_cast<float *>(&v1[1572880ull]);
            float * v57;
            v57 = reinterpret_cast<float *>(&v1[1835024ull]);
            int * v59;
            v59 = reinterpret_cast<int *>(&v0[6389760ull]);
            float * v61;
            v61 = reinterpret_cast<float *>(&v0[7962624ull]);
            int * v63;
            v63 = reinterpret_cast<int *>(&v0[9535488ull]);
            int * v65;
            v65 = reinterpret_cast<int *>(&v0[11108352ull]);
            double * v67;
            v67 = reinterpret_cast<double *>(&v0[12681216ull]);
            double * v69;
            v69 = reinterpret_cast<double *>(&v0[18972672ull]);
            double * v71;
            v71 = reinterpret_cast<double *>(&v1[2097168ull]);
            double * v73;
            v73 = reinterpret_cast<double *>(&v1[2490384ull]);
            int * v75;
            v75 = reinterpret_cast<int *>(&v1[2883600ull]);
            int v77;
            v77 = 0;
            while (while_method_0(v77)){
                int v79;
                v79 = threadIdx.x;
                int v80;
                v80 = blockIdx.x;
                int v81;
                v81 = v80 * 256;
                int v82;
                v82 = v79 + v81;
                float v83[2];
                int v84;
                v84 = 0;
                while (while_method_3(v84)){
                    float v86;
                    v86 = v40[v84];
                    v83[v84] = v86;
                    v84 += 1 ;
                }
                assert("Tensor range check" && 0 <= v77 && v77 < 4);
                assert("Tensor range check" && 0 <= v82 && v82 < 6144);
                int v88;
                v88 = 6144 * v77;
                int v89;
                v89 = v88 + v82;
                int v90;
                v90 = v75[v89];
                int v91;
                v91 = v90;
                while (while_method_10(v91)){
                    v91 -= 1 ;
                    assert("Tensor range check" && 0 <= v77 && v77 < 4);
                    assert("Tensor range check" && 0 <= v91 && v91 < 16);
                    assert("Tensor range check" && 0 <= v82 && v82 < 6144);
                    int v93;
                    v93 = 6144 * v91;
                    int v94;
                    v94 = v93 + v82;
                    int v95;
                    v95 = 98304 * v77;
                    int v96;
                    v96 = v95 + v94;
                    int v97;
                    v97 = v59[v96];
                    float v98;
                    v98 = v61[v96];
                    int v99;
                    v99 = v63[v96];
                    int v100;
                    v100 = v65[v96];
                    assert("Tensor range check" && 0 <= v99 && v99 < 2);
                    float v101;
                    v101 = v83[v99];
                    assert("Tensor range check" && 0 <= v77 && v77 < 4);
                    int v102;
                    v102 = 16384 * v77;
                    assert("Tensor range check" && 0 <= v100 && v100 < 4096);
                    int v103;
                    v103 = 4 * v100;
                    int v104;
                    v104 = v103 + v102;
                    float * v105;
                    v105 = v45+v104;
                    float * v107;
                    v107 = v47+v104;
                    float * v109;
                    v109 = v49+v104;
                    float * v111;
                    v111 = v51+v104;
                    float * v113;
                    v113 = v53+v104;
                    float * v115;
                    v115 = v55+v104;
                    float * v117;
                    v117 = v57+v104;
                    assert("Tensor range check" && 0 <= v77 && v77 < 4);
                    int v119;
                    v119 = 196608 * v77;
                    assert("Tensor range check" && 0 <= v91 && v91 < 16);
                    int v120;
                    v120 = 12288 * v91;
                    int v121;
                    v121 = v120 + v119;
                    assert("Tensor range check" && 0 <= v82 && v82 < 6144);
                    int v122;
                    v122 = 2 * v82;
                    int v123;
                    v123 = v122 + v121;
                    double v124[2];
                    int v125;
                    v125 = 0;
                    while (while_method_3(v125)){
                        assert("Tensor range check" && 0 <= v125 && v125 < 2);
                        int v127;
                        v127 = v125 + v123;
                        double v128;
                        v128 = v67[v127];
                        bool v129;
                        v129 = v99 == v125;
                        double v130;
                        if (v129){
                            v130 = 0.0;
                        } else {
                            v130 = v128;
                        }
                        assert("Tensor range check" && 0 <= v125 && v125 < 2);
                        v124[v125] = v130;
                        v125 += 1 ;
                    }
                    double v131;
                    v131 = 0.0;
                    int v132;
                    v132 = 0;
                    while (while_method_3(v132)){
                        assert("Tensor range check" && 0 <= v132 && v132 < 2);
                        double v134;
                        v134 = v124[v132];
                        double v135;
                        v135 = v131 + v134;
                        v131 = v135;
                        v132 += 1 ;
                    }
                    double v136;
                    v136 = 0.0;
                    int v137;
                    v137 = 0;
                    while (while_method_3(v137)){
                        assert("Tensor range check" && 0 <= v137 && v137 < 2);
                        int v139;
                        v139 = v137 + v123;
                        double v140;
                        v140 = v69[v139];
                        double v141;
                        v141 = v136 + v140;
                        v136 = v141;
                        v137 += 1 ;
                    }
                    double v142;
                    v142 = v131 - v136;
                    double v143;
                    v143 = exp(v142);
                    float v144;
                    v144 = (float)v143;
                    float v145;
                    v145 = v101 * v144;
                    assert("Tensor range check" && 0 <= v97 && v97 < 4);
                    float * v146;
                    v146 = v115+v97;
                    float * v148;
                    v148 = v117+v97;
                    float v150;
                    v150 = atomicAdd(v146,v145);
                    float v151;
                    v151 = atomicAdd(v148,v144);
                    float * v152;
                    v152 = v107+0;
                    float * v154;
                    v154 = v111+0;
                    float * v156;
                    v156 = v113+0;
                    int v158;
                    v158 = sizeof(float *);
                    unsigned long long v159;
                    v159 = (unsigned long long)v158;
                    unsigned long long v160;
                    v160 = 256ull * v159;
                    unsigned long long v161;
                    v161 = 4096ull + v160;
                    unsigned long long v162;
                    v162 = v161 + 16ull;
                    unsigned long long v163;
                    v163 = v162 - 1ull;
                    unsigned long long v164;
                    v164 = v163 % 16ull;
                    unsigned long long v165;
                    v165 = v163 - v164;
                    unsigned long long v166;
                    v166 = v165 + v160;
                    unsigned long long v167;
                    v167 = v166 + 16ull;
                    unsigned long long v168;
                    v168 = v167 - 1ull;
                    unsigned long long v169;
                    v169 = v168 % 16ull;
                    unsigned long long v170;
                    v170 = v168 - v169;
                    unsigned long long v171;
                    v171 = v170 + v160;
                    unsigned long long v172;
                    v172 = v171 + 16ull;
                    unsigned long long v173;
                    v173 = v172 - 1ull;
                    unsigned long long v174;
                    v174 = v173 % 16ull;
                    unsigned long long v175;
                    v175 = v173 - v174;
                    unsigned long long v176;
                    v176 = v175 + v160;
                    unsigned long long v177;
                    v177 = v176 + 16ull;
                    unsigned long long v178;
                    v178 = v177 - 1ull;
                    unsigned long long v179;
                    v179 = v178 % 16ull;
                    unsigned long long v180;
                    v180 = v178 - v179;
                    unsigned long long v181;
                    v181 = v180 + 1024ull;
                    bool v182;
                    v182 = v181 <= 98304ull;
                    bool v183;
                    v183 = v182 == false;
                    if (v183){
                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v182);
                    } else {
                    }
                    extern __shared__ unsigned char v185[];
                    bool v186;
                    v186 = v181 <= v181;
                    bool v187;
                    v187 = v186 == false;
                    if (v187){
                        assert("The length of the partition has to be less than or equal to the length of the base array." && v186);
                    } else {
                    }
                    float * v189;
                    v189 = reinterpret_cast<float *>(&v185[0ull]);
                    int * v191;
                    v191 = reinterpret_cast<int *>(&v185[1024ull]);
                    float * v193;
                    v193 = reinterpret_cast<float *>(&v185[2048ull]);
                    float * v195;
                    v195 = reinterpret_cast<float *>(&v185[3072ull]);
                    float * * v197;
                    v197 = reinterpret_cast<float * *>(&v185[4096ull]);
                    float * * v199;
                    v199 = reinterpret_cast<float * *>(&v185[v165]);
                    float * * v201;
                    v201 = reinterpret_cast<float * *>(&v185[v170]);
                    float * * v203;
                    v203 = reinterpret_cast<float * *>(&v185[v175]);
                    float * v205;
                    v205 = reinterpret_cast<float *>(&v185[v180]);
                    int v207;
                    v207 = threadIdx.x;
                    assert("Tensor range check" && 0 <= v207 && v207 < 256);
                    v189[v207] = v98;
                    v191[v207] = v97;
                    v193[v207] = v101;
                    v195[v207] = v144;
                    v197[v207] = v109;
                    v199[v207] = v152;
                    v201[v207] = v154;
                    v203[v207] = v156;
                    asm("barrier.cta.sync %0;" :: "r"(0));
                    bool v208;
                    v208 = 0 <= v207;
                    bool v209;
                    v209 = v208 == false;
                    if (v209){
                        assert("The index needs to be zero or positive." && v208);
                    } else {
                    }
                    int v211;
                    v211 = v207 % 1;
                    bool v212;
                    v212 = v207 < 256;
                    bool v213;
                    v213 = v212 == false;
                    if (v213){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v212);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v207 && v207 < 256);
                    int v215;
                    v215 = 0;
                    while (while_method_7(v215)){
                        bool v217;
                        v217 = v208 && v212;
                        bool v218;
                        v218 = v217 == false;
                        if (v218){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v217);
                        } else {
                        }
                        bool v220;
                        v220 = 0 <= v215;
                        bool v222;
                        if (v220){
                            bool v221;
                            v221 = v215 < 1;
                            v222 = v221;
                        } else {
                            v222 = false;
                        }
                        bool v223;
                        v223 = v222 == false;
                        if (v223){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v222);
                        } else {
                        }
                        int v225;
                        v225 = v215 * 256;
                        int v226;
                        v226 = v225 + v207;
                        assert("Tensor range check" && 0 <= v215 && v215 < 1);
                        int v227;
                        v227 = 256 * v215;
                        int v228;
                        v228 = v227 + v207;
                        float v229;
                        v229 = v189[v228];
                        int v230;
                        v230 = v191[v228];
                        float v231;
                        v231 = v193[v228];
                        float v232;
                        v232 = v195[v228];
                        float * v233;
                        v233 = v197[v228];
                        float * v234;
                        v234 = v199[v228];
                        float * v235;
                        v235 = v201[v228];
                        float * v236;
                        v236 = v203[v228];
                        int v237;
                        v237 = blockIdx.x;
                        int v238;
                        v238 = v237 * 256;
                        int v239;
                        v239 = v238 + v226;
                        assert("Tensor range check" && 0 <= v211 && v211 < 1);
                        int v240;
                        v240 = 4 * v211;
                        float v241[4];
                        float v242[4];
                        float v243[4];
                        int v244[4];
                        int v245;
                        v245 = 0;
                        while (while_method_7(v245)){
                            assert("Tensor range check" && 0 <= v245 && v245 < 1);
                            int v247;
                            v247 = 4 * v245;
                            assert("Tensor range check" && 0 <= v245 && v245 < 1);
                            int v248;
                            v248 = v247 + v240;
                            int4* v249;
                            v249 = reinterpret_cast<int4*>(v234 + v248);
                            int4* v250;
                            v250 = reinterpret_cast<int4*>(v241 + v247);
                            assert("Pointer alignment check" && (unsigned long long)(v249) % 4 == 0 && (unsigned long long)(v250) % 4 == 0);
                            *v250 = *v249;
                            int4* v251;
                            v251 = reinterpret_cast<int4*>(v235 + v248);
                            int4* v252;
                            v252 = reinterpret_cast<int4*>(v242 + v247);
                            assert("Pointer alignment check" && (unsigned long long)(v251) % 4 == 0 && (unsigned long long)(v252) % 4 == 0);
                            *v252 = *v251;
                            int4* v253;
                            v253 = reinterpret_cast<int4*>(v236 + v248);
                            int4* v254;
                            v254 = reinterpret_cast<int4*>(v243 + v247);
                            assert("Pointer alignment check" && (unsigned long long)(v253) % 4 == 0 && (unsigned long long)(v254) % 4 == 0);
                            *v254 = *v253;
                            v245 += 1 ;
                        }
                        int v255;
                        v255 = 0;
                        while (while_method_7(v255)){
                            int v257;
                            v257 = 0;
                            while (while_method_0(v257)){
                                bool v259;
                                v259 = 0 <= v257;
                                bool v261;
                                if (v259){
                                    bool v260;
                                    v260 = v257 < 4;
                                    v261 = v260;
                                } else {
                                    v261 = false;
                                }
                                bool v262;
                                v262 = v261 == false;
                                if (v262){
                                    assert("The indices should be inside the range of the dimension." && v261);
                                } else {
                                }
                                bool v264;
                                v264 = 0 <= v211;
                                bool v266;
                                if (v264){
                                    bool v265;
                                    v265 = v211 < 1;
                                    v266 = v265;
                                } else {
                                    v266 = false;
                                }
                                bool v267;
                                v267 = v266 == false;
                                if (v267){
                                    assert("The indices should be inside the range of the dimension." && v266);
                                } else {
                                }
                                int v269;
                                v269 = v211 * 4;
                                int v270;
                                v270 = v257 + v269;
                                bool v271;
                                v271 = 0 <= v255;
                                bool v273;
                                if (v271){
                                    bool v272;
                                    v272 = v255 < 1;
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
                                int v276;
                                v276 = v255 * 4;
                                int v277;
                                v277 = v270 + v276;
                                assert("Tensor range check" && 0 <= v255 && v255 < 1);
                                assert("Tensor range check" && 0 <= v257 && v257 < 4);
                                int v278;
                                v278 = 4 * v255;
                                int v279;
                                v279 = v278 + v257;
                                v244[v279] = v277;
                                v257 += 1 ;
                            }
                            v255 += 1 ;
                        }
                        float v280[4];
                        int v281;
                        v281 = 0;
                        while (while_method_7(v281)){
                            int v283;
                            v283 = 0;
                            while (while_method_0(v283)){
                                assert("Tensor range check" && 0 <= v281 && v281 < 1);
                                assert("Tensor range check" && 0 <= v283 && v283 < 4);
                                int v285;
                                v285 = 4 * v281;
                                int v286;
                                v286 = v285 + v283;
                                float v287;
                                v287 = v242[v286];
                                float v288;
                                v288 = v243[v286];
                                bool v289;
                                v289 = v288 == 0.0f;
                                bool v290;
                                v290 = v289 != true;
                                float v292;
                                if (v290){
                                    float v291;
                                    v291 = v287 / v288;
                                    v292 = v291;
                                } else {
                                    v292 = 0.0f;
                                }
                                assert("Tensor range check" && 0 <= v281 && v281 < 1);
                                assert("Tensor range check" && 0 <= v283 && v283 < 4);
                                v280[v286] = v292;
                                v283 += 1 ;
                            }
                            v281 += 1 ;
                        }
                        bool v293[4];
                        int v294;
                        v294 = 0;
                        while (while_method_7(v294)){
                            int v296;
                            v296 = 0;
                            while (while_method_0(v296)){
                                assert("Tensor range check" && 0 <= v294 && v294 < 1);
                                assert("Tensor range check" && 0 <= v296 && v296 < 4);
                                int v298;
                                v298 = 4 * v294;
                                int v299;
                                v299 = v298 + v296;
                                float v300;
                                v300 = v241[v299];
                                int v301;
                                v301 = v244[v299];
                                bool v302;
                                v302 = v301 < 3;
                                assert("Tensor range check" && 0 <= v294 && v294 < 1);
                                assert("Tensor range check" && 0 <= v296 && v296 < 4);
                                v293[v299] = v302;
                                v296 += 1 ;
                            }
                            v294 += 1 ;
                        }
                        float v303[4];
                        int v304;
                        v304 = 0;
                        while (while_method_7(v304)){
                            int v306;
                            v306 = 0;
                            while (while_method_0(v306)){
                                assert("Tensor range check" && 0 <= v304 && v304 < 1);
                                assert("Tensor range check" && 0 <= v306 && v306 < 4);
                                int v308;
                                v308 = 4 * v304;
                                int v309;
                                v309 = v308 + v306;
                                float v310;
                                v310 = v241[v309];
                                bool v311;
                                v311 = v293[v309];
                                float v314;
                                if (v311){
                                    bool v312;
                                    v312 = 0.0f >= v310;
                                    if (v312){
                                        v314 = 0.0f;
                                    } else {
                                        v314 = v310;
                                    }
                                } else {
                                    v314 = 0.0f;
                                }
                                assert("Tensor range check" && 0 <= v304 && v304 < 1);
                                assert("Tensor range check" && 0 <= v306 && v306 < 4);
                                v303[v309] = v314;
                                v306 += 1 ;
                            }
                            v304 += 1 ;
                        }
                        float v315;
                        v315 = 0.0f;
                        int v316;
                        v316 = 0;
                        while (while_method_7(v316)){
                            int v318;
                            v318 = 0;
                            while (while_method_0(v318)){
                                assert("Tensor range check" && 0 <= v316 && v316 < 1);
                                assert("Tensor range check" && 0 <= v318 && v318 < 4);
                                int v320;
                                v320 = 4 * v316;
                                int v321;
                                v321 = v320 + v318;
                                float v322;
                                v322 = v303[v321];
                                float v323;
                                v323 = v315 + v322;
                                v315 = v323;
                                v318 += 1 ;
                            }
                            v316 += 1 ;
                        }
                        auto v324 = cooperative_groups::coalesced_threads();
                        int v325;
                        v325 = threadIdx.x;
                        auto v326 = cooperative_groups::labeled_partition(v324,v325);
                        Closure1 v327{};
                        float v328;
                        v328 = cooperative_groups::reduce(v326, v315, v327);
                        int v329[4];
                        int v330;
                        v330 = 0;
                        while (while_method_7(v330)){
                            int v332;
                            v332 = 0;
                            while (while_method_0(v332)){
                                assert("Tensor range check" && 0 <= v330 && v330 < 1);
                                assert("Tensor range check" && 0 <= v332 && v332 < 4);
                                int v334;
                                v334 = 4 * v330;
                                int v335;
                                v335 = v334 + v332;
                                bool v336;
                                v336 = v293[v335];
                                int v337;
                                if (v336){
                                    v337 = 1;
                                } else {
                                    v337 = 0;
                                }
                                assert("Tensor range check" && 0 <= v330 && v330 < 1);
                                assert("Tensor range check" && 0 <= v332 && v332 < 4);
                                v329[v335] = v337;
                                v332 += 1 ;
                            }
                            v330 += 1 ;
                        }
                        int v338;
                        v338 = 0;
                        int v339;
                        v339 = 0;
                        while (while_method_7(v339)){
                            int v341;
                            v341 = 0;
                            while (while_method_0(v341)){
                                assert("Tensor range check" && 0 <= v339 && v339 < 1);
                                assert("Tensor range check" && 0 <= v341 && v341 < 4);
                                int v343;
                                v343 = 4 * v339;
                                int v344;
                                v344 = v343 + v341;
                                int v345;
                                v345 = v329[v344];
                                int v346;
                                v346 = v338 + v345;
                                v338 = v346;
                                v341 += 1 ;
                            }
                            v339 += 1 ;
                        }
                        auto v347 = cooperative_groups::coalesced_threads();
                        int v348;
                        v348 = threadIdx.x;
                        auto v349 = cooperative_groups::labeled_partition(v347,v348);
                        Closure2 v350{};
                        int v351;
                        v351 = cooperative_groups::reduce(v349, v338, v350);
                        float v352;
                        v352 = (float)v351;
                        float v353;
                        v353 = 1.0f / v352;
                        float v354[4];
                        int v355;
                        v355 = 0;
                        while (while_method_7(v355)){
                            int v357;
                            v357 = 0;
                            while (while_method_0(v357)){
                                assert("Tensor range check" && 0 <= v355 && v355 < 1);
                                assert("Tensor range check" && 0 <= v357 && v357 < 4);
                                int v359;
                                v359 = 4 * v355;
                                int v360;
                                v360 = v359 + v357;
                                float v361;
                                v361 = v303[v360];
                                bool v362;
                                v362 = v293[v360];
                                bool v363;
                                v363 = v362 == false;
                                float v368;
                                if (v363){
                                    v368 = 0.0f;
                                } else {
                                    bool v364;
                                    v364 = v328 == 0.0f;
                                    bool v365;
                                    v365 = v364 != true;
                                    if (v365){
                                        float v366;
                                        v366 = v361 / v328;
                                        v368 = v366;
                                    } else {
                                        v368 = v353;
                                    }
                                }
                                assert("Tensor range check" && 0 <= v355 && v355 < 1);
                                assert("Tensor range check" && 0 <= v357 && v357 < 4);
                                v354[v360] = v368;
                                v357 += 1 ;
                            }
                            v355 += 1 ;
                        }
                        float v369[4];
                        int v370;
                        v370 = 0;
                        while (while_method_7(v370)){
                            int v372;
                            v372 = 0;
                            while (while_method_0(v372)){
                                assert("Tensor range check" && 0 <= v370 && v370 < 1);
                                assert("Tensor range check" && 0 <= v372 && v372 < 4);
                                int v374;
                                v374 = 4 * v370;
                                int v375;
                                v375 = v374 + v372;
                                float v376;
                                v376 = v280[v375];
                                int v377;
                                v377 = v244[v375];
                                bool v378;
                                v378 = v230 == v377;
                                float v381;
                                if (v378){
                                    float v379;
                                    v379 = v231 - v376;
                                    float v380;
                                    v380 = v379 / v229;
                                    v381 = v380;
                                } else {
                                    v381 = 0.0f;
                                }
                                float v382;
                                v382 = v381 + v376;
                                assert("Tensor range check" && 0 <= v370 && v370 < 1);
                                assert("Tensor range check" && 0 <= v372 && v372 < 4);
                                v369[v375] = v382;
                                v372 += 1 ;
                            }
                            v370 += 1 ;
                        }
                        float v383[4];
                        int v384;
                        v384 = 0;
                        while (while_method_7(v384)){
                            int v386;
                            v386 = 0;
                            while (while_method_0(v386)){
                                assert("Tensor range check" && 0 <= v384 && v384 < 1);
                                assert("Tensor range check" && 0 <= v386 && v386 < 4);
                                int v388;
                                v388 = 4 * v384;
                                int v389;
                                v389 = v388 + v386;
                                float v390;
                                v390 = v354[v389];
                                float v391;
                                v391 = v369[v389];
                                float v392;
                                v392 = v390 * v391;
                                assert("Tensor range check" && 0 <= v384 && v384 < 1);
                                assert("Tensor range check" && 0 <= v386 && v386 < 4);
                                v383[v389] = v392;
                                v386 += 1 ;
                            }
                            v384 += 1 ;
                        }
                        float v393;
                        v393 = 0.0f;
                        int v394;
                        v394 = 0;
                        while (while_method_7(v394)){
                            int v396;
                            v396 = 0;
                            while (while_method_0(v396)){
                                assert("Tensor range check" && 0 <= v394 && v394 < 1);
                                assert("Tensor range check" && 0 <= v396 && v396 < 4);
                                int v398;
                                v398 = 4 * v394;
                                int v399;
                                v399 = v398 + v396;
                                float v400;
                                v400 = v383[v399];
                                float v401;
                                v401 = v393 + v400;
                                v393 = v401;
                                v396 += 1 ;
                            }
                            v394 += 1 ;
                        }
                        auto v402 = cooperative_groups::coalesced_threads();
                        int v403;
                        v403 = threadIdx.x;
                        auto v404 = cooperative_groups::labeled_partition(v402,v403);
                        float v405;
                        v405 = cooperative_groups::reduce(v404, v393, v327);
                        int v406;
                        v406 = 0;
                        while (while_method_7(v406)){
                            int v408;
                            v408 = 0;
                            while (while_method_0(v408)){
                                assert("Tensor range check" && 0 <= v406 && v406 < 1);
                                assert("Tensor range check" && 0 <= v408 && v408 < 4);
                                int v410;
                                v410 = 4 * v406;
                                int v411;
                                v411 = v410 + v408;
                                float v412;
                                v412 = v369[v411];
                                int v413;
                                v413 = v244[v411];
                                float v414;
                                v414 = v412 - v405;
                                float v415;
                                v415 = v232 * v414;
                                assert("Tensor range check" && 0 <= v413 && v413 < 4);
                                float * v416;
                                v416 = v233+v413;
                                float v418;
                                v418 = atomicAdd(v416,v415);
                                v408 += 1 ;
                            }
                            v406 += 1 ;
                        }
                        int v419;
                        v419 = 0;
                        while (while_method_7(v419)){
                            assert("Tensor range check" && 0 <= v419 && v419 < 1);
                            assert("Tensor range check" && 0 <= v419 && v419 < 1);
                            v419 += 1 ;
                        }
                        assert("Tensor range check" && 0 <= v226 && v226 < 256);
                        v205[v226] = v405;
                        v215 += 1 ;
                    }
                    asm("barrier.cta.sync %0;" :: "r"(0));
                    assert("Tensor range check" && 0 <= v207 && v207 < 256);
                    float v421;
                    v421 = v205[v207];
                    asm("barrier.cta.sync %0;" :: "r"(0));
                    assert("Tensor range check" && 0 <= v99 && v99 < 2);
                    v83[v99] = v421;
                }
                int v422;
                v422 = threadIdx.x;
                int v423;
                v423 = blockIdx.x;
                int v424;
                v424 = v423 * 256;
                int v425;
                v425 = v422 + v424;
                assert("Tensor range check" && 0 <= v77 && v77 < 4);
                int v426;
                v426 = 12288 * v77;
                assert("Tensor range check" && 0 <= v425 && v425 < 6144);
                int v427;
                v427 = 2 * v425;
                int v428;
                v428 = v427 + v426;
                double * v429;
                v429 = v71+v428;
                double * v431;
                v431 = v73+v428;
                double * v433;
                v433 = v429+0;
                double * v435;
                v435 = v431+0;
                double * v437;
                v437 = v429+0;
                double * v439;
                v439 = v431+0;
                int v441;
                v441 = sizeof(double *);
                unsigned long long v442;
                v442 = (unsigned long long)v441;
                unsigned long long v443;
                v443 = 256ull * v442;
                unsigned long long v444;
                v444 = v443 + 16ull;
                unsigned long long v445;
                v445 = v444 - 1ull;
                unsigned long long v446;
                v446 = v445 % 16ull;
                unsigned long long v447;
                v447 = v445 - v446;
                unsigned long long v448;
                v448 = v447 + v443;
                unsigned long long v449;
                v449 = v448 + 16ull;
                unsigned long long v450;
                v450 = v449 - 1ull;
                unsigned long long v451;
                v451 = v450 % 16ull;
                unsigned long long v452;
                v452 = v450 - v451;
                unsigned long long v453;
                v453 = v452 + v443;
                unsigned long long v454;
                v454 = v453 + 16ull;
                unsigned long long v455;
                v455 = v454 - 1ull;
                unsigned long long v456;
                v456 = v455 % 16ull;
                unsigned long long v457;
                v457 = v455 - v456;
                unsigned long long v458;
                v458 = v457 + v443;
                bool v459;
                v459 = v458 <= 98304ull;
                bool v460;
                v460 = v459 == false;
                if (v460){
                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v459);
                } else {
                }
                extern __shared__ unsigned char v462[];
                bool v463;
                v463 = v458 <= v458;
                bool v464;
                v464 = v463 == false;
                if (v464){
                    assert("The length of the partition has to be less than or equal to the length of the base array." && v463);
                } else {
                }
                double * * v466;
                v466 = reinterpret_cast<double * *>(&v462[0ull]);
                double * * v468;
                v468 = reinterpret_cast<double * *>(&v462[v447]);
                double * * v470;
                v470 = reinterpret_cast<double * *>(&v462[v452]);
                double * * v472;
                v472 = reinterpret_cast<double * *>(&v462[v457]);
                int v474;
                v474 = threadIdx.x;
                assert("Tensor range check" && 0 <= v474 && v474 < 256);
                v466[v474] = v433;
                v468[v474] = v435;
                v470[v474] = v437;
                v472[v474] = v439;
                asm("barrier.cta.sync %0;" :: "r"(0));
                bool v475;
                v475 = 0 <= v474;
                bool v476;
                v476 = v475 == false;
                if (v476){
                    assert("The index needs to be zero or positive." && v475);
                } else {
                }
                int v478;
                v478 = v474 % 1;
                bool v479;
                v479 = v474 < 256;
                bool v480;
                v480 = v479 == false;
                if (v480){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v479);
                } else {
                }
                assert("Tensor range check" && 0 <= v474 && v474 < 256);
                int v482;
                v482 = 0;
                while (while_method_7(v482)){
                    bool v484;
                    v484 = v475 && v479;
                    bool v485;
                    v485 = v484 == false;
                    if (v485){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v484);
                    } else {
                    }
                    bool v487;
                    v487 = 0 <= v482;
                    bool v489;
                    if (v487){
                        bool v488;
                        v488 = v482 < 1;
                        v489 = v488;
                    } else {
                        v489 = false;
                    }
                    bool v490;
                    v490 = v489 == false;
                    if (v490){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v489);
                    } else {
                    }
                    int v492;
                    v492 = v482 * 256;
                    int v493;
                    v493 = v492 + v474;
                    assert("Tensor range check" && 0 <= v482 && v482 < 1);
                    int v494;
                    v494 = 256 * v482;
                    int v495;
                    v495 = v494 + v474;
                    double * v496;
                    v496 = v466[v495];
                    double * v497;
                    v497 = v468[v495];
                    double * v498;
                    v498 = v470[v495];
                    double * v499;
                    v499 = v472[v495];
                    int v500;
                    v500 = blockIdx.x;
                    int v501;
                    v501 = v500 * 256;
                    int v502;
                    v502 = v501 + v493;
                    assert("Tensor range check" && 0 <= v478 && v478 < 1);
                    int v503;
                    v503 = 2 * v478;
                    double v504[2];
                    double v505[2];
                    int v506[2];
                    int v507;
                    v507 = 0;
                    while (while_method_7(v507)){
                        assert("Tensor range check" && 0 <= v507 && v507 < 1);
                        int v509;
                        v509 = 2 * v507;
                        assert("Tensor range check" && 0 <= v507 && v507 < 1);
                        int v510;
                        v510 = v509 + v503;
                        int4* v511;
                        v511 = reinterpret_cast<int4*>(v496 + v510);
                        int4* v512;
                        v512 = reinterpret_cast<int4*>(v504 + v509);
                        assert("Pointer alignment check" && (unsigned long long)(v511) % 2 == 0 && (unsigned long long)(v512) % 2 == 0);
                        *v512 = *v511;
                        int4* v513;
                        v513 = reinterpret_cast<int4*>(v497 + v510);
                        int4* v514;
                        v514 = reinterpret_cast<int4*>(v505 + v509);
                        assert("Pointer alignment check" && (unsigned long long)(v513) % 2 == 0 && (unsigned long long)(v514) % 2 == 0);
                        *v514 = *v513;
                        v507 += 1 ;
                    }
                    int v515;
                    v515 = 0;
                    while (while_method_7(v515)){
                        int v517;
                        v517 = 0;
                        while (while_method_3(v517)){
                            bool v519;
                            v519 = 0 <= v517;
                            bool v521;
                            if (v519){
                                bool v520;
                                v520 = v517 < 2;
                                v521 = v520;
                            } else {
                                v521 = false;
                            }
                            bool v522;
                            v522 = v521 == false;
                            if (v522){
                                assert("The indices should be inside the range of the dimension." && v521);
                            } else {
                            }
                            bool v524;
                            v524 = 0 <= v478;
                            bool v526;
                            if (v524){
                                bool v525;
                                v525 = v478 < 1;
                                v526 = v525;
                            } else {
                                v526 = false;
                            }
                            bool v527;
                            v527 = v526 == false;
                            if (v527){
                                assert("The indices should be inside the range of the dimension." && v526);
                            } else {
                            }
                            int v529;
                            v529 = v478 * 2;
                            int v530;
                            v530 = v517 + v529;
                            bool v531;
                            v531 = 0 <= v515;
                            bool v533;
                            if (v531){
                                bool v532;
                                v532 = v515 < 1;
                                v533 = v532;
                            } else {
                                v533 = false;
                            }
                            bool v534;
                            v534 = v533 == false;
                            if (v534){
                                assert("The indices should be inside the range of the dimension." && v533);
                            } else {
                            }
                            int v536;
                            v536 = v515 * 2;
                            int v537;
                            v537 = v530 + v536;
                            assert("Tensor range check" && 0 <= v515 && v515 < 1);
                            assert("Tensor range check" && 0 <= v517 && v517 < 2);
                            int v538;
                            v538 = 2 * v515;
                            int v539;
                            v539 = v538 + v517;
                            v506[v539] = v537;
                            v517 += 1 ;
                        }
                        v515 += 1 ;
                    }
                    double v540[2];
                    double v541[2];
                    int v542;
                    v542 = 0;
                    while (while_method_7(v542)){
                        int v544;
                        v544 = 0;
                        while (while_method_3(v544)){
                            assert("Tensor range check" && 0 <= v542 && v542 < 1);
                            assert("Tensor range check" && 0 <= v544 && v544 < 2);
                            int v546;
                            v546 = 2 * v542;
                            int v547;
                            v547 = v546 + v544;
                            double v548;
                            v548 = v504[v547];
                            double v549;
                            v549 = v505[v547];
                            assert("Tensor range check" && 0 <= v542 && v542 < 1);
                            assert("Tensor range check" && 0 <= v544 && v544 < 2);
                            v540[v547] = 0.0;
                            v541[v547] = 0.0;
                            v544 += 1 ;
                        }
                        v542 += 1 ;
                    }
                    int v550;
                    v550 = 0;
                    while (while_method_7(v550)){
                        assert("Tensor range check" && 0 <= v550 && v550 < 1);
                        int v552;
                        v552 = 2 * v550;
                        int v553;
                        v553 = v552 + v503;
                        assert("Tensor range check" && 0 <= v550 && v550 < 1);
                        int4* v554;
                        v554 = reinterpret_cast<int4*>(v540 + v552);
                        int4* v555;
                        v555 = reinterpret_cast<int4*>(v498 + v553);
                        assert("Pointer alignment check" && (unsigned long long)(v554) % 2 == 0 && (unsigned long long)(v555) % 2 == 0);
                        *v555 = *v554;
                        int4* v556;
                        v556 = reinterpret_cast<int4*>(v541 + v552);
                        int4* v557;
                        v557 = reinterpret_cast<int4*>(v499 + v553);
                        assert("Pointer alignment check" && (unsigned long long)(v556) % 2 == 0 && (unsigned long long)(v557) % 2 == 0);
                        *v557 = *v556;
                        v550 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v493 && v493 < 256);
                    v482 += 1 ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0));
                assert("Tensor range check" && 0 <= v474 && v474 < 256);
                asm("barrier.cta.sync %0;" :: "r"(0));
                assert("Tensor range check" && 0 <= v77 && v77 < 4);
                assert("Tensor range check" && 0 <= v425 && v425 < 6144);
                int v558;
                v558 = v88 + v425;
                v75[v558] = 0;
                v77 += 1 ;
            }
            v25 += 1 ;
        }
        cooperative_groups::grid_group & v559 = v22.base->v1;
        cooperative_groups::grid_group & v560 = v559;
        curandStatePhilox4_32_10_t & v561 = v22.base->v5;
        curandStatePhilox4_32_10_t & v562 = v561;
        unsigned int * v563;
        v563 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
        int * v565;
        v565 = reinterpret_cast<int *>(&v1[262144ull]);
        float * v567;
        v567 = reinterpret_cast<float *>(&v1[262160ull]);
        float * v569;
        v569 = reinterpret_cast<float *>(&v1[524304ull]);
        float * v571;
        v571 = reinterpret_cast<float *>(&v1[786448ull]);
        float * v573;
        v573 = reinterpret_cast<float *>(&v1[1048592ull]);
        float * v575;
        v575 = reinterpret_cast<float *>(&v1[1310736ull]);
        float * v577;
        v577 = reinterpret_cast<float *>(&v1[1572880ull]);
        float * v579;
        v579 = reinterpret_cast<float *>(&v1[1835024ull]);
        int * v581;
        v581 = reinterpret_cast<int *>(&v0[6389760ull]);
        float * v583;
        v583 = reinterpret_cast<float *>(&v0[7962624ull]);
        int * v585;
        v585 = reinterpret_cast<int *>(&v0[9535488ull]);
        int * v587;
        v587 = reinterpret_cast<int *>(&v0[11108352ull]);
        double * v589;
        v589 = reinterpret_cast<double *>(&v0[12681216ull]);
        double * v591;
        v591 = reinterpret_cast<double *>(&v0[18972672ull]);
        double * v593;
        v593 = reinterpret_cast<double *>(&v1[2097168ull]);
        double * v595;
        v595 = reinterpret_cast<double *>(&v1[2490384ull]);
        int * v597;
        v597 = reinterpret_cast<int *>(&v1[2883600ull]);
        v560.sync() ;
        int v599;
        v599 = threadIdx.x;
        int v600;
        v600 = blockIdx.x;
        int v601;
        v601 = v600 * 256;
        int v602;
        v602 = v599 + v601;
        bool v603;
        v603 = v602 == 0;
        if (v603){
            int v604;
            v604 = 0;
            int v605;
            v605 = 4;
            int v606;
            v606 = int_range_6(v605, v604, v562);
            v565[0] = v606;
        } else {
        }
        __syncwarp();
        int v607;
        v607 = threadIdx.x;
        bool v608;
        v608 = 0 <= v607;
        bool v609;
        v609 = v608 == false;
        if (v609){
            assert("The index needs to be zero or positive." && v608);
        } else {
        }
        int v611;
        v611 = v607 % 1;
        int v612;
        v612 = v607 % 256;
        int v613;
        v613 = v607 / 256;
        bool v614;
        v614 = v613 < 1;
        bool v615;
        v615 = v614 == false;
        if (v615){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v614);
        } else {
        }
        assert("Tensor range check" && 0 <= v613 && v613 < 1);
        assert("Tensor range check" && 0 <= v612 && v612 < 256);
        assert("Tensor range check" && 0 <= v611 && v611 < 1);
        int v617;
        v617 = 4 * v611;
        int v618;
        v618 = 4 * v612;
        int v619;
        v619 = v618 + v617;
        int v620;
        v620 = 16384 * v613;
        int v621;
        v621 = v620 + v619;
        assert("Tensor range check" && 0 <= v613 && v613 < 1);
        assert("Tensor range check" && 0 <= v612 && v612 < 256);
        assert("Tensor range check" && 0 <= v611 && v611 < 1);
        int v622;
        v622 = blockIdx.x;
        int v623;
        v623 = v622;
        while (while_method_11(v623)){
            bool v625;
            v625 = 0 <= v623;
            bool v626;
            v626 = v625 == false;
            if (v626){
                assert("The index needs to be zero or positive." && v625);
            } else {
            }
            int v628;
            v628 = v623 % 16;
            int v629;
            v629 = v623 / 16;
            bool v630;
            v630 = v629 < 4;
            bool v631;
            v631 = v630 == false;
            if (v631){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v630);
            } else {
            }
            assert("Tensor range check" && 0 <= v629 && v629 < 4);
            assert("Tensor range check" && 0 <= v628 && v628 < 16);
            int v633;
            v633 = 1024 * v628;
            int v634;
            v634 = v633 + v621;
            int v635;
            v635 = 16384 * v629;
            int v636;
            v636 = v635 + v634;
            float v637[4];
            float v638[4];
            float v639[4];
            float v640[4];
            float v641[4];
            float v642[4];
            float v643[4];
            int v644[4];
            int v645;
            v645 = 0;
            while (while_method_7(v645)){
                assert("Tensor range check" && 0 <= v645 && v645 < 1);
                int v647;
                v647 = 4 * v645;
                assert("Tensor range check" && 0 <= v645 && v645 < 1);
                int v648;
                v648 = v647 + v636;
                int4* v649;
                v649 = reinterpret_cast<int4*>(v567 + v648);
                int4* v650;
                v650 = reinterpret_cast<int4*>(v637 + v647);
                assert("Pointer alignment check" && (unsigned long long)(v649) % 4 == 0 && (unsigned long long)(v650) % 4 == 0);
                *v650 = *v649;
                int4* v651;
                v651 = reinterpret_cast<int4*>(v569 + v648);
                int4* v652;
                v652 = reinterpret_cast<int4*>(v638 + v647);
                assert("Pointer alignment check" && (unsigned long long)(v651) % 4 == 0 && (unsigned long long)(v652) % 4 == 0);
                *v652 = *v651;
                int4* v653;
                v653 = reinterpret_cast<int4*>(v571 + v648);
                int4* v654;
                v654 = reinterpret_cast<int4*>(v639 + v647);
                assert("Pointer alignment check" && (unsigned long long)(v653) % 4 == 0 && (unsigned long long)(v654) % 4 == 0);
                *v654 = *v653;
                int4* v655;
                v655 = reinterpret_cast<int4*>(v573 + v648);
                int4* v656;
                v656 = reinterpret_cast<int4*>(v640 + v647);
                assert("Pointer alignment check" && (unsigned long long)(v655) % 4 == 0 && (unsigned long long)(v656) % 4 == 0);
                *v656 = *v655;
                int4* v657;
                v657 = reinterpret_cast<int4*>(v575 + v648);
                int4* v658;
                v658 = reinterpret_cast<int4*>(v641 + v647);
                assert("Pointer alignment check" && (unsigned long long)(v657) % 4 == 0 && (unsigned long long)(v658) % 4 == 0);
                *v658 = *v657;
                int4* v659;
                v659 = reinterpret_cast<int4*>(v577 + v648);
                int4* v660;
                v660 = reinterpret_cast<int4*>(v642 + v647);
                assert("Pointer alignment check" && (unsigned long long)(v659) % 4 == 0 && (unsigned long long)(v660) % 4 == 0);
                *v660 = *v659;
                int4* v661;
                v661 = reinterpret_cast<int4*>(v579 + v648);
                int4* v662;
                v662 = reinterpret_cast<int4*>(v643 + v647);
                assert("Pointer alignment check" && (unsigned long long)(v661) % 4 == 0 && (unsigned long long)(v662) % 4 == 0);
                *v662 = *v661;
                v645 += 1 ;
            }
            int v663;
            v663 = 0;
            while (while_method_7(v663)){
                int v665;
                v665 = 0;
                while (while_method_0(v665)){
                    bool v667;
                    v667 = 0 <= v665;
                    bool v669;
                    if (v667){
                        bool v668;
                        v668 = v665 < 4;
                        v669 = v668;
                    } else {
                        v669 = false;
                    }
                    bool v670;
                    v670 = v669 == false;
                    if (v670){
                        assert("The indices should be inside the range of the dimension." && v669);
                    } else {
                    }
                    bool v672;
                    v672 = 0 <= v611;
                    bool v674;
                    if (v672){
                        bool v673;
                        v673 = v611 < 1;
                        v674 = v673;
                    } else {
                        v674 = false;
                    }
                    bool v675;
                    v675 = v674 == false;
                    if (v675){
                        assert("The indices should be inside the range of the dimension." && v674);
                    } else {
                    }
                    int v677;
                    v677 = v611 * 4;
                    int v678;
                    v678 = v665 + v677;
                    bool v679;
                    v679 = 0 <= v663;
                    bool v681;
                    if (v679){
                        bool v680;
                        v680 = v663 < 1;
                        v681 = v680;
                    } else {
                        v681 = false;
                    }
                    bool v682;
                    v682 = v681 == false;
                    if (v682){
                        assert("The indices should be inside the range of the dimension." && v681);
                    } else {
                    }
                    int v684;
                    v684 = v663 * 4;
                    int v685;
                    v685 = v678 + v684;
                    assert("Tensor range check" && 0 <= v663 && v663 < 1);
                    assert("Tensor range check" && 0 <= v665 && v665 < 4);
                    int v686;
                    v686 = 4 * v663;
                    int v687;
                    v687 = v686 + v665;
                    v644[v687] = v685;
                    v665 += 1 ;
                }
                v663 += 1 ;
            }
            bool v688;
            v688 = 0 <= v613;
            bool v689;
            v689 = v688 && v614;
            bool v690;
            v690 = v689 == false;
            if (v690){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v689);
            } else {
            }
            bool v692;
            v692 = 0 <= v612;
            bool v694;
            if (v692){
                bool v693;
                v693 = v612 < 256;
                v694 = v693;
            } else {
                v694 = false;
            }
            bool v695;
            v695 = v694 == false;
            if (v695){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v694);
            } else {
            }
            bool v697;
            v697 = 0 <= v629;
            bool v698;
            v698 = v697 && v630;
            bool v699;
            v699 = v698 == false;
            if (v699){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v698);
            } else {
            }
            bool v701;
            v701 = 0 <= v628;
            bool v703;
            if (v701){
                bool v702;
                v702 = v628 < 16;
                v703 = v702;
            } else {
                v703 = false;
            }
            bool v704;
            v704 = v703 == false;
            if (v704){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v703);
            } else {
            }
            int v706;
            v706 = v628 * 256;
            int v707;
            v707 = v629 + v613;
            int v708;
            v708 = v706 + v612;
            bool v709[4];
            int v710;
            v710 = 0;
            while (while_method_7(v710)){
                int v712;
                v712 = 0;
                while (while_method_0(v712)){
                    assert("Tensor range check" && 0 <= v710 && v710 < 1);
                    assert("Tensor range check" && 0 <= v712 && v712 < 4);
                    int v714;
                    v714 = 4 * v710;
                    int v715;
                    v715 = v714 + v712;
                    float v716;
                    v716 = v639[v715];
                    bool v717;
                    v717 = v716 == 0.0f;
                    bool v718;
                    v718 = v717 != true;
                    assert("Tensor range check" && 0 <= v710 && v710 < 1);
                    assert("Tensor range check" && 0 <= v712 && v712 < 4);
                    v709[v715] = v718;
                    v712 += 1 ;
                }
                v710 += 1 ;
            }
            bool v719;
            v719 = false;
            int v720;
            v720 = 0;
            while (while_method_7(v720)){
                int v722;
                v722 = 0;
                while (while_method_0(v722)){
                    assert("Tensor range check" && 0 <= v720 && v720 < 1);
                    assert("Tensor range check" && 0 <= v722 && v722 < 4);
                    int v724;
                    v724 = 4 * v720;
                    int v725;
                    v725 = v724 + v722;
                    bool v726;
                    v726 = v709[v725];
                    bool v727;
                    v727 = v719 || v726;
                    v719 = v727;
                    v722 += 1 ;
                }
                v720 += 1 ;
            }
            auto v728 = cooperative_groups::coalesced_threads();
            int v729;
            v729 = threadIdx.x;
            auto v730 = cooperative_groups::labeled_partition(v728,v729);
            Closure8 v731{};
            bool v732;
            v732 = cooperative_groups::reduce(v730, v719, v731);
            if (v732){
                float v733[4];
                int v734;
                v734 = 0;
                while (while_method_7(v734)){
                    int v736;
                    v736 = 0;
                    while (while_method_0(v736)){
                        assert("Tensor range check" && 0 <= v734 && v734 < 1);
                        assert("Tensor range check" && 0 <= v736 && v736 < 4);
                        int v738;
                        v738 = 4 * v734;
                        int v739;
                        v739 = v738 + v736;
                        float v740;
                        v740 = v638[v739];
                        float v741;
                        v741 = v639[v739];
                        float v742;
                        v742 = v740 + v741;
                        bool v743;
                        v743 = 0.0f >= v742;
                        float v744;
                        if (v743){
                            v744 = 0.0f;
                        } else {
                            v744 = v742;
                        }
                        assert("Tensor range check" && 0 <= v734 && v734 < 1);
                        assert("Tensor range check" && 0 <= v736 && v736 < 4);
                        v733[v739] = v744;
                        v736 += 1 ;
                    }
                    v734 += 1 ;
                }
                float v745[4];
                int v746;
                v746 = 0;
                while (while_method_7(v746)){
                    int v748;
                    v748 = 0;
                    while (while_method_0(v748)){
                        assert("Tensor range check" && 0 <= v746 && v746 < 1);
                        assert("Tensor range check" && 0 <= v748 && v748 < 4);
                        int v750;
                        v750 = 4 * v746;
                        int v751;
                        v751 = v750 + v748;
                        float v752;
                        v752 = v733[v751];
                        bool v753;
                        v753 = 0.0f >= v752;
                        float v754;
                        if (v753){
                            v754 = 0.0f;
                        } else {
                            v754 = v752;
                        }
                        assert("Tensor range check" && 0 <= v746 && v746 < 1);
                        assert("Tensor range check" && 0 <= v748 && v748 < 4);
                        v745[v751] = v754;
                        v748 += 1 ;
                    }
                    v746 += 1 ;
                }
                float v755;
                v755 = 0.0f;
                int v756;
                v756 = 0;
                while (while_method_7(v756)){
                    int v758;
                    v758 = 0;
                    while (while_method_0(v758)){
                        assert("Tensor range check" && 0 <= v756 && v756 < 1);
                        assert("Tensor range check" && 0 <= v758 && v758 < 4);
                        int v760;
                        v760 = 4 * v756;
                        int v761;
                        v761 = v760 + v758;
                        float v762;
                        v762 = v745[v761];
                        float v763;
                        v763 = v755 + v762;
                        v755 = v763;
                        v758 += 1 ;
                    }
                    v756 += 1 ;
                }
                auto v764 = cooperative_groups::coalesced_threads();
                int v765;
                v765 = threadIdx.x;
                auto v766 = cooperative_groups::labeled_partition(v764,v765);
                Closure1 v767{};
                float v768;
                v768 = cooperative_groups::reduce(v766, v755, v767);
                float v769[4];
                int v770;
                v770 = 0;
                while (while_method_7(v770)){
                    int v772;
                    v772 = 0;
                    while (while_method_0(v772)){
                        assert("Tensor range check" && 0 <= v770 && v770 < 1);
                        assert("Tensor range check" && 0 <= v772 && v772 < 4);
                        int v774;
                        v774 = 4 * v770;
                        int v775;
                        v775 = v774 + v772;
                        float v776;
                        v776 = v745[v775];
                        bool v777;
                        v777 = v768 == 0.0f;
                        bool v778;
                        v778 = v777 != true;
                        float v780;
                        if (v778){
                            float v779;
                            v779 = v776 / v768;
                            v780 = v779;
                        } else {
                            v780 = 0.25f;
                        }
                        assert("Tensor range check" && 0 <= v770 && v770 < 1);
                        assert("Tensor range check" && 0 <= v772 && v772 < 4);
                        v769[v775] = v780;
                        v772 += 1 ;
                    }
                    v770 += 1 ;
                }
                float v781[4];
                int v782;
                v782 = 0;
                while (while_method_7(v782)){
                    int v784;
                    v784 = 0;
                    while (while_method_0(v784)){
                        assert("Tensor range check" && 0 <= v782 && v782 < 1);
                        assert("Tensor range check" && 0 <= v784 && v784 < 4);
                        int v786;
                        v786 = 4 * v782;
                        int v787;
                        v787 = v786 + v784;
                        float v788;
                        v788 = v637[v787];
                        float v789;
                        v789 = v769[v787];
                        float v790;
                        v790 = v788 + v789;
                        assert("Tensor range check" && 0 <= v782 && v782 < 1);
                        assert("Tensor range check" && 0 <= v784 && v784 < 4);
                        v781[v787] = v790;
                        v784 += 1 ;
                    }
                    v782 += 1 ;
                }
                float v791[4];
                int v792;
                v792 = 0;
                while (while_method_7(v792)){
                    int v794;
                    v794 = 0;
                    while (while_method_0(v794)){
                        assert("Tensor range check" && 0 <= v792 && v792 < 1);
                        assert("Tensor range check" && 0 <= v794 && v794 < 4);
                        int v796;
                        v796 = 4 * v792;
                        int v797;
                        v797 = v796 + v794;
                        float v798;
                        v798 = v781[v797];
                        float v799;
                        v799 = -v798;
                        bool v800;
                        v800 = v798 >= v799;
                        float v801;
                        if (v800){
                            v801 = v798;
                        } else {
                            v801 = v799;
                        }
                        assert("Tensor range check" && 0 <= v792 && v792 < 1);
                        assert("Tensor range check" && 0 <= v794 && v794 < 4);
                        v791[v797] = v801;
                        v794 += 1 ;
                    }
                    v792 += 1 ;
                }
                float v802;
                v802 = 0.0f;
                int v803;
                v803 = 0;
                while (while_method_7(v803)){
                    int v805;
                    v805 = 0;
                    while (while_method_0(v805)){
                        assert("Tensor range check" && 0 <= v803 && v803 < 1);
                        assert("Tensor range check" && 0 <= v805 && v805 < 4);
                        int v807;
                        v807 = 4 * v803;
                        int v808;
                        v808 = v807 + v805;
                        float v809;
                        v809 = v791[v808];
                        float v810;
                        v810 = v802 + v809;
                        v802 = v810;
                        v805 += 1 ;
                    }
                    v803 += 1 ;
                }
                auto v811 = cooperative_groups::coalesced_threads();
                int v812;
                v812 = threadIdx.x;
                auto v813 = cooperative_groups::labeled_partition(v811,v812);
                float v814;
                v814 = cooperative_groups::reduce(v813, v802, v767);
                bool v815;
                v815 = v814 > 100.0f;
                float v817;
                if (v815){
                    float v816;
                    v816 = 100.0f / v814;
                    v817 = v816;
                } else {
                    v817 = 1.0f;
                }
                float v818[4];
                int v819;
                v819 = 0;
                while (while_method_7(v819)){
                    int v821;
                    v821 = 0;
                    while (while_method_0(v821)){
                        assert("Tensor range check" && 0 <= v819 && v819 < 1);
                        assert("Tensor range check" && 0 <= v821 && v821 < 4);
                        int v823;
                        v823 = 4 * v819;
                        int v824;
                        v824 = v823 + v821;
                        float v825;
                        v825 = v791[v824];
                        float v826;
                        v826 = v817 * v825;
                        assert("Tensor range check" && 0 <= v819 && v819 < 1);
                        assert("Tensor range check" && 0 <= v821 && v821 < 4);
                        v818[v824] = v826;
                        v821 += 1 ;
                    }
                    v819 += 1 ;
                }
                float v827[4];
                float v828[4];
                int v829;
                v829 = 0;
                while (while_method_7(v829)){
                    int v831;
                    v831 = 0;
                    while (while_method_0(v831)){
                        assert("Tensor range check" && 0 <= v829 && v829 < 1);
                        assert("Tensor range check" && 0 <= v831 && v831 < 4);
                        int v833;
                        v833 = 4 * v829;
                        int v834;
                        v834 = v833 + v831;
                        float v835;
                        v835 = v637[v834];
                        float v836;
                        v836 = v638[v834];
                        float v837;
                        v837 = v639[v834];
                        float v838;
                        v838 = v640[v834];
                        float v839;
                        v839 = v641[v834];
                        float v840;
                        v840 = v642[v834];
                        float v841;
                        v841 = v643[v834];
                        float v842;
                        v842 = v838 + v840;
                        float v843;
                        v843 = v839 + v841;
                        assert("Tensor range check" && 0 <= v829 && v829 < 1);
                        assert("Tensor range check" && 0 <= v831 && v831 < 4);
                        v827[v834] = v842;
                        v828[v834] = v843;
                        v831 += 1 ;
                    }
                    v829 += 1 ;
                }
                int v844;
                v844 = 0;
                while (while_method_7(v844)){
                    int v846;
                    v846 = 0;
                    while (while_method_0(v846)){
                        assert("Tensor range check" && 0 <= v844 && v844 < 1);
                        assert("Tensor range check" && 0 <= v846 && v846 < 4);
                        int v848;
                        v848 = 4 * v844;
                        int v849;
                        v849 = v848 + v846;
                        float v850;
                        v850 = v818[v849];
                        float v851;
                        v851 = v733[v849];
                        float v852;
                        v852 = v827[v849];
                        float v853;
                        v853 = v828[v849];
                        assert("Tensor range check" && 0 <= v844 && v844 < 1);
                        assert("Tensor range check" && 0 <= v846 && v846 < 4);
                        v637[v849] = v850;
                        v638[v849] = v851;
                        v639[v849] = 0.0f;
                        v640[v849] = v852;
                        v641[v849] = v853;
                        v642[v849] = 0.0f;
                        v643[v849] = 0.0f;
                        v846 += 1 ;
                    }
                    v844 += 1 ;
                }
            } else {
            }
            assert("Tensor range check" && 0 <= v629 && v629 < 4);
            assert("Tensor range check" && 0 <= v628 && v628 < 16);
            int v854;
            v854 = 0;
            while (while_method_7(v854)){
                assert("Tensor range check" && 0 <= v854 && v854 < 1);
                int v856;
                v856 = 4 * v854;
                int v857;
                v857 = v856 + v636;
                assert("Tensor range check" && 0 <= v854 && v854 < 1);
                int4* v858;
                v858 = reinterpret_cast<int4*>(v637 + v856);
                int4* v859;
                v859 = reinterpret_cast<int4*>(v567 + v857);
                assert("Pointer alignment check" && (unsigned long long)(v858) % 4 == 0 && (unsigned long long)(v859) % 4 == 0);
                *v859 = *v858;
                int4* v860;
                v860 = reinterpret_cast<int4*>(v638 + v856);
                int4* v861;
                v861 = reinterpret_cast<int4*>(v569 + v857);
                assert("Pointer alignment check" && (unsigned long long)(v860) % 4 == 0 && (unsigned long long)(v861) % 4 == 0);
                *v861 = *v860;
                int4* v862;
                v862 = reinterpret_cast<int4*>(v639 + v856);
                int4* v863;
                v863 = reinterpret_cast<int4*>(v571 + v857);
                assert("Pointer alignment check" && (unsigned long long)(v862) % 4 == 0 && (unsigned long long)(v863) % 4 == 0);
                *v863 = *v862;
                int4* v864;
                v864 = reinterpret_cast<int4*>(v640 + v856);
                int4* v865;
                v865 = reinterpret_cast<int4*>(v573 + v857);
                assert("Pointer alignment check" && (unsigned long long)(v864) % 4 == 0 && (unsigned long long)(v865) % 4 == 0);
                *v865 = *v864;
                int4* v866;
                v866 = reinterpret_cast<int4*>(v641 + v856);
                int4* v867;
                v867 = reinterpret_cast<int4*>(v575 + v857);
                assert("Pointer alignment check" && (unsigned long long)(v866) % 4 == 0 && (unsigned long long)(v867) % 4 == 0);
                *v867 = *v866;
                int4* v868;
                v868 = reinterpret_cast<int4*>(v642 + v856);
                int4* v869;
                v869 = reinterpret_cast<int4*>(v577 + v857);
                assert("Pointer alignment check" && (unsigned long long)(v868) % 4 == 0 && (unsigned long long)(v869) % 4 == 0);
                *v869 = *v868;
                int4* v870;
                v870 = reinterpret_cast<int4*>(v643 + v856);
                int4* v871;
                v871 = reinterpret_cast<int4*>(v579 + v857);
                assert("Pointer alignment check" && (unsigned long long)(v870) % 4 == 0 && (unsigned long long)(v871) % 4 == 0);
                *v871 = *v870;
                v854 += 1 ;
            }
            v623 += 24 ;
        }
        v560.sync() ;
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
    v42 = "Going to run the Leduc full kernel."
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
