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
struct Union7;
struct Tuple0;
__device__ unsigned int loop_2(unsigned int v0, curandStatePhilox4_32_10_t & v1);
__device__ Tuple0 draw_card_1(curandStatePhilox4_32_10_t & v0, unsigned int v1);
struct Union8;
struct Union9;
struct Union10;
__device__ void block_matmul_3(float * v0, float * v1, int v2, float * v3);
__device__ void block_map_4(float * v0, int v1, float * v2);
__device__ void block_row_map_5(float * v0, int v1, float * v2);
struct Tuple1;
struct Tuple2;
struct Tuple3;
struct Tuple4;
struct Union11;
struct Tuple5;
__device__ int int_range_6(int v0, int v1, curandStatePhilox4_32_10_t & v2);
struct Union12;
__device__ int tag_8(Union2 v0);
__device__ bool is_pair_9(int v0, int v1);
__device__ Tuple5 order_10(int v0, int v1);
__device__ Union12 compare_hands_7(Union5 v0, bool v1, static_array<Union2,2> v2, int v3, static_array<int,2> v4, int v5);
__device__ void method_0(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut0 & v3, int v4, Union4 v5);
struct Tuple6;
struct Tuple7;
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
struct Union7_0 { // T_game_chance_community_card
    Union5 v0;
    static_array<Union2,2> v2;
    static_array<int,2> v4;
    Union2 v6;
    int v3;
    int v5;
    bool v1;
    __device__ Union7_0(Union5 t0, bool t1, static_array<Union2,2> t2, int t3, static_array<int,2> t4, int t5, Union2 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union7_0() = delete;
};
struct Union7_1 { // T_game_chance_init
    Union2 v0;
    Union2 v1;
    __device__ Union7_1(Union2 t0, Union2 t1) : v0(t0), v1(t1) {}
    __device__ Union7_1() = delete;
};
struct Union7_2 { // T_game_round
    Union5 v0;
    static_array<Union2,2> v2;
    static_array<int,2> v4;
    Union3 v6;
    int v3;
    int v5;
    bool v1;
    __device__ Union7_2(Union5 t0, bool t1, static_array<Union2,2> t2, int t3, static_array<int,2> t4, int t5, Union3 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union7_2() = delete;
};
struct Union7_3 { // T_none
};
struct Union7 {
    union {
        Union7_0 case0; // T_game_chance_community_card
        Union7_1 case1; // T_game_chance_init
        Union7_2 case2; // T_game_round
        Union7_3 case3; // T_none
    };
    unsigned char tag{255};
    __device__ Union7() {}
    __device__ Union7(Union7_0 t) : tag(0), case0(t) {} // T_game_chance_community_card
    __device__ Union7(Union7_1 t) : tag(1), case1(t) {} // T_game_chance_init
    __device__ Union7(Union7_2 t) : tag(2), case2(t) {} // T_game_round
    __device__ Union7(Union7_3 t) : tag(3), case3(t) {} // T_none
    __device__ Union7(Union7 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(x.case0); break; // T_game_chance_community_card
            case 1: new (&this->case1) Union7_1(x.case1); break; // T_game_chance_init
            case 2: new (&this->case2) Union7_2(x.case2); break; // T_game_round
            case 3: new (&this->case3) Union7_3(x.case3); break; // T_none
        }
    }
    __device__ Union7(Union7 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(std::move(x.case0)); break; // T_game_chance_community_card
            case 1: new (&this->case1) Union7_1(std::move(x.case1)); break; // T_game_chance_init
            case 2: new (&this->case2) Union7_2(std::move(x.case2)); break; // T_game_round
            case 3: new (&this->case3) Union7_3(std::move(x.case3)); break; // T_none
        }
    }
    __device__ Union7 & operator=(Union7 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // T_game_chance_community_card
                case 1: this->case1 = x.case1; break; // T_game_chance_init
                case 2: this->case2 = x.case2; break; // T_game_round
                case 3: this->case3 = x.case3; break; // T_none
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
                case 0: this->case0 = std::move(x.case0); break; // T_game_chance_community_card
                case 1: this->case1 = std::move(x.case1); break; // T_game_chance_init
                case 2: this->case2 = std::move(x.case2); break; // T_game_round
                case 3: this->case3 = std::move(x.case3); break; // T_none
            }
        } else {
            this->~Union7();
            new (this) Union7{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union7() {
        switch(this->tag){
            case 0: this->case0.~Union7_0(); break; // T_game_chance_community_card
            case 1: this->case1.~Union7_1(); break; // T_game_chance_init
            case 2: this->case2.~Union7_2(); break; // T_game_round
            case 3: this->case3.~Union7_3(); break; // T_none
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
struct Union8_0 { // C1of2
    Union3 v0;
    __device__ Union8_0(Union3 t0) : v0(t0) {}
    __device__ Union8_0() = delete;
};
struct Union8_1 { // C2of2
    Union2 v0;
    __device__ Union8_1(Union2 t0) : v0(t0) {}
    __device__ Union8_1() = delete;
};
struct Union8 {
    union {
        Union8_0 case0; // C1of2
        Union8_1 case1; // C2of2
    };
    unsigned char tag{255};
    __device__ Union8() {}
    __device__ Union8(Union8_0 t) : tag(0), case0(t) {} // C1of2
    __device__ Union8(Union8_1 t) : tag(1), case1(t) {} // C2of2
    __device__ Union8(Union8 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(x.case0); break; // C1of2
            case 1: new (&this->case1) Union8_1(x.case1); break; // C2of2
        }
    }
    __device__ Union8(Union8 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(std::move(x.case0)); break; // C1of2
            case 1: new (&this->case1) Union8_1(std::move(x.case1)); break; // C2of2
        }
    }
    __device__ Union8 & operator=(Union8 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // C1of2
                case 1: this->case1 = x.case1; break; // C2of2
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
                case 0: this->case0 = std::move(x.case0); break; // C1of2
                case 1: this->case1 = std::move(x.case1); break; // C2of2
            }
        } else {
            this->~Union8();
            new (this) Union8{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union8() {
        switch(this->tag){
            case 0: this->case0.~Union8_0(); break; // C1of2
            case 1: this->case1.~Union8_1(); break; // C2of2
        }
        this->tag = 255;
    }
};
struct Union9_0 { // None
};
struct Union9_1 { // Some
    Union8 v0;
    __device__ Union9_1(Union8 t0) : v0(t0) {}
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
struct Union10_0 { // None
};
struct Union10_1 { // Some
    int v0;
    __device__ Union10_1(int t0) : v0(t0) {}
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
struct Tuple1 {
    int v0;
    float v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Closure2 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple2 {
    float v0;
    bool v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(float t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ Tuple2 operator()(Tuple2 tup0, Tuple2 tup1){
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
                return Tuple2{v5, true};
            } else {
                return Tuple2{v0, v1};
            }
        } else {
            if (v3){
                return Tuple2{v2, v3};
            } else {
                return Tuple2{v0, v1};
            }
        }
    }
};
struct Tuple3 {
    float v0;
    int v1;
    __device__ Tuple3() = default;
    __device__ Tuple3(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
    __device__ Tuple3 operator()(Tuple3 tup0, Tuple3 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple3{v0, v1};
        } else {
            return Tuple3{v2, v3};
        }
    }
};
struct Tuple4 {
    int v0;
    bool v1;
    __device__ Tuple4() = default;
    __device__ Tuple4(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure5 {
    __device__ Tuple4 operator()(Tuple4 tup0, Tuple4 tup1){
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
struct Closure6 {
    int v0;
    __device__ Tuple3 operator()(Tuple3 tup0, Tuple3 tup1){
        int & v0 = this->v0;
        float v1 = tup0.v0; int v2 = tup0.v1; float v3 = tup1.v0; int v4 = tup1.v1;
        bool v5;
        v5 = v2 == v0;
        if (v5){
            return Tuple3{v1, v2};
        } else {
            bool v6;
            v6 = v4 == v0;
            if (v6){
                return Tuple3{v3, v4};
            } else {
                return Tuple3{v1, v2};
            }
        }
    }
    __device__ Closure6(int _v0) : v0(_v0) { }
};
struct Union11_0 { // AA_Call
};
struct Union11_1 { // AA_Fold
};
struct Union11_2 { // AA_Raise
};
struct Union11 {
    union {
        Union11_0 case0; // AA_Call
        Union11_1 case1; // AA_Fold
        Union11_2 case2; // AA_Raise
    };
    unsigned char tag{255};
    __device__ Union11() {}
    __device__ Union11(Union11_0 t) : tag(0), case0(t) {} // AA_Call
    __device__ Union11(Union11_1 t) : tag(1), case1(t) {} // AA_Fold
    __device__ Union11(Union11_2 t) : tag(2), case2(t) {} // AA_Raise
    __device__ Union11(Union11 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union11_0(x.case0); break; // AA_Call
            case 1: new (&this->case1) Union11_1(x.case1); break; // AA_Fold
            case 2: new (&this->case2) Union11_2(x.case2); break; // AA_Raise
        }
    }
    __device__ Union11(Union11 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union11_0(std::move(x.case0)); break; // AA_Call
            case 1: new (&this->case1) Union11_1(std::move(x.case1)); break; // AA_Fold
            case 2: new (&this->case2) Union11_2(std::move(x.case2)); break; // AA_Raise
        }
    }
    __device__ Union11 & operator=(Union11 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // AA_Call
                case 1: this->case1 = x.case1; break; // AA_Fold
                case 2: this->case2 = x.case2; break; // AA_Raise
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
                case 0: this->case0 = std::move(x.case0); break; // AA_Call
                case 1: this->case1 = std::move(x.case1); break; // AA_Fold
                case 2: this->case2 = std::move(x.case2); break; // AA_Raise
            }
        } else {
            this->~Union11();
            new (this) Union11{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union11() {
        switch(this->tag){
            case 0: this->case0.~Union11_0(); break; // AA_Call
            case 1: this->case1.~Union11_1(); break; // AA_Fold
            case 2: this->case2.~Union11_2(); break; // AA_Raise
        }
        this->tag = 255;
    }
};
struct Tuple5 {
    int v0;
    int v1;
    __device__ Tuple5() = default;
    __device__ Tuple5(int t0, int t1) : v0(t0), v1(t1) {}
};
struct Union12_0 { // Eq
};
struct Union12_1 { // Gt
};
struct Union12_2 { // Lt
};
struct Union12 {
    union {
        Union12_0 case0; // Eq
        Union12_1 case1; // Gt
        Union12_2 case2; // Lt
    };
    unsigned char tag{255};
    __device__ Union12() {}
    __device__ Union12(Union12_0 t) : tag(0), case0(t) {} // Eq
    __device__ Union12(Union12_1 t) : tag(1), case1(t) {} // Gt
    __device__ Union12(Union12_2 t) : tag(2), case2(t) {} // Lt
    __device__ Union12(Union12 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union12_0(x.case0); break; // Eq
            case 1: new (&this->case1) Union12_1(x.case1); break; // Gt
            case 2: new (&this->case2) Union12_2(x.case2); break; // Lt
        }
    }
    __device__ Union12(Union12 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union12_0(std::move(x.case0)); break; // Eq
            case 1: new (&this->case1) Union12_1(std::move(x.case1)); break; // Gt
            case 2: new (&this->case2) Union12_2(std::move(x.case2)); break; // Lt
        }
    }
    __device__ Union12 & operator=(Union12 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Eq
                case 1: this->case1 = x.case1; break; // Gt
                case 2: this->case2 = x.case2; break; // Lt
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
                case 0: this->case0 = std::move(x.case0); break; // Eq
                case 1: this->case1 = std::move(x.case1); break; // Gt
                case 2: this->case2 = std::move(x.case2); break; // Lt
            }
        } else {
            this->~Union12();
            new (this) Union12{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union12() {
        switch(this->tag){
            case 0: this->case0.~Union12_0(); break; // Eq
            case 1: this->case1.~Union12_1(); break; // Gt
            case 2: this->case2.~Union12_2(); break; // Lt
        }
        this->tag = 255;
    }
};
struct Tuple6 {
    double v1;
    int v0;
    __device__ Tuple6() = default;
    __device__ Tuple6(int t0, double t1) : v0(t0), v1(t1) {}
};
struct Tuple7 {
    double v2;
    float v1;
    int v0;
    __device__ Tuple7() = default;
    __device__ Tuple7(int t0, float t1, double t2) : v0(t0), v1(t1), v2(t2) {}
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 32;
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
__device__ void block_matmul_3(float * v0, float * v1, int v2, float * v3){
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
            while (while_method_1(v74)){
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
                Union10 v97;
                if (v91){
                    bool v92;
                    v92 = 0 <= v82;
                    bool v93;
                    v93 = v92 == false;
                    if (v93){
                        assert("The index needs to be zero or positive." && v92);
                    } else {
                    }
                    v97 = Union10{Union10_1{v82}};
                } else {
                    v97 = Union10{Union10_0{}};
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
                    while (while_method_2(v127)){
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
                while (while_method_1(v155)){
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
                while (while_method_6(v168)){
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
                        while (while_method_2(v178)){
                            int v180;
                            v180 = 0;
                            #pragma unroll
                            while (while_method_2(v180)){
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
                        while (while_method_2(v222)){
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
                while (while_method_1(v232)){
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
                        while (while_method_2(v240)){
                            int v242;
                            v242 = 0;
                            #pragma unroll
                            while (while_method_2(v242)){
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
            while (while_method_1(v268)){
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
            while (while_method_9(v297)){
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
__device__ inline bool while_method_10(int v0){
    bool v1;
    v1 = v0 < 4096;
    return v1;
}
__device__ void block_map_4(float * v0, int v1, float * v2){
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
    while (while_method_10(v9)){
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
        while (while_method_8(v29)){
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
__device__ void block_row_map_5(float * v0, int v1, float * v2){
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
    while (while_method_9(v24)){
        assert("Tensor range check" && 0 <= v24 && v24 < 16);
        int v26;
        v26 = 1024 * v24;
        int v27;
        v27 = v26 + v21;
        float v28[4];
        int v29[4];
        int v30;
        v30 = 0;
        while (while_method_6(v30)){
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
        while (while_method_6(v37)){
            int v39;
            v39 = 0;
            while (while_method_8(v39)){
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
        while (while_method_6(v74)){
            int v76;
            v76 = 0;
            while (while_method_8(v76)){
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
        while (while_method_6(v84)){
            int v86;
            v86 = 0;
            while (while_method_8(v86)){
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
        while (while_method_6(v94)){
            int v96;
            v96 = 0;
            while (while_method_8(v96)){
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
        while (while_method_6(v109)){
            int v111;
            v111 = 0;
            while (while_method_8(v111)){
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
        while (while_method_6(v118)){
            int v120;
            v120 = 0;
            while (while_method_8(v120)){
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
        while (while_method_6(v135)){
            int v137;
            v137 = 0;
            while (while_method_8(v137)){
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
                bool v146;
                v146 = v145 < 1.0f / 0.0f;
                bool v147;
                v147 = v146 == false;
                if (v147){
                    assert("The softmax values must not grow too large." && v146);
                } else {
                }
                assert("Tensor range check" && 0 <= v135 && v135 < 1);
                assert("Tensor range check" && 0 <= v137 && v137 < 4);
                v134[v140] = v145;
                v137 += 1 ;
            }
            v135 += 1 ;
        }
        float v149;
        v149 = 0.0f;
        int v150;
        v150 = 0;
        while (while_method_6(v150)){
            int v152;
            v152 = 0;
            while (while_method_8(v152)){
                assert("Tensor range check" && 0 <= v150 && v150 < 1);
                assert("Tensor range check" && 0 <= v152 && v152 < 4);
                int v154;
                v154 = 4 * v150;
                int v155;
                v155 = v154 + v152;
                float v156;
                v156 = v134[v155];
                float v157;
                v157 = v149 + v156;
                v149 = v157;
                v152 += 1 ;
            }
            v150 += 1 ;
        }
        auto v158 = cooperative_groups::coalesced_threads();
        int v159;
        v159 = threadIdx.x;
        int v160;
        v160 = v159 / 16;
        auto v161 = cooperative_groups::labeled_partition(v158,v160);
        float v162;
        v162 = cooperative_groups::reduce(v161, v149, v106);
        float v163[4];
        int v164;
        v164 = 0;
        while (while_method_6(v164)){
            int v166;
            v166 = 0;
            while (while_method_8(v166)){
                assert("Tensor range check" && 0 <= v164 && v164 < 1);
                assert("Tensor range check" && 0 <= v166 && v166 < 4);
                int v168;
                v168 = 4 * v164;
                int v169;
                v169 = v168 + v166;
                float v170;
                v170 = v134[v169];
                float v171;
                v171 = v170 / v162;
                assert("Tensor range check" && 0 <= v164 && v164 < 1);
                assert("Tensor range check" && 0 <= v166 && v166 < 4);
                v163[v169] = v171;
                v166 += 1 ;
            }
            v164 += 1 ;
        }
        assert("Tensor range check" && 0 <= v24 && v24 < 16);
        int v172;
        v172 = v26 + v23;
        int v173;
        v173 = 0;
        while (while_method_6(v173)){
            assert("Tensor range check" && 0 <= v173 && v173 < 1);
            int v175;
            v175 = 64 * v173;
            int v176;
            v176 = v175 + v172;
            assert("Tensor range check" && 0 <= v173 && v173 < 1);
            int v177;
            v177 = 4 * v173;
            int4* v178;
            v178 = reinterpret_cast<int4*>(v163 + v177);
            int4* v179;
            v179 = reinterpret_cast<int4*>(v0 + v176);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v178) % 16 == 0 && reinterpret_cast<unsigned long long>(v179) % 16 == 0);
            *v179 = *v178;
            v173 += 1 ;
        }
        v24 += 1 ;
    }
    __syncthreads();
    return ;
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
__device__ Tuple5 order_10(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple5{v1, v0};
    } else {
        return Tuple5{v0, v1};
    }
}
__device__ Union12 compare_hands_7(Union5 v0, bool v1, static_array<Union2,2> v2, int v3, static_array<int,2> v4, int v5){
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
                        return Union12{Union12_2{}};
                    } else {
                        bool v19;
                        v19 = v11 > v14;
                        if (v19){
                            return Union12{Union12_1{}};
                        } else {
                            return Union12{Union12_0{}};
                        }
                    }
                } else {
                    return Union12{Union12_1{}};
                }
            } else {
                if (v16){
                    return Union12{Union12_2{}};
                } else {
                    int v27; int v28;
                    Tuple5 tmp14 = order_10(v8, v11);
                    v27 = tmp14.v0; v28 = tmp14.v1;
                    int v29; int v30;
                    Tuple5 tmp15 = order_10(v8, v14);
                    v29 = tmp15.v0; v30 = tmp15.v1;
                    bool v31;
                    v31 = v27 < v29;
                    Union12 v37;
                    if (v31){
                        v37 = Union12{Union12_2{}};
                    } else {
                        bool v33;
                        v33 = v27 > v29;
                        if (v33){
                            v37 = Union12{Union12_1{}};
                        } else {
                            v37 = Union12{Union12_0{}};
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
                            return Union12{Union12_2{}};
                        } else {
                            bool v41;
                            v41 = v28 > v30;
                            if (v41){
                                return Union12{Union12_1{}};
                            } else {
                                return Union12{Union12_0{}};
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
__device__ void method_0(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut0 & v3, int v4, Union4 v5){
    v3.v0 = 63u;
    static_array<float,2> v6;
    v6[0] = 0.0f;
    v6[1] = 0.0f;
    v3.v4 = v6;
    static_array_list<Union1,32> & v8 = v3.v2;
    v8.unsafe_set_length(0);
    static_array<Union0,2> v9;
    Union0 v11;
    v11 = Union0{Union0_0{}};
    v9[0] = v11;
    Union0 v13;
    v13 = Union0{Union0_0{}};
    v9[1] = v13;
    int v15;
    v15 = v4 ^ 1;
    Union0 v16;
    v16 = Union0{Union0_2{}};
    v9[v15] = v16;
    v3.v3 = v9;
    static_array_list<Union1,32> & v18 = v3.v2;
    Union6 v19;
    v19 = Union6{Union6_1{v5}};
    Union6 v20;
    v20 = v19;
    while (while_method_3(v20)){
        Union6 v792;
        switch (v20.tag) {
            case 0: { // None
                v792 = Union6{Union6_0{}};
                break;
            }
            case 1: { // Some
                Union4 v22 = v20.case1.v0;
                Union7 v632;
                switch (v22.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v601 = v22.case0.v0; bool v602 = v22.case0.v1; static_array<Union2,2> v603 = v22.case0.v2; int v604 = v22.case0.v3; static_array<int,2> v605 = v22.case0.v4; int v606 = v22.case0.v5;
                        curandStatePhilox4_32_10_t & v607 = v3.v5;
                        curandStatePhilox4_32_10_t & v608 = v607;
                        unsigned int & v609 = v3.v0;
                        Union2 v610; unsigned int v611;
                        Tuple0 tmp0 = draw_card_1(v608, v609);
                        v610 = tmp0.v0; v611 = tmp0.v1;
                        v3.v0 = v611;
                        Union1 v612;
                        v612 = Union1{Union1_0{v610}};
                        v18.push(v612);
                        v632 = Union7{Union7_0{v601, v602, v603, v604, v605, v606, v610}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v614 = v3.v5;
                        curandStatePhilox4_32_10_t & v615 = v614;
                        unsigned int & v616 = v3.v0;
                        Union2 v617; unsigned int v618;
                        Tuple0 tmp1 = draw_card_1(v615, v616);
                        v617 = tmp1.v0; v618 = tmp1.v1;
                        v3.v0 = v618;
                        curandStatePhilox4_32_10_t & v619 = v3.v5;
                        curandStatePhilox4_32_10_t & v620 = v619;
                        unsigned int & v621 = v3.v0;
                        Union2 v622; unsigned int v623;
                        Tuple0 tmp2 = draw_card_1(v620, v621);
                        v622 = tmp2.v0; v623 = tmp2.v1;
                        v3.v0 = v623;
                        Union1 v624;
                        v624 = Union1{Union1_2{0, v617}};
                        v18.push(v624);
                        Union1 v625;
                        v625 = Union1{Union1_2{1, v622}};
                        v18.push(v625);
                        v632 = Union7{Union7_1{v617, v622}};
                        break;
                    }
                    case 2: { // Round
                        Union5 v72 = v22.case2.v0; bool v73 = v22.case2.v1; static_array<Union2,2> v74 = v22.case2.v2; int v75 = v22.case2.v3; static_array<int,2> v76 = v22.case2.v4; int v77 = v22.case2.v5;
                        static_array<Union0,2> & v78 = v3.v3;
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
                        Union0 v84;
                        v84 = v78[v75];
                        Union3 v589;
                        switch (v84.tag) {
                            case 0: { // Computer
                                static_array_list<Union1,32> & v87 = v3.v2;
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
                                while (while_method_4(v99)){
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
                                static_array_list<Union8,10> v115;
                                v115 = static_array_list<Union8,10>{};
                                int v117;
                                v117 = v87.length;
                                int v118;
                                v118 = 0;
                                while (while_method_5(v117, v118)){
                                    Union1 v120;
                                    v120 = v87[v118];
                                    Union9 v139;
                                    switch (v120.tag) {
                                        case 0: { // CommunityCardIs
                                            Union2 v129 = v120.case0.v0;
                                            Union8 v130;
                                            v130 = Union8{Union8_1{v129}};
                                            v139 = Union9{Union9_1{v130}};
                                            break;
                                        }
                                        case 1: { // PlayerAction
                                            int v132 = v120.case1.v0; Union3 v133 = v120.case1.v1;
                                            Union8 v134;
                                            v134 = Union8{Union8_0{v133}};
                                            v139 = Union9{Union9_1{v134}};
                                            break;
                                        }
                                        case 2: { // PlayerGotCard
                                            int v122 = v120.case2.v0; Union2 v123 = v120.case2.v1;
                                            bool v124;
                                            v124 = v122 == v75;
                                            if (v124){
                                                Union8 v125;
                                                v125 = Union8{Union8_1{v123}};
                                                v139 = Union9{Union9_1{v125}};
                                            } else {
                                                v139 = Union9{Union9_0{}};
                                            }
                                            break;
                                        }
                                        default: {
                                            v139 = Union9{Union9_0{}};
                                        }
                                    }
                                    switch (v139.tag) {
                                        case 0: { // None
                                            break;
                                        }
                                        case 1: { // Some
                                            Union8 v140 = v139.case1.v0;
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
                                while (while_method_5(v145, v146)){
                                    Union8 v148;
                                    v148 = v115[v146];
                                    int v150;
                                    v150 = v146 * 6;
                                    int v151;
                                    v151 = 1 + v150;
                                    switch (v148.tag) {
                                        case 0: { // C1of2
                                            Union3 v152 = v148.case0.v0;
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
                                            Union2 v155 = v148.case1.v0;
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
                                while (while_method_0(v159)){
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
                                    block_matmul_3(v173, v168, v172, v166);
                                    block_map_4(v164, v163, v173);
                                    block_row_map_5(v161, v163, v164);
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
                                while (while_method_9(v238)){
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
                                    while (while_method_6(v260)){
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
                                    while (while_method_6(v267)){
                                        int v269;
                                        v269 = 0;
                                        while (while_method_8(v269)){
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
                                    while (while_method_6(v294)){
                                        assert("Tensor range check" && 0 <= v294 && v294 < 1);
                                        int v296;
                                        v296 = 4 * v294;
                                        assert("Tensor range check" && 0 <= v294 && v294 < 1);
                                        int v297; float v298;
                                        Tuple1 tmp3 = Tuple1{0, 0.0f};
                                        v297 = tmp3.v0; v298 = tmp3.v1;
                                        while (while_method_8(v297)){
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
                                        Tuple1 tmp4 = Tuple1{0, v313};
                                        v314 = tmp4.v0; v315 = tmp4.v1;
                                        while (while_method_8(v314)){
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
                                    while (while_method_6(v323)){
                                        int v325;
                                        v325 = 0;
                                        while (while_method_8(v325)){
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
                                    Tuple2 tmp5 = Tuple2{-1.0f / 0.0f, false};
                                    v332 = tmp5.v0; v333 = tmp5.v1;
                                    int v334;
                                    v334 = 0;
                                    while (while_method_6(v334)){
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
                                    Tuple2 tmp6 = cooperative_groups::reduce(v353, Tuple2{v332, v333}, v354);
                                    v355 = tmp6.v0; v356 = tmp6.v1;
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
                                    while (while_method_6(v361)){
                                        int v363;
                                        v363 = 0;
                                        while (while_method_8(v363)){
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
                                    Tuple3 tmp7 = Tuple3{0.0f, 2147483647};
                                    v369 = tmp7.v0; v370 = tmp7.v1;
                                    int v371;
                                    v371 = 0;
                                    while (while_method_6(v371)){
                                        int v373;
                                        v373 = 0;
                                        while (while_method_8(v373)){
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
                                    Tuple3 tmp8 = cooperative_groups::reduce(v385, Tuple3{v369, v370}, v386);
                                    v387 = tmp8.v0; v388 = tmp8.v1;
                                    float v389;
                                    v389 = v355 * v387;
                                    int v390[4];
                                    bool v391[4];
                                    int v392;
                                    v392 = 0;
                                    while (while_method_6(v392)){
                                        int v394;
                                        v394 = 0;
                                        while (while_method_8(v394)){
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
                                    Tuple4 tmp9 = Tuple4{2147483647, false};
                                    v405 = tmp9.v0; v406 = tmp9.v1;
                                    int v407;
                                    v407 = 0;
                                    while (while_method_6(v407)){
                                        int v409;
                                        v409 = 0;
                                        while (while_method_8(v409)){
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
                                    Tuple4 tmp10 = cooperative_groups::reduce(v426, Tuple4{v405, v406}, v427);
                                    v428 = tmp10.v0; v429 = tmp10.v1;
                                    bool v430;
                                    v430 = v429 == false;
                                    if (v430){
                                        assert("The local reduce must be true." && v429);
                                    } else {
                                    }
                                    float v432; int v433;
                                    Tuple3 tmp11 = Tuple3{0.0f, 2147483647};
                                    v432 = tmp11.v0; v433 = tmp11.v1;
                                    int v434;
                                    v434 = 0;
                                    while (while_method_6(v434)){
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
                                    Tuple3 tmp12 = cooperative_groups::reduce(v451, Tuple3{v432, v433}, v452);
                                    v453 = tmp12.v0; v454 = tmp12.v1;
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
                                    while (while_method_6(v459)){
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
                                while (while_method_0(v480)){
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
                                Union11 v514;
                                if (v505){
                                    v514 = Union11{Union11_1{}};
                                } else {
                                    bool v507;
                                    v507 = 1 == v471;
                                    if (v507){
                                        v514 = Union11{Union11_0{}};
                                    } else {
                                        bool v509;
                                        v509 = 2 == v471;
                                        if (v509){
                                            v514 = Union11{Union11_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v514.tag) {
                                    case 0: { // AA_Call
                                        v589 = Union3{Union3_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v515;
                                        v515 = v76[0];
                                        int v517; int v518;
                                        Tuple5 tmp13 = Tuple5{1, v515};
                                        v517 = tmp13.v0; v518 = tmp13.v1;
                                        while (while_method_2(v517)){
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
                                            v589 = Union3{Union3_0{}};
                                        } else {
                                            v589 = Union3{Union3_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v540;
                                        v540 = v77 > 0;
                                        if (v540){
                                            v589 = Union3{Union3_2{}};
                                        } else {
                                            v589 = Union3{Union3_0{}};
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
                                static_array_list<Union3,3> v549;
                                v549 = static_array_list<Union3,3>{};
                                v549.unsafe_set_length(1);
                                Union3 v551;
                                v551 = Union3{Union3_0{}};
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
                                    Union3 v559;
                                    v559 = Union3{Union3_1{}};
                                    v549.push(v559);
                                } else {
                                }
                                bool v560;
                                v560 = v77 > 0;
                                if (v560){
                                    Union3 v561;
                                    v561 = Union3{Union3_2{}};
                                    v549.push(v561);
                                } else {
                                }
                                int v562;
                                v562 = v549.length;
                                int v563;
                                v563 = v562 - 1;
                                int v564;
                                v564 = 0;
                                while (while_method_5(v563, v564)){
                                    int v566;
                                    v566 = v549.length;
                                    int v567;
                                    v567 = int_range_6(v566, v564, v548);
                                    Union3 v568;
                                    v568 = v549[v564];
                                    Union3 v570;
                                    v570 = v549[v567];
                                    v549[v564] = v570;
                                    v549[v567] = v568;
                                    v564 += 1 ;
                                }
                                Union3 v572;
                                v572 = v549.pop();
                                int v573;
                                v573 = sizeof(Union3);
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
                                Union3 * v582;
                                v582 = reinterpret_cast<Union3 *>(&v578[0ull]);
                                int v584;
                                v584 = threadIdx.x;
                                bool v585;
                                v585 = v584 == 0;
                                if (v585){
                                    v582[0] = v572;
                                } else {
                                }
                                __syncthreads();
                                Union3 v586;
                                v586 = v582[0];
                                __syncthreads();
                                v589 = v586;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union1 v590;
                        v590 = Union1{Union1_1{v75, v589}};
                        v18.push(v590);
                        v632 = Union7{Union7_2{v72, v73, v74, v75, v76, v77, v589}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v592 = v22.case3.v0; bool v593 = v22.case3.v1; static_array<Union2,2> v594 = v22.case3.v2; int v595 = v22.case3.v3; static_array<int,2> v596 = v22.case3.v4; int v597 = v22.case3.v5; Union3 v598 = v22.case3.v6;
                        Union1 v599;
                        v599 = Union1{Union1_1{v595, v598}};
                        v18.push(v599);
                        v632 = Union7{Union7_2{v592, v593, v594, v595, v596, v597, v598}};
                        break;
                    }
                    case 4: { // TerminalCall
                        Union5 v43 = v22.case4.v0; bool v44 = v22.case4.v1; static_array<Union2,2> v45 = v22.case4.v2; int v46 = v22.case4.v3; static_array<int,2> v47 = v22.case4.v4; int v48 = v22.case4.v5;
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
                        Union12 v56;
                        v56 = compare_hands_7(v43, v44, v45, v46, v47, v48);
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
                        Union1 v70;
                        v70 = Union1{Union1_3{v45, v61, v62}};
                        v18.push(v70);
                        v632 = Union7{Union7_3{}};
                        break;
                    }
                    case 5: { // TerminalFold
                        Union5 v23 = v22.case5.v0; bool v24 = v22.case5.v1; static_array<Union2,2> v25 = v22.case5.v2; int v26 = v22.case5.v3; static_array<int,2> v27 = v22.case5.v4; int v28 = v22.case5.v5;
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
                        Union1 v41;
                        v41 = Union1{Union1_3{v25, v34, v39}};
                        v18.push(v41);
                        v632 = Union7{Union7_3{}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                switch (v632.tag) {
                    case 0: { // T_game_chance_community_card
                        Union5 v634 = v632.case0.v0; bool v635 = v632.case0.v1; static_array<Union2,2> v636 = v632.case0.v2; int v637 = v632.case0.v3; static_array<int,2> v638 = v632.case0.v4; int v639 = v632.case0.v5; Union2 v640 = v632.case0.v6;
                        int v641;
                        v641 = 2;
                        int v642; int v643;
                        Tuple5 tmp16 = Tuple5{0, 0};
                        v642 = tmp16.v0; v643 = tmp16.v1;
                        while (while_method_2(v642)){
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
                        while (while_method_2(v656)){
                            v654[v656] = v643;
                            v656 += 1 ;
                        }
                        Union5 v658;
                        v658 = Union5{Union5_1{v640}};
                        Union4 v659;
                        v659 = Union4{Union4_2{v658, true, v636, 0, v654, v641}};
                        v792 = Union6{Union6_1{v659}};
                        break;
                    }
                    case 1: { // T_game_chance_init
                        Union2 v661 = v632.case1.v0; Union2 v662 = v632.case1.v1;
                        int v663;
                        v663 = 2;
                        static_array<int,2> v664;
                        v664[0] = 1;
                        v664[1] = 1;
                        static_array<Union2,2> v666;
                        v666[0] = v661;
                        v666[1] = v662;
                        Union5 v668;
                        v668 = Union5{Union5_0{}};
                        Union4 v669;
                        v669 = Union4{Union4_2{v668, true, v666, 0, v664, v663}};
                        v792 = Union6{Union6_1{v669}};
                        break;
                    }
                    case 2: { // T_game_round
                        Union5 v671 = v632.case2.v0; bool v672 = v632.case2.v1; static_array<Union2,2> v673 = v632.case2.v2; int v674 = v632.case2.v3; static_array<int,2> v675 = v632.case2.v4; int v676 = v632.case2.v5; Union3 v677 = v632.case2.v6;
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
                                            Tuple5 tmp17 = Tuple5{0, 0};
                                            v747 = tmp17.v0; v748 = tmp17.v1;
                                            while (while_method_2(v747)){
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
                                            while (while_method_2(v761)){
                                                v759[v761] = v748;
                                                v761 += 1 ;
                                            }
                                            static_array<int,2> v763;
                                            int v765;
                                            v765 = 0;
                                            while (while_method_2(v765)){
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
                                Union2 v678 = v671.case1.v0;
                                switch (v677.tag) {
                                    case 0: { // Call
                                        if (v672){
                                            int v680;
                                            v680 = v674 ^ 1;
                                            v784 = Union4{Union4_2{v671, false, v673, v680, v675, v676}};
                                        } else {
                                            int v682; int v683;
                                            Tuple5 tmp18 = Tuple5{0, 0};
                                            v682 = tmp18.v0; v683 = tmp18.v1;
                                            while (while_method_2(v682)){
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
                                            while (while_method_2(v696)){
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
                                            Tuple5 tmp19 = Tuple5{0, 0};
                                            v703 = tmp19.v0; v704 = tmp19.v1;
                                            while (while_method_2(v703)){
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
                                            while (while_method_2(v717)){
                                                v715[v717] = v704;
                                                v717 += 1 ;
                                            }
                                            static_array<int,2> v719;
                                            int v721;
                                            v721 = 0;
                                            while (while_method_2(v721)){
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
                        v792 = Union6{Union6_1{v784}};
                        break;
                    }
                    case 3: { // T_none
                        v792 = Union6{Union6_0{}};
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
__device__ inline bool while_method_11(int v0){
    bool v1;
    v1 = v0 < 16384;
    return v1;
}
__device__ inline bool while_method_12(int v0){
    bool v1;
    v1 = v0 < 256;
    return v1;
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1, unsigned char * v2, float * v3, float * v4, float * v5) {
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
    static_array<Union0,2> v14;
    Union0 v16;
    v16 = Union0{Union0_2{}};
    v14[0] = v16;
    Union0 v18;
    v18 = Union0{Union0_2{}};
    v14[1] = v18;
    static_array_list<Union1,32> v20;
    v20 = static_array_list<Union1,32>{};
    static_array<float,2> v22;
    v22[0] = 0.0f;
    v22[1] = 0.0f;
    cooperative_groups::grid_group & v24 = v6;
    curandStatePhilox4_32_10_t & v25 = v13;
    StackMut0 v26{63u, v24, v20, v14, v22, v25};
    int v27;
    v27 = 0;
    while (while_method_0(v27)){
        int v29;
        v29 = 0;
        while (while_method_1(v29)){
            int v31;
            v31 = 0;
            while (while_method_2(v31)){
                Union4 v33;
                v33 = Union4{Union4_1{}};
                method_0(v0, v1, v2, v26, v31, v33);
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
                while (while_method_2(v52)){
                    int v54; double v55;
                    Tuple6 tmp20 = Tuple6{0, 0.0};
                    v54 = tmp20.v0; v55 = tmp20.v1;
                    while (while_method_0(v54)){
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
                while (while_method_2(v66)){
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
                while (while_method_0(v71)){
                    int v73;
                    v73 = 0;
                    while (while_method_2(v73)){
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
                while (while_method_0(v87)){
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
                while (while_method_2(v125)){
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
                while (while_method_2(v135)){
                    int v137; double v138;
                    Tuple6 tmp21 = Tuple6{0, 0.0};
                    v137 = tmp21.v0; v138 = tmp21.v1;
                    while (while_method_0(v137)){
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
                while (while_method_2(v148)){
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
                while (while_method_0(v153)){
                    int v155;
                    v155 = 0;
                    while (while_method_2(v155)){
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
                while (while_method_0(v170)){
                    int v172; float v173; double v174;
                    Tuple7 tmp22 = Tuple7{0, 0.0f, 0.0};
                    v172 = tmp22.v0; v173 = tmp22.v1; v174 = tmp22.v2;
                    while (while_method_2(v172)){
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
                while (while_method_0(v185)){
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
                while (while_method_0(v200)){
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
                    while (while_method_6(v254)){
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
                        while (while_method_6(v279)){
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
                        while (while_method_6(v287)){
                            int v289;
                            v289 = 0;
                            while (while_method_2(v289)){
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
                        while (while_method_6(v314)){
                            int v316;
                            v316 = 0;
                            while (while_method_2(v316)){
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
                        while (while_method_6(v322)){
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
            v357 = int_range_6(v356, v355, v333);
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
        while (while_method_9(v362)){
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
            while (while_method_11(v379)){
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
            while (while_method_11(v396)){
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
        while (while_method_12(v429)){
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
            while (while_method_6(v445)){
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
            while (while_method_6(v452)){
                int v454;
                v454 = 0;
                while (while_method_8(v454)){
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
            while (while_method_6(v499)){
                int v501;
                v501 = 0;
                while (while_method_8(v501)){
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
            while (while_method_6(v513)){
                int v515;
                v515 = 0;
                while (while_method_8(v515)){
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
            while (while_method_6(v529)){
                int v531;
                v531 = 0;
                while (while_method_8(v531)){
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
            while (while_method_6(v538)){
                int v540;
                v540 = 0;
                while (while_method_8(v540)){
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
            while (while_method_6(v547)){
                int v549;
                v549 = 0;
                while (while_method_8(v549)){
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
            while (while_method_6(v563)){
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
            while (while_method_6(v573)){
                int v575;
                v575 = 0;
                while (while_method_8(v575)){
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
            while (while_method_6(v588)){
                int v590;
                v590 = 0;
                while (while_method_8(v590)){
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
            while (while_method_6(v598)){
                int v600;
                v600 = 0;
                while (while_method_8(v600)){
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
            while (while_method_6(v608)){
                int v610;
                v610 = 0;
                while (while_method_8(v610)){
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
            while (while_method_6(v629)){
                int v631;
                v631 = 0;
                while (while_method_8(v631)){
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
            while (while_method_6(v642)){
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
        while (while_method_9(v653)){
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
            while (while_method_11(v669)){
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
            while (while_method_11(v686)){
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
        while (while_method_1(v706)){
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
            while (while_method_8(v723)){
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
    while (while_method_12(v737)){
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
        while (while_method_8(v757)){
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
        del v11
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
                v26 = 0
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
                while method54(v32):
                    v34 = []
                    v35 = 0
                    while method54(v35):
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
            case t:
                raise Exception("WIP")
        del v2
        cp.cuda.get_current_stream().synchronize()
        v41 = time.perf_counter()
        v44 = "{}"
        v45 = "The time it took to run the kernel (in seconds) is: "
        print(v44.format(v45),end="")
        del v44, v45
        v46 = v41 - v17
        del v17, v41
        v49 = "{:.6f}\n"
        print(v49.format(v46),end="")
        del v46, v49
        v50, v51, v52, v53, v54 = method55(v12)
        del v12
        return method72(v50, v51, v52, v53, v54, v8, v9, v10, v18)
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
        return method111(v32, v33, v31, v25, v34, v2, v1, v0)
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
def method54(v0 : i32) -> bool:
    v1 = v0 < 32
    del v0
    return v1
def method56(v0 : cp.ndarray) -> u32:
    v2 = v0[0:].view(cp.uint32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method57(v0 : cp.ndarray) -> i32:
    v2 = v0[4:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method58(v0 : cp.ndarray) -> None:
    del v0
    return 
def method60(v0 : cp.ndarray) -> i32:
    v2 = v0[0:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method62(v0 : cp.ndarray) -> US6:
    v1 = method60(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method58(v3)
        del v3
        return US6_0()
    elif v1 == 1:
        del v1
        method58(v3)
        del v3
        return US6_1()
    elif v1 == 2:
        del v1
        method58(v3)
        del v3
        return US6_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method61(v0 : cp.ndarray) -> Tuple[US5, bool, static_array, i32, static_array, i32]:
    v1 = method60(v0)
    v3 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method58(v3)
        v8 = US5_0()
    elif v1 == 1:
        v6 = method62(v3)
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
        v21 = method62(v20)
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
        v34 = method60(v33)
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
def method64(v0 : cp.ndarray) -> i32:
    v2 = v0[36:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method63(v0 : cp.ndarray) -> Tuple[US5, bool, static_array, i32, static_array, i32, US1]:
    v1 = method60(v0)
    v3 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method58(v3)
        v8 = US5_0()
    elif v1 == 1:
        v6 = method62(v3)
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
        v21 = method62(v20)
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
        v34 = method60(v33)
        del v33
        v26[v27] = v34
        del v34
        v27 += 1 
    del v27
    v36 = v0[32:].view(cp.int32)
    v37 = v36[0].item()
    del v36
    v38 = method64(v0)
    v40 = v0[40:].view(cp.uint8)
    del v0
    if v38 == 0:
        method58(v40)
        v45 = US1_0()
    elif v38 == 1:
        method58(v40)
        v45 = US1_1()
    elif v38 == 2:
        method58(v40)
        v45 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v38, v40
    return v8, v11, v13, v24, v26, v37, v45
def method59(v0 : cp.ndarray) -> US4:
    v1 = method60(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5, v6, v7, v8, v9, v10 = method61(v3)
        del v3
        return US4_0(v5, v6, v7, v8, v9, v10)
    elif v1 == 1:
        del v1
        method58(v3)
        del v3
        return US4_1()
    elif v1 == 2:
        del v1
        v13, v14, v15, v16, v17, v18 = method61(v3)
        del v3
        return US4_2(v13, v14, v15, v16, v17, v18)
    elif v1 == 3:
        del v1
        v20, v21, v22, v23, v24, v25, v26 = method63(v3)
        del v3
        return US4_3(v20, v21, v22, v23, v24, v25, v26)
    elif v1 == 4:
        del v1
        v28, v29, v30, v31, v32, v33 = method61(v3)
        del v3
        return US4_4(v28, v29, v30, v31, v32, v33)
    elif v1 == 5:
        del v1
        v35, v36, v37, v38, v39, v40 = method61(v3)
        del v3
        return US4_5(v35, v36, v37, v38, v39, v40)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method65(v0 : cp.ndarray) -> i32:
    v2 = v0[80:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method67(v0 : cp.ndarray) -> Tuple[i32, US1]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method57(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method58(v6)
        v11 = US1_0()
    elif v4 == 1:
        method58(v6)
        v11 = US1_1()
    elif v4 == 2:
        method58(v6)
        v11 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v11
def method68(v0 : cp.ndarray) -> Tuple[i32, US6]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method57(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method58(v6)
        v11 = US6_0()
    elif v4 == 1:
        method58(v6)
        v11 = US6_1()
    elif v4 == 2:
        method58(v6)
        v11 = US6_2()
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v11
def method69(v0 : cp.ndarray) -> Tuple[static_array, i32, i32]:
    v2 = static_array(2)
    v3 = 0
    while method44(v3):
        v5 = u64(v3)
        v6 = v5 * 4
        del v5
        v8 = v0[v6:].view(cp.uint8)
        del v6
        v9 = method62(v8)
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
def method66(v0 : cp.ndarray) -> US8:
    v1 = method60(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5 = method62(v3)
        del v3
        return US8_0(v5)
    elif v1 == 1:
        del v1
        v7, v8 = method67(v3)
        del v3
        return US8_1(v7, v8)
    elif v1 == 2:
        del v1
        v10, v11 = method68(v3)
        del v3
        return US8_2(v10, v11)
    elif v1 == 3:
        del v1
        v13, v14, v15 = method69(v3)
        del v3
        return US8_3(v13, v14, v15)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method70(v0 : cp.ndarray) -> US2:
    v1 = method60(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method58(v3)
        del v3
        return US2_0()
    elif v1 == 1:
        del v1
        method58(v3)
        del v3
        return US2_1()
    elif v1 == 2:
        del v1
        method58(v3)
        del v3
        return US2_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method71(v0 : cp.ndarray) -> i32:
    v2 = v0[1128:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method55(v0 : cp.ndarray) -> Tuple[u32, US3, static_array_list, static_array, US7]:
    v1 = method56(v0)
    v2 = method57(v0)
    v4 = v0[16:].view(cp.uint8)
    if v2 == 0:
        method58(v4)
        v9 = US3_0()
    elif v2 == 1:
        v7 = method59(v4)
        v9 = US3_1(v7)
    else:
        raise Exception("Invalid tag.")
    del v2, v4
    v11 = static_array_list(32)
    v12 = method65(v0)
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
        v21 = method66(v20)
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
        v31 = method70(v30)
        del v30
        v23[v24] = v31
        del v31
        v24 += 1 
    del v24
    v32 = method71(v0)
    v34 = v0[1136:].view(cp.uint8)
    del v0
    if v32 == 0:
        method58(v34)
        v51 = US7_0()
    elif v32 == 1:
        v37, v38, v39, v40, v41, v42 = method61(v34)
        v51 = US7_1(v37, v38, v39, v40, v41, v42)
    elif v32 == 2:
        v44, v45, v46, v47, v48, v49 = method61(v34)
        v51 = US7_2(v44, v45, v46, v47, v48, v49)
    else:
        raise Exception("Invalid tag.")
    del v32, v34
    return v1, v9, v11, v23, v51
def method78(v0 : u32) -> object:
    v1 = v0
    del v0
    return v1
def method77(v0 : u32) -> object:
    return method78(v0)
def method80() -> object:
    v0 = []
    return v0
def method84(v0 : US6) -> object:
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
def method83(v0 : US5) -> object:
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
            v5 = method84(v4)
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
        v11 = method84(v10)
        del v10
        v1.append(v11)
        del v11
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
def method82(v0 : US5, v1 : bool, v2 : static_array, v3 : i32, v4 : static_array, v5 : i32) -> object:
    v6 = method83(v0)
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
    v8 = method82(v0, v1, v2, v3, v4, v5)
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
def method81(v0 : US4) -> object:
    match v0:
        case US4_0(v1, v2, v3, v4, v5, v6): # ChanceCommunityCard
            del v0
            v7 = method82(v1, v2, v3, v4, v5, v6)
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
            v19 = method82(v13, v14, v15, v16, v17, v18)
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
            v38 = method82(v32, v33, v34, v35, v36, v37)
            del v32, v33, v34, v35, v36, v37
            v39 = "TerminalCall"
            v40 = [v39,v38]
            del v38, v39
            return v40
        case US4_5(v41, v42, v43, v44, v45, v46): # TerminalFold
            del v0
            v47 = method82(v41, v42, v43, v44, v45, v46)
            del v41, v42, v43, v44, v45, v46
            v48 = "TerminalFold"
            v49 = [v48,v47]
            del v47, v48
            return v49
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method79(v0 : US3) -> object:
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
            v5 = method81(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method76(v0 : u32, v1 : US3) -> object:
    v2 = method77(v0)
    del v0
    v3 = method79(v1)
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
    v4 = method84(v1)
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
            v2 = method84(v1)
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
        v11 = method98(v10)
        del v10
        v1.append(v11)
        del v11
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
            v10 = method82(v4, v5, v6, v7, v8, v9)
            del v4, v5, v6, v7, v8, v9
            v11 = "GameOver"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US7_2(v13, v14, v15, v16, v17, v18): # WaitingForActionFromPlayerId
            del v0
            v19 = method82(v13, v14, v15, v16, v17, v18)
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
def method75(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7) -> object:
    v5 = method76(v0, v1)
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
def method103(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    v3 = []
    v4 = method104(v0)
    del v0
    v3.append(v4)
    del v4
    v5 = method104(v1)
    del v1
    v3.append(v5)
    del v5
    v6 = method104(v2)
    del v2
    v3.append(v6)
    del v6
    v7 = method80()
    v3.append(v7)
    del v7
    v8 = v3
    del v3
    return v8
def method102(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    return method103(v0, v1, v2)
def method101(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    return method102(v0, v1, v2)
def method100(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    v3 = method101(v0, v1, v2)
    del v0, v1, v2
    v4 = {'model_ptrs': v3}
    del v3
    return v4
def method74(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray) -> object:
    v8 = method75(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    v9 = method100(v5, v6, v7)
    del v5, v6, v7
    v10 = {'game': v8, 'neural': v9}
    del v8, v9
    return v10
def method110(v0 : f32) -> object:
    v1 = v0
    del v0
    return v1
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
def method108(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method5(v2, v3):
        v5 = v0[v3]
        v6 = method109(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method107(v0 : US9) -> object:
    match v0:
        case US9_0(v1): # AddRewardsRando
            del v0
            v2 = method108(v1)
            del v1
            v3 = "AddRewardsRando"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US9_1(v5): # AddRewardsSelf
            del v0
            v6 = method108(v5)
            del v5
            v7 = "AddRewardsSelf"
            v8 = [v7,v6]
            del v6, v7
            return v8
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method106(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method5(v2, v3):
        v5 = v0[v3]
        v6 = method107(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method73(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : list) -> object:
    v9 = []
    v10 = method74(v0, v1, v2, v3, v4, v5, v6, v7)
    del v0, v1, v2, v3, v4, v5, v6, v7
    v9.append(v10)
    del v10
    v11 = method106(v8)
    del v8
    v9.append(v11)
    del v11
    v12 = v9
    del v9
    return v12
def method72(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : list) -> object:
    v9 = method73(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8
    return v9
def method111(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray) -> object:
    v8 = method74(v0, v1, v2, v3, v4, v5, v6, v7)
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
