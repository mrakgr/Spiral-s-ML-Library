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
__device__ void block_row_map_4(float * v0, int v1, float * v2);
struct Tuple1;
struct Tuple2;
struct Tuple3;
struct Union11;
struct Tuple4;
__device__ int int_range_5(int v0, int v1, curandStatePhilox4_32_10_t & v2);
struct Union12;
__device__ int tag_7(Union2 v0);
__device__ bool is_pair_8(int v0, int v1);
__device__ Tuple4 order_9(int v0, int v1);
__device__ Union12 compare_hands_6(Union5 v0, bool v1, static_array<Union2,2> v2, int v3, static_array<int,2> v4, int v5);
__device__ void method_0(unsigned char * v0, unsigned char * v1, unsigned char * v2, StackMut0 & v3, int v4, Union4 v5);
struct Tuple5;
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
struct Closure2 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple1 {
    float v0;
    bool v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(float t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
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
                return Tuple1{v5, true};
            } else {
                return Tuple1{v0, v1};
            }
        } else {
            if (v3){
                return Tuple1{v2, v3};
            } else {
                return Tuple1{v0, v1};
            }
        }
    }
};
struct Tuple2 {
    float v0;
    int v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
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
struct Tuple3 {
    int v0;
    bool v1;
    __device__ Tuple3() = default;
    __device__ Tuple3(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure5 {
    __device__ Tuple3 operator()(Tuple3 tup0, Tuple3 tup1){
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
                return Tuple3{v5, true};
            } else {
                return Tuple3{v0, v1};
            }
        } else {
            if (v3){
                return Tuple3{v2, v3};
            } else {
                return Tuple3{v0, v1};
            }
        }
    }
};
struct Closure6 {
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
struct Tuple4 {
    int v0;
    int v1;
    __device__ Tuple4() = default;
    __device__ Tuple4(int t0, int t1) : v0(t0), v1(t1) {}
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
struct Tuple5 {
    double v1;
    int v0;
    __device__ Tuple5() = default;
    __device__ Tuple5(int t0, double t1) : v0(t0), v1(t1) {}
};
struct Tuple6 {
    double v2;
    float v1;
    int v0;
    __device__ Tuple6() = default;
    __device__ Tuple6(int t0, float t1, double t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple7 {
    int v0;
    float v1;
    __device__ Tuple7() = default;
    __device__ Tuple7(int t0, float t1) : v0(t0), v1(t1) {}
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
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 256;
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
    v1 = v0 < 32;
    return v1;
}
__device__ inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 1;
    return v1;
}
__device__ inline bool while_method_8(int v0){
    bool v1;
    v1 = v0 < 4;
    return v1;
}
__device__ inline bool while_method_9(int v0){
    bool v1;
    v1 = v0 < 4;
    return v1;
}
__device__ inline bool while_method_10(int v0){
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
    while (while_method_7(v64)){
        int v66;
        v66 = 0;
        while (while_method_7(v66)){
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
                while (while_method_7(v76)){
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
            while (while_method_8(v80)){
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
                        while (while_method_7(v129)){
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
                    while (while_method_7(v157)){
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
                while (while_method_7(v168)){
                    int v170;
                    v170 = 0;
                    #pragma unroll
                    while (while_method_9(v170)){
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
                            while (while_method_7(v224)){
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
                    while (while_method_9(v234)){
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
                        while (while_method_7(v261)){
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
                while (while_method_7(v270)){
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
            while (while_method_10(v297)){
                int v299;
                v299 = 0;
                #pragma unroll
                while (while_method_7(v299)){
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
__device__ void block_row_map_4(float * v0, int v1, float * v2){
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
    while (while_method_10(v23)){
        assert("Tensor range check" && 0 <= v23 && v23 < 16);
        int v25;
        v25 = 1024 * v23;
        int v26;
        v26 = v25 + v20;
        float v27[4];
        int v28[4];
        int v29;
        v29 = 0;
        while (while_method_7(v29)){
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
        while (while_method_7(v36)){
            int v38;
            v38 = 0;
            while (while_method_9(v38)){
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
        while (while_method_7(v73)){
            int v75;
            v75 = 0;
            while (while_method_9(v75)){
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
        while (while_method_7(v83)){
            int v85;
            v85 = 0;
            while (while_method_9(v85)){
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
        while (while_method_7(v99)){
            int v101;
            v101 = 0;
            while (while_method_9(v101)){
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
        while (while_method_7(v114)){
            int v116;
            v116 = 0;
            while (while_method_9(v116)){
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
        while (while_method_7(v123)){
            int v125;
            v125 = 0;
            while (while_method_9(v125)){
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
        while (while_method_7(v144)){
            int v146;
            v146 = 0;
            while (while_method_9(v146)){
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
        while (while_method_7(v169)){
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
__device__ int int_range_5(int v0, int v1, curandStatePhilox4_32_10_t & v2){
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
__device__ int tag_7(Union2 v0){
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
__device__ bool is_pair_8(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
__device__ Tuple4 order_9(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple4{v1, v0};
    } else {
        return Tuple4{v0, v1};
    }
}
__device__ Union12 compare_hands_6(Union5 v0, bool v1, static_array<Union2,2> v2, int v3, static_array<int,2> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            __trap();
            break;
        }
        case 1: { // Some
            Union2 v7 = v0.case1.v0;
            int v8;
            v8 = tag_7(v7);
            Union2 v9;
            v9 = v2[0];
            int v11;
            v11 = tag_7(v9);
            Union2 v12;
            v12 = v2[1];
            int v14;
            v14 = tag_7(v12);
            bool v15;
            v15 = is_pair_8(v8, v11);
            bool v16;
            v16 = is_pair_8(v8, v14);
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
                    Tuple4 tmp12 = order_9(v8, v11);
                    v27 = tmp12.v0; v28 = tmp12.v1;
                    int v29; int v30;
                    Tuple4 tmp13 = order_9(v8, v14);
                    v29 = tmp13.v0; v30 = tmp13.v1;
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
        Union6 v920;
        switch (v20.tag) {
            case 0: { // None
                v920 = Union6{Union6_0{}};
                break;
            }
            case 1: { // Some
                Union4 v22 = v20.case1.v0;
                Union7 v760;
                switch (v22.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v729 = v22.case0.v0; bool v730 = v22.case0.v1; static_array<Union2,2> v731 = v22.case0.v2; int v732 = v22.case0.v3; static_array<int,2> v733 = v22.case0.v4; int v734 = v22.case0.v5;
                        curandStatePhilox4_32_10_t & v735 = v3.v5;
                        curandStatePhilox4_32_10_t & v736 = v735;
                        unsigned int & v737 = v3.v0;
                        Union2 v738; unsigned int v739;
                        Tuple0 tmp0 = draw_card_1(v736, v737);
                        v738 = tmp0.v0; v739 = tmp0.v1;
                        v3.v0 = v739;
                        Union1 v740;
                        v740 = Union1{Union1_0{v738}};
                        v18.push(v740);
                        v760 = Union7{Union7_0{v729, v730, v731, v732, v733, v734, v738}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v742 = v3.v5;
                        curandStatePhilox4_32_10_t & v743 = v742;
                        unsigned int & v744 = v3.v0;
                        Union2 v745; unsigned int v746;
                        Tuple0 tmp1 = draw_card_1(v743, v744);
                        v745 = tmp1.v0; v746 = tmp1.v1;
                        v3.v0 = v746;
                        curandStatePhilox4_32_10_t & v747 = v3.v5;
                        curandStatePhilox4_32_10_t & v748 = v747;
                        unsigned int & v749 = v3.v0;
                        Union2 v750; unsigned int v751;
                        Tuple0 tmp2 = draw_card_1(v748, v749);
                        v750 = tmp2.v0; v751 = tmp2.v1;
                        v3.v0 = v751;
                        Union1 v752;
                        v752 = Union1{Union1_2{0, v745}};
                        v18.push(v752);
                        Union1 v753;
                        v753 = Union1{Union1_2{1, v750}};
                        v18.push(v753);
                        v760 = Union7{Union7_1{v745, v750}};
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
                        Union3 v717;
                        switch (v84.tag) {
                            case 0: { // Computer
                                static_array_list<Union1,32> & v87 = v3.v2;
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
                                while (while_method_6(v159)){
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
                                    block_matmul_3(v171, v166, v170, v164);
                                    block_row_map_4(v161, v163, v171);
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
                                while (while_method_10(v236)){
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
                                    while (while_method_7(v258)){
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
                                    while (while_method_7(v265)){
                                        int v267;
                                        v267 = 0;
                                        while (while_method_9(v267)){
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
                                    while (while_method_7(v292)){
                                        assert("Tensor range check" && 0 <= v292 && v292 < 1);
                                        int v294;
                                        v294 = 4 * v292;
                                        assert("Tensor range check" && 0 <= v292 && v292 < 1);
                                        float v295;
                                        v295 = 0.0f;
                                        int v296;
                                        v296 = 0;
                                        while (while_method_9(v296)){
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
                                        while (while_method_9(v313)){
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
                                    while (while_method_7(v321)){
                                        int v323;
                                        v323 = 0;
                                        while (while_method_9(v323)){
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
                                    Tuple1 tmp3 = Tuple1{-1.0f / 0.0f, false};
                                    v330 = tmp3.v0; v331 = tmp3.v1;
                                    int v332;
                                    v332 = 0;
                                    while (while_method_7(v332)){
                                        int v334;
                                        v334 = 0;
                                        while (while_method_9(v334)){
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
                                    Tuple1 tmp4 = cooperative_groups::reduce(v351, Tuple1{v330, v331}, v352);
                                    v353 = tmp4.v0; v354 = tmp4.v1;
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
                                        while (while_method_7(v363)){
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
                                            while (while_method_9(v369)){
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
                                    while (while_method_7(v420)){
                                        int v422;
                                        v422 = 0;
                                        while (while_method_9(v422)){
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
                                    Tuple2 tmp5 = Tuple2{0.0f, 2147483647};
                                    v428 = tmp5.v0; v429 = tmp5.v1;
                                    int v430;
                                    v430 = 0;
                                    while (while_method_7(v430)){
                                        int v432;
                                        v432 = 0;
                                        while (while_method_9(v432)){
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
                                    Tuple2 tmp6 = cooperative_groups::reduce(v444, Tuple2{v428, v429}, v445);
                                    v446 = tmp6.v0; v447 = tmp6.v1;
                                    float v448;
                                    v448 = v353 * v446;
                                    int v449[4];
                                    bool v450[4];
                                    int v451;
                                    v451 = 0;
                                    while (while_method_7(v451)){
                                        int v453;
                                        v453 = 0;
                                        while (while_method_9(v453)){
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
                                    Tuple3 tmp7 = Tuple3{2147483647, false};
                                    v464 = tmp7.v0; v465 = tmp7.v1;
                                    int v466;
                                    v466 = 0;
                                    while (while_method_7(v466)){
                                        int v468;
                                        v468 = 0;
                                        while (while_method_9(v468)){
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
                                    Tuple3 tmp8 = cooperative_groups::reduce(v485, Tuple3{v464, v465}, v486);
                                    v487 = tmp8.v0; v488 = tmp8.v1;
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
                                        while (while_method_7(v497)){
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
                                            while (while_method_9(v503)){
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
                                    Tuple2 tmp9 = Tuple2{0.0f, 2147483647};
                                    v552 = tmp9.v0; v553 = tmp9.v1;
                                    int v554;
                                    v554 = 0;
                                    while (while_method_7(v554)){
                                        int v556;
                                        v556 = 0;
                                        while (while_method_9(v556)){
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
                                    Tuple2 tmp10 = cooperative_groups::reduce(v571, Tuple2{v552, v553}, v572);
                                    v573 = tmp10.v0; v574 = tmp10.v1;
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
                                    while (while_method_7(v579)){
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
                                while (while_method_6(v600)){
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
                                Union11 v642;
                                if (v633){
                                    v642 = Union11{Union11_1{}};
                                } else {
                                    bool v635;
                                    v635 = 1 == v591;
                                    if (v635){
                                        v642 = Union11{Union11_0{}};
                                    } else {
                                        bool v637;
                                        v637 = 2 == v591;
                                        if (v637){
                                            v642 = Union11{Union11_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v642.tag) {
                                    case 0: { // AA_Call
                                        v717 = Union3{Union3_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v643;
                                        v643 = v76[0];
                                        int v645; int v646;
                                        Tuple4 tmp11 = Tuple4{1, v643};
                                        v645 = tmp11.v0; v646 = tmp11.v1;
                                        while (while_method_2(v645)){
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
                                            v717 = Union3{Union3_0{}};
                                        } else {
                                            v717 = Union3{Union3_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v668;
                                        v668 = v77 > 0;
                                        if (v668){
                                            v717 = Union3{Union3_2{}};
                                        } else {
                                            v717 = Union3{Union3_0{}};
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
                                static_array_list<Union3,3> v677;
                                v677 = static_array_list<Union3,3>{};
                                v677.unsafe_set_length(1);
                                Union3 v679;
                                v679 = Union3{Union3_0{}};
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
                                    Union3 v687;
                                    v687 = Union3{Union3_1{}};
                                    v677.push(v687);
                                } else {
                                }
                                bool v688;
                                v688 = v77 > 0;
                                if (v688){
                                    Union3 v689;
                                    v689 = Union3{Union3_2{}};
                                    v677.push(v689);
                                } else {
                                }
                                int v690;
                                v690 = v677.length;
                                int v691;
                                v691 = v690 - 1;
                                int v692;
                                v692 = 0;
                                while (while_method_5(v691, v692)){
                                    int v694;
                                    v694 = v677.length;
                                    int v695;
                                    v695 = int_range_5(v694, v692, v676);
                                    Union3 v696;
                                    v696 = v677[v692];
                                    Union3 v698;
                                    v698 = v677[v695];
                                    v677[v692] = v698;
                                    v677[v695] = v696;
                                    v692 += 1 ;
                                }
                                Union3 v700;
                                v700 = v677.pop();
                                int v701;
                                v701 = sizeof(Union3);
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
                                Union3 * v710;
                                v710 = reinterpret_cast<Union3 *>(&v706[0ull]);
                                int v712;
                                v712 = threadIdx.x;
                                bool v713;
                                v713 = v712 == 0;
                                if (v713){
                                    v710[0] = v700;
                                } else {
                                }
                                __syncthreads();
                                Union3 v714;
                                v714 = v710[0];
                                __syncthreads();
                                v717 = v714;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union1 v718;
                        v718 = Union1{Union1_1{v75, v717}};
                        v18.push(v718);
                        v760 = Union7{Union7_2{v72, v73, v74, v75, v76, v77, v717}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v720 = v22.case3.v0; bool v721 = v22.case3.v1; static_array<Union2,2> v722 = v22.case3.v2; int v723 = v22.case3.v3; static_array<int,2> v724 = v22.case3.v4; int v725 = v22.case3.v5; Union3 v726 = v22.case3.v6;
                        Union1 v727;
                        v727 = Union1{Union1_1{v723, v726}};
                        v18.push(v727);
                        v760 = Union7{Union7_2{v720, v721, v722, v723, v724, v725, v726}};
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
                        v56 = compare_hands_6(v43, v44, v45, v46, v47, v48);
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
                        v760 = Union7{Union7_3{}};
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
                        v760 = Union7{Union7_3{}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                switch (v760.tag) {
                    case 0: { // T_game_chance_community_card
                        Union5 v762 = v760.case0.v0; bool v763 = v760.case0.v1; static_array<Union2,2> v764 = v760.case0.v2; int v765 = v760.case0.v3; static_array<int,2> v766 = v760.case0.v4; int v767 = v760.case0.v5; Union2 v768 = v760.case0.v6;
                        int v769;
                        v769 = 2;
                        int v770; int v771;
                        Tuple4 tmp14 = Tuple4{0, 0};
                        v770 = tmp14.v0; v771 = tmp14.v1;
                        while (while_method_2(v770)){
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
                        while (while_method_2(v784)){
                            v782[v784] = v771;
                            v784 += 1 ;
                        }
                        Union5 v786;
                        v786 = Union5{Union5_1{v768}};
                        Union4 v787;
                        v787 = Union4{Union4_2{v786, true, v764, 0, v782, v769}};
                        v920 = Union6{Union6_1{v787}};
                        break;
                    }
                    case 1: { // T_game_chance_init
                        Union2 v789 = v760.case1.v0; Union2 v790 = v760.case1.v1;
                        int v791;
                        v791 = 2;
                        static_array<int,2> v792;
                        v792[0] = 1;
                        v792[1] = 1;
                        static_array<Union2,2> v794;
                        v794[0] = v789;
                        v794[1] = v790;
                        Union5 v796;
                        v796 = Union5{Union5_0{}};
                        Union4 v797;
                        v797 = Union4{Union4_2{v796, true, v794, 0, v792, v791}};
                        v920 = Union6{Union6_1{v797}};
                        break;
                    }
                    case 2: { // T_game_round
                        Union5 v799 = v760.case2.v0; bool v800 = v760.case2.v1; static_array<Union2,2> v801 = v760.case2.v2; int v802 = v760.case2.v3; static_array<int,2> v803 = v760.case2.v4; int v804 = v760.case2.v5; Union3 v805 = v760.case2.v6;
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
                                            Tuple4 tmp15 = Tuple4{0, 0};
                                            v875 = tmp15.v0; v876 = tmp15.v1;
                                            while (while_method_2(v875)){
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
                                            while (while_method_2(v889)){
                                                v887[v889] = v876;
                                                v889 += 1 ;
                                            }
                                            static_array<int,2> v891;
                                            int v893;
                                            v893 = 0;
                                            while (while_method_2(v893)){
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
                                Union2 v806 = v799.case1.v0;
                                switch (v805.tag) {
                                    case 0: { // Call
                                        if (v800){
                                            int v808;
                                            v808 = v802 ^ 1;
                                            v912 = Union4{Union4_2{v799, false, v801, v808, v803, v804}};
                                        } else {
                                            int v810; int v811;
                                            Tuple4 tmp16 = Tuple4{0, 0};
                                            v810 = tmp16.v0; v811 = tmp16.v1;
                                            while (while_method_2(v810)){
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
                                            while (while_method_2(v824)){
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
                                            Tuple4 tmp17 = Tuple4{0, 0};
                                            v831 = tmp17.v0; v832 = tmp17.v1;
                                            while (while_method_2(v831)){
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
                                            while (while_method_2(v845)){
                                                v843[v845] = v832;
                                                v845 += 1 ;
                                            }
                                            static_array<int,2> v847;
                                            int v849;
                                            v849 = 0;
                                            while (while_method_2(v849)){
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
                        v920 = Union6{Union6_1{v912}};
                        break;
                    }
                    case 3: { // T_none
                        v920 = Union6{Union6_0{}};
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
__device__ inline bool while_method_11(int v0){
    bool v1;
    v1 = v0 < 16384;
    return v1;
}
__device__ inline bool while_method_12(int v0){
    bool v1;
    v1 = v0 < 8192;
    return v1;
}
__device__ inline bool while_method_13(int v0){
    bool v1;
    v1 = v0 < 2048;
    return v1;
}
__device__ inline bool while_method_14(int v0){
    bool v1;
    v1 = v0 < 24;
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
                while (while_method_2(v52)){
                    int v54; double v55;
                    Tuple5 tmp18 = Tuple5{0, 0.0};
                    v54 = tmp18.v0; v55 = tmp18.v1;
                    while (while_method_6(v54)){
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
                while (while_method_6(v71)){
                    int v73;
                    v73 = 0;
                    while (while_method_2(v73)){
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
                while (while_method_6(v94)){
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
                while (while_method_2(v132)){
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
                while (while_method_2(v142)){
                    int v144; double v145;
                    Tuple5 tmp19 = Tuple5{0, 0.0};
                    v144 = tmp19.v0; v145 = tmp19.v1;
                    while (while_method_6(v144)){
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
                while (while_method_2(v155)){
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
                while (while_method_6(v160)){
                    int v162;
                    v162 = 0;
                    while (while_method_2(v162)){
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
                while (while_method_6(v184)){
                    int v186; float v187; double v188;
                    Tuple6 tmp20 = Tuple6{0, 0.0f, 0.0};
                    v186 = tmp20.v0; v187 = tmp20.v1; v188 = tmp20.v2;
                    while (while_method_2(v186)){
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
                while (while_method_6(v199)){
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
                while (while_method_6(v218)){
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
                    while (while_method_7(v272)){
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
                        while (while_method_7(v297)){
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
                        while (while_method_7(v305)){
                            int v307;
                            v307 = 0;
                            while (while_method_2(v307)){
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
                        while (while_method_7(v332)){
                            int v334;
                            v334 = 0;
                            while (while_method_2(v334)){
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
                        while (while_method_7(v340)){
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
            v375 = int_range_5(v374, v373, v351);
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
        while (while_method_10(v380)){
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
            while (while_method_11(v397)){
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
            while (while_method_11(v414)){
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
        while (while_method_0(v447)){
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
            while (while_method_7(v463)){
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
            while (while_method_7(v470)){
                int v472;
                v472 = 0;
                while (while_method_9(v472)){
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
            while (while_method_7(v517)){
                int v519;
                v519 = 0;
                while (while_method_9(v519)){
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
            while (while_method_7(v543)){
                int v545;
                v545 = 0;
                while (while_method_9(v545)){
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
            while (while_method_7(v559)){
                int v561;
                v561 = 0;
                while (while_method_9(v561)){
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
            while (while_method_7(v568)){
                int v570;
                v570 = 0;
                while (while_method_9(v570)){
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
            while (while_method_7(v577)){
                int v579;
                v579 = 0;
                while (while_method_9(v579)){
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
            while (while_method_7(v593)){
                int v595;
                v595 = 0;
                while (while_method_9(v595)){
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
            while (while_method_7(v603)){
                int v605;
                v605 = 0;
                while (while_method_9(v605)){
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
            while (while_method_7(v618)){
                int v620;
                v620 = 0;
                while (while_method_9(v620)){
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
            while (while_method_7(v628)){
                int v630;
                v630 = 0;
                while (while_method_9(v630)){
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
            while (while_method_7(v638)){
                int v640;
                v640 = 0;
                while (while_method_9(v640)){
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
            while (while_method_7(v659)){
                int v661;
                v661 = 0;
                while (while_method_9(v661)){
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
            while (while_method_7(v673)){
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
            while (while_method_1(v702)){
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
                while (while_method_9(v715)){
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
        while (while_method_13(v731)){
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
            Tuple7 tmp21 = Tuple7{0, v726};
            v747 = tmp21.v0; v748 = tmp21.v1;
            while (while_method_9(v747)){
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
        while (while_method_14(v773)){
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
        while (while_method_10(v799)){
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
            while (while_method_11(v815)){
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
            while (while_method_11(v832)){
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
        while (while_method_1(v852)){
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
            while (while_method_9(v869)){
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
    while (while_method_13(v883)){
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
        while (while_method_9(v903)){
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
                v19 = cp.zeros(8192,dtype=cp.float32) # type: ignore
                v20 = cp.zeros(8192,dtype=cp.float32) # type: ignore
                v21 = cp.empty(8192,dtype=cp.float32)
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
                    while method55(v35):
                        assert 0 <= v32 < 32, 'Tensor range check'
                        assert 0 <= v35 < 256, 'Tensor range check'
                        v37 = 256 * v32
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
        v50, v51, v52, v53, v54 = method56(v12)
        del v12
        return method73(v50, v51, v52, v53, v54, v8, v9, v10, v18)
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
        return method112(v30, v31, v29, v23, v32, v2, v1, v0)
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
def method55(v0 : i32) -> bool:
    v1 = v0 < 256
    del v0
    return v1
def method57(v0 : cp.ndarray) -> u32:
    v2 = v0[0:].view(cp.uint32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method58(v0 : cp.ndarray) -> i32:
    v2 = v0[4:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method59(v0 : cp.ndarray) -> None:
    del v0
    return 
def method61(v0 : cp.ndarray) -> i32:
    v2 = v0[0:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method63(v0 : cp.ndarray) -> US6:
    v1 = method61(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method59(v3)
        del v3
        return US6_0()
    elif v1 == 1:
        del v1
        method59(v3)
        del v3
        return US6_1()
    elif v1 == 2:
        del v1
        method59(v3)
        del v3
        return US6_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method62(v0 : cp.ndarray) -> Tuple[US5, bool, static_array, i32, static_array, i32]:
    v1 = method61(v0)
    v3 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method59(v3)
        v8 = US5_0()
    elif v1 == 1:
        v6 = method63(v3)
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
        v21 = method63(v20)
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
        v34 = method61(v33)
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
    v1 = method61(v0)
    v3 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method59(v3)
        v8 = US5_0()
    elif v1 == 1:
        v6 = method63(v3)
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
        v21 = method63(v20)
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
        v34 = method61(v33)
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
        method59(v40)
        v45 = US1_0()
    elif v38 == 1:
        method59(v40)
        v45 = US1_1()
    elif v38 == 2:
        method59(v40)
        v45 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v38, v40
    return v8, v11, v13, v24, v26, v37, v45
def method60(v0 : cp.ndarray) -> US4:
    v1 = method61(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5, v6, v7, v8, v9, v10 = method62(v3)
        del v3
        return US4_0(v5, v6, v7, v8, v9, v10)
    elif v1 == 1:
        del v1
        method59(v3)
        del v3
        return US4_1()
    elif v1 == 2:
        del v1
        v13, v14, v15, v16, v17, v18 = method62(v3)
        del v3
        return US4_2(v13, v14, v15, v16, v17, v18)
    elif v1 == 3:
        del v1
        v20, v21, v22, v23, v24, v25, v26 = method64(v3)
        del v3
        return US4_3(v20, v21, v22, v23, v24, v25, v26)
    elif v1 == 4:
        del v1
        v28, v29, v30, v31, v32, v33 = method62(v3)
        del v3
        return US4_4(v28, v29, v30, v31, v32, v33)
    elif v1 == 5:
        del v1
        v35, v36, v37, v38, v39, v40 = method62(v3)
        del v3
        return US4_5(v35, v36, v37, v38, v39, v40)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method66(v0 : cp.ndarray) -> i32:
    v2 = v0[80:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method68(v0 : cp.ndarray) -> Tuple[i32, US1]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method58(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method59(v6)
        v11 = US1_0()
    elif v4 == 1:
        method59(v6)
        v11 = US1_1()
    elif v4 == 2:
        method59(v6)
        v11 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v11
def method69(v0 : cp.ndarray) -> Tuple[i32, US6]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method58(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method59(v6)
        v11 = US6_0()
    elif v4 == 1:
        method59(v6)
        v11 = US6_1()
    elif v4 == 2:
        method59(v6)
        v11 = US6_2()
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v11
def method70(v0 : cp.ndarray) -> Tuple[static_array, i32, i32]:
    v2 = static_array(2)
    v3 = 0
    while method44(v3):
        v5 = u64(v3)
        v6 = v5 * 4
        del v5
        v8 = v0[v6:].view(cp.uint8)
        del v6
        v9 = method63(v8)
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
    v1 = method61(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5 = method63(v3)
        del v3
        return US8_0(v5)
    elif v1 == 1:
        del v1
        v7, v8 = method68(v3)
        del v3
        return US8_1(v7, v8)
    elif v1 == 2:
        del v1
        v10, v11 = method69(v3)
        del v3
        return US8_2(v10, v11)
    elif v1 == 3:
        del v1
        v13, v14, v15 = method70(v3)
        del v3
        return US8_3(v13, v14, v15)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method71(v0 : cp.ndarray) -> US2:
    v1 = method61(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method59(v3)
        del v3
        return US2_0()
    elif v1 == 1:
        del v1
        method59(v3)
        del v3
        return US2_1()
    elif v1 == 2:
        del v1
        method59(v3)
        del v3
        return US2_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method72(v0 : cp.ndarray) -> i32:
    v2 = v0[1128:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method56(v0 : cp.ndarray) -> Tuple[u32, US3, static_array_list, static_array, US7]:
    v1 = method57(v0)
    v2 = method58(v0)
    v4 = v0[16:].view(cp.uint8)
    if v2 == 0:
        method59(v4)
        v9 = US3_0()
    elif v2 == 1:
        v7 = method60(v4)
        v9 = US3_1(v7)
    else:
        raise Exception("Invalid tag.")
    del v2, v4
    v11 = static_array_list(32)
    v12 = method66(v0)
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
        v21 = method67(v20)
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
        v31 = method71(v30)
        del v30
        v23[v24] = v31
        del v31
        v24 += 1 
    del v24
    v32 = method72(v0)
    v34 = v0[1136:].view(cp.uint8)
    del v0
    if v32 == 0:
        method59(v34)
        v51 = US7_0()
    elif v32 == 1:
        v37, v38, v39, v40, v41, v42 = method62(v34)
        v51 = US7_1(v37, v38, v39, v40, v41, v42)
    elif v32 == 2:
        v44, v45, v46, v47, v48, v49 = method62(v34)
        v51 = US7_2(v44, v45, v46, v47, v48, v49)
    else:
        raise Exception("Invalid tag.")
    del v32, v34
    return v1, v9, v11, v23, v51
def method79(v0 : u32) -> object:
    v1 = v0
    del v0
    return v1
def method78(v0 : u32) -> object:
    return method79(v0)
def method81() -> object:
    v0 = []
    return v0
def method85(v0 : US6) -> object:
    match v0:
        case US6_0(): # Jack
            del v0
            v1 = method81()
            v2 = "Jack"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US6_1(): # King
            del v0
            v4 = method81()
            v5 = "King"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US6_2(): # Queen
            del v0
            v7 = method81()
            v8 = "Queen"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method84(v0 : US5) -> object:
    match v0:
        case US5_0(): # None
            del v0
            v1 = method81()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US5_1(v4): # Some
            del v0
            v5 = method85(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method86(v0 : bool) -> object:
    v1 = v0
    del v0
    return v1
def method87(v0 : static_array) -> object:
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
        v11 = method85(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method88(v0 : i32) -> object:
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
        v11 = method88(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method83(v0 : US5, v1 : bool, v2 : static_array, v3 : i32, v4 : static_array, v5 : i32) -> object:
    v6 = method84(v0)
    del v0
    v7 = method86(v1)
    del v1
    v8 = method87(v2)
    del v2
    v9 = method88(v3)
    del v3
    v10 = method89(v4)
    del v4
    v11 = method88(v5)
    del v5
    v12 = {'community_card': v6, 'is_button_s_first_move': v7, 'pl_card': v8, 'player_turn': v9, 'pot': v10, 'raises_left': v11}
    del v6, v7, v8, v9, v10, v11
    return v12
def method91(v0 : US1) -> object:
    match v0:
        case US1_0(): # Call
            del v0
            v1 = method81()
            v2 = "Call"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US1_1(): # Fold
            del v0
            v4 = method81()
            v5 = "Fold"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US1_2(): # Raise
            del v0
            v7 = method81()
            v8 = "Raise"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method90(v0 : US5, v1 : bool, v2 : static_array, v3 : i32, v4 : static_array, v5 : i32, v6 : US1) -> object:
    v7 = []
    v8 = method83(v0, v1, v2, v3, v4, v5)
    del v0, v1, v2, v3, v4, v5
    v7.append(v8)
    del v8
    v9 = method91(v6)
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
            v10 = method81()
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
            v29 = method90(v22, v23, v24, v25, v26, v27, v28)
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
def method80(v0 : US3) -> object:
    match v0:
        case US3_0(): # None
            del v0
            v1 = method81()
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
def method77(v0 : u32, v1 : US3) -> object:
    v2 = method78(v0)
    del v0
    v3 = method80(v1)
    del v1
    v4 = {'deck': v2, 'game': v3}
    del v2, v3
    return v4
def method95(v0 : i32, v1 : US1) -> object:
    v2 = []
    v3 = method88(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method91(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method96(v0 : i32, v1 : US6) -> object:
    v2 = []
    v3 = method88(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method85(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method97(v0 : static_array, v1 : i32, v2 : i32) -> object:
    v3 = method87(v0)
    del v0
    v4 = method88(v1)
    del v1
    v5 = method88(v2)
    del v2
    v6 = {'cards_shown': v3, 'chips_won': v4, 'winner_id': v5}
    del v3, v4, v5
    return v6
def method94(v0 : US8) -> object:
    match v0:
        case US8_0(v1): # CommunityCardIs
            del v0
            v2 = method85(v1)
            del v1
            v3 = "CommunityCardIs"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US8_1(v5, v6): # PlayerAction
            del v0
            v7 = method95(v5, v6)
            del v5, v6
            v8 = "PlayerAction"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US8_2(v10, v11): # PlayerGotCard
            del v0
            v12 = method96(v10, v11)
            del v10, v11
            v13 = "PlayerGotCard"
            v14 = [v13,v12]
            del v12, v13
            return v14
        case US8_3(v15, v16, v17): # Showdown
            del v0
            v18 = method97(v15, v16, v17)
            del v15, v16, v17
            v19 = "Showdown"
            v20 = [v19,v18]
            del v18, v19
            return v20
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method93(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method5(v2, v3):
        v6 = v0[v3]
        v7 = method94(v6)
        del v6
        v1.append(v7)
        del v7
        v3 += 1 
    del v0, v2, v3
    return v1
def method99(v0 : US2) -> object:
    match v0:
        case US2_0(): # Computer
            del v0
            v1 = method81()
            v2 = "Computer"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US2_1(): # Human
            del v0
            v4 = method81()
            v5 = "Human"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US2_2(): # Random
            del v0
            v7 = method81()
            v8 = "Random"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method98(v0 : static_array) -> object:
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
        v11 = method99(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method100(v0 : US7) -> object:
    match v0:
        case US7_0(): # GameNotStarted
            del v0
            v1 = method81()
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
def method92(v0 : static_array_list, v1 : static_array, v2 : US7) -> object:
    v3 = method93(v0)
    del v0
    v4 = method98(v1)
    del v1
    v5 = method100(v2)
    del v2
    v6 = {'messages': v3, 'pl_type': v4, 'ui_game_state': v5}
    del v3, v4, v5
    return v6
def method76(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7) -> object:
    v5 = method77(v0, v1)
    del v0, v1
    v6 = method92(v2, v3, v4)
    del v2, v3, v4
    v7 = {'private': v5, 'public': v6}
    del v5, v6
    return v7
def method106(v0 : cp.ndarray) -> object:
    v1 = v0
    del v0
    return v1
def method105(v0 : cp.ndarray) -> object:
    return method106(v0)
def method104(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    v3 = []
    v4 = method105(v0)
    del v0
    v3.append(v4)
    del v4
    v5 = method105(v1)
    del v1
    v3.append(v5)
    del v5
    v6 = method105(v2)
    del v2
    v3.append(v6)
    del v6
    v7 = method81()
    v3.append(v7)
    del v7
    v8 = v3
    del v3
    return v8
def method103(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    return method104(v0, v1, v2)
def method102(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    return method103(v0, v1, v2)
def method101(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> object:
    v3 = method102(v0, v1, v2)
    del v0, v1, v2
    v4 = {'model_ptrs': v3}
    del v3
    return v4
def method75(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray) -> object:
    v8 = method76(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    v9 = method101(v5, v6, v7)
    del v5, v6, v7
    v10 = {'game': v8, 'neural': v9}
    del v8, v9
    return v10
def method111(v0 : f32) -> object:
    v1 = v0
    del v0
    return v1
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
def method108(v0 : US9) -> object:
    match v0:
        case US9_0(v1): # AddRewardsRando
            del v0
            v2 = method109(v1)
            del v1
            v3 = "AddRewardsRando"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US9_1(v5): # AddRewardsSelf
            del v0
            v6 = method109(v5)
            del v5
            v7 = "AddRewardsSelf"
            v8 = [v7,v6]
            del v6, v7
            return v8
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method107(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method5(v2, v3):
        v5 = v0[v3]
        v6 = method108(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method74(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : list) -> object:
    v9 = []
    v10 = method75(v0, v1, v2, v3, v4, v5, v6, v7)
    del v0, v1, v2, v3, v4, v5, v6, v7
    v9.append(v10)
    del v10
    v11 = method107(v8)
    del v8
    v9.append(v11)
    del v11
    v12 = v9
    del v9
    return v12
def method73(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : list) -> object:
    v9 = method74(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8
    return v9
def method112(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray) -> object:
    v8 = method75(v0, v1, v2, v3, v4, v5, v6, v7)
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
