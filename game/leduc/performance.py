kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups.h>
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

struct Union0;
struct Union2;
struct Union3;
struct Union1;
struct Union5;
struct Union4;
struct Mut0;
struct Union6;
struct Union7;
struct Tuple0;
__device__ unsigned int loop_2(unsigned int v0, curandStatePhilox4_32_10_t & v1);
__device__ Tuple0 draw_card_1(curandStatePhilox4_32_10_t & v0, unsigned int v1);
struct Tuple1;
struct Union8;
struct Union9;
__device__ void method_4(float * v0, int v1, float * v2, int v3, float * v4, int v5);
__device__ void method_5(unsigned int * v0, int v1, float * v2);
__device__ void noinline_run_nn_model_3(unsigned char * v0, unsigned char * v1, curandStatePhilox4_32_10_t & v2);
struct Tuple2;
struct Tuple3;
struct Tuple4;
struct Tuple5;
__device__ Tuple2 method_6(curandStatePhilox4_32_10_t & v0, int * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int v9, int v10);
__device__ float method_7(int * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int v8, int v9, int v10);
struct Union10;
__device__ int int_range_8(int v0, int v1, curandStatePhilox4_32_10_t & v2);
struct Union11;
__device__ int tag_10(Union2 v0);
__device__ bool is_pair_11(int v0, int v1);
__device__ Tuple1 order_12(int v0, int v1);
__device__ Union11 compare_hands_9(Union5 v0, bool v1, static_array<Union2,2l> v2, int v3, static_array<int,2l> v4, int v5);
__device__ void method_0(unsigned char * v0, unsigned char * v1, sptr<Mut0> v2, Union6 v3);
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
    static_array<Union2,2l> v0;
    int v1;
    int v2;
    __device__ Union1_3(static_array<Union2,2l> t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
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
struct Union4_0 { // None
};
struct Union4_1 { // Some
    Union5 v0;
    static_array<Union2,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_1(Union5 t0, bool t1, static_array<Union2,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
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
struct Mut0 {
    int refc{0};
    Union4 v0;
    cooperative_groups::grid_group v2;
    static_array_list<Union1,32l> v3;
    static_array<Union0,2l> v4;
    static_array<float,2l> v5;
    curandStatePhilox4_32_10_t v6;
    unsigned int v1;
    __device__ Mut0() = default;
    __device__ Mut0(Union4 t0, unsigned int t1, cooperative_groups::grid_group t2, static_array_list<Union1,32l> t3, static_array<Union0,2l> t4, static_array<float,2l> t5, curandStatePhilox4_32_10_t t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
};
struct Union6_0 { // ChanceCommunityCard
    Union5 v0;
    static_array<Union2,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union6_0(Union5 t0, bool t1, static_array<Union2,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union6_0() = delete;
};
struct Union6_1 { // ChanceInit
};
struct Union6_2 { // Round
    Union5 v0;
    static_array<Union2,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union6_2(Union5 t0, bool t1, static_array<Union2,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union6_2() = delete;
};
struct Union6_3 { // RoundWithAction
    Union5 v0;
    static_array<Union2,2l> v2;
    static_array<int,2l> v4;
    Union3 v6;
    int v3;
    int v5;
    bool v1;
    __device__ Union6_3(Union5 t0, bool t1, static_array<Union2,2l> t2, int t3, static_array<int,2l> t4, int t5, Union3 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union6_3() = delete;
};
struct Union6_4 { // TerminalCall
    Union5 v0;
    static_array<Union2,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union6_4(Union5 t0, bool t1, static_array<Union2,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union6_4() = delete;
};
struct Union6_5 { // TerminalFold
    Union5 v0;
    static_array<Union2,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union6_5(Union5 t0, bool t1, static_array<Union2,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union6_5() = delete;
};
struct Union6 {
    union {
        Union6_0 case0; // ChanceCommunityCard
        Union6_1 case1; // ChanceInit
        Union6_2 case2; // Round
        Union6_3 case3; // RoundWithAction
        Union6_4 case4; // TerminalCall
        Union6_5 case5; // TerminalFold
    };
    unsigned char tag{255};
    __device__ Union6() {}
    __device__ Union6(Union6_0 t) : tag(0), case0(t) {} // ChanceCommunityCard
    __device__ Union6(Union6_1 t) : tag(1), case1(t) {} // ChanceInit
    __device__ Union6(Union6_2 t) : tag(2), case2(t) {} // Round
    __device__ Union6(Union6_3 t) : tag(3), case3(t) {} // RoundWithAction
    __device__ Union6(Union6_4 t) : tag(4), case4(t) {} // TerminalCall
    __device__ Union6(Union6_5 t) : tag(5), case5(t) {} // TerminalFold
    __device__ Union6(Union6 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(x.case0); break; // ChanceCommunityCard
            case 1: new (&this->case1) Union6_1(x.case1); break; // ChanceInit
            case 2: new (&this->case2) Union6_2(x.case2); break; // Round
            case 3: new (&this->case3) Union6_3(x.case3); break; // RoundWithAction
            case 4: new (&this->case4) Union6_4(x.case4); break; // TerminalCall
            case 5: new (&this->case5) Union6_5(x.case5); break; // TerminalFold
        }
    }
    __device__ Union6(Union6 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(std::move(x.case0)); break; // ChanceCommunityCard
            case 1: new (&this->case1) Union6_1(std::move(x.case1)); break; // ChanceInit
            case 2: new (&this->case2) Union6_2(std::move(x.case2)); break; // Round
            case 3: new (&this->case3) Union6_3(std::move(x.case3)); break; // RoundWithAction
            case 4: new (&this->case4) Union6_4(std::move(x.case4)); break; // TerminalCall
            case 5: new (&this->case5) Union6_5(std::move(x.case5)); break; // TerminalFold
        }
    }
    __device__ Union6 & operator=(Union6 & x) {
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
            this->~Union6();
            new (this) Union6{x};
        }
        return *this;
    }
    __device__ Union6 & operator=(Union6 && x) {
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
            this->~Union6();
            new (this) Union6{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union6() {
        switch(this->tag){
            case 0: this->case0.~Union6_0(); break; // ChanceCommunityCard
            case 1: this->case1.~Union6_1(); break; // ChanceInit
            case 2: this->case2.~Union6_2(); break; // Round
            case 3: this->case3.~Union6_3(); break; // RoundWithAction
            case 4: this->case4.~Union6_4(); break; // TerminalCall
            case 5: this->case5.~Union6_5(); break; // TerminalFold
        }
        this->tag = 255;
    }
};
struct Union7_0 { // None
};
struct Union7_1 { // Some
    Union6 v0;
    __device__ Union7_1(Union6 t0) : v0(t0) {}
    __device__ Union7_1() = delete;
};
struct Union7 {
    union {
        Union7_0 case0; // None
        Union7_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union7() {}
    __device__ Union7(Union7_0 t) : tag(0), case0(t) {} // None
    __device__ Union7(Union7_1 t) : tag(1), case1(t) {} // Some
    __device__ Union7(Union7 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(x.case0); break; // None
            case 1: new (&this->case1) Union7_1(x.case1); break; // Some
        }
    }
    __device__ Union7(Union7 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union7_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union7 & operator=(Union7 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union7();
            new (this) Union7{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union7() {
        switch(this->tag){
            case 0: this->case0.~Union7_0(); break; // None
            case 1: this->case1.~Union7_1(); break; // Some
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
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 16l;
    return v1;
}
__device__ inline bool while_method_2(Union7 v0){
    switch (v0.tag) {
        case 0: { // None
            return false;
            break;
        }
        case 1: { // Some
            Union6 v1 = v0.case1.v0;
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
    v5 = 0ul - v0;
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
        v8 = v5 + 1l;
        unsigned int v9;
        v9 = __fns(v1,0ul,v8);
        v12 = v9;
    } else {
        int v10;
        v10 = v5 - v6;
        printf("%s\n", "Cannot find the n-th set bit.");
        __trap();
    }
    bool v13;
    v13 = 0ul == v12;
    Union2 v31;
    if (v13){
        v31 = Union2{Union2_1{}};
    } else {
        bool v15;
        v15 = 1ul == v12;
        if (v15){
            v31 = Union2{Union2_1{}};
        } else {
            bool v17;
            v17 = 2ul == v12;
            if (v17){
                v31 = Union2{Union2_2{}};
            } else {
                bool v19;
                v19 = 3ul == v12;
                if (v19){
                    v31 = Union2{Union2_2{}};
                } else {
                    bool v21;
                    v21 = 4ul == v12;
                    if (v21){
                        v31 = Union2{Union2_0{}};
                    } else {
                        bool v23;
                        v23 = 5ul == v12;
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
    v33 = 1ul << v32;
    unsigned int v34;
    v34 = v1 ^ v33;
    return Tuple0{v31, v34};
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 32768l;
    return v1;
}
__device__ inline bool while_method_5(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 8l;
    return v1;
}
__device__ void method_4(float * v0, int v1, float * v2, int v3, float * v4, int v5){
    extern __shared__ unsigned char v6[];
    float * v7;
    v7 = reinterpret_cast<float *>(&v6[0ull]);
    float * v9;
    v9 = reinterpret_cast<float *>(&v6[34816ull]);
    float * v11;
    v11 = reinterpret_cast<float *>(&v6[0ull]);
    int v13;
    v13 = threadIdx.x;
    int v14;
    v14 = v13 / 32l;
    bool v15;
    v15 = 0l <= v14;
    bool v16;
    v16 = v15 == false;
    if (v16){
        assert("The index needs to be zero or positive." && v15);
    } else {
    }
    int v18;
    v18 = v14 % 8l;
    int v19;
    v19 = v14 / 8l;
    bool v20;
    v20 = v19 < 1l;
    bool v21;
    v21 = v20 == false;
    if (v21){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v20);
    } else {
    }
    assert("Tensor range check" && 0 <= v19 && v19 < 1l);
    assert("Tensor range check" && 0 <= v18 && v18 < 8l);
    int v23;
    v23 = 16l * v18;
    int v24;
    v24 = 17408l * v19;
    int v25;
    v25 = v24 + v23;
    float * v26;
    v26 = v11+v25;
    assert("Tensor range check" && 0 <= v19 && v19 < 1l);
    int v28;
    v28 = 8704l * v19;
    int v29;
    v29 = threadIdx.x;
    int v30;
    v30 = v29 % 32l;
    bool v31;
    v31 = 0l <= v30;
    bool v32;
    v32 = v31 == false;
    if (v32){
        assert("The index needs to be zero or positive." && v31);
    } else {
    }
    int v34;
    v34 = v30 % 4l;
    int v35;
    v35 = v30 / 4l;
    bool v36;
    v36 = v35 < 8l;
    bool v37;
    v37 = v36 == false;
    if (v37){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v36);
    } else {
    }
    assert("Tensor range check" && 0 <= v35 && v35 < 8l);
    assert("Tensor range check" && 0 <= v34 && v34 < 4l);
    int v39;
    v39 = v34 + v28;
    int v40;
    v40 = 68l * v35;
    int v41;
    v41 = v40 + v39;
    float * v42;
    v42 = v7+v41;
    assert("Tensor range check" && 0 <= v18 && v18 < 8l);
    int v44;
    v44 = 1088l * v18;
    int v45;
    v45 = threadIdx.x;
    int v46;
    v46 = v45 % 32l;
    bool v47;
    v47 = 0l <= v46;
    bool v48;
    v48 = v47 == false;
    if (v48){
        assert("The index needs to be zero or positive." && v47);
    } else {
    }
    int v50;
    v50 = v46 % 4l;
    int v51;
    v51 = v46 / 4l;
    bool v52;
    v52 = v51 < 8l;
    bool v53;
    v53 = v52 == false;
    if (v53){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v52);
    } else {
    }
    assert("Tensor range check" && 0 <= v51 && v51 < 8l);
    assert("Tensor range check" && 0 <= v50 && v50 < 4l);
    int v55;
    v55 = v50 + v44;
    int v56;
    v56 = 68l * v51;
    int v57;
    v57 = v56 + v55;
    float * v58;
    v58 = v9+v57;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v60[8l];
    int v61;
    v61 = 0l;
    while (while_method_3(v61)){
        int v63;
        v63 = 0l;
        while (while_method_6(v63)){
            assert("Tensor range check" && 0 <= v61 && v61 < 2l);
            assert("Tensor range check" && 0 <= v63 && v63 < 1l);
            int v65;
            v65 = 128l * v63;
            int v66;
            v66 = v65 + v3;
            int v67;
            v67 = 16384l * v61;
            int v68;
            v68 = v67 + v66;
            float * v69;
            v69 = v2+v68;
            // Pushing the loop unrolling to: 0
            int v71;
            v71 = 0l;
            #pragma unroll
            while (while_method_7(v71)){
                int v73;
                v73 = 0l;
                #pragma unroll
                while (while_method_6(v73)){
                    assert("Tensor range check" && 0 <= v71 && v71 < 8l);
                    assert("Tensor range check" && 0 <= v73 && v73 < 1l);
                    int v75;
                    v75 = v71 + v73;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v76 = v60[v75];
                    wmma::fill_fragment(v76, 0.0f);
                    v73 += 1l ;
                }
                v71 += 1l ;
            }
            int v77;
            v77 = 0l;
            #pragma unroll
            while (while_method_3(v77)){
                assert("Tensor range check" && 0 <= v61 && v61 < 2l);
                int v79;
                v79 = v67 + v5;
                assert("Tensor range check" && 0 <= v77 && v77 < 2l);
                int v80;
                v80 = 64l * v77;
                int v81;
                v81 = v80 + v79;
                float * v82;
                v82 = v4+v81;
                assert("Tensor range check" && 0 <= v63 && v63 < 1l);
                int v84;
                v84 = 16384l * v63;
                int v85;
                v85 = v84 + v1;
                assert("Tensor range check" && 0 <= v77 && v77 < 2l);
                int v86;
                v86 = v80 + v85;
                float * v87;
                v87 = v0+v86;
                int v89;
                v89 = threadIdx.x;
                bool v90;
                v90 = 0l <= v89;
                bool v91;
                v91 = v90 == false;
                if (v91){
                    assert("The index needs to be zero or positive." && v90);
                } else {
                }
                int v93;
                v93 = v89 % 16l;
                int v94;
                v94 = v89 / 16l;
                bool v95;
                v95 = v94 < 16l;
                bool v96;
                v96 = v95 == false;
                if (v96){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v95);
                } else {
                }
                assert("Tensor range check" && 0 <= v94 && v94 < 16l);
                assert("Tensor range check" && 0 <= v93 && v93 < 16l);
                int v98;
                v98 = 4l * v93;
                int v99;
                v99 = 68l * v94;
                int v100;
                v100 = v99 + v98;
                int v101;
                v101 = 128l * v94;
                int v102;
                v102 = v101 + v98;
                float * v103;
                v103 = v9+v100;
                float * v105;
                v105 = v87+v102;
                int v107;
                v107 = 0l;
                #pragma unroll
                while (while_method_7(v107)){
                    int v109;
                    v109 = 0l;
                    #pragma unroll
                    while (while_method_6(v109)){
                        assert("Tensor range check" && 0 <= v107 && v107 < 8l);
                        assert("Tensor range check" && 0 <= v109 && v109 < 1l);
                        int v111;
                        v111 = 64l * v109;
                        int v112;
                        v112 = 1088l * v107;
                        int v113;
                        v113 = v112 + v111;
                        int v114;
                        v114 = 2048l * v107;
                        int v115;
                        v115 = v114 + v111;
                        float v116[4l];
                        int v117;
                        v117 = 0l;
                        #pragma unroll
                        while (while_method_0(v117)){
                            assert("Tensor range check" && 0 <= v117 && v117 < 4l);
                            int v119;
                            v119 = v117 + v115;
                            float v120;
                            v120 = v105[v119];
                            float v121;
                            v121 = wmma::__float_to_tf32(v120);
                            assert("Tensor range check" && 0 <= v117 && v117 < 4l);
                            v116[v117] = v121;
                            v117 += 1l ;
                        }
                        int4* v122;
                        v122 = reinterpret_cast<int4*>(v116 + 0l);
                        int4* v123;
                        v123 = reinterpret_cast<int4*>(v103 + v113);
                        assert("Pointer alignment check" && (unsigned long long)(v122) % 4l == 0 && (unsigned long long)(v123) % 4l == 0);
                        *v123 = *v122;
                        v109 += 1l ;
                    }
                    v107 += 1l ;
                }
                int v124;
                v124 = threadIdx.x;
                bool v125;
                v125 = 0l <= v124;
                bool v126;
                v126 = v125 == false;
                if (v126){
                    assert("The index needs to be zero or positive." && v125);
                } else {
                }
                int v128;
                v128 = v124 % 16l;
                int v129;
                v129 = v124 / 16l;
                bool v130;
                v130 = v129 < 16l;
                bool v131;
                v131 = v130 == false;
                if (v131){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v130);
                } else {
                }
                assert("Tensor range check" && 0 <= v129 && v129 < 16l);
                assert("Tensor range check" && 0 <= v128 && v128 < 16l);
                int v133;
                v133 = 4l * v128;
                int v134;
                v134 = 68l * v129;
                int v135;
                v135 = v134 + v133;
                int v136;
                v136 = 128l * v129;
                int v137;
                v137 = v136 + v133;
                float * v138;
                v138 = v7+v135;
                float * v140;
                v140 = v82+v137;
                int v142;
                v142 = 0l;
                #pragma unroll
                while (while_method_7(v142)){
                    int v144;
                    v144 = 0l;
                    #pragma unroll
                    while (while_method_6(v144)){
                        assert("Tensor range check" && 0 <= v142 && v142 < 8l);
                        assert("Tensor range check" && 0 <= v144 && v144 < 1l);
                        int v146;
                        v146 = 64l * v144;
                        int v147;
                        v147 = 1088l * v142;
                        int v148;
                        v148 = v147 + v146;
                        int v149;
                        v149 = 2048l * v142;
                        int v150;
                        v150 = v149 + v146;
                        float v151[4l];
                        int v152;
                        v152 = 0l;
                        #pragma unroll
                        while (while_method_0(v152)){
                            assert("Tensor range check" && 0 <= v152 && v152 < 4l);
                            int v154;
                            v154 = v152 + v150;
                            float v155;
                            v155 = v140[v154];
                            float v156;
                            v156 = wmma::__float_to_tf32(v155);
                            assert("Tensor range check" && 0 <= v152 && v152 < 4l);
                            v151[v152] = v156;
                            v152 += 1l ;
                        }
                        int4* v157;
                        v157 = reinterpret_cast<int4*>(v151 + 0l);
                        int4* v158;
                        v158 = reinterpret_cast<int4*>(v138 + v148);
                        assert("Pointer alignment check" && (unsigned long long)(v157) % 4l == 0 && (unsigned long long)(v158) % 4l == 0);
                        *v158 = *v157;
                        v144 += 1l ;
                    }
                    v142 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v159[64l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v160[8l];
                int v161;
                v161 = 0l;
                #pragma unroll
                while (while_method_7(v161)){
                    int v163;
                    v163 = 0l;
                    #pragma unroll
                    while (while_method_7(v163)){
                        assert("Tensor range check" && 0 <= v161 && v161 < 8l);
                        assert("Tensor range check" && 0 <= v163 && v163 < 8l);
                        int v165;
                        v165 = 8l * v161;
                        int v166;
                        v166 = v165 + v163;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v167 = v159[v166];
                        assert("Tensor range check" && 0 <= v161 && v161 < 8l);
                        int v168;
                        v168 = 1088l * v161;
                        assert("Tensor range check" && 0 <= v163 && v163 < 8l);
                        int v169;
                        v169 = 8l * v163;
                        int v170;
                        v170 = v169 + v168;
                        int v171;
                        v171 = 0l;
                        #pragma unroll
                        while (while_method_3(v171)){
                            int v173;
                            v173 = 0l;
                            #pragma unroll
                            while (while_method_3(v173)){
                                assert("Tensor range check" && 0 <= v171 && v171 < 2l);
                                assert("Tensor range check" && 0 <= v173 && v173 < 2l);
                                int v175;
                                v175 = 544l * v173;
                                int v176;
                                v176 = v175 + v170;
                                int v177;
                                v177 = 4l * v171;
                                int v178;
                                v178 = v177 + v176;
                                float v179;
                                v179 = v42[v178];
                                bool v180;
                                v180 = 0l <= v173;
                                bool v182;
                                if (v180){
                                    bool v181;
                                    v181 = v173 < 2l;
                                    v182 = v181;
                                } else {
                                    v182 = false;
                                }
                                bool v183;
                                v183 = v182 == false;
                                if (v183){
                                    assert("The indices should be inside the range of the dimension." && v182);
                                } else {
                                }
                                bool v185;
                                v185 = 0l <= v171;
                                bool v187;
                                if (v185){
                                    bool v186;
                                    v186 = v171 < 2l;
                                    v187 = v186;
                                } else {
                                    v187 = false;
                                }
                                bool v188;
                                v188 = v187 == false;
                                if (v188){
                                    assert("The indices should be inside the range of the dimension." && v187);
                                } else {
                                }
                                int v190;
                                v190 = v171 * 2l;
                                int v191;
                                v191 = v173 + v190;
                                v167.x[v191] = v179;
                                v173 += 1l ;
                            }
                            v171 += 1l ;
                        }
                        v163 += 1l ;
                    }
                    v161 += 1l ;
                }
                int v192;
                v192 = 0l;
                #pragma unroll
                while (while_method_6(v192)){
                    int v194;
                    v194 = 0l;
                    #pragma unroll
                    while (while_method_7(v194)){
                        assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                        assert("Tensor range check" && 0 <= v194 && v194 < 8l);
                        int v196;
                        v196 = 8l * v192;
                        int v197;
                        v197 = v196 + v194;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v198 = v160[v197];
                        assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                        int v199;
                        v199 = 1088l * v192;
                        assert("Tensor range check" && 0 <= v194 && v194 < 8l);
                        int v200;
                        v200 = 8l * v194;
                        int v201;
                        v201 = v200 + v199;
                        int v202;
                        v202 = 0l;
                        #pragma unroll
                        while (while_method_3(v202)){
                            int v204;
                            v204 = 0l;
                            #pragma unroll
                            while (while_method_3(v204)){
                                assert("Tensor range check" && 0 <= v202 && v202 < 2l);
                                assert("Tensor range check" && 0 <= v204 && v204 < 2l);
                                int v206;
                                v206 = 4l * v204;
                                int v207;
                                v207 = v206 + v201;
                                int v208;
                                v208 = 544l * v202;
                                int v209;
                                v209 = v208 + v207;
                                float v210;
                                v210 = v58[v209];
                                bool v211;
                                v211 = 0l <= v204;
                                bool v213;
                                if (v211){
                                    bool v212;
                                    v212 = v204 < 2l;
                                    v213 = v212;
                                } else {
                                    v213 = false;
                                }
                                bool v214;
                                v214 = v213 == false;
                                if (v214){
                                    assert("The indices should be inside the range of the dimension." && v213);
                                } else {
                                }
                                bool v216;
                                v216 = 0l <= v202;
                                bool v218;
                                if (v216){
                                    bool v217;
                                    v217 = v202 < 2l;
                                    v218 = v217;
                                } else {
                                    v218 = false;
                                }
                                bool v219;
                                v219 = v218 == false;
                                if (v219){
                                    assert("The indices should be inside the range of the dimension." && v218);
                                } else {
                                }
                                int v221;
                                v221 = v202 * 2l;
                                int v222;
                                v222 = v204 + v221;
                                v198.x[v222] = v210;
                                v204 += 1l ;
                            }
                            v202 += 1l ;
                        }
                        v194 += 1l ;
                    }
                    v192 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                int v223;
                v223 = 0l;
                #pragma unroll
                while (while_method_7(v223)){
                    int v225;
                    v225 = 0l;
                    #pragma unroll
                    while (while_method_6(v225)){
                        int v227;
                        v227 = 0l;
                        #pragma unroll
                        while (while_method_7(v227)){
                            assert("Tensor range check" && 0 <= v223 && v223 < 8l);
                            assert("Tensor range check" && 0 <= v225 && v225 < 1l);
                            int v229;
                            v229 = v223 + v225;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v230 = v60[v229];
                            assert("Tensor range check" && 0 <= v223 && v223 < 8l);
                            assert("Tensor range check" && 0 <= v227 && v227 < 8l);
                            int v231;
                            v231 = 8l * v223;
                            int v232;
                            v232 = v231 + v227;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v233 = v159[v232];
                            assert("Tensor range check" && 0 <= v225 && v225 < 1l);
                            assert("Tensor range check" && 0 <= v227 && v227 < 8l);
                            int v234;
                            v234 = 8l * v225;
                            int v235;
                            v235 = v234 + v227;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v236 = v160[v235];
                            wmma::mma_sync(v230, v233, v236, v230);
                            v227 += 1l ;
                        }
                        v225 += 1l ;
                    }
                    v223 += 1l ;
                }
                v77 += 1l ;
            }
            int v237;
            v237 = 0l;
            #pragma unroll
            while (while_method_7(v237)){
                int v239;
                v239 = 0l;
                #pragma unroll
                while (while_method_6(v239)){
                    assert("Tensor range check" && 0 <= v237 && v237 < 8l);
                    assert("Tensor range check" && 0 <= v239 && v239 < 1l);
                    int v241;
                    v241 = v237 + v239;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v242 = v60[v241];
                    assert("Tensor range check" && 0 <= v237 && v237 < 8l);
                    assert("Tensor range check" && 0 <= v239 && v239 < 1l);
                    int v243;
                    v243 = 16l * v239;
                    int v244;
                    v244 = 2176l * v237;
                    int v245;
                    v245 = v244 + v243;
                    float * v246;
                    v246 = v26+v245;
                    wmma::store_matrix_sync(v246, v242, 136l, wmma::mem_row_major);
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
            v252 = v248 % 32l;
            int v253;
            v253 = v248 / 32l;
            bool v254;
            v254 = v253 < 8l;
            bool v255;
            v255 = v254 == false;
            if (v255){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v254);
            } else {
            }
            assert("Tensor range check" && 0 <= v253 && v253 < 8l);
            assert("Tensor range check" && 0 <= v252 && v252 < 32l);
            int v257;
            v257 = 4l * v252;
            int v258;
            v258 = 128l * v253;
            int v259;
            v259 = v258 + v257;
            int v260;
            v260 = 136l * v253;
            int v261;
            v261 = v260 + v257;
            float * v262;
            v262 = v69+v259;
            float * v264;
            v264 = v11+v261;
            int v266;
            v266 = 0l;
            #pragma unroll
            while (while_method_1(v266)){
                int v268;
                v268 = 0l;
                #pragma unroll
                while (while_method_6(v268)){
                    assert("Tensor range check" && 0 <= v266 && v266 < 16l);
                    assert("Tensor range check" && 0 <= v268 && v268 < 1l);
                    int v270;
                    v270 = 128l * v268;
                    int v271;
                    v271 = 1024l * v266;
                    int v272;
                    v272 = v271 + v270;
                    int v273;
                    v273 = 1088l * v266;
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
            v63 += 1l ;
        }
        v61 += 1l ;
    }
    return ;
}
__device__ inline bool while_method_8(int v0){
    bool v1;
    v1 = v0 < 32l;
    return v1;
}
__device__ void method_5(unsigned int * v0, int v1, float * v2){
    int v3;
    v3 = blockIdx.x;
    assert("Tensor range check" && 0 <= v3 && v3 < 24l);
    int v4;
    v4 = 32768l * v3;
    int v5;
    v5 = blockIdx.x;
    assert("Tensor range check" && 0 <= v5 && v5 < 24l);
    int v6;
    v6 = 256l * v5;
    int v7;
    v7 = v6 + v1;
    int v8;
    v8 = threadIdx.x;
    bool v9;
    v9 = 0l <= v8;
    bool v10;
    v10 = v9 == false;
    if (v10){
        assert("The index needs to be zero or positive." && v9);
    } else {
    }
    int v12;
    v12 = v8 % 32l;
    int v13;
    v13 = v8 / 32l;
    bool v14;
    v14 = v13 < 8l;
    bool v15;
    v15 = v14 == false;
    if (v15){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v14);
    } else {
    }
    assert("Tensor range check" && 0 <= v13 && v13 < 8l);
    assert("Tensor range check" && 0 <= v12 && v12 < 32l);
    int v17;
    v17 = 4l * v12;
    int v18;
    v18 = v17 + v4;
    int v19;
    v19 = 128l * v13;
    int v20;
    v20 = v19 + v18;
    assert("Tensor range check" && 0 <= v13 && v13 < 8l);
    int v21;
    v21 = v13 + v7;
    int v22;
    v22 = 0l;
    while (while_method_8(v22)){
        assert("Tensor range check" && 0 <= v22 && v22 < 32l);
        int v24;
        v24 = 1024l * v22;
        int v25;
        v25 = v24 + v20;
        float v26[4l];
        int v27[4l];
        int v28;
        v28 = 0l;
        while (while_method_6(v28)){
            assert("Tensor range check" && 0 <= v28 && v28 < 1l);
            int v30;
            v30 = 4l * v28;
            assert("Tensor range check" && 0 <= v28 && v28 < 1l);
            int v31;
            v31 = 128l * v28;
            int v32;
            v32 = v31 + v25;
            int4* v33;
            v33 = reinterpret_cast<int4*>(v2 + v32);
            int4* v34;
            v34 = reinterpret_cast<int4*>(v26 + v30);
            assert("Pointer alignment check" && (unsigned long long)(v33) % 4l == 0 && (unsigned long long)(v34) % 4l == 0);
            *v34 = *v33;
            v28 += 1l ;
        }
        int v35;
        v35 = 0l;
        while (while_method_6(v35)){
            int v37;
            v37 = 0l;
            while (while_method_0(v37)){
                bool v39;
                v39 = 0l <= v37;
                bool v41;
                if (v39){
                    bool v40;
                    v40 = v37 < 4l;
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
                v44 = 0l <= v12;
                bool v46;
                if (v44){
                    bool v45;
                    v45 = v12 < 32l;
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
                v49 = v12 * 4l;
                int v50;
                v50 = v37 + v49;
                bool v51;
                v51 = 0l <= v35;
                bool v53;
                if (v51){
                    bool v52;
                    v52 = v35 < 1l;
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
                v56 = v35 * 128l;
                int v57;
                v57 = v50 + v56;
                assert("Tensor range check" && 0 <= v35 && v35 < 1l);
                assert("Tensor range check" && 0 <= v37 && v37 < 4l);
                int v58;
                v58 = 4l * v35;
                int v59;
                v59 = v58 + v37;
                v27[v59] = v57;
                v37 += 1l ;
            }
            v35 += 1l ;
        }
        bool v60;
        v60 = 0l <= v13;
        bool v61;
        v61 = v60 && v14;
        bool v62;
        v62 = v61 == false;
        if (v62){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v61);
        } else {
        }
        bool v64;
        v64 = 0l <= v22;
        bool v66;
        if (v64){
            bool v65;
            v65 = v22 < 32l;
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
        v69 = v22 * 8l;
        int v70;
        v70 = v69 + v13;
        unsigned int v71[4l];
        int v72;
        v72 = 0l;
        while (while_method_6(v72)){
            int v74;
            v74 = 0l;
            while (while_method_0(v74)){
                assert("Tensor range check" && 0 <= v72 && v72 < 1l);
                assert("Tensor range check" && 0 <= v74 && v74 < 4l);
                int v76;
                v76 = 4l * v72;
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
                    v82 = 0ul;
                } else {
                    unsigned int v81;
                    v81 = 1ul << v79;
                    v82 = v81;
                }
                assert("Tensor range check" && 0 <= v72 && v72 < 1l);
                assert("Tensor range check" && 0 <= v74 && v74 < 4l);
                v71[v77] = v82;
                v74 += 1l ;
            }
            v72 += 1l ;
        }
        unsigned int v83;
        v83 = 0ul;
        int v84;
        v84 = 0l;
        while (while_method_6(v84)){
            int v86;
            v86 = 0l;
            while (while_method_0(v86)){
                assert("Tensor range check" && 0 <= v84 && v84 < 1l);
                assert("Tensor range check" && 0 <= v86 && v86 < 4l);
                int v88;
                v88 = 4l * v84;
                int v89;
                v89 = v88 + v86;
                unsigned int v90;
                v90 = v71[v89];
                unsigned int v91;
                v91 = v83 | v90;
                v83 = v91;
                v86 += 1l ;
            }
            v84 += 1l ;
        }
        auto v92 = cooperative_groups::coalesced_threads();
        int v93;
        v93 = threadIdx.x;
        int v94;
        v94 = v93 / 32l;
        auto v95 = cooperative_groups::labeled_partition(v92,v94);
        Closure0 v96{};
        unsigned int v97;
        v97 = cooperative_groups::reduce(v95, v83, v96);
        unsigned int v98;
        v98 = v97 % 4096ul;
        assert("Tensor range check" && 0 <= v22 && v22 < 32l);
        int v99;
        v99 = 8l * v22;
        int v100;
        v100 = v99 + v21;
        v0[v100] = v98;
        v22 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return ;
}
__device__ __noinline__ void noinline_run_nn_model_3(unsigned char * v0, unsigned char * v1, curandStatePhilox4_32_10_t & v2){
    int v3;
    v3 = 0l;
    while (while_method_0(v3)){
        float * v5;
        v5 = reinterpret_cast<float *>(&v0[0ull]);
        float * v7;
        v7 = reinterpret_cast<float *>(&v1[0ull]);
        assert("Tensor range check" && 0 <= v3 && v3 < 4l);
        int v9;
        v9 = 16384l * v3;
        float * v10;
        v10 = reinterpret_cast<float *>(&v0[3145728ull]);
        int v12;
        v12 = blockIdx.x;
        assert("Tensor range check" && 0 <= v12 && v12 < 24l);
        int v13;
        v13 = 32768l * v12;
        int v14;
        v14 = blockIdx.x;
        assert("Tensor range check" && 0 <= v14 && v14 < 24l);
        int v15;
        v15 = 32768l * v14;
        method_4(v7, v9, v10, v15, v5, v13);
        unsigned int * v16;
        v16 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
        assert("Tensor range check" && 0 <= v3 && v3 < 4l);
        int v18;
        v18 = 6144l * v3;
        method_5(v16, v18, v10);
        int * v19;
        v19 = reinterpret_cast<int *>(&v1[262144ull]);
        float * v21;
        v21 = reinterpret_cast<float *>(&v1[262160ull]);
        float * v23;
        v23 = reinterpret_cast<float *>(&v1[524304ull]);
        float * v25;
        v25 = reinterpret_cast<float *>(&v1[786448ull]);
        float * v27;
        v27 = reinterpret_cast<float *>(&v1[1048592ull]);
        float * v29;
        v29 = reinterpret_cast<float *>(&v1[1310736ull]);
        float * v31;
        v31 = reinterpret_cast<float *>(&v1[1572880ull]);
        float * v33;
        v33 = reinterpret_cast<float *>(&v1[1835024ull]);
        int * v35;
        v35 = reinterpret_cast<int *>(&v0[6389760ull]);
        float * v37;
        v37 = reinterpret_cast<float *>(&v0[7962624ull]);
        int * v39;
        v39 = reinterpret_cast<int *>(&v0[9535488ull]);
        int * v41;
        v41 = reinterpret_cast<int *>(&v0[11108352ull]);
        double * v43;
        v43 = reinterpret_cast<double *>(&v0[12681216ull]);
        double * v45;
        v45 = reinterpret_cast<double *>(&v0[18972672ull]);
        double * v47;
        v47 = reinterpret_cast<double *>(&v1[2097168ull]);
        double * v49;
        v49 = reinterpret_cast<double *>(&v1[2490384ull]);
        int * v51;
        v51 = reinterpret_cast<int *>(&v1[2883600ull]);
        v3 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return ;
}
__device__ Tuple2 method_6(curandStatePhilox4_32_10_t & v0, int * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int v9, int v10){
    assert("Tensor range check" && 0 <= v10 && v10 < 4l);
    int v11;
    v11 = 16384l * v10;
    assert("Tensor range check" && 0 <= v9 && v9 < 4096l);
    int v12;
    v12 = 4l * v9;
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
    v36 = v35 <= 81920ull;
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
    assert("Tensor range check" && 0 <= v51 && v51 < 256l);
    v43[v51] = v14;
    v45[v51] = v16;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v52;
    v52 = 0l <= v51;
    bool v53;
    v53 = v52 == false;
    if (v53){
        assert("The index needs to be zero or positive." && v52);
    } else {
    }
    int v55;
    v55 = v51 % 1l;
    bool v56;
    v56 = v51 < 256l;
    bool v57;
    v57 = v56 == false;
    if (v57){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v56);
    } else {
    }
    assert("Tensor range check" && 0 <= v51 && v51 < 256l);
    int v59;
    v59 = 0l;
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
        v64 = 0l <= v59;
        bool v66;
        if (v64){
            bool v65;
            v65 = v59 < 1l;
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
        v69 = v59 * 256l;
        int v70;
        v70 = v69 + v51;
        assert("Tensor range check" && 0 <= v59 && v59 < 1l);
        int v71;
        v71 = 256l * v59;
        int v72;
        v72 = v71 + v51;
        float * v73;
        v73 = v43[v72];
        float * v74;
        v74 = v45[v72];
        int v75;
        v75 = blockIdx.x;
        int v76;
        v76 = v75 * 256l;
        int v77;
        v77 = v76 + v70;
        assert("Tensor range check" && 0 <= v55 && v55 < 1l);
        int v78;
        v78 = 4l * v55;
        float v79[4l];
        float v80[4l];
        int v81[4l];
        int v82;
        v82 = 0l;
        while (while_method_6(v82)){
            assert("Tensor range check" && 0 <= v82 && v82 < 1l);
            int v84;
            v84 = 4l * v82;
            assert("Tensor range check" && 0 <= v82 && v82 < 1l);
            int v85;
            v85 = v84 + v78;
            int4* v86;
            v86 = reinterpret_cast<int4*>(v73 + v85);
            int4* v87;
            v87 = reinterpret_cast<int4*>(v79 + v84);
            assert("Pointer alignment check" && (unsigned long long)(v86) % 4l == 0 && (unsigned long long)(v87) % 4l == 0);
            *v87 = *v86;
            int4* v88;
            v88 = reinterpret_cast<int4*>(v74 + v85);
            int4* v89;
            v89 = reinterpret_cast<int4*>(v80 + v84);
            assert("Pointer alignment check" && (unsigned long long)(v88) % 4l == 0 && (unsigned long long)(v89) % 4l == 0);
            *v89 = *v88;
            v82 += 1l ;
        }
        int v90;
        v90 = 0l;
        while (while_method_6(v90)){
            int v92;
            v92 = 0l;
            while (while_method_0(v92)){
                bool v94;
                v94 = 0l <= v92;
                bool v96;
                if (v94){
                    bool v95;
                    v95 = v92 < 4l;
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
                v99 = 0l <= v55;
                bool v101;
                if (v99){
                    bool v100;
                    v100 = v55 < 1l;
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
                v104 = v55 * 4l;
                int v105;
                v105 = v92 + v104;
                bool v106;
                v106 = 0l <= v90;
                bool v108;
                if (v106){
                    bool v107;
                    v107 = v90 < 1l;
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
                v111 = v90 * 4l;
                int v112;
                v112 = v105 + v111;
                assert("Tensor range check" && 0 <= v90 && v90 < 1l);
                assert("Tensor range check" && 0 <= v92 && v92 < 4l);
                int v113;
                v113 = 4l * v90;
                int v114;
                v114 = v113 + v92;
                v81[v114] = v112;
                v92 += 1l ;
            }
            v90 += 1l ;
        }
        bool v115[4l];
        int v116;
        v116 = 0l;
        while (while_method_6(v116)){
            int v118;
            v118 = 0l;
            while (while_method_0(v118)){
                assert("Tensor range check" && 0 <= v116 && v116 < 1l);
                assert("Tensor range check" && 0 <= v118 && v118 < 4l);
                int v120;
                v120 = 4l * v116;
                int v121;
                v121 = v120 + v118;
                float v122;
                v122 = v79[v121];
                int v123;
                v123 = v81[v121];
                bool v124;
                v124 = v123 < 3l;
                assert("Tensor range check" && 0 <= v116 && v116 < 1l);
                assert("Tensor range check" && 0 <= v118 && v118 < 4l);
                v115[v121] = v124;
                v118 += 1l ;
            }
            v116 += 1l ;
        }
        float v125[4l];
        int v126;
        v126 = 0l;
        while (while_method_6(v126)){
            int v128;
            v128 = 0l;
            while (while_method_0(v128)){
                assert("Tensor range check" && 0 <= v126 && v126 < 1l);
                assert("Tensor range check" && 0 <= v128 && v128 < 4l);
                int v130;
                v130 = 4l * v126;
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
                assert("Tensor range check" && 0 <= v126 && v126 < 1l);
                assert("Tensor range check" && 0 <= v128 && v128 < 4l);
                v125[v131] = v136;
                v128 += 1l ;
            }
            v126 += 1l ;
        }
        float v137;
        v137 = 0.0f;
        int v138;
        v138 = 0l;
        while (while_method_6(v138)){
            int v140;
            v140 = 0l;
            while (while_method_0(v140)){
                assert("Tensor range check" && 0 <= v138 && v138 < 1l);
                assert("Tensor range check" && 0 <= v140 && v140 < 4l);
                int v142;
                v142 = 4l * v138;
                int v143;
                v143 = v142 + v140;
                float v144;
                v144 = v125[v143];
                float v145;
                v145 = v137 + v144;
                v137 = v145;
                v140 += 1l ;
            }
            v138 += 1l ;
        }
        auto v146 = cooperative_groups::coalesced_threads();
        int v147;
        v147 = threadIdx.x;
        auto v148 = cooperative_groups::labeled_partition(v146,v147);
        Closure1 v149{};
        float v150;
        v150 = cooperative_groups::reduce(v148, v137, v149);
        int v151[4l];
        int v152;
        v152 = 0l;
        while (while_method_6(v152)){
            int v154;
            v154 = 0l;
            while (while_method_0(v154)){
                assert("Tensor range check" && 0 <= v152 && v152 < 1l);
                assert("Tensor range check" && 0 <= v154 && v154 < 4l);
                int v156;
                v156 = 4l * v152;
                int v157;
                v157 = v156 + v154;
                bool v158;
                v158 = v115[v157];
                int v159;
                if (v158){
                    v159 = 1l;
                } else {
                    v159 = 0l;
                }
                assert("Tensor range check" && 0 <= v152 && v152 < 1l);
                assert("Tensor range check" && 0 <= v154 && v154 < 4l);
                v151[v157] = v159;
                v154 += 1l ;
            }
            v152 += 1l ;
        }
        int v160;
        v160 = 0l;
        int v161;
        v161 = 0l;
        while (while_method_6(v161)){
            int v163;
            v163 = 0l;
            while (while_method_0(v163)){
                assert("Tensor range check" && 0 <= v161 && v161 < 1l);
                assert("Tensor range check" && 0 <= v163 && v163 < 4l);
                int v165;
                v165 = 4l * v161;
                int v166;
                v166 = v165 + v163;
                int v167;
                v167 = v151[v166];
                int v168;
                v168 = v160 + v167;
                v160 = v168;
                v163 += 1l ;
            }
            v161 += 1l ;
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
        float v176[4l];
        int v177;
        v177 = 0l;
        while (while_method_6(v177)){
            int v179;
            v179 = 0l;
            while (while_method_0(v179)){
                assert("Tensor range check" && 0 <= v177 && v177 < 1l);
                assert("Tensor range check" && 0 <= v179 && v179 < 4l);
                int v181;
                v181 = 4l * v177;
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
                assert("Tensor range check" && 0 <= v177 && v177 < 1l);
                assert("Tensor range check" && 0 <= v179 && v179 < 4l);
                v176[v182] = v190;
                v179 += 1l ;
            }
            v177 += 1l ;
        }
        float v191[4l];
        float v192;
        v192 = 0.0f;
        int v193;
        v193 = 0l;
        while (while_method_6(v193)){
            assert("Tensor range check" && 0 <= v193 && v193 < 1l);
            int v195;
            v195 = 4l * v193;
            assert("Tensor range check" && 0 <= v193 && v193 < 1l);
            int v196; float v197;
            Tuple3 tmp4 = Tuple3{0l, 0.0f};
            v196 = tmp4.v0; v197 = tmp4.v1;
            while (while_method_0(v196)){
                assert("Tensor range check" && 0 <= v196 && v196 < 4l);
                int v199;
                v199 = v196 + v195;
                float v200;
                v200 = v176[v199];
                float v201;
                v201 = v197 + v200;
                v197 = v201;
                v196 += 1l ;
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
            Tuple3 tmp5 = Tuple3{0l, v211};
            v212 = tmp5.v0; v213 = tmp5.v1;
            while (while_method_0(v212)){
                assert("Tensor range check" && 0 <= v212 && v212 < 4l);
                int v215;
                v215 = v212 + v195;
                float v216;
                v216 = v176[v215];
                float v217;
                v217 = v213 + v216;
                assert("Tensor range check" && 0 <= v212 && v212 < 4l);
                v191[v215] = v217;
                v213 = v217;
                v212 += 1l ;
            }
            float v218;
            v218 = v192 + v210;
            v192 = v218;
            v193 += 1l ;
        }
        float v219[4l];
        bool v220[4l];
        int v221;
        v221 = 0l;
        while (while_method_6(v221)){
            int v223;
            v223 = 0l;
            while (while_method_0(v223)){
                assert("Tensor range check" && 0 <= v221 && v221 < 1l);
                assert("Tensor range check" && 0 <= v223 && v223 < 4l);
                int v225;
                v225 = 4l * v221;
                int v226;
                v226 = v225 + v223;
                float v227;
                v227 = v191[v226];
                float v228;
                v228 = v176[v226];
                bool v229;
                v229 = v228 > 0.0f;
                assert("Tensor range check" && 0 <= v221 && v221 < 1l);
                assert("Tensor range check" && 0 <= v223 && v223 < 4l);
                v219[v226] = v227;
                v220[v226] = v229;
                v223 += 1l ;
            }
            v221 += 1l ;
        }
        float v230; bool v231;
        Tuple4 tmp6 = Tuple4{-1.0f / 0.0f, false};
        v230 = tmp6.v0; v231 = tmp6.v1;
        int v232;
        v232 = 0l;
        while (while_method_6(v232)){
            int v234;
            v234 = 0l;
            while (while_method_0(v234)){
                assert("Tensor range check" && 0 <= v232 && v232 < 1l);
                assert("Tensor range check" && 0 <= v234 && v234 < 4l);
                int v236;
                v236 = 4l * v232;
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
                v234 += 1l ;
            }
            v232 += 1l ;
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
        float v256[4l];
        int v257[4l];
        int v258;
        v258 = 0l;
        while (while_method_6(v258)){
            int v260;
            v260 = 0l;
            while (while_method_0(v260)){
                assert("Tensor range check" && 0 <= v258 && v258 < 1l);
                assert("Tensor range check" && 0 <= v260 && v260 < 4l);
                int v262;
                v262 = 4l * v258;
                int v263;
                v263 = v262 + v260;
                int v264;
                v264 = v81[v263];
                float v265;
                v265 = curand_uniform(&v0);
                assert("Tensor range check" && 0 <= v258 && v258 < 1l);
                assert("Tensor range check" && 0 <= v260 && v260 < 4l);
                v256[v263] = v265;
                v257[v263] = v264;
                v260 += 1l ;
            }
            v258 += 1l ;
        }
        float v266; int v267;
        Tuple2 tmp8 = Tuple2{0.0f, 2147483647l};
        v266 = tmp8.v0; v267 = tmp8.v1;
        int v268;
        v268 = 0l;
        while (while_method_6(v268)){
            int v270;
            v270 = 0l;
            while (while_method_0(v270)){
                assert("Tensor range check" && 0 <= v268 && v268 < 1l);
                assert("Tensor range check" && 0 <= v270 && v270 < 4l);
                int v272;
                v272 = 4l * v268;
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
                v270 += 1l ;
            }
            v268 += 1l ;
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
        int v286[4l];
        bool v287[4l];
        int v288;
        v288 = 0l;
        while (while_method_6(v288)){
            int v290;
            v290 = 0l;
            while (while_method_0(v290)){
                assert("Tensor range check" && 0 <= v288 && v288 < 1l);
                assert("Tensor range check" && 0 <= v290 && v290 < 4l);
                int v292;
                v292 = 4l * v288;
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
                    v299 = 2147483647l; v300 = false;
                }
                assert("Tensor range check" && 0 <= v288 && v288 < 1l);
                assert("Tensor range check" && 0 <= v290 && v290 < 4l);
                v286[v293] = v299;
                v287[v293] = v300;
                v290 += 1l ;
            }
            v288 += 1l ;
        }
        int v301; bool v302;
        Tuple5 tmp10 = Tuple5{2147483647l, false};
        v301 = tmp10.v0; v302 = tmp10.v1;
        int v303;
        v303 = 0l;
        while (while_method_6(v303)){
            int v305;
            v305 = 0l;
            while (while_method_0(v305)){
                assert("Tensor range check" && 0 <= v303 && v303 < 1l);
                assert("Tensor range check" && 0 <= v305 && v305 < 4l);
                int v307;
                v307 = 4l * v303;
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
                v305 += 1l ;
            }
            v303 += 1l ;
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
        bool v327[4l];
        int v328;
        v328 = 0l;
        while (while_method_6(v328)){
            int v330;
            v330 = 0l;
            while (while_method_0(v330)){
                assert("Tensor range check" && 0 <= v328 && v328 < 1l);
                assert("Tensor range check" && 0 <= v330 && v330 < 4l);
                int v332;
                v332 = 4l * v328;
                int v333;
                v333 = v332 + v330;
                float v334;
                v334 = v80[v333];
                int v335;
                v335 = v81[v333];
                bool v336;
                v336 = v335 < 3l;
                assert("Tensor range check" && 0 <= v328 && v328 < 1l);
                assert("Tensor range check" && 0 <= v330 && v330 < 4l);
                v327[v333] = v336;
                v330 += 1l ;
            }
            v328 += 1l ;
        }
        float v337[4l];
        int v338;
        v338 = 0l;
        while (while_method_6(v338)){
            int v340;
            v340 = 0l;
            while (while_method_0(v340)){
                assert("Tensor range check" && 0 <= v338 && v338 < 1l);
                assert("Tensor range check" && 0 <= v340 && v340 < 4l);
                int v342;
                v342 = 4l * v338;
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
                assert("Tensor range check" && 0 <= v338 && v338 < 1l);
                assert("Tensor range check" && 0 <= v340 && v340 < 4l);
                v337[v343] = v348;
                v340 += 1l ;
            }
            v338 += 1l ;
        }
        float v349;
        v349 = 0.0f;
        int v350;
        v350 = 0l;
        while (while_method_6(v350)){
            int v352;
            v352 = 0l;
            while (while_method_0(v352)){
                assert("Tensor range check" && 0 <= v350 && v350 < 1l);
                assert("Tensor range check" && 0 <= v352 && v352 < 4l);
                int v354;
                v354 = 4l * v350;
                int v355;
                v355 = v354 + v352;
                float v356;
                v356 = v337[v355];
                float v357;
                v357 = v349 + v356;
                v349 = v357;
                v352 += 1l ;
            }
            v350 += 1l ;
        }
        auto v358 = cooperative_groups::coalesced_threads();
        int v359;
        v359 = threadIdx.x;
        auto v360 = cooperative_groups::labeled_partition(v358,v359);
        float v361;
        v361 = cooperative_groups::reduce(v360, v349, v149);
        int v362[4l];
        int v363;
        v363 = 0l;
        while (while_method_6(v363)){
            int v365;
            v365 = 0l;
            while (while_method_0(v365)){
                assert("Tensor range check" && 0 <= v363 && v363 < 1l);
                assert("Tensor range check" && 0 <= v365 && v365 < 4l);
                int v367;
                v367 = 4l * v363;
                int v368;
                v368 = v367 + v365;
                bool v369;
                v369 = v327[v368];
                int v370;
                if (v369){
                    v370 = 1l;
                } else {
                    v370 = 0l;
                }
                assert("Tensor range check" && 0 <= v363 && v363 < 1l);
                assert("Tensor range check" && 0 <= v365 && v365 < 4l);
                v362[v368] = v370;
                v365 += 1l ;
            }
            v363 += 1l ;
        }
        int v371;
        v371 = 0l;
        int v372;
        v372 = 0l;
        while (while_method_6(v372)){
            int v374;
            v374 = 0l;
            while (while_method_0(v374)){
                assert("Tensor range check" && 0 <= v372 && v372 < 1l);
                assert("Tensor range check" && 0 <= v374 && v374 < 4l);
                int v376;
                v376 = 4l * v372;
                int v377;
                v377 = v376 + v374;
                int v378;
                v378 = v362[v377];
                int v379;
                v379 = v371 + v378;
                v371 = v379;
                v374 += 1l ;
            }
            v372 += 1l ;
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
        float v386[4l];
        int v387;
        v387 = 0l;
        while (while_method_6(v387)){
            int v389;
            v389 = 0l;
            while (while_method_0(v389)){
                assert("Tensor range check" && 0 <= v387 && v387 < 1l);
                assert("Tensor range check" && 0 <= v389 && v389 < 4l);
                int v391;
                v391 = 4l * v387;
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
                assert("Tensor range check" && 0 <= v387 && v387 < 1l);
                assert("Tensor range check" && 0 <= v389 && v389 < 4l);
                v386[v392] = v400;
                v389 += 1l ;
            }
            v387 += 1l ;
        }
        float v401; int v402;
        Tuple2 tmp12 = Tuple2{0.0f, 2147483647l};
        v401 = tmp12.v0; v402 = tmp12.v1;
        int v403;
        v403 = 0l;
        while (while_method_6(v403)){
            int v405;
            v405 = 0l;
            while (while_method_0(v405)){
                assert("Tensor range check" && 0 <= v403 && v403 < 1l);
                assert("Tensor range check" && 0 <= v405 && v405 < 4l);
                int v407;
                v407 = 4l * v403;
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
                v405 += 1l ;
            }
            v403 += 1l ;
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
        v423 = v422 == 2147483647l;
        bool v424;
        v424 = v423 != true;
        bool v425;
        v425 = v424 == false;
        if (v425){
            assert("Expected a valid action id in get_action." && v424);
        } else {
        }
        int v427;
        v427 = 0l;
        while (while_method_6(v427)){
            assert("Tensor range check" && 0 <= v427 && v427 < 1l);
            assert("Tensor range check" && 0 <= v427 && v427 < 1l);
            v427 += 1l ;
        }
        assert("Tensor range check" && 0 <= v70 && v70 < 256l);
        v47[v70] = v421;
        v49[v70] = v323;
        v59 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v51 && v51 < 256l);
    float v429;
    v429 = v47[v51];
    int v430;
    v430 = v49[v51];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return Tuple2{v429, v430};
}
__device__ float method_7(int * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int v8, int v9, int v10){
    assert("Tensor range check" && 0 <= v9 && v9 < 4l);
    int v11;
    v11 = 16384l * v9;
    assert("Tensor range check" && 0 <= v8 && v8 < 4096l);
    int v12;
    v12 = 4l * v8;
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
    v25 = v24 <= 81920ull;
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
    assert("Tensor range check" && 0 <= v38 && v38 < 256l);
    v32[v38] = v10;
    v34[v38] = v14;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v39;
    v39 = 0l <= v38;
    bool v40;
    v40 = v39 == false;
    if (v40){
        assert("The index needs to be zero or positive." && v39);
    } else {
    }
    int v42;
    v42 = v38 % 1l;
    bool v43;
    v43 = v38 < 256l;
    bool v44;
    v44 = v43 == false;
    if (v44){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v43);
    } else {
    }
    assert("Tensor range check" && 0 <= v38 && v38 < 256l);
    int v46;
    v46 = 0l;
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
        v51 = 0l <= v46;
        bool v53;
        if (v51){
            bool v52;
            v52 = v46 < 1l;
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
        v56 = v46 * 256l;
        int v57;
        v57 = v56 + v38;
        assert("Tensor range check" && 0 <= v46 && v46 < 1l);
        int v58;
        v58 = 256l * v46;
        int v59;
        v59 = v58 + v38;
        int v60;
        v60 = v32[v59];
        float * v61;
        v61 = v34[v59];
        int v62;
        v62 = blockIdx.x;
        int v63;
        v63 = v62 * 256l;
        int v64;
        v64 = v63 + v57;
        assert("Tensor range check" && 0 <= v42 && v42 < 1l);
        int v65;
        v65 = 4l * v42;
        float v66[4l];
        int v67[4l];
        int v68;
        v68 = 0l;
        while (while_method_6(v68)){
            assert("Tensor range check" && 0 <= v68 && v68 < 1l);
            int v70;
            v70 = 4l * v68;
            assert("Tensor range check" && 0 <= v68 && v68 < 1l);
            int v71;
            v71 = v70 + v65;
            int4* v72;
            v72 = reinterpret_cast<int4*>(v61 + v71);
            int4* v73;
            v73 = reinterpret_cast<int4*>(v66 + v70);
            assert("Pointer alignment check" && (unsigned long long)(v72) % 4l == 0 && (unsigned long long)(v73) % 4l == 0);
            *v73 = *v72;
            v68 += 1l ;
        }
        int v74;
        v74 = 0l;
        while (while_method_6(v74)){
            int v76;
            v76 = 0l;
            while (while_method_0(v76)){
                bool v78;
                v78 = 0l <= v76;
                bool v80;
                if (v78){
                    bool v79;
                    v79 = v76 < 4l;
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
                v83 = 0l <= v42;
                bool v85;
                if (v83){
                    bool v84;
                    v84 = v42 < 1l;
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
                v88 = v42 * 4l;
                int v89;
                v89 = v76 + v88;
                bool v90;
                v90 = 0l <= v74;
                bool v92;
                if (v90){
                    bool v91;
                    v91 = v74 < 1l;
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
                v95 = v74 * 4l;
                int v96;
                v96 = v89 + v95;
                assert("Tensor range check" && 0 <= v74 && v74 < 1l);
                assert("Tensor range check" && 0 <= v76 && v76 < 4l);
                int v97;
                v97 = 4l * v74;
                int v98;
                v98 = v97 + v76;
                v67[v98] = v96;
                v76 += 1l ;
            }
            v74 += 1l ;
        }
        bool v99[4l];
        int v100;
        v100 = 0l;
        while (while_method_6(v100)){
            int v102;
            v102 = 0l;
            while (while_method_0(v102)){
                assert("Tensor range check" && 0 <= v100 && v100 < 1l);
                assert("Tensor range check" && 0 <= v102 && v102 < 4l);
                int v104;
                v104 = 4l * v100;
                int v105;
                v105 = v104 + v102;
                float v106;
                v106 = v66[v105];
                int v107;
                v107 = v67[v105];
                bool v108;
                v108 = v107 < 3l;
                assert("Tensor range check" && 0 <= v100 && v100 < 1l);
                assert("Tensor range check" && 0 <= v102 && v102 < 4l);
                v99[v105] = v108;
                v102 += 1l ;
            }
            v100 += 1l ;
        }
        float v109[4l];
        int v110;
        v110 = 0l;
        while (while_method_6(v110)){
            int v112;
            v112 = 0l;
            while (while_method_0(v112)){
                assert("Tensor range check" && 0 <= v110 && v110 < 1l);
                assert("Tensor range check" && 0 <= v112 && v112 < 4l);
                int v114;
                v114 = 4l * v110;
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
                assert("Tensor range check" && 0 <= v110 && v110 < 1l);
                assert("Tensor range check" && 0 <= v112 && v112 < 4l);
                v109[v115] = v120;
                v112 += 1l ;
            }
            v110 += 1l ;
        }
        float v121;
        v121 = 0.0f;
        int v122;
        v122 = 0l;
        while (while_method_6(v122)){
            int v124;
            v124 = 0l;
            while (while_method_0(v124)){
                assert("Tensor range check" && 0 <= v122 && v122 < 1l);
                assert("Tensor range check" && 0 <= v124 && v124 < 4l);
                int v126;
                v126 = 4l * v122;
                int v127;
                v127 = v126 + v124;
                float v128;
                v128 = v109[v127];
                float v129;
                v129 = v121 + v128;
                v121 = v129;
                v124 += 1l ;
            }
            v122 += 1l ;
        }
        auto v130 = cooperative_groups::coalesced_threads();
        int v131;
        v131 = threadIdx.x;
        auto v132 = cooperative_groups::labeled_partition(v130,v131);
        Closure1 v133{};
        float v134;
        v134 = cooperative_groups::reduce(v132, v121, v133);
        int v135[4l];
        int v136;
        v136 = 0l;
        while (while_method_6(v136)){
            int v138;
            v138 = 0l;
            while (while_method_0(v138)){
                assert("Tensor range check" && 0 <= v136 && v136 < 1l);
                assert("Tensor range check" && 0 <= v138 && v138 < 4l);
                int v140;
                v140 = 4l * v136;
                int v141;
                v141 = v140 + v138;
                bool v142;
                v142 = v99[v141];
                int v143;
                if (v142){
                    v143 = 1l;
                } else {
                    v143 = 0l;
                }
                assert("Tensor range check" && 0 <= v136 && v136 < 1l);
                assert("Tensor range check" && 0 <= v138 && v138 < 4l);
                v135[v141] = v143;
                v138 += 1l ;
            }
            v136 += 1l ;
        }
        int v144;
        v144 = 0l;
        int v145;
        v145 = 0l;
        while (while_method_6(v145)){
            int v147;
            v147 = 0l;
            while (while_method_0(v147)){
                assert("Tensor range check" && 0 <= v145 && v145 < 1l);
                assert("Tensor range check" && 0 <= v147 && v147 < 4l);
                int v149;
                v149 = 4l * v145;
                int v150;
                v150 = v149 + v147;
                int v151;
                v151 = v135[v150];
                int v152;
                v152 = v144 + v151;
                v144 = v152;
                v147 += 1l ;
            }
            v145 += 1l ;
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
        float v160[4l];
        int v161;
        v161 = 0l;
        while (while_method_6(v161)){
            int v163;
            v163 = 0l;
            while (while_method_0(v163)){
                assert("Tensor range check" && 0 <= v161 && v161 < 1l);
                assert("Tensor range check" && 0 <= v163 && v163 < 4l);
                int v165;
                v165 = 4l * v161;
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
                assert("Tensor range check" && 0 <= v161 && v161 < 1l);
                assert("Tensor range check" && 0 <= v163 && v163 < 4l);
                v160[v166] = v174;
                v163 += 1l ;
            }
            v161 += 1l ;
        }
        float v175; int v176;
        Tuple2 tmp15 = Tuple2{0.0f, 2147483647l};
        v175 = tmp15.v0; v176 = tmp15.v1;
        int v177;
        v177 = 0l;
        while (while_method_6(v177)){
            int v179;
            v179 = 0l;
            while (while_method_0(v179)){
                assert("Tensor range check" && 0 <= v177 && v177 < 1l);
                assert("Tensor range check" && 0 <= v179 && v179 < 4l);
                int v181;
                v181 = 4l * v177;
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
                v179 += 1l ;
            }
            v177 += 1l ;
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
        v197 = v196 == 2147483647l;
        bool v198;
        v198 = v197 != true;
        bool v199;
        v199 = v198 == false;
        if (v199){
            assert("Expected a valid action id in get_action." && v198);
        } else {
        }
        int v201;
        v201 = 0l;
        while (while_method_6(v201)){
            assert("Tensor range check" && 0 <= v201 && v201 < 1l);
            assert("Tensor range check" && 0 <= v201 && v201 < 1l);
            v201 += 1l ;
        }
        assert("Tensor range check" && 0 <= v57 && v57 < 256l);
        v36[v57] = v195;
        v46 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v38 && v38 < 256l);
    float v203;
    v203 = v36[v38];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return v203;
}
__device__ int int_range_8(int v0, int v1, curandStatePhilox4_32_10_t & v2){
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
__device__ int tag_10(Union2 v0){
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
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ bool is_pair_11(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
__device__ Tuple1 order_12(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple1{v1, v0};
    } else {
        return Tuple1{v0, v1};
    }
}
__device__ Union11 compare_hands_9(Union5 v0, bool v1, static_array<Union2,2l> v2, int v3, static_array<int,2l> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            __trap();
            break;
        }
        case 1: { // Some
            Union2 v7 = v0.case1.v0;
            int v8;
            v8 = tag_10(v7);
            Union2 v9;
            v9 = v2[0l];
            int v11;
            v11 = tag_10(v9);
            Union2 v12;
            v12 = v2[1l];
            int v14;
            v14 = tag_10(v12);
            bool v15;
            v15 = is_pair_11(v8, v11);
            bool v16;
            v16 = is_pair_11(v8, v14);
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
                    Tuple1 tmp24 = order_12(v8, v11);
                    v27 = tmp24.v0; v28 = tmp24.v1;
                    int v29; int v30;
                    Tuple1 tmp25 = order_12(v8, v14);
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
__device__ void method_0(unsigned char * v0, unsigned char * v1, sptr<Mut0> v2, Union6 v3){
    static_array_list<Union1,32l> & v4 = v2.base->v3;
    Union7 v5;
    v5 = Union7{Union7_1{v3}};
    Union7 v6;
    v6 = v5;
    while (while_method_2(v6)){
        Union7 v709;
        switch (v6.tag) {
            case 0: { // None
                v709 = Union7{Union7_0{}};
                break;
            }
            case 1: { // Some
                Union6 v8 = v6.case1.v0;
                switch (v8.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v654 = v8.case0.v0; bool v655 = v8.case0.v1; static_array<Union2,2l> v656 = v8.case0.v2; int v657 = v8.case0.v3; static_array<int,2l> v658 = v8.case0.v4; int v659 = v8.case0.v5;
                        curandStatePhilox4_32_10_t & v660 = v2.base->v6;
                        curandStatePhilox4_32_10_t & v661 = v660;
                        unsigned int & v662 = v2.base->v1;
                        Union2 v663; unsigned int v664;
                        Tuple0 tmp0 = draw_card_1(v661, v662);
                        v663 = tmp0.v0; v664 = tmp0.v1;
                        v2.base->v1 = v664;
                        Union1 v665;
                        v665 = Union1{Union1_0{v663}};
                        v4.push(v665);
                        int v666;
                        v666 = 2l;
                        int v667; int v668;
                        Tuple1 tmp1 = Tuple1{0l, 0l};
                        v667 = tmp1.v0; v668 = tmp1.v1;
                        while (while_method_3(v667)){
                            int v670;
                            v670 = v658[v667];
                            bool v672;
                            v672 = v668 >= v670;
                            int v673;
                            if (v672){
                                v673 = v668;
                            } else {
                                v673 = v670;
                            }
                            v668 = v673;
                            v667 += 1l ;
                        }
                        static_array<int,2l> v674;
                        int v676;
                        v676 = 0l;
                        while (while_method_3(v676)){
                            v674[v676] = v668;
                            v676 += 1l ;
                        }
                        Union5 v678;
                        v678 = Union5{Union5_1{v663}};
                        Union6 v679;
                        v679 = Union6{Union6_2{v678, true, v656, 0l, v674, v666}};
                        v709 = Union7{Union7_1{v679}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v681 = v2.base->v6;
                        curandStatePhilox4_32_10_t & v682 = v681;
                        unsigned int & v683 = v2.base->v1;
                        Union2 v684; unsigned int v685;
                        Tuple0 tmp2 = draw_card_1(v682, v683);
                        v684 = tmp2.v0; v685 = tmp2.v1;
                        v2.base->v1 = v685;
                        curandStatePhilox4_32_10_t & v686 = v2.base->v6;
                        curandStatePhilox4_32_10_t & v687 = v686;
                        unsigned int & v688 = v2.base->v1;
                        Union2 v689; unsigned int v690;
                        Tuple0 tmp3 = draw_card_1(v687, v688);
                        v689 = tmp3.v0; v690 = tmp3.v1;
                        v2.base->v1 = v690;
                        Union1 v691;
                        v691 = Union1{Union1_2{0l, v684}};
                        v4.push(v691);
                        Union1 v692;
                        v692 = Union1{Union1_2{1l, v689}};
                        v4.push(v692);
                        int v693;
                        v693 = 2l;
                        static_array<int,2l> v694;
                        v694[0l] = 1l;
                        v694[1l] = 1l;
                        static_array<Union2,2l> v696;
                        v696[0l] = v684;
                        v696[1l] = v689;
                        Union5 v698;
                        v698 = Union5{Union5_0{}};
                        Union6 v699;
                        v699 = Union6{Union6_2{v698, true, v696, 0l, v694, v693}};
                        v709 = Union7{Union7_1{v699}};
                        break;
                    }
                    case 2: { // Round
                        Union5 v51 = v8.case2.v0; bool v52 = v8.case2.v1; static_array<Union2,2l> v53 = v8.case2.v2; int v54 = v8.case2.v3; static_array<int,2l> v55 = v8.case2.v4; int v56 = v8.case2.v5;
                        static_array<Union0,2l> & v57 = v2.base->v4;
                        Union0 v58;
                        v58 = v57[v54];
                        Union3 v464; Union5 v465; bool v466; static_array<Union2,2l> v467; int v468; static_array<int,2l> v469; int v470;
                        switch (v58.tag) {
                            case 0: { // T_Computer
                                static_array_list<Union1,32l> & v60 = v2.base->v3;
                                Union4 & v61 = v2.base->v0;
                                Union4 & v62 = v61;
                                Union4 v63;
                                v63 = Union4{Union4_1{v51, v52, v53, v54, v55, v56}};
                                v62 = v63;
                                curandStatePhilox4_32_10_t & v64 = v2.base->v6;
                                curandStatePhilox4_32_10_t & v65 = v64;
                                unsigned int * v66;
                                v66 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
                                float * v68;
                                v68 = reinterpret_cast<float *>(&v0[0ull]);
                                int v70;
                                v70 = threadIdx.x;
                                int v71;
                                v71 = blockIdx.x;
                                int v72;
                                v72 = v71 * 256l;
                                int v73;
                                v73 = v70 + v72;
                                unsigned long long v74;
                                v74 = (unsigned long long)v73;
                                curandStatePhilox4_32_10_t v75;
                                curand_init(12344321ull,v74,0ull,&v75);
                                float * v76;
                                v76 = reinterpret_cast<float *>(&v0[0ull]);
                                int v78;
                                v78 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v78 && v78 < 24l);
                                int v79;
                                v79 = 32768l * v78;
                                int v80;
                                v80 = threadIdx.x;
                                int v81;
                                v81 = blockIdx.x;
                                int v82;
                                v82 = v81 * 256l;
                                int v83;
                                v83 = v80 + v82;
                                unsigned long long v84;
                                v84 = (unsigned long long)v83;
                                curandStatePhilox4_32_10_t v85;
                                curand_init(12344321ull,v84,0ull,&v85);
                                int v86;
                                v86 = threadIdx.x;
                                int v87;
                                v87 = v86;
                                while (while_method_4(v87)){
                                    bool v89;
                                    v89 = 0l <= v87;
                                    bool v90;
                                    v90 = v89 == false;
                                    if (v90){
                                        assert("The index needs to be zero or positive." && v89);
                                    } else {
                                    }
                                    int v92;
                                    v92 = v87 % 128l;
                                    int v93;
                                    v93 = v87 / 128l;
                                    bool v94;
                                    v94 = v93 < 256l;
                                    bool v95;
                                    v95 = v94 == false;
                                    if (v95){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v94);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v93 && v93 < 256l);
                                    assert("Tensor range check" && 0 <= v92 && v92 < 128l);
                                    int v97;
                                    v97 = v92 + v79;
                                    int v98;
                                    v98 = 128l * v93;
                                    int v99;
                                    v99 = v98 + v97;
                                    v76[v99] = 0.0f;
                                    v87 += 256l ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                int v100;
                                v100 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v100 && v100 < 256l);
                                int v101;
                                v101 = 128l * v100;
                                int v102;
                                v102 = v101 + v79;
                                static_array_list<Union8,10l> v103;
                                v103 = static_array_list<Union8,10l>{};
                                int v105;
                                v105 = v60.length;
                                int v106;
                                v106 = 0l;
                                while (while_method_5(v105, v106)){
                                    Union1 v108;
                                    v108 = v60[v106];
                                    Union9 v127;
                                    switch (v108.tag) {
                                        case 0: { // CommunityCardIs
                                            Union2 v117 = v108.case0.v0;
                                            Union8 v118;
                                            v118 = Union8{Union8_1{v117}};
                                            v127 = Union9{Union9_1{v118}};
                                            break;
                                        }
                                        case 1: { // PlayerAction
                                            int v120 = v108.case1.v0; Union3 v121 = v108.case1.v1;
                                            Union8 v122;
                                            v122 = Union8{Union8_0{v121}};
                                            v127 = Union9{Union9_1{v122}};
                                            break;
                                        }
                                        case 2: { // PlayerGotCard
                                            int v110 = v108.case2.v0; Union2 v111 = v108.case2.v1;
                                            bool v112;
                                            v112 = v110 == v54;
                                            if (v112){
                                                Union8 v113;
                                                v113 = Union8{Union8_1{v111}};
                                                v127 = Union9{Union9_1{v113}};
                                            } else {
                                                v127 = Union9{Union9_0{}};
                                            }
                                            break;
                                        }
                                        default: {
                                            v127 = Union9{Union9_0{}};
                                        }
                                    }
                                    switch (v127.tag) {
                                        case 0: { // None
                                            break;
                                        }
                                        case 1: { // Some
                                            Union8 v128 = v127.case1.v0;
                                            v103.push(v128);
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false); __trap();
                                        }
                                    }
                                    v106 += 1l ;
                                }
                                float * v129;
                                v129 = v76+v102;
                                int v131;
                                v131 = v103.length;
                                bool v132;
                                v132 = v131 == 0l;
                                if (v132){
                                    v129[0l] = 1.0f;
                                } else {
                                }
                                int v133;
                                v133 = v103.length;
                                int v134;
                                v134 = 0l;
                                while (while_method_5(v133, v134)){
                                    Union8 v136;
                                    v136 = v103[v134];
                                    int v138;
                                    v138 = v134 * 6l;
                                    int v139;
                                    v139 = 1l + v138;
                                    switch (v136.tag) {
                                        case 0: { // C1of2
                                            Union3 v140 = v136.case0.v0;
                                            switch (v140.tag) {
                                                case 0: { // Call
                                                    v129[v139] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // Fold
                                                    int v141;
                                                    v141 = v139 + 1l;
                                                    v129[v141] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // Raise
                                                    int v142;
                                                    v142 = v139 + 2l;
                                                    v129[v142] = 1.0f;
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false); __trap();
                                                }
                                            }
                                            break;
                                        }
                                        case 1: { // C2of2
                                            Union2 v143 = v136.case1.v0;
                                            int v144;
                                            v144 = v139 + 3l;
                                            switch (v143.tag) {
                                                case 0: { // Jack
                                                    v129[v144] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // King
                                                    int v145;
                                                    v145 = v144 + 1l;
                                                    v129[v145] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // Queen
                                                    int v146;
                                                    v146 = v144 + 2l;
                                                    v129[v146] = 1.0f;
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
                                    v134 += 1l ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                noinline_run_nn_model_3(v0, v1, v85);
                                int * v147;
                                v147 = reinterpret_cast<int *>(&v1[262144ull]);
                                float * v149;
                                v149 = reinterpret_cast<float *>(&v1[262160ull]);
                                float * v151;
                                v151 = reinterpret_cast<float *>(&v1[524304ull]);
                                float * v153;
                                v153 = reinterpret_cast<float *>(&v1[786448ull]);
                                float * v155;
                                v155 = reinterpret_cast<float *>(&v1[1048592ull]);
                                float * v157;
                                v157 = reinterpret_cast<float *>(&v1[1310736ull]);
                                float * v159;
                                v159 = reinterpret_cast<float *>(&v1[1572880ull]);
                                float * v161;
                                v161 = reinterpret_cast<float *>(&v1[1835024ull]);
                                int v163;
                                v163 = v147[0l];
                                unsigned int * v164;
                                v164 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
                                int v166;
                                v166 = blockIdx.x;
                                int v167;
                                v167 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v163 && v163 < 4l);
                                assert("Tensor range check" && 0 <= v166 && v166 < 24l);
                                assert("Tensor range check" && 0 <= v167 && v167 < 256l);
                                int v168;
                                v168 = 256l * v166;
                                int v169;
                                v169 = v168 + v167;
                                int v170;
                                v170 = 6144l * v163;
                                int v171;
                                v171 = v170 + v169;
                                unsigned int v172;
                                v172 = v164[v171];
                                int v173;
                                v173 = (int)v172;
                                float v174; int v175;
                                Tuple2 tmp14 = method_6(v65, v147, v149, v151, v153, v155, v157, v159, v161, v173, v163);
                                v174 = tmp14.v0; v175 = tmp14.v1;
                                extern __shared__ unsigned char v176[];
                                float * v177;
                                v177 = reinterpret_cast<float *>(&v176[0ull]);
                                int * v179;
                                v179 = reinterpret_cast<int *>(&v176[16ull]);
                                int v181;
                                v181 = threadIdx.x;
                                bool v182;
                                v182 = v181 == 0l;
                                if (v182){
                                    v177[0l] = v174;
                                    v179[0l] = v175;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                float v183;
                                v183 = v177[0l];
                                int v184;
                                v184 = v179[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                double * v185;
                                v185 = reinterpret_cast<double *>(&v1[2097168ull]);
                                double * v187;
                                v187 = reinterpret_cast<double *>(&v1[2490384ull]);
                                int * v189;
                                v189 = reinterpret_cast<int *>(&v1[2883600ull]);
                                int * v191;
                                v191 = reinterpret_cast<int *>(&v0[6389760ull]);
                                float * v193;
                                v193 = reinterpret_cast<float *>(&v0[7962624ull]);
                                int * v195;
                                v195 = reinterpret_cast<int *>(&v0[9535488ull]);
                                int * v197;
                                v197 = reinterpret_cast<int *>(&v0[11108352ull]);
                                double * v199;
                                v199 = reinterpret_cast<double *>(&v0[12681216ull]);
                                double * v201;
                                v201 = reinterpret_cast<double *>(&v0[18972672ull]);
                                int v203;
                                v203 = threadIdx.x;
                                int v204;
                                v204 = blockIdx.x;
                                int v205;
                                v205 = v204 * 256l;
                                int v206;
                                v206 = v203 + v205;
                                int v207;
                                v207 = 0l;
                                while (while_method_0(v207)){
                                    unsigned int * v209;
                                    v209 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
                                    int v211;
                                    v211 = blockIdx.x;
                                    int v212;
                                    v212 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v207 && v207 < 4l);
                                    assert("Tensor range check" && 0 <= v211 && v211 < 24l);
                                    assert("Tensor range check" && 0 <= v212 && v212 < 256l);
                                    int v213;
                                    v213 = 256l * v211;
                                    int v214;
                                    v214 = v213 + v212;
                                    int v215;
                                    v215 = 6144l * v207;
                                    int v216;
                                    v216 = v215 + v214;
                                    unsigned int v217;
                                    v217 = v209[v216];
                                    int v218;
                                    v218 = (int)v217;
                                    float v219;
                                    v219 = method_7(v147, v149, v151, v153, v155, v157, v159, v161, v218, v207, v184);
                                    assert("Tensor range check" && 0 <= v207 && v207 < 4l);
                                    assert("Tensor range check" && 0 <= v206 && v206 < 6144l);
                                    int v220;
                                    v220 = v215 + v206;
                                    int v221;
                                    v221 = v189[v220];
                                    int v222;
                                    v222 = v221 + 1l;
                                    assert("Tensor range check" && 0 <= v207 && v207 < 4l);
                                    assert("Tensor range check" && 0 <= v206 && v206 < 6144l);
                                    v189[v220] = v222;
                                    assert("Tensor range check" && 0 <= v207 && v207 < 4l);
                                    assert("Tensor range check" && 0 <= v221 && v221 < 16l);
                                    assert("Tensor range check" && 0 <= v206 && v206 < 6144l);
                                    int v223;
                                    v223 = 6144l * v221;
                                    int v224;
                                    v224 = v223 + v206;
                                    int v225;
                                    v225 = 98304l * v207;
                                    int v226;
                                    v226 = v225 + v224;
                                    v191[v226] = v184;
                                    v193[v226] = v183;
                                    v195[v226] = v54;
                                    v197[v226] = v218;
                                    assert("Tensor range check" && 0 <= v207 && v207 < 4l);
                                    int v227;
                                    v227 = 12288l * v207;
                                    assert("Tensor range check" && 0 <= v206 && v206 < 6144l);
                                    int v228;
                                    v228 = 2l * v206;
                                    int v229;
                                    v229 = v228 + v227;
                                    assert("Tensor range check" && 0 <= v207 && v207 < 4l);
                                    int v230;
                                    v230 = 196608l * v207;
                                    assert("Tensor range check" && 0 <= v221 && v221 < 16l);
                                    int v231;
                                    v231 = 12288l * v221;
                                    int v232;
                                    v232 = v231 + v230;
                                    assert("Tensor range check" && 0 <= v206 && v206 < 6144l);
                                    int v233;
                                    v233 = v228 + v232;
                                    double * v234;
                                    v234 = v185+v229;
                                    double * v236;
                                    v236 = v187+v229;
                                    double * v238;
                                    v238 = v199+v233;
                                    double * v240;
                                    v240 = v201+v233;
                                    int v242;
                                    v242 = sizeof(double *);
                                    unsigned long long v243;
                                    v243 = (unsigned long long)v242;
                                    unsigned long long v244;
                                    v244 = 256ull * v243;
                                    unsigned long long v245;
                                    v245 = v244 + 16ull;
                                    unsigned long long v246;
                                    v246 = v245 - 1ull;
                                    unsigned long long v247;
                                    v247 = v246 % 16ull;
                                    unsigned long long v248;
                                    v248 = v246 - v247;
                                    unsigned long long v249;
                                    v249 = v248 + v244;
                                    unsigned long long v250;
                                    v250 = v249 + 16ull;
                                    unsigned long long v251;
                                    v251 = v250 - 1ull;
                                    unsigned long long v252;
                                    v252 = v251 % 16ull;
                                    unsigned long long v253;
                                    v253 = v251 - v252;
                                    unsigned long long v254;
                                    v254 = v253 + v244;
                                    unsigned long long v255;
                                    v255 = v254 + 16ull;
                                    unsigned long long v256;
                                    v256 = v255 - 1ull;
                                    unsigned long long v257;
                                    v257 = v256 % 16ull;
                                    unsigned long long v258;
                                    v258 = v256 - v257;
                                    unsigned long long v259;
                                    v259 = v258 + v244;
                                    bool v260;
                                    v260 = v259 <= 81920ull;
                                    bool v261;
                                    v261 = v260 == false;
                                    if (v261){
                                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v260);
                                    } else {
                                    }
                                    extern __shared__ unsigned char v263[];
                                    bool v264;
                                    v264 = v259 <= v259;
                                    bool v265;
                                    v265 = v264 == false;
                                    if (v265){
                                        assert("The length of the partition has to be less than or equal to the length of the base array." && v264);
                                    } else {
                                    }
                                    double * * v267;
                                    v267 = reinterpret_cast<double * *>(&v263[0ull]);
                                    double * * v269;
                                    v269 = reinterpret_cast<double * *>(&v263[v248]);
                                    double * * v271;
                                    v271 = reinterpret_cast<double * *>(&v263[v253]);
                                    double * * v273;
                                    v273 = reinterpret_cast<double * *>(&v263[v258]);
                                    int v275;
                                    v275 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v275 && v275 < 256l);
                                    v267[v275] = v234;
                                    v269[v275] = v236;
                                    v271[v275] = v238;
                                    v273[v275] = v240;
                                    asm("barrier.cta.sync %0;" :: "r"(0l));
                                    bool v276;
                                    v276 = 0l <= v275;
                                    bool v277;
                                    v277 = v276 == false;
                                    if (v277){
                                        assert("The index needs to be zero or positive." && v276);
                                    } else {
                                    }
                                    int v279;
                                    v279 = v275 % 1l;
                                    bool v280;
                                    v280 = v275 < 256l;
                                    bool v281;
                                    v281 = v280 == false;
                                    if (v281){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v280);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v275 && v275 < 256l);
                                    int v283;
                                    v283 = 0l;
                                    while (while_method_6(v283)){
                                        bool v285;
                                        v285 = v276 && v280;
                                        bool v286;
                                        v286 = v285 == false;
                                        if (v286){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v285);
                                        } else {
                                        }
                                        bool v288;
                                        v288 = 0l <= v283;
                                        bool v290;
                                        if (v288){
                                            bool v289;
                                            v289 = v283 < 1l;
                                            v290 = v289;
                                        } else {
                                            v290 = false;
                                        }
                                        bool v291;
                                        v291 = v290 == false;
                                        if (v291){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v290);
                                        } else {
                                        }
                                        int v293;
                                        v293 = v283 * 256l;
                                        int v294;
                                        v294 = v293 + v275;
                                        assert("Tensor range check" && 0 <= v283 && v283 < 1l);
                                        int v295;
                                        v295 = 256l * v283;
                                        int v296;
                                        v296 = v295 + v275;
                                        double * v297;
                                        v297 = v267[v296];
                                        double * v298;
                                        v298 = v269[v296];
                                        double * v299;
                                        v299 = v271[v296];
                                        double * v300;
                                        v300 = v273[v296];
                                        int v301;
                                        v301 = blockIdx.x;
                                        int v302;
                                        v302 = v301 * 256l;
                                        int v303;
                                        v303 = v302 + v294;
                                        assert("Tensor range check" && 0 <= v279 && v279 < 1l);
                                        int v304;
                                        v304 = 2l * v279;
                                        double v305[2l];
                                        double v306[2l];
                                        int v307[2l];
                                        int v308;
                                        v308 = 0l;
                                        while (while_method_6(v308)){
                                            assert("Tensor range check" && 0 <= v308 && v308 < 1l);
                                            int v310;
                                            v310 = 2l * v308;
                                            assert("Tensor range check" && 0 <= v308 && v308 < 1l);
                                            int v311;
                                            v311 = v310 + v304;
                                            int4* v312;
                                            v312 = reinterpret_cast<int4*>(v297 + v311);
                                            int4* v313;
                                            v313 = reinterpret_cast<int4*>(v305 + v310);
                                            assert("Pointer alignment check" && (unsigned long long)(v312) % 2l == 0 && (unsigned long long)(v313) % 2l == 0);
                                            *v313 = *v312;
                                            int4* v314;
                                            v314 = reinterpret_cast<int4*>(v298 + v311);
                                            int4* v315;
                                            v315 = reinterpret_cast<int4*>(v306 + v310);
                                            assert("Pointer alignment check" && (unsigned long long)(v314) % 2l == 0 && (unsigned long long)(v315) % 2l == 0);
                                            *v315 = *v314;
                                            v308 += 1l ;
                                        }
                                        int v316;
                                        v316 = 0l;
                                        while (while_method_6(v316)){
                                            int v318;
                                            v318 = 0l;
                                            while (while_method_3(v318)){
                                                bool v320;
                                                v320 = 0l <= v318;
                                                bool v322;
                                                if (v320){
                                                    bool v321;
                                                    v321 = v318 < 2l;
                                                    v322 = v321;
                                                } else {
                                                    v322 = false;
                                                }
                                                bool v323;
                                                v323 = v322 == false;
                                                if (v323){
                                                    assert("The indices should be inside the range of the dimension." && v322);
                                                } else {
                                                }
                                                bool v325;
                                                v325 = 0l <= v279;
                                                bool v327;
                                                if (v325){
                                                    bool v326;
                                                    v326 = v279 < 1l;
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
                                                int v330;
                                                v330 = v279 * 2l;
                                                int v331;
                                                v331 = v318 + v330;
                                                bool v332;
                                                v332 = 0l <= v316;
                                                bool v334;
                                                if (v332){
                                                    bool v333;
                                                    v333 = v316 < 1l;
                                                    v334 = v333;
                                                } else {
                                                    v334 = false;
                                                }
                                                bool v335;
                                                v335 = v334 == false;
                                                if (v335){
                                                    assert("The indices should be inside the range of the dimension." && v334);
                                                } else {
                                                }
                                                int v337;
                                                v337 = v316 * 2l;
                                                int v338;
                                                v338 = v331 + v337;
                                                assert("Tensor range check" && 0 <= v316 && v316 < 1l);
                                                assert("Tensor range check" && 0 <= v318 && v318 < 2l);
                                                int v339;
                                                v339 = 2l * v316;
                                                int v340;
                                                v340 = v339 + v318;
                                                v307[v340] = v338;
                                                v318 += 1l ;
                                            }
                                            v316 += 1l ;
                                        }
                                        int v341;
                                        v341 = 0l;
                                        while (while_method_6(v341)){
                                            assert("Tensor range check" && 0 <= v341 && v341 < 1l);
                                            int v343;
                                            v343 = 2l * v341;
                                            int v344;
                                            v344 = v343 + v304;
                                            assert("Tensor range check" && 0 <= v341 && v341 < 1l);
                                            int4* v345;
                                            v345 = reinterpret_cast<int4*>(v305 + v343);
                                            int4* v346;
                                            v346 = reinterpret_cast<int4*>(v299 + v344);
                                            assert("Pointer alignment check" && (unsigned long long)(v345) % 2l == 0 && (unsigned long long)(v346) % 2l == 0);
                                            *v346 = *v345;
                                            int4* v347;
                                            v347 = reinterpret_cast<int4*>(v306 + v343);
                                            int4* v348;
                                            v348 = reinterpret_cast<int4*>(v300 + v344);
                                            assert("Pointer alignment check" && (unsigned long long)(v347) % 2l == 0 && (unsigned long long)(v348) % 2l == 0);
                                            *v348 = *v347;
                                            v341 += 1l ;
                                        }
                                        assert("Tensor range check" && 0 <= v294 && v294 < 256l);
                                        v283 += 1l ;
                                    }
                                    asm("barrier.cta.sync %0;" :: "r"(0l));
                                    assert("Tensor range check" && 0 <= v275 && v275 < 256l);
                                    asm("barrier.cta.sync %0;" :: "r"(0l));
                                    double v349;
                                    v349 = (double)v183;
                                    double v350;
                                    v350 = log(v349);
                                    double v351;
                                    v351 = (double)v219;
                                    double v352;
                                    v352 = log(v351);
                                    assert("Tensor range check" && 0 <= v207 && v207 < 4l);
                                    assert("Tensor range check" && 0 <= v206 && v206 < 6144l);
                                    assert("Tensor range check" && 0 <= v54 && v54 < 2l);
                                    int v353;
                                    v353 = v228 + v54;
                                    int v354;
                                    v354 = v227 + v353;
                                    double v355;
                                    v355 = v185[v354];
                                    double v356;
                                    v356 = v187[v354];
                                    double v357;
                                    v357 = v352 + v355;
                                    double v358;
                                    v358 = v350 + v356;
                                    assert("Tensor range check" && 0 <= v207 && v207 < 4l);
                                    assert("Tensor range check" && 0 <= v206 && v206 < 6144l);
                                    assert("Tensor range check" && 0 <= v54 && v54 < 2l);
                                    v185[v354] = v357;
                                    v187[v354] = v358;
                                    v207 += 1l ;
                                }
                                bool v359;
                                v359 = 0l == v184;
                                Union10 v368;
                                if (v359){
                                    v368 = Union10{Union10_1{}};
                                } else {
                                    bool v361;
                                    v361 = 1l == v184;
                                    if (v361){
                                        v368 = Union10{Union10_0{}};
                                    } else {
                                        bool v363;
                                        v363 = 2l == v184;
                                        if (v363){
                                            v368 = Union10{Union10_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                Union4 v369 = v62;
                                Union5 v388; bool v389; static_array<Union2,2l> v390; int v391; static_array<int,2l> v392; int v393;
                                switch (v369.tag) {
                                    case 0: { // None
                                        printf("%s\n", "Expect the env to be backedu up in the reference.");
                                        __trap();
                                        break;
                                    }
                                    case 1: { // Some
                                        Union5 v370 = v369.case1.v0; bool v371 = v369.case1.v1; static_array<Union2,2l> v372 = v369.case1.v2; int v373 = v369.case1.v3; static_array<int,2l> v374 = v369.case1.v4; int v375 = v369.case1.v5;
                                        v388 = v370; v389 = v371; v390 = v372; v391 = v373; v392 = v374; v393 = v375;
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                Union3 v416;
                                switch (v368.tag) {
                                    case 0: { // AA_Call
                                        v416 = Union3{Union3_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v394;
                                        v394 = v392[0l];
                                        int v396; int v397;
                                        Tuple1 tmp17 = Tuple1{1l, v394};
                                        v396 = tmp17.v0; v397 = tmp17.v1;
                                        while (while_method_3(v396)){
                                            int v399;
                                            v399 = v392[v396];
                                            bool v401;
                                            v401 = v397 >= v399;
                                            int v402;
                                            if (v401){
                                                v402 = v397;
                                            } else {
                                                v402 = v399;
                                            }
                                            v397 = v402;
                                            v396 += 1l ;
                                        }
                                        int v403;
                                        v403 = v392[v391];
                                        bool v405;
                                        v405 = v403 == v397;
                                        if (v405){
                                            v416 = Union3{Union3_0{}};
                                        } else {
                                            v416 = Union3{Union3_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v410;
                                        v410 = v393 > 0l;
                                        if (v410){
                                            v416 = Union3{Union3_2{}};
                                        } else {
                                            v416 = Union3{Union3_0{}};
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                v464 = v416; v465 = v388; v466 = v389; v467 = v390; v468 = v391; v469 = v392; v470 = v393;
                                break;
                            }
                            case 1: { // T_Random
                                curandStatePhilox4_32_10_t & v417 = v2.base->v6;
                                curandStatePhilox4_32_10_t & v418 = v417;
                                static_array_list<Union3,3l> v419;
                                v419 = static_array_list<Union3,3l>{};
                                v419.unsafe_set_length(1l);
                                Union3 v421;
                                v421 = Union3{Union3_0{}};
                                v419[0l] = v421;
                                int v423;
                                v423 = v55[0l];
                                int v425;
                                v425 = v55[1l];
                                bool v427;
                                v427 = v423 == v425;
                                bool v428;
                                v428 = v427 != true;
                                if (v428){
                                    Union3 v429;
                                    v429 = Union3{Union3_1{}};
                                    v419.push(v429);
                                } else {
                                }
                                bool v430;
                                v430 = v56 > 0l;
                                if (v430){
                                    Union3 v431;
                                    v431 = Union3{Union3_2{}};
                                    v419.push(v431);
                                } else {
                                }
                                int v432;
                                v432 = v419.length;
                                int v433;
                                v433 = v432 - 1l;
                                int v434;
                                v434 = 0l;
                                while (while_method_5(v433, v434)){
                                    int v436;
                                    v436 = v419.length;
                                    int v437;
                                    v437 = int_range_8(v436, v434, v418);
                                    Union3 v438;
                                    v438 = v419[v434];
                                    Union3 v440;
                                    v440 = v419[v437];
                                    v419[v434] = v440;
                                    v419[v437] = v438;
                                    v434 += 1l ;
                                }
                                Union3 v442;
                                v442 = v419.pop();
                                int v443;
                                v443 = sizeof(Union3);
                                unsigned long long v444;
                                v444 = (unsigned long long)v443;
                                bool v445;
                                v445 = v444 <= 81920ull;
                                bool v446;
                                v446 = v445 == false;
                                if (v446){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v445);
                                } else {
                                }
                                extern __shared__ unsigned char v448[];
                                bool v449;
                                v449 = v444 <= v444;
                                bool v450;
                                v450 = v449 == false;
                                if (v450){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v449);
                                } else {
                                }
                                Union3 * v452;
                                v452 = reinterpret_cast<Union3 *>(&v448[0ull]);
                                int v454;
                                v454 = threadIdx.x;
                                bool v455;
                                v455 = v454 == 0l;
                                if (v455){
                                    v452[0l] = v442;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                Union3 v456;
                                v456 = v452[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                v464 = v456; v465 = v51; v466 = v52; v467 = v53; v468 = v54; v469 = v55; v470 = v56;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union1 v471;
                        v471 = Union1{Union1_1{v468, v464}};
                        v4.push(v471);
                        Union6 v557;
                        switch (v465.tag) {
                            case 0: { // None
                                switch (v464.tag) {
                                    case 0: { // Call
                                        if (v466){
                                            bool v521;
                                            v521 = v468 == 0l;
                                            int v522;
                                            if (v521){
                                                v522 = 1l;
                                            } else {
                                                v522 = 0l;
                                            }
                                            v557 = Union6{Union6_2{v465, false, v467, v522, v469, v470}};
                                        } else {
                                            v557 = Union6{Union6_0{v465, v466, v467, v468, v469, v470}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v557 = Union6{Union6_5{v465, v466, v467, v468, v469, v470}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v526;
                                        v526 = v470 > 0l;
                                        if (v526){
                                            bool v527;
                                            v527 = v468 == 0l;
                                            int v528;
                                            if (v527){
                                                v528 = 1l;
                                            } else {
                                                v528 = 0l;
                                            }
                                            int v529;
                                            v529 = -1l + v470;
                                            int v530; int v531;
                                            Tuple1 tmp18 = Tuple1{0l, 0l};
                                            v530 = tmp18.v0; v531 = tmp18.v1;
                                            while (while_method_3(v530)){
                                                int v533;
                                                v533 = v469[v530];
                                                bool v535;
                                                v535 = v531 >= v533;
                                                int v536;
                                                if (v535){
                                                    v536 = v531;
                                                } else {
                                                    v536 = v533;
                                                }
                                                v531 = v536;
                                                v530 += 1l ;
                                            }
                                            static_array<int,2l> v537;
                                            int v539;
                                            v539 = 0l;
                                            while (while_method_3(v539)){
                                                v537[v539] = v531;
                                                v539 += 1l ;
                                            }
                                            static_array<int,2l> v541;
                                            int v543;
                                            v543 = 0l;
                                            while (while_method_3(v543)){
                                                int v545;
                                                v545 = v537[v543];
                                                bool v547;
                                                v547 = v543 == v468;
                                                int v549;
                                                if (v547){
                                                    int v548;
                                                    v548 = v545 + 2l;
                                                    v549 = v548;
                                                } else {
                                                    v549 = v545;
                                                }
                                                v541[v543] = v549;
                                                v543 += 1l ;
                                            }
                                            v557 = Union6{Union6_2{v465, false, v467, v528, v541, v529}};
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
                                Union2 v472 = v465.case1.v0;
                                switch (v464.tag) {
                                    case 0: { // Call
                                        if (v466){
                                            bool v474;
                                            v474 = v468 == 0l;
                                            int v475;
                                            if (v474){
                                                v475 = 1l;
                                            } else {
                                                v475 = 0l;
                                            }
                                            v557 = Union6{Union6_2{v465, false, v467, v475, v469, v470}};
                                        } else {
                                            int v477; int v478;
                                            Tuple1 tmp19 = Tuple1{0l, 0l};
                                            v477 = tmp19.v0; v478 = tmp19.v1;
                                            while (while_method_3(v477)){
                                                int v480;
                                                v480 = v469[v477];
                                                bool v482;
                                                v482 = v478 >= v480;
                                                int v483;
                                                if (v482){
                                                    v483 = v478;
                                                } else {
                                                    v483 = v480;
                                                }
                                                v478 = v483;
                                                v477 += 1l ;
                                            }
                                            static_array<int,2l> v484;
                                            int v486;
                                            v486 = 0l;
                                            while (while_method_3(v486)){
                                                v484[v486] = v478;
                                                v486 += 1l ;
                                            }
                                            v557 = Union6{Union6_4{v465, v466, v467, v468, v484, v470}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v557 = Union6{Union6_5{v465, v466, v467, v468, v469, v470}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v490;
                                        v490 = v470 > 0l;
                                        if (v490){
                                            bool v491;
                                            v491 = v468 == 0l;
                                            int v492;
                                            if (v491){
                                                v492 = 1l;
                                            } else {
                                                v492 = 0l;
                                            }
                                            int v493;
                                            v493 = -1l + v470;
                                            int v494; int v495;
                                            Tuple1 tmp20 = Tuple1{0l, 0l};
                                            v494 = tmp20.v0; v495 = tmp20.v1;
                                            while (while_method_3(v494)){
                                                int v497;
                                                v497 = v469[v494];
                                                bool v499;
                                                v499 = v495 >= v497;
                                                int v500;
                                                if (v499){
                                                    v500 = v495;
                                                } else {
                                                    v500 = v497;
                                                }
                                                v495 = v500;
                                                v494 += 1l ;
                                            }
                                            static_array<int,2l> v501;
                                            int v503;
                                            v503 = 0l;
                                            while (while_method_3(v503)){
                                                v501[v503] = v495;
                                                v503 += 1l ;
                                            }
                                            static_array<int,2l> v505;
                                            int v507;
                                            v507 = 0l;
                                            while (while_method_3(v507)){
                                                int v509;
                                                v509 = v501[v507];
                                                bool v511;
                                                v511 = v507 == v468;
                                                int v513;
                                                if (v511){
                                                    int v512;
                                                    v512 = v509 + 4l;
                                                    v513 = v512;
                                                } else {
                                                    v513 = v509;
                                                }
                                                v505[v507] = v513;
                                                v507 += 1l ;
                                            }
                                            v557 = Union6{Union6_2{v465, false, v467, v492, v505, v493}};
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
                        v709 = Union7{Union7_1{v557}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v559 = v8.case3.v0; bool v560 = v8.case3.v1; static_array<Union2,2l> v561 = v8.case3.v2; int v562 = v8.case3.v3; static_array<int,2l> v563 = v8.case3.v4; int v564 = v8.case3.v5; Union3 v565 = v8.case3.v6;
                        Union1 v566;
                        v566 = Union1{Union1_1{v562, v565}};
                        v4.push(v566);
                        Union6 v652;
                        switch (v559.tag) {
                            case 0: { // None
                                switch (v565.tag) {
                                    case 0: { // Call
                                        if (v560){
                                            bool v616;
                                            v616 = v562 == 0l;
                                            int v617;
                                            if (v616){
                                                v617 = 1l;
                                            } else {
                                                v617 = 0l;
                                            }
                                            v652 = Union6{Union6_2{v559, false, v561, v617, v563, v564}};
                                        } else {
                                            v652 = Union6{Union6_0{v559, v560, v561, v562, v563, v564}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v652 = Union6{Union6_5{v559, v560, v561, v562, v563, v564}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v621;
                                        v621 = v564 > 0l;
                                        if (v621){
                                            bool v622;
                                            v622 = v562 == 0l;
                                            int v623;
                                            if (v622){
                                                v623 = 1l;
                                            } else {
                                                v623 = 0l;
                                            }
                                            int v624;
                                            v624 = -1l + v564;
                                            int v625; int v626;
                                            Tuple1 tmp21 = Tuple1{0l, 0l};
                                            v625 = tmp21.v0; v626 = tmp21.v1;
                                            while (while_method_3(v625)){
                                                int v628;
                                                v628 = v563[v625];
                                                bool v630;
                                                v630 = v626 >= v628;
                                                int v631;
                                                if (v630){
                                                    v631 = v626;
                                                } else {
                                                    v631 = v628;
                                                }
                                                v626 = v631;
                                                v625 += 1l ;
                                            }
                                            static_array<int,2l> v632;
                                            int v634;
                                            v634 = 0l;
                                            while (while_method_3(v634)){
                                                v632[v634] = v626;
                                                v634 += 1l ;
                                            }
                                            static_array<int,2l> v636;
                                            int v638;
                                            v638 = 0l;
                                            while (while_method_3(v638)){
                                                int v640;
                                                v640 = v632[v638];
                                                bool v642;
                                                v642 = v638 == v562;
                                                int v644;
                                                if (v642){
                                                    int v643;
                                                    v643 = v640 + 2l;
                                                    v644 = v643;
                                                } else {
                                                    v644 = v640;
                                                }
                                                v636[v638] = v644;
                                                v638 += 1l ;
                                            }
                                            v652 = Union6{Union6_2{v559, false, v561, v623, v636, v624}};
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
                                Union2 v567 = v559.case1.v0;
                                switch (v565.tag) {
                                    case 0: { // Call
                                        if (v560){
                                            bool v569;
                                            v569 = v562 == 0l;
                                            int v570;
                                            if (v569){
                                                v570 = 1l;
                                            } else {
                                                v570 = 0l;
                                            }
                                            v652 = Union6{Union6_2{v559, false, v561, v570, v563, v564}};
                                        } else {
                                            int v572; int v573;
                                            Tuple1 tmp22 = Tuple1{0l, 0l};
                                            v572 = tmp22.v0; v573 = tmp22.v1;
                                            while (while_method_3(v572)){
                                                int v575;
                                                v575 = v563[v572];
                                                bool v577;
                                                v577 = v573 >= v575;
                                                int v578;
                                                if (v577){
                                                    v578 = v573;
                                                } else {
                                                    v578 = v575;
                                                }
                                                v573 = v578;
                                                v572 += 1l ;
                                            }
                                            static_array<int,2l> v579;
                                            int v581;
                                            v581 = 0l;
                                            while (while_method_3(v581)){
                                                v579[v581] = v573;
                                                v581 += 1l ;
                                            }
                                            v652 = Union6{Union6_4{v559, v560, v561, v562, v579, v564}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v652 = Union6{Union6_5{v559, v560, v561, v562, v563, v564}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v585;
                                        v585 = v564 > 0l;
                                        if (v585){
                                            bool v586;
                                            v586 = v562 == 0l;
                                            int v587;
                                            if (v586){
                                                v587 = 1l;
                                            } else {
                                                v587 = 0l;
                                            }
                                            int v588;
                                            v588 = -1l + v564;
                                            int v589; int v590;
                                            Tuple1 tmp23 = Tuple1{0l, 0l};
                                            v589 = tmp23.v0; v590 = tmp23.v1;
                                            while (while_method_3(v589)){
                                                int v592;
                                                v592 = v563[v589];
                                                bool v594;
                                                v594 = v590 >= v592;
                                                int v595;
                                                if (v594){
                                                    v595 = v590;
                                                } else {
                                                    v595 = v592;
                                                }
                                                v590 = v595;
                                                v589 += 1l ;
                                            }
                                            static_array<int,2l> v596;
                                            int v598;
                                            v598 = 0l;
                                            while (while_method_3(v598)){
                                                v596[v598] = v590;
                                                v598 += 1l ;
                                            }
                                            static_array<int,2l> v600;
                                            int v602;
                                            v602 = 0l;
                                            while (while_method_3(v602)){
                                                int v604;
                                                v604 = v596[v602];
                                                bool v606;
                                                v606 = v602 == v562;
                                                int v608;
                                                if (v606){
                                                    int v607;
                                                    v607 = v604 + 4l;
                                                    v608 = v607;
                                                } else {
                                                    v608 = v604;
                                                }
                                                v600[v602] = v608;
                                                v602 += 1l ;
                                            }
                                            v652 = Union6{Union6_2{v559, false, v561, v587, v600, v588}};
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
                        v709 = Union7{Union7_1{v652}};
                        break;
                    }
                    case 4: { // TerminalCall
                        Union5 v26 = v8.case4.v0; bool v27 = v8.case4.v1; static_array<Union2,2l> v28 = v8.case4.v2; int v29 = v8.case4.v3; static_array<int,2l> v30 = v8.case4.v4; int v31 = v8.case4.v5;
                        int v32;
                        v32 = v30[v29];
                        Union11 v34;
                        v34 = compare_hands_9(v26, v27, v28, v29, v30, v31);
                        int v39; int v40;
                        switch (v34.tag) {
                            case 0: { // Eq
                                v39 = 0l; v40 = -1l;
                                break;
                            }
                            case 1: { // Gt
                                v39 = v32; v40 = 0l;
                                break;
                            }
                            case 2: { // Lt
                                v39 = v32; v40 = 1l;
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
                        static_array<float,2l> & v45 = v2.base->v5;
                        v45[v43] = v44;
                        bool v46;
                        v46 = v43 == 0l;
                        int v47;
                        if (v46){
                            v47 = 1l;
                        } else {
                            v47 = 0l;
                        }
                        float v48;
                        v48 = -v44;
                        v45[v47] = v48;
                        Union1 v49;
                        v49 = Union1{Union1_3{v28, v39, v40}};
                        v4.push(v49);
                        v709 = Union7{Union7_0{}};
                        break;
                    }
                    case 5: { // TerminalFold
                        Union5 v9 = v8.case5.v0; bool v10 = v8.case5.v1; static_array<Union2,2l> v11 = v8.case5.v2; int v12 = v8.case5.v3; static_array<int,2l> v13 = v8.case5.v4; int v14 = v8.case5.v5;
                        int v15;
                        v15 = v13[v12];
                        int v17;
                        v17 = -v15;
                        float v18;
                        v18 = (float)v17;
                        static_array<float,2l> & v19 = v2.base->v5;
                        v19[v12] = v18;
                        bool v20;
                        v20 = v12 == 0l;
                        int v21;
                        if (v20){
                            v21 = 1l;
                        } else {
                            v21 = 0l;
                        }
                        float v22;
                        v22 = -v18;
                        v19[v21] = v22;
                        int v23;
                        if (v20){
                            v23 = 1l;
                        } else {
                            v23 = 0l;
                        }
                        Union1 v24;
                        v24 = Union1{Union1_3{v11, v15, v23}};
                        v4.push(v24);
                        v709 = Union7{Union7_0{}};
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
        v6 = v709;
    }
    return ;
}
__device__ inline bool while_method_9(int v0){
    bool v1;
    v1 = v0 > 0l;
    return v1;
}
__device__ inline bool while_method_10(int v0){
    bool v1;
    v1 = v0 < 64l;
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
    v6 = v5 * 256l;
    int v7;
    v7 = v4 + v6;
    unsigned long long v8;
    v8 = (unsigned long long)v7;
    curandStatePhilox4_32_10_t v9;
    curand_init(v3,v8,0ull,&v9);
    static_array<Union0,2l> v10;
    Union0 v12;
    v12 = Union0{Union0_1{}};
    v10[0l] = v12;
    Union0 v14;
    v14 = Union0{Union0_1{}};
    v10[1l] = v14;
    static_array_list<Union1,32l> v16;
    v16 = static_array_list<Union1,32l>{};
    static_array<float,2l> v18;
    v18[0l] = 0.0f;
    v18[1l] = 0.0f;
    cooperative_groups::grid_group & v20 = v2;
    curandStatePhilox4_32_10_t & v21 = v9;
    Union4 v22;
    v22 = Union4{Union4_0{}};
    sptr<Mut0> v23;
    v23 = sptr<Mut0>{new Mut0{v22, 63ul, v20, v16, v10, v18, v21}};
    int v24;
    v24 = 0l;
    while (while_method_0(v24)){
        int v26;
        v26 = 0l;
        while (while_method_1(v26)){
            v23.base->v1 = 63ul;
            static_array<float,2l> v28;
            v28[0l] = 0.0f;
            v28[1l] = 0.0f;
            v23.base->v5 = v28;
            static_array_list<Union1,32l> & v30 = v23.base->v3;
            v30.unsafe_set_length(0l);
            static_array<Union0,2l> v31;
            Union0 v33;
            v33 = Union0{Union0_1{}};
            v31[0l] = v33;
            Union0 v35;
            v35 = Union0{Union0_1{}};
            v31[1l] = v35;
            Union0 v37;
            v37 = Union0{Union0_0{}};
            v31[0l] = v37;
            v23.base->v4 = v31;
            Union6 v39;
            v39 = Union6{Union6_1{}};
            method_0(v0, v1, v23, v39);
            static_array<float,2l> & v40 = v23.base->v5;
            static_array<float,2l> v41;
            v41 = v40;
            unsigned int * v42;
            v42 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
            int * v44;
            v44 = reinterpret_cast<int *>(&v1[262144ull]);
            float * v46;
            v46 = reinterpret_cast<float *>(&v1[262160ull]);
            float * v48;
            v48 = reinterpret_cast<float *>(&v1[524304ull]);
            float * v50;
            v50 = reinterpret_cast<float *>(&v1[786448ull]);
            float * v52;
            v52 = reinterpret_cast<float *>(&v1[1048592ull]);
            float * v54;
            v54 = reinterpret_cast<float *>(&v1[1310736ull]);
            float * v56;
            v56 = reinterpret_cast<float *>(&v1[1572880ull]);
            float * v58;
            v58 = reinterpret_cast<float *>(&v1[1835024ull]);
            int * v60;
            v60 = reinterpret_cast<int *>(&v0[6389760ull]);
            float * v62;
            v62 = reinterpret_cast<float *>(&v0[7962624ull]);
            int * v64;
            v64 = reinterpret_cast<int *>(&v0[9535488ull]);
            int * v66;
            v66 = reinterpret_cast<int *>(&v0[11108352ull]);
            double * v68;
            v68 = reinterpret_cast<double *>(&v0[12681216ull]);
            double * v70;
            v70 = reinterpret_cast<double *>(&v0[18972672ull]);
            double * v72;
            v72 = reinterpret_cast<double *>(&v1[2097168ull]);
            double * v74;
            v74 = reinterpret_cast<double *>(&v1[2490384ull]);
            int * v76;
            v76 = reinterpret_cast<int *>(&v1[2883600ull]);
            int v78;
            v78 = 0l;
            while (while_method_0(v78)){
                int v80;
                v80 = threadIdx.x;
                int v81;
                v81 = blockIdx.x;
                int v82;
                v82 = v81 * 256l;
                int v83;
                v83 = v80 + v82;
                float v84[2l];
                int v85;
                v85 = 0l;
                while (while_method_3(v85)){
                    float v87;
                    v87 = v41[v85];
                    v84[v85] = v87;
                    v85 += 1l ;
                }
                assert("Tensor range check" && 0 <= v78 && v78 < 4l);
                assert("Tensor range check" && 0 <= v83 && v83 < 6144l);
                int v89;
                v89 = 6144l * v78;
                int v90;
                v90 = v89 + v83;
                int v91;
                v91 = v76[v90];
                int v92;
                v92 = v91;
                while (while_method_9(v92)){
                    v92 -= 1l ;
                    assert("Tensor range check" && 0 <= v78 && v78 < 4l);
                    assert("Tensor range check" && 0 <= v92 && v92 < 16l);
                    assert("Tensor range check" && 0 <= v83 && v83 < 6144l);
                    int v94;
                    v94 = 6144l * v92;
                    int v95;
                    v95 = v94 + v83;
                    int v96;
                    v96 = 98304l * v78;
                    int v97;
                    v97 = v96 + v95;
                    int v98;
                    v98 = v60[v97];
                    float v99;
                    v99 = v62[v97];
                    int v100;
                    v100 = v64[v97];
                    int v101;
                    v101 = v66[v97];
                    assert("Tensor range check" && 0 <= v100 && v100 < 2l);
                    float v102;
                    v102 = v84[v100];
                    assert("Tensor range check" && 0 <= v78 && v78 < 4l);
                    int v103;
                    v103 = 16384l * v78;
                    assert("Tensor range check" && 0 <= v101 && v101 < 4096l);
                    int v104;
                    v104 = 4l * v101;
                    int v105;
                    v105 = v104 + v103;
                    float * v106;
                    v106 = v46+v105;
                    float * v108;
                    v108 = v48+v105;
                    float * v110;
                    v110 = v50+v105;
                    float * v112;
                    v112 = v52+v105;
                    float * v114;
                    v114 = v54+v105;
                    float * v116;
                    v116 = v56+v105;
                    float * v118;
                    v118 = v58+v105;
                    assert("Tensor range check" && 0 <= v78 && v78 < 4l);
                    int v120;
                    v120 = 196608l * v78;
                    assert("Tensor range check" && 0 <= v92 && v92 < 16l);
                    int v121;
                    v121 = 12288l * v92;
                    int v122;
                    v122 = v121 + v120;
                    assert("Tensor range check" && 0 <= v83 && v83 < 6144l);
                    int v123;
                    v123 = 2l * v83;
                    int v124;
                    v124 = v123 + v122;
                    double v125[2l];
                    int v126;
                    v126 = 0l;
                    while (while_method_3(v126)){
                        assert("Tensor range check" && 0 <= v126 && v126 < 2l);
                        int v128;
                        v128 = v126 + v124;
                        double v129;
                        v129 = v68[v128];
                        bool v130;
                        v130 = v100 == v126;
                        double v131;
                        if (v130){
                            v131 = 0.0;
                        } else {
                            v131 = v129;
                        }
                        assert("Tensor range check" && 0 <= v126 && v126 < 2l);
                        v125[v126] = v131;
                        v126 += 1l ;
                    }
                    double v132;
                    v132 = 0.0;
                    int v133;
                    v133 = 0l;
                    while (while_method_3(v133)){
                        assert("Tensor range check" && 0 <= v133 && v133 < 2l);
                        double v135;
                        v135 = v125[v133];
                        double v136;
                        v136 = v132 + v135;
                        v132 = v136;
                        v133 += 1l ;
                    }
                    double v137;
                    v137 = 0.0;
                    int v138;
                    v138 = 0l;
                    while (while_method_3(v138)){
                        assert("Tensor range check" && 0 <= v138 && v138 < 2l);
                        int v140;
                        v140 = v138 + v124;
                        double v141;
                        v141 = v70[v140];
                        double v142;
                        v142 = v137 + v141;
                        v137 = v142;
                        v138 += 1l ;
                    }
                    double v143;
                    v143 = v132 - v137;
                    double v144;
                    v144 = exp(v143);
                    float v145;
                    v145 = (float)v144;
                    float v146;
                    v146 = v102 * v145;
                    assert("Tensor range check" && 0 <= v98 && v98 < 4l);
                    float * v147;
                    v147 = v116+v98;
                    float * v149;
                    v149 = v118+v98;
                    float v151;
                    v151 = atomicAdd(v147,v146);
                    float v152;
                    v152 = atomicAdd(v149,v145);
                    float * v153;
                    v153 = v108+0l;
                    float * v155;
                    v155 = v112+0l;
                    float * v157;
                    v157 = v114+0l;
                    int v159;
                    v159 = sizeof(float *);
                    unsigned long long v160;
                    v160 = (unsigned long long)v159;
                    unsigned long long v161;
                    v161 = 256ull * v160;
                    unsigned long long v162;
                    v162 = 4096ull + v161;
                    unsigned long long v163;
                    v163 = v162 + 16ull;
                    unsigned long long v164;
                    v164 = v163 - 1ull;
                    unsigned long long v165;
                    v165 = v164 % 16ull;
                    unsigned long long v166;
                    v166 = v164 - v165;
                    unsigned long long v167;
                    v167 = v166 + v161;
                    unsigned long long v168;
                    v168 = v167 + 16ull;
                    unsigned long long v169;
                    v169 = v168 - 1ull;
                    unsigned long long v170;
                    v170 = v169 % 16ull;
                    unsigned long long v171;
                    v171 = v169 - v170;
                    unsigned long long v172;
                    v172 = v171 + v161;
                    unsigned long long v173;
                    v173 = v172 + 16ull;
                    unsigned long long v174;
                    v174 = v173 - 1ull;
                    unsigned long long v175;
                    v175 = v174 % 16ull;
                    unsigned long long v176;
                    v176 = v174 - v175;
                    unsigned long long v177;
                    v177 = v176 + v161;
                    unsigned long long v178;
                    v178 = v177 + 16ull;
                    unsigned long long v179;
                    v179 = v178 - 1ull;
                    unsigned long long v180;
                    v180 = v179 % 16ull;
                    unsigned long long v181;
                    v181 = v179 - v180;
                    unsigned long long v182;
                    v182 = v181 + 1024ull;
                    bool v183;
                    v183 = v182 <= 81920ull;
                    bool v184;
                    v184 = v183 == false;
                    if (v184){
                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v183);
                    } else {
                    }
                    extern __shared__ unsigned char v186[];
                    bool v187;
                    v187 = v182 <= v182;
                    bool v188;
                    v188 = v187 == false;
                    if (v188){
                        assert("The length of the partition has to be less than or equal to the length of the base array." && v187);
                    } else {
                    }
                    float * v190;
                    v190 = reinterpret_cast<float *>(&v186[0ull]);
                    int * v192;
                    v192 = reinterpret_cast<int *>(&v186[1024ull]);
                    float * v194;
                    v194 = reinterpret_cast<float *>(&v186[2048ull]);
                    float * v196;
                    v196 = reinterpret_cast<float *>(&v186[3072ull]);
                    float * * v198;
                    v198 = reinterpret_cast<float * *>(&v186[4096ull]);
                    float * * v200;
                    v200 = reinterpret_cast<float * *>(&v186[v166]);
                    float * * v202;
                    v202 = reinterpret_cast<float * *>(&v186[v171]);
                    float * * v204;
                    v204 = reinterpret_cast<float * *>(&v186[v176]);
                    float * v206;
                    v206 = reinterpret_cast<float *>(&v186[v181]);
                    int v208;
                    v208 = threadIdx.x;
                    assert("Tensor range check" && 0 <= v208 && v208 < 256l);
                    v190[v208] = v99;
                    v192[v208] = v98;
                    v194[v208] = v102;
                    v196[v208] = v145;
                    v198[v208] = v110;
                    v200[v208] = v153;
                    v202[v208] = v155;
                    v204[v208] = v157;
                    asm("barrier.cta.sync %0;" :: "r"(0l));
                    bool v209;
                    v209 = 0l <= v208;
                    bool v210;
                    v210 = v209 == false;
                    if (v210){
                        assert("The index needs to be zero or positive." && v209);
                    } else {
                    }
                    int v212;
                    v212 = v208 % 1l;
                    bool v213;
                    v213 = v208 < 256l;
                    bool v214;
                    v214 = v213 == false;
                    if (v214){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v213);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v208 && v208 < 256l);
                    int v216;
                    v216 = 0l;
                    while (while_method_6(v216)){
                        bool v218;
                        v218 = v209 && v213;
                        bool v219;
                        v219 = v218 == false;
                        if (v219){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v218);
                        } else {
                        }
                        bool v221;
                        v221 = 0l <= v216;
                        bool v223;
                        if (v221){
                            bool v222;
                            v222 = v216 < 1l;
                            v223 = v222;
                        } else {
                            v223 = false;
                        }
                        bool v224;
                        v224 = v223 == false;
                        if (v224){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v223);
                        } else {
                        }
                        int v226;
                        v226 = v216 * 256l;
                        int v227;
                        v227 = v226 + v208;
                        assert("Tensor range check" && 0 <= v216 && v216 < 1l);
                        int v228;
                        v228 = 256l * v216;
                        int v229;
                        v229 = v228 + v208;
                        float v230;
                        v230 = v190[v229];
                        int v231;
                        v231 = v192[v229];
                        float v232;
                        v232 = v194[v229];
                        float v233;
                        v233 = v196[v229];
                        float * v234;
                        v234 = v198[v229];
                        float * v235;
                        v235 = v200[v229];
                        float * v236;
                        v236 = v202[v229];
                        float * v237;
                        v237 = v204[v229];
                        int v238;
                        v238 = blockIdx.x;
                        int v239;
                        v239 = v238 * 256l;
                        int v240;
                        v240 = v239 + v227;
                        assert("Tensor range check" && 0 <= v212 && v212 < 1l);
                        int v241;
                        v241 = 4l * v212;
                        float v242[4l];
                        float v243[4l];
                        float v244[4l];
                        int v245[4l];
                        int v246;
                        v246 = 0l;
                        while (while_method_6(v246)){
                            assert("Tensor range check" && 0 <= v246 && v246 < 1l);
                            int v248;
                            v248 = 4l * v246;
                            assert("Tensor range check" && 0 <= v246 && v246 < 1l);
                            int v249;
                            v249 = v248 + v241;
                            int4* v250;
                            v250 = reinterpret_cast<int4*>(v235 + v249);
                            int4* v251;
                            v251 = reinterpret_cast<int4*>(v242 + v248);
                            assert("Pointer alignment check" && (unsigned long long)(v250) % 4l == 0 && (unsigned long long)(v251) % 4l == 0);
                            *v251 = *v250;
                            int4* v252;
                            v252 = reinterpret_cast<int4*>(v236 + v249);
                            int4* v253;
                            v253 = reinterpret_cast<int4*>(v243 + v248);
                            assert("Pointer alignment check" && (unsigned long long)(v252) % 4l == 0 && (unsigned long long)(v253) % 4l == 0);
                            *v253 = *v252;
                            int4* v254;
                            v254 = reinterpret_cast<int4*>(v237 + v249);
                            int4* v255;
                            v255 = reinterpret_cast<int4*>(v244 + v248);
                            assert("Pointer alignment check" && (unsigned long long)(v254) % 4l == 0 && (unsigned long long)(v255) % 4l == 0);
                            *v255 = *v254;
                            v246 += 1l ;
                        }
                        int v256;
                        v256 = 0l;
                        while (while_method_6(v256)){
                            int v258;
                            v258 = 0l;
                            while (while_method_0(v258)){
                                bool v260;
                                v260 = 0l <= v258;
                                bool v262;
                                if (v260){
                                    bool v261;
                                    v261 = v258 < 4l;
                                    v262 = v261;
                                } else {
                                    v262 = false;
                                }
                                bool v263;
                                v263 = v262 == false;
                                if (v263){
                                    assert("The indices should be inside the range of the dimension." && v262);
                                } else {
                                }
                                bool v265;
                                v265 = 0l <= v212;
                                bool v267;
                                if (v265){
                                    bool v266;
                                    v266 = v212 < 1l;
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
                                int v270;
                                v270 = v212 * 4l;
                                int v271;
                                v271 = v258 + v270;
                                bool v272;
                                v272 = 0l <= v256;
                                bool v274;
                                if (v272){
                                    bool v273;
                                    v273 = v256 < 1l;
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
                                v277 = v256 * 4l;
                                int v278;
                                v278 = v271 + v277;
                                assert("Tensor range check" && 0 <= v256 && v256 < 1l);
                                assert("Tensor range check" && 0 <= v258 && v258 < 4l);
                                int v279;
                                v279 = 4l * v256;
                                int v280;
                                v280 = v279 + v258;
                                v245[v280] = v278;
                                v258 += 1l ;
                            }
                            v256 += 1l ;
                        }
                        float v281[4l];
                        int v282;
                        v282 = 0l;
                        while (while_method_6(v282)){
                            int v284;
                            v284 = 0l;
                            while (while_method_0(v284)){
                                assert("Tensor range check" && 0 <= v282 && v282 < 1l);
                                assert("Tensor range check" && 0 <= v284 && v284 < 4l);
                                int v286;
                                v286 = 4l * v282;
                                int v287;
                                v287 = v286 + v284;
                                float v288;
                                v288 = v243[v287];
                                float v289;
                                v289 = v244[v287];
                                bool v290;
                                v290 = v289 == 0.0f;
                                bool v291;
                                v291 = v290 != true;
                                float v293;
                                if (v291){
                                    float v292;
                                    v292 = v288 / v289;
                                    v293 = v292;
                                } else {
                                    v293 = 0.0f;
                                }
                                assert("Tensor range check" && 0 <= v282 && v282 < 1l);
                                assert("Tensor range check" && 0 <= v284 && v284 < 4l);
                                v281[v287] = v293;
                                v284 += 1l ;
                            }
                            v282 += 1l ;
                        }
                        bool v294[4l];
                        int v295;
                        v295 = 0l;
                        while (while_method_6(v295)){
                            int v297;
                            v297 = 0l;
                            while (while_method_0(v297)){
                                assert("Tensor range check" && 0 <= v295 && v295 < 1l);
                                assert("Tensor range check" && 0 <= v297 && v297 < 4l);
                                int v299;
                                v299 = 4l * v295;
                                int v300;
                                v300 = v299 + v297;
                                float v301;
                                v301 = v242[v300];
                                int v302;
                                v302 = v245[v300];
                                bool v303;
                                v303 = v302 < 3l;
                                assert("Tensor range check" && 0 <= v295 && v295 < 1l);
                                assert("Tensor range check" && 0 <= v297 && v297 < 4l);
                                v294[v300] = v303;
                                v297 += 1l ;
                            }
                            v295 += 1l ;
                        }
                        float v304[4l];
                        int v305;
                        v305 = 0l;
                        while (while_method_6(v305)){
                            int v307;
                            v307 = 0l;
                            while (while_method_0(v307)){
                                assert("Tensor range check" && 0 <= v305 && v305 < 1l);
                                assert("Tensor range check" && 0 <= v307 && v307 < 4l);
                                int v309;
                                v309 = 4l * v305;
                                int v310;
                                v310 = v309 + v307;
                                float v311;
                                v311 = v242[v310];
                                bool v312;
                                v312 = v294[v310];
                                float v315;
                                if (v312){
                                    bool v313;
                                    v313 = 0.0f >= v311;
                                    if (v313){
                                        v315 = 0.0f;
                                    } else {
                                        v315 = v311;
                                    }
                                } else {
                                    v315 = 0.0f;
                                }
                                assert("Tensor range check" && 0 <= v305 && v305 < 1l);
                                assert("Tensor range check" && 0 <= v307 && v307 < 4l);
                                v304[v310] = v315;
                                v307 += 1l ;
                            }
                            v305 += 1l ;
                        }
                        float v316;
                        v316 = 0.0f;
                        int v317;
                        v317 = 0l;
                        while (while_method_6(v317)){
                            int v319;
                            v319 = 0l;
                            while (while_method_0(v319)){
                                assert("Tensor range check" && 0 <= v317 && v317 < 1l);
                                assert("Tensor range check" && 0 <= v319 && v319 < 4l);
                                int v321;
                                v321 = 4l * v317;
                                int v322;
                                v322 = v321 + v319;
                                float v323;
                                v323 = v304[v322];
                                float v324;
                                v324 = v316 + v323;
                                v316 = v324;
                                v319 += 1l ;
                            }
                            v317 += 1l ;
                        }
                        auto v325 = cooperative_groups::coalesced_threads();
                        int v326;
                        v326 = threadIdx.x;
                        auto v327 = cooperative_groups::labeled_partition(v325,v326);
                        Closure1 v328{};
                        float v329;
                        v329 = cooperative_groups::reduce(v327, v316, v328);
                        int v330[4l];
                        int v331;
                        v331 = 0l;
                        while (while_method_6(v331)){
                            int v333;
                            v333 = 0l;
                            while (while_method_0(v333)){
                                assert("Tensor range check" && 0 <= v331 && v331 < 1l);
                                assert("Tensor range check" && 0 <= v333 && v333 < 4l);
                                int v335;
                                v335 = 4l * v331;
                                int v336;
                                v336 = v335 + v333;
                                bool v337;
                                v337 = v294[v336];
                                int v338;
                                if (v337){
                                    v338 = 1l;
                                } else {
                                    v338 = 0l;
                                }
                                assert("Tensor range check" && 0 <= v331 && v331 < 1l);
                                assert("Tensor range check" && 0 <= v333 && v333 < 4l);
                                v330[v336] = v338;
                                v333 += 1l ;
                            }
                            v331 += 1l ;
                        }
                        int v339;
                        v339 = 0l;
                        int v340;
                        v340 = 0l;
                        while (while_method_6(v340)){
                            int v342;
                            v342 = 0l;
                            while (while_method_0(v342)){
                                assert("Tensor range check" && 0 <= v340 && v340 < 1l);
                                assert("Tensor range check" && 0 <= v342 && v342 < 4l);
                                int v344;
                                v344 = 4l * v340;
                                int v345;
                                v345 = v344 + v342;
                                int v346;
                                v346 = v330[v345];
                                int v347;
                                v347 = v339 + v346;
                                v339 = v347;
                                v342 += 1l ;
                            }
                            v340 += 1l ;
                        }
                        auto v348 = cooperative_groups::coalesced_threads();
                        int v349;
                        v349 = threadIdx.x;
                        auto v350 = cooperative_groups::labeled_partition(v348,v349);
                        Closure2 v351{};
                        int v352;
                        v352 = cooperative_groups::reduce(v350, v339, v351);
                        float v353;
                        v353 = (float)v352;
                        float v354;
                        v354 = 1.0f / v353;
                        float v355[4l];
                        int v356;
                        v356 = 0l;
                        while (while_method_6(v356)){
                            int v358;
                            v358 = 0l;
                            while (while_method_0(v358)){
                                assert("Tensor range check" && 0 <= v356 && v356 < 1l);
                                assert("Tensor range check" && 0 <= v358 && v358 < 4l);
                                int v360;
                                v360 = 4l * v356;
                                int v361;
                                v361 = v360 + v358;
                                float v362;
                                v362 = v304[v361];
                                bool v363;
                                v363 = v294[v361];
                                bool v364;
                                v364 = v363 == false;
                                float v369;
                                if (v364){
                                    v369 = 0.0f;
                                } else {
                                    bool v365;
                                    v365 = v329 == 0.0f;
                                    bool v366;
                                    v366 = v365 != true;
                                    if (v366){
                                        float v367;
                                        v367 = v362 / v329;
                                        v369 = v367;
                                    } else {
                                        v369 = v354;
                                    }
                                }
                                assert("Tensor range check" && 0 <= v356 && v356 < 1l);
                                assert("Tensor range check" && 0 <= v358 && v358 < 4l);
                                v355[v361] = v369;
                                v358 += 1l ;
                            }
                            v356 += 1l ;
                        }
                        float v370[4l];
                        int v371;
                        v371 = 0l;
                        while (while_method_6(v371)){
                            int v373;
                            v373 = 0l;
                            while (while_method_0(v373)){
                                assert("Tensor range check" && 0 <= v371 && v371 < 1l);
                                assert("Tensor range check" && 0 <= v373 && v373 < 4l);
                                int v375;
                                v375 = 4l * v371;
                                int v376;
                                v376 = v375 + v373;
                                float v377;
                                v377 = v281[v376];
                                int v378;
                                v378 = v245[v376];
                                bool v379;
                                v379 = v231 == v378;
                                float v382;
                                if (v379){
                                    float v380;
                                    v380 = v232 - v377;
                                    float v381;
                                    v381 = v380 / v230;
                                    v382 = v381;
                                } else {
                                    v382 = 0.0f;
                                }
                                float v383;
                                v383 = v382 + v377;
                                assert("Tensor range check" && 0 <= v371 && v371 < 1l);
                                assert("Tensor range check" && 0 <= v373 && v373 < 4l);
                                v370[v376] = v383;
                                v373 += 1l ;
                            }
                            v371 += 1l ;
                        }
                        float v384[4l];
                        int v385;
                        v385 = 0l;
                        while (while_method_6(v385)){
                            int v387;
                            v387 = 0l;
                            while (while_method_0(v387)){
                                assert("Tensor range check" && 0 <= v385 && v385 < 1l);
                                assert("Tensor range check" && 0 <= v387 && v387 < 4l);
                                int v389;
                                v389 = 4l * v385;
                                int v390;
                                v390 = v389 + v387;
                                float v391;
                                v391 = v355[v390];
                                float v392;
                                v392 = v370[v390];
                                float v393;
                                v393 = v391 * v392;
                                assert("Tensor range check" && 0 <= v385 && v385 < 1l);
                                assert("Tensor range check" && 0 <= v387 && v387 < 4l);
                                v384[v390] = v393;
                                v387 += 1l ;
                            }
                            v385 += 1l ;
                        }
                        float v394;
                        v394 = 0.0f;
                        int v395;
                        v395 = 0l;
                        while (while_method_6(v395)){
                            int v397;
                            v397 = 0l;
                            while (while_method_0(v397)){
                                assert("Tensor range check" && 0 <= v395 && v395 < 1l);
                                assert("Tensor range check" && 0 <= v397 && v397 < 4l);
                                int v399;
                                v399 = 4l * v395;
                                int v400;
                                v400 = v399 + v397;
                                float v401;
                                v401 = v384[v400];
                                float v402;
                                v402 = v394 + v401;
                                v394 = v402;
                                v397 += 1l ;
                            }
                            v395 += 1l ;
                        }
                        auto v403 = cooperative_groups::coalesced_threads();
                        int v404;
                        v404 = threadIdx.x;
                        auto v405 = cooperative_groups::labeled_partition(v403,v404);
                        float v406;
                        v406 = cooperative_groups::reduce(v405, v394, v328);
                        int v407;
                        v407 = 0l;
                        while (while_method_6(v407)){
                            int v409;
                            v409 = 0l;
                            while (while_method_0(v409)){
                                assert("Tensor range check" && 0 <= v407 && v407 < 1l);
                                assert("Tensor range check" && 0 <= v409 && v409 < 4l);
                                int v411;
                                v411 = 4l * v407;
                                int v412;
                                v412 = v411 + v409;
                                float v413;
                                v413 = v370[v412];
                                int v414;
                                v414 = v245[v412];
                                float v415;
                                v415 = v413 - v406;
                                float v416;
                                v416 = v233 * v415;
                                assert("Tensor range check" && 0 <= v414 && v414 < 4l);
                                float * v417;
                                v417 = v234+v414;
                                float v419;
                                v419 = atomicAdd(v417,v416);
                                v409 += 1l ;
                            }
                            v407 += 1l ;
                        }
                        int v420;
                        v420 = 0l;
                        while (while_method_6(v420)){
                            assert("Tensor range check" && 0 <= v420 && v420 < 1l);
                            assert("Tensor range check" && 0 <= v420 && v420 < 1l);
                            v420 += 1l ;
                        }
                        assert("Tensor range check" && 0 <= v227 && v227 < 256l);
                        v206[v227] = v406;
                        v216 += 1l ;
                    }
                    asm("barrier.cta.sync %0;" :: "r"(0l));
                    assert("Tensor range check" && 0 <= v208 && v208 < 256l);
                    float v422;
                    v422 = v206[v208];
                    asm("barrier.cta.sync %0;" :: "r"(0l));
                    assert("Tensor range check" && 0 <= v100 && v100 < 2l);
                    v84[v100] = v422;
                }
                int v423;
                v423 = threadIdx.x;
                int v424;
                v424 = blockIdx.x;
                int v425;
                v425 = v424 * 256l;
                int v426;
                v426 = v423 + v425;
                assert("Tensor range check" && 0 <= v78 && v78 < 4l);
                int v427;
                v427 = 12288l * v78;
                assert("Tensor range check" && 0 <= v426 && v426 < 6144l);
                int v428;
                v428 = 2l * v426;
                int v429;
                v429 = v428 + v427;
                double * v430;
                v430 = v72+v429;
                double * v432;
                v432 = v74+v429;
                double * v434;
                v434 = v430+0l;
                double * v436;
                v436 = v432+0l;
                double * v438;
                v438 = v430+0l;
                double * v440;
                v440 = v432+0l;
                int v442;
                v442 = sizeof(double *);
                unsigned long long v443;
                v443 = (unsigned long long)v442;
                unsigned long long v444;
                v444 = 256ull * v443;
                unsigned long long v445;
                v445 = v444 + 16ull;
                unsigned long long v446;
                v446 = v445 - 1ull;
                unsigned long long v447;
                v447 = v446 % 16ull;
                unsigned long long v448;
                v448 = v446 - v447;
                unsigned long long v449;
                v449 = v448 + v444;
                unsigned long long v450;
                v450 = v449 + 16ull;
                unsigned long long v451;
                v451 = v450 - 1ull;
                unsigned long long v452;
                v452 = v451 % 16ull;
                unsigned long long v453;
                v453 = v451 - v452;
                unsigned long long v454;
                v454 = v453 + v444;
                unsigned long long v455;
                v455 = v454 + 16ull;
                unsigned long long v456;
                v456 = v455 - 1ull;
                unsigned long long v457;
                v457 = v456 % 16ull;
                unsigned long long v458;
                v458 = v456 - v457;
                unsigned long long v459;
                v459 = v458 + v444;
                bool v460;
                v460 = v459 <= 81920ull;
                bool v461;
                v461 = v460 == false;
                if (v461){
                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v460);
                } else {
                }
                extern __shared__ unsigned char v463[];
                bool v464;
                v464 = v459 <= v459;
                bool v465;
                v465 = v464 == false;
                if (v465){
                    assert("The length of the partition has to be less than or equal to the length of the base array." && v464);
                } else {
                }
                double * * v467;
                v467 = reinterpret_cast<double * *>(&v463[0ull]);
                double * * v469;
                v469 = reinterpret_cast<double * *>(&v463[v448]);
                double * * v471;
                v471 = reinterpret_cast<double * *>(&v463[v453]);
                double * * v473;
                v473 = reinterpret_cast<double * *>(&v463[v458]);
                int v475;
                v475 = threadIdx.x;
                assert("Tensor range check" && 0 <= v475 && v475 < 256l);
                v467[v475] = v434;
                v469[v475] = v436;
                v471[v475] = v438;
                v473[v475] = v440;
                asm("barrier.cta.sync %0;" :: "r"(0l));
                bool v476;
                v476 = 0l <= v475;
                bool v477;
                v477 = v476 == false;
                if (v477){
                    assert("The index needs to be zero or positive." && v476);
                } else {
                }
                int v479;
                v479 = v475 % 1l;
                bool v480;
                v480 = v475 < 256l;
                bool v481;
                v481 = v480 == false;
                if (v481){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v480);
                } else {
                }
                assert("Tensor range check" && 0 <= v475 && v475 < 256l);
                int v483;
                v483 = 0l;
                while (while_method_6(v483)){
                    bool v485;
                    v485 = v476 && v480;
                    bool v486;
                    v486 = v485 == false;
                    if (v486){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v485);
                    } else {
                    }
                    bool v488;
                    v488 = 0l <= v483;
                    bool v490;
                    if (v488){
                        bool v489;
                        v489 = v483 < 1l;
                        v490 = v489;
                    } else {
                        v490 = false;
                    }
                    bool v491;
                    v491 = v490 == false;
                    if (v491){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v490);
                    } else {
                    }
                    int v493;
                    v493 = v483 * 256l;
                    int v494;
                    v494 = v493 + v475;
                    assert("Tensor range check" && 0 <= v483 && v483 < 1l);
                    int v495;
                    v495 = 256l * v483;
                    int v496;
                    v496 = v495 + v475;
                    double * v497;
                    v497 = v467[v496];
                    double * v498;
                    v498 = v469[v496];
                    double * v499;
                    v499 = v471[v496];
                    double * v500;
                    v500 = v473[v496];
                    int v501;
                    v501 = blockIdx.x;
                    int v502;
                    v502 = v501 * 256l;
                    int v503;
                    v503 = v502 + v494;
                    assert("Tensor range check" && 0 <= v479 && v479 < 1l);
                    int v504;
                    v504 = 2l * v479;
                    double v505[2l];
                    double v506[2l];
                    int v507[2l];
                    int v508;
                    v508 = 0l;
                    while (while_method_6(v508)){
                        assert("Tensor range check" && 0 <= v508 && v508 < 1l);
                        int v510;
                        v510 = 2l * v508;
                        assert("Tensor range check" && 0 <= v508 && v508 < 1l);
                        int v511;
                        v511 = v510 + v504;
                        int4* v512;
                        v512 = reinterpret_cast<int4*>(v497 + v511);
                        int4* v513;
                        v513 = reinterpret_cast<int4*>(v505 + v510);
                        assert("Pointer alignment check" && (unsigned long long)(v512) % 2l == 0 && (unsigned long long)(v513) % 2l == 0);
                        *v513 = *v512;
                        int4* v514;
                        v514 = reinterpret_cast<int4*>(v498 + v511);
                        int4* v515;
                        v515 = reinterpret_cast<int4*>(v506 + v510);
                        assert("Pointer alignment check" && (unsigned long long)(v514) % 2l == 0 && (unsigned long long)(v515) % 2l == 0);
                        *v515 = *v514;
                        v508 += 1l ;
                    }
                    int v516;
                    v516 = 0l;
                    while (while_method_6(v516)){
                        int v518;
                        v518 = 0l;
                        while (while_method_3(v518)){
                            bool v520;
                            v520 = 0l <= v518;
                            bool v522;
                            if (v520){
                                bool v521;
                                v521 = v518 < 2l;
                                v522 = v521;
                            } else {
                                v522 = false;
                            }
                            bool v523;
                            v523 = v522 == false;
                            if (v523){
                                assert("The indices should be inside the range of the dimension." && v522);
                            } else {
                            }
                            bool v525;
                            v525 = 0l <= v479;
                            bool v527;
                            if (v525){
                                bool v526;
                                v526 = v479 < 1l;
                                v527 = v526;
                            } else {
                                v527 = false;
                            }
                            bool v528;
                            v528 = v527 == false;
                            if (v528){
                                assert("The indices should be inside the range of the dimension." && v527);
                            } else {
                            }
                            int v530;
                            v530 = v479 * 2l;
                            int v531;
                            v531 = v518 + v530;
                            bool v532;
                            v532 = 0l <= v516;
                            bool v534;
                            if (v532){
                                bool v533;
                                v533 = v516 < 1l;
                                v534 = v533;
                            } else {
                                v534 = false;
                            }
                            bool v535;
                            v535 = v534 == false;
                            if (v535){
                                assert("The indices should be inside the range of the dimension." && v534);
                            } else {
                            }
                            int v537;
                            v537 = v516 * 2l;
                            int v538;
                            v538 = v531 + v537;
                            assert("Tensor range check" && 0 <= v516 && v516 < 1l);
                            assert("Tensor range check" && 0 <= v518 && v518 < 2l);
                            int v539;
                            v539 = 2l * v516;
                            int v540;
                            v540 = v539 + v518;
                            v507[v540] = v538;
                            v518 += 1l ;
                        }
                        v516 += 1l ;
                    }
                    double v541[2l];
                    double v542[2l];
                    int v543;
                    v543 = 0l;
                    while (while_method_6(v543)){
                        int v545;
                        v545 = 0l;
                        while (while_method_3(v545)){
                            assert("Tensor range check" && 0 <= v543 && v543 < 1l);
                            assert("Tensor range check" && 0 <= v545 && v545 < 2l);
                            int v547;
                            v547 = 2l * v543;
                            int v548;
                            v548 = v547 + v545;
                            double v549;
                            v549 = v505[v548];
                            double v550;
                            v550 = v506[v548];
                            assert("Tensor range check" && 0 <= v543 && v543 < 1l);
                            assert("Tensor range check" && 0 <= v545 && v545 < 2l);
                            v541[v548] = 0.0;
                            v542[v548] = 0.0;
                            v545 += 1l ;
                        }
                        v543 += 1l ;
                    }
                    int v551;
                    v551 = 0l;
                    while (while_method_6(v551)){
                        assert("Tensor range check" && 0 <= v551 && v551 < 1l);
                        int v553;
                        v553 = 2l * v551;
                        int v554;
                        v554 = v553 + v504;
                        assert("Tensor range check" && 0 <= v551 && v551 < 1l);
                        int4* v555;
                        v555 = reinterpret_cast<int4*>(v541 + v553);
                        int4* v556;
                        v556 = reinterpret_cast<int4*>(v499 + v554);
                        assert("Pointer alignment check" && (unsigned long long)(v555) % 2l == 0 && (unsigned long long)(v556) % 2l == 0);
                        *v556 = *v555;
                        int4* v557;
                        v557 = reinterpret_cast<int4*>(v542 + v553);
                        int4* v558;
                        v558 = reinterpret_cast<int4*>(v500 + v554);
                        assert("Pointer alignment check" && (unsigned long long)(v557) % 2l == 0 && (unsigned long long)(v558) % 2l == 0);
                        *v558 = *v557;
                        v551 += 1l ;
                    }
                    assert("Tensor range check" && 0 <= v494 && v494 < 256l);
                    v483 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                assert("Tensor range check" && 0 <= v475 && v475 < 256l);
                asm("barrier.cta.sync %0;" :: "r"(0l));
                assert("Tensor range check" && 0 <= v78 && v78 < 4l);
                assert("Tensor range check" && 0 <= v426 && v426 < 6144l);
                int v559;
                v559 = v89 + v426;
                v76[v559] = 0l;
                v78 += 1l ;
            }
            v26 += 1l ;
        }
        cooperative_groups::grid_group & v560 = v23.base->v2;
        cooperative_groups::grid_group & v561 = v560;
        curandStatePhilox4_32_10_t & v562 = v23.base->v6;
        curandStatePhilox4_32_10_t & v563 = v562;
        unsigned int * v564;
        v564 = reinterpret_cast<unsigned int *>(&v0[6291456ull]);
        int * v566;
        v566 = reinterpret_cast<int *>(&v1[262144ull]);
        float * v568;
        v568 = reinterpret_cast<float *>(&v1[262160ull]);
        float * v570;
        v570 = reinterpret_cast<float *>(&v1[524304ull]);
        float * v572;
        v572 = reinterpret_cast<float *>(&v1[786448ull]);
        float * v574;
        v574 = reinterpret_cast<float *>(&v1[1048592ull]);
        float * v576;
        v576 = reinterpret_cast<float *>(&v1[1310736ull]);
        float * v578;
        v578 = reinterpret_cast<float *>(&v1[1572880ull]);
        float * v580;
        v580 = reinterpret_cast<float *>(&v1[1835024ull]);
        int * v582;
        v582 = reinterpret_cast<int *>(&v0[6389760ull]);
        float * v584;
        v584 = reinterpret_cast<float *>(&v0[7962624ull]);
        int * v586;
        v586 = reinterpret_cast<int *>(&v0[9535488ull]);
        int * v588;
        v588 = reinterpret_cast<int *>(&v0[11108352ull]);
        double * v590;
        v590 = reinterpret_cast<double *>(&v0[12681216ull]);
        double * v592;
        v592 = reinterpret_cast<double *>(&v0[18972672ull]);
        double * v594;
        v594 = reinterpret_cast<double *>(&v1[2097168ull]);
        double * v596;
        v596 = reinterpret_cast<double *>(&v1[2490384ull]);
        int * v598;
        v598 = reinterpret_cast<int *>(&v1[2883600ull]);
        v561.sync() ;
        int v600;
        v600 = threadIdx.x;
        int v601;
        v601 = blockIdx.x;
        int v602;
        v602 = v601 * 256l;
        int v603;
        v603 = v600 + v602;
        bool v604;
        v604 = v603 == 0l;
        if (v604){
            int v605;
            v605 = 0l;
            int v606;
            v606 = 4l;
            int v607;
            v607 = int_range_8(v606, v605, v563);
            v566[0l] = v607;
        } else {
        }
        __syncwarp();
        int v608;
        v608 = threadIdx.x;
        bool v609;
        v609 = 0l <= v608;
        bool v610;
        v610 = v609 == false;
        if (v610){
            assert("The index needs to be zero or positive." && v609);
        } else {
        }
        int v612;
        v612 = v608 % 1l;
        int v613;
        v613 = v608 % 256l;
        int v614;
        v614 = v608 / 256l;
        bool v615;
        v615 = v614 < 1l;
        bool v616;
        v616 = v615 == false;
        if (v616){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v615);
        } else {
        }
        assert("Tensor range check" && 0 <= v614 && v614 < 1l);
        assert("Tensor range check" && 0 <= v613 && v613 < 256l);
        assert("Tensor range check" && 0 <= v612 && v612 < 1l);
        int v618;
        v618 = 4l * v612;
        int v619;
        v619 = 4l * v613;
        int v620;
        v620 = v619 + v618;
        int v621;
        v621 = 16384l * v614;
        int v622;
        v622 = v621 + v620;
        assert("Tensor range check" && 0 <= v614 && v614 < 1l);
        assert("Tensor range check" && 0 <= v613 && v613 < 256l);
        assert("Tensor range check" && 0 <= v612 && v612 < 1l);
        int v623;
        v623 = blockIdx.x;
        int v624;
        v624 = v623;
        while (while_method_10(v624)){
            bool v626;
            v626 = 0l <= v624;
            bool v627;
            v627 = v626 == false;
            if (v627){
                assert("The index needs to be zero or positive." && v626);
            } else {
            }
            int v629;
            v629 = v624 % 16l;
            int v630;
            v630 = v624 / 16l;
            bool v631;
            v631 = v630 < 4l;
            bool v632;
            v632 = v631 == false;
            if (v632){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v631);
            } else {
            }
            assert("Tensor range check" && 0 <= v630 && v630 < 4l);
            assert("Tensor range check" && 0 <= v629 && v629 < 16l);
            int v634;
            v634 = 1024l * v629;
            int v635;
            v635 = v634 + v622;
            int v636;
            v636 = 16384l * v630;
            int v637;
            v637 = v636 + v635;
            float v638[4l];
            float v639[4l];
            float v640[4l];
            float v641[4l];
            float v642[4l];
            float v643[4l];
            float v644[4l];
            int v645[4l];
            int v646;
            v646 = 0l;
            while (while_method_6(v646)){
                assert("Tensor range check" && 0 <= v646 && v646 < 1l);
                int v648;
                v648 = 4l * v646;
                assert("Tensor range check" && 0 <= v646 && v646 < 1l);
                int v649;
                v649 = v648 + v637;
                int4* v650;
                v650 = reinterpret_cast<int4*>(v568 + v649);
                int4* v651;
                v651 = reinterpret_cast<int4*>(v638 + v648);
                assert("Pointer alignment check" && (unsigned long long)(v650) % 4l == 0 && (unsigned long long)(v651) % 4l == 0);
                *v651 = *v650;
                int4* v652;
                v652 = reinterpret_cast<int4*>(v570 + v649);
                int4* v653;
                v653 = reinterpret_cast<int4*>(v639 + v648);
                assert("Pointer alignment check" && (unsigned long long)(v652) % 4l == 0 && (unsigned long long)(v653) % 4l == 0);
                *v653 = *v652;
                int4* v654;
                v654 = reinterpret_cast<int4*>(v572 + v649);
                int4* v655;
                v655 = reinterpret_cast<int4*>(v640 + v648);
                assert("Pointer alignment check" && (unsigned long long)(v654) % 4l == 0 && (unsigned long long)(v655) % 4l == 0);
                *v655 = *v654;
                int4* v656;
                v656 = reinterpret_cast<int4*>(v574 + v649);
                int4* v657;
                v657 = reinterpret_cast<int4*>(v641 + v648);
                assert("Pointer alignment check" && (unsigned long long)(v656) % 4l == 0 && (unsigned long long)(v657) % 4l == 0);
                *v657 = *v656;
                int4* v658;
                v658 = reinterpret_cast<int4*>(v576 + v649);
                int4* v659;
                v659 = reinterpret_cast<int4*>(v642 + v648);
                assert("Pointer alignment check" && (unsigned long long)(v658) % 4l == 0 && (unsigned long long)(v659) % 4l == 0);
                *v659 = *v658;
                int4* v660;
                v660 = reinterpret_cast<int4*>(v578 + v649);
                int4* v661;
                v661 = reinterpret_cast<int4*>(v643 + v648);
                assert("Pointer alignment check" && (unsigned long long)(v660) % 4l == 0 && (unsigned long long)(v661) % 4l == 0);
                *v661 = *v660;
                int4* v662;
                v662 = reinterpret_cast<int4*>(v580 + v649);
                int4* v663;
                v663 = reinterpret_cast<int4*>(v644 + v648);
                assert("Pointer alignment check" && (unsigned long long)(v662) % 4l == 0 && (unsigned long long)(v663) % 4l == 0);
                *v663 = *v662;
                v646 += 1l ;
            }
            int v664;
            v664 = 0l;
            while (while_method_6(v664)){
                int v666;
                v666 = 0l;
                while (while_method_0(v666)){
                    bool v668;
                    v668 = 0l <= v666;
                    bool v670;
                    if (v668){
                        bool v669;
                        v669 = v666 < 4l;
                        v670 = v669;
                    } else {
                        v670 = false;
                    }
                    bool v671;
                    v671 = v670 == false;
                    if (v671){
                        assert("The indices should be inside the range of the dimension." && v670);
                    } else {
                    }
                    bool v673;
                    v673 = 0l <= v612;
                    bool v675;
                    if (v673){
                        bool v674;
                        v674 = v612 < 1l;
                        v675 = v674;
                    } else {
                        v675 = false;
                    }
                    bool v676;
                    v676 = v675 == false;
                    if (v676){
                        assert("The indices should be inside the range of the dimension." && v675);
                    } else {
                    }
                    int v678;
                    v678 = v612 * 4l;
                    int v679;
                    v679 = v666 + v678;
                    bool v680;
                    v680 = 0l <= v664;
                    bool v682;
                    if (v680){
                        bool v681;
                        v681 = v664 < 1l;
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
                    int v685;
                    v685 = v664 * 4l;
                    int v686;
                    v686 = v679 + v685;
                    assert("Tensor range check" && 0 <= v664 && v664 < 1l);
                    assert("Tensor range check" && 0 <= v666 && v666 < 4l);
                    int v687;
                    v687 = 4l * v664;
                    int v688;
                    v688 = v687 + v666;
                    v645[v688] = v686;
                    v666 += 1l ;
                }
                v664 += 1l ;
            }
            bool v689;
            v689 = 0l <= v614;
            bool v690;
            v690 = v689 && v615;
            bool v691;
            v691 = v690 == false;
            if (v691){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v690);
            } else {
            }
            bool v693;
            v693 = 0l <= v613;
            bool v695;
            if (v693){
                bool v694;
                v694 = v613 < 256l;
                v695 = v694;
            } else {
                v695 = false;
            }
            bool v696;
            v696 = v695 == false;
            if (v696){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v695);
            } else {
            }
            bool v698;
            v698 = 0l <= v630;
            bool v699;
            v699 = v698 && v631;
            bool v700;
            v700 = v699 == false;
            if (v700){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v699);
            } else {
            }
            bool v702;
            v702 = 0l <= v629;
            bool v704;
            if (v702){
                bool v703;
                v703 = v629 < 16l;
                v704 = v703;
            } else {
                v704 = false;
            }
            bool v705;
            v705 = v704 == false;
            if (v705){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v704);
            } else {
            }
            int v707;
            v707 = v629 * 256l;
            int v708;
            v708 = v630 + v614;
            int v709;
            v709 = v707 + v613;
            bool v710[4l];
            int v711;
            v711 = 0l;
            while (while_method_6(v711)){
                int v713;
                v713 = 0l;
                while (while_method_0(v713)){
                    assert("Tensor range check" && 0 <= v711 && v711 < 1l);
                    assert("Tensor range check" && 0 <= v713 && v713 < 4l);
                    int v715;
                    v715 = 4l * v711;
                    int v716;
                    v716 = v715 + v713;
                    float v717;
                    v717 = v640[v716];
                    bool v718;
                    v718 = v717 == 0.0f;
                    bool v719;
                    v719 = v718 != true;
                    assert("Tensor range check" && 0 <= v711 && v711 < 1l);
                    assert("Tensor range check" && 0 <= v713 && v713 < 4l);
                    v710[v716] = v719;
                    v713 += 1l ;
                }
                v711 += 1l ;
            }
            bool v720;
            v720 = false;
            int v721;
            v721 = 0l;
            while (while_method_6(v721)){
                int v723;
                v723 = 0l;
                while (while_method_0(v723)){
                    assert("Tensor range check" && 0 <= v721 && v721 < 1l);
                    assert("Tensor range check" && 0 <= v723 && v723 < 4l);
                    int v725;
                    v725 = 4l * v721;
                    int v726;
                    v726 = v725 + v723;
                    bool v727;
                    v727 = v710[v726];
                    bool v728;
                    v728 = v720 || v727;
                    v720 = v728;
                    v723 += 1l ;
                }
                v721 += 1l ;
            }
            auto v729 = cooperative_groups::coalesced_threads();
            int v730;
            v730 = threadIdx.x;
            auto v731 = cooperative_groups::labeled_partition(v729,v730);
            Closure8 v732{};
            bool v733;
            v733 = cooperative_groups::reduce(v731, v720, v732);
            if (v733){
                float v734[4l];
                int v735;
                v735 = 0l;
                while (while_method_6(v735)){
                    int v737;
                    v737 = 0l;
                    while (while_method_0(v737)){
                        assert("Tensor range check" && 0 <= v735 && v735 < 1l);
                        assert("Tensor range check" && 0 <= v737 && v737 < 4l);
                        int v739;
                        v739 = 4l * v735;
                        int v740;
                        v740 = v739 + v737;
                        float v741;
                        v741 = v639[v740];
                        float v742;
                        v742 = v640[v740];
                        float v743;
                        v743 = v741 + v742;
                        bool v744;
                        v744 = 0.0f >= v743;
                        float v745;
                        if (v744){
                            v745 = 0.0f;
                        } else {
                            v745 = v743;
                        }
                        assert("Tensor range check" && 0 <= v735 && v735 < 1l);
                        assert("Tensor range check" && 0 <= v737 && v737 < 4l);
                        v734[v740] = v745;
                        v737 += 1l ;
                    }
                    v735 += 1l ;
                }
                float v746[4l];
                int v747;
                v747 = 0l;
                while (while_method_6(v747)){
                    int v749;
                    v749 = 0l;
                    while (while_method_0(v749)){
                        assert("Tensor range check" && 0 <= v747 && v747 < 1l);
                        assert("Tensor range check" && 0 <= v749 && v749 < 4l);
                        int v751;
                        v751 = 4l * v747;
                        int v752;
                        v752 = v751 + v749;
                        float v753;
                        v753 = v734[v752];
                        bool v754;
                        v754 = 0.0f >= v753;
                        float v755;
                        if (v754){
                            v755 = 0.0f;
                        } else {
                            v755 = v753;
                        }
                        assert("Tensor range check" && 0 <= v747 && v747 < 1l);
                        assert("Tensor range check" && 0 <= v749 && v749 < 4l);
                        v746[v752] = v755;
                        v749 += 1l ;
                    }
                    v747 += 1l ;
                }
                float v756;
                v756 = 0.0f;
                int v757;
                v757 = 0l;
                while (while_method_6(v757)){
                    int v759;
                    v759 = 0l;
                    while (while_method_0(v759)){
                        assert("Tensor range check" && 0 <= v757 && v757 < 1l);
                        assert("Tensor range check" && 0 <= v759 && v759 < 4l);
                        int v761;
                        v761 = 4l * v757;
                        int v762;
                        v762 = v761 + v759;
                        float v763;
                        v763 = v746[v762];
                        float v764;
                        v764 = v756 + v763;
                        v756 = v764;
                        v759 += 1l ;
                    }
                    v757 += 1l ;
                }
                auto v765 = cooperative_groups::coalesced_threads();
                int v766;
                v766 = threadIdx.x;
                auto v767 = cooperative_groups::labeled_partition(v765,v766);
                Closure1 v768{};
                float v769;
                v769 = cooperative_groups::reduce(v767, v756, v768);
                float v770[4l];
                int v771;
                v771 = 0l;
                while (while_method_6(v771)){
                    int v773;
                    v773 = 0l;
                    while (while_method_0(v773)){
                        assert("Tensor range check" && 0 <= v771 && v771 < 1l);
                        assert("Tensor range check" && 0 <= v773 && v773 < 4l);
                        int v775;
                        v775 = 4l * v771;
                        int v776;
                        v776 = v775 + v773;
                        float v777;
                        v777 = v746[v776];
                        bool v778;
                        v778 = v769 == 0.0f;
                        bool v779;
                        v779 = v778 != true;
                        float v781;
                        if (v779){
                            float v780;
                            v780 = v777 / v769;
                            v781 = v780;
                        } else {
                            v781 = 0.25f;
                        }
                        assert("Tensor range check" && 0 <= v771 && v771 < 1l);
                        assert("Tensor range check" && 0 <= v773 && v773 < 4l);
                        v770[v776] = v781;
                        v773 += 1l ;
                    }
                    v771 += 1l ;
                }
                float v782[4l];
                int v783;
                v783 = 0l;
                while (while_method_6(v783)){
                    int v785;
                    v785 = 0l;
                    while (while_method_0(v785)){
                        assert("Tensor range check" && 0 <= v783 && v783 < 1l);
                        assert("Tensor range check" && 0 <= v785 && v785 < 4l);
                        int v787;
                        v787 = 4l * v783;
                        int v788;
                        v788 = v787 + v785;
                        float v789;
                        v789 = v638[v788];
                        float v790;
                        v790 = v770[v788];
                        float v791;
                        v791 = v789 + v790;
                        assert("Tensor range check" && 0 <= v783 && v783 < 1l);
                        assert("Tensor range check" && 0 <= v785 && v785 < 4l);
                        v782[v788] = v791;
                        v785 += 1l ;
                    }
                    v783 += 1l ;
                }
                float v792[4l];
                int v793;
                v793 = 0l;
                while (while_method_6(v793)){
                    int v795;
                    v795 = 0l;
                    while (while_method_0(v795)){
                        assert("Tensor range check" && 0 <= v793 && v793 < 1l);
                        assert("Tensor range check" && 0 <= v795 && v795 < 4l);
                        int v797;
                        v797 = 4l * v793;
                        int v798;
                        v798 = v797 + v795;
                        float v799;
                        v799 = v782[v798];
                        float v800;
                        v800 = -v799;
                        bool v801;
                        v801 = v799 >= v800;
                        float v802;
                        if (v801){
                            v802 = v799;
                        } else {
                            v802 = v800;
                        }
                        assert("Tensor range check" && 0 <= v793 && v793 < 1l);
                        assert("Tensor range check" && 0 <= v795 && v795 < 4l);
                        v792[v798] = v802;
                        v795 += 1l ;
                    }
                    v793 += 1l ;
                }
                float v803;
                v803 = 0.0f;
                int v804;
                v804 = 0l;
                while (while_method_6(v804)){
                    int v806;
                    v806 = 0l;
                    while (while_method_0(v806)){
                        assert("Tensor range check" && 0 <= v804 && v804 < 1l);
                        assert("Tensor range check" && 0 <= v806 && v806 < 4l);
                        int v808;
                        v808 = 4l * v804;
                        int v809;
                        v809 = v808 + v806;
                        float v810;
                        v810 = v792[v809];
                        float v811;
                        v811 = v803 + v810;
                        v803 = v811;
                        v806 += 1l ;
                    }
                    v804 += 1l ;
                }
                auto v812 = cooperative_groups::coalesced_threads();
                int v813;
                v813 = threadIdx.x;
                auto v814 = cooperative_groups::labeled_partition(v812,v813);
                float v815;
                v815 = cooperative_groups::reduce(v814, v803, v768);
                bool v816;
                v816 = v815 > 100.0f;
                float v818;
                if (v816){
                    float v817;
                    v817 = 100.0f / v815;
                    v818 = v817;
                } else {
                    v818 = 1.0f;
                }
                float v819[4l];
                int v820;
                v820 = 0l;
                while (while_method_6(v820)){
                    int v822;
                    v822 = 0l;
                    while (while_method_0(v822)){
                        assert("Tensor range check" && 0 <= v820 && v820 < 1l);
                        assert("Tensor range check" && 0 <= v822 && v822 < 4l);
                        int v824;
                        v824 = 4l * v820;
                        int v825;
                        v825 = v824 + v822;
                        float v826;
                        v826 = v792[v825];
                        float v827;
                        v827 = v818 * v826;
                        assert("Tensor range check" && 0 <= v820 && v820 < 1l);
                        assert("Tensor range check" && 0 <= v822 && v822 < 4l);
                        v819[v825] = v827;
                        v822 += 1l ;
                    }
                    v820 += 1l ;
                }
                float v828[4l];
                float v829[4l];
                int v830;
                v830 = 0l;
                while (while_method_6(v830)){
                    int v832;
                    v832 = 0l;
                    while (while_method_0(v832)){
                        assert("Tensor range check" && 0 <= v830 && v830 < 1l);
                        assert("Tensor range check" && 0 <= v832 && v832 < 4l);
                        int v834;
                        v834 = 4l * v830;
                        int v835;
                        v835 = v834 + v832;
                        float v836;
                        v836 = v638[v835];
                        float v837;
                        v837 = v639[v835];
                        float v838;
                        v838 = v640[v835];
                        float v839;
                        v839 = v641[v835];
                        float v840;
                        v840 = v642[v835];
                        float v841;
                        v841 = v643[v835];
                        float v842;
                        v842 = v644[v835];
                        float v843;
                        v843 = v839 + v841;
                        float v844;
                        v844 = v840 + v842;
                        assert("Tensor range check" && 0 <= v830 && v830 < 1l);
                        assert("Tensor range check" && 0 <= v832 && v832 < 4l);
                        v828[v835] = v843;
                        v829[v835] = v844;
                        v832 += 1l ;
                    }
                    v830 += 1l ;
                }
                int v845;
                v845 = 0l;
                while (while_method_6(v845)){
                    int v847;
                    v847 = 0l;
                    while (while_method_0(v847)){
                        assert("Tensor range check" && 0 <= v845 && v845 < 1l);
                        assert("Tensor range check" && 0 <= v847 && v847 < 4l);
                        int v849;
                        v849 = 4l * v845;
                        int v850;
                        v850 = v849 + v847;
                        float v851;
                        v851 = v819[v850];
                        float v852;
                        v852 = v734[v850];
                        float v853;
                        v853 = v828[v850];
                        float v854;
                        v854 = v829[v850];
                        assert("Tensor range check" && 0 <= v845 && v845 < 1l);
                        assert("Tensor range check" && 0 <= v847 && v847 < 4l);
                        v638[v850] = v851;
                        v639[v850] = v852;
                        v640[v850] = 0.0f;
                        v641[v850] = v853;
                        v642[v850] = v854;
                        v643[v850] = 0.0f;
                        v644[v850] = 0.0f;
                        v847 += 1l ;
                    }
                    v845 += 1l ;
                }
            } else {
            }
            assert("Tensor range check" && 0 <= v630 && v630 < 4l);
            assert("Tensor range check" && 0 <= v629 && v629 < 16l);
            int v855;
            v855 = 0l;
            while (while_method_6(v855)){
                assert("Tensor range check" && 0 <= v855 && v855 < 1l);
                int v857;
                v857 = 4l * v855;
                int v858;
                v858 = v857 + v637;
                assert("Tensor range check" && 0 <= v855 && v855 < 1l);
                int4* v859;
                v859 = reinterpret_cast<int4*>(v638 + v857);
                int4* v860;
                v860 = reinterpret_cast<int4*>(v568 + v858);
                assert("Pointer alignment check" && (unsigned long long)(v859) % 4l == 0 && (unsigned long long)(v860) % 4l == 0);
                *v860 = *v859;
                int4* v861;
                v861 = reinterpret_cast<int4*>(v639 + v857);
                int4* v862;
                v862 = reinterpret_cast<int4*>(v570 + v858);
                assert("Pointer alignment check" && (unsigned long long)(v861) % 4l == 0 && (unsigned long long)(v862) % 4l == 0);
                *v862 = *v861;
                int4* v863;
                v863 = reinterpret_cast<int4*>(v640 + v857);
                int4* v864;
                v864 = reinterpret_cast<int4*>(v572 + v858);
                assert("Pointer alignment check" && (unsigned long long)(v863) % 4l == 0 && (unsigned long long)(v864) % 4l == 0);
                *v864 = *v863;
                int4* v865;
                v865 = reinterpret_cast<int4*>(v641 + v857);
                int4* v866;
                v866 = reinterpret_cast<int4*>(v574 + v858);
                assert("Pointer alignment check" && (unsigned long long)(v865) % 4l == 0 && (unsigned long long)(v866) % 4l == 0);
                *v866 = *v865;
                int4* v867;
                v867 = reinterpret_cast<int4*>(v642 + v857);
                int4* v868;
                v868 = reinterpret_cast<int4*>(v576 + v858);
                assert("Pointer alignment check" && (unsigned long long)(v867) % 4l == 0 && (unsigned long long)(v868) % 4l == 0);
                *v868 = *v867;
                int4* v869;
                v869 = reinterpret_cast<int4*>(v643 + v857);
                int4* v870;
                v870 = reinterpret_cast<int4*>(v578 + v858);
                assert("Pointer alignment check" && (unsigned long long)(v869) % 4l == 0 && (unsigned long long)(v870) % 4l == 0);
                *v870 = *v869;
                int4* v871;
                v871 = reinterpret_cast<int4*>(v644 + v857);
                int4* v872;
                v872 = reinterpret_cast<int4*>(v580 + v858);
                assert("Pointer alignment check" && (unsigned long long)(v871) % 4l == 0 && (unsigned long long)(v872) % 4l == 0);
                *v872 = *v871;
                v855 += 1l ;
            }
            v624 += 24l ;
        }
        v561.sync() ;
        v24 += 1l ;
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
options.append('--maxrregcount=256')
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
    v53.max_dynamic_shared_size_bytes = 81920 
    v53((24,),(256,),(v9, v8),shared_mem=81920)
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
