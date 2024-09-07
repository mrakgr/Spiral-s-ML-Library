kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <mma.h>
using namespace nvcuda;
#include <cooperative_groups.h>
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
__device__ int f_1(unsigned char * v0);
__device__ void f_2(unsigned char * v0);
__device__ Union0 f_0(unsigned char * v0);
struct Union1;
struct Union3;
struct Union4;
struct Union2;
struct Union6;
struct Union5;
struct Union7;
struct Tuple0;
__device__ unsigned int loop_5(unsigned int v0, curandStatePhilox4_32_10_t & v1);
__device__ Tuple0 draw_card_4(curandStatePhilox4_32_10_t & v0, unsigned int v1);
struct Tuple1;
struct Union8;
struct Union9;
__device__ void method_6(float * v0, int v1, float * v2, int v3, float * v4, int v5);
__device__ void method_7(unsigned int * v0, int v1, float * v2);
struct Tuple2;
struct Tuple3;
struct Tuple4;
struct Tuple5;
__device__ Tuple2 method_8(curandStatePhilox4_32_10_t & v0, int * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int v9, int v10);
__device__ float method_9(int * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int v8, int v9, int v10);
struct Union10;
__device__ int int_range_10(int v0, int v1, curandStatePhilox4_32_10_t & v2);
struct Union11;
__device__ int tag_12(Union3 v0);
__device__ bool is_pair_13(int v0, int v1);
__device__ Tuple1 order_14(int v0, int v1);
__device__ Union11 compare_hands_11(Union6 v0, bool v1, static_array<Union3,2l> v2, int v3, static_array<int,2l> v4, int v5);
__device__ static_array<float,2l> method_3(unsigned char * v0, unsigned char * v1, unsigned int & v2, static_array_list<Union2,32l> & v3, static_array<Union1,2l> & v4, curandStatePhilox4_32_10_t & v5, Union5 v6);
__device__ float method_16(int * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int v8, int v9, int v10);
__device__ static_array<float,2l> method_15(unsigned char * v0, unsigned char * v1, unsigned int & v2, static_array_list<Union2,32l> & v3, static_array<Union1,2l> & v4, curandStatePhilox4_32_10_t & v5, Union5 v6);
struct Union0_0 { // StartTraining
};
struct Union0 {
    union {
        Union0_0 case0; // StartTraining
    };
    unsigned char tag{255};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // StartTraining
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // StartTraining
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // StartTraining
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // StartTraining
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
                case 0: this->case0 = std::move(x.case0); break; // StartTraining
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // StartTraining
        }
        this->tag = 255;
    }
};
struct Union1_0 { // Computer
};
struct Union1_1 { // Random
};
struct Union1 {
    union {
        Union1_0 case0; // Computer
        Union1_1 case1; // Random
    };
    unsigned char tag{255};
    __device__ Union1() {}
    __device__ Union1(Union1_0 t) : tag(0), case0(t) {} // Computer
    __device__ Union1(Union1_1 t) : tag(1), case1(t) {} // Random
    __device__ Union1(Union1 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(x.case0); break; // Computer
            case 1: new (&this->case1) Union1_1(x.case1); break; // Random
        }
    }
    __device__ Union1(Union1 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // Computer
            case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // Random
        }
    }
    __device__ Union1 & operator=(Union1 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Computer
                case 1: this->case1 = x.case1; break; // Random
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
                case 0: this->case0 = std::move(x.case0); break; // Computer
                case 1: this->case1 = std::move(x.case1); break; // Random
            }
        } else {
            this->~Union1();
            new (this) Union1{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union1() {
        switch(this->tag){
            case 0: this->case0.~Union1_0(); break; // Computer
            case 1: this->case1.~Union1_1(); break; // Random
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
struct Union4_0 { // Call
};
struct Union4_1 { // Fold
};
struct Union4_2 { // Raise
};
struct Union4 {
    union {
        Union4_0 case0; // Call
        Union4_1 case1; // Fold
        Union4_2 case2; // Raise
    };
    unsigned char tag{255};
    __device__ Union4() {}
    __device__ Union4(Union4_0 t) : tag(0), case0(t) {} // Call
    __device__ Union4(Union4_1 t) : tag(1), case1(t) {} // Fold
    __device__ Union4(Union4_2 t) : tag(2), case2(t) {} // Raise
    __device__ Union4(Union4 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(x.case0); break; // Call
            case 1: new (&this->case1) Union4_1(x.case1); break; // Fold
            case 2: new (&this->case2) Union4_2(x.case2); break; // Raise
        }
    }
    __device__ Union4(Union4 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(std::move(x.case0)); break; // Call
            case 1: new (&this->case1) Union4_1(std::move(x.case1)); break; // Fold
            case 2: new (&this->case2) Union4_2(std::move(x.case2)); break; // Raise
        }
    }
    __device__ Union4 & operator=(Union4 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Call
                case 1: this->case1 = x.case1; break; // Fold
                case 2: this->case2 = x.case2; break; // Raise
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
                case 0: this->case0 = std::move(x.case0); break; // Call
                case 1: this->case1 = std::move(x.case1); break; // Fold
                case 2: this->case2 = std::move(x.case2); break; // Raise
            }
        } else {
            this->~Union4();
            new (this) Union4{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union4() {
        switch(this->tag){
            case 0: this->case0.~Union4_0(); break; // Call
            case 1: this->case1.~Union4_1(); break; // Fold
            case 2: this->case2.~Union4_2(); break; // Raise
        }
        this->tag = 255;
    }
};
struct Union2_0 { // CommunityCardIs
    Union3 v0;
    __device__ Union2_0(Union3 t0) : v0(t0) {}
    __device__ Union2_0() = delete;
};
struct Union2_1 { // PlayerAction
    Union4 v1;
    int v0;
    __device__ Union2_1(int t0, Union4 t1) : v0(t0), v1(t1) {}
    __device__ Union2_1() = delete;
};
struct Union2_2 { // PlayerGotCard
    Union3 v1;
    int v0;
    __device__ Union2_2(int t0, Union3 t1) : v0(t0), v1(t1) {}
    __device__ Union2_2() = delete;
};
struct Union2_3 { // Showdown
    static_array<Union3,2l> v0;
    int v1;
    int v2;
    __device__ Union2_3(static_array<Union3,2l> t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
    __device__ Union2_3() = delete;
};
struct Union2 {
    union {
        Union2_0 case0; // CommunityCardIs
        Union2_1 case1; // PlayerAction
        Union2_2 case2; // PlayerGotCard
        Union2_3 case3; // Showdown
    };
    unsigned char tag{255};
    __device__ Union2() {}
    __device__ Union2(Union2_0 t) : tag(0), case0(t) {} // CommunityCardIs
    __device__ Union2(Union2_1 t) : tag(1), case1(t) {} // PlayerAction
    __device__ Union2(Union2_2 t) : tag(2), case2(t) {} // PlayerGotCard
    __device__ Union2(Union2_3 t) : tag(3), case3(t) {} // Showdown
    __device__ Union2(Union2 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(x.case0); break; // CommunityCardIs
            case 1: new (&this->case1) Union2_1(x.case1); break; // PlayerAction
            case 2: new (&this->case2) Union2_2(x.case2); break; // PlayerGotCard
            case 3: new (&this->case3) Union2_3(x.case3); break; // Showdown
        }
    }
    __device__ Union2(Union2 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(std::move(x.case0)); break; // CommunityCardIs
            case 1: new (&this->case1) Union2_1(std::move(x.case1)); break; // PlayerAction
            case 2: new (&this->case2) Union2_2(std::move(x.case2)); break; // PlayerGotCard
            case 3: new (&this->case3) Union2_3(std::move(x.case3)); break; // Showdown
        }
    }
    __device__ Union2 & operator=(Union2 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // CommunityCardIs
                case 1: this->case1 = x.case1; break; // PlayerAction
                case 2: this->case2 = x.case2; break; // PlayerGotCard
                case 3: this->case3 = x.case3; break; // Showdown
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
                case 0: this->case0 = std::move(x.case0); break; // CommunityCardIs
                case 1: this->case1 = std::move(x.case1); break; // PlayerAction
                case 2: this->case2 = std::move(x.case2); break; // PlayerGotCard
                case 3: this->case3 = std::move(x.case3); break; // Showdown
            }
        } else {
            this->~Union2();
            new (this) Union2{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union2() {
        switch(this->tag){
            case 0: this->case0.~Union2_0(); break; // CommunityCardIs
            case 1: this->case1.~Union2_1(); break; // PlayerAction
            case 2: this->case2.~Union2_2(); break; // PlayerGotCard
            case 3: this->case3.~Union2_3(); break; // Showdown
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
    Union4 v6;
    int v3;
    int v5;
    bool v1;
    __device__ Union5_3(Union6 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5, Union4 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
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
struct Union7_0 { // None
};
struct Union7_1 { // Some
    Union5 v0;
    __device__ Union7_1(Union5 t0) : v0(t0) {}
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
    Union3 v0;
    unsigned int v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(Union3 t0, unsigned int t1) : v0(t0), v1(t1) {}
};
struct Tuple1 {
    int v0;
    int v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(int t0, int t1) : v0(t0), v1(t1) {}
};
struct Union8_0 { // C1of2
    Union4 v0;
    __device__ Union8_0(Union4 t0) : v0(t0) {}
    __device__ Union8_0() = delete;
};
struct Union8_1 { // C2of2
    Union3 v0;
    __device__ Union8_1(Union3 t0) : v0(t0) {}
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
__device__ int f_1(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ void f_2(unsigned char * v0){
    return ;
}
__device__ Union0 f_0(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    switch (v1) {
        case 0: {
            f_2(v2);
            return Union0{Union0_0{}};
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
    v1 = v0 < 1024l;
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
            Union5 v1 = v0.case1.v0;
            return true;
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ unsigned int loop_5(unsigned int v0, curandStatePhilox4_32_10_t & v1){
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
        return loop_5(v0, v1);
    }
}
__device__ Tuple0 draw_card_4(curandStatePhilox4_32_10_t & v0, unsigned int v1){
    int v2;
    v2 = __popc(v1);
    unsigned int v3;
    v3 = (unsigned int)v2;
    unsigned int v4;
    v4 = loop_5(v3, v0);
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
    Union3 v31;
    if (v13){
        v31 = Union3{Union3_1{}};
    } else {
        bool v15;
        v15 = 1ul == v12;
        if (v15){
            v31 = Union3{Union3_1{}};
        } else {
            bool v17;
            v17 = 2ul == v12;
            if (v17){
                v31 = Union3{Union3_2{}};
            } else {
                bool v19;
                v19 = 3ul == v12;
                if (v19){
                    v31 = Union3{Union3_2{}};
                } else {
                    bool v21;
                    v21 = 4ul == v12;
                    if (v21){
                        v31 = Union3{Union3_0{}};
                    } else {
                        bool v23;
                        v23 = 5ul == v12;
                        if (v23){
                            v31 = Union3{Union3_0{}};
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
    v1 = v0 < 65536l;
    return v1;
}
__device__ inline bool while_method_5(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_8(int v0){
    bool v1;
    v1 = v0 < 8l;
    return v1;
}
__device__ void method_6(float * v0, int v1, float * v2, int v3, float * v4, int v5){
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
    v20 = v19 < 2l;
    bool v21;
    v21 = v20 == false;
    if (v21){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v20);
    } else {
    }
    assert("Tensor range check" && 0 <= v19 && v19 < 2l);
    assert("Tensor range check" && 0 <= v18 && v18 < 8l);
    int v23;
    v23 = 16l * v18;
    int v24;
    v24 = 8704l * v19;
    int v25;
    v25 = v24 + v23;
    float * v26;
    v26 = v11+v25;
    assert("Tensor range check" && 0 <= v19 && v19 < 2l);
    int v28;
    v28 = 4352l * v19;
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
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v60[4l];
    int v61;
    v61 = 0l;
    while (while_method_6(v61)){
        int v63;
        v63 = 0l;
        while (while_method_7(v63)){
            assert("Tensor range check" && 0 <= v61 && v61 < 4l);
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
            while (while_method_6(v71)){
                int v73;
                v73 = 0l;
                #pragma unroll
                while (while_method_7(v73)){
                    assert("Tensor range check" && 0 <= v71 && v71 < 4l);
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
                assert("Tensor range check" && 0 <= v61 && v61 < 4l);
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
                v95 = v94 < 32l;
                bool v96;
                v96 = v95 == false;
                if (v96){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v95);
                } else {
                }
                assert("Tensor range check" && 0 <= v94 && v94 < 32l);
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
                while (while_method_6(v107)){
                    int v109;
                    v109 = 0l;
                    #pragma unroll
                    while (while_method_7(v109)){
                        assert("Tensor range check" && 0 <= v107 && v107 < 4l);
                        assert("Tensor range check" && 0 <= v109 && v109 < 1l);
                        int v111;
                        v111 = 64l * v109;
                        int v112;
                        v112 = 2176l * v107;
                        int v113;
                        v113 = v112 + v111;
                        int v114;
                        v114 = 4096l * v107;
                        int v115;
                        v115 = v114 + v111;
                        float v116[4l];
                        int v117;
                        v117 = 0l;
                        #pragma unroll
                        while (while_method_6(v117)){
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
                v130 = v129 < 32l;
                bool v131;
                v131 = v130 == false;
                if (v131){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v130);
                } else {
                }
                assert("Tensor range check" && 0 <= v129 && v129 < 32l);
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
                while (while_method_6(v142)){
                    int v144;
                    v144 = 0l;
                    #pragma unroll
                    while (while_method_7(v144)){
                        assert("Tensor range check" && 0 <= v142 && v142 < 4l);
                        assert("Tensor range check" && 0 <= v144 && v144 < 1l);
                        int v146;
                        v146 = 64l * v144;
                        int v147;
                        v147 = 2176l * v142;
                        int v148;
                        v148 = v147 + v146;
                        int v149;
                        v149 = 4096l * v142;
                        int v150;
                        v150 = v149 + v146;
                        float v151[4l];
                        int v152;
                        v152 = 0l;
                        #pragma unroll
                        while (while_method_6(v152)){
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
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v159[32l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v160[8l];
                int v161;
                v161 = 0l;
                #pragma unroll
                while (while_method_6(v161)){
                    int v163;
                    v163 = 0l;
                    #pragma unroll
                    while (while_method_8(v163)){
                        assert("Tensor range check" && 0 <= v161 && v161 < 4l);
                        assert("Tensor range check" && 0 <= v163 && v163 < 8l);
                        int v165;
                        v165 = 8l * v161;
                        int v166;
                        v166 = v165 + v163;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v167 = v159[v166];
                        assert("Tensor range check" && 0 <= v161 && v161 < 4l);
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
                while (while_method_7(v192)){
                    int v194;
                    v194 = 0l;
                    #pragma unroll
                    while (while_method_8(v194)){
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
                while (while_method_6(v223)){
                    int v225;
                    v225 = 0l;
                    #pragma unroll
                    while (while_method_7(v225)){
                        int v227;
                        v227 = 0l;
                        #pragma unroll
                        while (while_method_8(v227)){
                            assert("Tensor range check" && 0 <= v223 && v223 < 4l);
                            assert("Tensor range check" && 0 <= v225 && v225 < 1l);
                            int v229;
                            v229 = v223 + v225;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v230 = v60[v229];
                            assert("Tensor range check" && 0 <= v223 && v223 < 4l);
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
            while (while_method_6(v237)){
                int v239;
                v239 = 0l;
                #pragma unroll
                while (while_method_7(v239)){
                    assert("Tensor range check" && 0 <= v237 && v237 < 4l);
                    assert("Tensor range check" && 0 <= v239 && v239 < 1l);
                    int v241;
                    v241 = v237 + v239;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v242 = v60[v241];
                    assert("Tensor range check" && 0 <= v237 && v237 < 4l);
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
            v254 = v253 < 16l;
            bool v255;
            v255 = v254 == false;
            if (v255){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v254);
            } else {
            }
            assert("Tensor range check" && 0 <= v253 && v253 < 16l);
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
            while (while_method_8(v266)){
                int v268;
                v268 = 0l;
                #pragma unroll
                while (while_method_7(v268)){
                    assert("Tensor range check" && 0 <= v266 && v266 < 8l);
                    assert("Tensor range check" && 0 <= v268 && v268 < 1l);
                    int v270;
                    v270 = 128l * v268;
                    int v271;
                    v271 = 2048l * v266;
                    int v272;
                    v272 = v271 + v270;
                    int v273;
                    v273 = 2176l * v266;
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
__device__ inline bool while_method_9(int v0){
    bool v1;
    v1 = v0 < 32l;
    return v1;
}
__device__ void method_7(unsigned int * v0, int v1, float * v2){
    int v3;
    v3 = blockIdx.x;
    assert("Tensor range check" && 0 <= v3 && v3 < 24l);
    int v4;
    v4 = 65536l * v3;
    int v5;
    v5 = blockIdx.x;
    assert("Tensor range check" && 0 <= v5 && v5 < 24l);
    int v6;
    v6 = 512l * v5;
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
    v14 = v13 < 16l;
    bool v15;
    v15 = v14 == false;
    if (v15){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v14);
    } else {
    }
    assert("Tensor range check" && 0 <= v13 && v13 < 16l);
    assert("Tensor range check" && 0 <= v12 && v12 < 32l);
    int v17;
    v17 = 4l * v12;
    int v18;
    v18 = v17 + v4;
    int v19;
    v19 = 128l * v13;
    int v20;
    v20 = v19 + v18;
    assert("Tensor range check" && 0 <= v13 && v13 < 16l);
    int v21;
    v21 = v13 + v7;
    int v22;
    v22 = 0l;
    while (while_method_9(v22)){
        assert("Tensor range check" && 0 <= v22 && v22 < 32l);
        int v24;
        v24 = 2048l * v22;
        int v25;
        v25 = v24 + v20;
        float v26[4l];
        int v27[4l];
        int v28;
        v28 = 0l;
        while (while_method_7(v28)){
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
        while (while_method_7(v35)){
            int v37;
            v37 = 0l;
            while (while_method_6(v37)){
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
        v69 = v22 * 16l;
        int v70;
        v70 = v69 + v13;
        unsigned int v71[4l];
        int v72;
        v72 = 0l;
        while (while_method_7(v72)){
            int v74;
            v74 = 0l;
            while (while_method_6(v74)){
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
        while (while_method_7(v84)){
            int v86;
            v86 = 0l;
            while (while_method_6(v86)){
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
        v99 = 16l * v22;
        int v100;
        v100 = v99 + v21;
        v0[v100] = v98;
        v22 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return ;
}
__device__ Tuple2 method_8(curandStatePhilox4_32_10_t & v0, int * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int v9, int v10){
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
    v20 = 512ull * v19;
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
    v30 = v29 + 2048ull;
    unsigned long long v31;
    v31 = v30 + 16ull;
    unsigned long long v32;
    v32 = v31 - 1ull;
    unsigned long long v33;
    v33 = v32 % 16ull;
    unsigned long long v34;
    v34 = v32 - v33;
    unsigned long long v35;
    v35 = v34 + 2048ull;
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
    assert("Tensor range check" && 0 <= v51 && v51 < 512l);
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
    v56 = v51 < 512l;
    bool v57;
    v57 = v56 == false;
    if (v57){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v56);
    } else {
    }
    assert("Tensor range check" && 0 <= v51 && v51 < 512l);
    int v59;
    v59 = 0l;
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
        v69 = v59 * 512l;
        int v70;
        v70 = v69 + v51;
        assert("Tensor range check" && 0 <= v59 && v59 < 1l);
        int v71;
        v71 = 512l * v59;
        int v72;
        v72 = v71 + v51;
        float * v73;
        v73 = v43[v72];
        float * v74;
        v74 = v45[v72];
        int v75;
        v75 = blockIdx.x;
        int v76;
        v76 = v75 * 512l;
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
        while (while_method_7(v82)){
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
        while (while_method_7(v90)){
            int v92;
            v92 = 0l;
            while (while_method_6(v92)){
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
        while (while_method_7(v116)){
            int v118;
            v118 = 0l;
            while (while_method_6(v118)){
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
        while (while_method_7(v126)){
            int v128;
            v128 = 0l;
            while (while_method_6(v128)){
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
        while (while_method_7(v138)){
            int v140;
            v140 = 0l;
            while (while_method_6(v140)){
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
        while (while_method_7(v152)){
            int v154;
            v154 = 0l;
            while (while_method_6(v154)){
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
        while (while_method_7(v161)){
            int v163;
            v163 = 0l;
            while (while_method_6(v163)){
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
        while (while_method_7(v177)){
            int v179;
            v179 = 0l;
            while (while_method_6(v179)){
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
        while (while_method_7(v193)){
            assert("Tensor range check" && 0 <= v193 && v193 < 1l);
            int v195;
            v195 = 4l * v193;
            assert("Tensor range check" && 0 <= v193 && v193 < 1l);
            int v196; float v197;
            Tuple3 tmp4 = Tuple3{0l, 0.0f};
            v196 = tmp4.v0; v197 = tmp4.v1;
            while (while_method_6(v196)){
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
            while (while_method_6(v212)){
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
        while (while_method_7(v221)){
            int v223;
            v223 = 0l;
            while (while_method_6(v223)){
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
        while (while_method_7(v232)){
            int v234;
            v234 = 0l;
            while (while_method_6(v234)){
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
        while (while_method_7(v258)){
            int v260;
            v260 = 0l;
            while (while_method_6(v260)){
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
        while (while_method_7(v268)){
            int v270;
            v270 = 0l;
            while (while_method_6(v270)){
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
        while (while_method_7(v288)){
            int v290;
            v290 = 0l;
            while (while_method_6(v290)){
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
        while (while_method_7(v303)){
            int v305;
            v305 = 0l;
            while (while_method_6(v305)){
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
        while (while_method_7(v328)){
            int v330;
            v330 = 0l;
            while (while_method_6(v330)){
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
        while (while_method_7(v338)){
            int v340;
            v340 = 0l;
            while (while_method_6(v340)){
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
        while (while_method_7(v350)){
            int v352;
            v352 = 0l;
            while (while_method_6(v352)){
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
        while (while_method_7(v363)){
            int v365;
            v365 = 0l;
            while (while_method_6(v365)){
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
        while (while_method_7(v372)){
            int v374;
            v374 = 0l;
            while (while_method_6(v374)){
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
        while (while_method_7(v387)){
            int v389;
            v389 = 0l;
            while (while_method_6(v389)){
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
        while (while_method_7(v403)){
            int v405;
            v405 = 0l;
            while (while_method_6(v405)){
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
        while (while_method_7(v427)){
            assert("Tensor range check" && 0 <= v427 && v427 < 1l);
            assert("Tensor range check" && 0 <= v427 && v427 < 1l);
            v427 += 1l ;
        }
        assert("Tensor range check" && 0 <= v70 && v70 < 512l);
        v47[v70] = v421;
        v49[v70] = v323;
        v59 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v51 && v51 < 512l);
    float v429;
    v429 = v47[v51];
    int v430;
    v430 = v49[v51];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return Tuple2{v429, v430};
}
__device__ float method_9(int * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int v8, int v9, int v10){
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
    v18 = 512ull * v17;
    unsigned long long v19;
    v19 = 2048ull + v18;
    unsigned long long v20;
    v20 = v19 + 16ull;
    unsigned long long v21;
    v21 = v20 - 1ull;
    unsigned long long v22;
    v22 = v21 % 16ull;
    unsigned long long v23;
    v23 = v21 - v22;
    unsigned long long v24;
    v24 = v23 + 2048ull;
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
    v34 = reinterpret_cast<float * *>(&v28[2048ull]);
    float * v36;
    v36 = reinterpret_cast<float *>(&v28[v23]);
    int v38;
    v38 = threadIdx.x;
    assert("Tensor range check" && 0 <= v38 && v38 < 512l);
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
    v43 = v38 < 512l;
    bool v44;
    v44 = v43 == false;
    if (v44){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v43);
    } else {
    }
    assert("Tensor range check" && 0 <= v38 && v38 < 512l);
    int v46;
    v46 = 0l;
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
        v56 = v46 * 512l;
        int v57;
        v57 = v56 + v38;
        assert("Tensor range check" && 0 <= v46 && v46 < 1l);
        int v58;
        v58 = 512l * v46;
        int v59;
        v59 = v58 + v38;
        int v60;
        v60 = v32[v59];
        float * v61;
        v61 = v34[v59];
        int v62;
        v62 = blockIdx.x;
        int v63;
        v63 = v62 * 512l;
        int v64;
        v64 = v63 + v57;
        assert("Tensor range check" && 0 <= v42 && v42 < 1l);
        int v65;
        v65 = 4l * v42;
        float v66[4l];
        int v67[4l];
        int v68;
        v68 = 0l;
        while (while_method_7(v68)){
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
        while (while_method_7(v74)){
            int v76;
            v76 = 0l;
            while (while_method_6(v76)){
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
        while (while_method_7(v100)){
            int v102;
            v102 = 0l;
            while (while_method_6(v102)){
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
        while (while_method_7(v110)){
            int v112;
            v112 = 0l;
            while (while_method_6(v112)){
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
        while (while_method_7(v122)){
            int v124;
            v124 = 0l;
            while (while_method_6(v124)){
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
        while (while_method_7(v136)){
            int v138;
            v138 = 0l;
            while (while_method_6(v138)){
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
        while (while_method_7(v145)){
            int v147;
            v147 = 0l;
            while (while_method_6(v147)){
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
        while (while_method_7(v161)){
            int v163;
            v163 = 0l;
            while (while_method_6(v163)){
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
        while (while_method_7(v177)){
            int v179;
            v179 = 0l;
            while (while_method_6(v179)){
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
        while (while_method_7(v201)){
            assert("Tensor range check" && 0 <= v201 && v201 < 1l);
            assert("Tensor range check" && 0 <= v201 && v201 < 1l);
            v201 += 1l ;
        }
        assert("Tensor range check" && 0 <= v57 && v57 < 512l);
        v36[v57] = v195;
        v46 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v38 && v38 < 512l);
    float v203;
    v203 = v36[v38];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return v203;
}
__device__ int int_range_10(int v0, int v1, curandStatePhilox4_32_10_t & v2){
    int v3;
    v3 = v0 - v1;
    unsigned int v4;
    v4 = (unsigned int)v3;
    unsigned int v5;
    v5 = loop_5(v4, v2);
    unsigned int v6;
    v6 = (unsigned int)v1;
    unsigned int v7;
    v7 = v5 + v6;
    int v8;
    v8 = (int)v7;
    return v8;
}
__device__ int tag_12(Union3 v0){
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
__device__ bool is_pair_13(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
__device__ Tuple1 order_14(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple1{v1, v0};
    } else {
        return Tuple1{v0, v1};
    }
}
__device__ Union11 compare_hands_11(Union6 v0, bool v1, static_array<Union3,2l> v2, int v3, static_array<int,2l> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            __trap();
            break;
        }
        case 1: { // Some
            Union3 v7 = v0.case1.v0;
            int v8;
            v8 = tag_12(v7);
            Union3 v9;
            v9 = v2[0l];
            int v11;
            v11 = tag_12(v9);
            Union3 v12;
            v12 = v2[1l];
            int v14;
            v14 = tag_12(v12);
            bool v15;
            v15 = is_pair_13(v8, v11);
            bool v16;
            v16 = is_pair_13(v8, v14);
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
                    Tuple1 tmp24 = order_14(v8, v11);
                    v27 = tmp24.v0; v28 = tmp24.v1;
                    int v29; int v30;
                    Tuple1 tmp25 = order_14(v8, v14);
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
__device__ static_array<float,2l> method_3(unsigned char * v0, unsigned char * v1, unsigned int & v2, static_array_list<Union2,32l> & v3, static_array<Union1,2l> & v4, curandStatePhilox4_32_10_t & v5, Union5 v6){
    static_array<float,2l> v7;
    static_array_list<Union2,32l> & v9 = v3;
    Union7 v10;
    v10 = Union7{Union7_1{v6}};
    Union7 v11;
    v11 = v10;
    while (while_method_2(v11)){
        Union7 v712;
        switch (v11.tag) {
            case 0: { // None
                v712 = Union7{Union7_0{}};
                break;
            }
            case 1: { // Some
                Union5 v13 = v11.case1.v0;
                switch (v13.tag) {
                    case 0: { // ChanceCommunityCard
                        Union6 v663 = v13.case0.v0; bool v664 = v13.case0.v1; static_array<Union3,2l> v665 = v13.case0.v2; int v666 = v13.case0.v3; static_array<int,2l> v667 = v13.case0.v4; int v668 = v13.case0.v5;
                        unsigned int v669 = v2;
                        Union3 v670; unsigned int v671;
                        Tuple0 tmp0 = draw_card_4(v5, v669);
                        v670 = tmp0.v0; v671 = tmp0.v1;
                        v2 = v671;
                        Union2 v672;
                        v672 = Union2{Union2_0{v670}};
                        v9.push(v672);
                        int v673;
                        v673 = 2l;
                        int v674; int v675;
                        Tuple1 tmp1 = Tuple1{0l, 0l};
                        v674 = tmp1.v0; v675 = tmp1.v1;
                        while (while_method_3(v674)){
                            int v677;
                            v677 = v667[v674];
                            bool v679;
                            v679 = v675 >= v677;
                            int v680;
                            if (v679){
                                v680 = v675;
                            } else {
                                v680 = v677;
                            }
                            v675 = v680;
                            v674 += 1l ;
                        }
                        static_array<int,2l> v681;
                        int v683;
                        v683 = 0l;
                        while (while_method_3(v683)){
                            v681[v683] = v675;
                            v683 += 1l ;
                        }
                        Union6 v685;
                        v685 = Union6{Union6_1{v670}};
                        Union5 v686;
                        v686 = Union5{Union5_2{v685, true, v665, 0l, v681, v673}};
                        v712 = Union7{Union7_1{v686}};
                        break;
                    }
                    case 1: { // ChanceInit
                        unsigned int v688 = v2;
                        Union3 v689; unsigned int v690;
                        Tuple0 tmp2 = draw_card_4(v5, v688);
                        v689 = tmp2.v0; v690 = tmp2.v1;
                        v2 = v690;
                        unsigned int v691 = v2;
                        Union3 v692; unsigned int v693;
                        Tuple0 tmp3 = draw_card_4(v5, v691);
                        v692 = tmp3.v0; v693 = tmp3.v1;
                        v2 = v693;
                        Union2 v694;
                        v694 = Union2{Union2_2{0l, v689}};
                        v9.push(v694);
                        Union2 v695;
                        v695 = Union2{Union2_2{1l, v692}};
                        v9.push(v695);
                        int v696;
                        v696 = 2l;
                        static_array<int,2l> v697;
                        v697[0l] = 1l;
                        v697[1l] = 1l;
                        static_array<Union3,2l> v699;
                        v699[0l] = v689;
                        v699[1l] = v692;
                        Union6 v701;
                        v701 = Union6{Union6_0{}};
                        Union5 v702;
                        v702 = Union5{Union5_2{v701, true, v699, 0l, v697, v696}};
                        v712 = Union7{Union7_1{v702}};
                        break;
                    }
                    case 2: { // Round
                        Union6 v54 = v13.case2.v0; bool v55 = v13.case2.v1; static_array<Union3,2l> v56 = v13.case2.v2; int v57 = v13.case2.v3; static_array<int,2l> v58 = v13.case2.v4; int v59 = v13.case2.v5;
                        static_array<Union1,2l> v60 = v4;
                        Union1 v61;
                        v61 = v60[v57];
                        Union4 v479;
                        switch (v61.tag) {
                            case 0: { // Computer
                                static_array_list<Union2,32l> & v63 = v3;
                                unsigned int * v64;
                                v64 = reinterpret_cast<unsigned int *>(&v0[12582912ull]);
                                float * v66;
                                v66 = reinterpret_cast<float *>(&v0[0ull]);
                                int v68;
                                v68 = threadIdx.x;
                                int v69;
                                v69 = blockIdx.x;
                                int v70;
                                v70 = v69 * 512l;
                                int v71;
                                v71 = v68 + v70;
                                unsigned long long v72;
                                v72 = (unsigned long long)v71;
                                curandStatePhilox4_32_10_t v73;
                                curand_init(12344321ull,v72,0ull,&v73);
                                float * v74;
                                v74 = reinterpret_cast<float *>(&v0[0ull]);
                                int v76;
                                v76 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v76 && v76 < 24l);
                                int v77;
                                v77 = 65536l * v76;
                                int v78;
                                v78 = threadIdx.x;
                                int v79;
                                v79 = blockIdx.x;
                                int v80;
                                v80 = v79 * 512l;
                                int v81;
                                v81 = v78 + v80;
                                unsigned long long v82;
                                v82 = (unsigned long long)v81;
                                curandStatePhilox4_32_10_t v83;
                                curand_init(12344321ull,v82,0ull,&v83);
                                int v84;
                                v84 = threadIdx.x;
                                int v85;
                                v85 = v84;
                                while (while_method_4(v85)){
                                    bool v87;
                                    v87 = 0l <= v85;
                                    bool v88;
                                    v88 = v87 == false;
                                    if (v88){
                                        assert("The index needs to be zero or positive." && v87);
                                    } else {
                                    }
                                    int v90;
                                    v90 = v85 % 128l;
                                    int v91;
                                    v91 = v85 / 128l;
                                    bool v92;
                                    v92 = v91 < 512l;
                                    bool v93;
                                    v93 = v92 == false;
                                    if (v93){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v92);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v91 && v91 < 512l);
                                    assert("Tensor range check" && 0 <= v90 && v90 < 128l);
                                    int v95;
                                    v95 = v90 + v77;
                                    int v96;
                                    v96 = 128l * v91;
                                    int v97;
                                    v97 = v96 + v95;
                                    v74[v97] = 0.0f;
                                    v85 += 512l ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                int v98;
                                v98 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v98 && v98 < 512l);
                                int v99;
                                v99 = 128l * v98;
                                int v100;
                                v100 = v99 + v77;
                                static_array_list<Union8,10l> v101;
                                v101 = static_array_list<Union8,10l>{};
                                int v103;
                                v103 = v63.length;
                                int v104;
                                v104 = 0l;
                                while (while_method_5(v103, v104)){
                                    Union2 v106;
                                    v106 = v63[v104];
                                    Union9 v125;
                                    switch (v106.tag) {
                                        case 0: { // CommunityCardIs
                                            Union3 v115 = v106.case0.v0;
                                            Union8 v116;
                                            v116 = Union8{Union8_1{v115}};
                                            v125 = Union9{Union9_1{v116}};
                                            break;
                                        }
                                        case 1: { // PlayerAction
                                            int v118 = v106.case1.v0; Union4 v119 = v106.case1.v1;
                                            Union8 v120;
                                            v120 = Union8{Union8_0{v119}};
                                            v125 = Union9{Union9_1{v120}};
                                            break;
                                        }
                                        case 2: { // PlayerGotCard
                                            int v108 = v106.case2.v0; Union3 v109 = v106.case2.v1;
                                            bool v110;
                                            v110 = v108 == v57;
                                            if (v110){
                                                Union8 v111;
                                                v111 = Union8{Union8_1{v109}};
                                                v125 = Union9{Union9_1{v111}};
                                            } else {
                                                v125 = Union9{Union9_0{}};
                                            }
                                            break;
                                        }
                                        default: {
                                            v125 = Union9{Union9_0{}};
                                        }
                                    }
                                    switch (v125.tag) {
                                        case 0: { // None
                                            break;
                                        }
                                        case 1: { // Some
                                            Union8 v126 = v125.case1.v0;
                                            v101.push(v126);
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false); __trap();
                                        }
                                    }
                                    v104 += 1l ;
                                }
                                float * v127;
                                v127 = v74+v100;
                                int v129;
                                v129 = v101.length;
                                bool v130;
                                v130 = v129 == 0l;
                                if (v130){
                                    v127[0l] = 1.0f;
                                } else {
                                }
                                int v131;
                                v131 = v101.length;
                                int v132;
                                v132 = 0l;
                                while (while_method_5(v131, v132)){
                                    Union8 v134;
                                    v134 = v101[v132];
                                    int v136;
                                    v136 = v132 * 6l;
                                    int v137;
                                    v137 = 1l + v136;
                                    switch (v134.tag) {
                                        case 0: { // C1of2
                                            Union4 v138 = v134.case0.v0;
                                            switch (v138.tag) {
                                                case 0: { // Call
                                                    v127[v137] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // Fold
                                                    int v139;
                                                    v139 = v137 + 1l;
                                                    v127[v139] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // Raise
                                                    int v140;
                                                    v140 = v137 + 2l;
                                                    v127[v140] = 1.0f;
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false); __trap();
                                                }
                                            }
                                            break;
                                        }
                                        case 1: { // C2of2
                                            Union3 v141 = v134.case1.v0;
                                            int v142;
                                            v142 = v137 + 3l;
                                            switch (v141.tag) {
                                                case 0: { // Jack
                                                    v127[v142] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // King
                                                    int v143;
                                                    v143 = v142 + 1l;
                                                    v127[v143] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // Queen
                                                    int v144;
                                                    v144 = v142 + 2l;
                                                    v127[v144] = 1.0f;
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
                                    v132 += 1l ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                int v145;
                                v145 = 0l;
                                while (while_method_6(v145)){
                                    float * v147;
                                    v147 = reinterpret_cast<float *>(&v0[0ull]);
                                    float * v149;
                                    v149 = reinterpret_cast<float *>(&v1[0ull]);
                                    assert("Tensor range check" && 0 <= v145 && v145 < 4l);
                                    int v151;
                                    v151 = 16384l * v145;
                                    float * v152;
                                    v152 = reinterpret_cast<float *>(&v0[6291456ull]);
                                    int v154;
                                    v154 = blockIdx.x;
                                    assert("Tensor range check" && 0 <= v154 && v154 < 24l);
                                    int v155;
                                    v155 = 65536l * v154;
                                    int v156;
                                    v156 = blockIdx.x;
                                    assert("Tensor range check" && 0 <= v156 && v156 < 24l);
                                    int v157;
                                    v157 = 65536l * v156;
                                    method_6(v149, v151, v152, v157, v147, v155);
                                    unsigned int * v158;
                                    v158 = reinterpret_cast<unsigned int *>(&v0[12582912ull]);
                                    assert("Tensor range check" && 0 <= v145 && v145 < 4l);
                                    int v160;
                                    v160 = 12288l * v145;
                                    method_7(v158, v160, v152);
                                    int * v161;
                                    v161 = reinterpret_cast<int *>(&v1[262144ull]);
                                    float * v163;
                                    v163 = reinterpret_cast<float *>(&v1[262160ull]);
                                    float * v165;
                                    v165 = reinterpret_cast<float *>(&v1[524304ull]);
                                    float * v167;
                                    v167 = reinterpret_cast<float *>(&v1[786448ull]);
                                    float * v169;
                                    v169 = reinterpret_cast<float *>(&v1[1048592ull]);
                                    float * v171;
                                    v171 = reinterpret_cast<float *>(&v1[1310736ull]);
                                    float * v173;
                                    v173 = reinterpret_cast<float *>(&v1[1572880ull]);
                                    float * v175;
                                    v175 = reinterpret_cast<float *>(&v1[1835024ull]);
                                    int * v177;
                                    v177 = reinterpret_cast<int *>(&v0[12779520ull]);
                                    float * v179;
                                    v179 = reinterpret_cast<float *>(&v0[15925248ull]);
                                    int * v181;
                                    v181 = reinterpret_cast<int *>(&v0[19070976ull]);
                                    int * v183;
                                    v183 = reinterpret_cast<int *>(&v0[22216704ull]);
                                    double * v185;
                                    v185 = reinterpret_cast<double *>(&v0[25362432ull]);
                                    double * v187;
                                    v187 = reinterpret_cast<double *>(&v0[37945344ull]);
                                    double * v189;
                                    v189 = reinterpret_cast<double *>(&v1[2097168ull]);
                                    double * v191;
                                    v191 = reinterpret_cast<double *>(&v1[2883600ull]);
                                    int * v193;
                                    v193 = reinterpret_cast<int *>(&v1[3670032ull]);
                                    v145 += 1l ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                int * v195;
                                v195 = reinterpret_cast<int *>(&v1[262144ull]);
                                float * v197;
                                v197 = reinterpret_cast<float *>(&v1[262160ull]);
                                float * v199;
                                v199 = reinterpret_cast<float *>(&v1[524304ull]);
                                float * v201;
                                v201 = reinterpret_cast<float *>(&v1[786448ull]);
                                float * v203;
                                v203 = reinterpret_cast<float *>(&v1[1048592ull]);
                                float * v205;
                                v205 = reinterpret_cast<float *>(&v1[1310736ull]);
                                float * v207;
                                v207 = reinterpret_cast<float *>(&v1[1572880ull]);
                                float * v209;
                                v209 = reinterpret_cast<float *>(&v1[1835024ull]);
                                int v211;
                                v211 = v195[0l];
                                unsigned int * v212;
                                v212 = reinterpret_cast<unsigned int *>(&v0[12582912ull]);
                                int v214;
                                v214 = blockIdx.x;
                                int v215;
                                v215 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v211 && v211 < 4l);
                                assert("Tensor range check" && 0 <= v214 && v214 < 24l);
                                assert("Tensor range check" && 0 <= v215 && v215 < 512l);
                                int v216;
                                v216 = 512l * v214;
                                int v217;
                                v217 = v216 + v215;
                                int v218;
                                v218 = 12288l * v211;
                                int v219;
                                v219 = v218 + v217;
                                unsigned int v220;
                                v220 = v212[v219];
                                int v221;
                                v221 = (int)v220;
                                float v222; int v223;
                                Tuple2 tmp14 = method_8(v5, v195, v197, v199, v201, v203, v205, v207, v209, v221, v211);
                                v222 = tmp14.v0; v223 = tmp14.v1;
                                extern __shared__ unsigned char v224[];
                                float * v225;
                                v225 = reinterpret_cast<float *>(&v224[0ull]);
                                int * v227;
                                v227 = reinterpret_cast<int *>(&v224[16ull]);
                                int v229;
                                v229 = threadIdx.x;
                                bool v230;
                                v230 = v229 == 0l;
                                if (v230){
                                    v225[0l] = v222;
                                    v227[0l] = v223;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                float v231;
                                v231 = v225[0l];
                                int v232;
                                v232 = v227[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                double * v233;
                                v233 = reinterpret_cast<double *>(&v1[2097168ull]);
                                double * v235;
                                v235 = reinterpret_cast<double *>(&v1[2883600ull]);
                                int * v237;
                                v237 = reinterpret_cast<int *>(&v1[3670032ull]);
                                int * v239;
                                v239 = reinterpret_cast<int *>(&v0[12779520ull]);
                                float * v241;
                                v241 = reinterpret_cast<float *>(&v0[15925248ull]);
                                int * v243;
                                v243 = reinterpret_cast<int *>(&v0[19070976ull]);
                                int * v245;
                                v245 = reinterpret_cast<int *>(&v0[22216704ull]);
                                double * v247;
                                v247 = reinterpret_cast<double *>(&v0[25362432ull]);
                                double * v249;
                                v249 = reinterpret_cast<double *>(&v0[37945344ull]);
                                int v251;
                                v251 = threadIdx.x;
                                int v252;
                                v252 = blockIdx.x;
                                int v253;
                                v253 = v252 * 512l;
                                int v254;
                                v254 = v251 + v253;
                                int v255;
                                v255 = 0l;
                                while (while_method_6(v255)){
                                    unsigned int * v257;
                                    v257 = reinterpret_cast<unsigned int *>(&v0[12582912ull]);
                                    int v259;
                                    v259 = blockIdx.x;
                                    int v260;
                                    v260 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v255 && v255 < 4l);
                                    assert("Tensor range check" && 0 <= v259 && v259 < 24l);
                                    assert("Tensor range check" && 0 <= v260 && v260 < 512l);
                                    int v261;
                                    v261 = 512l * v259;
                                    int v262;
                                    v262 = v261 + v260;
                                    int v263;
                                    v263 = 12288l * v255;
                                    int v264;
                                    v264 = v263 + v262;
                                    unsigned int v265;
                                    v265 = v257[v264];
                                    int v266;
                                    v266 = (int)v265;
                                    float v267;
                                    v267 = method_9(v195, v197, v199, v201, v203, v205, v207, v209, v266, v255, v232);
                                    assert("Tensor range check" && 0 <= v255 && v255 < 4l);
                                    assert("Tensor range check" && 0 <= v254 && v254 < 12288l);
                                    int v268;
                                    v268 = v263 + v254;
                                    int v269;
                                    v269 = v237[v268];
                                    int v270;
                                    v270 = v269 + 1l;
                                    assert("Tensor range check" && 0 <= v255 && v255 < 4l);
                                    assert("Tensor range check" && 0 <= v254 && v254 < 12288l);
                                    v237[v268] = v270;
                                    assert("Tensor range check" && 0 <= v255 && v255 < 4l);
                                    assert("Tensor range check" && 0 <= v269 && v269 < 16l);
                                    assert("Tensor range check" && 0 <= v254 && v254 < 12288l);
                                    int v271;
                                    v271 = 12288l * v269;
                                    int v272;
                                    v272 = v271 + v254;
                                    int v273;
                                    v273 = 196608l * v255;
                                    int v274;
                                    v274 = v273 + v272;
                                    v239[v274] = v232;
                                    v241[v274] = v231;
                                    v243[v274] = v57;
                                    v245[v274] = v266;
                                    assert("Tensor range check" && 0 <= v255 && v255 < 4l);
                                    int v275;
                                    v275 = 24576l * v255;
                                    assert("Tensor range check" && 0 <= v254 && v254 < 12288l);
                                    int v276;
                                    v276 = 2l * v254;
                                    int v277;
                                    v277 = v276 + v275;
                                    assert("Tensor range check" && 0 <= v255 && v255 < 4l);
                                    int v278;
                                    v278 = 393216l * v255;
                                    assert("Tensor range check" && 0 <= v269 && v269 < 16l);
                                    int v279;
                                    v279 = 24576l * v269;
                                    int v280;
                                    v280 = v279 + v278;
                                    assert("Tensor range check" && 0 <= v254 && v254 < 12288l);
                                    int v281;
                                    v281 = v276 + v280;
                                    double * v282;
                                    v282 = v233+v277;
                                    double * v284;
                                    v284 = v235+v277;
                                    double * v286;
                                    v286 = v247+v281;
                                    double * v288;
                                    v288 = v249+v281;
                                    int v290;
                                    v290 = sizeof(double *);
                                    unsigned long long v291;
                                    v291 = (unsigned long long)v290;
                                    unsigned long long v292;
                                    v292 = 512ull * v291;
                                    unsigned long long v293;
                                    v293 = v292 + 16ull;
                                    unsigned long long v294;
                                    v294 = v293 - 1ull;
                                    unsigned long long v295;
                                    v295 = v294 % 16ull;
                                    unsigned long long v296;
                                    v296 = v294 - v295;
                                    unsigned long long v297;
                                    v297 = v296 + v292;
                                    unsigned long long v298;
                                    v298 = v297 + 16ull;
                                    unsigned long long v299;
                                    v299 = v298 - 1ull;
                                    unsigned long long v300;
                                    v300 = v299 % 16ull;
                                    unsigned long long v301;
                                    v301 = v299 - v300;
                                    unsigned long long v302;
                                    v302 = v301 + v292;
                                    unsigned long long v303;
                                    v303 = v302 + 16ull;
                                    unsigned long long v304;
                                    v304 = v303 - 1ull;
                                    unsigned long long v305;
                                    v305 = v304 % 16ull;
                                    unsigned long long v306;
                                    v306 = v304 - v305;
                                    unsigned long long v307;
                                    v307 = v306 + v292;
                                    bool v308;
                                    v308 = v307 <= 81920ull;
                                    bool v309;
                                    v309 = v308 == false;
                                    if (v309){
                                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v308);
                                    } else {
                                    }
                                    extern __shared__ unsigned char v311[];
                                    bool v312;
                                    v312 = v307 <= v307;
                                    bool v313;
                                    v313 = v312 == false;
                                    if (v313){
                                        assert("The length of the partition has to be less than or equal to the length of the base array." && v312);
                                    } else {
                                    }
                                    double * * v315;
                                    v315 = reinterpret_cast<double * *>(&v311[0ull]);
                                    double * * v317;
                                    v317 = reinterpret_cast<double * *>(&v311[v296]);
                                    double * * v319;
                                    v319 = reinterpret_cast<double * *>(&v311[v301]);
                                    double * * v321;
                                    v321 = reinterpret_cast<double * *>(&v311[v306]);
                                    int v323;
                                    v323 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v323 && v323 < 512l);
                                    v315[v323] = v282;
                                    v317[v323] = v284;
                                    v319[v323] = v286;
                                    v321[v323] = v288;
                                    asm("barrier.cta.sync %0;" :: "r"(0l));
                                    bool v324;
                                    v324 = 0l <= v323;
                                    bool v325;
                                    v325 = v324 == false;
                                    if (v325){
                                        assert("The index needs to be zero or positive." && v324);
                                    } else {
                                    }
                                    int v327;
                                    v327 = v323 % 1l;
                                    bool v328;
                                    v328 = v323 < 512l;
                                    bool v329;
                                    v329 = v328 == false;
                                    if (v329){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v328);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v323 && v323 < 512l);
                                    int v331;
                                    v331 = 0l;
                                    while (while_method_7(v331)){
                                        bool v333;
                                        v333 = v324 && v328;
                                        bool v334;
                                        v334 = v333 == false;
                                        if (v334){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v333);
                                        } else {
                                        }
                                        bool v336;
                                        v336 = 0l <= v331;
                                        bool v338;
                                        if (v336){
                                            bool v337;
                                            v337 = v331 < 1l;
                                            v338 = v337;
                                        } else {
                                            v338 = false;
                                        }
                                        bool v339;
                                        v339 = v338 == false;
                                        if (v339){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v338);
                                        } else {
                                        }
                                        int v341;
                                        v341 = v331 * 512l;
                                        int v342;
                                        v342 = v341 + v323;
                                        assert("Tensor range check" && 0 <= v331 && v331 < 1l);
                                        int v343;
                                        v343 = 512l * v331;
                                        int v344;
                                        v344 = v343 + v323;
                                        double * v345;
                                        v345 = v315[v344];
                                        double * v346;
                                        v346 = v317[v344];
                                        double * v347;
                                        v347 = v319[v344];
                                        double * v348;
                                        v348 = v321[v344];
                                        int v349;
                                        v349 = blockIdx.x;
                                        int v350;
                                        v350 = v349 * 512l;
                                        int v351;
                                        v351 = v350 + v342;
                                        assert("Tensor range check" && 0 <= v327 && v327 < 1l);
                                        int v352;
                                        v352 = 2l * v327;
                                        double v353[2l];
                                        double v354[2l];
                                        int v355[2l];
                                        int v356;
                                        v356 = 0l;
                                        while (while_method_7(v356)){
                                            assert("Tensor range check" && 0 <= v356 && v356 < 1l);
                                            int v358;
                                            v358 = 2l * v356;
                                            assert("Tensor range check" && 0 <= v356 && v356 < 1l);
                                            int v359;
                                            v359 = v358 + v352;
                                            int4* v360;
                                            v360 = reinterpret_cast<int4*>(v345 + v359);
                                            int4* v361;
                                            v361 = reinterpret_cast<int4*>(v353 + v358);
                                            assert("Pointer alignment check" && (unsigned long long)(v360) % 2l == 0 && (unsigned long long)(v361) % 2l == 0);
                                            *v361 = *v360;
                                            int4* v362;
                                            v362 = reinterpret_cast<int4*>(v346 + v359);
                                            int4* v363;
                                            v363 = reinterpret_cast<int4*>(v354 + v358);
                                            assert("Pointer alignment check" && (unsigned long long)(v362) % 2l == 0 && (unsigned long long)(v363) % 2l == 0);
                                            *v363 = *v362;
                                            v356 += 1l ;
                                        }
                                        int v364;
                                        v364 = 0l;
                                        while (while_method_7(v364)){
                                            int v366;
                                            v366 = 0l;
                                            while (while_method_3(v366)){
                                                bool v368;
                                                v368 = 0l <= v366;
                                                bool v370;
                                                if (v368){
                                                    bool v369;
                                                    v369 = v366 < 2l;
                                                    v370 = v369;
                                                } else {
                                                    v370 = false;
                                                }
                                                bool v371;
                                                v371 = v370 == false;
                                                if (v371){
                                                    assert("The indices should be inside the range of the dimension." && v370);
                                                } else {
                                                }
                                                bool v373;
                                                v373 = 0l <= v327;
                                                bool v375;
                                                if (v373){
                                                    bool v374;
                                                    v374 = v327 < 1l;
                                                    v375 = v374;
                                                } else {
                                                    v375 = false;
                                                }
                                                bool v376;
                                                v376 = v375 == false;
                                                if (v376){
                                                    assert("The indices should be inside the range of the dimension." && v375);
                                                } else {
                                                }
                                                int v378;
                                                v378 = v327 * 2l;
                                                int v379;
                                                v379 = v366 + v378;
                                                bool v380;
                                                v380 = 0l <= v364;
                                                bool v382;
                                                if (v380){
                                                    bool v381;
                                                    v381 = v364 < 1l;
                                                    v382 = v381;
                                                } else {
                                                    v382 = false;
                                                }
                                                bool v383;
                                                v383 = v382 == false;
                                                if (v383){
                                                    assert("The indices should be inside the range of the dimension." && v382);
                                                } else {
                                                }
                                                int v385;
                                                v385 = v364 * 2l;
                                                int v386;
                                                v386 = v379 + v385;
                                                assert("Tensor range check" && 0 <= v364 && v364 < 1l);
                                                assert("Tensor range check" && 0 <= v366 && v366 < 2l);
                                                int v387;
                                                v387 = 2l * v364;
                                                int v388;
                                                v388 = v387 + v366;
                                                v355[v388] = v386;
                                                v366 += 1l ;
                                            }
                                            v364 += 1l ;
                                        }
                                        int v389;
                                        v389 = 0l;
                                        while (while_method_7(v389)){
                                            assert("Tensor range check" && 0 <= v389 && v389 < 1l);
                                            int v391;
                                            v391 = 2l * v389;
                                            int v392;
                                            v392 = v391 + v352;
                                            assert("Tensor range check" && 0 <= v389 && v389 < 1l);
                                            int4* v393;
                                            v393 = reinterpret_cast<int4*>(v353 + v391);
                                            int4* v394;
                                            v394 = reinterpret_cast<int4*>(v347 + v392);
                                            assert("Pointer alignment check" && (unsigned long long)(v393) % 2l == 0 && (unsigned long long)(v394) % 2l == 0);
                                            *v394 = *v393;
                                            int4* v395;
                                            v395 = reinterpret_cast<int4*>(v354 + v391);
                                            int4* v396;
                                            v396 = reinterpret_cast<int4*>(v348 + v392);
                                            assert("Pointer alignment check" && (unsigned long long)(v395) % 2l == 0 && (unsigned long long)(v396) % 2l == 0);
                                            *v396 = *v395;
                                            v389 += 1l ;
                                        }
                                        assert("Tensor range check" && 0 <= v342 && v342 < 512l);
                                        v331 += 1l ;
                                    }
                                    asm("barrier.cta.sync %0;" :: "r"(0l));
                                    assert("Tensor range check" && 0 <= v323 && v323 < 512l);
                                    asm("barrier.cta.sync %0;" :: "r"(0l));
                                    double v397;
                                    v397 = (double)v231;
                                    double v398;
                                    v398 = log(v397);
                                    double v399;
                                    v399 = (double)v267;
                                    double v400;
                                    v400 = log(v399);
                                    assert("Tensor range check" && 0 <= v255 && v255 < 4l);
                                    assert("Tensor range check" && 0 <= v254 && v254 < 12288l);
                                    assert("Tensor range check" && 0 <= v57 && v57 < 2l);
                                    int v401;
                                    v401 = v276 + v57;
                                    int v402;
                                    v402 = v275 + v401;
                                    double v403;
                                    v403 = v233[v402];
                                    double v404;
                                    v404 = v235[v402];
                                    double v405;
                                    v405 = v400 + v403;
                                    double v406;
                                    v406 = v398 + v404;
                                    assert("Tensor range check" && 0 <= v255 && v255 < 4l);
                                    assert("Tensor range check" && 0 <= v254 && v254 < 12288l);
                                    assert("Tensor range check" && 0 <= v57 && v57 < 2l);
                                    v233[v402] = v405;
                                    v235[v402] = v406;
                                    v255 += 1l ;
                                }
                                bool v407;
                                v407 = 0l == v232;
                                Union10 v416;
                                if (v407){
                                    v416 = Union10{Union10_1{}};
                                } else {
                                    bool v409;
                                    v409 = 1l == v232;
                                    if (v409){
                                        v416 = Union10{Union10_0{}};
                                    } else {
                                        bool v411;
                                        v411 = 2l == v232;
                                        if (v411){
                                            v416 = Union10{Union10_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v416.tag) {
                                    case 0: { // AA_Call
                                        v479 = Union4{Union4_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v417;
                                        v417 = v58[0l];
                                        int v419; int v420;
                                        Tuple1 tmp17 = Tuple1{1l, v417};
                                        v419 = tmp17.v0; v420 = tmp17.v1;
                                        while (while_method_3(v419)){
                                            int v422;
                                            v422 = v58[v419];
                                            bool v424;
                                            v424 = v420 >= v422;
                                            int v425;
                                            if (v424){
                                                v425 = v420;
                                            } else {
                                                v425 = v422;
                                            }
                                            v420 = v425;
                                            v419 += 1l ;
                                        }
                                        int v426;
                                        v426 = v58[v57];
                                        bool v428;
                                        v428 = v426 == v420;
                                        if (v428){
                                            v479 = Union4{Union4_0{}};
                                        } else {
                                            v479 = Union4{Union4_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v433;
                                        v433 = v59 > 0l;
                                        if (v433){
                                            v479 = Union4{Union4_2{}};
                                        } else {
                                            v479 = Union4{Union4_0{}};
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                break;
                            }
                            case 1: { // Random
                                static_array_list<Union4,3l> v440;
                                v440 = static_array_list<Union4,3l>{};
                                v440.unsafe_set_length(1l);
                                Union4 v442;
                                v442 = Union4{Union4_0{}};
                                v440[0l] = v442;
                                int v444;
                                v444 = v58[0l];
                                int v446;
                                v446 = v58[1l];
                                bool v448;
                                v448 = v444 == v446;
                                bool v449;
                                v449 = v448 != true;
                                if (v449){
                                    Union4 v450;
                                    v450 = Union4{Union4_1{}};
                                    v440.push(v450);
                                } else {
                                }
                                bool v451;
                                v451 = v59 > 0l;
                                if (v451){
                                    Union4 v452;
                                    v452 = Union4{Union4_2{}};
                                    v440.push(v452);
                                } else {
                                }
                                int v453;
                                v453 = v440.length;
                                int v454;
                                v454 = v453 - 1l;
                                int v455;
                                v455 = 0l;
                                while (while_method_5(v454, v455)){
                                    int v457;
                                    v457 = v440.length;
                                    int v458;
                                    v458 = int_range_10(v457, v455, v5);
                                    Union4 v459;
                                    v459 = v440[v455];
                                    Union4 v461;
                                    v461 = v440[v458];
                                    v440[v455] = v461;
                                    v440[v458] = v459;
                                    v455 += 1l ;
                                }
                                Union4 v463;
                                v463 = v440.pop();
                                int v464;
                                v464 = sizeof(Union4);
                                unsigned long long v465;
                                v465 = (unsigned long long)v464;
                                bool v466;
                                v466 = v465 <= 81920ull;
                                bool v467;
                                v467 = v466 == false;
                                if (v467){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v466);
                                } else {
                                }
                                extern __shared__ unsigned char v469[];
                                bool v470;
                                v470 = v465 <= v465;
                                bool v471;
                                v471 = v470 == false;
                                if (v471){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v470);
                                } else {
                                }
                                Union4 * v473;
                                v473 = reinterpret_cast<Union4 *>(&v469[0ull]);
                                int v475;
                                v475 = threadIdx.x;
                                bool v476;
                                v476 = v475 == 0l;
                                if (v476){
                                    v473[0l] = v463;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                Union4 v477;
                                v477 = v473[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                v479 = v477;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union2 v480;
                        v480 = Union2{Union2_1{v57, v479}};
                        v9.push(v480);
                        Union5 v566;
                        switch (v54.tag) {
                            case 0: { // None
                                switch (v479.tag) {
                                    case 0: { // Call
                                        if (v55){
                                            bool v530;
                                            v530 = v57 == 0l;
                                            int v531;
                                            if (v530){
                                                v531 = 1l;
                                            } else {
                                                v531 = 0l;
                                            }
                                            v566 = Union5{Union5_2{v54, false, v56, v531, v58, v59}};
                                        } else {
                                            v566 = Union5{Union5_0{v54, v55, v56, v57, v58, v59}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v566 = Union5{Union5_5{v54, v55, v56, v57, v58, v59}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v535;
                                        v535 = v59 > 0l;
                                        if (v535){
                                            bool v536;
                                            v536 = v57 == 0l;
                                            int v537;
                                            if (v536){
                                                v537 = 1l;
                                            } else {
                                                v537 = 0l;
                                            }
                                            int v538;
                                            v538 = -1l + v59;
                                            int v539; int v540;
                                            Tuple1 tmp18 = Tuple1{0l, 0l};
                                            v539 = tmp18.v0; v540 = tmp18.v1;
                                            while (while_method_3(v539)){
                                                int v542;
                                                v542 = v58[v539];
                                                bool v544;
                                                v544 = v540 >= v542;
                                                int v545;
                                                if (v544){
                                                    v545 = v540;
                                                } else {
                                                    v545 = v542;
                                                }
                                                v540 = v545;
                                                v539 += 1l ;
                                            }
                                            static_array<int,2l> v546;
                                            int v548;
                                            v548 = 0l;
                                            while (while_method_3(v548)){
                                                v546[v548] = v540;
                                                v548 += 1l ;
                                            }
                                            static_array<int,2l> v550;
                                            int v552;
                                            v552 = 0l;
                                            while (while_method_3(v552)){
                                                int v554;
                                                v554 = v546[v552];
                                                bool v556;
                                                v556 = v552 == v57;
                                                int v558;
                                                if (v556){
                                                    int v557;
                                                    v557 = v554 + 2l;
                                                    v558 = v557;
                                                } else {
                                                    v558 = v554;
                                                }
                                                v550[v552] = v558;
                                                v552 += 1l ;
                                            }
                                            v566 = Union5{Union5_2{v54, false, v56, v537, v550, v538}};
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
                                Union3 v481 = v54.case1.v0;
                                switch (v479.tag) {
                                    case 0: { // Call
                                        if (v55){
                                            bool v483;
                                            v483 = v57 == 0l;
                                            int v484;
                                            if (v483){
                                                v484 = 1l;
                                            } else {
                                                v484 = 0l;
                                            }
                                            v566 = Union5{Union5_2{v54, false, v56, v484, v58, v59}};
                                        } else {
                                            int v486; int v487;
                                            Tuple1 tmp19 = Tuple1{0l, 0l};
                                            v486 = tmp19.v0; v487 = tmp19.v1;
                                            while (while_method_3(v486)){
                                                int v489;
                                                v489 = v58[v486];
                                                bool v491;
                                                v491 = v487 >= v489;
                                                int v492;
                                                if (v491){
                                                    v492 = v487;
                                                } else {
                                                    v492 = v489;
                                                }
                                                v487 = v492;
                                                v486 += 1l ;
                                            }
                                            static_array<int,2l> v493;
                                            int v495;
                                            v495 = 0l;
                                            while (while_method_3(v495)){
                                                v493[v495] = v487;
                                                v495 += 1l ;
                                            }
                                            v566 = Union5{Union5_4{v54, v55, v56, v57, v493, v59}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v566 = Union5{Union5_5{v54, v55, v56, v57, v58, v59}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v499;
                                        v499 = v59 > 0l;
                                        if (v499){
                                            bool v500;
                                            v500 = v57 == 0l;
                                            int v501;
                                            if (v500){
                                                v501 = 1l;
                                            } else {
                                                v501 = 0l;
                                            }
                                            int v502;
                                            v502 = -1l + v59;
                                            int v503; int v504;
                                            Tuple1 tmp20 = Tuple1{0l, 0l};
                                            v503 = tmp20.v0; v504 = tmp20.v1;
                                            while (while_method_3(v503)){
                                                int v506;
                                                v506 = v58[v503];
                                                bool v508;
                                                v508 = v504 >= v506;
                                                int v509;
                                                if (v508){
                                                    v509 = v504;
                                                } else {
                                                    v509 = v506;
                                                }
                                                v504 = v509;
                                                v503 += 1l ;
                                            }
                                            static_array<int,2l> v510;
                                            int v512;
                                            v512 = 0l;
                                            while (while_method_3(v512)){
                                                v510[v512] = v504;
                                                v512 += 1l ;
                                            }
                                            static_array<int,2l> v514;
                                            int v516;
                                            v516 = 0l;
                                            while (while_method_3(v516)){
                                                int v518;
                                                v518 = v510[v516];
                                                bool v520;
                                                v520 = v516 == v57;
                                                int v522;
                                                if (v520){
                                                    int v521;
                                                    v521 = v518 + 4l;
                                                    v522 = v521;
                                                } else {
                                                    v522 = v518;
                                                }
                                                v514[v516] = v522;
                                                v516 += 1l ;
                                            }
                                            v566 = Union5{Union5_2{v54, false, v56, v501, v514, v502}};
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
                        v712 = Union7{Union7_1{v566}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union6 v568 = v13.case3.v0; bool v569 = v13.case3.v1; static_array<Union3,2l> v570 = v13.case3.v2; int v571 = v13.case3.v3; static_array<int,2l> v572 = v13.case3.v4; int v573 = v13.case3.v5; Union4 v574 = v13.case3.v6;
                        Union2 v575;
                        v575 = Union2{Union2_1{v571, v574}};
                        v9.push(v575);
                        Union5 v661;
                        switch (v568.tag) {
                            case 0: { // None
                                switch (v574.tag) {
                                    case 0: { // Call
                                        if (v569){
                                            bool v625;
                                            v625 = v571 == 0l;
                                            int v626;
                                            if (v625){
                                                v626 = 1l;
                                            } else {
                                                v626 = 0l;
                                            }
                                            v661 = Union5{Union5_2{v568, false, v570, v626, v572, v573}};
                                        } else {
                                            v661 = Union5{Union5_0{v568, v569, v570, v571, v572, v573}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v661 = Union5{Union5_5{v568, v569, v570, v571, v572, v573}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v630;
                                        v630 = v573 > 0l;
                                        if (v630){
                                            bool v631;
                                            v631 = v571 == 0l;
                                            int v632;
                                            if (v631){
                                                v632 = 1l;
                                            } else {
                                                v632 = 0l;
                                            }
                                            int v633;
                                            v633 = -1l + v573;
                                            int v634; int v635;
                                            Tuple1 tmp21 = Tuple1{0l, 0l};
                                            v634 = tmp21.v0; v635 = tmp21.v1;
                                            while (while_method_3(v634)){
                                                int v637;
                                                v637 = v572[v634];
                                                bool v639;
                                                v639 = v635 >= v637;
                                                int v640;
                                                if (v639){
                                                    v640 = v635;
                                                } else {
                                                    v640 = v637;
                                                }
                                                v635 = v640;
                                                v634 += 1l ;
                                            }
                                            static_array<int,2l> v641;
                                            int v643;
                                            v643 = 0l;
                                            while (while_method_3(v643)){
                                                v641[v643] = v635;
                                                v643 += 1l ;
                                            }
                                            static_array<int,2l> v645;
                                            int v647;
                                            v647 = 0l;
                                            while (while_method_3(v647)){
                                                int v649;
                                                v649 = v641[v647];
                                                bool v651;
                                                v651 = v647 == v571;
                                                int v653;
                                                if (v651){
                                                    int v652;
                                                    v652 = v649 + 2l;
                                                    v653 = v652;
                                                } else {
                                                    v653 = v649;
                                                }
                                                v645[v647] = v653;
                                                v647 += 1l ;
                                            }
                                            v661 = Union5{Union5_2{v568, false, v570, v632, v645, v633}};
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
                                Union3 v576 = v568.case1.v0;
                                switch (v574.tag) {
                                    case 0: { // Call
                                        if (v569){
                                            bool v578;
                                            v578 = v571 == 0l;
                                            int v579;
                                            if (v578){
                                                v579 = 1l;
                                            } else {
                                                v579 = 0l;
                                            }
                                            v661 = Union5{Union5_2{v568, false, v570, v579, v572, v573}};
                                        } else {
                                            int v581; int v582;
                                            Tuple1 tmp22 = Tuple1{0l, 0l};
                                            v581 = tmp22.v0; v582 = tmp22.v1;
                                            while (while_method_3(v581)){
                                                int v584;
                                                v584 = v572[v581];
                                                bool v586;
                                                v586 = v582 >= v584;
                                                int v587;
                                                if (v586){
                                                    v587 = v582;
                                                } else {
                                                    v587 = v584;
                                                }
                                                v582 = v587;
                                                v581 += 1l ;
                                            }
                                            static_array<int,2l> v588;
                                            int v590;
                                            v590 = 0l;
                                            while (while_method_3(v590)){
                                                v588[v590] = v582;
                                                v590 += 1l ;
                                            }
                                            v661 = Union5{Union5_4{v568, v569, v570, v571, v588, v573}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v661 = Union5{Union5_5{v568, v569, v570, v571, v572, v573}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v594;
                                        v594 = v573 > 0l;
                                        if (v594){
                                            bool v595;
                                            v595 = v571 == 0l;
                                            int v596;
                                            if (v595){
                                                v596 = 1l;
                                            } else {
                                                v596 = 0l;
                                            }
                                            int v597;
                                            v597 = -1l + v573;
                                            int v598; int v599;
                                            Tuple1 tmp23 = Tuple1{0l, 0l};
                                            v598 = tmp23.v0; v599 = tmp23.v1;
                                            while (while_method_3(v598)){
                                                int v601;
                                                v601 = v572[v598];
                                                bool v603;
                                                v603 = v599 >= v601;
                                                int v604;
                                                if (v603){
                                                    v604 = v599;
                                                } else {
                                                    v604 = v601;
                                                }
                                                v599 = v604;
                                                v598 += 1l ;
                                            }
                                            static_array<int,2l> v605;
                                            int v607;
                                            v607 = 0l;
                                            while (while_method_3(v607)){
                                                v605[v607] = v599;
                                                v607 += 1l ;
                                            }
                                            static_array<int,2l> v609;
                                            int v611;
                                            v611 = 0l;
                                            while (while_method_3(v611)){
                                                int v613;
                                                v613 = v605[v611];
                                                bool v615;
                                                v615 = v611 == v571;
                                                int v617;
                                                if (v615){
                                                    int v616;
                                                    v616 = v613 + 4l;
                                                    v617 = v616;
                                                } else {
                                                    v617 = v613;
                                                }
                                                v609[v611] = v617;
                                                v611 += 1l ;
                                            }
                                            v661 = Union5{Union5_2{v568, false, v570, v596, v609, v597}};
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
                        v712 = Union7{Union7_1{v661}};
                        break;
                    }
                    case 4: { // TerminalCall
                        Union6 v30 = v13.case4.v0; bool v31 = v13.case4.v1; static_array<Union3,2l> v32 = v13.case4.v2; int v33 = v13.case4.v3; static_array<int,2l> v34 = v13.case4.v4; int v35 = v13.case4.v5;
                        int v36;
                        v36 = v34[v33];
                        Union11 v38;
                        v38 = compare_hands_11(v30, v31, v32, v33, v34, v35);
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
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        int v45;
                        v45 = -v44;
                        bool v46;
                        v46 = v44 >= v45;
                        int v47;
                        if (v46){
                            v47 = v44;
                        } else {
                            v47 = v45;
                        }
                        float v48;
                        v48 = (float)v43;
                        v7[v47] = v48;
                        bool v49;
                        v49 = v47 == 0l;
                        int v50;
                        if (v49){
                            v50 = 1l;
                        } else {
                            v50 = 0l;
                        }
                        float v51;
                        v51 = -v48;
                        v7[v50] = v51;
                        Union2 v52;
                        v52 = Union2{Union2_3{v32, v43, v44}};
                        v9.push(v52);
                        v712 = Union7{Union7_0{}};
                        break;
                    }
                    case 5: { // TerminalFold
                        Union6 v14 = v13.case5.v0; bool v15 = v13.case5.v1; static_array<Union3,2l> v16 = v13.case5.v2; int v17 = v13.case5.v3; static_array<int,2l> v18 = v13.case5.v4; int v19 = v13.case5.v5;
                        int v20;
                        v20 = v18[v17];
                        int v22;
                        v22 = -v20;
                        float v23;
                        v23 = (float)v22;
                        v7[v17] = v23;
                        bool v24;
                        v24 = v17 == 0l;
                        int v25;
                        if (v24){
                            v25 = 1l;
                        } else {
                            v25 = 0l;
                        }
                        float v26;
                        v26 = -v23;
                        v7[v25] = v26;
                        int v27;
                        if (v24){
                            v27 = 1l;
                        } else {
                            v27 = 0l;
                        }
                        Union2 v28;
                        v28 = Union2{Union2_3{v16, v20, v27}};
                        v9.push(v28);
                        v712 = Union7{Union7_0{}};
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
        v11 = v712;
    }
    return v7;
}
__device__ inline bool while_method_10(int v0){
    bool v1;
    v1 = v0 > 0l;
    return v1;
}
__device__ float method_16(int * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int v8, int v9, int v10){
    assert("Tensor range check" && 0 <= v9 && v9 < 4l);
    int v11;
    v11 = 16384l * v9;
    assert("Tensor range check" && 0 <= v8 && v8 < 4096l);
    int v12;
    v12 = 4l * v8;
    int v13;
    v13 = v12 + v11;
    float * v14;
    v14 = v1+v13;
    int v16;
    v16 = sizeof(float *);
    unsigned long long v17;
    v17 = (unsigned long long)v16;
    unsigned long long v18;
    v18 = 512ull * v17;
    unsigned long long v19;
    v19 = 2048ull + v18;
    unsigned long long v20;
    v20 = v19 + 16ull;
    unsigned long long v21;
    v21 = v20 - 1ull;
    unsigned long long v22;
    v22 = v21 % 16ull;
    unsigned long long v23;
    v23 = v21 - v22;
    unsigned long long v24;
    v24 = v23 + 2048ull;
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
    v34 = reinterpret_cast<float * *>(&v28[2048ull]);
    float * v36;
    v36 = reinterpret_cast<float *>(&v28[v23]);
    int v38;
    v38 = threadIdx.x;
    assert("Tensor range check" && 0 <= v38 && v38 < 512l);
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
    v43 = v38 < 512l;
    bool v44;
    v44 = v43 == false;
    if (v44){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v43);
    } else {
    }
    assert("Tensor range check" && 0 <= v38 && v38 < 512l);
    int v46;
    v46 = 0l;
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
        v56 = v46 * 512l;
        int v57;
        v57 = v56 + v38;
        assert("Tensor range check" && 0 <= v46 && v46 < 1l);
        int v58;
        v58 = 512l * v46;
        int v59;
        v59 = v58 + v38;
        int v60;
        v60 = v32[v59];
        float * v61;
        v61 = v34[v59];
        int v62;
        v62 = blockIdx.x;
        int v63;
        v63 = v62 * 512l;
        int v64;
        v64 = v63 + v57;
        assert("Tensor range check" && 0 <= v42 && v42 < 1l);
        int v65;
        v65 = 4l * v42;
        float v66[4l];
        int v67[4l];
        int v68;
        v68 = 0l;
        while (while_method_7(v68)){
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
        while (while_method_7(v74)){
            int v76;
            v76 = 0l;
            while (while_method_6(v76)){
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
        while (while_method_7(v100)){
            int v102;
            v102 = 0l;
            while (while_method_6(v102)){
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
        while (while_method_7(v110)){
            int v112;
            v112 = 0l;
            while (while_method_6(v112)){
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
        while (while_method_7(v122)){
            int v124;
            v124 = 0l;
            while (while_method_6(v124)){
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
        while (while_method_7(v136)){
            int v138;
            v138 = 0l;
            while (while_method_6(v138)){
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
        while (while_method_7(v145)){
            int v147;
            v147 = 0l;
            while (while_method_6(v147)){
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
        while (while_method_7(v161)){
            int v163;
            v163 = 0l;
            while (while_method_6(v163)){
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
        Tuple2 tmp31 = Tuple2{0.0f, 2147483647l};
        v175 = tmp31.v0; v176 = tmp31.v1;
        int v177;
        v177 = 0l;
        while (while_method_7(v177)){
            int v179;
            v179 = 0l;
            while (while_method_6(v179)){
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
        Tuple2 tmp32 = cooperative_groups::reduce(v193, Tuple2{v175, v176}, v194);
        v195 = tmp32.v0; v196 = tmp32.v1;
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
        while (while_method_7(v201)){
            assert("Tensor range check" && 0 <= v201 && v201 < 1l);
            assert("Tensor range check" && 0 <= v201 && v201 < 1l);
            v201 += 1l ;
        }
        assert("Tensor range check" && 0 <= v57 && v57 < 512l);
        v36[v57] = v195;
        v46 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v38 && v38 < 512l);
    float v203;
    v203 = v36[v38];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return v203;
}
__device__ static_array<float,2l> method_15(unsigned char * v0, unsigned char * v1, unsigned int & v2, static_array_list<Union2,32l> & v3, static_array<Union1,2l> & v4, curandStatePhilox4_32_10_t & v5, Union5 v6){
    static_array<float,2l> v7;
    static_array_list<Union2,32l> & v9 = v3;
    Union7 v10;
    v10 = Union7{Union7_1{v6}};
    Union7 v11;
    v11 = v10;
    while (while_method_2(v11)){
        Union7 v593;
        switch (v11.tag) {
            case 0: { // None
                v593 = Union7{Union7_0{}};
                break;
            }
            case 1: { // Some
                Union5 v13 = v11.case1.v0;
                switch (v13.tag) {
                    case 0: { // ChanceCommunityCard
                        Union6 v544 = v13.case0.v0; bool v545 = v13.case0.v1; static_array<Union3,2l> v546 = v13.case0.v2; int v547 = v13.case0.v3; static_array<int,2l> v548 = v13.case0.v4; int v549 = v13.case0.v5;
                        unsigned int v550 = v2;
                        Union3 v551; unsigned int v552;
                        Tuple0 tmp26 = draw_card_4(v5, v550);
                        v551 = tmp26.v0; v552 = tmp26.v1;
                        v2 = v552;
                        Union2 v553;
                        v553 = Union2{Union2_0{v551}};
                        v9.push(v553);
                        int v554;
                        v554 = 2l;
                        int v555; int v556;
                        Tuple1 tmp27 = Tuple1{0l, 0l};
                        v555 = tmp27.v0; v556 = tmp27.v1;
                        while (while_method_3(v555)){
                            int v558;
                            v558 = v548[v555];
                            bool v560;
                            v560 = v556 >= v558;
                            int v561;
                            if (v560){
                                v561 = v556;
                            } else {
                                v561 = v558;
                            }
                            v556 = v561;
                            v555 += 1l ;
                        }
                        static_array<int,2l> v562;
                        int v564;
                        v564 = 0l;
                        while (while_method_3(v564)){
                            v562[v564] = v556;
                            v564 += 1l ;
                        }
                        Union6 v566;
                        v566 = Union6{Union6_1{v551}};
                        Union5 v567;
                        v567 = Union5{Union5_2{v566, true, v546, 0l, v562, v554}};
                        v593 = Union7{Union7_1{v567}};
                        break;
                    }
                    case 1: { // ChanceInit
                        unsigned int v569 = v2;
                        Union3 v570; unsigned int v571;
                        Tuple0 tmp28 = draw_card_4(v5, v569);
                        v570 = tmp28.v0; v571 = tmp28.v1;
                        v2 = v571;
                        unsigned int v572 = v2;
                        Union3 v573; unsigned int v574;
                        Tuple0 tmp29 = draw_card_4(v5, v572);
                        v573 = tmp29.v0; v574 = tmp29.v1;
                        v2 = v574;
                        Union2 v575;
                        v575 = Union2{Union2_2{0l, v570}};
                        v9.push(v575);
                        Union2 v576;
                        v576 = Union2{Union2_2{1l, v573}};
                        v9.push(v576);
                        int v577;
                        v577 = 2l;
                        static_array<int,2l> v578;
                        v578[0l] = 1l;
                        v578[1l] = 1l;
                        static_array<Union3,2l> v580;
                        v580[0l] = v570;
                        v580[1l] = v573;
                        Union6 v582;
                        v582 = Union6{Union6_0{}};
                        Union5 v583;
                        v583 = Union5{Union5_2{v582, true, v580, 0l, v578, v577}};
                        v593 = Union7{Union7_1{v583}};
                        break;
                    }
                    case 2: { // Round
                        Union6 v54 = v13.case2.v0; bool v55 = v13.case2.v1; static_array<Union3,2l> v56 = v13.case2.v2; int v57 = v13.case2.v3; static_array<int,2l> v58 = v13.case2.v4; int v59 = v13.case2.v5;
                        static_array<Union1,2l> v60 = v4;
                        Union1 v61;
                        v61 = v60[v57];
                        Union4 v360;
                        switch (v61.tag) {
                            case 0: { // Computer
                                static_array_list<Union2,32l> & v63 = v3;
                                unsigned int * v64;
                                v64 = reinterpret_cast<unsigned int *>(&v0[12582912ull]);
                                float * v66;
                                v66 = reinterpret_cast<float *>(&v0[0ull]);
                                int v68;
                                v68 = threadIdx.x;
                                int v69;
                                v69 = blockIdx.x;
                                int v70;
                                v70 = v69 * 512l;
                                int v71;
                                v71 = v68 + v70;
                                unsigned long long v72;
                                v72 = (unsigned long long)v71;
                                curandStatePhilox4_32_10_t v73;
                                curand_init(12344321ull,v72,0ull,&v73);
                                float * v74;
                                v74 = reinterpret_cast<float *>(&v0[0ull]);
                                int v76;
                                v76 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v76 && v76 < 24l);
                                int v77;
                                v77 = 65536l * v76;
                                int v78;
                                v78 = threadIdx.x;
                                int v79;
                                v79 = blockIdx.x;
                                int v80;
                                v80 = v79 * 512l;
                                int v81;
                                v81 = v78 + v80;
                                unsigned long long v82;
                                v82 = (unsigned long long)v81;
                                curandStatePhilox4_32_10_t v83;
                                curand_init(12344321ull,v82,0ull,&v83);
                                int v84;
                                v84 = threadIdx.x;
                                int v85;
                                v85 = v84;
                                while (while_method_4(v85)){
                                    bool v87;
                                    v87 = 0l <= v85;
                                    bool v88;
                                    v88 = v87 == false;
                                    if (v88){
                                        assert("The index needs to be zero or positive." && v87);
                                    } else {
                                    }
                                    int v90;
                                    v90 = v85 % 128l;
                                    int v91;
                                    v91 = v85 / 128l;
                                    bool v92;
                                    v92 = v91 < 512l;
                                    bool v93;
                                    v93 = v92 == false;
                                    if (v93){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v92);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v91 && v91 < 512l);
                                    assert("Tensor range check" && 0 <= v90 && v90 < 128l);
                                    int v95;
                                    v95 = v90 + v77;
                                    int v96;
                                    v96 = 128l * v91;
                                    int v97;
                                    v97 = v96 + v95;
                                    v74[v97] = 0.0f;
                                    v85 += 512l ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                int v98;
                                v98 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v98 && v98 < 512l);
                                int v99;
                                v99 = 128l * v98;
                                int v100;
                                v100 = v99 + v77;
                                static_array_list<Union8,10l> v101;
                                v101 = static_array_list<Union8,10l>{};
                                int v103;
                                v103 = v63.length;
                                int v104;
                                v104 = 0l;
                                while (while_method_5(v103, v104)){
                                    Union2 v106;
                                    v106 = v63[v104];
                                    Union9 v125;
                                    switch (v106.tag) {
                                        case 0: { // CommunityCardIs
                                            Union3 v115 = v106.case0.v0;
                                            Union8 v116;
                                            v116 = Union8{Union8_1{v115}};
                                            v125 = Union9{Union9_1{v116}};
                                            break;
                                        }
                                        case 1: { // PlayerAction
                                            int v118 = v106.case1.v0; Union4 v119 = v106.case1.v1;
                                            Union8 v120;
                                            v120 = Union8{Union8_0{v119}};
                                            v125 = Union9{Union9_1{v120}};
                                            break;
                                        }
                                        case 2: { // PlayerGotCard
                                            int v108 = v106.case2.v0; Union3 v109 = v106.case2.v1;
                                            bool v110;
                                            v110 = v108 == v57;
                                            if (v110){
                                                Union8 v111;
                                                v111 = Union8{Union8_1{v109}};
                                                v125 = Union9{Union9_1{v111}};
                                            } else {
                                                v125 = Union9{Union9_0{}};
                                            }
                                            break;
                                        }
                                        default: {
                                            v125 = Union9{Union9_0{}};
                                        }
                                    }
                                    switch (v125.tag) {
                                        case 0: { // None
                                            break;
                                        }
                                        case 1: { // Some
                                            Union8 v126 = v125.case1.v0;
                                            v101.push(v126);
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false); __trap();
                                        }
                                    }
                                    v104 += 1l ;
                                }
                                float * v127;
                                v127 = v74+v100;
                                int v129;
                                v129 = v101.length;
                                bool v130;
                                v130 = v129 == 0l;
                                if (v130){
                                    v127[0l] = 1.0f;
                                } else {
                                }
                                int v131;
                                v131 = v101.length;
                                int v132;
                                v132 = 0l;
                                while (while_method_5(v131, v132)){
                                    Union8 v134;
                                    v134 = v101[v132];
                                    int v136;
                                    v136 = v132 * 6l;
                                    int v137;
                                    v137 = 1l + v136;
                                    switch (v134.tag) {
                                        case 0: { // C1of2
                                            Union4 v138 = v134.case0.v0;
                                            switch (v138.tag) {
                                                case 0: { // Call
                                                    v127[v137] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // Fold
                                                    int v139;
                                                    v139 = v137 + 1l;
                                                    v127[v139] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // Raise
                                                    int v140;
                                                    v140 = v137 + 2l;
                                                    v127[v140] = 1.0f;
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false); __trap();
                                                }
                                            }
                                            break;
                                        }
                                        case 1: { // C2of2
                                            Union3 v141 = v134.case1.v0;
                                            int v142;
                                            v142 = v137 + 3l;
                                            switch (v141.tag) {
                                                case 0: { // Jack
                                                    v127[v142] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // King
                                                    int v143;
                                                    v143 = v142 + 1l;
                                                    v127[v143] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // Queen
                                                    int v144;
                                                    v144 = v142 + 2l;
                                                    v127[v144] = 1.0f;
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
                                    v132 += 1l ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                int v145;
                                v145 = 0l;
                                while (while_method_6(v145)){
                                    float * v147;
                                    v147 = reinterpret_cast<float *>(&v0[0ull]);
                                    float * v149;
                                    v149 = reinterpret_cast<float *>(&v1[0ull]);
                                    assert("Tensor range check" && 0 <= v145 && v145 < 4l);
                                    int v151;
                                    v151 = 16384l * v145;
                                    float * v152;
                                    v152 = reinterpret_cast<float *>(&v0[6291456ull]);
                                    int v154;
                                    v154 = blockIdx.x;
                                    assert("Tensor range check" && 0 <= v154 && v154 < 24l);
                                    int v155;
                                    v155 = 65536l * v154;
                                    int v156;
                                    v156 = blockIdx.x;
                                    assert("Tensor range check" && 0 <= v156 && v156 < 24l);
                                    int v157;
                                    v157 = 65536l * v156;
                                    method_6(v149, v151, v152, v157, v147, v155);
                                    unsigned int * v158;
                                    v158 = reinterpret_cast<unsigned int *>(&v0[12582912ull]);
                                    assert("Tensor range check" && 0 <= v145 && v145 < 4l);
                                    int v160;
                                    v160 = 12288l * v145;
                                    method_7(v158, v160, v152);
                                    int * v161;
                                    v161 = reinterpret_cast<int *>(&v1[262144ull]);
                                    float * v163;
                                    v163 = reinterpret_cast<float *>(&v1[262160ull]);
                                    float * v165;
                                    v165 = reinterpret_cast<float *>(&v1[524304ull]);
                                    float * v167;
                                    v167 = reinterpret_cast<float *>(&v1[786448ull]);
                                    float * v169;
                                    v169 = reinterpret_cast<float *>(&v1[1048592ull]);
                                    float * v171;
                                    v171 = reinterpret_cast<float *>(&v1[1310736ull]);
                                    float * v173;
                                    v173 = reinterpret_cast<float *>(&v1[1572880ull]);
                                    float * v175;
                                    v175 = reinterpret_cast<float *>(&v1[1835024ull]);
                                    int * v177;
                                    v177 = reinterpret_cast<int *>(&v0[12779520ull]);
                                    float * v179;
                                    v179 = reinterpret_cast<float *>(&v0[15925248ull]);
                                    int * v181;
                                    v181 = reinterpret_cast<int *>(&v0[19070976ull]);
                                    int * v183;
                                    v183 = reinterpret_cast<int *>(&v0[22216704ull]);
                                    double * v185;
                                    v185 = reinterpret_cast<double *>(&v0[25362432ull]);
                                    double * v187;
                                    v187 = reinterpret_cast<double *>(&v0[37945344ull]);
                                    double * v189;
                                    v189 = reinterpret_cast<double *>(&v1[2097168ull]);
                                    double * v191;
                                    v191 = reinterpret_cast<double *>(&v1[2883600ull]);
                                    int * v193;
                                    v193 = reinterpret_cast<int *>(&v1[3670032ull]);
                                    v145 += 1l ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                int * v195;
                                v195 = reinterpret_cast<int *>(&v1[262144ull]);
                                float * v197;
                                v197 = reinterpret_cast<float *>(&v1[262160ull]);
                                float * v199;
                                v199 = reinterpret_cast<float *>(&v1[524304ull]);
                                float * v201;
                                v201 = reinterpret_cast<float *>(&v1[786448ull]);
                                float * v203;
                                v203 = reinterpret_cast<float *>(&v1[1048592ull]);
                                float * v205;
                                v205 = reinterpret_cast<float *>(&v1[1310736ull]);
                                float * v207;
                                v207 = reinterpret_cast<float *>(&v1[1572880ull]);
                                float * v209;
                                v209 = reinterpret_cast<float *>(&v1[1835024ull]);
                                int v211;
                                v211 = 0l;
                                int v212;
                                v212 = 4l;
                                int v213;
                                v213 = int_range_10(v212, v211, v5);
                                extern __shared__ unsigned char v214[];
                                int * v215;
                                v215 = reinterpret_cast<int *>(&v214[0ull]);
                                int v217;
                                v217 = threadIdx.x;
                                bool v218;
                                v218 = v217 == 0l;
                                if (v218){
                                    v215[0l] = v213;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                int v219;
                                v219 = v215[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                unsigned int * v220;
                                v220 = reinterpret_cast<unsigned int *>(&v0[12582912ull]);
                                int v222;
                                v222 = blockIdx.x;
                                int v223;
                                v223 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v219 && v219 < 4l);
                                assert("Tensor range check" && 0 <= v222 && v222 < 24l);
                                assert("Tensor range check" && 0 <= v223 && v223 < 512l);
                                int v224;
                                v224 = 512l * v222;
                                int v225;
                                v225 = v224 + v223;
                                int v226;
                                v226 = 12288l * v219;
                                int v227;
                                v227 = v226 + v225;
                                unsigned int v228;
                                v228 = v220[v227];
                                int v229;
                                v229 = (int)v228;
                                float v230; int v231;
                                Tuple2 tmp30 = method_8(v5, v195, v197, v199, v201, v203, v205, v207, v209, v229, v219);
                                v230 = tmp30.v0; v231 = tmp30.v1;
                                extern __shared__ unsigned char v232[];
                                float * v233;
                                v233 = reinterpret_cast<float *>(&v232[0ull]);
                                int * v235;
                                v235 = reinterpret_cast<int *>(&v232[16ull]);
                                int v237;
                                v237 = threadIdx.x;
                                bool v238;
                                v238 = v237 == 0l;
                                if (v238){
                                    v233[0l] = v230;
                                    v235[0l] = v231;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                float v239;
                                v239 = v233[0l];
                                int v240;
                                v240 = v235[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                double * v241;
                                v241 = reinterpret_cast<double *>(&v1[2097168ull]);
                                double * v243;
                                v243 = reinterpret_cast<double *>(&v1[2883600ull]);
                                int * v245;
                                v245 = reinterpret_cast<int *>(&v1[3670032ull]);
                                int * v247;
                                v247 = reinterpret_cast<int *>(&v0[12779520ull]);
                                float * v249;
                                v249 = reinterpret_cast<float *>(&v0[15925248ull]);
                                int * v251;
                                v251 = reinterpret_cast<int *>(&v0[19070976ull]);
                                int * v253;
                                v253 = reinterpret_cast<int *>(&v0[22216704ull]);
                                double * v255;
                                v255 = reinterpret_cast<double *>(&v0[25362432ull]);
                                double * v257;
                                v257 = reinterpret_cast<double *>(&v0[37945344ull]);
                                int v259;
                                v259 = threadIdx.x;
                                int v260;
                                v260 = blockIdx.x;
                                int v261;
                                v261 = v260 * 512l;
                                int v262;
                                v262 = v259 + v261;
                                int v263;
                                v263 = 0l;
                                while (while_method_6(v263)){
                                    unsigned int * v265;
                                    v265 = reinterpret_cast<unsigned int *>(&v0[12582912ull]);
                                    int v267;
                                    v267 = blockIdx.x;
                                    int v268;
                                    v268 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v263 && v263 < 4l);
                                    assert("Tensor range check" && 0 <= v267 && v267 < 24l);
                                    assert("Tensor range check" && 0 <= v268 && v268 < 512l);
                                    int v269;
                                    v269 = 512l * v267;
                                    int v270;
                                    v270 = v269 + v268;
                                    int v271;
                                    v271 = 12288l * v263;
                                    int v272;
                                    v272 = v271 + v270;
                                    unsigned int v273;
                                    v273 = v265[v272];
                                    int v274;
                                    v274 = (int)v273;
                                    float v275;
                                    v275 = method_16(v195, v197, v199, v201, v203, v205, v207, v209, v274, v263, v240);
                                    double v276;
                                    v276 = (double)v239;
                                    double v277;
                                    v277 = log(v276);
                                    double v278;
                                    v278 = (double)v275;
                                    double v279;
                                    v279 = log(v278);
                                    assert("Tensor range check" && 0 <= v263 && v263 < 4l);
                                    assert("Tensor range check" && 0 <= v262 && v262 < 12288l);
                                    assert("Tensor range check" && 0 <= v57 && v57 < 2l);
                                    int v280;
                                    v280 = 2l * v262;
                                    int v281;
                                    v281 = v280 + v57;
                                    int v282;
                                    v282 = 24576l * v263;
                                    int v283;
                                    v283 = v282 + v281;
                                    double v284;
                                    v284 = v241[v283];
                                    double v285;
                                    v285 = v243[v283];
                                    double v286;
                                    v286 = v279 + v284;
                                    double v287;
                                    v287 = v277 + v285;
                                    assert("Tensor range check" && 0 <= v263 && v263 < 4l);
                                    assert("Tensor range check" && 0 <= v262 && v262 < 12288l);
                                    assert("Tensor range check" && 0 <= v57 && v57 < 2l);
                                    v241[v283] = v286;
                                    v243[v283] = v287;
                                    v263 += 1l ;
                                }
                                bool v288;
                                v288 = 0l == v240;
                                Union10 v297;
                                if (v288){
                                    v297 = Union10{Union10_1{}};
                                } else {
                                    bool v290;
                                    v290 = 1l == v240;
                                    if (v290){
                                        v297 = Union10{Union10_0{}};
                                    } else {
                                        bool v292;
                                        v292 = 2l == v240;
                                        if (v292){
                                            v297 = Union10{Union10_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v297.tag) {
                                    case 0: { // AA_Call
                                        v360 = Union4{Union4_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v298;
                                        v298 = v58[0l];
                                        int v300; int v301;
                                        Tuple1 tmp33 = Tuple1{1l, v298};
                                        v300 = tmp33.v0; v301 = tmp33.v1;
                                        while (while_method_3(v300)){
                                            int v303;
                                            v303 = v58[v300];
                                            bool v305;
                                            v305 = v301 >= v303;
                                            int v306;
                                            if (v305){
                                                v306 = v301;
                                            } else {
                                                v306 = v303;
                                            }
                                            v301 = v306;
                                            v300 += 1l ;
                                        }
                                        int v307;
                                        v307 = v58[v57];
                                        bool v309;
                                        v309 = v307 == v301;
                                        if (v309){
                                            v360 = Union4{Union4_0{}};
                                        } else {
                                            v360 = Union4{Union4_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v314;
                                        v314 = v59 > 0l;
                                        if (v314){
                                            v360 = Union4{Union4_2{}};
                                        } else {
                                            v360 = Union4{Union4_0{}};
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                break;
                            }
                            case 1: { // Random
                                static_array_list<Union4,3l> v321;
                                v321 = static_array_list<Union4,3l>{};
                                v321.unsafe_set_length(1l);
                                Union4 v323;
                                v323 = Union4{Union4_0{}};
                                v321[0l] = v323;
                                int v325;
                                v325 = v58[0l];
                                int v327;
                                v327 = v58[1l];
                                bool v329;
                                v329 = v325 == v327;
                                bool v330;
                                v330 = v329 != true;
                                if (v330){
                                    Union4 v331;
                                    v331 = Union4{Union4_1{}};
                                    v321.push(v331);
                                } else {
                                }
                                bool v332;
                                v332 = v59 > 0l;
                                if (v332){
                                    Union4 v333;
                                    v333 = Union4{Union4_2{}};
                                    v321.push(v333);
                                } else {
                                }
                                int v334;
                                v334 = v321.length;
                                int v335;
                                v335 = v334 - 1l;
                                int v336;
                                v336 = 0l;
                                while (while_method_5(v335, v336)){
                                    int v338;
                                    v338 = v321.length;
                                    int v339;
                                    v339 = int_range_10(v338, v336, v5);
                                    Union4 v340;
                                    v340 = v321[v336];
                                    Union4 v342;
                                    v342 = v321[v339];
                                    v321[v336] = v342;
                                    v321[v339] = v340;
                                    v336 += 1l ;
                                }
                                Union4 v344;
                                v344 = v321.pop();
                                int v345;
                                v345 = sizeof(Union4);
                                unsigned long long v346;
                                v346 = (unsigned long long)v345;
                                bool v347;
                                v347 = v346 <= 81920ull;
                                bool v348;
                                v348 = v347 == false;
                                if (v348){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v347);
                                } else {
                                }
                                extern __shared__ unsigned char v350[];
                                bool v351;
                                v351 = v346 <= v346;
                                bool v352;
                                v352 = v351 == false;
                                if (v352){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v351);
                                } else {
                                }
                                Union4 * v354;
                                v354 = reinterpret_cast<Union4 *>(&v350[0ull]);
                                int v356;
                                v356 = threadIdx.x;
                                bool v357;
                                v357 = v356 == 0l;
                                if (v357){
                                    v354[0l] = v344;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                Union4 v358;
                                v358 = v354[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                v360 = v358;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union2 v361;
                        v361 = Union2{Union2_1{v57, v360}};
                        v9.push(v361);
                        Union5 v447;
                        switch (v54.tag) {
                            case 0: { // None
                                switch (v360.tag) {
                                    case 0: { // Call
                                        if (v55){
                                            bool v411;
                                            v411 = v57 == 0l;
                                            int v412;
                                            if (v411){
                                                v412 = 1l;
                                            } else {
                                                v412 = 0l;
                                            }
                                            v447 = Union5{Union5_2{v54, false, v56, v412, v58, v59}};
                                        } else {
                                            v447 = Union5{Union5_0{v54, v55, v56, v57, v58, v59}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v447 = Union5{Union5_5{v54, v55, v56, v57, v58, v59}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v416;
                                        v416 = v59 > 0l;
                                        if (v416){
                                            bool v417;
                                            v417 = v57 == 0l;
                                            int v418;
                                            if (v417){
                                                v418 = 1l;
                                            } else {
                                                v418 = 0l;
                                            }
                                            int v419;
                                            v419 = -1l + v59;
                                            int v420; int v421;
                                            Tuple1 tmp34 = Tuple1{0l, 0l};
                                            v420 = tmp34.v0; v421 = tmp34.v1;
                                            while (while_method_3(v420)){
                                                int v423;
                                                v423 = v58[v420];
                                                bool v425;
                                                v425 = v421 >= v423;
                                                int v426;
                                                if (v425){
                                                    v426 = v421;
                                                } else {
                                                    v426 = v423;
                                                }
                                                v421 = v426;
                                                v420 += 1l ;
                                            }
                                            static_array<int,2l> v427;
                                            int v429;
                                            v429 = 0l;
                                            while (while_method_3(v429)){
                                                v427[v429] = v421;
                                                v429 += 1l ;
                                            }
                                            static_array<int,2l> v431;
                                            int v433;
                                            v433 = 0l;
                                            while (while_method_3(v433)){
                                                int v435;
                                                v435 = v427[v433];
                                                bool v437;
                                                v437 = v433 == v57;
                                                int v439;
                                                if (v437){
                                                    int v438;
                                                    v438 = v435 + 2l;
                                                    v439 = v438;
                                                } else {
                                                    v439 = v435;
                                                }
                                                v431[v433] = v439;
                                                v433 += 1l ;
                                            }
                                            v447 = Union5{Union5_2{v54, false, v56, v418, v431, v419}};
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
                                Union3 v362 = v54.case1.v0;
                                switch (v360.tag) {
                                    case 0: { // Call
                                        if (v55){
                                            bool v364;
                                            v364 = v57 == 0l;
                                            int v365;
                                            if (v364){
                                                v365 = 1l;
                                            } else {
                                                v365 = 0l;
                                            }
                                            v447 = Union5{Union5_2{v54, false, v56, v365, v58, v59}};
                                        } else {
                                            int v367; int v368;
                                            Tuple1 tmp35 = Tuple1{0l, 0l};
                                            v367 = tmp35.v0; v368 = tmp35.v1;
                                            while (while_method_3(v367)){
                                                int v370;
                                                v370 = v58[v367];
                                                bool v372;
                                                v372 = v368 >= v370;
                                                int v373;
                                                if (v372){
                                                    v373 = v368;
                                                } else {
                                                    v373 = v370;
                                                }
                                                v368 = v373;
                                                v367 += 1l ;
                                            }
                                            static_array<int,2l> v374;
                                            int v376;
                                            v376 = 0l;
                                            while (while_method_3(v376)){
                                                v374[v376] = v368;
                                                v376 += 1l ;
                                            }
                                            v447 = Union5{Union5_4{v54, v55, v56, v57, v374, v59}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v447 = Union5{Union5_5{v54, v55, v56, v57, v58, v59}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v380;
                                        v380 = v59 > 0l;
                                        if (v380){
                                            bool v381;
                                            v381 = v57 == 0l;
                                            int v382;
                                            if (v381){
                                                v382 = 1l;
                                            } else {
                                                v382 = 0l;
                                            }
                                            int v383;
                                            v383 = -1l + v59;
                                            int v384; int v385;
                                            Tuple1 tmp36 = Tuple1{0l, 0l};
                                            v384 = tmp36.v0; v385 = tmp36.v1;
                                            while (while_method_3(v384)){
                                                int v387;
                                                v387 = v58[v384];
                                                bool v389;
                                                v389 = v385 >= v387;
                                                int v390;
                                                if (v389){
                                                    v390 = v385;
                                                } else {
                                                    v390 = v387;
                                                }
                                                v385 = v390;
                                                v384 += 1l ;
                                            }
                                            static_array<int,2l> v391;
                                            int v393;
                                            v393 = 0l;
                                            while (while_method_3(v393)){
                                                v391[v393] = v385;
                                                v393 += 1l ;
                                            }
                                            static_array<int,2l> v395;
                                            int v397;
                                            v397 = 0l;
                                            while (while_method_3(v397)){
                                                int v399;
                                                v399 = v391[v397];
                                                bool v401;
                                                v401 = v397 == v57;
                                                int v403;
                                                if (v401){
                                                    int v402;
                                                    v402 = v399 + 4l;
                                                    v403 = v402;
                                                } else {
                                                    v403 = v399;
                                                }
                                                v395[v397] = v403;
                                                v397 += 1l ;
                                            }
                                            v447 = Union5{Union5_2{v54, false, v56, v382, v395, v383}};
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
                        v593 = Union7{Union7_1{v447}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union6 v449 = v13.case3.v0; bool v450 = v13.case3.v1; static_array<Union3,2l> v451 = v13.case3.v2; int v452 = v13.case3.v3; static_array<int,2l> v453 = v13.case3.v4; int v454 = v13.case3.v5; Union4 v455 = v13.case3.v6;
                        Union2 v456;
                        v456 = Union2{Union2_1{v452, v455}};
                        v9.push(v456);
                        Union5 v542;
                        switch (v449.tag) {
                            case 0: { // None
                                switch (v455.tag) {
                                    case 0: { // Call
                                        if (v450){
                                            bool v506;
                                            v506 = v452 == 0l;
                                            int v507;
                                            if (v506){
                                                v507 = 1l;
                                            } else {
                                                v507 = 0l;
                                            }
                                            v542 = Union5{Union5_2{v449, false, v451, v507, v453, v454}};
                                        } else {
                                            v542 = Union5{Union5_0{v449, v450, v451, v452, v453, v454}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v542 = Union5{Union5_5{v449, v450, v451, v452, v453, v454}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v511;
                                        v511 = v454 > 0l;
                                        if (v511){
                                            bool v512;
                                            v512 = v452 == 0l;
                                            int v513;
                                            if (v512){
                                                v513 = 1l;
                                            } else {
                                                v513 = 0l;
                                            }
                                            int v514;
                                            v514 = -1l + v454;
                                            int v515; int v516;
                                            Tuple1 tmp37 = Tuple1{0l, 0l};
                                            v515 = tmp37.v0; v516 = tmp37.v1;
                                            while (while_method_3(v515)){
                                                int v518;
                                                v518 = v453[v515];
                                                bool v520;
                                                v520 = v516 >= v518;
                                                int v521;
                                                if (v520){
                                                    v521 = v516;
                                                } else {
                                                    v521 = v518;
                                                }
                                                v516 = v521;
                                                v515 += 1l ;
                                            }
                                            static_array<int,2l> v522;
                                            int v524;
                                            v524 = 0l;
                                            while (while_method_3(v524)){
                                                v522[v524] = v516;
                                                v524 += 1l ;
                                            }
                                            static_array<int,2l> v526;
                                            int v528;
                                            v528 = 0l;
                                            while (while_method_3(v528)){
                                                int v530;
                                                v530 = v522[v528];
                                                bool v532;
                                                v532 = v528 == v452;
                                                int v534;
                                                if (v532){
                                                    int v533;
                                                    v533 = v530 + 2l;
                                                    v534 = v533;
                                                } else {
                                                    v534 = v530;
                                                }
                                                v526[v528] = v534;
                                                v528 += 1l ;
                                            }
                                            v542 = Union5{Union5_2{v449, false, v451, v513, v526, v514}};
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
                                Union3 v457 = v449.case1.v0;
                                switch (v455.tag) {
                                    case 0: { // Call
                                        if (v450){
                                            bool v459;
                                            v459 = v452 == 0l;
                                            int v460;
                                            if (v459){
                                                v460 = 1l;
                                            } else {
                                                v460 = 0l;
                                            }
                                            v542 = Union5{Union5_2{v449, false, v451, v460, v453, v454}};
                                        } else {
                                            int v462; int v463;
                                            Tuple1 tmp38 = Tuple1{0l, 0l};
                                            v462 = tmp38.v0; v463 = tmp38.v1;
                                            while (while_method_3(v462)){
                                                int v465;
                                                v465 = v453[v462];
                                                bool v467;
                                                v467 = v463 >= v465;
                                                int v468;
                                                if (v467){
                                                    v468 = v463;
                                                } else {
                                                    v468 = v465;
                                                }
                                                v463 = v468;
                                                v462 += 1l ;
                                            }
                                            static_array<int,2l> v469;
                                            int v471;
                                            v471 = 0l;
                                            while (while_method_3(v471)){
                                                v469[v471] = v463;
                                                v471 += 1l ;
                                            }
                                            v542 = Union5{Union5_4{v449, v450, v451, v452, v469, v454}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v542 = Union5{Union5_5{v449, v450, v451, v452, v453, v454}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v475;
                                        v475 = v454 > 0l;
                                        if (v475){
                                            bool v476;
                                            v476 = v452 == 0l;
                                            int v477;
                                            if (v476){
                                                v477 = 1l;
                                            } else {
                                                v477 = 0l;
                                            }
                                            int v478;
                                            v478 = -1l + v454;
                                            int v479; int v480;
                                            Tuple1 tmp39 = Tuple1{0l, 0l};
                                            v479 = tmp39.v0; v480 = tmp39.v1;
                                            while (while_method_3(v479)){
                                                int v482;
                                                v482 = v453[v479];
                                                bool v484;
                                                v484 = v480 >= v482;
                                                int v485;
                                                if (v484){
                                                    v485 = v480;
                                                } else {
                                                    v485 = v482;
                                                }
                                                v480 = v485;
                                                v479 += 1l ;
                                            }
                                            static_array<int,2l> v486;
                                            int v488;
                                            v488 = 0l;
                                            while (while_method_3(v488)){
                                                v486[v488] = v480;
                                                v488 += 1l ;
                                            }
                                            static_array<int,2l> v490;
                                            int v492;
                                            v492 = 0l;
                                            while (while_method_3(v492)){
                                                int v494;
                                                v494 = v486[v492];
                                                bool v496;
                                                v496 = v492 == v452;
                                                int v498;
                                                if (v496){
                                                    int v497;
                                                    v497 = v494 + 4l;
                                                    v498 = v497;
                                                } else {
                                                    v498 = v494;
                                                }
                                                v490[v492] = v498;
                                                v492 += 1l ;
                                            }
                                            v542 = Union5{Union5_2{v449, false, v451, v477, v490, v478}};
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
                        v593 = Union7{Union7_1{v542}};
                        break;
                    }
                    case 4: { // TerminalCall
                        Union6 v30 = v13.case4.v0; bool v31 = v13.case4.v1; static_array<Union3,2l> v32 = v13.case4.v2; int v33 = v13.case4.v3; static_array<int,2l> v34 = v13.case4.v4; int v35 = v13.case4.v5;
                        int v36;
                        v36 = v34[v33];
                        Union11 v38;
                        v38 = compare_hands_11(v30, v31, v32, v33, v34, v35);
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
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        int v45;
                        v45 = -v44;
                        bool v46;
                        v46 = v44 >= v45;
                        int v47;
                        if (v46){
                            v47 = v44;
                        } else {
                            v47 = v45;
                        }
                        float v48;
                        v48 = (float)v43;
                        v7[v47] = v48;
                        bool v49;
                        v49 = v47 == 0l;
                        int v50;
                        if (v49){
                            v50 = 1l;
                        } else {
                            v50 = 0l;
                        }
                        float v51;
                        v51 = -v48;
                        v7[v50] = v51;
                        Union2 v52;
                        v52 = Union2{Union2_3{v32, v43, v44}};
                        v9.push(v52);
                        v593 = Union7{Union7_0{}};
                        break;
                    }
                    case 5: { // TerminalFold
                        Union6 v14 = v13.case5.v0; bool v15 = v13.case5.v1; static_array<Union3,2l> v16 = v13.case5.v2; int v17 = v13.case5.v3; static_array<int,2l> v18 = v13.case5.v4; int v19 = v13.case5.v5;
                        int v20;
                        v20 = v18[v17];
                        int v22;
                        v22 = -v20;
                        float v23;
                        v23 = (float)v22;
                        v7[v17] = v23;
                        bool v24;
                        v24 = v17 == 0l;
                        int v25;
                        if (v24){
                            v25 = 1l;
                        } else {
                            v25 = 0l;
                        }
                        float v26;
                        v26 = -v23;
                        v7[v25] = v26;
                        int v27;
                        if (v24){
                            v27 = 1l;
                        } else {
                            v27 = 0l;
                        }
                        Union2 v28;
                        v28 = Union2{Union2_3{v16, v20, v27}};
                        v9.push(v28);
                        v593 = Union7{Union7_0{}};
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
        v11 = v593;
    }
    return v7;
}
__device__ inline bool while_method_11(int v0){
    bool v1;
    v1 = v0 < 256l;
    return v1;
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1, unsigned long long v2, unsigned char * v3, unsigned long long v4, float * v5, float * v6, float * v7) {
    bool v8;
    v8 = 3866640ull == v4;
    bool v9;
    v9 = v8 == false;
    if (v9){
        assert("The params needs to have matching offsets." && v8);
    } else {
    }
    bool v11;
    v11 = 50528256ull == v2;
    bool v12;
    v12 = v11 == false;
    if (v12){
        assert("The outputs needs to have matching offsets." && v11);
    } else {
    }
    auto v14 = cooperative_groups::this_grid();
    unsigned long long v15;
    v15 = clock64();
    int v16;
    v16 = threadIdx.x;
    int v17;
    v17 = blockIdx.x;
    int v18;
    v18 = v17 * 512l;
    int v19;
    v19 = v16 + v18;
    unsigned long long v20;
    v20 = (unsigned long long)v19;
    curandStatePhilox4_32_10_t v21;
    curand_init(v15,v20,0ull,&v21);
    Union0 v22;
    v22 = f_0(v0);
    switch (v22.tag) {
        case 0: { // StartTraining
            static_array<Union1,2l> v23;
            Union1 v25;
            v25 = Union1{Union1_0{}};
            v23[0l] = v25;
            Union1 v27;
            v27 = Union1{Union1_0{}};
            v23[1l] = v27;
            int v29;
            v29 = 0l;
            while (while_method_0(v29)){
                int v31;
                v31 = 0l;
                while (while_method_1(v31)){
                    static_array<Union1,2l> & v33 = v23;
                    unsigned int v34 = 63ul;
                    static_array_list<Union2,32l> v35;
                    v35 = static_array_list<Union2,32l>{};
                    static_array_list<Union2,32l> & v37 = v35;
                    Union5 v38;
                    v38 = Union5{Union5_1{}};
                    static_array<float,2l> v39;
                    v39 = method_3(v1, v3, v34, v37, v33, v21, v38);
                    unsigned int * v40;
                    v40 = reinterpret_cast<unsigned int *>(&v1[12582912ull]);
                    int * v42;
                    v42 = reinterpret_cast<int *>(&v3[262144ull]);
                    float * v44;
                    v44 = reinterpret_cast<float *>(&v3[262160ull]);
                    float * v46;
                    v46 = reinterpret_cast<float *>(&v3[524304ull]);
                    float * v48;
                    v48 = reinterpret_cast<float *>(&v3[786448ull]);
                    float * v50;
                    v50 = reinterpret_cast<float *>(&v3[1048592ull]);
                    float * v52;
                    v52 = reinterpret_cast<float *>(&v3[1310736ull]);
                    float * v54;
                    v54 = reinterpret_cast<float *>(&v3[1572880ull]);
                    float * v56;
                    v56 = reinterpret_cast<float *>(&v3[1835024ull]);
                    int * v58;
                    v58 = reinterpret_cast<int *>(&v1[12779520ull]);
                    float * v60;
                    v60 = reinterpret_cast<float *>(&v1[15925248ull]);
                    int * v62;
                    v62 = reinterpret_cast<int *>(&v1[19070976ull]);
                    int * v64;
                    v64 = reinterpret_cast<int *>(&v1[22216704ull]);
                    double * v66;
                    v66 = reinterpret_cast<double *>(&v1[25362432ull]);
                    double * v68;
                    v68 = reinterpret_cast<double *>(&v1[37945344ull]);
                    double * v70;
                    v70 = reinterpret_cast<double *>(&v3[2097168ull]);
                    double * v72;
                    v72 = reinterpret_cast<double *>(&v3[2883600ull]);
                    int * v74;
                    v74 = reinterpret_cast<int *>(&v3[3670032ull]);
                    int v76;
                    v76 = 0l;
                    while (while_method_6(v76)){
                        int v78;
                        v78 = threadIdx.x;
                        int v79;
                        v79 = blockIdx.x;
                        int v80;
                        v80 = v79 * 512l;
                        int v81;
                        v81 = v78 + v80;
                        float v82[2l];
                        int v83;
                        v83 = 0l;
                        while (while_method_3(v83)){
                            float v85;
                            v85 = v39[v83];
                            v82[v83] = v85;
                            v83 += 1l ;
                        }
                        assert("Tensor range check" && 0 <= v76 && v76 < 4l);
                        assert("Tensor range check" && 0 <= v81 && v81 < 12288l);
                        int v87;
                        v87 = 12288l * v76;
                        int v88;
                        v88 = v87 + v81;
                        int v89;
                        v89 = v74[v88];
                        int v90;
                        v90 = v89;
                        while (while_method_10(v90)){
                            v90 -= 1l ;
                            assert("Tensor range check" && 0 <= v76 && v76 < 4l);
                            assert("Tensor range check" && 0 <= v90 && v90 < 16l);
                            assert("Tensor range check" && 0 <= v81 && v81 < 12288l);
                            int v92;
                            v92 = 12288l * v90;
                            int v93;
                            v93 = v92 + v81;
                            int v94;
                            v94 = 196608l * v76;
                            int v95;
                            v95 = v94 + v93;
                            int v96;
                            v96 = v58[v95];
                            float v97;
                            v97 = v60[v95];
                            int v98;
                            v98 = v62[v95];
                            int v99;
                            v99 = v64[v95];
                            assert("Tensor range check" && 0 <= v98 && v98 < 2l);
                            float v100;
                            v100 = v82[v98];
                            assert("Tensor range check" && 0 <= v76 && v76 < 4l);
                            int v101;
                            v101 = 16384l * v76;
                            assert("Tensor range check" && 0 <= v99 && v99 < 4096l);
                            int v102;
                            v102 = 4l * v99;
                            int v103;
                            v103 = v102 + v101;
                            float * v104;
                            v104 = v44+v103;
                            float * v106;
                            v106 = v46+v103;
                            float * v108;
                            v108 = v48+v103;
                            float * v110;
                            v110 = v50+v103;
                            float * v112;
                            v112 = v52+v103;
                            float * v114;
                            v114 = v54+v103;
                            float * v116;
                            v116 = v56+v103;
                            assert("Tensor range check" && 0 <= v76 && v76 < 4l);
                            int v118;
                            v118 = 393216l * v76;
                            assert("Tensor range check" && 0 <= v90 && v90 < 16l);
                            int v119;
                            v119 = 24576l * v90;
                            int v120;
                            v120 = v119 + v118;
                            assert("Tensor range check" && 0 <= v81 && v81 < 12288l);
                            int v121;
                            v121 = 2l * v81;
                            int v122;
                            v122 = v121 + v120;
                            double v123[2l];
                            int v124;
                            v124 = 0l;
                            while (while_method_3(v124)){
                                assert("Tensor range check" && 0 <= v124 && v124 < 2l);
                                int v126;
                                v126 = v124 + v122;
                                double v127;
                                v127 = v66[v126];
                                bool v128;
                                v128 = v98 == v124;
                                double v129;
                                if (v128){
                                    v129 = 0.0;
                                } else {
                                    v129 = v127;
                                }
                                assert("Tensor range check" && 0 <= v124 && v124 < 2l);
                                v123[v124] = v129;
                                v124 += 1l ;
                            }
                            double v130;
                            v130 = 0.0;
                            int v131;
                            v131 = 0l;
                            while (while_method_3(v131)){
                                assert("Tensor range check" && 0 <= v131 && v131 < 2l);
                                double v133;
                                v133 = v123[v131];
                                double v134;
                                v134 = v130 + v133;
                                v130 = v134;
                                v131 += 1l ;
                            }
                            double v135;
                            v135 = 0.0;
                            int v136;
                            v136 = 0l;
                            while (while_method_3(v136)){
                                assert("Tensor range check" && 0 <= v136 && v136 < 2l);
                                int v138;
                                v138 = v136 + v122;
                                double v139;
                                v139 = v68[v138];
                                double v140;
                                v140 = v135 + v139;
                                v135 = v140;
                                v136 += 1l ;
                            }
                            double v141;
                            v141 = v130 - v135;
                            double v142;
                            v142 = exp(v141);
                            float v143;
                            v143 = (float)v142;
                            float v144;
                            v144 = v100 * v143;
                            assert("Tensor range check" && 0 <= v96 && v96 < 4l);
                            float * v145;
                            v145 = v114+v96;
                            float * v147;
                            v147 = v116+v96;
                            float v149;
                            v149 = atomicAdd(v145,v144);
                            float v150;
                            v150 = atomicAdd(v147,v143);
                            float * v151;
                            v151 = v106+0l;
                            float * v153;
                            v153 = v110+0l;
                            float * v155;
                            v155 = v112+0l;
                            int v157;
                            v157 = sizeof(float *);
                            unsigned long long v158;
                            v158 = (unsigned long long)v157;
                            unsigned long long v159;
                            v159 = 512ull * v158;
                            unsigned long long v160;
                            v160 = 8192ull + v159;
                            unsigned long long v161;
                            v161 = v160 + 16ull;
                            unsigned long long v162;
                            v162 = v161 - 1ull;
                            unsigned long long v163;
                            v163 = v162 % 16ull;
                            unsigned long long v164;
                            v164 = v162 - v163;
                            unsigned long long v165;
                            v165 = v164 + v159;
                            unsigned long long v166;
                            v166 = v165 + 16ull;
                            unsigned long long v167;
                            v167 = v166 - 1ull;
                            unsigned long long v168;
                            v168 = v167 % 16ull;
                            unsigned long long v169;
                            v169 = v167 - v168;
                            unsigned long long v170;
                            v170 = v169 + v159;
                            unsigned long long v171;
                            v171 = v170 + 16ull;
                            unsigned long long v172;
                            v172 = v171 - 1ull;
                            unsigned long long v173;
                            v173 = v172 % 16ull;
                            unsigned long long v174;
                            v174 = v172 - v173;
                            unsigned long long v175;
                            v175 = v174 + v159;
                            unsigned long long v176;
                            v176 = v175 + 16ull;
                            unsigned long long v177;
                            v177 = v176 - 1ull;
                            unsigned long long v178;
                            v178 = v177 % 16ull;
                            unsigned long long v179;
                            v179 = v177 - v178;
                            unsigned long long v180;
                            v180 = v179 + 2048ull;
                            bool v181;
                            v181 = v180 <= 81920ull;
                            bool v182;
                            v182 = v181 == false;
                            if (v182){
                                assert("The dynamic shared memory is insufficient to allocate the tensor." && v181);
                            } else {
                            }
                            extern __shared__ unsigned char v184[];
                            bool v185;
                            v185 = v180 <= v180;
                            bool v186;
                            v186 = v185 == false;
                            if (v186){
                                assert("The length of the partition has to be less than or equal to the length of the base array." && v185);
                            } else {
                            }
                            float * v188;
                            v188 = reinterpret_cast<float *>(&v184[0ull]);
                            int * v190;
                            v190 = reinterpret_cast<int *>(&v184[2048ull]);
                            float * v192;
                            v192 = reinterpret_cast<float *>(&v184[4096ull]);
                            float * v194;
                            v194 = reinterpret_cast<float *>(&v184[6144ull]);
                            float * * v196;
                            v196 = reinterpret_cast<float * *>(&v184[8192ull]);
                            float * * v198;
                            v198 = reinterpret_cast<float * *>(&v184[v164]);
                            float * * v200;
                            v200 = reinterpret_cast<float * *>(&v184[v169]);
                            float * * v202;
                            v202 = reinterpret_cast<float * *>(&v184[v174]);
                            float * v204;
                            v204 = reinterpret_cast<float *>(&v184[v179]);
                            int v206;
                            v206 = threadIdx.x;
                            assert("Tensor range check" && 0 <= v206 && v206 < 512l);
                            v188[v206] = v97;
                            v190[v206] = v96;
                            v192[v206] = v100;
                            v194[v206] = v143;
                            v196[v206] = v108;
                            v198[v206] = v151;
                            v200[v206] = v153;
                            v202[v206] = v155;
                            asm("barrier.cta.sync %0;" :: "r"(0l));
                            bool v207;
                            v207 = 0l <= v206;
                            bool v208;
                            v208 = v207 == false;
                            if (v208){
                                assert("The index needs to be zero or positive." && v207);
                            } else {
                            }
                            int v210;
                            v210 = v206 % 1l;
                            bool v211;
                            v211 = v206 < 512l;
                            bool v212;
                            v212 = v211 == false;
                            if (v212){
                                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v211);
                            } else {
                            }
                            assert("Tensor range check" && 0 <= v206 && v206 < 512l);
                            int v214;
                            v214 = 0l;
                            while (while_method_7(v214)){
                                bool v216;
                                v216 = v207 && v211;
                                bool v217;
                                v217 = v216 == false;
                                if (v217){
                                    assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v216);
                                } else {
                                }
                                bool v219;
                                v219 = 0l <= v214;
                                bool v221;
                                if (v219){
                                    bool v220;
                                    v220 = v214 < 1l;
                                    v221 = v220;
                                } else {
                                    v221 = false;
                                }
                                bool v222;
                                v222 = v221 == false;
                                if (v222){
                                    assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v221);
                                } else {
                                }
                                int v224;
                                v224 = v214 * 512l;
                                int v225;
                                v225 = v224 + v206;
                                assert("Tensor range check" && 0 <= v214 && v214 < 1l);
                                int v226;
                                v226 = 512l * v214;
                                int v227;
                                v227 = v226 + v206;
                                float v228;
                                v228 = v188[v227];
                                int v229;
                                v229 = v190[v227];
                                float v230;
                                v230 = v192[v227];
                                float v231;
                                v231 = v194[v227];
                                float * v232;
                                v232 = v196[v227];
                                float * v233;
                                v233 = v198[v227];
                                float * v234;
                                v234 = v200[v227];
                                float * v235;
                                v235 = v202[v227];
                                int v236;
                                v236 = blockIdx.x;
                                int v237;
                                v237 = v236 * 512l;
                                int v238;
                                v238 = v237 + v225;
                                assert("Tensor range check" && 0 <= v210 && v210 < 1l);
                                int v239;
                                v239 = 4l * v210;
                                float v240[4l];
                                float v241[4l];
                                float v242[4l];
                                int v243[4l];
                                int v244;
                                v244 = 0l;
                                while (while_method_7(v244)){
                                    assert("Tensor range check" && 0 <= v244 && v244 < 1l);
                                    int v246;
                                    v246 = 4l * v244;
                                    assert("Tensor range check" && 0 <= v244 && v244 < 1l);
                                    int v247;
                                    v247 = v246 + v239;
                                    int4* v248;
                                    v248 = reinterpret_cast<int4*>(v233 + v247);
                                    int4* v249;
                                    v249 = reinterpret_cast<int4*>(v240 + v246);
                                    assert("Pointer alignment check" && (unsigned long long)(v248) % 4l == 0 && (unsigned long long)(v249) % 4l == 0);
                                    *v249 = *v248;
                                    int4* v250;
                                    v250 = reinterpret_cast<int4*>(v234 + v247);
                                    int4* v251;
                                    v251 = reinterpret_cast<int4*>(v241 + v246);
                                    assert("Pointer alignment check" && (unsigned long long)(v250) % 4l == 0 && (unsigned long long)(v251) % 4l == 0);
                                    *v251 = *v250;
                                    int4* v252;
                                    v252 = reinterpret_cast<int4*>(v235 + v247);
                                    int4* v253;
                                    v253 = reinterpret_cast<int4*>(v242 + v246);
                                    assert("Pointer alignment check" && (unsigned long long)(v252) % 4l == 0 && (unsigned long long)(v253) % 4l == 0);
                                    *v253 = *v252;
                                    v244 += 1l ;
                                }
                                int v254;
                                v254 = 0l;
                                while (while_method_7(v254)){
                                    int v256;
                                    v256 = 0l;
                                    while (while_method_6(v256)){
                                        bool v258;
                                        v258 = 0l <= v256;
                                        bool v260;
                                        if (v258){
                                            bool v259;
                                            v259 = v256 < 4l;
                                            v260 = v259;
                                        } else {
                                            v260 = false;
                                        }
                                        bool v261;
                                        v261 = v260 == false;
                                        if (v261){
                                            assert("The indices should be inside the range of the dimension." && v260);
                                        } else {
                                        }
                                        bool v263;
                                        v263 = 0l <= v210;
                                        bool v265;
                                        if (v263){
                                            bool v264;
                                            v264 = v210 < 1l;
                                            v265 = v264;
                                        } else {
                                            v265 = false;
                                        }
                                        bool v266;
                                        v266 = v265 == false;
                                        if (v266){
                                            assert("The indices should be inside the range of the dimension." && v265);
                                        } else {
                                        }
                                        int v268;
                                        v268 = v210 * 4l;
                                        int v269;
                                        v269 = v256 + v268;
                                        bool v270;
                                        v270 = 0l <= v254;
                                        bool v272;
                                        if (v270){
                                            bool v271;
                                            v271 = v254 < 1l;
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
                                        v275 = v254 * 4l;
                                        int v276;
                                        v276 = v269 + v275;
                                        assert("Tensor range check" && 0 <= v254 && v254 < 1l);
                                        assert("Tensor range check" && 0 <= v256 && v256 < 4l);
                                        int v277;
                                        v277 = 4l * v254;
                                        int v278;
                                        v278 = v277 + v256;
                                        v243[v278] = v276;
                                        v256 += 1l ;
                                    }
                                    v254 += 1l ;
                                }
                                float v279[4l];
                                int v280;
                                v280 = 0l;
                                while (while_method_7(v280)){
                                    int v282;
                                    v282 = 0l;
                                    while (while_method_6(v282)){
                                        assert("Tensor range check" && 0 <= v280 && v280 < 1l);
                                        assert("Tensor range check" && 0 <= v282 && v282 < 4l);
                                        int v284;
                                        v284 = 4l * v280;
                                        int v285;
                                        v285 = v284 + v282;
                                        float v286;
                                        v286 = v241[v285];
                                        float v287;
                                        v287 = v242[v285];
                                        bool v288;
                                        v288 = v287 == 0.0f;
                                        bool v289;
                                        v289 = v288 != true;
                                        float v291;
                                        if (v289){
                                            float v290;
                                            v290 = v286 / v287;
                                            v291 = v290;
                                        } else {
                                            v291 = 0.0f;
                                        }
                                        assert("Tensor range check" && 0 <= v280 && v280 < 1l);
                                        assert("Tensor range check" && 0 <= v282 && v282 < 4l);
                                        v279[v285] = v291;
                                        v282 += 1l ;
                                    }
                                    v280 += 1l ;
                                }
                                bool v292[4l];
                                int v293;
                                v293 = 0l;
                                while (while_method_7(v293)){
                                    int v295;
                                    v295 = 0l;
                                    while (while_method_6(v295)){
                                        assert("Tensor range check" && 0 <= v293 && v293 < 1l);
                                        assert("Tensor range check" && 0 <= v295 && v295 < 4l);
                                        int v297;
                                        v297 = 4l * v293;
                                        int v298;
                                        v298 = v297 + v295;
                                        float v299;
                                        v299 = v240[v298];
                                        int v300;
                                        v300 = v243[v298];
                                        bool v301;
                                        v301 = v300 < 3l;
                                        assert("Tensor range check" && 0 <= v293 && v293 < 1l);
                                        assert("Tensor range check" && 0 <= v295 && v295 < 4l);
                                        v292[v298] = v301;
                                        v295 += 1l ;
                                    }
                                    v293 += 1l ;
                                }
                                float v302[4l];
                                int v303;
                                v303 = 0l;
                                while (while_method_7(v303)){
                                    int v305;
                                    v305 = 0l;
                                    while (while_method_6(v305)){
                                        assert("Tensor range check" && 0 <= v303 && v303 < 1l);
                                        assert("Tensor range check" && 0 <= v305 && v305 < 4l);
                                        int v307;
                                        v307 = 4l * v303;
                                        int v308;
                                        v308 = v307 + v305;
                                        float v309;
                                        v309 = v240[v308];
                                        bool v310;
                                        v310 = v292[v308];
                                        float v313;
                                        if (v310){
                                            bool v311;
                                            v311 = 0.0f >= v309;
                                            if (v311){
                                                v313 = 0.0f;
                                            } else {
                                                v313 = v309;
                                            }
                                        } else {
                                            v313 = 0.0f;
                                        }
                                        assert("Tensor range check" && 0 <= v303 && v303 < 1l);
                                        assert("Tensor range check" && 0 <= v305 && v305 < 4l);
                                        v302[v308] = v313;
                                        v305 += 1l ;
                                    }
                                    v303 += 1l ;
                                }
                                float v314;
                                v314 = 0.0f;
                                int v315;
                                v315 = 0l;
                                while (while_method_7(v315)){
                                    int v317;
                                    v317 = 0l;
                                    while (while_method_6(v317)){
                                        assert("Tensor range check" && 0 <= v315 && v315 < 1l);
                                        assert("Tensor range check" && 0 <= v317 && v317 < 4l);
                                        int v319;
                                        v319 = 4l * v315;
                                        int v320;
                                        v320 = v319 + v317;
                                        float v321;
                                        v321 = v302[v320];
                                        float v322;
                                        v322 = v314 + v321;
                                        v314 = v322;
                                        v317 += 1l ;
                                    }
                                    v315 += 1l ;
                                }
                                auto v323 = cooperative_groups::coalesced_threads();
                                int v324;
                                v324 = threadIdx.x;
                                auto v325 = cooperative_groups::labeled_partition(v323,v324);
                                Closure1 v326{};
                                float v327;
                                v327 = cooperative_groups::reduce(v325, v314, v326);
                                int v328[4l];
                                int v329;
                                v329 = 0l;
                                while (while_method_7(v329)){
                                    int v331;
                                    v331 = 0l;
                                    while (while_method_6(v331)){
                                        assert("Tensor range check" && 0 <= v329 && v329 < 1l);
                                        assert("Tensor range check" && 0 <= v331 && v331 < 4l);
                                        int v333;
                                        v333 = 4l * v329;
                                        int v334;
                                        v334 = v333 + v331;
                                        bool v335;
                                        v335 = v292[v334];
                                        int v336;
                                        if (v335){
                                            v336 = 1l;
                                        } else {
                                            v336 = 0l;
                                        }
                                        assert("Tensor range check" && 0 <= v329 && v329 < 1l);
                                        assert("Tensor range check" && 0 <= v331 && v331 < 4l);
                                        v328[v334] = v336;
                                        v331 += 1l ;
                                    }
                                    v329 += 1l ;
                                }
                                int v337;
                                v337 = 0l;
                                int v338;
                                v338 = 0l;
                                while (while_method_7(v338)){
                                    int v340;
                                    v340 = 0l;
                                    while (while_method_6(v340)){
                                        assert("Tensor range check" && 0 <= v338 && v338 < 1l);
                                        assert("Tensor range check" && 0 <= v340 && v340 < 4l);
                                        int v342;
                                        v342 = 4l * v338;
                                        int v343;
                                        v343 = v342 + v340;
                                        int v344;
                                        v344 = v328[v343];
                                        int v345;
                                        v345 = v337 + v344;
                                        v337 = v345;
                                        v340 += 1l ;
                                    }
                                    v338 += 1l ;
                                }
                                auto v346 = cooperative_groups::coalesced_threads();
                                int v347;
                                v347 = threadIdx.x;
                                auto v348 = cooperative_groups::labeled_partition(v346,v347);
                                Closure2 v349{};
                                int v350;
                                v350 = cooperative_groups::reduce(v348, v337, v349);
                                float v351;
                                v351 = (float)v350;
                                float v352;
                                v352 = 1.0f / v351;
                                float v353[4l];
                                int v354;
                                v354 = 0l;
                                while (while_method_7(v354)){
                                    int v356;
                                    v356 = 0l;
                                    while (while_method_6(v356)){
                                        assert("Tensor range check" && 0 <= v354 && v354 < 1l);
                                        assert("Tensor range check" && 0 <= v356 && v356 < 4l);
                                        int v358;
                                        v358 = 4l * v354;
                                        int v359;
                                        v359 = v358 + v356;
                                        float v360;
                                        v360 = v302[v359];
                                        bool v361;
                                        v361 = v292[v359];
                                        bool v362;
                                        v362 = v361 == false;
                                        float v367;
                                        if (v362){
                                            v367 = 0.0f;
                                        } else {
                                            bool v363;
                                            v363 = v327 == 0.0f;
                                            bool v364;
                                            v364 = v363 != true;
                                            if (v364){
                                                float v365;
                                                v365 = v360 / v327;
                                                v367 = v365;
                                            } else {
                                                v367 = v352;
                                            }
                                        }
                                        assert("Tensor range check" && 0 <= v354 && v354 < 1l);
                                        assert("Tensor range check" && 0 <= v356 && v356 < 4l);
                                        v353[v359] = v367;
                                        v356 += 1l ;
                                    }
                                    v354 += 1l ;
                                }
                                float v368[4l];
                                int v369;
                                v369 = 0l;
                                while (while_method_7(v369)){
                                    int v371;
                                    v371 = 0l;
                                    while (while_method_6(v371)){
                                        assert("Tensor range check" && 0 <= v369 && v369 < 1l);
                                        assert("Tensor range check" && 0 <= v371 && v371 < 4l);
                                        int v373;
                                        v373 = 4l * v369;
                                        int v374;
                                        v374 = v373 + v371;
                                        float v375;
                                        v375 = v279[v374];
                                        int v376;
                                        v376 = v243[v374];
                                        bool v377;
                                        v377 = v229 == v376;
                                        float v380;
                                        if (v377){
                                            float v378;
                                            v378 = v230 - v375;
                                            float v379;
                                            v379 = v378 / v228;
                                            v380 = v379;
                                        } else {
                                            v380 = 0.0f;
                                        }
                                        float v381;
                                        v381 = v380 + v375;
                                        assert("Tensor range check" && 0 <= v369 && v369 < 1l);
                                        assert("Tensor range check" && 0 <= v371 && v371 < 4l);
                                        v368[v374] = v381;
                                        v371 += 1l ;
                                    }
                                    v369 += 1l ;
                                }
                                float v382[4l];
                                int v383;
                                v383 = 0l;
                                while (while_method_7(v383)){
                                    int v385;
                                    v385 = 0l;
                                    while (while_method_6(v385)){
                                        assert("Tensor range check" && 0 <= v383 && v383 < 1l);
                                        assert("Tensor range check" && 0 <= v385 && v385 < 4l);
                                        int v387;
                                        v387 = 4l * v383;
                                        int v388;
                                        v388 = v387 + v385;
                                        float v389;
                                        v389 = v353[v388];
                                        float v390;
                                        v390 = v368[v388];
                                        float v391;
                                        v391 = v389 * v390;
                                        assert("Tensor range check" && 0 <= v383 && v383 < 1l);
                                        assert("Tensor range check" && 0 <= v385 && v385 < 4l);
                                        v382[v388] = v391;
                                        v385 += 1l ;
                                    }
                                    v383 += 1l ;
                                }
                                float v392;
                                v392 = 0.0f;
                                int v393;
                                v393 = 0l;
                                while (while_method_7(v393)){
                                    int v395;
                                    v395 = 0l;
                                    while (while_method_6(v395)){
                                        assert("Tensor range check" && 0 <= v393 && v393 < 1l);
                                        assert("Tensor range check" && 0 <= v395 && v395 < 4l);
                                        int v397;
                                        v397 = 4l * v393;
                                        int v398;
                                        v398 = v397 + v395;
                                        float v399;
                                        v399 = v382[v398];
                                        float v400;
                                        v400 = v392 + v399;
                                        v392 = v400;
                                        v395 += 1l ;
                                    }
                                    v393 += 1l ;
                                }
                                auto v401 = cooperative_groups::coalesced_threads();
                                int v402;
                                v402 = threadIdx.x;
                                auto v403 = cooperative_groups::labeled_partition(v401,v402);
                                float v404;
                                v404 = cooperative_groups::reduce(v403, v392, v326);
                                int v405;
                                v405 = 0l;
                                while (while_method_7(v405)){
                                    int v407;
                                    v407 = 0l;
                                    while (while_method_6(v407)){
                                        assert("Tensor range check" && 0 <= v405 && v405 < 1l);
                                        assert("Tensor range check" && 0 <= v407 && v407 < 4l);
                                        int v409;
                                        v409 = 4l * v405;
                                        int v410;
                                        v410 = v409 + v407;
                                        float v411;
                                        v411 = v368[v410];
                                        int v412;
                                        v412 = v243[v410];
                                        float v413;
                                        v413 = v411 - v404;
                                        float v414;
                                        v414 = v231 * v413;
                                        assert("Tensor range check" && 0 <= v412 && v412 < 4l);
                                        float * v415;
                                        v415 = v232+v412;
                                        float v417;
                                        v417 = atomicAdd(v415,v414);
                                        v407 += 1l ;
                                    }
                                    v405 += 1l ;
                                }
                                int v418;
                                v418 = 0l;
                                while (while_method_7(v418)){
                                    assert("Tensor range check" && 0 <= v418 && v418 < 1l);
                                    assert("Tensor range check" && 0 <= v418 && v418 < 1l);
                                    v418 += 1l ;
                                }
                                assert("Tensor range check" && 0 <= v225 && v225 < 512l);
                                v204[v225] = v404;
                                v214 += 1l ;
                            }
                            asm("barrier.cta.sync %0;" :: "r"(0l));
                            assert("Tensor range check" && 0 <= v206 && v206 < 512l);
                            float v420;
                            v420 = v204[v206];
                            asm("barrier.cta.sync %0;" :: "r"(0l));
                            assert("Tensor range check" && 0 <= v98 && v98 < 2l);
                            v82[v98] = v420;
                        }
                        int v421;
                        v421 = threadIdx.x;
                        int v422;
                        v422 = blockIdx.x;
                        int v423;
                        v423 = v422 * 512l;
                        int v424;
                        v424 = v421 + v423;
                        assert("Tensor range check" && 0 <= v76 && v76 < 4l);
                        int v425;
                        v425 = 24576l * v76;
                        assert("Tensor range check" && 0 <= v424 && v424 < 12288l);
                        int v426;
                        v426 = 2l * v424;
                        int v427;
                        v427 = v426 + v425;
                        double * v428;
                        v428 = v70+v427;
                        double * v430;
                        v430 = v72+v427;
                        double * v432;
                        v432 = v428+0l;
                        double * v434;
                        v434 = v430+0l;
                        double * v436;
                        v436 = v428+0l;
                        double * v438;
                        v438 = v430+0l;
                        int v440;
                        v440 = sizeof(double *);
                        unsigned long long v441;
                        v441 = (unsigned long long)v440;
                        unsigned long long v442;
                        v442 = 512ull * v441;
                        unsigned long long v443;
                        v443 = v442 + 16ull;
                        unsigned long long v444;
                        v444 = v443 - 1ull;
                        unsigned long long v445;
                        v445 = v444 % 16ull;
                        unsigned long long v446;
                        v446 = v444 - v445;
                        unsigned long long v447;
                        v447 = v446 + v442;
                        unsigned long long v448;
                        v448 = v447 + 16ull;
                        unsigned long long v449;
                        v449 = v448 - 1ull;
                        unsigned long long v450;
                        v450 = v449 % 16ull;
                        unsigned long long v451;
                        v451 = v449 - v450;
                        unsigned long long v452;
                        v452 = v451 + v442;
                        unsigned long long v453;
                        v453 = v452 + 16ull;
                        unsigned long long v454;
                        v454 = v453 - 1ull;
                        unsigned long long v455;
                        v455 = v454 % 16ull;
                        unsigned long long v456;
                        v456 = v454 - v455;
                        unsigned long long v457;
                        v457 = v456 + v442;
                        bool v458;
                        v458 = v457 <= 81920ull;
                        bool v459;
                        v459 = v458 == false;
                        if (v459){
                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v458);
                        } else {
                        }
                        extern __shared__ unsigned char v461[];
                        bool v462;
                        v462 = v457 <= v457;
                        bool v463;
                        v463 = v462 == false;
                        if (v463){
                            assert("The length of the partition has to be less than or equal to the length of the base array." && v462);
                        } else {
                        }
                        double * * v465;
                        v465 = reinterpret_cast<double * *>(&v461[0ull]);
                        double * * v467;
                        v467 = reinterpret_cast<double * *>(&v461[v446]);
                        double * * v469;
                        v469 = reinterpret_cast<double * *>(&v461[v451]);
                        double * * v471;
                        v471 = reinterpret_cast<double * *>(&v461[v456]);
                        int v473;
                        v473 = threadIdx.x;
                        assert("Tensor range check" && 0 <= v473 && v473 < 512l);
                        v465[v473] = v432;
                        v467[v473] = v434;
                        v469[v473] = v436;
                        v471[v473] = v438;
                        asm("barrier.cta.sync %0;" :: "r"(0l));
                        bool v474;
                        v474 = 0l <= v473;
                        bool v475;
                        v475 = v474 == false;
                        if (v475){
                            assert("The index needs to be zero or positive." && v474);
                        } else {
                        }
                        int v477;
                        v477 = v473 % 1l;
                        bool v478;
                        v478 = v473 < 512l;
                        bool v479;
                        v479 = v478 == false;
                        if (v479){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v478);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v473 && v473 < 512l);
                        int v481;
                        v481 = 0l;
                        while (while_method_7(v481)){
                            bool v483;
                            v483 = v474 && v478;
                            bool v484;
                            v484 = v483 == false;
                            if (v484){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v483);
                            } else {
                            }
                            bool v486;
                            v486 = 0l <= v481;
                            bool v488;
                            if (v486){
                                bool v487;
                                v487 = v481 < 1l;
                                v488 = v487;
                            } else {
                                v488 = false;
                            }
                            bool v489;
                            v489 = v488 == false;
                            if (v489){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v488);
                            } else {
                            }
                            int v491;
                            v491 = v481 * 512l;
                            int v492;
                            v492 = v491 + v473;
                            assert("Tensor range check" && 0 <= v481 && v481 < 1l);
                            int v493;
                            v493 = 512l * v481;
                            int v494;
                            v494 = v493 + v473;
                            double * v495;
                            v495 = v465[v494];
                            double * v496;
                            v496 = v467[v494];
                            double * v497;
                            v497 = v469[v494];
                            double * v498;
                            v498 = v471[v494];
                            int v499;
                            v499 = blockIdx.x;
                            int v500;
                            v500 = v499 * 512l;
                            int v501;
                            v501 = v500 + v492;
                            assert("Tensor range check" && 0 <= v477 && v477 < 1l);
                            int v502;
                            v502 = 2l * v477;
                            double v503[2l];
                            double v504[2l];
                            int v505[2l];
                            int v506;
                            v506 = 0l;
                            while (while_method_7(v506)){
                                assert("Tensor range check" && 0 <= v506 && v506 < 1l);
                                int v508;
                                v508 = 2l * v506;
                                assert("Tensor range check" && 0 <= v506 && v506 < 1l);
                                int v509;
                                v509 = v508 + v502;
                                int4* v510;
                                v510 = reinterpret_cast<int4*>(v495 + v509);
                                int4* v511;
                                v511 = reinterpret_cast<int4*>(v503 + v508);
                                assert("Pointer alignment check" && (unsigned long long)(v510) % 2l == 0 && (unsigned long long)(v511) % 2l == 0);
                                *v511 = *v510;
                                int4* v512;
                                v512 = reinterpret_cast<int4*>(v496 + v509);
                                int4* v513;
                                v513 = reinterpret_cast<int4*>(v504 + v508);
                                assert("Pointer alignment check" && (unsigned long long)(v512) % 2l == 0 && (unsigned long long)(v513) % 2l == 0);
                                *v513 = *v512;
                                v506 += 1l ;
                            }
                            int v514;
                            v514 = 0l;
                            while (while_method_7(v514)){
                                int v516;
                                v516 = 0l;
                                while (while_method_3(v516)){
                                    bool v518;
                                    v518 = 0l <= v516;
                                    bool v520;
                                    if (v518){
                                        bool v519;
                                        v519 = v516 < 2l;
                                        v520 = v519;
                                    } else {
                                        v520 = false;
                                    }
                                    bool v521;
                                    v521 = v520 == false;
                                    if (v521){
                                        assert("The indices should be inside the range of the dimension." && v520);
                                    } else {
                                    }
                                    bool v523;
                                    v523 = 0l <= v477;
                                    bool v525;
                                    if (v523){
                                        bool v524;
                                        v524 = v477 < 1l;
                                        v525 = v524;
                                    } else {
                                        v525 = false;
                                    }
                                    bool v526;
                                    v526 = v525 == false;
                                    if (v526){
                                        assert("The indices should be inside the range of the dimension." && v525);
                                    } else {
                                    }
                                    int v528;
                                    v528 = v477 * 2l;
                                    int v529;
                                    v529 = v516 + v528;
                                    bool v530;
                                    v530 = 0l <= v514;
                                    bool v532;
                                    if (v530){
                                        bool v531;
                                        v531 = v514 < 1l;
                                        v532 = v531;
                                    } else {
                                        v532 = false;
                                    }
                                    bool v533;
                                    v533 = v532 == false;
                                    if (v533){
                                        assert("The indices should be inside the range of the dimension." && v532);
                                    } else {
                                    }
                                    int v535;
                                    v535 = v514 * 2l;
                                    int v536;
                                    v536 = v529 + v535;
                                    assert("Tensor range check" && 0 <= v514 && v514 < 1l);
                                    assert("Tensor range check" && 0 <= v516 && v516 < 2l);
                                    int v537;
                                    v537 = 2l * v514;
                                    int v538;
                                    v538 = v537 + v516;
                                    v505[v538] = v536;
                                    v516 += 1l ;
                                }
                                v514 += 1l ;
                            }
                            double v539[2l];
                            double v540[2l];
                            int v541;
                            v541 = 0l;
                            while (while_method_7(v541)){
                                int v543;
                                v543 = 0l;
                                while (while_method_3(v543)){
                                    assert("Tensor range check" && 0 <= v541 && v541 < 1l);
                                    assert("Tensor range check" && 0 <= v543 && v543 < 2l);
                                    int v545;
                                    v545 = 2l * v541;
                                    int v546;
                                    v546 = v545 + v543;
                                    double v547;
                                    v547 = v503[v546];
                                    double v548;
                                    v548 = v504[v546];
                                    assert("Tensor range check" && 0 <= v541 && v541 < 1l);
                                    assert("Tensor range check" && 0 <= v543 && v543 < 2l);
                                    v539[v546] = 0.0;
                                    v540[v546] = 0.0;
                                    v543 += 1l ;
                                }
                                v541 += 1l ;
                            }
                            int v549;
                            v549 = 0l;
                            while (while_method_7(v549)){
                                assert("Tensor range check" && 0 <= v549 && v549 < 1l);
                                int v551;
                                v551 = 2l * v549;
                                int v552;
                                v552 = v551 + v502;
                                assert("Tensor range check" && 0 <= v549 && v549 < 1l);
                                int4* v553;
                                v553 = reinterpret_cast<int4*>(v539 + v551);
                                int4* v554;
                                v554 = reinterpret_cast<int4*>(v497 + v552);
                                assert("Pointer alignment check" && (unsigned long long)(v553) % 2l == 0 && (unsigned long long)(v554) % 2l == 0);
                                *v554 = *v553;
                                int4* v555;
                                v555 = reinterpret_cast<int4*>(v540 + v551);
                                int4* v556;
                                v556 = reinterpret_cast<int4*>(v498 + v552);
                                assert("Pointer alignment check" && (unsigned long long)(v555) % 2l == 0 && (unsigned long long)(v556) % 2l == 0);
                                *v556 = *v555;
                                v549 += 1l ;
                            }
                            assert("Tensor range check" && 0 <= v492 && v492 < 512l);
                            v481 += 1l ;
                        }
                        asm("barrier.cta.sync %0;" :: "r"(0l));
                        assert("Tensor range check" && 0 <= v473 && v473 < 512l);
                        asm("barrier.cta.sync %0;" :: "r"(0l));
                        assert("Tensor range check" && 0 <= v76 && v76 < 4l);
                        assert("Tensor range check" && 0 <= v424 && v424 < 12288l);
                        int v557;
                        v557 = v87 + v424;
                        v74[v557] = 0l;
                        v76 += 1l ;
                    }
                    static_array<Union1,2l> & v558 = v23;
                    unsigned int v559 = 63ul;
                    static_array_list<Union2,32l> v560;
                    v560 = static_array_list<Union2,32l>{};
                    static_array_list<Union2,32l> & v562 = v560;
                    Union5 v563;
                    v563 = Union5{Union5_1{}};
                    static_array<float,2l> v564;
                    v564 = method_15(v1, v3, v559, v562, v558, v21, v563);
                    float v565;
                    v565 = v564[0l];
                    int v567;
                    v567 = 0l;
                    while (while_method_6(v567)){
                        double * v569;
                        v569 = reinterpret_cast<double *>(&v3[2097168ull]);
                        double * v571;
                        v571 = reinterpret_cast<double *>(&v3[2883600ull]);
                        int * v573;
                        v573 = reinterpret_cast<int *>(&v3[3670032ull]);
                        assert("Tensor range check" && 0 <= v567 && v567 < 4l);
                        int v575;
                        v575 = 24576l * v567;
                        int v576;
                        v576 = threadIdx.x;
                        int v577;
                        v577 = blockIdx.x;
                        int v578;
                        v578 = v577 * 512l;
                        int v579;
                        v579 = v576 + v578;
                        assert("Tensor range check" && 0 <= v579 && v579 < 12288l);
                        int v580;
                        v580 = 2l * v579;
                        int v581;
                        v581 = v580 + v575;
                        double v582;
                        v582 = 0.0;
                        int v583;
                        v583 = 0l;
                        while (while_method_3(v583)){
                            assert("Tensor range check" && 0 <= v583 && v583 < 2l);
                            int v585;
                            v585 = v583 + v581;
                            double v586;
                            v586 = v569[v585];
                            double v587;
                            v587 = v582 + v586;
                            v582 = v587;
                            v583 += 1l ;
                        }
                        double v588;
                        v588 = 0.0;
                        int v589;
                        v589 = 0l;
                        while (while_method_3(v589)){
                            assert("Tensor range check" && 0 <= v589 && v589 < 2l);
                            int v591;
                            v591 = v589 + v581;
                            double v592;
                            v592 = v571[v591];
                            double v593;
                            v593 = v588 + v592;
                            v588 = v593;
                            v589 += 1l ;
                        }
                        double v594;
                        v594 = v582 - v588;
                        double v595;
                        v595 = exp(v594);
                        float v596;
                        v596 = (float)v595;
                        int v597;
                        v597 = v29 / 4l;
                        float v598;
                        v598 = v565 * v596;
                        assert("Tensor range check" && 0 <= v567 && v567 < 4l);
                        assert("Tensor range check" && 0 <= v597 && v597 < 256l);
                        int v599;
                        v599 = 256l * v567;
                        int v600;
                        v600 = v599 + v597;
                        float * v601;
                        v601 = v5+v600;
                        float * v603;
                        v603 = v6+v600;
                        float v605;
                        v605 = atomicAdd(v601,v598);
                        float v606;
                        v606 = atomicAdd(v603,v596);
                        v567 += 1l ;
                    }
                    double * v607;
                    v607 = reinterpret_cast<double *>(&v3[2097168ull]);
                    double * v609;
                    v609 = reinterpret_cast<double *>(&v3[2883600ull]);
                    int * v611;
                    v611 = reinterpret_cast<int *>(&v3[3670032ull]);
                    int v613;
                    v613 = 0l;
                    while (while_method_6(v613)){
                        int v615;
                        v615 = threadIdx.x;
                        int v616;
                        v616 = blockIdx.x;
                        int v617;
                        v617 = v616 * 512l;
                        int v618;
                        v618 = v615 + v617;
                        assert("Tensor range check" && 0 <= v613 && v613 < 4l);
                        int v619;
                        v619 = 24576l * v613;
                        assert("Tensor range check" && 0 <= v618 && v618 < 12288l);
                        int v620;
                        v620 = 2l * v618;
                        int v621;
                        v621 = v620 + v619;
                        double * v622;
                        v622 = v607+v621;
                        double * v624;
                        v624 = v609+v621;
                        double * v626;
                        v626 = v622+0l;
                        double * v628;
                        v628 = v624+0l;
                        double * v630;
                        v630 = v622+0l;
                        double * v632;
                        v632 = v624+0l;
                        int v634;
                        v634 = sizeof(double *);
                        unsigned long long v635;
                        v635 = (unsigned long long)v634;
                        unsigned long long v636;
                        v636 = 512ull * v635;
                        unsigned long long v637;
                        v637 = v636 + 16ull;
                        unsigned long long v638;
                        v638 = v637 - 1ull;
                        unsigned long long v639;
                        v639 = v638 % 16ull;
                        unsigned long long v640;
                        v640 = v638 - v639;
                        unsigned long long v641;
                        v641 = v640 + v636;
                        unsigned long long v642;
                        v642 = v641 + 16ull;
                        unsigned long long v643;
                        v643 = v642 - 1ull;
                        unsigned long long v644;
                        v644 = v643 % 16ull;
                        unsigned long long v645;
                        v645 = v643 - v644;
                        unsigned long long v646;
                        v646 = v645 + v636;
                        unsigned long long v647;
                        v647 = v646 + 16ull;
                        unsigned long long v648;
                        v648 = v647 - 1ull;
                        unsigned long long v649;
                        v649 = v648 % 16ull;
                        unsigned long long v650;
                        v650 = v648 - v649;
                        unsigned long long v651;
                        v651 = v650 + v636;
                        bool v652;
                        v652 = v651 <= 81920ull;
                        bool v653;
                        v653 = v652 == false;
                        if (v653){
                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v652);
                        } else {
                        }
                        extern __shared__ unsigned char v655[];
                        bool v656;
                        v656 = v651 <= v651;
                        bool v657;
                        v657 = v656 == false;
                        if (v657){
                            assert("The length of the partition has to be less than or equal to the length of the base array." && v656);
                        } else {
                        }
                        double * * v659;
                        v659 = reinterpret_cast<double * *>(&v655[0ull]);
                        double * * v661;
                        v661 = reinterpret_cast<double * *>(&v655[v640]);
                        double * * v663;
                        v663 = reinterpret_cast<double * *>(&v655[v645]);
                        double * * v665;
                        v665 = reinterpret_cast<double * *>(&v655[v650]);
                        int v667;
                        v667 = threadIdx.x;
                        assert("Tensor range check" && 0 <= v667 && v667 < 512l);
                        v659[v667] = v626;
                        v661[v667] = v628;
                        v663[v667] = v630;
                        v665[v667] = v632;
                        asm("barrier.cta.sync %0;" :: "r"(0l));
                        bool v668;
                        v668 = 0l <= v667;
                        bool v669;
                        v669 = v668 == false;
                        if (v669){
                            assert("The index needs to be zero or positive." && v668);
                        } else {
                        }
                        int v671;
                        v671 = v667 % 1l;
                        bool v672;
                        v672 = v667 < 512l;
                        bool v673;
                        v673 = v672 == false;
                        if (v673){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v672);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v667 && v667 < 512l);
                        int v675;
                        v675 = 0l;
                        while (while_method_7(v675)){
                            bool v677;
                            v677 = v668 && v672;
                            bool v678;
                            v678 = v677 == false;
                            if (v678){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v677);
                            } else {
                            }
                            bool v680;
                            v680 = 0l <= v675;
                            bool v682;
                            if (v680){
                                bool v681;
                                v681 = v675 < 1l;
                                v682 = v681;
                            } else {
                                v682 = false;
                            }
                            bool v683;
                            v683 = v682 == false;
                            if (v683){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v682);
                            } else {
                            }
                            int v685;
                            v685 = v675 * 512l;
                            int v686;
                            v686 = v685 + v667;
                            assert("Tensor range check" && 0 <= v675 && v675 < 1l);
                            int v687;
                            v687 = 512l * v675;
                            int v688;
                            v688 = v687 + v667;
                            double * v689;
                            v689 = v659[v688];
                            double * v690;
                            v690 = v661[v688];
                            double * v691;
                            v691 = v663[v688];
                            double * v692;
                            v692 = v665[v688];
                            int v693;
                            v693 = blockIdx.x;
                            int v694;
                            v694 = v693 * 512l;
                            int v695;
                            v695 = v694 + v686;
                            assert("Tensor range check" && 0 <= v671 && v671 < 1l);
                            int v696;
                            v696 = 2l * v671;
                            double v697[2l];
                            double v698[2l];
                            int v699[2l];
                            int v700;
                            v700 = 0l;
                            while (while_method_7(v700)){
                                assert("Tensor range check" && 0 <= v700 && v700 < 1l);
                                int v702;
                                v702 = 2l * v700;
                                assert("Tensor range check" && 0 <= v700 && v700 < 1l);
                                int v703;
                                v703 = v702 + v696;
                                int4* v704;
                                v704 = reinterpret_cast<int4*>(v689 + v703);
                                int4* v705;
                                v705 = reinterpret_cast<int4*>(v697 + v702);
                                assert("Pointer alignment check" && (unsigned long long)(v704) % 2l == 0 && (unsigned long long)(v705) % 2l == 0);
                                *v705 = *v704;
                                int4* v706;
                                v706 = reinterpret_cast<int4*>(v690 + v703);
                                int4* v707;
                                v707 = reinterpret_cast<int4*>(v698 + v702);
                                assert("Pointer alignment check" && (unsigned long long)(v706) % 2l == 0 && (unsigned long long)(v707) % 2l == 0);
                                *v707 = *v706;
                                v700 += 1l ;
                            }
                            int v708;
                            v708 = 0l;
                            while (while_method_7(v708)){
                                int v710;
                                v710 = 0l;
                                while (while_method_3(v710)){
                                    bool v712;
                                    v712 = 0l <= v710;
                                    bool v714;
                                    if (v712){
                                        bool v713;
                                        v713 = v710 < 2l;
                                        v714 = v713;
                                    } else {
                                        v714 = false;
                                    }
                                    bool v715;
                                    v715 = v714 == false;
                                    if (v715){
                                        assert("The indices should be inside the range of the dimension." && v714);
                                    } else {
                                    }
                                    bool v717;
                                    v717 = 0l <= v671;
                                    bool v719;
                                    if (v717){
                                        bool v718;
                                        v718 = v671 < 1l;
                                        v719 = v718;
                                    } else {
                                        v719 = false;
                                    }
                                    bool v720;
                                    v720 = v719 == false;
                                    if (v720){
                                        assert("The indices should be inside the range of the dimension." && v719);
                                    } else {
                                    }
                                    int v722;
                                    v722 = v671 * 2l;
                                    int v723;
                                    v723 = v710 + v722;
                                    bool v724;
                                    v724 = 0l <= v708;
                                    bool v726;
                                    if (v724){
                                        bool v725;
                                        v725 = v708 < 1l;
                                        v726 = v725;
                                    } else {
                                        v726 = false;
                                    }
                                    bool v727;
                                    v727 = v726 == false;
                                    if (v727){
                                        assert("The indices should be inside the range of the dimension." && v726);
                                    } else {
                                    }
                                    int v729;
                                    v729 = v708 * 2l;
                                    int v730;
                                    v730 = v723 + v729;
                                    assert("Tensor range check" && 0 <= v708 && v708 < 1l);
                                    assert("Tensor range check" && 0 <= v710 && v710 < 2l);
                                    int v731;
                                    v731 = 2l * v708;
                                    int v732;
                                    v732 = v731 + v710;
                                    v699[v732] = v730;
                                    v710 += 1l ;
                                }
                                v708 += 1l ;
                            }
                            double v733[2l];
                            double v734[2l];
                            int v735;
                            v735 = 0l;
                            while (while_method_7(v735)){
                                int v737;
                                v737 = 0l;
                                while (while_method_3(v737)){
                                    assert("Tensor range check" && 0 <= v735 && v735 < 1l);
                                    assert("Tensor range check" && 0 <= v737 && v737 < 2l);
                                    int v739;
                                    v739 = 2l * v735;
                                    int v740;
                                    v740 = v739 + v737;
                                    double v741;
                                    v741 = v697[v740];
                                    double v742;
                                    v742 = v698[v740];
                                    assert("Tensor range check" && 0 <= v735 && v735 < 1l);
                                    assert("Tensor range check" && 0 <= v737 && v737 < 2l);
                                    v733[v740] = 0.0;
                                    v734[v740] = 0.0;
                                    v737 += 1l ;
                                }
                                v735 += 1l ;
                            }
                            int v743;
                            v743 = 0l;
                            while (while_method_7(v743)){
                                assert("Tensor range check" && 0 <= v743 && v743 < 1l);
                                int v745;
                                v745 = 2l * v743;
                                int v746;
                                v746 = v745 + v696;
                                assert("Tensor range check" && 0 <= v743 && v743 < 1l);
                                int4* v747;
                                v747 = reinterpret_cast<int4*>(v733 + v745);
                                int4* v748;
                                v748 = reinterpret_cast<int4*>(v691 + v746);
                                assert("Pointer alignment check" && (unsigned long long)(v747) % 2l == 0 && (unsigned long long)(v748) % 2l == 0);
                                *v748 = *v747;
                                int4* v749;
                                v749 = reinterpret_cast<int4*>(v734 + v745);
                                int4* v750;
                                v750 = reinterpret_cast<int4*>(v692 + v746);
                                assert("Pointer alignment check" && (unsigned long long)(v749) % 2l == 0 && (unsigned long long)(v750) % 2l == 0);
                                *v750 = *v749;
                                v743 += 1l ;
                            }
                            assert("Tensor range check" && 0 <= v686 && v686 < 512l);
                            v675 += 1l ;
                        }
                        asm("barrier.cta.sync %0;" :: "r"(0l));
                        assert("Tensor range check" && 0 <= v667 && v667 < 512l);
                        asm("barrier.cta.sync %0;" :: "r"(0l));
                        assert("Tensor range check" && 0 <= v613 && v613 < 4l);
                        assert("Tensor range check" && 0 <= v618 && v618 < 12288l);
                        int v751;
                        v751 = 12288l * v613;
                        int v752;
                        v752 = v751 + v618;
                        v611[v752] = 0l;
                        v613 += 1l ;
                    }
                    v31 += 1l ;
                }
                unsigned int * v753;
                v753 = reinterpret_cast<unsigned int *>(&v1[12582912ull]);
                int * v755;
                v755 = reinterpret_cast<int *>(&v3[262144ull]);
                float * v757;
                v757 = reinterpret_cast<float *>(&v3[262160ull]);
                float * v759;
                v759 = reinterpret_cast<float *>(&v3[524304ull]);
                float * v761;
                v761 = reinterpret_cast<float *>(&v3[786448ull]);
                float * v763;
                v763 = reinterpret_cast<float *>(&v3[1048592ull]);
                float * v765;
                v765 = reinterpret_cast<float *>(&v3[1310736ull]);
                float * v767;
                v767 = reinterpret_cast<float *>(&v3[1572880ull]);
                float * v769;
                v769 = reinterpret_cast<float *>(&v3[1835024ull]);
                int * v771;
                v771 = reinterpret_cast<int *>(&v1[12779520ull]);
                float * v773;
                v773 = reinterpret_cast<float *>(&v1[15925248ull]);
                int * v775;
                v775 = reinterpret_cast<int *>(&v1[19070976ull]);
                int * v777;
                v777 = reinterpret_cast<int *>(&v1[22216704ull]);
                double * v779;
                v779 = reinterpret_cast<double *>(&v1[25362432ull]);
                double * v781;
                v781 = reinterpret_cast<double *>(&v1[37945344ull]);
                double * v783;
                v783 = reinterpret_cast<double *>(&v3[2097168ull]);
                double * v785;
                v785 = reinterpret_cast<double *>(&v3[2883600ull]);
                int * v787;
                v787 = reinterpret_cast<int *>(&v3[3670032ull]);
                v14.sync() ;
                int v789;
                v789 = threadIdx.x;
                int v790;
                v790 = blockIdx.x;
                int v791;
                v791 = v790 * 512l;
                int v792;
                v792 = v789 + v791;
                bool v793;
                v793 = v792 == 0l;
                if (v793){
                    int v794;
                    v794 = 0l;
                    int v795;
                    v795 = 4l;
                    int v796;
                    v796 = int_range_10(v795, v794, v21);
                    v755[0l] = v796;
                } else {
                }
                __syncwarp();
                int v797;
                v797 = threadIdx.x;
                bool v798;
                v798 = 0l <= v797;
                bool v799;
                v799 = v798 == false;
                if (v799){
                    assert("The index needs to be zero or positive." && v798);
                } else {
                }
                int v801;
                v801 = v797 % 1l;
                int v802;
                v802 = v797 % 512l;
                int v803;
                v803 = v797 / 512l;
                bool v804;
                v804 = v803 < 1l;
                bool v805;
                v805 = v804 == false;
                if (v805){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v804);
                } else {
                }
                assert("Tensor range check" && 0 <= v803 && v803 < 1l);
                assert("Tensor range check" && 0 <= v802 && v802 < 512l);
                assert("Tensor range check" && 0 <= v801 && v801 < 1l);
                int v807;
                v807 = 4l * v801;
                int v808;
                v808 = 4l * v802;
                int v809;
                v809 = v808 + v807;
                int v810;
                v810 = 16384l * v803;
                int v811;
                v811 = v810 + v809;
                assert("Tensor range check" && 0 <= v803 && v803 < 1l);
                assert("Tensor range check" && 0 <= v802 && v802 < 512l);
                assert("Tensor range check" && 0 <= v801 && v801 < 1l);
                int v812;
                v812 = blockIdx.x;
                int v813;
                v813 = v812;
                while (while_method_9(v813)){
                    bool v815;
                    v815 = 0l <= v813;
                    bool v816;
                    v816 = v815 == false;
                    if (v816){
                        assert("The index needs to be zero or positive." && v815);
                    } else {
                    }
                    int v818;
                    v818 = v813 % 8l;
                    int v819;
                    v819 = v813 / 8l;
                    bool v820;
                    v820 = v819 < 4l;
                    bool v821;
                    v821 = v820 == false;
                    if (v821){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v820);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v819 && v819 < 4l);
                    assert("Tensor range check" && 0 <= v818 && v818 < 8l);
                    int v823;
                    v823 = 2048l * v818;
                    int v824;
                    v824 = v823 + v811;
                    int v825;
                    v825 = 16384l * v819;
                    int v826;
                    v826 = v825 + v824;
                    float v827[4l];
                    float v828[4l];
                    float v829[4l];
                    float v830[4l];
                    float v831[4l];
                    float v832[4l];
                    float v833[4l];
                    int v834[4l];
                    int v835;
                    v835 = 0l;
                    while (while_method_7(v835)){
                        assert("Tensor range check" && 0 <= v835 && v835 < 1l);
                        int v837;
                        v837 = 4l * v835;
                        assert("Tensor range check" && 0 <= v835 && v835 < 1l);
                        int v838;
                        v838 = v837 + v826;
                        int4* v839;
                        v839 = reinterpret_cast<int4*>(v757 + v838);
                        int4* v840;
                        v840 = reinterpret_cast<int4*>(v827 + v837);
                        assert("Pointer alignment check" && (unsigned long long)(v839) % 4l == 0 && (unsigned long long)(v840) % 4l == 0);
                        *v840 = *v839;
                        int4* v841;
                        v841 = reinterpret_cast<int4*>(v759 + v838);
                        int4* v842;
                        v842 = reinterpret_cast<int4*>(v828 + v837);
                        assert("Pointer alignment check" && (unsigned long long)(v841) % 4l == 0 && (unsigned long long)(v842) % 4l == 0);
                        *v842 = *v841;
                        int4* v843;
                        v843 = reinterpret_cast<int4*>(v761 + v838);
                        int4* v844;
                        v844 = reinterpret_cast<int4*>(v829 + v837);
                        assert("Pointer alignment check" && (unsigned long long)(v843) % 4l == 0 && (unsigned long long)(v844) % 4l == 0);
                        *v844 = *v843;
                        int4* v845;
                        v845 = reinterpret_cast<int4*>(v763 + v838);
                        int4* v846;
                        v846 = reinterpret_cast<int4*>(v830 + v837);
                        assert("Pointer alignment check" && (unsigned long long)(v845) % 4l == 0 && (unsigned long long)(v846) % 4l == 0);
                        *v846 = *v845;
                        int4* v847;
                        v847 = reinterpret_cast<int4*>(v765 + v838);
                        int4* v848;
                        v848 = reinterpret_cast<int4*>(v831 + v837);
                        assert("Pointer alignment check" && (unsigned long long)(v847) % 4l == 0 && (unsigned long long)(v848) % 4l == 0);
                        *v848 = *v847;
                        int4* v849;
                        v849 = reinterpret_cast<int4*>(v767 + v838);
                        int4* v850;
                        v850 = reinterpret_cast<int4*>(v832 + v837);
                        assert("Pointer alignment check" && (unsigned long long)(v849) % 4l == 0 && (unsigned long long)(v850) % 4l == 0);
                        *v850 = *v849;
                        int4* v851;
                        v851 = reinterpret_cast<int4*>(v769 + v838);
                        int4* v852;
                        v852 = reinterpret_cast<int4*>(v833 + v837);
                        assert("Pointer alignment check" && (unsigned long long)(v851) % 4l == 0 && (unsigned long long)(v852) % 4l == 0);
                        *v852 = *v851;
                        v835 += 1l ;
                    }
                    int v853;
                    v853 = 0l;
                    while (while_method_7(v853)){
                        int v855;
                        v855 = 0l;
                        while (while_method_6(v855)){
                            bool v857;
                            v857 = 0l <= v855;
                            bool v859;
                            if (v857){
                                bool v858;
                                v858 = v855 < 4l;
                                v859 = v858;
                            } else {
                                v859 = false;
                            }
                            bool v860;
                            v860 = v859 == false;
                            if (v860){
                                assert("The indices should be inside the range of the dimension." && v859);
                            } else {
                            }
                            bool v862;
                            v862 = 0l <= v801;
                            bool v864;
                            if (v862){
                                bool v863;
                                v863 = v801 < 1l;
                                v864 = v863;
                            } else {
                                v864 = false;
                            }
                            bool v865;
                            v865 = v864 == false;
                            if (v865){
                                assert("The indices should be inside the range of the dimension." && v864);
                            } else {
                            }
                            int v867;
                            v867 = v801 * 4l;
                            int v868;
                            v868 = v855 + v867;
                            bool v869;
                            v869 = 0l <= v853;
                            bool v871;
                            if (v869){
                                bool v870;
                                v870 = v853 < 1l;
                                v871 = v870;
                            } else {
                                v871 = false;
                            }
                            bool v872;
                            v872 = v871 == false;
                            if (v872){
                                assert("The indices should be inside the range of the dimension." && v871);
                            } else {
                            }
                            int v874;
                            v874 = v853 * 4l;
                            int v875;
                            v875 = v868 + v874;
                            assert("Tensor range check" && 0 <= v853 && v853 < 1l);
                            assert("Tensor range check" && 0 <= v855 && v855 < 4l);
                            int v876;
                            v876 = 4l * v853;
                            int v877;
                            v877 = v876 + v855;
                            v834[v877] = v875;
                            v855 += 1l ;
                        }
                        v853 += 1l ;
                    }
                    bool v878;
                    v878 = 0l <= v803;
                    bool v879;
                    v879 = v878 && v804;
                    bool v880;
                    v880 = v879 == false;
                    if (v880){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v879);
                    } else {
                    }
                    bool v882;
                    v882 = 0l <= v802;
                    bool v884;
                    if (v882){
                        bool v883;
                        v883 = v802 < 512l;
                        v884 = v883;
                    } else {
                        v884 = false;
                    }
                    bool v885;
                    v885 = v884 == false;
                    if (v885){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v884);
                    } else {
                    }
                    bool v887;
                    v887 = 0l <= v819;
                    bool v888;
                    v888 = v887 && v820;
                    bool v889;
                    v889 = v888 == false;
                    if (v889){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v888);
                    } else {
                    }
                    bool v891;
                    v891 = 0l <= v818;
                    bool v893;
                    if (v891){
                        bool v892;
                        v892 = v818 < 8l;
                        v893 = v892;
                    } else {
                        v893 = false;
                    }
                    bool v894;
                    v894 = v893 == false;
                    if (v894){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v893);
                    } else {
                    }
                    int v896;
                    v896 = v818 * 512l;
                    int v897;
                    v897 = v819 + v803;
                    int v898;
                    v898 = v896 + v802;
                    bool v899[4l];
                    int v900;
                    v900 = 0l;
                    while (while_method_7(v900)){
                        int v902;
                        v902 = 0l;
                        while (while_method_6(v902)){
                            assert("Tensor range check" && 0 <= v900 && v900 < 1l);
                            assert("Tensor range check" && 0 <= v902 && v902 < 4l);
                            int v904;
                            v904 = 4l * v900;
                            int v905;
                            v905 = v904 + v902;
                            float v906;
                            v906 = v829[v905];
                            bool v907;
                            v907 = v906 == 0.0f;
                            bool v908;
                            v908 = v907 != true;
                            assert("Tensor range check" && 0 <= v900 && v900 < 1l);
                            assert("Tensor range check" && 0 <= v902 && v902 < 4l);
                            v899[v905] = v908;
                            v902 += 1l ;
                        }
                        v900 += 1l ;
                    }
                    bool v909;
                    v909 = false;
                    int v910;
                    v910 = 0l;
                    while (while_method_7(v910)){
                        int v912;
                        v912 = 0l;
                        while (while_method_6(v912)){
                            assert("Tensor range check" && 0 <= v910 && v910 < 1l);
                            assert("Tensor range check" && 0 <= v912 && v912 < 4l);
                            int v914;
                            v914 = 4l * v910;
                            int v915;
                            v915 = v914 + v912;
                            bool v916;
                            v916 = v899[v915];
                            bool v917;
                            v917 = v909 || v916;
                            v909 = v917;
                            v912 += 1l ;
                        }
                        v910 += 1l ;
                    }
                    auto v918 = cooperative_groups::coalesced_threads();
                    int v919;
                    v919 = threadIdx.x;
                    auto v920 = cooperative_groups::labeled_partition(v918,v919);
                    Closure8 v921{};
                    bool v922;
                    v922 = cooperative_groups::reduce(v920, v909, v921);
                    if (v922){
                        float v923[4l];
                        int v924;
                        v924 = 0l;
                        while (while_method_7(v924)){
                            int v926;
                            v926 = 0l;
                            while (while_method_6(v926)){
                                assert("Tensor range check" && 0 <= v924 && v924 < 1l);
                                assert("Tensor range check" && 0 <= v926 && v926 < 4l);
                                int v928;
                                v928 = 4l * v924;
                                int v929;
                                v929 = v928 + v926;
                                float v930;
                                v930 = v828[v929];
                                float v931;
                                v931 = v829[v929];
                                float v932;
                                v932 = v930 + v931;
                                bool v933;
                                v933 = 0.0f >= v932;
                                float v934;
                                if (v933){
                                    v934 = 0.0f;
                                } else {
                                    v934 = v932;
                                }
                                assert("Tensor range check" && 0 <= v924 && v924 < 1l);
                                assert("Tensor range check" && 0 <= v926 && v926 < 4l);
                                v923[v929] = v934;
                                v926 += 1l ;
                            }
                            v924 += 1l ;
                        }
                        float v935[4l];
                        int v936;
                        v936 = 0l;
                        while (while_method_7(v936)){
                            int v938;
                            v938 = 0l;
                            while (while_method_6(v938)){
                                assert("Tensor range check" && 0 <= v936 && v936 < 1l);
                                assert("Tensor range check" && 0 <= v938 && v938 < 4l);
                                int v940;
                                v940 = 4l * v936;
                                int v941;
                                v941 = v940 + v938;
                                float v942;
                                v942 = v923[v941];
                                bool v943;
                                v943 = 0.0f >= v942;
                                float v944;
                                if (v943){
                                    v944 = 0.0f;
                                } else {
                                    v944 = v942;
                                }
                                assert("Tensor range check" && 0 <= v936 && v936 < 1l);
                                assert("Tensor range check" && 0 <= v938 && v938 < 4l);
                                v935[v941] = v944;
                                v938 += 1l ;
                            }
                            v936 += 1l ;
                        }
                        float v945;
                        v945 = 0.0f;
                        int v946;
                        v946 = 0l;
                        while (while_method_7(v946)){
                            int v948;
                            v948 = 0l;
                            while (while_method_6(v948)){
                                assert("Tensor range check" && 0 <= v946 && v946 < 1l);
                                assert("Tensor range check" && 0 <= v948 && v948 < 4l);
                                int v950;
                                v950 = 4l * v946;
                                int v951;
                                v951 = v950 + v948;
                                float v952;
                                v952 = v935[v951];
                                float v953;
                                v953 = v945 + v952;
                                v945 = v953;
                                v948 += 1l ;
                            }
                            v946 += 1l ;
                        }
                        auto v954 = cooperative_groups::coalesced_threads();
                        int v955;
                        v955 = threadIdx.x;
                        auto v956 = cooperative_groups::labeled_partition(v954,v955);
                        Closure1 v957{};
                        float v958;
                        v958 = cooperative_groups::reduce(v956, v945, v957);
                        float v959[4l];
                        int v960;
                        v960 = 0l;
                        while (while_method_7(v960)){
                            int v962;
                            v962 = 0l;
                            while (while_method_6(v962)){
                                assert("Tensor range check" && 0 <= v960 && v960 < 1l);
                                assert("Tensor range check" && 0 <= v962 && v962 < 4l);
                                int v964;
                                v964 = 4l * v960;
                                int v965;
                                v965 = v964 + v962;
                                float v966;
                                v966 = v935[v965];
                                bool v967;
                                v967 = v958 == 0.0f;
                                bool v968;
                                v968 = v967 != true;
                                float v970;
                                if (v968){
                                    float v969;
                                    v969 = v966 / v958;
                                    v970 = v969;
                                } else {
                                    v970 = 0.25f;
                                }
                                assert("Tensor range check" && 0 <= v960 && v960 < 1l);
                                assert("Tensor range check" && 0 <= v962 && v962 < 4l);
                                v959[v965] = v970;
                                v962 += 1l ;
                            }
                            v960 += 1l ;
                        }
                        float v971[4l];
                        int v972;
                        v972 = 0l;
                        while (while_method_7(v972)){
                            int v974;
                            v974 = 0l;
                            while (while_method_6(v974)){
                                assert("Tensor range check" && 0 <= v972 && v972 < 1l);
                                assert("Tensor range check" && 0 <= v974 && v974 < 4l);
                                int v976;
                                v976 = 4l * v972;
                                int v977;
                                v977 = v976 + v974;
                                float v978;
                                v978 = v827[v977];
                                float v979;
                                v979 = v959[v977];
                                float v980;
                                v980 = v978 + v979;
                                assert("Tensor range check" && 0 <= v972 && v972 < 1l);
                                assert("Tensor range check" && 0 <= v974 && v974 < 4l);
                                v971[v977] = v980;
                                v974 += 1l ;
                            }
                            v972 += 1l ;
                        }
                        float v981[4l];
                        int v982;
                        v982 = 0l;
                        while (while_method_7(v982)){
                            int v984;
                            v984 = 0l;
                            while (while_method_6(v984)){
                                assert("Tensor range check" && 0 <= v982 && v982 < 1l);
                                assert("Tensor range check" && 0 <= v984 && v984 < 4l);
                                int v986;
                                v986 = 4l * v982;
                                int v987;
                                v987 = v986 + v984;
                                float v988;
                                v988 = v971[v987];
                                float v989;
                                v989 = -v988;
                                bool v990;
                                v990 = v988 >= v989;
                                float v991;
                                if (v990){
                                    v991 = v988;
                                } else {
                                    v991 = v989;
                                }
                                assert("Tensor range check" && 0 <= v982 && v982 < 1l);
                                assert("Tensor range check" && 0 <= v984 && v984 < 4l);
                                v981[v987] = v991;
                                v984 += 1l ;
                            }
                            v982 += 1l ;
                        }
                        float v992;
                        v992 = 0.0f;
                        int v993;
                        v993 = 0l;
                        while (while_method_7(v993)){
                            int v995;
                            v995 = 0l;
                            while (while_method_6(v995)){
                                assert("Tensor range check" && 0 <= v993 && v993 < 1l);
                                assert("Tensor range check" && 0 <= v995 && v995 < 4l);
                                int v997;
                                v997 = 4l * v993;
                                int v998;
                                v998 = v997 + v995;
                                float v999;
                                v999 = v981[v998];
                                float v1000;
                                v1000 = v992 + v999;
                                v992 = v1000;
                                v995 += 1l ;
                            }
                            v993 += 1l ;
                        }
                        auto v1001 = cooperative_groups::coalesced_threads();
                        int v1002;
                        v1002 = threadIdx.x;
                        auto v1003 = cooperative_groups::labeled_partition(v1001,v1002);
                        float v1004;
                        v1004 = cooperative_groups::reduce(v1003, v992, v957);
                        bool v1005;
                        v1005 = v1004 > 100.0f;
                        float v1007;
                        if (v1005){
                            float v1006;
                            v1006 = 100.0f / v1004;
                            v1007 = v1006;
                        } else {
                            v1007 = 1.0f;
                        }
                        float v1008[4l];
                        int v1009;
                        v1009 = 0l;
                        while (while_method_7(v1009)){
                            int v1011;
                            v1011 = 0l;
                            while (while_method_6(v1011)){
                                assert("Tensor range check" && 0 <= v1009 && v1009 < 1l);
                                assert("Tensor range check" && 0 <= v1011 && v1011 < 4l);
                                int v1013;
                                v1013 = 4l * v1009;
                                int v1014;
                                v1014 = v1013 + v1011;
                                float v1015;
                                v1015 = v981[v1014];
                                float v1016;
                                v1016 = v1007 * v1015;
                                assert("Tensor range check" && 0 <= v1009 && v1009 < 1l);
                                assert("Tensor range check" && 0 <= v1011 && v1011 < 4l);
                                v1008[v1014] = v1016;
                                v1011 += 1l ;
                            }
                            v1009 += 1l ;
                        }
                        float v1017[4l];
                        float v1018[4l];
                        int v1019;
                        v1019 = 0l;
                        while (while_method_7(v1019)){
                            int v1021;
                            v1021 = 0l;
                            while (while_method_6(v1021)){
                                assert("Tensor range check" && 0 <= v1019 && v1019 < 1l);
                                assert("Tensor range check" && 0 <= v1021 && v1021 < 4l);
                                int v1023;
                                v1023 = 4l * v1019;
                                int v1024;
                                v1024 = v1023 + v1021;
                                float v1025;
                                v1025 = v827[v1024];
                                float v1026;
                                v1026 = v828[v1024];
                                float v1027;
                                v1027 = v829[v1024];
                                float v1028;
                                v1028 = v830[v1024];
                                float v1029;
                                v1029 = v831[v1024];
                                float v1030;
                                v1030 = v832[v1024];
                                float v1031;
                                v1031 = v833[v1024];
                                float v1032;
                                v1032 = v1028 + v1030;
                                float v1033;
                                v1033 = v1029 + v1031;
                                assert("Tensor range check" && 0 <= v1019 && v1019 < 1l);
                                assert("Tensor range check" && 0 <= v1021 && v1021 < 4l);
                                v1017[v1024] = v1032;
                                v1018[v1024] = v1033;
                                v1021 += 1l ;
                            }
                            v1019 += 1l ;
                        }
                        int v1034;
                        v1034 = 0l;
                        while (while_method_7(v1034)){
                            int v1036;
                            v1036 = 0l;
                            while (while_method_6(v1036)){
                                assert("Tensor range check" && 0 <= v1034 && v1034 < 1l);
                                assert("Tensor range check" && 0 <= v1036 && v1036 < 4l);
                                int v1038;
                                v1038 = 4l * v1034;
                                int v1039;
                                v1039 = v1038 + v1036;
                                float v1040;
                                v1040 = v1008[v1039];
                                float v1041;
                                v1041 = v923[v1039];
                                float v1042;
                                v1042 = v1017[v1039];
                                float v1043;
                                v1043 = v1018[v1039];
                                assert("Tensor range check" && 0 <= v1034 && v1034 < 1l);
                                assert("Tensor range check" && 0 <= v1036 && v1036 < 4l);
                                v827[v1039] = v1040;
                                v828[v1039] = v1041;
                                v829[v1039] = 0.0f;
                                v830[v1039] = v1042;
                                v831[v1039] = v1043;
                                v832[v1039] = 0.0f;
                                v833[v1039] = 0.0f;
                                v1036 += 1l ;
                            }
                            v1034 += 1l ;
                        }
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v819 && v819 < 4l);
                    assert("Tensor range check" && 0 <= v818 && v818 < 8l);
                    int v1044;
                    v1044 = 0l;
                    while (while_method_7(v1044)){
                        assert("Tensor range check" && 0 <= v1044 && v1044 < 1l);
                        int v1046;
                        v1046 = 4l * v1044;
                        int v1047;
                        v1047 = v1046 + v826;
                        assert("Tensor range check" && 0 <= v1044 && v1044 < 1l);
                        int4* v1048;
                        v1048 = reinterpret_cast<int4*>(v827 + v1046);
                        int4* v1049;
                        v1049 = reinterpret_cast<int4*>(v757 + v1047);
                        assert("Pointer alignment check" && (unsigned long long)(v1048) % 4l == 0 && (unsigned long long)(v1049) % 4l == 0);
                        *v1049 = *v1048;
                        int4* v1050;
                        v1050 = reinterpret_cast<int4*>(v828 + v1046);
                        int4* v1051;
                        v1051 = reinterpret_cast<int4*>(v759 + v1047);
                        assert("Pointer alignment check" && (unsigned long long)(v1050) % 4l == 0 && (unsigned long long)(v1051) % 4l == 0);
                        *v1051 = *v1050;
                        int4* v1052;
                        v1052 = reinterpret_cast<int4*>(v829 + v1046);
                        int4* v1053;
                        v1053 = reinterpret_cast<int4*>(v761 + v1047);
                        assert("Pointer alignment check" && (unsigned long long)(v1052) % 4l == 0 && (unsigned long long)(v1053) % 4l == 0);
                        *v1053 = *v1052;
                        int4* v1054;
                        v1054 = reinterpret_cast<int4*>(v830 + v1046);
                        int4* v1055;
                        v1055 = reinterpret_cast<int4*>(v763 + v1047);
                        assert("Pointer alignment check" && (unsigned long long)(v1054) % 4l == 0 && (unsigned long long)(v1055) % 4l == 0);
                        *v1055 = *v1054;
                        int4* v1056;
                        v1056 = reinterpret_cast<int4*>(v831 + v1046);
                        int4* v1057;
                        v1057 = reinterpret_cast<int4*>(v765 + v1047);
                        assert("Pointer alignment check" && (unsigned long long)(v1056) % 4l == 0 && (unsigned long long)(v1057) % 4l == 0);
                        *v1057 = *v1056;
                        int4* v1058;
                        v1058 = reinterpret_cast<int4*>(v832 + v1046);
                        int4* v1059;
                        v1059 = reinterpret_cast<int4*>(v767 + v1047);
                        assert("Pointer alignment check" && (unsigned long long)(v1058) % 4l == 0 && (unsigned long long)(v1059) % 4l == 0);
                        *v1059 = *v1058;
                        int4* v1060;
                        v1060 = reinterpret_cast<int4*>(v833 + v1046);
                        int4* v1061;
                        v1061 = reinterpret_cast<int4*>(v769 + v1047);
                        assert("Pointer alignment check" && (unsigned long long)(v1060) % 4l == 0 && (unsigned long long)(v1061) % 4l == 0);
                        *v1061 = *v1060;
                        v1044 += 1l ;
                    }
                    v813 += 24l ;
                }
                v14.sync() ;
                v29 += 1l ;
            }
            int v1062;
            v1062 = threadIdx.x;
            int v1063;
            v1063 = blockIdx.x;
            int v1064;
            v1064 = v1063 * 512l;
            int v1065;
            v1065 = v1062 + v1064;
            int v1066;
            v1066 = v1065;
            while (while_method_11(v1066)){
                bool v1068;
                v1068 = 0l <= v1066;
                bool v1069;
                v1069 = v1068 == false;
                if (v1069){
                    assert("The index needs to be zero or positive." && v1068);
                } else {
                }
                int v1071;
                v1071 = v1066 % 64l;
                int v1072;
                v1072 = v1066 / 64l;
                bool v1073;
                v1073 = v1072 < 4l;
                bool v1074;
                v1074 = v1073 == false;
                if (v1074){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1073);
                } else {
                }
                assert("Tensor range check" && 0 <= v1072 && v1072 < 4l);
                assert("Tensor range check" && 0 <= v1071 && v1071 < 64l);
                int v1076;
                v1076 = 4l * v1071;
                int v1077;
                v1077 = 256l * v1072;
                int v1078;
                v1078 = v1077 + v1076;
                assert("Tensor range check" && 0 <= v1072 && v1072 < 4l);
                assert("Tensor range check" && 0 <= v1071 && v1071 < 64l);
                float v1079[4l];
                float v1080[4l];
                float v1081[4l];
                int4* v1082;
                v1082 = reinterpret_cast<int4*>(v5 + v1078);
                int4* v1083;
                v1083 = reinterpret_cast<int4*>(v1079 + 0l);
                assert("Pointer alignment check" && (unsigned long long)(v1082) % 4l == 0 && (unsigned long long)(v1083) % 4l == 0);
                *v1083 = *v1082;
                int4* v1084;
                v1084 = reinterpret_cast<int4*>(v6 + v1078);
                int4* v1085;
                v1085 = reinterpret_cast<int4*>(v1080 + 0l);
                assert("Pointer alignment check" && (unsigned long long)(v1084) % 4l == 0 && (unsigned long long)(v1085) % 4l == 0);
                *v1085 = *v1084;
                // Pushing the loop unrolling to: 0
                int v1086;
                v1086 = 0l;
                #pragma unroll
                while (while_method_6(v1086)){
                    assert("Tensor range check" && 0 <= v1086 && v1086 < 4l);
                    float v1088;
                    v1088 = v1079[v1086];
                    float v1089;
                    v1089 = v1080[v1086];
                    bool v1090;
                    v1090 = v1089 == 0.0f;
                    bool v1091;
                    v1091 = v1090 != true;
                    float v1093;
                    if (v1091){
                        float v1092;
                        v1092 = v1088 / v1089;
                        v1093 = v1092;
                    } else {
                        v1093 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v1086 && v1086 < 4l);
                    v1081[v1086] = v1093;
                    v1086 += 1l ;
                }
                // Poping the loop unrolling to: 0
                int4* v1094;
                v1094 = reinterpret_cast<int4*>(v1081 + 0l);
                int4* v1095;
                v1095 = reinterpret_cast<int4*>(v7 + v1078);
                assert("Pointer alignment check" && (unsigned long long)(v1094) % 4l == 0 && (unsigned long long)(v1095) % 4l == 0);
                *v1095 = *v1094;
                v1066 += 12288l ;
            }
            v14.sync() ;
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
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
options.append('--diag-suppress=550,20012,68,39')
options.append('--restrict')
options.append('--maxrregcount=128')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
import collections
class US0_0(NamedTuple): # StartTraining
    tag = 0
US0 = US0_0
class US1_0(NamedTuple): # Computer
    tag = 0
class US1_1(NamedTuple): # Random
    tag = 1
US1 = Union[US1_0, US1_1]
class US2_0(NamedTuple): # AddRewards
    v0 : list
    tag = 0
US2 = US2_0
def Closure0():
    def inner(v0 : object, v1 : object) -> object:
        v2 = cp.empty(4,dtype=cp.uint8)
        v3 = method0(v0)
        method3(v2, v3)
        del v3
        v4, v5, v6, v7 = method6(v1)
        v10 = "{}\n"
        v11 = "Going to run the Leduc training kernel (vs self.)"
        print(v10.format(v11),end="")
        del v10, v11
        v12 = time.perf_counter()
        v14 = static_array(2)
        v16 = US1_0()
        v14[0] = v16
        del v16
        v18 = US1_0()
        v14[1] = v18
        del v14, v18
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
        v27.max_dynamic_shared_size_bytes = 81920 
        v27((24,),(512,),(v2, v4, v5, v6, v7, v19, v20, v21),shared_mem=81920)
        del v2, v19, v20, v27
        cp.cuda.get_current_stream().synchronize()
        v30 = "{}"
        v31 = "The time it took to run the kernel (in seconds) is: "
        print(v30.format(v31),end="")
        del v30, v31
        v32 = time.perf_counter()
        v33 = v32 - v12
        del v12, v32
        v36 = "{:.6f}\n"
        print(v36.format(v33),end="")
        del v33, v36
        v37 = []
        v39 = v21[0:]
        del v21
        v40 = v39.get()
        del v39
        v41 = 0
        while method16(v41):
            v43 = []
            v44 = 0
            while method17(v44):
                assert 0 <= v41 < 4, 'Tensor range check'
                assert 0 <= v44 < 256, 'Tensor range check'
                v46 = 256 * v41
                v47 = v46 + v44
                del v46
                v48 = v40[v47].item()
                del v47
                v43.append(v48)
                del v48
                v44 += 1 
            del v44
            v37.append(v43)
            del v43
            v41 += 1 
        del v40, v41
        v49 = US2_0(v37)
        del v37
        v50 = [v49]
        del v49
        return method18(v4, v5, v6, v7, v50)
    return inner
def Closure1():
    def inner() -> object:
        v0 = cp.empty(3866640,dtype=cp.uint8)
        v1 = cp.empty(50528256,dtype=cp.uint8)
        v3 = v0[0:0+4*65536].view(cp.float32)
        v4 = cp.random.normal(0.0,0.00390625,65536,dtype=cp.float32) # type: ignore
        cp.copyto(v3[0:0+65536],v4[0:0+65536])
        del v3, v4
        v6 = v0[262144:262144+4*1].view(cp.int32)
        v8 = v0[262160:262160+4*65536].view(cp.float32)
        v10 = v0[524304:524304+4*65536].view(cp.float32)
        v12 = v0[786448:786448+4*65536].view(cp.float32)
        v14 = v0[1048592:1048592+4*65536].view(cp.float32)
        v16 = v0[1310736:1310736+4*65536].view(cp.float32)
        v18 = v0[1572880:1572880+4*65536].view(cp.float32)
        v20 = v0[1835024:1835024+4*65536].view(cp.float32)
        v6[:] = 0
        del v6
        v8[:] = 0
        del v8
        v10[:] = 0
        del v10
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
        v22 = v0[2097168:2097168+8*98304].view(cp.float64)
        v24 = v0[2883600:2883600+8*98304].view(cp.float64)
        v26 = v0[3670032:3670032+4*49152].view(cp.int32)
        v22[:] = 0
        del v22
        v24[:] = 0
        del v24
        v26[:] = 0
        del v26
        v27 = 50528256
        v28 = 3866640
        return method36(v1, v27, v0, v28)
    return inner
def method2(v0 : object) -> None:
    assert v0 == [], f'Expected an unit type. Got: {v0}'
    del v0
    return 
def method1(v0 : object) -> US0:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "StartTraining" == v1
    if v3:
        del v1, v3
        method2(v2)
        del v2
        return US0_0()
    else:
        del v2, v3
        raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
        del v1
        raise Exception("Error")
def method0(v0 : object) -> US0:
    return method1(v0)
def method4(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[0:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method5(v0 : cp.ndarray) -> None:
    del v0
    return 
def method3(v0 : cp.ndarray, v1 : US0) -> None:
    v2 = v1.tag
    method4(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US0_0(): # StartTraining
            del v1
            return method5(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method8(v0 : object) -> None:
    v1 = v0["private"] # type: ignore
    method2(v1)
    del v1
    v2 = v0["public"] # type: ignore
    del v0
    method2(v2)
    del v2
    return 
def method14(v0 : object) -> cp.ndarray:
    assert isinstance(v0,cp.ndarray), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method13(v0 : object) -> cp.ndarray:
    v1 = method14(v0)
    del v0
    return v1
def method15(v0 : object) -> u64:
    assert isinstance(v0,u64), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method12(v0 : object) -> Tuple[cp.ndarray, u64]:
    v1 = v0[0] # type: ignore
    v2 = method13(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method15(v3)
    del v3
    return v2, v4
def method11(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["output"] # type: ignore
    v2, v3 = method12(v1)
    del v1
    v4 = v0["param"] # type: ignore
    del v0
    v5, v6 = method12(v4)
    del v4
    return v2, v3, v5, v6
def method10(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1, v2, v3, v4 = method11(v0)
    del v0
    return v1, v2, v3, v4
def method9(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["model_data"] # type: ignore
    del v0
    v2, v3, v4, v5 = method10(v1)
    del v1
    return v2, v3, v4, v5
def method7(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["game"] # type: ignore
    method8(v1)
    del v1
    v2 = v0["neural"] # type: ignore
    del v0
    v3, v4, v5, v6 = method9(v2)
    del v2
    return v3, v4, v5, v6
def method6(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    return method7(v0)
def method16(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method17(v0 : i32) -> bool:
    v1 = v0 < 256
    del v0
    return v1
def method22() -> object:
    v0 = []
    return v0
def method21() -> object:
    v0 = method22()
    v1 = method22()
    v2 = {'private': v0, 'public': v1}
    del v0, v1
    return v2
def method28(v0 : cp.ndarray) -> object:
    v1 = v0
    del v0
    return v1
def method27(v0 : cp.ndarray) -> object:
    return method28(v0)
def method29(v0 : u64) -> object:
    v1 = v0
    del v0
    return v1
def method26(v0 : cp.ndarray, v1 : u64) -> object:
    v2 = []
    v3 = method27(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method29(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method25(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method26(v0, v1)
    del v0, v1
    v5 = method26(v2, v3)
    del v2, v3
    v6 = {'output': v4, 'param': v5}
    del v4, v5
    return v6
def method24(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    return method25(v0, v1, v2, v3)
def method23(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method24(v0, v1, v2, v3)
    del v0, v1, v2, v3
    v5 = {'model_data': v4}
    del v4
    return v5
def method20(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method21()
    v5 = method23(v0, v1, v2, v3)
    del v0, v1, v2, v3
    v6 = {'game': v4, 'neural': v5}
    del v4, v5
    return v6
def method31(v0 : i32, v1 : i32) -> bool:
    v2 = v1 < v0
    del v0, v1
    return v2
def method35(v0 : f32) -> object:
    v1 = v0
    del v0
    return v1
def method34(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method31(v2, v3):
        v5 = v0[v3]
        v6 = method35(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method33(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method31(v2, v3):
        v5 = v0[v3]
        v6 = method34(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method32(v0 : US2) -> object:
    match v0:
        case US2_0(v1): # AddRewards
            del v0
            v2 = method33(v1)
            del v1
            v3 = "AddRewards"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method30(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method31(v2, v3):
        v5 = v0[v3]
        v6 = method32(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method19(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64, v4 : list) -> object:
    v5 = []
    v6 = method20(v0, v1, v2, v3)
    del v0, v1, v2, v3
    v5.append(v6)
    del v6
    v7 = method30(v4)
    del v4
    v5.append(v7)
    del v7
    v8 = v5
    del v5
    return v8
def method18(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64, v4 : list) -> object:
    v5 = method19(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    return v5
def method36(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method20(v0, v1, v2, v3)
    del v0, v1, v2, v3
    return v4
def main_body():
    v0 = Closure0()
    v1 = Closure1()
    v2 = collections.namedtuple("Leduc_Game",['event_loop_gpu', 'init'])(v0, v1)
    del v0, v1
    return v2

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
