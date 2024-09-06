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
        Union7 v711;
        switch (v11.tag) {
            case 0: { // None
                v711 = Union7{Union7_0{}};
                break;
            }
            case 1: { // Some
                Union5 v13 = v11.case1.v0;
                switch (v13.tag) {
                    case 0: { // ChanceCommunityCard
                        Union6 v662 = v13.case0.v0; bool v663 = v13.case0.v1; static_array<Union3,2l> v664 = v13.case0.v2; int v665 = v13.case0.v3; static_array<int,2l> v666 = v13.case0.v4; int v667 = v13.case0.v5;
                        unsigned int v668 = v2;
                        Union3 v669; unsigned int v670;
                        Tuple0 tmp0 = draw_card_4(v5, v668);
                        v669 = tmp0.v0; v670 = tmp0.v1;
                        v2 = v670;
                        Union2 v671;
                        v671 = Union2{Union2_0{v669}};
                        v9.push(v671);
                        int v672;
                        v672 = 2l;
                        int v673; int v674;
                        Tuple1 tmp1 = Tuple1{0l, 0l};
                        v673 = tmp1.v0; v674 = tmp1.v1;
                        while (while_method_3(v673)){
                            int v676;
                            v676 = v666[v673];
                            bool v678;
                            v678 = v674 >= v676;
                            int v679;
                            if (v678){
                                v679 = v674;
                            } else {
                                v679 = v676;
                            }
                            v674 = v679;
                            v673 += 1l ;
                        }
                        static_array<int,2l> v680;
                        int v682;
                        v682 = 0l;
                        while (while_method_3(v682)){
                            v680[v682] = v674;
                            v682 += 1l ;
                        }
                        Union6 v684;
                        v684 = Union6{Union6_1{v669}};
                        Union5 v685;
                        v685 = Union5{Union5_2{v684, true, v664, 0l, v680, v672}};
                        v711 = Union7{Union7_1{v685}};
                        break;
                    }
                    case 1: { // ChanceInit
                        unsigned int v687 = v2;
                        Union3 v688; unsigned int v689;
                        Tuple0 tmp2 = draw_card_4(v5, v687);
                        v688 = tmp2.v0; v689 = tmp2.v1;
                        v2 = v689;
                        unsigned int v690 = v2;
                        Union3 v691; unsigned int v692;
                        Tuple0 tmp3 = draw_card_4(v5, v690);
                        v691 = tmp3.v0; v692 = tmp3.v1;
                        v2 = v692;
                        Union2 v693;
                        v693 = Union2{Union2_2{0l, v688}};
                        v9.push(v693);
                        Union2 v694;
                        v694 = Union2{Union2_2{1l, v691}};
                        v9.push(v694);
                        int v695;
                        v695 = 2l;
                        static_array<int,2l> v696;
                        v696[0l] = 1l;
                        v696[1l] = 1l;
                        static_array<Union3,2l> v698;
                        v698[0l] = v688;
                        v698[1l] = v691;
                        Union6 v700;
                        v700 = Union6{Union6_0{}};
                        Union5 v701;
                        v701 = Union5{Union5_2{v700, true, v698, 0l, v696, v695}};
                        v711 = Union7{Union7_1{v701}};
                        break;
                    }
                    case 2: { // Round
                        Union6 v54 = v13.case2.v0; bool v55 = v13.case2.v1; static_array<Union3,2l> v56 = v13.case2.v2; int v57 = v13.case2.v3; static_array<int,2l> v58 = v13.case2.v4; int v59 = v13.case2.v5;
                        static_array<Union1,2l> v60 = v4;
                        Union1 v61;
                        v61 = v60[v57];
                        Union4 v478;
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
                                    double v275;
                                    v275 = (double)v231;
                                    double v276;
                                    v276 = log(v275);
                                    double v277;
                                    v277 = (double)v267;
                                    double v278;
                                    v278 = log(v277);
                                    assert("Tensor range check" && 0 <= v255 && v255 < 4l);
                                    int v279;
                                    v279 = 24576l * v255;
                                    assert("Tensor range check" && 0 <= v254 && v254 < 12288l);
                                    int v280;
                                    v280 = 2l * v254;
                                    int v281;
                                    v281 = v280 + v279;
                                    assert("Tensor range check" && 0 <= v255 && v255 < 4l);
                                    int v282;
                                    v282 = 393216l * v255;
                                    assert("Tensor range check" && 0 <= v269 && v269 < 16l);
                                    int v283;
                                    v283 = 24576l * v269;
                                    int v284;
                                    v284 = v283 + v282;
                                    assert("Tensor range check" && 0 <= v254 && v254 < 12288l);
                                    int v285;
                                    v285 = v280 + v284;
                                    double * v286;
                                    v286 = v233+v281;
                                    double * v288;
                                    v288 = v235+v281;
                                    double * v290;
                                    v290 = v247+v285;
                                    double * v292;
                                    v292 = v249+v285;
                                    int v294;
                                    v294 = sizeof(double *);
                                    unsigned long long v295;
                                    v295 = (unsigned long long)v294;
                                    unsigned long long v296;
                                    v296 = 512ull * v295;
                                    unsigned long long v297;
                                    v297 = v296 + 16ull;
                                    unsigned long long v298;
                                    v298 = v297 - 1ull;
                                    unsigned long long v299;
                                    v299 = v298 % 16ull;
                                    unsigned long long v300;
                                    v300 = v298 - v299;
                                    unsigned long long v301;
                                    v301 = v300 + v296;
                                    unsigned long long v302;
                                    v302 = v301 + 16ull;
                                    unsigned long long v303;
                                    v303 = v302 - 1ull;
                                    unsigned long long v304;
                                    v304 = v303 % 16ull;
                                    unsigned long long v305;
                                    v305 = v303 - v304;
                                    unsigned long long v306;
                                    v306 = v305 + v296;
                                    unsigned long long v307;
                                    v307 = v306 + 16ull;
                                    unsigned long long v308;
                                    v308 = v307 - 1ull;
                                    unsigned long long v309;
                                    v309 = v308 % 16ull;
                                    unsigned long long v310;
                                    v310 = v308 - v309;
                                    unsigned long long v311;
                                    v311 = v310 + v296;
                                    bool v312;
                                    v312 = v311 <= 81920ull;
                                    bool v313;
                                    v313 = v312 == false;
                                    if (v313){
                                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v312);
                                    } else {
                                    }
                                    extern __shared__ unsigned char v315[];
                                    bool v316;
                                    v316 = v311 <= v311;
                                    bool v317;
                                    v317 = v316 == false;
                                    if (v317){
                                        assert("The length of the partition has to be less than or equal to the length of the base array." && v316);
                                    } else {
                                    }
                                    double * * v319;
                                    v319 = reinterpret_cast<double * *>(&v315[0ull]);
                                    double * * v321;
                                    v321 = reinterpret_cast<double * *>(&v315[v300]);
                                    double * * v323;
                                    v323 = reinterpret_cast<double * *>(&v315[v305]);
                                    double * * v325;
                                    v325 = reinterpret_cast<double * *>(&v315[v310]);
                                    int v327;
                                    v327 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v327 && v327 < 512l);
                                    v319[v327] = v286;
                                    v321[v327] = v288;
                                    v323[v327] = v290;
                                    v325[v327] = v292;
                                    asm("barrier.cta.sync %0;" :: "r"(0l));
                                    bool v328;
                                    v328 = 0l <= v327;
                                    bool v329;
                                    v329 = v328 == false;
                                    if (v329){
                                        assert("The index needs to be zero or positive." && v328);
                                    } else {
                                    }
                                    int v331;
                                    v331 = v327 % 1l;
                                    bool v332;
                                    v332 = v327 < 512l;
                                    bool v333;
                                    v333 = v332 == false;
                                    if (v333){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v332);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v327 && v327 < 512l);
                                    int v335;
                                    v335 = 0l;
                                    while (while_method_7(v335)){
                                        bool v337;
                                        v337 = v328 && v332;
                                        bool v338;
                                        v338 = v337 == false;
                                        if (v338){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v337);
                                        } else {
                                        }
                                        bool v340;
                                        v340 = 0l <= v335;
                                        bool v342;
                                        if (v340){
                                            bool v341;
                                            v341 = v335 < 1l;
                                            v342 = v341;
                                        } else {
                                            v342 = false;
                                        }
                                        bool v343;
                                        v343 = v342 == false;
                                        if (v343){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v342);
                                        } else {
                                        }
                                        int v345;
                                        v345 = v335 * 512l;
                                        int v346;
                                        v346 = v345 + v327;
                                        assert("Tensor range check" && 0 <= v335 && v335 < 1l);
                                        int v347;
                                        v347 = 512l * v335;
                                        int v348;
                                        v348 = v347 + v327;
                                        double * v349;
                                        v349 = v319[v348];
                                        double * v350;
                                        v350 = v321[v348];
                                        double * v351;
                                        v351 = v323[v348];
                                        double * v352;
                                        v352 = v325[v348];
                                        int v353;
                                        v353 = blockIdx.x;
                                        int v354;
                                        v354 = v353 * 512l;
                                        int v355;
                                        v355 = v354 + v346;
                                        assert("Tensor range check" && 0 <= v331 && v331 < 1l);
                                        int v356;
                                        v356 = 2l * v331;
                                        double v357[2l];
                                        double v358[2l];
                                        int v359[2l];
                                        int v360;
                                        v360 = 0l;
                                        while (while_method_7(v360)){
                                            assert("Tensor range check" && 0 <= v360 && v360 < 1l);
                                            int v362;
                                            v362 = 2l * v360;
                                            assert("Tensor range check" && 0 <= v360 && v360 < 1l);
                                            int v363;
                                            v363 = v362 + v356;
                                            int4* v364;
                                            v364 = reinterpret_cast<int4*>(v349 + v363);
                                            int4* v365;
                                            v365 = reinterpret_cast<int4*>(v357 + v362);
                                            assert("Pointer alignment check" && (unsigned long long)(v364) % 2l == 0 && (unsigned long long)(v365) % 2l == 0);
                                            *v365 = *v364;
                                            int4* v366;
                                            v366 = reinterpret_cast<int4*>(v350 + v363);
                                            int4* v367;
                                            v367 = reinterpret_cast<int4*>(v358 + v362);
                                            assert("Pointer alignment check" && (unsigned long long)(v366) % 2l == 0 && (unsigned long long)(v367) % 2l == 0);
                                            *v367 = *v366;
                                            v360 += 1l ;
                                        }
                                        int v368;
                                        v368 = 0l;
                                        while (while_method_7(v368)){
                                            int v370;
                                            v370 = 0l;
                                            while (while_method_3(v370)){
                                                bool v372;
                                                v372 = 0l <= v370;
                                                bool v374;
                                                if (v372){
                                                    bool v373;
                                                    v373 = v370 < 2l;
                                                    v374 = v373;
                                                } else {
                                                    v374 = false;
                                                }
                                                bool v375;
                                                v375 = v374 == false;
                                                if (v375){
                                                    assert("The indices should be inside the range of the dimension." && v374);
                                                } else {
                                                }
                                                bool v377;
                                                v377 = 0l <= v331;
                                                bool v379;
                                                if (v377){
                                                    bool v378;
                                                    v378 = v331 < 1l;
                                                    v379 = v378;
                                                } else {
                                                    v379 = false;
                                                }
                                                bool v380;
                                                v380 = v379 == false;
                                                if (v380){
                                                    assert("The indices should be inside the range of the dimension." && v379);
                                                } else {
                                                }
                                                int v382;
                                                v382 = v331 * 2l;
                                                int v383;
                                                v383 = v370 + v382;
                                                bool v384;
                                                v384 = 0l <= v368;
                                                bool v386;
                                                if (v384){
                                                    bool v385;
                                                    v385 = v368 < 1l;
                                                    v386 = v385;
                                                } else {
                                                    v386 = false;
                                                }
                                                bool v387;
                                                v387 = v386 == false;
                                                if (v387){
                                                    assert("The indices should be inside the range of the dimension." && v386);
                                                } else {
                                                }
                                                int v389;
                                                v389 = v368 * 2l;
                                                int v390;
                                                v390 = v383 + v389;
                                                assert("Tensor range check" && 0 <= v368 && v368 < 1l);
                                                assert("Tensor range check" && 0 <= v370 && v370 < 2l);
                                                int v391;
                                                v391 = 2l * v368;
                                                int v392;
                                                v392 = v391 + v370;
                                                v359[v392] = v390;
                                                v370 += 1l ;
                                            }
                                            v368 += 1l ;
                                        }
                                        int v393;
                                        v393 = 0l;
                                        while (while_method_7(v393)){
                                            assert("Tensor range check" && 0 <= v393 && v393 < 1l);
                                            int v395;
                                            v395 = 2l * v393;
                                            int v396;
                                            v396 = v395 + v356;
                                            assert("Tensor range check" && 0 <= v393 && v393 < 1l);
                                            int4* v397;
                                            v397 = reinterpret_cast<int4*>(v357 + v395);
                                            int4* v398;
                                            v398 = reinterpret_cast<int4*>(v351 + v396);
                                            assert("Pointer alignment check" && (unsigned long long)(v397) % 2l == 0 && (unsigned long long)(v398) % 2l == 0);
                                            *v398 = *v397;
                                            int4* v399;
                                            v399 = reinterpret_cast<int4*>(v358 + v395);
                                            int4* v400;
                                            v400 = reinterpret_cast<int4*>(v352 + v396);
                                            assert("Pointer alignment check" && (unsigned long long)(v399) % 2l == 0 && (unsigned long long)(v400) % 2l == 0);
                                            *v400 = *v399;
                                            v393 += 1l ;
                                        }
                                        assert("Tensor range check" && 0 <= v346 && v346 < 512l);
                                        v335 += 1l ;
                                    }
                                    asm("barrier.cta.sync %0;" :: "r"(0l));
                                    assert("Tensor range check" && 0 <= v327 && v327 < 512l);
                                    asm("barrier.cta.sync %0;" :: "r"(0l));
                                    assert("Tensor range check" && 0 <= v57 && v57 < 2l);
                                    int v401;
                                    v401 = v57 + v281;
                                    double v402;
                                    v402 = v233[v401];
                                    double v403;
                                    v403 = v235[v401];
                                    double v404;
                                    v404 = v278 + v402;
                                    double v405;
                                    v405 = v276 + v403;
                                    assert("Tensor range check" && 0 <= v57 && v57 < 2l);
                                    v233[v401] = v404;
                                    v235[v401] = v405;
                                    v255 += 1l ;
                                }
                                bool v406;
                                v406 = 0l == v232;
                                Union10 v415;
                                if (v406){
                                    v415 = Union10{Union10_1{}};
                                } else {
                                    bool v408;
                                    v408 = 1l == v232;
                                    if (v408){
                                        v415 = Union10{Union10_0{}};
                                    } else {
                                        bool v410;
                                        v410 = 2l == v232;
                                        if (v410){
                                            v415 = Union10{Union10_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v415.tag) {
                                    case 0: { // AA_Call
                                        v478 = Union4{Union4_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v416;
                                        v416 = v58[0l];
                                        int v418; int v419;
                                        Tuple1 tmp17 = Tuple1{1l, v416};
                                        v418 = tmp17.v0; v419 = tmp17.v1;
                                        while (while_method_3(v418)){
                                            int v421;
                                            v421 = v58[v418];
                                            bool v423;
                                            v423 = v419 >= v421;
                                            int v424;
                                            if (v423){
                                                v424 = v419;
                                            } else {
                                                v424 = v421;
                                            }
                                            v419 = v424;
                                            v418 += 1l ;
                                        }
                                        int v425;
                                        v425 = v58[v57];
                                        bool v427;
                                        v427 = v425 == v419;
                                        if (v427){
                                            v478 = Union4{Union4_0{}};
                                        } else {
                                            v478 = Union4{Union4_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v432;
                                        v432 = v59 > 0l;
                                        if (v432){
                                            v478 = Union4{Union4_2{}};
                                        } else {
                                            v478 = Union4{Union4_0{}};
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
                                static_array_list<Union4,3l> v439;
                                v439 = static_array_list<Union4,3l>{};
                                v439.unsafe_set_length(1l);
                                Union4 v441;
                                v441 = Union4{Union4_0{}};
                                v439[0l] = v441;
                                int v443;
                                v443 = v58[0l];
                                int v445;
                                v445 = v58[1l];
                                bool v447;
                                v447 = v443 == v445;
                                bool v448;
                                v448 = v447 != true;
                                if (v448){
                                    Union4 v449;
                                    v449 = Union4{Union4_1{}};
                                    v439.push(v449);
                                } else {
                                }
                                bool v450;
                                v450 = v59 > 0l;
                                if (v450){
                                    Union4 v451;
                                    v451 = Union4{Union4_2{}};
                                    v439.push(v451);
                                } else {
                                }
                                int v452;
                                v452 = v439.length;
                                int v453;
                                v453 = v452 - 1l;
                                int v454;
                                v454 = 0l;
                                while (while_method_5(v453, v454)){
                                    int v456;
                                    v456 = v439.length;
                                    int v457;
                                    v457 = int_range_10(v456, v454, v5);
                                    Union4 v458;
                                    v458 = v439[v454];
                                    Union4 v460;
                                    v460 = v439[v457];
                                    v439[v454] = v460;
                                    v439[v457] = v458;
                                    v454 += 1l ;
                                }
                                Union4 v462;
                                v462 = v439.pop();
                                int v463;
                                v463 = sizeof(Union4);
                                unsigned long long v464;
                                v464 = (unsigned long long)v463;
                                bool v465;
                                v465 = v464 <= 81920ull;
                                bool v466;
                                v466 = v465 == false;
                                if (v466){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v465);
                                } else {
                                }
                                extern __shared__ unsigned char v468[];
                                bool v469;
                                v469 = v464 <= v464;
                                bool v470;
                                v470 = v469 == false;
                                if (v470){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v469);
                                } else {
                                }
                                Union4 * v472;
                                v472 = reinterpret_cast<Union4 *>(&v468[0ull]);
                                int v474;
                                v474 = threadIdx.x;
                                bool v475;
                                v475 = v474 == 0l;
                                if (v475){
                                    v472[0l] = v462;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                Union4 v476;
                                v476 = v472[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                v478 = v476;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union2 v479;
                        v479 = Union2{Union2_1{v57, v478}};
                        v9.push(v479);
                        Union5 v565;
                        switch (v54.tag) {
                            case 0: { // None
                                switch (v478.tag) {
                                    case 0: { // Call
                                        if (v55){
                                            bool v529;
                                            v529 = v57 == 0l;
                                            int v530;
                                            if (v529){
                                                v530 = 1l;
                                            } else {
                                                v530 = 0l;
                                            }
                                            v565 = Union5{Union5_2{v54, false, v56, v530, v58, v59}};
                                        } else {
                                            v565 = Union5{Union5_0{v54, v55, v56, v57, v58, v59}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v565 = Union5{Union5_5{v54, v55, v56, v57, v58, v59}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v534;
                                        v534 = v59 > 0l;
                                        if (v534){
                                            bool v535;
                                            v535 = v57 == 0l;
                                            int v536;
                                            if (v535){
                                                v536 = 1l;
                                            } else {
                                                v536 = 0l;
                                            }
                                            int v537;
                                            v537 = -1l + v59;
                                            int v538; int v539;
                                            Tuple1 tmp18 = Tuple1{0l, 0l};
                                            v538 = tmp18.v0; v539 = tmp18.v1;
                                            while (while_method_3(v538)){
                                                int v541;
                                                v541 = v58[v538];
                                                bool v543;
                                                v543 = v539 >= v541;
                                                int v544;
                                                if (v543){
                                                    v544 = v539;
                                                } else {
                                                    v544 = v541;
                                                }
                                                v539 = v544;
                                                v538 += 1l ;
                                            }
                                            static_array<int,2l> v545;
                                            int v547;
                                            v547 = 0l;
                                            while (while_method_3(v547)){
                                                v545[v547] = v539;
                                                v547 += 1l ;
                                            }
                                            static_array<int,2l> v549;
                                            int v551;
                                            v551 = 0l;
                                            while (while_method_3(v551)){
                                                int v553;
                                                v553 = v545[v551];
                                                bool v555;
                                                v555 = v551 == v57;
                                                int v557;
                                                if (v555){
                                                    int v556;
                                                    v556 = v553 + 2l;
                                                    v557 = v556;
                                                } else {
                                                    v557 = v553;
                                                }
                                                v549[v551] = v557;
                                                v551 += 1l ;
                                            }
                                            v565 = Union5{Union5_2{v54, false, v56, v536, v549, v537}};
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
                                Union3 v480 = v54.case1.v0;
                                switch (v478.tag) {
                                    case 0: { // Call
                                        if (v55){
                                            bool v482;
                                            v482 = v57 == 0l;
                                            int v483;
                                            if (v482){
                                                v483 = 1l;
                                            } else {
                                                v483 = 0l;
                                            }
                                            v565 = Union5{Union5_2{v54, false, v56, v483, v58, v59}};
                                        } else {
                                            int v485; int v486;
                                            Tuple1 tmp19 = Tuple1{0l, 0l};
                                            v485 = tmp19.v0; v486 = tmp19.v1;
                                            while (while_method_3(v485)){
                                                int v488;
                                                v488 = v58[v485];
                                                bool v490;
                                                v490 = v486 >= v488;
                                                int v491;
                                                if (v490){
                                                    v491 = v486;
                                                } else {
                                                    v491 = v488;
                                                }
                                                v486 = v491;
                                                v485 += 1l ;
                                            }
                                            static_array<int,2l> v492;
                                            int v494;
                                            v494 = 0l;
                                            while (while_method_3(v494)){
                                                v492[v494] = v486;
                                                v494 += 1l ;
                                            }
                                            v565 = Union5{Union5_4{v54, v55, v56, v57, v492, v59}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v565 = Union5{Union5_5{v54, v55, v56, v57, v58, v59}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v498;
                                        v498 = v59 > 0l;
                                        if (v498){
                                            bool v499;
                                            v499 = v57 == 0l;
                                            int v500;
                                            if (v499){
                                                v500 = 1l;
                                            } else {
                                                v500 = 0l;
                                            }
                                            int v501;
                                            v501 = -1l + v59;
                                            int v502; int v503;
                                            Tuple1 tmp20 = Tuple1{0l, 0l};
                                            v502 = tmp20.v0; v503 = tmp20.v1;
                                            while (while_method_3(v502)){
                                                int v505;
                                                v505 = v58[v502];
                                                bool v507;
                                                v507 = v503 >= v505;
                                                int v508;
                                                if (v507){
                                                    v508 = v503;
                                                } else {
                                                    v508 = v505;
                                                }
                                                v503 = v508;
                                                v502 += 1l ;
                                            }
                                            static_array<int,2l> v509;
                                            int v511;
                                            v511 = 0l;
                                            while (while_method_3(v511)){
                                                v509[v511] = v503;
                                                v511 += 1l ;
                                            }
                                            static_array<int,2l> v513;
                                            int v515;
                                            v515 = 0l;
                                            while (while_method_3(v515)){
                                                int v517;
                                                v517 = v509[v515];
                                                bool v519;
                                                v519 = v515 == v57;
                                                int v521;
                                                if (v519){
                                                    int v520;
                                                    v520 = v517 + 4l;
                                                    v521 = v520;
                                                } else {
                                                    v521 = v517;
                                                }
                                                v513[v515] = v521;
                                                v515 += 1l ;
                                            }
                                            v565 = Union5{Union5_2{v54, false, v56, v500, v513, v501}};
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
                        v711 = Union7{Union7_1{v565}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union6 v567 = v13.case3.v0; bool v568 = v13.case3.v1; static_array<Union3,2l> v569 = v13.case3.v2; int v570 = v13.case3.v3; static_array<int,2l> v571 = v13.case3.v4; int v572 = v13.case3.v5; Union4 v573 = v13.case3.v6;
                        Union2 v574;
                        v574 = Union2{Union2_1{v570, v573}};
                        v9.push(v574);
                        Union5 v660;
                        switch (v567.tag) {
                            case 0: { // None
                                switch (v573.tag) {
                                    case 0: { // Call
                                        if (v568){
                                            bool v624;
                                            v624 = v570 == 0l;
                                            int v625;
                                            if (v624){
                                                v625 = 1l;
                                            } else {
                                                v625 = 0l;
                                            }
                                            v660 = Union5{Union5_2{v567, false, v569, v625, v571, v572}};
                                        } else {
                                            v660 = Union5{Union5_0{v567, v568, v569, v570, v571, v572}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v660 = Union5{Union5_5{v567, v568, v569, v570, v571, v572}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v629;
                                        v629 = v572 > 0l;
                                        if (v629){
                                            bool v630;
                                            v630 = v570 == 0l;
                                            int v631;
                                            if (v630){
                                                v631 = 1l;
                                            } else {
                                                v631 = 0l;
                                            }
                                            int v632;
                                            v632 = -1l + v572;
                                            int v633; int v634;
                                            Tuple1 tmp21 = Tuple1{0l, 0l};
                                            v633 = tmp21.v0; v634 = tmp21.v1;
                                            while (while_method_3(v633)){
                                                int v636;
                                                v636 = v571[v633];
                                                bool v638;
                                                v638 = v634 >= v636;
                                                int v639;
                                                if (v638){
                                                    v639 = v634;
                                                } else {
                                                    v639 = v636;
                                                }
                                                v634 = v639;
                                                v633 += 1l ;
                                            }
                                            static_array<int,2l> v640;
                                            int v642;
                                            v642 = 0l;
                                            while (while_method_3(v642)){
                                                v640[v642] = v634;
                                                v642 += 1l ;
                                            }
                                            static_array<int,2l> v644;
                                            int v646;
                                            v646 = 0l;
                                            while (while_method_3(v646)){
                                                int v648;
                                                v648 = v640[v646];
                                                bool v650;
                                                v650 = v646 == v570;
                                                int v652;
                                                if (v650){
                                                    int v651;
                                                    v651 = v648 + 2l;
                                                    v652 = v651;
                                                } else {
                                                    v652 = v648;
                                                }
                                                v644[v646] = v652;
                                                v646 += 1l ;
                                            }
                                            v660 = Union5{Union5_2{v567, false, v569, v631, v644, v632}};
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
                                Union3 v575 = v567.case1.v0;
                                switch (v573.tag) {
                                    case 0: { // Call
                                        if (v568){
                                            bool v577;
                                            v577 = v570 == 0l;
                                            int v578;
                                            if (v577){
                                                v578 = 1l;
                                            } else {
                                                v578 = 0l;
                                            }
                                            v660 = Union5{Union5_2{v567, false, v569, v578, v571, v572}};
                                        } else {
                                            int v580; int v581;
                                            Tuple1 tmp22 = Tuple1{0l, 0l};
                                            v580 = tmp22.v0; v581 = tmp22.v1;
                                            while (while_method_3(v580)){
                                                int v583;
                                                v583 = v571[v580];
                                                bool v585;
                                                v585 = v581 >= v583;
                                                int v586;
                                                if (v585){
                                                    v586 = v581;
                                                } else {
                                                    v586 = v583;
                                                }
                                                v581 = v586;
                                                v580 += 1l ;
                                            }
                                            static_array<int,2l> v587;
                                            int v589;
                                            v589 = 0l;
                                            while (while_method_3(v589)){
                                                v587[v589] = v581;
                                                v589 += 1l ;
                                            }
                                            v660 = Union5{Union5_4{v567, v568, v569, v570, v587, v572}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v660 = Union5{Union5_5{v567, v568, v569, v570, v571, v572}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v593;
                                        v593 = v572 > 0l;
                                        if (v593){
                                            bool v594;
                                            v594 = v570 == 0l;
                                            int v595;
                                            if (v594){
                                                v595 = 1l;
                                            } else {
                                                v595 = 0l;
                                            }
                                            int v596;
                                            v596 = -1l + v572;
                                            int v597; int v598;
                                            Tuple1 tmp23 = Tuple1{0l, 0l};
                                            v597 = tmp23.v0; v598 = tmp23.v1;
                                            while (while_method_3(v597)){
                                                int v600;
                                                v600 = v571[v597];
                                                bool v602;
                                                v602 = v598 >= v600;
                                                int v603;
                                                if (v602){
                                                    v603 = v598;
                                                } else {
                                                    v603 = v600;
                                                }
                                                v598 = v603;
                                                v597 += 1l ;
                                            }
                                            static_array<int,2l> v604;
                                            int v606;
                                            v606 = 0l;
                                            while (while_method_3(v606)){
                                                v604[v606] = v598;
                                                v606 += 1l ;
                                            }
                                            static_array<int,2l> v608;
                                            int v610;
                                            v610 = 0l;
                                            while (while_method_3(v610)){
                                                int v612;
                                                v612 = v604[v610];
                                                bool v614;
                                                v614 = v610 == v570;
                                                int v616;
                                                if (v614){
                                                    int v615;
                                                    v615 = v612 + 4l;
                                                    v616 = v615;
                                                } else {
                                                    v616 = v612;
                                                }
                                                v608[v610] = v616;
                                                v610 += 1l ;
                                            }
                                            v660 = Union5{Union5_2{v567, false, v569, v595, v608, v596}};
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
                        v711 = Union7{Union7_1{v660}};
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
                        v711 = Union7{Union7_0{}};
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
                        v711 = Union7{Union7_0{}};
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
        v11 = v711;
    }
    return v7;
}
__device__ inline bool while_method_10(int v0){
    bool v1;
    v1 = v0 > 0l;
    return v1;
}
__device__ inline bool while_method_11(int v0){
    bool v1;
    v1 = v0 < 16384l;
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
            v27 = Union1{Union1_1{}};
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
                    float v40;
                    v40 = v39[0l];
                    int v42;
                    v42 = 0l;
                    while (while_method_6(v42)){
                        double * v44;
                        v44 = reinterpret_cast<double *>(&v3[2097168ull]);
                        double * v46;
                        v46 = reinterpret_cast<double *>(&v3[2883600ull]);
                        int * v48;
                        v48 = reinterpret_cast<int *>(&v3[3670032ull]);
                        int v50;
                        v50 = threadIdx.x;
                        int v51;
                        v51 = blockIdx.x;
                        int v52;
                        v52 = v51 * 512l;
                        int v53;
                        v53 = v50 + v52;
                        assert("Tensor range check" && 0 <= v42 && v42 < 4l);
                        assert("Tensor range check" && 0 <= v53 && v53 < 12288l);
                        int v54;
                        v54 = 2l * v53;
                        int v55;
                        v55 = 24576l * v42;
                        int v56;
                        v56 = v55 + v54;
                        double v57;
                        v57 = v44[v56];
                        double v58;
                        v58 = v46[v56];
                        double v59;
                        v59 = v57 - v58;
                        double v60;
                        v60 = exp(v59);
                        float v61;
                        v61 = (float)v60;
                        float v62;
                        v62 = v40 * v61;
                        assert("Tensor range check" && 0 <= v42 && v42 < 4l);
                        assert("Tensor range check" && 0 <= v29 && v29 < 1024l);
                        assert("Tensor range check" && 0 <= v31 && v31 < 16l);
                        int v63;
                        v63 = 16l * v29;
                        int v64;
                        v64 = v63 + v31;
                        int v65;
                        v65 = 16384l * v42;
                        int v66;
                        v66 = v65 + v64;
                        float * v67;
                        v67 = v5+v66;
                        float * v69;
                        v69 = v6+v66;
                        float v71;
                        v71 = atomicAdd(v67,v62);
                        float v72;
                        v72 = atomicAdd(v69,v61);
                        v42 += 1l ;
                    }
                    unsigned int * v73;
                    v73 = reinterpret_cast<unsigned int *>(&v1[12582912ull]);
                    int * v75;
                    v75 = reinterpret_cast<int *>(&v3[262144ull]);
                    float * v77;
                    v77 = reinterpret_cast<float *>(&v3[262160ull]);
                    float * v79;
                    v79 = reinterpret_cast<float *>(&v3[524304ull]);
                    float * v81;
                    v81 = reinterpret_cast<float *>(&v3[786448ull]);
                    float * v83;
                    v83 = reinterpret_cast<float *>(&v3[1048592ull]);
                    float * v85;
                    v85 = reinterpret_cast<float *>(&v3[1310736ull]);
                    float * v87;
                    v87 = reinterpret_cast<float *>(&v3[1572880ull]);
                    float * v89;
                    v89 = reinterpret_cast<float *>(&v3[1835024ull]);
                    int * v91;
                    v91 = reinterpret_cast<int *>(&v1[12779520ull]);
                    float * v93;
                    v93 = reinterpret_cast<float *>(&v1[15925248ull]);
                    int * v95;
                    v95 = reinterpret_cast<int *>(&v1[19070976ull]);
                    int * v97;
                    v97 = reinterpret_cast<int *>(&v1[22216704ull]);
                    double * v99;
                    v99 = reinterpret_cast<double *>(&v1[25362432ull]);
                    double * v101;
                    v101 = reinterpret_cast<double *>(&v1[37945344ull]);
                    double * v103;
                    v103 = reinterpret_cast<double *>(&v3[2097168ull]);
                    double * v105;
                    v105 = reinterpret_cast<double *>(&v3[2883600ull]);
                    int * v107;
                    v107 = reinterpret_cast<int *>(&v3[3670032ull]);
                    int v109;
                    v109 = 0l;
                    while (while_method_6(v109)){
                        int v111;
                        v111 = threadIdx.x;
                        int v112;
                        v112 = blockIdx.x;
                        int v113;
                        v113 = v112 * 512l;
                        int v114;
                        v114 = v111 + v113;
                        float v115[2l];
                        int v116;
                        v116 = 0l;
                        while (while_method_3(v116)){
                            float v118;
                            v118 = v39[v116];
                            v115[v116] = v118;
                            v116 += 1l ;
                        }
                        assert("Tensor range check" && 0 <= v109 && v109 < 4l);
                        assert("Tensor range check" && 0 <= v114 && v114 < 12288l);
                        int v120;
                        v120 = 12288l * v109;
                        int v121;
                        v121 = v120 + v114;
                        int v122;
                        v122 = v107[v121];
                        int v123;
                        v123 = v122;
                        while (while_method_10(v123)){
                            v123 -= 1l ;
                            assert("Tensor range check" && 0 <= v109 && v109 < 4l);
                            assert("Tensor range check" && 0 <= v123 && v123 < 16l);
                            assert("Tensor range check" && 0 <= v114 && v114 < 12288l);
                            int v125;
                            v125 = 12288l * v123;
                            int v126;
                            v126 = v125 + v114;
                            int v127;
                            v127 = 196608l * v109;
                            int v128;
                            v128 = v127 + v126;
                            int v129;
                            v129 = v91[v128];
                            float v130;
                            v130 = v93[v128];
                            int v131;
                            v131 = v95[v128];
                            int v132;
                            v132 = v97[v128];
                            assert("Tensor range check" && 0 <= v131 && v131 < 2l);
                            float v133;
                            v133 = v115[v131];
                            assert("Tensor range check" && 0 <= v109 && v109 < 4l);
                            int v134;
                            v134 = 16384l * v109;
                            assert("Tensor range check" && 0 <= v132 && v132 < 4096l);
                            int v135;
                            v135 = 4l * v132;
                            int v136;
                            v136 = v135 + v134;
                            float * v137;
                            v137 = v77+v136;
                            float * v139;
                            v139 = v79+v136;
                            float * v141;
                            v141 = v81+v136;
                            float * v143;
                            v143 = v83+v136;
                            float * v145;
                            v145 = v85+v136;
                            float * v147;
                            v147 = v87+v136;
                            float * v149;
                            v149 = v89+v136;
                            assert("Tensor range check" && 0 <= v109 && v109 < 4l);
                            int v151;
                            v151 = 393216l * v109;
                            assert("Tensor range check" && 0 <= v123 && v123 < 16l);
                            int v152;
                            v152 = 24576l * v123;
                            int v153;
                            v153 = v152 + v151;
                            assert("Tensor range check" && 0 <= v114 && v114 < 12288l);
                            int v154;
                            v154 = 2l * v114;
                            int v155;
                            v155 = v154 + v153;
                            double v156[2l];
                            int v157;
                            v157 = 0l;
                            while (while_method_3(v157)){
                                assert("Tensor range check" && 0 <= v157 && v157 < 2l);
                                int v159;
                                v159 = v157 + v155;
                                double v160;
                                v160 = v99[v159];
                                bool v161;
                                v161 = v131 == v157;
                                double v162;
                                if (v161){
                                    v162 = 0.0;
                                } else {
                                    v162 = v160;
                                }
                                assert("Tensor range check" && 0 <= v157 && v157 < 2l);
                                v156[v157] = v162;
                                v157 += 1l ;
                            }
                            double v163;
                            v163 = 0.0;
                            int v164;
                            v164 = 0l;
                            while (while_method_3(v164)){
                                assert("Tensor range check" && 0 <= v164 && v164 < 2l);
                                double v166;
                                v166 = v156[v164];
                                double v167;
                                v167 = v163 + v166;
                                v163 = v167;
                                v164 += 1l ;
                            }
                            double v168;
                            v168 = 0.0;
                            int v169;
                            v169 = 0l;
                            while (while_method_3(v169)){
                                assert("Tensor range check" && 0 <= v169 && v169 < 2l);
                                int v171;
                                v171 = v169 + v155;
                                double v172;
                                v172 = v101[v171];
                                double v173;
                                v173 = v168 + v172;
                                v168 = v173;
                                v169 += 1l ;
                            }
                            double v174;
                            v174 = v163 - v168;
                            double v175;
                            v175 = exp(v174);
                            float v176;
                            v176 = (float)v175;
                            float v177;
                            v177 = v133 * v176;
                            assert("Tensor range check" && 0 <= v129 && v129 < 4l);
                            float * v178;
                            v178 = v147+v129;
                            float * v180;
                            v180 = v149+v129;
                            float v182;
                            v182 = atomicAdd(v178,v177);
                            float v183;
                            v183 = atomicAdd(v180,v176);
                            float * v184;
                            v184 = v139+0l;
                            float * v186;
                            v186 = v143+0l;
                            float * v188;
                            v188 = v145+0l;
                            int v190;
                            v190 = sizeof(float *);
                            unsigned long long v191;
                            v191 = (unsigned long long)v190;
                            unsigned long long v192;
                            v192 = 512ull * v191;
                            unsigned long long v193;
                            v193 = 8192ull + v192;
                            unsigned long long v194;
                            v194 = v193 + 16ull;
                            unsigned long long v195;
                            v195 = v194 - 1ull;
                            unsigned long long v196;
                            v196 = v195 % 16ull;
                            unsigned long long v197;
                            v197 = v195 - v196;
                            unsigned long long v198;
                            v198 = v197 + v192;
                            unsigned long long v199;
                            v199 = v198 + 16ull;
                            unsigned long long v200;
                            v200 = v199 - 1ull;
                            unsigned long long v201;
                            v201 = v200 % 16ull;
                            unsigned long long v202;
                            v202 = v200 - v201;
                            unsigned long long v203;
                            v203 = v202 + v192;
                            unsigned long long v204;
                            v204 = v203 + 16ull;
                            unsigned long long v205;
                            v205 = v204 - 1ull;
                            unsigned long long v206;
                            v206 = v205 % 16ull;
                            unsigned long long v207;
                            v207 = v205 - v206;
                            unsigned long long v208;
                            v208 = v207 + v192;
                            unsigned long long v209;
                            v209 = v208 + 16ull;
                            unsigned long long v210;
                            v210 = v209 - 1ull;
                            unsigned long long v211;
                            v211 = v210 % 16ull;
                            unsigned long long v212;
                            v212 = v210 - v211;
                            unsigned long long v213;
                            v213 = v212 + 2048ull;
                            bool v214;
                            v214 = v213 <= 81920ull;
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
                            float * v221;
                            v221 = reinterpret_cast<float *>(&v217[0ull]);
                            int * v223;
                            v223 = reinterpret_cast<int *>(&v217[2048ull]);
                            float * v225;
                            v225 = reinterpret_cast<float *>(&v217[4096ull]);
                            float * v227;
                            v227 = reinterpret_cast<float *>(&v217[6144ull]);
                            float * * v229;
                            v229 = reinterpret_cast<float * *>(&v217[8192ull]);
                            float * * v231;
                            v231 = reinterpret_cast<float * *>(&v217[v197]);
                            float * * v233;
                            v233 = reinterpret_cast<float * *>(&v217[v202]);
                            float * * v235;
                            v235 = reinterpret_cast<float * *>(&v217[v207]);
                            float * v237;
                            v237 = reinterpret_cast<float *>(&v217[v212]);
                            int v239;
                            v239 = threadIdx.x;
                            assert("Tensor range check" && 0 <= v239 && v239 < 512l);
                            v221[v239] = v130;
                            v223[v239] = v129;
                            v225[v239] = v133;
                            v227[v239] = v176;
                            v229[v239] = v141;
                            v231[v239] = v184;
                            v233[v239] = v186;
                            v235[v239] = v188;
                            asm("barrier.cta.sync %0;" :: "r"(0l));
                            bool v240;
                            v240 = 0l <= v239;
                            bool v241;
                            v241 = v240 == false;
                            if (v241){
                                assert("The index needs to be zero or positive." && v240);
                            } else {
                            }
                            int v243;
                            v243 = v239 % 1l;
                            bool v244;
                            v244 = v239 < 512l;
                            bool v245;
                            v245 = v244 == false;
                            if (v245){
                                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v244);
                            } else {
                            }
                            assert("Tensor range check" && 0 <= v239 && v239 < 512l);
                            int v247;
                            v247 = 0l;
                            while (while_method_7(v247)){
                                bool v249;
                                v249 = v240 && v244;
                                bool v250;
                                v250 = v249 == false;
                                if (v250){
                                    assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v249);
                                } else {
                                }
                                bool v252;
                                v252 = 0l <= v247;
                                bool v254;
                                if (v252){
                                    bool v253;
                                    v253 = v247 < 1l;
                                    v254 = v253;
                                } else {
                                    v254 = false;
                                }
                                bool v255;
                                v255 = v254 == false;
                                if (v255){
                                    assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v254);
                                } else {
                                }
                                int v257;
                                v257 = v247 * 512l;
                                int v258;
                                v258 = v257 + v239;
                                assert("Tensor range check" && 0 <= v247 && v247 < 1l);
                                int v259;
                                v259 = 512l * v247;
                                int v260;
                                v260 = v259 + v239;
                                float v261;
                                v261 = v221[v260];
                                int v262;
                                v262 = v223[v260];
                                float v263;
                                v263 = v225[v260];
                                float v264;
                                v264 = v227[v260];
                                float * v265;
                                v265 = v229[v260];
                                float * v266;
                                v266 = v231[v260];
                                float * v267;
                                v267 = v233[v260];
                                float * v268;
                                v268 = v235[v260];
                                int v269;
                                v269 = blockIdx.x;
                                int v270;
                                v270 = v269 * 512l;
                                int v271;
                                v271 = v270 + v258;
                                assert("Tensor range check" && 0 <= v243 && v243 < 1l);
                                int v272;
                                v272 = 4l * v243;
                                float v273[4l];
                                float v274[4l];
                                float v275[4l];
                                int v276[4l];
                                int v277;
                                v277 = 0l;
                                while (while_method_7(v277)){
                                    assert("Tensor range check" && 0 <= v277 && v277 < 1l);
                                    int v279;
                                    v279 = 4l * v277;
                                    assert("Tensor range check" && 0 <= v277 && v277 < 1l);
                                    int v280;
                                    v280 = v279 + v272;
                                    int4* v281;
                                    v281 = reinterpret_cast<int4*>(v266 + v280);
                                    int4* v282;
                                    v282 = reinterpret_cast<int4*>(v273 + v279);
                                    assert("Pointer alignment check" && (unsigned long long)(v281) % 4l == 0 && (unsigned long long)(v282) % 4l == 0);
                                    *v282 = *v281;
                                    int4* v283;
                                    v283 = reinterpret_cast<int4*>(v267 + v280);
                                    int4* v284;
                                    v284 = reinterpret_cast<int4*>(v274 + v279);
                                    assert("Pointer alignment check" && (unsigned long long)(v283) % 4l == 0 && (unsigned long long)(v284) % 4l == 0);
                                    *v284 = *v283;
                                    int4* v285;
                                    v285 = reinterpret_cast<int4*>(v268 + v280);
                                    int4* v286;
                                    v286 = reinterpret_cast<int4*>(v275 + v279);
                                    assert("Pointer alignment check" && (unsigned long long)(v285) % 4l == 0 && (unsigned long long)(v286) % 4l == 0);
                                    *v286 = *v285;
                                    v277 += 1l ;
                                }
                                int v287;
                                v287 = 0l;
                                while (while_method_7(v287)){
                                    int v289;
                                    v289 = 0l;
                                    while (while_method_6(v289)){
                                        bool v291;
                                        v291 = 0l <= v289;
                                        bool v293;
                                        if (v291){
                                            bool v292;
                                            v292 = v289 < 4l;
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
                                        v296 = 0l <= v243;
                                        bool v298;
                                        if (v296){
                                            bool v297;
                                            v297 = v243 < 1l;
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
                                        v301 = v243 * 4l;
                                        int v302;
                                        v302 = v289 + v301;
                                        bool v303;
                                        v303 = 0l <= v287;
                                        bool v305;
                                        if (v303){
                                            bool v304;
                                            v304 = v287 < 1l;
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
                                        v308 = v287 * 4l;
                                        int v309;
                                        v309 = v302 + v308;
                                        assert("Tensor range check" && 0 <= v287 && v287 < 1l);
                                        assert("Tensor range check" && 0 <= v289 && v289 < 4l);
                                        int v310;
                                        v310 = 4l * v287;
                                        int v311;
                                        v311 = v310 + v289;
                                        v276[v311] = v309;
                                        v289 += 1l ;
                                    }
                                    v287 += 1l ;
                                }
                                float v312[4l];
                                int v313;
                                v313 = 0l;
                                while (while_method_7(v313)){
                                    int v315;
                                    v315 = 0l;
                                    while (while_method_6(v315)){
                                        assert("Tensor range check" && 0 <= v313 && v313 < 1l);
                                        assert("Tensor range check" && 0 <= v315 && v315 < 4l);
                                        int v317;
                                        v317 = 4l * v313;
                                        int v318;
                                        v318 = v317 + v315;
                                        float v319;
                                        v319 = v274[v318];
                                        float v320;
                                        v320 = v275[v318];
                                        bool v321;
                                        v321 = v320 == 0.0f;
                                        bool v322;
                                        v322 = v321 != true;
                                        float v324;
                                        if (v322){
                                            float v323;
                                            v323 = v319 / v320;
                                            v324 = v323;
                                        } else {
                                            v324 = 0.0f;
                                        }
                                        assert("Tensor range check" && 0 <= v313 && v313 < 1l);
                                        assert("Tensor range check" && 0 <= v315 && v315 < 4l);
                                        v312[v318] = v324;
                                        v315 += 1l ;
                                    }
                                    v313 += 1l ;
                                }
                                bool v325[4l];
                                int v326;
                                v326 = 0l;
                                while (while_method_7(v326)){
                                    int v328;
                                    v328 = 0l;
                                    while (while_method_6(v328)){
                                        assert("Tensor range check" && 0 <= v326 && v326 < 1l);
                                        assert("Tensor range check" && 0 <= v328 && v328 < 4l);
                                        int v330;
                                        v330 = 4l * v326;
                                        int v331;
                                        v331 = v330 + v328;
                                        float v332;
                                        v332 = v273[v331];
                                        int v333;
                                        v333 = v276[v331];
                                        bool v334;
                                        v334 = v333 < 3l;
                                        assert("Tensor range check" && 0 <= v326 && v326 < 1l);
                                        assert("Tensor range check" && 0 <= v328 && v328 < 4l);
                                        v325[v331] = v334;
                                        v328 += 1l ;
                                    }
                                    v326 += 1l ;
                                }
                                float v335[4l];
                                int v336;
                                v336 = 0l;
                                while (while_method_7(v336)){
                                    int v338;
                                    v338 = 0l;
                                    while (while_method_6(v338)){
                                        assert("Tensor range check" && 0 <= v336 && v336 < 1l);
                                        assert("Tensor range check" && 0 <= v338 && v338 < 4l);
                                        int v340;
                                        v340 = 4l * v336;
                                        int v341;
                                        v341 = v340 + v338;
                                        float v342;
                                        v342 = v273[v341];
                                        bool v343;
                                        v343 = v325[v341];
                                        float v346;
                                        if (v343){
                                            bool v344;
                                            v344 = 0.0f >= v342;
                                            if (v344){
                                                v346 = 0.0f;
                                            } else {
                                                v346 = v342;
                                            }
                                        } else {
                                            v346 = 0.0f;
                                        }
                                        assert("Tensor range check" && 0 <= v336 && v336 < 1l);
                                        assert("Tensor range check" && 0 <= v338 && v338 < 4l);
                                        v335[v341] = v346;
                                        v338 += 1l ;
                                    }
                                    v336 += 1l ;
                                }
                                float v347;
                                v347 = 0.0f;
                                int v348;
                                v348 = 0l;
                                while (while_method_7(v348)){
                                    int v350;
                                    v350 = 0l;
                                    while (while_method_6(v350)){
                                        assert("Tensor range check" && 0 <= v348 && v348 < 1l);
                                        assert("Tensor range check" && 0 <= v350 && v350 < 4l);
                                        int v352;
                                        v352 = 4l * v348;
                                        int v353;
                                        v353 = v352 + v350;
                                        float v354;
                                        v354 = v335[v353];
                                        float v355;
                                        v355 = v347 + v354;
                                        v347 = v355;
                                        v350 += 1l ;
                                    }
                                    v348 += 1l ;
                                }
                                auto v356 = cooperative_groups::coalesced_threads();
                                int v357;
                                v357 = threadIdx.x;
                                auto v358 = cooperative_groups::labeled_partition(v356,v357);
                                Closure1 v359{};
                                float v360;
                                v360 = cooperative_groups::reduce(v358, v347, v359);
                                int v361[4l];
                                int v362;
                                v362 = 0l;
                                while (while_method_7(v362)){
                                    int v364;
                                    v364 = 0l;
                                    while (while_method_6(v364)){
                                        assert("Tensor range check" && 0 <= v362 && v362 < 1l);
                                        assert("Tensor range check" && 0 <= v364 && v364 < 4l);
                                        int v366;
                                        v366 = 4l * v362;
                                        int v367;
                                        v367 = v366 + v364;
                                        bool v368;
                                        v368 = v325[v367];
                                        int v369;
                                        if (v368){
                                            v369 = 1l;
                                        } else {
                                            v369 = 0l;
                                        }
                                        assert("Tensor range check" && 0 <= v362 && v362 < 1l);
                                        assert("Tensor range check" && 0 <= v364 && v364 < 4l);
                                        v361[v367] = v369;
                                        v364 += 1l ;
                                    }
                                    v362 += 1l ;
                                }
                                int v370;
                                v370 = 0l;
                                int v371;
                                v371 = 0l;
                                while (while_method_7(v371)){
                                    int v373;
                                    v373 = 0l;
                                    while (while_method_6(v373)){
                                        assert("Tensor range check" && 0 <= v371 && v371 < 1l);
                                        assert("Tensor range check" && 0 <= v373 && v373 < 4l);
                                        int v375;
                                        v375 = 4l * v371;
                                        int v376;
                                        v376 = v375 + v373;
                                        int v377;
                                        v377 = v361[v376];
                                        int v378;
                                        v378 = v370 + v377;
                                        v370 = v378;
                                        v373 += 1l ;
                                    }
                                    v371 += 1l ;
                                }
                                auto v379 = cooperative_groups::coalesced_threads();
                                int v380;
                                v380 = threadIdx.x;
                                auto v381 = cooperative_groups::labeled_partition(v379,v380);
                                Closure2 v382{};
                                int v383;
                                v383 = cooperative_groups::reduce(v381, v370, v382);
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
                                        v393 = v335[v392];
                                        bool v394;
                                        v394 = v325[v392];
                                        bool v395;
                                        v395 = v394 == false;
                                        float v400;
                                        if (v395){
                                            v400 = 0.0f;
                                        } else {
                                            bool v396;
                                            v396 = v360 == 0.0f;
                                            bool v397;
                                            v397 = v396 != true;
                                            if (v397){
                                                float v398;
                                                v398 = v393 / v360;
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
                                float v401[4l];
                                int v402;
                                v402 = 0l;
                                while (while_method_7(v402)){
                                    int v404;
                                    v404 = 0l;
                                    while (while_method_6(v404)){
                                        assert("Tensor range check" && 0 <= v402 && v402 < 1l);
                                        assert("Tensor range check" && 0 <= v404 && v404 < 4l);
                                        int v406;
                                        v406 = 4l * v402;
                                        int v407;
                                        v407 = v406 + v404;
                                        float v408;
                                        v408 = v312[v407];
                                        int v409;
                                        v409 = v276[v407];
                                        bool v410;
                                        v410 = v262 == v409;
                                        float v413;
                                        if (v410){
                                            float v411;
                                            v411 = v263 - v408;
                                            float v412;
                                            v412 = v411 / v261;
                                            v413 = v412;
                                        } else {
                                            v413 = 0.0f;
                                        }
                                        float v414;
                                        v414 = v413 + v408;
                                        assert("Tensor range check" && 0 <= v402 && v402 < 1l);
                                        assert("Tensor range check" && 0 <= v404 && v404 < 4l);
                                        v401[v407] = v414;
                                        v404 += 1l ;
                                    }
                                    v402 += 1l ;
                                }
                                float v415[4l];
                                int v416;
                                v416 = 0l;
                                while (while_method_7(v416)){
                                    int v418;
                                    v418 = 0l;
                                    while (while_method_6(v418)){
                                        assert("Tensor range check" && 0 <= v416 && v416 < 1l);
                                        assert("Tensor range check" && 0 <= v418 && v418 < 4l);
                                        int v420;
                                        v420 = 4l * v416;
                                        int v421;
                                        v421 = v420 + v418;
                                        float v422;
                                        v422 = v386[v421];
                                        float v423;
                                        v423 = v401[v421];
                                        float v424;
                                        v424 = v422 * v423;
                                        assert("Tensor range check" && 0 <= v416 && v416 < 1l);
                                        assert("Tensor range check" && 0 <= v418 && v418 < 4l);
                                        v415[v421] = v424;
                                        v418 += 1l ;
                                    }
                                    v416 += 1l ;
                                }
                                float v425;
                                v425 = 0.0f;
                                int v426;
                                v426 = 0l;
                                while (while_method_7(v426)){
                                    int v428;
                                    v428 = 0l;
                                    while (while_method_6(v428)){
                                        assert("Tensor range check" && 0 <= v426 && v426 < 1l);
                                        assert("Tensor range check" && 0 <= v428 && v428 < 4l);
                                        int v430;
                                        v430 = 4l * v426;
                                        int v431;
                                        v431 = v430 + v428;
                                        float v432;
                                        v432 = v415[v431];
                                        float v433;
                                        v433 = v425 + v432;
                                        v425 = v433;
                                        v428 += 1l ;
                                    }
                                    v426 += 1l ;
                                }
                                auto v434 = cooperative_groups::coalesced_threads();
                                int v435;
                                v435 = threadIdx.x;
                                auto v436 = cooperative_groups::labeled_partition(v434,v435);
                                float v437;
                                v437 = cooperative_groups::reduce(v436, v425, v359);
                                int v438;
                                v438 = 0l;
                                while (while_method_7(v438)){
                                    int v440;
                                    v440 = 0l;
                                    while (while_method_6(v440)){
                                        assert("Tensor range check" && 0 <= v438 && v438 < 1l);
                                        assert("Tensor range check" && 0 <= v440 && v440 < 4l);
                                        int v442;
                                        v442 = 4l * v438;
                                        int v443;
                                        v443 = v442 + v440;
                                        float v444;
                                        v444 = v401[v443];
                                        int v445;
                                        v445 = v276[v443];
                                        float v446;
                                        v446 = v444 - v437;
                                        float v447;
                                        v447 = v264 * v446;
                                        assert("Tensor range check" && 0 <= v445 && v445 < 4l);
                                        float * v448;
                                        v448 = v265+v445;
                                        float v450;
                                        v450 = atomicAdd(v448,v447);
                                        v440 += 1l ;
                                    }
                                    v438 += 1l ;
                                }
                                int v451;
                                v451 = 0l;
                                while (while_method_7(v451)){
                                    assert("Tensor range check" && 0 <= v451 && v451 < 1l);
                                    assert("Tensor range check" && 0 <= v451 && v451 < 1l);
                                    v451 += 1l ;
                                }
                                assert("Tensor range check" && 0 <= v258 && v258 < 512l);
                                v237[v258] = v437;
                                v247 += 1l ;
                            }
                            asm("barrier.cta.sync %0;" :: "r"(0l));
                            assert("Tensor range check" && 0 <= v239 && v239 < 512l);
                            float v453;
                            v453 = v237[v239];
                            asm("barrier.cta.sync %0;" :: "r"(0l));
                            assert("Tensor range check" && 0 <= v131 && v131 < 2l);
                            v115[v131] = v453;
                        }
                        assert("Tensor range check" && 0 <= v109 && v109 < 4l);
                        int v454;
                        v454 = 24576l * v109;
                        assert("Tensor range check" && 0 <= v114 && v114 < 12288l);
                        int v455;
                        v455 = 2l * v114;
                        int v456;
                        v456 = v455 + v454;
                        double * v457;
                        v457 = v103+v456;
                        double * v459;
                        v459 = v105+v456;
                        double * v461;
                        v461 = v457+0l;
                        double * v463;
                        v463 = v459+0l;
                        double * v465;
                        v465 = v457+0l;
                        double * v467;
                        v467 = v459+0l;
                        int v469;
                        v469 = sizeof(double *);
                        unsigned long long v470;
                        v470 = (unsigned long long)v469;
                        unsigned long long v471;
                        v471 = 512ull * v470;
                        unsigned long long v472;
                        v472 = v471 + 16ull;
                        unsigned long long v473;
                        v473 = v472 - 1ull;
                        unsigned long long v474;
                        v474 = v473 % 16ull;
                        unsigned long long v475;
                        v475 = v473 - v474;
                        unsigned long long v476;
                        v476 = v475 + v471;
                        unsigned long long v477;
                        v477 = v476 + 16ull;
                        unsigned long long v478;
                        v478 = v477 - 1ull;
                        unsigned long long v479;
                        v479 = v478 % 16ull;
                        unsigned long long v480;
                        v480 = v478 - v479;
                        unsigned long long v481;
                        v481 = v480 + v471;
                        unsigned long long v482;
                        v482 = v481 + 16ull;
                        unsigned long long v483;
                        v483 = v482 - 1ull;
                        unsigned long long v484;
                        v484 = v483 % 16ull;
                        unsigned long long v485;
                        v485 = v483 - v484;
                        unsigned long long v486;
                        v486 = v485 + v471;
                        bool v487;
                        v487 = v486 <= 81920ull;
                        bool v488;
                        v488 = v487 == false;
                        if (v488){
                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v487);
                        } else {
                        }
                        extern __shared__ unsigned char v490[];
                        bool v491;
                        v491 = v486 <= v486;
                        bool v492;
                        v492 = v491 == false;
                        if (v492){
                            assert("The length of the partition has to be less than or equal to the length of the base array." && v491);
                        } else {
                        }
                        double * * v494;
                        v494 = reinterpret_cast<double * *>(&v490[0ull]);
                        double * * v496;
                        v496 = reinterpret_cast<double * *>(&v490[v475]);
                        double * * v498;
                        v498 = reinterpret_cast<double * *>(&v490[v480]);
                        double * * v500;
                        v500 = reinterpret_cast<double * *>(&v490[v485]);
                        int v502;
                        v502 = threadIdx.x;
                        assert("Tensor range check" && 0 <= v502 && v502 < 512l);
                        v494[v502] = v461;
                        v496[v502] = v463;
                        v498[v502] = v465;
                        v500[v502] = v467;
                        asm("barrier.cta.sync %0;" :: "r"(0l));
                        bool v503;
                        v503 = 0l <= v502;
                        bool v504;
                        v504 = v503 == false;
                        if (v504){
                            assert("The index needs to be zero or positive." && v503);
                        } else {
                        }
                        int v506;
                        v506 = v502 % 1l;
                        bool v507;
                        v507 = v502 < 512l;
                        bool v508;
                        v508 = v507 == false;
                        if (v508){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v507);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v502 && v502 < 512l);
                        int v510;
                        v510 = 0l;
                        while (while_method_7(v510)){
                            bool v512;
                            v512 = v503 && v507;
                            bool v513;
                            v513 = v512 == false;
                            if (v513){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v512);
                            } else {
                            }
                            bool v515;
                            v515 = 0l <= v510;
                            bool v517;
                            if (v515){
                                bool v516;
                                v516 = v510 < 1l;
                                v517 = v516;
                            } else {
                                v517 = false;
                            }
                            bool v518;
                            v518 = v517 == false;
                            if (v518){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v517);
                            } else {
                            }
                            int v520;
                            v520 = v510 * 512l;
                            int v521;
                            v521 = v520 + v502;
                            assert("Tensor range check" && 0 <= v510 && v510 < 1l);
                            int v522;
                            v522 = 512l * v510;
                            int v523;
                            v523 = v522 + v502;
                            double * v524;
                            v524 = v494[v523];
                            double * v525;
                            v525 = v496[v523];
                            double * v526;
                            v526 = v498[v523];
                            double * v527;
                            v527 = v500[v523];
                            int v528;
                            v528 = blockIdx.x;
                            int v529;
                            v529 = v528 * 512l;
                            int v530;
                            v530 = v529 + v521;
                            assert("Tensor range check" && 0 <= v506 && v506 < 1l);
                            int v531;
                            v531 = 2l * v506;
                            double v532[2l];
                            double v533[2l];
                            int v534[2l];
                            int v535;
                            v535 = 0l;
                            while (while_method_7(v535)){
                                assert("Tensor range check" && 0 <= v535 && v535 < 1l);
                                int v537;
                                v537 = 2l * v535;
                                assert("Tensor range check" && 0 <= v535 && v535 < 1l);
                                int v538;
                                v538 = v537 + v531;
                                int4* v539;
                                v539 = reinterpret_cast<int4*>(v524 + v538);
                                int4* v540;
                                v540 = reinterpret_cast<int4*>(v532 + v537);
                                assert("Pointer alignment check" && (unsigned long long)(v539) % 2l == 0 && (unsigned long long)(v540) % 2l == 0);
                                *v540 = *v539;
                                int4* v541;
                                v541 = reinterpret_cast<int4*>(v525 + v538);
                                int4* v542;
                                v542 = reinterpret_cast<int4*>(v533 + v537);
                                assert("Pointer alignment check" && (unsigned long long)(v541) % 2l == 0 && (unsigned long long)(v542) % 2l == 0);
                                *v542 = *v541;
                                v535 += 1l ;
                            }
                            int v543;
                            v543 = 0l;
                            while (while_method_7(v543)){
                                int v545;
                                v545 = 0l;
                                while (while_method_3(v545)){
                                    bool v547;
                                    v547 = 0l <= v545;
                                    bool v549;
                                    if (v547){
                                        bool v548;
                                        v548 = v545 < 2l;
                                        v549 = v548;
                                    } else {
                                        v549 = false;
                                    }
                                    bool v550;
                                    v550 = v549 == false;
                                    if (v550){
                                        assert("The indices should be inside the range of the dimension." && v549);
                                    } else {
                                    }
                                    bool v552;
                                    v552 = 0l <= v506;
                                    bool v554;
                                    if (v552){
                                        bool v553;
                                        v553 = v506 < 1l;
                                        v554 = v553;
                                    } else {
                                        v554 = false;
                                    }
                                    bool v555;
                                    v555 = v554 == false;
                                    if (v555){
                                        assert("The indices should be inside the range of the dimension." && v554);
                                    } else {
                                    }
                                    int v557;
                                    v557 = v506 * 2l;
                                    int v558;
                                    v558 = v545 + v557;
                                    bool v559;
                                    v559 = 0l <= v543;
                                    bool v561;
                                    if (v559){
                                        bool v560;
                                        v560 = v543 < 1l;
                                        v561 = v560;
                                    } else {
                                        v561 = false;
                                    }
                                    bool v562;
                                    v562 = v561 == false;
                                    if (v562){
                                        assert("The indices should be inside the range of the dimension." && v561);
                                    } else {
                                    }
                                    int v564;
                                    v564 = v543 * 2l;
                                    int v565;
                                    v565 = v558 + v564;
                                    assert("Tensor range check" && 0 <= v543 && v543 < 1l);
                                    assert("Tensor range check" && 0 <= v545 && v545 < 2l);
                                    int v566;
                                    v566 = 2l * v543;
                                    int v567;
                                    v567 = v566 + v545;
                                    v534[v567] = v565;
                                    v545 += 1l ;
                                }
                                v543 += 1l ;
                            }
                            double v568[2l];
                            double v569[2l];
                            int v570;
                            v570 = 0l;
                            while (while_method_7(v570)){
                                int v572;
                                v572 = 0l;
                                while (while_method_3(v572)){
                                    assert("Tensor range check" && 0 <= v570 && v570 < 1l);
                                    assert("Tensor range check" && 0 <= v572 && v572 < 2l);
                                    int v574;
                                    v574 = 2l * v570;
                                    int v575;
                                    v575 = v574 + v572;
                                    double v576;
                                    v576 = v532[v575];
                                    double v577;
                                    v577 = v533[v575];
                                    assert("Tensor range check" && 0 <= v570 && v570 < 1l);
                                    assert("Tensor range check" && 0 <= v572 && v572 < 2l);
                                    v568[v575] = 0.0;
                                    v569[v575] = 0.0;
                                    v572 += 1l ;
                                }
                                v570 += 1l ;
                            }
                            int v578;
                            v578 = 0l;
                            while (while_method_7(v578)){
                                assert("Tensor range check" && 0 <= v578 && v578 < 1l);
                                int v580;
                                v580 = 2l * v578;
                                int v581;
                                v581 = v580 + v531;
                                assert("Tensor range check" && 0 <= v578 && v578 < 1l);
                                int4* v582;
                                v582 = reinterpret_cast<int4*>(v568 + v580);
                                int4* v583;
                                v583 = reinterpret_cast<int4*>(v526 + v581);
                                assert("Pointer alignment check" && (unsigned long long)(v582) % 2l == 0 && (unsigned long long)(v583) % 2l == 0);
                                *v583 = *v582;
                                int4* v584;
                                v584 = reinterpret_cast<int4*>(v569 + v580);
                                int4* v585;
                                v585 = reinterpret_cast<int4*>(v527 + v581);
                                assert("Pointer alignment check" && (unsigned long long)(v584) % 2l == 0 && (unsigned long long)(v585) % 2l == 0);
                                *v585 = *v584;
                                v578 += 1l ;
                            }
                            assert("Tensor range check" && 0 <= v521 && v521 < 512l);
                            v510 += 1l ;
                        }
                        asm("barrier.cta.sync %0;" :: "r"(0l));
                        assert("Tensor range check" && 0 <= v502 && v502 < 512l);
                        asm("barrier.cta.sync %0;" :: "r"(0l));
                        assert("Tensor range check" && 0 <= v109 && v109 < 4l);
                        assert("Tensor range check" && 0 <= v114 && v114 < 12288l);
                        v107[v121] = 0l;
                        v109 += 1l ;
                    }
                    v31 += 1l ;
                }
                v29 += 1l ;
            }
            int v586;
            v586 = threadIdx.x;
            int v587;
            v587 = blockIdx.x;
            int v588;
            v588 = v587 * 512l;
            int v589;
            v589 = v586 + v588;
            int v590;
            v590 = v589;
            while (while_method_11(v590)){
                bool v592;
                v592 = 0l <= v590;
                bool v593;
                v593 = v592 == false;
                if (v593){
                    assert("The index needs to be zero or positive." && v592);
                } else {
                }
                int v595;
                v595 = v590 % 4l;
                int v596;
                v596 = v590 / 4l;
                int v597;
                v597 = v596 % 1024l;
                int v598;
                v598 = v596 / 1024l;
                bool v599;
                v599 = v598 < 4l;
                bool v600;
                v600 = v599 == false;
                if (v600){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v599);
                } else {
                }
                assert("Tensor range check" && 0 <= v598 && v598 < 4l);
                assert("Tensor range check" && 0 <= v597 && v597 < 1024l);
                assert("Tensor range check" && 0 <= v595 && v595 < 4l);
                int v602;
                v602 = 4l * v595;
                int v603;
                v603 = 16l * v597;
                int v604;
                v604 = v603 + v602;
                int v605;
                v605 = 16384l * v598;
                int v606;
                v606 = v605 + v604;
                assert("Tensor range check" && 0 <= v598 && v598 < 4l);
                assert("Tensor range check" && 0 <= v597 && v597 < 1024l);
                assert("Tensor range check" && 0 <= v595 && v595 < 4l);
                float v607[4l];
                float v608[4l];
                float v609[4l];
                int4* v610;
                v610 = reinterpret_cast<int4*>(v5 + v606);
                int4* v611;
                v611 = reinterpret_cast<int4*>(v607 + 0l);
                assert("Pointer alignment check" && (unsigned long long)(v610) % 4l == 0 && (unsigned long long)(v611) % 4l == 0);
                *v611 = *v610;
                int4* v612;
                v612 = reinterpret_cast<int4*>(v6 + v606);
                int4* v613;
                v613 = reinterpret_cast<int4*>(v608 + 0l);
                assert("Pointer alignment check" && (unsigned long long)(v612) % 4l == 0 && (unsigned long long)(v613) % 4l == 0);
                *v613 = *v612;
                // Pushing the loop unrolling to: 0
                int v614;
                v614 = 0l;
                #pragma unroll
                while (while_method_6(v614)){
                    assert("Tensor range check" && 0 <= v614 && v614 < 4l);
                    float v616;
                    v616 = v607[v614];
                    float v617;
                    v617 = v608[v614];
                    bool v618;
                    v618 = v617 == 0.0f;
                    bool v619;
                    v619 = v618 != true;
                    float v621;
                    if (v619){
                        float v620;
                        v620 = v616 / v617;
                        v621 = v620;
                    } else {
                        v621 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v614 && v614 < 4l);
                    v609[v614] = v621;
                    v614 += 1l ;
                }
                // Poping the loop unrolling to: 0
                int4* v622;
                v622 = reinterpret_cast<int4*>(v609 + 0l);
                int4* v623;
                v623 = reinterpret_cast<int4*>(v7 + v606);
                assert("Pointer alignment check" && (unsigned long long)(v622) % 4l == 0 && (unsigned long long)(v623) % 4l == 0);
                *v623 = *v622;
                v590 += 12288l ;
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
        v11 = "Going to run the Leduc training kernel."
        print(v10.format(v11),end="")
        del v10, v11
        v12 = time.perf_counter()
        v14 = static_array(2)
        v16 = US1_0()
        v14[0] = v16
        del v16
        v18 = US1_1()
        v14[1] = v18
        del v14, v18
        v19 = cp.zeros(65536,dtype=cp.float32) # type: ignore
        v20 = cp.zeros(65536,dtype=cp.float32) # type: ignore
        v21 = cp.empty(65536,dtype=cp.float32)
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
                assert 0 <= v44 < 16384, 'Tensor range check'
                v46 = 16384 * v41
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
        v49 = [None] * 1
        v50 = US2_0(v37)
        del v37
        v49[0] = v50
        del v50
        return method18(v4, v5, v6, v7, v49)
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
    v1 = v0 < 16384
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
