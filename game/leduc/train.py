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
    v1 = v0 < 2l;
    return v1;
}
__device__ inline bool while_method_1(Union7 v0){
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
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 4096l;
    return v1;
}
__device__ inline bool while_method_3(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
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
__device__ void method_6(float * v0, int v1, float * v2, int v3, float * v4, int v5){
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
__device__ void method_7(unsigned int * v0, int v1, float * v2){
    int v3;
    v3 = blockIdx.x;
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v4;
    v4 = 4096l * v3;
    int v5;
    v5 = blockIdx.x;
    assert("Tensor range check" && 0 <= v5 && v5 < 1l);
    int v6;
    v6 = 32l * v5;
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
    v14 = v13 < 1l;
    bool v15;
    v15 = v14 == false;
    if (v15){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v14);
    } else {
    }
    assert("Tensor range check" && 0 <= v13 && v13 < 1l);
    assert("Tensor range check" && 0 <= v12 && v12 < 32l);
    int v17;
    v17 = 4l * v12;
    int v18;
    v18 = v17 + v4;
    int v19;
    v19 = 128l * v13;
    int v20;
    v20 = v19 + v18;
    assert("Tensor range check" && 0 <= v13 && v13 < 1l);
    int v21;
    v21 = v13 + v7;
    int v22;
    v22 = 0l;
    while (while_method_8(v22)){
        assert("Tensor range check" && 0 <= v22 && v22 < 32l);
        int v24;
        v24 = 128l * v22;
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
            while (while_method_4(v37)){
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
        v69 = v22 + v13;
        unsigned int v70[4l];
        int v71;
        v71 = 0l;
        while (while_method_6(v71)){
            int v73;
            v73 = 0l;
            while (while_method_4(v73)){
                assert("Tensor range check" && 0 <= v71 && v71 < 1l);
                assert("Tensor range check" && 0 <= v73 && v73 < 4l);
                int v75;
                v75 = 4l * v71;
                int v76;
                v76 = v75 + v73;
                float v77;
                v77 = v26[v76];
                int v78;
                v78 = v27[v76];
                bool v79;
                v79 = v77 <= 0.0f;
                unsigned int v81;
                if (v79){
                    v81 = 0ul;
                } else {
                    unsigned int v80;
                    v80 = 1ul << v78;
                    v81 = v80;
                }
                assert("Tensor range check" && 0 <= v71 && v71 < 1l);
                assert("Tensor range check" && 0 <= v73 && v73 < 4l);
                v70[v76] = v81;
                v73 += 1l ;
            }
            v71 += 1l ;
        }
        unsigned int v82;
        v82 = 0ul;
        int v83;
        v83 = 0l;
        while (while_method_6(v83)){
            int v85;
            v85 = 0l;
            while (while_method_4(v85)){
                assert("Tensor range check" && 0 <= v83 && v83 < 1l);
                assert("Tensor range check" && 0 <= v85 && v85 < 4l);
                int v87;
                v87 = 4l * v83;
                int v88;
                v88 = v87 + v85;
                unsigned int v89;
                v89 = v70[v88];
                unsigned int v90;
                v90 = v82 | v89;
                v82 = v90;
                v85 += 1l ;
            }
            v83 += 1l ;
        }
        auto v91 = cooperative_groups::coalesced_threads();
        int v92;
        v92 = threadIdx.x;
        int v93;
        v93 = v92 / 32l;
        auto v94 = cooperative_groups::labeled_partition(v91,v93);
        Closure0 v95{};
        unsigned int v96;
        v96 = cooperative_groups::reduce(v94, v82, v95);
        unsigned int v97;
        v97 = v96 % 4096ul;
        assert("Tensor range check" && 0 <= v22 && v22 < 32l);
        int v98;
        v98 = v22 + v21;
        v0[v98] = v97;
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
    __shared__ float * v18[32l];
    __shared__ float * v19[32l];
    /* void shared array create v20 */;
    __shared__ float v21[32l];
    __shared__ int v22[32l];
    int v23;
    v23 = threadIdx.x;
    assert("Tensor range check" && 0 <= v23 && v23 < 32l);
    v18[v23] = v14;
    v19[v23] = v16;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v24;
    v24 = 0l <= v23;
    bool v25;
    v25 = v24 == false;
    if (v25){
        assert("The index needs to be zero or positive." && v24);
    } else {
    }
    int v27;
    v27 = v23 % 1l;
    bool v28;
    v28 = v23 < 32l;
    bool v29;
    v29 = v28 == false;
    if (v29){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v28);
    } else {
    }
    assert("Tensor range check" && 0 <= v23 && v23 < 32l);
    int v31;
    v31 = 0l;
    while (while_method_6(v31)){
        bool v33;
        v33 = v24 && v28;
        bool v34;
        v34 = v33 == false;
        if (v34){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v33);
        } else {
        }
        bool v36;
        v36 = 0l <= v31;
        bool v38;
        if (v36){
            bool v37;
            v37 = v31 < 1l;
            v38 = v37;
        } else {
            v38 = false;
        }
        bool v39;
        v39 = v38 == false;
        if (v39){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v38);
        } else {
        }
        int v41;
        v41 = v31 * 32l;
        int v42;
        v42 = v41 + v23;
        assert("Tensor range check" && 0 <= v31 && v31 < 1l);
        int v43;
        v43 = 32l * v31;
        int v44;
        v44 = v43 + v23;
        float * v45;
        v45 = v18[v44];
        float * v46;
        v46 = v19[v44];
        /* void array index */;
        int v47;
        v47 = blockIdx.x;
        int v48;
        v48 = v47 * 32l;
        int v49;
        v49 = v48 + v42;
        assert("Tensor range check" && 0 <= v27 && v27 < 1l);
        int v50;
        v50 = 4l * v27;
        float v51[4l];
        float v52[4l];
        int v53[4l];
        int v54;
        v54 = 0l;
        while (while_method_6(v54)){
            assert("Tensor range check" && 0 <= v54 && v54 < 1l);
            int v56;
            v56 = 4l * v54;
            assert("Tensor range check" && 0 <= v54 && v54 < 1l);
            int v57;
            v57 = v56 + v50;
            int4* v58;
            v58 = reinterpret_cast<int4*>(v45 + v57);
            int4* v59;
            v59 = reinterpret_cast<int4*>(v51 + v56);
            assert("Pointer alignment check" && (unsigned long long)(v58) % 4l == 0 && (unsigned long long)(v59) % 4l == 0);
            *v59 = *v58;
            int4* v60;
            v60 = reinterpret_cast<int4*>(v46 + v57);
            int4* v61;
            v61 = reinterpret_cast<int4*>(v52 + v56);
            assert("Pointer alignment check" && (unsigned long long)(v60) % 4l == 0 && (unsigned long long)(v61) % 4l == 0);
            *v61 = *v60;
            v54 += 1l ;
        }
        int v62;
        v62 = 0l;
        while (while_method_6(v62)){
            int v64;
            v64 = 0l;
            while (while_method_4(v64)){
                bool v66;
                v66 = 0l <= v64;
                bool v68;
                if (v66){
                    bool v67;
                    v67 = v64 < 4l;
                    v68 = v67;
                } else {
                    v68 = false;
                }
                bool v69;
                v69 = v68 == false;
                if (v69){
                    assert("The indices should be inside the range of the dimension." && v68);
                } else {
                }
                bool v71;
                v71 = 0l <= v27;
                bool v73;
                if (v71){
                    bool v72;
                    v72 = v27 < 1l;
                    v73 = v72;
                } else {
                    v73 = false;
                }
                bool v74;
                v74 = v73 == false;
                if (v74){
                    assert("The indices should be inside the range of the dimension." && v73);
                } else {
                }
                int v76;
                v76 = v27 * 4l;
                int v77;
                v77 = v64 + v76;
                bool v78;
                v78 = 0l <= v62;
                bool v80;
                if (v78){
                    bool v79;
                    v79 = v62 < 1l;
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
                int v83;
                v83 = v62 * 4l;
                int v84;
                v84 = v77 + v83;
                assert("Tensor range check" && 0 <= v62 && v62 < 1l);
                assert("Tensor range check" && 0 <= v64 && v64 < 4l);
                int v85;
                v85 = 4l * v62;
                int v86;
                v86 = v85 + v64;
                v53[v86] = v84;
                v64 += 1l ;
            }
            v62 += 1l ;
        }
        bool v87[4l];
        int v88;
        v88 = 0l;
        while (while_method_6(v88)){
            int v90;
            v90 = 0l;
            while (while_method_4(v90)){
                assert("Tensor range check" && 0 <= v88 && v88 < 1l);
                assert("Tensor range check" && 0 <= v90 && v90 < 4l);
                int v92;
                v92 = 4l * v88;
                int v93;
                v93 = v92 + v90;
                float v94;
                v94 = v51[v93];
                int v95;
                v95 = v53[v93];
                bool v96;
                v96 = v95 < 3l;
                assert("Tensor range check" && 0 <= v88 && v88 < 1l);
                assert("Tensor range check" && 0 <= v90 && v90 < 4l);
                v87[v93] = v96;
                v90 += 1l ;
            }
            v88 += 1l ;
        }
        float v97[4l];
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
                float v104;
                v104 = v51[v103];
                bool v105;
                v105 = v87[v103];
                float v108;
                if (v105){
                    bool v106;
                    v106 = 0.0f >= v104;
                    if (v106){
                        v108 = 0.0f;
                    } else {
                        v108 = v104;
                    }
                } else {
                    v108 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v98 && v98 < 1l);
                assert("Tensor range check" && 0 <= v100 && v100 < 4l);
                v97[v103] = v108;
                v100 += 1l ;
            }
            v98 += 1l ;
        }
        float v109;
        v109 = 0.0f;
        int v110;
        v110 = 0l;
        while (while_method_6(v110)){
            int v112;
            v112 = 0l;
            while (while_method_4(v112)){
                assert("Tensor range check" && 0 <= v110 && v110 < 1l);
                assert("Tensor range check" && 0 <= v112 && v112 < 4l);
                int v114;
                v114 = 4l * v110;
                int v115;
                v115 = v114 + v112;
                float v116;
                v116 = v97[v115];
                float v117;
                v117 = v109 + v116;
                v109 = v117;
                v112 += 1l ;
            }
            v110 += 1l ;
        }
        auto v118 = cooperative_groups::coalesced_threads();
        int v119;
        v119 = threadIdx.x;
        auto v120 = cooperative_groups::labeled_partition(v118,v119);
        Closure1 v121{};
        float v122;
        v122 = cooperative_groups::reduce(v120, v109, v121);
        int v123[4l];
        int v124;
        v124 = 0l;
        while (while_method_6(v124)){
            int v126;
            v126 = 0l;
            while (while_method_4(v126)){
                assert("Tensor range check" && 0 <= v124 && v124 < 1l);
                assert("Tensor range check" && 0 <= v126 && v126 < 4l);
                int v128;
                v128 = 4l * v124;
                int v129;
                v129 = v128 + v126;
                bool v130;
                v130 = v87[v129];
                int v131;
                if (v130){
                    v131 = 1l;
                } else {
                    v131 = 0l;
                }
                assert("Tensor range check" && 0 <= v124 && v124 < 1l);
                assert("Tensor range check" && 0 <= v126 && v126 < 4l);
                v123[v129] = v131;
                v126 += 1l ;
            }
            v124 += 1l ;
        }
        int v132;
        v132 = 0l;
        int v133;
        v133 = 0l;
        while (while_method_6(v133)){
            int v135;
            v135 = 0l;
            while (while_method_4(v135)){
                assert("Tensor range check" && 0 <= v133 && v133 < 1l);
                assert("Tensor range check" && 0 <= v135 && v135 < 4l);
                int v137;
                v137 = 4l * v133;
                int v138;
                v138 = v137 + v135;
                int v139;
                v139 = v123[v138];
                int v140;
                v140 = v132 + v139;
                v132 = v140;
                v135 += 1l ;
            }
            v133 += 1l ;
        }
        auto v141 = cooperative_groups::coalesced_threads();
        int v142;
        v142 = threadIdx.x;
        auto v143 = cooperative_groups::labeled_partition(v141,v142);
        Closure2 v144{};
        int v145;
        v145 = cooperative_groups::reduce(v143, v132, v144);
        float v146;
        v146 = (float)v145;
        float v147;
        v147 = 1.0f / v146;
        float v148[4l];
        int v149;
        v149 = 0l;
        while (while_method_6(v149)){
            int v151;
            v151 = 0l;
            while (while_method_4(v151)){
                assert("Tensor range check" && 0 <= v149 && v149 < 1l);
                assert("Tensor range check" && 0 <= v151 && v151 < 4l);
                int v153;
                v153 = 4l * v149;
                int v154;
                v154 = v153 + v151;
                float v155;
                v155 = v97[v154];
                bool v156;
                v156 = v87[v154];
                bool v157;
                v157 = v156 == false;
                float v162;
                if (v157){
                    v162 = 0.0f;
                } else {
                    bool v158;
                    v158 = v122 == 0.0f;
                    bool v159;
                    v159 = v158 != true;
                    if (v159){
                        float v160;
                        v160 = v155 / v122;
                        v162 = v160;
                    } else {
                        v162 = v147;
                    }
                }
                assert("Tensor range check" && 0 <= v149 && v149 < 1l);
                assert("Tensor range check" && 0 <= v151 && v151 < 4l);
                v148[v154] = v162;
                v151 += 1l ;
            }
            v149 += 1l ;
        }
        float v163[4l];
        float v164;
        v164 = 0.0f;
        int v165;
        v165 = 0l;
        while (while_method_6(v165)){
            assert("Tensor range check" && 0 <= v165 && v165 < 1l);
            int v167;
            v167 = 4l * v165;
            assert("Tensor range check" && 0 <= v165 && v165 < 1l);
            int v168; float v169;
            Tuple3 tmp4 = Tuple3{0l, 0.0f};
            v168 = tmp4.v0; v169 = tmp4.v1;
            while (while_method_4(v168)){
                assert("Tensor range check" && 0 <= v168 && v168 < 4l);
                int v171;
                v171 = v168 + v167;
                float v172;
                v172 = v148[v171];
                float v173;
                v173 = v169 + v172;
                v169 = v173;
                v168 += 1l ;
            }
            auto v174 = cooperative_groups::coalesced_threads();
            int v175;
            v175 = threadIdx.x;
            auto v176 = cooperative_groups::labeled_partition(v174,v175);
            Closure3 v177{};
            float v178;
            v178 = cooperative_groups::inclusive_scan(v176, v169, v177);
            float v179;
            v179 = v176.shfl_up(v178,1);
            bool v180;
            v180 = v176.thread_rank() == 0;
            float v181;
            if (v180){
                v181 = 0.0f;
            } else {
                v181 = v179;
            }
            float v182;
            v182 = v176.shfl(v178,v176.num_threads()-1);
            float v183;
            v183 = v164 + v181;
            int v184; float v185;
            Tuple3 tmp5 = Tuple3{0l, v183};
            v184 = tmp5.v0; v185 = tmp5.v1;
            while (while_method_4(v184)){
                assert("Tensor range check" && 0 <= v184 && v184 < 4l);
                int v187;
                v187 = v184 + v167;
                float v188;
                v188 = v148[v187];
                float v189;
                v189 = v185 + v188;
                assert("Tensor range check" && 0 <= v184 && v184 < 4l);
                v163[v187] = v189;
                v185 = v189;
                v184 += 1l ;
            }
            float v190;
            v190 = v164 + v182;
            v164 = v190;
            v165 += 1l ;
        }
        float v191[4l];
        bool v192[4l];
        int v193;
        v193 = 0l;
        while (while_method_6(v193)){
            int v195;
            v195 = 0l;
            while (while_method_4(v195)){
                assert("Tensor range check" && 0 <= v193 && v193 < 1l);
                assert("Tensor range check" && 0 <= v195 && v195 < 4l);
                int v197;
                v197 = 4l * v193;
                int v198;
                v198 = v197 + v195;
                float v199;
                v199 = v163[v198];
                float v200;
                v200 = v148[v198];
                bool v201;
                v201 = v200 > 0.0f;
                assert("Tensor range check" && 0 <= v193 && v193 < 1l);
                assert("Tensor range check" && 0 <= v195 && v195 < 4l);
                v191[v198] = v199;
                v192[v198] = v201;
                v195 += 1l ;
            }
            v193 += 1l ;
        }
        float v202; bool v203;
        Tuple4 tmp6 = Tuple4{-1.0f / 0.0f, false};
        v202 = tmp6.v0; v203 = tmp6.v1;
        int v204;
        v204 = 0l;
        while (while_method_6(v204)){
            int v206;
            v206 = 0l;
            while (while_method_4(v206)){
                assert("Tensor range check" && 0 <= v204 && v204 < 1l);
                assert("Tensor range check" && 0 <= v206 && v206 < 4l);
                int v208;
                v208 = 4l * v204;
                int v209;
                v209 = v208 + v206;
                float v210;
                v210 = v191[v209];
                bool v211;
                v211 = v192[v209];
                float v218; bool v219;
                if (v203){
                    if (v211){
                        bool v212;
                        v212 = v202 >= v210;
                        float v213;
                        if (v212){
                            v213 = v202;
                        } else {
                            v213 = v210;
                        }
                        v218 = v213; v219 = true;
                    } else {
                        v218 = v202; v219 = v203;
                    }
                } else {
                    if (v211){
                        v218 = v210; v219 = v211;
                    } else {
                        v218 = v202; v219 = v203;
                    }
                }
                v202 = v218;
                v203 = v219;
                v206 += 1l ;
            }
            v204 += 1l ;
        }
        auto v220 = cooperative_groups::coalesced_threads();
        int v221;
        v221 = threadIdx.x;
        auto v222 = cooperative_groups::labeled_partition(v220,v221);
        Closure4 v223{};
        float v224; bool v225;
        Tuple4 tmp7 = cooperative_groups::reduce(v222, Tuple4{v202, v203}, v223);
        v224 = tmp7.v0; v225 = tmp7.v1;
        bool v226;
        v226 = v225 == false;
        if (v226){
            assert("The local reduce must be true." && v225);
        } else {
        }
        float v228[4l];
        int v229[4l];
        int v230;
        v230 = 0l;
        while (while_method_6(v230)){
            int v232;
            v232 = 0l;
            while (while_method_4(v232)){
                assert("Tensor range check" && 0 <= v230 && v230 < 1l);
                assert("Tensor range check" && 0 <= v232 && v232 < 4l);
                int v234;
                v234 = 4l * v230;
                int v235;
                v235 = v234 + v232;
                int v236;
                v236 = v53[v235];
                float v237;
                v237 = curand_uniform(&v0);
                assert("Tensor range check" && 0 <= v230 && v230 < 1l);
                assert("Tensor range check" && 0 <= v232 && v232 < 4l);
                v228[v235] = v237;
                v229[v235] = v236;
                v232 += 1l ;
            }
            v230 += 1l ;
        }
        float v238; int v239;
        Tuple2 tmp8 = Tuple2{0.0f, 2147483647l};
        v238 = tmp8.v0; v239 = tmp8.v1;
        int v240;
        v240 = 0l;
        while (while_method_6(v240)){
            int v242;
            v242 = 0l;
            while (while_method_4(v242)){
                assert("Tensor range check" && 0 <= v240 && v240 < 1l);
                assert("Tensor range check" && 0 <= v242 && v242 < 4l);
                int v244;
                v244 = 4l * v240;
                int v245;
                v245 = v244 + v242;
                float v246;
                v246 = v228[v245];
                int v247;
                v247 = v229[v245];
                bool v248;
                v248 = v239 < v247;
                float v249; int v250;
                if (v248){
                    v249 = v238; v250 = v239;
                } else {
                    v249 = v246; v250 = v247;
                }
                v238 = v249;
                v239 = v250;
                v242 += 1l ;
            }
            v240 += 1l ;
        }
        auto v251 = cooperative_groups::coalesced_threads();
        int v252;
        v252 = threadIdx.x;
        auto v253 = cooperative_groups::labeled_partition(v251,v252);
        Closure5 v254{};
        float v255; int v256;
        Tuple2 tmp9 = cooperative_groups::reduce(v253, Tuple2{v238, v239}, v254);
        v255 = tmp9.v0; v256 = tmp9.v1;
        float v257;
        v257 = v224 * v255;
        int v258[4l];
        bool v259[4l];
        int v260;
        v260 = 0l;
        while (while_method_6(v260)){
            int v262;
            v262 = 0l;
            while (while_method_4(v262)){
                assert("Tensor range check" && 0 <= v260 && v260 < 1l);
                assert("Tensor range check" && 0 <= v262 && v262 < 4l);
                int v264;
                v264 = 4l * v260;
                int v265;
                v265 = v264 + v262;
                float v266;
                v266 = v191[v265];
                bool v267;
                v267 = v192[v265];
                int v268;
                v268 = v53[v265];
                int v271; bool v272;
                if (v267){
                    float v269;
                    v269 = v266 - v257;
                    bool v270;
                    v270 = v269 >= 0.0f;
                    v271 = v268; v272 = v270;
                } else {
                    v271 = 2147483647l; v272 = false;
                }
                assert("Tensor range check" && 0 <= v260 && v260 < 1l);
                assert("Tensor range check" && 0 <= v262 && v262 < 4l);
                v258[v265] = v271;
                v259[v265] = v272;
                v262 += 1l ;
            }
            v260 += 1l ;
        }
        int v273; bool v274;
        Tuple5 tmp10 = Tuple5{2147483647l, false};
        v273 = tmp10.v0; v274 = tmp10.v1;
        int v275;
        v275 = 0l;
        while (while_method_6(v275)){
            int v277;
            v277 = 0l;
            while (while_method_4(v277)){
                assert("Tensor range check" && 0 <= v275 && v275 < 1l);
                assert("Tensor range check" && 0 <= v277 && v277 < 4l);
                int v279;
                v279 = 4l * v275;
                int v280;
                v280 = v279 + v277;
                int v281;
                v281 = v258[v280];
                bool v282;
                v282 = v259[v280];
                int v289; bool v290;
                if (v274){
                    if (v282){
                        bool v283;
                        v283 = v273 < v281;
                        int v284;
                        if (v283){
                            v284 = v273;
                        } else {
                            v284 = v281;
                        }
                        v289 = v284; v290 = true;
                    } else {
                        v289 = v273; v290 = v274;
                    }
                } else {
                    if (v282){
                        v289 = v281; v290 = v282;
                    } else {
                        v289 = v273; v290 = v274;
                    }
                }
                v273 = v289;
                v274 = v290;
                v277 += 1l ;
            }
            v275 += 1l ;
        }
        auto v291 = cooperative_groups::coalesced_threads();
        int v292;
        v292 = threadIdx.x;
        auto v293 = cooperative_groups::labeled_partition(v291,v292);
        Closure6 v294{};
        int v295; bool v296;
        Tuple5 tmp11 = cooperative_groups::reduce(v293, Tuple5{v273, v274}, v294);
        v295 = tmp11.v0; v296 = tmp11.v1;
        bool v297;
        v297 = v296 == false;
        if (v297){
            assert("The local reduce must be true." && v296);
        } else {
        }
        bool v299[4l];
        int v300;
        v300 = 0l;
        while (while_method_6(v300)){
            int v302;
            v302 = 0l;
            while (while_method_4(v302)){
                assert("Tensor range check" && 0 <= v300 && v300 < 1l);
                assert("Tensor range check" && 0 <= v302 && v302 < 4l);
                int v304;
                v304 = 4l * v300;
                int v305;
                v305 = v304 + v302;
                float v306;
                v306 = v52[v305];
                int v307;
                v307 = v53[v305];
                bool v308;
                v308 = v307 < 3l;
                assert("Tensor range check" && 0 <= v300 && v300 < 1l);
                assert("Tensor range check" && 0 <= v302 && v302 < 4l);
                v299[v305] = v308;
                v302 += 1l ;
            }
            v300 += 1l ;
        }
        float v309[4l];
        int v310;
        v310 = 0l;
        while (while_method_6(v310)){
            int v312;
            v312 = 0l;
            while (while_method_4(v312)){
                assert("Tensor range check" && 0 <= v310 && v310 < 1l);
                assert("Tensor range check" && 0 <= v312 && v312 < 4l);
                int v314;
                v314 = 4l * v310;
                int v315;
                v315 = v314 + v312;
                float v316;
                v316 = v52[v315];
                bool v317;
                v317 = v299[v315];
                float v320;
                if (v317){
                    bool v318;
                    v318 = 0.0f >= v316;
                    if (v318){
                        v320 = 0.0f;
                    } else {
                        v320 = v316;
                    }
                } else {
                    v320 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v310 && v310 < 1l);
                assert("Tensor range check" && 0 <= v312 && v312 < 4l);
                v309[v315] = v320;
                v312 += 1l ;
            }
            v310 += 1l ;
        }
        float v321;
        v321 = 0.0f;
        int v322;
        v322 = 0l;
        while (while_method_6(v322)){
            int v324;
            v324 = 0l;
            while (while_method_4(v324)){
                assert("Tensor range check" && 0 <= v322 && v322 < 1l);
                assert("Tensor range check" && 0 <= v324 && v324 < 4l);
                int v326;
                v326 = 4l * v322;
                int v327;
                v327 = v326 + v324;
                float v328;
                v328 = v309[v327];
                float v329;
                v329 = v321 + v328;
                v321 = v329;
                v324 += 1l ;
            }
            v322 += 1l ;
        }
        auto v330 = cooperative_groups::coalesced_threads();
        int v331;
        v331 = threadIdx.x;
        auto v332 = cooperative_groups::labeled_partition(v330,v331);
        float v333;
        v333 = cooperative_groups::reduce(v332, v321, v121);
        int v334[4l];
        int v335;
        v335 = 0l;
        while (while_method_6(v335)){
            int v337;
            v337 = 0l;
            while (while_method_4(v337)){
                assert("Tensor range check" && 0 <= v335 && v335 < 1l);
                assert("Tensor range check" && 0 <= v337 && v337 < 4l);
                int v339;
                v339 = 4l * v335;
                int v340;
                v340 = v339 + v337;
                bool v341;
                v341 = v299[v340];
                int v342;
                if (v341){
                    v342 = 1l;
                } else {
                    v342 = 0l;
                }
                assert("Tensor range check" && 0 <= v335 && v335 < 1l);
                assert("Tensor range check" && 0 <= v337 && v337 < 4l);
                v334[v340] = v342;
                v337 += 1l ;
            }
            v335 += 1l ;
        }
        int v343;
        v343 = 0l;
        int v344;
        v344 = 0l;
        while (while_method_6(v344)){
            int v346;
            v346 = 0l;
            while (while_method_4(v346)){
                assert("Tensor range check" && 0 <= v344 && v344 < 1l);
                assert("Tensor range check" && 0 <= v346 && v346 < 4l);
                int v348;
                v348 = 4l * v344;
                int v349;
                v349 = v348 + v346;
                int v350;
                v350 = v334[v349];
                int v351;
                v351 = v343 + v350;
                v343 = v351;
                v346 += 1l ;
            }
            v344 += 1l ;
        }
        auto v352 = cooperative_groups::coalesced_threads();
        int v353;
        v353 = threadIdx.x;
        auto v354 = cooperative_groups::labeled_partition(v352,v353);
        int v355;
        v355 = cooperative_groups::reduce(v354, v343, v144);
        float v356;
        v356 = (float)v355;
        float v357;
        v357 = 1.0f / v356;
        float v358[4l];
        int v359;
        v359 = 0l;
        while (while_method_6(v359)){
            int v361;
            v361 = 0l;
            while (while_method_4(v361)){
                assert("Tensor range check" && 0 <= v359 && v359 < 1l);
                assert("Tensor range check" && 0 <= v361 && v361 < 4l);
                int v363;
                v363 = 4l * v359;
                int v364;
                v364 = v363 + v361;
                float v365;
                v365 = v309[v364];
                bool v366;
                v366 = v299[v364];
                bool v367;
                v367 = v366 == false;
                float v372;
                if (v367){
                    v372 = 0.0f;
                } else {
                    bool v368;
                    v368 = v333 == 0.0f;
                    bool v369;
                    v369 = v368 != true;
                    if (v369){
                        float v370;
                        v370 = v365 / v333;
                        v372 = v370;
                    } else {
                        v372 = v357;
                    }
                }
                assert("Tensor range check" && 0 <= v359 && v359 < 1l);
                assert("Tensor range check" && 0 <= v361 && v361 < 4l);
                v358[v364] = v372;
                v361 += 1l ;
            }
            v359 += 1l ;
        }
        float v373; int v374;
        Tuple2 tmp12 = Tuple2{0.0f, 2147483647l};
        v373 = tmp12.v0; v374 = tmp12.v1;
        int v375;
        v375 = 0l;
        while (while_method_6(v375)){
            int v377;
            v377 = 0l;
            while (while_method_4(v377)){
                assert("Tensor range check" && 0 <= v375 && v375 < 1l);
                assert("Tensor range check" && 0 <= v377 && v377 < 4l);
                int v379;
                v379 = 4l * v375;
                int v380;
                v380 = v379 + v377;
                float v381;
                v381 = v148[v380];
                int v382;
                v382 = v53[v380];
                bool v383;
                v383 = v374 == v295;
                float v387; int v388;
                if (v383){
                    v387 = v373; v388 = v374;
                } else {
                    bool v384;
                    v384 = v382 == v295;
                    if (v384){
                        v387 = v381; v388 = v382;
                    } else {
                        v387 = v373; v388 = v374;
                    }
                }
                v373 = v387;
                v374 = v388;
                v377 += 1l ;
            }
            v375 += 1l ;
        }
        auto v389 = cooperative_groups::coalesced_threads();
        int v390;
        v390 = threadIdx.x;
        auto v391 = cooperative_groups::labeled_partition(v389,v390);
        Closure7 v392{v295};
        float v393; int v394;
        Tuple2 tmp13 = cooperative_groups::reduce(v391, Tuple2{v373, v374}, v392);
        v393 = tmp13.v0; v394 = tmp13.v1;
        bool v395;
        v395 = v394 == 2147483647l;
        bool v396;
        v396 = v395 != true;
        bool v397;
        v397 = v396 == false;
        if (v397){
            assert("Expected a valid action id in get_action." && v396);
        } else {
        }
        int v399;
        v399 = 0l;
        while (while_method_6(v399)){
            assert("Tensor range check" && 0 <= v399 && v399 < 1l);
            assert("Tensor range check" && 0 <= v399 && v399 < 1l);
            v399 += 1l ;
        }
        assert("Tensor range check" && 0 <= v42 && v42 < 32l);
        v21[v42] = v393;
        v22[v42] = v295;
        v31 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v23 && v23 < 32l);
    float v401;
    v401 = v21[v23];
    int v402;
    v402 = v22[v23];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return Tuple2{v401, v402};
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
    __shared__ int v16[32l];
    __shared__ float * v17[32l];
    /* void shared array create v18 */;
    __shared__ float v19[32l];
    int v20;
    v20 = threadIdx.x;
    assert("Tensor range check" && 0 <= v20 && v20 < 32l);
    v16[v20] = v10;
    v17[v20] = v14;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v21;
    v21 = 0l <= v20;
    bool v22;
    v22 = v21 == false;
    if (v22){
        assert("The index needs to be zero or positive." && v21);
    } else {
    }
    int v24;
    v24 = v20 % 1l;
    bool v25;
    v25 = v20 < 32l;
    bool v26;
    v26 = v25 == false;
    if (v26){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v25);
    } else {
    }
    assert("Tensor range check" && 0 <= v20 && v20 < 32l);
    int v28;
    v28 = 0l;
    while (while_method_6(v28)){
        bool v30;
        v30 = v21 && v25;
        bool v31;
        v31 = v30 == false;
        if (v31){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v30);
        } else {
        }
        bool v33;
        v33 = 0l <= v28;
        bool v35;
        if (v33){
            bool v34;
            v34 = v28 < 1l;
            v35 = v34;
        } else {
            v35 = false;
        }
        bool v36;
        v36 = v35 == false;
        if (v36){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v35);
        } else {
        }
        int v38;
        v38 = v28 * 32l;
        int v39;
        v39 = v38 + v20;
        assert("Tensor range check" && 0 <= v28 && v28 < 1l);
        int v40;
        v40 = 32l * v28;
        int v41;
        v41 = v40 + v20;
        int v42;
        v42 = v16[v41];
        float * v43;
        v43 = v17[v41];
        /* void array index */;
        int v44;
        v44 = blockIdx.x;
        int v45;
        v45 = v44 * 32l;
        int v46;
        v46 = v45 + v39;
        assert("Tensor range check" && 0 <= v24 && v24 < 1l);
        int v47;
        v47 = 4l * v24;
        float v48[4l];
        int v49[4l];
        int v50;
        v50 = 0l;
        while (while_method_6(v50)){
            assert("Tensor range check" && 0 <= v50 && v50 < 1l);
            int v52;
            v52 = 4l * v50;
            assert("Tensor range check" && 0 <= v50 && v50 < 1l);
            int v53;
            v53 = v52 + v47;
            int4* v54;
            v54 = reinterpret_cast<int4*>(v43 + v53);
            int4* v55;
            v55 = reinterpret_cast<int4*>(v48 + v52);
            assert("Pointer alignment check" && (unsigned long long)(v54) % 4l == 0 && (unsigned long long)(v55) % 4l == 0);
            *v55 = *v54;
            v50 += 1l ;
        }
        int v56;
        v56 = 0l;
        while (while_method_6(v56)){
            int v58;
            v58 = 0l;
            while (while_method_4(v58)){
                bool v60;
                v60 = 0l <= v58;
                bool v62;
                if (v60){
                    bool v61;
                    v61 = v58 < 4l;
                    v62 = v61;
                } else {
                    v62 = false;
                }
                bool v63;
                v63 = v62 == false;
                if (v63){
                    assert("The indices should be inside the range of the dimension." && v62);
                } else {
                }
                bool v65;
                v65 = 0l <= v24;
                bool v67;
                if (v65){
                    bool v66;
                    v66 = v24 < 1l;
                    v67 = v66;
                } else {
                    v67 = false;
                }
                bool v68;
                v68 = v67 == false;
                if (v68){
                    assert("The indices should be inside the range of the dimension." && v67);
                } else {
                }
                int v70;
                v70 = v24 * 4l;
                int v71;
                v71 = v58 + v70;
                bool v72;
                v72 = 0l <= v56;
                bool v74;
                if (v72){
                    bool v73;
                    v73 = v56 < 1l;
                    v74 = v73;
                } else {
                    v74 = false;
                }
                bool v75;
                v75 = v74 == false;
                if (v75){
                    assert("The indices should be inside the range of the dimension." && v74);
                } else {
                }
                int v77;
                v77 = v56 * 4l;
                int v78;
                v78 = v71 + v77;
                assert("Tensor range check" && 0 <= v56 && v56 < 1l);
                assert("Tensor range check" && 0 <= v58 && v58 < 4l);
                int v79;
                v79 = 4l * v56;
                int v80;
                v80 = v79 + v58;
                v49[v80] = v78;
                v58 += 1l ;
            }
            v56 += 1l ;
        }
        bool v81[4l];
        int v82;
        v82 = 0l;
        while (while_method_6(v82)){
            int v84;
            v84 = 0l;
            while (while_method_4(v84)){
                assert("Tensor range check" && 0 <= v82 && v82 < 1l);
                assert("Tensor range check" && 0 <= v84 && v84 < 4l);
                int v86;
                v86 = 4l * v82;
                int v87;
                v87 = v86 + v84;
                float v88;
                v88 = v48[v87];
                int v89;
                v89 = v49[v87];
                bool v90;
                v90 = v89 < 3l;
                assert("Tensor range check" && 0 <= v82 && v82 < 1l);
                assert("Tensor range check" && 0 <= v84 && v84 < 4l);
                v81[v87] = v90;
                v84 += 1l ;
            }
            v82 += 1l ;
        }
        float v91[4l];
        int v92;
        v92 = 0l;
        while (while_method_6(v92)){
            int v94;
            v94 = 0l;
            while (while_method_4(v94)){
                assert("Tensor range check" && 0 <= v92 && v92 < 1l);
                assert("Tensor range check" && 0 <= v94 && v94 < 4l);
                int v96;
                v96 = 4l * v92;
                int v97;
                v97 = v96 + v94;
                float v98;
                v98 = v48[v97];
                bool v99;
                v99 = v81[v97];
                float v102;
                if (v99){
                    bool v100;
                    v100 = 0.0f >= v98;
                    if (v100){
                        v102 = 0.0f;
                    } else {
                        v102 = v98;
                    }
                } else {
                    v102 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v92 && v92 < 1l);
                assert("Tensor range check" && 0 <= v94 && v94 < 4l);
                v91[v97] = v102;
                v94 += 1l ;
            }
            v92 += 1l ;
        }
        float v103;
        v103 = 0.0f;
        int v104;
        v104 = 0l;
        while (while_method_6(v104)){
            int v106;
            v106 = 0l;
            while (while_method_4(v106)){
                assert("Tensor range check" && 0 <= v104 && v104 < 1l);
                assert("Tensor range check" && 0 <= v106 && v106 < 4l);
                int v108;
                v108 = 4l * v104;
                int v109;
                v109 = v108 + v106;
                float v110;
                v110 = v91[v109];
                float v111;
                v111 = v103 + v110;
                v103 = v111;
                v106 += 1l ;
            }
            v104 += 1l ;
        }
        auto v112 = cooperative_groups::coalesced_threads();
        int v113;
        v113 = threadIdx.x;
        auto v114 = cooperative_groups::labeled_partition(v112,v113);
        Closure1 v115{};
        float v116;
        v116 = cooperative_groups::reduce(v114, v103, v115);
        int v117[4l];
        int v118;
        v118 = 0l;
        while (while_method_6(v118)){
            int v120;
            v120 = 0l;
            while (while_method_4(v120)){
                assert("Tensor range check" && 0 <= v118 && v118 < 1l);
                assert("Tensor range check" && 0 <= v120 && v120 < 4l);
                int v122;
                v122 = 4l * v118;
                int v123;
                v123 = v122 + v120;
                bool v124;
                v124 = v81[v123];
                int v125;
                if (v124){
                    v125 = 1l;
                } else {
                    v125 = 0l;
                }
                assert("Tensor range check" && 0 <= v118 && v118 < 1l);
                assert("Tensor range check" && 0 <= v120 && v120 < 4l);
                v117[v123] = v125;
                v120 += 1l ;
            }
            v118 += 1l ;
        }
        int v126;
        v126 = 0l;
        int v127;
        v127 = 0l;
        while (while_method_6(v127)){
            int v129;
            v129 = 0l;
            while (while_method_4(v129)){
                assert("Tensor range check" && 0 <= v127 && v127 < 1l);
                assert("Tensor range check" && 0 <= v129 && v129 < 4l);
                int v131;
                v131 = 4l * v127;
                int v132;
                v132 = v131 + v129;
                int v133;
                v133 = v117[v132];
                int v134;
                v134 = v126 + v133;
                v126 = v134;
                v129 += 1l ;
            }
            v127 += 1l ;
        }
        auto v135 = cooperative_groups::coalesced_threads();
        int v136;
        v136 = threadIdx.x;
        auto v137 = cooperative_groups::labeled_partition(v135,v136);
        Closure2 v138{};
        int v139;
        v139 = cooperative_groups::reduce(v137, v126, v138);
        float v140;
        v140 = (float)v139;
        float v141;
        v141 = 1.0f / v140;
        float v142[4l];
        int v143;
        v143 = 0l;
        while (while_method_6(v143)){
            int v145;
            v145 = 0l;
            while (while_method_4(v145)){
                assert("Tensor range check" && 0 <= v143 && v143 < 1l);
                assert("Tensor range check" && 0 <= v145 && v145 < 4l);
                int v147;
                v147 = 4l * v143;
                int v148;
                v148 = v147 + v145;
                float v149;
                v149 = v91[v148];
                bool v150;
                v150 = v81[v148];
                bool v151;
                v151 = v150 == false;
                float v156;
                if (v151){
                    v156 = 0.0f;
                } else {
                    bool v152;
                    v152 = v116 == 0.0f;
                    bool v153;
                    v153 = v152 != true;
                    if (v153){
                        float v154;
                        v154 = v149 / v116;
                        v156 = v154;
                    } else {
                        v156 = v141;
                    }
                }
                assert("Tensor range check" && 0 <= v143 && v143 < 1l);
                assert("Tensor range check" && 0 <= v145 && v145 < 4l);
                v142[v148] = v156;
                v145 += 1l ;
            }
            v143 += 1l ;
        }
        float v157; int v158;
        Tuple2 tmp15 = Tuple2{0.0f, 2147483647l};
        v157 = tmp15.v0; v158 = tmp15.v1;
        int v159;
        v159 = 0l;
        while (while_method_6(v159)){
            int v161;
            v161 = 0l;
            while (while_method_4(v161)){
                assert("Tensor range check" && 0 <= v159 && v159 < 1l);
                assert("Tensor range check" && 0 <= v161 && v161 < 4l);
                int v163;
                v163 = 4l * v159;
                int v164;
                v164 = v163 + v161;
                float v165;
                v165 = v142[v164];
                int v166;
                v166 = v49[v164];
                bool v167;
                v167 = v158 == v42;
                float v171; int v172;
                if (v167){
                    v171 = v157; v172 = v158;
                } else {
                    bool v168;
                    v168 = v166 == v42;
                    if (v168){
                        v171 = v165; v172 = v166;
                    } else {
                        v171 = v157; v172 = v158;
                    }
                }
                v157 = v171;
                v158 = v172;
                v161 += 1l ;
            }
            v159 += 1l ;
        }
        auto v173 = cooperative_groups::coalesced_threads();
        int v174;
        v174 = threadIdx.x;
        auto v175 = cooperative_groups::labeled_partition(v173,v174);
        Closure7 v176{v42};
        float v177; int v178;
        Tuple2 tmp16 = cooperative_groups::reduce(v175, Tuple2{v157, v158}, v176);
        v177 = tmp16.v0; v178 = tmp16.v1;
        bool v179;
        v179 = v178 == 2147483647l;
        bool v180;
        v180 = v179 != true;
        bool v181;
        v181 = v180 == false;
        if (v181){
            assert("Expected a valid action id in get_action." && v180);
        } else {
        }
        int v183;
        v183 = 0l;
        while (while_method_6(v183)){
            assert("Tensor range check" && 0 <= v183 && v183 < 1l);
            assert("Tensor range check" && 0 <= v183 && v183 < 1l);
            v183 += 1l ;
        }
        assert("Tensor range check" && 0 <= v39 && v39 < 32l);
        v19[v39] = v177;
        v28 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v20 && v20 < 32l);
    float v185;
    v185 = v19[v20];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return v185;
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
                    Tuple1 tmp27 = order_14(v8, v11);
                    v27 = tmp27.v0; v28 = tmp27.v1;
                    int v29; int v30;
                    Tuple1 tmp28 = order_14(v8, v14);
                    v29 = tmp28.v0; v30 = tmp28.v1;
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
    while (while_method_1(v11)){
        Union7 v757;
        switch (v11.tag) {
            case 0: { // None
                v757 = Union7{Union7_0{}};
                break;
            }
            case 1: { // Some
                Union5 v13 = v11.case1.v0;
                switch (v13.tag) {
                    case 0: { // ChanceCommunityCard
                        Union6 v708 = v13.case0.v0; bool v709 = v13.case0.v1; static_array<Union3,2l> v710 = v13.case0.v2; int v711 = v13.case0.v3; static_array<int,2l> v712 = v13.case0.v4; int v713 = v13.case0.v5;
                        unsigned int v714 = v2;
                        Union3 v715; unsigned int v716;
                        Tuple0 tmp0 = draw_card_4(v5, v714);
                        v715 = tmp0.v0; v716 = tmp0.v1;
                        v2 = v716;
                        Union2 v717;
                        v717 = Union2{Union2_0{v715}};
                        v9.push(v717);
                        int v718;
                        v718 = 2l;
                        int v719; int v720;
                        Tuple1 tmp1 = Tuple1{0l, 0l};
                        v719 = tmp1.v0; v720 = tmp1.v1;
                        while (while_method_0(v719)){
                            int v722;
                            v722 = v712[v719];
                            bool v724;
                            v724 = v720 >= v722;
                            int v725;
                            if (v724){
                                v725 = v720;
                            } else {
                                v725 = v722;
                            }
                            v720 = v725;
                            v719 += 1l ;
                        }
                        static_array<int,2l> v726;
                        int v728;
                        v728 = 0l;
                        while (while_method_0(v728)){
                            v726[v728] = v720;
                            v728 += 1l ;
                        }
                        Union6 v730;
                        v730 = Union6{Union6_1{v715}};
                        Union5 v731;
                        v731 = Union5{Union5_2{v730, true, v710, 0l, v726, v718}};
                        v757 = Union7{Union7_1{v731}};
                        break;
                    }
                    case 1: { // ChanceInit
                        unsigned int v733 = v2;
                        Union3 v734; unsigned int v735;
                        Tuple0 tmp2 = draw_card_4(v5, v733);
                        v734 = tmp2.v0; v735 = tmp2.v1;
                        v2 = v735;
                        unsigned int v736 = v2;
                        Union3 v737; unsigned int v738;
                        Tuple0 tmp3 = draw_card_4(v5, v736);
                        v737 = tmp3.v0; v738 = tmp3.v1;
                        v2 = v738;
                        Union2 v739;
                        v739 = Union2{Union2_2{0l, v734}};
                        v9.push(v739);
                        Union2 v740;
                        v740 = Union2{Union2_2{1l, v737}};
                        v9.push(v740);
                        int v741;
                        v741 = 2l;
                        static_array<int,2l> v742;
                        v742[0l] = 1l;
                        v742[1l] = 1l;
                        static_array<Union3,2l> v744;
                        v744[0l] = v734;
                        v744[1l] = v737;
                        Union6 v746;
                        v746 = Union6{Union6_0{}};
                        Union5 v747;
                        v747 = Union5{Union5_2{v746, true, v744, 0l, v742, v741}};
                        v757 = Union7{Union7_1{v747}};
                        break;
                    }
                    case 2: { // Round
                        Union6 v54 = v13.case2.v0; bool v55 = v13.case2.v1; static_array<Union3,2l> v56 = v13.case2.v2; int v57 = v13.case2.v3; static_array<int,2l> v58 = v13.case2.v4; int v59 = v13.case2.v5;
                        static_array<Union1,2l> v60 = v4;
                        Union1 v61;
                        v61 = v60[v57];
                        switch (v61.tag) {
                            case 0: { // Computer
                                static_array_list<Union2,32l> & v63 = v3;
                                unsigned int * v64;
                                v64 = reinterpret_cast<unsigned int *>(&v0[32768ull]);
                                float * v66;
                                v66 = reinterpret_cast<float *>(&v0[0ull]);
                                int v68;
                                v68 = threadIdx.x;
                                int v69;
                                v69 = blockIdx.x;
                                int v70;
                                v70 = v69 * 32l;
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
                                assert("Tensor range check" && 0 <= v76 && v76 < 1l);
                                int v77;
                                v77 = 4096l * v76;
                                int v78;
                                v78 = threadIdx.x;
                                int v79;
                                v79 = blockIdx.x;
                                int v80;
                                v80 = v79 * 32l;
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
                                while (while_method_2(v85)){
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
                                    v92 = v91 < 32l;
                                    bool v93;
                                    v93 = v92 == false;
                                    if (v93){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v92);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v91 && v91 < 32l);
                                    assert("Tensor range check" && 0 <= v90 && v90 < 128l);
                                    int v95;
                                    v95 = v90 + v77;
                                    int v96;
                                    v96 = 128l * v91;
                                    int v97;
                                    v97 = v96 + v95;
                                    v74[v97] = 0.0f;
                                    v85 += 32l ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                int v98;
                                v98 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v98 && v98 < 32l);
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
                                while (while_method_3(v103, v104)){
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
                                while (while_method_3(v131, v132)){
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
                                while (while_method_4(v145)){
                                    float * v147;
                                    v147 = reinterpret_cast<float *>(&v0[0ull]);
                                    float * v149;
                                    v149 = reinterpret_cast<float *>(&v1[0ull]);
                                    assert("Tensor range check" && 0 <= v145 && v145 < 4l);
                                    int v151;
                                    v151 = 16384l * v145;
                                    float * v152;
                                    v152 = reinterpret_cast<float *>(&v0[16384ull]);
                                    int v154;
                                    v154 = blockIdx.x;
                                    assert("Tensor range check" && 0 <= v154 && v154 < 1l);
                                    int v155;
                                    v155 = 4096l * v154;
                                    int v156;
                                    v156 = blockIdx.x;
                                    assert("Tensor range check" && 0 <= v156 && v156 < 1l);
                                    int v157;
                                    v157 = 4096l * v156;
                                    method_6(v149, v151, v152, v157, v147, v155);
                                    unsigned int * v158;
                                    v158 = reinterpret_cast<unsigned int *>(&v0[32768ull]);
                                    assert("Tensor range check" && 0 <= v145 && v145 < 4l);
                                    int v160;
                                    v160 = 32l * v145;
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
                                    v177 = reinterpret_cast<int *>(&v0[33280ull]);
                                    float * v179;
                                    v179 = reinterpret_cast<float *>(&v0[41472ull]);
                                    int * v181;
                                    v181 = reinterpret_cast<int *>(&v0[49664ull]);
                                    int * v183;
                                    v183 = reinterpret_cast<int *>(&v0[57856ull]);
                                    double * v185;
                                    v185 = reinterpret_cast<double *>(&v0[66048ull]);
                                    double * v187;
                                    v187 = reinterpret_cast<double *>(&v0[98816ull]);
                                    double * v189;
                                    v189 = reinterpret_cast<double *>(&v1[2097168ull]);
                                    double * v191;
                                    v191 = reinterpret_cast<double *>(&v1[2099216ull]);
                                    int * v193;
                                    v193 = reinterpret_cast<int *>(&v1[2101264ull]);
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
                                v212 = reinterpret_cast<unsigned int *>(&v0[32768ull]);
                                int v214;
                                v214 = blockIdx.x;
                                int v215;
                                v215 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v211 && v211 < 4l);
                                assert("Tensor range check" && 0 <= v214 && v214 < 1l);
                                assert("Tensor range check" && 0 <= v215 && v215 < 32l);
                                int v216;
                                v216 = 32l * v214;
                                int v217;
                                v217 = v216 + v215;
                                int v218;
                                v218 = 32l * v211;
                                int v219;
                                v219 = v218 + v217;
                                unsigned int v220;
                                v220 = v212[v219];
                                int v221;
                                v221 = (int)v220;
                                float v222; int v223;
                                Tuple2 tmp14 = method_8(v5, v195, v197, v199, v201, v203, v205, v207, v209, v221, v211);
                                v222 = tmp14.v0; v223 = tmp14.v1;
                                __shared__ float v224[1l];
                                __shared__ int v225[1l];
                                int v226;
                                v226 = threadIdx.x;
                                bool v227;
                                v227 = v226 == 0l;
                                if (v227){
                                    v224[0l] = v222;
                                    v225[0l] = v223;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                float v228;
                                v228 = v224[0l];
                                int v229;
                                v229 = v225[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                double * v230;
                                v230 = reinterpret_cast<double *>(&v1[2097168ull]);
                                double * v232;
                                v232 = reinterpret_cast<double *>(&v1[2099216ull]);
                                int * v234;
                                v234 = reinterpret_cast<int *>(&v1[2101264ull]);
                                int * v236;
                                v236 = reinterpret_cast<int *>(&v0[33280ull]);
                                float * v238;
                                v238 = reinterpret_cast<float *>(&v0[41472ull]);
                                int * v240;
                                v240 = reinterpret_cast<int *>(&v0[49664ull]);
                                int * v242;
                                v242 = reinterpret_cast<int *>(&v0[57856ull]);
                                double * v244;
                                v244 = reinterpret_cast<double *>(&v0[66048ull]);
                                double * v246;
                                v246 = reinterpret_cast<double *>(&v0[98816ull]);
                                int v248;
                                v248 = threadIdx.x;
                                int v249;
                                v249 = blockIdx.x;
                                int v250;
                                v250 = v249 * 32l;
                                int v251;
                                v251 = v248 + v250;
                                int v252;
                                v252 = 0l;
                                while (while_method_4(v252)){
                                    unsigned int * v254;
                                    v254 = reinterpret_cast<unsigned int *>(&v0[32768ull]);
                                    int v256;
                                    v256 = blockIdx.x;
                                    int v257;
                                    v257 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v252 && v252 < 4l);
                                    assert("Tensor range check" && 0 <= v256 && v256 < 1l);
                                    assert("Tensor range check" && 0 <= v257 && v257 < 32l);
                                    int v258;
                                    v258 = 32l * v256;
                                    int v259;
                                    v259 = v258 + v257;
                                    int v260;
                                    v260 = 32l * v252;
                                    int v261;
                                    v261 = v260 + v259;
                                    unsigned int v262;
                                    v262 = v254[v261];
                                    int v263;
                                    v263 = (int)v262;
                                    float v264;
                                    v264 = method_9(v195, v197, v199, v201, v203, v205, v207, v209, v263, v252, v229);
                                    assert("Tensor range check" && 0 <= v252 && v252 < 4l);
                                    assert("Tensor range check" && 0 <= v251 && v251 < 32l);
                                    int v265;
                                    v265 = v260 + v251;
                                    int v266;
                                    v266 = v234[v265];
                                    int v267;
                                    v267 = v266 + 1l;
                                    assert("Tensor range check" && 0 <= v252 && v252 < 4l);
                                    assert("Tensor range check" && 0 <= v251 && v251 < 32l);
                                    v234[v265] = v267;
                                    assert("Tensor range check" && 0 <= v252 && v252 < 4l);
                                    assert("Tensor range check" && 0 <= v266 && v266 < 16l);
                                    assert("Tensor range check" && 0 <= v251 && v251 < 32l);
                                    int v268;
                                    v268 = 32l * v266;
                                    int v269;
                                    v269 = v268 + v251;
                                    int v270;
                                    v270 = 512l * v252;
                                    int v271;
                                    v271 = v270 + v269;
                                    v236[v271] = v229;
                                    v238[v271] = v228;
                                    v240[v271] = v57;
                                    v242[v271] = v263;
                                    double v272;
                                    v272 = (double)v228;
                                    double v273;
                                    v273 = log(v272);
                                    double v274;
                                    v274 = (double)v264;
                                    double v275;
                                    v275 = log(v274);
                                    assert("Tensor range check" && 0 <= v252 && v252 < 4l);
                                    int v276;
                                    v276 = 64l * v252;
                                    assert("Tensor range check" && 0 <= v251 && v251 < 32l);
                                    int v277;
                                    v277 = 2l * v251;
                                    int v278;
                                    v278 = v277 + v276;
                                    assert("Tensor range check" && 0 <= v252 && v252 < 4l);
                                    int v279;
                                    v279 = 1024l * v252;
                                    assert("Tensor range check" && 0 <= v266 && v266 < 16l);
                                    int v280;
                                    v280 = 64l * v266;
                                    int v281;
                                    v281 = v280 + v279;
                                    assert("Tensor range check" && 0 <= v251 && v251 < 32l);
                                    int v282;
                                    v282 = v277 + v281;
                                    double * v283;
                                    v283 = v230+v278;
                                    double * v285;
                                    v285 = v232+v278;
                                    double * v287;
                                    v287 = v244+v282;
                                    double * v289;
                                    v289 = v246+v282;
                                    __shared__ double * v291[32l];
                                    __shared__ double * v292[32l];
                                    __shared__ double * v293[32l];
                                    __shared__ double * v294[32l];
                                    /* void shared array create v295 */;
                                    /* void shared array create v296 */;
                                    int v297;
                                    v297 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v297 && v297 < 32l);
                                    v291[v297] = v283;
                                    v292[v297] = v285;
                                    v293[v297] = v287;
                                    v294[v297] = v289;
                                    /* void array set */;
                                    asm("barrier.cta.sync %0;" :: "r"(0l));
                                    bool v298;
                                    v298 = 0l <= v297;
                                    bool v299;
                                    v299 = v298 == false;
                                    if (v299){
                                        assert("The index needs to be zero or positive." && v298);
                                    } else {
                                    }
                                    int v301;
                                    v301 = v297 % 1l;
                                    bool v302;
                                    v302 = v297 < 32l;
                                    bool v303;
                                    v303 = v302 == false;
                                    if (v303){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v302);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v297 && v297 < 32l);
                                    int v305;
                                    v305 = 0l;
                                    while (while_method_6(v305)){
                                        bool v307;
                                        v307 = v298 && v302;
                                        bool v308;
                                        v308 = v307 == false;
                                        if (v308){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v307);
                                        } else {
                                        }
                                        bool v310;
                                        v310 = 0l <= v305;
                                        bool v312;
                                        if (v310){
                                            bool v311;
                                            v311 = v305 < 1l;
                                            v312 = v311;
                                        } else {
                                            v312 = false;
                                        }
                                        bool v313;
                                        v313 = v312 == false;
                                        if (v313){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v312);
                                        } else {
                                        }
                                        int v315;
                                        v315 = v305 * 32l;
                                        int v316;
                                        v316 = v315 + v297;
                                        assert("Tensor range check" && 0 <= v305 && v305 < 1l);
                                        int v317;
                                        v317 = 32l * v305;
                                        int v318;
                                        v318 = v317 + v297;
                                        double * v319;
                                        v319 = v291[v318];
                                        double * v320;
                                        v320 = v292[v318];
                                        double * v321;
                                        v321 = v293[v318];
                                        double * v322;
                                        v322 = v294[v318];
                                        /* void array index */;
                                        int v323;
                                        v323 = blockIdx.x;
                                        int v324;
                                        v324 = v323 * 32l;
                                        int v325;
                                        v325 = v324 + v316;
                                        assert("Tensor range check" && 0 <= v301 && v301 < 1l);
                                        int v326;
                                        v326 = 2l * v301;
                                        double v327[2l];
                                        double v328[2l];
                                        int v329[2l];
                                        int v330;
                                        v330 = 0l;
                                        while (while_method_6(v330)){
                                            assert("Tensor range check" && 0 <= v330 && v330 < 1l);
                                            int v332;
                                            v332 = 2l * v330;
                                            assert("Tensor range check" && 0 <= v330 && v330 < 1l);
                                            int v333;
                                            v333 = v332 + v326;
                                            int4* v334;
                                            v334 = reinterpret_cast<int4*>(v319 + v333);
                                            int4* v335;
                                            v335 = reinterpret_cast<int4*>(v327 + v332);
                                            assert("Pointer alignment check" && (unsigned long long)(v334) % 2l == 0 && (unsigned long long)(v335) % 2l == 0);
                                            *v335 = *v334;
                                            int4* v336;
                                            v336 = reinterpret_cast<int4*>(v320 + v333);
                                            int4* v337;
                                            v337 = reinterpret_cast<int4*>(v328 + v332);
                                            assert("Pointer alignment check" && (unsigned long long)(v336) % 2l == 0 && (unsigned long long)(v337) % 2l == 0);
                                            *v337 = *v336;
                                            v330 += 1l ;
                                        }
                                        int v338;
                                        v338 = 0l;
                                        while (while_method_6(v338)){
                                            int v340;
                                            v340 = 0l;
                                            while (while_method_0(v340)){
                                                bool v342;
                                                v342 = 0l <= v340;
                                                bool v344;
                                                if (v342){
                                                    bool v343;
                                                    v343 = v340 < 2l;
                                                    v344 = v343;
                                                } else {
                                                    v344 = false;
                                                }
                                                bool v345;
                                                v345 = v344 == false;
                                                if (v345){
                                                    assert("The indices should be inside the range of the dimension." && v344);
                                                } else {
                                                }
                                                bool v347;
                                                v347 = 0l <= v301;
                                                bool v349;
                                                if (v347){
                                                    bool v348;
                                                    v348 = v301 < 1l;
                                                    v349 = v348;
                                                } else {
                                                    v349 = false;
                                                }
                                                bool v350;
                                                v350 = v349 == false;
                                                if (v350){
                                                    assert("The indices should be inside the range of the dimension." && v349);
                                                } else {
                                                }
                                                int v352;
                                                v352 = v301 * 2l;
                                                int v353;
                                                v353 = v340 + v352;
                                                bool v354;
                                                v354 = 0l <= v338;
                                                bool v356;
                                                if (v354){
                                                    bool v355;
                                                    v355 = v338 < 1l;
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
                                                v359 = v338 * 2l;
                                                int v360;
                                                v360 = v353 + v359;
                                                assert("Tensor range check" && 0 <= v338 && v338 < 1l);
                                                assert("Tensor range check" && 0 <= v340 && v340 < 2l);
                                                int v361;
                                                v361 = 2l * v338;
                                                int v362;
                                                v362 = v361 + v340;
                                                v329[v362] = v360;
                                                v340 += 1l ;
                                            }
                                            v338 += 1l ;
                                        }
                                        int v363;
                                        v363 = 0l;
                                        while (while_method_6(v363)){
                                            assert("Tensor range check" && 0 <= v363 && v363 < 1l);
                                            int v365;
                                            v365 = 2l * v363;
                                            int v366;
                                            v366 = v365 + v326;
                                            assert("Tensor range check" && 0 <= v363 && v363 < 1l);
                                            int4* v367;
                                            v367 = reinterpret_cast<int4*>(v327 + v365);
                                            int4* v368;
                                            v368 = reinterpret_cast<int4*>(v321 + v366);
                                            assert("Pointer alignment check" && (unsigned long long)(v367) % 2l == 0 && (unsigned long long)(v368) % 2l == 0);
                                            *v368 = *v367;
                                            int4* v369;
                                            v369 = reinterpret_cast<int4*>(v328 + v365);
                                            int4* v370;
                                            v370 = reinterpret_cast<int4*>(v322 + v366);
                                            assert("Pointer alignment check" && (unsigned long long)(v369) % 2l == 0 && (unsigned long long)(v370) % 2l == 0);
                                            *v370 = *v369;
                                            v363 += 1l ;
                                        }
                                        assert("Tensor range check" && 0 <= v316 && v316 < 32l);
                                        /* void array set */;
                                        v305 += 1l ;
                                    }
                                    asm("barrier.cta.sync %0;" :: "r"(0l));
                                    assert("Tensor range check" && 0 <= v297 && v297 < 32l);
                                    /* void array index */;
                                    asm("barrier.cta.sync %0;" :: "r"(0l));
                                    assert("Tensor range check" && 0 <= v57 && v57 < 2l);
                                    int v371;
                                    v371 = v57 + v278;
                                    double v372;
                                    v372 = v230[v371];
                                    double v373;
                                    v373 = v232[v371];
                                    double v374;
                                    v374 = v275 + v372;
                                    double v375;
                                    v375 = v273 + v373;
                                    assert("Tensor range check" && 0 <= v57 && v57 < 2l);
                                    v230[v371] = v374;
                                    v232[v371] = v375;
                                    v252 += 1l ;
                                }
                                bool v376;
                                v376 = 0l == v229;
                                Union10 v385;
                                if (v376){
                                    v385 = Union10{Union10_1{}};
                                } else {
                                    bool v378;
                                    v378 = 1l == v229;
                                    if (v378){
                                        v385 = Union10{Union10_0{}};
                                    } else {
                                        bool v380;
                                        v380 = 2l == v229;
                                        if (v380){
                                            v385 = Union10{Union10_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                Union4 v408;
                                switch (v385.tag) {
                                    case 0: { // AA_Call
                                        v408 = Union4{Union4_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v386;
                                        v386 = v58[0l];
                                        int v388; int v389;
                                        Tuple1 tmp17 = Tuple1{1l, v386};
                                        v388 = tmp17.v0; v389 = tmp17.v1;
                                        while (while_method_0(v388)){
                                            int v391;
                                            v391 = v58[v388];
                                            bool v393;
                                            v393 = v389 >= v391;
                                            int v394;
                                            if (v393){
                                                v394 = v389;
                                            } else {
                                                v394 = v391;
                                            }
                                            v389 = v394;
                                            v388 += 1l ;
                                        }
                                        int v395;
                                        v395 = v58[v57];
                                        bool v397;
                                        v397 = v395 == v389;
                                        if (v397){
                                            v408 = Union4{Union4_0{}};
                                        } else {
                                            v408 = Union4{Union4_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v402;
                                        v402 = v59 > 0l;
                                        if (v402){
                                            v408 = Union4{Union4_2{}};
                                        } else {
                                            v408 = Union4{Union4_0{}};
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                Union2 v409;
                                v409 = Union2{Union2_1{v57, v408}};
                                v9.push(v409);
                                Union5 v495;
                                switch (v54.tag) {
                                    case 0: { // None
                                        switch (v408.tag) {
                                            case 0: { // Call
                                                if (v55){
                                                    bool v459;
                                                    v459 = v57 == 0l;
                                                    int v460;
                                                    if (v459){
                                                        v460 = 1l;
                                                    } else {
                                                        v460 = 0l;
                                                    }
                                                    v495 = Union5{Union5_2{v54, false, v56, v460, v58, v59}};
                                                } else {
                                                    v495 = Union5{Union5_0{v54, v55, v56, v57, v58, v59}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v495 = Union5{Union5_5{v54, v55, v56, v57, v58, v59}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                bool v464;
                                                v464 = v59 > 0l;
                                                if (v464){
                                                    bool v465;
                                                    v465 = v57 == 0l;
                                                    int v466;
                                                    if (v465){
                                                        v466 = 1l;
                                                    } else {
                                                        v466 = 0l;
                                                    }
                                                    int v467;
                                                    v467 = -1l + v59;
                                                    int v468; int v469;
                                                    Tuple1 tmp18 = Tuple1{0l, 0l};
                                                    v468 = tmp18.v0; v469 = tmp18.v1;
                                                    while (while_method_0(v468)){
                                                        int v471;
                                                        v471 = v58[v468];
                                                        bool v473;
                                                        v473 = v469 >= v471;
                                                        int v474;
                                                        if (v473){
                                                            v474 = v469;
                                                        } else {
                                                            v474 = v471;
                                                        }
                                                        v469 = v474;
                                                        v468 += 1l ;
                                                    }
                                                    static_array<int,2l> v475;
                                                    int v477;
                                                    v477 = 0l;
                                                    while (while_method_0(v477)){
                                                        v475[v477] = v469;
                                                        v477 += 1l ;
                                                    }
                                                    static_array<int,2l> v479;
                                                    int v481;
                                                    v481 = 0l;
                                                    while (while_method_0(v481)){
                                                        int v483;
                                                        v483 = v475[v481];
                                                        bool v485;
                                                        v485 = v481 == v57;
                                                        int v487;
                                                        if (v485){
                                                            int v486;
                                                            v486 = v483 + 2l;
                                                            v487 = v486;
                                                        } else {
                                                            v487 = v483;
                                                        }
                                                        v479[v481] = v487;
                                                        v481 += 1l ;
                                                    }
                                                    v495 = Union5{Union5_2{v54, false, v56, v466, v479, v467}};
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
                                        Union3 v410 = v54.case1.v0;
                                        switch (v408.tag) {
                                            case 0: { // Call
                                                if (v55){
                                                    bool v412;
                                                    v412 = v57 == 0l;
                                                    int v413;
                                                    if (v412){
                                                        v413 = 1l;
                                                    } else {
                                                        v413 = 0l;
                                                    }
                                                    v495 = Union5{Union5_2{v54, false, v56, v413, v58, v59}};
                                                } else {
                                                    int v415; int v416;
                                                    Tuple1 tmp19 = Tuple1{0l, 0l};
                                                    v415 = tmp19.v0; v416 = tmp19.v1;
                                                    while (while_method_0(v415)){
                                                        int v418;
                                                        v418 = v58[v415];
                                                        bool v420;
                                                        v420 = v416 >= v418;
                                                        int v421;
                                                        if (v420){
                                                            v421 = v416;
                                                        } else {
                                                            v421 = v418;
                                                        }
                                                        v416 = v421;
                                                        v415 += 1l ;
                                                    }
                                                    static_array<int,2l> v422;
                                                    int v424;
                                                    v424 = 0l;
                                                    while (while_method_0(v424)){
                                                        v422[v424] = v416;
                                                        v424 += 1l ;
                                                    }
                                                    v495 = Union5{Union5_4{v54, v55, v56, v57, v422, v59}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v495 = Union5{Union5_5{v54, v55, v56, v57, v58, v59}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                bool v428;
                                                v428 = v59 > 0l;
                                                if (v428){
                                                    bool v429;
                                                    v429 = v57 == 0l;
                                                    int v430;
                                                    if (v429){
                                                        v430 = 1l;
                                                    } else {
                                                        v430 = 0l;
                                                    }
                                                    int v431;
                                                    v431 = -1l + v59;
                                                    int v432; int v433;
                                                    Tuple1 tmp20 = Tuple1{0l, 0l};
                                                    v432 = tmp20.v0; v433 = tmp20.v1;
                                                    while (while_method_0(v432)){
                                                        int v435;
                                                        v435 = v58[v432];
                                                        bool v437;
                                                        v437 = v433 >= v435;
                                                        int v438;
                                                        if (v437){
                                                            v438 = v433;
                                                        } else {
                                                            v438 = v435;
                                                        }
                                                        v433 = v438;
                                                        v432 += 1l ;
                                                    }
                                                    static_array<int,2l> v439;
                                                    int v441;
                                                    v441 = 0l;
                                                    while (while_method_0(v441)){
                                                        v439[v441] = v433;
                                                        v441 += 1l ;
                                                    }
                                                    static_array<int,2l> v443;
                                                    int v445;
                                                    v445 = 0l;
                                                    while (while_method_0(v445)){
                                                        int v447;
                                                        v447 = v439[v445];
                                                        bool v449;
                                                        v449 = v445 == v57;
                                                        int v451;
                                                        if (v449){
                                                            int v450;
                                                            v450 = v447 + 4l;
                                                            v451 = v450;
                                                        } else {
                                                            v451 = v447;
                                                        }
                                                        v443[v445] = v451;
                                                        v445 += 1l ;
                                                    }
                                                    v495 = Union5{Union5_2{v54, false, v56, v430, v443, v431}};
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
                                v757 = Union7{Union7_1{v495}};
                                break;
                            }
                            case 1: { // Random
                                static_array_list<Union4,3l> v497;
                                v497 = static_array_list<Union4,3l>{};
                                v497.unsafe_set_length(1l);
                                Union4 v499;
                                v499 = Union4{Union4_0{}};
                                v497[0l] = v499;
                                int v501;
                                v501 = v58[0l];
                                int v503;
                                v503 = v58[1l];
                                bool v505;
                                v505 = v501 == v503;
                                bool v506;
                                v506 = v505 != true;
                                if (v506){
                                    Union4 v507;
                                    v507 = Union4{Union4_1{}};
                                    v497.push(v507);
                                } else {
                                }
                                bool v508;
                                v508 = v59 > 0l;
                                if (v508){
                                    Union4 v509;
                                    v509 = Union4{Union4_2{}};
                                    v497.push(v509);
                                } else {
                                }
                                int v510;
                                v510 = v497.length;
                                int v511;
                                v511 = v510 - 1l;
                                int v512;
                                v512 = 0l;
                                while (while_method_3(v511, v512)){
                                    int v514;
                                    v514 = v497.length;
                                    int v515;
                                    v515 = int_range_10(v514, v512, v5);
                                    Union4 v516;
                                    v516 = v497[v512];
                                    Union4 v518;
                                    v518 = v497[v515];
                                    v497[v512] = v518;
                                    v497[v515] = v516;
                                    v512 += 1l ;
                                }
                                Union4 v520;
                                v520 = v497.pop();
                                __shared__ Union4 v521[1l];
                                int v522;
                                v522 = threadIdx.x;
                                bool v523;
                                v523 = v522 == 0l;
                                if (v523){
                                    v521[0l] = v520;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                Union4 v524;
                                v524 = v521[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                Union2 v525;
                                v525 = Union2{Union2_1{v57, v524}};
                                v9.push(v525);
                                Union5 v609;
                                switch (v54.tag) {
                                    case 0: { // None
                                        switch (v524.tag) {
                                            case 0: { // Call
                                                if (v55){
                                                    bool v574;
                                                    v574 = v57 == 0l;
                                                    int v575;
                                                    if (v574){
                                                        v575 = 1l;
                                                    } else {
                                                        v575 = 0l;
                                                    }
                                                    v609 = Union5{Union5_2{v54, false, v56, v575, v58, v59}};
                                                } else {
                                                    v609 = Union5{Union5_0{v54, v55, v56, v57, v58, v59}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v609 = Union5{Union5_5{v54, v55, v56, v57, v58, v59}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                if (v508){
                                                    bool v579;
                                                    v579 = v57 == 0l;
                                                    int v580;
                                                    if (v579){
                                                        v580 = 1l;
                                                    } else {
                                                        v580 = 0l;
                                                    }
                                                    int v581;
                                                    v581 = -1l + v59;
                                                    int v582; int v583;
                                                    Tuple1 tmp21 = Tuple1{0l, 0l};
                                                    v582 = tmp21.v0; v583 = tmp21.v1;
                                                    while (while_method_0(v582)){
                                                        int v585;
                                                        v585 = v58[v582];
                                                        bool v587;
                                                        v587 = v583 >= v585;
                                                        int v588;
                                                        if (v587){
                                                            v588 = v583;
                                                        } else {
                                                            v588 = v585;
                                                        }
                                                        v583 = v588;
                                                        v582 += 1l ;
                                                    }
                                                    static_array<int,2l> v589;
                                                    int v591;
                                                    v591 = 0l;
                                                    while (while_method_0(v591)){
                                                        v589[v591] = v583;
                                                        v591 += 1l ;
                                                    }
                                                    static_array<int,2l> v593;
                                                    int v595;
                                                    v595 = 0l;
                                                    while (while_method_0(v595)){
                                                        int v597;
                                                        v597 = v589[v595];
                                                        bool v599;
                                                        v599 = v595 == v57;
                                                        int v601;
                                                        if (v599){
                                                            int v600;
                                                            v600 = v597 + 2l;
                                                            v601 = v600;
                                                        } else {
                                                            v601 = v597;
                                                        }
                                                        v593[v595] = v601;
                                                        v595 += 1l ;
                                                    }
                                                    v609 = Union5{Union5_2{v54, false, v56, v580, v593, v581}};
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
                                        Union3 v526 = v54.case1.v0;
                                        switch (v524.tag) {
                                            case 0: { // Call
                                                if (v55){
                                                    bool v528;
                                                    v528 = v57 == 0l;
                                                    int v529;
                                                    if (v528){
                                                        v529 = 1l;
                                                    } else {
                                                        v529 = 0l;
                                                    }
                                                    v609 = Union5{Union5_2{v54, false, v56, v529, v58, v59}};
                                                } else {
                                                    int v531; int v532;
                                                    Tuple1 tmp22 = Tuple1{0l, 0l};
                                                    v531 = tmp22.v0; v532 = tmp22.v1;
                                                    while (while_method_0(v531)){
                                                        int v534;
                                                        v534 = v58[v531];
                                                        bool v536;
                                                        v536 = v532 >= v534;
                                                        int v537;
                                                        if (v536){
                                                            v537 = v532;
                                                        } else {
                                                            v537 = v534;
                                                        }
                                                        v532 = v537;
                                                        v531 += 1l ;
                                                    }
                                                    static_array<int,2l> v538;
                                                    int v540;
                                                    v540 = 0l;
                                                    while (while_method_0(v540)){
                                                        v538[v540] = v532;
                                                        v540 += 1l ;
                                                    }
                                                    v609 = Union5{Union5_4{v54, v55, v56, v57, v538, v59}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v609 = Union5{Union5_5{v54, v55, v56, v57, v58, v59}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                if (v508){
                                                    bool v544;
                                                    v544 = v57 == 0l;
                                                    int v545;
                                                    if (v544){
                                                        v545 = 1l;
                                                    } else {
                                                        v545 = 0l;
                                                    }
                                                    int v546;
                                                    v546 = -1l + v59;
                                                    int v547; int v548;
                                                    Tuple1 tmp23 = Tuple1{0l, 0l};
                                                    v547 = tmp23.v0; v548 = tmp23.v1;
                                                    while (while_method_0(v547)){
                                                        int v550;
                                                        v550 = v58[v547];
                                                        bool v552;
                                                        v552 = v548 >= v550;
                                                        int v553;
                                                        if (v552){
                                                            v553 = v548;
                                                        } else {
                                                            v553 = v550;
                                                        }
                                                        v548 = v553;
                                                        v547 += 1l ;
                                                    }
                                                    static_array<int,2l> v554;
                                                    int v556;
                                                    v556 = 0l;
                                                    while (while_method_0(v556)){
                                                        v554[v556] = v548;
                                                        v556 += 1l ;
                                                    }
                                                    static_array<int,2l> v558;
                                                    int v560;
                                                    v560 = 0l;
                                                    while (while_method_0(v560)){
                                                        int v562;
                                                        v562 = v554[v560];
                                                        bool v564;
                                                        v564 = v560 == v57;
                                                        int v566;
                                                        if (v564){
                                                            int v565;
                                                            v565 = v562 + 4l;
                                                            v566 = v565;
                                                        } else {
                                                            v566 = v562;
                                                        }
                                                        v558[v560] = v566;
                                                        v560 += 1l ;
                                                    }
                                                    v609 = Union5{Union5_2{v54, false, v56, v545, v558, v546}};
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
                                v757 = Union7{Union7_1{v609}};
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union6 v613 = v13.case3.v0; bool v614 = v13.case3.v1; static_array<Union3,2l> v615 = v13.case3.v2; int v616 = v13.case3.v3; static_array<int,2l> v617 = v13.case3.v4; int v618 = v13.case3.v5; Union4 v619 = v13.case3.v6;
                        Union2 v620;
                        v620 = Union2{Union2_1{v616, v619}};
                        v9.push(v620);
                        Union5 v706;
                        switch (v613.tag) {
                            case 0: { // None
                                switch (v619.tag) {
                                    case 0: { // Call
                                        if (v614){
                                            bool v670;
                                            v670 = v616 == 0l;
                                            int v671;
                                            if (v670){
                                                v671 = 1l;
                                            } else {
                                                v671 = 0l;
                                            }
                                            v706 = Union5{Union5_2{v613, false, v615, v671, v617, v618}};
                                        } else {
                                            v706 = Union5{Union5_0{v613, v614, v615, v616, v617, v618}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v706 = Union5{Union5_5{v613, v614, v615, v616, v617, v618}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v675;
                                        v675 = v618 > 0l;
                                        if (v675){
                                            bool v676;
                                            v676 = v616 == 0l;
                                            int v677;
                                            if (v676){
                                                v677 = 1l;
                                            } else {
                                                v677 = 0l;
                                            }
                                            int v678;
                                            v678 = -1l + v618;
                                            int v679; int v680;
                                            Tuple1 tmp24 = Tuple1{0l, 0l};
                                            v679 = tmp24.v0; v680 = tmp24.v1;
                                            while (while_method_0(v679)){
                                                int v682;
                                                v682 = v617[v679];
                                                bool v684;
                                                v684 = v680 >= v682;
                                                int v685;
                                                if (v684){
                                                    v685 = v680;
                                                } else {
                                                    v685 = v682;
                                                }
                                                v680 = v685;
                                                v679 += 1l ;
                                            }
                                            static_array<int,2l> v686;
                                            int v688;
                                            v688 = 0l;
                                            while (while_method_0(v688)){
                                                v686[v688] = v680;
                                                v688 += 1l ;
                                            }
                                            static_array<int,2l> v690;
                                            int v692;
                                            v692 = 0l;
                                            while (while_method_0(v692)){
                                                int v694;
                                                v694 = v686[v692];
                                                bool v696;
                                                v696 = v692 == v616;
                                                int v698;
                                                if (v696){
                                                    int v697;
                                                    v697 = v694 + 2l;
                                                    v698 = v697;
                                                } else {
                                                    v698 = v694;
                                                }
                                                v690[v692] = v698;
                                                v692 += 1l ;
                                            }
                                            v706 = Union5{Union5_2{v613, false, v615, v677, v690, v678}};
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
                                Union3 v621 = v613.case1.v0;
                                switch (v619.tag) {
                                    case 0: { // Call
                                        if (v614){
                                            bool v623;
                                            v623 = v616 == 0l;
                                            int v624;
                                            if (v623){
                                                v624 = 1l;
                                            } else {
                                                v624 = 0l;
                                            }
                                            v706 = Union5{Union5_2{v613, false, v615, v624, v617, v618}};
                                        } else {
                                            int v626; int v627;
                                            Tuple1 tmp25 = Tuple1{0l, 0l};
                                            v626 = tmp25.v0; v627 = tmp25.v1;
                                            while (while_method_0(v626)){
                                                int v629;
                                                v629 = v617[v626];
                                                bool v631;
                                                v631 = v627 >= v629;
                                                int v632;
                                                if (v631){
                                                    v632 = v627;
                                                } else {
                                                    v632 = v629;
                                                }
                                                v627 = v632;
                                                v626 += 1l ;
                                            }
                                            static_array<int,2l> v633;
                                            int v635;
                                            v635 = 0l;
                                            while (while_method_0(v635)){
                                                v633[v635] = v627;
                                                v635 += 1l ;
                                            }
                                            v706 = Union5{Union5_4{v613, v614, v615, v616, v633, v618}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v706 = Union5{Union5_5{v613, v614, v615, v616, v617, v618}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v639;
                                        v639 = v618 > 0l;
                                        if (v639){
                                            bool v640;
                                            v640 = v616 == 0l;
                                            int v641;
                                            if (v640){
                                                v641 = 1l;
                                            } else {
                                                v641 = 0l;
                                            }
                                            int v642;
                                            v642 = -1l + v618;
                                            int v643; int v644;
                                            Tuple1 tmp26 = Tuple1{0l, 0l};
                                            v643 = tmp26.v0; v644 = tmp26.v1;
                                            while (while_method_0(v643)){
                                                int v646;
                                                v646 = v617[v643];
                                                bool v648;
                                                v648 = v644 >= v646;
                                                int v649;
                                                if (v648){
                                                    v649 = v644;
                                                } else {
                                                    v649 = v646;
                                                }
                                                v644 = v649;
                                                v643 += 1l ;
                                            }
                                            static_array<int,2l> v650;
                                            int v652;
                                            v652 = 0l;
                                            while (while_method_0(v652)){
                                                v650[v652] = v644;
                                                v652 += 1l ;
                                            }
                                            static_array<int,2l> v654;
                                            int v656;
                                            v656 = 0l;
                                            while (while_method_0(v656)){
                                                int v658;
                                                v658 = v650[v656];
                                                bool v660;
                                                v660 = v656 == v616;
                                                int v662;
                                                if (v660){
                                                    int v661;
                                                    v661 = v658 + 4l;
                                                    v662 = v661;
                                                } else {
                                                    v662 = v658;
                                                }
                                                v654[v656] = v662;
                                                v656 += 1l ;
                                            }
                                            v706 = Union5{Union5_2{v613, false, v615, v641, v654, v642}};
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
                        v757 = Union7{Union7_1{v706}};
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
                        v757 = Union7{Union7_0{}};
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
                        v757 = Union7{Union7_0{}};
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
        v11 = v757;
    }
    return v7;
}
__device__ inline bool while_method_9(int v0){
    bool v1;
    v1 = v0 > 0l;
    return v1;
}
__device__ inline bool while_method_10(int v0){
    bool v1;
    v1 = v0 < 512l;
    return v1;
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1, unsigned long long v2, unsigned char * v3, unsigned long long v4, float * v5, float * v6, float * v7) {
    bool v8;
    v8 = 2101776ull == v4;
    bool v9;
    v9 = v8 == false;
    if (v9){
        assert("The params needs to have matching offsets." && v8);
    } else {
    }
    bool v11;
    v11 = 131584ull == v2;
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
    v18 = v17 * 32l;
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
                while (while_method_0(v31)){
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
                    while (while_method_4(v42)){
                        double * v44;
                        v44 = reinterpret_cast<double *>(&v3[2097168ull]);
                        double * v46;
                        v46 = reinterpret_cast<double *>(&v3[2099216ull]);
                        int * v48;
                        v48 = reinterpret_cast<int *>(&v3[2101264ull]);
                        int v50;
                        v50 = threadIdx.x;
                        int v51;
                        v51 = blockIdx.x;
                        int v52;
                        v52 = v51 * 32l;
                        int v53;
                        v53 = v50 + v52;
                        assert("Tensor range check" && 0 <= v42 && v42 < 4l);
                        assert("Tensor range check" && 0 <= v53 && v53 < 32l);
                        int v54;
                        v54 = 2l * v53;
                        int v55;
                        v55 = 64l * v42;
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
                        assert("Tensor range check" && 0 <= v29 && v29 < 2l);
                        assert("Tensor range check" && 0 <= v31 && v31 < 2l);
                        assert("Tensor range check" && 0 <= v42 && v42 < 4l);
                        int v63;
                        v63 = 4l * v31;
                        int v64;
                        v64 = v63 + v42;
                        int v65;
                        v65 = 8l * v29;
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
                    v73 = reinterpret_cast<unsigned int *>(&v1[32768ull]);
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
                    v91 = reinterpret_cast<int *>(&v1[33280ull]);
                    float * v93;
                    v93 = reinterpret_cast<float *>(&v1[41472ull]);
                    int * v95;
                    v95 = reinterpret_cast<int *>(&v1[49664ull]);
                    int * v97;
                    v97 = reinterpret_cast<int *>(&v1[57856ull]);
                    double * v99;
                    v99 = reinterpret_cast<double *>(&v1[66048ull]);
                    double * v101;
                    v101 = reinterpret_cast<double *>(&v1[98816ull]);
                    double * v103;
                    v103 = reinterpret_cast<double *>(&v3[2097168ull]);
                    double * v105;
                    v105 = reinterpret_cast<double *>(&v3[2099216ull]);
                    int * v107;
                    v107 = reinterpret_cast<int *>(&v3[2101264ull]);
                    int v109;
                    v109 = 0l;
                    while (while_method_4(v109)){
                        int v111;
                        v111 = threadIdx.x;
                        int v112;
                        v112 = blockIdx.x;
                        int v113;
                        v113 = v112 * 32l;
                        int v114;
                        v114 = v111 + v113;
                        float v115[2l];
                        int v116;
                        v116 = 0l;
                        while (while_method_0(v116)){
                            float v118;
                            v118 = v39[v116];
                            v115[v116] = v118;
                            v116 += 1l ;
                        }
                        assert("Tensor range check" && 0 <= v109 && v109 < 4l);
                        assert("Tensor range check" && 0 <= v114 && v114 < 32l);
                        int v120;
                        v120 = 32l * v109;
                        int v121;
                        v121 = v120 + v114;
                        int v122;
                        v122 = v107[v121];
                        int v123;
                        v123 = v122;
                        while (while_method_9(v123)){
                            v123 -= 1l ;
                            assert("Tensor range check" && 0 <= v109 && v109 < 4l);
                            assert("Tensor range check" && 0 <= v123 && v123 < 16l);
                            assert("Tensor range check" && 0 <= v114 && v114 < 32l);
                            int v125;
                            v125 = 32l * v123;
                            int v126;
                            v126 = v125 + v114;
                            int v127;
                            v127 = 512l * v109;
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
                            v151 = 1024l * v109;
                            assert("Tensor range check" && 0 <= v123 && v123 < 16l);
                            int v152;
                            v152 = 64l * v123;
                            int v153;
                            v153 = v152 + v151;
                            assert("Tensor range check" && 0 <= v114 && v114 < 32l);
                            int v154;
                            v154 = 2l * v114;
                            int v155;
                            v155 = v154 + v153;
                            double v156[2l];
                            int v157;
                            v157 = 0l;
                            while (while_method_0(v157)){
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
                            while (while_method_0(v164)){
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
                            while (while_method_0(v169)){
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
                            __shared__ float v190[32l];
                            __shared__ int v191[32l];
                            __shared__ float v192[32l];
                            __shared__ float v193[32l];
                            __shared__ float * v194[32l];
                            __shared__ float * v195[32l];
                            __shared__ float * v196[32l];
                            __shared__ float * v197[32l];
                            /* void shared array create v198 */;
                            __shared__ float v199[32l];
                            int v200;
                            v200 = threadIdx.x;
                            assert("Tensor range check" && 0 <= v200 && v200 < 32l);
                            v190[v200] = v130;
                            v191[v200] = v129;
                            v192[v200] = v133;
                            v193[v200] = v176;
                            v194[v200] = v141;
                            v195[v200] = v184;
                            v196[v200] = v186;
                            v197[v200] = v188;
                            /* void array set */;
                            asm("barrier.cta.sync %0;" :: "r"(0l));
                            bool v201;
                            v201 = 0l <= v200;
                            bool v202;
                            v202 = v201 == false;
                            if (v202){
                                assert("The index needs to be zero or positive." && v201);
                            } else {
                            }
                            int v204;
                            v204 = v200 % 1l;
                            bool v205;
                            v205 = v200 < 32l;
                            bool v206;
                            v206 = v205 == false;
                            if (v206){
                                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v205);
                            } else {
                            }
                            assert("Tensor range check" && 0 <= v200 && v200 < 32l);
                            int v208;
                            v208 = 0l;
                            while (while_method_6(v208)){
                                bool v210;
                                v210 = v201 && v205;
                                bool v211;
                                v211 = v210 == false;
                                if (v211){
                                    assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v210);
                                } else {
                                }
                                bool v213;
                                v213 = 0l <= v208;
                                bool v215;
                                if (v213){
                                    bool v214;
                                    v214 = v208 < 1l;
                                    v215 = v214;
                                } else {
                                    v215 = false;
                                }
                                bool v216;
                                v216 = v215 == false;
                                if (v216){
                                    assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v215);
                                } else {
                                }
                                int v218;
                                v218 = v208 * 32l;
                                int v219;
                                v219 = v218 + v200;
                                assert("Tensor range check" && 0 <= v208 && v208 < 1l);
                                int v220;
                                v220 = 32l * v208;
                                int v221;
                                v221 = v220 + v200;
                                float v222;
                                v222 = v190[v221];
                                int v223;
                                v223 = v191[v221];
                                float v224;
                                v224 = v192[v221];
                                float v225;
                                v225 = v193[v221];
                                float * v226;
                                v226 = v194[v221];
                                float * v227;
                                v227 = v195[v221];
                                float * v228;
                                v228 = v196[v221];
                                float * v229;
                                v229 = v197[v221];
                                /* void array index */;
                                int v230;
                                v230 = blockIdx.x;
                                int v231;
                                v231 = v230 * 32l;
                                int v232;
                                v232 = v231 + v219;
                                assert("Tensor range check" && 0 <= v204 && v204 < 1l);
                                int v233;
                                v233 = 4l * v204;
                                float v234[4l];
                                float v235[4l];
                                float v236[4l];
                                int v237[4l];
                                int v238;
                                v238 = 0l;
                                while (while_method_6(v238)){
                                    assert("Tensor range check" && 0 <= v238 && v238 < 1l);
                                    int v240;
                                    v240 = 4l * v238;
                                    assert("Tensor range check" && 0 <= v238 && v238 < 1l);
                                    int v241;
                                    v241 = v240 + v233;
                                    int4* v242;
                                    v242 = reinterpret_cast<int4*>(v227 + v241);
                                    int4* v243;
                                    v243 = reinterpret_cast<int4*>(v234 + v240);
                                    assert("Pointer alignment check" && (unsigned long long)(v242) % 4l == 0 && (unsigned long long)(v243) % 4l == 0);
                                    *v243 = *v242;
                                    int4* v244;
                                    v244 = reinterpret_cast<int4*>(v228 + v241);
                                    int4* v245;
                                    v245 = reinterpret_cast<int4*>(v235 + v240);
                                    assert("Pointer alignment check" && (unsigned long long)(v244) % 4l == 0 && (unsigned long long)(v245) % 4l == 0);
                                    *v245 = *v244;
                                    int4* v246;
                                    v246 = reinterpret_cast<int4*>(v229 + v241);
                                    int4* v247;
                                    v247 = reinterpret_cast<int4*>(v236 + v240);
                                    assert("Pointer alignment check" && (unsigned long long)(v246) % 4l == 0 && (unsigned long long)(v247) % 4l == 0);
                                    *v247 = *v246;
                                    v238 += 1l ;
                                }
                                int v248;
                                v248 = 0l;
                                while (while_method_6(v248)){
                                    int v250;
                                    v250 = 0l;
                                    while (while_method_4(v250)){
                                        bool v252;
                                        v252 = 0l <= v250;
                                        bool v254;
                                        if (v252){
                                            bool v253;
                                            v253 = v250 < 4l;
                                            v254 = v253;
                                        } else {
                                            v254 = false;
                                        }
                                        bool v255;
                                        v255 = v254 == false;
                                        if (v255){
                                            assert("The indices should be inside the range of the dimension." && v254);
                                        } else {
                                        }
                                        bool v257;
                                        v257 = 0l <= v204;
                                        bool v259;
                                        if (v257){
                                            bool v258;
                                            v258 = v204 < 1l;
                                            v259 = v258;
                                        } else {
                                            v259 = false;
                                        }
                                        bool v260;
                                        v260 = v259 == false;
                                        if (v260){
                                            assert("The indices should be inside the range of the dimension." && v259);
                                        } else {
                                        }
                                        int v262;
                                        v262 = v204 * 4l;
                                        int v263;
                                        v263 = v250 + v262;
                                        bool v264;
                                        v264 = 0l <= v248;
                                        bool v266;
                                        if (v264){
                                            bool v265;
                                            v265 = v248 < 1l;
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
                                        v269 = v248 * 4l;
                                        int v270;
                                        v270 = v263 + v269;
                                        assert("Tensor range check" && 0 <= v248 && v248 < 1l);
                                        assert("Tensor range check" && 0 <= v250 && v250 < 4l);
                                        int v271;
                                        v271 = 4l * v248;
                                        int v272;
                                        v272 = v271 + v250;
                                        v237[v272] = v270;
                                        v250 += 1l ;
                                    }
                                    v248 += 1l ;
                                }
                                float v273[4l];
                                int v274;
                                v274 = 0l;
                                while (while_method_6(v274)){
                                    int v276;
                                    v276 = 0l;
                                    while (while_method_4(v276)){
                                        assert("Tensor range check" && 0 <= v274 && v274 < 1l);
                                        assert("Tensor range check" && 0 <= v276 && v276 < 4l);
                                        int v278;
                                        v278 = 4l * v274;
                                        int v279;
                                        v279 = v278 + v276;
                                        float v280;
                                        v280 = v235[v279];
                                        float v281;
                                        v281 = v236[v279];
                                        bool v282;
                                        v282 = v281 == 0.0f;
                                        bool v283;
                                        v283 = v282 != true;
                                        float v285;
                                        if (v283){
                                            float v284;
                                            v284 = v280 / v281;
                                            v285 = v284;
                                        } else {
                                            v285 = 0.0f;
                                        }
                                        assert("Tensor range check" && 0 <= v274 && v274 < 1l);
                                        assert("Tensor range check" && 0 <= v276 && v276 < 4l);
                                        v273[v279] = v285;
                                        v276 += 1l ;
                                    }
                                    v274 += 1l ;
                                }
                                bool v286[4l];
                                int v287;
                                v287 = 0l;
                                while (while_method_6(v287)){
                                    int v289;
                                    v289 = 0l;
                                    while (while_method_4(v289)){
                                        assert("Tensor range check" && 0 <= v287 && v287 < 1l);
                                        assert("Tensor range check" && 0 <= v289 && v289 < 4l);
                                        int v291;
                                        v291 = 4l * v287;
                                        int v292;
                                        v292 = v291 + v289;
                                        float v293;
                                        v293 = v234[v292];
                                        int v294;
                                        v294 = v237[v292];
                                        bool v295;
                                        v295 = v294 < 3l;
                                        assert("Tensor range check" && 0 <= v287 && v287 < 1l);
                                        assert("Tensor range check" && 0 <= v289 && v289 < 4l);
                                        v286[v292] = v295;
                                        v289 += 1l ;
                                    }
                                    v287 += 1l ;
                                }
                                float v296[4l];
                                int v297;
                                v297 = 0l;
                                while (while_method_6(v297)){
                                    int v299;
                                    v299 = 0l;
                                    while (while_method_4(v299)){
                                        assert("Tensor range check" && 0 <= v297 && v297 < 1l);
                                        assert("Tensor range check" && 0 <= v299 && v299 < 4l);
                                        int v301;
                                        v301 = 4l * v297;
                                        int v302;
                                        v302 = v301 + v299;
                                        float v303;
                                        v303 = v234[v302];
                                        bool v304;
                                        v304 = v286[v302];
                                        float v307;
                                        if (v304){
                                            bool v305;
                                            v305 = 0.0f >= v303;
                                            if (v305){
                                                v307 = 0.0f;
                                            } else {
                                                v307 = v303;
                                            }
                                        } else {
                                            v307 = 0.0f;
                                        }
                                        assert("Tensor range check" && 0 <= v297 && v297 < 1l);
                                        assert("Tensor range check" && 0 <= v299 && v299 < 4l);
                                        v296[v302] = v307;
                                        v299 += 1l ;
                                    }
                                    v297 += 1l ;
                                }
                                float v308;
                                v308 = 0.0f;
                                int v309;
                                v309 = 0l;
                                while (while_method_6(v309)){
                                    int v311;
                                    v311 = 0l;
                                    while (while_method_4(v311)){
                                        assert("Tensor range check" && 0 <= v309 && v309 < 1l);
                                        assert("Tensor range check" && 0 <= v311 && v311 < 4l);
                                        int v313;
                                        v313 = 4l * v309;
                                        int v314;
                                        v314 = v313 + v311;
                                        float v315;
                                        v315 = v296[v314];
                                        float v316;
                                        v316 = v308 + v315;
                                        v308 = v316;
                                        v311 += 1l ;
                                    }
                                    v309 += 1l ;
                                }
                                auto v317 = cooperative_groups::coalesced_threads();
                                int v318;
                                v318 = threadIdx.x;
                                auto v319 = cooperative_groups::labeled_partition(v317,v318);
                                Closure1 v320{};
                                float v321;
                                v321 = cooperative_groups::reduce(v319, v308, v320);
                                int v322[4l];
                                int v323;
                                v323 = 0l;
                                while (while_method_6(v323)){
                                    int v325;
                                    v325 = 0l;
                                    while (while_method_4(v325)){
                                        assert("Tensor range check" && 0 <= v323 && v323 < 1l);
                                        assert("Tensor range check" && 0 <= v325 && v325 < 4l);
                                        int v327;
                                        v327 = 4l * v323;
                                        int v328;
                                        v328 = v327 + v325;
                                        bool v329;
                                        v329 = v286[v328];
                                        int v330;
                                        if (v329){
                                            v330 = 1l;
                                        } else {
                                            v330 = 0l;
                                        }
                                        assert("Tensor range check" && 0 <= v323 && v323 < 1l);
                                        assert("Tensor range check" && 0 <= v325 && v325 < 4l);
                                        v322[v328] = v330;
                                        v325 += 1l ;
                                    }
                                    v323 += 1l ;
                                }
                                int v331;
                                v331 = 0l;
                                int v332;
                                v332 = 0l;
                                while (while_method_6(v332)){
                                    int v334;
                                    v334 = 0l;
                                    while (while_method_4(v334)){
                                        assert("Tensor range check" && 0 <= v332 && v332 < 1l);
                                        assert("Tensor range check" && 0 <= v334 && v334 < 4l);
                                        int v336;
                                        v336 = 4l * v332;
                                        int v337;
                                        v337 = v336 + v334;
                                        int v338;
                                        v338 = v322[v337];
                                        int v339;
                                        v339 = v331 + v338;
                                        v331 = v339;
                                        v334 += 1l ;
                                    }
                                    v332 += 1l ;
                                }
                                auto v340 = cooperative_groups::coalesced_threads();
                                int v341;
                                v341 = threadIdx.x;
                                auto v342 = cooperative_groups::labeled_partition(v340,v341);
                                Closure2 v343{};
                                int v344;
                                v344 = cooperative_groups::reduce(v342, v331, v343);
                                float v345;
                                v345 = (float)v344;
                                float v346;
                                v346 = 1.0f / v345;
                                float v347[4l];
                                int v348;
                                v348 = 0l;
                                while (while_method_6(v348)){
                                    int v350;
                                    v350 = 0l;
                                    while (while_method_4(v350)){
                                        assert("Tensor range check" && 0 <= v348 && v348 < 1l);
                                        assert("Tensor range check" && 0 <= v350 && v350 < 4l);
                                        int v352;
                                        v352 = 4l * v348;
                                        int v353;
                                        v353 = v352 + v350;
                                        float v354;
                                        v354 = v296[v353];
                                        bool v355;
                                        v355 = v286[v353];
                                        bool v356;
                                        v356 = v355 == false;
                                        float v361;
                                        if (v356){
                                            v361 = 0.0f;
                                        } else {
                                            bool v357;
                                            v357 = v321 == 0.0f;
                                            bool v358;
                                            v358 = v357 != true;
                                            if (v358){
                                                float v359;
                                                v359 = v354 / v321;
                                                v361 = v359;
                                            } else {
                                                v361 = v346;
                                            }
                                        }
                                        assert("Tensor range check" && 0 <= v348 && v348 < 1l);
                                        assert("Tensor range check" && 0 <= v350 && v350 < 4l);
                                        v347[v353] = v361;
                                        v350 += 1l ;
                                    }
                                    v348 += 1l ;
                                }
                                float v362[4l];
                                int v363;
                                v363 = 0l;
                                while (while_method_6(v363)){
                                    int v365;
                                    v365 = 0l;
                                    while (while_method_4(v365)){
                                        assert("Tensor range check" && 0 <= v363 && v363 < 1l);
                                        assert("Tensor range check" && 0 <= v365 && v365 < 4l);
                                        int v367;
                                        v367 = 4l * v363;
                                        int v368;
                                        v368 = v367 + v365;
                                        float v369;
                                        v369 = v273[v368];
                                        int v370;
                                        v370 = v237[v368];
                                        bool v371;
                                        v371 = v223 == v370;
                                        float v374;
                                        if (v371){
                                            float v372;
                                            v372 = v224 - v369;
                                            float v373;
                                            v373 = v372 / v222;
                                            v374 = v373;
                                        } else {
                                            v374 = 0.0f;
                                        }
                                        float v375;
                                        v375 = v374 + v369;
                                        assert("Tensor range check" && 0 <= v363 && v363 < 1l);
                                        assert("Tensor range check" && 0 <= v365 && v365 < 4l);
                                        v362[v368] = v375;
                                        v365 += 1l ;
                                    }
                                    v363 += 1l ;
                                }
                                float v376[4l];
                                int v377;
                                v377 = 0l;
                                while (while_method_6(v377)){
                                    int v379;
                                    v379 = 0l;
                                    while (while_method_4(v379)){
                                        assert("Tensor range check" && 0 <= v377 && v377 < 1l);
                                        assert("Tensor range check" && 0 <= v379 && v379 < 4l);
                                        int v381;
                                        v381 = 4l * v377;
                                        int v382;
                                        v382 = v381 + v379;
                                        float v383;
                                        v383 = v347[v382];
                                        float v384;
                                        v384 = v362[v382];
                                        float v385;
                                        v385 = v383 * v384;
                                        assert("Tensor range check" && 0 <= v377 && v377 < 1l);
                                        assert("Tensor range check" && 0 <= v379 && v379 < 4l);
                                        v376[v382] = v385;
                                        v379 += 1l ;
                                    }
                                    v377 += 1l ;
                                }
                                float v386;
                                v386 = 0.0f;
                                int v387;
                                v387 = 0l;
                                while (while_method_6(v387)){
                                    int v389;
                                    v389 = 0l;
                                    while (while_method_4(v389)){
                                        assert("Tensor range check" && 0 <= v387 && v387 < 1l);
                                        assert("Tensor range check" && 0 <= v389 && v389 < 4l);
                                        int v391;
                                        v391 = 4l * v387;
                                        int v392;
                                        v392 = v391 + v389;
                                        float v393;
                                        v393 = v376[v392];
                                        float v394;
                                        v394 = v386 + v393;
                                        v386 = v394;
                                        v389 += 1l ;
                                    }
                                    v387 += 1l ;
                                }
                                auto v395 = cooperative_groups::coalesced_threads();
                                int v396;
                                v396 = threadIdx.x;
                                auto v397 = cooperative_groups::labeled_partition(v395,v396);
                                float v398;
                                v398 = cooperative_groups::reduce(v397, v386, v320);
                                int v399;
                                v399 = 0l;
                                while (while_method_6(v399)){
                                    int v401;
                                    v401 = 0l;
                                    while (while_method_4(v401)){
                                        assert("Tensor range check" && 0 <= v399 && v399 < 1l);
                                        assert("Tensor range check" && 0 <= v401 && v401 < 4l);
                                        int v403;
                                        v403 = 4l * v399;
                                        int v404;
                                        v404 = v403 + v401;
                                        float v405;
                                        v405 = v362[v404];
                                        int v406;
                                        v406 = v237[v404];
                                        float v407;
                                        v407 = v405 - v398;
                                        float v408;
                                        v408 = v225 * v407;
                                        assert("Tensor range check" && 0 <= v406 && v406 < 4l);
                                        float * v409;
                                        v409 = v226+v406;
                                        float v411;
                                        v411 = atomicAdd(v409,v408);
                                        v401 += 1l ;
                                    }
                                    v399 += 1l ;
                                }
                                int v412;
                                v412 = 0l;
                                while (while_method_6(v412)){
                                    assert("Tensor range check" && 0 <= v412 && v412 < 1l);
                                    assert("Tensor range check" && 0 <= v412 && v412 < 1l);
                                    v412 += 1l ;
                                }
                                assert("Tensor range check" && 0 <= v219 && v219 < 32l);
                                v199[v219] = v398;
                                v208 += 1l ;
                            }
                            asm("barrier.cta.sync %0;" :: "r"(0l));
                            assert("Tensor range check" && 0 <= v200 && v200 < 32l);
                            float v414;
                            v414 = v199[v200];
                            asm("barrier.cta.sync %0;" :: "r"(0l));
                            assert("Tensor range check" && 0 <= v131 && v131 < 2l);
                            v115[v131] = v414;
                        }
                        assert("Tensor range check" && 0 <= v109 && v109 < 4l);
                        int v415;
                        v415 = 64l * v109;
                        assert("Tensor range check" && 0 <= v114 && v114 < 32l);
                        int v416;
                        v416 = 2l * v114;
                        int v417;
                        v417 = v416 + v415;
                        double * v418;
                        v418 = v103+v417;
                        double * v420;
                        v420 = v105+v417;
                        double * v422;
                        v422 = v418+0l;
                        double * v424;
                        v424 = v420+0l;
                        double * v426;
                        v426 = v418+0l;
                        double * v428;
                        v428 = v420+0l;
                        __shared__ double * v430[32l];
                        __shared__ double * v431[32l];
                        __shared__ double * v432[32l];
                        __shared__ double * v433[32l];
                        /* void shared array create v434 */;
                        /* void shared array create v435 */;
                        int v436;
                        v436 = threadIdx.x;
                        assert("Tensor range check" && 0 <= v436 && v436 < 32l);
                        v430[v436] = v422;
                        v431[v436] = v424;
                        v432[v436] = v426;
                        v433[v436] = v428;
                        /* void array set */;
                        asm("barrier.cta.sync %0;" :: "r"(0l));
                        bool v437;
                        v437 = 0l <= v436;
                        bool v438;
                        v438 = v437 == false;
                        if (v438){
                            assert("The index needs to be zero or positive." && v437);
                        } else {
                        }
                        int v440;
                        v440 = v436 % 1l;
                        bool v441;
                        v441 = v436 < 32l;
                        bool v442;
                        v442 = v441 == false;
                        if (v442){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v441);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v436 && v436 < 32l);
                        int v444;
                        v444 = 0l;
                        while (while_method_6(v444)){
                            bool v446;
                            v446 = v437 && v441;
                            bool v447;
                            v447 = v446 == false;
                            if (v447){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v446);
                            } else {
                            }
                            bool v449;
                            v449 = 0l <= v444;
                            bool v451;
                            if (v449){
                                bool v450;
                                v450 = v444 < 1l;
                                v451 = v450;
                            } else {
                                v451 = false;
                            }
                            bool v452;
                            v452 = v451 == false;
                            if (v452){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v451);
                            } else {
                            }
                            int v454;
                            v454 = v444 * 32l;
                            int v455;
                            v455 = v454 + v436;
                            assert("Tensor range check" && 0 <= v444 && v444 < 1l);
                            int v456;
                            v456 = 32l * v444;
                            int v457;
                            v457 = v456 + v436;
                            double * v458;
                            v458 = v430[v457];
                            double * v459;
                            v459 = v431[v457];
                            double * v460;
                            v460 = v432[v457];
                            double * v461;
                            v461 = v433[v457];
                            /* void array index */;
                            int v462;
                            v462 = blockIdx.x;
                            int v463;
                            v463 = v462 * 32l;
                            int v464;
                            v464 = v463 + v455;
                            assert("Tensor range check" && 0 <= v440 && v440 < 1l);
                            int v465;
                            v465 = 2l * v440;
                            double v466[2l];
                            double v467[2l];
                            int v468[2l];
                            int v469;
                            v469 = 0l;
                            while (while_method_6(v469)){
                                assert("Tensor range check" && 0 <= v469 && v469 < 1l);
                                int v471;
                                v471 = 2l * v469;
                                assert("Tensor range check" && 0 <= v469 && v469 < 1l);
                                int v472;
                                v472 = v471 + v465;
                                int4* v473;
                                v473 = reinterpret_cast<int4*>(v458 + v472);
                                int4* v474;
                                v474 = reinterpret_cast<int4*>(v466 + v471);
                                assert("Pointer alignment check" && (unsigned long long)(v473) % 2l == 0 && (unsigned long long)(v474) % 2l == 0);
                                *v474 = *v473;
                                int4* v475;
                                v475 = reinterpret_cast<int4*>(v459 + v472);
                                int4* v476;
                                v476 = reinterpret_cast<int4*>(v467 + v471);
                                assert("Pointer alignment check" && (unsigned long long)(v475) % 2l == 0 && (unsigned long long)(v476) % 2l == 0);
                                *v476 = *v475;
                                v469 += 1l ;
                            }
                            int v477;
                            v477 = 0l;
                            while (while_method_6(v477)){
                                int v479;
                                v479 = 0l;
                                while (while_method_0(v479)){
                                    bool v481;
                                    v481 = 0l <= v479;
                                    bool v483;
                                    if (v481){
                                        bool v482;
                                        v482 = v479 < 2l;
                                        v483 = v482;
                                    } else {
                                        v483 = false;
                                    }
                                    bool v484;
                                    v484 = v483 == false;
                                    if (v484){
                                        assert("The indices should be inside the range of the dimension." && v483);
                                    } else {
                                    }
                                    bool v486;
                                    v486 = 0l <= v440;
                                    bool v488;
                                    if (v486){
                                        bool v487;
                                        v487 = v440 < 1l;
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
                                    v491 = v440 * 2l;
                                    int v492;
                                    v492 = v479 + v491;
                                    bool v493;
                                    v493 = 0l <= v477;
                                    bool v495;
                                    if (v493){
                                        bool v494;
                                        v494 = v477 < 1l;
                                        v495 = v494;
                                    } else {
                                        v495 = false;
                                    }
                                    bool v496;
                                    v496 = v495 == false;
                                    if (v496){
                                        assert("The indices should be inside the range of the dimension." && v495);
                                    } else {
                                    }
                                    int v498;
                                    v498 = v477 * 2l;
                                    int v499;
                                    v499 = v492 + v498;
                                    assert("Tensor range check" && 0 <= v477 && v477 < 1l);
                                    assert("Tensor range check" && 0 <= v479 && v479 < 2l);
                                    int v500;
                                    v500 = 2l * v477;
                                    int v501;
                                    v501 = v500 + v479;
                                    v468[v501] = v499;
                                    v479 += 1l ;
                                }
                                v477 += 1l ;
                            }
                            double v502[2l];
                            double v503[2l];
                            int v504;
                            v504 = 0l;
                            while (while_method_6(v504)){
                                int v506;
                                v506 = 0l;
                                while (while_method_0(v506)){
                                    assert("Tensor range check" && 0 <= v504 && v504 < 1l);
                                    assert("Tensor range check" && 0 <= v506 && v506 < 2l);
                                    int v508;
                                    v508 = 2l * v504;
                                    int v509;
                                    v509 = v508 + v506;
                                    double v510;
                                    v510 = v466[v509];
                                    double v511;
                                    v511 = v467[v509];
                                    assert("Tensor range check" && 0 <= v504 && v504 < 1l);
                                    assert("Tensor range check" && 0 <= v506 && v506 < 2l);
                                    v502[v509] = 0.0;
                                    v503[v509] = 0.0;
                                    v506 += 1l ;
                                }
                                v504 += 1l ;
                            }
                            int v512;
                            v512 = 0l;
                            while (while_method_6(v512)){
                                assert("Tensor range check" && 0 <= v512 && v512 < 1l);
                                int v514;
                                v514 = 2l * v512;
                                int v515;
                                v515 = v514 + v465;
                                assert("Tensor range check" && 0 <= v512 && v512 < 1l);
                                int4* v516;
                                v516 = reinterpret_cast<int4*>(v502 + v514);
                                int4* v517;
                                v517 = reinterpret_cast<int4*>(v460 + v515);
                                assert("Pointer alignment check" && (unsigned long long)(v516) % 2l == 0 && (unsigned long long)(v517) % 2l == 0);
                                *v517 = *v516;
                                int4* v518;
                                v518 = reinterpret_cast<int4*>(v503 + v514);
                                int4* v519;
                                v519 = reinterpret_cast<int4*>(v461 + v515);
                                assert("Pointer alignment check" && (unsigned long long)(v518) % 2l == 0 && (unsigned long long)(v519) % 2l == 0);
                                *v519 = *v518;
                                v512 += 1l ;
                            }
                            assert("Tensor range check" && 0 <= v455 && v455 < 32l);
                            /* void array set */;
                            v444 += 1l ;
                        }
                        asm("barrier.cta.sync %0;" :: "r"(0l));
                        assert("Tensor range check" && 0 <= v436 && v436 < 32l);
                        /* void array index */;
                        asm("barrier.cta.sync %0;" :: "r"(0l));
                        assert("Tensor range check" && 0 <= v109 && v109 < 4l);
                        assert("Tensor range check" && 0 <= v114 && v114 < 32l);
                        v107[v121] = 0l;
                        v109 += 1l ;
                    }
                    v31 += 1l ;
                }
                unsigned int * v520;
                v520 = reinterpret_cast<unsigned int *>(&v1[32768ull]);
                int * v522;
                v522 = reinterpret_cast<int *>(&v3[262144ull]);
                float * v524;
                v524 = reinterpret_cast<float *>(&v3[262160ull]);
                float * v526;
                v526 = reinterpret_cast<float *>(&v3[524304ull]);
                float * v528;
                v528 = reinterpret_cast<float *>(&v3[786448ull]);
                float * v530;
                v530 = reinterpret_cast<float *>(&v3[1048592ull]);
                float * v532;
                v532 = reinterpret_cast<float *>(&v3[1310736ull]);
                float * v534;
                v534 = reinterpret_cast<float *>(&v3[1572880ull]);
                float * v536;
                v536 = reinterpret_cast<float *>(&v3[1835024ull]);
                int * v538;
                v538 = reinterpret_cast<int *>(&v1[33280ull]);
                float * v540;
                v540 = reinterpret_cast<float *>(&v1[41472ull]);
                int * v542;
                v542 = reinterpret_cast<int *>(&v1[49664ull]);
                int * v544;
                v544 = reinterpret_cast<int *>(&v1[57856ull]);
                double * v546;
                v546 = reinterpret_cast<double *>(&v1[66048ull]);
                double * v548;
                v548 = reinterpret_cast<double *>(&v1[98816ull]);
                double * v550;
                v550 = reinterpret_cast<double *>(&v3[2097168ull]);
                double * v552;
                v552 = reinterpret_cast<double *>(&v3[2099216ull]);
                int * v554;
                v554 = reinterpret_cast<int *>(&v3[2101264ull]);
                v14.sync() ;
                int v556;
                v556 = threadIdx.x;
                int v557;
                v557 = blockIdx.x;
                int v558;
                v558 = v557 * 32l;
                int v559;
                v559 = v556 + v558;
                bool v560;
                v560 = v559 == 0l;
                if (v560){
                    int v561;
                    v561 = 0l;
                    int v562;
                    v562 = 4l;
                    int v563;
                    v563 = int_range_10(v562, v561, v21);
                    v522[0l] = v563;
                } else {
                }
                __syncwarp();
                int v564;
                v564 = threadIdx.x;
                bool v565;
                v565 = 0l <= v564;
                bool v566;
                v566 = v565 == false;
                if (v566){
                    assert("The index needs to be zero or positive." && v565);
                } else {
                }
                int v568;
                v568 = v564 % 1l;
                int v569;
                v569 = v564 % 32l;
                int v570;
                v570 = v564 / 32l;
                bool v571;
                v571 = v570 < 1l;
                bool v572;
                v572 = v571 == false;
                if (v572){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v571);
                } else {
                }
                assert("Tensor range check" && 0 <= v570 && v570 < 1l);
                assert("Tensor range check" && 0 <= v569 && v569 < 32l);
                assert("Tensor range check" && 0 <= v568 && v568 < 1l);
                int v574;
                v574 = 4l * v568;
                int v575;
                v575 = 4l * v569;
                int v576;
                v576 = v575 + v574;
                int v577;
                v577 = 16384l * v570;
                int v578;
                v578 = v577 + v576;
                assert("Tensor range check" && 0 <= v570 && v570 < 1l);
                assert("Tensor range check" && 0 <= v569 && v569 < 32l);
                assert("Tensor range check" && 0 <= v568 && v568 < 1l);
                int v579;
                v579 = blockIdx.x;
                int v580;
                v580 = v579;
                while (while_method_10(v580)){
                    bool v582;
                    v582 = 0l <= v580;
                    bool v583;
                    v583 = v582 == false;
                    if (v583){
                        assert("The index needs to be zero or positive." && v582);
                    } else {
                    }
                    int v585;
                    v585 = v580 % 128l;
                    int v586;
                    v586 = v580 / 128l;
                    bool v587;
                    v587 = v586 < 4l;
                    bool v588;
                    v588 = v587 == false;
                    if (v588){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v587);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v586 && v586 < 4l);
                    assert("Tensor range check" && 0 <= v585 && v585 < 128l);
                    int v590;
                    v590 = 128l * v585;
                    int v591;
                    v591 = v590 + v578;
                    int v592;
                    v592 = 16384l * v586;
                    int v593;
                    v593 = v592 + v591;
                    float v594[4l];
                    float v595[4l];
                    float v596[4l];
                    float v597[4l];
                    float v598[4l];
                    float v599[4l];
                    float v600[4l];
                    int v601[4l];
                    int v602;
                    v602 = 0l;
                    while (while_method_6(v602)){
                        assert("Tensor range check" && 0 <= v602 && v602 < 1l);
                        int v604;
                        v604 = 4l * v602;
                        assert("Tensor range check" && 0 <= v602 && v602 < 1l);
                        int v605;
                        v605 = v604 + v593;
                        int4* v606;
                        v606 = reinterpret_cast<int4*>(v524 + v605);
                        int4* v607;
                        v607 = reinterpret_cast<int4*>(v594 + v604);
                        assert("Pointer alignment check" && (unsigned long long)(v606) % 4l == 0 && (unsigned long long)(v607) % 4l == 0);
                        *v607 = *v606;
                        int4* v608;
                        v608 = reinterpret_cast<int4*>(v526 + v605);
                        int4* v609;
                        v609 = reinterpret_cast<int4*>(v595 + v604);
                        assert("Pointer alignment check" && (unsigned long long)(v608) % 4l == 0 && (unsigned long long)(v609) % 4l == 0);
                        *v609 = *v608;
                        int4* v610;
                        v610 = reinterpret_cast<int4*>(v528 + v605);
                        int4* v611;
                        v611 = reinterpret_cast<int4*>(v596 + v604);
                        assert("Pointer alignment check" && (unsigned long long)(v610) % 4l == 0 && (unsigned long long)(v611) % 4l == 0);
                        *v611 = *v610;
                        int4* v612;
                        v612 = reinterpret_cast<int4*>(v530 + v605);
                        int4* v613;
                        v613 = reinterpret_cast<int4*>(v597 + v604);
                        assert("Pointer alignment check" && (unsigned long long)(v612) % 4l == 0 && (unsigned long long)(v613) % 4l == 0);
                        *v613 = *v612;
                        int4* v614;
                        v614 = reinterpret_cast<int4*>(v532 + v605);
                        int4* v615;
                        v615 = reinterpret_cast<int4*>(v598 + v604);
                        assert("Pointer alignment check" && (unsigned long long)(v614) % 4l == 0 && (unsigned long long)(v615) % 4l == 0);
                        *v615 = *v614;
                        int4* v616;
                        v616 = reinterpret_cast<int4*>(v534 + v605);
                        int4* v617;
                        v617 = reinterpret_cast<int4*>(v599 + v604);
                        assert("Pointer alignment check" && (unsigned long long)(v616) % 4l == 0 && (unsigned long long)(v617) % 4l == 0);
                        *v617 = *v616;
                        int4* v618;
                        v618 = reinterpret_cast<int4*>(v536 + v605);
                        int4* v619;
                        v619 = reinterpret_cast<int4*>(v600 + v604);
                        assert("Pointer alignment check" && (unsigned long long)(v618) % 4l == 0 && (unsigned long long)(v619) % 4l == 0);
                        *v619 = *v618;
                        v602 += 1l ;
                    }
                    int v620;
                    v620 = 0l;
                    while (while_method_6(v620)){
                        int v622;
                        v622 = 0l;
                        while (while_method_4(v622)){
                            bool v624;
                            v624 = 0l <= v622;
                            bool v626;
                            if (v624){
                                bool v625;
                                v625 = v622 < 4l;
                                v626 = v625;
                            } else {
                                v626 = false;
                            }
                            bool v627;
                            v627 = v626 == false;
                            if (v627){
                                assert("The indices should be inside the range of the dimension." && v626);
                            } else {
                            }
                            bool v629;
                            v629 = 0l <= v568;
                            bool v631;
                            if (v629){
                                bool v630;
                                v630 = v568 < 1l;
                                v631 = v630;
                            } else {
                                v631 = false;
                            }
                            bool v632;
                            v632 = v631 == false;
                            if (v632){
                                assert("The indices should be inside the range of the dimension." && v631);
                            } else {
                            }
                            int v634;
                            v634 = v568 * 4l;
                            int v635;
                            v635 = v622 + v634;
                            bool v636;
                            v636 = 0l <= v620;
                            bool v638;
                            if (v636){
                                bool v637;
                                v637 = v620 < 1l;
                                v638 = v637;
                            } else {
                                v638 = false;
                            }
                            bool v639;
                            v639 = v638 == false;
                            if (v639){
                                assert("The indices should be inside the range of the dimension." && v638);
                            } else {
                            }
                            int v641;
                            v641 = v620 * 4l;
                            int v642;
                            v642 = v635 + v641;
                            assert("Tensor range check" && 0 <= v620 && v620 < 1l);
                            assert("Tensor range check" && 0 <= v622 && v622 < 4l);
                            int v643;
                            v643 = 4l * v620;
                            int v644;
                            v644 = v643 + v622;
                            v601[v644] = v642;
                            v622 += 1l ;
                        }
                        v620 += 1l ;
                    }
                    bool v645;
                    v645 = 0l <= v570;
                    bool v646;
                    v646 = v645 && v571;
                    bool v647;
                    v647 = v646 == false;
                    if (v647){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v646);
                    } else {
                    }
                    bool v649;
                    v649 = 0l <= v569;
                    bool v651;
                    if (v649){
                        bool v650;
                        v650 = v569 < 32l;
                        v651 = v650;
                    } else {
                        v651 = false;
                    }
                    bool v652;
                    v652 = v651 == false;
                    if (v652){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v651);
                    } else {
                    }
                    bool v654;
                    v654 = 0l <= v586;
                    bool v655;
                    v655 = v654 && v587;
                    bool v656;
                    v656 = v655 == false;
                    if (v656){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v655);
                    } else {
                    }
                    bool v658;
                    v658 = 0l <= v585;
                    bool v660;
                    if (v658){
                        bool v659;
                        v659 = v585 < 128l;
                        v660 = v659;
                    } else {
                        v660 = false;
                    }
                    bool v661;
                    v661 = v660 == false;
                    if (v661){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v660);
                    } else {
                    }
                    int v663;
                    v663 = v585 * 32l;
                    int v664;
                    v664 = v586 + v570;
                    int v665;
                    v665 = v663 + v569;
                    float v666[4l];
                    int v667;
                    v667 = 0l;
                    while (while_method_6(v667)){
                        int v669;
                        v669 = 0l;
                        while (while_method_4(v669)){
                            assert("Tensor range check" && 0 <= v667 && v667 < 1l);
                            assert("Tensor range check" && 0 <= v669 && v669 < 4l);
                            int v671;
                            v671 = 4l * v667;
                            int v672;
                            v672 = v671 + v669;
                            float v673;
                            v673 = v595[v672];
                            float v674;
                            v674 = v596[v672];
                            float v675;
                            v675 = v673 + v674;
                            bool v676;
                            v676 = 0.0f >= v675;
                            float v677;
                            if (v676){
                                v677 = 0.0f;
                            } else {
                                v677 = v675;
                            }
                            assert("Tensor range check" && 0 <= v667 && v667 < 1l);
                            assert("Tensor range check" && 0 <= v669 && v669 < 4l);
                            v666[v672] = v677;
                            v669 += 1l ;
                        }
                        v667 += 1l ;
                    }
                    float v678[4l];
                    int v679;
                    v679 = 0l;
                    while (while_method_6(v679)){
                        int v681;
                        v681 = 0l;
                        while (while_method_4(v681)){
                            assert("Tensor range check" && 0 <= v679 && v679 < 1l);
                            assert("Tensor range check" && 0 <= v681 && v681 < 4l);
                            int v683;
                            v683 = 4l * v679;
                            int v684;
                            v684 = v683 + v681;
                            float v685;
                            v685 = v666[v684];
                            bool v686;
                            v686 = 0.0f >= v685;
                            float v687;
                            if (v686){
                                v687 = 0.0f;
                            } else {
                                v687 = v685;
                            }
                            assert("Tensor range check" && 0 <= v679 && v679 < 1l);
                            assert("Tensor range check" && 0 <= v681 && v681 < 4l);
                            v678[v684] = v687;
                            v681 += 1l ;
                        }
                        v679 += 1l ;
                    }
                    float v688;
                    v688 = 0.0f;
                    int v689;
                    v689 = 0l;
                    while (while_method_6(v689)){
                        int v691;
                        v691 = 0l;
                        while (while_method_4(v691)){
                            assert("Tensor range check" && 0 <= v689 && v689 < 1l);
                            assert("Tensor range check" && 0 <= v691 && v691 < 4l);
                            int v693;
                            v693 = 4l * v689;
                            int v694;
                            v694 = v693 + v691;
                            float v695;
                            v695 = v678[v694];
                            float v696;
                            v696 = v688 + v695;
                            v688 = v696;
                            v691 += 1l ;
                        }
                        v689 += 1l ;
                    }
                    auto v697 = cooperative_groups::coalesced_threads();
                    int v698;
                    v698 = threadIdx.x;
                    auto v699 = cooperative_groups::labeled_partition(v697,v698);
                    Closure1 v700{};
                    float v701;
                    v701 = cooperative_groups::reduce(v699, v688, v700);
                    float v702[4l];
                    int v703;
                    v703 = 0l;
                    while (while_method_6(v703)){
                        int v705;
                        v705 = 0l;
                        while (while_method_4(v705)){
                            assert("Tensor range check" && 0 <= v703 && v703 < 1l);
                            assert("Tensor range check" && 0 <= v705 && v705 < 4l);
                            int v707;
                            v707 = 4l * v703;
                            int v708;
                            v708 = v707 + v705;
                            float v709;
                            v709 = v678[v708];
                            bool v710;
                            v710 = v701 == 0.0f;
                            bool v711;
                            v711 = v710 != true;
                            float v713;
                            if (v711){
                                float v712;
                                v712 = v709 / v701;
                                v713 = v712;
                            } else {
                                v713 = 0.25f;
                            }
                            assert("Tensor range check" && 0 <= v703 && v703 < 1l);
                            assert("Tensor range check" && 0 <= v705 && v705 < 4l);
                            v702[v708] = v713;
                            v705 += 1l ;
                        }
                        v703 += 1l ;
                    }
                    float v714[4l];
                    int v715;
                    v715 = 0l;
                    while (while_method_6(v715)){
                        int v717;
                        v717 = 0l;
                        while (while_method_4(v717)){
                            assert("Tensor range check" && 0 <= v715 && v715 < 1l);
                            assert("Tensor range check" && 0 <= v717 && v717 < 4l);
                            int v719;
                            v719 = 4l * v715;
                            int v720;
                            v720 = v719 + v717;
                            float v721;
                            v721 = v594[v720];
                            float v722;
                            v722 = v702[v720];
                            float v723;
                            v723 = v721 + v722;
                            assert("Tensor range check" && 0 <= v715 && v715 < 1l);
                            assert("Tensor range check" && 0 <= v717 && v717 < 4l);
                            v714[v720] = v723;
                            v717 += 1l ;
                        }
                        v715 += 1l ;
                    }
                    float v724[4l];
                    int v725;
                    v725 = 0l;
                    while (while_method_6(v725)){
                        int v727;
                        v727 = 0l;
                        while (while_method_4(v727)){
                            assert("Tensor range check" && 0 <= v725 && v725 < 1l);
                            assert("Tensor range check" && 0 <= v727 && v727 < 4l);
                            int v729;
                            v729 = 4l * v725;
                            int v730;
                            v730 = v729 + v727;
                            float v731;
                            v731 = v714[v730];
                            float v732;
                            v732 = -v731;
                            bool v733;
                            v733 = v731 >= v732;
                            float v734;
                            if (v733){
                                v734 = v731;
                            } else {
                                v734 = v732;
                            }
                            assert("Tensor range check" && 0 <= v725 && v725 < 1l);
                            assert("Tensor range check" && 0 <= v727 && v727 < 4l);
                            v724[v730] = v734;
                            v727 += 1l ;
                        }
                        v725 += 1l ;
                    }
                    float v735;
                    v735 = 0.0f;
                    int v736;
                    v736 = 0l;
                    while (while_method_6(v736)){
                        int v738;
                        v738 = 0l;
                        while (while_method_4(v738)){
                            assert("Tensor range check" && 0 <= v736 && v736 < 1l);
                            assert("Tensor range check" && 0 <= v738 && v738 < 4l);
                            int v740;
                            v740 = 4l * v736;
                            int v741;
                            v741 = v740 + v738;
                            float v742;
                            v742 = v724[v741];
                            float v743;
                            v743 = v735 + v742;
                            v735 = v743;
                            v738 += 1l ;
                        }
                        v736 += 1l ;
                    }
                    auto v744 = cooperative_groups::coalesced_threads();
                    int v745;
                    v745 = threadIdx.x;
                    auto v746 = cooperative_groups::labeled_partition(v744,v745);
                    float v747;
                    v747 = cooperative_groups::reduce(v746, v735, v700);
                    bool v748;
                    v748 = v747 > 100.0f;
                    float v750;
                    if (v748){
                        float v749;
                        v749 = 100.0f / v747;
                        v750 = v749;
                    } else {
                        v750 = 1.0f;
                    }
                    float v751[4l];
                    int v752;
                    v752 = 0l;
                    while (while_method_6(v752)){
                        int v754;
                        v754 = 0l;
                        while (while_method_4(v754)){
                            assert("Tensor range check" && 0 <= v752 && v752 < 1l);
                            assert("Tensor range check" && 0 <= v754 && v754 < 4l);
                            int v756;
                            v756 = 4l * v752;
                            int v757;
                            v757 = v756 + v754;
                            float v758;
                            v758 = v724[v757];
                            float v759;
                            v759 = v750 * v758;
                            assert("Tensor range check" && 0 <= v752 && v752 < 1l);
                            assert("Tensor range check" && 0 <= v754 && v754 < 4l);
                            v751[v757] = v759;
                            v754 += 1l ;
                        }
                        v752 += 1l ;
                    }
                    float v760[4l];
                    float v761[4l];
                    int v762;
                    v762 = 0l;
                    while (while_method_6(v762)){
                        int v764;
                        v764 = 0l;
                        while (while_method_4(v764)){
                            assert("Tensor range check" && 0 <= v762 && v762 < 1l);
                            assert("Tensor range check" && 0 <= v764 && v764 < 4l);
                            int v766;
                            v766 = 4l * v762;
                            int v767;
                            v767 = v766 + v764;
                            float v768;
                            v768 = v594[v767];
                            float v769;
                            v769 = v595[v767];
                            float v770;
                            v770 = v596[v767];
                            float v771;
                            v771 = v597[v767];
                            float v772;
                            v772 = v598[v767];
                            float v773;
                            v773 = v599[v767];
                            float v774;
                            v774 = v600[v767];
                            float v775;
                            v775 = v771 + v773;
                            float v776;
                            v776 = v772 + v774;
                            assert("Tensor range check" && 0 <= v762 && v762 < 1l);
                            assert("Tensor range check" && 0 <= v764 && v764 < 4l);
                            v760[v767] = v775;
                            v761[v767] = v776;
                            v764 += 1l ;
                        }
                        v762 += 1l ;
                    }
                    float v777[4l];
                    float v778[4l];
                    float v779[4l];
                    float v780[4l];
                    float v781[4l];
                    float v782[4l];
                    float v783[4l];
                    int v784;
                    v784 = 0l;
                    while (while_method_6(v784)){
                        int v786;
                        v786 = 0l;
                        while (while_method_4(v786)){
                            assert("Tensor range check" && 0 <= v784 && v784 < 1l);
                            assert("Tensor range check" && 0 <= v786 && v786 < 4l);
                            int v788;
                            v788 = 4l * v784;
                            int v789;
                            v789 = v788 + v786;
                            float v790;
                            v790 = v751[v789];
                            float v791;
                            v791 = v666[v789];
                            float v792;
                            v792 = v760[v789];
                            float v793;
                            v793 = v761[v789];
                            assert("Tensor range check" && 0 <= v784 && v784 < 1l);
                            assert("Tensor range check" && 0 <= v786 && v786 < 4l);
                            v777[v789] = v790;
                            v778[v789] = v791;
                            v779[v789] = 0.0f;
                            v780[v789] = v792;
                            v781[v789] = v793;
                            v782[v789] = 0.0f;
                            v783[v789] = 0.0f;
                            v786 += 1l ;
                        }
                        v784 += 1l ;
                    }
                    assert("Tensor range check" && 0 <= v586 && v586 < 4l);
                    assert("Tensor range check" && 0 <= v585 && v585 < 128l);
                    int v794;
                    v794 = 0l;
                    while (while_method_6(v794)){
                        assert("Tensor range check" && 0 <= v794 && v794 < 1l);
                        int v796;
                        v796 = 4l * v794;
                        int v797;
                        v797 = v796 + v593;
                        assert("Tensor range check" && 0 <= v794 && v794 < 1l);
                        int4* v798;
                        v798 = reinterpret_cast<int4*>(v777 + v796);
                        int4* v799;
                        v799 = reinterpret_cast<int4*>(v524 + v797);
                        assert("Pointer alignment check" && (unsigned long long)(v798) % 4l == 0 && (unsigned long long)(v799) % 4l == 0);
                        *v799 = *v798;
                        int4* v800;
                        v800 = reinterpret_cast<int4*>(v778 + v796);
                        int4* v801;
                        v801 = reinterpret_cast<int4*>(v526 + v797);
                        assert("Pointer alignment check" && (unsigned long long)(v800) % 4l == 0 && (unsigned long long)(v801) % 4l == 0);
                        *v801 = *v800;
                        int4* v802;
                        v802 = reinterpret_cast<int4*>(v779 + v796);
                        int4* v803;
                        v803 = reinterpret_cast<int4*>(v528 + v797);
                        assert("Pointer alignment check" && (unsigned long long)(v802) % 4l == 0 && (unsigned long long)(v803) % 4l == 0);
                        *v803 = *v802;
                        int4* v804;
                        v804 = reinterpret_cast<int4*>(v780 + v796);
                        int4* v805;
                        v805 = reinterpret_cast<int4*>(v530 + v797);
                        assert("Pointer alignment check" && (unsigned long long)(v804) % 4l == 0 && (unsigned long long)(v805) % 4l == 0);
                        *v805 = *v804;
                        int4* v806;
                        v806 = reinterpret_cast<int4*>(v781 + v796);
                        int4* v807;
                        v807 = reinterpret_cast<int4*>(v532 + v797);
                        assert("Pointer alignment check" && (unsigned long long)(v806) % 4l == 0 && (unsigned long long)(v807) % 4l == 0);
                        *v807 = *v806;
                        int4* v808;
                        v808 = reinterpret_cast<int4*>(v782 + v796);
                        int4* v809;
                        v809 = reinterpret_cast<int4*>(v534 + v797);
                        assert("Pointer alignment check" && (unsigned long long)(v808) % 4l == 0 && (unsigned long long)(v809) % 4l == 0);
                        *v809 = *v808;
                        int4* v810;
                        v810 = reinterpret_cast<int4*>(v783 + v796);
                        int4* v811;
                        v811 = reinterpret_cast<int4*>(v536 + v797);
                        assert("Pointer alignment check" && (unsigned long long)(v810) % 4l == 0 && (unsigned long long)(v811) % 4l == 0);
                        *v811 = *v810;
                        v794 += 1l ;
                    }
                    v580 += 1l ;
                }
                v14.sync() ;
                v29 += 1l ;
            }
            int v812;
            v812 = threadIdx.x;
            int v813;
            v813 = blockIdx.x;
            int v814;
            v814 = v813 * 32l;
            int v815;
            v815 = v812 + v814;
            int v816;
            v816 = v815;
            while (while_method_4(v816)){
                bool v818;
                v818 = 0l <= v816;
                bool v819;
                v819 = v818 == false;
                if (v819){
                    assert("The index needs to be zero or positive." && v818);
                } else {
                }
                int v821;
                v821 = v816 % 1l;
                int v822;
                v822 = v816 % 2l;
                int v823;
                v823 = v816 / 2l;
                bool v824;
                v824 = v823 < 2l;
                bool v825;
                v825 = v824 == false;
                if (v825){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v824);
                } else {
                }
                assert("Tensor range check" && 0 <= v823 && v823 < 2l);
                assert("Tensor range check" && 0 <= v822 && v822 < 2l);
                assert("Tensor range check" && 0 <= v821 && v821 < 1l);
                int v827;
                v827 = 4l * v821;
                int v828;
                v828 = 4l * v822;
                int v829;
                v829 = v828 + v827;
                int v830;
                v830 = 8l * v823;
                int v831;
                v831 = v830 + v829;
                assert("Tensor range check" && 0 <= v823 && v823 < 2l);
                assert("Tensor range check" && 0 <= v822 && v822 < 2l);
                assert("Tensor range check" && 0 <= v821 && v821 < 1l);
                float v832[4l];
                float v833[4l];
                float v834[4l];
                int4* v835;
                v835 = reinterpret_cast<int4*>(v5 + v831);
                int4* v836;
                v836 = reinterpret_cast<int4*>(v832 + 0l);
                assert("Pointer alignment check" && (unsigned long long)(v835) % 4l == 0 && (unsigned long long)(v836) % 4l == 0);
                *v836 = *v835;
                int4* v837;
                v837 = reinterpret_cast<int4*>(v6 + v831);
                int4* v838;
                v838 = reinterpret_cast<int4*>(v833 + 0l);
                assert("Pointer alignment check" && (unsigned long long)(v837) % 4l == 0 && (unsigned long long)(v838) % 4l == 0);
                *v838 = *v837;
                // Pushing the loop unrolling to: 0
                int v839;
                v839 = 0l;
                #pragma unroll
                while (while_method_4(v839)){
                    assert("Tensor range check" && 0 <= v839 && v839 < 4l);
                    float v841;
                    v841 = v832[v839];
                    float v842;
                    v842 = v833[v839];
                    bool v843;
                    v843 = v842 == 0.0f;
                    bool v844;
                    v844 = v843 != true;
                    float v846;
                    if (v844){
                        float v845;
                        v845 = v841 / v842;
                        v846 = v845;
                    } else {
                        v846 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v839 && v839 < 4l);
                    v834[v839] = v846;
                    v839 += 1l ;
                }
                // Poping the loop unrolling to: 0
                int4* v847;
                v847 = reinterpret_cast<int4*>(v834 + 0l);
                int4* v848;
                v848 = reinterpret_cast<int4*>(v7 + v831);
                assert("Pointer alignment check" && (unsigned long long)(v847) % 4l == 0 && (unsigned long long)(v848) % 4l == 0);
                *v848 = *v847;
                v816 += 32l ;
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
        v19 = cp.zeros(16,dtype=cp.float32) # type: ignore
        v20 = cp.zeros(16,dtype=cp.float32) # type: ignore
        v21 = cp.empty(16,dtype=cp.float32)
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
        v27.max_dynamic_shared_size_bytes = 1536 
        v27((1,),(32,),(v2, v4, v5, v6, v7, v19, v20, v21),shared_mem=1536)
        del v2, v19, v20, v27
        v28 = time.perf_counter()
        cp.cuda.get_current_stream().synchronize()
        v31 = "{}"
        v32 = "The time it took to run the kernel (in seconds) is: "
        print(v31.format(v32),end="")
        del v31, v32
        v33 = v28 - v12
        del v12, v28
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
            while method16(v44):
                assert 0 <= v41 < 4, 'Tensor range check'
                assert 0 <= v44 < 4, 'Tensor range check'
                v46 = 4 * v41
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
        return method17(v4, v5, v6, v7, v49)
    return inner
def Closure1():
    def inner() -> object:
        v0 = cp.empty(2101776,dtype=cp.uint8)
        v1 = cp.empty(131584,dtype=cp.uint8)
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
        v22 = v0[2097168:2097168+8*256].view(cp.float64)
        v24 = v0[2099216:2099216+8*256].view(cp.float64)
        v26 = v0[2101264:2101264+4*128].view(cp.int32)
        v22[:] = 0
        del v22
        v24[:] = 0
        del v24
        v26[:] = 0
        del v26
        v27 = 131584
        v28 = 2101776
        return method31(v1, v27, v0, v28)
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
def method21() -> object:
    v0 = []
    return v0
def method20() -> object:
    v0 = method21()
    v1 = method21()
    v2 = {'private': v0, 'public': v1}
    del v0, v1
    return v2
def method27(v0 : cp.ndarray) -> object:
    v1 = v0
    del v0
    return v1
def method26(v0 : cp.ndarray) -> object:
    return method27(v0)
def method28(v0 : u64) -> object:
    v1 = v0
    del v0
    return v1
def method25(v0 : cp.ndarray, v1 : u64) -> object:
    v2 = []
    v3 = method26(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method28(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method24(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method25(v0, v1)
    del v0, v1
    v5 = method25(v2, v3)
    del v2, v3
    v6 = {'output': v4, 'param': v5}
    del v4, v5
    return v6
def method23(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    return method24(v0, v1, v2, v3)
def method22(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method23(v0, v1, v2, v3)
    del v0, v1, v2, v3
    v5 = {'model_data': v4}
    del v4
    return v5
def method19(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method20()
    v5 = method22(v0, v1, v2, v3)
    del v0, v1, v2, v3
    v6 = {'game': v4, 'neural': v5}
    del v4, v5
    return v6
def method30(v0 : list) -> object:
    v1 = v0
    del v0
    return v1
def method29(v0 : list) -> object:
    return method30(v0)
def method18(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64, v4 : list) -> object:
    v5 = []
    v6 = method19(v0, v1, v2, v3)
    del v0, v1, v2, v3
    v5.append(v6)
    del v6
    v7 = method29(v4)
    del v4
    v5.append(v7)
    del v7
    v8 = v5
    del v5
    return v8
def method17(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64, v4 : list) -> object:
    v5 = method18(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    return v5
def method31(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method19(v0, v1, v2, v3)
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
