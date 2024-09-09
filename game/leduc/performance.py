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
struct Union2;
struct Union3;
struct Union1;
struct Union5;
struct Union4;
struct Union6;
struct Tuple0;
__device__ unsigned int loop_3(unsigned int v0, curandStatePhilox4_32_10_t & v1);
__device__ Tuple0 draw_card_2(curandStatePhilox4_32_10_t & v0, unsigned int v1);
struct Tuple1;
struct Union7;
struct Union8;
__device__ void method_4(float * v0, int v1, float * v2, int v3, float * v4, int v5);
__device__ void method_5(unsigned int * v0, int v1, float * v2);
struct Tuple2;
struct Tuple3;
struct Tuple4;
struct Tuple5;
__device__ Tuple2 method_6(curandStatePhilox4_32_10_t & v0, int * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int v9, int v10);
__device__ float method_7(int * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int v8, int v9, int v10);
struct Union9;
__device__ int int_range_8(int v0, int v1, curandStatePhilox4_32_10_t & v2);
struct Union10;
__device__ int tag_10(Union2 v0);
__device__ bool is_pair_11(int v0, int v1);
__device__ Tuple1 order_12(int v0, int v1);
__device__ Union10 compare_hands_9(Union5 v0, bool v1, static_array<Union2,2l> v2, int v3, static_array<int,2l> v4, int v5);
__device__ static_array<float,2l> method_1(unsigned char * v0, unsigned char * v1, unsigned int & v2, static_array_list<Union1,32l> & v3, static_array<Union0,2l> & v4, curandStatePhilox4_32_10_t & v5, Union4 v6);
__device__ void noinline_train_0(unsigned char * v0, unsigned char * v1, curandStatePhilox4_32_10_t & v2, int v3);
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
struct Union4_0 { // ChanceCommunityCard
    Union5 v0;
    static_array<Union2,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_0(Union5 t0, bool t1, static_array<Union2,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_0() = delete;
};
struct Union4_1 { // ChanceInit
};
struct Union4_2 { // Round
    Union5 v0;
    static_array<Union2,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_2(Union5 t0, bool t1, static_array<Union2,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_2() = delete;
};
struct Union4_3 { // RoundWithAction
    Union5 v0;
    static_array<Union2,2l> v2;
    static_array<int,2l> v4;
    Union3 v6;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_3(Union5 t0, bool t1, static_array<Union2,2l> t2, int t3, static_array<int,2l> t4, int t5, Union3 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union4_3() = delete;
};
struct Union4_4 { // TerminalCall
    Union5 v0;
    static_array<Union2,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_4(Union5 t0, bool t1, static_array<Union2,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_4() = delete;
};
struct Union4_5 { // TerminalFold
    Union5 v0;
    static_array<Union2,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_5(Union5 t0, bool t1, static_array<Union2,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
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
struct Union9_0 { // AA_Call
};
struct Union9_1 { // AA_Fold
};
struct Union9_2 { // AA_Raise
};
struct Union9 {
    union {
        Union9_0 case0; // AA_Call
        Union9_1 case1; // AA_Fold
        Union9_2 case2; // AA_Raise
    };
    unsigned char tag{255};
    __device__ Union9() {}
    __device__ Union9(Union9_0 t) : tag(0), case0(t) {} // AA_Call
    __device__ Union9(Union9_1 t) : tag(1), case1(t) {} // AA_Fold
    __device__ Union9(Union9_2 t) : tag(2), case2(t) {} // AA_Raise
    __device__ Union9(Union9 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union9_0(x.case0); break; // AA_Call
            case 1: new (&this->case1) Union9_1(x.case1); break; // AA_Fold
            case 2: new (&this->case2) Union9_2(x.case2); break; // AA_Raise
        }
    }
    __device__ Union9(Union9 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union9_0(std::move(x.case0)); break; // AA_Call
            case 1: new (&this->case1) Union9_1(std::move(x.case1)); break; // AA_Fold
            case 2: new (&this->case2) Union9_2(std::move(x.case2)); break; // AA_Raise
        }
    }
    __device__ Union9 & operator=(Union9 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // AA_Call
                case 1: this->case1 = x.case1; break; // AA_Fold
                case 2: this->case2 = x.case2; break; // AA_Raise
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
                case 0: this->case0 = std::move(x.case0); break; // AA_Call
                case 1: this->case1 = std::move(x.case1); break; // AA_Fold
                case 2: this->case2 = std::move(x.case2); break; // AA_Raise
            }
        } else {
            this->~Union9();
            new (this) Union9{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union9() {
        switch(this->tag){
            case 0: this->case0.~Union9_0(); break; // AA_Call
            case 1: this->case1.~Union9_1(); break; // AA_Fold
            case 2: this->case2.~Union9_2(); break; // AA_Raise
        }
        this->tag = 255;
    }
};
struct Union10_0 { // Eq
};
struct Union10_1 { // Gt
};
struct Union10_2 { // Lt
};
struct Union10 {
    union {
        Union10_0 case0; // Eq
        Union10_1 case1; // Gt
        Union10_2 case2; // Lt
    };
    unsigned char tag{255};
    __device__ Union10() {}
    __device__ Union10(Union10_0 t) : tag(0), case0(t) {} // Eq
    __device__ Union10(Union10_1 t) : tag(1), case1(t) {} // Gt
    __device__ Union10(Union10_2 t) : tag(2), case2(t) {} // Lt
    __device__ Union10(Union10 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union10_0(x.case0); break; // Eq
            case 1: new (&this->case1) Union10_1(x.case1); break; // Gt
            case 2: new (&this->case2) Union10_2(x.case2); break; // Lt
        }
    }
    __device__ Union10(Union10 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union10_0(std::move(x.case0)); break; // Eq
            case 1: new (&this->case1) Union10_1(std::move(x.case1)); break; // Gt
            case 2: new (&this->case2) Union10_2(std::move(x.case2)); break; // Lt
        }
    }
    __device__ Union10 & operator=(Union10 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Eq
                case 1: this->case1 = x.case1; break; // Gt
                case 2: this->case2 = x.case2; break; // Lt
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
                case 0: this->case0 = std::move(x.case0); break; // Eq
                case 1: this->case1 = std::move(x.case1); break; // Gt
                case 2: this->case2 = std::move(x.case2); break; // Lt
            }
        } else {
            this->~Union10();
            new (this) Union10{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union10() {
        switch(this->tag){
            case 0: this->case0.~Union10_0(); break; // Eq
            case 1: this->case1.~Union10_1(); break; // Gt
            case 2: this->case2.~Union10_2(); break; // Lt
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
__device__ unsigned int loop_3(unsigned int v0, curandStatePhilox4_32_10_t & v1){
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
        return loop_3(v0, v1);
    }
}
__device__ Tuple0 draw_card_2(curandStatePhilox4_32_10_t & v0, unsigned int v1){
    int v2;
    v2 = __popc(v1);
    unsigned int v3;
    v3 = (unsigned int)v2;
    unsigned int v4;
    v4 = loop_3(v3, v0);
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
    while (while_method_0(v61)){
        int v63;
        v63 = 0l;
        while (while_method_6(v63)){
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
            while (while_method_0(v71)){
                int v73;
                v73 = 0l;
                #pragma unroll
                while (while_method_6(v73)){
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
                while (while_method_0(v107)){
                    int v109;
                    v109 = 0l;
                    #pragma unroll
                    while (while_method_6(v109)){
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
                while (while_method_0(v142)){
                    int v144;
                    v144 = 0l;
                    #pragma unroll
                    while (while_method_6(v144)){
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
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v159[32l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v160[8l];
                int v161;
                v161 = 0l;
                #pragma unroll
                while (while_method_0(v161)){
                    int v163;
                    v163 = 0l;
                    #pragma unroll
                    while (while_method_7(v163)){
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
                while (while_method_0(v223)){
                    int v225;
                    v225 = 0l;
                    #pragma unroll
                    while (while_method_6(v225)){
                        int v227;
                        v227 = 0l;
                        #pragma unroll
                        while (while_method_7(v227)){
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
            while (while_method_0(v237)){
                int v239;
                v239 = 0l;
                #pragma unroll
                while (while_method_6(v239)){
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
            while (while_method_7(v266)){
                int v268;
                v268 = 0l;
                #pragma unroll
                while (while_method_6(v268)){
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
    while (while_method_8(v22)){
        assert("Tensor range check" && 0 <= v22 && v22 < 32l);
        int v24;
        v24 = 2048l * v22;
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
        v69 = v22 * 16l;
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
        v99 = 16l * v22;
        int v100;
        v100 = v99 + v21;
        v0[v100] = v98;
        v22 += 1l ;
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
__device__ int int_range_8(int v0, int v1, curandStatePhilox4_32_10_t & v2){
    int v3;
    v3 = v0 - v1;
    unsigned int v4;
    v4 = (unsigned int)v3;
    unsigned int v5;
    v5 = loop_3(v4, v2);
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
__device__ Union10 compare_hands_9(Union5 v0, bool v1, static_array<Union2,2l> v2, int v3, static_array<int,2l> v4, int v5){
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
                        return Union10{Union10_2{}};
                    } else {
                        bool v19;
                        v19 = v11 > v14;
                        if (v19){
                            return Union10{Union10_1{}};
                        } else {
                            return Union10{Union10_0{}};
                        }
                    }
                } else {
                    return Union10{Union10_1{}};
                }
            } else {
                if (v16){
                    return Union10{Union10_2{}};
                } else {
                    int v27; int v28;
                    Tuple1 tmp24 = order_12(v8, v11);
                    v27 = tmp24.v0; v28 = tmp24.v1;
                    int v29; int v30;
                    Tuple1 tmp25 = order_12(v8, v14);
                    v29 = tmp25.v0; v30 = tmp25.v1;
                    bool v31;
                    v31 = v27 < v29;
                    Union10 v37;
                    if (v31){
                        v37 = Union10{Union10_2{}};
                    } else {
                        bool v33;
                        v33 = v27 > v29;
                        if (v33){
                            v37 = Union10{Union10_1{}};
                        } else {
                            v37 = Union10{Union10_0{}};
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
                            return Union10{Union10_2{}};
                        } else {
                            bool v41;
                            v41 = v28 > v30;
                            if (v41){
                                return Union10{Union10_1{}};
                            } else {
                                return Union10{Union10_0{}};
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
__device__ static_array<float,2l> method_1(unsigned char * v0, unsigned char * v1, unsigned int & v2, static_array_list<Union1,32l> & v3, static_array<Union0,2l> & v4, curandStatePhilox4_32_10_t & v5, Union4 v6){
    static_array<float,2l> v7;
    static_array_list<Union1,32l> & v9 = v3;
    Union6 v10;
    v10 = Union6{Union6_1{v6}};
    Union6 v11;
    v11 = v10;
    while (while_method_2(v11)){
        Union6 v712;
        switch (v11.tag) {
            case 0: { // None
                v712 = Union6{Union6_0{}};
                break;
            }
            case 1: { // Some
                Union4 v13 = v11.case1.v0;
                switch (v13.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v663 = v13.case0.v0; bool v664 = v13.case0.v1; static_array<Union2,2l> v665 = v13.case0.v2; int v666 = v13.case0.v3; static_array<int,2l> v667 = v13.case0.v4; int v668 = v13.case0.v5;
                        unsigned int v669 = v2;
                        Union2 v670; unsigned int v671;
                        Tuple0 tmp0 = draw_card_2(v5, v669);
                        v670 = tmp0.v0; v671 = tmp0.v1;
                        v2 = v671;
                        Union1 v672;
                        v672 = Union1{Union1_0{v670}};
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
                        Union5 v685;
                        v685 = Union5{Union5_1{v670}};
                        Union4 v686;
                        v686 = Union4{Union4_2{v685, true, v665, 0l, v681, v673}};
                        v712 = Union6{Union6_1{v686}};
                        break;
                    }
                    case 1: { // ChanceInit
                        unsigned int v688 = v2;
                        Union2 v689; unsigned int v690;
                        Tuple0 tmp2 = draw_card_2(v5, v688);
                        v689 = tmp2.v0; v690 = tmp2.v1;
                        v2 = v690;
                        unsigned int v691 = v2;
                        Union2 v692; unsigned int v693;
                        Tuple0 tmp3 = draw_card_2(v5, v691);
                        v692 = tmp3.v0; v693 = tmp3.v1;
                        v2 = v693;
                        Union1 v694;
                        v694 = Union1{Union1_2{0l, v689}};
                        v9.push(v694);
                        Union1 v695;
                        v695 = Union1{Union1_2{1l, v692}};
                        v9.push(v695);
                        int v696;
                        v696 = 2l;
                        static_array<int,2l> v697;
                        v697[0l] = 1l;
                        v697[1l] = 1l;
                        static_array<Union2,2l> v699;
                        v699[0l] = v689;
                        v699[1l] = v692;
                        Union5 v701;
                        v701 = Union5{Union5_0{}};
                        Union4 v702;
                        v702 = Union4{Union4_2{v701, true, v699, 0l, v697, v696}};
                        v712 = Union6{Union6_1{v702}};
                        break;
                    }
                    case 2: { // Round
                        Union5 v54 = v13.case2.v0; bool v55 = v13.case2.v1; static_array<Union2,2l> v56 = v13.case2.v2; int v57 = v13.case2.v3; static_array<int,2l> v58 = v13.case2.v4; int v59 = v13.case2.v5;
                        static_array<Union0,2l> v60 = v4;
                        Union0 v61;
                        v61 = v60[v57];
                        Union3 v479;
                        switch (v61.tag) {
                            case 0: { // T_Computer
                                static_array_list<Union1,32l> & v63 = v3;
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
                                static_array_list<Union7,10l> v101;
                                v101 = static_array_list<Union7,10l>{};
                                int v103;
                                v103 = v63.length;
                                int v104;
                                v104 = 0l;
                                while (while_method_5(v103, v104)){
                                    Union1 v106;
                                    v106 = v63[v104];
                                    Union8 v125;
                                    switch (v106.tag) {
                                        case 0: { // CommunityCardIs
                                            Union2 v115 = v106.case0.v0;
                                            Union7 v116;
                                            v116 = Union7{Union7_1{v115}};
                                            v125 = Union8{Union8_1{v116}};
                                            break;
                                        }
                                        case 1: { // PlayerAction
                                            int v118 = v106.case1.v0; Union3 v119 = v106.case1.v1;
                                            Union7 v120;
                                            v120 = Union7{Union7_0{v119}};
                                            v125 = Union8{Union8_1{v120}};
                                            break;
                                        }
                                        case 2: { // PlayerGotCard
                                            int v108 = v106.case2.v0; Union2 v109 = v106.case2.v1;
                                            bool v110;
                                            v110 = v108 == v57;
                                            if (v110){
                                                Union7 v111;
                                                v111 = Union7{Union7_1{v109}};
                                                v125 = Union8{Union8_1{v111}};
                                            } else {
                                                v125 = Union8{Union8_0{}};
                                            }
                                            break;
                                        }
                                        default: {
                                            v125 = Union8{Union8_0{}};
                                        }
                                    }
                                    switch (v125.tag) {
                                        case 0: { // None
                                            break;
                                        }
                                        case 1: { // Some
                                            Union7 v126 = v125.case1.v0;
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
                                    Union7 v134;
                                    v134 = v101[v132];
                                    int v136;
                                    v136 = v132 * 6l;
                                    int v137;
                                    v137 = 1l + v136;
                                    switch (v134.tag) {
                                        case 0: { // C1of2
                                            Union3 v138 = v134.case0.v0;
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
                                            Union2 v141 = v134.case1.v0;
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
                                while (while_method_0(v145)){
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
                                    method_4(v149, v151, v152, v157, v147, v155);
                                    unsigned int * v158;
                                    v158 = reinterpret_cast<unsigned int *>(&v0[12582912ull]);
                                    assert("Tensor range check" && 0 <= v145 && v145 < 4l);
                                    int v160;
                                    v160 = 12288l * v145;
                                    method_5(v158, v160, v152);
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
                                Tuple2 tmp14 = method_6(v5, v195, v197, v199, v201, v203, v205, v207, v209, v221, v211);
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
                                while (while_method_0(v255)){
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
                                    v267 = method_7(v195, v197, v199, v201, v203, v205, v207, v209, v266, v255, v232);
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
                                    while (while_method_6(v331)){
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
                                        while (while_method_6(v356)){
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
                                        while (while_method_6(v364)){
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
                                        while (while_method_6(v389)){
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
                                Union9 v416;
                                if (v407){
                                    v416 = Union9{Union9_1{}};
                                } else {
                                    bool v409;
                                    v409 = 1l == v232;
                                    if (v409){
                                        v416 = Union9{Union9_0{}};
                                    } else {
                                        bool v411;
                                        v411 = 2l == v232;
                                        if (v411){
                                            v416 = Union9{Union9_2{}};
                                        } else {
                                            printf("%s\n", "Invalid output id in the Leduc model.");
                                            __trap();
                                        }
                                    }
                                }
                                switch (v416.tag) {
                                    case 0: { // AA_Call
                                        v479 = Union3{Union3_0{}};
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
                                            v479 = Union3{Union3_0{}};
                                        } else {
                                            v479 = Union3{Union3_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v433;
                                        v433 = v59 > 0l;
                                        if (v433){
                                            v479 = Union3{Union3_2{}};
                                        } else {
                                            v479 = Union3{Union3_0{}};
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
                                static_array_list<Union3,3l> v440;
                                v440 = static_array_list<Union3,3l>{};
                                v440.unsafe_set_length(1l);
                                Union3 v442;
                                v442 = Union3{Union3_0{}};
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
                                    Union3 v450;
                                    v450 = Union3{Union3_1{}};
                                    v440.push(v450);
                                } else {
                                }
                                bool v451;
                                v451 = v59 > 0l;
                                if (v451){
                                    Union3 v452;
                                    v452 = Union3{Union3_2{}};
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
                                    v458 = int_range_8(v457, v455, v5);
                                    Union3 v459;
                                    v459 = v440[v455];
                                    Union3 v461;
                                    v461 = v440[v458];
                                    v440[v455] = v461;
                                    v440[v458] = v459;
                                    v455 += 1l ;
                                }
                                Union3 v463;
                                v463 = v440.pop();
                                int v464;
                                v464 = sizeof(Union3);
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
                                Union3 * v473;
                                v473 = reinterpret_cast<Union3 *>(&v469[0ull]);
                                int v475;
                                v475 = threadIdx.x;
                                bool v476;
                                v476 = v475 == 0l;
                                if (v476){
                                    v473[0l] = v463;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                Union3 v477;
                                v477 = v473[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                v479 = v477;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union1 v480;
                        v480 = Union1{Union1_1{v57, v479}};
                        v9.push(v480);
                        Union4 v566;
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
                                            v566 = Union4{Union4_2{v54, false, v56, v531, v58, v59}};
                                        } else {
                                            v566 = Union4{Union4_0{v54, v55, v56, v57, v58, v59}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v566 = Union4{Union4_5{v54, v55, v56, v57, v58, v59}};
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
                                            v566 = Union4{Union4_2{v54, false, v56, v537, v550, v538}};
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
                                Union2 v481 = v54.case1.v0;
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
                                            v566 = Union4{Union4_2{v54, false, v56, v484, v58, v59}};
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
                                            v566 = Union4{Union4_4{v54, v55, v56, v57, v493, v59}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v566 = Union4{Union4_5{v54, v55, v56, v57, v58, v59}};
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
                                            v566 = Union4{Union4_2{v54, false, v56, v501, v514, v502}};
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
                        v712 = Union6{Union6_1{v566}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v568 = v13.case3.v0; bool v569 = v13.case3.v1; static_array<Union2,2l> v570 = v13.case3.v2; int v571 = v13.case3.v3; static_array<int,2l> v572 = v13.case3.v4; int v573 = v13.case3.v5; Union3 v574 = v13.case3.v6;
                        Union1 v575;
                        v575 = Union1{Union1_1{v571, v574}};
                        v9.push(v575);
                        Union4 v661;
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
                                            v661 = Union4{Union4_2{v568, false, v570, v626, v572, v573}};
                                        } else {
                                            v661 = Union4{Union4_0{v568, v569, v570, v571, v572, v573}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v661 = Union4{Union4_5{v568, v569, v570, v571, v572, v573}};
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
                                            v661 = Union4{Union4_2{v568, false, v570, v632, v645, v633}};
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
                                Union2 v576 = v568.case1.v0;
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
                                            v661 = Union4{Union4_2{v568, false, v570, v579, v572, v573}};
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
                                            v661 = Union4{Union4_4{v568, v569, v570, v571, v588, v573}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v661 = Union4{Union4_5{v568, v569, v570, v571, v572, v573}};
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
                                            v661 = Union4{Union4_2{v568, false, v570, v596, v609, v597}};
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
                        v712 = Union6{Union6_1{v661}};
                        break;
                    }
                    case 4: { // TerminalCall
                        Union5 v30 = v13.case4.v0; bool v31 = v13.case4.v1; static_array<Union2,2l> v32 = v13.case4.v2; int v33 = v13.case4.v3; static_array<int,2l> v34 = v13.case4.v4; int v35 = v13.case4.v5;
                        int v36;
                        v36 = v34[v33];
                        Union10 v38;
                        v38 = compare_hands_9(v30, v31, v32, v33, v34, v35);
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
                        Union1 v52;
                        v52 = Union1{Union1_3{v32, v43, v44}};
                        v9.push(v52);
                        v712 = Union6{Union6_0{}};
                        break;
                    }
                    case 5: { // TerminalFold
                        Union5 v14 = v13.case5.v0; bool v15 = v13.case5.v1; static_array<Union2,2l> v16 = v13.case5.v2; int v17 = v13.case5.v3; static_array<int,2l> v18 = v13.case5.v4; int v19 = v13.case5.v5;
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
                        Union1 v28;
                        v28 = Union1{Union1_3{v16, v20, v27}};
                        v9.push(v28);
                        v712 = Union6{Union6_0{}};
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
__device__ inline bool while_method_9(int v0){
    bool v1;
    v1 = v0 > 0l;
    return v1;
}
__device__ __noinline__ void noinline_train_0(unsigned char * v0, unsigned char * v1, curandStatePhilox4_32_10_t & v2, int v3){
    static_array<Union0,2l> v4;
    Union0 v6;
    v6 = Union0{Union0_1{}};
    v4[0l] = v6;
    Union0 v8;
    v8 = Union0{Union0_1{}};
    v4[1l] = v8;
    Union0 v10;
    v10 = Union0{Union0_0{}};
    v4[v3] = v10;
    static_array<Union0,2l> & v12 = v4;
    unsigned int v13 = 63ul;
    static_array_list<Union1,32l> v14;
    v14 = static_array_list<Union1,32l>{};
    static_array_list<Union1,32l> & v16 = v14;
    Union4 v17;
    v17 = Union4{Union4_1{}};
    static_array<float,2l> v18;
    v18 = method_1(v0, v1, v13, v16, v12, v2, v17);
    unsigned int * v19;
    v19 = reinterpret_cast<unsigned int *>(&v0[12582912ull]);
    int * v21;
    v21 = reinterpret_cast<int *>(&v1[262144ull]);
    float * v23;
    v23 = reinterpret_cast<float *>(&v1[262160ull]);
    float * v25;
    v25 = reinterpret_cast<float *>(&v1[524304ull]);
    float * v27;
    v27 = reinterpret_cast<float *>(&v1[786448ull]);
    float * v29;
    v29 = reinterpret_cast<float *>(&v1[1048592ull]);
    float * v31;
    v31 = reinterpret_cast<float *>(&v1[1310736ull]);
    float * v33;
    v33 = reinterpret_cast<float *>(&v1[1572880ull]);
    float * v35;
    v35 = reinterpret_cast<float *>(&v1[1835024ull]);
    int * v37;
    v37 = reinterpret_cast<int *>(&v0[12779520ull]);
    float * v39;
    v39 = reinterpret_cast<float *>(&v0[15925248ull]);
    int * v41;
    v41 = reinterpret_cast<int *>(&v0[19070976ull]);
    int * v43;
    v43 = reinterpret_cast<int *>(&v0[22216704ull]);
    double * v45;
    v45 = reinterpret_cast<double *>(&v0[25362432ull]);
    double * v47;
    v47 = reinterpret_cast<double *>(&v0[37945344ull]);
    double * v49;
    v49 = reinterpret_cast<double *>(&v1[2097168ull]);
    double * v51;
    v51 = reinterpret_cast<double *>(&v1[2883600ull]);
    int * v53;
    v53 = reinterpret_cast<int *>(&v1[3670032ull]);
    int v55;
    v55 = 0l;
    while (while_method_0(v55)){
        int v57;
        v57 = threadIdx.x;
        int v58;
        v58 = blockIdx.x;
        int v59;
        v59 = v58 * 512l;
        int v60;
        v60 = v57 + v59;
        float v61[2l];
        int v62;
        v62 = 0l;
        while (while_method_3(v62)){
            float v64;
            v64 = v18[v62];
            v61[v62] = v64;
            v62 += 1l ;
        }
        assert("Tensor range check" && 0 <= v55 && v55 < 4l);
        assert("Tensor range check" && 0 <= v60 && v60 < 12288l);
        int v66;
        v66 = 12288l * v55;
        int v67;
        v67 = v66 + v60;
        int v68;
        v68 = v53[v67];
        int v69;
        v69 = v68;
        while (while_method_9(v69)){
            v69 -= 1l ;
            assert("Tensor range check" && 0 <= v55 && v55 < 4l);
            assert("Tensor range check" && 0 <= v69 && v69 < 16l);
            assert("Tensor range check" && 0 <= v60 && v60 < 12288l);
            int v71;
            v71 = 12288l * v69;
            int v72;
            v72 = v71 + v60;
            int v73;
            v73 = 196608l * v55;
            int v74;
            v74 = v73 + v72;
            int v75;
            v75 = v37[v74];
            float v76;
            v76 = v39[v74];
            int v77;
            v77 = v41[v74];
            int v78;
            v78 = v43[v74];
            assert("Tensor range check" && 0 <= v77 && v77 < 2l);
            float v79;
            v79 = v61[v77];
            assert("Tensor range check" && 0 <= v55 && v55 < 4l);
            int v80;
            v80 = 16384l * v55;
            assert("Tensor range check" && 0 <= v78 && v78 < 4096l);
            int v81;
            v81 = 4l * v78;
            int v82;
            v82 = v81 + v80;
            float * v83;
            v83 = v23+v82;
            float * v85;
            v85 = v25+v82;
            float * v87;
            v87 = v27+v82;
            float * v89;
            v89 = v29+v82;
            float * v91;
            v91 = v31+v82;
            float * v93;
            v93 = v33+v82;
            float * v95;
            v95 = v35+v82;
            assert("Tensor range check" && 0 <= v55 && v55 < 4l);
            int v97;
            v97 = 393216l * v55;
            assert("Tensor range check" && 0 <= v69 && v69 < 16l);
            int v98;
            v98 = 24576l * v69;
            int v99;
            v99 = v98 + v97;
            assert("Tensor range check" && 0 <= v60 && v60 < 12288l);
            int v100;
            v100 = 2l * v60;
            int v101;
            v101 = v100 + v99;
            double v102[2l];
            int v103;
            v103 = 0l;
            while (while_method_3(v103)){
                assert("Tensor range check" && 0 <= v103 && v103 < 2l);
                int v105;
                v105 = v103 + v101;
                double v106;
                v106 = v45[v105];
                bool v107;
                v107 = v77 == v103;
                double v108;
                if (v107){
                    v108 = 0.0;
                } else {
                    v108 = v106;
                }
                assert("Tensor range check" && 0 <= v103 && v103 < 2l);
                v102[v103] = v108;
                v103 += 1l ;
            }
            double v109;
            v109 = 0.0;
            int v110;
            v110 = 0l;
            while (while_method_3(v110)){
                assert("Tensor range check" && 0 <= v110 && v110 < 2l);
                double v112;
                v112 = v102[v110];
                double v113;
                v113 = v109 + v112;
                v109 = v113;
                v110 += 1l ;
            }
            double v114;
            v114 = 0.0;
            int v115;
            v115 = 0l;
            while (while_method_3(v115)){
                assert("Tensor range check" && 0 <= v115 && v115 < 2l);
                int v117;
                v117 = v115 + v101;
                double v118;
                v118 = v47[v117];
                double v119;
                v119 = v114 + v118;
                v114 = v119;
                v115 += 1l ;
            }
            double v120;
            v120 = v109 - v114;
            double v121;
            v121 = exp(v120);
            float v122;
            v122 = (float)v121;
            float v123;
            v123 = v79 * v122;
            assert("Tensor range check" && 0 <= v75 && v75 < 4l);
            float * v124;
            v124 = v93+v75;
            float * v126;
            v126 = v95+v75;
            float v128;
            v128 = atomicAdd(v124,v123);
            float v129;
            v129 = atomicAdd(v126,v122);
            float * v130;
            v130 = v85+0l;
            float * v132;
            v132 = v89+0l;
            float * v134;
            v134 = v91+0l;
            int v136;
            v136 = sizeof(float *);
            unsigned long long v137;
            v137 = (unsigned long long)v136;
            unsigned long long v138;
            v138 = 512ull * v137;
            unsigned long long v139;
            v139 = 8192ull + v138;
            unsigned long long v140;
            v140 = v139 + 16ull;
            unsigned long long v141;
            v141 = v140 - 1ull;
            unsigned long long v142;
            v142 = v141 % 16ull;
            unsigned long long v143;
            v143 = v141 - v142;
            unsigned long long v144;
            v144 = v143 + v138;
            unsigned long long v145;
            v145 = v144 + 16ull;
            unsigned long long v146;
            v146 = v145 - 1ull;
            unsigned long long v147;
            v147 = v146 % 16ull;
            unsigned long long v148;
            v148 = v146 - v147;
            unsigned long long v149;
            v149 = v148 + v138;
            unsigned long long v150;
            v150 = v149 + 16ull;
            unsigned long long v151;
            v151 = v150 - 1ull;
            unsigned long long v152;
            v152 = v151 % 16ull;
            unsigned long long v153;
            v153 = v151 - v152;
            unsigned long long v154;
            v154 = v153 + v138;
            unsigned long long v155;
            v155 = v154 + 16ull;
            unsigned long long v156;
            v156 = v155 - 1ull;
            unsigned long long v157;
            v157 = v156 % 16ull;
            unsigned long long v158;
            v158 = v156 - v157;
            unsigned long long v159;
            v159 = v158 + 2048ull;
            bool v160;
            v160 = v159 <= 81920ull;
            bool v161;
            v161 = v160 == false;
            if (v161){
                assert("The dynamic shared memory is insufficient to allocate the tensor." && v160);
            } else {
            }
            extern __shared__ unsigned char v163[];
            bool v164;
            v164 = v159 <= v159;
            bool v165;
            v165 = v164 == false;
            if (v165){
                assert("The length of the partition has to be less than or equal to the length of the base array." && v164);
            } else {
            }
            float * v167;
            v167 = reinterpret_cast<float *>(&v163[0ull]);
            int * v169;
            v169 = reinterpret_cast<int *>(&v163[2048ull]);
            float * v171;
            v171 = reinterpret_cast<float *>(&v163[4096ull]);
            float * v173;
            v173 = reinterpret_cast<float *>(&v163[6144ull]);
            float * * v175;
            v175 = reinterpret_cast<float * *>(&v163[8192ull]);
            float * * v177;
            v177 = reinterpret_cast<float * *>(&v163[v143]);
            float * * v179;
            v179 = reinterpret_cast<float * *>(&v163[v148]);
            float * * v181;
            v181 = reinterpret_cast<float * *>(&v163[v153]);
            float * v183;
            v183 = reinterpret_cast<float *>(&v163[v158]);
            int v185;
            v185 = threadIdx.x;
            assert("Tensor range check" && 0 <= v185 && v185 < 512l);
            v167[v185] = v76;
            v169[v185] = v75;
            v171[v185] = v79;
            v173[v185] = v122;
            v175[v185] = v87;
            v177[v185] = v130;
            v179[v185] = v132;
            v181[v185] = v134;
            asm("barrier.cta.sync %0;" :: "r"(0l));
            bool v186;
            v186 = 0l <= v185;
            bool v187;
            v187 = v186 == false;
            if (v187){
                assert("The index needs to be zero or positive." && v186);
            } else {
            }
            int v189;
            v189 = v185 % 1l;
            bool v190;
            v190 = v185 < 512l;
            bool v191;
            v191 = v190 == false;
            if (v191){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v190);
            } else {
            }
            assert("Tensor range check" && 0 <= v185 && v185 < 512l);
            int v193;
            v193 = 0l;
            while (while_method_6(v193)){
                bool v195;
                v195 = v186 && v190;
                bool v196;
                v196 = v195 == false;
                if (v196){
                    assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v195);
                } else {
                }
                bool v198;
                v198 = 0l <= v193;
                bool v200;
                if (v198){
                    bool v199;
                    v199 = v193 < 1l;
                    v200 = v199;
                } else {
                    v200 = false;
                }
                bool v201;
                v201 = v200 == false;
                if (v201){
                    assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v200);
                } else {
                }
                int v203;
                v203 = v193 * 512l;
                int v204;
                v204 = v203 + v185;
                assert("Tensor range check" && 0 <= v193 && v193 < 1l);
                int v205;
                v205 = 512l * v193;
                int v206;
                v206 = v205 + v185;
                float v207;
                v207 = v167[v206];
                int v208;
                v208 = v169[v206];
                float v209;
                v209 = v171[v206];
                float v210;
                v210 = v173[v206];
                float * v211;
                v211 = v175[v206];
                float * v212;
                v212 = v177[v206];
                float * v213;
                v213 = v179[v206];
                float * v214;
                v214 = v181[v206];
                int v215;
                v215 = blockIdx.x;
                int v216;
                v216 = v215 * 512l;
                int v217;
                v217 = v216 + v204;
                assert("Tensor range check" && 0 <= v189 && v189 < 1l);
                int v218;
                v218 = 4l * v189;
                float v219[4l];
                float v220[4l];
                float v221[4l];
                int v222[4l];
                int v223;
                v223 = 0l;
                while (while_method_6(v223)){
                    assert("Tensor range check" && 0 <= v223 && v223 < 1l);
                    int v225;
                    v225 = 4l * v223;
                    assert("Tensor range check" && 0 <= v223 && v223 < 1l);
                    int v226;
                    v226 = v225 + v218;
                    int4* v227;
                    v227 = reinterpret_cast<int4*>(v212 + v226);
                    int4* v228;
                    v228 = reinterpret_cast<int4*>(v219 + v225);
                    assert("Pointer alignment check" && (unsigned long long)(v227) % 4l == 0 && (unsigned long long)(v228) % 4l == 0);
                    *v228 = *v227;
                    int4* v229;
                    v229 = reinterpret_cast<int4*>(v213 + v226);
                    int4* v230;
                    v230 = reinterpret_cast<int4*>(v220 + v225);
                    assert("Pointer alignment check" && (unsigned long long)(v229) % 4l == 0 && (unsigned long long)(v230) % 4l == 0);
                    *v230 = *v229;
                    int4* v231;
                    v231 = reinterpret_cast<int4*>(v214 + v226);
                    int4* v232;
                    v232 = reinterpret_cast<int4*>(v221 + v225);
                    assert("Pointer alignment check" && (unsigned long long)(v231) % 4l == 0 && (unsigned long long)(v232) % 4l == 0);
                    *v232 = *v231;
                    v223 += 1l ;
                }
                int v233;
                v233 = 0l;
                while (while_method_6(v233)){
                    int v235;
                    v235 = 0l;
                    while (while_method_0(v235)){
                        bool v237;
                        v237 = 0l <= v235;
                        bool v239;
                        if (v237){
                            bool v238;
                            v238 = v235 < 4l;
                            v239 = v238;
                        } else {
                            v239 = false;
                        }
                        bool v240;
                        v240 = v239 == false;
                        if (v240){
                            assert("The indices should be inside the range of the dimension." && v239);
                        } else {
                        }
                        bool v242;
                        v242 = 0l <= v189;
                        bool v244;
                        if (v242){
                            bool v243;
                            v243 = v189 < 1l;
                            v244 = v243;
                        } else {
                            v244 = false;
                        }
                        bool v245;
                        v245 = v244 == false;
                        if (v245){
                            assert("The indices should be inside the range of the dimension." && v244);
                        } else {
                        }
                        int v247;
                        v247 = v189 * 4l;
                        int v248;
                        v248 = v235 + v247;
                        bool v249;
                        v249 = 0l <= v233;
                        bool v251;
                        if (v249){
                            bool v250;
                            v250 = v233 < 1l;
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
                        int v254;
                        v254 = v233 * 4l;
                        int v255;
                        v255 = v248 + v254;
                        assert("Tensor range check" && 0 <= v233 && v233 < 1l);
                        assert("Tensor range check" && 0 <= v235 && v235 < 4l);
                        int v256;
                        v256 = 4l * v233;
                        int v257;
                        v257 = v256 + v235;
                        v222[v257] = v255;
                        v235 += 1l ;
                    }
                    v233 += 1l ;
                }
                float v258[4l];
                int v259;
                v259 = 0l;
                while (while_method_6(v259)){
                    int v261;
                    v261 = 0l;
                    while (while_method_0(v261)){
                        assert("Tensor range check" && 0 <= v259 && v259 < 1l);
                        assert("Tensor range check" && 0 <= v261 && v261 < 4l);
                        int v263;
                        v263 = 4l * v259;
                        int v264;
                        v264 = v263 + v261;
                        float v265;
                        v265 = v220[v264];
                        float v266;
                        v266 = v221[v264];
                        bool v267;
                        v267 = v266 == 0.0f;
                        bool v268;
                        v268 = v267 != true;
                        float v270;
                        if (v268){
                            float v269;
                            v269 = v265 / v266;
                            v270 = v269;
                        } else {
                            v270 = 0.0f;
                        }
                        assert("Tensor range check" && 0 <= v259 && v259 < 1l);
                        assert("Tensor range check" && 0 <= v261 && v261 < 4l);
                        v258[v264] = v270;
                        v261 += 1l ;
                    }
                    v259 += 1l ;
                }
                bool v271[4l];
                int v272;
                v272 = 0l;
                while (while_method_6(v272)){
                    int v274;
                    v274 = 0l;
                    while (while_method_0(v274)){
                        assert("Tensor range check" && 0 <= v272 && v272 < 1l);
                        assert("Tensor range check" && 0 <= v274 && v274 < 4l);
                        int v276;
                        v276 = 4l * v272;
                        int v277;
                        v277 = v276 + v274;
                        float v278;
                        v278 = v219[v277];
                        int v279;
                        v279 = v222[v277];
                        bool v280;
                        v280 = v279 < 3l;
                        assert("Tensor range check" && 0 <= v272 && v272 < 1l);
                        assert("Tensor range check" && 0 <= v274 && v274 < 4l);
                        v271[v277] = v280;
                        v274 += 1l ;
                    }
                    v272 += 1l ;
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
                        v288 = v219[v287];
                        bool v289;
                        v289 = v271[v287];
                        float v292;
                        if (v289){
                            bool v290;
                            v290 = 0.0f >= v288;
                            if (v290){
                                v292 = 0.0f;
                            } else {
                                v292 = v288;
                            }
                        } else {
                            v292 = 0.0f;
                        }
                        assert("Tensor range check" && 0 <= v282 && v282 < 1l);
                        assert("Tensor range check" && 0 <= v284 && v284 < 4l);
                        v281[v287] = v292;
                        v284 += 1l ;
                    }
                    v282 += 1l ;
                }
                float v293;
                v293 = 0.0f;
                int v294;
                v294 = 0l;
                while (while_method_6(v294)){
                    int v296;
                    v296 = 0l;
                    while (while_method_0(v296)){
                        assert("Tensor range check" && 0 <= v294 && v294 < 1l);
                        assert("Tensor range check" && 0 <= v296 && v296 < 4l);
                        int v298;
                        v298 = 4l * v294;
                        int v299;
                        v299 = v298 + v296;
                        float v300;
                        v300 = v281[v299];
                        float v301;
                        v301 = v293 + v300;
                        v293 = v301;
                        v296 += 1l ;
                    }
                    v294 += 1l ;
                }
                auto v302 = cooperative_groups::coalesced_threads();
                int v303;
                v303 = threadIdx.x;
                auto v304 = cooperative_groups::labeled_partition(v302,v303);
                Closure1 v305{};
                float v306;
                v306 = cooperative_groups::reduce(v304, v293, v305);
                int v307[4l];
                int v308;
                v308 = 0l;
                while (while_method_6(v308)){
                    int v310;
                    v310 = 0l;
                    while (while_method_0(v310)){
                        assert("Tensor range check" && 0 <= v308 && v308 < 1l);
                        assert("Tensor range check" && 0 <= v310 && v310 < 4l);
                        int v312;
                        v312 = 4l * v308;
                        int v313;
                        v313 = v312 + v310;
                        bool v314;
                        v314 = v271[v313];
                        int v315;
                        if (v314){
                            v315 = 1l;
                        } else {
                            v315 = 0l;
                        }
                        assert("Tensor range check" && 0 <= v308 && v308 < 1l);
                        assert("Tensor range check" && 0 <= v310 && v310 < 4l);
                        v307[v313] = v315;
                        v310 += 1l ;
                    }
                    v308 += 1l ;
                }
                int v316;
                v316 = 0l;
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
                        int v323;
                        v323 = v307[v322];
                        int v324;
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
                Closure2 v328{};
                int v329;
                v329 = cooperative_groups::reduce(v327, v316, v328);
                float v330;
                v330 = (float)v329;
                float v331;
                v331 = 1.0f / v330;
                float v332[4l];
                int v333;
                v333 = 0l;
                while (while_method_6(v333)){
                    int v335;
                    v335 = 0l;
                    while (while_method_0(v335)){
                        assert("Tensor range check" && 0 <= v333 && v333 < 1l);
                        assert("Tensor range check" && 0 <= v335 && v335 < 4l);
                        int v337;
                        v337 = 4l * v333;
                        int v338;
                        v338 = v337 + v335;
                        float v339;
                        v339 = v281[v338];
                        bool v340;
                        v340 = v271[v338];
                        bool v341;
                        v341 = v340 == false;
                        float v346;
                        if (v341){
                            v346 = 0.0f;
                        } else {
                            bool v342;
                            v342 = v306 == 0.0f;
                            bool v343;
                            v343 = v342 != true;
                            if (v343){
                                float v344;
                                v344 = v339 / v306;
                                v346 = v344;
                            } else {
                                v346 = v331;
                            }
                        }
                        assert("Tensor range check" && 0 <= v333 && v333 < 1l);
                        assert("Tensor range check" && 0 <= v335 && v335 < 4l);
                        v332[v338] = v346;
                        v335 += 1l ;
                    }
                    v333 += 1l ;
                }
                float v347[4l];
                int v348;
                v348 = 0l;
                while (while_method_6(v348)){
                    int v350;
                    v350 = 0l;
                    while (while_method_0(v350)){
                        assert("Tensor range check" && 0 <= v348 && v348 < 1l);
                        assert("Tensor range check" && 0 <= v350 && v350 < 4l);
                        int v352;
                        v352 = 4l * v348;
                        int v353;
                        v353 = v352 + v350;
                        float v354;
                        v354 = v258[v353];
                        int v355;
                        v355 = v222[v353];
                        bool v356;
                        v356 = v208 == v355;
                        float v359;
                        if (v356){
                            float v357;
                            v357 = v209 - v354;
                            float v358;
                            v358 = v357 / v207;
                            v359 = v358;
                        } else {
                            v359 = 0.0f;
                        }
                        float v360;
                        v360 = v359 + v354;
                        assert("Tensor range check" && 0 <= v348 && v348 < 1l);
                        assert("Tensor range check" && 0 <= v350 && v350 < 4l);
                        v347[v353] = v360;
                        v350 += 1l ;
                    }
                    v348 += 1l ;
                }
                float v361[4l];
                int v362;
                v362 = 0l;
                while (while_method_6(v362)){
                    int v364;
                    v364 = 0l;
                    while (while_method_0(v364)){
                        assert("Tensor range check" && 0 <= v362 && v362 < 1l);
                        assert("Tensor range check" && 0 <= v364 && v364 < 4l);
                        int v366;
                        v366 = 4l * v362;
                        int v367;
                        v367 = v366 + v364;
                        float v368;
                        v368 = v332[v367];
                        float v369;
                        v369 = v347[v367];
                        float v370;
                        v370 = v368 * v369;
                        assert("Tensor range check" && 0 <= v362 && v362 < 1l);
                        assert("Tensor range check" && 0 <= v364 && v364 < 4l);
                        v361[v367] = v370;
                        v364 += 1l ;
                    }
                    v362 += 1l ;
                }
                float v371;
                v371 = 0.0f;
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
                        float v378;
                        v378 = v361[v377];
                        float v379;
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
                float v383;
                v383 = cooperative_groups::reduce(v382, v371, v305);
                int v384;
                v384 = 0l;
                while (while_method_6(v384)){
                    int v386;
                    v386 = 0l;
                    while (while_method_0(v386)){
                        assert("Tensor range check" && 0 <= v384 && v384 < 1l);
                        assert("Tensor range check" && 0 <= v386 && v386 < 4l);
                        int v388;
                        v388 = 4l * v384;
                        int v389;
                        v389 = v388 + v386;
                        float v390;
                        v390 = v347[v389];
                        int v391;
                        v391 = v222[v389];
                        float v392;
                        v392 = v390 - v383;
                        float v393;
                        v393 = v210 * v392;
                        assert("Tensor range check" && 0 <= v391 && v391 < 4l);
                        float * v394;
                        v394 = v211+v391;
                        float v396;
                        v396 = atomicAdd(v394,v393);
                        v386 += 1l ;
                    }
                    v384 += 1l ;
                }
                int v397;
                v397 = 0l;
                while (while_method_6(v397)){
                    assert("Tensor range check" && 0 <= v397 && v397 < 1l);
                    assert("Tensor range check" && 0 <= v397 && v397 < 1l);
                    v397 += 1l ;
                }
                assert("Tensor range check" && 0 <= v204 && v204 < 512l);
                v183[v204] = v383;
                v193 += 1l ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0l));
            assert("Tensor range check" && 0 <= v185 && v185 < 512l);
            float v399;
            v399 = v183[v185];
            asm("barrier.cta.sync %0;" :: "r"(0l));
            assert("Tensor range check" && 0 <= v77 && v77 < 2l);
            v61[v77] = v399;
        }
        int v400;
        v400 = threadIdx.x;
        int v401;
        v401 = blockIdx.x;
        int v402;
        v402 = v401 * 512l;
        int v403;
        v403 = v400 + v402;
        assert("Tensor range check" && 0 <= v55 && v55 < 4l);
        int v404;
        v404 = 24576l * v55;
        assert("Tensor range check" && 0 <= v403 && v403 < 12288l);
        int v405;
        v405 = 2l * v403;
        int v406;
        v406 = v405 + v404;
        double * v407;
        v407 = v49+v406;
        double * v409;
        v409 = v51+v406;
        double * v411;
        v411 = v407+0l;
        double * v413;
        v413 = v409+0l;
        double * v415;
        v415 = v407+0l;
        double * v417;
        v417 = v409+0l;
        int v419;
        v419 = sizeof(double *);
        unsigned long long v420;
        v420 = (unsigned long long)v419;
        unsigned long long v421;
        v421 = 512ull * v420;
        unsigned long long v422;
        v422 = v421 + 16ull;
        unsigned long long v423;
        v423 = v422 - 1ull;
        unsigned long long v424;
        v424 = v423 % 16ull;
        unsigned long long v425;
        v425 = v423 - v424;
        unsigned long long v426;
        v426 = v425 + v421;
        unsigned long long v427;
        v427 = v426 + 16ull;
        unsigned long long v428;
        v428 = v427 - 1ull;
        unsigned long long v429;
        v429 = v428 % 16ull;
        unsigned long long v430;
        v430 = v428 - v429;
        unsigned long long v431;
        v431 = v430 + v421;
        unsigned long long v432;
        v432 = v431 + 16ull;
        unsigned long long v433;
        v433 = v432 - 1ull;
        unsigned long long v434;
        v434 = v433 % 16ull;
        unsigned long long v435;
        v435 = v433 - v434;
        unsigned long long v436;
        v436 = v435 + v421;
        bool v437;
        v437 = v436 <= 81920ull;
        bool v438;
        v438 = v437 == false;
        if (v438){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v437);
        } else {
        }
        extern __shared__ unsigned char v440[];
        bool v441;
        v441 = v436 <= v436;
        bool v442;
        v442 = v441 == false;
        if (v442){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v441);
        } else {
        }
        double * * v444;
        v444 = reinterpret_cast<double * *>(&v440[0ull]);
        double * * v446;
        v446 = reinterpret_cast<double * *>(&v440[v425]);
        double * * v448;
        v448 = reinterpret_cast<double * *>(&v440[v430]);
        double * * v450;
        v450 = reinterpret_cast<double * *>(&v440[v435]);
        int v452;
        v452 = threadIdx.x;
        assert("Tensor range check" && 0 <= v452 && v452 < 512l);
        v444[v452] = v411;
        v446[v452] = v413;
        v448[v452] = v415;
        v450[v452] = v417;
        asm("barrier.cta.sync %0;" :: "r"(0l));
        bool v453;
        v453 = 0l <= v452;
        bool v454;
        v454 = v453 == false;
        if (v454){
            assert("The index needs to be zero or positive." && v453);
        } else {
        }
        int v456;
        v456 = v452 % 1l;
        bool v457;
        v457 = v452 < 512l;
        bool v458;
        v458 = v457 == false;
        if (v458){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v457);
        } else {
        }
        assert("Tensor range check" && 0 <= v452 && v452 < 512l);
        int v460;
        v460 = 0l;
        while (while_method_6(v460)){
            bool v462;
            v462 = v453 && v457;
            bool v463;
            v463 = v462 == false;
            if (v463){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v462);
            } else {
            }
            bool v465;
            v465 = 0l <= v460;
            bool v467;
            if (v465){
                bool v466;
                v466 = v460 < 1l;
                v467 = v466;
            } else {
                v467 = false;
            }
            bool v468;
            v468 = v467 == false;
            if (v468){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v467);
            } else {
            }
            int v470;
            v470 = v460 * 512l;
            int v471;
            v471 = v470 + v452;
            assert("Tensor range check" && 0 <= v460 && v460 < 1l);
            int v472;
            v472 = 512l * v460;
            int v473;
            v473 = v472 + v452;
            double * v474;
            v474 = v444[v473];
            double * v475;
            v475 = v446[v473];
            double * v476;
            v476 = v448[v473];
            double * v477;
            v477 = v450[v473];
            int v478;
            v478 = blockIdx.x;
            int v479;
            v479 = v478 * 512l;
            int v480;
            v480 = v479 + v471;
            assert("Tensor range check" && 0 <= v456 && v456 < 1l);
            int v481;
            v481 = 2l * v456;
            double v482[2l];
            double v483[2l];
            int v484[2l];
            int v485;
            v485 = 0l;
            while (while_method_6(v485)){
                assert("Tensor range check" && 0 <= v485 && v485 < 1l);
                int v487;
                v487 = 2l * v485;
                assert("Tensor range check" && 0 <= v485 && v485 < 1l);
                int v488;
                v488 = v487 + v481;
                int4* v489;
                v489 = reinterpret_cast<int4*>(v474 + v488);
                int4* v490;
                v490 = reinterpret_cast<int4*>(v482 + v487);
                assert("Pointer alignment check" && (unsigned long long)(v489) % 2l == 0 && (unsigned long long)(v490) % 2l == 0);
                *v490 = *v489;
                int4* v491;
                v491 = reinterpret_cast<int4*>(v475 + v488);
                int4* v492;
                v492 = reinterpret_cast<int4*>(v483 + v487);
                assert("Pointer alignment check" && (unsigned long long)(v491) % 2l == 0 && (unsigned long long)(v492) % 2l == 0);
                *v492 = *v491;
                v485 += 1l ;
            }
            int v493;
            v493 = 0l;
            while (while_method_6(v493)){
                int v495;
                v495 = 0l;
                while (while_method_3(v495)){
                    bool v497;
                    v497 = 0l <= v495;
                    bool v499;
                    if (v497){
                        bool v498;
                        v498 = v495 < 2l;
                        v499 = v498;
                    } else {
                        v499 = false;
                    }
                    bool v500;
                    v500 = v499 == false;
                    if (v500){
                        assert("The indices should be inside the range of the dimension." && v499);
                    } else {
                    }
                    bool v502;
                    v502 = 0l <= v456;
                    bool v504;
                    if (v502){
                        bool v503;
                        v503 = v456 < 1l;
                        v504 = v503;
                    } else {
                        v504 = false;
                    }
                    bool v505;
                    v505 = v504 == false;
                    if (v505){
                        assert("The indices should be inside the range of the dimension." && v504);
                    } else {
                    }
                    int v507;
                    v507 = v456 * 2l;
                    int v508;
                    v508 = v495 + v507;
                    bool v509;
                    v509 = 0l <= v493;
                    bool v511;
                    if (v509){
                        bool v510;
                        v510 = v493 < 1l;
                        v511 = v510;
                    } else {
                        v511 = false;
                    }
                    bool v512;
                    v512 = v511 == false;
                    if (v512){
                        assert("The indices should be inside the range of the dimension." && v511);
                    } else {
                    }
                    int v514;
                    v514 = v493 * 2l;
                    int v515;
                    v515 = v508 + v514;
                    assert("Tensor range check" && 0 <= v493 && v493 < 1l);
                    assert("Tensor range check" && 0 <= v495 && v495 < 2l);
                    int v516;
                    v516 = 2l * v493;
                    int v517;
                    v517 = v516 + v495;
                    v484[v517] = v515;
                    v495 += 1l ;
                }
                v493 += 1l ;
            }
            double v518[2l];
            double v519[2l];
            int v520;
            v520 = 0l;
            while (while_method_6(v520)){
                int v522;
                v522 = 0l;
                while (while_method_3(v522)){
                    assert("Tensor range check" && 0 <= v520 && v520 < 1l);
                    assert("Tensor range check" && 0 <= v522 && v522 < 2l);
                    int v524;
                    v524 = 2l * v520;
                    int v525;
                    v525 = v524 + v522;
                    double v526;
                    v526 = v482[v525];
                    double v527;
                    v527 = v483[v525];
                    assert("Tensor range check" && 0 <= v520 && v520 < 1l);
                    assert("Tensor range check" && 0 <= v522 && v522 < 2l);
                    v518[v525] = 0.0;
                    v519[v525] = 0.0;
                    v522 += 1l ;
                }
                v520 += 1l ;
            }
            int v528;
            v528 = 0l;
            while (while_method_6(v528)){
                assert("Tensor range check" && 0 <= v528 && v528 < 1l);
                int v530;
                v530 = 2l * v528;
                int v531;
                v531 = v530 + v481;
                assert("Tensor range check" && 0 <= v528 && v528 < 1l);
                int4* v532;
                v532 = reinterpret_cast<int4*>(v518 + v530);
                int4* v533;
                v533 = reinterpret_cast<int4*>(v476 + v531);
                assert("Pointer alignment check" && (unsigned long long)(v532) % 2l == 0 && (unsigned long long)(v533) % 2l == 0);
                *v533 = *v532;
                int4* v534;
                v534 = reinterpret_cast<int4*>(v519 + v530);
                int4* v535;
                v535 = reinterpret_cast<int4*>(v477 + v531);
                assert("Pointer alignment check" && (unsigned long long)(v534) % 2l == 0 && (unsigned long long)(v535) % 2l == 0);
                *v535 = *v534;
                v528 += 1l ;
            }
            assert("Tensor range check" && 0 <= v471 && v471 < 512l);
            v460 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        assert("Tensor range check" && 0 <= v452 && v452 < 512l);
        asm("barrier.cta.sync %0;" :: "r"(0l));
        assert("Tensor range check" && 0 <= v55 && v55 < 4l);
        assert("Tensor range check" && 0 <= v403 && v403 < 12288l);
        int v536;
        v536 = v66 + v403;
        v53[v536] = 0l;
        v55 += 1l ;
    }
    return ;
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1, float * v2, float * v3, float * v4) {
    auto v5 = cooperative_groups::this_grid();
    unsigned long long v6;
    v6 = clock64();
    int v7;
    v7 = threadIdx.x;
    int v8;
    v8 = blockIdx.x;
    int v9;
    v9 = v8 * 512l;
    int v10;
    v10 = v7 + v9;
    unsigned long long v11;
    v11 = (unsigned long long)v10;
    curandStatePhilox4_32_10_t v12;
    curand_init(v6,v11,0ull,&v12);
    int v13;
    v13 = 0l;
    while (while_method_0(v13)){
        int v15;
        v15 = 0l;
        while (while_method_1(v15)){
            int v17;
            v17 = 0l;
            noinline_train_0(v0, v1, v12, v17);
            int v18;
            v18 = 1l;
            noinline_train_0(v0, v1, v12, v18);
            v15 += 1l ;
        }
        unsigned int * v19;
        v19 = reinterpret_cast<unsigned int *>(&v0[12582912ull]);
        int * v21;
        v21 = reinterpret_cast<int *>(&v1[262144ull]);
        float * v23;
        v23 = reinterpret_cast<float *>(&v1[262160ull]);
        float * v25;
        v25 = reinterpret_cast<float *>(&v1[524304ull]);
        float * v27;
        v27 = reinterpret_cast<float *>(&v1[786448ull]);
        float * v29;
        v29 = reinterpret_cast<float *>(&v1[1048592ull]);
        float * v31;
        v31 = reinterpret_cast<float *>(&v1[1310736ull]);
        float * v33;
        v33 = reinterpret_cast<float *>(&v1[1572880ull]);
        float * v35;
        v35 = reinterpret_cast<float *>(&v1[1835024ull]);
        int * v37;
        v37 = reinterpret_cast<int *>(&v0[12779520ull]);
        float * v39;
        v39 = reinterpret_cast<float *>(&v0[15925248ull]);
        int * v41;
        v41 = reinterpret_cast<int *>(&v0[19070976ull]);
        int * v43;
        v43 = reinterpret_cast<int *>(&v0[22216704ull]);
        double * v45;
        v45 = reinterpret_cast<double *>(&v0[25362432ull]);
        double * v47;
        v47 = reinterpret_cast<double *>(&v0[37945344ull]);
        double * v49;
        v49 = reinterpret_cast<double *>(&v1[2097168ull]);
        double * v51;
        v51 = reinterpret_cast<double *>(&v1[2883600ull]);
        int * v53;
        v53 = reinterpret_cast<int *>(&v1[3670032ull]);
        v5.sync() ;
        int v55;
        v55 = threadIdx.x;
        int v56;
        v56 = blockIdx.x;
        int v57;
        v57 = v56 * 512l;
        int v58;
        v58 = v55 + v57;
        bool v59;
        v59 = v58 == 0l;
        if (v59){
            int v60;
            v60 = 0l;
            int v61;
            v61 = 4l;
            int v62;
            v62 = int_range_8(v61, v60, v12);
            v21[0l] = v62;
        } else {
        }
        __syncwarp();
        int v63;
        v63 = threadIdx.x;
        bool v64;
        v64 = 0l <= v63;
        bool v65;
        v65 = v64 == false;
        if (v65){
            assert("The index needs to be zero or positive." && v64);
        } else {
        }
        int v67;
        v67 = v63 % 1l;
        int v68;
        v68 = v63 % 512l;
        int v69;
        v69 = v63 / 512l;
        bool v70;
        v70 = v69 < 1l;
        bool v71;
        v71 = v70 == false;
        if (v71){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v70);
        } else {
        }
        assert("Tensor range check" && 0 <= v69 && v69 < 1l);
        assert("Tensor range check" && 0 <= v68 && v68 < 512l);
        assert("Tensor range check" && 0 <= v67 && v67 < 1l);
        int v73;
        v73 = 4l * v67;
        int v74;
        v74 = 4l * v68;
        int v75;
        v75 = v74 + v73;
        int v76;
        v76 = 16384l * v69;
        int v77;
        v77 = v76 + v75;
        assert("Tensor range check" && 0 <= v69 && v69 < 1l);
        assert("Tensor range check" && 0 <= v68 && v68 < 512l);
        assert("Tensor range check" && 0 <= v67 && v67 < 1l);
        int v78;
        v78 = blockIdx.x;
        int v79;
        v79 = v78;
        while (while_method_8(v79)){
            bool v81;
            v81 = 0l <= v79;
            bool v82;
            v82 = v81 == false;
            if (v82){
                assert("The index needs to be zero or positive." && v81);
            } else {
            }
            int v84;
            v84 = v79 % 8l;
            int v85;
            v85 = v79 / 8l;
            bool v86;
            v86 = v85 < 4l;
            bool v87;
            v87 = v86 == false;
            if (v87){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v86);
            } else {
            }
            assert("Tensor range check" && 0 <= v85 && v85 < 4l);
            assert("Tensor range check" && 0 <= v84 && v84 < 8l);
            int v89;
            v89 = 2048l * v84;
            int v90;
            v90 = v89 + v77;
            int v91;
            v91 = 16384l * v85;
            int v92;
            v92 = v91 + v90;
            float v93[4l];
            float v94[4l];
            float v95[4l];
            float v96[4l];
            float v97[4l];
            float v98[4l];
            float v99[4l];
            int v100[4l];
            int v101;
            v101 = 0l;
            while (while_method_6(v101)){
                assert("Tensor range check" && 0 <= v101 && v101 < 1l);
                int v103;
                v103 = 4l * v101;
                assert("Tensor range check" && 0 <= v101 && v101 < 1l);
                int v104;
                v104 = v103 + v92;
                int4* v105;
                v105 = reinterpret_cast<int4*>(v23 + v104);
                int4* v106;
                v106 = reinterpret_cast<int4*>(v93 + v103);
                assert("Pointer alignment check" && (unsigned long long)(v105) % 4l == 0 && (unsigned long long)(v106) % 4l == 0);
                *v106 = *v105;
                int4* v107;
                v107 = reinterpret_cast<int4*>(v25 + v104);
                int4* v108;
                v108 = reinterpret_cast<int4*>(v94 + v103);
                assert("Pointer alignment check" && (unsigned long long)(v107) % 4l == 0 && (unsigned long long)(v108) % 4l == 0);
                *v108 = *v107;
                int4* v109;
                v109 = reinterpret_cast<int4*>(v27 + v104);
                int4* v110;
                v110 = reinterpret_cast<int4*>(v95 + v103);
                assert("Pointer alignment check" && (unsigned long long)(v109) % 4l == 0 && (unsigned long long)(v110) % 4l == 0);
                *v110 = *v109;
                int4* v111;
                v111 = reinterpret_cast<int4*>(v29 + v104);
                int4* v112;
                v112 = reinterpret_cast<int4*>(v96 + v103);
                assert("Pointer alignment check" && (unsigned long long)(v111) % 4l == 0 && (unsigned long long)(v112) % 4l == 0);
                *v112 = *v111;
                int4* v113;
                v113 = reinterpret_cast<int4*>(v31 + v104);
                int4* v114;
                v114 = reinterpret_cast<int4*>(v97 + v103);
                assert("Pointer alignment check" && (unsigned long long)(v113) % 4l == 0 && (unsigned long long)(v114) % 4l == 0);
                *v114 = *v113;
                int4* v115;
                v115 = reinterpret_cast<int4*>(v33 + v104);
                int4* v116;
                v116 = reinterpret_cast<int4*>(v98 + v103);
                assert("Pointer alignment check" && (unsigned long long)(v115) % 4l == 0 && (unsigned long long)(v116) % 4l == 0);
                *v116 = *v115;
                int4* v117;
                v117 = reinterpret_cast<int4*>(v35 + v104);
                int4* v118;
                v118 = reinterpret_cast<int4*>(v99 + v103);
                assert("Pointer alignment check" && (unsigned long long)(v117) % 4l == 0 && (unsigned long long)(v118) % 4l == 0);
                *v118 = *v117;
                v101 += 1l ;
            }
            int v119;
            v119 = 0l;
            while (while_method_6(v119)){
                int v121;
                v121 = 0l;
                while (while_method_0(v121)){
                    bool v123;
                    v123 = 0l <= v121;
                    bool v125;
                    if (v123){
                        bool v124;
                        v124 = v121 < 4l;
                        v125 = v124;
                    } else {
                        v125 = false;
                    }
                    bool v126;
                    v126 = v125 == false;
                    if (v126){
                        assert("The indices should be inside the range of the dimension." && v125);
                    } else {
                    }
                    bool v128;
                    v128 = 0l <= v67;
                    bool v130;
                    if (v128){
                        bool v129;
                        v129 = v67 < 1l;
                        v130 = v129;
                    } else {
                        v130 = false;
                    }
                    bool v131;
                    v131 = v130 == false;
                    if (v131){
                        assert("The indices should be inside the range of the dimension." && v130);
                    } else {
                    }
                    int v133;
                    v133 = v67 * 4l;
                    int v134;
                    v134 = v121 + v133;
                    bool v135;
                    v135 = 0l <= v119;
                    bool v137;
                    if (v135){
                        bool v136;
                        v136 = v119 < 1l;
                        v137 = v136;
                    } else {
                        v137 = false;
                    }
                    bool v138;
                    v138 = v137 == false;
                    if (v138){
                        assert("The indices should be inside the range of the dimension." && v137);
                    } else {
                    }
                    int v140;
                    v140 = v119 * 4l;
                    int v141;
                    v141 = v134 + v140;
                    assert("Tensor range check" && 0 <= v119 && v119 < 1l);
                    assert("Tensor range check" && 0 <= v121 && v121 < 4l);
                    int v142;
                    v142 = 4l * v119;
                    int v143;
                    v143 = v142 + v121;
                    v100[v143] = v141;
                    v121 += 1l ;
                }
                v119 += 1l ;
            }
            bool v144;
            v144 = 0l <= v69;
            bool v145;
            v145 = v144 && v70;
            bool v146;
            v146 = v145 == false;
            if (v146){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v145);
            } else {
            }
            bool v148;
            v148 = 0l <= v68;
            bool v150;
            if (v148){
                bool v149;
                v149 = v68 < 512l;
                v150 = v149;
            } else {
                v150 = false;
            }
            bool v151;
            v151 = v150 == false;
            if (v151){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v150);
            } else {
            }
            bool v153;
            v153 = 0l <= v85;
            bool v154;
            v154 = v153 && v86;
            bool v155;
            v155 = v154 == false;
            if (v155){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v154);
            } else {
            }
            bool v157;
            v157 = 0l <= v84;
            bool v159;
            if (v157){
                bool v158;
                v158 = v84 < 8l;
                v159 = v158;
            } else {
                v159 = false;
            }
            bool v160;
            v160 = v159 == false;
            if (v160){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v159);
            } else {
            }
            int v162;
            v162 = v84 * 512l;
            int v163;
            v163 = v85 + v69;
            int v164;
            v164 = v162 + v68;
            bool v165[4l];
            int v166;
            v166 = 0l;
            while (while_method_6(v166)){
                int v168;
                v168 = 0l;
                while (while_method_0(v168)){
                    assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                    assert("Tensor range check" && 0 <= v168 && v168 < 4l);
                    int v170;
                    v170 = 4l * v166;
                    int v171;
                    v171 = v170 + v168;
                    float v172;
                    v172 = v95[v171];
                    bool v173;
                    v173 = v172 == 0.0f;
                    bool v174;
                    v174 = v173 != true;
                    assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                    assert("Tensor range check" && 0 <= v168 && v168 < 4l);
                    v165[v171] = v174;
                    v168 += 1l ;
                }
                v166 += 1l ;
            }
            bool v175;
            v175 = false;
            int v176;
            v176 = 0l;
            while (while_method_6(v176)){
                int v178;
                v178 = 0l;
                while (while_method_0(v178)){
                    assert("Tensor range check" && 0 <= v176 && v176 < 1l);
                    assert("Tensor range check" && 0 <= v178 && v178 < 4l);
                    int v180;
                    v180 = 4l * v176;
                    int v181;
                    v181 = v180 + v178;
                    bool v182;
                    v182 = v165[v181];
                    bool v183;
                    v183 = v175 || v182;
                    v175 = v183;
                    v178 += 1l ;
                }
                v176 += 1l ;
            }
            auto v184 = cooperative_groups::coalesced_threads();
            int v185;
            v185 = threadIdx.x;
            auto v186 = cooperative_groups::labeled_partition(v184,v185);
            Closure8 v187{};
            bool v188;
            v188 = cooperative_groups::reduce(v186, v175, v187);
            if (v188){
                float v189[4l];
                int v190;
                v190 = 0l;
                while (while_method_6(v190)){
                    int v192;
                    v192 = 0l;
                    while (while_method_0(v192)){
                        assert("Tensor range check" && 0 <= v190 && v190 < 1l);
                        assert("Tensor range check" && 0 <= v192 && v192 < 4l);
                        int v194;
                        v194 = 4l * v190;
                        int v195;
                        v195 = v194 + v192;
                        float v196;
                        v196 = v94[v195];
                        float v197;
                        v197 = v95[v195];
                        float v198;
                        v198 = v196 + v197;
                        bool v199;
                        v199 = 0.0f >= v198;
                        float v200;
                        if (v199){
                            v200 = 0.0f;
                        } else {
                            v200 = v198;
                        }
                        assert("Tensor range check" && 0 <= v190 && v190 < 1l);
                        assert("Tensor range check" && 0 <= v192 && v192 < 4l);
                        v189[v195] = v200;
                        v192 += 1l ;
                    }
                    v190 += 1l ;
                }
                float v201[4l];
                int v202;
                v202 = 0l;
                while (while_method_6(v202)){
                    int v204;
                    v204 = 0l;
                    while (while_method_0(v204)){
                        assert("Tensor range check" && 0 <= v202 && v202 < 1l);
                        assert("Tensor range check" && 0 <= v204 && v204 < 4l);
                        int v206;
                        v206 = 4l * v202;
                        int v207;
                        v207 = v206 + v204;
                        float v208;
                        v208 = v189[v207];
                        bool v209;
                        v209 = 0.0f >= v208;
                        float v210;
                        if (v209){
                            v210 = 0.0f;
                        } else {
                            v210 = v208;
                        }
                        assert("Tensor range check" && 0 <= v202 && v202 < 1l);
                        assert("Tensor range check" && 0 <= v204 && v204 < 4l);
                        v201[v207] = v210;
                        v204 += 1l ;
                    }
                    v202 += 1l ;
                }
                float v211;
                v211 = 0.0f;
                int v212;
                v212 = 0l;
                while (while_method_6(v212)){
                    int v214;
                    v214 = 0l;
                    while (while_method_0(v214)){
                        assert("Tensor range check" && 0 <= v212 && v212 < 1l);
                        assert("Tensor range check" && 0 <= v214 && v214 < 4l);
                        int v216;
                        v216 = 4l * v212;
                        int v217;
                        v217 = v216 + v214;
                        float v218;
                        v218 = v201[v217];
                        float v219;
                        v219 = v211 + v218;
                        v211 = v219;
                        v214 += 1l ;
                    }
                    v212 += 1l ;
                }
                auto v220 = cooperative_groups::coalesced_threads();
                int v221;
                v221 = threadIdx.x;
                auto v222 = cooperative_groups::labeled_partition(v220,v221);
                Closure1 v223{};
                float v224;
                v224 = cooperative_groups::reduce(v222, v211, v223);
                float v225[4l];
                int v226;
                v226 = 0l;
                while (while_method_6(v226)){
                    int v228;
                    v228 = 0l;
                    while (while_method_0(v228)){
                        assert("Tensor range check" && 0 <= v226 && v226 < 1l);
                        assert("Tensor range check" && 0 <= v228 && v228 < 4l);
                        int v230;
                        v230 = 4l * v226;
                        int v231;
                        v231 = v230 + v228;
                        float v232;
                        v232 = v201[v231];
                        bool v233;
                        v233 = v224 == 0.0f;
                        bool v234;
                        v234 = v233 != true;
                        float v236;
                        if (v234){
                            float v235;
                            v235 = v232 / v224;
                            v236 = v235;
                        } else {
                            v236 = 0.25f;
                        }
                        assert("Tensor range check" && 0 <= v226 && v226 < 1l);
                        assert("Tensor range check" && 0 <= v228 && v228 < 4l);
                        v225[v231] = v236;
                        v228 += 1l ;
                    }
                    v226 += 1l ;
                }
                float v237[4l];
                int v238;
                v238 = 0l;
                while (while_method_6(v238)){
                    int v240;
                    v240 = 0l;
                    while (while_method_0(v240)){
                        assert("Tensor range check" && 0 <= v238 && v238 < 1l);
                        assert("Tensor range check" && 0 <= v240 && v240 < 4l);
                        int v242;
                        v242 = 4l * v238;
                        int v243;
                        v243 = v242 + v240;
                        float v244;
                        v244 = v93[v243];
                        float v245;
                        v245 = v225[v243];
                        float v246;
                        v246 = v244 + v245;
                        assert("Tensor range check" && 0 <= v238 && v238 < 1l);
                        assert("Tensor range check" && 0 <= v240 && v240 < 4l);
                        v237[v243] = v246;
                        v240 += 1l ;
                    }
                    v238 += 1l ;
                }
                float v247[4l];
                int v248;
                v248 = 0l;
                while (while_method_6(v248)){
                    int v250;
                    v250 = 0l;
                    while (while_method_0(v250)){
                        assert("Tensor range check" && 0 <= v248 && v248 < 1l);
                        assert("Tensor range check" && 0 <= v250 && v250 < 4l);
                        int v252;
                        v252 = 4l * v248;
                        int v253;
                        v253 = v252 + v250;
                        float v254;
                        v254 = v237[v253];
                        float v255;
                        v255 = -v254;
                        bool v256;
                        v256 = v254 >= v255;
                        float v257;
                        if (v256){
                            v257 = v254;
                        } else {
                            v257 = v255;
                        }
                        assert("Tensor range check" && 0 <= v248 && v248 < 1l);
                        assert("Tensor range check" && 0 <= v250 && v250 < 4l);
                        v247[v253] = v257;
                        v250 += 1l ;
                    }
                    v248 += 1l ;
                }
                float v258;
                v258 = 0.0f;
                int v259;
                v259 = 0l;
                while (while_method_6(v259)){
                    int v261;
                    v261 = 0l;
                    while (while_method_0(v261)){
                        assert("Tensor range check" && 0 <= v259 && v259 < 1l);
                        assert("Tensor range check" && 0 <= v261 && v261 < 4l);
                        int v263;
                        v263 = 4l * v259;
                        int v264;
                        v264 = v263 + v261;
                        float v265;
                        v265 = v247[v264];
                        float v266;
                        v266 = v258 + v265;
                        v258 = v266;
                        v261 += 1l ;
                    }
                    v259 += 1l ;
                }
                auto v267 = cooperative_groups::coalesced_threads();
                int v268;
                v268 = threadIdx.x;
                auto v269 = cooperative_groups::labeled_partition(v267,v268);
                float v270;
                v270 = cooperative_groups::reduce(v269, v258, v223);
                bool v271;
                v271 = v270 > 100.0f;
                float v273;
                if (v271){
                    float v272;
                    v272 = 100.0f / v270;
                    v273 = v272;
                } else {
                    v273 = 1.0f;
                }
                float v274[4l];
                int v275;
                v275 = 0l;
                while (while_method_6(v275)){
                    int v277;
                    v277 = 0l;
                    while (while_method_0(v277)){
                        assert("Tensor range check" && 0 <= v275 && v275 < 1l);
                        assert("Tensor range check" && 0 <= v277 && v277 < 4l);
                        int v279;
                        v279 = 4l * v275;
                        int v280;
                        v280 = v279 + v277;
                        float v281;
                        v281 = v247[v280];
                        float v282;
                        v282 = v273 * v281;
                        assert("Tensor range check" && 0 <= v275 && v275 < 1l);
                        assert("Tensor range check" && 0 <= v277 && v277 < 4l);
                        v274[v280] = v282;
                        v277 += 1l ;
                    }
                    v275 += 1l ;
                }
                float v283[4l];
                float v284[4l];
                int v285;
                v285 = 0l;
                while (while_method_6(v285)){
                    int v287;
                    v287 = 0l;
                    while (while_method_0(v287)){
                        assert("Tensor range check" && 0 <= v285 && v285 < 1l);
                        assert("Tensor range check" && 0 <= v287 && v287 < 4l);
                        int v289;
                        v289 = 4l * v285;
                        int v290;
                        v290 = v289 + v287;
                        float v291;
                        v291 = v93[v290];
                        float v292;
                        v292 = v94[v290];
                        float v293;
                        v293 = v95[v290];
                        float v294;
                        v294 = v96[v290];
                        float v295;
                        v295 = v97[v290];
                        float v296;
                        v296 = v98[v290];
                        float v297;
                        v297 = v99[v290];
                        float v298;
                        v298 = v294 + v296;
                        float v299;
                        v299 = v295 + v297;
                        assert("Tensor range check" && 0 <= v285 && v285 < 1l);
                        assert("Tensor range check" && 0 <= v287 && v287 < 4l);
                        v283[v290] = v298;
                        v284[v290] = v299;
                        v287 += 1l ;
                    }
                    v285 += 1l ;
                }
                int v300;
                v300 = 0l;
                while (while_method_6(v300)){
                    int v302;
                    v302 = 0l;
                    while (while_method_0(v302)){
                        assert("Tensor range check" && 0 <= v300 && v300 < 1l);
                        assert("Tensor range check" && 0 <= v302 && v302 < 4l);
                        int v304;
                        v304 = 4l * v300;
                        int v305;
                        v305 = v304 + v302;
                        float v306;
                        v306 = v274[v305];
                        float v307;
                        v307 = v189[v305];
                        float v308;
                        v308 = v283[v305];
                        float v309;
                        v309 = v284[v305];
                        assert("Tensor range check" && 0 <= v300 && v300 < 1l);
                        assert("Tensor range check" && 0 <= v302 && v302 < 4l);
                        v93[v305] = v306;
                        v94[v305] = v307;
                        v95[v305] = 0.0f;
                        v96[v305] = v308;
                        v97[v305] = v309;
                        v98[v305] = 0.0f;
                        v99[v305] = 0.0f;
                        v302 += 1l ;
                    }
                    v300 += 1l ;
                }
            } else {
            }
            assert("Tensor range check" && 0 <= v85 && v85 < 4l);
            assert("Tensor range check" && 0 <= v84 && v84 < 8l);
            int v310;
            v310 = 0l;
            while (while_method_6(v310)){
                assert("Tensor range check" && 0 <= v310 && v310 < 1l);
                int v312;
                v312 = 4l * v310;
                int v313;
                v313 = v312 + v92;
                assert("Tensor range check" && 0 <= v310 && v310 < 1l);
                int4* v314;
                v314 = reinterpret_cast<int4*>(v93 + v312);
                int4* v315;
                v315 = reinterpret_cast<int4*>(v23 + v313);
                assert("Pointer alignment check" && (unsigned long long)(v314) % 4l == 0 && (unsigned long long)(v315) % 4l == 0);
                *v315 = *v314;
                int4* v316;
                v316 = reinterpret_cast<int4*>(v94 + v312);
                int4* v317;
                v317 = reinterpret_cast<int4*>(v25 + v313);
                assert("Pointer alignment check" && (unsigned long long)(v316) % 4l == 0 && (unsigned long long)(v317) % 4l == 0);
                *v317 = *v316;
                int4* v318;
                v318 = reinterpret_cast<int4*>(v95 + v312);
                int4* v319;
                v319 = reinterpret_cast<int4*>(v27 + v313);
                assert("Pointer alignment check" && (unsigned long long)(v318) % 4l == 0 && (unsigned long long)(v319) % 4l == 0);
                *v319 = *v318;
                int4* v320;
                v320 = reinterpret_cast<int4*>(v96 + v312);
                int4* v321;
                v321 = reinterpret_cast<int4*>(v29 + v313);
                assert("Pointer alignment check" && (unsigned long long)(v320) % 4l == 0 && (unsigned long long)(v321) % 4l == 0);
                *v321 = *v320;
                int4* v322;
                v322 = reinterpret_cast<int4*>(v97 + v312);
                int4* v323;
                v323 = reinterpret_cast<int4*>(v31 + v313);
                assert("Pointer alignment check" && (unsigned long long)(v322) % 4l == 0 && (unsigned long long)(v323) % 4l == 0);
                *v323 = *v322;
                int4* v324;
                v324 = reinterpret_cast<int4*>(v98 + v312);
                int4* v325;
                v325 = reinterpret_cast<int4*>(v33 + v313);
                assert("Pointer alignment check" && (unsigned long long)(v324) % 4l == 0 && (unsigned long long)(v325) % 4l == 0);
                *v325 = *v324;
                int4* v326;
                v326 = reinterpret_cast<int4*>(v99 + v312);
                int4* v327;
                v327 = reinterpret_cast<int4*>(v35 + v313);
                assert("Pointer alignment check" && (unsigned long long)(v326) % 4l == 0 && (unsigned long long)(v327) % 4l == 0);
                *v327 = *v326;
                v310 += 1l ;
            }
            v79 += 24l ;
        }
        v5.sync() ;
        v13 += 1l ;
    }
    int v328;
    v328 = threadIdx.x;
    int v329;
    v329 = blockIdx.x;
    int v330;
    v330 = v329 * 512l;
    int v331;
    v331 = v328 + v330;
    int v332;
    v332 = v331;
    while (while_method_0(v332)){
        bool v334;
        v334 = 0l <= v332;
        bool v335;
        v335 = v334 == false;
        if (v335){
            assert("The index needs to be zero or positive." && v334);
        } else {
        }
        int v337;
        v337 = v332 % 1l;
        bool v338;
        v338 = v332 < 4l;
        bool v339;
        v339 = v338 == false;
        if (v339){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v338);
        } else {
        }
        assert("Tensor range check" && 0 <= v332 && v332 < 4l);
        assert("Tensor range check" && 0 <= v337 && v337 < 1l);
        int v341;
        v341 = 4l * v337;
        int v342;
        v342 = 4l * v332;
        int v343;
        v343 = v342 + v341;
        assert("Tensor range check" && 0 <= v332 && v332 < 4l);
        assert("Tensor range check" && 0 <= v337 && v337 < 1l);
        float v344[4l];
        float v345[4l];
        float v346[4l];
        int4* v347;
        v347 = reinterpret_cast<int4*>(v2 + v343);
        int4* v348;
        v348 = reinterpret_cast<int4*>(v344 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v347) % 4l == 0 && (unsigned long long)(v348) % 4l == 0);
        *v348 = *v347;
        int4* v349;
        v349 = reinterpret_cast<int4*>(v3 + v343);
        int4* v350;
        v350 = reinterpret_cast<int4*>(v345 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v349) % 4l == 0 && (unsigned long long)(v350) % 4l == 0);
        *v350 = *v349;
        // Pushing the loop unrolling to: 0
        int v351;
        v351 = 0l;
        #pragma unroll
        while (while_method_0(v351)){
            assert("Tensor range check" && 0 <= v351 && v351 < 4l);
            float v353;
            v353 = v344[v351];
            float v354;
            v354 = v345[v351];
            bool v355;
            v355 = v354 == 0.0f;
            bool v356;
            v356 = v355 != true;
            float v358;
            if (v356){
                float v357;
                v357 = v353 / v354;
                v358 = v357;
            } else {
                v358 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v351 && v351 < 4l);
            v346[v351] = v358;
            v351 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v359;
        v359 = reinterpret_cast<int4*>(v346 + 0l);
        int4* v360;
        v360 = reinterpret_cast<int4*>(v4 + v343);
        assert("Pointer alignment check" && (unsigned long long)(v359) % 4l == 0 && (unsigned long long)(v360) % 4l == 0);
        *v360 = *v359;
        v332 += 12288l ;
    }
    v5.sync() ;
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
options.append('--diag-suppress=550,20012,68,39')
options.append('--restrict')
options.append('--maxrregcount=128')
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
    v8 = cp.empty(3866640,dtype=cp.uint8)
    v9 = cp.empty(50528256,dtype=cp.uint8)
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
    v30 = v8[2097168:2097168+8*98304].view(cp.float64)
    v32 = v8[2883600:2883600+8*98304].view(cp.float64)
    v34 = v8[3670032:3670032+4*49152].view(cp.int32)
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
    v46 = cp.zeros(16,dtype=cp.float32) # type: ignore
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
    v53((24,),(512,),(v9, v8, v45, v46, v47),shared_mem=81920)
    del v8, v9, v45, v46, v53
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
