kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>
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
__device__ int int_range_4(int v0, int v1, curandStatePhilox4_32_10_t & v2);
__device__ Union8 noinline_eval_3(sptr<Mut0> v0);
struct Union9;
__device__ int tag_6(Union2 v0);
__device__ bool is_pair_7(int v0, int v1);
__device__ Tuple1 order_8(int v0, int v1);
__device__ Union9 compare_hands_5(Union5 v0, bool v1, static_array<Union2,2l> v2, int v3, static_array<int,2l> v4, int v5);
__device__ void method_0(sptr<Mut0> v0, Union6 v1);
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
struct Union8_0 { // AA_Call
};
struct Union8_1 { // AA_Fold
};
struct Union8_2 { // AA_Raise
};
struct Union8 {
    union {
        Union8_0 case0; // AA_Call
        Union8_1 case1; // AA_Fold
        Union8_2 case2; // AA_Raise
    };
    unsigned char tag{255};
    __device__ Union8() {}
    __device__ Union8(Union8_0 t) : tag(0), case0(t) {} // AA_Call
    __device__ Union8(Union8_1 t) : tag(1), case1(t) {} // AA_Fold
    __device__ Union8(Union8_2 t) : tag(2), case2(t) {} // AA_Raise
    __device__ Union8(Union8 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(x.case0); break; // AA_Call
            case 1: new (&this->case1) Union8_1(x.case1); break; // AA_Fold
            case 2: new (&this->case2) Union8_2(x.case2); break; // AA_Raise
        }
    }
    __device__ Union8(Union8 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(std::move(x.case0)); break; // AA_Call
            case 1: new (&this->case1) Union8_1(std::move(x.case1)); break; // AA_Fold
            case 2: new (&this->case2) Union8_2(std::move(x.case2)); break; // AA_Raise
        }
    }
    __device__ Union8 & operator=(Union8 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // AA_Call
                case 1: this->case1 = x.case1; break; // AA_Fold
                case 2: this->case2 = x.case2; break; // AA_Raise
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
                case 0: this->case0 = std::move(x.case0); break; // AA_Call
                case 1: this->case1 = std::move(x.case1); break; // AA_Fold
                case 2: this->case2 = std::move(x.case2); break; // AA_Raise
            }
        } else {
            this->~Union8();
            new (this) Union8{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union8() {
        switch(this->tag){
            case 0: this->case0.~Union8_0(); break; // AA_Call
            case 1: this->case1.~Union8_1(); break; // AA_Fold
            case 2: this->case2.~Union8_2(); break; // AA_Raise
        }
        this->tag = 255;
    }
};
struct Union9_0 { // Eq
};
struct Union9_1 { // Gt
};
struct Union9_2 { // Lt
};
struct Union9 {
    union {
        Union9_0 case0; // Eq
        Union9_1 case1; // Gt
        Union9_2 case2; // Lt
    };
    unsigned char tag{255};
    __device__ Union9() {}
    __device__ Union9(Union9_0 t) : tag(0), case0(t) {} // Eq
    __device__ Union9(Union9_1 t) : tag(1), case1(t) {} // Gt
    __device__ Union9(Union9_2 t) : tag(2), case2(t) {} // Lt
    __device__ Union9(Union9 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union9_0(x.case0); break; // Eq
            case 1: new (&this->case1) Union9_1(x.case1); break; // Gt
            case 2: new (&this->case2) Union9_2(x.case2); break; // Lt
        }
    }
    __device__ Union9(Union9 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union9_0(std::move(x.case0)); break; // Eq
            case 1: new (&this->case1) Union9_1(std::move(x.case1)); break; // Gt
            case 2: new (&this->case2) Union9_2(std::move(x.case2)); break; // Lt
        }
    }
    __device__ Union9 & operator=(Union9 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Eq
                case 1: this->case1 = x.case1; break; // Gt
                case 2: this->case2 = x.case2; break; // Lt
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
                case 0: this->case0 = std::move(x.case0); break; // Eq
                case 1: this->case1 = std::move(x.case1); break; // Gt
                case 2: this->case2 = std::move(x.case2); break; // Lt
            }
        } else {
            this->~Union9();
            new (this) Union9{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union9() {
        switch(this->tag){
            case 0: this->case0.~Union9_0(); break; // Eq
            case 1: this->case1.~Union9_1(); break; // Gt
            case 2: this->case2.~Union9_2(); break; // Lt
        }
        this->tag = 255;
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
__device__ int int_range_4(int v0, int v1, curandStatePhilox4_32_10_t & v2){
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
__device__ __noinline__ Union8 noinline_eval_3(sptr<Mut0> v0){
    static_array<Union8,3l> v1;
    Union8 v3;
    v3 = Union8{Union8_1{}};
    v1[0l] = v3;
    Union8 v5;
    v5 = Union8{Union8_0{}};
    v1[1l] = v5;
    Union8 v7;
    v7 = Union8{Union8_2{}};
    v1[2l] = v7;
    int v9;
    v9 = 0l;
    int v10;
    v10 = 3l;
    curandStatePhilox4_32_10_t & v11 = v0.base->v6;
    curandStatePhilox4_32_10_t & v12 = v11;
    int v13;
    v13 = int_range_4(v10, v9, v12);
    Union8 v14;
    v14 = v1[v13];
    int v16;
    v16 = sizeof(Union8);
    unsigned long long v17;
    v17 = (unsigned long long)v16;
    bool v18;
    v18 = v17 <= 81920ull;
    bool v19;
    v19 = v18 == false;
    if (v19){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v18);
    } else {
    }
    extern __shared__ unsigned char v21[];
    bool v22;
    v22 = v17 <= v17;
    bool v23;
    v23 = v22 == false;
    if (v23){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v22);
    } else {
    }
    Union8 * v25;
    v25 = reinterpret_cast<Union8 *>(&v21[0ull]);
    int v27;
    v27 = threadIdx.x;
    bool v28;
    v28 = v27 == 0l;
    if (v28){
        v25[0l] = v14;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    Union8 v29;
    v29 = v25[0l];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return v29;
}
__device__ inline bool while_method_4(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ int tag_6(Union2 v0){
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
__device__ bool is_pair_7(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
__device__ Tuple1 order_8(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple1{v1, v0};
    } else {
        return Tuple1{v0, v1};
    }
}
__device__ Union9 compare_hands_5(Union5 v0, bool v1, static_array<Union2,2l> v2, int v3, static_array<int,2l> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            __trap();
            break;
        }
        case 1: { // Some
            Union2 v7 = v0.case1.v0;
            int v8;
            v8 = tag_6(v7);
            Union2 v9;
            v9 = v2[0l];
            int v11;
            v11 = tag_6(v9);
            Union2 v12;
            v12 = v2[1l];
            int v14;
            v14 = tag_6(v12);
            bool v15;
            v15 = is_pair_7(v8, v11);
            bool v16;
            v16 = is_pair_7(v8, v14);
            if (v15){
                if (v16){
                    bool v17;
                    v17 = v11 < v14;
                    if (v17){
                        return Union9{Union9_2{}};
                    } else {
                        bool v19;
                        v19 = v11 > v14;
                        if (v19){
                            return Union9{Union9_1{}};
                        } else {
                            return Union9{Union9_0{}};
                        }
                    }
                } else {
                    return Union9{Union9_1{}};
                }
            } else {
                if (v16){
                    return Union9{Union9_2{}};
                } else {
                    int v27; int v28;
                    Tuple1 tmp11 = order_8(v8, v11);
                    v27 = tmp11.v0; v28 = tmp11.v1;
                    int v29; int v30;
                    Tuple1 tmp12 = order_8(v8, v14);
                    v29 = tmp12.v0; v30 = tmp12.v1;
                    bool v31;
                    v31 = v27 < v29;
                    Union9 v37;
                    if (v31){
                        v37 = Union9{Union9_2{}};
                    } else {
                        bool v33;
                        v33 = v27 > v29;
                        if (v33){
                            v37 = Union9{Union9_1{}};
                        } else {
                            v37 = Union9{Union9_0{}};
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
                            return Union9{Union9_2{}};
                        } else {
                            bool v41;
                            v41 = v28 > v30;
                            if (v41){
                                return Union9{Union9_1{}};
                            } else {
                                return Union9{Union9_0{}};
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
__device__ void method_0(sptr<Mut0> v0, Union6 v1){
    static_array_list<Union1,32l> & v2 = v0.base->v3;
    Union7 v3;
    v3 = Union7{Union7_1{v1}};
    Union7 v4;
    v4 = v3;
    while (while_method_2(v4)){
        Union7 v363;
        switch (v4.tag) {
            case 0: { // None
                v363 = Union7{Union7_0{}};
                break;
            }
            case 1: { // Some
                Union6 v6 = v4.case1.v0;
                switch (v6.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v308 = v6.case0.v0; bool v309 = v6.case0.v1; static_array<Union2,2l> v310 = v6.case0.v2; int v311 = v6.case0.v3; static_array<int,2l> v312 = v6.case0.v4; int v313 = v6.case0.v5;
                        curandStatePhilox4_32_10_t & v314 = v0.base->v6;
                        curandStatePhilox4_32_10_t & v315 = v314;
                        unsigned int & v316 = v0.base->v1;
                        Union2 v317; unsigned int v318;
                        Tuple0 tmp0 = draw_card_1(v315, v316);
                        v317 = tmp0.v0; v318 = tmp0.v1;
                        v0.base->v1 = v318;
                        Union1 v319;
                        v319 = Union1{Union1_0{v317}};
                        v2.push(v319);
                        int v320;
                        v320 = 2l;
                        int v321; int v322;
                        Tuple1 tmp1 = Tuple1{0l, 0l};
                        v321 = tmp1.v0; v322 = tmp1.v1;
                        while (while_method_3(v321)){
                            int v324;
                            v324 = v312[v321];
                            bool v326;
                            v326 = v322 >= v324;
                            int v327;
                            if (v326){
                                v327 = v322;
                            } else {
                                v327 = v324;
                            }
                            v322 = v327;
                            v321 += 1l ;
                        }
                        static_array<int,2l> v328;
                        int v330;
                        v330 = 0l;
                        while (while_method_3(v330)){
                            v328[v330] = v322;
                            v330 += 1l ;
                        }
                        Union5 v332;
                        v332 = Union5{Union5_1{v317}};
                        Union6 v333;
                        v333 = Union6{Union6_2{v332, true, v310, 0l, v328, v320}};
                        v363 = Union7{Union7_1{v333}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t & v335 = v0.base->v6;
                        curandStatePhilox4_32_10_t & v336 = v335;
                        unsigned int & v337 = v0.base->v1;
                        Union2 v338; unsigned int v339;
                        Tuple0 tmp2 = draw_card_1(v336, v337);
                        v338 = tmp2.v0; v339 = tmp2.v1;
                        v0.base->v1 = v339;
                        curandStatePhilox4_32_10_t & v340 = v0.base->v6;
                        curandStatePhilox4_32_10_t & v341 = v340;
                        unsigned int & v342 = v0.base->v1;
                        Union2 v343; unsigned int v344;
                        Tuple0 tmp3 = draw_card_1(v341, v342);
                        v343 = tmp3.v0; v344 = tmp3.v1;
                        v0.base->v1 = v344;
                        Union1 v345;
                        v345 = Union1{Union1_2{0l, v338}};
                        v2.push(v345);
                        Union1 v346;
                        v346 = Union1{Union1_2{1l, v343}};
                        v2.push(v346);
                        int v347;
                        v347 = 2l;
                        static_array<int,2l> v348;
                        v348[0l] = 1l;
                        v348[1l] = 1l;
                        static_array<Union2,2l> v350;
                        v350[0l] = v338;
                        v350[1l] = v343;
                        Union5 v352;
                        v352 = Union5{Union5_0{}};
                        Union6 v353;
                        v353 = Union6{Union6_2{v352, true, v350, 0l, v348, v347}};
                        v363 = Union7{Union7_1{v353}};
                        break;
                    }
                    case 2: { // Round
                        Union5 v49 = v6.case2.v0; bool v50 = v6.case2.v1; static_array<Union2,2l> v51 = v6.case2.v2; int v52 = v6.case2.v3; static_array<int,2l> v53 = v6.case2.v4; int v54 = v6.case2.v5;
                        static_array<Union0,2l> & v55 = v0.base->v4;
                        Union0 v56;
                        v56 = v55[v52];
                        Union3 v124;
                        switch (v56.tag) {
                            case 0: { // T_Computer
                                static_array_list<Union1,32l> & v58 = v0.base->v3;
                                Union8 v59;
                                v59 = noinline_eval_3(v0);
                                switch (v59.tag) {
                                    case 0: { // AA_Call
                                        v124 = Union3{Union3_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v60;
                                        v60 = v53[0l];
                                        int v62; int v63;
                                        Tuple1 tmp4 = Tuple1{1l, v60};
                                        v62 = tmp4.v0; v63 = tmp4.v1;
                                        while (while_method_3(v62)){
                                            int v65;
                                            v65 = v53[v62];
                                            bool v67;
                                            v67 = v63 >= v65;
                                            int v68;
                                            if (v67){
                                                v68 = v63;
                                            } else {
                                                v68 = v65;
                                            }
                                            v63 = v68;
                                            v62 += 1l ;
                                        }
                                        int v69;
                                        v69 = v53[v52];
                                        bool v71;
                                        v71 = v69 == v63;
                                        if (v71){
                                            v124 = Union3{Union3_0{}};
                                        } else {
                                            v124 = Union3{Union3_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v76;
                                        v76 = v54 > 0l;
                                        if (v76){
                                            v124 = Union3{Union3_2{}};
                                        } else {
                                            v124 = Union3{Union3_0{}};
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
                                curandStatePhilox4_32_10_t & v83 = v0.base->v6;
                                curandStatePhilox4_32_10_t & v84 = v83;
                                static_array_list<Union3,3l> v85;
                                v85 = static_array_list<Union3,3l>{};
                                v85.unsafe_set_length(1l);
                                Union3 v87;
                                v87 = Union3{Union3_0{}};
                                v85[0l] = v87;
                                int v89;
                                v89 = v53[0l];
                                int v91;
                                v91 = v53[1l];
                                bool v93;
                                v93 = v89 == v91;
                                bool v94;
                                v94 = v93 != true;
                                if (v94){
                                    Union3 v95;
                                    v95 = Union3{Union3_1{}};
                                    v85.push(v95);
                                } else {
                                }
                                bool v96;
                                v96 = v54 > 0l;
                                if (v96){
                                    Union3 v97;
                                    v97 = Union3{Union3_2{}};
                                    v85.push(v97);
                                } else {
                                }
                                int v98;
                                v98 = v85.length;
                                int v99;
                                v99 = v98 - 1l;
                                int v100;
                                v100 = 0l;
                                while (while_method_4(v99, v100)){
                                    int v102;
                                    v102 = v85.length;
                                    int v103;
                                    v103 = int_range_4(v102, v100, v84);
                                    Union3 v104;
                                    v104 = v85[v100];
                                    Union3 v106;
                                    v106 = v85[v103];
                                    v85[v100] = v106;
                                    v85[v103] = v104;
                                    v100 += 1l ;
                                }
                                Union3 v108;
                                v108 = v85.pop();
                                int v109;
                                v109 = sizeof(Union3);
                                unsigned long long v110;
                                v110 = (unsigned long long)v109;
                                bool v111;
                                v111 = v110 <= 81920ull;
                                bool v112;
                                v112 = v111 == false;
                                if (v112){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v111);
                                } else {
                                }
                                extern __shared__ unsigned char v114[];
                                bool v115;
                                v115 = v110 <= v110;
                                bool v116;
                                v116 = v115 == false;
                                if (v116){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v115);
                                } else {
                                }
                                Union3 * v118;
                                v118 = reinterpret_cast<Union3 *>(&v114[0ull]);
                                int v120;
                                v120 = threadIdx.x;
                                bool v121;
                                v121 = v120 == 0l;
                                if (v121){
                                    v118[0l] = v108;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                Union3 v122;
                                v122 = v118[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                v124 = v122;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union1 v125;
                        v125 = Union1{Union1_1{v52, v124}};
                        v2.push(v125);
                        Union6 v211;
                        switch (v49.tag) {
                            case 0: { // None
                                switch (v124.tag) {
                                    case 0: { // Call
                                        if (v50){
                                            bool v175;
                                            v175 = v52 == 0l;
                                            int v176;
                                            if (v175){
                                                v176 = 1l;
                                            } else {
                                                v176 = 0l;
                                            }
                                            v211 = Union6{Union6_2{v49, false, v51, v176, v53, v54}};
                                        } else {
                                            v211 = Union6{Union6_0{v49, v50, v51, v52, v53, v54}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v211 = Union6{Union6_5{v49, v50, v51, v52, v53, v54}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v180;
                                        v180 = v54 > 0l;
                                        if (v180){
                                            bool v181;
                                            v181 = v52 == 0l;
                                            int v182;
                                            if (v181){
                                                v182 = 1l;
                                            } else {
                                                v182 = 0l;
                                            }
                                            int v183;
                                            v183 = -1l + v54;
                                            int v184; int v185;
                                            Tuple1 tmp5 = Tuple1{0l, 0l};
                                            v184 = tmp5.v0; v185 = tmp5.v1;
                                            while (while_method_3(v184)){
                                                int v187;
                                                v187 = v53[v184];
                                                bool v189;
                                                v189 = v185 >= v187;
                                                int v190;
                                                if (v189){
                                                    v190 = v185;
                                                } else {
                                                    v190 = v187;
                                                }
                                                v185 = v190;
                                                v184 += 1l ;
                                            }
                                            static_array<int,2l> v191;
                                            int v193;
                                            v193 = 0l;
                                            while (while_method_3(v193)){
                                                v191[v193] = v185;
                                                v193 += 1l ;
                                            }
                                            static_array<int,2l> v195;
                                            int v197;
                                            v197 = 0l;
                                            while (while_method_3(v197)){
                                                int v199;
                                                v199 = v191[v197];
                                                bool v201;
                                                v201 = v197 == v52;
                                                int v203;
                                                if (v201){
                                                    int v202;
                                                    v202 = v199 + 2l;
                                                    v203 = v202;
                                                } else {
                                                    v203 = v199;
                                                }
                                                v195[v197] = v203;
                                                v197 += 1l ;
                                            }
                                            v211 = Union6{Union6_2{v49, false, v51, v182, v195, v183}};
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
                                Union2 v126 = v49.case1.v0;
                                switch (v124.tag) {
                                    case 0: { // Call
                                        if (v50){
                                            bool v128;
                                            v128 = v52 == 0l;
                                            int v129;
                                            if (v128){
                                                v129 = 1l;
                                            } else {
                                                v129 = 0l;
                                            }
                                            v211 = Union6{Union6_2{v49, false, v51, v129, v53, v54}};
                                        } else {
                                            int v131; int v132;
                                            Tuple1 tmp6 = Tuple1{0l, 0l};
                                            v131 = tmp6.v0; v132 = tmp6.v1;
                                            while (while_method_3(v131)){
                                                int v134;
                                                v134 = v53[v131];
                                                bool v136;
                                                v136 = v132 >= v134;
                                                int v137;
                                                if (v136){
                                                    v137 = v132;
                                                } else {
                                                    v137 = v134;
                                                }
                                                v132 = v137;
                                                v131 += 1l ;
                                            }
                                            static_array<int,2l> v138;
                                            int v140;
                                            v140 = 0l;
                                            while (while_method_3(v140)){
                                                v138[v140] = v132;
                                                v140 += 1l ;
                                            }
                                            v211 = Union6{Union6_4{v49, v50, v51, v52, v138, v54}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v211 = Union6{Union6_5{v49, v50, v51, v52, v53, v54}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v144;
                                        v144 = v54 > 0l;
                                        if (v144){
                                            bool v145;
                                            v145 = v52 == 0l;
                                            int v146;
                                            if (v145){
                                                v146 = 1l;
                                            } else {
                                                v146 = 0l;
                                            }
                                            int v147;
                                            v147 = -1l + v54;
                                            int v148; int v149;
                                            Tuple1 tmp7 = Tuple1{0l, 0l};
                                            v148 = tmp7.v0; v149 = tmp7.v1;
                                            while (while_method_3(v148)){
                                                int v151;
                                                v151 = v53[v148];
                                                bool v153;
                                                v153 = v149 >= v151;
                                                int v154;
                                                if (v153){
                                                    v154 = v149;
                                                } else {
                                                    v154 = v151;
                                                }
                                                v149 = v154;
                                                v148 += 1l ;
                                            }
                                            static_array<int,2l> v155;
                                            int v157;
                                            v157 = 0l;
                                            while (while_method_3(v157)){
                                                v155[v157] = v149;
                                                v157 += 1l ;
                                            }
                                            static_array<int,2l> v159;
                                            int v161;
                                            v161 = 0l;
                                            while (while_method_3(v161)){
                                                int v163;
                                                v163 = v155[v161];
                                                bool v165;
                                                v165 = v161 == v52;
                                                int v167;
                                                if (v165){
                                                    int v166;
                                                    v166 = v163 + 4l;
                                                    v167 = v166;
                                                } else {
                                                    v167 = v163;
                                                }
                                                v159[v161] = v167;
                                                v161 += 1l ;
                                            }
                                            v211 = Union6{Union6_2{v49, false, v51, v146, v159, v147}};
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
                        v363 = Union7{Union7_1{v211}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v213 = v6.case3.v0; bool v214 = v6.case3.v1; static_array<Union2,2l> v215 = v6.case3.v2; int v216 = v6.case3.v3; static_array<int,2l> v217 = v6.case3.v4; int v218 = v6.case3.v5; Union3 v219 = v6.case3.v6;
                        Union1 v220;
                        v220 = Union1{Union1_1{v216, v219}};
                        v2.push(v220);
                        Union6 v306;
                        switch (v213.tag) {
                            case 0: { // None
                                switch (v219.tag) {
                                    case 0: { // Call
                                        if (v214){
                                            bool v270;
                                            v270 = v216 == 0l;
                                            int v271;
                                            if (v270){
                                                v271 = 1l;
                                            } else {
                                                v271 = 0l;
                                            }
                                            v306 = Union6{Union6_2{v213, false, v215, v271, v217, v218}};
                                        } else {
                                            v306 = Union6{Union6_0{v213, v214, v215, v216, v217, v218}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v306 = Union6{Union6_5{v213, v214, v215, v216, v217, v218}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v275;
                                        v275 = v218 > 0l;
                                        if (v275){
                                            bool v276;
                                            v276 = v216 == 0l;
                                            int v277;
                                            if (v276){
                                                v277 = 1l;
                                            } else {
                                                v277 = 0l;
                                            }
                                            int v278;
                                            v278 = -1l + v218;
                                            int v279; int v280;
                                            Tuple1 tmp8 = Tuple1{0l, 0l};
                                            v279 = tmp8.v0; v280 = tmp8.v1;
                                            while (while_method_3(v279)){
                                                int v282;
                                                v282 = v217[v279];
                                                bool v284;
                                                v284 = v280 >= v282;
                                                int v285;
                                                if (v284){
                                                    v285 = v280;
                                                } else {
                                                    v285 = v282;
                                                }
                                                v280 = v285;
                                                v279 += 1l ;
                                            }
                                            static_array<int,2l> v286;
                                            int v288;
                                            v288 = 0l;
                                            while (while_method_3(v288)){
                                                v286[v288] = v280;
                                                v288 += 1l ;
                                            }
                                            static_array<int,2l> v290;
                                            int v292;
                                            v292 = 0l;
                                            while (while_method_3(v292)){
                                                int v294;
                                                v294 = v286[v292];
                                                bool v296;
                                                v296 = v292 == v216;
                                                int v298;
                                                if (v296){
                                                    int v297;
                                                    v297 = v294 + 2l;
                                                    v298 = v297;
                                                } else {
                                                    v298 = v294;
                                                }
                                                v290[v292] = v298;
                                                v292 += 1l ;
                                            }
                                            v306 = Union6{Union6_2{v213, false, v215, v277, v290, v278}};
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
                                Union2 v221 = v213.case1.v0;
                                switch (v219.tag) {
                                    case 0: { // Call
                                        if (v214){
                                            bool v223;
                                            v223 = v216 == 0l;
                                            int v224;
                                            if (v223){
                                                v224 = 1l;
                                            } else {
                                                v224 = 0l;
                                            }
                                            v306 = Union6{Union6_2{v213, false, v215, v224, v217, v218}};
                                        } else {
                                            int v226; int v227;
                                            Tuple1 tmp9 = Tuple1{0l, 0l};
                                            v226 = tmp9.v0; v227 = tmp9.v1;
                                            while (while_method_3(v226)){
                                                int v229;
                                                v229 = v217[v226];
                                                bool v231;
                                                v231 = v227 >= v229;
                                                int v232;
                                                if (v231){
                                                    v232 = v227;
                                                } else {
                                                    v232 = v229;
                                                }
                                                v227 = v232;
                                                v226 += 1l ;
                                            }
                                            static_array<int,2l> v233;
                                            int v235;
                                            v235 = 0l;
                                            while (while_method_3(v235)){
                                                v233[v235] = v227;
                                                v235 += 1l ;
                                            }
                                            v306 = Union6{Union6_4{v213, v214, v215, v216, v233, v218}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v306 = Union6{Union6_5{v213, v214, v215, v216, v217, v218}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v239;
                                        v239 = v218 > 0l;
                                        if (v239){
                                            bool v240;
                                            v240 = v216 == 0l;
                                            int v241;
                                            if (v240){
                                                v241 = 1l;
                                            } else {
                                                v241 = 0l;
                                            }
                                            int v242;
                                            v242 = -1l + v218;
                                            int v243; int v244;
                                            Tuple1 tmp10 = Tuple1{0l, 0l};
                                            v243 = tmp10.v0; v244 = tmp10.v1;
                                            while (while_method_3(v243)){
                                                int v246;
                                                v246 = v217[v243];
                                                bool v248;
                                                v248 = v244 >= v246;
                                                int v249;
                                                if (v248){
                                                    v249 = v244;
                                                } else {
                                                    v249 = v246;
                                                }
                                                v244 = v249;
                                                v243 += 1l ;
                                            }
                                            static_array<int,2l> v250;
                                            int v252;
                                            v252 = 0l;
                                            while (while_method_3(v252)){
                                                v250[v252] = v244;
                                                v252 += 1l ;
                                            }
                                            static_array<int,2l> v254;
                                            int v256;
                                            v256 = 0l;
                                            while (while_method_3(v256)){
                                                int v258;
                                                v258 = v250[v256];
                                                bool v260;
                                                v260 = v256 == v216;
                                                int v262;
                                                if (v260){
                                                    int v261;
                                                    v261 = v258 + 4l;
                                                    v262 = v261;
                                                } else {
                                                    v262 = v258;
                                                }
                                                v254[v256] = v262;
                                                v256 += 1l ;
                                            }
                                            v306 = Union6{Union6_2{v213, false, v215, v241, v254, v242}};
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
                        v363 = Union7{Union7_1{v306}};
                        break;
                    }
                    case 4: { // TerminalCall
                        Union5 v24 = v6.case4.v0; bool v25 = v6.case4.v1; static_array<Union2,2l> v26 = v6.case4.v2; int v27 = v6.case4.v3; static_array<int,2l> v28 = v6.case4.v4; int v29 = v6.case4.v5;
                        int v30;
                        v30 = v28[v27];
                        Union9 v32;
                        v32 = compare_hands_5(v24, v25, v26, v27, v28, v29);
                        int v37; int v38;
                        switch (v32.tag) {
                            case 0: { // Eq
                                v37 = 0l; v38 = -1l;
                                break;
                            }
                            case 1: { // Gt
                                v37 = v30; v38 = 0l;
                                break;
                            }
                            case 2: { // Lt
                                v37 = v30; v38 = 1l;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        int v39;
                        v39 = -v38;
                        bool v40;
                        v40 = v38 >= v39;
                        int v41;
                        if (v40){
                            v41 = v38;
                        } else {
                            v41 = v39;
                        }
                        float v42;
                        v42 = (float)v37;
                        static_array<float,2l> & v43 = v0.base->v5;
                        v43[v41] = v42;
                        bool v44;
                        v44 = v41 == 0l;
                        int v45;
                        if (v44){
                            v45 = 1l;
                        } else {
                            v45 = 0l;
                        }
                        float v46;
                        v46 = -v42;
                        v43[v45] = v46;
                        Union1 v47;
                        v47 = Union1{Union1_3{v26, v37, v38}};
                        v2.push(v47);
                        v363 = Union7{Union7_0{}};
                        break;
                    }
                    case 5: { // TerminalFold
                        Union5 v7 = v6.case5.v0; bool v8 = v6.case5.v1; static_array<Union2,2l> v9 = v6.case5.v2; int v10 = v6.case5.v3; static_array<int,2l> v11 = v6.case5.v4; int v12 = v6.case5.v5;
                        int v13;
                        v13 = v11[v10];
                        int v15;
                        v15 = -v13;
                        float v16;
                        v16 = (float)v15;
                        static_array<float,2l> & v17 = v0.base->v5;
                        v17[v10] = v16;
                        bool v18;
                        v18 = v10 == 0l;
                        int v19;
                        if (v18){
                            v19 = 1l;
                        } else {
                            v19 = 0l;
                        }
                        float v20;
                        v20 = -v16;
                        v17[v19] = v20;
                        int v21;
                        if (v18){
                            v21 = 1l;
                        } else {
                            v21 = 0l;
                        }
                        Union1 v22;
                        v22 = Union1{Union1_3{v9, v13, v21}};
                        v2.push(v22);
                        v363 = Union7{Union7_0{}};
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
        v4 = v363;
    }
    return ;
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
            method_0(v23, v39);
            static_array<float,2l> & v40 = v23.base->v5;
            static_array<float,2l> v41;
            v41 = v40;
            v26 += 1l ;
        }
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
