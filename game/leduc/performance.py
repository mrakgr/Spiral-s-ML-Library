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
struct Mut0;
struct Union5;
struct Union4;
struct Union6;
struct Tuple0;
__device__ unsigned int loop_2(unsigned int v0, curandStatePhilox4_32_10_t & v1);
__device__ Tuple0 draw_card_1(curandStatePhilox4_32_10_t & v0, unsigned int v1);
struct Tuple1;
__device__ int int_range_3(int v0, int v1, curandStatePhilox4_32_10_t & v2);
struct Union7;
__device__ int tag_5(Union2 v0);
__device__ bool is_pair_6(int v0, int v1);
__device__ Tuple1 order_7(int v0, int v1);
__device__ Union7 compare_hands_4(Union5 v0, bool v1, static_array<Union2,2l> v2, int v3, static_array<int,2l> v4, int v5);
__device__ void method_0(sptr<Mut0> v0, Union4 v1);
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
struct Mut0 {
    int refc{0};
    cooperative_groups::grid_group v1;
    static_array_list<Union1,32l> v2;
    static_array<Union0,2l> v3;
    static_array<float,2l> v4;
    curandStatePhilox4_32_10_t v5;
    unsigned int v0;
    __device__ Mut0() = default;
    __device__ Mut0(unsigned int t0, cooperative_groups::grid_group t1, static_array_list<Union1,32l> t2, static_array<Union0,2l> t3, static_array<float,2l> t4, curandStatePhilox4_32_10_t t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
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
struct Union7_0 { // Eq
};
struct Union7_1 { // Gt
};
struct Union7_2 { // Lt
};
struct Union7 {
    union {
        Union7_0 case0; // Eq
        Union7_1 case1; // Gt
        Union7_2 case2; // Lt
    };
    unsigned char tag{255};
    __device__ Union7() {}
    __device__ Union7(Union7_0 t) : tag(0), case0(t) {} // Eq
    __device__ Union7(Union7_1 t) : tag(1), case1(t) {} // Gt
    __device__ Union7(Union7_2 t) : tag(2), case2(t) {} // Lt
    __device__ Union7(Union7 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(x.case0); break; // Eq
            case 1: new (&this->case1) Union7_1(x.case1); break; // Gt
            case 2: new (&this->case2) Union7_2(x.case2); break; // Lt
        }
    }
    __device__ Union7(Union7 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(std::move(x.case0)); break; // Eq
            case 1: new (&this->case1) Union7_1(std::move(x.case1)); break; // Gt
            case 2: new (&this->case2) Union7_2(std::move(x.case2)); break; // Lt
        }
    }
    __device__ Union7 & operator=(Union7 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Eq
                case 1: this->case1 = x.case1; break; // Gt
                case 2: this->case2 = x.case2; break; // Lt
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
                case 0: this->case0 = std::move(x.case0); break; // Eq
                case 1: this->case1 = std::move(x.case1); break; // Gt
                case 2: this->case2 = std::move(x.case2); break; // Lt
            }
        } else {
            this->~Union7();
            new (this) Union7{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union7() {
        switch(this->tag){
            case 0: this->case0.~Union7_0(); break; // Eq
            case 1: this->case1.~Union7_1(); break; // Gt
            case 2: this->case2.~Union7_2(); break; // Lt
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
__device__ inline bool while_method_4(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ int int_range_3(int v0, int v1, curandStatePhilox4_32_10_t & v2){
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
__device__ int tag_5(Union2 v0){
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
__device__ bool is_pair_6(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
__device__ Tuple1 order_7(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple1{v1, v0};
    } else {
        return Tuple1{v0, v1};
    }
}
__device__ Union7 compare_hands_4(Union5 v0, bool v1, static_array<Union2,2l> v2, int v3, static_array<int,2l> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            __trap();
            break;
        }
        case 1: { // Some
            Union2 v7 = v0.case1.v0;
            int v8;
            v8 = tag_5(v7);
            Union2 v9;
            v9 = v2[0l];
            int v11;
            v11 = tag_5(v9);
            Union2 v12;
            v12 = v2[1l];
            int v14;
            v14 = tag_5(v12);
            bool v15;
            v15 = is_pair_6(v8, v11);
            bool v16;
            v16 = is_pair_6(v8, v14);
            if (v15){
                if (v16){
                    bool v17;
                    v17 = v11 < v14;
                    if (v17){
                        return Union7{Union7_2{}};
                    } else {
                        bool v19;
                        v19 = v11 > v14;
                        if (v19){
                            return Union7{Union7_1{}};
                        } else {
                            return Union7{Union7_0{}};
                        }
                    }
                } else {
                    return Union7{Union7_1{}};
                }
            } else {
                if (v16){
                    return Union7{Union7_2{}};
                } else {
                    int v27; int v28;
                    Tuple1 tmp10 = order_7(v8, v11);
                    v27 = tmp10.v0; v28 = tmp10.v1;
                    int v29; int v30;
                    Tuple1 tmp11 = order_7(v8, v14);
                    v29 = tmp11.v0; v30 = tmp11.v1;
                    bool v31;
                    v31 = v27 < v29;
                    Union7 v37;
                    if (v31){
                        v37 = Union7{Union7_2{}};
                    } else {
                        bool v33;
                        v33 = v27 > v29;
                        if (v33){
                            v37 = Union7{Union7_1{}};
                        } else {
                            v37 = Union7{Union7_0{}};
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
                            return Union7{Union7_2{}};
                        } else {
                            bool v41;
                            v41 = v28 > v30;
                            if (v41){
                                return Union7{Union7_1{}};
                            } else {
                                return Union7{Union7_0{}};
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
__device__ void method_0(sptr<Mut0> v0, Union4 v1){
    static_array_list<Union1,32l> v2;
    v2 = v0.base->v2;
    Union6 v3;
    v3 = Union6{Union6_1{v1}};
    Union6 v4;
    v4 = v3;
    while (while_method_2(v4)){
        Union6 v378;
        switch (v4.tag) {
            case 0: { // None
                v378 = Union6{Union6_0{}};
                break;
            }
            case 1: { // Some
                Union4 v6 = v4.case1.v0;
                switch (v6.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v323 = v6.case0.v0; bool v324 = v6.case0.v1; static_array<Union2,2l> v325 = v6.case0.v2; int v326 = v6.case0.v3; static_array<int,2l> v327 = v6.case0.v4; int v328 = v6.case0.v5;
                        curandStatePhilox4_32_10_t v329;
                        v329 = v0.base->v5;
                        curandStatePhilox4_32_10_t & v330 = v329;
                        unsigned int v331;
                        v331 = v0.base->v0;
                        Union2 v332; unsigned int v333;
                        Tuple0 tmp0 = draw_card_1(v330, v331);
                        v332 = tmp0.v0; v333 = tmp0.v1;
                        v0.base->v0 = v333;
                        Union1 v334;
                        v334 = Union1{Union1_0{v332}};
                        v2.push(v334);
                        int v335;
                        v335 = 2l;
                        int v336; int v337;
                        Tuple1 tmp1 = Tuple1{0l, 0l};
                        v336 = tmp1.v0; v337 = tmp1.v1;
                        while (while_method_3(v336)){
                            int v339;
                            v339 = v327[v336];
                            bool v341;
                            v341 = v337 >= v339;
                            int v342;
                            if (v341){
                                v342 = v337;
                            } else {
                                v342 = v339;
                            }
                            v337 = v342;
                            v336 += 1l ;
                        }
                        static_array<int,2l> v343;
                        int v345;
                        v345 = 0l;
                        while (while_method_3(v345)){
                            v343[v345] = v337;
                            v345 += 1l ;
                        }
                        Union5 v347;
                        v347 = Union5{Union5_1{v332}};
                        Union4 v348;
                        v348 = Union4{Union4_2{v347, true, v325, 0l, v343, v335}};
                        v378 = Union6{Union6_1{v348}};
                        break;
                    }
                    case 1: { // ChanceInit
                        curandStatePhilox4_32_10_t v350;
                        v350 = v0.base->v5;
                        curandStatePhilox4_32_10_t & v351 = v350;
                        unsigned int v352;
                        v352 = v0.base->v0;
                        Union2 v353; unsigned int v354;
                        Tuple0 tmp2 = draw_card_1(v351, v352);
                        v353 = tmp2.v0; v354 = tmp2.v1;
                        v0.base->v0 = v354;
                        curandStatePhilox4_32_10_t v355;
                        v355 = v0.base->v5;
                        curandStatePhilox4_32_10_t & v356 = v355;
                        unsigned int v357;
                        v357 = v0.base->v0;
                        Union2 v358; unsigned int v359;
                        Tuple0 tmp3 = draw_card_1(v356, v357);
                        v358 = tmp3.v0; v359 = tmp3.v1;
                        v0.base->v0 = v359;
                        Union1 v360;
                        v360 = Union1{Union1_2{0l, v353}};
                        v2.push(v360);
                        Union1 v361;
                        v361 = Union1{Union1_2{1l, v358}};
                        v2.push(v361);
                        int v362;
                        v362 = 2l;
                        static_array<int,2l> v363;
                        v363[0l] = 1l;
                        v363[1l] = 1l;
                        static_array<Union2,2l> v365;
                        v365[0l] = v353;
                        v365[1l] = v358;
                        Union5 v367;
                        v367 = Union5{Union5_0{}};
                        Union4 v368;
                        v368 = Union4{Union4_2{v367, true, v365, 0l, v363, v362}};
                        v378 = Union6{Union6_1{v368}};
                        break;
                    }
                    case 2: { // Round
                        Union5 v49 = v6.case2.v0; bool v50 = v6.case2.v1; static_array<Union2,2l> v51 = v6.case2.v2; int v52 = v6.case2.v3; static_array<int,2l> v53 = v6.case2.v4; int v54 = v6.case2.v5;
                        static_array<Union0,2l> v55;
                        v55 = v0.base->v3;
                        Union0 v56;
                        v56 = v55[v52];
                        Union3 v139;
                        switch (v56.tag) {
                            case 0: { // T_Computer
                                curandStatePhilox4_32_10_t v58;
                                v58 = v0.base->v5;
                                curandStatePhilox4_32_10_t & v59 = v58;
                                static_array_list<Union3,3l> v60;
                                v60 = static_array_list<Union3,3l>{};
                                v60.unsafe_set_length(1l);
                                Union3 v62;
                                v62 = Union3{Union3_0{}};
                                v60[0l] = v62;
                                int v64;
                                v64 = v53[0l];
                                int v66;
                                v66 = v53[1l];
                                bool v68;
                                v68 = v64 == v66;
                                bool v69;
                                v69 = v68 != true;
                                if (v69){
                                    Union3 v70;
                                    v70 = Union3{Union3_1{}};
                                    v60.push(v70);
                                } else {
                                }
                                bool v71;
                                v71 = v54 > 0l;
                                if (v71){
                                    Union3 v72;
                                    v72 = Union3{Union3_2{}};
                                    v60.push(v72);
                                } else {
                                }
                                int v73;
                                v73 = v60.length;
                                int v74;
                                v74 = v73 - 1l;
                                int v75;
                                v75 = 0l;
                                while (while_method_4(v74, v75)){
                                    int v77;
                                    v77 = v60.length;
                                    int v78;
                                    v78 = int_range_3(v77, v75, v59);
                                    Union3 v79;
                                    v79 = v60[v75];
                                    Union3 v81;
                                    v81 = v60[v78];
                                    v60[v75] = v81;
                                    v60[v78] = v79;
                                    v75 += 1l ;
                                }
                                Union3 v83;
                                v83 = v60.pop();
                                int v84;
                                v84 = sizeof(Union3);
                                unsigned long long v85;
                                v85 = (unsigned long long)v84;
                                bool v86;
                                v86 = v85 <= 81920ull;
                                bool v87;
                                v87 = v86 == false;
                                if (v87){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v86);
                                } else {
                                }
                                extern __shared__ unsigned char v89[];
                                bool v90;
                                v90 = v85 <= v85;
                                bool v91;
                                v91 = v90 == false;
                                if (v91){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v90);
                                } else {
                                }
                                Union3 * v93;
                                v93 = reinterpret_cast<Union3 *>(&v89[0ull]);
                                int v95;
                                v95 = threadIdx.x;
                                bool v96;
                                v96 = v95 == 0l;
                                if (v96){
                                    v93[0l] = v83;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                Union3 v97;
                                v97 = v93[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                v139 = v97;
                                break;
                            }
                            case 1: { // T_Random
                                curandStatePhilox4_32_10_t v98;
                                v98 = v0.base->v5;
                                curandStatePhilox4_32_10_t & v99 = v98;
                                static_array_list<Union3,3l> v100;
                                v100 = static_array_list<Union3,3l>{};
                                v100.unsafe_set_length(1l);
                                Union3 v102;
                                v102 = Union3{Union3_0{}};
                                v100[0l] = v102;
                                int v104;
                                v104 = v53[0l];
                                int v106;
                                v106 = v53[1l];
                                bool v108;
                                v108 = v104 == v106;
                                bool v109;
                                v109 = v108 != true;
                                if (v109){
                                    Union3 v110;
                                    v110 = Union3{Union3_1{}};
                                    v100.push(v110);
                                } else {
                                }
                                bool v111;
                                v111 = v54 > 0l;
                                if (v111){
                                    Union3 v112;
                                    v112 = Union3{Union3_2{}};
                                    v100.push(v112);
                                } else {
                                }
                                int v113;
                                v113 = v100.length;
                                int v114;
                                v114 = v113 - 1l;
                                int v115;
                                v115 = 0l;
                                while (while_method_4(v114, v115)){
                                    int v117;
                                    v117 = v100.length;
                                    int v118;
                                    v118 = int_range_3(v117, v115, v99);
                                    Union3 v119;
                                    v119 = v100[v115];
                                    Union3 v121;
                                    v121 = v100[v118];
                                    v100[v115] = v121;
                                    v100[v118] = v119;
                                    v115 += 1l ;
                                }
                                Union3 v123;
                                v123 = v100.pop();
                                int v124;
                                v124 = sizeof(Union3);
                                unsigned long long v125;
                                v125 = (unsigned long long)v124;
                                bool v126;
                                v126 = v125 <= 81920ull;
                                bool v127;
                                v127 = v126 == false;
                                if (v127){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v126);
                                } else {
                                }
                                extern __shared__ unsigned char v129[];
                                bool v130;
                                v130 = v125 <= v125;
                                bool v131;
                                v131 = v130 == false;
                                if (v131){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v130);
                                } else {
                                }
                                Union3 * v133;
                                v133 = reinterpret_cast<Union3 *>(&v129[0ull]);
                                int v135;
                                v135 = threadIdx.x;
                                bool v136;
                                v136 = v135 == 0l;
                                if (v136){
                                    v133[0l] = v123;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                Union3 v137;
                                v137 = v133[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                v139 = v137;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union1 v140;
                        v140 = Union1{Union1_1{v52, v139}};
                        v2.push(v140);
                        Union4 v226;
                        switch (v49.tag) {
                            case 0: { // None
                                switch (v139.tag) {
                                    case 0: { // Call
                                        if (v50){
                                            bool v190;
                                            v190 = v52 == 0l;
                                            int v191;
                                            if (v190){
                                                v191 = 1l;
                                            } else {
                                                v191 = 0l;
                                            }
                                            v226 = Union4{Union4_2{v49, false, v51, v191, v53, v54}};
                                        } else {
                                            v226 = Union4{Union4_0{v49, v50, v51, v52, v53, v54}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v226 = Union4{Union4_5{v49, v50, v51, v52, v53, v54}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v195;
                                        v195 = v54 > 0l;
                                        if (v195){
                                            bool v196;
                                            v196 = v52 == 0l;
                                            int v197;
                                            if (v196){
                                                v197 = 1l;
                                            } else {
                                                v197 = 0l;
                                            }
                                            int v198;
                                            v198 = -1l + v54;
                                            int v199; int v200;
                                            Tuple1 tmp4 = Tuple1{0l, 0l};
                                            v199 = tmp4.v0; v200 = tmp4.v1;
                                            while (while_method_3(v199)){
                                                int v202;
                                                v202 = v53[v199];
                                                bool v204;
                                                v204 = v200 >= v202;
                                                int v205;
                                                if (v204){
                                                    v205 = v200;
                                                } else {
                                                    v205 = v202;
                                                }
                                                v200 = v205;
                                                v199 += 1l ;
                                            }
                                            static_array<int,2l> v206;
                                            int v208;
                                            v208 = 0l;
                                            while (while_method_3(v208)){
                                                v206[v208] = v200;
                                                v208 += 1l ;
                                            }
                                            static_array<int,2l> v210;
                                            int v212;
                                            v212 = 0l;
                                            while (while_method_3(v212)){
                                                int v214;
                                                v214 = v206[v212];
                                                bool v216;
                                                v216 = v212 == v52;
                                                int v218;
                                                if (v216){
                                                    int v217;
                                                    v217 = v214 + 2l;
                                                    v218 = v217;
                                                } else {
                                                    v218 = v214;
                                                }
                                                v210[v212] = v218;
                                                v212 += 1l ;
                                            }
                                            v226 = Union4{Union4_2{v49, false, v51, v197, v210, v198}};
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
                                Union2 v141 = v49.case1.v0;
                                switch (v139.tag) {
                                    case 0: { // Call
                                        if (v50){
                                            bool v143;
                                            v143 = v52 == 0l;
                                            int v144;
                                            if (v143){
                                                v144 = 1l;
                                            } else {
                                                v144 = 0l;
                                            }
                                            v226 = Union4{Union4_2{v49, false, v51, v144, v53, v54}};
                                        } else {
                                            int v146; int v147;
                                            Tuple1 tmp5 = Tuple1{0l, 0l};
                                            v146 = tmp5.v0; v147 = tmp5.v1;
                                            while (while_method_3(v146)){
                                                int v149;
                                                v149 = v53[v146];
                                                bool v151;
                                                v151 = v147 >= v149;
                                                int v152;
                                                if (v151){
                                                    v152 = v147;
                                                } else {
                                                    v152 = v149;
                                                }
                                                v147 = v152;
                                                v146 += 1l ;
                                            }
                                            static_array<int,2l> v153;
                                            int v155;
                                            v155 = 0l;
                                            while (while_method_3(v155)){
                                                v153[v155] = v147;
                                                v155 += 1l ;
                                            }
                                            v226 = Union4{Union4_4{v49, v50, v51, v52, v153, v54}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v226 = Union4{Union4_5{v49, v50, v51, v52, v53, v54}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v159;
                                        v159 = v54 > 0l;
                                        if (v159){
                                            bool v160;
                                            v160 = v52 == 0l;
                                            int v161;
                                            if (v160){
                                                v161 = 1l;
                                            } else {
                                                v161 = 0l;
                                            }
                                            int v162;
                                            v162 = -1l + v54;
                                            int v163; int v164;
                                            Tuple1 tmp6 = Tuple1{0l, 0l};
                                            v163 = tmp6.v0; v164 = tmp6.v1;
                                            while (while_method_3(v163)){
                                                int v166;
                                                v166 = v53[v163];
                                                bool v168;
                                                v168 = v164 >= v166;
                                                int v169;
                                                if (v168){
                                                    v169 = v164;
                                                } else {
                                                    v169 = v166;
                                                }
                                                v164 = v169;
                                                v163 += 1l ;
                                            }
                                            static_array<int,2l> v170;
                                            int v172;
                                            v172 = 0l;
                                            while (while_method_3(v172)){
                                                v170[v172] = v164;
                                                v172 += 1l ;
                                            }
                                            static_array<int,2l> v174;
                                            int v176;
                                            v176 = 0l;
                                            while (while_method_3(v176)){
                                                int v178;
                                                v178 = v170[v176];
                                                bool v180;
                                                v180 = v176 == v52;
                                                int v182;
                                                if (v180){
                                                    int v181;
                                                    v181 = v178 + 4l;
                                                    v182 = v181;
                                                } else {
                                                    v182 = v178;
                                                }
                                                v174[v176] = v182;
                                                v176 += 1l ;
                                            }
                                            v226 = Union4{Union4_2{v49, false, v51, v161, v174, v162}};
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
                        v378 = Union6{Union6_1{v226}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v228 = v6.case3.v0; bool v229 = v6.case3.v1; static_array<Union2,2l> v230 = v6.case3.v2; int v231 = v6.case3.v3; static_array<int,2l> v232 = v6.case3.v4; int v233 = v6.case3.v5; Union3 v234 = v6.case3.v6;
                        Union1 v235;
                        v235 = Union1{Union1_1{v231, v234}};
                        v2.push(v235);
                        Union4 v321;
                        switch (v228.tag) {
                            case 0: { // None
                                switch (v234.tag) {
                                    case 0: { // Call
                                        if (v229){
                                            bool v285;
                                            v285 = v231 == 0l;
                                            int v286;
                                            if (v285){
                                                v286 = 1l;
                                            } else {
                                                v286 = 0l;
                                            }
                                            v321 = Union4{Union4_2{v228, false, v230, v286, v232, v233}};
                                        } else {
                                            v321 = Union4{Union4_0{v228, v229, v230, v231, v232, v233}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v321 = Union4{Union4_5{v228, v229, v230, v231, v232, v233}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v290;
                                        v290 = v233 > 0l;
                                        if (v290){
                                            bool v291;
                                            v291 = v231 == 0l;
                                            int v292;
                                            if (v291){
                                                v292 = 1l;
                                            } else {
                                                v292 = 0l;
                                            }
                                            int v293;
                                            v293 = -1l + v233;
                                            int v294; int v295;
                                            Tuple1 tmp7 = Tuple1{0l, 0l};
                                            v294 = tmp7.v0; v295 = tmp7.v1;
                                            while (while_method_3(v294)){
                                                int v297;
                                                v297 = v232[v294];
                                                bool v299;
                                                v299 = v295 >= v297;
                                                int v300;
                                                if (v299){
                                                    v300 = v295;
                                                } else {
                                                    v300 = v297;
                                                }
                                                v295 = v300;
                                                v294 += 1l ;
                                            }
                                            static_array<int,2l> v301;
                                            int v303;
                                            v303 = 0l;
                                            while (while_method_3(v303)){
                                                v301[v303] = v295;
                                                v303 += 1l ;
                                            }
                                            static_array<int,2l> v305;
                                            int v307;
                                            v307 = 0l;
                                            while (while_method_3(v307)){
                                                int v309;
                                                v309 = v301[v307];
                                                bool v311;
                                                v311 = v307 == v231;
                                                int v313;
                                                if (v311){
                                                    int v312;
                                                    v312 = v309 + 2l;
                                                    v313 = v312;
                                                } else {
                                                    v313 = v309;
                                                }
                                                v305[v307] = v313;
                                                v307 += 1l ;
                                            }
                                            v321 = Union4{Union4_2{v228, false, v230, v292, v305, v293}};
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
                                Union2 v236 = v228.case1.v0;
                                switch (v234.tag) {
                                    case 0: { // Call
                                        if (v229){
                                            bool v238;
                                            v238 = v231 == 0l;
                                            int v239;
                                            if (v238){
                                                v239 = 1l;
                                            } else {
                                                v239 = 0l;
                                            }
                                            v321 = Union4{Union4_2{v228, false, v230, v239, v232, v233}};
                                        } else {
                                            int v241; int v242;
                                            Tuple1 tmp8 = Tuple1{0l, 0l};
                                            v241 = tmp8.v0; v242 = tmp8.v1;
                                            while (while_method_3(v241)){
                                                int v244;
                                                v244 = v232[v241];
                                                bool v246;
                                                v246 = v242 >= v244;
                                                int v247;
                                                if (v246){
                                                    v247 = v242;
                                                } else {
                                                    v247 = v244;
                                                }
                                                v242 = v247;
                                                v241 += 1l ;
                                            }
                                            static_array<int,2l> v248;
                                            int v250;
                                            v250 = 0l;
                                            while (while_method_3(v250)){
                                                v248[v250] = v242;
                                                v250 += 1l ;
                                            }
                                            v321 = Union4{Union4_4{v228, v229, v230, v231, v248, v233}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v321 = Union4{Union4_5{v228, v229, v230, v231, v232, v233}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v254;
                                        v254 = v233 > 0l;
                                        if (v254){
                                            bool v255;
                                            v255 = v231 == 0l;
                                            int v256;
                                            if (v255){
                                                v256 = 1l;
                                            } else {
                                                v256 = 0l;
                                            }
                                            int v257;
                                            v257 = -1l + v233;
                                            int v258; int v259;
                                            Tuple1 tmp9 = Tuple1{0l, 0l};
                                            v258 = tmp9.v0; v259 = tmp9.v1;
                                            while (while_method_3(v258)){
                                                int v261;
                                                v261 = v232[v258];
                                                bool v263;
                                                v263 = v259 >= v261;
                                                int v264;
                                                if (v263){
                                                    v264 = v259;
                                                } else {
                                                    v264 = v261;
                                                }
                                                v259 = v264;
                                                v258 += 1l ;
                                            }
                                            static_array<int,2l> v265;
                                            int v267;
                                            v267 = 0l;
                                            while (while_method_3(v267)){
                                                v265[v267] = v259;
                                                v267 += 1l ;
                                            }
                                            static_array<int,2l> v269;
                                            int v271;
                                            v271 = 0l;
                                            while (while_method_3(v271)){
                                                int v273;
                                                v273 = v265[v271];
                                                bool v275;
                                                v275 = v271 == v231;
                                                int v277;
                                                if (v275){
                                                    int v276;
                                                    v276 = v273 + 4l;
                                                    v277 = v276;
                                                } else {
                                                    v277 = v273;
                                                }
                                                v269[v271] = v277;
                                                v271 += 1l ;
                                            }
                                            v321 = Union4{Union4_2{v228, false, v230, v256, v269, v257}};
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
                        v378 = Union6{Union6_1{v321}};
                        break;
                    }
                    case 4: { // TerminalCall
                        Union5 v24 = v6.case4.v0; bool v25 = v6.case4.v1; static_array<Union2,2l> v26 = v6.case4.v2; int v27 = v6.case4.v3; static_array<int,2l> v28 = v6.case4.v4; int v29 = v6.case4.v5;
                        int v30;
                        v30 = v28[v27];
                        Union7 v32;
                        v32 = compare_hands_4(v24, v25, v26, v27, v28, v29);
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
                        static_array<float,2l> v43;
                        v43 = v0.base->v4;
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
                        v378 = Union6{Union6_0{}};
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
                        static_array<float,2l> v17;
                        v17 = v0.base->v4;
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
                        v378 = Union6{Union6_0{}};
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
        v4 = v378;
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
    sptr<Mut0> v22;
    v22 = sptr<Mut0>{new Mut0{63ul, v20, v16, v10, v18, v21}};
    int v23;
    v23 = 0l;
    while (while_method_0(v23)){
        int v25;
        v25 = 0l;
        while (while_method_1(v25)){
            v22.base->v0 = 63ul;
            static_array<float,2l> v27;
            v27[0l] = 0.0f;
            v27[1l] = 0.0f;
            v22.base->v4 = v27;
            static_array_list<Union1,32l> v29;
            v29 = v22.base->v2;
            v29.unsafe_set_length(0l);
            static_array<Union0,2l> v30;
            Union0 v32;
            v32 = Union0{Union0_1{}};
            v30[0l] = v32;
            Union0 v34;
            v34 = Union0{Union0_1{}};
            v30[1l] = v34;
            Union0 v36;
            v36 = Union0{Union0_0{}};
            v30[0l] = v36;
            v22.base->v3 = v30;
            Union4 v38;
            v38 = Union4{Union4_1{}};
            method_0(v22, v38);
            static_array<float,2l> v39;
            v39 = v22.base->v4;
            static_array<float,2l> v40;
            v40 = v39;
            v25 += 1l ;
        }
        v23 += 1l ;
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
options.append('--diag-suppress=550,20012,68,39')
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
