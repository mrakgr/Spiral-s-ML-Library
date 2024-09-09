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
__device__ static_array<float,2l> method_0(unsigned int & v0, static_array_list<Union1,32l> & v1, static_array<Union0,2l> & v2, curandStatePhilox4_32_10_t & v3, Union4 v4);
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
__device__ static_array<float,2l> method_0(unsigned int & v0, static_array_list<Union1,32l> & v1, static_array<Union0,2l> & v2, curandStatePhilox4_32_10_t & v3, Union4 v4){
    static_array<float,2l> v5;
    static_array_list<Union1,32l> & v7 = v1;
    Union6 v8;
    v8 = Union6{Union6_1{v4}};
    Union6 v9;
    v9 = v8;
    while (while_method_2(v9)){
        Union6 v371;
        switch (v9.tag) {
            case 0: { // None
                v371 = Union6{Union6_0{}};
                break;
            }
            case 1: { // Some
                Union4 v11 = v9.case1.v0;
                switch (v11.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v322 = v11.case0.v0; bool v323 = v11.case0.v1; static_array<Union2,2l> v324 = v11.case0.v2; int v325 = v11.case0.v3; static_array<int,2l> v326 = v11.case0.v4; int v327 = v11.case0.v5;
                        unsigned int v328 = v0;
                        Union2 v329; unsigned int v330;
                        Tuple0 tmp0 = draw_card_1(v3, v328);
                        v329 = tmp0.v0; v330 = tmp0.v1;
                        v0 = v330;
                        Union1 v331;
                        v331 = Union1{Union1_0{v329}};
                        v7.push(v331);
                        int v332;
                        v332 = 2l;
                        int v333; int v334;
                        Tuple1 tmp1 = Tuple1{0l, 0l};
                        v333 = tmp1.v0; v334 = tmp1.v1;
                        while (while_method_3(v333)){
                            int v336;
                            v336 = v326[v333];
                            bool v338;
                            v338 = v334 >= v336;
                            int v339;
                            if (v338){
                                v339 = v334;
                            } else {
                                v339 = v336;
                            }
                            v334 = v339;
                            v333 += 1l ;
                        }
                        static_array<int,2l> v340;
                        int v342;
                        v342 = 0l;
                        while (while_method_3(v342)){
                            v340[v342] = v334;
                            v342 += 1l ;
                        }
                        Union5 v344;
                        v344 = Union5{Union5_1{v329}};
                        Union4 v345;
                        v345 = Union4{Union4_2{v344, true, v324, 0l, v340, v332}};
                        v371 = Union6{Union6_1{v345}};
                        break;
                    }
                    case 1: { // ChanceInit
                        unsigned int v347 = v0;
                        Union2 v348; unsigned int v349;
                        Tuple0 tmp2 = draw_card_1(v3, v347);
                        v348 = tmp2.v0; v349 = tmp2.v1;
                        v0 = v349;
                        unsigned int v350 = v0;
                        Union2 v351; unsigned int v352;
                        Tuple0 tmp3 = draw_card_1(v3, v350);
                        v351 = tmp3.v0; v352 = tmp3.v1;
                        v0 = v352;
                        Union1 v353;
                        v353 = Union1{Union1_2{0l, v348}};
                        v7.push(v353);
                        Union1 v354;
                        v354 = Union1{Union1_2{1l, v351}};
                        v7.push(v354);
                        int v355;
                        v355 = 2l;
                        static_array<int,2l> v356;
                        v356[0l] = 1l;
                        v356[1l] = 1l;
                        static_array<Union2,2l> v358;
                        v358[0l] = v348;
                        v358[1l] = v351;
                        Union5 v360;
                        v360 = Union5{Union5_0{}};
                        Union4 v361;
                        v361 = Union4{Union4_2{v360, true, v358, 0l, v356, v355}};
                        v371 = Union6{Union6_1{v361}};
                        break;
                    }
                    case 2: { // Round
                        Union5 v52 = v11.case2.v0; bool v53 = v11.case2.v1; static_array<Union2,2l> v54 = v11.case2.v2; int v55 = v11.case2.v3; static_array<int,2l> v56 = v11.case2.v4; int v57 = v11.case2.v5;
                        static_array<Union0,2l> v58 = v2;
                        Union0 v59;
                        v59 = v58[v55];
                        Union3 v138;
                        switch (v59.tag) {
                            case 0: { // T_Computer
                                static_array_list<Union3,3l> v61;
                                v61 = static_array_list<Union3,3l>{};
                                v61.unsafe_set_length(1l);
                                Union3 v63;
                                v63 = Union3{Union3_0{}};
                                v61[0l] = v63;
                                int v65;
                                v65 = v56[0l];
                                int v67;
                                v67 = v56[1l];
                                bool v69;
                                v69 = v65 == v67;
                                bool v70;
                                v70 = v69 != true;
                                if (v70){
                                    Union3 v71;
                                    v71 = Union3{Union3_1{}};
                                    v61.push(v71);
                                } else {
                                }
                                bool v72;
                                v72 = v57 > 0l;
                                if (v72){
                                    Union3 v73;
                                    v73 = Union3{Union3_2{}};
                                    v61.push(v73);
                                } else {
                                }
                                int v74;
                                v74 = v61.length;
                                int v75;
                                v75 = v74 - 1l;
                                int v76;
                                v76 = 0l;
                                while (while_method_4(v75, v76)){
                                    int v78;
                                    v78 = v61.length;
                                    int v79;
                                    v79 = int_range_3(v78, v76, v3);
                                    Union3 v80;
                                    v80 = v61[v76];
                                    Union3 v82;
                                    v82 = v61[v79];
                                    v61[v76] = v82;
                                    v61[v79] = v80;
                                    v76 += 1l ;
                                }
                                Union3 v84;
                                v84 = v61.pop();
                                int v85;
                                v85 = sizeof(Union3);
                                unsigned long long v86;
                                v86 = (unsigned long long)v85;
                                bool v87;
                                v87 = v86 <= 81920ull;
                                bool v88;
                                v88 = v87 == false;
                                if (v88){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v87);
                                } else {
                                }
                                extern __shared__ unsigned char v90[];
                                bool v91;
                                v91 = v86 <= v86;
                                bool v92;
                                v92 = v91 == false;
                                if (v92){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v91);
                                } else {
                                }
                                Union3 * v94;
                                v94 = reinterpret_cast<Union3 *>(&v90[0ull]);
                                int v96;
                                v96 = threadIdx.x;
                                bool v97;
                                v97 = v96 == 0l;
                                if (v97){
                                    v94[0l] = v84;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                Union3 v98;
                                v98 = v94[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                v138 = v98;
                                break;
                            }
                            case 1: { // T_Random
                                static_array_list<Union3,3l> v99;
                                v99 = static_array_list<Union3,3l>{};
                                v99.unsafe_set_length(1l);
                                Union3 v101;
                                v101 = Union3{Union3_0{}};
                                v99[0l] = v101;
                                int v103;
                                v103 = v56[0l];
                                int v105;
                                v105 = v56[1l];
                                bool v107;
                                v107 = v103 == v105;
                                bool v108;
                                v108 = v107 != true;
                                if (v108){
                                    Union3 v109;
                                    v109 = Union3{Union3_1{}};
                                    v99.push(v109);
                                } else {
                                }
                                bool v110;
                                v110 = v57 > 0l;
                                if (v110){
                                    Union3 v111;
                                    v111 = Union3{Union3_2{}};
                                    v99.push(v111);
                                } else {
                                }
                                int v112;
                                v112 = v99.length;
                                int v113;
                                v113 = v112 - 1l;
                                int v114;
                                v114 = 0l;
                                while (while_method_4(v113, v114)){
                                    int v116;
                                    v116 = v99.length;
                                    int v117;
                                    v117 = int_range_3(v116, v114, v3);
                                    Union3 v118;
                                    v118 = v99[v114];
                                    Union3 v120;
                                    v120 = v99[v117];
                                    v99[v114] = v120;
                                    v99[v117] = v118;
                                    v114 += 1l ;
                                }
                                Union3 v122;
                                v122 = v99.pop();
                                int v123;
                                v123 = sizeof(Union3);
                                unsigned long long v124;
                                v124 = (unsigned long long)v123;
                                bool v125;
                                v125 = v124 <= 81920ull;
                                bool v126;
                                v126 = v125 == false;
                                if (v126){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v125);
                                } else {
                                }
                                extern __shared__ unsigned char v128[];
                                bool v129;
                                v129 = v124 <= v124;
                                bool v130;
                                v130 = v129 == false;
                                if (v130){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v129);
                                } else {
                                }
                                Union3 * v132;
                                v132 = reinterpret_cast<Union3 *>(&v128[0ull]);
                                int v134;
                                v134 = threadIdx.x;
                                bool v135;
                                v135 = v134 == 0l;
                                if (v135){
                                    v132[0l] = v122;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                Union3 v136;
                                v136 = v132[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                v138 = v136;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union1 v139;
                        v139 = Union1{Union1_1{v55, v138}};
                        v7.push(v139);
                        Union4 v225;
                        switch (v52.tag) {
                            case 0: { // None
                                switch (v138.tag) {
                                    case 0: { // Call
                                        if (v53){
                                            bool v189;
                                            v189 = v55 == 0l;
                                            int v190;
                                            if (v189){
                                                v190 = 1l;
                                            } else {
                                                v190 = 0l;
                                            }
                                            v225 = Union4{Union4_2{v52, false, v54, v190, v56, v57}};
                                        } else {
                                            v225 = Union4{Union4_0{v52, v53, v54, v55, v56, v57}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v225 = Union4{Union4_5{v52, v53, v54, v55, v56, v57}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v194;
                                        v194 = v57 > 0l;
                                        if (v194){
                                            bool v195;
                                            v195 = v55 == 0l;
                                            int v196;
                                            if (v195){
                                                v196 = 1l;
                                            } else {
                                                v196 = 0l;
                                            }
                                            int v197;
                                            v197 = -1l + v57;
                                            int v198; int v199;
                                            Tuple1 tmp4 = Tuple1{0l, 0l};
                                            v198 = tmp4.v0; v199 = tmp4.v1;
                                            while (while_method_3(v198)){
                                                int v201;
                                                v201 = v56[v198];
                                                bool v203;
                                                v203 = v199 >= v201;
                                                int v204;
                                                if (v203){
                                                    v204 = v199;
                                                } else {
                                                    v204 = v201;
                                                }
                                                v199 = v204;
                                                v198 += 1l ;
                                            }
                                            static_array<int,2l> v205;
                                            int v207;
                                            v207 = 0l;
                                            while (while_method_3(v207)){
                                                v205[v207] = v199;
                                                v207 += 1l ;
                                            }
                                            static_array<int,2l> v209;
                                            int v211;
                                            v211 = 0l;
                                            while (while_method_3(v211)){
                                                int v213;
                                                v213 = v205[v211];
                                                bool v215;
                                                v215 = v211 == v55;
                                                int v217;
                                                if (v215){
                                                    int v216;
                                                    v216 = v213 + 2l;
                                                    v217 = v216;
                                                } else {
                                                    v217 = v213;
                                                }
                                                v209[v211] = v217;
                                                v211 += 1l ;
                                            }
                                            v225 = Union4{Union4_2{v52, false, v54, v196, v209, v197}};
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
                                Union2 v140 = v52.case1.v0;
                                switch (v138.tag) {
                                    case 0: { // Call
                                        if (v53){
                                            bool v142;
                                            v142 = v55 == 0l;
                                            int v143;
                                            if (v142){
                                                v143 = 1l;
                                            } else {
                                                v143 = 0l;
                                            }
                                            v225 = Union4{Union4_2{v52, false, v54, v143, v56, v57}};
                                        } else {
                                            int v145; int v146;
                                            Tuple1 tmp5 = Tuple1{0l, 0l};
                                            v145 = tmp5.v0; v146 = tmp5.v1;
                                            while (while_method_3(v145)){
                                                int v148;
                                                v148 = v56[v145];
                                                bool v150;
                                                v150 = v146 >= v148;
                                                int v151;
                                                if (v150){
                                                    v151 = v146;
                                                } else {
                                                    v151 = v148;
                                                }
                                                v146 = v151;
                                                v145 += 1l ;
                                            }
                                            static_array<int,2l> v152;
                                            int v154;
                                            v154 = 0l;
                                            while (while_method_3(v154)){
                                                v152[v154] = v146;
                                                v154 += 1l ;
                                            }
                                            v225 = Union4{Union4_4{v52, v53, v54, v55, v152, v57}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v225 = Union4{Union4_5{v52, v53, v54, v55, v56, v57}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v158;
                                        v158 = v57 > 0l;
                                        if (v158){
                                            bool v159;
                                            v159 = v55 == 0l;
                                            int v160;
                                            if (v159){
                                                v160 = 1l;
                                            } else {
                                                v160 = 0l;
                                            }
                                            int v161;
                                            v161 = -1l + v57;
                                            int v162; int v163;
                                            Tuple1 tmp6 = Tuple1{0l, 0l};
                                            v162 = tmp6.v0; v163 = tmp6.v1;
                                            while (while_method_3(v162)){
                                                int v165;
                                                v165 = v56[v162];
                                                bool v167;
                                                v167 = v163 >= v165;
                                                int v168;
                                                if (v167){
                                                    v168 = v163;
                                                } else {
                                                    v168 = v165;
                                                }
                                                v163 = v168;
                                                v162 += 1l ;
                                            }
                                            static_array<int,2l> v169;
                                            int v171;
                                            v171 = 0l;
                                            while (while_method_3(v171)){
                                                v169[v171] = v163;
                                                v171 += 1l ;
                                            }
                                            static_array<int,2l> v173;
                                            int v175;
                                            v175 = 0l;
                                            while (while_method_3(v175)){
                                                int v177;
                                                v177 = v169[v175];
                                                bool v179;
                                                v179 = v175 == v55;
                                                int v181;
                                                if (v179){
                                                    int v180;
                                                    v180 = v177 + 4l;
                                                    v181 = v180;
                                                } else {
                                                    v181 = v177;
                                                }
                                                v173[v175] = v181;
                                                v175 += 1l ;
                                            }
                                            v225 = Union4{Union4_2{v52, false, v54, v160, v173, v161}};
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
                        v371 = Union6{Union6_1{v225}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v227 = v11.case3.v0; bool v228 = v11.case3.v1; static_array<Union2,2l> v229 = v11.case3.v2; int v230 = v11.case3.v3; static_array<int,2l> v231 = v11.case3.v4; int v232 = v11.case3.v5; Union3 v233 = v11.case3.v6;
                        Union1 v234;
                        v234 = Union1{Union1_1{v230, v233}};
                        v7.push(v234);
                        Union4 v320;
                        switch (v227.tag) {
                            case 0: { // None
                                switch (v233.tag) {
                                    case 0: { // Call
                                        if (v228){
                                            bool v284;
                                            v284 = v230 == 0l;
                                            int v285;
                                            if (v284){
                                                v285 = 1l;
                                            } else {
                                                v285 = 0l;
                                            }
                                            v320 = Union4{Union4_2{v227, false, v229, v285, v231, v232}};
                                        } else {
                                            v320 = Union4{Union4_0{v227, v228, v229, v230, v231, v232}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v320 = Union4{Union4_5{v227, v228, v229, v230, v231, v232}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v289;
                                        v289 = v232 > 0l;
                                        if (v289){
                                            bool v290;
                                            v290 = v230 == 0l;
                                            int v291;
                                            if (v290){
                                                v291 = 1l;
                                            } else {
                                                v291 = 0l;
                                            }
                                            int v292;
                                            v292 = -1l + v232;
                                            int v293; int v294;
                                            Tuple1 tmp7 = Tuple1{0l, 0l};
                                            v293 = tmp7.v0; v294 = tmp7.v1;
                                            while (while_method_3(v293)){
                                                int v296;
                                                v296 = v231[v293];
                                                bool v298;
                                                v298 = v294 >= v296;
                                                int v299;
                                                if (v298){
                                                    v299 = v294;
                                                } else {
                                                    v299 = v296;
                                                }
                                                v294 = v299;
                                                v293 += 1l ;
                                            }
                                            static_array<int,2l> v300;
                                            int v302;
                                            v302 = 0l;
                                            while (while_method_3(v302)){
                                                v300[v302] = v294;
                                                v302 += 1l ;
                                            }
                                            static_array<int,2l> v304;
                                            int v306;
                                            v306 = 0l;
                                            while (while_method_3(v306)){
                                                int v308;
                                                v308 = v300[v306];
                                                bool v310;
                                                v310 = v306 == v230;
                                                int v312;
                                                if (v310){
                                                    int v311;
                                                    v311 = v308 + 2l;
                                                    v312 = v311;
                                                } else {
                                                    v312 = v308;
                                                }
                                                v304[v306] = v312;
                                                v306 += 1l ;
                                            }
                                            v320 = Union4{Union4_2{v227, false, v229, v291, v304, v292}};
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
                                Union2 v235 = v227.case1.v0;
                                switch (v233.tag) {
                                    case 0: { // Call
                                        if (v228){
                                            bool v237;
                                            v237 = v230 == 0l;
                                            int v238;
                                            if (v237){
                                                v238 = 1l;
                                            } else {
                                                v238 = 0l;
                                            }
                                            v320 = Union4{Union4_2{v227, false, v229, v238, v231, v232}};
                                        } else {
                                            int v240; int v241;
                                            Tuple1 tmp8 = Tuple1{0l, 0l};
                                            v240 = tmp8.v0; v241 = tmp8.v1;
                                            while (while_method_3(v240)){
                                                int v243;
                                                v243 = v231[v240];
                                                bool v245;
                                                v245 = v241 >= v243;
                                                int v246;
                                                if (v245){
                                                    v246 = v241;
                                                } else {
                                                    v246 = v243;
                                                }
                                                v241 = v246;
                                                v240 += 1l ;
                                            }
                                            static_array<int,2l> v247;
                                            int v249;
                                            v249 = 0l;
                                            while (while_method_3(v249)){
                                                v247[v249] = v241;
                                                v249 += 1l ;
                                            }
                                            v320 = Union4{Union4_4{v227, v228, v229, v230, v247, v232}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v320 = Union4{Union4_5{v227, v228, v229, v230, v231, v232}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v253;
                                        v253 = v232 > 0l;
                                        if (v253){
                                            bool v254;
                                            v254 = v230 == 0l;
                                            int v255;
                                            if (v254){
                                                v255 = 1l;
                                            } else {
                                                v255 = 0l;
                                            }
                                            int v256;
                                            v256 = -1l + v232;
                                            int v257; int v258;
                                            Tuple1 tmp9 = Tuple1{0l, 0l};
                                            v257 = tmp9.v0; v258 = tmp9.v1;
                                            while (while_method_3(v257)){
                                                int v260;
                                                v260 = v231[v257];
                                                bool v262;
                                                v262 = v258 >= v260;
                                                int v263;
                                                if (v262){
                                                    v263 = v258;
                                                } else {
                                                    v263 = v260;
                                                }
                                                v258 = v263;
                                                v257 += 1l ;
                                            }
                                            static_array<int,2l> v264;
                                            int v266;
                                            v266 = 0l;
                                            while (while_method_3(v266)){
                                                v264[v266] = v258;
                                                v266 += 1l ;
                                            }
                                            static_array<int,2l> v268;
                                            int v270;
                                            v270 = 0l;
                                            while (while_method_3(v270)){
                                                int v272;
                                                v272 = v264[v270];
                                                bool v274;
                                                v274 = v270 == v230;
                                                int v276;
                                                if (v274){
                                                    int v275;
                                                    v275 = v272 + 4l;
                                                    v276 = v275;
                                                } else {
                                                    v276 = v272;
                                                }
                                                v268[v270] = v276;
                                                v270 += 1l ;
                                            }
                                            v320 = Union4{Union4_2{v227, false, v229, v255, v268, v256}};
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
                        v371 = Union6{Union6_1{v320}};
                        break;
                    }
                    case 4: { // TerminalCall
                        Union5 v28 = v11.case4.v0; bool v29 = v11.case4.v1; static_array<Union2,2l> v30 = v11.case4.v2; int v31 = v11.case4.v3; static_array<int,2l> v32 = v11.case4.v4; int v33 = v11.case4.v5;
                        int v34;
                        v34 = v32[v31];
                        Union7 v36;
                        v36 = compare_hands_4(v28, v29, v30, v31, v32, v33);
                        int v41; int v42;
                        switch (v36.tag) {
                            case 0: { // Eq
                                v41 = 0l; v42 = -1l;
                                break;
                            }
                            case 1: { // Gt
                                v41 = v34; v42 = 0l;
                                break;
                            }
                            case 2: { // Lt
                                v41 = v34; v42 = 1l;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        int v43;
                        v43 = -v42;
                        bool v44;
                        v44 = v42 >= v43;
                        int v45;
                        if (v44){
                            v45 = v42;
                        } else {
                            v45 = v43;
                        }
                        float v46;
                        v46 = (float)v41;
                        v5[v45] = v46;
                        bool v47;
                        v47 = v45 == 0l;
                        int v48;
                        if (v47){
                            v48 = 1l;
                        } else {
                            v48 = 0l;
                        }
                        float v49;
                        v49 = -v46;
                        v5[v48] = v49;
                        Union1 v50;
                        v50 = Union1{Union1_3{v30, v41, v42}};
                        v7.push(v50);
                        v371 = Union6{Union6_0{}};
                        break;
                    }
                    case 5: { // TerminalFold
                        Union5 v12 = v11.case5.v0; bool v13 = v11.case5.v1; static_array<Union2,2l> v14 = v11.case5.v2; int v15 = v11.case5.v3; static_array<int,2l> v16 = v11.case5.v4; int v17 = v11.case5.v5;
                        int v18;
                        v18 = v16[v15];
                        int v20;
                        v20 = -v18;
                        float v21;
                        v21 = (float)v20;
                        v5[v15] = v21;
                        bool v22;
                        v22 = v15 == 0l;
                        int v23;
                        if (v22){
                            v23 = 1l;
                        } else {
                            v23 = 0l;
                        }
                        float v24;
                        v24 = -v21;
                        v5[v23] = v24;
                        int v25;
                        if (v22){
                            v25 = 1l;
                        } else {
                            v25 = 0l;
                        }
                        Union1 v26;
                        v26 = Union1{Union1_3{v14, v18, v25}};
                        v7.push(v26);
                        v371 = Union6{Union6_0{}};
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
        v9 = v371;
    }
    return v5;
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
    int v10;
    v10 = 0l;
    while (while_method_0(v10)){
        int v12;
        v12 = 0l;
        while (while_method_1(v12)){
            static_array<Union0,2l> v14;
            Union0 v16;
            v16 = Union0{Union0_1{}};
            v14[0l] = v16;
            Union0 v18;
            v18 = Union0{Union0_1{}};
            v14[1l] = v18;
            static_array<Union0,2l> & v20 = v14;
            unsigned int v21 = 63ul;
            static_array_list<Union1,32l> v22;
            v22 = static_array_list<Union1,32l>{};
            static_array_list<Union1,32l> & v24 = v22;
            Union4 v25;
            v25 = Union4{Union4_1{}};
            static_array<float,2l> v26;
            v26 = method_0(v21, v24, v20, v9, v25);
            v12 += 1l ;
        }
        v10 += 1l ;
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
