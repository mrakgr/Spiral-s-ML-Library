kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
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

struct Union1;
struct Union2;
struct Union0;
__device__ int f_1(unsigned char * v0);
__device__ void f_3(unsigned char * v0);
__device__ Union1 f_2(unsigned char * v0);
__device__ Union2 f_5(unsigned char * v0);
__device__ static_array<Union2,2l> f_4(unsigned char * v0);
__device__ Union0 f_0(unsigned char * v0);
struct Union3;
struct Union4;
struct Union7;
struct Union6;
struct Union5;
struct Union8;
struct Tuple0;
__device__ Union3 f_7(unsigned char * v0);
__device__ int f_8(unsigned char * v0);
struct Tuple1;
__device__ int f_11(unsigned char * v0);
__device__ Tuple1 f_10(unsigned char * v0);
struct Tuple2;
__device__ Tuple2 f_12(unsigned char * v0);
struct Tuple3;
__device__ Tuple3 f_13(unsigned char * v0);
__device__ Union4 f_9(unsigned char * v0);
__device__ int f_14(unsigned char * v0);
struct Tuple4;
__device__ Tuple4 f_16(unsigned char * v0);
struct Tuple5;
__device__ int f_18(unsigned char * v0);
__device__ Tuple5 f_17(unsigned char * v0);
__device__ Union6 f_15(unsigned char * v0);
__device__ int f_19(unsigned char * v0);
__device__ Tuple0 f_6(unsigned char * v0);
struct Tuple6;
struct Tuple7;
struct Tuple8;
__device__ unsigned int loop_22(unsigned int v0, curandStatePhilox4_32_10_t & v1);
struct Union9;
__device__ int tag_24(Union3 v0);
__device__ bool is_pair_25(int v0, int v1);
__device__ Tuple8 order_26(int v0, int v1);
__device__ Union9 compare_hands_23(Union7 v0, bool v1, static_array<Union3,2l> v2, int v3, static_array<int,2l> v4, int v5);
__device__ Union6 play_loop_inner_21(static_array_list<Union3,6l> & v0, static_array_list<Union4,32l> & v1, static_array<Union2,2l> v2, Union6 v3);
__device__ Tuple6 play_loop_20(Union5 v0, static_array<Union2,2l> v1, Union8 v2, static_array_list<Union3,6l> & v3, static_array_list<Union4,32l> & v4, Union6 v5);
__device__ void f_28(unsigned char * v0, int v1);
__device__ void f_30(unsigned char * v0);
__device__ void f_29(unsigned char * v0, Union3 v1);
__device__ void f_31(unsigned char * v0, int v1);
__device__ void f_34(unsigned char * v0, int v1);
__device__ void f_33(unsigned char * v0, int v1, Union1 v2);
__device__ void f_35(unsigned char * v0, int v1, Union3 v2);
__device__ void f_36(unsigned char * v0, static_array<Union3,2l> v1, int v2, int v3);
__device__ void f_32(unsigned char * v0, Union4 v1);
__device__ void f_37(unsigned char * v0, int v1);
__device__ void f_39(unsigned char * v0, Union7 v1, bool v2, static_array<Union3,2l> v3, int v4, static_array<int,2l> v5, int v6);
__device__ void f_41(unsigned char * v0, int v1);
__device__ void f_40(unsigned char * v0, Union7 v1, bool v2, static_array<Union3,2l> v3, int v4, static_array<int,2l> v5, int v6, Union1 v7);
__device__ void f_38(unsigned char * v0, Union6 v1);
__device__ void f_42(unsigned char * v0, Union2 v1);
__device__ void f_43(unsigned char * v0, int v1);
__device__ void f_27(unsigned char * v0, static_array_list<Union3,6l> v1, static_array_list<Union4,32l> v2, Union5 v3, static_array<Union2,2l> v4, Union8 v5);
__device__ void f_45(unsigned char * v0, int v1);
__device__ void f_44(unsigned char * v0, static_array_list<Union4,32l> v1, static_array<Union2,2l> v2, Union8 v3);
struct Union1_0 { // Call
};
struct Union1_1 { // Fold
};
struct Union1_2 { // Raise
};
struct Union1 {
    union {
        Union1_0 case0; // Call
        Union1_1 case1; // Fold
        Union1_2 case2; // Raise
    };
    unsigned char tag{255};
    __device__ Union1() {}
    __device__ Union1(Union1_0 t) : tag(0), case0(t) {} // Call
    __device__ Union1(Union1_1 t) : tag(1), case1(t) {} // Fold
    __device__ Union1(Union1_2 t) : tag(2), case2(t) {} // Raise
    __device__ Union1(Union1 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(x.case0); break; // Call
            case 1: new (&this->case1) Union1_1(x.case1); break; // Fold
            case 2: new (&this->case2) Union1_2(x.case2); break; // Raise
        }
    }
    __device__ Union1(Union1 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // Call
            case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // Fold
            case 2: new (&this->case2) Union1_2(std::move(x.case2)); break; // Raise
        }
    }
    __device__ Union1 & operator=(Union1 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Call
                case 1: this->case1 = x.case1; break; // Fold
                case 2: this->case2 = x.case2; break; // Raise
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
                case 0: this->case0 = std::move(x.case0); break; // Call
                case 1: this->case1 = std::move(x.case1); break; // Fold
                case 2: this->case2 = std::move(x.case2); break; // Raise
            }
        } else {
            this->~Union1();
            new (this) Union1{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union1() {
        switch(this->tag){
            case 0: this->case0.~Union1_0(); break; // Call
            case 1: this->case1.~Union1_1(); break; // Fold
            case 2: this->case2.~Union1_2(); break; // Raise
        }
        this->tag = 255;
    }
};
struct Union2_0 { // Computer
};
struct Union2_1 { // Human
};
struct Union2 {
    union {
        Union2_0 case0; // Computer
        Union2_1 case1; // Human
    };
    unsigned char tag{255};
    __device__ Union2() {}
    __device__ Union2(Union2_0 t) : tag(0), case0(t) {} // Computer
    __device__ Union2(Union2_1 t) : tag(1), case1(t) {} // Human
    __device__ Union2(Union2 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(x.case0); break; // Computer
            case 1: new (&this->case1) Union2_1(x.case1); break; // Human
        }
    }
    __device__ Union2(Union2 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(std::move(x.case0)); break; // Computer
            case 1: new (&this->case1) Union2_1(std::move(x.case1)); break; // Human
        }
    }
    __device__ Union2 & operator=(Union2 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Computer
                case 1: this->case1 = x.case1; break; // Human
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
                case 0: this->case0 = std::move(x.case0); break; // Computer
                case 1: this->case1 = std::move(x.case1); break; // Human
            }
        } else {
            this->~Union2();
            new (this) Union2{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union2() {
        switch(this->tag){
            case 0: this->case0.~Union2_0(); break; // Computer
            case 1: this->case1.~Union2_1(); break; // Human
        }
        this->tag = 255;
    }
};
struct Union0_0 { // ActionSelected
    Union1 v0;
    __device__ Union0_0(Union1 t0) : v0(t0) {}
    __device__ Union0_0() = delete;
};
struct Union0_1 { // PlayerChanged
    static_array<Union2,2l> v0;
    __device__ Union0_1(static_array<Union2,2l> t0) : v0(t0) {}
    __device__ Union0_1() = delete;
};
struct Union0_2 { // StartGame
};
struct Union0 {
    union {
        Union0_0 case0; // ActionSelected
        Union0_1 case1; // PlayerChanged
        Union0_2 case2; // StartGame
    };
    unsigned char tag{255};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // ActionSelected
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // PlayerChanged
    __device__ Union0(Union0_2 t) : tag(2), case2(t) {} // StartGame
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // ActionSelected
            case 1: new (&this->case1) Union0_1(x.case1); break; // PlayerChanged
            case 2: new (&this->case2) Union0_2(x.case2); break; // StartGame
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // ActionSelected
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // PlayerChanged
            case 2: new (&this->case2) Union0_2(std::move(x.case2)); break; // StartGame
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // ActionSelected
                case 1: this->case1 = x.case1; break; // PlayerChanged
                case 2: this->case2 = x.case2; break; // StartGame
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
                case 0: this->case0 = std::move(x.case0); break; // ActionSelected
                case 1: this->case1 = std::move(x.case1); break; // PlayerChanged
                case 2: this->case2 = std::move(x.case2); break; // StartGame
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // ActionSelected
            case 1: this->case1.~Union0_1(); break; // PlayerChanged
            case 2: this->case2.~Union0_2(); break; // StartGame
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
struct Union4_0 { // CommunityCardIs
    Union3 v0;
    __device__ Union4_0(Union3 t0) : v0(t0) {}
    __device__ Union4_0() = delete;
};
struct Union4_1 { // PlayerAction
    Union1 v1;
    int v0;
    __device__ Union4_1(int t0, Union1 t1) : v0(t0), v1(t1) {}
    __device__ Union4_1() = delete;
};
struct Union4_2 { // PlayerGotCard
    Union3 v1;
    int v0;
    __device__ Union4_2(int t0, Union3 t1) : v0(t0), v1(t1) {}
    __device__ Union4_2() = delete;
};
struct Union4_3 { // Showdown
    static_array<Union3,2l> v0;
    int v1;
    int v2;
    __device__ Union4_3(static_array<Union3,2l> t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
    __device__ Union4_3() = delete;
};
struct Union4 {
    union {
        Union4_0 case0; // CommunityCardIs
        Union4_1 case1; // PlayerAction
        Union4_2 case2; // PlayerGotCard
        Union4_3 case3; // Showdown
    };
    unsigned char tag{255};
    __device__ Union4() {}
    __device__ Union4(Union4_0 t) : tag(0), case0(t) {} // CommunityCardIs
    __device__ Union4(Union4_1 t) : tag(1), case1(t) {} // PlayerAction
    __device__ Union4(Union4_2 t) : tag(2), case2(t) {} // PlayerGotCard
    __device__ Union4(Union4_3 t) : tag(3), case3(t) {} // Showdown
    __device__ Union4(Union4 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(x.case0); break; // CommunityCardIs
            case 1: new (&this->case1) Union4_1(x.case1); break; // PlayerAction
            case 2: new (&this->case2) Union4_2(x.case2); break; // PlayerGotCard
            case 3: new (&this->case3) Union4_3(x.case3); break; // Showdown
        }
    }
    __device__ Union4(Union4 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(std::move(x.case0)); break; // CommunityCardIs
            case 1: new (&this->case1) Union4_1(std::move(x.case1)); break; // PlayerAction
            case 2: new (&this->case2) Union4_2(std::move(x.case2)); break; // PlayerGotCard
            case 3: new (&this->case3) Union4_3(std::move(x.case3)); break; // Showdown
        }
    }
    __device__ Union4 & operator=(Union4 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // CommunityCardIs
                case 1: this->case1 = x.case1; break; // PlayerAction
                case 2: this->case2 = x.case2; break; // PlayerGotCard
                case 3: this->case3 = x.case3; break; // Showdown
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
                case 0: this->case0 = std::move(x.case0); break; // CommunityCardIs
                case 1: this->case1 = std::move(x.case1); break; // PlayerAction
                case 2: this->case2 = std::move(x.case2); break; // PlayerGotCard
                case 3: this->case3 = std::move(x.case3); break; // Showdown
            }
        } else {
            this->~Union4();
            new (this) Union4{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union4() {
        switch(this->tag){
            case 0: this->case0.~Union4_0(); break; // CommunityCardIs
            case 1: this->case1.~Union4_1(); break; // PlayerAction
            case 2: this->case2.~Union4_2(); break; // PlayerGotCard
            case 3: this->case3.~Union4_3(); break; // Showdown
        }
        this->tag = 255;
    }
};
struct Union7_0 { // None
};
struct Union7_1 { // Some
    Union3 v0;
    __device__ Union7_1(Union3 t0) : v0(t0) {}
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
struct Union6_0 { // ChanceCommunityCard
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union6_0(Union7 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union6_0() = delete;
};
struct Union6_1 { // ChanceInit
};
struct Union6_2 { // Round
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union6_2(Union7 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union6_2() = delete;
};
struct Union6_3 { // RoundWithAction
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    Union1 v6;
    int v3;
    int v5;
    bool v1;
    __device__ Union6_3(Union7 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union6_3() = delete;
};
struct Union6_4 { // TerminalCall
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union6_4(Union7 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union6_4() = delete;
};
struct Union6_5 { // TerminalFold
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union6_5(Union7 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
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
struct Union5_0 { // None
};
struct Union5_1 { // Some
    Union6 v0;
    __device__ Union5_1(Union6 t0) : v0(t0) {}
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
struct Union8_0 { // GameNotStarted
};
struct Union8_1 { // GameOver
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union8_1(Union7 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union8_1() = delete;
};
struct Union8_2 { // WaitingForActionFromPlayerId
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union8_2(Union7 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union8_2() = delete;
};
struct Union8 {
    union {
        Union8_0 case0; // GameNotStarted
        Union8_1 case1; // GameOver
        Union8_2 case2; // WaitingForActionFromPlayerId
    };
    unsigned char tag{255};
    __device__ Union8() {}
    __device__ Union8(Union8_0 t) : tag(0), case0(t) {} // GameNotStarted
    __device__ Union8(Union8_1 t) : tag(1), case1(t) {} // GameOver
    __device__ Union8(Union8_2 t) : tag(2), case2(t) {} // WaitingForActionFromPlayerId
    __device__ Union8(Union8 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(x.case0); break; // GameNotStarted
            case 1: new (&this->case1) Union8_1(x.case1); break; // GameOver
            case 2: new (&this->case2) Union8_2(x.case2); break; // WaitingForActionFromPlayerId
        }
    }
    __device__ Union8(Union8 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(std::move(x.case0)); break; // GameNotStarted
            case 1: new (&this->case1) Union8_1(std::move(x.case1)); break; // GameOver
            case 2: new (&this->case2) Union8_2(std::move(x.case2)); break; // WaitingForActionFromPlayerId
        }
    }
    __device__ Union8 & operator=(Union8 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // GameNotStarted
                case 1: this->case1 = x.case1; break; // GameOver
                case 2: this->case2 = x.case2; break; // WaitingForActionFromPlayerId
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
                case 0: this->case0 = std::move(x.case0); break; // GameNotStarted
                case 1: this->case1 = std::move(x.case1); break; // GameOver
                case 2: this->case2 = std::move(x.case2); break; // WaitingForActionFromPlayerId
            }
        } else {
            this->~Union8();
            new (this) Union8{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union8() {
        switch(this->tag){
            case 0: this->case0.~Union8_0(); break; // GameNotStarted
            case 1: this->case1.~Union8_1(); break; // GameOver
            case 2: this->case2.~Union8_2(); break; // WaitingForActionFromPlayerId
        }
        this->tag = 255;
    }
};
struct Tuple0 {
    static_array_list<Union3,6l> v0;
    static_array_list<Union4,32l> v1;
    Union5 v2;
    static_array<Union2,2l> v3;
    Union8 v4;
    __device__ Tuple0() = default;
    __device__ Tuple0(static_array_list<Union3,6l> t0, static_array_list<Union4,32l> t1, Union5 t2, static_array<Union2,2l> t3, Union8 t4) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4) {}
};
struct Tuple1 {
    Union1 v1;
    int v0;
    __device__ Tuple1() = default;
    __device__ Tuple1(int t0, Union1 t1) : v0(t0), v1(t1) {}
};
struct Tuple2 {
    Union3 v1;
    int v0;
    __device__ Tuple2() = default;
    __device__ Tuple2(int t0, Union3 t1) : v0(t0), v1(t1) {}
};
struct Tuple3 {
    static_array<Union3,2l> v0;
    int v1;
    int v2;
    __device__ Tuple3() = default;
    __device__ Tuple3(static_array<Union3,2l> t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple4 {
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Tuple4() = default;
    __device__ Tuple4(Union7 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Tuple5 {
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<int,2l> v4;
    Union1 v6;
    int v3;
    int v5;
    bool v1;
    __device__ Tuple5() = default;
    __device__ Tuple5(Union7 t0, bool t1, static_array<Union3,2l> t2, int t3, static_array<int,2l> t4, int t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
};
struct Tuple6 {
    Union5 v0;
    static_array<Union2,2l> v1;
    Union8 v2;
    __device__ Tuple6() = default;
    __device__ Tuple6(Union5 t0, static_array<Union2,2l> t1, Union8 t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple7 {
    Union6 v1;
    bool v0;
    __device__ Tuple7() = default;
    __device__ Tuple7(bool t0, Union6 t1) : v0(t0), v1(t1) {}
};
struct Tuple8 {
    int v0;
    int v1;
    __device__ Tuple8() = default;
    __device__ Tuple8(int t0, int t1) : v0(t0), v1(t1) {}
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
__device__ int f_1(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ void f_3(unsigned char * v0){
    return ;
}
__device__ Union1 f_2(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    switch (v1) {
        case 0: {
            f_3(v2);
            return Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v2);
            return Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v2);
            return Union1{Union1_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
__device__ Union2 f_5(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    switch (v1) {
        case 0: {
            f_3(v2);
            return Union2{Union2_0{}};
            break;
        }
        case 1: {
            f_3(v2);
            return Union2{Union2_1{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ static_array<Union2,2l> f_4(unsigned char * v0){
    static_array<Union2,2l> v1;
    int v3;
    v3 = 0l;
    while (while_method_0(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned long long v6;
        v6 = v5 * 4ull;
        unsigned char * v7;
        v7 = (unsigned char *)(v0+v6);
        Union2 v9;
        v9 = f_5(v7);
        v1[v3] = v9;
        v3 += 1l ;
    }
    return v1;
}
__device__ Union0 f_0(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+8ull);
    switch (v1) {
        case 0: {
            Union1 v5;
            v5 = f_2(v2);
            return Union0{Union0_0{v5}};
            break;
        }
        case 1: {
            static_array<Union2,2l> v7;
            v7 = f_4(v2);
            return Union0{Union0_1{v7}};
            break;
        }
        case 2: {
            f_3(v2);
            return Union0{Union0_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ inline bool while_method_1(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ Union3 f_7(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    switch (v1) {
        case 0: {
            f_3(v2);
            return Union3{Union3_0{}};
            break;
        }
        case 1: {
            f_3(v2);
            return Union3{Union3_1{}};
            break;
        }
        case 2: {
            f_3(v2);
            return Union3{Union3_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ int f_8(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+28ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ int f_11(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+4ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ Tuple1 f_10(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    int v4;
    v4 = f_11(v0);
    unsigned char * v5;
    v5 = (unsigned char *)(v0+8ull);
    Union1 v11;
    switch (v4) {
        case 0: {
            f_3(v5);
            v11 = Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v5);
            v11 = Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v5);
            v11 = Union1{Union1_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple1{v3, v11};
}
__device__ Tuple2 f_12(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    int v4;
    v4 = f_11(v0);
    unsigned char * v5;
    v5 = (unsigned char *)(v0+8ull);
    Union3 v11;
    switch (v4) {
        case 0: {
            f_3(v5);
            v11 = Union3{Union3_0{}};
            break;
        }
        case 1: {
            f_3(v5);
            v11 = Union3{Union3_1{}};
            break;
        }
        case 2: {
            f_3(v5);
            v11 = Union3{Union3_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple2{v3, v11};
}
__device__ Tuple3 f_13(unsigned char * v0){
    static_array<Union3,2l> v1;
    int v3;
    v3 = 0l;
    while (while_method_0(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned long long v6;
        v6 = v5 * 4ull;
        unsigned char * v7;
        v7 = (unsigned char *)(v0+v6);
        Union3 v9;
        v9 = f_7(v7);
        v1[v3] = v9;
        v3 += 1l ;
    }
    int * v10;
    v10 = (int *)(v0+8ull);
    int v12;
    v12 = v10[0l];
    int * v13;
    v13 = (int *)(v0+12ull);
    int v15;
    v15 = v13[0l];
    return Tuple3{v1, v12, v15};
}
__device__ Union4 f_9(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            Union3 v5;
            v5 = f_7(v2);
            return Union4{Union4_0{v5}};
            break;
        }
        case 1: {
            int v7; Union1 v8;
            Tuple1 tmp0 = f_10(v2);
            v7 = tmp0.v0; v8 = tmp0.v1;
            return Union4{Union4_1{v7, v8}};
            break;
        }
        case 2: {
            int v10; Union3 v11;
            Tuple2 tmp1 = f_12(v2);
            v10 = tmp1.v0; v11 = tmp1.v1;
            return Union4{Union4_2{v10, v11}};
            break;
        }
        case 3: {
            static_array<Union3,2l> v13; int v14; int v15;
            Tuple3 tmp2 = f_13(v2);
            v13 = tmp2.v0; v14 = tmp2.v1; v15 = tmp2.v2;
            return Union4{Union4_3{v13, v14, v15}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ int f_14(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+1056ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ Tuple4 f_16(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    Union7 v8;
    switch (v1) {
        case 0: {
            f_3(v2);
            v8 = Union7{Union7_0{}};
            break;
        }
        case 1: {
            Union3 v6;
            v6 = f_7(v2);
            v8 = Union7{Union7_1{v6}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    bool * v9;
    v9 = (bool *)(v0+8ull);
    bool v11;
    v11 = v9[0l];
    static_array<Union3,2l> v12;
    int v14;
    v14 = 0l;
    while (while_method_0(v14)){
        unsigned long long v16;
        v16 = (unsigned long long)v14;
        unsigned long long v17;
        v17 = v16 * 4ull;
        unsigned long long v18;
        v18 = 12ull + v17;
        unsigned char * v19;
        v19 = (unsigned char *)(v0+v18);
        Union3 v21;
        v21 = f_7(v19);
        v12[v14] = v21;
        v14 += 1l ;
    }
    int * v22;
    v22 = (int *)(v0+20ull);
    int v24;
    v24 = v22[0l];
    static_array<int,2l> v25;
    int v27;
    v27 = 0l;
    while (while_method_0(v27)){
        unsigned long long v29;
        v29 = (unsigned long long)v27;
        unsigned long long v30;
        v30 = v29 * 4ull;
        unsigned long long v31;
        v31 = 24ull + v30;
        unsigned char * v32;
        v32 = (unsigned char *)(v0+v31);
        int v34;
        v34 = f_1(v32);
        v25[v27] = v34;
        v27 += 1l ;
    }
    int * v35;
    v35 = (int *)(v0+32ull);
    int v37;
    v37 = v35[0l];
    return Tuple4{v8, v11, v12, v24, v25, v37};
}
__device__ int f_18(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+36ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ Tuple5 f_17(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    Union7 v8;
    switch (v1) {
        case 0: {
            f_3(v2);
            v8 = Union7{Union7_0{}};
            break;
        }
        case 1: {
            Union3 v6;
            v6 = f_7(v2);
            v8 = Union7{Union7_1{v6}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    bool * v9;
    v9 = (bool *)(v0+8ull);
    bool v11;
    v11 = v9[0l];
    static_array<Union3,2l> v12;
    int v14;
    v14 = 0l;
    while (while_method_0(v14)){
        unsigned long long v16;
        v16 = (unsigned long long)v14;
        unsigned long long v17;
        v17 = v16 * 4ull;
        unsigned long long v18;
        v18 = 12ull + v17;
        unsigned char * v19;
        v19 = (unsigned char *)(v0+v18);
        Union3 v21;
        v21 = f_7(v19);
        v12[v14] = v21;
        v14 += 1l ;
    }
    int * v22;
    v22 = (int *)(v0+20ull);
    int v24;
    v24 = v22[0l];
    static_array<int,2l> v25;
    int v27;
    v27 = 0l;
    while (while_method_0(v27)){
        unsigned long long v29;
        v29 = (unsigned long long)v27;
        unsigned long long v30;
        v30 = v29 * 4ull;
        unsigned long long v31;
        v31 = 24ull + v30;
        unsigned char * v32;
        v32 = (unsigned char *)(v0+v31);
        int v34;
        v34 = f_1(v32);
        v25[v27] = v34;
        v27 += 1l ;
    }
    int * v35;
    v35 = (int *)(v0+32ull);
    int v37;
    v37 = v35[0l];
    int v38;
    v38 = f_18(v0);
    unsigned char * v39;
    v39 = (unsigned char *)(v0+40ull);
    Union1 v45;
    switch (v38) {
        case 0: {
            f_3(v39);
            v45 = Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v39);
            v45 = Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v39);
            v45 = Union1{Union1_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple5{v8, v11, v12, v24, v25, v37, v45};
}
__device__ Union6 f_15(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            Union7 v5; bool v6; static_array<Union3,2l> v7; int v8; static_array<int,2l> v9; int v10;
            Tuple4 tmp3 = f_16(v2);
            v5 = tmp3.v0; v6 = tmp3.v1; v7 = tmp3.v2; v8 = tmp3.v3; v9 = tmp3.v4; v10 = tmp3.v5;
            return Union6{Union6_0{v5, v6, v7, v8, v9, v10}};
            break;
        }
        case 1: {
            f_3(v2);
            return Union6{Union6_1{}};
            break;
        }
        case 2: {
            Union7 v13; bool v14; static_array<Union3,2l> v15; int v16; static_array<int,2l> v17; int v18;
            Tuple4 tmp4 = f_16(v2);
            v13 = tmp4.v0; v14 = tmp4.v1; v15 = tmp4.v2; v16 = tmp4.v3; v17 = tmp4.v4; v18 = tmp4.v5;
            return Union6{Union6_2{v13, v14, v15, v16, v17, v18}};
            break;
        }
        case 3: {
            Union7 v20; bool v21; static_array<Union3,2l> v22; int v23; static_array<int,2l> v24; int v25; Union1 v26;
            Tuple5 tmp5 = f_17(v2);
            v20 = tmp5.v0; v21 = tmp5.v1; v22 = tmp5.v2; v23 = tmp5.v3; v24 = tmp5.v4; v25 = tmp5.v5; v26 = tmp5.v6;
            return Union6{Union6_3{v20, v21, v22, v23, v24, v25, v26}};
            break;
        }
        case 4: {
            Union7 v28; bool v29; static_array<Union3,2l> v30; int v31; static_array<int,2l> v32; int v33;
            Tuple4 tmp6 = f_16(v2);
            v28 = tmp6.v0; v29 = tmp6.v1; v30 = tmp6.v2; v31 = tmp6.v3; v32 = tmp6.v4; v33 = tmp6.v5;
            return Union6{Union6_4{v28, v29, v30, v31, v32, v33}};
            break;
        }
        case 5: {
            Union7 v35; bool v36; static_array<Union3,2l> v37; int v38; static_array<int,2l> v39; int v40;
            Tuple4 tmp7 = f_16(v2);
            v35 = tmp7.v0; v36 = tmp7.v1; v37 = tmp7.v2; v38 = tmp7.v3; v39 = tmp7.v4; v40 = tmp7.v5;
            return Union6{Union6_5{v35, v36, v37, v38, v39, v40}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ int f_19(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+1144ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ Tuple0 f_6(unsigned char * v0){
    static_array_list<Union3,6l> v1;
    v1 = static_array_list<Union3,6l>{};
    int v3;
    v3 = f_1(v0);
    v1.unsafe_set_length(v3);
    int v4;
    v4 = v1.length;
    int v5;
    v5 = 0l;
    while (while_method_1(v4, v5)){
        unsigned long long v7;
        v7 = (unsigned long long)v5;
        unsigned long long v8;
        v8 = v7 * 4ull;
        unsigned long long v9;
        v9 = 4ull + v8;
        unsigned char * v10;
        v10 = (unsigned char *)(v0+v9);
        Union3 v12;
        v12 = f_7(v10);
        v1[v5] = v12;
        v5 += 1l ;
    }
    static_array_list<Union4,32l> v13;
    v13 = static_array_list<Union4,32l>{};
    int v15;
    v15 = f_8(v0);
    v13.unsafe_set_length(v15);
    int v16;
    v16 = v13.length;
    int v17;
    v17 = 0l;
    while (while_method_1(v16, v17)){
        unsigned long long v19;
        v19 = (unsigned long long)v17;
        unsigned long long v20;
        v20 = v19 * 32ull;
        unsigned long long v21;
        v21 = 32ull + v20;
        unsigned char * v22;
        v22 = (unsigned char *)(v0+v21);
        Union4 v24;
        v24 = f_9(v22);
        v13[v17] = v24;
        v17 += 1l ;
    }
    int v25;
    v25 = f_14(v0);
    unsigned char * v26;
    v26 = (unsigned char *)(v0+1072ull);
    Union5 v32;
    switch (v25) {
        case 0: {
            f_3(v26);
            v32 = Union5{Union5_0{}};
            break;
        }
        case 1: {
            Union6 v30;
            v30 = f_15(v26);
            v32 = Union5{Union5_1{v30}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    static_array<Union2,2l> v33;
    int v35;
    v35 = 0l;
    while (while_method_0(v35)){
        unsigned long long v37;
        v37 = (unsigned long long)v35;
        unsigned long long v38;
        v38 = v37 * 4ull;
        unsigned long long v39;
        v39 = 1136ull + v38;
        unsigned char * v40;
        v40 = (unsigned char *)(v0+v39);
        Union2 v42;
        v42 = f_5(v40);
        v33[v35] = v42;
        v35 += 1l ;
    }
    int v43;
    v43 = f_19(v0);
    unsigned char * v44;
    v44 = (unsigned char *)(v0+1152ull);
    Union8 v62;
    switch (v43) {
        case 0: {
            f_3(v44);
            v62 = Union8{Union8_0{}};
            break;
        }
        case 1: {
            Union7 v48; bool v49; static_array<Union3,2l> v50; int v51; static_array<int,2l> v52; int v53;
            Tuple4 tmp8 = f_16(v44);
            v48 = tmp8.v0; v49 = tmp8.v1; v50 = tmp8.v2; v51 = tmp8.v3; v52 = tmp8.v4; v53 = tmp8.v5;
            v62 = Union8{Union8_1{v48, v49, v50, v51, v52, v53}};
            break;
        }
        case 2: {
            Union7 v55; bool v56; static_array<Union3,2l> v57; int v58; static_array<int,2l> v59; int v60;
            Tuple4 tmp9 = f_16(v44);
            v55 = tmp9.v0; v56 = tmp9.v1; v57 = tmp9.v2; v58 = tmp9.v3; v59 = tmp9.v4; v60 = tmp9.v5;
            v62 = Union8{Union8_2{v55, v56, v57, v58, v59, v60}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple0{v1, v13, v32, v33, v62};
}
__device__ inline bool while_method_2(bool v0, Union6 v1){
    return v0;
}
__device__ unsigned int loop_22(unsigned int v0, curandStatePhilox4_32_10_t & v1){
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
        return loop_22(v0, v1);
    }
}
__device__ int tag_24(Union3 v0){
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
            assert("Invalid tag." && false);
        }
    }
}
__device__ bool is_pair_25(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
__device__ Tuple8 order_26(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple8{v1, v0};
    } else {
        return Tuple8{v0, v1};
    }
}
__device__ Union9 compare_hands_23(Union7 v0, bool v1, static_array<Union3,2l> v2, int v3, static_array<int,2l> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            asm("exit;");
            break;
        }
        case 1: { // Some
            Union3 v7 = v0.case1.v0;
            int v8;
            v8 = tag_24(v7);
            Union3 v9;
            v9 = v2[0l];
            int v11;
            v11 = tag_24(v9);
            Union3 v12;
            v12 = v2[1l];
            int v14;
            v14 = tag_24(v12);
            bool v15;
            v15 = is_pair_25(v8, v11);
            bool v16;
            v16 = is_pair_25(v8, v14);
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
                    Tuple8 tmp19 = order_26(v8, v11);
                    v27 = tmp19.v0; v28 = tmp19.v1;
                    int v29; int v30;
                    Tuple8 tmp20 = order_26(v8, v14);
                    v29 = tmp20.v0; v30 = tmp20.v1;
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
            assert("Invalid tag." && false);
        }
    }
}
__device__ Union6 play_loop_inner_21(static_array_list<Union3,6l> & v0, static_array_list<Union4,32l> & v1, static_array<Union2,2l> v2, Union6 v3){
    static_array_list<Union4,32l> & v4 = v1;
    static_array_list<Union3,6l> & v5 = v0;
    bool v6; Union6 v7;
    Tuple7 tmp11 = Tuple7{true, v3};
    v6 = tmp11.v0; v7 = tmp11.v1;
    while (while_method_2(v6, v7)){
        bool v311; Union6 v312;
        switch (v7.tag) {
            case 0: { // ChanceCommunityCard
                Union7 v268 = v7.case0.v0; bool v269 = v7.case0.v1; static_array<Union3,2l> v270 = v7.case0.v2; int v271 = v7.case0.v3; static_array<int,2l> v272 = v7.case0.v4; int v273 = v7.case0.v5;
                Union3 v274;
                v274 = v5.pop();
                Union4 v275;
                v275 = Union4{Union4_0{v274}};
                v4.push(v275);
                int v276;
                v276 = 2l;
                int v277; int v278;
                Tuple8 tmp12 = Tuple8{0l, 0l};
                v277 = tmp12.v0; v278 = tmp12.v1;
                while (while_method_0(v277)){
                    int v280;
                    v280 = v272[v277];
                    bool v282;
                    v282 = v278 >= v280;
                    int v283;
                    if (v282){
                        v283 = v278;
                    } else {
                        v283 = v280;
                    }
                    v278 = v283;
                    v277 += 1l ;
                }
                static_array<int,2l> v284;
                int v286;
                v286 = 0l;
                while (while_method_0(v286)){
                    v284[v286] = v278;
                    v286 += 1l ;
                }
                Union7 v288;
                v288 = Union7{Union7_1{v274}};
                Union6 v289;
                v289 = Union6{Union6_2{v288, true, v270, 0l, v284, v276}};
                v311 = true; v312 = v289;
                break;
            }
            case 1: { // ChanceInit
                Union3 v290;
                v290 = v5.pop();
                Union3 v291;
                v291 = v5.pop();
                Union4 v292;
                v292 = Union4{Union4_2{0l, v290}};
                v4.push(v292);
                Union4 v293;
                v293 = Union4{Union4_2{1l, v291}};
                v4.push(v293);
                int v294;
                v294 = 2l;
                static_array<int,2l> v295;
                v295[0l] = 1l;
                v295[1l] = 1l;
                static_array<Union3,2l> v297;
                v297[0l] = v290;
                v297[1l] = v291;
                Union7 v299;
                v299 = Union7{Union7_0{}};
                Union6 v300;
                v300 = Union6{Union6_2{v299, true, v297, 0l, v295, v294}};
                v311 = true; v312 = v300;
                break;
            }
            case 2: { // Round
                Union7 v36 = v7.case2.v0; bool v37 = v7.case2.v1; static_array<Union3,2l> v38 = v7.case2.v2; int v39 = v7.case2.v3; static_array<int,2l> v40 = v7.case2.v4; int v41 = v7.case2.v5;
                Union2 v42;
                v42 = v2[v39];
                switch (v42.tag) {
                    case 0: { // Computer
                        static_array_list<Union1,3l> v44;
                        v44 = static_array_list<Union1,3l>{};
                        v44.unsafe_set_length(1l);
                        Union1 v46;
                        v46 = Union1{Union1_0{}};
                        v44[0l] = v46;
                        int v48;
                        v48 = v40[0l];
                        int v50;
                        v50 = v40[1l];
                        bool v52;
                        v52 = v48 == v50;
                        bool v53;
                        v53 = v52 != true;
                        if (v53){
                            Union1 v54;
                            v54 = Union1{Union1_1{}};
                            v44.push(v54);
                        } else {
                        }
                        bool v55;
                        v55 = v41 > 0l;
                        if (v55){
                            Union1 v56;
                            v56 = Union1{Union1_2{}};
                            v44.push(v56);
                        } else {
                        }
                        unsigned long long v57;
                        v57 = clock64();
                        curandStatePhilox4_32_10_t v58;
                        curand_init(v57,0ull,0ull,&v58);
                        int v59;
                        v59 = v44.length;
                        int v60;
                        v60 = v59 - 1l;
                        int v61;
                        v61 = 0l;
                        while (while_method_1(v60, v61)){
                            int v63;
                            v63 = v44.length;
                            int v64;
                            v64 = v63 - v61;
                            unsigned int v65;
                            v65 = (unsigned int)v64;
                            unsigned int v66;
                            v66 = loop_22(v65, v58);
                            unsigned int v67;
                            v67 = (unsigned int)v61;
                            unsigned int v68;
                            v68 = v66 + v67;
                            int v69;
                            v69 = (int)v68;
                            Union1 v70;
                            v70 = v44[v61];
                            Union1 v72;
                            v72 = v44[v69];
                            v44[v61] = v72;
                            v44[v69] = v70;
                            v61 += 1l ;
                        }
                        Union1 v84;
                        v84 = v44.pop();
                        Union4 v85;
                        v85 = Union4{Union4_1{v39, v84}};
                        v4.push(v85);
                        Union6 v169;
                        switch (v36.tag) {
                            case 0: { // None
                                switch (v84.tag) {
                                    case 0: { // Call
                                        if (v37){
                                            bool v134;
                                            v134 = v39 == 0l;
                                            int v135;
                                            if (v134){
                                                v135 = 1l;
                                            } else {
                                                v135 = 0l;
                                            }
                                            v169 = Union6{Union6_2{v36, false, v38, v135, v40, v41}};
                                        } else {
                                            v169 = Union6{Union6_0{v36, v37, v38, v39, v40, v41}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v169 = Union6{Union6_5{v36, v37, v38, v39, v40, v41}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        if (v55){
                                            bool v139;
                                            v139 = v39 == 0l;
                                            int v140;
                                            if (v139){
                                                v140 = 1l;
                                            } else {
                                                v140 = 0l;
                                            }
                                            int v141;
                                            v141 = -1l + v41;
                                            int v142; int v143;
                                            Tuple8 tmp13 = Tuple8{0l, 0l};
                                            v142 = tmp13.v0; v143 = tmp13.v1;
                                            while (while_method_0(v142)){
                                                int v145;
                                                v145 = v40[v142];
                                                bool v147;
                                                v147 = v143 >= v145;
                                                int v148;
                                                if (v147){
                                                    v148 = v143;
                                                } else {
                                                    v148 = v145;
                                                }
                                                v143 = v148;
                                                v142 += 1l ;
                                            }
                                            static_array<int,2l> v149;
                                            int v151;
                                            v151 = 0l;
                                            while (while_method_0(v151)){
                                                v149[v151] = v143;
                                                v151 += 1l ;
                                            }
                                            static_array<int,2l> v153;
                                            int v155;
                                            v155 = 0l;
                                            while (while_method_0(v155)){
                                                int v157;
                                                v157 = v149[v155];
                                                bool v159;
                                                v159 = v155 == v39;
                                                int v161;
                                                if (v159){
                                                    int v160;
                                                    v160 = v157 + 2l;
                                                    v161 = v160;
                                                } else {
                                                    v161 = v157;
                                                }
                                                v153[v155] = v161;
                                                v155 += 1l ;
                                            }
                                            v169 = Union6{Union6_2{v36, false, v38, v140, v153, v141}};
                                        } else {
                                            printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                            asm("exit;");
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                    }
                                }
                                break;
                            }
                            case 1: { // Some
                                Union3 v86 = v36.case1.v0;
                                switch (v84.tag) {
                                    case 0: { // Call
                                        if (v37){
                                            bool v88;
                                            v88 = v39 == 0l;
                                            int v89;
                                            if (v88){
                                                v89 = 1l;
                                            } else {
                                                v89 = 0l;
                                            }
                                            v169 = Union6{Union6_2{v36, false, v38, v89, v40, v41}};
                                        } else {
                                            int v91; int v92;
                                            Tuple8 tmp14 = Tuple8{0l, 0l};
                                            v91 = tmp14.v0; v92 = tmp14.v1;
                                            while (while_method_0(v91)){
                                                int v94;
                                                v94 = v40[v91];
                                                bool v96;
                                                v96 = v92 >= v94;
                                                int v97;
                                                if (v96){
                                                    v97 = v92;
                                                } else {
                                                    v97 = v94;
                                                }
                                                v92 = v97;
                                                v91 += 1l ;
                                            }
                                            static_array<int,2l> v98;
                                            int v100;
                                            v100 = 0l;
                                            while (while_method_0(v100)){
                                                v98[v100] = v92;
                                                v100 += 1l ;
                                            }
                                            v169 = Union6{Union6_4{v36, v37, v38, v39, v98, v41}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v169 = Union6{Union6_5{v36, v37, v38, v39, v40, v41}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        if (v55){
                                            bool v104;
                                            v104 = v39 == 0l;
                                            int v105;
                                            if (v104){
                                                v105 = 1l;
                                            } else {
                                                v105 = 0l;
                                            }
                                            int v106;
                                            v106 = -1l + v41;
                                            int v107; int v108;
                                            Tuple8 tmp15 = Tuple8{0l, 0l};
                                            v107 = tmp15.v0; v108 = tmp15.v1;
                                            while (while_method_0(v107)){
                                                int v110;
                                                v110 = v40[v107];
                                                bool v112;
                                                v112 = v108 >= v110;
                                                int v113;
                                                if (v112){
                                                    v113 = v108;
                                                } else {
                                                    v113 = v110;
                                                }
                                                v108 = v113;
                                                v107 += 1l ;
                                            }
                                            static_array<int,2l> v114;
                                            int v116;
                                            v116 = 0l;
                                            while (while_method_0(v116)){
                                                v114[v116] = v108;
                                                v116 += 1l ;
                                            }
                                            static_array<int,2l> v118;
                                            int v120;
                                            v120 = 0l;
                                            while (while_method_0(v120)){
                                                int v122;
                                                v122 = v114[v120];
                                                bool v124;
                                                v124 = v120 == v39;
                                                int v126;
                                                if (v124){
                                                    int v125;
                                                    v125 = v122 + 4l;
                                                    v126 = v125;
                                                } else {
                                                    v126 = v122;
                                                }
                                                v118[v120] = v126;
                                                v120 += 1l ;
                                            }
                                            v169 = Union6{Union6_2{v36, false, v38, v105, v118, v106}};
                                        } else {
                                            printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                            asm("exit;");
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                    }
                                }
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                            }
                        }
                        v311 = true; v312 = v169;
                        break;
                    }
                    case 1: { // Human
                        v311 = false; v312 = v7;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                break;
            }
            case 3: { // RoundWithAction
                Union7 v174 = v7.case3.v0; bool v175 = v7.case3.v1; static_array<Union3,2l> v176 = v7.case3.v2; int v177 = v7.case3.v3; static_array<int,2l> v178 = v7.case3.v4; int v179 = v7.case3.v5; Union1 v180 = v7.case3.v6;
                Union4 v181;
                v181 = Union4{Union4_1{v177, v180}};
                v4.push(v181);
                Union6 v267;
                switch (v174.tag) {
                    case 0: { // None
                        switch (v180.tag) {
                            case 0: { // Call
                                if (v175){
                                    bool v231;
                                    v231 = v177 == 0l;
                                    int v232;
                                    if (v231){
                                        v232 = 1l;
                                    } else {
                                        v232 = 0l;
                                    }
                                    v267 = Union6{Union6_2{v174, false, v176, v232, v178, v179}};
                                } else {
                                    v267 = Union6{Union6_0{v174, v175, v176, v177, v178, v179}};
                                }
                                break;
                            }
                            case 1: { // Fold
                                v267 = Union6{Union6_5{v174, v175, v176, v177, v178, v179}};
                                break;
                            }
                            case 2: { // Raise
                                bool v236;
                                v236 = v179 > 0l;
                                if (v236){
                                    bool v237;
                                    v237 = v177 == 0l;
                                    int v238;
                                    if (v237){
                                        v238 = 1l;
                                    } else {
                                        v238 = 0l;
                                    }
                                    int v239;
                                    v239 = -1l + v179;
                                    int v240; int v241;
                                    Tuple8 tmp16 = Tuple8{0l, 0l};
                                    v240 = tmp16.v0; v241 = tmp16.v1;
                                    while (while_method_0(v240)){
                                        int v243;
                                        v243 = v178[v240];
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
                                    while (while_method_0(v249)){
                                        v247[v249] = v241;
                                        v249 += 1l ;
                                    }
                                    static_array<int,2l> v251;
                                    int v253;
                                    v253 = 0l;
                                    while (while_method_0(v253)){
                                        int v255;
                                        v255 = v247[v253];
                                        bool v257;
                                        v257 = v253 == v177;
                                        int v259;
                                        if (v257){
                                            int v258;
                                            v258 = v255 + 2l;
                                            v259 = v258;
                                        } else {
                                            v259 = v255;
                                        }
                                        v251[v253] = v259;
                                        v253 += 1l ;
                                    }
                                    v267 = Union6{Union6_2{v174, false, v176, v238, v251, v239}};
                                } else {
                                    printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                    asm("exit;");
                                }
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                            }
                        }
                        break;
                    }
                    case 1: { // Some
                        Union3 v182 = v174.case1.v0;
                        switch (v180.tag) {
                            case 0: { // Call
                                if (v175){
                                    bool v184;
                                    v184 = v177 == 0l;
                                    int v185;
                                    if (v184){
                                        v185 = 1l;
                                    } else {
                                        v185 = 0l;
                                    }
                                    v267 = Union6{Union6_2{v174, false, v176, v185, v178, v179}};
                                } else {
                                    int v187; int v188;
                                    Tuple8 tmp17 = Tuple8{0l, 0l};
                                    v187 = tmp17.v0; v188 = tmp17.v1;
                                    while (while_method_0(v187)){
                                        int v190;
                                        v190 = v178[v187];
                                        bool v192;
                                        v192 = v188 >= v190;
                                        int v193;
                                        if (v192){
                                            v193 = v188;
                                        } else {
                                            v193 = v190;
                                        }
                                        v188 = v193;
                                        v187 += 1l ;
                                    }
                                    static_array<int,2l> v194;
                                    int v196;
                                    v196 = 0l;
                                    while (while_method_0(v196)){
                                        v194[v196] = v188;
                                        v196 += 1l ;
                                    }
                                    v267 = Union6{Union6_4{v174, v175, v176, v177, v194, v179}};
                                }
                                break;
                            }
                            case 1: { // Fold
                                v267 = Union6{Union6_5{v174, v175, v176, v177, v178, v179}};
                                break;
                            }
                            case 2: { // Raise
                                bool v200;
                                v200 = v179 > 0l;
                                if (v200){
                                    bool v201;
                                    v201 = v177 == 0l;
                                    int v202;
                                    if (v201){
                                        v202 = 1l;
                                    } else {
                                        v202 = 0l;
                                    }
                                    int v203;
                                    v203 = -1l + v179;
                                    int v204; int v205;
                                    Tuple8 tmp18 = Tuple8{0l, 0l};
                                    v204 = tmp18.v0; v205 = tmp18.v1;
                                    while (while_method_0(v204)){
                                        int v207;
                                        v207 = v178[v204];
                                        bool v209;
                                        v209 = v205 >= v207;
                                        int v210;
                                        if (v209){
                                            v210 = v205;
                                        } else {
                                            v210 = v207;
                                        }
                                        v205 = v210;
                                        v204 += 1l ;
                                    }
                                    static_array<int,2l> v211;
                                    int v213;
                                    v213 = 0l;
                                    while (while_method_0(v213)){
                                        v211[v213] = v205;
                                        v213 += 1l ;
                                    }
                                    static_array<int,2l> v215;
                                    int v217;
                                    v217 = 0l;
                                    while (while_method_0(v217)){
                                        int v219;
                                        v219 = v211[v217];
                                        bool v221;
                                        v221 = v217 == v177;
                                        int v223;
                                        if (v221){
                                            int v222;
                                            v222 = v219 + 4l;
                                            v223 = v222;
                                        } else {
                                            v223 = v219;
                                        }
                                        v215[v217] = v223;
                                        v217 += 1l ;
                                    }
                                    v267 = Union6{Union6_2{v174, false, v176, v202, v215, v203}};
                                } else {
                                    printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                    asm("exit;");
                                }
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                            }
                        }
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                v311 = true; v312 = v267;
                break;
            }
            case 4: { // TerminalCall
                Union7 v20 = v7.case4.v0; bool v21 = v7.case4.v1; static_array<Union3,2l> v22 = v7.case4.v2; int v23 = v7.case4.v3; static_array<int,2l> v24 = v7.case4.v4; int v25 = v7.case4.v5;
                int v26;
                v26 = v24[v23];
                Union9 v28;
                v28 = compare_hands_23(v20, v21, v22, v23, v24, v25);
                int v33; int v34;
                switch (v28.tag) {
                    case 0: { // Eq
                        v33 = 0l; v34 = -1l;
                        break;
                    }
                    case 1: { // Gt
                        v33 = v26; v34 = 0l;
                        break;
                    }
                    case 2: { // Lt
                        v33 = v26; v34 = 1l;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                Union4 v35;
                v35 = Union4{Union4_3{v22, v33, v34}};
                v4.push(v35);
                v311 = false; v312 = v7;
                break;
            }
            case 5: { // TerminalFold
                Union7 v9 = v7.case5.v0; bool v10 = v7.case5.v1; static_array<Union3,2l> v11 = v7.case5.v2; int v12 = v7.case5.v3; static_array<int,2l> v13 = v7.case5.v4; int v14 = v7.case5.v5;
                int v15;
                v15 = v13[v12];
                bool v17;
                v17 = v12 == 0l;
                int v18;
                if (v17){
                    v18 = 1l;
                } else {
                    v18 = 0l;
                }
                Union4 v19;
                v19 = Union4{Union4_3{v11, v15, v18}};
                v4.push(v19);
                v311 = false; v312 = v7;
                break;
            }
            default: {
                assert("Invalid tag." && false);
            }
        }
        v6 = v311;
        v7 = v312;
    }
    return v7;
}
__device__ Tuple6 play_loop_20(Union5 v0, static_array<Union2,2l> v1, Union8 v2, static_array_list<Union3,6l> & v3, static_array_list<Union4,32l> & v4, Union6 v5){
    Union6 v6;
    v6 = play_loop_inner_21(v3, v4, v1, v5);
    switch (v6.tag) {
        case 2: { // Round
            Union7 v7 = v6.case2.v0; bool v8 = v6.case2.v1; static_array<Union3,2l> v9 = v6.case2.v2; int v10 = v6.case2.v3; static_array<int,2l> v11 = v6.case2.v4; int v12 = v6.case2.v5;
            Union5 v13;
            v13 = Union5{Union5_1{v6}};
            Union8 v14;
            v14 = Union8{Union8_2{v7, v8, v9, v10, v11, v12}};
            return Tuple6{v13, v1, v14};
            break;
        }
        case 4: { // TerminalCall
            Union7 v15 = v6.case4.v0; bool v16 = v6.case4.v1; static_array<Union3,2l> v17 = v6.case4.v2; int v18 = v6.case4.v3; static_array<int,2l> v19 = v6.case4.v4; int v20 = v6.case4.v5;
            Union5 v21;
            v21 = Union5{Union5_0{}};
            Union8 v22;
            v22 = Union8{Union8_1{v15, v16, v17, v18, v19, v20}};
            return Tuple6{v21, v1, v22};
            break;
        }
        case 5: { // TerminalFold
            Union7 v23 = v6.case5.v0; bool v24 = v6.case5.v1; static_array<Union3,2l> v25 = v6.case5.v2; int v26 = v6.case5.v3; static_array<int,2l> v27 = v6.case5.v4; int v28 = v6.case5.v5;
            Union5 v29;
            v29 = Union5{Union5_0{}};
            Union8 v30;
            v30 = Union8{Union8_1{v23, v24, v25, v26, v27, v28}};
            return Tuple6{v29, v1, v30};
            break;
        }
        default: {
            printf("%s\n", "Unexpected node received in play_loop.");
            asm("exit;");
        }
    }
}
__device__ void f_28(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+0ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_30(unsigned char * v0){
    return ;
}
__device__ void f_29(unsigned char * v0, Union3 v1){
    int v2;
    v2 = v1.tag;
    f_28(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // Jack
            return f_30(v3);
            break;
        }
        case 1: { // King
            return f_30(v3);
            break;
        }
        case 2: { // Queen
            return f_30(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_31(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+28ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_34(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+4ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_33(unsigned char * v0, int v1, Union1 v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0l] = v1;
    int v5;
    v5 = v2.tag;
    f_34(v0, v5);
    unsigned char * v6;
    v6 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // Call
            return f_30(v6);
            break;
        }
        case 1: { // Fold
            return f_30(v6);
            break;
        }
        case 2: { // Raise
            return f_30(v6);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_35(unsigned char * v0, int v1, Union3 v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0l] = v1;
    int v5;
    v5 = v2.tag;
    f_34(v0, v5);
    unsigned char * v6;
    v6 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // Jack
            return f_30(v6);
            break;
        }
        case 1: { // King
            return f_30(v6);
            break;
        }
        case 2: { // Queen
            return f_30(v6);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_36(unsigned char * v0, static_array<Union3,2l> v1, int v2, int v3){
    int v4;
    v4 = 0l;
    while (while_method_0(v4)){
        unsigned long long v6;
        v6 = (unsigned long long)v4;
        unsigned long long v7;
        v7 = v6 * 4ull;
        unsigned char * v8;
        v8 = (unsigned char *)(v0+v7);
        Union3 v10;
        v10 = v1[v4];
        f_29(v8, v10);
        v4 += 1l ;
    }
    int * v12;
    v12 = (int *)(v0+8ull);
    v12[0l] = v2;
    int * v14;
    v14 = (int *)(v0+12ull);
    v14[0l] = v3;
    return ;
}
__device__ void f_32(unsigned char * v0, Union4 v1){
    int v2;
    v2 = v1.tag;
    f_28(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // CommunityCardIs
            Union3 v5 = v1.case0.v0;
            return f_29(v3, v5);
            break;
        }
        case 1: { // PlayerAction
            int v6 = v1.case1.v0; Union1 v7 = v1.case1.v1;
            return f_33(v3, v6, v7);
            break;
        }
        case 2: { // PlayerGotCard
            int v8 = v1.case2.v0; Union3 v9 = v1.case2.v1;
            return f_35(v3, v8, v9);
            break;
        }
        case 3: { // Showdown
            static_array<Union3,2l> v10 = v1.case3.v0; int v11 = v1.case3.v1; int v12 = v1.case3.v2;
            return f_36(v3, v10, v11, v12);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_37(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+1056ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_39(unsigned char * v0, Union7 v1, bool v2, static_array<Union3,2l> v3, int v4, static_array<int,2l> v5, int v6){
    int v7;
    v7 = v1.tag;
    f_28(v0, v7);
    unsigned char * v8;
    v8 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // None
            f_30(v8);
            break;
        }
        case 1: { // Some
            Union3 v10 = v1.case1.v0;
            f_29(v8, v10);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    bool * v11;
    v11 = (bool *)(v0+8ull);
    v11[0l] = v2;
    int v13;
    v13 = 0l;
    while (while_method_0(v13)){
        unsigned long long v15;
        v15 = (unsigned long long)v13;
        unsigned long long v16;
        v16 = v15 * 4ull;
        unsigned long long v17;
        v17 = 12ull + v16;
        unsigned char * v18;
        v18 = (unsigned char *)(v0+v17);
        Union3 v20;
        v20 = v3[v13];
        f_29(v18, v20);
        v13 += 1l ;
    }
    int * v22;
    v22 = (int *)(v0+20ull);
    v22[0l] = v4;
    int v24;
    v24 = 0l;
    while (while_method_0(v24)){
        unsigned long long v26;
        v26 = (unsigned long long)v24;
        unsigned long long v27;
        v27 = v26 * 4ull;
        unsigned long long v28;
        v28 = 24ull + v27;
        unsigned char * v29;
        v29 = (unsigned char *)(v0+v28);
        int v31;
        v31 = v5[v24];
        f_28(v29, v31);
        v24 += 1l ;
    }
    int * v33;
    v33 = (int *)(v0+32ull);
    v33[0l] = v6;
    return ;
}
__device__ void f_41(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+36ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_40(unsigned char * v0, Union7 v1, bool v2, static_array<Union3,2l> v3, int v4, static_array<int,2l> v5, int v6, Union1 v7){
    int v8;
    v8 = v1.tag;
    f_28(v0, v8);
    unsigned char * v9;
    v9 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // None
            f_30(v9);
            break;
        }
        case 1: { // Some
            Union3 v11 = v1.case1.v0;
            f_29(v9, v11);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    bool * v12;
    v12 = (bool *)(v0+8ull);
    v12[0l] = v2;
    int v14;
    v14 = 0l;
    while (while_method_0(v14)){
        unsigned long long v16;
        v16 = (unsigned long long)v14;
        unsigned long long v17;
        v17 = v16 * 4ull;
        unsigned long long v18;
        v18 = 12ull + v17;
        unsigned char * v19;
        v19 = (unsigned char *)(v0+v18);
        Union3 v21;
        v21 = v3[v14];
        f_29(v19, v21);
        v14 += 1l ;
    }
    int * v23;
    v23 = (int *)(v0+20ull);
    v23[0l] = v4;
    int v25;
    v25 = 0l;
    while (while_method_0(v25)){
        unsigned long long v27;
        v27 = (unsigned long long)v25;
        unsigned long long v28;
        v28 = v27 * 4ull;
        unsigned long long v29;
        v29 = 24ull + v28;
        unsigned char * v30;
        v30 = (unsigned char *)(v0+v29);
        int v32;
        v32 = v5[v25];
        f_28(v30, v32);
        v25 += 1l ;
    }
    int * v34;
    v34 = (int *)(v0+32ull);
    v34[0l] = v6;
    int v36;
    v36 = v7.tag;
    f_41(v0, v36);
    unsigned char * v37;
    v37 = (unsigned char *)(v0+40ull);
    switch (v7.tag) {
        case 0: { // Call
            return f_30(v37);
            break;
        }
        case 1: { // Fold
            return f_30(v37);
            break;
        }
        case 2: { // Raise
            return f_30(v37);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_38(unsigned char * v0, Union6 v1){
    int v2;
    v2 = v1.tag;
    f_28(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // ChanceCommunityCard
            Union7 v5 = v1.case0.v0; bool v6 = v1.case0.v1; static_array<Union3,2l> v7 = v1.case0.v2; int v8 = v1.case0.v3; static_array<int,2l> v9 = v1.case0.v4; int v10 = v1.case0.v5;
            return f_39(v3, v5, v6, v7, v8, v9, v10);
            break;
        }
        case 1: { // ChanceInit
            return f_30(v3);
            break;
        }
        case 2: { // Round
            Union7 v11 = v1.case2.v0; bool v12 = v1.case2.v1; static_array<Union3,2l> v13 = v1.case2.v2; int v14 = v1.case2.v3; static_array<int,2l> v15 = v1.case2.v4; int v16 = v1.case2.v5;
            return f_39(v3, v11, v12, v13, v14, v15, v16);
            break;
        }
        case 3: { // RoundWithAction
            Union7 v17 = v1.case3.v0; bool v18 = v1.case3.v1; static_array<Union3,2l> v19 = v1.case3.v2; int v20 = v1.case3.v3; static_array<int,2l> v21 = v1.case3.v4; int v22 = v1.case3.v5; Union1 v23 = v1.case3.v6;
            return f_40(v3, v17, v18, v19, v20, v21, v22, v23);
            break;
        }
        case 4: { // TerminalCall
            Union7 v24 = v1.case4.v0; bool v25 = v1.case4.v1; static_array<Union3,2l> v26 = v1.case4.v2; int v27 = v1.case4.v3; static_array<int,2l> v28 = v1.case4.v4; int v29 = v1.case4.v5;
            return f_39(v3, v24, v25, v26, v27, v28, v29);
            break;
        }
        case 5: { // TerminalFold
            Union7 v30 = v1.case5.v0; bool v31 = v1.case5.v1; static_array<Union3,2l> v32 = v1.case5.v2; int v33 = v1.case5.v3; static_array<int,2l> v34 = v1.case5.v4; int v35 = v1.case5.v5;
            return f_39(v3, v30, v31, v32, v33, v34, v35);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_42(unsigned char * v0, Union2 v1){
    int v2;
    v2 = v1.tag;
    f_28(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // Computer
            return f_30(v3);
            break;
        }
        case 1: { // Human
            return f_30(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_43(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+1144ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_27(unsigned char * v0, static_array_list<Union3,6l> v1, static_array_list<Union4,32l> v2, Union5 v3, static_array<Union2,2l> v4, Union8 v5){
    int v6;
    v6 = v1.length;
    f_28(v0, v6);
    int v7;
    v7 = v1.length;
    int v8;
    v8 = 0l;
    while (while_method_1(v7, v8)){
        unsigned long long v10;
        v10 = (unsigned long long)v8;
        unsigned long long v11;
        v11 = v10 * 4ull;
        unsigned long long v12;
        v12 = 4ull + v11;
        unsigned char * v13;
        v13 = (unsigned char *)(v0+v12);
        Union3 v15;
        v15 = v1[v8];
        f_29(v13, v15);
        v8 += 1l ;
    }
    int v17;
    v17 = v2.length;
    f_31(v0, v17);
    int v18;
    v18 = v2.length;
    int v19;
    v19 = 0l;
    while (while_method_1(v18, v19)){
        unsigned long long v21;
        v21 = (unsigned long long)v19;
        unsigned long long v22;
        v22 = v21 * 32ull;
        unsigned long long v23;
        v23 = 32ull + v22;
        unsigned char * v24;
        v24 = (unsigned char *)(v0+v23);
        Union4 v26;
        v26 = v2[v19];
        f_32(v24, v26);
        v19 += 1l ;
    }
    int v28;
    v28 = v3.tag;
    f_37(v0, v28);
    unsigned char * v29;
    v29 = (unsigned char *)(v0+1072ull);
    switch (v3.tag) {
        case 0: { // None
            f_30(v29);
            break;
        }
        case 1: { // Some
            Union6 v31 = v3.case1.v0;
            f_38(v29, v31);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    int v32;
    v32 = 0l;
    while (while_method_0(v32)){
        unsigned long long v34;
        v34 = (unsigned long long)v32;
        unsigned long long v35;
        v35 = v34 * 4ull;
        unsigned long long v36;
        v36 = 1136ull + v35;
        unsigned char * v37;
        v37 = (unsigned char *)(v0+v36);
        Union2 v39;
        v39 = v4[v32];
        f_42(v37, v39);
        v32 += 1l ;
    }
    int v41;
    v41 = v5.tag;
    f_43(v0, v41);
    unsigned char * v42;
    v42 = (unsigned char *)(v0+1152ull);
    switch (v5.tag) {
        case 0: { // GameNotStarted
            return f_30(v42);
            break;
        }
        case 1: { // GameOver
            Union7 v44 = v5.case1.v0; bool v45 = v5.case1.v1; static_array<Union3,2l> v46 = v5.case1.v2; int v47 = v5.case1.v3; static_array<int,2l> v48 = v5.case1.v4; int v49 = v5.case1.v5;
            return f_39(v42, v44, v45, v46, v47, v48, v49);
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            Union7 v50 = v5.case2.v0; bool v51 = v5.case2.v1; static_array<Union3,2l> v52 = v5.case2.v2; int v53 = v5.case2.v3; static_array<int,2l> v54 = v5.case2.v4; int v55 = v5.case2.v5;
            return f_39(v42, v50, v51, v52, v53, v54, v55);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_45(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+1048ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_44(unsigned char * v0, static_array_list<Union4,32l> v1, static_array<Union2,2l> v2, Union8 v3){
    int v4;
    v4 = v1.length;
    f_28(v0, v4);
    int v5;
    v5 = v1.length;
    int v6;
    v6 = 0l;
    while (while_method_1(v5, v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = v8 * 32ull;
        unsigned long long v10;
        v10 = 16ull + v9;
        unsigned char * v11;
        v11 = (unsigned char *)(v0+v10);
        Union4 v13;
        v13 = v1[v6];
        f_32(v11, v13);
        v6 += 1l ;
    }
    int v15;
    v15 = 0l;
    while (while_method_0(v15)){
        unsigned long long v17;
        v17 = (unsigned long long)v15;
        unsigned long long v18;
        v18 = v17 * 4ull;
        unsigned long long v19;
        v19 = 1040ull + v18;
        unsigned char * v20;
        v20 = (unsigned char *)(v0+v19);
        Union2 v22;
        v22 = v2[v15];
        f_42(v20, v22);
        v15 += 1l ;
    }
    int v24;
    v24 = v3.tag;
    f_45(v0, v24);
    unsigned char * v25;
    v25 = (unsigned char *)(v0+1056ull);
    switch (v3.tag) {
        case 0: { // GameNotStarted
            return f_30(v25);
            break;
        }
        case 1: { // GameOver
            Union7 v27 = v3.case1.v0; bool v28 = v3.case1.v1; static_array<Union3,2l> v29 = v3.case1.v2; int v30 = v3.case1.v3; static_array<int,2l> v31 = v3.case1.v4; int v32 = v3.case1.v5;
            return f_39(v25, v27, v28, v29, v30, v31, v32);
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            Union7 v33 = v3.case2.v0; bool v34 = v3.case2.v1; static_array<Union3,2l> v35 = v3.case2.v2; int v36 = v3.case2.v3; static_array<int,2l> v37 = v3.case2.v4; int v38 = v3.case2.v5;
            return f_39(v25, v33, v34, v35, v36, v37, v38);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1, unsigned char * v2) {
    int v3;
    v3 = threadIdx.x;
    int v4;
    v4 = blockIdx.x;
    int v5;
    v5 = v4 * 32l;
    int v6;
    v6 = v3 + v5;
    bool v7;
    v7 = v6 == 0l;
    if (v7){
        Union0 v8;
        v8 = f_0(v1);
        static_array_list<Union3,6l> v9; static_array_list<Union4,32l> v10; Union5 v11; static_array<Union2,2l> v12; Union8 v13;
        Tuple0 tmp10 = f_6(v0);
        v9 = tmp10.v0; v10 = tmp10.v1; v11 = tmp10.v2; v12 = tmp10.v3; v13 = tmp10.v4;
        static_array_list<Union3,6l> & v14 = v9;
        static_array_list<Union4,32l> & v15 = v10;
        Union5 v102; static_array<Union2,2l> v103; Union8 v104;
        switch (v8.tag) {
            case 0: { // ActionSelected
                Union1 v72 = v8.case0.v0;
                switch (v11.tag) {
                    case 0: { // None
                        v102 = v11; v103 = v12; v104 = v13;
                        break;
                    }
                    case 1: { // Some
                        Union6 v73 = v11.case1.v0;
                        switch (v73.tag) {
                            case 2: { // Round
                                Union7 v74 = v73.case2.v0; bool v75 = v73.case2.v1; static_array<Union3,2l> v76 = v73.case2.v2; int v77 = v73.case2.v3; static_array<int,2l> v78 = v73.case2.v4; int v79 = v73.case2.v5;
                                Union6 v80;
                                v80 = Union6{Union6_3{v74, v75, v76, v77, v78, v79, v72}};
                                Tuple6 tmp21 = play_loop_20(v11, v12, v13, v14, v15, v80);
                                v102 = tmp21.v0; v103 = tmp21.v1; v104 = tmp21.v2;
                                break;
                            }
                            default: {
                                printf("%s\n", "Unexpected game node in ActionSelected.");
                                asm("exit;");
                            }
                        }
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                break;
            }
            case 1: { // PlayerChanged
                static_array<Union2,2l> v71 = v8.case1.v0;
                v102 = v11; v103 = v71; v104 = v13;
                break;
            }
            case 2: { // StartGame
                static_array<Union2,2l> v16;
                Union2 v18;
                v18 = Union2{Union2_0{}};
                v16[0l] = v18;
                Union2 v20;
                v20 = Union2{Union2_1{}};
                v16[1l] = v20;
                static_array_list<Union3,6l> v22;
                v22 = static_array_list<Union3,6l>{};
                v22.unsafe_set_length(6l);
                Union3 v24;
                v24 = Union3{Union3_1{}};
                v22[0l] = v24;
                Union3 v26;
                v26 = Union3{Union3_1{}};
                v22[1l] = v26;
                Union3 v28;
                v28 = Union3{Union3_2{}};
                v22[2l] = v28;
                Union3 v30;
                v30 = Union3{Union3_2{}};
                v22[3l] = v30;
                Union3 v32;
                v32 = Union3{Union3_0{}};
                v22[4l] = v32;
                Union3 v34;
                v34 = Union3{Union3_0{}};
                v22[5l] = v34;
                unsigned long long v36;
                v36 = clock64();
                curandStatePhilox4_32_10_t v37;
                curand_init(v36,0ull,0ull,&v37);
                int v38;
                v38 = v22.length;
                int v39;
                v39 = v38 - 1l;
                int v40;
                v40 = 0l;
                while (while_method_1(v39, v40)){
                    int v42;
                    v42 = v22.length;
                    int v43;
                    v43 = v42 - v40;
                    unsigned int v44;
                    v44 = (unsigned int)v43;
                    unsigned int v45;
                    v45 = loop_22(v44, v37);
                    unsigned int v46;
                    v46 = (unsigned int)v40;
                    unsigned int v47;
                    v47 = v45 + v46;
                    int v48;
                    v48 = (int)v47;
                    Union3 v49;
                    v49 = v22[v40];
                    Union3 v51;
                    v51 = v22[v48];
                    v22[v40] = v51;
                    v22[v48] = v49;
                    v40 += 1l ;
                }
                static_array_list<Union4,32l> v63;
                v63 = static_array_list<Union4,32l>{};
                v14 = v22;
                v15 = v63;
                Union5 v65;
                v65 = Union5{Union5_0{}};
                Union8 v66;
                v66 = Union8{Union8_0{}};
                Union6 v67;
                v67 = Union6{Union6_1{}};
                Tuple6 tmp22 = play_loop_20(v65, v16, v66, v14, v15, v67);
                v102 = tmp22.v0; v103 = tmp22.v1; v104 = tmp22.v2;
                break;
            }
            default: {
                assert("Invalid tag." && false);
            }
        }
        f_27(v0, v9, v10, v102, v103, v104);
        return f_44(v2, v10, v103, v104);
    } else {
        return ;
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

options = []
options.append('--diag-suppress=550,20012,68')
options.append('--dopt=on')
options.append('--restrict')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
import random
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
US0 = Union[US0_0, US0_1, US0_2]
class US2_0(NamedTuple): # Computer
    tag = 0
class US2_1(NamedTuple): # Human
    tag = 1
US2 = Union[US2_0, US2_1]
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
def Closure0():
    def inner(v0 : object, v1 : object) -> object:
        v2 = cp.empty(16,dtype=cp.uint8)
        v3 = cp.empty(1200,dtype=cp.uint8)
        v4 = cp.empty(1104,dtype=cp.uint8)
        v5 = method0(v0)
        v6, v7, v8, v9, v10 = method7(v1)
        method28(v2, v5)
        del v5
        method35(v3, v6, v7, v8, v9, v10)
        del v6, v7, v8, v9, v10
        v11 = 0
        v12 = raw_module.get_function(f"entry{v11}")
        del v11
        v12.max_dynamic_shared_size_bytes = 0 
        v12((1,),(32,),(v3, v2, v4),shared_mem=0)
        del v2, v12
        v13, v14, v15, v16, v17 = method49(v3)
        del v3
        v18, v19, v20 = method66(v4)
        del v4
        return method68(v13, v14, v15, v16, v17, v18, v19, v20)
    return inner
def Closure1():
    def inner() -> object:
        v1 = static_array(2)
        v3 = US2_0()
        v1[0] = v3
        del v3
        v5 = US2_1()
        v1[1] = v5
        del v5
        v7 = static_array_list(6)
        v7.unsafe_set_length(6)
        v9 = US6_1()
        v7[0] = v9
        del v9
        v11 = US6_1()
        v7[1] = v11
        del v11
        v13 = US6_2()
        v7[2] = v13
        del v13
        v15 = US6_2()
        v7[3] = v15
        del v15
        v17 = US6_0()
        v7[4] = v17
        del v17
        v19 = US6_0()
        v7[5] = v19
        del v19
        v37 = v7.length
        v38 = v37 - 1
        del v37
        v39 = 0
        while method5(v38, v39):
            v41 = v7.length
            v42 = random.randrange(v39, v41)
            del v41
            v44 = v7[v39]
            v46 = v7[v42]
            v7[v39] = v46
            del v46
            v7[v42] = v44
            del v42, v44
            v39 += 1 
        del v38, v39
        v48 = static_array_list(32)
        v49 = US3_0()
        v50 = US7_0()
        return method95(v7, v48, v49, v1, v50)
    return inner
def method3(v0 : object) -> None:
    assert v0 == [], f'Expected an unit type. Got: {v0}'
    del v0
    return 
def method2(v0 : object) -> US1:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "Call" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US1_0()
    else:
        del v4
        v7 = "Fold" == v1
        if v7:
            del v1, v7
            method3(v2)
            del v2
            return US1_1()
        else:
            del v7
            v10 = "Raise" == v1
            if v10:
                del v1, v10
                method3(v2)
                del v2
                return US1_2()
            else:
                del v2, v10
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
    v4 = "Computer" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US2_0()
    else:
        del v4
        v7 = "Human" == v1
        if v7:
            del v1, v7
            method3(v2)
            del v2
            return US2_1()
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
    v4 = "ActionSelected" == v1
    if v4:
        del v1, v4
        v5 = method2(v2)
        del v2
        return US0_0(v5)
    else:
        del v4
        v8 = "PlayerChanged" == v1
        if v8:
            del v1, v8
            v9 = method4(v2)
            del v2
            return US0_1(v9)
        else:
            del v8
            v12 = "StartGame" == v1
            if v12:
                del v1, v12
                method3(v2)
                del v2
                return US0_2()
            else:
                del v2, v12
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method0(v0 : object) -> US0:
    return method1(v0)
def method11(v0 : object) -> US6:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "Jack" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US6_0()
    else:
        del v4
        v7 = "King" == v1
        if v7:
            del v1, v7
            method3(v2)
            del v2
            return US6_1()
        else:
            del v7
            v10 = "Queen" == v1
            if v10:
                del v1, v10
                method3(v2)
                del v2
                return US6_2()
            else:
                del v2, v10
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method10(v0 : object) -> static_array_list:
    v1 = len(v0) # type: ignore
    assert (6 >= v1), f'The length of the original object has to be greater than or equal to the static array dimension.\nExpected: 6\nGot: {v1} '
    del v1
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v2 = len(v0) # type: ignore
    v3 = 6 >= v2
    v4 = v3 == False
    if v4:
        v5 = "The type level dimension has to equal the value passed at runtime into create."
        assert v3, v5
        del v5
    else:
        pass
    del v3, v4
    v7 = static_array_list(6)
    v7.unsafe_set_length(v2)
    v8 = 0
    while method5(v2, v8):
        v10 = v0[v8]
        v11 = method11(v10)
        del v10
        v7[v8] = v11
        del v11
        v8 += 1 
    del v0, v2, v8
    return v7
def method15(v0 : object) -> i32:
    assert isinstance(v0,i32), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method14(v0 : object) -> Tuple[i32, US1]:
    v1 = v0[0] # type: ignore
    v2 = method15(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method2(v3)
    del v3
    return v2, v4
def method16(v0 : object) -> Tuple[i32, US6]:
    v1 = v0[0] # type: ignore
    v2 = method15(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method11(v3)
    del v3
    return v2, v4
def method18(v0 : object) -> static_array:
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
        v10 = method11(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method17(v0 : object) -> Tuple[static_array, i32, i32]:
    v1 = v0["cards_shown"] # type: ignore
    v2 = method18(v1)
    del v1
    v3 = v0["chips_won"] # type: ignore
    v4 = method15(v3)
    del v3
    v5 = v0["winner_id"] # type: ignore
    del v0
    v6 = method15(v5)
    del v5
    return v2, v4, v6
def method13(v0 : object) -> US8:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "CommunityCardIs" == v1
    if v4:
        del v1, v4
        v5 = method11(v2)
        del v2
        return US8_0(v5)
    else:
        del v4
        v8 = "PlayerAction" == v1
        if v8:
            del v1, v8
            v9, v10 = method14(v2)
            del v2
            return US8_1(v9, v10)
        else:
            del v8
            v13 = "PlayerGotCard" == v1
            if v13:
                del v1, v13
                v14, v15 = method16(v2)
                del v2
                return US8_2(v14, v15)
            else:
                del v13
                v18 = "Showdown" == v1
                if v18:
                    del v1, v18
                    v19, v20, v21 = method17(v2)
                    del v2
                    return US8_3(v19, v20, v21)
                else:
                    del v2, v18
                    raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                    del v1
                    raise Exception("Error")
def method12(v0 : object) -> static_array_list:
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
        v11 = method13(v10)
        del v10
        v7[v8] = v11
        del v11
        v8 += 1 
    del v0, v2, v8
    return v7
def method9(v0 : object) -> Tuple[static_array_list, static_array_list]:
    v1 = v0["deck"] # type: ignore
    v2 = method10(v1)
    del v1
    v3 = v0["messages"] # type: ignore
    del v0
    v4 = method12(v3)
    del v3
    return v2, v4
def method23(v0 : object) -> US5:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "None" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US5_0()
    else:
        del v4
        v7 = "Some" == v1
        if v7:
            del v1, v7
            v8 = method11(v2)
            del v2
            return US5_1(v8)
        else:
            del v2, v7
            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
            del v1
            raise Exception("Error")
def method24(v0 : object) -> bool:
    assert isinstance(v0,bool), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method25(v0 : object) -> static_array:
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
        v10 = method15(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method22(v0 : object) -> Tuple[US5, bool, static_array, i32, static_array, i32]:
    v1 = v0["community_card"] # type: ignore
    v2 = method23(v1)
    del v1
    v3 = v0["is_button_s_first_move"] # type: ignore
    v4 = method24(v3)
    del v3
    v5 = v0["pl_card"] # type: ignore
    v6 = method18(v5)
    del v5
    v7 = v0["player_turn"] # type: ignore
    v8 = method15(v7)
    del v7
    v9 = v0["pot"] # type: ignore
    v10 = method25(v9)
    del v9
    v11 = v0["raises_left"] # type: ignore
    del v0
    v12 = method15(v11)
    del v11
    return v2, v4, v6, v8, v10, v12
def method26(v0 : object) -> Tuple[US5, bool, static_array, i32, static_array, i32, US1]:
    v1 = v0[0] # type: ignore
    v2, v3, v4, v5, v6, v7 = method22(v1)
    del v1
    v8 = v0[1] # type: ignore
    del v0
    v9 = method2(v8)
    del v8
    return v2, v3, v4, v5, v6, v7, v9
def method21(v0 : object) -> US4:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "ChanceCommunityCard" == v1
    if v4:
        del v1, v4
        v5, v6, v7, v8, v9, v10 = method22(v2)
        del v2
        return US4_0(v5, v6, v7, v8, v9, v10)
    else:
        del v4
        v13 = "ChanceInit" == v1
        if v13:
            del v1, v13
            method3(v2)
            del v2
            return US4_1()
        else:
            del v13
            v16 = "Round" == v1
            if v16:
                del v1, v16
                v17, v18, v19, v20, v21, v22 = method22(v2)
                del v2
                return US4_2(v17, v18, v19, v20, v21, v22)
            else:
                del v16
                v25 = "RoundWithAction" == v1
                if v25:
                    del v1, v25
                    v26, v27, v28, v29, v30, v31, v32 = method26(v2)
                    del v2
                    return US4_3(v26, v27, v28, v29, v30, v31, v32)
                else:
                    del v25
                    v35 = "TerminalCall" == v1
                    if v35:
                        del v1, v35
                        v36, v37, v38, v39, v40, v41 = method22(v2)
                        del v2
                        return US4_4(v36, v37, v38, v39, v40, v41)
                    else:
                        del v35
                        v44 = "TerminalFold" == v1
                        if v44:
                            del v1, v44
                            v45, v46, v47, v48, v49, v50 = method22(v2)
                            del v2
                            return US4_5(v45, v46, v47, v48, v49, v50)
                        else:
                            del v2, v44
                            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                            del v1
                            raise Exception("Error")
def method20(v0 : object) -> US3:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "None" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US3_0()
    else:
        del v4
        v7 = "Some" == v1
        if v7:
            del v1, v7
            v8 = method21(v2)
            del v2
            return US3_1(v8)
        else:
            del v2, v7
            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
            del v1
            raise Exception("Error")
def method27(v0 : object) -> US7:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "GameNotStarted" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US7_0()
    else:
        del v4
        v7 = "GameOver" == v1
        if v7:
            del v1, v7
            v8, v9, v10, v11, v12, v13 = method22(v2)
            del v2
            return US7_1(v8, v9, v10, v11, v12, v13)
        else:
            del v7
            v16 = "WaitingForActionFromPlayerId" == v1
            if v16:
                del v1, v16
                v17, v18, v19, v20, v21, v22 = method22(v2)
                del v2
                return US7_2(v17, v18, v19, v20, v21, v22)
            else:
                del v2, v16
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method19(v0 : object) -> Tuple[US3, static_array, US7]:
    v1 = v0["game"] # type: ignore
    v2 = method20(v1)
    del v1
    v3 = v0["pl_type"] # type: ignore
    v4 = method4(v3)
    del v3
    v5 = v0["ui_game_state"] # type: ignore
    del v0
    v6 = method27(v5)
    del v5
    return v2, v4, v6
def method8(v0 : object) -> Tuple[static_array_list, static_array_list, US3, static_array, US7]:
    v1 = v0["large"] # type: ignore
    v2, v3 = method9(v1)
    del v1
    v4 = v0["small"] # type: ignore
    del v0
    v5, v6, v7 = method19(v4)
    del v4
    return v2, v3, v5, v6, v7
def method7(v0 : object) -> Tuple[static_array_list, static_array_list, US3, static_array, US7]:
    return method8(v0)
def method29(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[0:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method31(v0 : cp.ndarray) -> None:
    del v0
    return 
def method30(v0 : cp.ndarray, v1 : US1) -> None:
    v2 = v1.tag
    method29(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US1_0(): # Call
            del v1
            return method31(v4)
        case US1_1(): # Fold
            del v1
            return method31(v4)
        case US1_2(): # Raise
            del v1
            return method31(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method33(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method34(v0 : cp.ndarray, v1 : US2) -> None:
    v2 = v1.tag
    method29(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US2_0(): # Computer
            del v1
            return method31(v4)
        case US2_1(): # Human
            del v1
            return method31(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method32(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method33(v2):
        v4 = u64(v2)
        v5 = v4 * 4
        del v4
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v9 = v1[v2]
        method34(v7, v9)
        del v7, v9
        v2 += 1 
    del v0, v1, v2
    return 
def method28(v0 : cp.ndarray, v1 : US0) -> None:
    v2 = v1.tag
    method29(v0, v2)
    del v2
    v4 = v0[8:].view(cp.uint8)
    del v0
    match v1:
        case US0_0(v5): # ActionSelected
            del v1
            return method30(v4, v5)
        case US0_1(v6): # PlayerChanged
            del v1
            return method32(v4, v6)
        case US0_2(): # StartGame
            del v1
            return method31(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method36(v0 : cp.ndarray, v1 : US6) -> None:
    v2 = v1.tag
    method29(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US6_0(): # Jack
            del v1
            return method31(v4)
        case US6_1(): # King
            del v1
            return method31(v4)
        case US6_2(): # Queen
            del v1
            return method31(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method37(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[28:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method40(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[4:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method39(v0 : cp.ndarray, v1 : i32, v2 : US1) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = v2.tag
    method40(v0, v5)
    del v5
    v7 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US1_0(): # Call
            del v2
            return method31(v7)
        case US1_1(): # Fold
            del v2
            return method31(v7)
        case US1_2(): # Raise
            del v2
            return method31(v7)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method41(v0 : cp.ndarray, v1 : i32, v2 : US6) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = v2.tag
    method40(v0, v5)
    del v5
    v7 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US6_0(): # Jack
            del v2
            return method31(v7)
        case US6_1(): # King
            del v2
            return method31(v7)
        case US6_2(): # Queen
            del v2
            return method31(v7)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method42(v0 : cp.ndarray, v1 : static_array, v2 : i32, v3 : i32) -> None:
    v4 = 0
    while method33(v4):
        v6 = u64(v4)
        v7 = v6 * 4
        del v6
        v9 = v0[v7:].view(cp.uint8)
        del v7
        v11 = v1[v4]
        method36(v9, v11)
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
def method38(v0 : cp.ndarray, v1 : US8) -> None:
    v2 = v1.tag
    method29(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US8_0(v5): # CommunityCardIs
            del v1
            return method36(v4, v5)
        case US8_1(v6, v7): # PlayerAction
            del v1
            return method39(v4, v6, v7)
        case US8_2(v8, v9): # PlayerGotCard
            del v1
            return method41(v4, v8, v9)
        case US8_3(v10, v11, v12): # Showdown
            del v1
            return method42(v4, v10, v11, v12)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method43(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[1056:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method45(v0 : cp.ndarray, v1 : US5, v2 : bool, v3 : static_array, v4 : i32, v5 : static_array, v6 : i32) -> None:
    v7 = v1.tag
    method29(v0, v7)
    del v7
    v9 = v0[4:].view(cp.uint8)
    match v1:
        case US5_0(): # None
            method31(v9)
        case US5_1(v10): # Some
            method36(v9, v10)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v1, v9
    v12 = v0[8:].view(cp.bool_)
    v12[0] = v2
    del v2, v12
    v13 = 0
    while method33(v13):
        v15 = u64(v13)
        v16 = v15 * 4
        del v15
        v17 = 12 + v16
        del v16
        v19 = v0[v17:].view(cp.uint8)
        del v17
        v21 = v3[v13]
        method36(v19, v21)
        del v19, v21
        v13 += 1 
    del v3, v13
    v23 = v0[20:].view(cp.int32)
    v23[0] = v4
    del v4, v23
    v24 = 0
    while method33(v24):
        v26 = u64(v24)
        v27 = v26 * 4
        del v26
        v28 = 24 + v27
        del v27
        v30 = v0[v28:].view(cp.uint8)
        del v28
        v32 = v5[v24]
        method29(v30, v32)
        del v30, v32
        v24 += 1 
    del v5, v24
    v34 = v0[32:].view(cp.int32)
    del v0
    v34[0] = v6
    del v6, v34
    return 
def method47(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[36:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method46(v0 : cp.ndarray, v1 : US5, v2 : bool, v3 : static_array, v4 : i32, v5 : static_array, v6 : i32, v7 : US1) -> None:
    v8 = v1.tag
    method29(v0, v8)
    del v8
    v10 = v0[4:].view(cp.uint8)
    match v1:
        case US5_0(): # None
            method31(v10)
        case US5_1(v11): # Some
            method36(v10, v11)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v1, v10
    v13 = v0[8:].view(cp.bool_)
    v13[0] = v2
    del v2, v13
    v14 = 0
    while method33(v14):
        v16 = u64(v14)
        v17 = v16 * 4
        del v16
        v18 = 12 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v22 = v3[v14]
        method36(v20, v22)
        del v20, v22
        v14 += 1 
    del v3, v14
    v24 = v0[20:].view(cp.int32)
    v24[0] = v4
    del v4, v24
    v25 = 0
    while method33(v25):
        v27 = u64(v25)
        v28 = v27 * 4
        del v27
        v29 = 24 + v28
        del v28
        v31 = v0[v29:].view(cp.uint8)
        del v29
        v33 = v5[v25]
        method29(v31, v33)
        del v31, v33
        v25 += 1 
    del v5, v25
    v35 = v0[32:].view(cp.int32)
    v35[0] = v6
    del v6, v35
    v36 = v7.tag
    method47(v0, v36)
    del v36
    v38 = v0[40:].view(cp.uint8)
    del v0
    match v7:
        case US1_0(): # Call
            del v7
            return method31(v38)
        case US1_1(): # Fold
            del v7
            return method31(v38)
        case US1_2(): # Raise
            del v7
            return method31(v38)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method44(v0 : cp.ndarray, v1 : US4) -> None:
    v2 = v1.tag
    method29(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US4_0(v5, v6, v7, v8, v9, v10): # ChanceCommunityCard
            del v1
            return method45(v4, v5, v6, v7, v8, v9, v10)
        case US4_1(): # ChanceInit
            del v1
            return method31(v4)
        case US4_2(v11, v12, v13, v14, v15, v16): # Round
            del v1
            return method45(v4, v11, v12, v13, v14, v15, v16)
        case US4_3(v17, v18, v19, v20, v21, v22, v23): # RoundWithAction
            del v1
            return method46(v4, v17, v18, v19, v20, v21, v22, v23)
        case US4_4(v24, v25, v26, v27, v28, v29): # TerminalCall
            del v1
            return method45(v4, v24, v25, v26, v27, v28, v29)
        case US4_5(v30, v31, v32, v33, v34, v35): # TerminalFold
            del v1
            return method45(v4, v30, v31, v32, v33, v34, v35)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method48(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[1144:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method35(v0 : cp.ndarray, v1 : static_array_list, v2 : static_array_list, v3 : US3, v4 : static_array, v5 : US7) -> None:
    v6 = v1.length
    method29(v0, v6)
    del v6
    v7 = v1.length
    v8 = 0
    while method5(v7, v8):
        v10 = u64(v8)
        v11 = v10 * 4
        del v10
        v12 = 4 + v11
        del v11
        v14 = v0[v12:].view(cp.uint8)
        del v12
        v16 = v1[v8]
        method36(v14, v16)
        del v14, v16
        v8 += 1 
    del v1, v7, v8
    v17 = v2.length
    method37(v0, v17)
    del v17
    v18 = v2.length
    v19 = 0
    while method5(v18, v19):
        v21 = u64(v19)
        v22 = v21 * 32
        del v21
        v23 = 32 + v22
        del v22
        v25 = v0[v23:].view(cp.uint8)
        del v23
        v27 = v2[v19]
        method38(v25, v27)
        del v25, v27
        v19 += 1 
    del v2, v18, v19
    v28 = v3.tag
    method43(v0, v28)
    del v28
    v30 = v0[1072:].view(cp.uint8)
    match v3:
        case US3_0(): # None
            method31(v30)
        case US3_1(v31): # Some
            method44(v30, v31)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v3, v30
    v32 = 0
    while method33(v32):
        v34 = u64(v32)
        v35 = v34 * 4
        del v34
        v36 = 1136 + v35
        del v35
        v38 = v0[v36:].view(cp.uint8)
        del v36
        v40 = v4[v32]
        method34(v38, v40)
        del v38, v40
        v32 += 1 
    del v4, v32
    v41 = v5.tag
    method48(v0, v41)
    del v41
    v43 = v0[1152:].view(cp.uint8)
    del v0
    match v5:
        case US7_0(): # GameNotStarted
            del v5
            return method31(v43)
        case US7_1(v44, v45, v46, v47, v48, v49): # GameOver
            del v5
            return method45(v43, v44, v45, v46, v47, v48, v49)
        case US7_2(v50, v51, v52, v53, v54, v55): # WaitingForActionFromPlayerId
            del v5
            return method45(v43, v50, v51, v52, v53, v54, v55)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method50(v0 : cp.ndarray) -> i32:
    v2 = v0[0:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method52(v0 : cp.ndarray) -> None:
    del v0
    return 
def method51(v0 : cp.ndarray) -> US6:
    v1 = method50(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method52(v3)
        del v3
        return US6_0()
    elif v1 == 1:
        del v1
        method52(v3)
        del v3
        return US6_1()
    elif v1 == 2:
        del v1
        method52(v3)
        del v3
        return US6_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method53(v0 : cp.ndarray) -> i32:
    v2 = v0[28:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method56(v0 : cp.ndarray) -> i32:
    v2 = v0[4:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method55(v0 : cp.ndarray) -> Tuple[i32, US1]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method56(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method52(v6)
        v11 = US1_0()
    elif v4 == 1:
        method52(v6)
        v11 = US1_1()
    elif v4 == 2:
        method52(v6)
        v11 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v11
def method57(v0 : cp.ndarray) -> Tuple[i32, US6]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method56(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method52(v6)
        v11 = US6_0()
    elif v4 == 1:
        method52(v6)
        v11 = US6_1()
    elif v4 == 2:
        method52(v6)
        v11 = US6_2()
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v11
def method58(v0 : cp.ndarray) -> Tuple[static_array, i32, i32]:
    v2 = static_array(2)
    v3 = 0
    while method33(v3):
        v5 = u64(v3)
        v6 = v5 * 4
        del v5
        v8 = v0[v6:].view(cp.uint8)
        del v6
        v9 = method51(v8)
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
def method54(v0 : cp.ndarray) -> US8:
    v1 = method50(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5 = method51(v3)
        del v3
        return US8_0(v5)
    elif v1 == 1:
        del v1
        v7, v8 = method55(v3)
        del v3
        return US8_1(v7, v8)
    elif v1 == 2:
        del v1
        v10, v11 = method57(v3)
        del v3
        return US8_2(v10, v11)
    elif v1 == 3:
        del v1
        v13, v14, v15 = method58(v3)
        del v3
        return US8_3(v13, v14, v15)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method59(v0 : cp.ndarray) -> i32:
    v2 = v0[1056:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method61(v0 : cp.ndarray) -> Tuple[US5, bool, static_array, i32, static_array, i32]:
    v1 = method50(v0)
    v3 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method52(v3)
        v8 = US5_0()
    elif v1 == 1:
        v6 = method51(v3)
        v8 = US5_1(v6)
    else:
        raise Exception("Invalid tag.")
    del v1, v3
    v10 = v0[8:].view(cp.bool_)
    v11 = v10[0].item()
    del v10
    v13 = static_array(2)
    v14 = 0
    while method33(v14):
        v16 = u64(v14)
        v17 = v16 * 4
        del v16
        v18 = 12 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v21 = method51(v20)
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
    while method33(v27):
        v29 = u64(v27)
        v30 = v29 * 4
        del v29
        v31 = 24 + v30
        del v30
        v33 = v0[v31:].view(cp.uint8)
        del v31
        v34 = method50(v33)
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
def method63(v0 : cp.ndarray) -> i32:
    v2 = v0[36:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method62(v0 : cp.ndarray) -> Tuple[US5, bool, static_array, i32, static_array, i32, US1]:
    v1 = method50(v0)
    v3 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method52(v3)
        v8 = US5_0()
    elif v1 == 1:
        v6 = method51(v3)
        v8 = US5_1(v6)
    else:
        raise Exception("Invalid tag.")
    del v1, v3
    v10 = v0[8:].view(cp.bool_)
    v11 = v10[0].item()
    del v10
    v13 = static_array(2)
    v14 = 0
    while method33(v14):
        v16 = u64(v14)
        v17 = v16 * 4
        del v16
        v18 = 12 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v21 = method51(v20)
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
    while method33(v27):
        v29 = u64(v27)
        v30 = v29 * 4
        del v29
        v31 = 24 + v30
        del v30
        v33 = v0[v31:].view(cp.uint8)
        del v31
        v34 = method50(v33)
        del v33
        v26[v27] = v34
        del v34
        v27 += 1 
    del v27
    v36 = v0[32:].view(cp.int32)
    v37 = v36[0].item()
    del v36
    v38 = method63(v0)
    v40 = v0[40:].view(cp.uint8)
    del v0
    if v38 == 0:
        method52(v40)
        v45 = US1_0()
    elif v38 == 1:
        method52(v40)
        v45 = US1_1()
    elif v38 == 2:
        method52(v40)
        v45 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v38, v40
    return v8, v11, v13, v24, v26, v37, v45
def method60(v0 : cp.ndarray) -> US4:
    v1 = method50(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5, v6, v7, v8, v9, v10 = method61(v3)
        del v3
        return US4_0(v5, v6, v7, v8, v9, v10)
    elif v1 == 1:
        del v1
        method52(v3)
        del v3
        return US4_1()
    elif v1 == 2:
        del v1
        v13, v14, v15, v16, v17, v18 = method61(v3)
        del v3
        return US4_2(v13, v14, v15, v16, v17, v18)
    elif v1 == 3:
        del v1
        v20, v21, v22, v23, v24, v25, v26 = method62(v3)
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
def method64(v0 : cp.ndarray) -> US2:
    v1 = method50(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method52(v3)
        del v3
        return US2_0()
    elif v1 == 1:
        del v1
        method52(v3)
        del v3
        return US2_1()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method65(v0 : cp.ndarray) -> i32:
    v2 = v0[1144:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method49(v0 : cp.ndarray) -> Tuple[static_array_list, static_array_list, US3, static_array, US7]:
    v2 = static_array_list(6)
    v3 = method50(v0)
    v2.unsafe_set_length(v3)
    del v3
    v4 = v2.length
    v5 = 0
    while method5(v4, v5):
        v7 = u64(v5)
        v8 = v7 * 4
        del v7
        v9 = 4 + v8
        del v8
        v11 = v0[v9:].view(cp.uint8)
        del v9
        v12 = method51(v11)
        del v11
        v2[v5] = v12
        del v12
        v5 += 1 
    del v4, v5
    v14 = static_array_list(32)
    v15 = method53(v0)
    v14.unsafe_set_length(v15)
    del v15
    v16 = v14.length
    v17 = 0
    while method5(v16, v17):
        v19 = u64(v17)
        v20 = v19 * 32
        del v19
        v21 = 32 + v20
        del v20
        v23 = v0[v21:].view(cp.uint8)
        del v21
        v24 = method54(v23)
        del v23
        v14[v17] = v24
        del v24
        v17 += 1 
    del v16, v17
    v25 = method59(v0)
    v27 = v0[1072:].view(cp.uint8)
    if v25 == 0:
        method52(v27)
        v32 = US3_0()
    elif v25 == 1:
        v30 = method60(v27)
        v32 = US3_1(v30)
    else:
        raise Exception("Invalid tag.")
    del v25, v27
    v34 = static_array(2)
    v35 = 0
    while method33(v35):
        v37 = u64(v35)
        v38 = v37 * 4
        del v37
        v39 = 1136 + v38
        del v38
        v41 = v0[v39:].view(cp.uint8)
        del v39
        v42 = method64(v41)
        del v41
        v34[v35] = v42
        del v42
        v35 += 1 
    del v35
    v43 = method65(v0)
    v45 = v0[1152:].view(cp.uint8)
    del v0
    if v43 == 0:
        method52(v45)
        v62 = US7_0()
    elif v43 == 1:
        v48, v49, v50, v51, v52, v53 = method61(v45)
        v62 = US7_1(v48, v49, v50, v51, v52, v53)
    elif v43 == 2:
        v55, v56, v57, v58, v59, v60 = method61(v45)
        v62 = US7_2(v55, v56, v57, v58, v59, v60)
    else:
        raise Exception("Invalid tag.")
    del v43, v45
    return v2, v14, v32, v34, v62
def method67(v0 : cp.ndarray) -> i32:
    v2 = v0[1048:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method66(v0 : cp.ndarray) -> Tuple[static_array_list, static_array, US7]:
    v2 = static_array_list(32)
    v3 = method50(v0)
    v2.unsafe_set_length(v3)
    del v3
    v4 = v2.length
    v5 = 0
    while method5(v4, v5):
        v7 = u64(v5)
        v8 = v7 * 32
        del v7
        v9 = 16 + v8
        del v8
        v11 = v0[v9:].view(cp.uint8)
        del v9
        v12 = method54(v11)
        del v11
        v2[v5] = v12
        del v12
        v5 += 1 
    del v4, v5
    v14 = static_array(2)
    v15 = 0
    while method33(v15):
        v17 = u64(v15)
        v18 = v17 * 4
        del v17
        v19 = 1040 + v18
        del v18
        v21 = v0[v19:].view(cp.uint8)
        del v19
        v22 = method64(v21)
        del v21
        v14[v15] = v22
        del v22
        v15 += 1 
    del v15
    v23 = method67(v0)
    v25 = v0[1056:].view(cp.uint8)
    del v0
    if v23 == 0:
        method52(v25)
        v42 = US7_0()
    elif v23 == 1:
        v28, v29, v30, v31, v32, v33 = method61(v25)
        v42 = US7_1(v28, v29, v30, v31, v32, v33)
    elif v23 == 2:
        v35, v36, v37, v38, v39, v40 = method61(v25)
        v42 = US7_2(v35, v36, v37, v38, v39, v40)
    else:
        raise Exception("Invalid tag.")
    del v23, v25
    return v2, v14, v42
def method74() -> object:
    v0 = []
    return v0
def method73(v0 : US6) -> object:
    match v0:
        case US6_0(): # Jack
            del v0
            v1 = method74()
            v2 = "Jack"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US6_1(): # King
            del v0
            v4 = method74()
            v5 = "King"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US6_2(): # Queen
            del v0
            v7 = method74()
            v8 = "Queen"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method72(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method5(v2, v3):
        v6 = v0[v3]
        v7 = method73(v6)
        del v6
        v1.append(v7)
        del v7
        v3 += 1 
    del v0, v2, v3
    return v1
def method78(v0 : i32) -> object:
    v1 = v0
    del v0
    return v1
def method79(v0 : US1) -> object:
    match v0:
        case US1_0(): # Call
            del v0
            v1 = method74()
            v2 = "Call"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US1_1(): # Fold
            del v0
            v4 = method74()
            v5 = "Fold"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US1_2(): # Raise
            del v0
            v7 = method74()
            v8 = "Raise"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method77(v0 : i32, v1 : US1) -> object:
    v2 = []
    v3 = method78(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method79(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method80(v0 : i32, v1 : US6) -> object:
    v2 = []
    v3 = method78(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method73(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method82(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method33(v2):
        v5 = v0[v2]
        v6 = method73(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method81(v0 : static_array, v1 : i32, v2 : i32) -> object:
    v3 = method82(v0)
    del v0
    v4 = method78(v1)
    del v1
    v5 = method78(v2)
    del v2
    v6 = {'cards_shown': v3, 'chips_won': v4, 'winner_id': v5}
    del v3, v4, v5
    return v6
def method76(v0 : US8) -> object:
    match v0:
        case US8_0(v1): # CommunityCardIs
            del v0
            v2 = method73(v1)
            del v1
            v3 = "CommunityCardIs"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US8_1(v5, v6): # PlayerAction
            del v0
            v7 = method77(v5, v6)
            del v5, v6
            v8 = "PlayerAction"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US8_2(v10, v11): # PlayerGotCard
            del v0
            v12 = method80(v10, v11)
            del v10, v11
            v13 = "PlayerGotCard"
            v14 = [v13,v12]
            del v12, v13
            return v14
        case US8_3(v15, v16, v17): # Showdown
            del v0
            v18 = method81(v15, v16, v17)
            del v15, v16, v17
            v19 = "Showdown"
            v20 = [v19,v18]
            del v18, v19
            return v20
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method75(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method5(v2, v3):
        v6 = v0[v3]
        v7 = method76(v6)
        del v6
        v1.append(v7)
        del v7
        v3 += 1 
    del v0, v2, v3
    return v1
def method71(v0 : static_array_list, v1 : static_array_list) -> object:
    v2 = method72(v0)
    del v0
    v3 = method75(v1)
    del v1
    v4 = {'deck': v2, 'messages': v3}
    del v2, v3
    return v4
def method87(v0 : US5) -> object:
    match v0:
        case US5_0(): # None
            del v0
            v1 = method74()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US5_1(v4): # Some
            del v0
            v5 = method73(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method88(v0 : bool) -> object:
    v1 = v0
    del v0
    return v1
def method89(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method33(v2):
        v5 = v0[v2]
        v6 = method78(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method86(v0 : US5, v1 : bool, v2 : static_array, v3 : i32, v4 : static_array, v5 : i32) -> object:
    v6 = method87(v0)
    del v0
    v7 = method88(v1)
    del v1
    v8 = method82(v2)
    del v2
    v9 = method78(v3)
    del v3
    v10 = method89(v4)
    del v4
    v11 = method78(v5)
    del v5
    v12 = {'community_card': v6, 'is_button_s_first_move': v7, 'pl_card': v8, 'player_turn': v9, 'pot': v10, 'raises_left': v11}
    del v6, v7, v8, v9, v10, v11
    return v12
def method90(v0 : US5, v1 : bool, v2 : static_array, v3 : i32, v4 : static_array, v5 : i32, v6 : US1) -> object:
    v7 = []
    v8 = method86(v0, v1, v2, v3, v4, v5)
    del v0, v1, v2, v3, v4, v5
    v7.append(v8)
    del v8
    v9 = method79(v6)
    del v6
    v7.append(v9)
    del v9
    v10 = v7
    del v7
    return v10
def method85(v0 : US4) -> object:
    match v0:
        case US4_0(v1, v2, v3, v4, v5, v6): # ChanceCommunityCard
            del v0
            v7 = method86(v1, v2, v3, v4, v5, v6)
            del v1, v2, v3, v4, v5, v6
            v8 = "ChanceCommunityCard"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US4_1(): # ChanceInit
            del v0
            v10 = method74()
            v11 = "ChanceInit"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US4_2(v13, v14, v15, v16, v17, v18): # Round
            del v0
            v19 = method86(v13, v14, v15, v16, v17, v18)
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
            v38 = method86(v32, v33, v34, v35, v36, v37)
            del v32, v33, v34, v35, v36, v37
            v39 = "TerminalCall"
            v40 = [v39,v38]
            del v38, v39
            return v40
        case US4_5(v41, v42, v43, v44, v45, v46): # TerminalFold
            del v0
            v47 = method86(v41, v42, v43, v44, v45, v46)
            del v41, v42, v43, v44, v45, v46
            v48 = "TerminalFold"
            v49 = [v48,v47]
            del v47, v48
            return v49
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method84(v0 : US3) -> object:
    match v0:
        case US3_0(): # None
            del v0
            v1 = method74()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US3_1(v4): # Some
            del v0
            v5 = method85(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method92(v0 : US2) -> object:
    match v0:
        case US2_0(): # Computer
            del v0
            v1 = method74()
            v2 = "Computer"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US2_1(): # Human
            del v0
            v4 = method74()
            v5 = "Human"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method91(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method33(v2):
        v5 = v0[v2]
        v6 = method92(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method93(v0 : US7) -> object:
    match v0:
        case US7_0(): # GameNotStarted
            del v0
            v1 = method74()
            v2 = "GameNotStarted"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US7_1(v4, v5, v6, v7, v8, v9): # GameOver
            del v0
            v10 = method86(v4, v5, v6, v7, v8, v9)
            del v4, v5, v6, v7, v8, v9
            v11 = "GameOver"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US7_2(v13, v14, v15, v16, v17, v18): # WaitingForActionFromPlayerId
            del v0
            v19 = method86(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "WaitingForActionFromPlayerId"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method83(v0 : US3, v1 : static_array, v2 : US7) -> object:
    v3 = method84(v0)
    del v0
    v4 = method91(v1)
    del v1
    v5 = method93(v2)
    del v2
    v6 = {'game': v3, 'pl_type': v4, 'ui_game_state': v5}
    del v3, v4, v5
    return v6
def method70(v0 : static_array_list, v1 : static_array_list, v2 : US3, v3 : static_array, v4 : US7) -> object:
    v5 = method71(v0, v1)
    del v0, v1
    v6 = method83(v2, v3, v4)
    del v2, v3, v4
    v7 = {'large': v5, 'small': v6}
    del v5, v6
    return v7
def method94(v0 : static_array_list, v1 : static_array, v2 : US7) -> object:
    v3 = method75(v0)
    del v0
    v4 = method91(v1)
    del v1
    v5 = method93(v2)
    del v2
    v6 = {'messages': v3, 'pl_type': v4, 'ui_game_state': v5}
    del v3, v4, v5
    return v6
def method69(v0 : static_array_list, v1 : static_array_list, v2 : US3, v3 : static_array, v4 : US7, v5 : static_array_list, v6 : static_array, v7 : US7) -> object:
    v8 = method70(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    v9 = method94(v5, v6, v7)
    del v5, v6, v7
    v10 = {'game_state': v8, 'ui_state': v9}
    del v8, v9
    return v10
def method68(v0 : static_array_list, v1 : static_array_list, v2 : US3, v3 : static_array, v4 : US7, v5 : static_array_list, v6 : static_array, v7 : US7) -> object:
    v8 = method69(v0, v1, v2, v3, v4, v5, v6, v7)
    del v0, v1, v2, v3, v4, v5, v6, v7
    return v8
def method96(v0 : static_array_list, v1 : static_array_list, v2 : US3, v3 : static_array, v4 : US7) -> object:
    v5 = method70(v0, v1, v2, v3, v4)
    del v0, v2
    v6 = method94(v1, v3, v4)
    del v1, v3, v4
    v7 = {'game_state': v5, 'ui_state': v6}
    del v5, v6
    return v7
def method95(v0 : static_array_list, v1 : static_array_list, v2 : US3, v3 : static_array, v4 : US7) -> object:
    v5 = method96(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    return v5
def main():
    v0 = Closure0()
    v1 = Closure1()
    v2 = collections.namedtuple("Leduc_Game",['event_loop_gpu', 'init'])(v0, v1)
    del v0, v1
    return v2

if __name__ == '__main__': print(main())
