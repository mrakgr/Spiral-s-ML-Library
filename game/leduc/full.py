kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
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

struct Union1;
struct Union2;
struct Union0;
__device__ int f_1(unsigned char * v0);
__device__ void f_3(unsigned char * v0);
__device__ Union1 f_2(unsigned char * v0);
__device__ Union2 f_5(unsigned char * v0);
__device__ static_array<Union2,2l> f_4(unsigned char * v0);
__device__ Union0 f_0(unsigned char * v0);
struct Union6;
struct Union5;
struct Union4;
struct Union3;
struct Union7;
struct Union8;
struct Tuple0;
__device__ unsigned int f_7(unsigned char * v0);
__device__ int f_8(unsigned char * v0);
struct Tuple1;
__device__ Union6 f_11(unsigned char * v0);
__device__ Tuple1 f_10(unsigned char * v0);
struct Tuple2;
__device__ int f_13(unsigned char * v0);
__device__ Tuple2 f_12(unsigned char * v0);
__device__ Union4 f_9(unsigned char * v0);
__device__ int f_14(unsigned char * v0);
struct Tuple3;
__device__ Tuple3 f_16(unsigned char * v0);
struct Tuple4;
__device__ Tuple4 f_17(unsigned char * v0);
struct Tuple5;
__device__ Tuple5 f_18(unsigned char * v0);
__device__ Union7 f_15(unsigned char * v0);
__device__ int f_19(unsigned char * v0);
__device__ Tuple0 f_6(unsigned char * v0);
struct Tuple6;
__device__ unsigned int loop_22(unsigned int v0, curandStatePhilox4_32_10_t & v1);
__device__ Tuple6 draw_card_21(curandStatePhilox4_32_10_t & v0, unsigned int v1);
struct Tuple7;
struct Union9;
struct Union10;
struct Union11;
struct Union12;
__device__ int int_range_24(int v0, int v1, curandStatePhilox4_32_10_t & v2);
__device__ void method_25(float * v0, int v1, float * v2, int v3, float * v4, int v5);
__device__ void method_26(unsigned int * v0, int v1, float * v2);
struct Tuple8;
struct Tuple9;
struct Tuple10;
struct Tuple11;
__device__ Tuple8 method_27(curandStatePhilox4_32_10_t & v0, int * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int v9, int v10);
__device__ Union10 noinline_run_23(curandStatePhilox4_32_10_t & v0, unsigned char * v1, unsigned char * v2, Union9 v3);
__device__ void method_28(Union1 v0);
struct Union13;
__device__ int tag_30(Union6 v0);
__device__ bool is_pair_31(int v0, int v1);
__device__ Tuple7 order_32(int v0, int v1);
__device__ Union13 compare_hands_29(Union5 v0, bool v1, static_array<Union6,2l> v2, int v3, static_array<int,2l> v4, int v5);
__device__ void play_loop_20(unsigned char * v0, unsigned long long v1, unsigned char * v2, unsigned long long v3, unsigned int & v4, Union3 & v5, static_array_list<Union7,32l> & v6, static_array<Union2,2l> & v7, curandStatePhilox4_32_10_t & v8, Union8 & v9, Union4 v10);
__device__ void f_34(unsigned char * v0, unsigned int v1);
__device__ void f_35(unsigned char * v0, int v1);
__device__ void f_36(unsigned char * v0);
__device__ void f_38(unsigned char * v0, int v1);
__device__ void f_40(unsigned char * v0, Union6 v1);
__device__ void f_39(unsigned char * v0, Union5 v1, bool v2, static_array<Union6,2l> v3, int v4, static_array<int,2l> v5, int v6);
__device__ void f_42(unsigned char * v0, int v1);
__device__ void f_41(unsigned char * v0, Union5 v1, bool v2, static_array<Union6,2l> v3, int v4, static_array<int,2l> v5, int v6, Union1 v7);
__device__ void f_37(unsigned char * v0, Union4 v1);
__device__ void f_43(unsigned char * v0, int v1);
__device__ void f_45(unsigned char * v0, int v1, Union1 v2);
__device__ void f_46(unsigned char * v0, int v1, Union6 v2);
__device__ void f_47(unsigned char * v0, static_array<Union6,2l> v1, int v2, int v3);
__device__ void f_44(unsigned char * v0, Union7 v1);
__device__ void f_48(unsigned char * v0, Union2 v1);
__device__ void f_49(unsigned char * v0, int v1);
__device__ void f_33(unsigned char * v0, unsigned int v1, Union3 v2, static_array_list<Union7,32l> v3, static_array<Union2,2l> v4, Union8 v5);
struct Union14;
__device__ float method_52(int * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int v8, int v9, int v10);
__device__ static_array<float,2l> method_51(unsigned char * v0, unsigned char * v1, unsigned int & v2, static_array_list<Union7,32l> & v3, static_array<Union14,2l> & v4, curandStatePhilox4_32_10_t & v5, Union4 v6);
__device__ void noinline_train_50(float * v0, float * v1, unsigned char * v2, unsigned char * v3, curandStatePhilox4_32_10_t & v4, int v5, int v6);
__device__ float method_54(int * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int v8, int v9, int v10);
__device__ static_array<float,2l> method_53(unsigned char * v0, unsigned char * v1, unsigned int & v2, static_array_list<Union7,32l> & v3, static_array<Union14,2l> & v4, curandStatePhilox4_32_10_t & v5, Union4 v6);
struct Tuple12;
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
struct Union2_2 { // Random
};
struct Union2 {
    union {
        Union2_0 case0; // Computer
        Union2_1 case1; // Human
        Union2_2 case2; // Random
    };
    unsigned char tag{255};
    __device__ Union2() {}
    __device__ Union2(Union2_0 t) : tag(0), case0(t) {} // Computer
    __device__ Union2(Union2_1 t) : tag(1), case1(t) {} // Human
    __device__ Union2(Union2_2 t) : tag(2), case2(t) {} // Random
    __device__ Union2(Union2 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(x.case0); break; // Computer
            case 1: new (&this->case1) Union2_1(x.case1); break; // Human
            case 2: new (&this->case2) Union2_2(x.case2); break; // Random
        }
    }
    __device__ Union2(Union2 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(std::move(x.case0)); break; // Computer
            case 1: new (&this->case1) Union2_1(std::move(x.case1)); break; // Human
            case 2: new (&this->case2) Union2_2(std::move(x.case2)); break; // Random
        }
    }
    __device__ Union2 & operator=(Union2 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Computer
                case 1: this->case1 = x.case1; break; // Human
                case 2: this->case2 = x.case2; break; // Random
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
                case 2: this->case2 = std::move(x.case2); break; // Random
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
            case 2: this->case2.~Union2_2(); break; // Random
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
struct Union0_3 { // StartTrainingVsRando
};
struct Union0_4 { // StartTrainingVsSelf
};
struct Union0 {
    union {
        Union0_0 case0; // ActionSelected
        Union0_1 case1; // PlayerChanged
        Union0_2 case2; // StartGame
        Union0_3 case3; // StartTrainingVsRando
        Union0_4 case4; // StartTrainingVsSelf
    };
    unsigned char tag{255};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // ActionSelected
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // PlayerChanged
    __device__ Union0(Union0_2 t) : tag(2), case2(t) {} // StartGame
    __device__ Union0(Union0_3 t) : tag(3), case3(t) {} // StartTrainingVsRando
    __device__ Union0(Union0_4 t) : tag(4), case4(t) {} // StartTrainingVsSelf
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // ActionSelected
            case 1: new (&this->case1) Union0_1(x.case1); break; // PlayerChanged
            case 2: new (&this->case2) Union0_2(x.case2); break; // StartGame
            case 3: new (&this->case3) Union0_3(x.case3); break; // StartTrainingVsRando
            case 4: new (&this->case4) Union0_4(x.case4); break; // StartTrainingVsSelf
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // ActionSelected
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // PlayerChanged
            case 2: new (&this->case2) Union0_2(std::move(x.case2)); break; // StartGame
            case 3: new (&this->case3) Union0_3(std::move(x.case3)); break; // StartTrainingVsRando
            case 4: new (&this->case4) Union0_4(std::move(x.case4)); break; // StartTrainingVsSelf
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // ActionSelected
                case 1: this->case1 = x.case1; break; // PlayerChanged
                case 2: this->case2 = x.case2; break; // StartGame
                case 3: this->case3 = x.case3; break; // StartTrainingVsRando
                case 4: this->case4 = x.case4; break; // StartTrainingVsSelf
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
                case 3: this->case3 = std::move(x.case3); break; // StartTrainingVsRando
                case 4: this->case4 = std::move(x.case4); break; // StartTrainingVsSelf
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
            case 3: this->case3.~Union0_3(); break; // StartTrainingVsRando
            case 4: this->case4.~Union0_4(); break; // StartTrainingVsSelf
        }
        this->tag = 255;
    }
};
struct Union6_0 { // Jack
};
struct Union6_1 { // King
};
struct Union6_2 { // Queen
};
struct Union6 {
    union {
        Union6_0 case0; // Jack
        Union6_1 case1; // King
        Union6_2 case2; // Queen
    };
    unsigned char tag{255};
    __device__ Union6() {}
    __device__ Union6(Union6_0 t) : tag(0), case0(t) {} // Jack
    __device__ Union6(Union6_1 t) : tag(1), case1(t) {} // King
    __device__ Union6(Union6_2 t) : tag(2), case2(t) {} // Queen
    __device__ Union6(Union6 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(x.case0); break; // Jack
            case 1: new (&this->case1) Union6_1(x.case1); break; // King
            case 2: new (&this->case2) Union6_2(x.case2); break; // Queen
        }
    }
    __device__ Union6(Union6 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(std::move(x.case0)); break; // Jack
            case 1: new (&this->case1) Union6_1(std::move(x.case1)); break; // King
            case 2: new (&this->case2) Union6_2(std::move(x.case2)); break; // Queen
        }
    }
    __device__ Union6 & operator=(Union6 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Jack
                case 1: this->case1 = x.case1; break; // King
                case 2: this->case2 = x.case2; break; // Queen
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
                case 0: this->case0 = std::move(x.case0); break; // Jack
                case 1: this->case1 = std::move(x.case1); break; // King
                case 2: this->case2 = std::move(x.case2); break; // Queen
            }
        } else {
            this->~Union6();
            new (this) Union6{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union6() {
        switch(this->tag){
            case 0: this->case0.~Union6_0(); break; // Jack
            case 1: this->case1.~Union6_1(); break; // King
            case 2: this->case2.~Union6_2(); break; // Queen
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
struct Union4_0 { // ChanceCommunityCard
    Union5 v0;
    static_array<Union6,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_0(Union5 t0, bool t1, static_array<Union6,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_0() = delete;
};
struct Union4_1 { // ChanceInit
};
struct Union4_2 { // Round
    Union5 v0;
    static_array<Union6,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_2(Union5 t0, bool t1, static_array<Union6,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_2() = delete;
};
struct Union4_3 { // RoundWithAction
    Union5 v0;
    static_array<Union6,2l> v2;
    static_array<int,2l> v4;
    Union1 v6;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_3(Union5 t0, bool t1, static_array<Union6,2l> t2, int t3, static_array<int,2l> t4, int t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union4_3() = delete;
};
struct Union4_4 { // TerminalCall
    Union5 v0;
    static_array<Union6,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_4(Union5 t0, bool t1, static_array<Union6,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_4() = delete;
};
struct Union4_5 { // TerminalFold
    Union5 v0;
    static_array<Union6,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union4_5(Union5 t0, bool t1, static_array<Union6,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
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
struct Union3_0 { // None
};
struct Union3_1 { // Some
    Union4 v0;
    __device__ Union3_1(Union4 t0) : v0(t0) {}
    __device__ Union3_1() = delete;
};
struct Union3 {
    union {
        Union3_0 case0; // None
        Union3_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union3() {}
    __device__ Union3(Union3_0 t) : tag(0), case0(t) {} // None
    __device__ Union3(Union3_1 t) : tag(1), case1(t) {} // Some
    __device__ Union3(Union3 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(x.case0); break; // None
            case 1: new (&this->case1) Union3_1(x.case1); break; // Some
        }
    }
    __device__ Union3(Union3 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union3_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union3 & operator=(Union3 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union3();
            new (this) Union3{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union3() {
        switch(this->tag){
            case 0: this->case0.~Union3_0(); break; // None
            case 1: this->case1.~Union3_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union7_0 { // CommunityCardIs
    Union6 v0;
    __device__ Union7_0(Union6 t0) : v0(t0) {}
    __device__ Union7_0() = delete;
};
struct Union7_1 { // PlayerAction
    Union1 v1;
    int v0;
    __device__ Union7_1(int t0, Union1 t1) : v0(t0), v1(t1) {}
    __device__ Union7_1() = delete;
};
struct Union7_2 { // PlayerGotCard
    Union6 v1;
    int v0;
    __device__ Union7_2(int t0, Union6 t1) : v0(t0), v1(t1) {}
    __device__ Union7_2() = delete;
};
struct Union7_3 { // Showdown
    static_array<Union6,2l> v0;
    int v1;
    int v2;
    __device__ Union7_3(static_array<Union6,2l> t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
    __device__ Union7_3() = delete;
};
struct Union7 {
    union {
        Union7_0 case0; // CommunityCardIs
        Union7_1 case1; // PlayerAction
        Union7_2 case2; // PlayerGotCard
        Union7_3 case3; // Showdown
    };
    unsigned char tag{255};
    __device__ Union7() {}
    __device__ Union7(Union7_0 t) : tag(0), case0(t) {} // CommunityCardIs
    __device__ Union7(Union7_1 t) : tag(1), case1(t) {} // PlayerAction
    __device__ Union7(Union7_2 t) : tag(2), case2(t) {} // PlayerGotCard
    __device__ Union7(Union7_3 t) : tag(3), case3(t) {} // Showdown
    __device__ Union7(Union7 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(x.case0); break; // CommunityCardIs
            case 1: new (&this->case1) Union7_1(x.case1); break; // PlayerAction
            case 2: new (&this->case2) Union7_2(x.case2); break; // PlayerGotCard
            case 3: new (&this->case3) Union7_3(x.case3); break; // Showdown
        }
    }
    __device__ Union7(Union7 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(std::move(x.case0)); break; // CommunityCardIs
            case 1: new (&this->case1) Union7_1(std::move(x.case1)); break; // PlayerAction
            case 2: new (&this->case2) Union7_2(std::move(x.case2)); break; // PlayerGotCard
            case 3: new (&this->case3) Union7_3(std::move(x.case3)); break; // Showdown
        }
    }
    __device__ Union7 & operator=(Union7 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // CommunityCardIs
                case 1: this->case1 = x.case1; break; // PlayerAction
                case 2: this->case2 = x.case2; break; // PlayerGotCard
                case 3: this->case3 = x.case3; break; // Showdown
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
                case 0: this->case0 = std::move(x.case0); break; // CommunityCardIs
                case 1: this->case1 = std::move(x.case1); break; // PlayerAction
                case 2: this->case2 = std::move(x.case2); break; // PlayerGotCard
                case 3: this->case3 = std::move(x.case3); break; // Showdown
            }
        } else {
            this->~Union7();
            new (this) Union7{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union7() {
        switch(this->tag){
            case 0: this->case0.~Union7_0(); break; // CommunityCardIs
            case 1: this->case1.~Union7_1(); break; // PlayerAction
            case 2: this->case2.~Union7_2(); break; // PlayerGotCard
            case 3: this->case3.~Union7_3(); break; // Showdown
        }
        this->tag = 255;
    }
};
struct Union8_0 { // GameNotStarted
};
struct Union8_1 { // GameOver
    Union5 v0;
    static_array<Union6,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union8_1(Union5 t0, bool t1, static_array<Union6,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union8_1() = delete;
};
struct Union8_2 { // WaitingForActionFromPlayerId
    Union5 v0;
    static_array<Union6,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Union8_2(Union5 t0, bool t1, static_array<Union6,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
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
    Union3 v1;
    static_array_list<Union7,32l> v2;
    static_array<Union2,2l> v3;
    Union8 v4;
    unsigned int v0;
    __device__ Tuple0() = default;
    __device__ Tuple0(unsigned int t0, Union3 t1, static_array_list<Union7,32l> t2, static_array<Union2,2l> t3, Union8 t4) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4) {}
};
struct Tuple1 {
    Union5 v0;
    static_array<Union6,2l> v2;
    static_array<int,2l> v4;
    int v3;
    int v5;
    bool v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(Union5 t0, bool t1, static_array<Union6,2l> t2, int t3, static_array<int,2l> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Tuple2 {
    Union5 v0;
    static_array<Union6,2l> v2;
    static_array<int,2l> v4;
    Union1 v6;
    int v3;
    int v5;
    bool v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(Union5 t0, bool t1, static_array<Union6,2l> t2, int t3, static_array<int,2l> t4, int t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
};
struct Tuple3 {
    Union1 v1;
    int v0;
    __device__ Tuple3() = default;
    __device__ Tuple3(int t0, Union1 t1) : v0(t0), v1(t1) {}
};
struct Tuple4 {
    Union6 v1;
    int v0;
    __device__ Tuple4() = default;
    __device__ Tuple4(int t0, Union6 t1) : v0(t0), v1(t1) {}
};
struct Tuple5 {
    static_array<Union6,2l> v0;
    int v1;
    int v2;
    __device__ Tuple5() = default;
    __device__ Tuple5(static_array<Union6,2l> t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple6 {
    Union6 v0;
    unsigned int v1;
    __device__ Tuple6() = default;
    __device__ Tuple6(Union6 t0, unsigned int t1) : v0(t0), v1(t1) {}
};
struct Tuple7 {
    int v0;
    int v1;
    __device__ Tuple7() = default;
    __device__ Tuple7(int t0, int t1) : v0(t0), v1(t1) {}
};
struct Union9_0 { // None
};
struct Union9_1 { // Some
    Union5 v0;
    static_array<Union6,2l> v2;
    static_array<int,2l> v4;
    static_array_list<Union7,32l> v6;
    int v3;
    int v5;
    bool v1;
    __device__ Union9_1(Union5 t0, bool t1, static_array<Union6,2l> t2, int t3, static_array<int,2l> t4, int t5, static_array_list<Union7,32l> t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
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
struct Union11_0 { // C1of2
    Union1 v0;
    __device__ Union11_0(Union1 t0) : v0(t0) {}
    __device__ Union11_0() = delete;
};
struct Union11_1 { // C2of2
    Union6 v0;
    __device__ Union11_1(Union6 t0) : v0(t0) {}
    __device__ Union11_1() = delete;
};
struct Union11 {
    union {
        Union11_0 case0; // C1of2
        Union11_1 case1; // C2of2
    };
    unsigned char tag{255};
    __device__ Union11() {}
    __device__ Union11(Union11_0 t) : tag(0), case0(t) {} // C1of2
    __device__ Union11(Union11_1 t) : tag(1), case1(t) {} // C2of2
    __device__ Union11(Union11 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union11_0(x.case0); break; // C1of2
            case 1: new (&this->case1) Union11_1(x.case1); break; // C2of2
        }
    }
    __device__ Union11(Union11 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union11_0(std::move(x.case0)); break; // C1of2
            case 1: new (&this->case1) Union11_1(std::move(x.case1)); break; // C2of2
        }
    }
    __device__ Union11 & operator=(Union11 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // C1of2
                case 1: this->case1 = x.case1; break; // C2of2
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
                case 0: this->case0 = std::move(x.case0); break; // C1of2
                case 1: this->case1 = std::move(x.case1); break; // C2of2
            }
        } else {
            this->~Union11();
            new (this) Union11{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union11() {
        switch(this->tag){
            case 0: this->case0.~Union11_0(); break; // C1of2
            case 1: this->case1.~Union11_1(); break; // C2of2
        }
        this->tag = 255;
    }
};
struct Union12_0 { // None
};
struct Union12_1 { // Some
    Union11 v0;
    __device__ Union12_1(Union11 t0) : v0(t0) {}
    __device__ Union12_1() = delete;
};
struct Union12 {
    union {
        Union12_0 case0; // None
        Union12_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union12() {}
    __device__ Union12(Union12_0 t) : tag(0), case0(t) {} // None
    __device__ Union12(Union12_1 t) : tag(1), case1(t) {} // Some
    __device__ Union12(Union12 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union12_0(x.case0); break; // None
            case 1: new (&this->case1) Union12_1(x.case1); break; // Some
        }
    }
    __device__ Union12(Union12 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union12_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union12_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union12 & operator=(Union12 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union12();
            new (this) Union12{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union12() {
        switch(this->tag){
            case 0: this->case0.~Union12_0(); break; // None
            case 1: this->case1.~Union12_1(); break; // Some
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
struct Tuple8 {
    float v0;
    int v1;
    __device__ Tuple8() = default;
    __device__ Tuple8(float t0, int t1) : v0(t0), v1(t1) {}
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
struct Tuple9 {
    int v0;
    float v1;
    __device__ Tuple9() = default;
    __device__ Tuple9(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple10 {
    float v0;
    bool v1;
    __device__ Tuple10() = default;
    __device__ Tuple10(float t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
    __device__ Tuple10 operator()(Tuple10 tup0, Tuple10 tup1){
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
                return Tuple10{v5, true};
            } else {
                return Tuple10{v0, v1};
            }
        } else {
            if (v3){
                return Tuple10{v2, v3};
            } else {
                return Tuple10{v0, v1};
            }
        }
    }
};
struct Closure5 {
    __device__ Tuple8 operator()(Tuple8 tup0, Tuple8 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple8{v0, v1};
        } else {
            return Tuple8{v2, v3};
        }
    }
};
struct Tuple11 {
    int v0;
    bool v1;
    __device__ Tuple11() = default;
    __device__ Tuple11(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure6 {
    __device__ Tuple11 operator()(Tuple11 tup0, Tuple11 tup1){
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
                return Tuple11{v5, true};
            } else {
                return Tuple11{v0, v1};
            }
        } else {
            if (v3){
                return Tuple11{v2, v3};
            } else {
                return Tuple11{v0, v1};
            }
        }
    }
};
struct Closure7 {
    int v0;
    __device__ Tuple8 operator()(Tuple8 tup0, Tuple8 tup1){
        int & v0 = this->v0;
        float v1 = tup0.v0; int v2 = tup0.v1; float v3 = tup1.v0; int v4 = tup1.v1;
        bool v5;
        v5 = v2 == v0;
        if (v5){
            return Tuple8{v1, v2};
        } else {
            bool v6;
            v6 = v4 == v0;
            if (v6){
                return Tuple8{v3, v4};
            } else {
                return Tuple8{v1, v2};
            }
        }
    }
    __device__ Closure7(int _v0) : v0(_v0) { }
};
struct Union13_0 { // Eq
};
struct Union13_1 { // Gt
};
struct Union13_2 { // Lt
};
struct Union13 {
    union {
        Union13_0 case0; // Eq
        Union13_1 case1; // Gt
        Union13_2 case2; // Lt
    };
    unsigned char tag{255};
    __device__ Union13() {}
    __device__ Union13(Union13_0 t) : tag(0), case0(t) {} // Eq
    __device__ Union13(Union13_1 t) : tag(1), case1(t) {} // Gt
    __device__ Union13(Union13_2 t) : tag(2), case2(t) {} // Lt
    __device__ Union13(Union13 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union13_0(x.case0); break; // Eq
            case 1: new (&this->case1) Union13_1(x.case1); break; // Gt
            case 2: new (&this->case2) Union13_2(x.case2); break; // Lt
        }
    }
    __device__ Union13(Union13 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union13_0(std::move(x.case0)); break; // Eq
            case 1: new (&this->case1) Union13_1(std::move(x.case1)); break; // Gt
            case 2: new (&this->case2) Union13_2(std::move(x.case2)); break; // Lt
        }
    }
    __device__ Union13 & operator=(Union13 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Eq
                case 1: this->case1 = x.case1; break; // Gt
                case 2: this->case2 = x.case2; break; // Lt
            }
        } else {
            this->~Union13();
            new (this) Union13{x};
        }
        return *this;
    }
    __device__ Union13 & operator=(Union13 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Eq
                case 1: this->case1 = std::move(x.case1); break; // Gt
                case 2: this->case2 = std::move(x.case2); break; // Lt
            }
        } else {
            this->~Union13();
            new (this) Union13{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union13() {
        switch(this->tag){
            case 0: this->case0.~Union13_0(); break; // Eq
            case 1: this->case1.~Union13_1(); break; // Gt
            case 2: this->case2.~Union13_2(); break; // Lt
        }
        this->tag = 255;
    }
};
struct Union14_0 { // T_Computer
};
struct Union14_1 { // T_Random
};
struct Union14 {
    union {
        Union14_0 case0; // T_Computer
        Union14_1 case1; // T_Random
    };
    unsigned char tag{255};
    __device__ Union14() {}
    __device__ Union14(Union14_0 t) : tag(0), case0(t) {} // T_Computer
    __device__ Union14(Union14_1 t) : tag(1), case1(t) {} // T_Random
    __device__ Union14(Union14 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union14_0(x.case0); break; // T_Computer
            case 1: new (&this->case1) Union14_1(x.case1); break; // T_Random
        }
    }
    __device__ Union14(Union14 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union14_0(std::move(x.case0)); break; // T_Computer
            case 1: new (&this->case1) Union14_1(std::move(x.case1)); break; // T_Random
        }
    }
    __device__ Union14 & operator=(Union14 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // T_Computer
                case 1: this->case1 = x.case1; break; // T_Random
            }
        } else {
            this->~Union14();
            new (this) Union14{x};
        }
        return *this;
    }
    __device__ Union14 & operator=(Union14 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // T_Computer
                case 1: this->case1 = std::move(x.case1); break; // T_Random
            }
        } else {
            this->~Union14();
            new (this) Union14{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union14() {
        switch(this->tag){
            case 0: this->case0.~Union14_0(); break; // T_Computer
            case 1: this->case1.~Union14_1(); break; // T_Random
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
struct Tuple12 {
    double v1;
    int v0;
    __device__ Tuple12() = default;
    __device__ Tuple12(int t0, double t1) : v0(t0), v1(t1) {}
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
            __trap();
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
        case 2: {
            f_3(v2);
            return Union2{Union2_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
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
        case 3: {
            f_3(v2);
            return Union0{Union0_3{}};
            break;
        }
        case 4: {
            f_3(v2);
            return Union0{Union0_4{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
}
__device__ unsigned int f_7(unsigned char * v0){
    unsigned int * v1;
    v1 = (unsigned int *)(v0+0ull);
    unsigned int v3;
    v3 = v1[0l];
    return v3;
}
__device__ int f_8(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+4ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ Union6 f_11(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    switch (v1) {
        case 0: {
            f_3(v2);
            return Union6{Union6_0{}};
            break;
        }
        case 1: {
            f_3(v2);
            return Union6{Union6_1{}};
            break;
        }
        case 2: {
            f_3(v2);
            return Union6{Union6_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
}
__device__ Tuple1 f_10(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    Union5 v8;
    switch (v1) {
        case 0: {
            f_3(v2);
            v8 = Union5{Union5_0{}};
            break;
        }
        case 1: {
            Union6 v6;
            v6 = f_11(v2);
            v8 = Union5{Union5_1{v6}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
    bool * v9;
    v9 = (bool *)(v0+8ull);
    bool v11;
    v11 = v9[0l];
    static_array<Union6,2l> v12;
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
        Union6 v21;
        v21 = f_11(v19);
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
    return Tuple1{v8, v11, v12, v24, v25, v37};
}
__device__ int f_13(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+36ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ Tuple2 f_12(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    Union5 v8;
    switch (v1) {
        case 0: {
            f_3(v2);
            v8 = Union5{Union5_0{}};
            break;
        }
        case 1: {
            Union6 v6;
            v6 = f_11(v2);
            v8 = Union5{Union5_1{v6}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
    bool * v9;
    v9 = (bool *)(v0+8ull);
    bool v11;
    v11 = v9[0l];
    static_array<Union6,2l> v12;
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
        Union6 v21;
        v21 = f_11(v19);
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
    v38 = f_13(v0);
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
            __trap();
        }
    }
    return Tuple2{v8, v11, v12, v24, v25, v37, v45};
}
__device__ Union4 f_9(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            Union5 v5; bool v6; static_array<Union6,2l> v7; int v8; static_array<int,2l> v9; int v10;
            Tuple1 tmp0 = f_10(v2);
            v5 = tmp0.v0; v6 = tmp0.v1; v7 = tmp0.v2; v8 = tmp0.v3; v9 = tmp0.v4; v10 = tmp0.v5;
            return Union4{Union4_0{v5, v6, v7, v8, v9, v10}};
            break;
        }
        case 1: {
            f_3(v2);
            return Union4{Union4_1{}};
            break;
        }
        case 2: {
            Union5 v13; bool v14; static_array<Union6,2l> v15; int v16; static_array<int,2l> v17; int v18;
            Tuple1 tmp1 = f_10(v2);
            v13 = tmp1.v0; v14 = tmp1.v1; v15 = tmp1.v2; v16 = tmp1.v3; v17 = tmp1.v4; v18 = tmp1.v5;
            return Union4{Union4_2{v13, v14, v15, v16, v17, v18}};
            break;
        }
        case 3: {
            Union5 v20; bool v21; static_array<Union6,2l> v22; int v23; static_array<int,2l> v24; int v25; Union1 v26;
            Tuple2 tmp2 = f_12(v2);
            v20 = tmp2.v0; v21 = tmp2.v1; v22 = tmp2.v2; v23 = tmp2.v3; v24 = tmp2.v4; v25 = tmp2.v5; v26 = tmp2.v6;
            return Union4{Union4_3{v20, v21, v22, v23, v24, v25, v26}};
            break;
        }
        case 4: {
            Union5 v28; bool v29; static_array<Union6,2l> v30; int v31; static_array<int,2l> v32; int v33;
            Tuple1 tmp3 = f_10(v2);
            v28 = tmp3.v0; v29 = tmp3.v1; v30 = tmp3.v2; v31 = tmp3.v3; v32 = tmp3.v4; v33 = tmp3.v5;
            return Union4{Union4_4{v28, v29, v30, v31, v32, v33}};
            break;
        }
        case 5: {
            Union5 v35; bool v36; static_array<Union6,2l> v37; int v38; static_array<int,2l> v39; int v40;
            Tuple1 tmp4 = f_10(v2);
            v35 = tmp4.v0; v36 = tmp4.v1; v37 = tmp4.v2; v38 = tmp4.v3; v39 = tmp4.v4; v40 = tmp4.v5;
            return Union4{Union4_5{v35, v36, v37, v38, v39, v40}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
}
__device__ int f_14(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+80ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ inline bool while_method_1(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ Tuple3 f_16(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    int v4;
    v4 = f_8(v0);
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
            __trap();
        }
    }
    return Tuple3{v3, v11};
}
__device__ Tuple4 f_17(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    int v4;
    v4 = f_8(v0);
    unsigned char * v5;
    v5 = (unsigned char *)(v0+8ull);
    Union6 v11;
    switch (v4) {
        case 0: {
            f_3(v5);
            v11 = Union6{Union6_0{}};
            break;
        }
        case 1: {
            f_3(v5);
            v11 = Union6{Union6_1{}};
            break;
        }
        case 2: {
            f_3(v5);
            v11 = Union6{Union6_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
    return Tuple4{v3, v11};
}
__device__ Tuple5 f_18(unsigned char * v0){
    static_array<Union6,2l> v1;
    int v3;
    v3 = 0l;
    while (while_method_0(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned long long v6;
        v6 = v5 * 4ull;
        unsigned char * v7;
        v7 = (unsigned char *)(v0+v6);
        Union6 v9;
        v9 = f_11(v7);
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
    return Tuple5{v1, v12, v15};
}
__device__ Union7 f_15(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            Union6 v5;
            v5 = f_11(v2);
            return Union7{Union7_0{v5}};
            break;
        }
        case 1: {
            int v7; Union1 v8;
            Tuple3 tmp5 = f_16(v2);
            v7 = tmp5.v0; v8 = tmp5.v1;
            return Union7{Union7_1{v7, v8}};
            break;
        }
        case 2: {
            int v10; Union6 v11;
            Tuple4 tmp6 = f_17(v2);
            v10 = tmp6.v0; v11 = tmp6.v1;
            return Union7{Union7_2{v10, v11}};
            break;
        }
        case 3: {
            static_array<Union6,2l> v13; int v14; int v15;
            Tuple5 tmp7 = f_18(v2);
            v13 = tmp7.v0; v14 = tmp7.v1; v15 = tmp7.v2;
            return Union7{Union7_3{v13, v14, v15}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
}
__device__ int f_19(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+1128ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ Tuple0 f_6(unsigned char * v0){
    unsigned int v1;
    v1 = f_7(v0);
    int v2;
    v2 = f_8(v0);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    Union3 v9;
    switch (v2) {
        case 0: {
            f_3(v3);
            v9 = Union3{Union3_0{}};
            break;
        }
        case 1: {
            Union4 v7;
            v7 = f_9(v3);
            v9 = Union3{Union3_1{v7}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
    static_array_list<Union7,32l> v10;
    v10 = static_array_list<Union7,32l>{};
    int v12;
    v12 = f_14(v0);
    v10.unsafe_set_length(v12);
    int v13;
    v13 = v10.length;
    int v14;
    v14 = 0l;
    while (while_method_1(v13, v14)){
        unsigned long long v16;
        v16 = (unsigned long long)v14;
        unsigned long long v17;
        v17 = v16 * 32ull;
        unsigned long long v18;
        v18 = 96ull + v17;
        unsigned char * v19;
        v19 = (unsigned char *)(v0+v18);
        Union7 v21;
        v21 = f_15(v19);
        v10[v14] = v21;
        v14 += 1l ;
    }
    static_array<Union2,2l> v22;
    int v24;
    v24 = 0l;
    while (while_method_0(v24)){
        unsigned long long v26;
        v26 = (unsigned long long)v24;
        unsigned long long v27;
        v27 = v26 * 4ull;
        unsigned long long v28;
        v28 = 1120ull + v27;
        unsigned char * v29;
        v29 = (unsigned char *)(v0+v28);
        Union2 v31;
        v31 = f_5(v29);
        v22[v24] = v31;
        v24 += 1l ;
    }
    int v32;
    v32 = f_19(v0);
    unsigned char * v33;
    v33 = (unsigned char *)(v0+1136ull);
    Union8 v51;
    switch (v32) {
        case 0: {
            f_3(v33);
            v51 = Union8{Union8_0{}};
            break;
        }
        case 1: {
            Union5 v37; bool v38; static_array<Union6,2l> v39; int v40; static_array<int,2l> v41; int v42;
            Tuple1 tmp8 = f_10(v33);
            v37 = tmp8.v0; v38 = tmp8.v1; v39 = tmp8.v2; v40 = tmp8.v3; v41 = tmp8.v4; v42 = tmp8.v5;
            v51 = Union8{Union8_1{v37, v38, v39, v40, v41, v42}};
            break;
        }
        case 2: {
            Union5 v44; bool v45; static_array<Union6,2l> v46; int v47; static_array<int,2l> v48; int v49;
            Tuple1 tmp9 = f_10(v33);
            v44 = tmp9.v0; v45 = tmp9.v1; v46 = tmp9.v2; v47 = tmp9.v3; v48 = tmp9.v4; v49 = tmp9.v5;
            v51 = Union8{Union8_2{v44, v45, v46, v47, v48, v49}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
    return Tuple0{v1, v9, v10, v22, v51};
}
__device__ inline bool while_method_2(Union3 v0){
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
__device__ Tuple6 draw_card_21(curandStatePhilox4_32_10_t & v0, unsigned int v1){
    int v2;
    v2 = __popc(v1);
    unsigned int v3;
    v3 = (unsigned int)v2;
    unsigned int v4;
    v4 = loop_22(v3, v0);
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
    Union6 v31;
    if (v13){
        v31 = Union6{Union6_1{}};
    } else {
        bool v15;
        v15 = 1ul == v12;
        if (v15){
            v31 = Union6{Union6_1{}};
        } else {
            bool v17;
            v17 = 2ul == v12;
            if (v17){
                v31 = Union6{Union6_2{}};
            } else {
                bool v19;
                v19 = 3ul == v12;
                if (v19){
                    v31 = Union6{Union6_2{}};
                } else {
                    bool v21;
                    v21 = 4ul == v12;
                    if (v21){
                        v31 = Union6{Union6_0{}};
                    } else {
                        bool v23;
                        v23 = 5ul == v12;
                        if (v23){
                            v31 = Union6{Union6_0{}};
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
    return Tuple6{v31, v34};
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 65536l;
    return v1;
}
__device__ int int_range_24(int v0, int v1, curandStatePhilox4_32_10_t & v2){
    int v3;
    v3 = v0 - v1;
    unsigned int v4;
    v4 = (unsigned int)v3;
    unsigned int v5;
    v5 = loop_22(v4, v2);
    unsigned int v6;
    v6 = (unsigned int)v1;
    unsigned int v7;
    v7 = v5 + v6;
    int v8;
    v8 = (int)v7;
    return v8;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 8l;
    return v1;
}
__device__ void method_25(float * v0, int v1, float * v2, int v3, float * v4, int v5){
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
    while (while_method_4(v61)){
        int v63;
        v63 = 0l;
        while (while_method_5(v63)){
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
            while (while_method_4(v71)){
                int v73;
                v73 = 0l;
                #pragma unroll
                while (while_method_5(v73)){
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
            while (while_method_0(v77)){
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
                while (while_method_4(v107)){
                    int v109;
                    v109 = 0l;
                    #pragma unroll
                    while (while_method_5(v109)){
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
                        while (while_method_4(v117)){
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
                while (while_method_4(v142)){
                    int v144;
                    v144 = 0l;
                    #pragma unroll
                    while (while_method_5(v144)){
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
                        while (while_method_4(v152)){
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
                while (while_method_4(v161)){
                    int v163;
                    v163 = 0l;
                    #pragma unroll
                    while (while_method_6(v163)){
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
                        while (while_method_0(v171)){
                            int v173;
                            v173 = 0l;
                            #pragma unroll
                            while (while_method_0(v173)){
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
                while (while_method_5(v192)){
                    int v194;
                    v194 = 0l;
                    #pragma unroll
                    while (while_method_6(v194)){
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
                        while (while_method_0(v202)){
                            int v204;
                            v204 = 0l;
                            #pragma unroll
                            while (while_method_0(v204)){
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
                while (while_method_4(v223)){
                    int v225;
                    v225 = 0l;
                    #pragma unroll
                    while (while_method_5(v225)){
                        int v227;
                        v227 = 0l;
                        #pragma unroll
                        while (while_method_6(v227)){
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
            while (while_method_4(v237)){
                int v239;
                v239 = 0l;
                #pragma unroll
                while (while_method_5(v239)){
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
            while (while_method_6(v266)){
                int v268;
                v268 = 0l;
                #pragma unroll
                while (while_method_5(v268)){
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
__device__ inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 32l;
    return v1;
}
__device__ void method_26(unsigned int * v0, int v1, float * v2){
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
    while (while_method_7(v22)){
        assert("Tensor range check" && 0 <= v22 && v22 < 32l);
        int v24;
        v24 = 2048l * v22;
        int v25;
        v25 = v24 + v20;
        float v26[4l];
        int v27[4l];
        int v28;
        v28 = 0l;
        while (while_method_5(v28)){
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
        while (while_method_5(v35)){
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
        v69 = v22 * 16l;
        int v70;
        v70 = v69 + v13;
        unsigned int v71[4l];
        int v72;
        v72 = 0l;
        while (while_method_5(v72)){
            int v74;
            v74 = 0l;
            while (while_method_4(v74)){
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
        while (while_method_5(v84)){
            int v86;
            v86 = 0l;
            while (while_method_4(v86)){
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
__device__ Tuple8 method_27(curandStatePhilox4_32_10_t & v0, int * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int v9, int v10){
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
    while (while_method_5(v59)){
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
        while (while_method_5(v82)){
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
        while (while_method_5(v90)){
            int v92;
            v92 = 0l;
            while (while_method_4(v92)){
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
        while (while_method_5(v116)){
            int v118;
            v118 = 0l;
            while (while_method_4(v118)){
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
        while (while_method_5(v126)){
            int v128;
            v128 = 0l;
            while (while_method_4(v128)){
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
        while (while_method_5(v138)){
            int v140;
            v140 = 0l;
            while (while_method_4(v140)){
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
        while (while_method_5(v152)){
            int v154;
            v154 = 0l;
            while (while_method_4(v154)){
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
        while (while_method_5(v161)){
            int v163;
            v163 = 0l;
            while (while_method_4(v163)){
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
        while (while_method_5(v177)){
            int v179;
            v179 = 0l;
            while (while_method_4(v179)){
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
        while (while_method_5(v193)){
            assert("Tensor range check" && 0 <= v193 && v193 < 1l);
            int v195;
            v195 = 4l * v193;
            assert("Tensor range check" && 0 <= v193 && v193 < 1l);
            int v196; float v197;
            Tuple9 tmp15 = Tuple9{0l, 0.0f};
            v196 = tmp15.v0; v197 = tmp15.v1;
            while (while_method_4(v196)){
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
            Tuple9 tmp16 = Tuple9{0l, v211};
            v212 = tmp16.v0; v213 = tmp16.v1;
            while (while_method_4(v212)){
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
        while (while_method_5(v221)){
            int v223;
            v223 = 0l;
            while (while_method_4(v223)){
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
        Tuple10 tmp17 = Tuple10{-1.0f / 0.0f, false};
        v230 = tmp17.v0; v231 = tmp17.v1;
        int v232;
        v232 = 0l;
        while (while_method_5(v232)){
            int v234;
            v234 = 0l;
            while (while_method_4(v234)){
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
        Tuple10 tmp18 = cooperative_groups::reduce(v250, Tuple10{v230, v231}, v251);
        v252 = tmp18.v0; v253 = tmp18.v1;
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
        while (while_method_5(v258)){
            int v260;
            v260 = 0l;
            while (while_method_4(v260)){
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
        Tuple8 tmp19 = Tuple8{0.0f, 2147483647l};
        v266 = tmp19.v0; v267 = tmp19.v1;
        int v268;
        v268 = 0l;
        while (while_method_5(v268)){
            int v270;
            v270 = 0l;
            while (while_method_4(v270)){
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
        Tuple8 tmp20 = cooperative_groups::reduce(v281, Tuple8{v266, v267}, v282);
        v283 = tmp20.v0; v284 = tmp20.v1;
        float v285;
        v285 = v252 * v283;
        int v286[4l];
        bool v287[4l];
        int v288;
        v288 = 0l;
        while (while_method_5(v288)){
            int v290;
            v290 = 0l;
            while (while_method_4(v290)){
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
        Tuple11 tmp21 = Tuple11{2147483647l, false};
        v301 = tmp21.v0; v302 = tmp21.v1;
        int v303;
        v303 = 0l;
        while (while_method_5(v303)){
            int v305;
            v305 = 0l;
            while (while_method_4(v305)){
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
        Tuple11 tmp22 = cooperative_groups::reduce(v321, Tuple11{v301, v302}, v322);
        v323 = tmp22.v0; v324 = tmp22.v1;
        bool v325;
        v325 = v324 == false;
        if (v325){
            assert("The local reduce must be true." && v324);
        } else {
        }
        bool v327[4l];
        int v328;
        v328 = 0l;
        while (while_method_5(v328)){
            int v330;
            v330 = 0l;
            while (while_method_4(v330)){
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
        while (while_method_5(v338)){
            int v340;
            v340 = 0l;
            while (while_method_4(v340)){
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
        while (while_method_5(v350)){
            int v352;
            v352 = 0l;
            while (while_method_4(v352)){
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
        while (while_method_5(v363)){
            int v365;
            v365 = 0l;
            while (while_method_4(v365)){
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
        while (while_method_5(v372)){
            int v374;
            v374 = 0l;
            while (while_method_4(v374)){
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
        while (while_method_5(v387)){
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
        Tuple8 tmp23 = Tuple8{0.0f, 2147483647l};
        v401 = tmp23.v0; v402 = tmp23.v1;
        int v403;
        v403 = 0l;
        while (while_method_5(v403)){
            int v405;
            v405 = 0l;
            while (while_method_4(v405)){
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
        Tuple8 tmp24 = cooperative_groups::reduce(v419, Tuple8{v401, v402}, v420);
        v421 = tmp24.v0; v422 = tmp24.v1;
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
        while (while_method_5(v427)){
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
    return Tuple8{v429, v430};
}
__device__ __noinline__ Union10 noinline_run_23(curandStatePhilox4_32_10_t & v0, unsigned char * v1, unsigned char * v2, Union9 v3){
    asm("barrier.cta.sync %0;" :: "r"(0l));
    unsigned int * v4;
    v4 = reinterpret_cast<unsigned int *>(&v1[12582912ull]);
    float * v6;
    v6 = reinterpret_cast<float *>(&v1[0ull]);
    int v8;
    v8 = threadIdx.x;
    int v9;
    v9 = blockIdx.x;
    int v10;
    v10 = v9 * 512l;
    int v11;
    v11 = v8 + v10;
    unsigned long long v12;
    v12 = (unsigned long long)v11;
    curandStatePhilox4_32_10_t v13;
    curand_init(12344321ull,v12,0ull,&v13);
    float * v14;
    v14 = reinterpret_cast<float *>(&v1[0ull]);
    int v16;
    v16 = blockIdx.x;
    assert("Tensor range check" && 0 <= v16 && v16 < 24l);
    int v17;
    v17 = 65536l * v16;
    int v18;
    v18 = threadIdx.x;
    int v19;
    v19 = blockIdx.x;
    int v20;
    v20 = v19 * 512l;
    int v21;
    v21 = v18 + v20;
    unsigned long long v22;
    v22 = (unsigned long long)v21;
    curandStatePhilox4_32_10_t v23;
    curand_init(12344321ull,v22,0ull,&v23);
    int v24;
    v24 = threadIdx.x;
    int v25;
    v25 = v24;
    while (while_method_3(v25)){
        bool v27;
        v27 = 0l <= v25;
        bool v28;
        v28 = v27 == false;
        if (v28){
            assert("The index needs to be zero or positive." && v27);
        } else {
        }
        int v30;
        v30 = v25 % 128l;
        int v31;
        v31 = v25 / 128l;
        bool v32;
        v32 = v31 < 512l;
        bool v33;
        v33 = v32 == false;
        if (v33){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v32);
        } else {
        }
        assert("Tensor range check" && 0 <= v31 && v31 < 512l);
        assert("Tensor range check" && 0 <= v30 && v30 < 128l);
        int v35;
        v35 = v30 + v17;
        int v36;
        v36 = 128l * v31;
        int v37;
        v37 = v36 + v35;
        v14[v37] = 0.0f;
        v25 += 512l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    switch (v3.tag) {
        case 0: { // None
            break;
        }
        case 1: { // Some
            Union5 v38 = v3.case1.v0; bool v39 = v3.case1.v1; static_array<Union6,2l> v40 = v3.case1.v2; int v41 = v3.case1.v3; static_array<int,2l> v42 = v3.case1.v4; int v43 = v3.case1.v5; static_array_list<Union7,32l> v44 = v3.case1.v6;
            int v45;
            v45 = threadIdx.x;
            assert("Tensor range check" && 0 <= v45 && v45 < 512l);
            int v46;
            v46 = 128l * v45;
            int v47;
            v47 = v46 + v17;
            static_array_list<Union11,10l> v48;
            v48 = static_array_list<Union11,10l>{};
            int v50;
            v50 = v44.length;
            int v51;
            v51 = 0l;
            while (while_method_1(v50, v51)){
                Union7 v53;
                v53 = v44[v51];
                Union12 v72;
                switch (v53.tag) {
                    case 0: { // CommunityCardIs
                        Union6 v62 = v53.case0.v0;
                        Union11 v63;
                        v63 = Union11{Union11_1{v62}};
                        v72 = Union12{Union12_1{v63}};
                        break;
                    }
                    case 1: { // PlayerAction
                        int v65 = v53.case1.v0; Union1 v66 = v53.case1.v1;
                        Union11 v67;
                        v67 = Union11{Union11_0{v66}};
                        v72 = Union12{Union12_1{v67}};
                        break;
                    }
                    case 2: { // PlayerGotCard
                        int v55 = v53.case2.v0; Union6 v56 = v53.case2.v1;
                        bool v57;
                        v57 = v55 == v41;
                        if (v57){
                            Union11 v58;
                            v58 = Union11{Union11_1{v56}};
                            v72 = Union12{Union12_1{v58}};
                        } else {
                            v72 = Union12{Union12_0{}};
                        }
                        break;
                    }
                    default: {
                        v72 = Union12{Union12_0{}};
                    }
                }
                switch (v72.tag) {
                    case 0: { // None
                        break;
                    }
                    case 1: { // Some
                        Union11 v73 = v72.case1.v0;
                        v48.push(v73);
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                v51 += 1l ;
            }
            float * v74;
            v74 = v14+v47;
            int v76;
            v76 = v48.length;
            bool v77;
            v77 = v76 == 0l;
            if (v77){
                v74[0l] = 1.0f;
            } else {
            }
            int v78;
            v78 = v48.length;
            int v79;
            v79 = 0l;
            while (while_method_1(v78, v79)){
                Union11 v81;
                v81 = v48[v79];
                int v83;
                v83 = v79 * 6l;
                int v84;
                v84 = 1l + v83;
                switch (v81.tag) {
                    case 0: { // C1of2
                        Union1 v85 = v81.case0.v0;
                        switch (v85.tag) {
                            case 0: { // Call
                                v74[v84] = 1.0f;
                                break;
                            }
                            case 1: { // Fold
                                int v86;
                                v86 = v84 + 1l;
                                v74[v86] = 1.0f;
                                break;
                            }
                            case 2: { // Raise
                                int v87;
                                v87 = v84 + 2l;
                                v74[v87] = 1.0f;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        break;
                    }
                    case 1: { // C2of2
                        Union6 v88 = v81.case1.v0;
                        int v89;
                        v89 = v84 + 3l;
                        switch (v88.tag) {
                            case 0: { // Jack
                                v74[v89] = 1.0f;
                                break;
                            }
                            case 1: { // King
                                int v90;
                                v90 = v89 + 1l;
                                v74[v90] = 1.0f;
                                break;
                            }
                            case 2: { // Queen
                                int v91;
                                v91 = v89 + 2l;
                                v74[v91] = 1.0f;
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
                v79 += 1l ;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v92;
    v92 = 0l;
    int v93;
    v93 = 4l;
    int v94;
    v94 = int_range_24(v93, v92, v23);
    extern __shared__ unsigned char v95[];
    int * v96;
    v96 = reinterpret_cast<int *>(&v95[0ull]);
    int v98;
    v98 = threadIdx.x;
    bool v99;
    v99 = v98 == 0l;
    if (v99){
        v96[0l] = v94;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v100;
    v100 = v96[0l];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v101;
    v101 = reinterpret_cast<float *>(&v1[0ull]);
    float * v103;
    v103 = reinterpret_cast<float *>(&v2[0ull]);
    assert("Tensor range check" && 0 <= v100 && v100 < 4l);
    int v105;
    v105 = 16384l * v100;
    float * v106;
    v106 = reinterpret_cast<float *>(&v1[6291456ull]);
    int v108;
    v108 = blockIdx.x;
    assert("Tensor range check" && 0 <= v108 && v108 < 24l);
    int v109;
    v109 = 65536l * v108;
    int v110;
    v110 = blockIdx.x;
    assert("Tensor range check" && 0 <= v110 && v110 < 24l);
    int v111;
    v111 = 65536l * v110;
    method_25(v103, v105, v106, v111, v101, v109);
    unsigned int * v112;
    v112 = reinterpret_cast<unsigned int *>(&v1[12582912ull]);
    assert("Tensor range check" && 0 <= v100 && v100 < 4l);
    int v114;
    v114 = 12288l * v100;
    method_26(v112, v114, v106);
    int * v115;
    v115 = reinterpret_cast<int *>(&v2[262144ull]);
    float * v117;
    v117 = reinterpret_cast<float *>(&v2[262160ull]);
    float * v119;
    v119 = reinterpret_cast<float *>(&v2[524304ull]);
    float * v121;
    v121 = reinterpret_cast<float *>(&v2[786448ull]);
    float * v123;
    v123 = reinterpret_cast<float *>(&v2[1048592ull]);
    float * v125;
    v125 = reinterpret_cast<float *>(&v2[1310736ull]);
    float * v127;
    v127 = reinterpret_cast<float *>(&v2[1572880ull]);
    float * v129;
    v129 = reinterpret_cast<float *>(&v2[1835024ull]);
    int * v131;
    v131 = reinterpret_cast<int *>(&v1[12779520ull]);
    float * v133;
    v133 = reinterpret_cast<float *>(&v1[15925248ull]);
    int * v135;
    v135 = reinterpret_cast<int *>(&v1[19070976ull]);
    int * v137;
    v137 = reinterpret_cast<int *>(&v1[22216704ull]);
    double * v139;
    v139 = reinterpret_cast<double *>(&v1[25362432ull]);
    double * v141;
    v141 = reinterpret_cast<double *>(&v1[37945344ull]);
    double * v143;
    v143 = reinterpret_cast<double *>(&v2[2097168ull]);
    double * v145;
    v145 = reinterpret_cast<double *>(&v2[2883600ull]);
    int * v147;
    v147 = reinterpret_cast<int *>(&v2[3670032ull]);
    asm("barrier.cta.sync %0;" :: "r"(0l));
    unsigned int * v149;
    v149 = reinterpret_cast<unsigned int *>(&v1[12582912ull]);
    int v151;
    v151 = blockIdx.x;
    int v152;
    v152 = threadIdx.x;
    assert("Tensor range check" && 0 <= v100 && v100 < 4l);
    assert("Tensor range check" && 0 <= v151 && v151 < 24l);
    assert("Tensor range check" && 0 <= v152 && v152 < 512l);
    int v153;
    v153 = 512l * v151;
    int v154;
    v154 = v153 + v152;
    int v155;
    v155 = v114 + v154;
    unsigned int v156;
    v156 = v149[v155];
    int * v157;
    v157 = reinterpret_cast<int *>(&v2[262144ull]);
    float * v159;
    v159 = reinterpret_cast<float *>(&v2[262160ull]);
    float * v161;
    v161 = reinterpret_cast<float *>(&v2[524304ull]);
    float * v163;
    v163 = reinterpret_cast<float *>(&v2[786448ull]);
    float * v165;
    v165 = reinterpret_cast<float *>(&v2[1048592ull]);
    float * v167;
    v167 = reinterpret_cast<float *>(&v2[1310736ull]);
    float * v169;
    v169 = reinterpret_cast<float *>(&v2[1572880ull]);
    float * v171;
    v171 = reinterpret_cast<float *>(&v2[1835024ull]);
    int v173;
    v173 = (int)v156;
    float v174; int v175;
    Tuple8 tmp25 = method_27(v0, v157, v159, v161, v163, v165, v167, v169, v171, v173, v100);
    v174 = tmp25.v0; v175 = tmp25.v1;
    bool v176;
    v176 = 0l == v175;
    if (v176){
        return Union10{Union10_1{}};
    } else {
        bool v178;
        v178 = 1l == v175;
        if (v178){
            return Union10{Union10_0{}};
        } else {
            bool v180;
            v180 = 2l == v175;
            if (v180){
                return Union10{Union10_2{}};
            } else {
                printf("%s\n", "Invalid output id in the Leduc model.");
                __trap();
            }
        }
    }
}
__device__ void method_28(Union1 v0){
    switch (v0.tag) {
        case 0: { // Call
            printf("%s","Call");
            return ;
            break;
        }
        case 1: { // Fold
            printf("%s","Fold");
            return ;
            break;
        }
        case 2: { // Raise
            printf("%s","Raise");
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ int tag_30(Union6 v0){
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
__device__ bool is_pair_31(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
__device__ Tuple7 order_32(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple7{v1, v0};
    } else {
        return Tuple7{v0, v1};
    }
}
__device__ Union13 compare_hands_29(Union5 v0, bool v1, static_array<Union6,2l> v2, int v3, static_array<int,2l> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            __trap();
            break;
        }
        case 1: { // Some
            Union6 v7 = v0.case1.v0;
            int v8;
            v8 = tag_30(v7);
            Union6 v9;
            v9 = v2[0l];
            int v11;
            v11 = tag_30(v9);
            Union6 v12;
            v12 = v2[1l];
            int v14;
            v14 = tag_30(v12);
            bool v15;
            v15 = is_pair_31(v8, v11);
            bool v16;
            v16 = is_pair_31(v8, v14);
            if (v15){
                if (v16){
                    bool v17;
                    v17 = v11 < v14;
                    if (v17){
                        return Union13{Union13_2{}};
                    } else {
                        bool v19;
                        v19 = v11 > v14;
                        if (v19){
                            return Union13{Union13_1{}};
                        } else {
                            return Union13{Union13_0{}};
                        }
                    }
                } else {
                    return Union13{Union13_1{}};
                }
            } else {
                if (v16){
                    return Union13{Union13_2{}};
                } else {
                    int v27; int v28;
                    Tuple7 tmp36 = order_32(v8, v11);
                    v27 = tmp36.v0; v28 = tmp36.v1;
                    int v29; int v30;
                    Tuple7 tmp37 = order_32(v8, v14);
                    v29 = tmp37.v0; v30 = tmp37.v1;
                    bool v31;
                    v31 = v27 < v29;
                    Union13 v37;
                    if (v31){
                        v37 = Union13{Union13_2{}};
                    } else {
                        bool v33;
                        v33 = v27 > v29;
                        if (v33){
                            v37 = Union13{Union13_1{}};
                        } else {
                            v37 = Union13{Union13_0{}};
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
                            return Union13{Union13_2{}};
                        } else {
                            bool v41;
                            v41 = v28 > v30;
                            if (v41){
                                return Union13{Union13_1{}};
                            } else {
                                return Union13{Union13_0{}};
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
__device__ void play_loop_20(unsigned char * v0, unsigned long long v1, unsigned char * v2, unsigned long long v3, unsigned int & v4, Union3 & v5, static_array_list<Union7,32l> & v6, static_array<Union2,2l> & v7, curandStatePhilox4_32_10_t & v8, Union8 & v9, Union4 v10){
    static_array_list<Union7,32l> & v11 = v6;
    Union3 v12;
    v12 = Union3{Union3_1{v10}};
    Union3 v13;
    v13 = v12;
    while (while_method_2(v13)){
        Union3 v450;
        switch (v13.tag) {
            case 0: { // None
                v450 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v15 = v13.case1.v0;
                switch (v15.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v401 = v15.case0.v0; bool v402 = v15.case0.v1; static_array<Union6,2l> v403 = v15.case0.v2; int v404 = v15.case0.v3; static_array<int,2l> v405 = v15.case0.v4; int v406 = v15.case0.v5;
                        unsigned int v407 = v4;
                        Union6 v408; unsigned int v409;
                        Tuple6 tmp11 = draw_card_21(v8, v407);
                        v408 = tmp11.v0; v409 = tmp11.v1;
                        v4 = v409;
                        Union7 v410;
                        v410 = Union7{Union7_0{v408}};
                        v11.push(v410);
                        int v411;
                        v411 = 2l;
                        int v412; int v413;
                        Tuple7 tmp12 = Tuple7{0l, 0l};
                        v412 = tmp12.v0; v413 = tmp12.v1;
                        while (while_method_0(v412)){
                            int v415;
                            v415 = v405[v412];
                            bool v417;
                            v417 = v413 >= v415;
                            int v418;
                            if (v417){
                                v418 = v413;
                            } else {
                                v418 = v415;
                            }
                            v413 = v418;
                            v412 += 1l ;
                        }
                        static_array<int,2l> v419;
                        int v421;
                        v421 = 0l;
                        while (while_method_0(v421)){
                            v419[v421] = v413;
                            v421 += 1l ;
                        }
                        Union5 v423;
                        v423 = Union5{Union5_1{v408}};
                        Union4 v424;
                        v424 = Union4{Union4_2{v423, true, v403, 0l, v419, v411}};
                        v450 = Union3{Union3_1{v424}};
                        break;
                    }
                    case 1: { // ChanceInit
                        unsigned int v426 = v4;
                        Union6 v427; unsigned int v428;
                        Tuple6 tmp13 = draw_card_21(v8, v426);
                        v427 = tmp13.v0; v428 = tmp13.v1;
                        v4 = v428;
                        unsigned int v429 = v4;
                        Union6 v430; unsigned int v431;
                        Tuple6 tmp14 = draw_card_21(v8, v429);
                        v430 = tmp14.v0; v431 = tmp14.v1;
                        v4 = v431;
                        Union7 v432;
                        v432 = Union7{Union7_2{0l, v427}};
                        v11.push(v432);
                        Union7 v433;
                        v433 = Union7{Union7_2{1l, v430}};
                        v11.push(v433);
                        int v434;
                        v434 = 2l;
                        static_array<int,2l> v435;
                        v435[0l] = 1l;
                        v435[1l] = 1l;
                        static_array<Union6,2l> v437;
                        v437[0l] = v427;
                        v437[1l] = v430;
                        Union5 v439;
                        v439 = Union5{Union5_0{}};
                        Union4 v440;
                        v440 = Union4{Union4_2{v439, true, v437, 0l, v435, v434}};
                        v450 = Union3{Union3_1{v440}};
                        break;
                    }
                    case 2: { // Round
                        Union5 v49 = v15.case2.v0; bool v50 = v15.case2.v1; static_array<Union6,2l> v51 = v15.case2.v2; int v52 = v15.case2.v3; static_array<int,2l> v53 = v15.case2.v4; int v54 = v15.case2.v5;
                        static_array<Union2,2l> v55 = v7;
                        Union2 v56;
                        v56 = v55[v52];
                        switch (v56.tag) {
                            case 0: { // Computer
                                bool v58;
                                v58 = 3866640ull == v3;
                                bool v59;
                                v59 = v58 == false;
                                if (v59){
                                    assert("The params needs to have matching offsets." && v58);
                                } else {
                                }
                                bool v61;
                                v61 = 50528256ull == v1;
                                bool v62;
                                v62 = v61 == false;
                                if (v62){
                                    assert("The outputs needs to have matching offsets." && v61);
                                } else {
                                }
                                static_array_list<Union7,32l> & v64 = v6;
                                cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v65 = console_lock;
                                auto v66 = cooperative_groups::coalesced_threads();
                                v65.acquire();
                                printf("%s\n","Running the GPU model.");
                                v65.release();
                                v66.sync() ;
                                Union9 v69;
                                v69 = Union9{Union9_1{v49, v50, v51, v52, v53, v54, v64}};
                                Union10 v70;
                                v70 = noinline_run_23(v8, v0, v2, v69);
                                Union1 v93;
                                switch (v70.tag) {
                                    case 0: { // AA_Call
                                        v93 = Union1{Union1_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v71;
                                        v71 = v53[0l];
                                        int v73; int v74;
                                        Tuple7 tmp26 = Tuple7{1l, v71};
                                        v73 = tmp26.v0; v74 = tmp26.v1;
                                        while (while_method_0(v73)){
                                            int v76;
                                            v76 = v53[v73];
                                            bool v78;
                                            v78 = v74 >= v76;
                                            int v79;
                                            if (v78){
                                                v79 = v74;
                                            } else {
                                                v79 = v76;
                                            }
                                            v74 = v79;
                                            v73 += 1l ;
                                        }
                                        int v80;
                                        v80 = v53[v52];
                                        bool v82;
                                        v82 = v80 == v74;
                                        if (v82){
                                            v93 = Union1{Union1_0{}};
                                        } else {
                                            v93 = Union1{Union1_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v87;
                                        v87 = v54 > 0l;
                                        if (v87){
                                            v93 = Union1{Union1_2{}};
                                        } else {
                                            v93 = Union1{Union1_0{}};
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v94 = console_lock;
                                auto v95 = cooperative_groups::coalesced_threads();
                                v94.acquire();
                                printf("%s","The action is: ");
                                v94.release();
                                v95.sync() ;
                                cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v98 = console_lock;
                                auto v99 = cooperative_groups::coalesced_threads();
                                v98.acquire();
                                printf("");
                                method_28(v93);
                                printf("\n");
                                v98.release();
                                v99.sync() ;
                                Union7 v102;
                                v102 = Union7{Union7_1{v52, v93}};
                                v11.push(v102);
                                Union4 v188;
                                switch (v49.tag) {
                                    case 0: { // None
                                        switch (v93.tag) {
                                            case 0: { // Call
                                                if (v50){
                                                    bool v152;
                                                    v152 = v52 == 0l;
                                                    int v153;
                                                    if (v152){
                                                        v153 = 1l;
                                                    } else {
                                                        v153 = 0l;
                                                    }
                                                    v188 = Union4{Union4_2{v49, false, v51, v153, v53, v54}};
                                                } else {
                                                    v188 = Union4{Union4_0{v49, v50, v51, v52, v53, v54}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v188 = Union4{Union4_5{v49, v50, v51, v52, v53, v54}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                bool v157;
                                                v157 = v54 > 0l;
                                                if (v157){
                                                    bool v158;
                                                    v158 = v52 == 0l;
                                                    int v159;
                                                    if (v158){
                                                        v159 = 1l;
                                                    } else {
                                                        v159 = 0l;
                                                    }
                                                    int v160;
                                                    v160 = -1l + v54;
                                                    int v161; int v162;
                                                    Tuple7 tmp27 = Tuple7{0l, 0l};
                                                    v161 = tmp27.v0; v162 = tmp27.v1;
                                                    while (while_method_0(v161)){
                                                        int v164;
                                                        v164 = v53[v161];
                                                        bool v166;
                                                        v166 = v162 >= v164;
                                                        int v167;
                                                        if (v166){
                                                            v167 = v162;
                                                        } else {
                                                            v167 = v164;
                                                        }
                                                        v162 = v167;
                                                        v161 += 1l ;
                                                    }
                                                    static_array<int,2l> v168;
                                                    int v170;
                                                    v170 = 0l;
                                                    while (while_method_0(v170)){
                                                        v168[v170] = v162;
                                                        v170 += 1l ;
                                                    }
                                                    static_array<int,2l> v172;
                                                    int v174;
                                                    v174 = 0l;
                                                    while (while_method_0(v174)){
                                                        int v176;
                                                        v176 = v168[v174];
                                                        bool v178;
                                                        v178 = v174 == v52;
                                                        int v180;
                                                        if (v178){
                                                            int v179;
                                                            v179 = v176 + 2l;
                                                            v180 = v179;
                                                        } else {
                                                            v180 = v176;
                                                        }
                                                        v172[v174] = v180;
                                                        v174 += 1l ;
                                                    }
                                                    v188 = Union4{Union4_2{v49, false, v51, v159, v172, v160}};
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
                                        Union6 v103 = v49.case1.v0;
                                        switch (v93.tag) {
                                            case 0: { // Call
                                                if (v50){
                                                    bool v105;
                                                    v105 = v52 == 0l;
                                                    int v106;
                                                    if (v105){
                                                        v106 = 1l;
                                                    } else {
                                                        v106 = 0l;
                                                    }
                                                    v188 = Union4{Union4_2{v49, false, v51, v106, v53, v54}};
                                                } else {
                                                    int v108; int v109;
                                                    Tuple7 tmp28 = Tuple7{0l, 0l};
                                                    v108 = tmp28.v0; v109 = tmp28.v1;
                                                    while (while_method_0(v108)){
                                                        int v111;
                                                        v111 = v53[v108];
                                                        bool v113;
                                                        v113 = v109 >= v111;
                                                        int v114;
                                                        if (v113){
                                                            v114 = v109;
                                                        } else {
                                                            v114 = v111;
                                                        }
                                                        v109 = v114;
                                                        v108 += 1l ;
                                                    }
                                                    static_array<int,2l> v115;
                                                    int v117;
                                                    v117 = 0l;
                                                    while (while_method_0(v117)){
                                                        v115[v117] = v109;
                                                        v117 += 1l ;
                                                    }
                                                    v188 = Union4{Union4_4{v49, v50, v51, v52, v115, v54}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v188 = Union4{Union4_5{v49, v50, v51, v52, v53, v54}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                bool v121;
                                                v121 = v54 > 0l;
                                                if (v121){
                                                    bool v122;
                                                    v122 = v52 == 0l;
                                                    int v123;
                                                    if (v122){
                                                        v123 = 1l;
                                                    } else {
                                                        v123 = 0l;
                                                    }
                                                    int v124;
                                                    v124 = -1l + v54;
                                                    int v125; int v126;
                                                    Tuple7 tmp29 = Tuple7{0l, 0l};
                                                    v125 = tmp29.v0; v126 = tmp29.v1;
                                                    while (while_method_0(v125)){
                                                        int v128;
                                                        v128 = v53[v125];
                                                        bool v130;
                                                        v130 = v126 >= v128;
                                                        int v131;
                                                        if (v130){
                                                            v131 = v126;
                                                        } else {
                                                            v131 = v128;
                                                        }
                                                        v126 = v131;
                                                        v125 += 1l ;
                                                    }
                                                    static_array<int,2l> v132;
                                                    int v134;
                                                    v134 = 0l;
                                                    while (while_method_0(v134)){
                                                        v132[v134] = v126;
                                                        v134 += 1l ;
                                                    }
                                                    static_array<int,2l> v136;
                                                    int v138;
                                                    v138 = 0l;
                                                    while (while_method_0(v138)){
                                                        int v140;
                                                        v140 = v132[v138];
                                                        bool v142;
                                                        v142 = v138 == v52;
                                                        int v144;
                                                        if (v142){
                                                            int v143;
                                                            v143 = v140 + 4l;
                                                            v144 = v143;
                                                        } else {
                                                            v144 = v140;
                                                        }
                                                        v136[v138] = v144;
                                                        v138 += 1l ;
                                                    }
                                                    v188 = Union4{Union4_2{v49, false, v51, v123, v136, v124}};
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
                                v450 = Union3{Union3_1{v188}};
                                break;
                            }
                            case 1: { // Human
                                Union8 v190;
                                v190 = Union8{Union8_2{v49, v50, v51, v52, v53, v54}};
                                v9 = v190;
                                Union3 v191;
                                v191 = Union3{Union3_1{v15}};
                                v5 = v191;
                                v450 = Union3{Union3_0{}};
                                break;
                            }
                            case 2: { // Random
                                static_array_list<Union1,3l> v193;
                                v193 = static_array_list<Union1,3l>{};
                                v193.unsafe_set_length(1l);
                                Union1 v195;
                                v195 = Union1{Union1_0{}};
                                v193[0l] = v195;
                                int v197;
                                v197 = v53[0l];
                                int v199;
                                v199 = v53[1l];
                                bool v201;
                                v201 = v197 == v199;
                                bool v202;
                                v202 = v201 != true;
                                if (v202){
                                    Union1 v203;
                                    v203 = Union1{Union1_1{}};
                                    v193.push(v203);
                                } else {
                                }
                                bool v204;
                                v204 = v54 > 0l;
                                if (v204){
                                    Union1 v205;
                                    v205 = Union1{Union1_2{}};
                                    v193.push(v205);
                                } else {
                                }
                                int v206;
                                v206 = v193.length;
                                int v207;
                                v207 = v206 - 1l;
                                int v208;
                                v208 = 0l;
                                while (while_method_1(v207, v208)){
                                    int v210;
                                    v210 = v193.length;
                                    int v211;
                                    v211 = int_range_24(v210, v208, v8);
                                    Union1 v212;
                                    v212 = v193[v208];
                                    Union1 v214;
                                    v214 = v193[v211];
                                    v193[v208] = v214;
                                    v193[v211] = v212;
                                    v208 += 1l ;
                                }
                                Union1 v216;
                                v216 = v193.pop();
                                Union7 v217;
                                v217 = Union7{Union7_1{v52, v216}};
                                v11.push(v217);
                                Union4 v301;
                                switch (v49.tag) {
                                    case 0: { // None
                                        switch (v216.tag) {
                                            case 0: { // Call
                                                if (v50){
                                                    bool v266;
                                                    v266 = v52 == 0l;
                                                    int v267;
                                                    if (v266){
                                                        v267 = 1l;
                                                    } else {
                                                        v267 = 0l;
                                                    }
                                                    v301 = Union4{Union4_2{v49, false, v51, v267, v53, v54}};
                                                } else {
                                                    v301 = Union4{Union4_0{v49, v50, v51, v52, v53, v54}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v301 = Union4{Union4_5{v49, v50, v51, v52, v53, v54}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                if (v204){
                                                    bool v271;
                                                    v271 = v52 == 0l;
                                                    int v272;
                                                    if (v271){
                                                        v272 = 1l;
                                                    } else {
                                                        v272 = 0l;
                                                    }
                                                    int v273;
                                                    v273 = -1l + v54;
                                                    int v274; int v275;
                                                    Tuple7 tmp30 = Tuple7{0l, 0l};
                                                    v274 = tmp30.v0; v275 = tmp30.v1;
                                                    while (while_method_0(v274)){
                                                        int v277;
                                                        v277 = v53[v274];
                                                        bool v279;
                                                        v279 = v275 >= v277;
                                                        int v280;
                                                        if (v279){
                                                            v280 = v275;
                                                        } else {
                                                            v280 = v277;
                                                        }
                                                        v275 = v280;
                                                        v274 += 1l ;
                                                    }
                                                    static_array<int,2l> v281;
                                                    int v283;
                                                    v283 = 0l;
                                                    while (while_method_0(v283)){
                                                        v281[v283] = v275;
                                                        v283 += 1l ;
                                                    }
                                                    static_array<int,2l> v285;
                                                    int v287;
                                                    v287 = 0l;
                                                    while (while_method_0(v287)){
                                                        int v289;
                                                        v289 = v281[v287];
                                                        bool v291;
                                                        v291 = v287 == v52;
                                                        int v293;
                                                        if (v291){
                                                            int v292;
                                                            v292 = v289 + 2l;
                                                            v293 = v292;
                                                        } else {
                                                            v293 = v289;
                                                        }
                                                        v285[v287] = v293;
                                                        v287 += 1l ;
                                                    }
                                                    v301 = Union4{Union4_2{v49, false, v51, v272, v285, v273}};
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
                                        Union6 v218 = v49.case1.v0;
                                        switch (v216.tag) {
                                            case 0: { // Call
                                                if (v50){
                                                    bool v220;
                                                    v220 = v52 == 0l;
                                                    int v221;
                                                    if (v220){
                                                        v221 = 1l;
                                                    } else {
                                                        v221 = 0l;
                                                    }
                                                    v301 = Union4{Union4_2{v49, false, v51, v221, v53, v54}};
                                                } else {
                                                    int v223; int v224;
                                                    Tuple7 tmp31 = Tuple7{0l, 0l};
                                                    v223 = tmp31.v0; v224 = tmp31.v1;
                                                    while (while_method_0(v223)){
                                                        int v226;
                                                        v226 = v53[v223];
                                                        bool v228;
                                                        v228 = v224 >= v226;
                                                        int v229;
                                                        if (v228){
                                                            v229 = v224;
                                                        } else {
                                                            v229 = v226;
                                                        }
                                                        v224 = v229;
                                                        v223 += 1l ;
                                                    }
                                                    static_array<int,2l> v230;
                                                    int v232;
                                                    v232 = 0l;
                                                    while (while_method_0(v232)){
                                                        v230[v232] = v224;
                                                        v232 += 1l ;
                                                    }
                                                    v301 = Union4{Union4_4{v49, v50, v51, v52, v230, v54}};
                                                }
                                                break;
                                            }
                                            case 1: { // Fold
                                                v301 = Union4{Union4_5{v49, v50, v51, v52, v53, v54}};
                                                break;
                                            }
                                            case 2: { // Raise
                                                if (v204){
                                                    bool v236;
                                                    v236 = v52 == 0l;
                                                    int v237;
                                                    if (v236){
                                                        v237 = 1l;
                                                    } else {
                                                        v237 = 0l;
                                                    }
                                                    int v238;
                                                    v238 = -1l + v54;
                                                    int v239; int v240;
                                                    Tuple7 tmp32 = Tuple7{0l, 0l};
                                                    v239 = tmp32.v0; v240 = tmp32.v1;
                                                    while (while_method_0(v239)){
                                                        int v242;
                                                        v242 = v53[v239];
                                                        bool v244;
                                                        v244 = v240 >= v242;
                                                        int v245;
                                                        if (v244){
                                                            v245 = v240;
                                                        } else {
                                                            v245 = v242;
                                                        }
                                                        v240 = v245;
                                                        v239 += 1l ;
                                                    }
                                                    static_array<int,2l> v246;
                                                    int v248;
                                                    v248 = 0l;
                                                    while (while_method_0(v248)){
                                                        v246[v248] = v240;
                                                        v248 += 1l ;
                                                    }
                                                    static_array<int,2l> v250;
                                                    int v252;
                                                    v252 = 0l;
                                                    while (while_method_0(v252)){
                                                        int v254;
                                                        v254 = v246[v252];
                                                        bool v256;
                                                        v256 = v252 == v52;
                                                        int v258;
                                                        if (v256){
                                                            int v257;
                                                            v257 = v254 + 4l;
                                                            v258 = v257;
                                                        } else {
                                                            v258 = v254;
                                                        }
                                                        v250[v252] = v258;
                                                        v252 += 1l ;
                                                    }
                                                    v301 = Union4{Union4_2{v49, false, v51, v237, v250, v238}};
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
                                v450 = Union3{Union3_1{v301}};
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v306 = v15.case3.v0; bool v307 = v15.case3.v1; static_array<Union6,2l> v308 = v15.case3.v2; int v309 = v15.case3.v3; static_array<int,2l> v310 = v15.case3.v4; int v311 = v15.case3.v5; Union1 v312 = v15.case3.v6;
                        Union7 v313;
                        v313 = Union7{Union7_1{v309, v312}};
                        v11.push(v313);
                        Union4 v399;
                        switch (v306.tag) {
                            case 0: { // None
                                switch (v312.tag) {
                                    case 0: { // Call
                                        if (v307){
                                            bool v363;
                                            v363 = v309 == 0l;
                                            int v364;
                                            if (v363){
                                                v364 = 1l;
                                            } else {
                                                v364 = 0l;
                                            }
                                            v399 = Union4{Union4_2{v306, false, v308, v364, v310, v311}};
                                        } else {
                                            v399 = Union4{Union4_0{v306, v307, v308, v309, v310, v311}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v399 = Union4{Union4_5{v306, v307, v308, v309, v310, v311}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v368;
                                        v368 = v311 > 0l;
                                        if (v368){
                                            bool v369;
                                            v369 = v309 == 0l;
                                            int v370;
                                            if (v369){
                                                v370 = 1l;
                                            } else {
                                                v370 = 0l;
                                            }
                                            int v371;
                                            v371 = -1l + v311;
                                            int v372; int v373;
                                            Tuple7 tmp33 = Tuple7{0l, 0l};
                                            v372 = tmp33.v0; v373 = tmp33.v1;
                                            while (while_method_0(v372)){
                                                int v375;
                                                v375 = v310[v372];
                                                bool v377;
                                                v377 = v373 >= v375;
                                                int v378;
                                                if (v377){
                                                    v378 = v373;
                                                } else {
                                                    v378 = v375;
                                                }
                                                v373 = v378;
                                                v372 += 1l ;
                                            }
                                            static_array<int,2l> v379;
                                            int v381;
                                            v381 = 0l;
                                            while (while_method_0(v381)){
                                                v379[v381] = v373;
                                                v381 += 1l ;
                                            }
                                            static_array<int,2l> v383;
                                            int v385;
                                            v385 = 0l;
                                            while (while_method_0(v385)){
                                                int v387;
                                                v387 = v379[v385];
                                                bool v389;
                                                v389 = v385 == v309;
                                                int v391;
                                                if (v389){
                                                    int v390;
                                                    v390 = v387 + 2l;
                                                    v391 = v390;
                                                } else {
                                                    v391 = v387;
                                                }
                                                v383[v385] = v391;
                                                v385 += 1l ;
                                            }
                                            v399 = Union4{Union4_2{v306, false, v308, v370, v383, v371}};
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
                                Union6 v314 = v306.case1.v0;
                                switch (v312.tag) {
                                    case 0: { // Call
                                        if (v307){
                                            bool v316;
                                            v316 = v309 == 0l;
                                            int v317;
                                            if (v316){
                                                v317 = 1l;
                                            } else {
                                                v317 = 0l;
                                            }
                                            v399 = Union4{Union4_2{v306, false, v308, v317, v310, v311}};
                                        } else {
                                            int v319; int v320;
                                            Tuple7 tmp34 = Tuple7{0l, 0l};
                                            v319 = tmp34.v0; v320 = tmp34.v1;
                                            while (while_method_0(v319)){
                                                int v322;
                                                v322 = v310[v319];
                                                bool v324;
                                                v324 = v320 >= v322;
                                                int v325;
                                                if (v324){
                                                    v325 = v320;
                                                } else {
                                                    v325 = v322;
                                                }
                                                v320 = v325;
                                                v319 += 1l ;
                                            }
                                            static_array<int,2l> v326;
                                            int v328;
                                            v328 = 0l;
                                            while (while_method_0(v328)){
                                                v326[v328] = v320;
                                                v328 += 1l ;
                                            }
                                            v399 = Union4{Union4_4{v306, v307, v308, v309, v326, v311}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v399 = Union4{Union4_5{v306, v307, v308, v309, v310, v311}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        bool v332;
                                        v332 = v311 > 0l;
                                        if (v332){
                                            bool v333;
                                            v333 = v309 == 0l;
                                            int v334;
                                            if (v333){
                                                v334 = 1l;
                                            } else {
                                                v334 = 0l;
                                            }
                                            int v335;
                                            v335 = -1l + v311;
                                            int v336; int v337;
                                            Tuple7 tmp35 = Tuple7{0l, 0l};
                                            v336 = tmp35.v0; v337 = tmp35.v1;
                                            while (while_method_0(v336)){
                                                int v339;
                                                v339 = v310[v336];
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
                                            while (while_method_0(v345)){
                                                v343[v345] = v337;
                                                v345 += 1l ;
                                            }
                                            static_array<int,2l> v347;
                                            int v349;
                                            v349 = 0l;
                                            while (while_method_0(v349)){
                                                int v351;
                                                v351 = v343[v349];
                                                bool v353;
                                                v353 = v349 == v309;
                                                int v355;
                                                if (v353){
                                                    int v354;
                                                    v354 = v351 + 4l;
                                                    v355 = v354;
                                                } else {
                                                    v355 = v351;
                                                }
                                                v347[v349] = v355;
                                                v349 += 1l ;
                                            }
                                            v399 = Union4{Union4_2{v306, false, v308, v334, v347, v335}};
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
                        v450 = Union3{Union3_1{v399}};
                        break;
                    }
                    case 4: { // TerminalCall
                        Union5 v30 = v15.case4.v0; bool v31 = v15.case4.v1; static_array<Union6,2l> v32 = v15.case4.v2; int v33 = v15.case4.v3; static_array<int,2l> v34 = v15.case4.v4; int v35 = v15.case4.v5;
                        int v36;
                        v36 = v34[v33];
                        Union13 v38;
                        v38 = compare_hands_29(v30, v31, v32, v33, v34, v35);
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
                        Union7 v45;
                        v45 = Union7{Union7_3{v32, v43, v44}};
                        v11.push(v45);
                        Union8 v46;
                        v46 = Union8{Union8_1{v30, v31, v32, v33, v34, v35}};
                        v9 = v46;
                        Union3 v47;
                        v47 = Union3{Union3_0{}};
                        v5 = v47;
                        v450 = Union3{Union3_0{}};
                        break;
                    }
                    case 5: { // TerminalFold
                        Union5 v16 = v15.case5.v0; bool v17 = v15.case5.v1; static_array<Union6,2l> v18 = v15.case5.v2; int v19 = v15.case5.v3; static_array<int,2l> v20 = v15.case5.v4; int v21 = v15.case5.v5;
                        int v22;
                        v22 = v20[v19];
                        bool v24;
                        v24 = v19 == 0l;
                        int v25;
                        if (v24){
                            v25 = 1l;
                        } else {
                            v25 = 0l;
                        }
                        Union7 v26;
                        v26 = Union7{Union7_3{v18, v22, v25}};
                        v11.push(v26);
                        Union8 v27;
                        v27 = Union8{Union8_1{v16, v17, v18, v19, v20, v21}};
                        v9 = v27;
                        Union3 v28;
                        v28 = Union3{Union3_0{}};
                        v5 = v28;
                        v450 = Union3{Union3_0{}};
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
        v13 = v450;
    }
    return ;
}
__device__ void f_34(unsigned char * v0, unsigned int v1){
    unsigned int * v2;
    v2 = (unsigned int *)(v0+0ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_35(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+4ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_36(unsigned char * v0){
    return ;
}
__device__ void f_38(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+0ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_40(unsigned char * v0, Union6 v1){
    int v2;
    v2 = v1.tag;
    f_38(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // Jack
            return f_36(v3);
            break;
        }
        case 1: { // King
            return f_36(v3);
            break;
        }
        case 2: { // Queen
            return f_36(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_39(unsigned char * v0, Union5 v1, bool v2, static_array<Union6,2l> v3, int v4, static_array<int,2l> v5, int v6){
    int v7;
    v7 = v1.tag;
    f_38(v0, v7);
    unsigned char * v8;
    v8 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // None
            f_36(v8);
            break;
        }
        case 1: { // Some
            Union6 v10 = v1.case1.v0;
            f_40(v8, v10);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
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
        Union6 v20;
        v20 = v3[v13];
        f_40(v18, v20);
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
        f_38(v29, v31);
        v24 += 1l ;
    }
    int * v33;
    v33 = (int *)(v0+32ull);
    v33[0l] = v6;
    return ;
}
__device__ void f_42(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+36ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_41(unsigned char * v0, Union5 v1, bool v2, static_array<Union6,2l> v3, int v4, static_array<int,2l> v5, int v6, Union1 v7){
    int v8;
    v8 = v1.tag;
    f_38(v0, v8);
    unsigned char * v9;
    v9 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // None
            f_36(v9);
            break;
        }
        case 1: { // Some
            Union6 v11 = v1.case1.v0;
            f_40(v9, v11);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
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
        Union6 v21;
        v21 = v3[v14];
        f_40(v19, v21);
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
        f_38(v30, v32);
        v25 += 1l ;
    }
    int * v34;
    v34 = (int *)(v0+32ull);
    v34[0l] = v6;
    int v36;
    v36 = v7.tag;
    f_42(v0, v36);
    unsigned char * v37;
    v37 = (unsigned char *)(v0+40ull);
    switch (v7.tag) {
        case 0: { // Call
            return f_36(v37);
            break;
        }
        case 1: { // Fold
            return f_36(v37);
            break;
        }
        case 2: { // Raise
            return f_36(v37);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_37(unsigned char * v0, Union4 v1){
    int v2;
    v2 = v1.tag;
    f_38(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // ChanceCommunityCard
            Union5 v5 = v1.case0.v0; bool v6 = v1.case0.v1; static_array<Union6,2l> v7 = v1.case0.v2; int v8 = v1.case0.v3; static_array<int,2l> v9 = v1.case0.v4; int v10 = v1.case0.v5;
            return f_39(v3, v5, v6, v7, v8, v9, v10);
            break;
        }
        case 1: { // ChanceInit
            return f_36(v3);
            break;
        }
        case 2: { // Round
            Union5 v11 = v1.case2.v0; bool v12 = v1.case2.v1; static_array<Union6,2l> v13 = v1.case2.v2; int v14 = v1.case2.v3; static_array<int,2l> v15 = v1.case2.v4; int v16 = v1.case2.v5;
            return f_39(v3, v11, v12, v13, v14, v15, v16);
            break;
        }
        case 3: { // RoundWithAction
            Union5 v17 = v1.case3.v0; bool v18 = v1.case3.v1; static_array<Union6,2l> v19 = v1.case3.v2; int v20 = v1.case3.v3; static_array<int,2l> v21 = v1.case3.v4; int v22 = v1.case3.v5; Union1 v23 = v1.case3.v6;
            return f_41(v3, v17, v18, v19, v20, v21, v22, v23);
            break;
        }
        case 4: { // TerminalCall
            Union5 v24 = v1.case4.v0; bool v25 = v1.case4.v1; static_array<Union6,2l> v26 = v1.case4.v2; int v27 = v1.case4.v3; static_array<int,2l> v28 = v1.case4.v4; int v29 = v1.case4.v5;
            return f_39(v3, v24, v25, v26, v27, v28, v29);
            break;
        }
        case 5: { // TerminalFold
            Union5 v30 = v1.case5.v0; bool v31 = v1.case5.v1; static_array<Union6,2l> v32 = v1.case5.v2; int v33 = v1.case5.v3; static_array<int,2l> v34 = v1.case5.v4; int v35 = v1.case5.v5;
            return f_39(v3, v30, v31, v32, v33, v34, v35);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_43(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+80ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_45(unsigned char * v0, int v1, Union1 v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0l] = v1;
    int v5;
    v5 = v2.tag;
    f_35(v0, v5);
    unsigned char * v6;
    v6 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // Call
            return f_36(v6);
            break;
        }
        case 1: { // Fold
            return f_36(v6);
            break;
        }
        case 2: { // Raise
            return f_36(v6);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_46(unsigned char * v0, int v1, Union6 v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0l] = v1;
    int v5;
    v5 = v2.tag;
    f_35(v0, v5);
    unsigned char * v6;
    v6 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // Jack
            return f_36(v6);
            break;
        }
        case 1: { // King
            return f_36(v6);
            break;
        }
        case 2: { // Queen
            return f_36(v6);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_47(unsigned char * v0, static_array<Union6,2l> v1, int v2, int v3){
    int v4;
    v4 = 0l;
    while (while_method_0(v4)){
        unsigned long long v6;
        v6 = (unsigned long long)v4;
        unsigned long long v7;
        v7 = v6 * 4ull;
        unsigned char * v8;
        v8 = (unsigned char *)(v0+v7);
        Union6 v10;
        v10 = v1[v4];
        f_40(v8, v10);
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
__device__ void f_44(unsigned char * v0, Union7 v1){
    int v2;
    v2 = v1.tag;
    f_38(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // CommunityCardIs
            Union6 v5 = v1.case0.v0;
            return f_40(v3, v5);
            break;
        }
        case 1: { // PlayerAction
            int v6 = v1.case1.v0; Union1 v7 = v1.case1.v1;
            return f_45(v3, v6, v7);
            break;
        }
        case 2: { // PlayerGotCard
            int v8 = v1.case2.v0; Union6 v9 = v1.case2.v1;
            return f_46(v3, v8, v9);
            break;
        }
        case 3: { // Showdown
            static_array<Union6,2l> v10 = v1.case3.v0; int v11 = v1.case3.v1; int v12 = v1.case3.v2;
            return f_47(v3, v10, v11, v12);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_48(unsigned char * v0, Union2 v1){
    int v2;
    v2 = v1.tag;
    f_38(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // Computer
            return f_36(v3);
            break;
        }
        case 1: { // Human
            return f_36(v3);
            break;
        }
        case 2: { // Random
            return f_36(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_49(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+1128ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_33(unsigned char * v0, unsigned int v1, Union3 v2, static_array_list<Union7,32l> v3, static_array<Union2,2l> v4, Union8 v5){
    f_34(v0, v1);
    int v6;
    v6 = v2.tag;
    f_35(v0, v6);
    unsigned char * v7;
    v7 = (unsigned char *)(v0+16ull);
    switch (v2.tag) {
        case 0: { // None
            f_36(v7);
            break;
        }
        case 1: { // Some
            Union4 v9 = v2.case1.v0;
            f_37(v7, v9);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v10;
    v10 = v3.length;
    f_43(v0, v10);
    int v11;
    v11 = v3.length;
    int v12;
    v12 = 0l;
    while (while_method_1(v11, v12)){
        unsigned long long v14;
        v14 = (unsigned long long)v12;
        unsigned long long v15;
        v15 = v14 * 32ull;
        unsigned long long v16;
        v16 = 96ull + v15;
        unsigned char * v17;
        v17 = (unsigned char *)(v0+v16);
        Union7 v19;
        v19 = v3[v12];
        f_44(v17, v19);
        v12 += 1l ;
    }
    int v21;
    v21 = 0l;
    while (while_method_0(v21)){
        unsigned long long v23;
        v23 = (unsigned long long)v21;
        unsigned long long v24;
        v24 = v23 * 4ull;
        unsigned long long v25;
        v25 = 1120ull + v24;
        unsigned char * v26;
        v26 = (unsigned char *)(v0+v25);
        Union2 v28;
        v28 = v4[v21];
        f_48(v26, v28);
        v21 += 1l ;
    }
    int v30;
    v30 = v5.tag;
    f_49(v0, v30);
    unsigned char * v31;
    v31 = (unsigned char *)(v0+1136ull);
    switch (v5.tag) {
        case 0: { // GameNotStarted
            return f_36(v31);
            break;
        }
        case 1: { // GameOver
            Union5 v33 = v5.case1.v0; bool v34 = v5.case1.v1; static_array<Union6,2l> v35 = v5.case1.v2; int v36 = v5.case1.v3; static_array<int,2l> v37 = v5.case1.v4; int v38 = v5.case1.v5;
            return f_39(v31, v33, v34, v35, v36, v37, v38);
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            Union5 v39 = v5.case2.v0; bool v40 = v5.case2.v1; static_array<Union6,2l> v41 = v5.case2.v2; int v42 = v5.case2.v3; static_array<int,2l> v43 = v5.case2.v4; int v44 = v5.case2.v5;
            return f_39(v31, v39, v40, v41, v42, v43, v44);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ inline bool while_method_8(){
    return true;
}
__device__ inline bool while_method_9(int v0){
    bool v1;
    v1 = v0 < 1024l;
    return v1;
}
__device__ inline bool while_method_10(int v0){
    bool v1;
    v1 = v0 < 16l;
    return v1;
}
__device__ inline bool while_method_11(Union3 v0){
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
__device__ float method_52(int * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int v8, int v9, int v10){
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
    while (while_method_5(v46)){
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
        while (while_method_5(v68)){
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
        while (while_method_5(v74)){
            int v76;
            v76 = 0l;
            while (while_method_4(v76)){
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
        while (while_method_5(v100)){
            int v102;
            v102 = 0l;
            while (while_method_4(v102)){
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
        while (while_method_5(v110)){
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
        while (while_method_5(v122)){
            int v124;
            v124 = 0l;
            while (while_method_4(v124)){
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
        while (while_method_5(v136)){
            int v138;
            v138 = 0l;
            while (while_method_4(v138)){
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
        while (while_method_5(v145)){
            int v147;
            v147 = 0l;
            while (while_method_4(v147)){
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
        while (while_method_5(v161)){
            int v163;
            v163 = 0l;
            while (while_method_4(v163)){
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
        Tuple8 tmp43 = Tuple8{0.0f, 2147483647l};
        v175 = tmp43.v0; v176 = tmp43.v1;
        int v177;
        v177 = 0l;
        while (while_method_5(v177)){
            int v179;
            v179 = 0l;
            while (while_method_4(v179)){
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
        Tuple8 tmp44 = cooperative_groups::reduce(v193, Tuple8{v175, v176}, v194);
        v195 = tmp44.v0; v196 = tmp44.v1;
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
        while (while_method_5(v201)){
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
__device__ static_array<float,2l> method_51(unsigned char * v0, unsigned char * v1, unsigned int & v2, static_array_list<Union7,32l> & v3, static_array<Union14,2l> & v4, curandStatePhilox4_32_10_t & v5, Union4 v6){
    static_array<float,2l> v7;
    static_array_list<Union7,32l> & v9 = v3;
    Union3 v10;
    v10 = Union3{Union3_1{v6}};
    Union3 v11;
    v11 = v10;
    while (while_method_11(v11)){
        Union3 v712;
        switch (v11.tag) {
            case 0: { // None
                v712 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v13 = v11.case1.v0;
                switch (v13.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v663 = v13.case0.v0; bool v664 = v13.case0.v1; static_array<Union6,2l> v665 = v13.case0.v2; int v666 = v13.case0.v3; static_array<int,2l> v667 = v13.case0.v4; int v668 = v13.case0.v5;
                        unsigned int v669 = v2;
                        Union6 v670; unsigned int v671;
                        Tuple6 tmp38 = draw_card_21(v5, v669);
                        v670 = tmp38.v0; v671 = tmp38.v1;
                        v2 = v671;
                        Union7 v672;
                        v672 = Union7{Union7_0{v670}};
                        v9.push(v672);
                        int v673;
                        v673 = 2l;
                        int v674; int v675;
                        Tuple7 tmp39 = Tuple7{0l, 0l};
                        v674 = tmp39.v0; v675 = tmp39.v1;
                        while (while_method_0(v674)){
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
                        while (while_method_0(v683)){
                            v681[v683] = v675;
                            v683 += 1l ;
                        }
                        Union5 v685;
                        v685 = Union5{Union5_1{v670}};
                        Union4 v686;
                        v686 = Union4{Union4_2{v685, true, v665, 0l, v681, v673}};
                        v712 = Union3{Union3_1{v686}};
                        break;
                    }
                    case 1: { // ChanceInit
                        unsigned int v688 = v2;
                        Union6 v689; unsigned int v690;
                        Tuple6 tmp40 = draw_card_21(v5, v688);
                        v689 = tmp40.v0; v690 = tmp40.v1;
                        v2 = v690;
                        unsigned int v691 = v2;
                        Union6 v692; unsigned int v693;
                        Tuple6 tmp41 = draw_card_21(v5, v691);
                        v692 = tmp41.v0; v693 = tmp41.v1;
                        v2 = v693;
                        Union7 v694;
                        v694 = Union7{Union7_2{0l, v689}};
                        v9.push(v694);
                        Union7 v695;
                        v695 = Union7{Union7_2{1l, v692}};
                        v9.push(v695);
                        int v696;
                        v696 = 2l;
                        static_array<int,2l> v697;
                        v697[0l] = 1l;
                        v697[1l] = 1l;
                        static_array<Union6,2l> v699;
                        v699[0l] = v689;
                        v699[1l] = v692;
                        Union5 v701;
                        v701 = Union5{Union5_0{}};
                        Union4 v702;
                        v702 = Union4{Union4_2{v701, true, v699, 0l, v697, v696}};
                        v712 = Union3{Union3_1{v702}};
                        break;
                    }
                    case 2: { // Round
                        Union5 v54 = v13.case2.v0; bool v55 = v13.case2.v1; static_array<Union6,2l> v56 = v13.case2.v2; int v57 = v13.case2.v3; static_array<int,2l> v58 = v13.case2.v4; int v59 = v13.case2.v5;
                        static_array<Union14,2l> v60 = v4;
                        Union14 v61;
                        v61 = v60[v57];
                        Union1 v479;
                        switch (v61.tag) {
                            case 0: { // T_Computer
                                static_array_list<Union7,32l> & v63 = v3;
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
                                while (while_method_3(v85)){
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
                                static_array_list<Union11,10l> v101;
                                v101 = static_array_list<Union11,10l>{};
                                int v103;
                                v103 = v63.length;
                                int v104;
                                v104 = 0l;
                                while (while_method_1(v103, v104)){
                                    Union7 v106;
                                    v106 = v63[v104];
                                    Union12 v125;
                                    switch (v106.tag) {
                                        case 0: { // CommunityCardIs
                                            Union6 v115 = v106.case0.v0;
                                            Union11 v116;
                                            v116 = Union11{Union11_1{v115}};
                                            v125 = Union12{Union12_1{v116}};
                                            break;
                                        }
                                        case 1: { // PlayerAction
                                            int v118 = v106.case1.v0; Union1 v119 = v106.case1.v1;
                                            Union11 v120;
                                            v120 = Union11{Union11_0{v119}};
                                            v125 = Union12{Union12_1{v120}};
                                            break;
                                        }
                                        case 2: { // PlayerGotCard
                                            int v108 = v106.case2.v0; Union6 v109 = v106.case2.v1;
                                            bool v110;
                                            v110 = v108 == v57;
                                            if (v110){
                                                Union11 v111;
                                                v111 = Union11{Union11_1{v109}};
                                                v125 = Union12{Union12_1{v111}};
                                            } else {
                                                v125 = Union12{Union12_0{}};
                                            }
                                            break;
                                        }
                                        default: {
                                            v125 = Union12{Union12_0{}};
                                        }
                                    }
                                    switch (v125.tag) {
                                        case 0: { // None
                                            break;
                                        }
                                        case 1: { // Some
                                            Union11 v126 = v125.case1.v0;
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
                                while (while_method_1(v131, v132)){
                                    Union11 v134;
                                    v134 = v101[v132];
                                    int v136;
                                    v136 = v132 * 6l;
                                    int v137;
                                    v137 = 1l + v136;
                                    switch (v134.tag) {
                                        case 0: { // C1of2
                                            Union1 v138 = v134.case0.v0;
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
                                            Union6 v141 = v134.case1.v0;
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
                                    method_25(v149, v151, v152, v157, v147, v155);
                                    unsigned int * v158;
                                    v158 = reinterpret_cast<unsigned int *>(&v0[12582912ull]);
                                    assert("Tensor range check" && 0 <= v145 && v145 < 4l);
                                    int v160;
                                    v160 = 12288l * v145;
                                    method_26(v158, v160, v152);
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
                                Tuple8 tmp42 = method_27(v5, v195, v197, v199, v201, v203, v205, v207, v209, v221, v211);
                                v222 = tmp42.v0; v223 = tmp42.v1;
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
                                while (while_method_4(v255)){
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
                                    v267 = method_52(v195, v197, v199, v201, v203, v205, v207, v209, v266, v255, v232);
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
                                    while (while_method_5(v331)){
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
                                        while (while_method_5(v356)){
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
                                        while (while_method_5(v364)){
                                            int v366;
                                            v366 = 0l;
                                            while (while_method_0(v366)){
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
                                        while (while_method_5(v389)){
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
                                        v479 = Union1{Union1_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v417;
                                        v417 = v58[0l];
                                        int v419; int v420;
                                        Tuple7 tmp45 = Tuple7{1l, v417};
                                        v419 = tmp45.v0; v420 = tmp45.v1;
                                        while (while_method_0(v419)){
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
                                            v479 = Union1{Union1_0{}};
                                        } else {
                                            v479 = Union1{Union1_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v433;
                                        v433 = v59 > 0l;
                                        if (v433){
                                            v479 = Union1{Union1_2{}};
                                        } else {
                                            v479 = Union1{Union1_0{}};
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
                                static_array_list<Union1,3l> v440;
                                v440 = static_array_list<Union1,3l>{};
                                v440.unsafe_set_length(1l);
                                Union1 v442;
                                v442 = Union1{Union1_0{}};
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
                                    Union1 v450;
                                    v450 = Union1{Union1_1{}};
                                    v440.push(v450);
                                } else {
                                }
                                bool v451;
                                v451 = v59 > 0l;
                                if (v451){
                                    Union1 v452;
                                    v452 = Union1{Union1_2{}};
                                    v440.push(v452);
                                } else {
                                }
                                int v453;
                                v453 = v440.length;
                                int v454;
                                v454 = v453 - 1l;
                                int v455;
                                v455 = 0l;
                                while (while_method_1(v454, v455)){
                                    int v457;
                                    v457 = v440.length;
                                    int v458;
                                    v458 = int_range_24(v457, v455, v5);
                                    Union1 v459;
                                    v459 = v440[v455];
                                    Union1 v461;
                                    v461 = v440[v458];
                                    v440[v455] = v461;
                                    v440[v458] = v459;
                                    v455 += 1l ;
                                }
                                Union1 v463;
                                v463 = v440.pop();
                                int v464;
                                v464 = sizeof(Union1);
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
                                Union1 * v473;
                                v473 = reinterpret_cast<Union1 *>(&v469[0ull]);
                                int v475;
                                v475 = threadIdx.x;
                                bool v476;
                                v476 = v475 == 0l;
                                if (v476){
                                    v473[0l] = v463;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                Union1 v477;
                                v477 = v473[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                v479 = v477;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union7 v480;
                        v480 = Union7{Union7_1{v57, v479}};
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
                                            Tuple7 tmp46 = Tuple7{0l, 0l};
                                            v539 = tmp46.v0; v540 = tmp46.v1;
                                            while (while_method_0(v539)){
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
                                            while (while_method_0(v548)){
                                                v546[v548] = v540;
                                                v548 += 1l ;
                                            }
                                            static_array<int,2l> v550;
                                            int v552;
                                            v552 = 0l;
                                            while (while_method_0(v552)){
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
                                Union6 v481 = v54.case1.v0;
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
                                            Tuple7 tmp47 = Tuple7{0l, 0l};
                                            v486 = tmp47.v0; v487 = tmp47.v1;
                                            while (while_method_0(v486)){
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
                                            while (while_method_0(v495)){
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
                                            Tuple7 tmp48 = Tuple7{0l, 0l};
                                            v503 = tmp48.v0; v504 = tmp48.v1;
                                            while (while_method_0(v503)){
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
                                            while (while_method_0(v512)){
                                                v510[v512] = v504;
                                                v512 += 1l ;
                                            }
                                            static_array<int,2l> v514;
                                            int v516;
                                            v516 = 0l;
                                            while (while_method_0(v516)){
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
                        v712 = Union3{Union3_1{v566}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v568 = v13.case3.v0; bool v569 = v13.case3.v1; static_array<Union6,2l> v570 = v13.case3.v2; int v571 = v13.case3.v3; static_array<int,2l> v572 = v13.case3.v4; int v573 = v13.case3.v5; Union1 v574 = v13.case3.v6;
                        Union7 v575;
                        v575 = Union7{Union7_1{v571, v574}};
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
                                            Tuple7 tmp49 = Tuple7{0l, 0l};
                                            v634 = tmp49.v0; v635 = tmp49.v1;
                                            while (while_method_0(v634)){
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
                                            while (while_method_0(v643)){
                                                v641[v643] = v635;
                                                v643 += 1l ;
                                            }
                                            static_array<int,2l> v645;
                                            int v647;
                                            v647 = 0l;
                                            while (while_method_0(v647)){
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
                                Union6 v576 = v568.case1.v0;
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
                                            Tuple7 tmp50 = Tuple7{0l, 0l};
                                            v581 = tmp50.v0; v582 = tmp50.v1;
                                            while (while_method_0(v581)){
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
                                            while (while_method_0(v590)){
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
                                            Tuple7 tmp51 = Tuple7{0l, 0l};
                                            v598 = tmp51.v0; v599 = tmp51.v1;
                                            while (while_method_0(v598)){
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
                                            while (while_method_0(v607)){
                                                v605[v607] = v599;
                                                v607 += 1l ;
                                            }
                                            static_array<int,2l> v609;
                                            int v611;
                                            v611 = 0l;
                                            while (while_method_0(v611)){
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
                        v712 = Union3{Union3_1{v661}};
                        break;
                    }
                    case 4: { // TerminalCall
                        Union5 v30 = v13.case4.v0; bool v31 = v13.case4.v1; static_array<Union6,2l> v32 = v13.case4.v2; int v33 = v13.case4.v3; static_array<int,2l> v34 = v13.case4.v4; int v35 = v13.case4.v5;
                        int v36;
                        v36 = v34[v33];
                        Union13 v38;
                        v38 = compare_hands_29(v30, v31, v32, v33, v34, v35);
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
                        Union7 v52;
                        v52 = Union7{Union7_3{v32, v43, v44}};
                        v9.push(v52);
                        v712 = Union3{Union3_0{}};
                        break;
                    }
                    case 5: { // TerminalFold
                        Union5 v14 = v13.case5.v0; bool v15 = v13.case5.v1; static_array<Union6,2l> v16 = v13.case5.v2; int v17 = v13.case5.v3; static_array<int,2l> v18 = v13.case5.v4; int v19 = v13.case5.v5;
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
                        Union7 v28;
                        v28 = Union7{Union7_3{v16, v20, v27}};
                        v9.push(v28);
                        v712 = Union3{Union3_0{}};
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
__device__ inline bool while_method_12(int v0){
    bool v1;
    v1 = v0 > 0l;
    return v1;
}
__device__ __noinline__ void noinline_train_50(float * v0, float * v1, unsigned char * v2, unsigned char * v3, curandStatePhilox4_32_10_t & v4, int v5, int v6){
    static_array<Union14,2l> v7;
    Union14 v9;
    v9 = Union14{Union14_1{}};
    v7[0l] = v9;
    Union14 v11;
    v11 = Union14{Union14_1{}};
    v7[1l] = v11;
    Union14 v13;
    v13 = Union14{Union14_0{}};
    v7[v6] = v13;
    static_array<Union14,2l> & v15 = v7;
    unsigned int v16 = 63ul;
    static_array_list<Union7,32l> v17;
    v17 = static_array_list<Union7,32l>{};
    static_array_list<Union7,32l> & v19 = v17;
    Union4 v20;
    v20 = Union4{Union4_1{}};
    static_array<float,2l> v21;
    v21 = method_51(v2, v3, v16, v19, v15, v4, v20);
    float v22;
    v22 = v21[v6];
    int v24;
    v24 = 0l;
    while (while_method_4(v24)){
        double * v26;
        v26 = reinterpret_cast<double *>(&v3[2097168ull]);
        double * v28;
        v28 = reinterpret_cast<double *>(&v3[2883600ull]);
        int * v30;
        v30 = reinterpret_cast<int *>(&v3[3670032ull]);
        assert("Tensor range check" && 0 <= v24 && v24 < 4l);
        int v32;
        v32 = 24576l * v24;
        int v33;
        v33 = threadIdx.x;
        int v34;
        v34 = blockIdx.x;
        int v35;
        v35 = v34 * 512l;
        int v36;
        v36 = v33 + v35;
        assert("Tensor range check" && 0 <= v36 && v36 < 12288l);
        int v37;
        v37 = 2l * v36;
        int v38;
        v38 = v37 + v32;
        double v39;
        v39 = 0.0;
        int v40;
        v40 = 0l;
        while (while_method_0(v40)){
            assert("Tensor range check" && 0 <= v40 && v40 < 2l);
            int v42;
            v42 = v40 + v38;
            double v43;
            v43 = v26[v42];
            double v44;
            v44 = v39 + v43;
            v39 = v44;
            v40 += 1l ;
        }
        double v45;
        v45 = 0.0;
        int v46;
        v46 = 0l;
        while (while_method_0(v46)){
            assert("Tensor range check" && 0 <= v46 && v46 < 2l);
            int v48;
            v48 = v46 + v38;
            double v49;
            v49 = v28[v48];
            double v50;
            v50 = v45 + v49;
            v45 = v50;
            v46 += 1l ;
        }
        double v51;
        v51 = v39 - v45;
        double v52;
        v52 = exp(v51);
        float v53;
        v53 = (float)v52;
        int v54;
        v54 = v5 / 4l;
        float v55;
        v55 = v22 * v53;
        assert("Tensor range check" && 0 <= v24 && v24 < 4l);
        assert("Tensor range check" && 0 <= v54 && v54 < 256l);
        int v56;
        v56 = 256l * v24;
        int v57;
        v57 = v56 + v54;
        float * v58;
        v58 = v0+v57;
        float * v60;
        v60 = v1+v57;
        float v62;
        v62 = atomicAdd(v58,v55);
        float v63;
        v63 = atomicAdd(v60,v53);
        v24 += 1l ;
    }
    unsigned int * v64;
    v64 = reinterpret_cast<unsigned int *>(&v2[12582912ull]);
    int * v66;
    v66 = reinterpret_cast<int *>(&v3[262144ull]);
    float * v68;
    v68 = reinterpret_cast<float *>(&v3[262160ull]);
    float * v70;
    v70 = reinterpret_cast<float *>(&v3[524304ull]);
    float * v72;
    v72 = reinterpret_cast<float *>(&v3[786448ull]);
    float * v74;
    v74 = reinterpret_cast<float *>(&v3[1048592ull]);
    float * v76;
    v76 = reinterpret_cast<float *>(&v3[1310736ull]);
    float * v78;
    v78 = reinterpret_cast<float *>(&v3[1572880ull]);
    float * v80;
    v80 = reinterpret_cast<float *>(&v3[1835024ull]);
    int * v82;
    v82 = reinterpret_cast<int *>(&v2[12779520ull]);
    float * v84;
    v84 = reinterpret_cast<float *>(&v2[15925248ull]);
    int * v86;
    v86 = reinterpret_cast<int *>(&v2[19070976ull]);
    int * v88;
    v88 = reinterpret_cast<int *>(&v2[22216704ull]);
    double * v90;
    v90 = reinterpret_cast<double *>(&v2[25362432ull]);
    double * v92;
    v92 = reinterpret_cast<double *>(&v2[37945344ull]);
    double * v94;
    v94 = reinterpret_cast<double *>(&v3[2097168ull]);
    double * v96;
    v96 = reinterpret_cast<double *>(&v3[2883600ull]);
    int * v98;
    v98 = reinterpret_cast<int *>(&v3[3670032ull]);
    int v100;
    v100 = 0l;
    while (while_method_4(v100)){
        int v102;
        v102 = threadIdx.x;
        int v103;
        v103 = blockIdx.x;
        int v104;
        v104 = v103 * 512l;
        int v105;
        v105 = v102 + v104;
        float v106[2l];
        int v107;
        v107 = 0l;
        while (while_method_0(v107)){
            float v109;
            v109 = v21[v107];
            v106[v107] = v109;
            v107 += 1l ;
        }
        assert("Tensor range check" && 0 <= v100 && v100 < 4l);
        assert("Tensor range check" && 0 <= v105 && v105 < 12288l);
        int v111;
        v111 = 12288l * v100;
        int v112;
        v112 = v111 + v105;
        int v113;
        v113 = v98[v112];
        int v114;
        v114 = v113;
        while (while_method_12(v114)){
            v114 -= 1l ;
            assert("Tensor range check" && 0 <= v100 && v100 < 4l);
            assert("Tensor range check" && 0 <= v114 && v114 < 16l);
            assert("Tensor range check" && 0 <= v105 && v105 < 12288l);
            int v116;
            v116 = 12288l * v114;
            int v117;
            v117 = v116 + v105;
            int v118;
            v118 = 196608l * v100;
            int v119;
            v119 = v118 + v117;
            int v120;
            v120 = v82[v119];
            float v121;
            v121 = v84[v119];
            int v122;
            v122 = v86[v119];
            int v123;
            v123 = v88[v119];
            assert("Tensor range check" && 0 <= v122 && v122 < 2l);
            float v124;
            v124 = v106[v122];
            assert("Tensor range check" && 0 <= v100 && v100 < 4l);
            int v125;
            v125 = 16384l * v100;
            assert("Tensor range check" && 0 <= v123 && v123 < 4096l);
            int v126;
            v126 = 4l * v123;
            int v127;
            v127 = v126 + v125;
            float * v128;
            v128 = v68+v127;
            float * v130;
            v130 = v70+v127;
            float * v132;
            v132 = v72+v127;
            float * v134;
            v134 = v74+v127;
            float * v136;
            v136 = v76+v127;
            float * v138;
            v138 = v78+v127;
            float * v140;
            v140 = v80+v127;
            assert("Tensor range check" && 0 <= v100 && v100 < 4l);
            int v142;
            v142 = 393216l * v100;
            assert("Tensor range check" && 0 <= v114 && v114 < 16l);
            int v143;
            v143 = 24576l * v114;
            int v144;
            v144 = v143 + v142;
            assert("Tensor range check" && 0 <= v105 && v105 < 12288l);
            int v145;
            v145 = 2l * v105;
            int v146;
            v146 = v145 + v144;
            double v147[2l];
            int v148;
            v148 = 0l;
            while (while_method_0(v148)){
                assert("Tensor range check" && 0 <= v148 && v148 < 2l);
                int v150;
                v150 = v148 + v146;
                double v151;
                v151 = v90[v150];
                bool v152;
                v152 = v122 == v148;
                double v153;
                if (v152){
                    v153 = 0.0;
                } else {
                    v153 = v151;
                }
                assert("Tensor range check" && 0 <= v148 && v148 < 2l);
                v147[v148] = v153;
                v148 += 1l ;
            }
            double v154;
            v154 = 0.0;
            int v155;
            v155 = 0l;
            while (while_method_0(v155)){
                assert("Tensor range check" && 0 <= v155 && v155 < 2l);
                double v157;
                v157 = v147[v155];
                double v158;
                v158 = v154 + v157;
                v154 = v158;
                v155 += 1l ;
            }
            double v159;
            v159 = 0.0;
            int v160;
            v160 = 0l;
            while (while_method_0(v160)){
                assert("Tensor range check" && 0 <= v160 && v160 < 2l);
                int v162;
                v162 = v160 + v146;
                double v163;
                v163 = v92[v162];
                double v164;
                v164 = v159 + v163;
                v159 = v164;
                v160 += 1l ;
            }
            double v165;
            v165 = v154 - v159;
            double v166;
            v166 = exp(v165);
            float v167;
            v167 = (float)v166;
            float v168;
            v168 = v124 * v167;
            assert("Tensor range check" && 0 <= v120 && v120 < 4l);
            float * v169;
            v169 = v138+v120;
            float * v171;
            v171 = v140+v120;
            float v173;
            v173 = atomicAdd(v169,v168);
            float v174;
            v174 = atomicAdd(v171,v167);
            float * v175;
            v175 = v130+0l;
            float * v177;
            v177 = v134+0l;
            float * v179;
            v179 = v136+0l;
            int v181;
            v181 = sizeof(float *);
            unsigned long long v182;
            v182 = (unsigned long long)v181;
            unsigned long long v183;
            v183 = 512ull * v182;
            unsigned long long v184;
            v184 = 8192ull + v183;
            unsigned long long v185;
            v185 = v184 + 16ull;
            unsigned long long v186;
            v186 = v185 - 1ull;
            unsigned long long v187;
            v187 = v186 % 16ull;
            unsigned long long v188;
            v188 = v186 - v187;
            unsigned long long v189;
            v189 = v188 + v183;
            unsigned long long v190;
            v190 = v189 + 16ull;
            unsigned long long v191;
            v191 = v190 - 1ull;
            unsigned long long v192;
            v192 = v191 % 16ull;
            unsigned long long v193;
            v193 = v191 - v192;
            unsigned long long v194;
            v194 = v193 + v183;
            unsigned long long v195;
            v195 = v194 + 16ull;
            unsigned long long v196;
            v196 = v195 - 1ull;
            unsigned long long v197;
            v197 = v196 % 16ull;
            unsigned long long v198;
            v198 = v196 - v197;
            unsigned long long v199;
            v199 = v198 + v183;
            unsigned long long v200;
            v200 = v199 + 16ull;
            unsigned long long v201;
            v201 = v200 - 1ull;
            unsigned long long v202;
            v202 = v201 % 16ull;
            unsigned long long v203;
            v203 = v201 - v202;
            unsigned long long v204;
            v204 = v203 + 2048ull;
            bool v205;
            v205 = v204 <= 81920ull;
            bool v206;
            v206 = v205 == false;
            if (v206){
                assert("The dynamic shared memory is insufficient to allocate the tensor." && v205);
            } else {
            }
            extern __shared__ unsigned char v208[];
            bool v209;
            v209 = v204 <= v204;
            bool v210;
            v210 = v209 == false;
            if (v210){
                assert("The length of the partition has to be less than or equal to the length of the base array." && v209);
            } else {
            }
            float * v212;
            v212 = reinterpret_cast<float *>(&v208[0ull]);
            int * v214;
            v214 = reinterpret_cast<int *>(&v208[2048ull]);
            float * v216;
            v216 = reinterpret_cast<float *>(&v208[4096ull]);
            float * v218;
            v218 = reinterpret_cast<float *>(&v208[6144ull]);
            float * * v220;
            v220 = reinterpret_cast<float * *>(&v208[8192ull]);
            float * * v222;
            v222 = reinterpret_cast<float * *>(&v208[v188]);
            float * * v224;
            v224 = reinterpret_cast<float * *>(&v208[v193]);
            float * * v226;
            v226 = reinterpret_cast<float * *>(&v208[v198]);
            float * v228;
            v228 = reinterpret_cast<float *>(&v208[v203]);
            int v230;
            v230 = threadIdx.x;
            assert("Tensor range check" && 0 <= v230 && v230 < 512l);
            v212[v230] = v121;
            v214[v230] = v120;
            v216[v230] = v124;
            v218[v230] = v167;
            v220[v230] = v132;
            v222[v230] = v175;
            v224[v230] = v177;
            v226[v230] = v179;
            asm("barrier.cta.sync %0;" :: "r"(0l));
            bool v231;
            v231 = 0l <= v230;
            bool v232;
            v232 = v231 == false;
            if (v232){
                assert("The index needs to be zero or positive." && v231);
            } else {
            }
            int v234;
            v234 = v230 % 1l;
            bool v235;
            v235 = v230 < 512l;
            bool v236;
            v236 = v235 == false;
            if (v236){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v235);
            } else {
            }
            assert("Tensor range check" && 0 <= v230 && v230 < 512l);
            int v238;
            v238 = 0l;
            while (while_method_5(v238)){
                bool v240;
                v240 = v231 && v235;
                bool v241;
                v241 = v240 == false;
                if (v241){
                    assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v240);
                } else {
                }
                bool v243;
                v243 = 0l <= v238;
                bool v245;
                if (v243){
                    bool v244;
                    v244 = v238 < 1l;
                    v245 = v244;
                } else {
                    v245 = false;
                }
                bool v246;
                v246 = v245 == false;
                if (v246){
                    assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v245);
                } else {
                }
                int v248;
                v248 = v238 * 512l;
                int v249;
                v249 = v248 + v230;
                assert("Tensor range check" && 0 <= v238 && v238 < 1l);
                int v250;
                v250 = 512l * v238;
                int v251;
                v251 = v250 + v230;
                float v252;
                v252 = v212[v251];
                int v253;
                v253 = v214[v251];
                float v254;
                v254 = v216[v251];
                float v255;
                v255 = v218[v251];
                float * v256;
                v256 = v220[v251];
                float * v257;
                v257 = v222[v251];
                float * v258;
                v258 = v224[v251];
                float * v259;
                v259 = v226[v251];
                int v260;
                v260 = blockIdx.x;
                int v261;
                v261 = v260 * 512l;
                int v262;
                v262 = v261 + v249;
                assert("Tensor range check" && 0 <= v234 && v234 < 1l);
                int v263;
                v263 = 4l * v234;
                float v264[4l];
                float v265[4l];
                float v266[4l];
                int v267[4l];
                int v268;
                v268 = 0l;
                while (while_method_5(v268)){
                    assert("Tensor range check" && 0 <= v268 && v268 < 1l);
                    int v270;
                    v270 = 4l * v268;
                    assert("Tensor range check" && 0 <= v268 && v268 < 1l);
                    int v271;
                    v271 = v270 + v263;
                    int4* v272;
                    v272 = reinterpret_cast<int4*>(v257 + v271);
                    int4* v273;
                    v273 = reinterpret_cast<int4*>(v264 + v270);
                    assert("Pointer alignment check" && (unsigned long long)(v272) % 4l == 0 && (unsigned long long)(v273) % 4l == 0);
                    *v273 = *v272;
                    int4* v274;
                    v274 = reinterpret_cast<int4*>(v258 + v271);
                    int4* v275;
                    v275 = reinterpret_cast<int4*>(v265 + v270);
                    assert("Pointer alignment check" && (unsigned long long)(v274) % 4l == 0 && (unsigned long long)(v275) % 4l == 0);
                    *v275 = *v274;
                    int4* v276;
                    v276 = reinterpret_cast<int4*>(v259 + v271);
                    int4* v277;
                    v277 = reinterpret_cast<int4*>(v266 + v270);
                    assert("Pointer alignment check" && (unsigned long long)(v276) % 4l == 0 && (unsigned long long)(v277) % 4l == 0);
                    *v277 = *v276;
                    v268 += 1l ;
                }
                int v278;
                v278 = 0l;
                while (while_method_5(v278)){
                    int v280;
                    v280 = 0l;
                    while (while_method_4(v280)){
                        bool v282;
                        v282 = 0l <= v280;
                        bool v284;
                        if (v282){
                            bool v283;
                            v283 = v280 < 4l;
                            v284 = v283;
                        } else {
                            v284 = false;
                        }
                        bool v285;
                        v285 = v284 == false;
                        if (v285){
                            assert("The indices should be inside the range of the dimension." && v284);
                        } else {
                        }
                        bool v287;
                        v287 = 0l <= v234;
                        bool v289;
                        if (v287){
                            bool v288;
                            v288 = v234 < 1l;
                            v289 = v288;
                        } else {
                            v289 = false;
                        }
                        bool v290;
                        v290 = v289 == false;
                        if (v290){
                            assert("The indices should be inside the range of the dimension." && v289);
                        } else {
                        }
                        int v292;
                        v292 = v234 * 4l;
                        int v293;
                        v293 = v280 + v292;
                        bool v294;
                        v294 = 0l <= v278;
                        bool v296;
                        if (v294){
                            bool v295;
                            v295 = v278 < 1l;
                            v296 = v295;
                        } else {
                            v296 = false;
                        }
                        bool v297;
                        v297 = v296 == false;
                        if (v297){
                            assert("The indices should be inside the range of the dimension." && v296);
                        } else {
                        }
                        int v299;
                        v299 = v278 * 4l;
                        int v300;
                        v300 = v293 + v299;
                        assert("Tensor range check" && 0 <= v278 && v278 < 1l);
                        assert("Tensor range check" && 0 <= v280 && v280 < 4l);
                        int v301;
                        v301 = 4l * v278;
                        int v302;
                        v302 = v301 + v280;
                        v267[v302] = v300;
                        v280 += 1l ;
                    }
                    v278 += 1l ;
                }
                float v303[4l];
                int v304;
                v304 = 0l;
                while (while_method_5(v304)){
                    int v306;
                    v306 = 0l;
                    while (while_method_4(v306)){
                        assert("Tensor range check" && 0 <= v304 && v304 < 1l);
                        assert("Tensor range check" && 0 <= v306 && v306 < 4l);
                        int v308;
                        v308 = 4l * v304;
                        int v309;
                        v309 = v308 + v306;
                        float v310;
                        v310 = v265[v309];
                        float v311;
                        v311 = v266[v309];
                        bool v312;
                        v312 = v311 == 0.0f;
                        bool v313;
                        v313 = v312 != true;
                        float v315;
                        if (v313){
                            float v314;
                            v314 = v310 / v311;
                            v315 = v314;
                        } else {
                            v315 = 0.0f;
                        }
                        assert("Tensor range check" && 0 <= v304 && v304 < 1l);
                        assert("Tensor range check" && 0 <= v306 && v306 < 4l);
                        v303[v309] = v315;
                        v306 += 1l ;
                    }
                    v304 += 1l ;
                }
                bool v316[4l];
                int v317;
                v317 = 0l;
                while (while_method_5(v317)){
                    int v319;
                    v319 = 0l;
                    while (while_method_4(v319)){
                        assert("Tensor range check" && 0 <= v317 && v317 < 1l);
                        assert("Tensor range check" && 0 <= v319 && v319 < 4l);
                        int v321;
                        v321 = 4l * v317;
                        int v322;
                        v322 = v321 + v319;
                        float v323;
                        v323 = v264[v322];
                        int v324;
                        v324 = v267[v322];
                        bool v325;
                        v325 = v324 < 3l;
                        assert("Tensor range check" && 0 <= v317 && v317 < 1l);
                        assert("Tensor range check" && 0 <= v319 && v319 < 4l);
                        v316[v322] = v325;
                        v319 += 1l ;
                    }
                    v317 += 1l ;
                }
                float v326[4l];
                int v327;
                v327 = 0l;
                while (while_method_5(v327)){
                    int v329;
                    v329 = 0l;
                    while (while_method_4(v329)){
                        assert("Tensor range check" && 0 <= v327 && v327 < 1l);
                        assert("Tensor range check" && 0 <= v329 && v329 < 4l);
                        int v331;
                        v331 = 4l * v327;
                        int v332;
                        v332 = v331 + v329;
                        float v333;
                        v333 = v264[v332];
                        bool v334;
                        v334 = v316[v332];
                        float v337;
                        if (v334){
                            bool v335;
                            v335 = 0.0f >= v333;
                            if (v335){
                                v337 = 0.0f;
                            } else {
                                v337 = v333;
                            }
                        } else {
                            v337 = 0.0f;
                        }
                        assert("Tensor range check" && 0 <= v327 && v327 < 1l);
                        assert("Tensor range check" && 0 <= v329 && v329 < 4l);
                        v326[v332] = v337;
                        v329 += 1l ;
                    }
                    v327 += 1l ;
                }
                float v338;
                v338 = 0.0f;
                int v339;
                v339 = 0l;
                while (while_method_5(v339)){
                    int v341;
                    v341 = 0l;
                    while (while_method_4(v341)){
                        assert("Tensor range check" && 0 <= v339 && v339 < 1l);
                        assert("Tensor range check" && 0 <= v341 && v341 < 4l);
                        int v343;
                        v343 = 4l * v339;
                        int v344;
                        v344 = v343 + v341;
                        float v345;
                        v345 = v326[v344];
                        float v346;
                        v346 = v338 + v345;
                        v338 = v346;
                        v341 += 1l ;
                    }
                    v339 += 1l ;
                }
                auto v347 = cooperative_groups::coalesced_threads();
                int v348;
                v348 = threadIdx.x;
                auto v349 = cooperative_groups::labeled_partition(v347,v348);
                Closure1 v350{};
                float v351;
                v351 = cooperative_groups::reduce(v349, v338, v350);
                int v352[4l];
                int v353;
                v353 = 0l;
                while (while_method_5(v353)){
                    int v355;
                    v355 = 0l;
                    while (while_method_4(v355)){
                        assert("Tensor range check" && 0 <= v353 && v353 < 1l);
                        assert("Tensor range check" && 0 <= v355 && v355 < 4l);
                        int v357;
                        v357 = 4l * v353;
                        int v358;
                        v358 = v357 + v355;
                        bool v359;
                        v359 = v316[v358];
                        int v360;
                        if (v359){
                            v360 = 1l;
                        } else {
                            v360 = 0l;
                        }
                        assert("Tensor range check" && 0 <= v353 && v353 < 1l);
                        assert("Tensor range check" && 0 <= v355 && v355 < 4l);
                        v352[v358] = v360;
                        v355 += 1l ;
                    }
                    v353 += 1l ;
                }
                int v361;
                v361 = 0l;
                int v362;
                v362 = 0l;
                while (while_method_5(v362)){
                    int v364;
                    v364 = 0l;
                    while (while_method_4(v364)){
                        assert("Tensor range check" && 0 <= v362 && v362 < 1l);
                        assert("Tensor range check" && 0 <= v364 && v364 < 4l);
                        int v366;
                        v366 = 4l * v362;
                        int v367;
                        v367 = v366 + v364;
                        int v368;
                        v368 = v352[v367];
                        int v369;
                        v369 = v361 + v368;
                        v361 = v369;
                        v364 += 1l ;
                    }
                    v362 += 1l ;
                }
                auto v370 = cooperative_groups::coalesced_threads();
                int v371;
                v371 = threadIdx.x;
                auto v372 = cooperative_groups::labeled_partition(v370,v371);
                Closure2 v373{};
                int v374;
                v374 = cooperative_groups::reduce(v372, v361, v373);
                float v375;
                v375 = (float)v374;
                float v376;
                v376 = 1.0f / v375;
                float v377[4l];
                int v378;
                v378 = 0l;
                while (while_method_5(v378)){
                    int v380;
                    v380 = 0l;
                    while (while_method_4(v380)){
                        assert("Tensor range check" && 0 <= v378 && v378 < 1l);
                        assert("Tensor range check" && 0 <= v380 && v380 < 4l);
                        int v382;
                        v382 = 4l * v378;
                        int v383;
                        v383 = v382 + v380;
                        float v384;
                        v384 = v326[v383];
                        bool v385;
                        v385 = v316[v383];
                        bool v386;
                        v386 = v385 == false;
                        float v391;
                        if (v386){
                            v391 = 0.0f;
                        } else {
                            bool v387;
                            v387 = v351 == 0.0f;
                            bool v388;
                            v388 = v387 != true;
                            if (v388){
                                float v389;
                                v389 = v384 / v351;
                                v391 = v389;
                            } else {
                                v391 = v376;
                            }
                        }
                        assert("Tensor range check" && 0 <= v378 && v378 < 1l);
                        assert("Tensor range check" && 0 <= v380 && v380 < 4l);
                        v377[v383] = v391;
                        v380 += 1l ;
                    }
                    v378 += 1l ;
                }
                float v392[4l];
                int v393;
                v393 = 0l;
                while (while_method_5(v393)){
                    int v395;
                    v395 = 0l;
                    while (while_method_4(v395)){
                        assert("Tensor range check" && 0 <= v393 && v393 < 1l);
                        assert("Tensor range check" && 0 <= v395 && v395 < 4l);
                        int v397;
                        v397 = 4l * v393;
                        int v398;
                        v398 = v397 + v395;
                        float v399;
                        v399 = v303[v398];
                        int v400;
                        v400 = v267[v398];
                        bool v401;
                        v401 = v253 == v400;
                        float v404;
                        if (v401){
                            float v402;
                            v402 = v254 - v399;
                            float v403;
                            v403 = v402 / v252;
                            v404 = v403;
                        } else {
                            v404 = 0.0f;
                        }
                        float v405;
                        v405 = v404 + v399;
                        assert("Tensor range check" && 0 <= v393 && v393 < 1l);
                        assert("Tensor range check" && 0 <= v395 && v395 < 4l);
                        v392[v398] = v405;
                        v395 += 1l ;
                    }
                    v393 += 1l ;
                }
                float v406[4l];
                int v407;
                v407 = 0l;
                while (while_method_5(v407)){
                    int v409;
                    v409 = 0l;
                    while (while_method_4(v409)){
                        assert("Tensor range check" && 0 <= v407 && v407 < 1l);
                        assert("Tensor range check" && 0 <= v409 && v409 < 4l);
                        int v411;
                        v411 = 4l * v407;
                        int v412;
                        v412 = v411 + v409;
                        float v413;
                        v413 = v377[v412];
                        float v414;
                        v414 = v392[v412];
                        float v415;
                        v415 = v413 * v414;
                        assert("Tensor range check" && 0 <= v407 && v407 < 1l);
                        assert("Tensor range check" && 0 <= v409 && v409 < 4l);
                        v406[v412] = v415;
                        v409 += 1l ;
                    }
                    v407 += 1l ;
                }
                float v416;
                v416 = 0.0f;
                int v417;
                v417 = 0l;
                while (while_method_5(v417)){
                    int v419;
                    v419 = 0l;
                    while (while_method_4(v419)){
                        assert("Tensor range check" && 0 <= v417 && v417 < 1l);
                        assert("Tensor range check" && 0 <= v419 && v419 < 4l);
                        int v421;
                        v421 = 4l * v417;
                        int v422;
                        v422 = v421 + v419;
                        float v423;
                        v423 = v406[v422];
                        float v424;
                        v424 = v416 + v423;
                        v416 = v424;
                        v419 += 1l ;
                    }
                    v417 += 1l ;
                }
                auto v425 = cooperative_groups::coalesced_threads();
                int v426;
                v426 = threadIdx.x;
                auto v427 = cooperative_groups::labeled_partition(v425,v426);
                float v428;
                v428 = cooperative_groups::reduce(v427, v416, v350);
                int v429;
                v429 = 0l;
                while (while_method_5(v429)){
                    int v431;
                    v431 = 0l;
                    while (while_method_4(v431)){
                        assert("Tensor range check" && 0 <= v429 && v429 < 1l);
                        assert("Tensor range check" && 0 <= v431 && v431 < 4l);
                        int v433;
                        v433 = 4l * v429;
                        int v434;
                        v434 = v433 + v431;
                        float v435;
                        v435 = v392[v434];
                        int v436;
                        v436 = v267[v434];
                        float v437;
                        v437 = v435 - v428;
                        float v438;
                        v438 = v255 * v437;
                        assert("Tensor range check" && 0 <= v436 && v436 < 4l);
                        float * v439;
                        v439 = v256+v436;
                        float v441;
                        v441 = atomicAdd(v439,v438);
                        v431 += 1l ;
                    }
                    v429 += 1l ;
                }
                int v442;
                v442 = 0l;
                while (while_method_5(v442)){
                    assert("Tensor range check" && 0 <= v442 && v442 < 1l);
                    assert("Tensor range check" && 0 <= v442 && v442 < 1l);
                    v442 += 1l ;
                }
                assert("Tensor range check" && 0 <= v249 && v249 < 512l);
                v228[v249] = v428;
                v238 += 1l ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0l));
            assert("Tensor range check" && 0 <= v230 && v230 < 512l);
            float v444;
            v444 = v228[v230];
            asm("barrier.cta.sync %0;" :: "r"(0l));
            assert("Tensor range check" && 0 <= v122 && v122 < 2l);
            v106[v122] = v444;
        }
        int v445;
        v445 = threadIdx.x;
        int v446;
        v446 = blockIdx.x;
        int v447;
        v447 = v446 * 512l;
        int v448;
        v448 = v445 + v447;
        assert("Tensor range check" && 0 <= v100 && v100 < 4l);
        int v449;
        v449 = 24576l * v100;
        assert("Tensor range check" && 0 <= v448 && v448 < 12288l);
        int v450;
        v450 = 2l * v448;
        int v451;
        v451 = v450 + v449;
        double * v452;
        v452 = v94+v451;
        double * v454;
        v454 = v96+v451;
        double * v456;
        v456 = v452+0l;
        double * v458;
        v458 = v454+0l;
        double * v460;
        v460 = v452+0l;
        double * v462;
        v462 = v454+0l;
        int v464;
        v464 = sizeof(double *);
        unsigned long long v465;
        v465 = (unsigned long long)v464;
        unsigned long long v466;
        v466 = 512ull * v465;
        unsigned long long v467;
        v467 = v466 + 16ull;
        unsigned long long v468;
        v468 = v467 - 1ull;
        unsigned long long v469;
        v469 = v468 % 16ull;
        unsigned long long v470;
        v470 = v468 - v469;
        unsigned long long v471;
        v471 = v470 + v466;
        unsigned long long v472;
        v472 = v471 + 16ull;
        unsigned long long v473;
        v473 = v472 - 1ull;
        unsigned long long v474;
        v474 = v473 % 16ull;
        unsigned long long v475;
        v475 = v473 - v474;
        unsigned long long v476;
        v476 = v475 + v466;
        unsigned long long v477;
        v477 = v476 + 16ull;
        unsigned long long v478;
        v478 = v477 - 1ull;
        unsigned long long v479;
        v479 = v478 % 16ull;
        unsigned long long v480;
        v480 = v478 - v479;
        unsigned long long v481;
        v481 = v480 + v466;
        bool v482;
        v482 = v481 <= 81920ull;
        bool v483;
        v483 = v482 == false;
        if (v483){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v482);
        } else {
        }
        extern __shared__ unsigned char v485[];
        bool v486;
        v486 = v481 <= v481;
        bool v487;
        v487 = v486 == false;
        if (v487){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v486);
        } else {
        }
        double * * v489;
        v489 = reinterpret_cast<double * *>(&v485[0ull]);
        double * * v491;
        v491 = reinterpret_cast<double * *>(&v485[v470]);
        double * * v493;
        v493 = reinterpret_cast<double * *>(&v485[v475]);
        double * * v495;
        v495 = reinterpret_cast<double * *>(&v485[v480]);
        int v497;
        v497 = threadIdx.x;
        assert("Tensor range check" && 0 <= v497 && v497 < 512l);
        v489[v497] = v456;
        v491[v497] = v458;
        v493[v497] = v460;
        v495[v497] = v462;
        asm("barrier.cta.sync %0;" :: "r"(0l));
        bool v498;
        v498 = 0l <= v497;
        bool v499;
        v499 = v498 == false;
        if (v499){
            assert("The index needs to be zero or positive." && v498);
        } else {
        }
        int v501;
        v501 = v497 % 1l;
        bool v502;
        v502 = v497 < 512l;
        bool v503;
        v503 = v502 == false;
        if (v503){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v502);
        } else {
        }
        assert("Tensor range check" && 0 <= v497 && v497 < 512l);
        int v505;
        v505 = 0l;
        while (while_method_5(v505)){
            bool v507;
            v507 = v498 && v502;
            bool v508;
            v508 = v507 == false;
            if (v508){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v507);
            } else {
            }
            bool v510;
            v510 = 0l <= v505;
            bool v512;
            if (v510){
                bool v511;
                v511 = v505 < 1l;
                v512 = v511;
            } else {
                v512 = false;
            }
            bool v513;
            v513 = v512 == false;
            if (v513){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v512);
            } else {
            }
            int v515;
            v515 = v505 * 512l;
            int v516;
            v516 = v515 + v497;
            assert("Tensor range check" && 0 <= v505 && v505 < 1l);
            int v517;
            v517 = 512l * v505;
            int v518;
            v518 = v517 + v497;
            double * v519;
            v519 = v489[v518];
            double * v520;
            v520 = v491[v518];
            double * v521;
            v521 = v493[v518];
            double * v522;
            v522 = v495[v518];
            int v523;
            v523 = blockIdx.x;
            int v524;
            v524 = v523 * 512l;
            int v525;
            v525 = v524 + v516;
            assert("Tensor range check" && 0 <= v501 && v501 < 1l);
            int v526;
            v526 = 2l * v501;
            double v527[2l];
            double v528[2l];
            int v529[2l];
            int v530;
            v530 = 0l;
            while (while_method_5(v530)){
                assert("Tensor range check" && 0 <= v530 && v530 < 1l);
                int v532;
                v532 = 2l * v530;
                assert("Tensor range check" && 0 <= v530 && v530 < 1l);
                int v533;
                v533 = v532 + v526;
                int4* v534;
                v534 = reinterpret_cast<int4*>(v519 + v533);
                int4* v535;
                v535 = reinterpret_cast<int4*>(v527 + v532);
                assert("Pointer alignment check" && (unsigned long long)(v534) % 2l == 0 && (unsigned long long)(v535) % 2l == 0);
                *v535 = *v534;
                int4* v536;
                v536 = reinterpret_cast<int4*>(v520 + v533);
                int4* v537;
                v537 = reinterpret_cast<int4*>(v528 + v532);
                assert("Pointer alignment check" && (unsigned long long)(v536) % 2l == 0 && (unsigned long long)(v537) % 2l == 0);
                *v537 = *v536;
                v530 += 1l ;
            }
            int v538;
            v538 = 0l;
            while (while_method_5(v538)){
                int v540;
                v540 = 0l;
                while (while_method_0(v540)){
                    bool v542;
                    v542 = 0l <= v540;
                    bool v544;
                    if (v542){
                        bool v543;
                        v543 = v540 < 2l;
                        v544 = v543;
                    } else {
                        v544 = false;
                    }
                    bool v545;
                    v545 = v544 == false;
                    if (v545){
                        assert("The indices should be inside the range of the dimension." && v544);
                    } else {
                    }
                    bool v547;
                    v547 = 0l <= v501;
                    bool v549;
                    if (v547){
                        bool v548;
                        v548 = v501 < 1l;
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
                    int v552;
                    v552 = v501 * 2l;
                    int v553;
                    v553 = v540 + v552;
                    bool v554;
                    v554 = 0l <= v538;
                    bool v556;
                    if (v554){
                        bool v555;
                        v555 = v538 < 1l;
                        v556 = v555;
                    } else {
                        v556 = false;
                    }
                    bool v557;
                    v557 = v556 == false;
                    if (v557){
                        assert("The indices should be inside the range of the dimension." && v556);
                    } else {
                    }
                    int v559;
                    v559 = v538 * 2l;
                    int v560;
                    v560 = v553 + v559;
                    assert("Tensor range check" && 0 <= v538 && v538 < 1l);
                    assert("Tensor range check" && 0 <= v540 && v540 < 2l);
                    int v561;
                    v561 = 2l * v538;
                    int v562;
                    v562 = v561 + v540;
                    v529[v562] = v560;
                    v540 += 1l ;
                }
                v538 += 1l ;
            }
            double v563[2l];
            double v564[2l];
            int v565;
            v565 = 0l;
            while (while_method_5(v565)){
                int v567;
                v567 = 0l;
                while (while_method_0(v567)){
                    assert("Tensor range check" && 0 <= v565 && v565 < 1l);
                    assert("Tensor range check" && 0 <= v567 && v567 < 2l);
                    int v569;
                    v569 = 2l * v565;
                    int v570;
                    v570 = v569 + v567;
                    double v571;
                    v571 = v527[v570];
                    double v572;
                    v572 = v528[v570];
                    assert("Tensor range check" && 0 <= v565 && v565 < 1l);
                    assert("Tensor range check" && 0 <= v567 && v567 < 2l);
                    v563[v570] = 0.0;
                    v564[v570] = 0.0;
                    v567 += 1l ;
                }
                v565 += 1l ;
            }
            int v573;
            v573 = 0l;
            while (while_method_5(v573)){
                assert("Tensor range check" && 0 <= v573 && v573 < 1l);
                int v575;
                v575 = 2l * v573;
                int v576;
                v576 = v575 + v526;
                assert("Tensor range check" && 0 <= v573 && v573 < 1l);
                int4* v577;
                v577 = reinterpret_cast<int4*>(v563 + v575);
                int4* v578;
                v578 = reinterpret_cast<int4*>(v521 + v576);
                assert("Pointer alignment check" && (unsigned long long)(v577) % 2l == 0 && (unsigned long long)(v578) % 2l == 0);
                *v578 = *v577;
                int4* v579;
                v579 = reinterpret_cast<int4*>(v564 + v575);
                int4* v580;
                v580 = reinterpret_cast<int4*>(v522 + v576);
                assert("Pointer alignment check" && (unsigned long long)(v579) % 2l == 0 && (unsigned long long)(v580) % 2l == 0);
                *v580 = *v579;
                v573 += 1l ;
            }
            assert("Tensor range check" && 0 <= v516 && v516 < 512l);
            v505 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        assert("Tensor range check" && 0 <= v497 && v497 < 512l);
        asm("barrier.cta.sync %0;" :: "r"(0l));
        assert("Tensor range check" && 0 <= v100 && v100 < 4l);
        assert("Tensor range check" && 0 <= v448 && v448 < 12288l);
        int v581;
        v581 = v111 + v448;
        v98[v581] = 0l;
        v100 += 1l ;
    }
    return ;
}
__device__ inline bool while_method_13(int v0){
    bool v1;
    v1 = v0 < 256l;
    return v1;
}
__device__ float method_54(int * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int v8, int v9, int v10){
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
    while (while_method_5(v46)){
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
        while (while_method_5(v68)){
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
        while (while_method_5(v74)){
            int v76;
            v76 = 0l;
            while (while_method_4(v76)){
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
        while (while_method_5(v100)){
            int v102;
            v102 = 0l;
            while (while_method_4(v102)){
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
        while (while_method_5(v110)){
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
        while (while_method_5(v122)){
            int v124;
            v124 = 0l;
            while (while_method_4(v124)){
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
        while (while_method_5(v136)){
            int v138;
            v138 = 0l;
            while (while_method_4(v138)){
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
        while (while_method_5(v145)){
            int v147;
            v147 = 0l;
            while (while_method_4(v147)){
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
        while (while_method_5(v161)){
            int v163;
            v163 = 0l;
            while (while_method_4(v163)){
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
        Tuple8 tmp57 = Tuple8{0.0f, 2147483647l};
        v175 = tmp57.v0; v176 = tmp57.v1;
        int v177;
        v177 = 0l;
        while (while_method_5(v177)){
            int v179;
            v179 = 0l;
            while (while_method_4(v179)){
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
        Tuple8 tmp58 = cooperative_groups::reduce(v193, Tuple8{v175, v176}, v194);
        v195 = tmp58.v0; v196 = tmp58.v1;
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
        while (while_method_5(v201)){
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
__device__ static_array<float,2l> method_53(unsigned char * v0, unsigned char * v1, unsigned int & v2, static_array_list<Union7,32l> & v3, static_array<Union14,2l> & v4, curandStatePhilox4_32_10_t & v5, Union4 v6){
    static_array<float,2l> v7;
    static_array_list<Union7,32l> & v9 = v3;
    Union3 v10;
    v10 = Union3{Union3_1{v6}};
    Union3 v11;
    v11 = v10;
    while (while_method_11(v11)){
        Union3 v593;
        switch (v11.tag) {
            case 0: { // None
                v593 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v13 = v11.case1.v0;
                switch (v13.tag) {
                    case 0: { // ChanceCommunityCard
                        Union5 v544 = v13.case0.v0; bool v545 = v13.case0.v1; static_array<Union6,2l> v546 = v13.case0.v2; int v547 = v13.case0.v3; static_array<int,2l> v548 = v13.case0.v4; int v549 = v13.case0.v5;
                        unsigned int v550 = v2;
                        Union6 v551; unsigned int v552;
                        Tuple6 tmp52 = draw_card_21(v5, v550);
                        v551 = tmp52.v0; v552 = tmp52.v1;
                        v2 = v552;
                        Union7 v553;
                        v553 = Union7{Union7_0{v551}};
                        v9.push(v553);
                        int v554;
                        v554 = 2l;
                        int v555; int v556;
                        Tuple7 tmp53 = Tuple7{0l, 0l};
                        v555 = tmp53.v0; v556 = tmp53.v1;
                        while (while_method_0(v555)){
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
                        while (while_method_0(v564)){
                            v562[v564] = v556;
                            v564 += 1l ;
                        }
                        Union5 v566;
                        v566 = Union5{Union5_1{v551}};
                        Union4 v567;
                        v567 = Union4{Union4_2{v566, true, v546, 0l, v562, v554}};
                        v593 = Union3{Union3_1{v567}};
                        break;
                    }
                    case 1: { // ChanceInit
                        unsigned int v569 = v2;
                        Union6 v570; unsigned int v571;
                        Tuple6 tmp54 = draw_card_21(v5, v569);
                        v570 = tmp54.v0; v571 = tmp54.v1;
                        v2 = v571;
                        unsigned int v572 = v2;
                        Union6 v573; unsigned int v574;
                        Tuple6 tmp55 = draw_card_21(v5, v572);
                        v573 = tmp55.v0; v574 = tmp55.v1;
                        v2 = v574;
                        Union7 v575;
                        v575 = Union7{Union7_2{0l, v570}};
                        v9.push(v575);
                        Union7 v576;
                        v576 = Union7{Union7_2{1l, v573}};
                        v9.push(v576);
                        int v577;
                        v577 = 2l;
                        static_array<int,2l> v578;
                        v578[0l] = 1l;
                        v578[1l] = 1l;
                        static_array<Union6,2l> v580;
                        v580[0l] = v570;
                        v580[1l] = v573;
                        Union5 v582;
                        v582 = Union5{Union5_0{}};
                        Union4 v583;
                        v583 = Union4{Union4_2{v582, true, v580, 0l, v578, v577}};
                        v593 = Union3{Union3_1{v583}};
                        break;
                    }
                    case 2: { // Round
                        Union5 v54 = v13.case2.v0; bool v55 = v13.case2.v1; static_array<Union6,2l> v56 = v13.case2.v2; int v57 = v13.case2.v3; static_array<int,2l> v58 = v13.case2.v4; int v59 = v13.case2.v5;
                        static_array<Union14,2l> v60 = v4;
                        Union14 v61;
                        v61 = v60[v57];
                        Union1 v360;
                        switch (v61.tag) {
                            case 0: { // T_Computer
                                static_array_list<Union7,32l> & v63 = v3;
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
                                while (while_method_3(v85)){
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
                                static_array_list<Union11,10l> v101;
                                v101 = static_array_list<Union11,10l>{};
                                int v103;
                                v103 = v63.length;
                                int v104;
                                v104 = 0l;
                                while (while_method_1(v103, v104)){
                                    Union7 v106;
                                    v106 = v63[v104];
                                    Union12 v125;
                                    switch (v106.tag) {
                                        case 0: { // CommunityCardIs
                                            Union6 v115 = v106.case0.v0;
                                            Union11 v116;
                                            v116 = Union11{Union11_1{v115}};
                                            v125 = Union12{Union12_1{v116}};
                                            break;
                                        }
                                        case 1: { // PlayerAction
                                            int v118 = v106.case1.v0; Union1 v119 = v106.case1.v1;
                                            Union11 v120;
                                            v120 = Union11{Union11_0{v119}};
                                            v125 = Union12{Union12_1{v120}};
                                            break;
                                        }
                                        case 2: { // PlayerGotCard
                                            int v108 = v106.case2.v0; Union6 v109 = v106.case2.v1;
                                            bool v110;
                                            v110 = v108 == v57;
                                            if (v110){
                                                Union11 v111;
                                                v111 = Union11{Union11_1{v109}};
                                                v125 = Union12{Union12_1{v111}};
                                            } else {
                                                v125 = Union12{Union12_0{}};
                                            }
                                            break;
                                        }
                                        default: {
                                            v125 = Union12{Union12_0{}};
                                        }
                                    }
                                    switch (v125.tag) {
                                        case 0: { // None
                                            break;
                                        }
                                        case 1: { // Some
                                            Union11 v126 = v125.case1.v0;
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
                                while (while_method_1(v131, v132)){
                                    Union11 v134;
                                    v134 = v101[v132];
                                    int v136;
                                    v136 = v132 * 6l;
                                    int v137;
                                    v137 = 1l + v136;
                                    switch (v134.tag) {
                                        case 0: { // C1of2
                                            Union1 v138 = v134.case0.v0;
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
                                            Union6 v141 = v134.case1.v0;
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
                                    method_25(v149, v151, v152, v157, v147, v155);
                                    unsigned int * v158;
                                    v158 = reinterpret_cast<unsigned int *>(&v0[12582912ull]);
                                    assert("Tensor range check" && 0 <= v145 && v145 < 4l);
                                    int v160;
                                    v160 = 12288l * v145;
                                    method_26(v158, v160, v152);
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
                                v213 = int_range_24(v212, v211, v5);
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
                                Tuple8 tmp56 = method_27(v5, v195, v197, v199, v201, v203, v205, v207, v209, v229, v219);
                                v230 = tmp56.v0; v231 = tmp56.v1;
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
                                while (while_method_4(v263)){
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
                                    v275 = method_54(v195, v197, v199, v201, v203, v205, v207, v209, v274, v263, v240);
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
                                        v360 = Union1{Union1_0{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v298;
                                        v298 = v58[0l];
                                        int v300; int v301;
                                        Tuple7 tmp59 = Tuple7{1l, v298};
                                        v300 = tmp59.v0; v301 = tmp59.v1;
                                        while (while_method_0(v300)){
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
                                            v360 = Union1{Union1_0{}};
                                        } else {
                                            v360 = Union1{Union1_1{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        bool v314;
                                        v314 = v59 > 0l;
                                        if (v314){
                                            v360 = Union1{Union1_2{}};
                                        } else {
                                            v360 = Union1{Union1_0{}};
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
                                static_array_list<Union1,3l> v321;
                                v321 = static_array_list<Union1,3l>{};
                                v321.unsafe_set_length(1l);
                                Union1 v323;
                                v323 = Union1{Union1_0{}};
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
                                    Union1 v331;
                                    v331 = Union1{Union1_1{}};
                                    v321.push(v331);
                                } else {
                                }
                                bool v332;
                                v332 = v59 > 0l;
                                if (v332){
                                    Union1 v333;
                                    v333 = Union1{Union1_2{}};
                                    v321.push(v333);
                                } else {
                                }
                                int v334;
                                v334 = v321.length;
                                int v335;
                                v335 = v334 - 1l;
                                int v336;
                                v336 = 0l;
                                while (while_method_1(v335, v336)){
                                    int v338;
                                    v338 = v321.length;
                                    int v339;
                                    v339 = int_range_24(v338, v336, v5);
                                    Union1 v340;
                                    v340 = v321[v336];
                                    Union1 v342;
                                    v342 = v321[v339];
                                    v321[v336] = v342;
                                    v321[v339] = v340;
                                    v336 += 1l ;
                                }
                                Union1 v344;
                                v344 = v321.pop();
                                int v345;
                                v345 = sizeof(Union1);
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
                                Union1 * v354;
                                v354 = reinterpret_cast<Union1 *>(&v350[0ull]);
                                int v356;
                                v356 = threadIdx.x;
                                bool v357;
                                v357 = v356 == 0l;
                                if (v357){
                                    v354[0l] = v344;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                Union1 v358;
                                v358 = v354[0l];
                                asm("barrier.cta.sync %0;" :: "r"(0l));
                                v360 = v358;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union7 v361;
                        v361 = Union7{Union7_1{v57, v360}};
                        v9.push(v361);
                        Union4 v447;
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
                                            v447 = Union4{Union4_2{v54, false, v56, v412, v58, v59}};
                                        } else {
                                            v447 = Union4{Union4_0{v54, v55, v56, v57, v58, v59}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v447 = Union4{Union4_5{v54, v55, v56, v57, v58, v59}};
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
                                            Tuple7 tmp60 = Tuple7{0l, 0l};
                                            v420 = tmp60.v0; v421 = tmp60.v1;
                                            while (while_method_0(v420)){
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
                                            while (while_method_0(v429)){
                                                v427[v429] = v421;
                                                v429 += 1l ;
                                            }
                                            static_array<int,2l> v431;
                                            int v433;
                                            v433 = 0l;
                                            while (while_method_0(v433)){
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
                                            v447 = Union4{Union4_2{v54, false, v56, v418, v431, v419}};
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
                                Union6 v362 = v54.case1.v0;
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
                                            v447 = Union4{Union4_2{v54, false, v56, v365, v58, v59}};
                                        } else {
                                            int v367; int v368;
                                            Tuple7 tmp61 = Tuple7{0l, 0l};
                                            v367 = tmp61.v0; v368 = tmp61.v1;
                                            while (while_method_0(v367)){
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
                                            while (while_method_0(v376)){
                                                v374[v376] = v368;
                                                v376 += 1l ;
                                            }
                                            v447 = Union4{Union4_4{v54, v55, v56, v57, v374, v59}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v447 = Union4{Union4_5{v54, v55, v56, v57, v58, v59}};
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
                                            Tuple7 tmp62 = Tuple7{0l, 0l};
                                            v384 = tmp62.v0; v385 = tmp62.v1;
                                            while (while_method_0(v384)){
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
                                            while (while_method_0(v393)){
                                                v391[v393] = v385;
                                                v393 += 1l ;
                                            }
                                            static_array<int,2l> v395;
                                            int v397;
                                            v397 = 0l;
                                            while (while_method_0(v397)){
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
                                            v447 = Union4{Union4_2{v54, false, v56, v382, v395, v383}};
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
                        v593 = Union3{Union3_1{v447}};
                        break;
                    }
                    case 3: { // RoundWithAction
                        Union5 v449 = v13.case3.v0; bool v450 = v13.case3.v1; static_array<Union6,2l> v451 = v13.case3.v2; int v452 = v13.case3.v3; static_array<int,2l> v453 = v13.case3.v4; int v454 = v13.case3.v5; Union1 v455 = v13.case3.v6;
                        Union7 v456;
                        v456 = Union7{Union7_1{v452, v455}};
                        v9.push(v456);
                        Union4 v542;
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
                                            v542 = Union4{Union4_2{v449, false, v451, v507, v453, v454}};
                                        } else {
                                            v542 = Union4{Union4_0{v449, v450, v451, v452, v453, v454}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v542 = Union4{Union4_5{v449, v450, v451, v452, v453, v454}};
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
                                            Tuple7 tmp63 = Tuple7{0l, 0l};
                                            v515 = tmp63.v0; v516 = tmp63.v1;
                                            while (while_method_0(v515)){
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
                                            while (while_method_0(v524)){
                                                v522[v524] = v516;
                                                v524 += 1l ;
                                            }
                                            static_array<int,2l> v526;
                                            int v528;
                                            v528 = 0l;
                                            while (while_method_0(v528)){
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
                                            v542 = Union4{Union4_2{v449, false, v451, v513, v526, v514}};
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
                                Union6 v457 = v449.case1.v0;
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
                                            v542 = Union4{Union4_2{v449, false, v451, v460, v453, v454}};
                                        } else {
                                            int v462; int v463;
                                            Tuple7 tmp64 = Tuple7{0l, 0l};
                                            v462 = tmp64.v0; v463 = tmp64.v1;
                                            while (while_method_0(v462)){
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
                                            while (while_method_0(v471)){
                                                v469[v471] = v463;
                                                v471 += 1l ;
                                            }
                                            v542 = Union4{Union4_4{v449, v450, v451, v452, v469, v454}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v542 = Union4{Union4_5{v449, v450, v451, v452, v453, v454}};
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
                                            Tuple7 tmp65 = Tuple7{0l, 0l};
                                            v479 = tmp65.v0; v480 = tmp65.v1;
                                            while (while_method_0(v479)){
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
                                            while (while_method_0(v488)){
                                                v486[v488] = v480;
                                                v488 += 1l ;
                                            }
                                            static_array<int,2l> v490;
                                            int v492;
                                            v492 = 0l;
                                            while (while_method_0(v492)){
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
                                            v542 = Union4{Union4_2{v449, false, v451, v477, v490, v478}};
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
                        v593 = Union3{Union3_1{v542}};
                        break;
                    }
                    case 4: { // TerminalCall
                        Union5 v30 = v13.case4.v0; bool v31 = v13.case4.v1; static_array<Union6,2l> v32 = v13.case4.v2; int v33 = v13.case4.v3; static_array<int,2l> v34 = v13.case4.v4; int v35 = v13.case4.v5;
                        int v36;
                        v36 = v34[v33];
                        Union13 v38;
                        v38 = compare_hands_29(v30, v31, v32, v33, v34, v35);
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
                        Union7 v52;
                        v52 = Union7{Union7_3{v32, v43, v44}};
                        v9.push(v52);
                        v593 = Union3{Union3_0{}};
                        break;
                    }
                    case 5: { // TerminalFold
                        Union5 v14 = v13.case5.v0; bool v15 = v13.case5.v1; static_array<Union6,2l> v16 = v13.case5.v2; int v17 = v13.case5.v3; static_array<int,2l> v18 = v13.case5.v4; int v19 = v13.case5.v5;
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
                        Union7 v28;
                        v28 = Union7{Union7_3{v16, v20, v27}};
                        v9.push(v28);
                        v593 = Union3{Union3_0{}};
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
__device__ inline bool while_method_14(int v0){
    bool v1;
    v1 = v0 < 128l;
    return v1;
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1, unsigned char * v2, unsigned long long v3, unsigned char * v4, unsigned long long v5) {
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
    __shared__ int v13;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v14;
    v14 = threadIdx.x;
    bool v15;
    v15 = v14 == 0l;
    if (v15){
        new(&v13) int{512l};
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v16;
    v16 = threadIdx.x;
    int v17;
    v17 = blockIdx.x;
    int v18;
    v18 = v17 * 512l;
    int v19;
    v19 = v16 + v18;
    bool v20;
    v20 = v19 == 0l;
    if (v20){
        Union0 v21;
        v21 = f_0(v1);
        unsigned int v22; Union3 v23; static_array_list<Union7,32l> v24; static_array<Union2,2l> v25; Union8 v26;
        Tuple0 tmp10 = f_6(v0);
        v22 = tmp10.v0; v23 = tmp10.v1; v24 = tmp10.v2; v25 = tmp10.v3; v26 = tmp10.v4;
        Union8 & v27 = v26;
        static_array<Union2,2l> & v28 = v25;
        Union3 & v29 = v23;
        unsigned int & v30 = v22;
        static_array_list<Union7,32l> & v31 = v24;
        switch (v21.tag) {
            case 0: { // ActionSelected
                Union1 v44 = v21.case0.v0;
                Union3 v45 = v29;
                switch (v45.tag) {
                    case 0: { // None
                        printf("%s\n", "The game hasn't been started in ActionSelected.");
                        __trap();
                        break;
                    }
                    case 1: { // Some
                        Union4 v46 = v45.case1.v0;
                        switch (v46.tag) {
                            case 2: { // Round
                                Union5 v47 = v46.case2.v0; bool v48 = v46.case2.v1; static_array<Union6,2l> v49 = v46.case2.v2; int v50 = v46.case2.v3; static_array<int,2l> v51 = v46.case2.v4; int v52 = v46.case2.v5;
                                Union4 v53;
                                v53 = Union4{Union4_3{v47, v48, v49, v50, v51, v52, v44}};
                                play_loop_20(v2, v3, v4, v5, v30, v29, v31, v28, v12, v27, v53);
                                break;
                            }
                            default: {
                                printf("%s\n", "Unexpected game node in ActionSelected.");
                                __trap();
                            }
                        }
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                break;
            }
            case 1: { // PlayerChanged
                static_array<Union2,2l> v43 = v21.case1.v0;
                v28 = v43;
                break;
            }
            case 2: { // StartGame
                static_array<Union2,2l> v32;
                Union2 v34;
                v34 = Union2{Union2_0{}};
                v32[0l] = v34;
                Union2 v36;
                v36 = Union2{Union2_1{}};
                v32[1l] = v36;
                static_array_list<Union7,32l> v38;
                v38 = static_array_list<Union7,32l>{};
                Union8 v40;
                v40 = Union8{Union8_0{}};
                v27 = v40;
                v28 = v32;
                Union3 v41;
                v41 = Union3{Union3_0{}};
                v29 = v41;
                v30 = 63ul;
                v31 = v38;
                Union4 v42;
                v42 = Union4{Union4_1{}};
                play_loop_20(v2, v3, v4, v5, v30, v29, v31, v28, v12, v27, v42);
                break;
            }
            case 3: { // StartTrainingVsRando
                printf("%s\n", "Training is not supported in the `event_loop_play` function.");
                __trap();
                break;
            }
            case 4: { // StartTrainingVsSelf
                printf("%s\n", "Training is not supported in the `event_loop_play` function.");
                __trap();
                break;
            }
            default: {
                assert("Invalid tag." && false); __trap();
            }
        }
        f_33(v0, v22, v23, v24, v25, v26);
    } else {
    }
    int v54;
    v54 = atomicAdd(&v13,-1l);
    while (while_method_8()){
        bool v56;
        v56 = 3866640ull == v5;
        bool v57;
        v57 = v56 == false;
        if (v57){
            assert("The params needs to have matching offsets." && v56);
        } else {
        }
        bool v59;
        v59 = 50528256ull == v3;
        bool v60;
        v60 = v59 == false;
        if (v60){
            assert("The outputs needs to have matching offsets." && v59);
        } else {
        }
        Union9 v62;
        v62 = Union9{Union9_0{}};
        Union10 v63;
        v63 = noinline_run_23(v12, v2, v4, v62);
        int v64 = v13;
        bool v65;
        v65 = v64 == 0l;
        if (v65){
            break;
        } else {
        }
    }
    return ;
}
extern "C" __global__ void entry1(unsigned char * v0, unsigned long long v1, unsigned char * v2, unsigned long long v3, float * v4, float * v5, float * v6) {
    bool v7;
    v7 = 3866640ull == v3;
    bool v8;
    v8 = v7 == false;
    if (v8){
        assert("The params needs to have matching offsets." && v7);
    } else {
    }
    bool v10;
    v10 = 50528256ull == v1;
    bool v11;
    v11 = v10 == false;
    if (v11){
        assert("The outputs needs to have matching offsets." && v10);
    } else {
    }
    auto v13 = cooperative_groups::this_grid();
    unsigned long long v14;
    v14 = clock64();
    int v15;
    v15 = threadIdx.x;
    int v16;
    v16 = blockIdx.x;
    int v17;
    v17 = v16 * 512l;
    int v18;
    v18 = v15 + v17;
    unsigned long long v19;
    v19 = (unsigned long long)v18;
    curandStatePhilox4_32_10_t v20;
    curand_init(v14,v19,0ull,&v20);
    int v21;
    v21 = 0l;
    while (while_method_9(v21)){
        int v23;
        v23 = 0l;
        while (while_method_10(v23)){
            int v25;
            v25 = 0l;
            noinline_train_50(v4, v5, v0, v2, v20, v21, v25);
            int v26;
            v26 = 1l;
            noinline_train_50(v4, v5, v0, v2, v20, v21, v26);
            v23 += 1l ;
        }
        unsigned int * v27;
        v27 = reinterpret_cast<unsigned int *>(&v0[12582912ull]);
        int * v29;
        v29 = reinterpret_cast<int *>(&v2[262144ull]);
        float * v31;
        v31 = reinterpret_cast<float *>(&v2[262160ull]);
        float * v33;
        v33 = reinterpret_cast<float *>(&v2[524304ull]);
        float * v35;
        v35 = reinterpret_cast<float *>(&v2[786448ull]);
        float * v37;
        v37 = reinterpret_cast<float *>(&v2[1048592ull]);
        float * v39;
        v39 = reinterpret_cast<float *>(&v2[1310736ull]);
        float * v41;
        v41 = reinterpret_cast<float *>(&v2[1572880ull]);
        float * v43;
        v43 = reinterpret_cast<float *>(&v2[1835024ull]);
        int * v45;
        v45 = reinterpret_cast<int *>(&v0[12779520ull]);
        float * v47;
        v47 = reinterpret_cast<float *>(&v0[15925248ull]);
        int * v49;
        v49 = reinterpret_cast<int *>(&v0[19070976ull]);
        int * v51;
        v51 = reinterpret_cast<int *>(&v0[22216704ull]);
        double * v53;
        v53 = reinterpret_cast<double *>(&v0[25362432ull]);
        double * v55;
        v55 = reinterpret_cast<double *>(&v0[37945344ull]);
        double * v57;
        v57 = reinterpret_cast<double *>(&v2[2097168ull]);
        double * v59;
        v59 = reinterpret_cast<double *>(&v2[2883600ull]);
        int * v61;
        v61 = reinterpret_cast<int *>(&v2[3670032ull]);
        v13.sync() ;
        int v63;
        v63 = threadIdx.x;
        int v64;
        v64 = blockIdx.x;
        int v65;
        v65 = v64 * 512l;
        int v66;
        v66 = v63 + v65;
        bool v67;
        v67 = v66 == 0l;
        if (v67){
            int v68;
            v68 = 0l;
            int v69;
            v69 = 4l;
            int v70;
            v70 = int_range_24(v69, v68, v20);
            v29[0l] = v70;
        } else {
        }
        __syncwarp();
        int v71;
        v71 = threadIdx.x;
        bool v72;
        v72 = 0l <= v71;
        bool v73;
        v73 = v72 == false;
        if (v73){
            assert("The index needs to be zero or positive." && v72);
        } else {
        }
        int v75;
        v75 = v71 % 1l;
        int v76;
        v76 = v71 % 512l;
        int v77;
        v77 = v71 / 512l;
        bool v78;
        v78 = v77 < 1l;
        bool v79;
        v79 = v78 == false;
        if (v79){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v78);
        } else {
        }
        assert("Tensor range check" && 0 <= v77 && v77 < 1l);
        assert("Tensor range check" && 0 <= v76 && v76 < 512l);
        assert("Tensor range check" && 0 <= v75 && v75 < 1l);
        int v81;
        v81 = 4l * v75;
        int v82;
        v82 = 4l * v76;
        int v83;
        v83 = v82 + v81;
        int v84;
        v84 = 16384l * v77;
        int v85;
        v85 = v84 + v83;
        assert("Tensor range check" && 0 <= v77 && v77 < 1l);
        assert("Tensor range check" && 0 <= v76 && v76 < 512l);
        assert("Tensor range check" && 0 <= v75 && v75 < 1l);
        int v86;
        v86 = blockIdx.x;
        int v87;
        v87 = v86;
        while (while_method_7(v87)){
            bool v89;
            v89 = 0l <= v87;
            bool v90;
            v90 = v89 == false;
            if (v90){
                assert("The index needs to be zero or positive." && v89);
            } else {
            }
            int v92;
            v92 = v87 % 8l;
            int v93;
            v93 = v87 / 8l;
            bool v94;
            v94 = v93 < 4l;
            bool v95;
            v95 = v94 == false;
            if (v95){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v94);
            } else {
            }
            assert("Tensor range check" && 0 <= v93 && v93 < 4l);
            assert("Tensor range check" && 0 <= v92 && v92 < 8l);
            int v97;
            v97 = 2048l * v92;
            int v98;
            v98 = v97 + v85;
            int v99;
            v99 = 16384l * v93;
            int v100;
            v100 = v99 + v98;
            float v101[4l];
            float v102[4l];
            float v103[4l];
            float v104[4l];
            float v105[4l];
            float v106[4l];
            float v107[4l];
            int v108[4l];
            int v109;
            v109 = 0l;
            while (while_method_5(v109)){
                assert("Tensor range check" && 0 <= v109 && v109 < 1l);
                int v111;
                v111 = 4l * v109;
                assert("Tensor range check" && 0 <= v109 && v109 < 1l);
                int v112;
                v112 = v111 + v100;
                int4* v113;
                v113 = reinterpret_cast<int4*>(v31 + v112);
                int4* v114;
                v114 = reinterpret_cast<int4*>(v101 + v111);
                assert("Pointer alignment check" && (unsigned long long)(v113) % 4l == 0 && (unsigned long long)(v114) % 4l == 0);
                *v114 = *v113;
                int4* v115;
                v115 = reinterpret_cast<int4*>(v33 + v112);
                int4* v116;
                v116 = reinterpret_cast<int4*>(v102 + v111);
                assert("Pointer alignment check" && (unsigned long long)(v115) % 4l == 0 && (unsigned long long)(v116) % 4l == 0);
                *v116 = *v115;
                int4* v117;
                v117 = reinterpret_cast<int4*>(v35 + v112);
                int4* v118;
                v118 = reinterpret_cast<int4*>(v103 + v111);
                assert("Pointer alignment check" && (unsigned long long)(v117) % 4l == 0 && (unsigned long long)(v118) % 4l == 0);
                *v118 = *v117;
                int4* v119;
                v119 = reinterpret_cast<int4*>(v37 + v112);
                int4* v120;
                v120 = reinterpret_cast<int4*>(v104 + v111);
                assert("Pointer alignment check" && (unsigned long long)(v119) % 4l == 0 && (unsigned long long)(v120) % 4l == 0);
                *v120 = *v119;
                int4* v121;
                v121 = reinterpret_cast<int4*>(v39 + v112);
                int4* v122;
                v122 = reinterpret_cast<int4*>(v105 + v111);
                assert("Pointer alignment check" && (unsigned long long)(v121) % 4l == 0 && (unsigned long long)(v122) % 4l == 0);
                *v122 = *v121;
                int4* v123;
                v123 = reinterpret_cast<int4*>(v41 + v112);
                int4* v124;
                v124 = reinterpret_cast<int4*>(v106 + v111);
                assert("Pointer alignment check" && (unsigned long long)(v123) % 4l == 0 && (unsigned long long)(v124) % 4l == 0);
                *v124 = *v123;
                int4* v125;
                v125 = reinterpret_cast<int4*>(v43 + v112);
                int4* v126;
                v126 = reinterpret_cast<int4*>(v107 + v111);
                assert("Pointer alignment check" && (unsigned long long)(v125) % 4l == 0 && (unsigned long long)(v126) % 4l == 0);
                *v126 = *v125;
                v109 += 1l ;
            }
            int v127;
            v127 = 0l;
            while (while_method_5(v127)){
                int v129;
                v129 = 0l;
                while (while_method_4(v129)){
                    bool v131;
                    v131 = 0l <= v129;
                    bool v133;
                    if (v131){
                        bool v132;
                        v132 = v129 < 4l;
                        v133 = v132;
                    } else {
                        v133 = false;
                    }
                    bool v134;
                    v134 = v133 == false;
                    if (v134){
                        assert("The indices should be inside the range of the dimension." && v133);
                    } else {
                    }
                    bool v136;
                    v136 = 0l <= v75;
                    bool v138;
                    if (v136){
                        bool v137;
                        v137 = v75 < 1l;
                        v138 = v137;
                    } else {
                        v138 = false;
                    }
                    bool v139;
                    v139 = v138 == false;
                    if (v139){
                        assert("The indices should be inside the range of the dimension." && v138);
                    } else {
                    }
                    int v141;
                    v141 = v75 * 4l;
                    int v142;
                    v142 = v129 + v141;
                    bool v143;
                    v143 = 0l <= v127;
                    bool v145;
                    if (v143){
                        bool v144;
                        v144 = v127 < 1l;
                        v145 = v144;
                    } else {
                        v145 = false;
                    }
                    bool v146;
                    v146 = v145 == false;
                    if (v146){
                        assert("The indices should be inside the range of the dimension." && v145);
                    } else {
                    }
                    int v148;
                    v148 = v127 * 4l;
                    int v149;
                    v149 = v142 + v148;
                    assert("Tensor range check" && 0 <= v127 && v127 < 1l);
                    assert("Tensor range check" && 0 <= v129 && v129 < 4l);
                    int v150;
                    v150 = 4l * v127;
                    int v151;
                    v151 = v150 + v129;
                    v108[v151] = v149;
                    v129 += 1l ;
                }
                v127 += 1l ;
            }
            bool v152;
            v152 = 0l <= v77;
            bool v153;
            v153 = v152 && v78;
            bool v154;
            v154 = v153 == false;
            if (v154){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v153);
            } else {
            }
            bool v156;
            v156 = 0l <= v76;
            bool v158;
            if (v156){
                bool v157;
                v157 = v76 < 512l;
                v158 = v157;
            } else {
                v158 = false;
            }
            bool v159;
            v159 = v158 == false;
            if (v159){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v158);
            } else {
            }
            bool v161;
            v161 = 0l <= v93;
            bool v162;
            v162 = v161 && v94;
            bool v163;
            v163 = v162 == false;
            if (v163){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v162);
            } else {
            }
            bool v165;
            v165 = 0l <= v92;
            bool v167;
            if (v165){
                bool v166;
                v166 = v92 < 8l;
                v167 = v166;
            } else {
                v167 = false;
            }
            bool v168;
            v168 = v167 == false;
            if (v168){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v167);
            } else {
            }
            int v170;
            v170 = v92 * 512l;
            int v171;
            v171 = v93 + v77;
            int v172;
            v172 = v170 + v76;
            bool v173[4l];
            int v174;
            v174 = 0l;
            while (while_method_5(v174)){
                int v176;
                v176 = 0l;
                while (while_method_4(v176)){
                    assert("Tensor range check" && 0 <= v174 && v174 < 1l);
                    assert("Tensor range check" && 0 <= v176 && v176 < 4l);
                    int v178;
                    v178 = 4l * v174;
                    int v179;
                    v179 = v178 + v176;
                    float v180;
                    v180 = v103[v179];
                    bool v181;
                    v181 = v180 == 0.0f;
                    bool v182;
                    v182 = v181 != true;
                    assert("Tensor range check" && 0 <= v174 && v174 < 1l);
                    assert("Tensor range check" && 0 <= v176 && v176 < 4l);
                    v173[v179] = v182;
                    v176 += 1l ;
                }
                v174 += 1l ;
            }
            bool v183;
            v183 = false;
            int v184;
            v184 = 0l;
            while (while_method_5(v184)){
                int v186;
                v186 = 0l;
                while (while_method_4(v186)){
                    assert("Tensor range check" && 0 <= v184 && v184 < 1l);
                    assert("Tensor range check" && 0 <= v186 && v186 < 4l);
                    int v188;
                    v188 = 4l * v184;
                    int v189;
                    v189 = v188 + v186;
                    bool v190;
                    v190 = v173[v189];
                    bool v191;
                    v191 = v183 || v190;
                    v183 = v191;
                    v186 += 1l ;
                }
                v184 += 1l ;
            }
            auto v192 = cooperative_groups::coalesced_threads();
            int v193;
            v193 = threadIdx.x;
            auto v194 = cooperative_groups::labeled_partition(v192,v193);
            Closure8 v195{};
            bool v196;
            v196 = cooperative_groups::reduce(v194, v183, v195);
            if (v196){
                float v197[4l];
                int v198;
                v198 = 0l;
                while (while_method_5(v198)){
                    int v200;
                    v200 = 0l;
                    while (while_method_4(v200)){
                        assert("Tensor range check" && 0 <= v198 && v198 < 1l);
                        assert("Tensor range check" && 0 <= v200 && v200 < 4l);
                        int v202;
                        v202 = 4l * v198;
                        int v203;
                        v203 = v202 + v200;
                        float v204;
                        v204 = v102[v203];
                        float v205;
                        v205 = v103[v203];
                        float v206;
                        v206 = v204 + v205;
                        bool v207;
                        v207 = 0.0f >= v206;
                        float v208;
                        if (v207){
                            v208 = 0.0f;
                        } else {
                            v208 = v206;
                        }
                        assert("Tensor range check" && 0 <= v198 && v198 < 1l);
                        assert("Tensor range check" && 0 <= v200 && v200 < 4l);
                        v197[v203] = v208;
                        v200 += 1l ;
                    }
                    v198 += 1l ;
                }
                float v209[4l];
                int v210;
                v210 = 0l;
                while (while_method_5(v210)){
                    int v212;
                    v212 = 0l;
                    while (while_method_4(v212)){
                        assert("Tensor range check" && 0 <= v210 && v210 < 1l);
                        assert("Tensor range check" && 0 <= v212 && v212 < 4l);
                        int v214;
                        v214 = 4l * v210;
                        int v215;
                        v215 = v214 + v212;
                        float v216;
                        v216 = v197[v215];
                        bool v217;
                        v217 = 0.0f >= v216;
                        float v218;
                        if (v217){
                            v218 = 0.0f;
                        } else {
                            v218 = v216;
                        }
                        assert("Tensor range check" && 0 <= v210 && v210 < 1l);
                        assert("Tensor range check" && 0 <= v212 && v212 < 4l);
                        v209[v215] = v218;
                        v212 += 1l ;
                    }
                    v210 += 1l ;
                }
                float v219;
                v219 = 0.0f;
                int v220;
                v220 = 0l;
                while (while_method_5(v220)){
                    int v222;
                    v222 = 0l;
                    while (while_method_4(v222)){
                        assert("Tensor range check" && 0 <= v220 && v220 < 1l);
                        assert("Tensor range check" && 0 <= v222 && v222 < 4l);
                        int v224;
                        v224 = 4l * v220;
                        int v225;
                        v225 = v224 + v222;
                        float v226;
                        v226 = v209[v225];
                        float v227;
                        v227 = v219 + v226;
                        v219 = v227;
                        v222 += 1l ;
                    }
                    v220 += 1l ;
                }
                auto v228 = cooperative_groups::coalesced_threads();
                int v229;
                v229 = threadIdx.x;
                auto v230 = cooperative_groups::labeled_partition(v228,v229);
                Closure1 v231{};
                float v232;
                v232 = cooperative_groups::reduce(v230, v219, v231);
                float v233[4l];
                int v234;
                v234 = 0l;
                while (while_method_5(v234)){
                    int v236;
                    v236 = 0l;
                    while (while_method_4(v236)){
                        assert("Tensor range check" && 0 <= v234 && v234 < 1l);
                        assert("Tensor range check" && 0 <= v236 && v236 < 4l);
                        int v238;
                        v238 = 4l * v234;
                        int v239;
                        v239 = v238 + v236;
                        float v240;
                        v240 = v209[v239];
                        bool v241;
                        v241 = v232 == 0.0f;
                        bool v242;
                        v242 = v241 != true;
                        float v244;
                        if (v242){
                            float v243;
                            v243 = v240 / v232;
                            v244 = v243;
                        } else {
                            v244 = 0.25f;
                        }
                        assert("Tensor range check" && 0 <= v234 && v234 < 1l);
                        assert("Tensor range check" && 0 <= v236 && v236 < 4l);
                        v233[v239] = v244;
                        v236 += 1l ;
                    }
                    v234 += 1l ;
                }
                float v245[4l];
                int v246;
                v246 = 0l;
                while (while_method_5(v246)){
                    int v248;
                    v248 = 0l;
                    while (while_method_4(v248)){
                        assert("Tensor range check" && 0 <= v246 && v246 < 1l);
                        assert("Tensor range check" && 0 <= v248 && v248 < 4l);
                        int v250;
                        v250 = 4l * v246;
                        int v251;
                        v251 = v250 + v248;
                        float v252;
                        v252 = v101[v251];
                        float v253;
                        v253 = v233[v251];
                        float v254;
                        v254 = v252 + v253;
                        assert("Tensor range check" && 0 <= v246 && v246 < 1l);
                        assert("Tensor range check" && 0 <= v248 && v248 < 4l);
                        v245[v251] = v254;
                        v248 += 1l ;
                    }
                    v246 += 1l ;
                }
                float v255[4l];
                int v256;
                v256 = 0l;
                while (while_method_5(v256)){
                    int v258;
                    v258 = 0l;
                    while (while_method_4(v258)){
                        assert("Tensor range check" && 0 <= v256 && v256 < 1l);
                        assert("Tensor range check" && 0 <= v258 && v258 < 4l);
                        int v260;
                        v260 = 4l * v256;
                        int v261;
                        v261 = v260 + v258;
                        float v262;
                        v262 = v245[v261];
                        float v263;
                        v263 = -v262;
                        bool v264;
                        v264 = v262 >= v263;
                        float v265;
                        if (v264){
                            v265 = v262;
                        } else {
                            v265 = v263;
                        }
                        assert("Tensor range check" && 0 <= v256 && v256 < 1l);
                        assert("Tensor range check" && 0 <= v258 && v258 < 4l);
                        v255[v261] = v265;
                        v258 += 1l ;
                    }
                    v256 += 1l ;
                }
                float v266;
                v266 = 0.0f;
                int v267;
                v267 = 0l;
                while (while_method_5(v267)){
                    int v269;
                    v269 = 0l;
                    while (while_method_4(v269)){
                        assert("Tensor range check" && 0 <= v267 && v267 < 1l);
                        assert("Tensor range check" && 0 <= v269 && v269 < 4l);
                        int v271;
                        v271 = 4l * v267;
                        int v272;
                        v272 = v271 + v269;
                        float v273;
                        v273 = v255[v272];
                        float v274;
                        v274 = v266 + v273;
                        v266 = v274;
                        v269 += 1l ;
                    }
                    v267 += 1l ;
                }
                auto v275 = cooperative_groups::coalesced_threads();
                int v276;
                v276 = threadIdx.x;
                auto v277 = cooperative_groups::labeled_partition(v275,v276);
                float v278;
                v278 = cooperative_groups::reduce(v277, v266, v231);
                bool v279;
                v279 = v278 > 100.0f;
                float v281;
                if (v279){
                    float v280;
                    v280 = 100.0f / v278;
                    v281 = v280;
                } else {
                    v281 = 1.0f;
                }
                float v282[4l];
                int v283;
                v283 = 0l;
                while (while_method_5(v283)){
                    int v285;
                    v285 = 0l;
                    while (while_method_4(v285)){
                        assert("Tensor range check" && 0 <= v283 && v283 < 1l);
                        assert("Tensor range check" && 0 <= v285 && v285 < 4l);
                        int v287;
                        v287 = 4l * v283;
                        int v288;
                        v288 = v287 + v285;
                        float v289;
                        v289 = v255[v288];
                        float v290;
                        v290 = v281 * v289;
                        assert("Tensor range check" && 0 <= v283 && v283 < 1l);
                        assert("Tensor range check" && 0 <= v285 && v285 < 4l);
                        v282[v288] = v290;
                        v285 += 1l ;
                    }
                    v283 += 1l ;
                }
                float v291[4l];
                float v292[4l];
                int v293;
                v293 = 0l;
                while (while_method_5(v293)){
                    int v295;
                    v295 = 0l;
                    while (while_method_4(v295)){
                        assert("Tensor range check" && 0 <= v293 && v293 < 1l);
                        assert("Tensor range check" && 0 <= v295 && v295 < 4l);
                        int v297;
                        v297 = 4l * v293;
                        int v298;
                        v298 = v297 + v295;
                        float v299;
                        v299 = v101[v298];
                        float v300;
                        v300 = v102[v298];
                        float v301;
                        v301 = v103[v298];
                        float v302;
                        v302 = v104[v298];
                        float v303;
                        v303 = v105[v298];
                        float v304;
                        v304 = v106[v298];
                        float v305;
                        v305 = v107[v298];
                        float v306;
                        v306 = v302 + v304;
                        float v307;
                        v307 = v303 + v305;
                        assert("Tensor range check" && 0 <= v293 && v293 < 1l);
                        assert("Tensor range check" && 0 <= v295 && v295 < 4l);
                        v291[v298] = v306;
                        v292[v298] = v307;
                        v295 += 1l ;
                    }
                    v293 += 1l ;
                }
                int v308;
                v308 = 0l;
                while (while_method_5(v308)){
                    int v310;
                    v310 = 0l;
                    while (while_method_4(v310)){
                        assert("Tensor range check" && 0 <= v308 && v308 < 1l);
                        assert("Tensor range check" && 0 <= v310 && v310 < 4l);
                        int v312;
                        v312 = 4l * v308;
                        int v313;
                        v313 = v312 + v310;
                        float v314;
                        v314 = v282[v313];
                        float v315;
                        v315 = v197[v313];
                        float v316;
                        v316 = v291[v313];
                        float v317;
                        v317 = v292[v313];
                        assert("Tensor range check" && 0 <= v308 && v308 < 1l);
                        assert("Tensor range check" && 0 <= v310 && v310 < 4l);
                        v101[v313] = v314;
                        v102[v313] = v315;
                        v103[v313] = 0.0f;
                        v104[v313] = v316;
                        v105[v313] = v317;
                        v106[v313] = 0.0f;
                        v107[v313] = 0.0f;
                        v310 += 1l ;
                    }
                    v308 += 1l ;
                }
            } else {
            }
            assert("Tensor range check" && 0 <= v93 && v93 < 4l);
            assert("Tensor range check" && 0 <= v92 && v92 < 8l);
            int v318;
            v318 = 0l;
            while (while_method_5(v318)){
                assert("Tensor range check" && 0 <= v318 && v318 < 1l);
                int v320;
                v320 = 4l * v318;
                int v321;
                v321 = v320 + v100;
                assert("Tensor range check" && 0 <= v318 && v318 < 1l);
                int4* v322;
                v322 = reinterpret_cast<int4*>(v101 + v320);
                int4* v323;
                v323 = reinterpret_cast<int4*>(v31 + v321);
                assert("Pointer alignment check" && (unsigned long long)(v322) % 4l == 0 && (unsigned long long)(v323) % 4l == 0);
                *v323 = *v322;
                int4* v324;
                v324 = reinterpret_cast<int4*>(v102 + v320);
                int4* v325;
                v325 = reinterpret_cast<int4*>(v33 + v321);
                assert("Pointer alignment check" && (unsigned long long)(v324) % 4l == 0 && (unsigned long long)(v325) % 4l == 0);
                *v325 = *v324;
                int4* v326;
                v326 = reinterpret_cast<int4*>(v103 + v320);
                int4* v327;
                v327 = reinterpret_cast<int4*>(v35 + v321);
                assert("Pointer alignment check" && (unsigned long long)(v326) % 4l == 0 && (unsigned long long)(v327) % 4l == 0);
                *v327 = *v326;
                int4* v328;
                v328 = reinterpret_cast<int4*>(v104 + v320);
                int4* v329;
                v329 = reinterpret_cast<int4*>(v37 + v321);
                assert("Pointer alignment check" && (unsigned long long)(v328) % 4l == 0 && (unsigned long long)(v329) % 4l == 0);
                *v329 = *v328;
                int4* v330;
                v330 = reinterpret_cast<int4*>(v105 + v320);
                int4* v331;
                v331 = reinterpret_cast<int4*>(v39 + v321);
                assert("Pointer alignment check" && (unsigned long long)(v330) % 4l == 0 && (unsigned long long)(v331) % 4l == 0);
                *v331 = *v330;
                int4* v332;
                v332 = reinterpret_cast<int4*>(v106 + v320);
                int4* v333;
                v333 = reinterpret_cast<int4*>(v41 + v321);
                assert("Pointer alignment check" && (unsigned long long)(v332) % 4l == 0 && (unsigned long long)(v333) % 4l == 0);
                *v333 = *v332;
                int4* v334;
                v334 = reinterpret_cast<int4*>(v107 + v320);
                int4* v335;
                v335 = reinterpret_cast<int4*>(v43 + v321);
                assert("Pointer alignment check" && (unsigned long long)(v334) % 4l == 0 && (unsigned long long)(v335) % 4l == 0);
                *v335 = *v334;
                v318 += 1l ;
            }
            v87 += 24l ;
        }
        v13.sync() ;
        v21 += 1l ;
    }
    int v336;
    v336 = threadIdx.x;
    int v337;
    v337 = blockIdx.x;
    int v338;
    v338 = v337 * 512l;
    int v339;
    v339 = v336 + v338;
    int v340;
    v340 = v339;
    while (while_method_13(v340)){
        bool v342;
        v342 = 0l <= v340;
        bool v343;
        v343 = v342 == false;
        if (v343){
            assert("The index needs to be zero or positive." && v342);
        } else {
        }
        int v345;
        v345 = v340 % 64l;
        int v346;
        v346 = v340 / 64l;
        bool v347;
        v347 = v346 < 4l;
        bool v348;
        v348 = v347 == false;
        if (v348){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v347);
        } else {
        }
        assert("Tensor range check" && 0 <= v346 && v346 < 4l);
        assert("Tensor range check" && 0 <= v345 && v345 < 64l);
        int v350;
        v350 = 4l * v345;
        int v351;
        v351 = 256l * v346;
        int v352;
        v352 = v351 + v350;
        assert("Tensor range check" && 0 <= v346 && v346 < 4l);
        assert("Tensor range check" && 0 <= v345 && v345 < 64l);
        float v353[4l];
        float v354[4l];
        float v355[4l];
        int4* v356;
        v356 = reinterpret_cast<int4*>(v4 + v352);
        int4* v357;
        v357 = reinterpret_cast<int4*>(v353 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v356) % 4l == 0 && (unsigned long long)(v357) % 4l == 0);
        *v357 = *v356;
        int4* v358;
        v358 = reinterpret_cast<int4*>(v5 + v352);
        int4* v359;
        v359 = reinterpret_cast<int4*>(v354 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v358) % 4l == 0 && (unsigned long long)(v359) % 4l == 0);
        *v359 = *v358;
        // Pushing the loop unrolling to: 0
        int v360;
        v360 = 0l;
        #pragma unroll
        while (while_method_4(v360)){
            assert("Tensor range check" && 0 <= v360 && v360 < 4l);
            float v362;
            v362 = v353[v360];
            float v363;
            v363 = v354[v360];
            bool v364;
            v364 = v363 == 0.0f;
            bool v365;
            v365 = v364 != true;
            float v367;
            if (v365){
                float v366;
                v366 = v362 / v363;
                v367 = v366;
            } else {
                v367 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v360 && v360 < 4l);
            v355[v360] = v367;
            v360 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v368;
        v368 = reinterpret_cast<int4*>(v355 + 0l);
        int4* v369;
        v369 = reinterpret_cast<int4*>(v6 + v352);
        assert("Pointer alignment check" && (unsigned long long)(v368) % 4l == 0 && (unsigned long long)(v369) % 4l == 0);
        *v369 = *v368;
        v340 += 12288l ;
    }
    v13.sync() ;
    return ;
}
extern "C" __global__ void entry2(unsigned char * v0, unsigned long long v1, unsigned char * v2, unsigned long long v3, float * v4, float * v5, float * v6) {
    bool v7;
    v7 = 3866640ull == v3;
    bool v8;
    v8 = v7 == false;
    if (v8){
        assert("The params needs to have matching offsets." && v7);
    } else {
    }
    bool v10;
    v10 = 50528256ull == v1;
    bool v11;
    v11 = v10 == false;
    if (v11){
        assert("The outputs needs to have matching offsets." && v10);
    } else {
    }
    auto v13 = cooperative_groups::this_grid();
    unsigned long long v14;
    v14 = clock64();
    int v15;
    v15 = threadIdx.x;
    int v16;
    v16 = blockIdx.x;
    int v17;
    v17 = v16 * 512l;
    int v18;
    v18 = v15 + v17;
    unsigned long long v19;
    v19 = (unsigned long long)v18;
    curandStatePhilox4_32_10_t v20;
    curand_init(v14,v19,0ull,&v20);
    int v21;
    v21 = 0l;
    while (while_method_9(v21)){
        int v23;
        v23 = 0l;
        while (while_method_10(v23)){
            static_array<Union14,2l> v25;
            Union14 v27;
            v27 = Union14{Union14_0{}};
            v25[0l] = v27;
            Union14 v29;
            v29 = Union14{Union14_0{}};
            v25[1l] = v29;
            static_array<Union14,2l> & v31 = v25;
            unsigned int v32 = 63ul;
            static_array_list<Union7,32l> v33;
            v33 = static_array_list<Union7,32l>{};
            static_array_list<Union7,32l> & v35 = v33;
            Union4 v36;
            v36 = Union4{Union4_1{}};
            static_array<float,2l> v37;
            v37 = method_51(v0, v2, v32, v35, v31, v20, v36);
            unsigned int * v38;
            v38 = reinterpret_cast<unsigned int *>(&v0[12582912ull]);
            int * v40;
            v40 = reinterpret_cast<int *>(&v2[262144ull]);
            float * v42;
            v42 = reinterpret_cast<float *>(&v2[262160ull]);
            float * v44;
            v44 = reinterpret_cast<float *>(&v2[524304ull]);
            float * v46;
            v46 = reinterpret_cast<float *>(&v2[786448ull]);
            float * v48;
            v48 = reinterpret_cast<float *>(&v2[1048592ull]);
            float * v50;
            v50 = reinterpret_cast<float *>(&v2[1310736ull]);
            float * v52;
            v52 = reinterpret_cast<float *>(&v2[1572880ull]);
            float * v54;
            v54 = reinterpret_cast<float *>(&v2[1835024ull]);
            int * v56;
            v56 = reinterpret_cast<int *>(&v0[12779520ull]);
            float * v58;
            v58 = reinterpret_cast<float *>(&v0[15925248ull]);
            int * v60;
            v60 = reinterpret_cast<int *>(&v0[19070976ull]);
            int * v62;
            v62 = reinterpret_cast<int *>(&v0[22216704ull]);
            double * v64;
            v64 = reinterpret_cast<double *>(&v0[25362432ull]);
            double * v66;
            v66 = reinterpret_cast<double *>(&v0[37945344ull]);
            double * v68;
            v68 = reinterpret_cast<double *>(&v2[2097168ull]);
            double * v70;
            v70 = reinterpret_cast<double *>(&v2[2883600ull]);
            int * v72;
            v72 = reinterpret_cast<int *>(&v2[3670032ull]);
            int v74;
            v74 = 0l;
            while (while_method_4(v74)){
                int v76;
                v76 = threadIdx.x;
                int v77;
                v77 = blockIdx.x;
                int v78;
                v78 = v77 * 512l;
                int v79;
                v79 = v76 + v78;
                float v80[2l];
                int v81;
                v81 = 0l;
                while (while_method_0(v81)){
                    float v83;
                    v83 = v37[v81];
                    v80[v81] = v83;
                    v81 += 1l ;
                }
                assert("Tensor range check" && 0 <= v74 && v74 < 4l);
                assert("Tensor range check" && 0 <= v79 && v79 < 12288l);
                int v85;
                v85 = 12288l * v74;
                int v86;
                v86 = v85 + v79;
                int v87;
                v87 = v72[v86];
                int v88;
                v88 = v87;
                while (while_method_12(v88)){
                    v88 -= 1l ;
                    assert("Tensor range check" && 0 <= v74 && v74 < 4l);
                    assert("Tensor range check" && 0 <= v88 && v88 < 16l);
                    assert("Tensor range check" && 0 <= v79 && v79 < 12288l);
                    int v90;
                    v90 = 12288l * v88;
                    int v91;
                    v91 = v90 + v79;
                    int v92;
                    v92 = 196608l * v74;
                    int v93;
                    v93 = v92 + v91;
                    int v94;
                    v94 = v56[v93];
                    float v95;
                    v95 = v58[v93];
                    int v96;
                    v96 = v60[v93];
                    int v97;
                    v97 = v62[v93];
                    assert("Tensor range check" && 0 <= v96 && v96 < 2l);
                    float v98;
                    v98 = v80[v96];
                    assert("Tensor range check" && 0 <= v74 && v74 < 4l);
                    int v99;
                    v99 = 16384l * v74;
                    assert("Tensor range check" && 0 <= v97 && v97 < 4096l);
                    int v100;
                    v100 = 4l * v97;
                    int v101;
                    v101 = v100 + v99;
                    float * v102;
                    v102 = v42+v101;
                    float * v104;
                    v104 = v44+v101;
                    float * v106;
                    v106 = v46+v101;
                    float * v108;
                    v108 = v48+v101;
                    float * v110;
                    v110 = v50+v101;
                    float * v112;
                    v112 = v52+v101;
                    float * v114;
                    v114 = v54+v101;
                    assert("Tensor range check" && 0 <= v74 && v74 < 4l);
                    int v116;
                    v116 = 393216l * v74;
                    assert("Tensor range check" && 0 <= v88 && v88 < 16l);
                    int v117;
                    v117 = 24576l * v88;
                    int v118;
                    v118 = v117 + v116;
                    assert("Tensor range check" && 0 <= v79 && v79 < 12288l);
                    int v119;
                    v119 = 2l * v79;
                    int v120;
                    v120 = v119 + v118;
                    double v121[2l];
                    int v122;
                    v122 = 0l;
                    while (while_method_0(v122)){
                        assert("Tensor range check" && 0 <= v122 && v122 < 2l);
                        int v124;
                        v124 = v122 + v120;
                        double v125;
                        v125 = v64[v124];
                        bool v126;
                        v126 = v96 == v122;
                        double v127;
                        if (v126){
                            v127 = 0.0;
                        } else {
                            v127 = v125;
                        }
                        assert("Tensor range check" && 0 <= v122 && v122 < 2l);
                        v121[v122] = v127;
                        v122 += 1l ;
                    }
                    double v128;
                    v128 = 0.0;
                    int v129;
                    v129 = 0l;
                    while (while_method_0(v129)){
                        assert("Tensor range check" && 0 <= v129 && v129 < 2l);
                        double v131;
                        v131 = v121[v129];
                        double v132;
                        v132 = v128 + v131;
                        v128 = v132;
                        v129 += 1l ;
                    }
                    double v133;
                    v133 = 0.0;
                    int v134;
                    v134 = 0l;
                    while (while_method_0(v134)){
                        assert("Tensor range check" && 0 <= v134 && v134 < 2l);
                        int v136;
                        v136 = v134 + v120;
                        double v137;
                        v137 = v66[v136];
                        double v138;
                        v138 = v133 + v137;
                        v133 = v138;
                        v134 += 1l ;
                    }
                    double v139;
                    v139 = v128 - v133;
                    double v140;
                    v140 = exp(v139);
                    float v141;
                    v141 = (float)v140;
                    float v142;
                    v142 = v98 * v141;
                    assert("Tensor range check" && 0 <= v94 && v94 < 4l);
                    float * v143;
                    v143 = v112+v94;
                    float * v145;
                    v145 = v114+v94;
                    float v147;
                    v147 = atomicAdd(v143,v142);
                    float v148;
                    v148 = atomicAdd(v145,v141);
                    float * v149;
                    v149 = v104+0l;
                    float * v151;
                    v151 = v108+0l;
                    float * v153;
                    v153 = v110+0l;
                    int v155;
                    v155 = sizeof(float *);
                    unsigned long long v156;
                    v156 = (unsigned long long)v155;
                    unsigned long long v157;
                    v157 = 512ull * v156;
                    unsigned long long v158;
                    v158 = 8192ull + v157;
                    unsigned long long v159;
                    v159 = v158 + 16ull;
                    unsigned long long v160;
                    v160 = v159 - 1ull;
                    unsigned long long v161;
                    v161 = v160 % 16ull;
                    unsigned long long v162;
                    v162 = v160 - v161;
                    unsigned long long v163;
                    v163 = v162 + v157;
                    unsigned long long v164;
                    v164 = v163 + 16ull;
                    unsigned long long v165;
                    v165 = v164 - 1ull;
                    unsigned long long v166;
                    v166 = v165 % 16ull;
                    unsigned long long v167;
                    v167 = v165 - v166;
                    unsigned long long v168;
                    v168 = v167 + v157;
                    unsigned long long v169;
                    v169 = v168 + 16ull;
                    unsigned long long v170;
                    v170 = v169 - 1ull;
                    unsigned long long v171;
                    v171 = v170 % 16ull;
                    unsigned long long v172;
                    v172 = v170 - v171;
                    unsigned long long v173;
                    v173 = v172 + v157;
                    unsigned long long v174;
                    v174 = v173 + 16ull;
                    unsigned long long v175;
                    v175 = v174 - 1ull;
                    unsigned long long v176;
                    v176 = v175 % 16ull;
                    unsigned long long v177;
                    v177 = v175 - v176;
                    unsigned long long v178;
                    v178 = v177 + 2048ull;
                    bool v179;
                    v179 = v178 <= 81920ull;
                    bool v180;
                    v180 = v179 == false;
                    if (v180){
                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v179);
                    } else {
                    }
                    extern __shared__ unsigned char v182[];
                    bool v183;
                    v183 = v178 <= v178;
                    bool v184;
                    v184 = v183 == false;
                    if (v184){
                        assert("The length of the partition has to be less than or equal to the length of the base array." && v183);
                    } else {
                    }
                    float * v186;
                    v186 = reinterpret_cast<float *>(&v182[0ull]);
                    int * v188;
                    v188 = reinterpret_cast<int *>(&v182[2048ull]);
                    float * v190;
                    v190 = reinterpret_cast<float *>(&v182[4096ull]);
                    float * v192;
                    v192 = reinterpret_cast<float *>(&v182[6144ull]);
                    float * * v194;
                    v194 = reinterpret_cast<float * *>(&v182[8192ull]);
                    float * * v196;
                    v196 = reinterpret_cast<float * *>(&v182[v162]);
                    float * * v198;
                    v198 = reinterpret_cast<float * *>(&v182[v167]);
                    float * * v200;
                    v200 = reinterpret_cast<float * *>(&v182[v172]);
                    float * v202;
                    v202 = reinterpret_cast<float *>(&v182[v177]);
                    int v204;
                    v204 = threadIdx.x;
                    assert("Tensor range check" && 0 <= v204 && v204 < 512l);
                    v186[v204] = v95;
                    v188[v204] = v94;
                    v190[v204] = v98;
                    v192[v204] = v141;
                    v194[v204] = v106;
                    v196[v204] = v149;
                    v198[v204] = v151;
                    v200[v204] = v153;
                    asm("barrier.cta.sync %0;" :: "r"(0l));
                    bool v205;
                    v205 = 0l <= v204;
                    bool v206;
                    v206 = v205 == false;
                    if (v206){
                        assert("The index needs to be zero or positive." && v205);
                    } else {
                    }
                    int v208;
                    v208 = v204 % 1l;
                    bool v209;
                    v209 = v204 < 512l;
                    bool v210;
                    v210 = v209 == false;
                    if (v210){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v209);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v204 && v204 < 512l);
                    int v212;
                    v212 = 0l;
                    while (while_method_5(v212)){
                        bool v214;
                        v214 = v205 && v209;
                        bool v215;
                        v215 = v214 == false;
                        if (v215){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v214);
                        } else {
                        }
                        bool v217;
                        v217 = 0l <= v212;
                        bool v219;
                        if (v217){
                            bool v218;
                            v218 = v212 < 1l;
                            v219 = v218;
                        } else {
                            v219 = false;
                        }
                        bool v220;
                        v220 = v219 == false;
                        if (v220){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v219);
                        } else {
                        }
                        int v222;
                        v222 = v212 * 512l;
                        int v223;
                        v223 = v222 + v204;
                        assert("Tensor range check" && 0 <= v212 && v212 < 1l);
                        int v224;
                        v224 = 512l * v212;
                        int v225;
                        v225 = v224 + v204;
                        float v226;
                        v226 = v186[v225];
                        int v227;
                        v227 = v188[v225];
                        float v228;
                        v228 = v190[v225];
                        float v229;
                        v229 = v192[v225];
                        float * v230;
                        v230 = v194[v225];
                        float * v231;
                        v231 = v196[v225];
                        float * v232;
                        v232 = v198[v225];
                        float * v233;
                        v233 = v200[v225];
                        int v234;
                        v234 = blockIdx.x;
                        int v235;
                        v235 = v234 * 512l;
                        int v236;
                        v236 = v235 + v223;
                        assert("Tensor range check" && 0 <= v208 && v208 < 1l);
                        int v237;
                        v237 = 4l * v208;
                        float v238[4l];
                        float v239[4l];
                        float v240[4l];
                        int v241[4l];
                        int v242;
                        v242 = 0l;
                        while (while_method_5(v242)){
                            assert("Tensor range check" && 0 <= v242 && v242 < 1l);
                            int v244;
                            v244 = 4l * v242;
                            assert("Tensor range check" && 0 <= v242 && v242 < 1l);
                            int v245;
                            v245 = v244 + v237;
                            int4* v246;
                            v246 = reinterpret_cast<int4*>(v231 + v245);
                            int4* v247;
                            v247 = reinterpret_cast<int4*>(v238 + v244);
                            assert("Pointer alignment check" && (unsigned long long)(v246) % 4l == 0 && (unsigned long long)(v247) % 4l == 0);
                            *v247 = *v246;
                            int4* v248;
                            v248 = reinterpret_cast<int4*>(v232 + v245);
                            int4* v249;
                            v249 = reinterpret_cast<int4*>(v239 + v244);
                            assert("Pointer alignment check" && (unsigned long long)(v248) % 4l == 0 && (unsigned long long)(v249) % 4l == 0);
                            *v249 = *v248;
                            int4* v250;
                            v250 = reinterpret_cast<int4*>(v233 + v245);
                            int4* v251;
                            v251 = reinterpret_cast<int4*>(v240 + v244);
                            assert("Pointer alignment check" && (unsigned long long)(v250) % 4l == 0 && (unsigned long long)(v251) % 4l == 0);
                            *v251 = *v250;
                            v242 += 1l ;
                        }
                        int v252;
                        v252 = 0l;
                        while (while_method_5(v252)){
                            int v254;
                            v254 = 0l;
                            while (while_method_4(v254)){
                                bool v256;
                                v256 = 0l <= v254;
                                bool v258;
                                if (v256){
                                    bool v257;
                                    v257 = v254 < 4l;
                                    v258 = v257;
                                } else {
                                    v258 = false;
                                }
                                bool v259;
                                v259 = v258 == false;
                                if (v259){
                                    assert("The indices should be inside the range of the dimension." && v258);
                                } else {
                                }
                                bool v261;
                                v261 = 0l <= v208;
                                bool v263;
                                if (v261){
                                    bool v262;
                                    v262 = v208 < 1l;
                                    v263 = v262;
                                } else {
                                    v263 = false;
                                }
                                bool v264;
                                v264 = v263 == false;
                                if (v264){
                                    assert("The indices should be inside the range of the dimension." && v263);
                                } else {
                                }
                                int v266;
                                v266 = v208 * 4l;
                                int v267;
                                v267 = v254 + v266;
                                bool v268;
                                v268 = 0l <= v252;
                                bool v270;
                                if (v268){
                                    bool v269;
                                    v269 = v252 < 1l;
                                    v270 = v269;
                                } else {
                                    v270 = false;
                                }
                                bool v271;
                                v271 = v270 == false;
                                if (v271){
                                    assert("The indices should be inside the range of the dimension." && v270);
                                } else {
                                }
                                int v273;
                                v273 = v252 * 4l;
                                int v274;
                                v274 = v267 + v273;
                                assert("Tensor range check" && 0 <= v252 && v252 < 1l);
                                assert("Tensor range check" && 0 <= v254 && v254 < 4l);
                                int v275;
                                v275 = 4l * v252;
                                int v276;
                                v276 = v275 + v254;
                                v241[v276] = v274;
                                v254 += 1l ;
                            }
                            v252 += 1l ;
                        }
                        float v277[4l];
                        int v278;
                        v278 = 0l;
                        while (while_method_5(v278)){
                            int v280;
                            v280 = 0l;
                            while (while_method_4(v280)){
                                assert("Tensor range check" && 0 <= v278 && v278 < 1l);
                                assert("Tensor range check" && 0 <= v280 && v280 < 4l);
                                int v282;
                                v282 = 4l * v278;
                                int v283;
                                v283 = v282 + v280;
                                float v284;
                                v284 = v239[v283];
                                float v285;
                                v285 = v240[v283];
                                bool v286;
                                v286 = v285 == 0.0f;
                                bool v287;
                                v287 = v286 != true;
                                float v289;
                                if (v287){
                                    float v288;
                                    v288 = v284 / v285;
                                    v289 = v288;
                                } else {
                                    v289 = 0.0f;
                                }
                                assert("Tensor range check" && 0 <= v278 && v278 < 1l);
                                assert("Tensor range check" && 0 <= v280 && v280 < 4l);
                                v277[v283] = v289;
                                v280 += 1l ;
                            }
                            v278 += 1l ;
                        }
                        bool v290[4l];
                        int v291;
                        v291 = 0l;
                        while (while_method_5(v291)){
                            int v293;
                            v293 = 0l;
                            while (while_method_4(v293)){
                                assert("Tensor range check" && 0 <= v291 && v291 < 1l);
                                assert("Tensor range check" && 0 <= v293 && v293 < 4l);
                                int v295;
                                v295 = 4l * v291;
                                int v296;
                                v296 = v295 + v293;
                                float v297;
                                v297 = v238[v296];
                                int v298;
                                v298 = v241[v296];
                                bool v299;
                                v299 = v298 < 3l;
                                assert("Tensor range check" && 0 <= v291 && v291 < 1l);
                                assert("Tensor range check" && 0 <= v293 && v293 < 4l);
                                v290[v296] = v299;
                                v293 += 1l ;
                            }
                            v291 += 1l ;
                        }
                        float v300[4l];
                        int v301;
                        v301 = 0l;
                        while (while_method_5(v301)){
                            int v303;
                            v303 = 0l;
                            while (while_method_4(v303)){
                                assert("Tensor range check" && 0 <= v301 && v301 < 1l);
                                assert("Tensor range check" && 0 <= v303 && v303 < 4l);
                                int v305;
                                v305 = 4l * v301;
                                int v306;
                                v306 = v305 + v303;
                                float v307;
                                v307 = v238[v306];
                                bool v308;
                                v308 = v290[v306];
                                float v311;
                                if (v308){
                                    bool v309;
                                    v309 = 0.0f >= v307;
                                    if (v309){
                                        v311 = 0.0f;
                                    } else {
                                        v311 = v307;
                                    }
                                } else {
                                    v311 = 0.0f;
                                }
                                assert("Tensor range check" && 0 <= v301 && v301 < 1l);
                                assert("Tensor range check" && 0 <= v303 && v303 < 4l);
                                v300[v306] = v311;
                                v303 += 1l ;
                            }
                            v301 += 1l ;
                        }
                        float v312;
                        v312 = 0.0f;
                        int v313;
                        v313 = 0l;
                        while (while_method_5(v313)){
                            int v315;
                            v315 = 0l;
                            while (while_method_4(v315)){
                                assert("Tensor range check" && 0 <= v313 && v313 < 1l);
                                assert("Tensor range check" && 0 <= v315 && v315 < 4l);
                                int v317;
                                v317 = 4l * v313;
                                int v318;
                                v318 = v317 + v315;
                                float v319;
                                v319 = v300[v318];
                                float v320;
                                v320 = v312 + v319;
                                v312 = v320;
                                v315 += 1l ;
                            }
                            v313 += 1l ;
                        }
                        auto v321 = cooperative_groups::coalesced_threads();
                        int v322;
                        v322 = threadIdx.x;
                        auto v323 = cooperative_groups::labeled_partition(v321,v322);
                        Closure1 v324{};
                        float v325;
                        v325 = cooperative_groups::reduce(v323, v312, v324);
                        int v326[4l];
                        int v327;
                        v327 = 0l;
                        while (while_method_5(v327)){
                            int v329;
                            v329 = 0l;
                            while (while_method_4(v329)){
                                assert("Tensor range check" && 0 <= v327 && v327 < 1l);
                                assert("Tensor range check" && 0 <= v329 && v329 < 4l);
                                int v331;
                                v331 = 4l * v327;
                                int v332;
                                v332 = v331 + v329;
                                bool v333;
                                v333 = v290[v332];
                                int v334;
                                if (v333){
                                    v334 = 1l;
                                } else {
                                    v334 = 0l;
                                }
                                assert("Tensor range check" && 0 <= v327 && v327 < 1l);
                                assert("Tensor range check" && 0 <= v329 && v329 < 4l);
                                v326[v332] = v334;
                                v329 += 1l ;
                            }
                            v327 += 1l ;
                        }
                        int v335;
                        v335 = 0l;
                        int v336;
                        v336 = 0l;
                        while (while_method_5(v336)){
                            int v338;
                            v338 = 0l;
                            while (while_method_4(v338)){
                                assert("Tensor range check" && 0 <= v336 && v336 < 1l);
                                assert("Tensor range check" && 0 <= v338 && v338 < 4l);
                                int v340;
                                v340 = 4l * v336;
                                int v341;
                                v341 = v340 + v338;
                                int v342;
                                v342 = v326[v341];
                                int v343;
                                v343 = v335 + v342;
                                v335 = v343;
                                v338 += 1l ;
                            }
                            v336 += 1l ;
                        }
                        auto v344 = cooperative_groups::coalesced_threads();
                        int v345;
                        v345 = threadIdx.x;
                        auto v346 = cooperative_groups::labeled_partition(v344,v345);
                        Closure2 v347{};
                        int v348;
                        v348 = cooperative_groups::reduce(v346, v335, v347);
                        float v349;
                        v349 = (float)v348;
                        float v350;
                        v350 = 1.0f / v349;
                        float v351[4l];
                        int v352;
                        v352 = 0l;
                        while (while_method_5(v352)){
                            int v354;
                            v354 = 0l;
                            while (while_method_4(v354)){
                                assert("Tensor range check" && 0 <= v352 && v352 < 1l);
                                assert("Tensor range check" && 0 <= v354 && v354 < 4l);
                                int v356;
                                v356 = 4l * v352;
                                int v357;
                                v357 = v356 + v354;
                                float v358;
                                v358 = v300[v357];
                                bool v359;
                                v359 = v290[v357];
                                bool v360;
                                v360 = v359 == false;
                                float v365;
                                if (v360){
                                    v365 = 0.0f;
                                } else {
                                    bool v361;
                                    v361 = v325 == 0.0f;
                                    bool v362;
                                    v362 = v361 != true;
                                    if (v362){
                                        float v363;
                                        v363 = v358 / v325;
                                        v365 = v363;
                                    } else {
                                        v365 = v350;
                                    }
                                }
                                assert("Tensor range check" && 0 <= v352 && v352 < 1l);
                                assert("Tensor range check" && 0 <= v354 && v354 < 4l);
                                v351[v357] = v365;
                                v354 += 1l ;
                            }
                            v352 += 1l ;
                        }
                        float v366[4l];
                        int v367;
                        v367 = 0l;
                        while (while_method_5(v367)){
                            int v369;
                            v369 = 0l;
                            while (while_method_4(v369)){
                                assert("Tensor range check" && 0 <= v367 && v367 < 1l);
                                assert("Tensor range check" && 0 <= v369 && v369 < 4l);
                                int v371;
                                v371 = 4l * v367;
                                int v372;
                                v372 = v371 + v369;
                                float v373;
                                v373 = v277[v372];
                                int v374;
                                v374 = v241[v372];
                                bool v375;
                                v375 = v227 == v374;
                                float v378;
                                if (v375){
                                    float v376;
                                    v376 = v228 - v373;
                                    float v377;
                                    v377 = v376 / v226;
                                    v378 = v377;
                                } else {
                                    v378 = 0.0f;
                                }
                                float v379;
                                v379 = v378 + v373;
                                assert("Tensor range check" && 0 <= v367 && v367 < 1l);
                                assert("Tensor range check" && 0 <= v369 && v369 < 4l);
                                v366[v372] = v379;
                                v369 += 1l ;
                            }
                            v367 += 1l ;
                        }
                        float v380[4l];
                        int v381;
                        v381 = 0l;
                        while (while_method_5(v381)){
                            int v383;
                            v383 = 0l;
                            while (while_method_4(v383)){
                                assert("Tensor range check" && 0 <= v381 && v381 < 1l);
                                assert("Tensor range check" && 0 <= v383 && v383 < 4l);
                                int v385;
                                v385 = 4l * v381;
                                int v386;
                                v386 = v385 + v383;
                                float v387;
                                v387 = v351[v386];
                                float v388;
                                v388 = v366[v386];
                                float v389;
                                v389 = v387 * v388;
                                assert("Tensor range check" && 0 <= v381 && v381 < 1l);
                                assert("Tensor range check" && 0 <= v383 && v383 < 4l);
                                v380[v386] = v389;
                                v383 += 1l ;
                            }
                            v381 += 1l ;
                        }
                        float v390;
                        v390 = 0.0f;
                        int v391;
                        v391 = 0l;
                        while (while_method_5(v391)){
                            int v393;
                            v393 = 0l;
                            while (while_method_4(v393)){
                                assert("Tensor range check" && 0 <= v391 && v391 < 1l);
                                assert("Tensor range check" && 0 <= v393 && v393 < 4l);
                                int v395;
                                v395 = 4l * v391;
                                int v396;
                                v396 = v395 + v393;
                                float v397;
                                v397 = v380[v396];
                                float v398;
                                v398 = v390 + v397;
                                v390 = v398;
                                v393 += 1l ;
                            }
                            v391 += 1l ;
                        }
                        auto v399 = cooperative_groups::coalesced_threads();
                        int v400;
                        v400 = threadIdx.x;
                        auto v401 = cooperative_groups::labeled_partition(v399,v400);
                        float v402;
                        v402 = cooperative_groups::reduce(v401, v390, v324);
                        int v403;
                        v403 = 0l;
                        while (while_method_5(v403)){
                            int v405;
                            v405 = 0l;
                            while (while_method_4(v405)){
                                assert("Tensor range check" && 0 <= v403 && v403 < 1l);
                                assert("Tensor range check" && 0 <= v405 && v405 < 4l);
                                int v407;
                                v407 = 4l * v403;
                                int v408;
                                v408 = v407 + v405;
                                float v409;
                                v409 = v366[v408];
                                int v410;
                                v410 = v241[v408];
                                float v411;
                                v411 = v409 - v402;
                                float v412;
                                v412 = v229 * v411;
                                assert("Tensor range check" && 0 <= v410 && v410 < 4l);
                                float * v413;
                                v413 = v230+v410;
                                float v415;
                                v415 = atomicAdd(v413,v412);
                                v405 += 1l ;
                            }
                            v403 += 1l ;
                        }
                        int v416;
                        v416 = 0l;
                        while (while_method_5(v416)){
                            assert("Tensor range check" && 0 <= v416 && v416 < 1l);
                            assert("Tensor range check" && 0 <= v416 && v416 < 1l);
                            v416 += 1l ;
                        }
                        assert("Tensor range check" && 0 <= v223 && v223 < 512l);
                        v202[v223] = v402;
                        v212 += 1l ;
                    }
                    asm("barrier.cta.sync %0;" :: "r"(0l));
                    assert("Tensor range check" && 0 <= v204 && v204 < 512l);
                    float v418;
                    v418 = v202[v204];
                    asm("barrier.cta.sync %0;" :: "r"(0l));
                    assert("Tensor range check" && 0 <= v96 && v96 < 2l);
                    v80[v96] = v418;
                }
                int v419;
                v419 = threadIdx.x;
                int v420;
                v420 = blockIdx.x;
                int v421;
                v421 = v420 * 512l;
                int v422;
                v422 = v419 + v421;
                assert("Tensor range check" && 0 <= v74 && v74 < 4l);
                int v423;
                v423 = 24576l * v74;
                assert("Tensor range check" && 0 <= v422 && v422 < 12288l);
                int v424;
                v424 = 2l * v422;
                int v425;
                v425 = v424 + v423;
                double * v426;
                v426 = v68+v425;
                double * v428;
                v428 = v70+v425;
                double * v430;
                v430 = v426+0l;
                double * v432;
                v432 = v428+0l;
                double * v434;
                v434 = v426+0l;
                double * v436;
                v436 = v428+0l;
                int v438;
                v438 = sizeof(double *);
                unsigned long long v439;
                v439 = (unsigned long long)v438;
                unsigned long long v440;
                v440 = 512ull * v439;
                unsigned long long v441;
                v441 = v440 + 16ull;
                unsigned long long v442;
                v442 = v441 - 1ull;
                unsigned long long v443;
                v443 = v442 % 16ull;
                unsigned long long v444;
                v444 = v442 - v443;
                unsigned long long v445;
                v445 = v444 + v440;
                unsigned long long v446;
                v446 = v445 + 16ull;
                unsigned long long v447;
                v447 = v446 - 1ull;
                unsigned long long v448;
                v448 = v447 % 16ull;
                unsigned long long v449;
                v449 = v447 - v448;
                unsigned long long v450;
                v450 = v449 + v440;
                unsigned long long v451;
                v451 = v450 + 16ull;
                unsigned long long v452;
                v452 = v451 - 1ull;
                unsigned long long v453;
                v453 = v452 % 16ull;
                unsigned long long v454;
                v454 = v452 - v453;
                unsigned long long v455;
                v455 = v454 + v440;
                bool v456;
                v456 = v455 <= 81920ull;
                bool v457;
                v457 = v456 == false;
                if (v457){
                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v456);
                } else {
                }
                extern __shared__ unsigned char v459[];
                bool v460;
                v460 = v455 <= v455;
                bool v461;
                v461 = v460 == false;
                if (v461){
                    assert("The length of the partition has to be less than or equal to the length of the base array." && v460);
                } else {
                }
                double * * v463;
                v463 = reinterpret_cast<double * *>(&v459[0ull]);
                double * * v465;
                v465 = reinterpret_cast<double * *>(&v459[v444]);
                double * * v467;
                v467 = reinterpret_cast<double * *>(&v459[v449]);
                double * * v469;
                v469 = reinterpret_cast<double * *>(&v459[v454]);
                int v471;
                v471 = threadIdx.x;
                assert("Tensor range check" && 0 <= v471 && v471 < 512l);
                v463[v471] = v430;
                v465[v471] = v432;
                v467[v471] = v434;
                v469[v471] = v436;
                asm("barrier.cta.sync %0;" :: "r"(0l));
                bool v472;
                v472 = 0l <= v471;
                bool v473;
                v473 = v472 == false;
                if (v473){
                    assert("The index needs to be zero or positive." && v472);
                } else {
                }
                int v475;
                v475 = v471 % 1l;
                bool v476;
                v476 = v471 < 512l;
                bool v477;
                v477 = v476 == false;
                if (v477){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v476);
                } else {
                }
                assert("Tensor range check" && 0 <= v471 && v471 < 512l);
                int v479;
                v479 = 0l;
                while (while_method_5(v479)){
                    bool v481;
                    v481 = v472 && v476;
                    bool v482;
                    v482 = v481 == false;
                    if (v482){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v481);
                    } else {
                    }
                    bool v484;
                    v484 = 0l <= v479;
                    bool v486;
                    if (v484){
                        bool v485;
                        v485 = v479 < 1l;
                        v486 = v485;
                    } else {
                        v486 = false;
                    }
                    bool v487;
                    v487 = v486 == false;
                    if (v487){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v486);
                    } else {
                    }
                    int v489;
                    v489 = v479 * 512l;
                    int v490;
                    v490 = v489 + v471;
                    assert("Tensor range check" && 0 <= v479 && v479 < 1l);
                    int v491;
                    v491 = 512l * v479;
                    int v492;
                    v492 = v491 + v471;
                    double * v493;
                    v493 = v463[v492];
                    double * v494;
                    v494 = v465[v492];
                    double * v495;
                    v495 = v467[v492];
                    double * v496;
                    v496 = v469[v492];
                    int v497;
                    v497 = blockIdx.x;
                    int v498;
                    v498 = v497 * 512l;
                    int v499;
                    v499 = v498 + v490;
                    assert("Tensor range check" && 0 <= v475 && v475 < 1l);
                    int v500;
                    v500 = 2l * v475;
                    double v501[2l];
                    double v502[2l];
                    int v503[2l];
                    int v504;
                    v504 = 0l;
                    while (while_method_5(v504)){
                        assert("Tensor range check" && 0 <= v504 && v504 < 1l);
                        int v506;
                        v506 = 2l * v504;
                        assert("Tensor range check" && 0 <= v504 && v504 < 1l);
                        int v507;
                        v507 = v506 + v500;
                        int4* v508;
                        v508 = reinterpret_cast<int4*>(v493 + v507);
                        int4* v509;
                        v509 = reinterpret_cast<int4*>(v501 + v506);
                        assert("Pointer alignment check" && (unsigned long long)(v508) % 2l == 0 && (unsigned long long)(v509) % 2l == 0);
                        *v509 = *v508;
                        int4* v510;
                        v510 = reinterpret_cast<int4*>(v494 + v507);
                        int4* v511;
                        v511 = reinterpret_cast<int4*>(v502 + v506);
                        assert("Pointer alignment check" && (unsigned long long)(v510) % 2l == 0 && (unsigned long long)(v511) % 2l == 0);
                        *v511 = *v510;
                        v504 += 1l ;
                    }
                    int v512;
                    v512 = 0l;
                    while (while_method_5(v512)){
                        int v514;
                        v514 = 0l;
                        while (while_method_0(v514)){
                            bool v516;
                            v516 = 0l <= v514;
                            bool v518;
                            if (v516){
                                bool v517;
                                v517 = v514 < 2l;
                                v518 = v517;
                            } else {
                                v518 = false;
                            }
                            bool v519;
                            v519 = v518 == false;
                            if (v519){
                                assert("The indices should be inside the range of the dimension." && v518);
                            } else {
                            }
                            bool v521;
                            v521 = 0l <= v475;
                            bool v523;
                            if (v521){
                                bool v522;
                                v522 = v475 < 1l;
                                v523 = v522;
                            } else {
                                v523 = false;
                            }
                            bool v524;
                            v524 = v523 == false;
                            if (v524){
                                assert("The indices should be inside the range of the dimension." && v523);
                            } else {
                            }
                            int v526;
                            v526 = v475 * 2l;
                            int v527;
                            v527 = v514 + v526;
                            bool v528;
                            v528 = 0l <= v512;
                            bool v530;
                            if (v528){
                                bool v529;
                                v529 = v512 < 1l;
                                v530 = v529;
                            } else {
                                v530 = false;
                            }
                            bool v531;
                            v531 = v530 == false;
                            if (v531){
                                assert("The indices should be inside the range of the dimension." && v530);
                            } else {
                            }
                            int v533;
                            v533 = v512 * 2l;
                            int v534;
                            v534 = v527 + v533;
                            assert("Tensor range check" && 0 <= v512 && v512 < 1l);
                            assert("Tensor range check" && 0 <= v514 && v514 < 2l);
                            int v535;
                            v535 = 2l * v512;
                            int v536;
                            v536 = v535 + v514;
                            v503[v536] = v534;
                            v514 += 1l ;
                        }
                        v512 += 1l ;
                    }
                    double v537[2l];
                    double v538[2l];
                    int v539;
                    v539 = 0l;
                    while (while_method_5(v539)){
                        int v541;
                        v541 = 0l;
                        while (while_method_0(v541)){
                            assert("Tensor range check" && 0 <= v539 && v539 < 1l);
                            assert("Tensor range check" && 0 <= v541 && v541 < 2l);
                            int v543;
                            v543 = 2l * v539;
                            int v544;
                            v544 = v543 + v541;
                            double v545;
                            v545 = v501[v544];
                            double v546;
                            v546 = v502[v544];
                            assert("Tensor range check" && 0 <= v539 && v539 < 1l);
                            assert("Tensor range check" && 0 <= v541 && v541 < 2l);
                            v537[v544] = 0.0;
                            v538[v544] = 0.0;
                            v541 += 1l ;
                        }
                        v539 += 1l ;
                    }
                    int v547;
                    v547 = 0l;
                    while (while_method_5(v547)){
                        assert("Tensor range check" && 0 <= v547 && v547 < 1l);
                        int v549;
                        v549 = 2l * v547;
                        int v550;
                        v550 = v549 + v500;
                        assert("Tensor range check" && 0 <= v547 && v547 < 1l);
                        int4* v551;
                        v551 = reinterpret_cast<int4*>(v537 + v549);
                        int4* v552;
                        v552 = reinterpret_cast<int4*>(v495 + v550);
                        assert("Pointer alignment check" && (unsigned long long)(v551) % 2l == 0 && (unsigned long long)(v552) % 2l == 0);
                        *v552 = *v551;
                        int4* v553;
                        v553 = reinterpret_cast<int4*>(v538 + v549);
                        int4* v554;
                        v554 = reinterpret_cast<int4*>(v496 + v550);
                        assert("Pointer alignment check" && (unsigned long long)(v553) % 2l == 0 && (unsigned long long)(v554) % 2l == 0);
                        *v554 = *v553;
                        v547 += 1l ;
                    }
                    assert("Tensor range check" && 0 <= v490 && v490 < 512l);
                    v479 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                assert("Tensor range check" && 0 <= v471 && v471 < 512l);
                asm("barrier.cta.sync %0;" :: "r"(0l));
                assert("Tensor range check" && 0 <= v74 && v74 < 4l);
                assert("Tensor range check" && 0 <= v422 && v422 < 12288l);
                int v555;
                v555 = v85 + v422;
                v72[v555] = 0l;
                v74 += 1l ;
            }
            static_array<Union14,2l> & v556 = v25;
            unsigned int v557 = 63ul;
            static_array_list<Union7,32l> v558;
            v558 = static_array_list<Union7,32l>{};
            static_array_list<Union7,32l> & v560 = v558;
            Union4 v561;
            v561 = Union4{Union4_1{}};
            static_array<float,2l> v562;
            v562 = method_53(v0, v2, v557, v560, v556, v20, v561);
            double * v563;
            v563 = reinterpret_cast<double *>(&v2[2097168ull]);
            double * v565;
            v565 = reinterpret_cast<double *>(&v2[2883600ull]);
            int * v567;
            v567 = reinterpret_cast<int *>(&v2[3670032ull]);
            int v569;
            v569 = threadIdx.x;
            int v570;
            v570 = blockIdx.x;
            int v571;
            v571 = v570 * 512l;
            int v572;
            v572 = v569 + v571;
            assert("Tensor range check" && 0 <= v572 && v572 < 12288l);
            int v573;
            v573 = 2l * v572;
            int v574; double v575;
            Tuple12 tmp66 = Tuple12{0l, 1.0};
            v574 = tmp66.v0; v575 = tmp66.v1;
            while (while_method_0(v574)){
                assert("Tensor range check" && 0 <= v574 && v574 < 2l);
                int v577;
                v577 = v574 + v573;
                int v578; double v579;
                Tuple12 tmp67 = Tuple12{0l, 0.0};
                v578 = tmp67.v0; v579 = tmp67.v1;
                while (while_method_4(v578)){
                    assert("Tensor range check" && 0 <= v578 && v578 < 4l);
                    int v581;
                    v581 = 24576l * v578;
                    int v582;
                    v582 = v581 + v577;
                    double v583;
                    v583 = v563[v582];
                    double v584;
                    v584 = v565[v582];
                    double v585;
                    v585 = v583 - v584;
                    double v586;
                    v586 = exp(v585);
                    double v587;
                    v587 = v579 + v586;
                    v579 = v587;
                    v578 += 1l ;
                }
                double v588;
                v588 = v575 * v579;
                v575 = v588;
                v574 += 1l ;
            }
            float v589;
            v589 = (float)v575;
            int v590;
            v590 = 0l;
            while (while_method_0(v590)){
                float v592;
                v592 = v562[v590];
                int v594;
                v594 = v21 / 4l;
                float v595;
                v595 = v592 * v589;
                assert("Tensor range check" && 0 <= v590 && v590 < 2l);
                assert("Tensor range check" && 0 <= v594 && v594 < 256l);
                int v596;
                v596 = 256l * v590;
                int v597;
                v597 = v596 + v594;
                float * v598;
                v598 = v4+v597;
                float * v600;
                v600 = v5+v597;
                float v602;
                v602 = atomicAdd(v598,v595);
                float v603;
                v603 = atomicAdd(v600,v589);
                v590 += 1l ;
            }
            double * v604;
            v604 = reinterpret_cast<double *>(&v2[2097168ull]);
            double * v606;
            v606 = reinterpret_cast<double *>(&v2[2883600ull]);
            int * v608;
            v608 = reinterpret_cast<int *>(&v2[3670032ull]);
            int v610;
            v610 = 0l;
            while (while_method_4(v610)){
                int v612;
                v612 = threadIdx.x;
                int v613;
                v613 = blockIdx.x;
                int v614;
                v614 = v613 * 512l;
                int v615;
                v615 = v612 + v614;
                assert("Tensor range check" && 0 <= v610 && v610 < 4l);
                int v616;
                v616 = 24576l * v610;
                assert("Tensor range check" && 0 <= v615 && v615 < 12288l);
                int v617;
                v617 = 2l * v615;
                int v618;
                v618 = v617 + v616;
                double * v619;
                v619 = v604+v618;
                double * v621;
                v621 = v606+v618;
                double * v623;
                v623 = v619+0l;
                double * v625;
                v625 = v621+0l;
                double * v627;
                v627 = v619+0l;
                double * v629;
                v629 = v621+0l;
                int v631;
                v631 = sizeof(double *);
                unsigned long long v632;
                v632 = (unsigned long long)v631;
                unsigned long long v633;
                v633 = 512ull * v632;
                unsigned long long v634;
                v634 = v633 + 16ull;
                unsigned long long v635;
                v635 = v634 - 1ull;
                unsigned long long v636;
                v636 = v635 % 16ull;
                unsigned long long v637;
                v637 = v635 - v636;
                unsigned long long v638;
                v638 = v637 + v633;
                unsigned long long v639;
                v639 = v638 + 16ull;
                unsigned long long v640;
                v640 = v639 - 1ull;
                unsigned long long v641;
                v641 = v640 % 16ull;
                unsigned long long v642;
                v642 = v640 - v641;
                unsigned long long v643;
                v643 = v642 + v633;
                unsigned long long v644;
                v644 = v643 + 16ull;
                unsigned long long v645;
                v645 = v644 - 1ull;
                unsigned long long v646;
                v646 = v645 % 16ull;
                unsigned long long v647;
                v647 = v645 - v646;
                unsigned long long v648;
                v648 = v647 + v633;
                bool v649;
                v649 = v648 <= 81920ull;
                bool v650;
                v650 = v649 == false;
                if (v650){
                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v649);
                } else {
                }
                extern __shared__ unsigned char v652[];
                bool v653;
                v653 = v648 <= v648;
                bool v654;
                v654 = v653 == false;
                if (v654){
                    assert("The length of the partition has to be less than or equal to the length of the base array." && v653);
                } else {
                }
                double * * v656;
                v656 = reinterpret_cast<double * *>(&v652[0ull]);
                double * * v658;
                v658 = reinterpret_cast<double * *>(&v652[v637]);
                double * * v660;
                v660 = reinterpret_cast<double * *>(&v652[v642]);
                double * * v662;
                v662 = reinterpret_cast<double * *>(&v652[v647]);
                int v664;
                v664 = threadIdx.x;
                assert("Tensor range check" && 0 <= v664 && v664 < 512l);
                v656[v664] = v623;
                v658[v664] = v625;
                v660[v664] = v627;
                v662[v664] = v629;
                asm("barrier.cta.sync %0;" :: "r"(0l));
                bool v665;
                v665 = 0l <= v664;
                bool v666;
                v666 = v665 == false;
                if (v666){
                    assert("The index needs to be zero or positive." && v665);
                } else {
                }
                int v668;
                v668 = v664 % 1l;
                bool v669;
                v669 = v664 < 512l;
                bool v670;
                v670 = v669 == false;
                if (v670){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v669);
                } else {
                }
                assert("Tensor range check" && 0 <= v664 && v664 < 512l);
                int v672;
                v672 = 0l;
                while (while_method_5(v672)){
                    bool v674;
                    v674 = v665 && v669;
                    bool v675;
                    v675 = v674 == false;
                    if (v675){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v674);
                    } else {
                    }
                    bool v677;
                    v677 = 0l <= v672;
                    bool v679;
                    if (v677){
                        bool v678;
                        v678 = v672 < 1l;
                        v679 = v678;
                    } else {
                        v679 = false;
                    }
                    bool v680;
                    v680 = v679 == false;
                    if (v680){
                        assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v679);
                    } else {
                    }
                    int v682;
                    v682 = v672 * 512l;
                    int v683;
                    v683 = v682 + v664;
                    assert("Tensor range check" && 0 <= v672 && v672 < 1l);
                    int v684;
                    v684 = 512l * v672;
                    int v685;
                    v685 = v684 + v664;
                    double * v686;
                    v686 = v656[v685];
                    double * v687;
                    v687 = v658[v685];
                    double * v688;
                    v688 = v660[v685];
                    double * v689;
                    v689 = v662[v685];
                    int v690;
                    v690 = blockIdx.x;
                    int v691;
                    v691 = v690 * 512l;
                    int v692;
                    v692 = v691 + v683;
                    assert("Tensor range check" && 0 <= v668 && v668 < 1l);
                    int v693;
                    v693 = 2l * v668;
                    double v694[2l];
                    double v695[2l];
                    int v696[2l];
                    int v697;
                    v697 = 0l;
                    while (while_method_5(v697)){
                        assert("Tensor range check" && 0 <= v697 && v697 < 1l);
                        int v699;
                        v699 = 2l * v697;
                        assert("Tensor range check" && 0 <= v697 && v697 < 1l);
                        int v700;
                        v700 = v699 + v693;
                        int4* v701;
                        v701 = reinterpret_cast<int4*>(v686 + v700);
                        int4* v702;
                        v702 = reinterpret_cast<int4*>(v694 + v699);
                        assert("Pointer alignment check" && (unsigned long long)(v701) % 2l == 0 && (unsigned long long)(v702) % 2l == 0);
                        *v702 = *v701;
                        int4* v703;
                        v703 = reinterpret_cast<int4*>(v687 + v700);
                        int4* v704;
                        v704 = reinterpret_cast<int4*>(v695 + v699);
                        assert("Pointer alignment check" && (unsigned long long)(v703) % 2l == 0 && (unsigned long long)(v704) % 2l == 0);
                        *v704 = *v703;
                        v697 += 1l ;
                    }
                    int v705;
                    v705 = 0l;
                    while (while_method_5(v705)){
                        int v707;
                        v707 = 0l;
                        while (while_method_0(v707)){
                            bool v709;
                            v709 = 0l <= v707;
                            bool v711;
                            if (v709){
                                bool v710;
                                v710 = v707 < 2l;
                                v711 = v710;
                            } else {
                                v711 = false;
                            }
                            bool v712;
                            v712 = v711 == false;
                            if (v712){
                                assert("The indices should be inside the range of the dimension." && v711);
                            } else {
                            }
                            bool v714;
                            v714 = 0l <= v668;
                            bool v716;
                            if (v714){
                                bool v715;
                                v715 = v668 < 1l;
                                v716 = v715;
                            } else {
                                v716 = false;
                            }
                            bool v717;
                            v717 = v716 == false;
                            if (v717){
                                assert("The indices should be inside the range of the dimension." && v716);
                            } else {
                            }
                            int v719;
                            v719 = v668 * 2l;
                            int v720;
                            v720 = v707 + v719;
                            bool v721;
                            v721 = 0l <= v705;
                            bool v723;
                            if (v721){
                                bool v722;
                                v722 = v705 < 1l;
                                v723 = v722;
                            } else {
                                v723 = false;
                            }
                            bool v724;
                            v724 = v723 == false;
                            if (v724){
                                assert("The indices should be inside the range of the dimension." && v723);
                            } else {
                            }
                            int v726;
                            v726 = v705 * 2l;
                            int v727;
                            v727 = v720 + v726;
                            assert("Tensor range check" && 0 <= v705 && v705 < 1l);
                            assert("Tensor range check" && 0 <= v707 && v707 < 2l);
                            int v728;
                            v728 = 2l * v705;
                            int v729;
                            v729 = v728 + v707;
                            v696[v729] = v727;
                            v707 += 1l ;
                        }
                        v705 += 1l ;
                    }
                    double v730[2l];
                    double v731[2l];
                    int v732;
                    v732 = 0l;
                    while (while_method_5(v732)){
                        int v734;
                        v734 = 0l;
                        while (while_method_0(v734)){
                            assert("Tensor range check" && 0 <= v732 && v732 < 1l);
                            assert("Tensor range check" && 0 <= v734 && v734 < 2l);
                            int v736;
                            v736 = 2l * v732;
                            int v737;
                            v737 = v736 + v734;
                            double v738;
                            v738 = v694[v737];
                            double v739;
                            v739 = v695[v737];
                            assert("Tensor range check" && 0 <= v732 && v732 < 1l);
                            assert("Tensor range check" && 0 <= v734 && v734 < 2l);
                            v730[v737] = 0.0;
                            v731[v737] = 0.0;
                            v734 += 1l ;
                        }
                        v732 += 1l ;
                    }
                    int v740;
                    v740 = 0l;
                    while (while_method_5(v740)){
                        assert("Tensor range check" && 0 <= v740 && v740 < 1l);
                        int v742;
                        v742 = 2l * v740;
                        int v743;
                        v743 = v742 + v693;
                        assert("Tensor range check" && 0 <= v740 && v740 < 1l);
                        int4* v744;
                        v744 = reinterpret_cast<int4*>(v730 + v742);
                        int4* v745;
                        v745 = reinterpret_cast<int4*>(v688 + v743);
                        assert("Pointer alignment check" && (unsigned long long)(v744) % 2l == 0 && (unsigned long long)(v745) % 2l == 0);
                        *v745 = *v744;
                        int4* v746;
                        v746 = reinterpret_cast<int4*>(v731 + v742);
                        int4* v747;
                        v747 = reinterpret_cast<int4*>(v689 + v743);
                        assert("Pointer alignment check" && (unsigned long long)(v746) % 2l == 0 && (unsigned long long)(v747) % 2l == 0);
                        *v747 = *v746;
                        v740 += 1l ;
                    }
                    assert("Tensor range check" && 0 <= v683 && v683 < 512l);
                    v672 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                assert("Tensor range check" && 0 <= v664 && v664 < 512l);
                asm("barrier.cta.sync %0;" :: "r"(0l));
                assert("Tensor range check" && 0 <= v610 && v610 < 4l);
                assert("Tensor range check" && 0 <= v615 && v615 < 12288l);
                int v748;
                v748 = 12288l * v610;
                int v749;
                v749 = v748 + v615;
                v608[v749] = 0l;
                v610 += 1l ;
            }
            v23 += 1l ;
        }
        unsigned int * v750;
        v750 = reinterpret_cast<unsigned int *>(&v0[12582912ull]);
        int * v752;
        v752 = reinterpret_cast<int *>(&v2[262144ull]);
        float * v754;
        v754 = reinterpret_cast<float *>(&v2[262160ull]);
        float * v756;
        v756 = reinterpret_cast<float *>(&v2[524304ull]);
        float * v758;
        v758 = reinterpret_cast<float *>(&v2[786448ull]);
        float * v760;
        v760 = reinterpret_cast<float *>(&v2[1048592ull]);
        float * v762;
        v762 = reinterpret_cast<float *>(&v2[1310736ull]);
        float * v764;
        v764 = reinterpret_cast<float *>(&v2[1572880ull]);
        float * v766;
        v766 = reinterpret_cast<float *>(&v2[1835024ull]);
        int * v768;
        v768 = reinterpret_cast<int *>(&v0[12779520ull]);
        float * v770;
        v770 = reinterpret_cast<float *>(&v0[15925248ull]);
        int * v772;
        v772 = reinterpret_cast<int *>(&v0[19070976ull]);
        int * v774;
        v774 = reinterpret_cast<int *>(&v0[22216704ull]);
        double * v776;
        v776 = reinterpret_cast<double *>(&v0[25362432ull]);
        double * v778;
        v778 = reinterpret_cast<double *>(&v0[37945344ull]);
        double * v780;
        v780 = reinterpret_cast<double *>(&v2[2097168ull]);
        double * v782;
        v782 = reinterpret_cast<double *>(&v2[2883600ull]);
        int * v784;
        v784 = reinterpret_cast<int *>(&v2[3670032ull]);
        v13.sync() ;
        int v786;
        v786 = threadIdx.x;
        int v787;
        v787 = blockIdx.x;
        int v788;
        v788 = v787 * 512l;
        int v789;
        v789 = v786 + v788;
        bool v790;
        v790 = v789 == 0l;
        if (v790){
            int v791;
            v791 = 0l;
            int v792;
            v792 = 4l;
            int v793;
            v793 = int_range_24(v792, v791, v20);
            v752[0l] = v793;
        } else {
        }
        __syncwarp();
        int v794;
        v794 = threadIdx.x;
        bool v795;
        v795 = 0l <= v794;
        bool v796;
        v796 = v795 == false;
        if (v796){
            assert("The index needs to be zero or positive." && v795);
        } else {
        }
        int v798;
        v798 = v794 % 1l;
        int v799;
        v799 = v794 % 512l;
        int v800;
        v800 = v794 / 512l;
        bool v801;
        v801 = v800 < 1l;
        bool v802;
        v802 = v801 == false;
        if (v802){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v801);
        } else {
        }
        assert("Tensor range check" && 0 <= v800 && v800 < 1l);
        assert("Tensor range check" && 0 <= v799 && v799 < 512l);
        assert("Tensor range check" && 0 <= v798 && v798 < 1l);
        int v804;
        v804 = 4l * v798;
        int v805;
        v805 = 4l * v799;
        int v806;
        v806 = v805 + v804;
        int v807;
        v807 = 16384l * v800;
        int v808;
        v808 = v807 + v806;
        assert("Tensor range check" && 0 <= v800 && v800 < 1l);
        assert("Tensor range check" && 0 <= v799 && v799 < 512l);
        assert("Tensor range check" && 0 <= v798 && v798 < 1l);
        int v809;
        v809 = blockIdx.x;
        int v810;
        v810 = v809;
        while (while_method_7(v810)){
            bool v812;
            v812 = 0l <= v810;
            bool v813;
            v813 = v812 == false;
            if (v813){
                assert("The index needs to be zero or positive." && v812);
            } else {
            }
            int v815;
            v815 = v810 % 8l;
            int v816;
            v816 = v810 / 8l;
            bool v817;
            v817 = v816 < 4l;
            bool v818;
            v818 = v817 == false;
            if (v818){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v817);
            } else {
            }
            assert("Tensor range check" && 0 <= v816 && v816 < 4l);
            assert("Tensor range check" && 0 <= v815 && v815 < 8l);
            int v820;
            v820 = 2048l * v815;
            int v821;
            v821 = v820 + v808;
            int v822;
            v822 = 16384l * v816;
            int v823;
            v823 = v822 + v821;
            float v824[4l];
            float v825[4l];
            float v826[4l];
            float v827[4l];
            float v828[4l];
            float v829[4l];
            float v830[4l];
            int v831[4l];
            int v832;
            v832 = 0l;
            while (while_method_5(v832)){
                assert("Tensor range check" && 0 <= v832 && v832 < 1l);
                int v834;
                v834 = 4l * v832;
                assert("Tensor range check" && 0 <= v832 && v832 < 1l);
                int v835;
                v835 = v834 + v823;
                int4* v836;
                v836 = reinterpret_cast<int4*>(v754 + v835);
                int4* v837;
                v837 = reinterpret_cast<int4*>(v824 + v834);
                assert("Pointer alignment check" && (unsigned long long)(v836) % 4l == 0 && (unsigned long long)(v837) % 4l == 0);
                *v837 = *v836;
                int4* v838;
                v838 = reinterpret_cast<int4*>(v756 + v835);
                int4* v839;
                v839 = reinterpret_cast<int4*>(v825 + v834);
                assert("Pointer alignment check" && (unsigned long long)(v838) % 4l == 0 && (unsigned long long)(v839) % 4l == 0);
                *v839 = *v838;
                int4* v840;
                v840 = reinterpret_cast<int4*>(v758 + v835);
                int4* v841;
                v841 = reinterpret_cast<int4*>(v826 + v834);
                assert("Pointer alignment check" && (unsigned long long)(v840) % 4l == 0 && (unsigned long long)(v841) % 4l == 0);
                *v841 = *v840;
                int4* v842;
                v842 = reinterpret_cast<int4*>(v760 + v835);
                int4* v843;
                v843 = reinterpret_cast<int4*>(v827 + v834);
                assert("Pointer alignment check" && (unsigned long long)(v842) % 4l == 0 && (unsigned long long)(v843) % 4l == 0);
                *v843 = *v842;
                int4* v844;
                v844 = reinterpret_cast<int4*>(v762 + v835);
                int4* v845;
                v845 = reinterpret_cast<int4*>(v828 + v834);
                assert("Pointer alignment check" && (unsigned long long)(v844) % 4l == 0 && (unsigned long long)(v845) % 4l == 0);
                *v845 = *v844;
                int4* v846;
                v846 = reinterpret_cast<int4*>(v764 + v835);
                int4* v847;
                v847 = reinterpret_cast<int4*>(v829 + v834);
                assert("Pointer alignment check" && (unsigned long long)(v846) % 4l == 0 && (unsigned long long)(v847) % 4l == 0);
                *v847 = *v846;
                int4* v848;
                v848 = reinterpret_cast<int4*>(v766 + v835);
                int4* v849;
                v849 = reinterpret_cast<int4*>(v830 + v834);
                assert("Pointer alignment check" && (unsigned long long)(v848) % 4l == 0 && (unsigned long long)(v849) % 4l == 0);
                *v849 = *v848;
                v832 += 1l ;
            }
            int v850;
            v850 = 0l;
            while (while_method_5(v850)){
                int v852;
                v852 = 0l;
                while (while_method_4(v852)){
                    bool v854;
                    v854 = 0l <= v852;
                    bool v856;
                    if (v854){
                        bool v855;
                        v855 = v852 < 4l;
                        v856 = v855;
                    } else {
                        v856 = false;
                    }
                    bool v857;
                    v857 = v856 == false;
                    if (v857){
                        assert("The indices should be inside the range of the dimension." && v856);
                    } else {
                    }
                    bool v859;
                    v859 = 0l <= v798;
                    bool v861;
                    if (v859){
                        bool v860;
                        v860 = v798 < 1l;
                        v861 = v860;
                    } else {
                        v861 = false;
                    }
                    bool v862;
                    v862 = v861 == false;
                    if (v862){
                        assert("The indices should be inside the range of the dimension." && v861);
                    } else {
                    }
                    int v864;
                    v864 = v798 * 4l;
                    int v865;
                    v865 = v852 + v864;
                    bool v866;
                    v866 = 0l <= v850;
                    bool v868;
                    if (v866){
                        bool v867;
                        v867 = v850 < 1l;
                        v868 = v867;
                    } else {
                        v868 = false;
                    }
                    bool v869;
                    v869 = v868 == false;
                    if (v869){
                        assert("The indices should be inside the range of the dimension." && v868);
                    } else {
                    }
                    int v871;
                    v871 = v850 * 4l;
                    int v872;
                    v872 = v865 + v871;
                    assert("Tensor range check" && 0 <= v850 && v850 < 1l);
                    assert("Tensor range check" && 0 <= v852 && v852 < 4l);
                    int v873;
                    v873 = 4l * v850;
                    int v874;
                    v874 = v873 + v852;
                    v831[v874] = v872;
                    v852 += 1l ;
                }
                v850 += 1l ;
            }
            bool v875;
            v875 = 0l <= v800;
            bool v876;
            v876 = v875 && v801;
            bool v877;
            v877 = v876 == false;
            if (v877){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v876);
            } else {
            }
            bool v879;
            v879 = 0l <= v799;
            bool v881;
            if (v879){
                bool v880;
                v880 = v799 < 512l;
                v881 = v880;
            } else {
                v881 = false;
            }
            bool v882;
            v882 = v881 == false;
            if (v882){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v881);
            } else {
            }
            bool v884;
            v884 = 0l <= v816;
            bool v885;
            v885 = v884 && v817;
            bool v886;
            v886 = v885 == false;
            if (v886){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v885);
            } else {
            }
            bool v888;
            v888 = 0l <= v815;
            bool v890;
            if (v888){
                bool v889;
                v889 = v815 < 8l;
                v890 = v889;
            } else {
                v890 = false;
            }
            bool v891;
            v891 = v890 == false;
            if (v891){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v890);
            } else {
            }
            int v893;
            v893 = v815 * 512l;
            int v894;
            v894 = v816 + v800;
            int v895;
            v895 = v893 + v799;
            bool v896[4l];
            int v897;
            v897 = 0l;
            while (while_method_5(v897)){
                int v899;
                v899 = 0l;
                while (while_method_4(v899)){
                    assert("Tensor range check" && 0 <= v897 && v897 < 1l);
                    assert("Tensor range check" && 0 <= v899 && v899 < 4l);
                    int v901;
                    v901 = 4l * v897;
                    int v902;
                    v902 = v901 + v899;
                    float v903;
                    v903 = v826[v902];
                    bool v904;
                    v904 = v903 == 0.0f;
                    bool v905;
                    v905 = v904 != true;
                    assert("Tensor range check" && 0 <= v897 && v897 < 1l);
                    assert("Tensor range check" && 0 <= v899 && v899 < 4l);
                    v896[v902] = v905;
                    v899 += 1l ;
                }
                v897 += 1l ;
            }
            bool v906;
            v906 = false;
            int v907;
            v907 = 0l;
            while (while_method_5(v907)){
                int v909;
                v909 = 0l;
                while (while_method_4(v909)){
                    assert("Tensor range check" && 0 <= v907 && v907 < 1l);
                    assert("Tensor range check" && 0 <= v909 && v909 < 4l);
                    int v911;
                    v911 = 4l * v907;
                    int v912;
                    v912 = v911 + v909;
                    bool v913;
                    v913 = v896[v912];
                    bool v914;
                    v914 = v906 || v913;
                    v906 = v914;
                    v909 += 1l ;
                }
                v907 += 1l ;
            }
            auto v915 = cooperative_groups::coalesced_threads();
            int v916;
            v916 = threadIdx.x;
            auto v917 = cooperative_groups::labeled_partition(v915,v916);
            Closure8 v918{};
            bool v919;
            v919 = cooperative_groups::reduce(v917, v906, v918);
            if (v919){
                float v920[4l];
                int v921;
                v921 = 0l;
                while (while_method_5(v921)){
                    int v923;
                    v923 = 0l;
                    while (while_method_4(v923)){
                        assert("Tensor range check" && 0 <= v921 && v921 < 1l);
                        assert("Tensor range check" && 0 <= v923 && v923 < 4l);
                        int v925;
                        v925 = 4l * v921;
                        int v926;
                        v926 = v925 + v923;
                        float v927;
                        v927 = v825[v926];
                        float v928;
                        v928 = v826[v926];
                        float v929;
                        v929 = v927 + v928;
                        bool v930;
                        v930 = 0.0f >= v929;
                        float v931;
                        if (v930){
                            v931 = 0.0f;
                        } else {
                            v931 = v929;
                        }
                        assert("Tensor range check" && 0 <= v921 && v921 < 1l);
                        assert("Tensor range check" && 0 <= v923 && v923 < 4l);
                        v920[v926] = v931;
                        v923 += 1l ;
                    }
                    v921 += 1l ;
                }
                float v932[4l];
                int v933;
                v933 = 0l;
                while (while_method_5(v933)){
                    int v935;
                    v935 = 0l;
                    while (while_method_4(v935)){
                        assert("Tensor range check" && 0 <= v933 && v933 < 1l);
                        assert("Tensor range check" && 0 <= v935 && v935 < 4l);
                        int v937;
                        v937 = 4l * v933;
                        int v938;
                        v938 = v937 + v935;
                        float v939;
                        v939 = v920[v938];
                        bool v940;
                        v940 = 0.0f >= v939;
                        float v941;
                        if (v940){
                            v941 = 0.0f;
                        } else {
                            v941 = v939;
                        }
                        assert("Tensor range check" && 0 <= v933 && v933 < 1l);
                        assert("Tensor range check" && 0 <= v935 && v935 < 4l);
                        v932[v938] = v941;
                        v935 += 1l ;
                    }
                    v933 += 1l ;
                }
                float v942;
                v942 = 0.0f;
                int v943;
                v943 = 0l;
                while (while_method_5(v943)){
                    int v945;
                    v945 = 0l;
                    while (while_method_4(v945)){
                        assert("Tensor range check" && 0 <= v943 && v943 < 1l);
                        assert("Tensor range check" && 0 <= v945 && v945 < 4l);
                        int v947;
                        v947 = 4l * v943;
                        int v948;
                        v948 = v947 + v945;
                        float v949;
                        v949 = v932[v948];
                        float v950;
                        v950 = v942 + v949;
                        v942 = v950;
                        v945 += 1l ;
                    }
                    v943 += 1l ;
                }
                auto v951 = cooperative_groups::coalesced_threads();
                int v952;
                v952 = threadIdx.x;
                auto v953 = cooperative_groups::labeled_partition(v951,v952);
                Closure1 v954{};
                float v955;
                v955 = cooperative_groups::reduce(v953, v942, v954);
                float v956[4l];
                int v957;
                v957 = 0l;
                while (while_method_5(v957)){
                    int v959;
                    v959 = 0l;
                    while (while_method_4(v959)){
                        assert("Tensor range check" && 0 <= v957 && v957 < 1l);
                        assert("Tensor range check" && 0 <= v959 && v959 < 4l);
                        int v961;
                        v961 = 4l * v957;
                        int v962;
                        v962 = v961 + v959;
                        float v963;
                        v963 = v932[v962];
                        bool v964;
                        v964 = v955 == 0.0f;
                        bool v965;
                        v965 = v964 != true;
                        float v967;
                        if (v965){
                            float v966;
                            v966 = v963 / v955;
                            v967 = v966;
                        } else {
                            v967 = 0.25f;
                        }
                        assert("Tensor range check" && 0 <= v957 && v957 < 1l);
                        assert("Tensor range check" && 0 <= v959 && v959 < 4l);
                        v956[v962] = v967;
                        v959 += 1l ;
                    }
                    v957 += 1l ;
                }
                float v968[4l];
                int v969;
                v969 = 0l;
                while (while_method_5(v969)){
                    int v971;
                    v971 = 0l;
                    while (while_method_4(v971)){
                        assert("Tensor range check" && 0 <= v969 && v969 < 1l);
                        assert("Tensor range check" && 0 <= v971 && v971 < 4l);
                        int v973;
                        v973 = 4l * v969;
                        int v974;
                        v974 = v973 + v971;
                        float v975;
                        v975 = v824[v974];
                        float v976;
                        v976 = v956[v974];
                        float v977;
                        v977 = v975 + v976;
                        assert("Tensor range check" && 0 <= v969 && v969 < 1l);
                        assert("Tensor range check" && 0 <= v971 && v971 < 4l);
                        v968[v974] = v977;
                        v971 += 1l ;
                    }
                    v969 += 1l ;
                }
                float v978[4l];
                int v979;
                v979 = 0l;
                while (while_method_5(v979)){
                    int v981;
                    v981 = 0l;
                    while (while_method_4(v981)){
                        assert("Tensor range check" && 0 <= v979 && v979 < 1l);
                        assert("Tensor range check" && 0 <= v981 && v981 < 4l);
                        int v983;
                        v983 = 4l * v979;
                        int v984;
                        v984 = v983 + v981;
                        float v985;
                        v985 = v968[v984];
                        float v986;
                        v986 = -v985;
                        bool v987;
                        v987 = v985 >= v986;
                        float v988;
                        if (v987){
                            v988 = v985;
                        } else {
                            v988 = v986;
                        }
                        assert("Tensor range check" && 0 <= v979 && v979 < 1l);
                        assert("Tensor range check" && 0 <= v981 && v981 < 4l);
                        v978[v984] = v988;
                        v981 += 1l ;
                    }
                    v979 += 1l ;
                }
                float v989;
                v989 = 0.0f;
                int v990;
                v990 = 0l;
                while (while_method_5(v990)){
                    int v992;
                    v992 = 0l;
                    while (while_method_4(v992)){
                        assert("Tensor range check" && 0 <= v990 && v990 < 1l);
                        assert("Tensor range check" && 0 <= v992 && v992 < 4l);
                        int v994;
                        v994 = 4l * v990;
                        int v995;
                        v995 = v994 + v992;
                        float v996;
                        v996 = v978[v995];
                        float v997;
                        v997 = v989 + v996;
                        v989 = v997;
                        v992 += 1l ;
                    }
                    v990 += 1l ;
                }
                auto v998 = cooperative_groups::coalesced_threads();
                int v999;
                v999 = threadIdx.x;
                auto v1000 = cooperative_groups::labeled_partition(v998,v999);
                float v1001;
                v1001 = cooperative_groups::reduce(v1000, v989, v954);
                bool v1002;
                v1002 = v1001 > 100.0f;
                float v1004;
                if (v1002){
                    float v1003;
                    v1003 = 100.0f / v1001;
                    v1004 = v1003;
                } else {
                    v1004 = 1.0f;
                }
                float v1005[4l];
                int v1006;
                v1006 = 0l;
                while (while_method_5(v1006)){
                    int v1008;
                    v1008 = 0l;
                    while (while_method_4(v1008)){
                        assert("Tensor range check" && 0 <= v1006 && v1006 < 1l);
                        assert("Tensor range check" && 0 <= v1008 && v1008 < 4l);
                        int v1010;
                        v1010 = 4l * v1006;
                        int v1011;
                        v1011 = v1010 + v1008;
                        float v1012;
                        v1012 = v978[v1011];
                        float v1013;
                        v1013 = v1004 * v1012;
                        assert("Tensor range check" && 0 <= v1006 && v1006 < 1l);
                        assert("Tensor range check" && 0 <= v1008 && v1008 < 4l);
                        v1005[v1011] = v1013;
                        v1008 += 1l ;
                    }
                    v1006 += 1l ;
                }
                float v1014[4l];
                float v1015[4l];
                int v1016;
                v1016 = 0l;
                while (while_method_5(v1016)){
                    int v1018;
                    v1018 = 0l;
                    while (while_method_4(v1018)){
                        assert("Tensor range check" && 0 <= v1016 && v1016 < 1l);
                        assert("Tensor range check" && 0 <= v1018 && v1018 < 4l);
                        int v1020;
                        v1020 = 4l * v1016;
                        int v1021;
                        v1021 = v1020 + v1018;
                        float v1022;
                        v1022 = v824[v1021];
                        float v1023;
                        v1023 = v825[v1021];
                        float v1024;
                        v1024 = v826[v1021];
                        float v1025;
                        v1025 = v827[v1021];
                        float v1026;
                        v1026 = v828[v1021];
                        float v1027;
                        v1027 = v829[v1021];
                        float v1028;
                        v1028 = v830[v1021];
                        float v1029;
                        v1029 = v1025 + v1027;
                        float v1030;
                        v1030 = v1026 + v1028;
                        assert("Tensor range check" && 0 <= v1016 && v1016 < 1l);
                        assert("Tensor range check" && 0 <= v1018 && v1018 < 4l);
                        v1014[v1021] = v1029;
                        v1015[v1021] = v1030;
                        v1018 += 1l ;
                    }
                    v1016 += 1l ;
                }
                int v1031;
                v1031 = 0l;
                while (while_method_5(v1031)){
                    int v1033;
                    v1033 = 0l;
                    while (while_method_4(v1033)){
                        assert("Tensor range check" && 0 <= v1031 && v1031 < 1l);
                        assert("Tensor range check" && 0 <= v1033 && v1033 < 4l);
                        int v1035;
                        v1035 = 4l * v1031;
                        int v1036;
                        v1036 = v1035 + v1033;
                        float v1037;
                        v1037 = v1005[v1036];
                        float v1038;
                        v1038 = v920[v1036];
                        float v1039;
                        v1039 = v1014[v1036];
                        float v1040;
                        v1040 = v1015[v1036];
                        assert("Tensor range check" && 0 <= v1031 && v1031 < 1l);
                        assert("Tensor range check" && 0 <= v1033 && v1033 < 4l);
                        v824[v1036] = v1037;
                        v825[v1036] = v1038;
                        v826[v1036] = 0.0f;
                        v827[v1036] = v1039;
                        v828[v1036] = v1040;
                        v829[v1036] = 0.0f;
                        v830[v1036] = 0.0f;
                        v1033 += 1l ;
                    }
                    v1031 += 1l ;
                }
            } else {
            }
            assert("Tensor range check" && 0 <= v816 && v816 < 4l);
            assert("Tensor range check" && 0 <= v815 && v815 < 8l);
            int v1041;
            v1041 = 0l;
            while (while_method_5(v1041)){
                assert("Tensor range check" && 0 <= v1041 && v1041 < 1l);
                int v1043;
                v1043 = 4l * v1041;
                int v1044;
                v1044 = v1043 + v823;
                assert("Tensor range check" && 0 <= v1041 && v1041 < 1l);
                int4* v1045;
                v1045 = reinterpret_cast<int4*>(v824 + v1043);
                int4* v1046;
                v1046 = reinterpret_cast<int4*>(v754 + v1044);
                assert("Pointer alignment check" && (unsigned long long)(v1045) % 4l == 0 && (unsigned long long)(v1046) % 4l == 0);
                *v1046 = *v1045;
                int4* v1047;
                v1047 = reinterpret_cast<int4*>(v825 + v1043);
                int4* v1048;
                v1048 = reinterpret_cast<int4*>(v756 + v1044);
                assert("Pointer alignment check" && (unsigned long long)(v1047) % 4l == 0 && (unsigned long long)(v1048) % 4l == 0);
                *v1048 = *v1047;
                int4* v1049;
                v1049 = reinterpret_cast<int4*>(v826 + v1043);
                int4* v1050;
                v1050 = reinterpret_cast<int4*>(v758 + v1044);
                assert("Pointer alignment check" && (unsigned long long)(v1049) % 4l == 0 && (unsigned long long)(v1050) % 4l == 0);
                *v1050 = *v1049;
                int4* v1051;
                v1051 = reinterpret_cast<int4*>(v827 + v1043);
                int4* v1052;
                v1052 = reinterpret_cast<int4*>(v760 + v1044);
                assert("Pointer alignment check" && (unsigned long long)(v1051) % 4l == 0 && (unsigned long long)(v1052) % 4l == 0);
                *v1052 = *v1051;
                int4* v1053;
                v1053 = reinterpret_cast<int4*>(v828 + v1043);
                int4* v1054;
                v1054 = reinterpret_cast<int4*>(v762 + v1044);
                assert("Pointer alignment check" && (unsigned long long)(v1053) % 4l == 0 && (unsigned long long)(v1054) % 4l == 0);
                *v1054 = *v1053;
                int4* v1055;
                v1055 = reinterpret_cast<int4*>(v829 + v1043);
                int4* v1056;
                v1056 = reinterpret_cast<int4*>(v764 + v1044);
                assert("Pointer alignment check" && (unsigned long long)(v1055) % 4l == 0 && (unsigned long long)(v1056) % 4l == 0);
                *v1056 = *v1055;
                int4* v1057;
                v1057 = reinterpret_cast<int4*>(v830 + v1043);
                int4* v1058;
                v1058 = reinterpret_cast<int4*>(v766 + v1044);
                assert("Pointer alignment check" && (unsigned long long)(v1057) % 4l == 0 && (unsigned long long)(v1058) % 4l == 0);
                *v1058 = *v1057;
                v1041 += 1l ;
            }
            v810 += 24l ;
        }
        v13.sync() ;
        v21 += 1l ;
    }
    int v1059;
    v1059 = threadIdx.x;
    int v1060;
    v1060 = blockIdx.x;
    int v1061;
    v1061 = v1060 * 512l;
    int v1062;
    v1062 = v1059 + v1061;
    int v1063;
    v1063 = v1062;
    while (while_method_14(v1063)){
        bool v1065;
        v1065 = 0l <= v1063;
        bool v1066;
        v1066 = v1065 == false;
        if (v1066){
            assert("The index needs to be zero or positive." && v1065);
        } else {
        }
        int v1068;
        v1068 = v1063 % 64l;
        int v1069;
        v1069 = v1063 / 64l;
        bool v1070;
        v1070 = v1069 < 2l;
        bool v1071;
        v1071 = v1070 == false;
        if (v1071){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1070);
        } else {
        }
        assert("Tensor range check" && 0 <= v1069 && v1069 < 2l);
        assert("Tensor range check" && 0 <= v1068 && v1068 < 64l);
        int v1073;
        v1073 = 4l * v1068;
        int v1074;
        v1074 = 256l * v1069;
        int v1075;
        v1075 = v1074 + v1073;
        assert("Tensor range check" && 0 <= v1069 && v1069 < 2l);
        assert("Tensor range check" && 0 <= v1068 && v1068 < 64l);
        float v1076[4l];
        float v1077[4l];
        float v1078[4l];
        int4* v1079;
        v1079 = reinterpret_cast<int4*>(v4 + v1075);
        int4* v1080;
        v1080 = reinterpret_cast<int4*>(v1076 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v1079) % 4l == 0 && (unsigned long long)(v1080) % 4l == 0);
        *v1080 = *v1079;
        int4* v1081;
        v1081 = reinterpret_cast<int4*>(v5 + v1075);
        int4* v1082;
        v1082 = reinterpret_cast<int4*>(v1077 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v1081) % 4l == 0 && (unsigned long long)(v1082) % 4l == 0);
        *v1082 = *v1081;
        // Pushing the loop unrolling to: 0
        int v1083;
        v1083 = 0l;
        #pragma unroll
        while (while_method_4(v1083)){
            assert("Tensor range check" && 0 <= v1083 && v1083 < 4l);
            float v1085;
            v1085 = v1076[v1083];
            float v1086;
            v1086 = v1077[v1083];
            bool v1087;
            v1087 = v1086 == 0.0f;
            bool v1088;
            v1088 = v1087 != true;
            float v1090;
            if (v1088){
                float v1089;
                v1089 = v1085 / v1086;
                v1090 = v1089;
            } else {
                v1090 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v1083 && v1083 < 4l);
            v1078[v1083] = v1090;
            v1083 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v1091;
        v1091 = reinterpret_cast<int4*>(v1078 + 0l);
        int4* v1092;
        v1092 = reinterpret_cast<int4*>(v6 + v1075);
        assert("Pointer alignment check" && (unsigned long long)(v1091) % 4l == 0 && (unsigned long long)(v1092) % 4l == 0);
        *v1092 = *v1091;
        v1063 += 12288l ;
    }
    v13.sync() ;
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
        v2 = cp.empty(16,dtype=cp.uint8)
        v3 = cp.empty(1184,dtype=cp.uint8)
        v4 = method0(v0)
        v5, v6, v7, v8, v9, v10, v11, v12, v13 = method7(v1)
        method37(v3, v5, v6, v7, v8, v9)
        del v5, v6, v7, v8, v9
        v16 = "{}\n"
        v17 = "Going to run the Leduc full kernel."
        print(v16.format(v17),end="")
        del v16, v17
        v18 = time.perf_counter()
        v19 = []
        match v4:
            case US0_0(_): # ActionSelected
                method55(v2, v4)
                v78 = cp.cuda.Device().attributes['MultiProcessorCount']
                v79 = v78 == 24
                del v78
                v80 = v79 == False
                if v80:
                    v81 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v79, v81
                    del v81
                else:
                    pass
                del v79, v80
                v82 = 0
                v83 = raw_module.get_function(f"entry{v82}")
                del v82
                v83.max_dynamic_shared_size_bytes = 81920 
                v83((24,),(512,),(v3, v2, v10, v11, v12, v13),shared_mem=81920)
                del v83
            case US0_1(_): # PlayerChanged
                method55(v2, v4)
                v71 = cp.cuda.Device().attributes['MultiProcessorCount']
                v72 = v71 == 24
                del v71
                v73 = v72 == False
                if v73:
                    v74 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v72, v74
                    del v74
                else:
                    pass
                del v72, v73
                v75 = 0
                v76 = raw_module.get_function(f"entry{v75}")
                del v75
                v76.max_dynamic_shared_size_bytes = 81920 
                v76((24,),(512,),(v3, v2, v10, v11, v12, v13),shared_mem=81920)
                del v76
            case US0_2(): # StartGame
                method55(v2, v4)
                v64 = cp.cuda.Device().attributes['MultiProcessorCount']
                v65 = v64 == 24
                del v64
                v66 = v65 == False
                if v66:
                    v67 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v65, v67
                    del v67
                else:
                    pass
                del v65, v66
                v68 = 0
                v69 = raw_module.get_function(f"entry{v68}")
                del v68
                v69.max_dynamic_shared_size_bytes = 81920 
                v69((24,),(512,),(v3, v2, v10, v11, v12, v13),shared_mem=81920)
                del v69
            case US0_3(): # StartTrainingVsRando
                v20 = cp.zeros(1024,dtype=cp.float32) # type: ignore
                v21 = cp.zeros(1024,dtype=cp.float32) # type: ignore
                v22 = cp.empty(1024,dtype=cp.float32)
                v23 = cp.cuda.Device().attributes['MultiProcessorCount']
                v24 = v23 == 24
                del v23
                v25 = v24 == False
                if v25:
                    v26 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v24, v26
                    del v26
                else:
                    pass
                del v24, v25
                v27 = 1
                v28 = raw_module.get_function(f"entry{v27}")
                del v27
                v28.max_dynamic_shared_size_bytes = 81920 
                v28((24,),(512,),(v10, v11, v12, v13, v20, v21, v22),shared_mem=81920)
                del v20, v21, v28
                v29 = []
                v31 = v22[0:]
                del v22
                v32 = v31.get()
                del v31
                v33 = 0
                while method58(v33):
                    v35 = []
                    v36 = 0
                    while method59(v36):
                        assert 0 <= v33 < 4, 'Tensor range check'
                        assert 0 <= v36 < 256, 'Tensor range check'
                        v38 = 256 * v33
                        v39 = v38 + v36
                        del v38
                        v40 = v32[v39].item()
                        del v39
                        v35.append(v40)
                        del v40
                        v36 += 1 
                    del v36
                    v29.append(v35)
                    del v35
                    v33 += 1 
                del v32, v33
                v41 = US9_0(v29)
                del v29
                v19.append(v41)
                del v41
            case US0_4(): # StartTrainingVsSelf
                v42 = cp.zeros(512,dtype=cp.float32) # type: ignore
                v43 = cp.zeros(512,dtype=cp.float32) # type: ignore
                v44 = cp.empty(512,dtype=cp.float32)
                v45 = cp.cuda.Device().attributes['MultiProcessorCount']
                v46 = v45 == 24
                del v45
                v47 = v46 == False
                if v47:
                    v48 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v46, v48
                    del v48
                else:
                    pass
                del v46, v47
                v49 = 2
                v50 = raw_module.get_function(f"entry{v49}")
                del v49
                v50.max_dynamic_shared_size_bytes = 81920 
                v50((24,),(512,),(v10, v11, v12, v13, v42, v43, v44),shared_mem=81920)
                del v42, v43, v50
                v51 = []
                v53 = v44[0:]
                del v44
                v54 = v53.get()
                del v53
                v55 = 0
                while method45(v55):
                    v57 = []
                    v58 = 0
                    while method59(v58):
                        assert 0 <= v55 < 2, 'Tensor range check'
                        assert 0 <= v58 < 256, 'Tensor range check'
                        v60 = 256 * v55
                        v61 = v60 + v58
                        del v60
                        v62 = v54[v61].item()
                        del v61
                        v57.append(v62)
                        del v62
                        v58 += 1 
                    del v58
                    v51.append(v57)
                    del v57
                    v55 += 1 
                del v54, v55
                v63 = US9_1(v51)
                del v51
                v19.append(v63)
                del v63
            case t:
                raise Exception(f'Pattern matching miss. Got: {t}')
        del v2, v4
        cp.cuda.get_current_stream().synchronize()
        v86 = "{}"
        v87 = "The time it took to run the kernel (in seconds) is: "
        print(v86.format(v87),end="")
        del v86, v87
        v88 = time.perf_counter()
        v89 = v88 - v18
        del v18, v88
        v92 = "{:.6f}\n"
        print(v92.format(v89),end="")
        del v89, v92
        v93, v94, v95, v96, v97 = method60(v3)
        del v3
        return method77(v93, v94, v95, v96, v97, v10, v11, v12, v13, v19)
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
        v35 = 63
        v36 = US3_0()
        v37 = US7_0()
        v38 = 50528256
        v39 = 3866640
        return method117(v35, v36, v7, v1, v37, v9, v38, v8, v39)
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
def method36(v0 : object) -> u64:
    assert isinstance(v0,u64), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method33(v0 : object) -> Tuple[cp.ndarray, u64]:
    v1 = v0[0] # type: ignore
    v2 = method34(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method36(v3)
    del v3
    return v2, v4
def method32(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["output"] # type: ignore
    v2, v3 = method33(v1)
    del v1
    v4 = v0["param"] # type: ignore
    del v0
    v5, v6 = method33(v4)
    del v4
    return v2, v3, v5, v6
def method31(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1, v2, v3, v4 = method32(v0)
    del v0
    return v1, v2, v3, v4
def method30(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["model_data"] # type: ignore
    del v0
    v2, v3, v4, v5 = method31(v1)
    del v1
    return v2, v3, v4, v5
def method8(v0 : object) -> Tuple[u32, US3, static_array_list, static_array, US7, cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["game"] # type: ignore
    v2, v3, v4, v5, v6 = method9(v1)
    del v1
    v7 = v0["neural"] # type: ignore
    del v0
    v8, v9, v10, v11 = method30(v7)
    del v7
    return v2, v3, v4, v5, v6, v8, v9, v10, v11
def method7(v0 : object) -> Tuple[u32, US3, static_array_list, static_array, US7, cp.ndarray, u64, cp.ndarray, u64]:
    return method8(v0)
def method38(v0 : cp.ndarray, v1 : u32) -> None:
    v3 = v0[0:].view(cp.uint32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method39(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[4:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method40(v0 : cp.ndarray) -> None:
    del v0
    return 
def method42(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[0:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method44(v0 : cp.ndarray, v1 : US6) -> None:
    v2 = v1.tag
    method42(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US6_0(): # Jack
            del v1
            return method40(v4)
        case US6_1(): # King
            del v1
            return method40(v4)
        case US6_2(): # Queen
            del v1
            return method40(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method45(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method43(v0 : cp.ndarray, v1 : US5, v2 : bool, v3 : static_array, v4 : i32, v5 : static_array, v6 : i32) -> None:
    v7 = v1.tag
    method42(v0, v7)
    del v7
    v9 = v0[4:].view(cp.uint8)
    match v1:
        case US5_0(): # None
            method40(v9)
        case US5_1(v10): # Some
            method44(v9, v10)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v1, v9
    v12 = v0[8:].view(cp.bool_)
    v12[0] = v2
    del v2, v12
    v13 = 0
    while method45(v13):
        v15 = u64(v13)
        v16 = v15 * 4
        del v15
        v17 = 12 + v16
        del v16
        v19 = v0[v17:].view(cp.uint8)
        del v17
        v21 = v3[v13]
        method44(v19, v21)
        del v19, v21
        v13 += 1 
    del v3, v13
    v23 = v0[20:].view(cp.int32)
    v23[0] = v4
    del v4, v23
    v24 = 0
    while method45(v24):
        v26 = u64(v24)
        v27 = v26 * 4
        del v26
        v28 = 24 + v27
        del v27
        v30 = v0[v28:].view(cp.uint8)
        del v28
        v32 = v5[v24]
        method42(v30, v32)
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
    method42(v0, v8)
    del v8
    v10 = v0[4:].view(cp.uint8)
    match v1:
        case US5_0(): # None
            method40(v10)
        case US5_1(v11): # Some
            method44(v10, v11)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v1, v10
    v13 = v0[8:].view(cp.bool_)
    v13[0] = v2
    del v2, v13
    v14 = 0
    while method45(v14):
        v16 = u64(v14)
        v17 = v16 * 4
        del v16
        v18 = 12 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v22 = v3[v14]
        method44(v20, v22)
        del v20, v22
        v14 += 1 
    del v3, v14
    v24 = v0[20:].view(cp.int32)
    v24[0] = v4
    del v4, v24
    v25 = 0
    while method45(v25):
        v27 = u64(v25)
        v28 = v27 * 4
        del v27
        v29 = 24 + v28
        del v28
        v31 = v0[v29:].view(cp.uint8)
        del v29
        v33 = v5[v25]
        method42(v31, v33)
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
            return method40(v38)
        case US1_1(): # Fold
            del v7
            return method40(v38)
        case US1_2(): # Raise
            del v7
            return method40(v38)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method41(v0 : cp.ndarray, v1 : US4) -> None:
    v2 = v1.tag
    method42(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US4_0(v5, v6, v7, v8, v9, v10): # ChanceCommunityCard
            del v1
            return method43(v4, v5, v6, v7, v8, v9, v10)
        case US4_1(): # ChanceInit
            del v1
            return method40(v4)
        case US4_2(v11, v12, v13, v14, v15, v16): # Round
            del v1
            return method43(v4, v11, v12, v13, v14, v15, v16)
        case US4_3(v17, v18, v19, v20, v21, v22, v23): # RoundWithAction
            del v1
            return method46(v4, v17, v18, v19, v20, v21, v22, v23)
        case US4_4(v24, v25, v26, v27, v28, v29): # TerminalCall
            del v1
            return method43(v4, v24, v25, v26, v27, v28, v29)
        case US4_5(v30, v31, v32, v33, v34, v35): # TerminalFold
            del v1
            return method43(v4, v30, v31, v32, v33, v34, v35)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method48(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[80:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method50(v0 : cp.ndarray, v1 : i32, v2 : US1) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = v2.tag
    method39(v0, v5)
    del v5
    v7 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US1_0(): # Call
            del v2
            return method40(v7)
        case US1_1(): # Fold
            del v2
            return method40(v7)
        case US1_2(): # Raise
            del v2
            return method40(v7)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method51(v0 : cp.ndarray, v1 : i32, v2 : US6) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = v2.tag
    method39(v0, v5)
    del v5
    v7 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US6_0(): # Jack
            del v2
            return method40(v7)
        case US6_1(): # King
            del v2
            return method40(v7)
        case US6_2(): # Queen
            del v2
            return method40(v7)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method52(v0 : cp.ndarray, v1 : static_array, v2 : i32, v3 : i32) -> None:
    v4 = 0
    while method45(v4):
        v6 = u64(v4)
        v7 = v6 * 4
        del v6
        v9 = v0[v7:].view(cp.uint8)
        del v7
        v11 = v1[v4]
        method44(v9, v11)
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
def method49(v0 : cp.ndarray, v1 : US8) -> None:
    v2 = v1.tag
    method42(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US8_0(v5): # CommunityCardIs
            del v1
            return method44(v4, v5)
        case US8_1(v6, v7): # PlayerAction
            del v1
            return method50(v4, v6, v7)
        case US8_2(v8, v9): # PlayerGotCard
            del v1
            return method51(v4, v8, v9)
        case US8_3(v10, v11, v12): # Showdown
            del v1
            return method52(v4, v10, v11, v12)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method53(v0 : cp.ndarray, v1 : US2) -> None:
    v2 = v1.tag
    method42(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US2_0(): # Computer
            del v1
            return method40(v4)
        case US2_1(): # Human
            del v1
            return method40(v4)
        case US2_2(): # Random
            del v1
            return method40(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method54(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[1128:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method37(v0 : cp.ndarray, v1 : u32, v2 : US3, v3 : static_array_list, v4 : static_array, v5 : US7) -> None:
    method38(v0, v1)
    del v1
    v6 = v2.tag
    method39(v0, v6)
    del v6
    v8 = v0[16:].view(cp.uint8)
    match v2:
        case US3_0(): # None
            method40(v8)
        case US3_1(v9): # Some
            method41(v8, v9)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v2, v8
    v10 = v3.length
    method48(v0, v10)
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
        method49(v18, v20)
        del v18, v20
        v12 += 1 
    del v3, v11, v12
    v21 = 0
    while method45(v21):
        v23 = u64(v21)
        v24 = v23 * 4
        del v23
        v25 = 1120 + v24
        del v24
        v27 = v0[v25:].view(cp.uint8)
        del v25
        v29 = v4[v21]
        method53(v27, v29)
        del v27, v29
        v21 += 1 
    del v4, v21
    v30 = v5.tag
    method54(v0, v30)
    del v30
    v32 = v0[1136:].view(cp.uint8)
    del v0
    match v5:
        case US7_0(): # GameNotStarted
            del v5
            return method40(v32)
        case US7_1(v33, v34, v35, v36, v37, v38): # GameOver
            del v5
            return method43(v32, v33, v34, v35, v36, v37, v38)
        case US7_2(v39, v40, v41, v42, v43, v44): # WaitingForActionFromPlayerId
            del v5
            return method43(v32, v39, v40, v41, v42, v43, v44)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method56(v0 : cp.ndarray, v1 : US1) -> None:
    v2 = v1.tag
    method42(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US1_0(): # Call
            del v1
            return method40(v4)
        case US1_1(): # Fold
            del v1
            return method40(v4)
        case US1_2(): # Raise
            del v1
            return method40(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method57(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method45(v2):
        v4 = u64(v2)
        v5 = v4 * 4
        del v4
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v9 = v1[v2]
        method53(v7, v9)
        del v7, v9
        v2 += 1 
    del v0, v1, v2
    return 
def method55(v0 : cp.ndarray, v1 : US0) -> None:
    v2 = v1.tag
    method42(v0, v2)
    del v2
    v4 = v0[8:].view(cp.uint8)
    del v0
    match v1:
        case US0_0(v5): # ActionSelected
            del v1
            return method56(v4, v5)
        case US0_1(v6): # PlayerChanged
            del v1
            return method57(v4, v6)
        case US0_2(): # StartGame
            del v1
            return method40(v4)
        case US0_3(): # StartTrainingVsRando
            del v1
            return method40(v4)
        case US0_4(): # StartTrainingVsSelf
            del v1
            return method40(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method58(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method59(v0 : i32) -> bool:
    v1 = v0 < 256
    del v0
    return v1
def method61(v0 : cp.ndarray) -> u32:
    v2 = v0[0:].view(cp.uint32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method62(v0 : cp.ndarray) -> i32:
    v2 = v0[4:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method63(v0 : cp.ndarray) -> None:
    del v0
    return 
def method65(v0 : cp.ndarray) -> i32:
    v2 = v0[0:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method67(v0 : cp.ndarray) -> US6:
    v1 = method65(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method63(v3)
        del v3
        return US6_0()
    elif v1 == 1:
        del v1
        method63(v3)
        del v3
        return US6_1()
    elif v1 == 2:
        del v1
        method63(v3)
        del v3
        return US6_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method66(v0 : cp.ndarray) -> Tuple[US5, bool, static_array, i32, static_array, i32]:
    v1 = method65(v0)
    v3 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method63(v3)
        v8 = US5_0()
    elif v1 == 1:
        v6 = method67(v3)
        v8 = US5_1(v6)
    else:
        raise Exception("Invalid tag.")
    del v1, v3
    v10 = v0[8:].view(cp.bool_)
    v11 = v10[0].item()
    del v10
    v13 = static_array(2)
    v14 = 0
    while method45(v14):
        v16 = u64(v14)
        v17 = v16 * 4
        del v16
        v18 = 12 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v21 = method67(v20)
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
    while method45(v27):
        v29 = u64(v27)
        v30 = v29 * 4
        del v29
        v31 = 24 + v30
        del v30
        v33 = v0[v31:].view(cp.uint8)
        del v31
        v34 = method65(v33)
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
def method69(v0 : cp.ndarray) -> i32:
    v2 = v0[36:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method68(v0 : cp.ndarray) -> Tuple[US5, bool, static_array, i32, static_array, i32, US1]:
    v1 = method65(v0)
    v3 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method63(v3)
        v8 = US5_0()
    elif v1 == 1:
        v6 = method67(v3)
        v8 = US5_1(v6)
    else:
        raise Exception("Invalid tag.")
    del v1, v3
    v10 = v0[8:].view(cp.bool_)
    v11 = v10[0].item()
    del v10
    v13 = static_array(2)
    v14 = 0
    while method45(v14):
        v16 = u64(v14)
        v17 = v16 * 4
        del v16
        v18 = 12 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v21 = method67(v20)
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
    while method45(v27):
        v29 = u64(v27)
        v30 = v29 * 4
        del v29
        v31 = 24 + v30
        del v30
        v33 = v0[v31:].view(cp.uint8)
        del v31
        v34 = method65(v33)
        del v33
        v26[v27] = v34
        del v34
        v27 += 1 
    del v27
    v36 = v0[32:].view(cp.int32)
    v37 = v36[0].item()
    del v36
    v38 = method69(v0)
    v40 = v0[40:].view(cp.uint8)
    del v0
    if v38 == 0:
        method63(v40)
        v45 = US1_0()
    elif v38 == 1:
        method63(v40)
        v45 = US1_1()
    elif v38 == 2:
        method63(v40)
        v45 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v38, v40
    return v8, v11, v13, v24, v26, v37, v45
def method64(v0 : cp.ndarray) -> US4:
    v1 = method65(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5, v6, v7, v8, v9, v10 = method66(v3)
        del v3
        return US4_0(v5, v6, v7, v8, v9, v10)
    elif v1 == 1:
        del v1
        method63(v3)
        del v3
        return US4_1()
    elif v1 == 2:
        del v1
        v13, v14, v15, v16, v17, v18 = method66(v3)
        del v3
        return US4_2(v13, v14, v15, v16, v17, v18)
    elif v1 == 3:
        del v1
        v20, v21, v22, v23, v24, v25, v26 = method68(v3)
        del v3
        return US4_3(v20, v21, v22, v23, v24, v25, v26)
    elif v1 == 4:
        del v1
        v28, v29, v30, v31, v32, v33 = method66(v3)
        del v3
        return US4_4(v28, v29, v30, v31, v32, v33)
    elif v1 == 5:
        del v1
        v35, v36, v37, v38, v39, v40 = method66(v3)
        del v3
        return US4_5(v35, v36, v37, v38, v39, v40)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method70(v0 : cp.ndarray) -> i32:
    v2 = v0[80:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method72(v0 : cp.ndarray) -> Tuple[i32, US1]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method62(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method63(v6)
        v11 = US1_0()
    elif v4 == 1:
        method63(v6)
        v11 = US1_1()
    elif v4 == 2:
        method63(v6)
        v11 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v11
def method73(v0 : cp.ndarray) -> Tuple[i32, US6]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method62(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method63(v6)
        v11 = US6_0()
    elif v4 == 1:
        method63(v6)
        v11 = US6_1()
    elif v4 == 2:
        method63(v6)
        v11 = US6_2()
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v11
def method74(v0 : cp.ndarray) -> Tuple[static_array, i32, i32]:
    v2 = static_array(2)
    v3 = 0
    while method45(v3):
        v5 = u64(v3)
        v6 = v5 * 4
        del v5
        v8 = v0[v6:].view(cp.uint8)
        del v6
        v9 = method67(v8)
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
def method71(v0 : cp.ndarray) -> US8:
    v1 = method65(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5 = method67(v3)
        del v3
        return US8_0(v5)
    elif v1 == 1:
        del v1
        v7, v8 = method72(v3)
        del v3
        return US8_1(v7, v8)
    elif v1 == 2:
        del v1
        v10, v11 = method73(v3)
        del v3
        return US8_2(v10, v11)
    elif v1 == 3:
        del v1
        v13, v14, v15 = method74(v3)
        del v3
        return US8_3(v13, v14, v15)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method75(v0 : cp.ndarray) -> US2:
    v1 = method65(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method63(v3)
        del v3
        return US2_0()
    elif v1 == 1:
        del v1
        method63(v3)
        del v3
        return US2_1()
    elif v1 == 2:
        del v1
        method63(v3)
        del v3
        return US2_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method76(v0 : cp.ndarray) -> i32:
    v2 = v0[1128:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method60(v0 : cp.ndarray) -> Tuple[u32, US3, static_array_list, static_array, US7]:
    v1 = method61(v0)
    v2 = method62(v0)
    v4 = v0[16:].view(cp.uint8)
    if v2 == 0:
        method63(v4)
        v9 = US3_0()
    elif v2 == 1:
        v7 = method64(v4)
        v9 = US3_1(v7)
    else:
        raise Exception("Invalid tag.")
    del v2, v4
    v11 = static_array_list(32)
    v12 = method70(v0)
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
        v21 = method71(v20)
        del v20
        v11[v14] = v21
        del v21
        v14 += 1 
    del v13, v14
    v23 = static_array(2)
    v24 = 0
    while method45(v24):
        v26 = u64(v24)
        v27 = v26 * 4
        del v26
        v28 = 1120 + v27
        del v27
        v30 = v0[v28:].view(cp.uint8)
        del v28
        v31 = method75(v30)
        del v30
        v23[v24] = v31
        del v31
        v24 += 1 
    del v24
    v32 = method76(v0)
    v34 = v0[1136:].view(cp.uint8)
    del v0
    if v32 == 0:
        method63(v34)
        v51 = US7_0()
    elif v32 == 1:
        v37, v38, v39, v40, v41, v42 = method66(v34)
        v51 = US7_1(v37, v38, v39, v40, v41, v42)
    elif v32 == 2:
        v44, v45, v46, v47, v48, v49 = method66(v34)
        v51 = US7_2(v44, v45, v46, v47, v48, v49)
    else:
        raise Exception("Invalid tag.")
    del v32, v34
    return v1, v9, v11, v23, v51
def method83(v0 : u32) -> object:
    v1 = v0
    del v0
    return v1
def method82(v0 : u32) -> object:
    return method83(v0)
def method85() -> object:
    v0 = []
    return v0
def method89(v0 : US6) -> object:
    match v0:
        case US6_0(): # Jack
            del v0
            v1 = method85()
            v2 = "Jack"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US6_1(): # King
            del v0
            v4 = method85()
            v5 = "King"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US6_2(): # Queen
            del v0
            v7 = method85()
            v8 = "Queen"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method88(v0 : US5) -> object:
    match v0:
        case US5_0(): # None
            del v0
            v1 = method85()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US5_1(v4): # Some
            del v0
            v5 = method89(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method90(v0 : bool) -> object:
    v1 = v0
    del v0
    return v1
def method91(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method45(v2):
        v5 = v0[v2]
        v6 = method89(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method92(v0 : i32) -> object:
    v1 = v0
    del v0
    return v1
def method93(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method45(v2):
        v5 = v0[v2]
        v6 = method92(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method87(v0 : US5, v1 : bool, v2 : static_array, v3 : i32, v4 : static_array, v5 : i32) -> object:
    v6 = method88(v0)
    del v0
    v7 = method90(v1)
    del v1
    v8 = method91(v2)
    del v2
    v9 = method92(v3)
    del v3
    v10 = method93(v4)
    del v4
    v11 = method92(v5)
    del v5
    v12 = {'community_card': v6, 'is_button_s_first_move': v7, 'pl_card': v8, 'player_turn': v9, 'pot': v10, 'raises_left': v11}
    del v6, v7, v8, v9, v10, v11
    return v12
def method95(v0 : US1) -> object:
    match v0:
        case US1_0(): # Call
            del v0
            v1 = method85()
            v2 = "Call"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US1_1(): # Fold
            del v0
            v4 = method85()
            v5 = "Fold"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US1_2(): # Raise
            del v0
            v7 = method85()
            v8 = "Raise"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method94(v0 : US5, v1 : bool, v2 : static_array, v3 : i32, v4 : static_array, v5 : i32, v6 : US1) -> object:
    v7 = []
    v8 = method87(v0, v1, v2, v3, v4, v5)
    del v0, v1, v2, v3, v4, v5
    v7.append(v8)
    del v8
    v9 = method95(v6)
    del v6
    v7.append(v9)
    del v9
    v10 = v7
    del v7
    return v10
def method86(v0 : US4) -> object:
    match v0:
        case US4_0(v1, v2, v3, v4, v5, v6): # ChanceCommunityCard
            del v0
            v7 = method87(v1, v2, v3, v4, v5, v6)
            del v1, v2, v3, v4, v5, v6
            v8 = "ChanceCommunityCard"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US4_1(): # ChanceInit
            del v0
            v10 = method85()
            v11 = "ChanceInit"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US4_2(v13, v14, v15, v16, v17, v18): # Round
            del v0
            v19 = method87(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "Round"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case US4_3(v22, v23, v24, v25, v26, v27, v28): # RoundWithAction
            del v0
            v29 = method94(v22, v23, v24, v25, v26, v27, v28)
            del v22, v23, v24, v25, v26, v27, v28
            v30 = "RoundWithAction"
            v31 = [v30,v29]
            del v29, v30
            return v31
        case US4_4(v32, v33, v34, v35, v36, v37): # TerminalCall
            del v0
            v38 = method87(v32, v33, v34, v35, v36, v37)
            del v32, v33, v34, v35, v36, v37
            v39 = "TerminalCall"
            v40 = [v39,v38]
            del v38, v39
            return v40
        case US4_5(v41, v42, v43, v44, v45, v46): # TerminalFold
            del v0
            v47 = method87(v41, v42, v43, v44, v45, v46)
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
            v1 = method85()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US3_1(v4): # Some
            del v0
            v5 = method86(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method81(v0 : u32, v1 : US3) -> object:
    v2 = method82(v0)
    del v0
    v3 = method84(v1)
    del v1
    v4 = {'deck': v2, 'game': v3}
    del v2, v3
    return v4
def method99(v0 : i32, v1 : US1) -> object:
    v2 = []
    v3 = method92(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method95(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method100(v0 : i32, v1 : US6) -> object:
    v2 = []
    v3 = method92(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method89(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method101(v0 : static_array, v1 : i32, v2 : i32) -> object:
    v3 = method91(v0)
    del v0
    v4 = method92(v1)
    del v1
    v5 = method92(v2)
    del v2
    v6 = {'cards_shown': v3, 'chips_won': v4, 'winner_id': v5}
    del v3, v4, v5
    return v6
def method98(v0 : US8) -> object:
    match v0:
        case US8_0(v1): # CommunityCardIs
            del v0
            v2 = method89(v1)
            del v1
            v3 = "CommunityCardIs"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US8_1(v5, v6): # PlayerAction
            del v0
            v7 = method99(v5, v6)
            del v5, v6
            v8 = "PlayerAction"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US8_2(v10, v11): # PlayerGotCard
            del v0
            v12 = method100(v10, v11)
            del v10, v11
            v13 = "PlayerGotCard"
            v14 = [v13,v12]
            del v12, v13
            return v14
        case US8_3(v15, v16, v17): # Showdown
            del v0
            v18 = method101(v15, v16, v17)
            del v15, v16, v17
            v19 = "Showdown"
            v20 = [v19,v18]
            del v18, v19
            return v20
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method97(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method5(v2, v3):
        v6 = v0[v3]
        v7 = method98(v6)
        del v6
        v1.append(v7)
        del v7
        v3 += 1 
    del v0, v2, v3
    return v1
def method103(v0 : US2) -> object:
    match v0:
        case US2_0(): # Computer
            del v0
            v1 = method85()
            v2 = "Computer"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US2_1(): # Human
            del v0
            v4 = method85()
            v5 = "Human"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US2_2(): # Random
            del v0
            v7 = method85()
            v8 = "Random"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method102(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method45(v2):
        v5 = v0[v2]
        v6 = method103(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method104(v0 : US7) -> object:
    match v0:
        case US7_0(): # GameNotStarted
            del v0
            v1 = method85()
            v2 = "GameNotStarted"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US7_1(v4, v5, v6, v7, v8, v9): # GameOver
            del v0
            v10 = method87(v4, v5, v6, v7, v8, v9)
            del v4, v5, v6, v7, v8, v9
            v11 = "GameOver"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US7_2(v13, v14, v15, v16, v17, v18): # WaitingForActionFromPlayerId
            del v0
            v19 = method87(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "WaitingForActionFromPlayerId"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method96(v0 : static_array_list, v1 : static_array, v2 : US7) -> object:
    v3 = method97(v0)
    del v0
    v4 = method102(v1)
    del v1
    v5 = method104(v2)
    del v2
    v6 = {'messages': v3, 'pl_type': v4, 'ui_game_state': v5}
    del v3, v4, v5
    return v6
def method80(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7) -> object:
    v5 = method81(v0, v1)
    del v0, v1
    v6 = method96(v2, v3, v4)
    del v2, v3, v4
    v7 = {'private': v5, 'public': v6}
    del v5, v6
    return v7
def method110(v0 : cp.ndarray) -> object:
    v1 = v0
    del v0
    return v1
def method109(v0 : cp.ndarray) -> object:
    return method110(v0)
def method111(v0 : u64) -> object:
    v1 = v0
    del v0
    return v1
def method108(v0 : cp.ndarray, v1 : u64) -> object:
    v2 = []
    v3 = method109(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method111(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method107(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method108(v0, v1)
    del v0, v1
    v5 = method108(v2, v3)
    del v2, v3
    v6 = {'output': v4, 'param': v5}
    del v4, v5
    return v6
def method106(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    return method107(v0, v1, v2, v3)
def method105(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method106(v0, v1, v2, v3)
    del v0, v1, v2, v3
    v5 = {'model_data': v4}
    del v4
    return v5
def method79(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64) -> object:
    v9 = method80(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    v10 = method105(v5, v6, v7, v8)
    del v5, v6, v7, v8
    v11 = {'game': v9, 'neural': v10}
    del v9, v10
    return v11
def method116(v0 : f32) -> object:
    v1 = v0
    del v0
    return v1
def method115(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method5(v2, v3):
        v5 = v0[v3]
        v6 = method116(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method114(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method5(v2, v3):
        v5 = v0[v3]
        v6 = method115(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method113(v0 : US9) -> object:
    match v0:
        case US9_0(v1): # AddRewardsRando
            del v0
            v2 = method114(v1)
            del v1
            v3 = "AddRewardsRando"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US9_1(v5): # AddRewardsSelf
            del v0
            v6 = method114(v5)
            del v5
            v7 = "AddRewardsSelf"
            v8 = [v7,v6]
            del v6, v7
            return v8
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method112(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method5(v2, v3):
        v5 = v0[v3]
        v6 = method113(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method78(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64, v9 : list) -> object:
    v10 = []
    v11 = method79(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8
    v10.append(v11)
    del v11
    v12 = method112(v9)
    del v9
    v10.append(v12)
    del v12
    v13 = v10
    del v10
    return v13
def method77(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64, v9 : list) -> object:
    v10 = method78(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9
    return v10
def method117(v0 : u32, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US7, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64) -> object:
    v9 = method79(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8
    return v9
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
