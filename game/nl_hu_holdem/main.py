kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
using default_int = long;
using default_uint = unsigned long;
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
__device__ long f_1(unsigned char * v0);
__device__ void f_3(unsigned char * v0);
__device__ Union1 f_2(unsigned char * v0);
__device__ Union2 f_5(unsigned char * v0);
__device__ static_array<Union2,2l> f_4(unsigned char * v0);
__device__ Union0 f_0(unsigned char * v0);
struct Tuple0;
struct Union3;
struct Union6;
struct Union5;
struct Union4;
struct Union7;
struct Tuple1;
__device__ unsigned long long f_7(unsigned char * v0);
__device__ long f_8(unsigned char * v0);
__device__ unsigned char f_12(unsigned char * v0);
__device__ unsigned char f_11(unsigned char * v0);
__device__ static_array_list<unsigned char,5l> f_10(unsigned char * v0);
struct Tuple2;
__device__ Tuple2 f_13(unsigned char * v0);
struct Tuple3;
__device__ long f_15(unsigned char * v0);
__device__ Tuple3 f_14(unsigned char * v0);
struct Tuple4;
__device__ Tuple4 f_16(unsigned char * v0);
struct Tuple5;
__device__ Tuple0 f_19(unsigned char * v0);
__device__ Tuple0 f_18(unsigned char * v0);
__device__ Tuple5 f_17(unsigned char * v0);
__device__ Union3 f_9(unsigned char * v0);
__device__ long f_20(unsigned char * v0);
struct Tuple6;
__device__ static_array<unsigned char,2l> f_23(unsigned char * v0);
__device__ long f_24(unsigned char * v0);
__device__ static_array<unsigned char,3l> f_25(unsigned char * v0);
__device__ static_array<unsigned char,5l> f_26(unsigned char * v0);
__device__ static_array<unsigned char,4l> f_27(unsigned char * v0);
__device__ Tuple6 f_22(unsigned char * v0);
struct Tuple7;
__device__ long f_29(unsigned char * v0);
__device__ Tuple7 f_28(unsigned char * v0);
__device__ Union5 f_21(unsigned char * v0);
__device__ long f_30(unsigned char * v0);
__device__ Tuple1 f_6(unsigned char * v0);
struct Tuple8;
struct Tuple9;
struct Tuple10;
struct Tuple11;
struct Tuple12;
__device__ unsigned long loop_35(unsigned long v0, curandStatePhilox4_32_10_t & v1);
__device__ Tuple12 draw_card_34(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
__device__ Tuple10 draw_cards_33(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
__device__ static_array_list<unsigned char,5l> get_community_cards_36(Union6 v0, static_array<unsigned char,3l> v1);
__device__ bool player_can_act_38(long v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<long,2l> v2, long v3, static_array<long,2l> v4, Union6 v5);
__device__ Union5 go_next_street_39(long v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<long,2l> v2, long v3, static_array<long,2l> v4, Union6 v5);
__device__ Union5 try_round_37(long v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<long,2l> v2, long v3, static_array<long,2l> v4, Union6 v5);
struct Tuple13;
__device__ Tuple13 draw_cards_40(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
struct Tuple14;
__device__ Tuple14 draw_cards_41(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
__device__ static_array_list<unsigned char,5l> get_community_cards_42(Union6 v0, static_array<unsigned char,1l> v1);
struct Tuple15;
__device__ long loop_45(static_array<float,6l> v0, float v1, long v2);
__device__ long sample_discrete__44(static_array<float,6l> v0, curandStatePhilox4_32_10_t & v1);
__device__ Union1 sample_discrete_43(static_array<Tuple15,6l> v0, curandStatePhilox4_32_10_t & v1);
struct Tuple16;
struct Tuple17;
struct Union8;
struct Tuple18;
struct Union9;
struct Tuple19;
struct Tuple20;
struct Union10;
struct Union11;
struct Union12;
struct Union13;
struct Union14;
__device__ Tuple0 score_46(static_array<unsigned char,7l> v0);
__device__ Union5 play_loop_inner_32(unsigned long long & v0, dynamic_array_list<Union3,128l> & v1, curandStatePhilox4_32_10_t & v2, static_array<Union2,2l> v3, Union5 v4);
__device__ Tuple8 play_loop_31(Union4 v0, static_array<Union2,2l> v1, Union7 v2, unsigned long long & v3, dynamic_array_list<Union3,128l> & v4, curandStatePhilox4_32_10_t & v5, Union5 v6);
__device__ void f_48(unsigned char * v0, unsigned long long v1);
__device__ void f_49(unsigned char * v0, long v1);
__device__ void f_51(unsigned char * v0, long v1);
__device__ void f_54(unsigned char * v0, unsigned char v1);
__device__ void f_53(unsigned char * v0, unsigned char v1);
__device__ void f_52(unsigned char * v0, static_array_list<unsigned char,5l> v1);
__device__ void f_55(unsigned char * v0, long v1, long v2);
__device__ void f_57(unsigned char * v0, long v1);
__device__ void f_58(unsigned char * v0);
__device__ void f_56(unsigned char * v0, long v1, Union1 v2);
__device__ void f_59(unsigned char * v0, long v1, static_array<unsigned char,2l> v2);
__device__ void f_62(unsigned char * v0, static_array<unsigned char,5l> v1, char v2);
__device__ void f_61(unsigned char * v0, static_array<unsigned char,5l> v1, char v2);
__device__ void f_60(unsigned char * v0, long v1, static_array<Tuple0,2l> v2, long v3);
__device__ void f_50(unsigned char * v0, Union3 v1);
__device__ void f_63(unsigned char * v0, long v1);
__device__ void f_66(unsigned char * v0, static_array<unsigned char,2l> v1);
__device__ void f_67(unsigned char * v0, long v1);
__device__ void f_68(unsigned char * v0, static_array<unsigned char,3l> v1);
__device__ void f_69(unsigned char * v0, static_array<unsigned char,5l> v1);
__device__ void f_70(unsigned char * v0, static_array<unsigned char,4l> v1);
__device__ void f_65(unsigned char * v0, long v1, static_array<static_array<unsigned char,2l>,2l> v2, static_array<long,2l> v3, long v4, static_array<long,2l> v5, Union6 v6);
__device__ void f_72(unsigned char * v0, long v1);
__device__ void f_71(unsigned char * v0, long v1, static_array<static_array<unsigned char,2l>,2l> v2, static_array<long,2l> v3, long v4, static_array<long,2l> v5, Union6 v6, Union1 v7);
__device__ void f_64(unsigned char * v0, Union5 v1);
__device__ void f_73(unsigned char * v0, Union2 v1);
__device__ void f_74(unsigned char * v0, long v1);
__device__ void f_47(unsigned char * v0, unsigned long long v1, dynamic_array_list<Union3,128l> v2, Union4 v3, static_array<Union2,2l> v4, Union7 v5);
__device__ void f_76(unsigned char * v0, long v1);
__device__ void f_75(unsigned char * v0, dynamic_array_list<Union3,128l> v1, static_array<Union2,2l> v2, Union7 v3);
struct Union1_0 { // A_All_In
};
struct Union1_1 { // A_Call
};
struct Union1_2 { // A_Fold
};
struct Union1_3 { // A_Raise
    long v0;
    __device__ Union1_3(long t0) : v0(t0) {}
    __device__ Union1_3() = delete;
};
struct Union1 {
    union {
        Union1_0 case0; // A_All_In
        Union1_1 case1; // A_Call
        Union1_2 case2; // A_Fold
        Union1_3 case3; // A_Raise
    };
    unsigned char tag{255};
    __device__ Union1() {}
    __device__ Union1(Union1_0 t) : tag(0), case0(t) {} // A_All_In
    __device__ Union1(Union1_1 t) : tag(1), case1(t) {} // A_Call
    __device__ Union1(Union1_2 t) : tag(2), case2(t) {} // A_Fold
    __device__ Union1(Union1_3 t) : tag(3), case3(t) {} // A_Raise
    __device__ Union1(Union1 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(x.case0); break; // A_All_In
            case 1: new (&this->case1) Union1_1(x.case1); break; // A_Call
            case 2: new (&this->case2) Union1_2(x.case2); break; // A_Fold
            case 3: new (&this->case3) Union1_3(x.case3); break; // A_Raise
        }
    }
    __device__ Union1(Union1 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // A_All_In
            case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // A_Call
            case 2: new (&this->case2) Union1_2(std::move(x.case2)); break; // A_Fold
            case 3: new (&this->case3) Union1_3(std::move(x.case3)); break; // A_Raise
        }
    }
    __device__ Union1 & operator=(Union1 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // A_All_In
                case 1: this->case1 = x.case1; break; // A_Call
                case 2: this->case2 = x.case2; break; // A_Fold
                case 3: this->case3 = x.case3; break; // A_Raise
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
                case 0: this->case0 = std::move(x.case0); break; // A_All_In
                case 1: this->case1 = std::move(x.case1); break; // A_Call
                case 2: this->case2 = std::move(x.case2); break; // A_Fold
                case 3: this->case3 = std::move(x.case3); break; // A_Raise
            }
        } else {
            this->~Union1();
            new (this) Union1{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union1() {
        switch(this->tag){
            case 0: this->case0.~Union1_0(); break; // A_All_In
            case 1: this->case1.~Union1_1(); break; // A_Call
            case 2: this->case2.~Union1_2(); break; // A_Fold
            case 3: this->case3.~Union1_3(); break; // A_Raise
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
struct Tuple0 {
    static_array<unsigned char,5l> v0;
    char v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(static_array<unsigned char,5l> t0, char t1) : v0(t0), v1(t1) {}
};
struct Union3_0 { // CommunityCardsAre
    static_array_list<unsigned char,5l> v0;
    __device__ Union3_0(static_array_list<unsigned char,5l> t0) : v0(t0) {}
    __device__ Union3_0() = delete;
};
struct Union3_1 { // Fold
    long v0;
    long v1;
    __device__ Union3_1(long t0, long t1) : v0(t0), v1(t1) {}
    __device__ Union3_1() = delete;
};
struct Union3_2 { // PlayerAction
    Union1 v1;
    long v0;
    __device__ Union3_2(long t0, Union1 t1) : v0(t0), v1(t1) {}
    __device__ Union3_2() = delete;
};
struct Union3_3 { // PlayerGotCards
    static_array<unsigned char,2l> v1;
    long v0;
    __device__ Union3_3(long t0, static_array<unsigned char,2l> t1) : v0(t0), v1(t1) {}
    __device__ Union3_3() = delete;
};
struct Union3_4 { // Showdown
    static_array<Tuple0,2l> v1;
    long v0;
    long v2;
    __device__ Union3_4(long t0, static_array<Tuple0,2l> t1, long t2) : v0(t0), v1(t1), v2(t2) {}
    __device__ Union3_4() = delete;
};
struct Union3 {
    union {
        Union3_0 case0; // CommunityCardsAre
        Union3_1 case1; // Fold
        Union3_2 case2; // PlayerAction
        Union3_3 case3; // PlayerGotCards
        Union3_4 case4; // Showdown
    };
    unsigned char tag{255};
    __device__ Union3() {}
    __device__ Union3(Union3_0 t) : tag(0), case0(t) {} // CommunityCardsAre
    __device__ Union3(Union3_1 t) : tag(1), case1(t) {} // Fold
    __device__ Union3(Union3_2 t) : tag(2), case2(t) {} // PlayerAction
    __device__ Union3(Union3_3 t) : tag(3), case3(t) {} // PlayerGotCards
    __device__ Union3(Union3_4 t) : tag(4), case4(t) {} // Showdown
    __device__ Union3(Union3 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(x.case0); break; // CommunityCardsAre
            case 1: new (&this->case1) Union3_1(x.case1); break; // Fold
            case 2: new (&this->case2) Union3_2(x.case2); break; // PlayerAction
            case 3: new (&this->case3) Union3_3(x.case3); break; // PlayerGotCards
            case 4: new (&this->case4) Union3_4(x.case4); break; // Showdown
        }
    }
    __device__ Union3(Union3 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(std::move(x.case0)); break; // CommunityCardsAre
            case 1: new (&this->case1) Union3_1(std::move(x.case1)); break; // Fold
            case 2: new (&this->case2) Union3_2(std::move(x.case2)); break; // PlayerAction
            case 3: new (&this->case3) Union3_3(std::move(x.case3)); break; // PlayerGotCards
            case 4: new (&this->case4) Union3_4(std::move(x.case4)); break; // Showdown
        }
    }
    __device__ Union3 & operator=(Union3 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // CommunityCardsAre
                case 1: this->case1 = x.case1; break; // Fold
                case 2: this->case2 = x.case2; break; // PlayerAction
                case 3: this->case3 = x.case3; break; // PlayerGotCards
                case 4: this->case4 = x.case4; break; // Showdown
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
                case 0: this->case0 = std::move(x.case0); break; // CommunityCardsAre
                case 1: this->case1 = std::move(x.case1); break; // Fold
                case 2: this->case2 = std::move(x.case2); break; // PlayerAction
                case 3: this->case3 = std::move(x.case3); break; // PlayerGotCards
                case 4: this->case4 = std::move(x.case4); break; // Showdown
            }
        } else {
            this->~Union3();
            new (this) Union3{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union3() {
        switch(this->tag){
            case 0: this->case0.~Union3_0(); break; // CommunityCardsAre
            case 1: this->case1.~Union3_1(); break; // Fold
            case 2: this->case2.~Union3_2(); break; // PlayerAction
            case 3: this->case3.~Union3_3(); break; // PlayerGotCards
            case 4: this->case4.~Union3_4(); break; // Showdown
        }
        this->tag = 255;
    }
};
struct Union6_0 { // Flop
    static_array<unsigned char,3l> v0;
    __device__ Union6_0(static_array<unsigned char,3l> t0) : v0(t0) {}
    __device__ Union6_0() = delete;
};
struct Union6_1 { // Preflop
};
struct Union6_2 { // River
    static_array<unsigned char,5l> v0;
    __device__ Union6_2(static_array<unsigned char,5l> t0) : v0(t0) {}
    __device__ Union6_2() = delete;
};
struct Union6_3 { // Turn
    static_array<unsigned char,4l> v0;
    __device__ Union6_3(static_array<unsigned char,4l> t0) : v0(t0) {}
    __device__ Union6_3() = delete;
};
struct Union6 {
    union {
        Union6_0 case0; // Flop
        Union6_1 case1; // Preflop
        Union6_2 case2; // River
        Union6_3 case3; // Turn
    };
    unsigned char tag{255};
    __device__ Union6() {}
    __device__ Union6(Union6_0 t) : tag(0), case0(t) {} // Flop
    __device__ Union6(Union6_1 t) : tag(1), case1(t) {} // Preflop
    __device__ Union6(Union6_2 t) : tag(2), case2(t) {} // River
    __device__ Union6(Union6_3 t) : tag(3), case3(t) {} // Turn
    __device__ Union6(Union6 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(x.case0); break; // Flop
            case 1: new (&this->case1) Union6_1(x.case1); break; // Preflop
            case 2: new (&this->case2) Union6_2(x.case2); break; // River
            case 3: new (&this->case3) Union6_3(x.case3); break; // Turn
        }
    }
    __device__ Union6(Union6 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(std::move(x.case0)); break; // Flop
            case 1: new (&this->case1) Union6_1(std::move(x.case1)); break; // Preflop
            case 2: new (&this->case2) Union6_2(std::move(x.case2)); break; // River
            case 3: new (&this->case3) Union6_3(std::move(x.case3)); break; // Turn
        }
    }
    __device__ Union6 & operator=(Union6 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Flop
                case 1: this->case1 = x.case1; break; // Preflop
                case 2: this->case2 = x.case2; break; // River
                case 3: this->case3 = x.case3; break; // Turn
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
                case 0: this->case0 = std::move(x.case0); break; // Flop
                case 1: this->case1 = std::move(x.case1); break; // Preflop
                case 2: this->case2 = std::move(x.case2); break; // River
                case 3: this->case3 = std::move(x.case3); break; // Turn
            }
        } else {
            this->~Union6();
            new (this) Union6{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union6() {
        switch(this->tag){
            case 0: this->case0.~Union6_0(); break; // Flop
            case 1: this->case1.~Union6_1(); break; // Preflop
            case 2: this->case2.~Union6_2(); break; // River
            case 3: this->case3.~Union6_3(); break; // Turn
        }
        this->tag = 255;
    }
};
struct Union5_0 { // G_Flop
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<long,2l> v2;
    static_array<long,2l> v4;
    Union6 v5;
    long v0;
    long v3;
    __device__ Union5_0(long t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<long,2l> t2, long t3, static_array<long,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union5_0() = delete;
};
struct Union5_1 { // G_Fold
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<long,2l> v2;
    static_array<long,2l> v4;
    Union6 v5;
    long v0;
    long v3;
    __device__ Union5_1(long t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<long,2l> t2, long t3, static_array<long,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union5_1() = delete;
};
struct Union5_2 { // G_Preflop
};
struct Union5_3 { // G_River
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<long,2l> v2;
    static_array<long,2l> v4;
    Union6 v5;
    long v0;
    long v3;
    __device__ Union5_3(long t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<long,2l> t2, long t3, static_array<long,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union5_3() = delete;
};
struct Union5_4 { // G_Round
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<long,2l> v2;
    static_array<long,2l> v4;
    Union6 v5;
    long v0;
    long v3;
    __device__ Union5_4(long t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<long,2l> t2, long t3, static_array<long,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union5_4() = delete;
};
struct Union5_5 { // G_Round'
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<long,2l> v2;
    static_array<long,2l> v4;
    Union6 v5;
    Union1 v6;
    long v0;
    long v3;
    __device__ Union5_5(long t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<long,2l> t2, long t3, static_array<long,2l> t4, Union6 t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union5_5() = delete;
};
struct Union5_6 { // G_Showdown
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<long,2l> v2;
    static_array<long,2l> v4;
    Union6 v5;
    long v0;
    long v3;
    __device__ Union5_6(long t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<long,2l> t2, long t3, static_array<long,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union5_6() = delete;
};
struct Union5_7 { // G_Turn
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<long,2l> v2;
    static_array<long,2l> v4;
    Union6 v5;
    long v0;
    long v3;
    __device__ Union5_7(long t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<long,2l> t2, long t3, static_array<long,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union5_7() = delete;
};
struct Union5 {
    union {
        Union5_0 case0; // G_Flop
        Union5_1 case1; // G_Fold
        Union5_2 case2; // G_Preflop
        Union5_3 case3; // G_River
        Union5_4 case4; // G_Round
        Union5_5 case5; // G_Round'
        Union5_6 case6; // G_Showdown
        Union5_7 case7; // G_Turn
    };
    unsigned char tag{255};
    __device__ Union5() {}
    __device__ Union5(Union5_0 t) : tag(0), case0(t) {} // G_Flop
    __device__ Union5(Union5_1 t) : tag(1), case1(t) {} // G_Fold
    __device__ Union5(Union5_2 t) : tag(2), case2(t) {} // G_Preflop
    __device__ Union5(Union5_3 t) : tag(3), case3(t) {} // G_River
    __device__ Union5(Union5_4 t) : tag(4), case4(t) {} // G_Round
    __device__ Union5(Union5_5 t) : tag(5), case5(t) {} // G_Round'
    __device__ Union5(Union5_6 t) : tag(6), case6(t) {} // G_Showdown
    __device__ Union5(Union5_7 t) : tag(7), case7(t) {} // G_Turn
    __device__ Union5(Union5 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(x.case0); break; // G_Flop
            case 1: new (&this->case1) Union5_1(x.case1); break; // G_Fold
            case 2: new (&this->case2) Union5_2(x.case2); break; // G_Preflop
            case 3: new (&this->case3) Union5_3(x.case3); break; // G_River
            case 4: new (&this->case4) Union5_4(x.case4); break; // G_Round
            case 5: new (&this->case5) Union5_5(x.case5); break; // G_Round'
            case 6: new (&this->case6) Union5_6(x.case6); break; // G_Showdown
            case 7: new (&this->case7) Union5_7(x.case7); break; // G_Turn
        }
    }
    __device__ Union5(Union5 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(std::move(x.case0)); break; // G_Flop
            case 1: new (&this->case1) Union5_1(std::move(x.case1)); break; // G_Fold
            case 2: new (&this->case2) Union5_2(std::move(x.case2)); break; // G_Preflop
            case 3: new (&this->case3) Union5_3(std::move(x.case3)); break; // G_River
            case 4: new (&this->case4) Union5_4(std::move(x.case4)); break; // G_Round
            case 5: new (&this->case5) Union5_5(std::move(x.case5)); break; // G_Round'
            case 6: new (&this->case6) Union5_6(std::move(x.case6)); break; // G_Showdown
            case 7: new (&this->case7) Union5_7(std::move(x.case7)); break; // G_Turn
        }
    }
    __device__ Union5 & operator=(Union5 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // G_Flop
                case 1: this->case1 = x.case1; break; // G_Fold
                case 2: this->case2 = x.case2; break; // G_Preflop
                case 3: this->case3 = x.case3; break; // G_River
                case 4: this->case4 = x.case4; break; // G_Round
                case 5: this->case5 = x.case5; break; // G_Round'
                case 6: this->case6 = x.case6; break; // G_Showdown
                case 7: this->case7 = x.case7; break; // G_Turn
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
                case 0: this->case0 = std::move(x.case0); break; // G_Flop
                case 1: this->case1 = std::move(x.case1); break; // G_Fold
                case 2: this->case2 = std::move(x.case2); break; // G_Preflop
                case 3: this->case3 = std::move(x.case3); break; // G_River
                case 4: this->case4 = std::move(x.case4); break; // G_Round
                case 5: this->case5 = std::move(x.case5); break; // G_Round'
                case 6: this->case6 = std::move(x.case6); break; // G_Showdown
                case 7: this->case7 = std::move(x.case7); break; // G_Turn
            }
        } else {
            this->~Union5();
            new (this) Union5{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union5() {
        switch(this->tag){
            case 0: this->case0.~Union5_0(); break; // G_Flop
            case 1: this->case1.~Union5_1(); break; // G_Fold
            case 2: this->case2.~Union5_2(); break; // G_Preflop
            case 3: this->case3.~Union5_3(); break; // G_River
            case 4: this->case4.~Union5_4(); break; // G_Round
            case 5: this->case5.~Union5_5(); break; // G_Round'
            case 6: this->case6.~Union5_6(); break; // G_Showdown
            case 7: this->case7.~Union5_7(); break; // G_Turn
        }
        this->tag = 255;
    }
};
struct Union4_0 { // None
};
struct Union4_1 { // Some
    Union5 v0;
    __device__ Union4_1(Union5 t0) : v0(t0) {}
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
struct Union7_0 { // GameNotStarted
};
struct Union7_1 { // GameOver
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<long,2l> v2;
    static_array<long,2l> v4;
    Union6 v5;
    long v0;
    long v3;
    __device__ Union7_1(long t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<long,2l> t2, long t3, static_array<long,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union7_1() = delete;
};
struct Union7_2 { // WaitingForActionFromPlayerId
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<long,2l> v2;
    static_array<long,2l> v4;
    Union6 v5;
    long v0;
    long v3;
    __device__ Union7_2(long t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<long,2l> t2, long t3, static_array<long,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union7_2() = delete;
};
struct Union7 {
    union {
        Union7_0 case0; // GameNotStarted
        Union7_1 case1; // GameOver
        Union7_2 case2; // WaitingForActionFromPlayerId
    };
    unsigned char tag{255};
    __device__ Union7() {}
    __device__ Union7(Union7_0 t) : tag(0), case0(t) {} // GameNotStarted
    __device__ Union7(Union7_1 t) : tag(1), case1(t) {} // GameOver
    __device__ Union7(Union7_2 t) : tag(2), case2(t) {} // WaitingForActionFromPlayerId
    __device__ Union7(Union7 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(x.case0); break; // GameNotStarted
            case 1: new (&this->case1) Union7_1(x.case1); break; // GameOver
            case 2: new (&this->case2) Union7_2(x.case2); break; // WaitingForActionFromPlayerId
        }
    }
    __device__ Union7(Union7 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(std::move(x.case0)); break; // GameNotStarted
            case 1: new (&this->case1) Union7_1(std::move(x.case1)); break; // GameOver
            case 2: new (&this->case2) Union7_2(std::move(x.case2)); break; // WaitingForActionFromPlayerId
        }
    }
    __device__ Union7 & operator=(Union7 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // GameNotStarted
                case 1: this->case1 = x.case1; break; // GameOver
                case 2: this->case2 = x.case2; break; // WaitingForActionFromPlayerId
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
                case 0: this->case0 = std::move(x.case0); break; // GameNotStarted
                case 1: this->case1 = std::move(x.case1); break; // GameOver
                case 2: this->case2 = std::move(x.case2); break; // WaitingForActionFromPlayerId
            }
        } else {
            this->~Union7();
            new (this) Union7{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union7() {
        switch(this->tag){
            case 0: this->case0.~Union7_0(); break; // GameNotStarted
            case 1: this->case1.~Union7_1(); break; // GameOver
            case 2: this->case2.~Union7_2(); break; // WaitingForActionFromPlayerId
        }
        this->tag = 255;
    }
};
struct Tuple1 {
    unsigned long long v0;
    dynamic_array_list<Union3,128l> v1;
    Union4 v2;
    static_array<Union2,2l> v3;
    Union7 v4;
    __device__ Tuple1() = default;
    __device__ Tuple1(unsigned long long t0, dynamic_array_list<Union3,128l> t1, Union4 t2, static_array<Union2,2l> t3, Union7 t4) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4) {}
};
struct Tuple2 {
    long v0;
    long v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(long t0, long t1) : v0(t0), v1(t1) {}
};
struct Tuple3 {
    Union1 v1;
    long v0;
    __device__ Tuple3() = default;
    __device__ Tuple3(long t0, Union1 t1) : v0(t0), v1(t1) {}
};
struct Tuple4 {
    static_array<unsigned char,2l> v1;
    long v0;
    __device__ Tuple4() = default;
    __device__ Tuple4(long t0, static_array<unsigned char,2l> t1) : v0(t0), v1(t1) {}
};
struct Tuple5 {
    static_array<Tuple0,2l> v1;
    long v0;
    long v2;
    __device__ Tuple5() = default;
    __device__ Tuple5(long t0, static_array<Tuple0,2l> t1, long t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple6 {
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<long,2l> v2;
    static_array<long,2l> v4;
    Union6 v5;
    long v0;
    long v3;
    __device__ Tuple6() = default;
    __device__ Tuple6(long t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<long,2l> t2, long t3, static_array<long,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Tuple7 {
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<long,2l> v2;
    static_array<long,2l> v4;
    Union6 v5;
    Union1 v6;
    long v0;
    long v3;
    __device__ Tuple7() = default;
    __device__ Tuple7(long t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<long,2l> t2, long t3, static_array<long,2l> t4, Union6 t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
};
struct Tuple8 {
    Union4 v0;
    static_array<Union2,2l> v1;
    Union7 v2;
    __device__ Tuple8() = default;
    __device__ Tuple8(Union4 t0, static_array<Union2,2l> t1, Union7 t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple9 {
    Union5 v1;
    bool v0;
    __device__ Tuple9() = default;
    __device__ Tuple9(bool t0, Union5 t1) : v0(t0), v1(t1) {}
};
struct Tuple10 {
    static_array<unsigned char,3l> v0;
    unsigned long long v1;
    __device__ Tuple10() = default;
    __device__ Tuple10(static_array<unsigned char,3l> t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Tuple11 {
    unsigned long long v1;
    long v0;
    __device__ Tuple11() = default;
    __device__ Tuple11(long t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Tuple12 {
    unsigned long long v1;
    unsigned char v0;
    __device__ Tuple12() = default;
    __device__ Tuple12(unsigned char t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Tuple13 {
    static_array<unsigned char,2l> v0;
    unsigned long long v1;
    __device__ Tuple13() = default;
    __device__ Tuple13(static_array<unsigned char,2l> t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Tuple14 {
    static_array<unsigned char,1l> v0;
    unsigned long long v1;
    __device__ Tuple14() = default;
    __device__ Tuple14(static_array<unsigned char,1l> t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Tuple15 {
    Union1 v0;
    float v1;
    __device__ Tuple15() = default;
    __device__ Tuple15(Union1 t0, float t1) : v0(t0), v1(t1) {}
};
struct Tuple16 {
    long v1;
    bool v0;
    __device__ Tuple16() = default;
    __device__ Tuple16(bool t0, long t1) : v0(t0), v1(t1) {}
};
struct Tuple17 {
    long v0;
    long v1;
    long v2;
    __device__ Tuple17() = default;
    __device__ Tuple17(long t0, long t1, long t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Union8_0 { // Eq
};
struct Union8_1 { // Gt
};
struct Union8_2 { // Lt
};
struct Union8 {
    union {
        Union8_0 case0; // Eq
        Union8_1 case1; // Gt
        Union8_2 case2; // Lt
    };
    unsigned char tag{255};
    __device__ Union8() {}
    __device__ Union8(Union8_0 t) : tag(0), case0(t) {} // Eq
    __device__ Union8(Union8_1 t) : tag(1), case1(t) {} // Gt
    __device__ Union8(Union8_2 t) : tag(2), case2(t) {} // Lt
    __device__ Union8(Union8 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(x.case0); break; // Eq
            case 1: new (&this->case1) Union8_1(x.case1); break; // Gt
            case 2: new (&this->case2) Union8_2(x.case2); break; // Lt
        }
    }
    __device__ Union8(Union8 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(std::move(x.case0)); break; // Eq
            case 1: new (&this->case1) Union8_1(std::move(x.case1)); break; // Gt
            case 2: new (&this->case2) Union8_2(std::move(x.case2)); break; // Lt
        }
    }
    __device__ Union8 & operator=(Union8 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Eq
                case 1: this->case1 = x.case1; break; // Gt
                case 2: this->case2 = x.case2; break; // Lt
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
                case 0: this->case0 = std::move(x.case0); break; // Eq
                case 1: this->case1 = std::move(x.case1); break; // Gt
                case 2: this->case2 = std::move(x.case2); break; // Lt
            }
        } else {
            this->~Union8();
            new (this) Union8{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union8() {
        switch(this->tag){
            case 0: this->case0.~Union8_0(); break; // Eq
            case 1: this->case1.~Union8_1(); break; // Gt
            case 2: this->case2.~Union8_2(); break; // Lt
        }
        this->tag = 255;
    }
};
struct Tuple18 {
    long v0;
    long v1;
    unsigned char v2;
    __device__ Tuple18() = default;
    __device__ Tuple18(long t0, long t1, unsigned char t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Union9_0 { // None
};
struct Union9_1 { // Some
    static_array<unsigned char,5l> v0;
    __device__ Union9_1(static_array<unsigned char,5l> t0) : v0(t0) {}
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
struct Tuple19 {
    Union8 v1;
    long v0;
    __device__ Tuple19() = default;
    __device__ Tuple19(long t0, Union8 t1) : v0(t0), v1(t1) {}
};
struct Tuple20 {
    long v0;
    long v1;
    long v2;
    unsigned char v3;
    __device__ Tuple20() = default;
    __device__ Tuple20(long t0, long t1, long t2, unsigned char t3) : v0(t0), v1(t1), v2(t2), v3(t3) {}
};
struct Union10_0 { // None
};
struct Union10_1 { // Some
    static_array<unsigned char,4l> v0;
    static_array<unsigned char,3l> v1;
    __device__ Union10_1(static_array<unsigned char,4l> t0, static_array<unsigned char,3l> t1) : v0(t0), v1(t1) {}
    __device__ Union10_1() = delete;
};
struct Union10 {
    union {
        Union10_0 case0; // None
        Union10_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union10() {}
    __device__ Union10(Union10_0 t) : tag(0), case0(t) {} // None
    __device__ Union10(Union10_1 t) : tag(1), case1(t) {} // Some
    __device__ Union10(Union10 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union10_0(x.case0); break; // None
            case 1: new (&this->case1) Union10_1(x.case1); break; // Some
        }
    }
    __device__ Union10(Union10 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union10_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union10_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union10 & operator=(Union10 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union10();
            new (this) Union10{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union10() {
        switch(this->tag){
            case 0: this->case0.~Union10_0(); break; // None
            case 1: this->case1.~Union10_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union11_0 { // None
};
struct Union11_1 { // Some
    static_array<unsigned char,3l> v0;
    static_array<unsigned char,4l> v1;
    __device__ Union11_1(static_array<unsigned char,3l> t0, static_array<unsigned char,4l> t1) : v0(t0), v1(t1) {}
    __device__ Union11_1() = delete;
};
struct Union11 {
    union {
        Union11_0 case0; // None
        Union11_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union11() {}
    __device__ Union11(Union11_0 t) : tag(0), case0(t) {} // None
    __device__ Union11(Union11_1 t) : tag(1), case1(t) {} // Some
    __device__ Union11(Union11 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union11_0(x.case0); break; // None
            case 1: new (&this->case1) Union11_1(x.case1); break; // Some
        }
    }
    __device__ Union11(Union11 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union11_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union11_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union11 & operator=(Union11 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union11();
            new (this) Union11{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union11() {
        switch(this->tag){
            case 0: this->case0.~Union11_0(); break; // None
            case 1: this->case1.~Union11_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union12_0 { // None
};
struct Union12_1 { // Some
    static_array<unsigned char,2l> v0;
    static_array<unsigned char,2l> v1;
    __device__ Union12_1(static_array<unsigned char,2l> t0, static_array<unsigned char,2l> t1) : v0(t0), v1(t1) {}
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
struct Union13_0 { // None
};
struct Union13_1 { // Some
    static_array<unsigned char,2l> v0;
    static_array<unsigned char,5l> v1;
    __device__ Union13_1(static_array<unsigned char,2l> t0, static_array<unsigned char,5l> t1) : v0(t0), v1(t1) {}
    __device__ Union13_1() = delete;
};
struct Union13 {
    union {
        Union13_0 case0; // None
        Union13_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union13() {}
    __device__ Union13(Union13_0 t) : tag(0), case0(t) {} // None
    __device__ Union13(Union13_1 t) : tag(1), case1(t) {} // Some
    __device__ Union13(Union13 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union13_0(x.case0); break; // None
            case 1: new (&this->case1) Union13_1(x.case1); break; // Some
        }
    }
    __device__ Union13(Union13 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union13_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union13_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union13 & operator=(Union13 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union13();
            new (this) Union13{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union13() {
        switch(this->tag){
            case 0: this->case0.~Union13_0(); break; // None
            case 1: this->case1.~Union13_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union14_0 { // None
};
struct Union14_1 { // Some
    static_array<unsigned char,2l> v0;
    static_array<unsigned char,3l> v1;
    __device__ Union14_1(static_array<unsigned char,2l> t0, static_array<unsigned char,3l> t1) : v0(t0), v1(t1) {}
    __device__ Union14_1() = delete;
};
struct Union14 {
    union {
        Union14_0 case0; // None
        Union14_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union14() {}
    __device__ Union14(Union14_0 t) : tag(0), case0(t) {} // None
    __device__ Union14(Union14_1 t) : tag(1), case1(t) {} // Some
    __device__ Union14(Union14 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union14_0(x.case0); break; // None
            case 1: new (&this->case1) Union14_1(x.case1); break; // Some
        }
    }
    __device__ Union14(Union14 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union14_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union14_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union14 & operator=(Union14 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union14();
            new (this) Union14{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union14() {
        switch(this->tag){
            case 0: this->case0.~Union14_0(); break; // None
            case 1: this->case1.~Union14_1(); break; // Some
        }
        this->tag = 255;
    }
};
__device__ long f_1(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+0ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ void f_3(unsigned char * v0){
    return ;
}
__device__ Union1 f_2(unsigned char * v0){
    long v1;
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
        case 3: {
            long v7;
            v7 = f_1(v2);
            return Union1{Union1_3{v7}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ inline bool while_method_0(long v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
__device__ Union2 f_5(unsigned char * v0){
    long v1;
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
    long v2;
    v2 = 0l;
    while (while_method_0(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned long long v5;
        v5 = v4 * 4ull;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        Union2 v7;
        v7 = f_5(v6);
        v1[v2] = v7;
        v2 += 1l ;
    }
    return v1;
}
__device__ Union0 f_0(unsigned char * v0){
    long v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+8ull);
    switch (v1) {
        case 0: {
            Union1 v4;
            v4 = f_2(v2);
            return Union0{Union0_0{v4}};
            break;
        }
        case 1: {
            static_array<Union2,2l> v6;
            v6 = f_4(v2);
            return Union0{Union0_1{v6}};
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
__device__ unsigned long long f_7(unsigned char * v0){
    unsigned long long * v1;
    v1 = (unsigned long long *)(v0+0ull);
    unsigned long long v2;
    v2 = v1[0l];
    return v2;
}
__device__ long f_8(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+8ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ inline bool while_method_1(long v0, long v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ unsigned char f_12(unsigned char * v0){
    unsigned char * v1;
    v1 = (unsigned char *)(v0+0ull);
    unsigned char v2;
    v2 = v1[0l];
    return v2;
}
__device__ unsigned char f_11(unsigned char * v0){
    unsigned char v1;
    v1 = f_12(v0);
    return v1;
}
__device__ static_array_list<unsigned char,5l> f_10(unsigned char * v0){
    static_array_list<unsigned char,5l> v1;
    v1 = static_array_list<unsigned char,5l>{};
    long v2;
    v2 = f_1(v0);
    v1.unsafe_set_length(v2);
    long v3;
    v3 = v1.length;
    long v4;
    v4 = 0l;
    while (while_method_1(v3, v4)){
        unsigned long long v6;
        v6 = (unsigned long long)v4;
        unsigned long long v7;
        v7 = 4ull + v6;
        unsigned char * v8;
        v8 = (unsigned char *)(v0+v7);
        unsigned char v9;
        v9 = f_11(v8);
        v1[v4] = v9;
        v4 += 1l ;
    }
    return v1;
}
__device__ Tuple2 f_13(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+0ull);
    long v2;
    v2 = v1[0l];
    long * v3;
    v3 = (long *)(v0+4ull);
    long v4;
    v4 = v3[0l];
    return Tuple2{v2, v4};
}
__device__ long f_15(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+4ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ Tuple3 f_14(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+0ull);
    long v2;
    v2 = v1[0l];
    long v3;
    v3 = f_15(v0);
    unsigned char * v4;
    v4 = (unsigned char *)(v0+8ull);
    Union1 v11;
    switch (v3) {
        case 0: {
            f_3(v4);
            v11 = Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v4);
            v11 = Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v4);
            v11 = Union1{Union1_2{}};
            break;
        }
        case 3: {
            long v9;
            v9 = f_1(v4);
            v11 = Union1{Union1_3{v9}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple3{v2, v11};
}
__device__ Tuple4 f_16(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+0ull);
    long v2;
    v2 = v1[0l];
    static_array<unsigned char,2l> v3;
    long v4;
    v4 = 0l;
    while (while_method_0(v4)){
        unsigned long long v6;
        v6 = (unsigned long long)v4;
        unsigned long long v7;
        v7 = 4ull + v6;
        unsigned char * v8;
        v8 = (unsigned char *)(v0+v7);
        unsigned char v9;
        v9 = f_11(v8);
        v3[v4] = v9;
        v4 += 1l ;
    }
    return Tuple4{v2, v3};
}
__device__ inline bool while_method_2(long v0){
    bool v1;
    v1 = v0 < 5l;
    return v1;
}
__device__ Tuple0 f_19(unsigned char * v0){
    static_array<unsigned char,5l> v1;
    long v2;
    v2 = 0l;
    while (while_method_2(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v6;
        v6 = f_11(v5);
        v1[v2] = v6;
        v2 += 1l ;
    }
    char * v7;
    v7 = (char *)(v0+5ull);
    char v8;
    v8 = v7[0l];
    return Tuple0{v1, v8};
}
__device__ Tuple0 f_18(unsigned char * v0){
    static_array<unsigned char,5l> v1; char v2;
    Tuple0 tmp3 = f_19(v0);
    v1 = tmp3.v0; v2 = tmp3.v1;
    return Tuple0{v1, v2};
}
__device__ Tuple5 f_17(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+0ull);
    long v2;
    v2 = v1[0l];
    static_array<Tuple0,2l> v3;
    long v4;
    v4 = 0l;
    while (while_method_0(v4)){
        unsigned long long v6;
        v6 = (unsigned long long)v4;
        unsigned long long v7;
        v7 = v6 * 8ull;
        unsigned long long v8;
        v8 = 8ull + v7;
        unsigned char * v9;
        v9 = (unsigned char *)(v0+v8);
        static_array<unsigned char,5l> v10; char v11;
        Tuple0 tmp4 = f_18(v9);
        v10 = tmp4.v0; v11 = tmp4.v1;
        v3[v4] = Tuple0{v10, v11};
        v4 += 1l ;
    }
    long * v12;
    v12 = (long *)(v0+24ull);
    long v13;
    v13 = v12[0l];
    return Tuple5{v2, v3, v13};
}
__device__ Union3 f_9(unsigned char * v0){
    long v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            static_array_list<unsigned char,5l> v4;
            v4 = f_10(v2);
            return Union3{Union3_0{v4}};
            break;
        }
        case 1: {
            long v6; long v7;
            Tuple2 tmp0 = f_13(v2);
            v6 = tmp0.v0; v7 = tmp0.v1;
            return Union3{Union3_1{v6, v7}};
            break;
        }
        case 2: {
            long v9; Union1 v10;
            Tuple3 tmp1 = f_14(v2);
            v9 = tmp1.v0; v10 = tmp1.v1;
            return Union3{Union3_2{v9, v10}};
            break;
        }
        case 3: {
            long v12; static_array<unsigned char,2l> v13;
            Tuple4 tmp2 = f_16(v2);
            v12 = tmp2.v0; v13 = tmp2.v1;
            return Union3{Union3_3{v12, v13}};
            break;
        }
        case 4: {
            long v15; static_array<Tuple0,2l> v16; long v17;
            Tuple5 tmp5 = f_17(v2);
            v15 = tmp5.v0; v16 = tmp5.v1; v17 = tmp5.v2;
            return Union3{Union3_4{v15, v16, v17}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ long f_20(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+6160ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ static_array<unsigned char,2l> f_23(unsigned char * v0){
    static_array<unsigned char,2l> v1;
    long v2;
    v2 = 0l;
    while (while_method_0(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v6;
        v6 = f_11(v5);
        v1[v2] = v6;
        v2 += 1l ;
    }
    return v1;
}
__device__ long f_24(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+28ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ inline bool while_method_3(long v0){
    bool v1;
    v1 = v0 < 3l;
    return v1;
}
__device__ static_array<unsigned char,3l> f_25(unsigned char * v0){
    static_array<unsigned char,3l> v1;
    long v2;
    v2 = 0l;
    while (while_method_3(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v6;
        v6 = f_11(v5);
        v1[v2] = v6;
        v2 += 1l ;
    }
    return v1;
}
__device__ static_array<unsigned char,5l> f_26(unsigned char * v0){
    static_array<unsigned char,5l> v1;
    long v2;
    v2 = 0l;
    while (while_method_2(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v6;
        v6 = f_11(v5);
        v1[v2] = v6;
        v2 += 1l ;
    }
    return v1;
}
__device__ inline bool while_method_4(long v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ static_array<unsigned char,4l> f_27(unsigned char * v0){
    static_array<unsigned char,4l> v1;
    long v2;
    v2 = 0l;
    while (while_method_4(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v6;
        v6 = f_11(v5);
        v1[v2] = v6;
        v2 += 1l ;
    }
    return v1;
}
__device__ Tuple6 f_22(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+0ull);
    long v2;
    v2 = v1[0l];
    static_array<static_array<unsigned char,2l>,2l> v3;
    long v4;
    v4 = 0l;
    while (while_method_0(v4)){
        unsigned long long v6;
        v6 = (unsigned long long)v4;
        unsigned long long v7;
        v7 = v6 * 2ull;
        unsigned long long v8;
        v8 = 4ull + v7;
        unsigned char * v9;
        v9 = (unsigned char *)(v0+v8);
        static_array<unsigned char,2l> v10;
        v10 = f_23(v9);
        v3[v4] = v10;
        v4 += 1l ;
    }
    static_array<long,2l> v11;
    long v12;
    v12 = 0l;
    while (while_method_0(v12)){
        unsigned long long v14;
        v14 = (unsigned long long)v12;
        unsigned long long v15;
        v15 = v14 * 4ull;
        unsigned long long v16;
        v16 = 8ull + v15;
        unsigned char * v17;
        v17 = (unsigned char *)(v0+v16);
        long v18;
        v18 = f_1(v17);
        v11[v12] = v18;
        v12 += 1l ;
    }
    long * v19;
    v19 = (long *)(v0+16ull);
    long v20;
    v20 = v19[0l];
    static_array<long,2l> v21;
    long v22;
    v22 = 0l;
    while (while_method_0(v22)){
        unsigned long long v24;
        v24 = (unsigned long long)v22;
        unsigned long long v25;
        v25 = v24 * 4ull;
        unsigned long long v26;
        v26 = 20ull + v25;
        unsigned char * v27;
        v27 = (unsigned char *)(v0+v26);
        long v28;
        v28 = f_1(v27);
        v21[v22] = v28;
        v22 += 1l ;
    }
    long v29;
    v29 = f_24(v0);
    unsigned char * v30;
    v30 = (unsigned char *)(v0+32ull);
    Union6 v39;
    switch (v29) {
        case 0: {
            static_array<unsigned char,3l> v32;
            v32 = f_25(v30);
            v39 = Union6{Union6_0{v32}};
            break;
        }
        case 1: {
            f_3(v30);
            v39 = Union6{Union6_1{}};
            break;
        }
        case 2: {
            static_array<unsigned char,5l> v35;
            v35 = f_26(v30);
            v39 = Union6{Union6_2{v35}};
            break;
        }
        case 3: {
            static_array<unsigned char,4l> v37;
            v37 = f_27(v30);
            v39 = Union6{Union6_3{v37}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple6{v2, v3, v11, v20, v21, v39};
}
__device__ long f_29(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+40ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ Tuple7 f_28(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+0ull);
    long v2;
    v2 = v1[0l];
    static_array<static_array<unsigned char,2l>,2l> v3;
    long v4;
    v4 = 0l;
    while (while_method_0(v4)){
        unsigned long long v6;
        v6 = (unsigned long long)v4;
        unsigned long long v7;
        v7 = v6 * 2ull;
        unsigned long long v8;
        v8 = 4ull + v7;
        unsigned char * v9;
        v9 = (unsigned char *)(v0+v8);
        static_array<unsigned char,2l> v10;
        v10 = f_23(v9);
        v3[v4] = v10;
        v4 += 1l ;
    }
    static_array<long,2l> v11;
    long v12;
    v12 = 0l;
    while (while_method_0(v12)){
        unsigned long long v14;
        v14 = (unsigned long long)v12;
        unsigned long long v15;
        v15 = v14 * 4ull;
        unsigned long long v16;
        v16 = 8ull + v15;
        unsigned char * v17;
        v17 = (unsigned char *)(v0+v16);
        long v18;
        v18 = f_1(v17);
        v11[v12] = v18;
        v12 += 1l ;
    }
    long * v19;
    v19 = (long *)(v0+16ull);
    long v20;
    v20 = v19[0l];
    static_array<long,2l> v21;
    long v22;
    v22 = 0l;
    while (while_method_0(v22)){
        unsigned long long v24;
        v24 = (unsigned long long)v22;
        unsigned long long v25;
        v25 = v24 * 4ull;
        unsigned long long v26;
        v26 = 20ull + v25;
        unsigned char * v27;
        v27 = (unsigned char *)(v0+v26);
        long v28;
        v28 = f_1(v27);
        v21[v22] = v28;
        v22 += 1l ;
    }
    long v29;
    v29 = f_24(v0);
    unsigned char * v30;
    v30 = (unsigned char *)(v0+32ull);
    Union6 v39;
    switch (v29) {
        case 0: {
            static_array<unsigned char,3l> v32;
            v32 = f_25(v30);
            v39 = Union6{Union6_0{v32}};
            break;
        }
        case 1: {
            f_3(v30);
            v39 = Union6{Union6_1{}};
            break;
        }
        case 2: {
            static_array<unsigned char,5l> v35;
            v35 = f_26(v30);
            v39 = Union6{Union6_2{v35}};
            break;
        }
        case 3: {
            static_array<unsigned char,4l> v37;
            v37 = f_27(v30);
            v39 = Union6{Union6_3{v37}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    long v40;
    v40 = f_29(v0);
    unsigned char * v41;
    v41 = (unsigned char *)(v0+44ull);
    Union1 v48;
    switch (v40) {
        case 0: {
            f_3(v41);
            v48 = Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v41);
            v48 = Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v41);
            v48 = Union1{Union1_2{}};
            break;
        }
        case 3: {
            long v46;
            v46 = f_1(v41);
            v48 = Union1{Union1_3{v46}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple7{v2, v3, v11, v20, v21, v39, v48};
}
__device__ Union5 f_21(unsigned char * v0){
    long v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            long v4; static_array<static_array<unsigned char,2l>,2l> v5; static_array<long,2l> v6; long v7; static_array<long,2l> v8; Union6 v9;
            Tuple6 tmp6 = f_22(v2);
            v4 = tmp6.v0; v5 = tmp6.v1; v6 = tmp6.v2; v7 = tmp6.v3; v8 = tmp6.v4; v9 = tmp6.v5;
            return Union5{Union5_0{v4, v5, v6, v7, v8, v9}};
            break;
        }
        case 1: {
            long v11; static_array<static_array<unsigned char,2l>,2l> v12; static_array<long,2l> v13; long v14; static_array<long,2l> v15; Union6 v16;
            Tuple6 tmp7 = f_22(v2);
            v11 = tmp7.v0; v12 = tmp7.v1; v13 = tmp7.v2; v14 = tmp7.v3; v15 = tmp7.v4; v16 = tmp7.v5;
            return Union5{Union5_1{v11, v12, v13, v14, v15, v16}};
            break;
        }
        case 2: {
            f_3(v2);
            return Union5{Union5_2{}};
            break;
        }
        case 3: {
            long v19; static_array<static_array<unsigned char,2l>,2l> v20; static_array<long,2l> v21; long v22; static_array<long,2l> v23; Union6 v24;
            Tuple6 tmp8 = f_22(v2);
            v19 = tmp8.v0; v20 = tmp8.v1; v21 = tmp8.v2; v22 = tmp8.v3; v23 = tmp8.v4; v24 = tmp8.v5;
            return Union5{Union5_3{v19, v20, v21, v22, v23, v24}};
            break;
        }
        case 4: {
            long v26; static_array<static_array<unsigned char,2l>,2l> v27; static_array<long,2l> v28; long v29; static_array<long,2l> v30; Union6 v31;
            Tuple6 tmp9 = f_22(v2);
            v26 = tmp9.v0; v27 = tmp9.v1; v28 = tmp9.v2; v29 = tmp9.v3; v30 = tmp9.v4; v31 = tmp9.v5;
            return Union5{Union5_4{v26, v27, v28, v29, v30, v31}};
            break;
        }
        case 5: {
            long v33; static_array<static_array<unsigned char,2l>,2l> v34; static_array<long,2l> v35; long v36; static_array<long,2l> v37; Union6 v38; Union1 v39;
            Tuple7 tmp10 = f_28(v2);
            v33 = tmp10.v0; v34 = tmp10.v1; v35 = tmp10.v2; v36 = tmp10.v3; v37 = tmp10.v4; v38 = tmp10.v5; v39 = tmp10.v6;
            return Union5{Union5_5{v33, v34, v35, v36, v37, v38, v39}};
            break;
        }
        case 6: {
            long v41; static_array<static_array<unsigned char,2l>,2l> v42; static_array<long,2l> v43; long v44; static_array<long,2l> v45; Union6 v46;
            Tuple6 tmp11 = f_22(v2);
            v41 = tmp11.v0; v42 = tmp11.v1; v43 = tmp11.v2; v44 = tmp11.v3; v45 = tmp11.v4; v46 = tmp11.v5;
            return Union5{Union5_6{v41, v42, v43, v44, v45, v46}};
            break;
        }
        case 7: {
            long v48; static_array<static_array<unsigned char,2l>,2l> v49; static_array<long,2l> v50; long v51; static_array<long,2l> v52; Union6 v53;
            Tuple6 tmp12 = f_22(v2);
            v48 = tmp12.v0; v49 = tmp12.v1; v50 = tmp12.v2; v51 = tmp12.v3; v52 = tmp12.v4; v53 = tmp12.v5;
            return Union5{Union5_7{v48, v49, v50, v51, v52, v53}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ long f_30(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+6248ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ Tuple1 f_6(unsigned char * v0){
    unsigned long long v1;
    v1 = f_7(v0);
    dynamic_array_list<Union3,128l> v2{0};
    long v3;
    v3 = f_8(v0);
    v2.unsafe_set_length(v3);
    long v4;
    v4 = v2.length_();
    long v5;
    v5 = 0l;
    while (while_method_1(v4, v5)){
        unsigned long long v7;
        v7 = (unsigned long long)v5;
        unsigned long long v8;
        v8 = v7 * 48ull;
        unsigned long long v9;
        v9 = 16ull + v8;
        unsigned char * v10;
        v10 = (unsigned char *)(v0+v9);
        Union3 v11;
        v11 = f_9(v10);
        v2[v5] = v11;
        v5 += 1l ;
    }
    long v12;
    v12 = f_20(v0);
    unsigned char * v13;
    v13 = (unsigned char *)(v0+6176ull);
    Union4 v18;
    switch (v12) {
        case 0: {
            f_3(v13);
            v18 = Union4{Union4_0{}};
            break;
        }
        case 1: {
            Union5 v16;
            v16 = f_21(v13);
            v18 = Union4{Union4_1{v16}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    static_array<Union2,2l> v19;
    long v20;
    v20 = 0l;
    while (while_method_0(v20)){
        unsigned long long v22;
        v22 = (unsigned long long)v20;
        unsigned long long v23;
        v23 = v22 * 4ull;
        unsigned long long v24;
        v24 = 6240ull + v23;
        unsigned char * v25;
        v25 = (unsigned char *)(v0+v24);
        Union2 v26;
        v26 = f_5(v25);
        v19[v20] = v26;
        v20 += 1l ;
    }
    long v27;
    v27 = f_30(v0);
    unsigned char * v28;
    v28 = (unsigned char *)(v0+6256ull);
    Union7 v45;
    switch (v27) {
        case 0: {
            f_3(v28);
            v45 = Union7{Union7_0{}};
            break;
        }
        case 1: {
            long v31; static_array<static_array<unsigned char,2l>,2l> v32; static_array<long,2l> v33; long v34; static_array<long,2l> v35; Union6 v36;
            Tuple6 tmp13 = f_22(v28);
            v31 = tmp13.v0; v32 = tmp13.v1; v33 = tmp13.v2; v34 = tmp13.v3; v35 = tmp13.v4; v36 = tmp13.v5;
            v45 = Union7{Union7_1{v31, v32, v33, v34, v35, v36}};
            break;
        }
        case 2: {
            long v38; static_array<static_array<unsigned char,2l>,2l> v39; static_array<long,2l> v40; long v41; static_array<long,2l> v42; Union6 v43;
            Tuple6 tmp14 = f_22(v28);
            v38 = tmp14.v0; v39 = tmp14.v1; v40 = tmp14.v2; v41 = tmp14.v3; v42 = tmp14.v4; v43 = tmp14.v5;
            v45 = Union7{Union7_2{v38, v39, v40, v41, v42, v43}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple1{v1, v2, v18, v19, v45};
}
__device__ inline bool while_method_5(bool v0, Union5 v1){
    return v0;
}
__device__ unsigned long loop_35(unsigned long v0, curandStatePhilox4_32_10_t & v1){
    unsigned long v2;
    v2 = curand(&v1);
    unsigned long v3;
    v3 = v2 % v0;
    unsigned long v4;
    v4 = v2 - v3;
    unsigned long v5;
    v5 = 0ul - v0;
    bool v6;
    v6 = v4 <= v5;
    if (v6){
        return v3;
    } else {
        return loop_35(v0, v1);
    }
}
__device__ Tuple12 draw_card_34(curandStatePhilox4_32_10_t & v0, unsigned long long v1){
    long v2;
    v2 = __popcll(v1);
    unsigned long v3;
    v3 = (unsigned long)v2;
    unsigned long v4;
    v4 = loop_35(v3, v0);
    long v5;
    v5 = (long)v4;
    unsigned long v6;
    v6 = (unsigned long)v1;
    unsigned long long v7;
    v7 = v1 >> 32l;
    unsigned long v8;
    v8 = (unsigned long)v7;
    long v9;
    v9 = __popc(v6);
    bool v10;
    v10 = v5 < v9;
    unsigned long v17;
    if (v10){
        long v11;
        v11 = v5 + 1l;
        unsigned long v12;
        v12 = __fns(v6,0ul,v11);
        v17 = v12;
    } else {
        long v13;
        v13 = v5 - v9;
        long v14;
        v14 = v13 + 1l;
        unsigned long v15;
        v15 = __fns(v8,0ul,v14);
        unsigned long v16;
        v16 = v15 + 32ul;
        v17 = v16;
    }
    unsigned char v18;
    v18 = (unsigned char)v17;
    long v19;
    v19 = (long)v17;
    unsigned long long v20;
    v20 = 1ull << v19;
    unsigned long long v21;
    v21 = v1 ^ v20;
    return Tuple12{v18, v21};
}
__device__ Tuple10 draw_cards_33(curandStatePhilox4_32_10_t & v0, unsigned long long v1){
    static_array<unsigned char,3l> v2;
    long v3; unsigned long long v4;
    Tuple11 tmp17 = Tuple11{0l, v1};
    v3 = tmp17.v0; v4 = tmp17.v1;
    while (while_method_3(v3)){
        unsigned char v6; unsigned long long v7;
        Tuple12 tmp18 = draw_card_34(v0, v4);
        v6 = tmp18.v0; v7 = tmp18.v1;
        v2[v3] = v6;
        v4 = v7;
        v3 += 1l ;
    }
    return Tuple10{v2, v4};
}
__device__ static_array_list<unsigned char,5l> get_community_cards_36(Union6 v0, static_array<unsigned char,3l> v1){
    static_array_list<unsigned char,5l> v2;
    v2 = static_array_list<unsigned char,5l>{};
    switch (v0.tag) {
        case 0: { // Flop
            static_array<unsigned char,3l> v3 = v0.case0.v0;
            long v4;
            v4 = 0l;
            while (while_method_3(v4)){
                unsigned char v6;
                v6 = v3[v4];
                v2.push(v6);
                v4 += 1l ;
            }
            break;
        }
        case 1: { // Preflop
            break;
        }
        case 2: { // River
            static_array<unsigned char,5l> v11 = v0.case2.v0;
            long v12;
            v12 = 0l;
            while (while_method_2(v12)){
                unsigned char v14;
                v14 = v11[v12];
                v2.push(v14);
                v12 += 1l ;
            }
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4l> v7 = v0.case3.v0;
            long v8;
            v8 = 0l;
            while (while_method_4(v8)){
                unsigned char v10;
                v10 = v7[v8];
                v2.push(v10);
                v8 += 1l ;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    long v15;
    v15 = 0l;
    while (while_method_3(v15)){
        unsigned char v17;
        v17 = v1[v15];
        v2.push(v17);
        v15 += 1l ;
    }
    return v2;
}
__device__ bool player_can_act_38(long v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<long,2l> v2, long v3, static_array<long,2l> v4, Union6 v5){
    long v6;
    v6 = v3 % 2l;
    long v7;
    v7 = v4[v6];
    bool v8;
    v8 = v7 > 0l;
    long v9;
    v9 = v2[v6];
    long v10;
    v10 = v2[0l];
    long v11; long v12;
    Tuple2 tmp20 = Tuple2{1l, v10};
    v11 = tmp20.v0; v12 = tmp20.v1;
    while (while_method_0(v11)){
        long v14;
        v14 = v2[v11];
        bool v15;
        v15 = v12 >= v14;
        long v16;
        if (v15){
            v16 = v12;
        } else {
            v16 = v14;
        }
        v12 = v16;
        v11 += 1l ;
    }
    bool v17;
    v17 = v9 < v12;
    long v18; long v19;
    Tuple2 tmp21 = Tuple2{0l, 0l};
    v18 = tmp21.v0; v19 = tmp21.v1;
    while (while_method_0(v18)){
        long v21;
        v21 = v4[v18];
        bool v22;
        v22 = 0l < v21;
        long v23;
        if (v22){
            v23 = 1l;
        } else {
            v23 = 0l;
        }
        long v24;
        v24 = v19 + v23;
        v19 = v24;
        v18 += 1l ;
    }
    if (v8){
        if (v17){
            return true;
        } else {
            bool v25;
            v25 = v3 < 2l;
            if (v25){
                bool v26;
                v26 = 0l < v19;
                return v26;
            } else {
                return false;
            }
        }
    } else {
        return false;
    }
}
__device__ Union5 go_next_street_39(long v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<long,2l> v2, long v3, static_array<long,2l> v4, Union6 v5){
    switch (v5.tag) {
        case 0: { // Flop
            static_array<unsigned char,3l> v7 = v5.case0.v0;
            return Union5{Union5_7{v0, v1, v2, v3, v4, v5}};
            break;
        }
        case 1: { // Preflop
            return Union5{Union5_0{v0, v1, v2, v3, v4, v5}};
            break;
        }
        case 2: { // River
            static_array<unsigned char,5l> v11 = v5.case2.v0;
            return Union5{Union5_6{v0, v1, v2, v3, v4, v5}};
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4l> v9 = v5.case3.v0;
            return Union5{Union5_3{v0, v1, v2, v3, v4, v5}};
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ Union5 try_round_37(long v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<long,2l> v2, long v3, static_array<long,2l> v4, Union6 v5){
    long v6;
    v6 = v3 + 1l;
    bool v7;
    v7 = player_can_act_38(v0, v1, v2, v3, v4, v5);
    if (v7){
        return Union5{Union5_4{v0, v1, v2, v3, v4, v5}};
    } else {
        bool v9;
        v9 = player_can_act_38(v0, v1, v2, v6, v4, v5);
        if (v9){
            return Union5{Union5_4{v0, v1, v2, v6, v4, v5}};
        } else {
            return go_next_street_39(v0, v1, v2, v3, v4, v5);
        }
    }
}
__device__ Tuple13 draw_cards_40(curandStatePhilox4_32_10_t & v0, unsigned long long v1){
    static_array<unsigned char,2l> v2;
    long v3; unsigned long long v4;
    Tuple11 tmp22 = Tuple11{0l, v1};
    v3 = tmp22.v0; v4 = tmp22.v1;
    while (while_method_0(v3)){
        unsigned char v6; unsigned long long v7;
        Tuple12 tmp23 = draw_card_34(v0, v4);
        v6 = tmp23.v0; v7 = tmp23.v1;
        v2[v3] = v6;
        v4 = v7;
        v3 += 1l ;
    }
    return Tuple13{v2, v4};
}
__device__ inline bool while_method_6(long v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ Tuple14 draw_cards_41(curandStatePhilox4_32_10_t & v0, unsigned long long v1){
    static_array<unsigned char,1l> v2;
    long v3; unsigned long long v4;
    Tuple11 tmp26 = Tuple11{0l, v1};
    v3 = tmp26.v0; v4 = tmp26.v1;
    while (while_method_6(v3)){
        unsigned char v6; unsigned long long v7;
        Tuple12 tmp27 = draw_card_34(v0, v4);
        v6 = tmp27.v0; v7 = tmp27.v1;
        v2[v3] = v6;
        v4 = v7;
        v3 += 1l ;
    }
    return Tuple14{v2, v4};
}
__device__ static_array_list<unsigned char,5l> get_community_cards_42(Union6 v0, static_array<unsigned char,1l> v1){
    static_array_list<unsigned char,5l> v2;
    v2 = static_array_list<unsigned char,5l>{};
    switch (v0.tag) {
        case 0: { // Flop
            static_array<unsigned char,3l> v3 = v0.case0.v0;
            long v4;
            v4 = 0l;
            while (while_method_3(v4)){
                unsigned char v6;
                v6 = v3[v4];
                v2.push(v6);
                v4 += 1l ;
            }
            break;
        }
        case 1: { // Preflop
            break;
        }
        case 2: { // River
            static_array<unsigned char,5l> v11 = v0.case2.v0;
            long v12;
            v12 = 0l;
            while (while_method_2(v12)){
                unsigned char v14;
                v14 = v11[v12];
                v2.push(v14);
                v12 += 1l ;
            }
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4l> v7 = v0.case3.v0;
            long v8;
            v8 = 0l;
            while (while_method_4(v8)){
                unsigned char v10;
                v10 = v7[v8];
                v2.push(v10);
                v8 += 1l ;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    long v15;
    v15 = 0l;
    while (while_method_6(v15)){
        unsigned char v17;
        v17 = v1[v15];
        v2.push(v17);
        v15 += 1l ;
    }
    return v2;
}
__device__ inline bool while_method_7(long v0){
    bool v1;
    v1 = v0 < 6l;
    return v1;
}
__device__ inline bool while_method_8(static_array<float,6l> v0, long v1){
    bool v2;
    v2 = v1 < 6l;
    return v2;
}
__device__ inline bool while_method_9(long v0, long v1){
    bool v2;
    v2 = v1 > v0;
    return v2;
}
__device__ long loop_45(static_array<float,6l> v0, float v1, long v2){
    bool v3;
    v3 = v2 < 6l;
    if (v3){
        float v4;
        v4 = v0[v2];
        bool v5;
        v5 = v1 <= v4;
        if (v5){
            return v2;
        } else {
            long v6;
            v6 = v2 + 1l;
            return loop_45(v0, v1, v6);
        }
    } else {
        return 5l;
    }
}
__device__ long sample_discrete__44(static_array<float,6l> v0, curandStatePhilox4_32_10_t & v1){
    static_array<float,6l> v2;
    long v3;
    v3 = 0l;
    while (while_method_7(v3)){
        float v5;
        v5 = v0[v3];
        v2[v3] = v5;
        v3 += 1l ;
    }
    long v6;
    v6 = 1l;
    while (while_method_8(v2, v6)){
        long v8;
        v8 = 6l;
        while (while_method_9(v6, v8)){
            v8 -= 1l ;
            long v10;
            v10 = v8 - v6;
            float v11;
            v11 = v2[v10];
            float v12;
            v12 = v2[v8];
            float v13;
            v13 = v11 + v12;
            v2[v8] = v13;
        }
        long v14;
        v14 = v6 * 2l;
        v6 = v14;
    }
    float v15;
    v15 = v2[5l];
    float v16;
    v16 = curand_uniform(&v1);
    float v17;
    v17 = v16 * v15;
    long v18;
    v18 = 0l;
    return loop_45(v2, v17, v18);
}
__device__ Union1 sample_discrete_43(static_array<Tuple15,6l> v0, curandStatePhilox4_32_10_t & v1){
    static_array<float,6l> v2;
    long v3;
    v3 = 0l;
    while (while_method_7(v3)){
        Union1 v5; float v6;
        Tuple15 tmp31 = v0[v3];
        v5 = tmp31.v0; v6 = tmp31.v1;
        v2[v3] = v6;
        v3 += 1l ;
    }
    long v7;
    v7 = sample_discrete__44(v2, v1);
    Union1 v8; float v9;
    Tuple15 tmp32 = v0[v7];
    v8 = tmp32.v0; v9 = tmp32.v1;
    return v8;
}
__device__ inline bool while_method_10(long v0){
    bool v1;
    v1 = v0 < 7l;
    return v1;
}
__device__ inline bool while_method_11(static_array<unsigned char,7l> v0, bool v1, long v2){
    bool v3;
    v3 = v2 < 7l;
    return v3;
}
__device__ inline bool while_method_12(static_array<unsigned char,7l> v0, long v1){
    bool v2;
    v2 = v1 < 7l;
    return v2;
}
__device__ inline bool while_method_13(long v0, long v1, long v2, long v3){
    bool v4;
    v4 = v3 < v0;
    return v4;
}
__device__ Tuple0 score_46(static_array<unsigned char,7l> v0){
    static_array<unsigned char,7l> v1;
    long v2;
    v2 = 0l;
    while (while_method_10(v2)){
        unsigned char v4;
        v4 = v0[v2];
        v1[v2] = v4;
        v2 += 1l ;
    }
    static_array<unsigned char,7l> v5;
    bool v6; long v7;
    Tuple16 tmp39 = Tuple16{true, 1l};
    v6 = tmp39.v0; v7 = tmp39.v1;
    while (while_method_11(v1, v6, v7)){
        long v9;
        v9 = 0l;
        while (while_method_12(v1, v9)){
            long v11;
            v11 = v9 + v7;
            bool v12;
            v12 = v11 < 7l;
            long v13;
            if (v12){
                v13 = v11;
            } else {
                v13 = 7l;
            }
            long v14;
            v14 = v7 * 2l;
            long v15;
            v15 = v9 + v14;
            bool v16;
            v16 = v15 < 7l;
            long v17;
            if (v16){
                v17 = v15;
            } else {
                v17 = 7l;
            }
            long v18; long v19; long v20;
            Tuple17 tmp40 = Tuple17{v9, v13, v9};
            v18 = tmp40.v0; v19 = tmp40.v1; v20 = tmp40.v2;
            while (while_method_13(v17, v18, v19, v20)){
                bool v22;
                v22 = v18 < v13;
                bool v24;
                if (v22){
                    bool v23;
                    v23 = v19 < v17;
                    v24 = v23;
                } else {
                    v24 = false;
                }
                unsigned char v66; long v67; long v68;
                if (v24){
                    unsigned char v27;
                    if (v6){
                        unsigned char v25;
                        v25 = v1[v18];
                        v27 = v25;
                    } else {
                        unsigned char v26;
                        v26 = v5[v18];
                        v27 = v26;
                    }
                    unsigned char v30;
                    if (v6){
                        unsigned char v28;
                        v28 = v1[v19];
                        v30 = v28;
                    } else {
                        unsigned char v29;
                        v29 = v5[v19];
                        v30 = v29;
                    }
                    unsigned char v31;
                    v31 = v30 / 4u;
                    unsigned char v32;
                    v32 = v27 / 4u;
                    bool v33;
                    v33 = v31 < v32;
                    Union8 v39;
                    if (v33){
                        v39 = Union8{Union8_2{}};
                    } else {
                        bool v35;
                        v35 = v31 > v32;
                        if (v35){
                            v39 = Union8{Union8_1{}};
                        } else {
                            v39 = Union8{Union8_0{}};
                        }
                    }
                    Union8 v49;
                    switch (v39.tag) {
                        case 0: { // Eq
                            unsigned char v40;
                            v40 = v27 % 4u;
                            unsigned char v41;
                            v41 = v30 % 4u;
                            bool v42;
                            v42 = v40 < v41;
                            if (v42){
                                v49 = Union8{Union8_2{}};
                            } else {
                                bool v44;
                                v44 = v40 > v41;
                                if (v44){
                                    v49 = Union8{Union8_1{}};
                                } else {
                                    v49 = Union8{Union8_0{}};
                                }
                            }
                            break;
                        }
                        default: {
                            v49 = v39;
                        }
                    }
                    switch (v49.tag) {
                        case 1: { // Gt
                            long v50;
                            v50 = v19 + 1l;
                            v66 = v30; v67 = v18; v68 = v50;
                            break;
                        }
                        default: {
                            long v51;
                            v51 = v18 + 1l;
                            v66 = v27; v67 = v51; v68 = v19;
                        }
                    }
                } else {
                    if (v22){
                        unsigned char v57;
                        if (v6){
                            unsigned char v55;
                            v55 = v1[v18];
                            v57 = v55;
                        } else {
                            unsigned char v56;
                            v56 = v5[v18];
                            v57 = v56;
                        }
                        long v58;
                        v58 = v18 + 1l;
                        v66 = v57; v67 = v58; v68 = v19;
                    } else {
                        unsigned char v61;
                        if (v6){
                            unsigned char v59;
                            v59 = v1[v19];
                            v61 = v59;
                        } else {
                            unsigned char v60;
                            v60 = v5[v19];
                            v61 = v60;
                        }
                        long v62;
                        v62 = v19 + 1l;
                        v66 = v61; v67 = v18; v68 = v62;
                    }
                }
                if (v6){
                    v5[v20] = v66;
                } else {
                    v1[v20] = v66;
                }
                long v69;
                v69 = v20 + 1l;
                v18 = v67;
                v19 = v68;
                v20 = v69;
            }
            v9 = v15;
        }
        bool v70;
        v70 = v6 == false;
        long v71;
        v71 = v7 * 2l;
        v6 = v70;
        v7 = v71;
    }
    bool v72;
    v72 = v6 == false;
    static_array<unsigned char,7l> v73;
    if (v72){
        v73 = v5;
    } else {
        v73 = v1;
    }
    static_array<unsigned char,5l> v74;
    long v75; long v76; unsigned char v77;
    Tuple18 tmp41 = Tuple18{0l, 0l, 12u};
    v75 = tmp41.v0; v76 = tmp41.v1; v77 = tmp41.v2;
    while (while_method_10(v75)){
        unsigned char v79;
        v79 = v73[v75];
        bool v80;
        v80 = v76 < 5l;
        long v92; unsigned char v93;
        if (v80){
            unsigned char v81;
            v81 = v79 % 4u;
            bool v82;
            v82 = 0u == v81;
            if (v82){
                unsigned char v83;
                v83 = v79 / 4u;
                bool v84;
                v84 = v77 == v83;
                long v85;
                if (v84){
                    v85 = v76;
                } else {
                    v85 = 0l;
                }
                v74[v85] = v79;
                long v86;
                v86 = v85 + 1l;
                unsigned char v87;
                v87 = v83 - 1u;
                v92 = v86; v93 = v87;
            } else {
                v92 = v76; v93 = v77;
            }
        } else {
            break;
        }
        v76 = v92;
        v77 = v93;
        v75 += 1l ;
    }
    bool v94;
    v94 = v76 == 4l;
    bool v129;
    if (v94){
        unsigned char v95;
        v95 = v77 + 1u;
        bool v96;
        v96 = v95 == 0u;
        if (v96){
            unsigned char v97;
            v97 = v73[0l];
            unsigned char v98;
            v98 = v97 % 4u;
            bool v99;
            v99 = 0u == v98;
            bool v103;
            if (v99){
                unsigned char v100;
                v100 = v97 / 4u;
                bool v101;
                v101 = v100 == 12u;
                if (v101){
                    v74[4l] = v97;
                    v103 = true;
                } else {
                    v103 = false;
                }
            } else {
                v103 = false;
            }
            if (v103){
                v129 = true;
            } else {
                unsigned char v104;
                v104 = v73[1l];
                unsigned char v105;
                v105 = v104 % 4u;
                bool v106;
                v106 = 0u == v105;
                bool v110;
                if (v106){
                    unsigned char v107;
                    v107 = v104 / 4u;
                    bool v108;
                    v108 = v107 == 12u;
                    if (v108){
                        v74[4l] = v104;
                        v110 = true;
                    } else {
                        v110 = false;
                    }
                } else {
                    v110 = false;
                }
                if (v110){
                    v129 = true;
                } else {
                    unsigned char v111;
                    v111 = v73[2l];
                    unsigned char v112;
                    v112 = v111 % 4u;
                    bool v113;
                    v113 = 0u == v112;
                    bool v117;
                    if (v113){
                        unsigned char v114;
                        v114 = v111 / 4u;
                        bool v115;
                        v115 = v114 == 12u;
                        if (v115){
                            v74[4l] = v111;
                            v117 = true;
                        } else {
                            v117 = false;
                        }
                    } else {
                        v117 = false;
                    }
                    if (v117){
                        v129 = true;
                    } else {
                        unsigned char v118;
                        v118 = v73[3l];
                        unsigned char v119;
                        v119 = v118 % 4u;
                        bool v120;
                        v120 = 0u == v119;
                        if (v120){
                            unsigned char v121;
                            v121 = v118 / 4u;
                            bool v122;
                            v122 = v121 == 12u;
                            if (v122){
                                v74[4l] = v118;
                                v129 = true;
                            } else {
                                v129 = false;
                            }
                        } else {
                            v129 = false;
                        }
                    }
                }
            }
        } else {
            v129 = false;
        }
    } else {
        v129 = false;
    }
    Union9 v135;
    if (v129){
        v135 = Union9{Union9_1{v74}};
    } else {
        bool v131;
        v131 = v76 == 5l;
        if (v131){
            v135 = Union9{Union9_1{v74}};
        } else {
            v135 = Union9{Union9_0{}};
        }
    }
    static_array<unsigned char,5l> v136;
    long v137; long v138; unsigned char v139;
    Tuple18 tmp42 = Tuple18{0l, 0l, 12u};
    v137 = tmp42.v0; v138 = tmp42.v1; v139 = tmp42.v2;
    while (while_method_10(v137)){
        unsigned char v141;
        v141 = v73[v137];
        bool v142;
        v142 = v138 < 5l;
        long v154; unsigned char v155;
        if (v142){
            unsigned char v143;
            v143 = v141 % 4u;
            bool v144;
            v144 = 1u == v143;
            if (v144){
                unsigned char v145;
                v145 = v141 / 4u;
                bool v146;
                v146 = v139 == v145;
                long v147;
                if (v146){
                    v147 = v138;
                } else {
                    v147 = 0l;
                }
                v136[v147] = v141;
                long v148;
                v148 = v147 + 1l;
                unsigned char v149;
                v149 = v145 - 1u;
                v154 = v148; v155 = v149;
            } else {
                v154 = v138; v155 = v139;
            }
        } else {
            break;
        }
        v138 = v154;
        v139 = v155;
        v137 += 1l ;
    }
    bool v156;
    v156 = v138 == 4l;
    bool v191;
    if (v156){
        unsigned char v157;
        v157 = v139 + 1u;
        bool v158;
        v158 = v157 == 0u;
        if (v158){
            unsigned char v159;
            v159 = v73[0l];
            unsigned char v160;
            v160 = v159 % 4u;
            bool v161;
            v161 = 1u == v160;
            bool v165;
            if (v161){
                unsigned char v162;
                v162 = v159 / 4u;
                bool v163;
                v163 = v162 == 12u;
                if (v163){
                    v136[4l] = v159;
                    v165 = true;
                } else {
                    v165 = false;
                }
            } else {
                v165 = false;
            }
            if (v165){
                v191 = true;
            } else {
                unsigned char v166;
                v166 = v73[1l];
                unsigned char v167;
                v167 = v166 % 4u;
                bool v168;
                v168 = 1u == v167;
                bool v172;
                if (v168){
                    unsigned char v169;
                    v169 = v166 / 4u;
                    bool v170;
                    v170 = v169 == 12u;
                    if (v170){
                        v136[4l] = v166;
                        v172 = true;
                    } else {
                        v172 = false;
                    }
                } else {
                    v172 = false;
                }
                if (v172){
                    v191 = true;
                } else {
                    unsigned char v173;
                    v173 = v73[2l];
                    unsigned char v174;
                    v174 = v173 % 4u;
                    bool v175;
                    v175 = 1u == v174;
                    bool v179;
                    if (v175){
                        unsigned char v176;
                        v176 = v173 / 4u;
                        bool v177;
                        v177 = v176 == 12u;
                        if (v177){
                            v136[4l] = v173;
                            v179 = true;
                        } else {
                            v179 = false;
                        }
                    } else {
                        v179 = false;
                    }
                    if (v179){
                        v191 = true;
                    } else {
                        unsigned char v180;
                        v180 = v73[3l];
                        unsigned char v181;
                        v181 = v180 % 4u;
                        bool v182;
                        v182 = 1u == v181;
                        if (v182){
                            unsigned char v183;
                            v183 = v180 / 4u;
                            bool v184;
                            v184 = v183 == 12u;
                            if (v184){
                                v136[4l] = v180;
                                v191 = true;
                            } else {
                                v191 = false;
                            }
                        } else {
                            v191 = false;
                        }
                    }
                }
            }
        } else {
            v191 = false;
        }
    } else {
        v191 = false;
    }
    Union9 v197;
    if (v191){
        v197 = Union9{Union9_1{v136}};
    } else {
        bool v193;
        v193 = v138 == 5l;
        if (v193){
            v197 = Union9{Union9_1{v136}};
        } else {
            v197 = Union9{Union9_0{}};
        }
    }
    Union9 v223;
    switch (v135.tag) {
        case 0: { // None
            v223 = v197;
            break;
        }
        case 1: { // Some
            static_array<unsigned char,5l> v198 = v135.case1.v0;
            switch (v197.tag) {
                case 0: { // None
                    v223 = v135;
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5l> v199 = v197.case1.v0;
                    Union8 v200;
                    v200 = Union8{Union8_0{}};
                    long v201; Union8 v202;
                    Tuple19 tmp43 = Tuple19{0l, v200};
                    v201 = tmp43.v0; v202 = tmp43.v1;
                    while (while_method_2(v201)){
                        unsigned char v204;
                        v204 = v198[v201];
                        unsigned char v205;
                        v205 = v199[v201];
                        Union8 v216;
                        switch (v202.tag) {
                            case 0: { // Eq
                                unsigned char v206;
                                v206 = v204 / 4u;
                                unsigned char v207;
                                v207 = v205 / 4u;
                                bool v208;
                                v208 = v206 < v207;
                                if (v208){
                                    v216 = Union8{Union8_2{}};
                                } else {
                                    bool v210;
                                    v210 = v206 > v207;
                                    if (v210){
                                        v216 = Union8{Union8_1{}};
                                    } else {
                                        v216 = Union8{Union8_0{}};
                                    }
                                }
                                break;
                            }
                            default: {
                                break;
                            }
                        }
                        v202 = v216;
                        v201 += 1l ;
                    }
                    bool v217;
                    switch (v202.tag) {
                        case 1: { // Gt
                            v217 = true;
                            break;
                        }
                        default: {
                            v217 = false;
                        }
                    }
                    static_array<unsigned char,5l> v218;
                    if (v217){
                        v218 = v198;
                    } else {
                        v218 = v199;
                    }
                    v223 = Union9{Union9_1{v218}};
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
    static_array<unsigned char,5l> v224;
    long v225; long v226; unsigned char v227;
    Tuple18 tmp44 = Tuple18{0l, 0l, 12u};
    v225 = tmp44.v0; v226 = tmp44.v1; v227 = tmp44.v2;
    while (while_method_10(v225)){
        unsigned char v229;
        v229 = v73[v225];
        bool v230;
        v230 = v226 < 5l;
        long v242; unsigned char v243;
        if (v230){
            unsigned char v231;
            v231 = v229 % 4u;
            bool v232;
            v232 = 2u == v231;
            if (v232){
                unsigned char v233;
                v233 = v229 / 4u;
                bool v234;
                v234 = v227 == v233;
                long v235;
                if (v234){
                    v235 = v226;
                } else {
                    v235 = 0l;
                }
                v224[v235] = v229;
                long v236;
                v236 = v235 + 1l;
                unsigned char v237;
                v237 = v233 - 1u;
                v242 = v236; v243 = v237;
            } else {
                v242 = v226; v243 = v227;
            }
        } else {
            break;
        }
        v226 = v242;
        v227 = v243;
        v225 += 1l ;
    }
    bool v244;
    v244 = v226 == 4l;
    bool v279;
    if (v244){
        unsigned char v245;
        v245 = v227 + 1u;
        bool v246;
        v246 = v245 == 0u;
        if (v246){
            unsigned char v247;
            v247 = v73[0l];
            unsigned char v248;
            v248 = v247 % 4u;
            bool v249;
            v249 = 2u == v248;
            bool v253;
            if (v249){
                unsigned char v250;
                v250 = v247 / 4u;
                bool v251;
                v251 = v250 == 12u;
                if (v251){
                    v224[4l] = v247;
                    v253 = true;
                } else {
                    v253 = false;
                }
            } else {
                v253 = false;
            }
            if (v253){
                v279 = true;
            } else {
                unsigned char v254;
                v254 = v73[1l];
                unsigned char v255;
                v255 = v254 % 4u;
                bool v256;
                v256 = 2u == v255;
                bool v260;
                if (v256){
                    unsigned char v257;
                    v257 = v254 / 4u;
                    bool v258;
                    v258 = v257 == 12u;
                    if (v258){
                        v224[4l] = v254;
                        v260 = true;
                    } else {
                        v260 = false;
                    }
                } else {
                    v260 = false;
                }
                if (v260){
                    v279 = true;
                } else {
                    unsigned char v261;
                    v261 = v73[2l];
                    unsigned char v262;
                    v262 = v261 % 4u;
                    bool v263;
                    v263 = 2u == v262;
                    bool v267;
                    if (v263){
                        unsigned char v264;
                        v264 = v261 / 4u;
                        bool v265;
                        v265 = v264 == 12u;
                        if (v265){
                            v224[4l] = v261;
                            v267 = true;
                        } else {
                            v267 = false;
                        }
                    } else {
                        v267 = false;
                    }
                    if (v267){
                        v279 = true;
                    } else {
                        unsigned char v268;
                        v268 = v73[3l];
                        unsigned char v269;
                        v269 = v268 % 4u;
                        bool v270;
                        v270 = 2u == v269;
                        if (v270){
                            unsigned char v271;
                            v271 = v268 / 4u;
                            bool v272;
                            v272 = v271 == 12u;
                            if (v272){
                                v224[4l] = v268;
                                v279 = true;
                            } else {
                                v279 = false;
                            }
                        } else {
                            v279 = false;
                        }
                    }
                }
            }
        } else {
            v279 = false;
        }
    } else {
        v279 = false;
    }
    Union9 v285;
    if (v279){
        v285 = Union9{Union9_1{v224}};
    } else {
        bool v281;
        v281 = v226 == 5l;
        if (v281){
            v285 = Union9{Union9_1{v224}};
        } else {
            v285 = Union9{Union9_0{}};
        }
    }
    Union9 v311;
    switch (v223.tag) {
        case 0: { // None
            v311 = v285;
            break;
        }
        case 1: { // Some
            static_array<unsigned char,5l> v286 = v223.case1.v0;
            switch (v285.tag) {
                case 0: { // None
                    v311 = v223;
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5l> v287 = v285.case1.v0;
                    Union8 v288;
                    v288 = Union8{Union8_0{}};
                    long v289; Union8 v290;
                    Tuple19 tmp45 = Tuple19{0l, v288};
                    v289 = tmp45.v0; v290 = tmp45.v1;
                    while (while_method_2(v289)){
                        unsigned char v292;
                        v292 = v286[v289];
                        unsigned char v293;
                        v293 = v287[v289];
                        Union8 v304;
                        switch (v290.tag) {
                            case 0: { // Eq
                                unsigned char v294;
                                v294 = v292 / 4u;
                                unsigned char v295;
                                v295 = v293 / 4u;
                                bool v296;
                                v296 = v294 < v295;
                                if (v296){
                                    v304 = Union8{Union8_2{}};
                                } else {
                                    bool v298;
                                    v298 = v294 > v295;
                                    if (v298){
                                        v304 = Union8{Union8_1{}};
                                    } else {
                                        v304 = Union8{Union8_0{}};
                                    }
                                }
                                break;
                            }
                            default: {
                                break;
                            }
                        }
                        v290 = v304;
                        v289 += 1l ;
                    }
                    bool v305;
                    switch (v290.tag) {
                        case 1: { // Gt
                            v305 = true;
                            break;
                        }
                        default: {
                            v305 = false;
                        }
                    }
                    static_array<unsigned char,5l> v306;
                    if (v305){
                        v306 = v286;
                    } else {
                        v306 = v287;
                    }
                    v311 = Union9{Union9_1{v306}};
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
    static_array<unsigned char,5l> v312;
    long v313; long v314; unsigned char v315;
    Tuple18 tmp46 = Tuple18{0l, 0l, 12u};
    v313 = tmp46.v0; v314 = tmp46.v1; v315 = tmp46.v2;
    while (while_method_10(v313)){
        unsigned char v317;
        v317 = v73[v313];
        bool v318;
        v318 = v314 < 5l;
        long v330; unsigned char v331;
        if (v318){
            unsigned char v319;
            v319 = v317 % 4u;
            bool v320;
            v320 = 3u == v319;
            if (v320){
                unsigned char v321;
                v321 = v317 / 4u;
                bool v322;
                v322 = v315 == v321;
                long v323;
                if (v322){
                    v323 = v314;
                } else {
                    v323 = 0l;
                }
                v312[v323] = v317;
                long v324;
                v324 = v323 + 1l;
                unsigned char v325;
                v325 = v321 - 1u;
                v330 = v324; v331 = v325;
            } else {
                v330 = v314; v331 = v315;
            }
        } else {
            break;
        }
        v314 = v330;
        v315 = v331;
        v313 += 1l ;
    }
    bool v332;
    v332 = v314 == 4l;
    bool v367;
    if (v332){
        unsigned char v333;
        v333 = v315 + 1u;
        bool v334;
        v334 = v333 == 0u;
        if (v334){
            unsigned char v335;
            v335 = v73[0l];
            unsigned char v336;
            v336 = v335 % 4u;
            bool v337;
            v337 = 3u == v336;
            bool v341;
            if (v337){
                unsigned char v338;
                v338 = v335 / 4u;
                bool v339;
                v339 = v338 == 12u;
                if (v339){
                    v312[4l] = v335;
                    v341 = true;
                } else {
                    v341 = false;
                }
            } else {
                v341 = false;
            }
            if (v341){
                v367 = true;
            } else {
                unsigned char v342;
                v342 = v73[1l];
                unsigned char v343;
                v343 = v342 % 4u;
                bool v344;
                v344 = 3u == v343;
                bool v348;
                if (v344){
                    unsigned char v345;
                    v345 = v342 / 4u;
                    bool v346;
                    v346 = v345 == 12u;
                    if (v346){
                        v312[4l] = v342;
                        v348 = true;
                    } else {
                        v348 = false;
                    }
                } else {
                    v348 = false;
                }
                if (v348){
                    v367 = true;
                } else {
                    unsigned char v349;
                    v349 = v73[2l];
                    unsigned char v350;
                    v350 = v349 % 4u;
                    bool v351;
                    v351 = 3u == v350;
                    bool v355;
                    if (v351){
                        unsigned char v352;
                        v352 = v349 / 4u;
                        bool v353;
                        v353 = v352 == 12u;
                        if (v353){
                            v312[4l] = v349;
                            v355 = true;
                        } else {
                            v355 = false;
                        }
                    } else {
                        v355 = false;
                    }
                    if (v355){
                        v367 = true;
                    } else {
                        unsigned char v356;
                        v356 = v73[3l];
                        unsigned char v357;
                        v357 = v356 % 4u;
                        bool v358;
                        v358 = 3u == v357;
                        if (v358){
                            unsigned char v359;
                            v359 = v356 / 4u;
                            bool v360;
                            v360 = v359 == 12u;
                            if (v360){
                                v312[4l] = v356;
                                v367 = true;
                            } else {
                                v367 = false;
                            }
                        } else {
                            v367 = false;
                        }
                    }
                }
            }
        } else {
            v367 = false;
        }
    } else {
        v367 = false;
    }
    Union9 v373;
    if (v367){
        v373 = Union9{Union9_1{v312}};
    } else {
        bool v369;
        v369 = v314 == 5l;
        if (v369){
            v373 = Union9{Union9_1{v312}};
        } else {
            v373 = Union9{Union9_0{}};
        }
    }
    Union9 v399;
    switch (v311.tag) {
        case 0: { // None
            v399 = v373;
            break;
        }
        case 1: { // Some
            static_array<unsigned char,5l> v374 = v311.case1.v0;
            switch (v373.tag) {
                case 0: { // None
                    v399 = v311;
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5l> v375 = v373.case1.v0;
                    Union8 v376;
                    v376 = Union8{Union8_0{}};
                    long v377; Union8 v378;
                    Tuple19 tmp47 = Tuple19{0l, v376};
                    v377 = tmp47.v0; v378 = tmp47.v1;
                    while (while_method_2(v377)){
                        unsigned char v380;
                        v380 = v374[v377];
                        unsigned char v381;
                        v381 = v375[v377];
                        Union8 v392;
                        switch (v378.tag) {
                            case 0: { // Eq
                                unsigned char v382;
                                v382 = v380 / 4u;
                                unsigned char v383;
                                v383 = v381 / 4u;
                                bool v384;
                                v384 = v382 < v383;
                                if (v384){
                                    v392 = Union8{Union8_2{}};
                                } else {
                                    bool v386;
                                    v386 = v382 > v383;
                                    if (v386){
                                        v392 = Union8{Union8_1{}};
                                    } else {
                                        v392 = Union8{Union8_0{}};
                                    }
                                }
                                break;
                            }
                            default: {
                                break;
                            }
                        }
                        v378 = v392;
                        v377 += 1l ;
                    }
                    bool v393;
                    switch (v378.tag) {
                        case 1: { // Gt
                            v393 = true;
                            break;
                        }
                        default: {
                            v393 = false;
                        }
                    }
                    static_array<unsigned char,5l> v394;
                    if (v393){
                        v394 = v374;
                    } else {
                        v394 = v375;
                    }
                    v399 = Union9{Union9_1{v394}};
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
    static_array<unsigned char,5l> v925; char v926;
    switch (v399.tag) {
        case 0: { // None
            static_array<unsigned char,4l> v401;
            static_array<unsigned char,3l> v402;
            long v403; long v404; long v405; unsigned char v406;
            Tuple20 tmp48 = Tuple20{0l, 0l, 0l, 12u};
            v403 = tmp48.v0; v404 = tmp48.v1; v405 = tmp48.v2; v406 = tmp48.v3;
            while (while_method_10(v403)){
                unsigned char v408;
                v408 = v73[v403];
                bool v409;
                v409 = v405 < 4l;
                long v417; long v418; unsigned char v419;
                if (v409){
                    unsigned char v410;
                    v410 = v408 / 4u;
                    bool v411;
                    v411 = v406 == v410;
                    long v412;
                    if (v411){
                        v412 = v405;
                    } else {
                        v412 = 0l;
                    }
                    v401[v412] = v408;
                    long v413;
                    v413 = v412 + 1l;
                    v417 = v403; v418 = v413; v419 = v410;
                } else {
                    break;
                }
                v404 = v417;
                v405 = v418;
                v406 = v419;
                v403 += 1l ;
            }
            bool v420;
            v420 = v405 == 4l;
            Union10 v430;
            if (v420){
                long v421;
                v421 = 0l;
                while (while_method_3(v421)){
                    long v423;
                    v423 = v404 + -3l;
                    bool v424;
                    v424 = v421 < v423;
                    long v425;
                    if (v424){
                        v425 = 0l;
                    } else {
                        v425 = 4l;
                    }
                    long v426;
                    v426 = v425 + v421;
                    unsigned char v427;
                    v427 = v73[v426];
                    v402[v421] = v427;
                    v421 += 1l ;
                }
                v430 = Union10{Union10_1{v401, v402}};
            } else {
                v430 = Union10{Union10_0{}};
            }
            Union9 v448;
            switch (v430.tag) {
                case 0: { // None
                    v448 = Union9{Union9_0{}};
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,4l> v431 = v430.case1.v0; static_array<unsigned char,3l> v432 = v430.case1.v1;
                    static_array<unsigned char,1l> v433;
                    long v434;
                    v434 = 0l;
                    while (while_method_6(v434)){
                        unsigned char v436;
                        v436 = v432[v434];
                        v433[v434] = v436;
                        v434 += 1l ;
                    }
                    static_array<unsigned char,5l> v437;
                    long v438;
                    v438 = 0l;
                    while (while_method_4(v438)){
                        unsigned char v440;
                        v440 = v431[v438];
                        v437[v438] = v440;
                        v438 += 1l ;
                    }
                    long v441;
                    v441 = 0l;
                    while (while_method_6(v441)){
                        unsigned char v443;
                        v443 = v433[v441];
                        long v444;
                        v444 = 4l + v441;
                        v437[v444] = v443;
                        v441 += 1l ;
                    }
                    v448 = Union9{Union9_1{v437}};
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                }
            }
            switch (v448.tag) {
                case 0: { // None
                    static_array<unsigned char,3l> v450;
                    static_array<unsigned char,4l> v451;
                    long v452; long v453; long v454; unsigned char v455;
                    Tuple20 tmp49 = Tuple20{0l, 0l, 0l, 12u};
                    v452 = tmp49.v0; v453 = tmp49.v1; v454 = tmp49.v2; v455 = tmp49.v3;
                    while (while_method_10(v452)){
                        unsigned char v457;
                        v457 = v73[v452];
                        bool v458;
                        v458 = v454 < 3l;
                        long v466; long v467; unsigned char v468;
                        if (v458){
                            unsigned char v459;
                            v459 = v457 / 4u;
                            bool v460;
                            v460 = v455 == v459;
                            long v461;
                            if (v460){
                                v461 = v454;
                            } else {
                                v461 = 0l;
                            }
                            v450[v461] = v457;
                            long v462;
                            v462 = v461 + 1l;
                            v466 = v452; v467 = v462; v468 = v459;
                        } else {
                            break;
                        }
                        v453 = v466;
                        v454 = v467;
                        v455 = v468;
                        v452 += 1l ;
                    }
                    bool v469;
                    v469 = v454 == 3l;
                    Union11 v479;
                    if (v469){
                        long v470;
                        v470 = 0l;
                        while (while_method_4(v470)){
                            long v472;
                            v472 = v453 + -2l;
                            bool v473;
                            v473 = v470 < v472;
                            long v474;
                            if (v473){
                                v474 = 0l;
                            } else {
                                v474 = 3l;
                            }
                            long v475;
                            v475 = v474 + v470;
                            unsigned char v476;
                            v476 = v73[v475];
                            v451[v470] = v476;
                            v470 += 1l ;
                        }
                        v479 = Union11{Union11_1{v450, v451}};
                    } else {
                        v479 = Union11{Union11_0{}};
                    }
                    Union9 v528;
                    switch (v479.tag) {
                        case 0: { // None
                            v528 = Union9{Union9_0{}};
                            break;
                        }
                        case 1: { // Some
                            static_array<unsigned char,3l> v480 = v479.case1.v0; static_array<unsigned char,4l> v481 = v479.case1.v1;
                            static_array<unsigned char,2l> v482;
                            static_array<unsigned char,2l> v483;
                            long v484; long v485; long v486; unsigned char v487;
                            Tuple20 tmp50 = Tuple20{0l, 0l, 0l, 12u};
                            v484 = tmp50.v0; v485 = tmp50.v1; v486 = tmp50.v2; v487 = tmp50.v3;
                            while (while_method_4(v484)){
                                unsigned char v489;
                                v489 = v481[v484];
                                bool v490;
                                v490 = v486 < 2l;
                                long v498; long v499; unsigned char v500;
                                if (v490){
                                    unsigned char v491;
                                    v491 = v489 / 4u;
                                    bool v492;
                                    v492 = v487 == v491;
                                    long v493;
                                    if (v492){
                                        v493 = v486;
                                    } else {
                                        v493 = 0l;
                                    }
                                    v482[v493] = v489;
                                    long v494;
                                    v494 = v493 + 1l;
                                    v498 = v484; v499 = v494; v500 = v491;
                                } else {
                                    break;
                                }
                                v485 = v498;
                                v486 = v499;
                                v487 = v500;
                                v484 += 1l ;
                            }
                            bool v501;
                            v501 = v486 == 2l;
                            Union12 v511;
                            if (v501){
                                long v502;
                                v502 = 0l;
                                while (while_method_0(v502)){
                                    long v504;
                                    v504 = v485 + -1l;
                                    bool v505;
                                    v505 = v502 < v504;
                                    long v506;
                                    if (v505){
                                        v506 = 0l;
                                    } else {
                                        v506 = 2l;
                                    }
                                    long v507;
                                    v507 = v506 + v502;
                                    unsigned char v508;
                                    v508 = v481[v507];
                                    v483[v502] = v508;
                                    v502 += 1l ;
                                }
                                v511 = Union12{Union12_1{v482, v483}};
                            } else {
                                v511 = Union12{Union12_0{}};
                            }
                            switch (v511.tag) {
                                case 0: { // None
                                    v528 = Union9{Union9_0{}};
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,2l> v512 = v511.case1.v0; static_array<unsigned char,2l> v513 = v511.case1.v1;
                                    static_array<unsigned char,5l> v514;
                                    long v515;
                                    v515 = 0l;
                                    while (while_method_3(v515)){
                                        unsigned char v517;
                                        v517 = v480[v515];
                                        v514[v515] = v517;
                                        v515 += 1l ;
                                    }
                                    long v518;
                                    v518 = 0l;
                                    while (while_method_0(v518)){
                                        unsigned char v520;
                                        v520 = v512[v518];
                                        long v521;
                                        v521 = 3l + v518;
                                        v514[v521] = v520;
                                        v518 += 1l ;
                                    }
                                    v528 = Union9{Union9_1{v514}};
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
                    switch (v528.tag) {
                        case 0: { // None
                            static_array<unsigned char,5l> v530;
                            long v531; long v532;
                            Tuple2 tmp51 = Tuple2{0l, 0l};
                            v531 = tmp51.v0; v532 = tmp51.v1;
                            while (while_method_10(v531)){
                                unsigned char v534;
                                v534 = v73[v531];
                                unsigned char v535;
                                v535 = v534 % 4u;
                                bool v536;
                                v536 = v535 == 0u;
                                bool v538;
                                if (v536){
                                    bool v537;
                                    v537 = v532 < 5l;
                                    v538 = v537;
                                } else {
                                    v538 = false;
                                }
                                long v540;
                                if (v538){
                                    v530[v532] = v534;
                                    long v539;
                                    v539 = v532 + 1l;
                                    v540 = v539;
                                } else {
                                    v540 = v532;
                                }
                                v532 = v540;
                                v531 += 1l ;
                            }
                            bool v541;
                            v541 = v532 == 5l;
                            Union9 v544;
                            if (v541){
                                v544 = Union9{Union9_1{v530}};
                            } else {
                                v544 = Union9{Union9_0{}};
                            }
                            static_array<unsigned char,5l> v545;
                            long v546; long v547;
                            Tuple2 tmp52 = Tuple2{0l, 0l};
                            v546 = tmp52.v0; v547 = tmp52.v1;
                            while (while_method_10(v546)){
                                unsigned char v549;
                                v549 = v73[v546];
                                unsigned char v550;
                                v550 = v549 % 4u;
                                bool v551;
                                v551 = v550 == 1u;
                                bool v553;
                                if (v551){
                                    bool v552;
                                    v552 = v547 < 5l;
                                    v553 = v552;
                                } else {
                                    v553 = false;
                                }
                                long v555;
                                if (v553){
                                    v545[v547] = v549;
                                    long v554;
                                    v554 = v547 + 1l;
                                    v555 = v554;
                                } else {
                                    v555 = v547;
                                }
                                v547 = v555;
                                v546 += 1l ;
                            }
                            bool v556;
                            v556 = v547 == 5l;
                            Union9 v559;
                            if (v556){
                                v559 = Union9{Union9_1{v545}};
                            } else {
                                v559 = Union9{Union9_0{}};
                            }
                            Union9 v585;
                            switch (v544.tag) {
                                case 0: { // None
                                    v585 = v559;
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,5l> v560 = v544.case1.v0;
                                    switch (v559.tag) {
                                        case 0: { // None
                                            v585 = v544;
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5l> v561 = v559.case1.v0;
                                            Union8 v562;
                                            v562 = Union8{Union8_0{}};
                                            long v563; Union8 v564;
                                            Tuple19 tmp53 = Tuple19{0l, v562};
                                            v563 = tmp53.v0; v564 = tmp53.v1;
                                            while (while_method_2(v563)){
                                                unsigned char v566;
                                                v566 = v560[v563];
                                                unsigned char v567;
                                                v567 = v561[v563];
                                                Union8 v578;
                                                switch (v564.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v568;
                                                        v568 = v566 / 4u;
                                                        unsigned char v569;
                                                        v569 = v567 / 4u;
                                                        bool v570;
                                                        v570 = v568 < v569;
                                                        if (v570){
                                                            v578 = Union8{Union8_2{}};
                                                        } else {
                                                            bool v572;
                                                            v572 = v568 > v569;
                                                            if (v572){
                                                                v578 = Union8{Union8_1{}};
                                                            } else {
                                                                v578 = Union8{Union8_0{}};
                                                            }
                                                        }
                                                        break;
                                                    }
                                                    default: {
                                                        break;
                                                    }
                                                }
                                                v564 = v578;
                                                v563 += 1l ;
                                            }
                                            bool v579;
                                            switch (v564.tag) {
                                                case 1: { // Gt
                                                    v579 = true;
                                                    break;
                                                }
                                                default: {
                                                    v579 = false;
                                                }
                                            }
                                            static_array<unsigned char,5l> v580;
                                            if (v579){
                                                v580 = v560;
                                            } else {
                                                v580 = v561;
                                            }
                                            v585 = Union9{Union9_1{v580}};
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
                            static_array<unsigned char,5l> v586;
                            long v587; long v588;
                            Tuple2 tmp54 = Tuple2{0l, 0l};
                            v587 = tmp54.v0; v588 = tmp54.v1;
                            while (while_method_10(v587)){
                                unsigned char v590;
                                v590 = v73[v587];
                                unsigned char v591;
                                v591 = v590 % 4u;
                                bool v592;
                                v592 = v591 == 2u;
                                bool v594;
                                if (v592){
                                    bool v593;
                                    v593 = v588 < 5l;
                                    v594 = v593;
                                } else {
                                    v594 = false;
                                }
                                long v596;
                                if (v594){
                                    v586[v588] = v590;
                                    long v595;
                                    v595 = v588 + 1l;
                                    v596 = v595;
                                } else {
                                    v596 = v588;
                                }
                                v588 = v596;
                                v587 += 1l ;
                            }
                            bool v597;
                            v597 = v588 == 5l;
                            Union9 v600;
                            if (v597){
                                v600 = Union9{Union9_1{v586}};
                            } else {
                                v600 = Union9{Union9_0{}};
                            }
                            Union9 v626;
                            switch (v585.tag) {
                                case 0: { // None
                                    v626 = v600;
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,5l> v601 = v585.case1.v0;
                                    switch (v600.tag) {
                                        case 0: { // None
                                            v626 = v585;
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5l> v602 = v600.case1.v0;
                                            Union8 v603;
                                            v603 = Union8{Union8_0{}};
                                            long v604; Union8 v605;
                                            Tuple19 tmp55 = Tuple19{0l, v603};
                                            v604 = tmp55.v0; v605 = tmp55.v1;
                                            while (while_method_2(v604)){
                                                unsigned char v607;
                                                v607 = v601[v604];
                                                unsigned char v608;
                                                v608 = v602[v604];
                                                Union8 v619;
                                                switch (v605.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v609;
                                                        v609 = v607 / 4u;
                                                        unsigned char v610;
                                                        v610 = v608 / 4u;
                                                        bool v611;
                                                        v611 = v609 < v610;
                                                        if (v611){
                                                            v619 = Union8{Union8_2{}};
                                                        } else {
                                                            bool v613;
                                                            v613 = v609 > v610;
                                                            if (v613){
                                                                v619 = Union8{Union8_1{}};
                                                            } else {
                                                                v619 = Union8{Union8_0{}};
                                                            }
                                                        }
                                                        break;
                                                    }
                                                    default: {
                                                        break;
                                                    }
                                                }
                                                v605 = v619;
                                                v604 += 1l ;
                                            }
                                            bool v620;
                                            switch (v605.tag) {
                                                case 1: { // Gt
                                                    v620 = true;
                                                    break;
                                                }
                                                default: {
                                                    v620 = false;
                                                }
                                            }
                                            static_array<unsigned char,5l> v621;
                                            if (v620){
                                                v621 = v601;
                                            } else {
                                                v621 = v602;
                                            }
                                            v626 = Union9{Union9_1{v621}};
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
                            static_array<unsigned char,5l> v627;
                            long v628; long v629;
                            Tuple2 tmp56 = Tuple2{0l, 0l};
                            v628 = tmp56.v0; v629 = tmp56.v1;
                            while (while_method_10(v628)){
                                unsigned char v631;
                                v631 = v73[v628];
                                unsigned char v632;
                                v632 = v631 % 4u;
                                bool v633;
                                v633 = v632 == 3u;
                                bool v635;
                                if (v633){
                                    bool v634;
                                    v634 = v629 < 5l;
                                    v635 = v634;
                                } else {
                                    v635 = false;
                                }
                                long v637;
                                if (v635){
                                    v627[v629] = v631;
                                    long v636;
                                    v636 = v629 + 1l;
                                    v637 = v636;
                                } else {
                                    v637 = v629;
                                }
                                v629 = v637;
                                v628 += 1l ;
                            }
                            bool v638;
                            v638 = v629 == 5l;
                            Union9 v641;
                            if (v638){
                                v641 = Union9{Union9_1{v627}};
                            } else {
                                v641 = Union9{Union9_0{}};
                            }
                            Union9 v667;
                            switch (v626.tag) {
                                case 0: { // None
                                    v667 = v641;
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,5l> v642 = v626.case1.v0;
                                    switch (v641.tag) {
                                        case 0: { // None
                                            v667 = v626;
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5l> v643 = v641.case1.v0;
                                            Union8 v644;
                                            v644 = Union8{Union8_0{}};
                                            long v645; Union8 v646;
                                            Tuple19 tmp57 = Tuple19{0l, v644};
                                            v645 = tmp57.v0; v646 = tmp57.v1;
                                            while (while_method_2(v645)){
                                                unsigned char v648;
                                                v648 = v642[v645];
                                                unsigned char v649;
                                                v649 = v643[v645];
                                                Union8 v660;
                                                switch (v646.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v650;
                                                        v650 = v648 / 4u;
                                                        unsigned char v651;
                                                        v651 = v649 / 4u;
                                                        bool v652;
                                                        v652 = v650 < v651;
                                                        if (v652){
                                                            v660 = Union8{Union8_2{}};
                                                        } else {
                                                            bool v654;
                                                            v654 = v650 > v651;
                                                            if (v654){
                                                                v660 = Union8{Union8_1{}};
                                                            } else {
                                                                v660 = Union8{Union8_0{}};
                                                            }
                                                        }
                                                        break;
                                                    }
                                                    default: {
                                                        break;
                                                    }
                                                }
                                                v646 = v660;
                                                v645 += 1l ;
                                            }
                                            bool v661;
                                            switch (v646.tag) {
                                                case 1: { // Gt
                                                    v661 = true;
                                                    break;
                                                }
                                                default: {
                                                    v661 = false;
                                                }
                                            }
                                            static_array<unsigned char,5l> v662;
                                            if (v661){
                                                v662 = v642;
                                            } else {
                                                v662 = v643;
                                            }
                                            v667 = Union9{Union9_1{v662}};
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
                            switch (v667.tag) {
                                case 0: { // None
                                    static_array<unsigned char,5l> v669;
                                    long v670; long v671; unsigned char v672;
                                    Tuple18 tmp58 = Tuple18{0l, 0l, 12u};
                                    v670 = tmp58.v0; v671 = tmp58.v1; v672 = tmp58.v2;
                                    while (while_method_10(v670)){
                                        unsigned char v674;
                                        v674 = v73[v670];
                                        bool v675;
                                        v675 = v671 < 5l;
                                        long v687; unsigned char v688;
                                        if (v675){
                                            unsigned char v676;
                                            v676 = v674 / 4u;
                                            unsigned char v677;
                                            v677 = v676 - 1u;
                                            bool v678;
                                            v678 = v672 == v677;
                                            bool v679;
                                            v679 = v678 != true;
                                            if (v679){
                                                bool v680;
                                                v680 = v672 == v676;
                                                long v681;
                                                if (v680){
                                                    v681 = v671;
                                                } else {
                                                    v681 = 0l;
                                                }
                                                v669[v681] = v674;
                                                long v682;
                                                v682 = v681 + 1l;
                                                v687 = v682; v688 = v677;
                                            } else {
                                                v687 = v671; v688 = v672;
                                            }
                                        } else {
                                            break;
                                        }
                                        v671 = v687;
                                        v672 = v688;
                                        v670 += 1l ;
                                    }
                                    bool v689;
                                    v689 = v671 == 4l;
                                    bool v697;
                                    if (v689){
                                        unsigned char v690;
                                        v690 = v672 + 1u;
                                        bool v691;
                                        v691 = v690 == 0u;
                                        if (v691){
                                            unsigned char v692;
                                            v692 = v73[0l];
                                            unsigned char v693;
                                            v693 = v692 / 4u;
                                            bool v694;
                                            v694 = v693 == 12u;
                                            if (v694){
                                                v669[4l] = v692;
                                                v697 = true;
                                            } else {
                                                v697 = false;
                                            }
                                        } else {
                                            v697 = false;
                                        }
                                    } else {
                                        v697 = false;
                                    }
                                    Union9 v703;
                                    if (v697){
                                        v703 = Union9{Union9_1{v669}};
                                    } else {
                                        bool v699;
                                        v699 = v671 == 5l;
                                        if (v699){
                                            v703 = Union9{Union9_1{v669}};
                                        } else {
                                            v703 = Union9{Union9_0{}};
                                        }
                                    }
                                    switch (v703.tag) {
                                        case 0: { // None
                                            static_array<unsigned char,3l> v705;
                                            static_array<unsigned char,4l> v706;
                                            long v707; long v708; long v709; unsigned char v710;
                                            Tuple20 tmp59 = Tuple20{0l, 0l, 0l, 12u};
                                            v707 = tmp59.v0; v708 = tmp59.v1; v709 = tmp59.v2; v710 = tmp59.v3;
                                            while (while_method_10(v707)){
                                                unsigned char v712;
                                                v712 = v73[v707];
                                                bool v713;
                                                v713 = v709 < 3l;
                                                long v721; long v722; unsigned char v723;
                                                if (v713){
                                                    unsigned char v714;
                                                    v714 = v712 / 4u;
                                                    bool v715;
                                                    v715 = v710 == v714;
                                                    long v716;
                                                    if (v715){
                                                        v716 = v709;
                                                    } else {
                                                        v716 = 0l;
                                                    }
                                                    v705[v716] = v712;
                                                    long v717;
                                                    v717 = v716 + 1l;
                                                    v721 = v707; v722 = v717; v723 = v714;
                                                } else {
                                                    break;
                                                }
                                                v708 = v721;
                                                v709 = v722;
                                                v710 = v723;
                                                v707 += 1l ;
                                            }
                                            bool v724;
                                            v724 = v709 == 3l;
                                            Union11 v734;
                                            if (v724){
                                                long v725;
                                                v725 = 0l;
                                                while (while_method_4(v725)){
                                                    long v727;
                                                    v727 = v708 + -2l;
                                                    bool v728;
                                                    v728 = v725 < v727;
                                                    long v729;
                                                    if (v728){
                                                        v729 = 0l;
                                                    } else {
                                                        v729 = 3l;
                                                    }
                                                    long v730;
                                                    v730 = v729 + v725;
                                                    unsigned char v731;
                                                    v731 = v73[v730];
                                                    v706[v725] = v731;
                                                    v725 += 1l ;
                                                }
                                                v734 = Union11{Union11_1{v705, v706}};
                                            } else {
                                                v734 = Union11{Union11_0{}};
                                            }
                                            Union9 v752;
                                            switch (v734.tag) {
                                                case 0: { // None
                                                    v752 = Union9{Union9_0{}};
                                                    break;
                                                }
                                                case 1: { // Some
                                                    static_array<unsigned char,3l> v735 = v734.case1.v0; static_array<unsigned char,4l> v736 = v734.case1.v1;
                                                    static_array<unsigned char,2l> v737;
                                                    long v738;
                                                    v738 = 0l;
                                                    while (while_method_0(v738)){
                                                        unsigned char v740;
                                                        v740 = v736[v738];
                                                        v737[v738] = v740;
                                                        v738 += 1l ;
                                                    }
                                                    static_array<unsigned char,5l> v741;
                                                    long v742;
                                                    v742 = 0l;
                                                    while (while_method_3(v742)){
                                                        unsigned char v744;
                                                        v744 = v735[v742];
                                                        v741[v742] = v744;
                                                        v742 += 1l ;
                                                    }
                                                    long v745;
                                                    v745 = 0l;
                                                    while (while_method_0(v745)){
                                                        unsigned char v747;
                                                        v747 = v737[v745];
                                                        long v748;
                                                        v748 = 3l + v745;
                                                        v741[v748] = v747;
                                                        v745 += 1l ;
                                                    }
                                                    v752 = Union9{Union9_1{v741}};
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false);
                                                }
                                            }
                                            switch (v752.tag) {
                                                case 0: { // None
                                                    static_array<unsigned char,2l> v754;
                                                    static_array<unsigned char,5l> v755;
                                                    long v756; long v757; long v758; unsigned char v759;
                                                    Tuple20 tmp60 = Tuple20{0l, 0l, 0l, 12u};
                                                    v756 = tmp60.v0; v757 = tmp60.v1; v758 = tmp60.v2; v759 = tmp60.v3;
                                                    while (while_method_10(v756)){
                                                        unsigned char v761;
                                                        v761 = v73[v756];
                                                        bool v762;
                                                        v762 = v758 < 2l;
                                                        long v770; long v771; unsigned char v772;
                                                        if (v762){
                                                            unsigned char v763;
                                                            v763 = v761 / 4u;
                                                            bool v764;
                                                            v764 = v759 == v763;
                                                            long v765;
                                                            if (v764){
                                                                v765 = v758;
                                                            } else {
                                                                v765 = 0l;
                                                            }
                                                            v754[v765] = v761;
                                                            long v766;
                                                            v766 = v765 + 1l;
                                                            v770 = v756; v771 = v766; v772 = v763;
                                                        } else {
                                                            break;
                                                        }
                                                        v757 = v770;
                                                        v758 = v771;
                                                        v759 = v772;
                                                        v756 += 1l ;
                                                    }
                                                    bool v773;
                                                    v773 = v758 == 2l;
                                                    Union13 v783;
                                                    if (v773){
                                                        long v774;
                                                        v774 = 0l;
                                                        while (while_method_2(v774)){
                                                            long v776;
                                                            v776 = v757 + -1l;
                                                            bool v777;
                                                            v777 = v774 < v776;
                                                            long v778;
                                                            if (v777){
                                                                v778 = 0l;
                                                            } else {
                                                                v778 = 2l;
                                                            }
                                                            long v779;
                                                            v779 = v778 + v774;
                                                            unsigned char v780;
                                                            v780 = v73[v779];
                                                            v755[v774] = v780;
                                                            v774 += 1l ;
                                                        }
                                                        v783 = Union13{Union13_1{v754, v755}};
                                                    } else {
                                                        v783 = Union13{Union13_0{}};
                                                    }
                                                    Union9 v840;
                                                    switch (v783.tag) {
                                                        case 0: { // None
                                                            v840 = Union9{Union9_0{}};
                                                            break;
                                                        }
                                                        case 1: { // Some
                                                            static_array<unsigned char,2l> v784 = v783.case1.v0; static_array<unsigned char,5l> v785 = v783.case1.v1;
                                                            static_array<unsigned char,2l> v786;
                                                            static_array<unsigned char,3l> v787;
                                                            long v788; long v789; long v790; unsigned char v791;
                                                            Tuple20 tmp61 = Tuple20{0l, 0l, 0l, 12u};
                                                            v788 = tmp61.v0; v789 = tmp61.v1; v790 = tmp61.v2; v791 = tmp61.v3;
                                                            while (while_method_2(v788)){
                                                                unsigned char v793;
                                                                v793 = v785[v788];
                                                                bool v794;
                                                                v794 = v790 < 2l;
                                                                long v802; long v803; unsigned char v804;
                                                                if (v794){
                                                                    unsigned char v795;
                                                                    v795 = v793 / 4u;
                                                                    bool v796;
                                                                    v796 = v791 == v795;
                                                                    long v797;
                                                                    if (v796){
                                                                        v797 = v790;
                                                                    } else {
                                                                        v797 = 0l;
                                                                    }
                                                                    v786[v797] = v793;
                                                                    long v798;
                                                                    v798 = v797 + 1l;
                                                                    v802 = v788; v803 = v798; v804 = v795;
                                                                } else {
                                                                    break;
                                                                }
                                                                v789 = v802;
                                                                v790 = v803;
                                                                v791 = v804;
                                                                v788 += 1l ;
                                                            }
                                                            bool v805;
                                                            v805 = v790 == 2l;
                                                            Union14 v815;
                                                            if (v805){
                                                                long v806;
                                                                v806 = 0l;
                                                                while (while_method_3(v806)){
                                                                    long v808;
                                                                    v808 = v789 + -1l;
                                                                    bool v809;
                                                                    v809 = v806 < v808;
                                                                    long v810;
                                                                    if (v809){
                                                                        v810 = 0l;
                                                                    } else {
                                                                        v810 = 2l;
                                                                    }
                                                                    long v811;
                                                                    v811 = v810 + v806;
                                                                    unsigned char v812;
                                                                    v812 = v785[v811];
                                                                    v787[v806] = v812;
                                                                    v806 += 1l ;
                                                                }
                                                                v815 = Union14{Union14_1{v786, v787}};
                                                            } else {
                                                                v815 = Union14{Union14_0{}};
                                                            }
                                                            switch (v815.tag) {
                                                                case 0: { // None
                                                                    v840 = Union9{Union9_0{}};
                                                                    break;
                                                                }
                                                                case 1: { // Some
                                                                    static_array<unsigned char,2l> v816 = v815.case1.v0; static_array<unsigned char,3l> v817 = v815.case1.v1;
                                                                    static_array<unsigned char,1l> v818;
                                                                    long v819;
                                                                    v819 = 0l;
                                                                    while (while_method_6(v819)){
                                                                        unsigned char v821;
                                                                        v821 = v817[v819];
                                                                        v818[v819] = v821;
                                                                        v819 += 1l ;
                                                                    }
                                                                    static_array<unsigned char,5l> v822;
                                                                    long v823;
                                                                    v823 = 0l;
                                                                    while (while_method_0(v823)){
                                                                        unsigned char v825;
                                                                        v825 = v784[v823];
                                                                        v822[v823] = v825;
                                                                        v823 += 1l ;
                                                                    }
                                                                    long v826;
                                                                    v826 = 0l;
                                                                    while (while_method_0(v826)){
                                                                        unsigned char v828;
                                                                        v828 = v816[v826];
                                                                        long v829;
                                                                        v829 = 2l + v826;
                                                                        v822[v829] = v828;
                                                                        v826 += 1l ;
                                                                    }
                                                                    long v830;
                                                                    v830 = 0l;
                                                                    while (while_method_6(v830)){
                                                                        unsigned char v832;
                                                                        v832 = v818[v830];
                                                                        long v833;
                                                                        v833 = 4l + v830;
                                                                        v822[v833] = v832;
                                                                        v830 += 1l ;
                                                                    }
                                                                    v840 = Union9{Union9_1{v822}};
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
                                                    switch (v840.tag) {
                                                        case 0: { // None
                                                            static_array<unsigned char,2l> v842;
                                                            static_array<unsigned char,5l> v843;
                                                            long v844; long v845; long v846; unsigned char v847;
                                                            Tuple20 tmp62 = Tuple20{0l, 0l, 0l, 12u};
                                                            v844 = tmp62.v0; v845 = tmp62.v1; v846 = tmp62.v2; v847 = tmp62.v3;
                                                            while (while_method_10(v844)){
                                                                unsigned char v849;
                                                                v849 = v73[v844];
                                                                bool v850;
                                                                v850 = v846 < 2l;
                                                                long v858; long v859; unsigned char v860;
                                                                if (v850){
                                                                    unsigned char v851;
                                                                    v851 = v849 / 4u;
                                                                    bool v852;
                                                                    v852 = v847 == v851;
                                                                    long v853;
                                                                    if (v852){
                                                                        v853 = v846;
                                                                    } else {
                                                                        v853 = 0l;
                                                                    }
                                                                    v842[v853] = v849;
                                                                    long v854;
                                                                    v854 = v853 + 1l;
                                                                    v858 = v844; v859 = v854; v860 = v851;
                                                                } else {
                                                                    break;
                                                                }
                                                                v845 = v858;
                                                                v846 = v859;
                                                                v847 = v860;
                                                                v844 += 1l ;
                                                            }
                                                            bool v861;
                                                            v861 = v846 == 2l;
                                                            Union13 v871;
                                                            if (v861){
                                                                long v862;
                                                                v862 = 0l;
                                                                while (while_method_2(v862)){
                                                                    long v864;
                                                                    v864 = v845 + -1l;
                                                                    bool v865;
                                                                    v865 = v862 < v864;
                                                                    long v866;
                                                                    if (v865){
                                                                        v866 = 0l;
                                                                    } else {
                                                                        v866 = 2l;
                                                                    }
                                                                    long v867;
                                                                    v867 = v866 + v862;
                                                                    unsigned char v868;
                                                                    v868 = v73[v867];
                                                                    v843[v862] = v868;
                                                                    v862 += 1l ;
                                                                }
                                                                v871 = Union13{Union13_1{v842, v843}};
                                                            } else {
                                                                v871 = Union13{Union13_0{}};
                                                            }
                                                            Union9 v889;
                                                            switch (v871.tag) {
                                                                case 0: { // None
                                                                    v889 = Union9{Union9_0{}};
                                                                    break;
                                                                }
                                                                case 1: { // Some
                                                                    static_array<unsigned char,2l> v872 = v871.case1.v0; static_array<unsigned char,5l> v873 = v871.case1.v1;
                                                                    static_array<unsigned char,3l> v874;
                                                                    long v875;
                                                                    v875 = 0l;
                                                                    while (while_method_3(v875)){
                                                                        unsigned char v877;
                                                                        v877 = v873[v875];
                                                                        v874[v875] = v877;
                                                                        v875 += 1l ;
                                                                    }
                                                                    static_array<unsigned char,5l> v878;
                                                                    long v879;
                                                                    v879 = 0l;
                                                                    while (while_method_0(v879)){
                                                                        unsigned char v881;
                                                                        v881 = v872[v879];
                                                                        v878[v879] = v881;
                                                                        v879 += 1l ;
                                                                    }
                                                                    long v882;
                                                                    v882 = 0l;
                                                                    while (while_method_3(v882)){
                                                                        unsigned char v884;
                                                                        v884 = v874[v882];
                                                                        long v885;
                                                                        v885 = 2l + v882;
                                                                        v878[v885] = v884;
                                                                        v882 += 1l ;
                                                                    }
                                                                    v889 = Union9{Union9_1{v878}};
                                                                    break;
                                                                }
                                                                default: {
                                                                    assert("Invalid tag." && false);
                                                                }
                                                            }
                                                            switch (v889.tag) {
                                                                case 0: { // None
                                                                    static_array<unsigned char,5l> v891;
                                                                    long v892;
                                                                    v892 = 0l;
                                                                    while (while_method_2(v892)){
                                                                        unsigned char v894;
                                                                        v894 = v73[v892];
                                                                        v891[v892] = v894;
                                                                        v892 += 1l ;
                                                                    }
                                                                    v925 = v891; v926 = 0;
                                                                    break;
                                                                }
                                                                case 1: { // Some
                                                                    static_array<unsigned char,5l> v890 = v889.case1.v0;
                                                                    v925 = v890; v926 = 1;
                                                                    break;
                                                                }
                                                                default: {
                                                                    assert("Invalid tag." && false);
                                                                }
                                                            }
                                                            break;
                                                        }
                                                        case 1: { // Some
                                                            static_array<unsigned char,5l> v841 = v840.case1.v0;
                                                            v925 = v841; v926 = 2;
                                                            break;
                                                        }
                                                        default: {
                                                            assert("Invalid tag." && false);
                                                        }
                                                    }
                                                    break;
                                                }
                                                case 1: { // Some
                                                    static_array<unsigned char,5l> v753 = v752.case1.v0;
                                                    v925 = v753; v926 = 3;
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false);
                                                }
                                            }
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5l> v704 = v703.case1.v0;
                                            v925 = v704; v926 = 4;
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false);
                                        }
                                    }
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,5l> v668 = v667.case1.v0;
                                    v925 = v668; v926 = 5;
                                    break;
                                }
                                default: {
                                    assert("Invalid tag." && false);
                                }
                            }
                            break;
                        }
                        case 1: { // Some
                            static_array<unsigned char,5l> v529 = v528.case1.v0;
                            v925 = v529; v926 = 6;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                        }
                    }
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5l> v449 = v448.case1.v0;
                    v925 = v449; v926 = 7;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                }
            }
            break;
        }
        case 1: { // Some
            static_array<unsigned char,5l> v400 = v399.case1.v0;
            v925 = v400; v926 = 8;
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    return Tuple0{v925, v926};
}
__device__ Union5 play_loop_inner_32(unsigned long long & v0, dynamic_array_list<Union3,128l> & v1, curandStatePhilox4_32_10_t & v2, static_array<Union2,2l> v3, Union5 v4){
    dynamic_array_list<Union3,128l> & v5 = v1;
    unsigned long long & v6 = v0;
    bool v7; Union5 v8;
    Tuple9 tmp16 = Tuple9{true, v4};
    v7 = tmp16.v0; v8 = tmp16.v1;
    while (while_method_5(v7, v8)){
        bool v543; Union5 v544;
        switch (v8.tag) {
            case 0: { // G_Flop
                long v446 = v8.case0.v0; static_array<static_array<unsigned char,2l>,2l> v447 = v8.case0.v1; static_array<long,2l> v448 = v8.case0.v2; long v449 = v8.case0.v3; static_array<long,2l> v450 = v8.case0.v4; Union6 v451 = v8.case0.v5;
                static_array<unsigned char,3l> v452; unsigned long long v453;
                Tuple10 tmp19 = draw_cards_33(v2, v6);
                v452 = tmp19.v0; v453 = tmp19.v1;
                v0 = v453;
                static_array_list<unsigned char,5l> v454;
                v454 = get_community_cards_36(v451, v452);
                Union3 v455;
                v455 = Union3{Union3_0{v454}};
                v5.push(v455);
                Union6 v458;
                switch (v451.tag) {
                    case 1: { // Preflop
                        v458 = Union6{Union6_0{v452}};
                        break;
                    }
                    default: {
                        printf("%s\n", "Invalid street in flop.");
                        asm("exit;");
                    }
                }
                long v459;
                v459 = 2l;
                long v460;
                v460 = 0l;
                Union5 v461;
                v461 = try_round_37(v459, v447, v448, v460, v450, v458);
                v543 = true; v544 = v461;
                break;
            }
            case 1: { // G_Fold
                long v10 = v8.case1.v0; static_array<static_array<unsigned char,2l>,2l> v11 = v8.case1.v1; static_array<long,2l> v12 = v8.case1.v2; long v13 = v8.case1.v3; static_array<long,2l> v14 = v8.case1.v4; Union6 v15 = v8.case1.v5;
                long v16;
                v16 = v13 % 2l;
                long v17;
                v17 = v12[v16];
                long v18;
                v18 = v13 + 1l;
                long v19;
                v19 = v18 % 2l;
                Union3 v20;
                v20 = Union3{Union3_1{v17, v19}};
                v5.push(v20);
                v543 = false; v544 = v8;
                break;
            }
            case 2: { // G_Preflop
                static_array<unsigned char,2l> v512; unsigned long long v513;
                Tuple13 tmp24 = draw_cards_40(v2, v6);
                v512 = tmp24.v0; v513 = tmp24.v1;
                v0 = v513;
                static_array<unsigned char,2l> v514; unsigned long long v515;
                Tuple13 tmp25 = draw_cards_40(v2, v6);
                v514 = tmp25.v0; v515 = tmp25.v1;
                v0 = v515;
                Union3 v516;
                v516 = Union3{Union3_3{0l, v512}};
                v5.push(v516);
                Union3 v517;
                v517 = Union3{Union3_3{1l, v514}};
                v5.push(v517);
                static_array<static_array<unsigned char,2l>,2l> v518;
                v518[0l] = v512;
                v518[1l] = v514;
                static_array<long,2l> v519;
                v519[0l] = 2l;
                v519[1l] = 1l;
                static_array<long,2l> v520;
                long v521;
                v521 = 0l;
                while (while_method_0(v521)){
                    long v523;
                    v523 = v519[v521];
                    long v524;
                    v524 = 100l - v523;
                    v520[v521] = v524;
                    v521 += 1l ;
                }
                long v525;
                v525 = 2l;
                long v526;
                v526 = 0l;
                Union6 v527;
                v527 = Union6{Union6_1{}};
                Union5 v528;
                v528 = try_round_37(v525, v518, v519, v526, v520, v527);
                v543 = true; v544 = v528;
                break;
            }
            case 3: { // G_River
                long v487 = v8.case3.v0; static_array<static_array<unsigned char,2l>,2l> v488 = v8.case3.v1; static_array<long,2l> v489 = v8.case3.v2; long v490 = v8.case3.v3; static_array<long,2l> v491 = v8.case3.v4; Union6 v492 = v8.case3.v5;
                static_array<unsigned char,1l> v493; unsigned long long v494;
                Tuple14 tmp28 = draw_cards_41(v2, v6);
                v493 = tmp28.v0; v494 = tmp28.v1;
                v0 = v494;
                static_array_list<unsigned char,5l> v495;
                v495 = get_community_cards_42(v492, v493);
                Union3 v496;
                v496 = Union3{Union3_0{v495}};
                v5.push(v496);
                Union6 v508;
                switch (v492.tag) {
                    case 3: { // Turn
                        static_array<unsigned char,4l> v497 = v492.case3.v0;
                        static_array<unsigned char,5l> v498;
                        long v499;
                        v499 = 0l;
                        while (while_method_4(v499)){
                            unsigned char v501;
                            v501 = v497[v499];
                            v498[v499] = v501;
                            v499 += 1l ;
                        }
                        long v502;
                        v502 = 0l;
                        while (while_method_6(v502)){
                            unsigned char v504;
                            v504 = v493[v502];
                            long v505;
                            v505 = 4l + v502;
                            v498[v505] = v504;
                            v502 += 1l ;
                        }
                        v508 = Union6{Union6_2{v498}};
                        break;
                    }
                    default: {
                        printf("%s\n", "Invalid street in river.");
                        asm("exit;");
                    }
                }
                long v509;
                v509 = 2l;
                long v510;
                v510 = 0l;
                Union5 v511;
                v511 = try_round_37(v509, v488, v489, v510, v491, v508);
                v543 = true; v544 = v511;
                break;
            }
            case 4: { // G_Round
                long v84 = v8.case4.v0; static_array<static_array<unsigned char,2l>,2l> v85 = v8.case4.v1; static_array<long,2l> v86 = v8.case4.v2; long v87 = v8.case4.v3; static_array<long,2l> v88 = v8.case4.v4; Union6 v89 = v8.case4.v5;
                long v90;
                v90 = v87 % 2l;
                Union2 v91;
                v91 = v3[v90];
                switch (v91.tag) {
                    case 0: { // Computer
                        static_array<long,2l> v92;
                        long v93;
                        v93 = 0l;
                        while (while_method_0(v93)){
                            long v95;
                            v95 = v88[v93];
                            long v96;
                            v96 = v86[v93];
                            long v97;
                            v97 = v95 + v96;
                            v92[v93] = v97;
                            v93 += 1l ;
                        }
                        long v98;
                        v98 = v86[0l];
                        long v99; long v100;
                        Tuple2 tmp29 = Tuple2{1l, v98};
                        v99 = tmp29.v0; v100 = tmp29.v1;
                        while (while_method_0(v99)){
                            long v102;
                            v102 = v86[v99];
                            bool v103;
                            v103 = v100 >= v102;
                            long v104;
                            if (v103){
                                v104 = v100;
                            } else {
                                v104 = v102;
                            }
                            v100 = v104;
                            v99 += 1l ;
                        }
                        long v105;
                        v105 = v92[v90];
                        bool v106;
                        v106 = v100 < v105;
                        long v107;
                        if (v106){
                            v107 = v100;
                        } else {
                            v107 = v105;
                        }
                        static_array<long,2l> v108;
                        long v109;
                        v109 = 0l;
                        while (while_method_0(v109)){
                            long v111;
                            v111 = v86[v109];
                            bool v112;
                            v112 = v90 == v109;
                            long v113;
                            if (v112){
                                v113 = v107;
                            } else {
                                v113 = v111;
                            }
                            v108[v109] = v113;
                            v109 += 1l ;
                        }
                        static_array<long,2l> v114;
                        long v115;
                        v115 = 0l;
                        while (while_method_0(v115)){
                            long v117;
                            v117 = v92[v115];
                            long v118;
                            v118 = v108[v115];
                            long v119;
                            v119 = v117 - v118;
                            v114[v115] = v119;
                            v115 += 1l ;
                        }
                        long v120;
                        v120 = v108[0l];
                        long v121; long v122;
                        Tuple2 tmp30 = Tuple2{1l, v120};
                        v121 = tmp30.v0; v122 = tmp30.v1;
                        while (while_method_0(v121)){
                            long v124;
                            v124 = v108[v121];
                            long v125;
                            v125 = v122 + v124;
                            v122 = v125;
                            v121 += 1l ;
                        }
                        long v126;
                        v126 = v86[v90];
                        bool v127;
                        v127 = v126 < v100;
                        float v128;
                        if (v127){
                            v128 = 1.0f;
                        } else {
                            v128 = 0.0f;
                        }
                        long v129;
                        v129 = v122 / 3l;
                        bool v130;
                        v130 = v84 <= v129;
                        bool v133;
                        if (v130){
                            long v131;
                            v131 = v114[v90];
                            bool v132;
                            v132 = v129 < v131;
                            v133 = v132;
                        } else {
                            v133 = false;
                        }
                        float v134;
                        if (v133){
                            v134 = 1.0f;
                        } else {
                            v134 = 0.0f;
                        }
                        long v135;
                        v135 = v122 / 2l;
                        bool v136;
                        v136 = v84 <= v135;
                        bool v139;
                        if (v136){
                            long v137;
                            v137 = v114[v90];
                            bool v138;
                            v138 = v135 < v137;
                            v139 = v138;
                        } else {
                            v139 = false;
                        }
                        float v140;
                        if (v139){
                            v140 = 1.0f;
                        } else {
                            v140 = 0.0f;
                        }
                        bool v141;
                        v141 = v84 <= v122;
                        bool v144;
                        if (v141){
                            long v142;
                            v142 = v114[v90];
                            bool v143;
                            v143 = v122 < v142;
                            v144 = v143;
                        } else {
                            v144 = false;
                        }
                        float v145;
                        if (v144){
                            v145 = 1.0f;
                        } else {
                            v145 = 0.0f;
                        }
                        static_array<Tuple15,6l> v146;
                        Union1 v147;
                        v147 = Union1{Union1_2{}};
                        v146[0l] = Tuple15{v147, v128};
                        Union1 v148;
                        v148 = Union1{Union1_1{}};
                        v146[1l] = Tuple15{v148, 4.0f};
                        Union1 v149;
                        v149 = Union1{Union1_3{v129}};
                        v146[2l] = Tuple15{v149, v134};
                        Union1 v150;
                        v150 = Union1{Union1_3{v135}};
                        v146[3l] = Tuple15{v150, v140};
                        Union1 v151;
                        v151 = Union1{Union1_3{v122}};
                        v146[4l] = Tuple15{v151, v145};
                        Union1 v152;
                        v152 = Union1{Union1_0{}};
                        v146[5l] = Tuple15{v152, 1.0f};
                        Union1 v153;
                        v153 = sample_discrete_43(v146, v2);
                        Union3 v154;
                        v154 = Union3{Union3_2{v90, v153}};
                        v5.push(v154);
                        Union5 v293;
                        switch (v153.tag) {
                            case 0: { // A_All_In
                                static_array<long,2l> v241;
                                long v242;
                                v242 = 0l;
                                while (while_method_0(v242)){
                                    long v244;
                                    v244 = v88[v242];
                                    long v245;
                                    v245 = v86[v242];
                                    long v246;
                                    v246 = v244 + v245;
                                    v241[v242] = v246;
                                    v242 += 1l ;
                                }
                                long v247;
                                v247 = v86[0l];
                                long v248; long v249;
                                Tuple2 tmp33 = Tuple2{1l, v247};
                                v248 = tmp33.v0; v249 = tmp33.v1;
                                while (while_method_0(v248)){
                                    long v251;
                                    v251 = v86[v248];
                                    bool v252;
                                    v252 = v249 >= v251;
                                    long v253;
                                    if (v252){
                                        v253 = v249;
                                    } else {
                                        v253 = v251;
                                    }
                                    v249 = v253;
                                    v248 += 1l ;
                                }
                                long v254;
                                v254 = v241[v90];
                                bool v255;
                                v255 = v249 < v254;
                                long v256;
                                if (v255){
                                    v256 = v249;
                                } else {
                                    v256 = v254;
                                }
                                static_array<long,2l> v257;
                                long v258;
                                v258 = 0l;
                                while (while_method_0(v258)){
                                    long v260;
                                    v260 = v86[v258];
                                    bool v261;
                                    v261 = v90 == v258;
                                    long v262;
                                    if (v261){
                                        v262 = v256;
                                    } else {
                                        v262 = v260;
                                    }
                                    v257[v258] = v262;
                                    v258 += 1l ;
                                }
                                static_array<long,2l> v263;
                                long v264;
                                v264 = 0l;
                                while (while_method_0(v264)){
                                    long v266;
                                    v266 = v241[v264];
                                    long v267;
                                    v267 = v257[v264];
                                    long v268;
                                    v268 = v266 - v267;
                                    v263[v264] = v268;
                                    v264 += 1l ;
                                }
                                long v269;
                                v269 = v263[v90];
                                long v270;
                                v270 = v249 + v269;
                                long v271;
                                v271 = v241[v90];
                                bool v272;
                                v272 = v270 < v271;
                                long v273;
                                if (v272){
                                    v273 = v270;
                                } else {
                                    v273 = v271;
                                }
                                static_array<long,2l> v274;
                                long v275;
                                v275 = 0l;
                                while (while_method_0(v275)){
                                    long v277;
                                    v277 = v86[v275];
                                    bool v278;
                                    v278 = v90 == v275;
                                    long v279;
                                    if (v278){
                                        v279 = v273;
                                    } else {
                                        v279 = v277;
                                    }
                                    v274[v275] = v279;
                                    v275 += 1l ;
                                }
                                static_array<long,2l> v280;
                                long v281;
                                v281 = 0l;
                                while (while_method_0(v281)){
                                    long v283;
                                    v283 = v241[v281];
                                    long v284;
                                    v284 = v274[v281];
                                    long v285;
                                    v285 = v283 - v284;
                                    v280[v281] = v285;
                                    v281 += 1l ;
                                }
                                bool v286;
                                v286 = v269 >= v84;
                                long v287;
                                if (v286){
                                    v287 = v269;
                                } else {
                                    v287 = v84;
                                }
                                long v288;
                                v288 = v87 + 1l;
                                v293 = try_round_37(v287, v85, v274, v288, v280, v89);
                                break;
                            }
                            case 1: { // A_Call
                                static_array<long,2l> v156;
                                long v157;
                                v157 = 0l;
                                while (while_method_0(v157)){
                                    long v159;
                                    v159 = v88[v157];
                                    long v160;
                                    v160 = v86[v157];
                                    long v161;
                                    v161 = v159 + v160;
                                    v156[v157] = v161;
                                    v157 += 1l ;
                                }
                                long v162;
                                v162 = v86[0l];
                                long v163; long v164;
                                Tuple2 tmp34 = Tuple2{1l, v162};
                                v163 = tmp34.v0; v164 = tmp34.v1;
                                while (while_method_0(v163)){
                                    long v166;
                                    v166 = v86[v163];
                                    bool v167;
                                    v167 = v164 >= v166;
                                    long v168;
                                    if (v167){
                                        v168 = v164;
                                    } else {
                                        v168 = v166;
                                    }
                                    v164 = v168;
                                    v163 += 1l ;
                                }
                                long v169;
                                v169 = v156[v90];
                                bool v170;
                                v170 = v164 < v169;
                                long v171;
                                if (v170){
                                    v171 = v164;
                                } else {
                                    v171 = v169;
                                }
                                static_array<long,2l> v172;
                                long v173;
                                v173 = 0l;
                                while (while_method_0(v173)){
                                    long v175;
                                    v175 = v86[v173];
                                    bool v176;
                                    v176 = v90 == v173;
                                    long v177;
                                    if (v176){
                                        v177 = v171;
                                    } else {
                                        v177 = v175;
                                    }
                                    v172[v173] = v177;
                                    v173 += 1l ;
                                }
                                static_array<long,2l> v178;
                                long v179;
                                v179 = 0l;
                                while (while_method_0(v179)){
                                    long v181;
                                    v181 = v156[v179];
                                    long v182;
                                    v182 = v172[v179];
                                    long v183;
                                    v183 = v181 - v182;
                                    v178[v179] = v183;
                                    v179 += 1l ;
                                }
                                bool v184;
                                v184 = v90 < 2l;
                                if (v184){
                                    long v185;
                                    v185 = v87 + 1l;
                                    v293 = try_round_37(v84, v85, v172, v185, v178, v89);
                                } else {
                                    v293 = go_next_street_39(v84, v85, v172, v87, v178, v89);
                                }
                                break;
                            }
                            case 2: { // A_Fold
                                v293 = Union5{Union5_1{v84, v85, v86, v87, v88, v89}};
                                break;
                            }
                            case 3: { // A_Raise
                                long v189 = v153.case3.v0;
                                bool v190;
                                v190 = v84 <= v189;
                                bool v191;
                                v191 = v190 == false;
                                if (v191){
                                    assert("The raise amount must match the minimum." && v190);
                                } else {
                                }
                                static_array<long,2l> v192;
                                long v193;
                                v193 = 0l;
                                while (while_method_0(v193)){
                                    long v195;
                                    v195 = v88[v193];
                                    long v196;
                                    v196 = v86[v193];
                                    long v197;
                                    v197 = v195 + v196;
                                    v192[v193] = v197;
                                    v193 += 1l ;
                                }
                                long v198;
                                v198 = v86[0l];
                                long v199; long v200;
                                Tuple2 tmp35 = Tuple2{1l, v198};
                                v199 = tmp35.v0; v200 = tmp35.v1;
                                while (while_method_0(v199)){
                                    long v202;
                                    v202 = v86[v199];
                                    bool v203;
                                    v203 = v200 >= v202;
                                    long v204;
                                    if (v203){
                                        v204 = v200;
                                    } else {
                                        v204 = v202;
                                    }
                                    v200 = v204;
                                    v199 += 1l ;
                                }
                                long v205;
                                v205 = v192[v90];
                                bool v206;
                                v206 = v200 < v205;
                                long v207;
                                if (v206){
                                    v207 = v200;
                                } else {
                                    v207 = v205;
                                }
                                static_array<long,2l> v208;
                                long v209;
                                v209 = 0l;
                                while (while_method_0(v209)){
                                    long v211;
                                    v211 = v86[v209];
                                    bool v212;
                                    v212 = v90 == v209;
                                    long v213;
                                    if (v212){
                                        v213 = v207;
                                    } else {
                                        v213 = v211;
                                    }
                                    v208[v209] = v213;
                                    v209 += 1l ;
                                }
                                static_array<long,2l> v214;
                                long v215;
                                v215 = 0l;
                                while (while_method_0(v215)){
                                    long v217;
                                    v217 = v192[v215];
                                    long v218;
                                    v218 = v208[v215];
                                    long v219;
                                    v219 = v217 - v218;
                                    v214[v215] = v219;
                                    v215 += 1l ;
                                }
                                long v220;
                                v220 = v214[v90];
                                bool v221;
                                v221 = v189 < v220;
                                bool v222;
                                v222 = v221 == false;
                                if (v222){
                                    assert("The raise amount must be less than the stack size after calling." && v221);
                                } else {
                                }
                                long v223;
                                v223 = v200 + v189;
                                long v224;
                                v224 = v192[v90];
                                bool v225;
                                v225 = v223 < v224;
                                long v226;
                                if (v225){
                                    v226 = v223;
                                } else {
                                    v226 = v224;
                                }
                                static_array<long,2l> v227;
                                long v228;
                                v228 = 0l;
                                while (while_method_0(v228)){
                                    long v230;
                                    v230 = v86[v228];
                                    bool v231;
                                    v231 = v90 == v228;
                                    long v232;
                                    if (v231){
                                        v232 = v226;
                                    } else {
                                        v232 = v230;
                                    }
                                    v227[v228] = v232;
                                    v228 += 1l ;
                                }
                                static_array<long,2l> v233;
                                long v234;
                                v234 = 0l;
                                while (while_method_0(v234)){
                                    long v236;
                                    v236 = v192[v234];
                                    long v237;
                                    v237 = v227[v234];
                                    long v238;
                                    v238 = v236 - v237;
                                    v233[v234] = v238;
                                    v234 += 1l ;
                                }
                                long v239;
                                v239 = v87 + 1l;
                                v293 = try_round_37(v189, v85, v227, v239, v233, v89);
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                            }
                        }
                        v543 = true; v544 = v293;
                        break;
                    }
                    case 1: { // Human
                        v543 = false; v544 = v8;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                break;
            }
            case 5: { // G_Round'
                long v298 = v8.case5.v0; static_array<static_array<unsigned char,2l>,2l> v299 = v8.case5.v1; static_array<long,2l> v300 = v8.case5.v2; long v301 = v8.case5.v3; static_array<long,2l> v302 = v8.case5.v4; Union6 v303 = v8.case5.v5; Union1 v304 = v8.case5.v6;
                long v305;
                v305 = v301 % 2l;
                Union3 v306;
                v306 = Union3{Union3_2{v305, v304}};
                v5.push(v306);
                Union5 v445;
                switch (v304.tag) {
                    case 0: { // A_All_In
                        static_array<long,2l> v393;
                        long v394;
                        v394 = 0l;
                        while (while_method_0(v394)){
                            long v396;
                            v396 = v302[v394];
                            long v397;
                            v397 = v300[v394];
                            long v398;
                            v398 = v396 + v397;
                            v393[v394] = v398;
                            v394 += 1l ;
                        }
                        long v399;
                        v399 = v300[0l];
                        long v400; long v401;
                        Tuple2 tmp36 = Tuple2{1l, v399};
                        v400 = tmp36.v0; v401 = tmp36.v1;
                        while (while_method_0(v400)){
                            long v403;
                            v403 = v300[v400];
                            bool v404;
                            v404 = v401 >= v403;
                            long v405;
                            if (v404){
                                v405 = v401;
                            } else {
                                v405 = v403;
                            }
                            v401 = v405;
                            v400 += 1l ;
                        }
                        long v406;
                        v406 = v393[v305];
                        bool v407;
                        v407 = v401 < v406;
                        long v408;
                        if (v407){
                            v408 = v401;
                        } else {
                            v408 = v406;
                        }
                        static_array<long,2l> v409;
                        long v410;
                        v410 = 0l;
                        while (while_method_0(v410)){
                            long v412;
                            v412 = v300[v410];
                            bool v413;
                            v413 = v305 == v410;
                            long v414;
                            if (v413){
                                v414 = v408;
                            } else {
                                v414 = v412;
                            }
                            v409[v410] = v414;
                            v410 += 1l ;
                        }
                        static_array<long,2l> v415;
                        long v416;
                        v416 = 0l;
                        while (while_method_0(v416)){
                            long v418;
                            v418 = v393[v416];
                            long v419;
                            v419 = v409[v416];
                            long v420;
                            v420 = v418 - v419;
                            v415[v416] = v420;
                            v416 += 1l ;
                        }
                        long v421;
                        v421 = v415[v305];
                        long v422;
                        v422 = v401 + v421;
                        long v423;
                        v423 = v393[v305];
                        bool v424;
                        v424 = v422 < v423;
                        long v425;
                        if (v424){
                            v425 = v422;
                        } else {
                            v425 = v423;
                        }
                        static_array<long,2l> v426;
                        long v427;
                        v427 = 0l;
                        while (while_method_0(v427)){
                            long v429;
                            v429 = v300[v427];
                            bool v430;
                            v430 = v305 == v427;
                            long v431;
                            if (v430){
                                v431 = v425;
                            } else {
                                v431 = v429;
                            }
                            v426[v427] = v431;
                            v427 += 1l ;
                        }
                        static_array<long,2l> v432;
                        long v433;
                        v433 = 0l;
                        while (while_method_0(v433)){
                            long v435;
                            v435 = v393[v433];
                            long v436;
                            v436 = v426[v433];
                            long v437;
                            v437 = v435 - v436;
                            v432[v433] = v437;
                            v433 += 1l ;
                        }
                        bool v438;
                        v438 = v421 >= v298;
                        long v439;
                        if (v438){
                            v439 = v421;
                        } else {
                            v439 = v298;
                        }
                        long v440;
                        v440 = v301 + 1l;
                        v445 = try_round_37(v439, v299, v426, v440, v432, v303);
                        break;
                    }
                    case 1: { // A_Call
                        static_array<long,2l> v308;
                        long v309;
                        v309 = 0l;
                        while (while_method_0(v309)){
                            long v311;
                            v311 = v302[v309];
                            long v312;
                            v312 = v300[v309];
                            long v313;
                            v313 = v311 + v312;
                            v308[v309] = v313;
                            v309 += 1l ;
                        }
                        long v314;
                        v314 = v300[0l];
                        long v315; long v316;
                        Tuple2 tmp37 = Tuple2{1l, v314};
                        v315 = tmp37.v0; v316 = tmp37.v1;
                        while (while_method_0(v315)){
                            long v318;
                            v318 = v300[v315];
                            bool v319;
                            v319 = v316 >= v318;
                            long v320;
                            if (v319){
                                v320 = v316;
                            } else {
                                v320 = v318;
                            }
                            v316 = v320;
                            v315 += 1l ;
                        }
                        long v321;
                        v321 = v308[v305];
                        bool v322;
                        v322 = v316 < v321;
                        long v323;
                        if (v322){
                            v323 = v316;
                        } else {
                            v323 = v321;
                        }
                        static_array<long,2l> v324;
                        long v325;
                        v325 = 0l;
                        while (while_method_0(v325)){
                            long v327;
                            v327 = v300[v325];
                            bool v328;
                            v328 = v305 == v325;
                            long v329;
                            if (v328){
                                v329 = v323;
                            } else {
                                v329 = v327;
                            }
                            v324[v325] = v329;
                            v325 += 1l ;
                        }
                        static_array<long,2l> v330;
                        long v331;
                        v331 = 0l;
                        while (while_method_0(v331)){
                            long v333;
                            v333 = v308[v331];
                            long v334;
                            v334 = v324[v331];
                            long v335;
                            v335 = v333 - v334;
                            v330[v331] = v335;
                            v331 += 1l ;
                        }
                        bool v336;
                        v336 = v305 < 2l;
                        if (v336){
                            long v337;
                            v337 = v301 + 1l;
                            v445 = try_round_37(v298, v299, v324, v337, v330, v303);
                        } else {
                            v445 = go_next_street_39(v298, v299, v324, v301, v330, v303);
                        }
                        break;
                    }
                    case 2: { // A_Fold
                        v445 = Union5{Union5_1{v298, v299, v300, v301, v302, v303}};
                        break;
                    }
                    case 3: { // A_Raise
                        long v341 = v304.case3.v0;
                        bool v342;
                        v342 = v298 <= v341;
                        bool v343;
                        v343 = v342 == false;
                        if (v343){
                            assert("The raise amount must match the minimum." && v342);
                        } else {
                        }
                        static_array<long,2l> v344;
                        long v345;
                        v345 = 0l;
                        while (while_method_0(v345)){
                            long v347;
                            v347 = v302[v345];
                            long v348;
                            v348 = v300[v345];
                            long v349;
                            v349 = v347 + v348;
                            v344[v345] = v349;
                            v345 += 1l ;
                        }
                        long v350;
                        v350 = v300[0l];
                        long v351; long v352;
                        Tuple2 tmp38 = Tuple2{1l, v350};
                        v351 = tmp38.v0; v352 = tmp38.v1;
                        while (while_method_0(v351)){
                            long v354;
                            v354 = v300[v351];
                            bool v355;
                            v355 = v352 >= v354;
                            long v356;
                            if (v355){
                                v356 = v352;
                            } else {
                                v356 = v354;
                            }
                            v352 = v356;
                            v351 += 1l ;
                        }
                        long v357;
                        v357 = v344[v305];
                        bool v358;
                        v358 = v352 < v357;
                        long v359;
                        if (v358){
                            v359 = v352;
                        } else {
                            v359 = v357;
                        }
                        static_array<long,2l> v360;
                        long v361;
                        v361 = 0l;
                        while (while_method_0(v361)){
                            long v363;
                            v363 = v300[v361];
                            bool v364;
                            v364 = v305 == v361;
                            long v365;
                            if (v364){
                                v365 = v359;
                            } else {
                                v365 = v363;
                            }
                            v360[v361] = v365;
                            v361 += 1l ;
                        }
                        static_array<long,2l> v366;
                        long v367;
                        v367 = 0l;
                        while (while_method_0(v367)){
                            long v369;
                            v369 = v344[v367];
                            long v370;
                            v370 = v360[v367];
                            long v371;
                            v371 = v369 - v370;
                            v366[v367] = v371;
                            v367 += 1l ;
                        }
                        long v372;
                        v372 = v366[v305];
                        bool v373;
                        v373 = v341 < v372;
                        bool v374;
                        v374 = v373 == false;
                        if (v374){
                            assert("The raise amount must be less than the stack size after calling." && v373);
                        } else {
                        }
                        long v375;
                        v375 = v352 + v341;
                        long v376;
                        v376 = v344[v305];
                        bool v377;
                        v377 = v375 < v376;
                        long v378;
                        if (v377){
                            v378 = v375;
                        } else {
                            v378 = v376;
                        }
                        static_array<long,2l> v379;
                        long v380;
                        v380 = 0l;
                        while (while_method_0(v380)){
                            long v382;
                            v382 = v300[v380];
                            bool v383;
                            v383 = v305 == v380;
                            long v384;
                            if (v383){
                                v384 = v378;
                            } else {
                                v384 = v382;
                            }
                            v379[v380] = v384;
                            v380 += 1l ;
                        }
                        static_array<long,2l> v385;
                        long v386;
                        v386 = 0l;
                        while (while_method_0(v386)){
                            long v388;
                            v388 = v344[v386];
                            long v389;
                            v389 = v379[v386];
                            long v390;
                            v390 = v388 - v389;
                            v385[v386] = v390;
                            v386 += 1l ;
                        }
                        long v391;
                        v391 = v301 + 1l;
                        v445 = try_round_37(v341, v299, v379, v391, v385, v303);
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                v543 = true; v544 = v445;
                break;
            }
            case 6: { // G_Showdown
                long v21 = v8.case6.v0; static_array<static_array<unsigned char,2l>,2l> v22 = v8.case6.v1; static_array<long,2l> v23 = v8.case6.v2; long v24 = v8.case6.v3; static_array<long,2l> v25 = v8.case6.v4; Union6 v26 = v8.case6.v5;
                static_array<unsigned char,5l> v29;
                switch (v26.tag) {
                    case 2: { // River
                        static_array<unsigned char,5l> v27 = v26.case2.v0;
                        v29 = v27;
                        break;
                    }
                    default: {
                        printf("%s\n", "Invalid street in showdown.");
                        asm("exit;");
                    }
                }
                static_array<unsigned char,2l> v30;
                v30 = v22[0l];
                static_array<unsigned char,7l> v31;
                long v32;
                v32 = 0l;
                while (while_method_0(v32)){
                    unsigned char v34;
                    v34 = v30[v32];
                    v31[v32] = v34;
                    v32 += 1l ;
                }
                long v35;
                v35 = 0l;
                while (while_method_2(v35)){
                    unsigned char v37;
                    v37 = v29[v35];
                    long v38;
                    v38 = 2l + v35;
                    v31[v38] = v37;
                    v35 += 1l ;
                }
                static_array<unsigned char,5l> v39; char v40;
                Tuple0 tmp63 = score_46(v31);
                v39 = tmp63.v0; v40 = tmp63.v1;
                static_array<unsigned char,2l> v41;
                v41 = v22[1l];
                static_array<unsigned char,7l> v42;
                long v43;
                v43 = 0l;
                while (while_method_0(v43)){
                    unsigned char v45;
                    v45 = v41[v43];
                    v42[v43] = v45;
                    v43 += 1l ;
                }
                long v46;
                v46 = 0l;
                while (while_method_2(v46)){
                    unsigned char v48;
                    v48 = v29[v46];
                    long v49;
                    v49 = 2l + v46;
                    v42[v49] = v48;
                    v46 += 1l ;
                }
                static_array<unsigned char,5l> v50; char v51;
                Tuple0 tmp64 = score_46(v42);
                v50 = tmp64.v0; v51 = tmp64.v1;
                long v52;
                v52 = v24 % 2l;
                long v53;
                v53 = v23[v52];
                bool v54;
                v54 = v40 < v51;
                Union8 v60;
                if (v54){
                    v60 = Union8{Union8_2{}};
                } else {
                    bool v56;
                    v56 = v40 > v51;
                    if (v56){
                        v60 = Union8{Union8_1{}};
                    } else {
                        v60 = Union8{Union8_0{}};
                    }
                }
                Union8 v75;
                switch (v60.tag) {
                    case 0: { // Eq
                        Union8 v61;
                        v61 = Union8{Union8_0{}};
                        long v62;
                        v62 = 0l;
                        while (while_method_2(v62)){
                            unsigned char v64;
                            v64 = v39[v62];
                            unsigned char v65;
                            v65 = v50[v62];
                            bool v66;
                            v66 = v64 < v65;
                            Union8 v72;
                            if (v66){
                                v72 = Union8{Union8_2{}};
                            } else {
                                bool v68;
                                v68 = v64 > v65;
                                if (v68){
                                    v72 = Union8{Union8_1{}};
                                } else {
                                    v72 = Union8{Union8_0{}};
                                }
                            }
                            bool v73;
                            switch (v72.tag) {
                                case 0: { // Eq
                                    v73 = true;
                                    break;
                                }
                                default: {
                                    v73 = false;
                                }
                            }
                            bool v74;
                            v74 = v73 == false;
                            if (v74){
                                v61 = v72;
                                break;
                            } else {
                            }
                            v62 += 1l ;
                        }
                        v75 = v61;
                        break;
                    }
                    default: {
                        v75 = v60;
                    }
                }
                long v80; long v81;
                switch (v75.tag) {
                    case 0: { // Eq
                        v80 = 0l; v81 = -1l;
                        break;
                    }
                    case 1: { // Gt
                        v80 = v53; v81 = 0l;
                        break;
                    }
                    case 2: { // Lt
                        v80 = v53; v81 = 1l;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                static_array<Tuple0,2l> v82;
                v82[0l] = Tuple0{v39, v40};
                v82[1l] = Tuple0{v50, v51};
                Union3 v83;
                v83 = Union3{Union3_4{v80, v82, v81}};
                v5.push(v83);
                v543 = false; v544 = v8;
                break;
            }
            case 7: { // G_Turn
                long v462 = v8.case7.v0; static_array<static_array<unsigned char,2l>,2l> v463 = v8.case7.v1; static_array<long,2l> v464 = v8.case7.v2; long v465 = v8.case7.v3; static_array<long,2l> v466 = v8.case7.v4; Union6 v467 = v8.case7.v5;
                static_array<unsigned char,1l> v468; unsigned long long v469;
                Tuple14 tmp65 = draw_cards_41(v2, v6);
                v468 = tmp65.v0; v469 = tmp65.v1;
                v0 = v469;
                static_array_list<unsigned char,5l> v470;
                v470 = get_community_cards_42(v467, v468);
                Union3 v471;
                v471 = Union3{Union3_0{v470}};
                v5.push(v471);
                Union6 v483;
                switch (v467.tag) {
                    case 0: { // Flop
                        static_array<unsigned char,3l> v472 = v467.case0.v0;
                        static_array<unsigned char,4l> v473;
                        long v474;
                        v474 = 0l;
                        while (while_method_3(v474)){
                            unsigned char v476;
                            v476 = v472[v474];
                            v473[v474] = v476;
                            v474 += 1l ;
                        }
                        long v477;
                        v477 = 0l;
                        while (while_method_6(v477)){
                            unsigned char v479;
                            v479 = v468[v477];
                            long v480;
                            v480 = 3l + v477;
                            v473[v480] = v479;
                            v477 += 1l ;
                        }
                        v483 = Union6{Union6_3{v473}};
                        break;
                    }
                    default: {
                        printf("%s\n", "Invalid street in turn.");
                        asm("exit;");
                    }
                }
                long v484;
                v484 = 2l;
                long v485;
                v485 = 0l;
                Union5 v486;
                v486 = try_round_37(v484, v463, v464, v485, v466, v483);
                v543 = true; v544 = v486;
                break;
            }
            default: {
                assert("Invalid tag." && false);
            }
        }
        v7 = v543;
        v8 = v544;
    }
    return v8;
}
__device__ Tuple8 play_loop_31(Union4 v0, static_array<Union2,2l> v1, Union7 v2, unsigned long long & v3, dynamic_array_list<Union3,128l> & v4, curandStatePhilox4_32_10_t & v5, Union5 v6){
    Union5 v7;
    v7 = play_loop_inner_32(v3, v4, v5, v1, v6);
    switch (v7.tag) {
        case 1: { // G_Fold
            long v24 = v7.case1.v0; static_array<static_array<unsigned char,2l>,2l> v25 = v7.case1.v1; static_array<long,2l> v26 = v7.case1.v2; long v27 = v7.case1.v3; static_array<long,2l> v28 = v7.case1.v4; Union6 v29 = v7.case1.v5;
            Union4 v30;
            v30 = Union4{Union4_0{}};
            Union7 v31;
            v31 = Union7{Union7_1{v24, v25, v26, v27, v28, v29}};
            return Tuple8{v30, v1, v31};
            break;
        }
        case 4: { // G_Round
            long v8 = v7.case4.v0; static_array<static_array<unsigned char,2l>,2l> v9 = v7.case4.v1; static_array<long,2l> v10 = v7.case4.v2; long v11 = v7.case4.v3; static_array<long,2l> v12 = v7.case4.v4; Union6 v13 = v7.case4.v5;
            Union4 v14;
            v14 = Union4{Union4_1{v7}};
            Union7 v15;
            v15 = Union7{Union7_2{v8, v9, v10, v11, v12, v13}};
            return Tuple8{v14, v1, v15};
            break;
        }
        case 6: { // G_Showdown
            long v16 = v7.case6.v0; static_array<static_array<unsigned char,2l>,2l> v17 = v7.case6.v1; static_array<long,2l> v18 = v7.case6.v2; long v19 = v7.case6.v3; static_array<long,2l> v20 = v7.case6.v4; Union6 v21 = v7.case6.v5;
            Union4 v22;
            v22 = Union4{Union4_0{}};
            Union7 v23;
            v23 = Union7{Union7_1{v16, v17, v18, v19, v20, v21}};
            return Tuple8{v22, v1, v23};
            break;
        }
        default: {
            printf("%s\n", "Unexpected node received in play_loop.");
            asm("exit;");
        }
    }
}
__device__ void f_48(unsigned char * v0, unsigned long long v1){
    unsigned long long * v2;
    v2 = (unsigned long long *)(v0+0ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_49(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+8ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_51(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+0ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_54(unsigned char * v0, unsigned char v1){
    unsigned char * v2;
    v2 = (unsigned char *)(v0+0ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_53(unsigned char * v0, unsigned char v1){
    return f_54(v0, v1);
}
__device__ void f_52(unsigned char * v0, static_array_list<unsigned char,5l> v1){
    long v2;
    v2 = v1.length;
    f_51(v0, v2);
    long v3;
    v3 = v1.length;
    long v4;
    v4 = 0l;
    while (while_method_1(v3, v4)){
        unsigned long long v6;
        v6 = (unsigned long long)v4;
        unsigned long long v7;
        v7 = 4ull + v6;
        unsigned char * v8;
        v8 = (unsigned char *)(v0+v7);
        unsigned char v9;
        v9 = v1[v4];
        f_53(v8, v9);
        v4 += 1l ;
    }
    return ;
}
__device__ void f_55(unsigned char * v0, long v1, long v2){
    long * v3;
    v3 = (long *)(v0+0ull);
    v3[0l] = v1;
    long * v4;
    v4 = (long *)(v0+4ull);
    v4[0l] = v2;
    return ;
}
__device__ void f_57(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+4ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_58(unsigned char * v0){
    return ;
}
__device__ void f_56(unsigned char * v0, long v1, Union1 v2){
    long * v3;
    v3 = (long *)(v0+0ull);
    v3[0l] = v1;
    long v4;
    v4 = v2.tag;
    f_57(v0, v4);
    unsigned char * v5;
    v5 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // A_All_In
            return f_58(v5);
            break;
        }
        case 1: { // A_Call
            return f_58(v5);
            break;
        }
        case 2: { // A_Fold
            return f_58(v5);
            break;
        }
        case 3: { // A_Raise
            long v6 = v2.case3.v0;
            return f_51(v5, v6);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_59(unsigned char * v0, long v1, static_array<unsigned char,2l> v2){
    long * v3;
    v3 = (long *)(v0+0ull);
    v3[0l] = v1;
    long v4;
    v4 = 0l;
    while (while_method_0(v4)){
        unsigned long long v6;
        v6 = (unsigned long long)v4;
        unsigned long long v7;
        v7 = 4ull + v6;
        unsigned char * v8;
        v8 = (unsigned char *)(v0+v7);
        unsigned char v9;
        v9 = v2[v4];
        f_53(v8, v9);
        v4 += 1l ;
    }
    return ;
}
__device__ void f_62(unsigned char * v0, static_array<unsigned char,5l> v1, char v2){
    long v3;
    v3 = 0l;
    while (while_method_2(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v7;
        v7 = v1[v3];
        f_53(v6, v7);
        v3 += 1l ;
    }
    char * v8;
    v8 = (char *)(v0+5ull);
    v8[0l] = v2;
    return ;
}
__device__ void f_61(unsigned char * v0, static_array<unsigned char,5l> v1, char v2){
    return f_62(v0, v1, v2);
}
__device__ void f_60(unsigned char * v0, long v1, static_array<Tuple0,2l> v2, long v3){
    long * v4;
    v4 = (long *)(v0+0ull);
    v4[0l] = v1;
    long v5;
    v5 = 0l;
    while (while_method_0(v5)){
        unsigned long long v7;
        v7 = (unsigned long long)v5;
        unsigned long long v8;
        v8 = v7 * 8ull;
        unsigned long long v9;
        v9 = 8ull + v8;
        unsigned char * v10;
        v10 = (unsigned char *)(v0+v9);
        static_array<unsigned char,5l> v11; char v12;
        Tuple0 tmp68 = v2[v5];
        v11 = tmp68.v0; v12 = tmp68.v1;
        f_61(v10, v11, v12);
        v5 += 1l ;
    }
    long * v13;
    v13 = (long *)(v0+24ull);
    v13[0l] = v3;
    return ;
}
__device__ void f_50(unsigned char * v0, Union3 v1){
    long v2;
    v2 = v1.tag;
    f_51(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // CommunityCardsAre
            static_array_list<unsigned char,5l> v4 = v1.case0.v0;
            return f_52(v3, v4);
            break;
        }
        case 1: { // Fold
            long v5 = v1.case1.v0; long v6 = v1.case1.v1;
            return f_55(v3, v5, v6);
            break;
        }
        case 2: { // PlayerAction
            long v7 = v1.case2.v0; Union1 v8 = v1.case2.v1;
            return f_56(v3, v7, v8);
            break;
        }
        case 3: { // PlayerGotCards
            long v9 = v1.case3.v0; static_array<unsigned char,2l> v10 = v1.case3.v1;
            return f_59(v3, v9, v10);
            break;
        }
        case 4: { // Showdown
            long v11 = v1.case4.v0; static_array<Tuple0,2l> v12 = v1.case4.v1; long v13 = v1.case4.v2;
            return f_60(v3, v11, v12, v13);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_63(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+6160ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_66(unsigned char * v0, static_array<unsigned char,2l> v1){
    long v2;
    v2 = 0l;
    while (while_method_0(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v6;
        v6 = v1[v2];
        f_53(v5, v6);
        v2 += 1l ;
    }
    return ;
}
__device__ void f_67(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+28ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_68(unsigned char * v0, static_array<unsigned char,3l> v1){
    long v2;
    v2 = 0l;
    while (while_method_3(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v6;
        v6 = v1[v2];
        f_53(v5, v6);
        v2 += 1l ;
    }
    return ;
}
__device__ void f_69(unsigned char * v0, static_array<unsigned char,5l> v1){
    long v2;
    v2 = 0l;
    while (while_method_2(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v6;
        v6 = v1[v2];
        f_53(v5, v6);
        v2 += 1l ;
    }
    return ;
}
__device__ void f_70(unsigned char * v0, static_array<unsigned char,4l> v1){
    long v2;
    v2 = 0l;
    while (while_method_4(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v6;
        v6 = v1[v2];
        f_53(v5, v6);
        v2 += 1l ;
    }
    return ;
}
__device__ void f_65(unsigned char * v0, long v1, static_array<static_array<unsigned char,2l>,2l> v2, static_array<long,2l> v3, long v4, static_array<long,2l> v5, Union6 v6){
    long * v7;
    v7 = (long *)(v0+0ull);
    v7[0l] = v1;
    long v8;
    v8 = 0l;
    while (while_method_0(v8)){
        unsigned long long v10;
        v10 = (unsigned long long)v8;
        unsigned long long v11;
        v11 = v10 * 2ull;
        unsigned long long v12;
        v12 = 4ull + v11;
        unsigned char * v13;
        v13 = (unsigned char *)(v0+v12);
        static_array<unsigned char,2l> v14;
        v14 = v2[v8];
        f_66(v13, v14);
        v8 += 1l ;
    }
    long v15;
    v15 = 0l;
    while (while_method_0(v15)){
        unsigned long long v17;
        v17 = (unsigned long long)v15;
        unsigned long long v18;
        v18 = v17 * 4ull;
        unsigned long long v19;
        v19 = 8ull + v18;
        unsigned char * v20;
        v20 = (unsigned char *)(v0+v19);
        long v21;
        v21 = v3[v15];
        f_51(v20, v21);
        v15 += 1l ;
    }
    long * v22;
    v22 = (long *)(v0+16ull);
    v22[0l] = v4;
    long v23;
    v23 = 0l;
    while (while_method_0(v23)){
        unsigned long long v25;
        v25 = (unsigned long long)v23;
        unsigned long long v26;
        v26 = v25 * 4ull;
        unsigned long long v27;
        v27 = 20ull + v26;
        unsigned char * v28;
        v28 = (unsigned char *)(v0+v27);
        long v29;
        v29 = v5[v23];
        f_51(v28, v29);
        v23 += 1l ;
    }
    long v30;
    v30 = v6.tag;
    f_67(v0, v30);
    unsigned char * v31;
    v31 = (unsigned char *)(v0+32ull);
    switch (v6.tag) {
        case 0: { // Flop
            static_array<unsigned char,3l> v32 = v6.case0.v0;
            return f_68(v31, v32);
            break;
        }
        case 1: { // Preflop
            return f_58(v31);
            break;
        }
        case 2: { // River
            static_array<unsigned char,5l> v33 = v6.case2.v0;
            return f_69(v31, v33);
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4l> v34 = v6.case3.v0;
            return f_70(v31, v34);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_72(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+40ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_71(unsigned char * v0, long v1, static_array<static_array<unsigned char,2l>,2l> v2, static_array<long,2l> v3, long v4, static_array<long,2l> v5, Union6 v6, Union1 v7){
    long * v8;
    v8 = (long *)(v0+0ull);
    v8[0l] = v1;
    long v9;
    v9 = 0l;
    while (while_method_0(v9)){
        unsigned long long v11;
        v11 = (unsigned long long)v9;
        unsigned long long v12;
        v12 = v11 * 2ull;
        unsigned long long v13;
        v13 = 4ull + v12;
        unsigned char * v14;
        v14 = (unsigned char *)(v0+v13);
        static_array<unsigned char,2l> v15;
        v15 = v2[v9];
        f_66(v14, v15);
        v9 += 1l ;
    }
    long v16;
    v16 = 0l;
    while (while_method_0(v16)){
        unsigned long long v18;
        v18 = (unsigned long long)v16;
        unsigned long long v19;
        v19 = v18 * 4ull;
        unsigned long long v20;
        v20 = 8ull + v19;
        unsigned char * v21;
        v21 = (unsigned char *)(v0+v20);
        long v22;
        v22 = v3[v16];
        f_51(v21, v22);
        v16 += 1l ;
    }
    long * v23;
    v23 = (long *)(v0+16ull);
    v23[0l] = v4;
    long v24;
    v24 = 0l;
    while (while_method_0(v24)){
        unsigned long long v26;
        v26 = (unsigned long long)v24;
        unsigned long long v27;
        v27 = v26 * 4ull;
        unsigned long long v28;
        v28 = 20ull + v27;
        unsigned char * v29;
        v29 = (unsigned char *)(v0+v28);
        long v30;
        v30 = v5[v24];
        f_51(v29, v30);
        v24 += 1l ;
    }
    long v31;
    v31 = v6.tag;
    f_67(v0, v31);
    unsigned char * v32;
    v32 = (unsigned char *)(v0+32ull);
    switch (v6.tag) {
        case 0: { // Flop
            static_array<unsigned char,3l> v33 = v6.case0.v0;
            f_68(v32, v33);
            break;
        }
        case 1: { // Preflop
            f_58(v32);
            break;
        }
        case 2: { // River
            static_array<unsigned char,5l> v34 = v6.case2.v0;
            f_69(v32, v34);
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4l> v35 = v6.case3.v0;
            f_70(v32, v35);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    long v36;
    v36 = v7.tag;
    f_72(v0, v36);
    unsigned char * v37;
    v37 = (unsigned char *)(v0+44ull);
    switch (v7.tag) {
        case 0: { // A_All_In
            return f_58(v37);
            break;
        }
        case 1: { // A_Call
            return f_58(v37);
            break;
        }
        case 2: { // A_Fold
            return f_58(v37);
            break;
        }
        case 3: { // A_Raise
            long v38 = v7.case3.v0;
            return f_51(v37, v38);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_64(unsigned char * v0, Union5 v1){
    long v2;
    v2 = v1.tag;
    f_51(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // G_Flop
            long v4 = v1.case0.v0; static_array<static_array<unsigned char,2l>,2l> v5 = v1.case0.v1; static_array<long,2l> v6 = v1.case0.v2; long v7 = v1.case0.v3; static_array<long,2l> v8 = v1.case0.v4; Union6 v9 = v1.case0.v5;
            return f_65(v3, v4, v5, v6, v7, v8, v9);
            break;
        }
        case 1: { // G_Fold
            long v10 = v1.case1.v0; static_array<static_array<unsigned char,2l>,2l> v11 = v1.case1.v1; static_array<long,2l> v12 = v1.case1.v2; long v13 = v1.case1.v3; static_array<long,2l> v14 = v1.case1.v4; Union6 v15 = v1.case1.v5;
            return f_65(v3, v10, v11, v12, v13, v14, v15);
            break;
        }
        case 2: { // G_Preflop
            return f_58(v3);
            break;
        }
        case 3: { // G_River
            long v16 = v1.case3.v0; static_array<static_array<unsigned char,2l>,2l> v17 = v1.case3.v1; static_array<long,2l> v18 = v1.case3.v2; long v19 = v1.case3.v3; static_array<long,2l> v20 = v1.case3.v4; Union6 v21 = v1.case3.v5;
            return f_65(v3, v16, v17, v18, v19, v20, v21);
            break;
        }
        case 4: { // G_Round
            long v22 = v1.case4.v0; static_array<static_array<unsigned char,2l>,2l> v23 = v1.case4.v1; static_array<long,2l> v24 = v1.case4.v2; long v25 = v1.case4.v3; static_array<long,2l> v26 = v1.case4.v4; Union6 v27 = v1.case4.v5;
            return f_65(v3, v22, v23, v24, v25, v26, v27);
            break;
        }
        case 5: { // G_Round'
            long v28 = v1.case5.v0; static_array<static_array<unsigned char,2l>,2l> v29 = v1.case5.v1; static_array<long,2l> v30 = v1.case5.v2; long v31 = v1.case5.v3; static_array<long,2l> v32 = v1.case5.v4; Union6 v33 = v1.case5.v5; Union1 v34 = v1.case5.v6;
            return f_71(v3, v28, v29, v30, v31, v32, v33, v34);
            break;
        }
        case 6: { // G_Showdown
            long v35 = v1.case6.v0; static_array<static_array<unsigned char,2l>,2l> v36 = v1.case6.v1; static_array<long,2l> v37 = v1.case6.v2; long v38 = v1.case6.v3; static_array<long,2l> v39 = v1.case6.v4; Union6 v40 = v1.case6.v5;
            return f_65(v3, v35, v36, v37, v38, v39, v40);
            break;
        }
        case 7: { // G_Turn
            long v41 = v1.case7.v0; static_array<static_array<unsigned char,2l>,2l> v42 = v1.case7.v1; static_array<long,2l> v43 = v1.case7.v2; long v44 = v1.case7.v3; static_array<long,2l> v45 = v1.case7.v4; Union6 v46 = v1.case7.v5;
            return f_65(v3, v41, v42, v43, v44, v45, v46);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_73(unsigned char * v0, Union2 v1){
    long v2;
    v2 = v1.tag;
    f_51(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // Computer
            return f_58(v3);
            break;
        }
        case 1: { // Human
            return f_58(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_74(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+6248ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_47(unsigned char * v0, unsigned long long v1, dynamic_array_list<Union3,128l> v2, Union4 v3, static_array<Union2,2l> v4, Union7 v5){
    f_48(v0, v1);
    long v6;
    v6 = v2.length_();
    f_49(v0, v6);
    long v7;
    v7 = v2.length_();
    long v8;
    v8 = 0l;
    while (while_method_1(v7, v8)){
        unsigned long long v10;
        v10 = (unsigned long long)v8;
        unsigned long long v11;
        v11 = v10 * 48ull;
        unsigned long long v12;
        v12 = 16ull + v11;
        unsigned char * v13;
        v13 = (unsigned char *)(v0+v12);
        Union3 v14;
        v14 = v2[v8];
        f_50(v13, v14);
        v8 += 1l ;
    }
    long v15;
    v15 = v3.tag;
    f_63(v0, v15);
    unsigned char * v16;
    v16 = (unsigned char *)(v0+6176ull);
    switch (v3.tag) {
        case 0: { // None
            f_58(v16);
            break;
        }
        case 1: { // Some
            Union5 v17 = v3.case1.v0;
            f_64(v16, v17);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    long v18;
    v18 = 0l;
    while (while_method_0(v18)){
        unsigned long long v20;
        v20 = (unsigned long long)v18;
        unsigned long long v21;
        v21 = v20 * 4ull;
        unsigned long long v22;
        v22 = 6240ull + v21;
        unsigned char * v23;
        v23 = (unsigned char *)(v0+v22);
        Union2 v24;
        v24 = v4[v18];
        f_73(v23, v24);
        v18 += 1l ;
    }
    long v25;
    v25 = v5.tag;
    f_74(v0, v25);
    unsigned char * v26;
    v26 = (unsigned char *)(v0+6256ull);
    switch (v5.tag) {
        case 0: { // GameNotStarted
            return f_58(v26);
            break;
        }
        case 1: { // GameOver
            long v27 = v5.case1.v0; static_array<static_array<unsigned char,2l>,2l> v28 = v5.case1.v1; static_array<long,2l> v29 = v5.case1.v2; long v30 = v5.case1.v3; static_array<long,2l> v31 = v5.case1.v4; Union6 v32 = v5.case1.v5;
            return f_65(v26, v27, v28, v29, v30, v31, v32);
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            long v33 = v5.case2.v0; static_array<static_array<unsigned char,2l>,2l> v34 = v5.case2.v1; static_array<long,2l> v35 = v5.case2.v2; long v36 = v5.case2.v3; static_array<long,2l> v37 = v5.case2.v4; Union6 v38 = v5.case2.v5;
            return f_65(v26, v33, v34, v35, v36, v37, v38);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_76(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+6168ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_75(unsigned char * v0, dynamic_array_list<Union3,128l> v1, static_array<Union2,2l> v2, Union7 v3){
    long v4;
    v4 = v1.length_();
    f_51(v0, v4);
    long v5;
    v5 = v1.length_();
    long v6;
    v6 = 0l;
    while (while_method_1(v5, v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = v8 * 48ull;
        unsigned long long v10;
        v10 = 16ull + v9;
        unsigned char * v11;
        v11 = (unsigned char *)(v0+v10);
        Union3 v12;
        v12 = v1[v6];
        f_50(v11, v12);
        v6 += 1l ;
    }
    long v13;
    v13 = 0l;
    while (while_method_0(v13)){
        unsigned long long v15;
        v15 = (unsigned long long)v13;
        unsigned long long v16;
        v16 = v15 * 4ull;
        unsigned long long v17;
        v17 = 6160ull + v16;
        unsigned char * v18;
        v18 = (unsigned char *)(v0+v17);
        Union2 v19;
        v19 = v2[v13];
        f_73(v18, v19);
        v13 += 1l ;
    }
    long v20;
    v20 = v3.tag;
    f_76(v0, v20);
    unsigned char * v21;
    v21 = (unsigned char *)(v0+6176ull);
    switch (v3.tag) {
        case 0: { // GameNotStarted
            return f_58(v21);
            break;
        }
        case 1: { // GameOver
            long v22 = v3.case1.v0; static_array<static_array<unsigned char,2l>,2l> v23 = v3.case1.v1; static_array<long,2l> v24 = v3.case1.v2; long v25 = v3.case1.v3; static_array<long,2l> v26 = v3.case1.v4; Union6 v27 = v3.case1.v5;
            return f_65(v21, v22, v23, v24, v25, v26, v27);
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            long v28 = v3.case2.v0; static_array<static_array<unsigned char,2l>,2l> v29 = v3.case2.v1; static_array<long,2l> v30 = v3.case2.v2; long v31 = v3.case2.v3; static_array<long,2l> v32 = v3.case2.v4; Union6 v33 = v3.case2.v5;
            return f_65(v21, v28, v29, v30, v31, v32, v33);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1, unsigned char * v2) {
    long v3;
    v3 = threadIdx.x;
    long v4;
    v4 = blockIdx.x;
    long v5;
    v5 = v4 * 32l;
    long v6;
    v6 = v3 + v5;
    bool v7;
    v7 = v6 == 0l;
    if (v7){
        Union0 v8;
        v8 = f_0(v1);
        unsigned long long v9; dynamic_array_list<Union3,128l> v10; Union4 v11; static_array<Union2,2l> v12; Union7 v13;
        Tuple1 tmp15 = f_6(v0);
        v9 = tmp15.v0; v10 = tmp15.v1; v11 = tmp15.v2; v12 = tmp15.v3; v13 = tmp15.v4;
        unsigned long long & v14 = v9;
        dynamic_array_list<Union3,128l> & v15 = v10;
        unsigned long long v16;
        v16 = clock64();
        long v17;
        v17 = threadIdx.x;
        long v18;
        v18 = blockIdx.x;
        long v19;
        v19 = v18 * 32l;
        long v20;
        v20 = v17 + v19;
        unsigned long long v21;
        v21 = (unsigned long long)v20;
        curandStatePhilox4_32_10_t v22;
        curand_init(v16,v21,0ull,&v22);
        Union4 v64; static_array<Union2,2l> v65; Union7 v66;
        switch (v8.tag) {
            case 0: { // ActionSelected
                Union1 v34 = v8.case0.v0;
                switch (v11.tag) {
                    case 0: { // None
                        v64 = v11; v65 = v12; v66 = v13;
                        break;
                    }
                    case 1: { // Some
                        Union5 v35 = v11.case1.v0;
                        switch (v35.tag) {
                            case 4: { // G_Round
                                long v36 = v35.case4.v0; static_array<static_array<unsigned char,2l>,2l> v37 = v35.case4.v1; static_array<long,2l> v38 = v35.case4.v2; long v39 = v35.case4.v3; static_array<long,2l> v40 = v35.case4.v4; Union6 v41 = v35.case4.v5;
                                Union5 v42;
                                v42 = Union5{Union5_5{v36, v37, v38, v39, v40, v41, v34}};
                                Tuple8 tmp66 = play_loop_31(v11, v12, v13, v14, v15, v22, v42);
                                v64 = tmp66.v0; v65 = tmp66.v1; v66 = tmp66.v2;
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
                static_array<Union2,2l> v33 = v8.case1.v0;
                v64 = v11; v65 = v33; v66 = v13;
                break;
            }
            case 2: { // StartGame
                static_array<Union2,2l> v23;
                Union2 v24;
                v24 = Union2{Union2_0{}};
                v23[0l] = v24;
                Union2 v25;
                v25 = Union2{Union2_1{}};
                v23[1l] = v25;
                dynamic_array_list<Union3,128l> v26{0};
                v14 = 4503599627370495ull;
                v15 = v26;
                Union4 v27;
                v27 = Union4{Union4_0{}};
                Union7 v28;
                v28 = Union7{Union7_0{}};
                Union5 v29;
                v29 = Union5{Union5_2{}};
                Tuple8 tmp67 = play_loop_31(v27, v23, v28, v14, v15, v22, v29);
                v64 = tmp67.v0; v65 = tmp67.v1; v66 = tmp67.v2;
                break;
            }
            default: {
                assert("Invalid tag." && false);
            }
        }
        f_47(v0, v9, v10, v64, v65, v66);
        return f_75(v2, v10, v65, v66);
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

import time
options = []
options.append('--diag-suppress=550,20012,68')
options.append('--dopt=on')
options.append('--restrict')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
import collections
class US1_0(NamedTuple): # A_All_In
    tag = 0
class US1_1(NamedTuple): # A_Call
    tag = 1
class US1_2(NamedTuple): # A_Fold
    tag = 2
class US1_3(NamedTuple): # A_Raise
    v0 : i32
    tag = 3
US1 = Union[US1_0, US1_1, US1_2, US1_3]
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
class US5_0(NamedTuple): # Flop
    v0 : static_array
    tag = 0
class US5_1(NamedTuple): # Preflop
    tag = 1
class US5_2(NamedTuple): # River
    v0 : static_array
    tag = 2
class US5_3(NamedTuple): # Turn
    v0 : static_array
    tag = 3
US5 = Union[US5_0, US5_1, US5_2, US5_3]
class US4_0(NamedTuple): # G_Flop
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    tag = 0
class US4_1(NamedTuple): # G_Fold
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    tag = 1
class US4_2(NamedTuple): # G_Preflop
    tag = 2
class US4_3(NamedTuple): # G_River
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    tag = 3
class US4_4(NamedTuple): # G_Round
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    tag = 4
class US4_5(NamedTuple): # G_Round'
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    v6 : US1
    tag = 5
class US4_6(NamedTuple): # G_Showdown
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    tag = 6
class US4_7(NamedTuple): # G_Turn
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    tag = 7
US4 = Union[US4_0, US4_1, US4_2, US4_3, US4_4, US4_5, US4_6, US4_7]
class US3_0(NamedTuple): # None
    tag = 0
class US3_1(NamedTuple): # Some
    v0 : US4
    tag = 1
US3 = Union[US3_0, US3_1]
class US6_0(NamedTuple): # GameNotStarted
    tag = 0
class US6_1(NamedTuple): # GameOver
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    tag = 1
class US6_2(NamedTuple): # WaitingForActionFromPlayerId
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    tag = 2
US6 = Union[US6_0, US6_1, US6_2]
class US7_0(NamedTuple): # CommunityCardsAre
    v0 : static_array_list
    tag = 0
class US7_1(NamedTuple): # Fold
    v0 : i32
    v1 : i32
    tag = 1
class US7_2(NamedTuple): # PlayerAction
    v0 : i32
    v1 : US1
    tag = 2
class US7_3(NamedTuple): # PlayerGotCards
    v0 : i32
    v1 : static_array
    tag = 3
class US7_4(NamedTuple): # Showdown
    v0 : i32
    v1 : static_array
    v2 : i32
    tag = 4
US7 = Union[US7_0, US7_1, US7_2, US7_3, US7_4]
def Closure0():
    def inner(v0 : object, v1 : object) -> object:
        v2 = cp.empty(16,dtype=cp.uint8)
        v3 = cp.empty(6304,dtype=cp.uint8)
        v4 = cp.empty(6224,dtype=cp.uint8)
        v5 = method0(v0)
        v6, v7, v8, v9, v10 = method8(v1)
        method39(v2, v5)
        del v5
        method46(v3, v6, v7, v8, v9, v10)
        del v6, v7, v8, v9, v10
        v11 = "Going to run the kernel."
        method74(v11)
        del v11
        print()
        v12 = time.perf_counter()
        v13 = 0
        v14 = raw_module.get_function(f"entry{v13}")
        del v13
        v14.max_dynamic_shared_size_bytes = 0 
        v14((1,),(32,),(v3, v2, v4),shared_mem=0)
        del v2, v14
        v15 = time.perf_counter()
        v16 = "The time it took to run the kernel (in seconds) is: "
        method74(v16)
        del v16
        v17 = v15 - v12
        del v12, v15
        method75(v17)
        del v17
        print()
        v18, v19, v20, v21, v22 = method76(v3)
        del v3
        v23, v24, v25 = method104(v4)
        del v4
        return method106(v18, v19, v20, v21, v22, v23, v24, v25)
    return inner
def Closure1():
    def inner() -> object:
        v0 = static_array(2)
        v1 = US2_0()
        v0[0] = v1
        del v1
        v2 = US2_1()
        v0[1] = v2
        del v2
        v3 = dynamic_array_list(128)
        v4 = 4503599627370495
        v5 = US3_0()
        v6 = US6_0()
        return method144(v4, v3, v5, v0, v6)
    return inner
def method3(v0 : object) -> None:
    assert v0 == [], f'Expected an unit type. Got: {v0}'
    del v0
    return 
def method4(v0 : object) -> i32:
    assert isinstance(v0,i32), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method2(v0 : object) -> US1:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "A_All_In" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US1_0()
    else:
        del v4
        v7 = "A_Call" == v1
        if v7:
            del v1, v7
            method3(v2)
            del v2
            return US1_1()
        else:
            del v7
            v10 = "A_Fold" == v1
            if v10:
                del v1, v10
                method3(v2)
                del v2
                return US1_2()
            else:
                del v10
                v13 = "A_Raise" == v1
                if v13:
                    del v1, v13
                    v14 = method4(v2)
                    del v2
                    return US1_3(v14)
                else:
                    del v2, v13
                    raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                    del v1
                    raise Exception("Error")
def method6(v0 : i32, v1 : i32) -> bool:
    v2 = v1 < v0
    del v0, v1
    return v2
def method7(v0 : object) -> US2:
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
def method5(v0 : object) -> static_array:
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
    v5 = static_array(2)
    v6 = 0
    while method6(v1, v6):
        v8 = v0[v6]
        v9 = method7(v8)
        del v8
        v5[v6] = v9
        del v9
        v6 += 1 
    del v0, v1, v6
    return v5
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
            v9 = method5(v2)
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
def method12(v0 : object) -> u64:
    assert isinstance(v0,u64), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method11(v0 : object) -> u64:
    v1 = method12(v0)
    del v0
    return v1
def method17(v0 : object) -> u8:
    assert isinstance(v0,u8), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method16(v0 : object) -> u8:
    v1 = method17(v0)
    del v0
    return v1
def method15(v0 : object) -> static_array_list:
    v1 = len(v0) # type: ignore
    assert (5 >= v1), f'The length of the original object has to be greater than or equal to the static array dimension.\nExpected: 5\nGot: {v1} '
    del v1
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v2 = len(v0) # type: ignore
    v3 = 5 >= v2
    v4 = v3 == False
    if v4:
        v5 = "The type level dimension has to equal the value passed at runtime into create."
        assert v3, v5
        del v5
    else:
        pass
    del v3, v4
    v6 = static_array_list(5)
    v6.unsafe_set_length(v2)
    v7 = 0
    while method6(v2, v7):
        v9 = v0[v7]
        v10 = method16(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v2, v7
    return v6
def method18(v0 : object) -> Tuple[i32, i32]:
    v1 = v0["chips_won"] # type: ignore
    v2 = method4(v1)
    del v1
    v3 = v0["winner_id"] # type: ignore
    del v0
    v4 = method4(v3)
    del v3
    return v2, v4
def method19(v0 : object) -> Tuple[i32, US1]:
    v1 = v0[0] # type: ignore
    v2 = method4(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method2(v3)
    del v3
    return v2, v4
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
    v5 = static_array(2)
    v6 = 0
    while method6(v1, v6):
        v8 = v0[v6]
        v9 = method16(v8)
        del v8
        v5[v6] = v9
        del v9
        v6 += 1 
    del v0, v1, v6
    return v5
def method20(v0 : object) -> Tuple[i32, static_array]:
    v1 = v0[0] # type: ignore
    v2 = method4(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method21(v3)
    del v3
    return v2, v4
def method26(v0 : object) -> static_array:
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v1 = len(v0) # type: ignore
    v2 = 5 == v1
    v3 = v2 == False
    if v3:
        v4 = "The type level dimension has to equal the value passed at runtime into create."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = static_array(5)
    v6 = 0
    while method6(v1, v6):
        v8 = v0[v6]
        v9 = method16(v8)
        del v8
        v5[v6] = v9
        del v9
        v6 += 1 
    del v0, v1, v6
    return v5
def method27(v0 : object) -> i8:
    assert isinstance(v0,i8), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method25(v0 : object) -> Tuple[static_array, i8]:
    v1 = v0["hand"] # type: ignore
    v2 = method26(v1)
    del v1
    v3 = v0["score"] # type: ignore
    del v0
    v4 = method27(v3)
    del v3
    return v2, v4
def method24(v0 : object) -> Tuple[static_array, i8]:
    v1, v2 = method25(v0)
    del v0
    return v1, v2
def method23(v0 : object) -> static_array:
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
    v5 = static_array(2)
    v6 = 0
    while method6(v1, v6):
        v8 = v0[v6]
        v9, v10 = method24(v8)
        del v8
        v5[v6] = (v9, v10)
        del v9, v10
        v6 += 1 
    del v0, v1, v6
    return v5
def method22(v0 : object) -> Tuple[i32, static_array, i32]:
    v1 = v0["chips_won"] # type: ignore
    v2 = method4(v1)
    del v1
    v3 = v0["hands_shown"] # type: ignore
    v4 = method23(v3)
    del v3
    v5 = v0["winner_id"] # type: ignore
    del v0
    v6 = method4(v5)
    del v5
    return v2, v4, v6
def method14(v0 : object) -> US7:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "CommunityCardsAre" == v1
    if v4:
        del v1, v4
        v5 = method15(v2)
        del v2
        return US7_0(v5)
    else:
        del v4
        v8 = "Fold" == v1
        if v8:
            del v1, v8
            v9, v10 = method18(v2)
            del v2
            return US7_1(v9, v10)
        else:
            del v8
            v13 = "PlayerAction" == v1
            if v13:
                del v1, v13
                v14, v15 = method19(v2)
                del v2
                return US7_2(v14, v15)
            else:
                del v13
                v18 = "PlayerGotCards" == v1
                if v18:
                    del v1, v18
                    v19, v20 = method20(v2)
                    del v2
                    return US7_3(v19, v20)
                else:
                    del v18
                    v23 = "Showdown" == v1
                    if v23:
                        del v1, v23
                        v24, v25, v26 = method22(v2)
                        del v2
                        return US7_4(v24, v25, v26)
                    else:
                        del v2, v23
                        raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                        del v1
                        raise Exception("Error")
def method13(v0 : object) -> dynamic_array_list:
    v1 = len(v0) # type: ignore
    assert (128 >= v1), f'The length of the original object has to be greater than or equal to the static array dimension.\nExpected: 128\nGot: {v1} '
    del v1
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v2 = len(v0) # type: ignore
    v3 = 128 >= v2
    v4 = v3 == False
    if v4:
        v5 = "The type level dimension has to equal the value passed at runtime into create."
        assert v3, v5
        del v5
    else:
        pass
    del v3, v4
    v6 = dynamic_array_list(128)
    v6.unsafe_set_length(v2)
    v7 = 0
    while method6(v2, v7):
        v9 = v0[v7]
        v10 = method14(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v2, v7
    return v6
def method10(v0 : object) -> Tuple[u64, dynamic_array_list]:
    v1 = v0["deck"] # type: ignore
    v2 = method11(v1)
    del v1
    v3 = v0["messages"] # type: ignore
    del v0
    v4 = method13(v3)
    del v3
    return v2, v4
def method32(v0 : object) -> static_array:
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
    v5 = static_array(2)
    v6 = 0
    while method6(v1, v6):
        v8 = v0[v6]
        v9 = method21(v8)
        del v8
        v5[v6] = v9
        del v9
        v6 += 1 
    del v0, v1, v6
    return v5
def method33(v0 : object) -> static_array:
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
    v5 = static_array(2)
    v6 = 0
    while method6(v1, v6):
        v8 = v0[v6]
        v9 = method4(v8)
        del v8
        v5[v6] = v9
        del v9
        v6 += 1 
    del v0, v1, v6
    return v5
def method35(v0 : object) -> static_array:
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v1 = len(v0) # type: ignore
    v2 = 3 == v1
    v3 = v2 == False
    if v3:
        v4 = "The type level dimension has to equal the value passed at runtime into create."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = static_array(3)
    v6 = 0
    while method6(v1, v6):
        v8 = v0[v6]
        v9 = method16(v8)
        del v8
        v5[v6] = v9
        del v9
        v6 += 1 
    del v0, v1, v6
    return v5
def method36(v0 : object) -> static_array:
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v1 = len(v0) # type: ignore
    v2 = 4 == v1
    v3 = v2 == False
    if v3:
        v4 = "The type level dimension has to equal the value passed at runtime into create."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = static_array(4)
    v6 = 0
    while method6(v1, v6):
        v8 = v0[v6]
        v9 = method16(v8)
        del v8
        v5[v6] = v9
        del v9
        v6 += 1 
    del v0, v1, v6
    return v5
def method34(v0 : object) -> US5:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "Flop" == v1
    if v4:
        del v1, v4
        v5 = method35(v2)
        del v2
        return US5_0(v5)
    else:
        del v4
        v8 = "Preflop" == v1
        if v8:
            del v1, v8
            method3(v2)
            del v2
            return US5_1()
        else:
            del v8
            v11 = "River" == v1
            if v11:
                del v1, v11
                v12 = method26(v2)
                del v2
                return US5_2(v12)
            else:
                del v11
                v15 = "Turn" == v1
                if v15:
                    del v1, v15
                    v16 = method36(v2)
                    del v2
                    return US5_3(v16)
                else:
                    del v2, v15
                    raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                    del v1
                    raise Exception("Error")
def method31(v0 : object) -> Tuple[i32, static_array, static_array, i32, static_array, US5]:
    v1 = v0["min_raise"] # type: ignore
    v2 = method4(v1)
    del v1
    v3 = v0["pl_card"] # type: ignore
    v4 = method32(v3)
    del v3
    v5 = v0["pot"] # type: ignore
    v6 = method33(v5)
    del v5
    v7 = v0["round_turn"] # type: ignore
    v8 = method4(v7)
    del v7
    v9 = v0["stack"] # type: ignore
    v10 = method33(v9)
    del v9
    v11 = v0["street"] # type: ignore
    del v0
    v12 = method34(v11)
    del v11
    return v2, v4, v6, v8, v10, v12
def method37(v0 : object) -> Tuple[i32, static_array, static_array, i32, static_array, US5, US1]:
    v1 = v0[0] # type: ignore
    v2, v3, v4, v5, v6, v7 = method31(v1)
    del v1
    v8 = v0[1] # type: ignore
    del v0
    v9 = method2(v8)
    del v8
    return v2, v3, v4, v5, v6, v7, v9
def method30(v0 : object) -> US4:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "G_Flop" == v1
    if v4:
        del v1, v4
        v5, v6, v7, v8, v9, v10 = method31(v2)
        del v2
        return US4_0(v5, v6, v7, v8, v9, v10)
    else:
        del v4
        v13 = "G_Fold" == v1
        if v13:
            del v1, v13
            v14, v15, v16, v17, v18, v19 = method31(v2)
            del v2
            return US4_1(v14, v15, v16, v17, v18, v19)
        else:
            del v13
            v22 = "G_Preflop" == v1
            if v22:
                del v1, v22
                method3(v2)
                del v2
                return US4_2()
            else:
                del v22
                v25 = "G_River" == v1
                if v25:
                    del v1, v25
                    v26, v27, v28, v29, v30, v31 = method31(v2)
                    del v2
                    return US4_3(v26, v27, v28, v29, v30, v31)
                else:
                    del v25
                    v34 = "G_Round" == v1
                    if v34:
                        del v1, v34
                        v35, v36, v37, v38, v39, v40 = method31(v2)
                        del v2
                        return US4_4(v35, v36, v37, v38, v39, v40)
                    else:
                        del v34
                        v43 = "G_Round'" == v1
                        if v43:
                            del v1, v43
                            v44, v45, v46, v47, v48, v49, v50 = method37(v2)
                            del v2
                            return US4_5(v44, v45, v46, v47, v48, v49, v50)
                        else:
                            del v43
                            v53 = "G_Showdown" == v1
                            if v53:
                                del v1, v53
                                v54, v55, v56, v57, v58, v59 = method31(v2)
                                del v2
                                return US4_6(v54, v55, v56, v57, v58, v59)
                            else:
                                del v53
                                v62 = "G_Turn" == v1
                                if v62:
                                    del v1, v62
                                    v63, v64, v65, v66, v67, v68 = method31(v2)
                                    del v2
                                    return US4_7(v63, v64, v65, v66, v67, v68)
                                else:
                                    del v2, v62
                                    raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                                    del v1
                                    raise Exception("Error")
def method29(v0 : object) -> US3:
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
            v8 = method30(v2)
            del v2
            return US3_1(v8)
        else:
            del v2, v7
            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
            del v1
            raise Exception("Error")
def method38(v0 : object) -> US6:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "GameNotStarted" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US6_0()
    else:
        del v4
        v7 = "GameOver" == v1
        if v7:
            del v1, v7
            v8, v9, v10, v11, v12, v13 = method31(v2)
            del v2
            return US6_1(v8, v9, v10, v11, v12, v13)
        else:
            del v7
            v16 = "WaitingForActionFromPlayerId" == v1
            if v16:
                del v1, v16
                v17, v18, v19, v20, v21, v22 = method31(v2)
                del v2
                return US6_2(v17, v18, v19, v20, v21, v22)
            else:
                del v2, v16
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method28(v0 : object) -> Tuple[US3, static_array, US6]:
    v1 = v0["game"] # type: ignore
    v2 = method29(v1)
    del v1
    v3 = v0["pl_type"] # type: ignore
    v4 = method5(v3)
    del v3
    v5 = v0["ui_game_state"] # type: ignore
    del v0
    v6 = method38(v5)
    del v5
    return v2, v4, v6
def method9(v0 : object) -> Tuple[u64, dynamic_array_list, US3, static_array, US6]:
    v1 = v0["large"] # type: ignore
    v2, v3 = method10(v1)
    del v1
    v4 = v0["small"] # type: ignore
    del v0
    v5, v6, v7 = method28(v4)
    del v4
    return v2, v3, v5, v6, v7
def method8(v0 : object) -> Tuple[u64, dynamic_array_list, US3, static_array, US6]:
    return method9(v0)
def method40(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[0:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method42(v0 : cp.ndarray) -> None:
    del v0
    return 
def method41(v0 : cp.ndarray, v1 : US1) -> None:
    v2 = v1.tag
    method40(v0, v2)
    del v2
    v3 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US1_0(): # A_All_In
            del v1
            return method42(v3)
        case US1_1(): # A_Call
            del v1
            return method42(v3)
        case US1_2(): # A_Fold
            del v1
            return method42(v3)
        case US1_3(v4): # A_Raise
            del v1
            return method40(v3, v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method44(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method45(v0 : cp.ndarray, v1 : US2) -> None:
    v2 = v1.tag
    method40(v0, v2)
    del v2
    v3 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US2_0(): # Computer
            del v1
            return method42(v3)
        case US2_1(): # Human
            del v1
            return method42(v3)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method43(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method44(v2):
        v4 = u64(v2)
        v5 = v4 * 4
        del v4
        v6 = v0[v5:].view(cp.uint8)
        del v5
        v7 = v1[v2]
        method45(v6, v7)
        del v6, v7
        v2 += 1 
    del v0, v1, v2
    return 
def method39(v0 : cp.ndarray, v1 : US0) -> None:
    v2 = v1.tag
    method40(v0, v2)
    del v2
    v3 = v0[8:].view(cp.uint8)
    del v0
    match v1:
        case US0_0(v4): # ActionSelected
            del v1
            return method41(v3, v4)
        case US0_1(v5): # PlayerChanged
            del v1
            return method43(v3, v5)
        case US0_2(): # StartGame
            del v1
            return method42(v3)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method47(v0 : cp.ndarray, v1 : u64) -> None:
    v2 = v0[0:].view(cp.uint64)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method48(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[8:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method52(v0 : cp.ndarray, v1 : u8) -> None:
    v2 = v0[0:].view(cp.uint8)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method51(v0 : cp.ndarray, v1 : u8) -> None:
    return method52(v0, v1)
def method50(v0 : cp.ndarray, v1 : static_array_list) -> None:
    v2 = v1.length
    method40(v0, v2)
    del v2
    v3 = v1.length
    v4 = 0
    while method6(v3, v4):
        v6 = u64(v4)
        v7 = 4 + v6
        del v6
        v8 = v0[v7:].view(cp.uint8)
        del v7
        v9 = v1[v4]
        method51(v8, v9)
        del v8, v9
        v4 += 1 
    del v0, v1, v3, v4
    return 
def method53(v0 : cp.ndarray, v1 : i32, v2 : i32) -> None:
    v3 = v0[0:].view(cp.int32)
    v3[0] = v1
    del v1, v3
    v4 = v0[4:].view(cp.int32)
    del v0
    v4[0] = v2
    del v2, v4
    return 
def method55(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[4:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method54(v0 : cp.ndarray, v1 : i32, v2 : US1) -> None:
    v3 = v0[0:].view(cp.int32)
    v3[0] = v1
    del v1, v3
    v4 = v2.tag
    method55(v0, v4)
    del v4
    v5 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US1_0(): # A_All_In
            del v2
            return method42(v5)
        case US1_1(): # A_Call
            del v2
            return method42(v5)
        case US1_2(): # A_Fold
            del v2
            return method42(v5)
        case US1_3(v6): # A_Raise
            del v2
            return method40(v5, v6)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method56(v0 : cp.ndarray, v1 : i32, v2 : static_array) -> None:
    v3 = v0[0:].view(cp.int32)
    v3[0] = v1
    del v1, v3
    v4 = 0
    while method44(v4):
        v6 = u64(v4)
        v7 = 4 + v6
        del v6
        v8 = v0[v7:].view(cp.uint8)
        del v7
        v9 = v2[v4]
        method51(v8, v9)
        del v8, v9
        v4 += 1 
    del v0, v2, v4
    return 
def method60(v0 : i32) -> bool:
    v1 = v0 < 5
    del v0
    return v1
def method59(v0 : cp.ndarray, v1 : static_array, v2 : i8) -> None:
    v3 = 0
    while method60(v3):
        v5 = u64(v3)
        v6 = v0[v5:].view(cp.uint8)
        del v5
        v7 = v1[v3]
        method51(v6, v7)
        del v6, v7
        v3 += 1 
    del v1, v3
    v8 = v0[5:].view(cp.int8)
    del v0
    v8[0] = v2
    del v2, v8
    return 
def method58(v0 : cp.ndarray, v1 : static_array, v2 : i8) -> None:
    return method59(v0, v1, v2)
def method57(v0 : cp.ndarray, v1 : i32, v2 : static_array, v3 : i32) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = 0
    while method44(v5):
        v7 = u64(v5)
        v8 = v7 * 8
        del v7
        v9 = 8 + v8
        del v8
        v10 = v0[v9:].view(cp.uint8)
        del v9
        v11, v12 = v2[v5]
        method58(v10, v11, v12)
        del v10, v11, v12
        v5 += 1 
    del v2, v5
    v13 = v0[24:].view(cp.int32)
    del v0
    v13[0] = v3
    del v3, v13
    return 
def method49(v0 : cp.ndarray, v1 : US7) -> None:
    v2 = v1.tag
    method40(v0, v2)
    del v2
    v3 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US7_0(v4): # CommunityCardsAre
            del v1
            return method50(v3, v4)
        case US7_1(v5, v6): # Fold
            del v1
            return method53(v3, v5, v6)
        case US7_2(v7, v8): # PlayerAction
            del v1
            return method54(v3, v7, v8)
        case US7_3(v9, v10): # PlayerGotCards
            del v1
            return method56(v3, v9, v10)
        case US7_4(v11, v12, v13): # Showdown
            del v1
            return method57(v3, v11, v12, v13)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method61(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[6160:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method64(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method44(v2):
        v4 = u64(v2)
        v5 = v0[v4:].view(cp.uint8)
        del v4
        v6 = v1[v2]
        method51(v5, v6)
        del v5, v6
        v2 += 1 
    del v0, v1, v2
    return 
def method65(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[28:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method67(v0 : i32) -> bool:
    v1 = v0 < 3
    del v0
    return v1
def method66(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method67(v2):
        v4 = u64(v2)
        v5 = v0[v4:].view(cp.uint8)
        del v4
        v6 = v1[v2]
        method51(v5, v6)
        del v5, v6
        v2 += 1 
    del v0, v1, v2
    return 
def method68(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method60(v2):
        v4 = u64(v2)
        v5 = v0[v4:].view(cp.uint8)
        del v4
        v6 = v1[v2]
        method51(v5, v6)
        del v5, v6
        v2 += 1 
    del v0, v1, v2
    return 
def method70(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method69(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method70(v2):
        v4 = u64(v2)
        v5 = v0[v4:].view(cp.uint8)
        del v4
        v6 = v1[v2]
        method51(v5, v6)
        del v5, v6
        v2 += 1 
    del v0, v1, v2
    return 
def method63(v0 : cp.ndarray, v1 : i32, v2 : static_array, v3 : static_array, v4 : i32, v5 : static_array, v6 : US5) -> None:
    v7 = v0[0:].view(cp.int32)
    v7[0] = v1
    del v1, v7
    v8 = 0
    while method44(v8):
        v10 = u64(v8)
        v11 = v10 * 2
        del v10
        v12 = 4 + v11
        del v11
        v13 = v0[v12:].view(cp.uint8)
        del v12
        v14 = v2[v8]
        method64(v13, v14)
        del v13, v14
        v8 += 1 
    del v2, v8
    v15 = 0
    while method44(v15):
        v17 = u64(v15)
        v18 = v17 * 4
        del v17
        v19 = 8 + v18
        del v18
        v20 = v0[v19:].view(cp.uint8)
        del v19
        v21 = v3[v15]
        method40(v20, v21)
        del v20, v21
        v15 += 1 
    del v3, v15
    v22 = v0[16:].view(cp.int32)
    v22[0] = v4
    del v4, v22
    v23 = 0
    while method44(v23):
        v25 = u64(v23)
        v26 = v25 * 4
        del v25
        v27 = 20 + v26
        del v26
        v28 = v0[v27:].view(cp.uint8)
        del v27
        v29 = v5[v23]
        method40(v28, v29)
        del v28, v29
        v23 += 1 
    del v5, v23
    v30 = v6.tag
    method65(v0, v30)
    del v30
    v31 = v0[32:].view(cp.uint8)
    del v0
    match v6:
        case US5_0(v32): # Flop
            del v6
            return method66(v31, v32)
        case US5_1(): # Preflop
            del v6
            return method42(v31)
        case US5_2(v33): # River
            del v6
            return method68(v31, v33)
        case US5_3(v34): # Turn
            del v6
            return method69(v31, v34)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method72(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[40:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method71(v0 : cp.ndarray, v1 : i32, v2 : static_array, v3 : static_array, v4 : i32, v5 : static_array, v6 : US5, v7 : US1) -> None:
    v8 = v0[0:].view(cp.int32)
    v8[0] = v1
    del v1, v8
    v9 = 0
    while method44(v9):
        v11 = u64(v9)
        v12 = v11 * 2
        del v11
        v13 = 4 + v12
        del v12
        v14 = v0[v13:].view(cp.uint8)
        del v13
        v15 = v2[v9]
        method64(v14, v15)
        del v14, v15
        v9 += 1 
    del v2, v9
    v16 = 0
    while method44(v16):
        v18 = u64(v16)
        v19 = v18 * 4
        del v18
        v20 = 8 + v19
        del v19
        v21 = v0[v20:].view(cp.uint8)
        del v20
        v22 = v3[v16]
        method40(v21, v22)
        del v21, v22
        v16 += 1 
    del v3, v16
    v23 = v0[16:].view(cp.int32)
    v23[0] = v4
    del v4, v23
    v24 = 0
    while method44(v24):
        v26 = u64(v24)
        v27 = v26 * 4
        del v26
        v28 = 20 + v27
        del v27
        v29 = v0[v28:].view(cp.uint8)
        del v28
        v30 = v5[v24]
        method40(v29, v30)
        del v29, v30
        v24 += 1 
    del v5, v24
    v31 = v6.tag
    method65(v0, v31)
    del v31
    v32 = v0[32:].view(cp.uint8)
    match v6:
        case US5_0(v33): # Flop
            method66(v32, v33)
        case US5_1(): # Preflop
            method42(v32)
        case US5_2(v34): # River
            method68(v32, v34)
        case US5_3(v35): # Turn
            method69(v32, v35)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v6, v32
    v36 = v7.tag
    method72(v0, v36)
    del v36
    v37 = v0[44:].view(cp.uint8)
    del v0
    match v7:
        case US1_0(): # A_All_In
            del v7
            return method42(v37)
        case US1_1(): # A_Call
            del v7
            return method42(v37)
        case US1_2(): # A_Fold
            del v7
            return method42(v37)
        case US1_3(v38): # A_Raise
            del v7
            return method40(v37, v38)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method62(v0 : cp.ndarray, v1 : US4) -> None:
    v2 = v1.tag
    method40(v0, v2)
    del v2
    v3 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US4_0(v4, v5, v6, v7, v8, v9): # G_Flop
            del v1
            return method63(v3, v4, v5, v6, v7, v8, v9)
        case US4_1(v10, v11, v12, v13, v14, v15): # G_Fold
            del v1
            return method63(v3, v10, v11, v12, v13, v14, v15)
        case US4_2(): # G_Preflop
            del v1
            return method42(v3)
        case US4_3(v16, v17, v18, v19, v20, v21): # G_River
            del v1
            return method63(v3, v16, v17, v18, v19, v20, v21)
        case US4_4(v22, v23, v24, v25, v26, v27): # G_Round
            del v1
            return method63(v3, v22, v23, v24, v25, v26, v27)
        case US4_5(v28, v29, v30, v31, v32, v33, v34): # G_Round'
            del v1
            return method71(v3, v28, v29, v30, v31, v32, v33, v34)
        case US4_6(v35, v36, v37, v38, v39, v40): # G_Showdown
            del v1
            return method63(v3, v35, v36, v37, v38, v39, v40)
        case US4_7(v41, v42, v43, v44, v45, v46): # G_Turn
            del v1
            return method63(v3, v41, v42, v43, v44, v45, v46)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method73(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[6248:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method46(v0 : cp.ndarray, v1 : u64, v2 : dynamic_array_list, v3 : US3, v4 : static_array, v5 : US6) -> None:
    method47(v0, v1)
    del v1
    v6 = v2.length_()
    method48(v0, v6)
    del v6
    v7 = v2.length_()
    v8 = 0
    while method6(v7, v8):
        v10 = u64(v8)
        v11 = v10 * 48
        del v10
        v12 = 16 + v11
        del v11
        v13 = v0[v12:].view(cp.uint8)
        del v12
        v14 = v2[v8]
        method49(v13, v14)
        del v13, v14
        v8 += 1 
    del v2, v7, v8
    v15 = v3.tag
    method61(v0, v15)
    del v15
    v16 = v0[6176:].view(cp.uint8)
    match v3:
        case US3_0(): # None
            method42(v16)
        case US3_1(v17): # Some
            method62(v16, v17)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v3, v16
    v18 = 0
    while method44(v18):
        v20 = u64(v18)
        v21 = v20 * 4
        del v20
        v22 = 6240 + v21
        del v21
        v23 = v0[v22:].view(cp.uint8)
        del v22
        v24 = v4[v18]
        method45(v23, v24)
        del v23, v24
        v18 += 1 
    del v4, v18
    v25 = v5.tag
    method73(v0, v25)
    del v25
    v26 = v0[6256:].view(cp.uint8)
    del v0
    match v5:
        case US6_0(): # GameNotStarted
            del v5
            return method42(v26)
        case US6_1(v27, v28, v29, v30, v31, v32): # GameOver
            del v5
            return method63(v26, v27, v28, v29, v30, v31, v32)
        case US6_2(v33, v34, v35, v36, v37, v38): # WaitingForActionFromPlayerId
            del v5
            return method63(v26, v33, v34, v35, v36, v37, v38)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method74(v0 : string) -> None:
    print(v0, end="")
    del v0
    return 
def method75(v0 : f64) -> None:
    print("{:.6f}".format(v0), end="")
    del v0
    return 
def method77(v0 : cp.ndarray) -> u64:
    v1 = v0[0:].view(cp.uint64)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method78(v0 : cp.ndarray) -> i32:
    v1 = v0[8:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method80(v0 : cp.ndarray) -> i32:
    v1 = v0[0:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method83(v0 : cp.ndarray) -> u8:
    v1 = v0[0:].view(cp.uint8)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method82(v0 : cp.ndarray) -> u8:
    v1 = method83(v0)
    del v0
    return v1
def method81(v0 : cp.ndarray) -> static_array_list:
    v1 = static_array_list(5)
    v2 = method80(v0)
    v1.unsafe_set_length(v2)
    del v2
    v3 = v1.length
    v4 = 0
    while method6(v3, v4):
        v6 = u64(v4)
        v7 = 4 + v6
        del v6
        v8 = v0[v7:].view(cp.uint8)
        del v7
        v9 = method82(v8)
        del v8
        v1[v4] = v9
        del v9
        v4 += 1 
    del v0, v3, v4
    return v1
def method84(v0 : cp.ndarray) -> Tuple[i32, i32]:
    v1 = v0[0:].view(cp.int32)
    v2 = v1[0].item()
    del v1
    v3 = v0[4:].view(cp.int32)
    del v0
    v4 = v3[0].item()
    del v3
    return v2, v4
def method86(v0 : cp.ndarray) -> i32:
    v1 = v0[4:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method87(v0 : cp.ndarray) -> None:
    del v0
    return 
def method85(v0 : cp.ndarray) -> Tuple[i32, US1]:
    v1 = v0[0:].view(cp.int32)
    v2 = v1[0].item()
    del v1
    v3 = method86(v0)
    v4 = v0[8:].view(cp.uint8)
    del v0
    if v3 == 0:
        method87(v4)
        v11 = US1_0()
    elif v3 == 1:
        method87(v4)
        v11 = US1_1()
    elif v3 == 2:
        method87(v4)
        v11 = US1_2()
    elif v3 == 3:
        v9 = method80(v4)
        v11 = US1_3(v9)
    else:
        raise Exception("Invalid tag.")
    del v3, v4
    return v2, v11
def method88(v0 : cp.ndarray) -> Tuple[i32, static_array]:
    v1 = v0[0:].view(cp.int32)
    v2 = v1[0].item()
    del v1
    v3 = static_array(2)
    v4 = 0
    while method44(v4):
        v6 = u64(v4)
        v7 = 4 + v6
        del v6
        v8 = v0[v7:].view(cp.uint8)
        del v7
        v9 = method82(v8)
        del v8
        v3[v4] = v9
        del v9
        v4 += 1 
    del v0, v4
    return v2, v3
def method91(v0 : cp.ndarray) -> Tuple[static_array, i8]:
    v1 = static_array(5)
    v2 = 0
    while method60(v2):
        v4 = u64(v2)
        v5 = v0[v4:].view(cp.uint8)
        del v4
        v6 = method82(v5)
        del v5
        v1[v2] = v6
        del v6
        v2 += 1 
    del v2
    v7 = v0[5:].view(cp.int8)
    del v0
    v8 = v7[0].item()
    del v7
    return v1, v8
def method90(v0 : cp.ndarray) -> Tuple[static_array, i8]:
    v1, v2 = method91(v0)
    del v0
    return v1, v2
def method89(v0 : cp.ndarray) -> Tuple[i32, static_array, i32]:
    v1 = v0[0:].view(cp.int32)
    v2 = v1[0].item()
    del v1
    v3 = static_array(2)
    v4 = 0
    while method44(v4):
        v6 = u64(v4)
        v7 = v6 * 8
        del v6
        v8 = 8 + v7
        del v7
        v9 = v0[v8:].view(cp.uint8)
        del v8
        v10, v11 = method90(v9)
        del v9
        v3[v4] = (v10, v11)
        del v10, v11
        v4 += 1 
    del v4
    v12 = v0[24:].view(cp.int32)
    del v0
    v13 = v12[0].item()
    del v12
    return v2, v3, v13
def method79(v0 : cp.ndarray) -> US7:
    v1 = method80(v0)
    v2 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v4 = method81(v2)
        del v2
        return US7_0(v4)
    elif v1 == 1:
        del v1
        v6, v7 = method84(v2)
        del v2
        return US7_1(v6, v7)
    elif v1 == 2:
        del v1
        v9, v10 = method85(v2)
        del v2
        return US7_2(v9, v10)
    elif v1 == 3:
        del v1
        v12, v13 = method88(v2)
        del v2
        return US7_3(v12, v13)
    elif v1 == 4:
        del v1
        v15, v16, v17 = method89(v2)
        del v2
        return US7_4(v15, v16, v17)
    else:
        del v1, v2
        raise Exception("Invalid tag.")
def method92(v0 : cp.ndarray) -> i32:
    v1 = v0[6160:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method95(v0 : cp.ndarray) -> static_array:
    v1 = static_array(2)
    v2 = 0
    while method44(v2):
        v4 = u64(v2)
        v5 = v0[v4:].view(cp.uint8)
        del v4
        v6 = method82(v5)
        del v5
        v1[v2] = v6
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method96(v0 : cp.ndarray) -> i32:
    v1 = v0[28:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method97(v0 : cp.ndarray) -> static_array:
    v1 = static_array(3)
    v2 = 0
    while method67(v2):
        v4 = u64(v2)
        v5 = v0[v4:].view(cp.uint8)
        del v4
        v6 = method82(v5)
        del v5
        v1[v2] = v6
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method98(v0 : cp.ndarray) -> static_array:
    v1 = static_array(5)
    v2 = 0
    while method60(v2):
        v4 = u64(v2)
        v5 = v0[v4:].view(cp.uint8)
        del v4
        v6 = method82(v5)
        del v5
        v1[v2] = v6
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method99(v0 : cp.ndarray) -> static_array:
    v1 = static_array(4)
    v2 = 0
    while method70(v2):
        v4 = u64(v2)
        v5 = v0[v4:].view(cp.uint8)
        del v4
        v6 = method82(v5)
        del v5
        v1[v2] = v6
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method94(v0 : cp.ndarray) -> Tuple[i32, static_array, static_array, i32, static_array, US5]:
    v1 = v0[0:].view(cp.int32)
    v2 = v1[0].item()
    del v1
    v3 = static_array(2)
    v4 = 0
    while method44(v4):
        v6 = u64(v4)
        v7 = v6 * 2
        del v6
        v8 = 4 + v7
        del v7
        v9 = v0[v8:].view(cp.uint8)
        del v8
        v10 = method95(v9)
        del v9
        v3[v4] = v10
        del v10
        v4 += 1 
    del v4
    v11 = static_array(2)
    v12 = 0
    while method44(v12):
        v14 = u64(v12)
        v15 = v14 * 4
        del v14
        v16 = 8 + v15
        del v15
        v17 = v0[v16:].view(cp.uint8)
        del v16
        v18 = method80(v17)
        del v17
        v11[v12] = v18
        del v18
        v12 += 1 
    del v12
    v19 = v0[16:].view(cp.int32)
    v20 = v19[0].item()
    del v19
    v21 = static_array(2)
    v22 = 0
    while method44(v22):
        v24 = u64(v22)
        v25 = v24 * 4
        del v24
        v26 = 20 + v25
        del v25
        v27 = v0[v26:].view(cp.uint8)
        del v26
        v28 = method80(v27)
        del v27
        v21[v22] = v28
        del v28
        v22 += 1 
    del v22
    v29 = method96(v0)
    v30 = v0[32:].view(cp.uint8)
    del v0
    if v29 == 0:
        v32 = method97(v30)
        v39 = US5_0(v32)
    elif v29 == 1:
        method87(v30)
        v39 = US5_1()
    elif v29 == 2:
        v35 = method98(v30)
        v39 = US5_2(v35)
    elif v29 == 3:
        v37 = method99(v30)
        v39 = US5_3(v37)
    else:
        raise Exception("Invalid tag.")
    del v29, v30
    return v2, v3, v11, v20, v21, v39
def method101(v0 : cp.ndarray) -> i32:
    v1 = v0[40:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method100(v0 : cp.ndarray) -> Tuple[i32, static_array, static_array, i32, static_array, US5, US1]:
    v1 = v0[0:].view(cp.int32)
    v2 = v1[0].item()
    del v1
    v3 = static_array(2)
    v4 = 0
    while method44(v4):
        v6 = u64(v4)
        v7 = v6 * 2
        del v6
        v8 = 4 + v7
        del v7
        v9 = v0[v8:].view(cp.uint8)
        del v8
        v10 = method95(v9)
        del v9
        v3[v4] = v10
        del v10
        v4 += 1 
    del v4
    v11 = static_array(2)
    v12 = 0
    while method44(v12):
        v14 = u64(v12)
        v15 = v14 * 4
        del v14
        v16 = 8 + v15
        del v15
        v17 = v0[v16:].view(cp.uint8)
        del v16
        v18 = method80(v17)
        del v17
        v11[v12] = v18
        del v18
        v12 += 1 
    del v12
    v19 = v0[16:].view(cp.int32)
    v20 = v19[0].item()
    del v19
    v21 = static_array(2)
    v22 = 0
    while method44(v22):
        v24 = u64(v22)
        v25 = v24 * 4
        del v24
        v26 = 20 + v25
        del v25
        v27 = v0[v26:].view(cp.uint8)
        del v26
        v28 = method80(v27)
        del v27
        v21[v22] = v28
        del v28
        v22 += 1 
    del v22
    v29 = method96(v0)
    v30 = v0[32:].view(cp.uint8)
    if v29 == 0:
        v32 = method97(v30)
        v39 = US5_0(v32)
    elif v29 == 1:
        method87(v30)
        v39 = US5_1()
    elif v29 == 2:
        v35 = method98(v30)
        v39 = US5_2(v35)
    elif v29 == 3:
        v37 = method99(v30)
        v39 = US5_3(v37)
    else:
        raise Exception("Invalid tag.")
    del v29, v30
    v40 = method101(v0)
    v41 = v0[44:].view(cp.uint8)
    del v0
    if v40 == 0:
        method87(v41)
        v48 = US1_0()
    elif v40 == 1:
        method87(v41)
        v48 = US1_1()
    elif v40 == 2:
        method87(v41)
        v48 = US1_2()
    elif v40 == 3:
        v46 = method80(v41)
        v48 = US1_3(v46)
    else:
        raise Exception("Invalid tag.")
    del v40, v41
    return v2, v3, v11, v20, v21, v39, v48
def method93(v0 : cp.ndarray) -> US4:
    v1 = method80(v0)
    v2 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v4, v5, v6, v7, v8, v9 = method94(v2)
        del v2
        return US4_0(v4, v5, v6, v7, v8, v9)
    elif v1 == 1:
        del v1
        v11, v12, v13, v14, v15, v16 = method94(v2)
        del v2
        return US4_1(v11, v12, v13, v14, v15, v16)
    elif v1 == 2:
        del v1
        method87(v2)
        del v2
        return US4_2()
    elif v1 == 3:
        del v1
        v19, v20, v21, v22, v23, v24 = method94(v2)
        del v2
        return US4_3(v19, v20, v21, v22, v23, v24)
    elif v1 == 4:
        del v1
        v26, v27, v28, v29, v30, v31 = method94(v2)
        del v2
        return US4_4(v26, v27, v28, v29, v30, v31)
    elif v1 == 5:
        del v1
        v33, v34, v35, v36, v37, v38, v39 = method100(v2)
        del v2
        return US4_5(v33, v34, v35, v36, v37, v38, v39)
    elif v1 == 6:
        del v1
        v41, v42, v43, v44, v45, v46 = method94(v2)
        del v2
        return US4_6(v41, v42, v43, v44, v45, v46)
    elif v1 == 7:
        del v1
        v48, v49, v50, v51, v52, v53 = method94(v2)
        del v2
        return US4_7(v48, v49, v50, v51, v52, v53)
    else:
        del v1, v2
        raise Exception("Invalid tag.")
def method102(v0 : cp.ndarray) -> US2:
    v1 = method80(v0)
    v2 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method87(v2)
        del v2
        return US2_0()
    elif v1 == 1:
        del v1
        method87(v2)
        del v2
        return US2_1()
    else:
        del v1, v2
        raise Exception("Invalid tag.")
def method103(v0 : cp.ndarray) -> i32:
    v1 = v0[6248:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method76(v0 : cp.ndarray) -> Tuple[u64, dynamic_array_list, US3, static_array, US6]:
    v1 = method77(v0)
    v2 = dynamic_array_list(128)
    v3 = method78(v0)
    v2.unsafe_set_length(v3)
    del v3
    v4 = v2.length_()
    v5 = 0
    while method6(v4, v5):
        v7 = u64(v5)
        v8 = v7 * 48
        del v7
        v9 = 16 + v8
        del v8
        v10 = v0[v9:].view(cp.uint8)
        del v9
        v11 = method79(v10)
        del v10
        v2[v5] = v11
        del v11
        v5 += 1 
    del v4, v5
    v12 = method92(v0)
    v13 = v0[6176:].view(cp.uint8)
    if v12 == 0:
        method87(v13)
        v18 = US3_0()
    elif v12 == 1:
        v16 = method93(v13)
        v18 = US3_1(v16)
    else:
        raise Exception("Invalid tag.")
    del v12, v13
    v19 = static_array(2)
    v20 = 0
    while method44(v20):
        v22 = u64(v20)
        v23 = v22 * 4
        del v22
        v24 = 6240 + v23
        del v23
        v25 = v0[v24:].view(cp.uint8)
        del v24
        v26 = method102(v25)
        del v25
        v19[v20] = v26
        del v26
        v20 += 1 
    del v20
    v27 = method103(v0)
    v28 = v0[6256:].view(cp.uint8)
    del v0
    if v27 == 0:
        method87(v28)
        v45 = US6_0()
    elif v27 == 1:
        v31, v32, v33, v34, v35, v36 = method94(v28)
        v45 = US6_1(v31, v32, v33, v34, v35, v36)
    elif v27 == 2:
        v38, v39, v40, v41, v42, v43 = method94(v28)
        v45 = US6_2(v38, v39, v40, v41, v42, v43)
    else:
        raise Exception("Invalid tag.")
    del v27, v28
    return v1, v2, v18, v19, v45
def method105(v0 : cp.ndarray) -> i32:
    v1 = v0[6168:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method104(v0 : cp.ndarray) -> Tuple[dynamic_array_list, static_array, US6]:
    v1 = dynamic_array_list(128)
    v2 = method80(v0)
    v1.unsafe_set_length(v2)
    del v2
    v3 = v1.length_()
    v4 = 0
    while method6(v3, v4):
        v6 = u64(v4)
        v7 = v6 * 48
        del v6
        v8 = 16 + v7
        del v7
        v9 = v0[v8:].view(cp.uint8)
        del v8
        v10 = method79(v9)
        del v9
        v1[v4] = v10
        del v10
        v4 += 1 
    del v3, v4
    v11 = static_array(2)
    v12 = 0
    while method44(v12):
        v14 = u64(v12)
        v15 = v14 * 4
        del v14
        v16 = 6160 + v15
        del v15
        v17 = v0[v16:].view(cp.uint8)
        del v16
        v18 = method102(v17)
        del v17
        v11[v12] = v18
        del v18
        v12 += 1 
    del v12
    v19 = method105(v0)
    v20 = v0[6176:].view(cp.uint8)
    del v0
    if v19 == 0:
        method87(v20)
        v37 = US6_0()
    elif v19 == 1:
        v23, v24, v25, v26, v27, v28 = method94(v20)
        v37 = US6_1(v23, v24, v25, v26, v27, v28)
    elif v19 == 2:
        v30, v31, v32, v33, v34, v35 = method94(v20)
        v37 = US6_2(v30, v31, v32, v33, v34, v35)
    else:
        raise Exception("Invalid tag.")
    del v19, v20
    return v1, v11, v37
def method111(v0 : u64) -> object:
    v1 = v0
    del v0
    return v1
def method110(v0 : u64) -> object:
    return method111(v0)
def method116(v0 : u8) -> object:
    v1 = v0
    del v0
    return v1
def method115(v0 : u8) -> object:
    return method116(v0)
def method114(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method6(v2, v3):
        v5 = v0[v3]
        v6 = method115(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method118(v0 : i32) -> object:
    v1 = v0
    del v0
    return v1
def method117(v0 : i32, v1 : i32) -> object:
    v2 = method118(v0)
    del v0
    v3 = method118(v1)
    del v1
    v4 = {'chips_won': v2, 'winner_id': v3}
    del v2, v3
    return v4
def method121() -> object:
    v0 = []
    return v0
def method120(v0 : US1) -> object:
    match v0:
        case US1_0(): # A_All_In
            del v0
            v1 = method121()
            v2 = "A_All_In"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US1_1(): # A_Call
            del v0
            v4 = method121()
            v5 = "A_Call"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US1_2(): # A_Fold
            del v0
            v7 = method121()
            v8 = "A_Fold"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US1_3(v10): # A_Raise
            del v0
            v11 = method118(v10)
            del v10
            v12 = "A_Raise"
            v13 = [v12,v11]
            del v11, v12
            return v13
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method119(v0 : i32, v1 : US1) -> object:
    v2 = []
    v3 = method118(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method120(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method123(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method44(v2):
        v4 = v0[v2]
        v5 = method115(v4)
        del v4
        v1.append(v5)
        del v5
        v2 += 1 
    del v0, v2
    return v1
def method122(v0 : i32, v1 : static_array) -> object:
    v2 = []
    v3 = method118(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method123(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method128(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method60(v2):
        v4 = v0[v2]
        v5 = method115(v4)
        del v4
        v1.append(v5)
        del v5
        v2 += 1 
    del v0, v2
    return v1
def method129(v0 : i8) -> object:
    v1 = v0
    del v0
    return v1
def method127(v0 : static_array, v1 : i8) -> object:
    v2 = method128(v0)
    del v0
    v3 = method129(v1)
    del v1
    v4 = {'hand': v2, 'score': v3}
    del v2, v3
    return v4
def method126(v0 : static_array, v1 : i8) -> object:
    return method127(v0, v1)
def method125(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method44(v2):
        v4, v5 = v0[v2]
        v6 = method126(v4, v5)
        del v4, v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method124(v0 : i32, v1 : static_array, v2 : i32) -> object:
    v3 = method118(v0)
    del v0
    v4 = method125(v1)
    del v1
    v5 = method118(v2)
    del v2
    v6 = {'chips_won': v3, 'hands_shown': v4, 'winner_id': v5}
    del v3, v4, v5
    return v6
def method113(v0 : US7) -> object:
    match v0:
        case US7_0(v1): # CommunityCardsAre
            del v0
            v2 = method114(v1)
            del v1
            v3 = "CommunityCardsAre"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US7_1(v5, v6): # Fold
            del v0
            v7 = method117(v5, v6)
            del v5, v6
            v8 = "Fold"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US7_2(v10, v11): # PlayerAction
            del v0
            v12 = method119(v10, v11)
            del v10, v11
            v13 = "PlayerAction"
            v14 = [v13,v12]
            del v12, v13
            return v14
        case US7_3(v15, v16): # PlayerGotCards
            del v0
            v17 = method122(v15, v16)
            del v15, v16
            v18 = "PlayerGotCards"
            v19 = [v18,v17]
            del v17, v18
            return v19
        case US7_4(v20, v21, v22): # Showdown
            del v0
            v23 = method124(v20, v21, v22)
            del v20, v21, v22
            v24 = "Showdown"
            v25 = [v24,v23]
            del v23, v24
            return v25
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method112(v0 : dynamic_array_list) -> object:
    v1 = []
    v2 = v0.length_()
    v3 = 0
    while method6(v2, v3):
        v5 = v0[v3]
        v6 = method113(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method109(v0 : u64, v1 : dynamic_array_list) -> object:
    v2 = method110(v0)
    del v0
    v3 = method112(v1)
    del v1
    v4 = {'deck': v2, 'messages': v3}
    del v2, v3
    return v4
def method134(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method44(v2):
        v4 = v0[v2]
        v5 = method123(v4)
        del v4
        v1.append(v5)
        del v5
        v2 += 1 
    del v0, v2
    return v1
def method135(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method44(v2):
        v4 = v0[v2]
        v5 = method118(v4)
        del v4
        v1.append(v5)
        del v5
        v2 += 1 
    del v0, v2
    return v1
def method137(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method67(v2):
        v4 = v0[v2]
        v5 = method115(v4)
        del v4
        v1.append(v5)
        del v5
        v2 += 1 
    del v0, v2
    return v1
def method138(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method70(v2):
        v4 = v0[v2]
        v5 = method115(v4)
        del v4
        v1.append(v5)
        del v5
        v2 += 1 
    del v0, v2
    return v1
def method136(v0 : US5) -> object:
    match v0:
        case US5_0(v1): # Flop
            del v0
            v2 = method137(v1)
            del v1
            v3 = "Flop"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US5_1(): # Preflop
            del v0
            v5 = method121()
            v6 = "Preflop"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case US5_2(v8): # River
            del v0
            v9 = method128(v8)
            del v8
            v10 = "River"
            v11 = [v10,v9]
            del v9, v10
            return v11
        case US5_3(v12): # Turn
            del v0
            v13 = method138(v12)
            del v12
            v14 = "Turn"
            v15 = [v14,v13]
            del v13, v14
            return v15
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method133(v0 : i32, v1 : static_array, v2 : static_array, v3 : i32, v4 : static_array, v5 : US5) -> object:
    v6 = method118(v0)
    del v0
    v7 = method134(v1)
    del v1
    v8 = method135(v2)
    del v2
    v9 = method118(v3)
    del v3
    v10 = method135(v4)
    del v4
    v11 = method136(v5)
    del v5
    v12 = {'min_raise': v6, 'pl_card': v7, 'pot': v8, 'round_turn': v9, 'stack': v10, 'street': v11}
    del v6, v7, v8, v9, v10, v11
    return v12
def method139(v0 : i32, v1 : static_array, v2 : static_array, v3 : i32, v4 : static_array, v5 : US5, v6 : US1) -> object:
    v7 = []
    v8 = method133(v0, v1, v2, v3, v4, v5)
    del v0, v1, v2, v3, v4, v5
    v7.append(v8)
    del v8
    v9 = method120(v6)
    del v6
    v7.append(v9)
    del v9
    v10 = v7
    del v7
    return v10
def method132(v0 : US4) -> object:
    match v0:
        case US4_0(v1, v2, v3, v4, v5, v6): # G_Flop
            del v0
            v7 = method133(v1, v2, v3, v4, v5, v6)
            del v1, v2, v3, v4, v5, v6
            v8 = "G_Flop"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US4_1(v10, v11, v12, v13, v14, v15): # G_Fold
            del v0
            v16 = method133(v10, v11, v12, v13, v14, v15)
            del v10, v11, v12, v13, v14, v15
            v17 = "G_Fold"
            v18 = [v17,v16]
            del v16, v17
            return v18
        case US4_2(): # G_Preflop
            del v0
            v19 = method121()
            v20 = "G_Preflop"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case US4_3(v22, v23, v24, v25, v26, v27): # G_River
            del v0
            v28 = method133(v22, v23, v24, v25, v26, v27)
            del v22, v23, v24, v25, v26, v27
            v29 = "G_River"
            v30 = [v29,v28]
            del v28, v29
            return v30
        case US4_4(v31, v32, v33, v34, v35, v36): # G_Round
            del v0
            v37 = method133(v31, v32, v33, v34, v35, v36)
            del v31, v32, v33, v34, v35, v36
            v38 = "G_Round"
            v39 = [v38,v37]
            del v37, v38
            return v39
        case US4_5(v40, v41, v42, v43, v44, v45, v46): # G_Round'
            del v0
            v47 = method139(v40, v41, v42, v43, v44, v45, v46)
            del v40, v41, v42, v43, v44, v45, v46
            v48 = "G_Round'"
            v49 = [v48,v47]
            del v47, v48
            return v49
        case US4_6(v50, v51, v52, v53, v54, v55): # G_Showdown
            del v0
            v56 = method133(v50, v51, v52, v53, v54, v55)
            del v50, v51, v52, v53, v54, v55
            v57 = "G_Showdown"
            v58 = [v57,v56]
            del v56, v57
            return v58
        case US4_7(v59, v60, v61, v62, v63, v64): # G_Turn
            del v0
            v65 = method133(v59, v60, v61, v62, v63, v64)
            del v59, v60, v61, v62, v63, v64
            v66 = "G_Turn"
            v67 = [v66,v65]
            del v65, v66
            return v67
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method131(v0 : US3) -> object:
    match v0:
        case US3_0(): # None
            del v0
            v1 = method121()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US3_1(v4): # Some
            del v0
            v5 = method132(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method141(v0 : US2) -> object:
    match v0:
        case US2_0(): # Computer
            del v0
            v1 = method121()
            v2 = "Computer"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US2_1(): # Human
            del v0
            v4 = method121()
            v5 = "Human"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method140(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method44(v2):
        v4 = v0[v2]
        v5 = method141(v4)
        del v4
        v1.append(v5)
        del v5
        v2 += 1 
    del v0, v2
    return v1
def method142(v0 : US6) -> object:
    match v0:
        case US6_0(): # GameNotStarted
            del v0
            v1 = method121()
            v2 = "GameNotStarted"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US6_1(v4, v5, v6, v7, v8, v9): # GameOver
            del v0
            v10 = method133(v4, v5, v6, v7, v8, v9)
            del v4, v5, v6, v7, v8, v9
            v11 = "GameOver"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US6_2(v13, v14, v15, v16, v17, v18): # WaitingForActionFromPlayerId
            del v0
            v19 = method133(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "WaitingForActionFromPlayerId"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method130(v0 : US3, v1 : static_array, v2 : US6) -> object:
    v3 = method131(v0)
    del v0
    v4 = method140(v1)
    del v1
    v5 = method142(v2)
    del v2
    v6 = {'game': v3, 'pl_type': v4, 'ui_game_state': v5}
    del v3, v4, v5
    return v6
def method108(v0 : u64, v1 : dynamic_array_list, v2 : US3, v3 : static_array, v4 : US6) -> object:
    v5 = method109(v0, v1)
    del v0, v1
    v6 = method130(v2, v3, v4)
    del v2, v3, v4
    v7 = {'large': v5, 'small': v6}
    del v5, v6
    return v7
def method143(v0 : dynamic_array_list, v1 : static_array, v2 : US6) -> object:
    v3 = method112(v0)
    del v0
    v4 = method140(v1)
    del v1
    v5 = method142(v2)
    del v2
    v6 = {'messages': v3, 'pl_type': v4, 'ui_game_state': v5}
    del v3, v4, v5
    return v6
def method107(v0 : u64, v1 : dynamic_array_list, v2 : US3, v3 : static_array, v4 : US6, v5 : dynamic_array_list, v6 : static_array, v7 : US6) -> object:
    v8 = method108(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    v9 = method143(v5, v6, v7)
    del v5, v6, v7
    v10 = {'game_state': v8, 'ui_state': v9}
    del v8, v9
    return v10
def method106(v0 : u64, v1 : dynamic_array_list, v2 : US3, v3 : static_array, v4 : US6, v5 : dynamic_array_list, v6 : static_array, v7 : US6) -> object:
    v8 = method107(v0, v1, v2, v3, v4, v5, v6, v7)
    del v0, v1, v2, v3, v4, v5, v6, v7
    return v8
def method145(v0 : u64, v1 : dynamic_array_list, v2 : US3, v3 : static_array, v4 : US6) -> object:
    v5 = method108(v0, v1, v2, v3, v4)
    del v0, v2
    v6 = method143(v1, v3, v4)
    del v1, v3, v4
    v7 = {'game_state': v5, 'ui_state': v6}
    del v5, v6
    return v7
def method144(v0 : u64, v1 : dynamic_array_list, v2 : US3, v3 : static_array, v4 : US6) -> object:
    v5 = method145(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    return v5
def main():
    v0 = Closure0()
    v1 = Closure1()
    v2 = collections.namedtuple("HU_Holdem_Game",['event_loop_gpu', 'init'])(v0, v1)
    del v0, v1
    return v2

if __name__ == '__main__': print(main())
