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
struct Tuple0;
struct Union3;
struct Union6;
struct Union5;
struct Union4;
struct Union7;
struct Tuple1;
__device__ unsigned long long f_7(unsigned char * v0);
__device__ int f_8(unsigned char * v0);
__device__ unsigned char f_12(unsigned char * v0);
__device__ unsigned char f_11(unsigned char * v0);
__device__ static_array_list<unsigned char,5l> f_10(unsigned char * v0);
struct Tuple2;
__device__ Tuple2 f_13(unsigned char * v0);
struct Tuple3;
__device__ int f_15(unsigned char * v0);
__device__ Tuple3 f_14(unsigned char * v0);
struct Tuple4;
__device__ Tuple4 f_16(unsigned char * v0);
struct Tuple5;
__device__ Tuple0 f_19(unsigned char * v0);
__device__ Tuple0 f_18(unsigned char * v0);
__device__ Tuple5 f_17(unsigned char * v0);
__device__ Union3 f_9(unsigned char * v0);
__device__ int f_20(unsigned char * v0);
struct Tuple6;
__device__ static_array<unsigned char,2l> f_23(unsigned char * v0);
__device__ int f_24(unsigned char * v0);
__device__ static_array<unsigned char,3l> f_25(unsigned char * v0);
__device__ static_array<unsigned char,5l> f_26(unsigned char * v0);
__device__ static_array<unsigned char,4l> f_27(unsigned char * v0);
__device__ Tuple6 f_22(unsigned char * v0);
struct Tuple7;
__device__ int f_29(unsigned char * v0);
__device__ Tuple7 f_28(unsigned char * v0);
__device__ Union5 f_21(unsigned char * v0);
__device__ int f_30(unsigned char * v0);
__device__ Tuple1 f_6(unsigned char * v0);
struct Tuple8;
struct Tuple9;
struct Tuple10;
struct Tuple11;
struct Tuple12;
__device__ unsigned int loop_35(unsigned int v0, curandStatePhilox4_32_10_t & v1);
__device__ Tuple12 draw_card_34(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
__device__ Tuple10 draw_cards_33(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
__device__ static_array_list<unsigned char,5l> get_community_cards_36(Union6 v0, static_array<unsigned char,3l> v1);
__device__ bool player_can_act_38(int v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<int,2l> v2, int v3, static_array<int,2l> v4, Union6 v5);
__device__ Union5 go_next_street_39(int v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<int,2l> v2, int v3, static_array<int,2l> v4, Union6 v5);
__device__ Union5 try_round_37(int v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<int,2l> v2, int v3, static_array<int,2l> v4, Union6 v5);
struct Tuple13;
__device__ Tuple13 draw_cards_40(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
struct Tuple14;
__device__ Tuple14 draw_cards_41(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
__device__ static_array_list<unsigned char,5l> get_community_cards_42(Union6 v0, static_array<unsigned char,1l> v1);
struct Tuple15;
__device__ int loop_45(static_array<float,6l> v0, float v1, int v2);
__device__ int sample_discrete__44(static_array<float,6l> v0, curandStatePhilox4_32_10_t & v1);
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
__device__ void f_49(unsigned char * v0, int v1);
__device__ void f_51(unsigned char * v0, int v1);
__device__ void f_54(unsigned char * v0, unsigned char v1);
__device__ void f_53(unsigned char * v0, unsigned char v1);
__device__ void f_52(unsigned char * v0, static_array_list<unsigned char,5l> v1);
__device__ void f_55(unsigned char * v0, int v1, int v2);
__device__ void f_57(unsigned char * v0, int v1);
__device__ void f_58(unsigned char * v0);
__device__ void f_56(unsigned char * v0, int v1, Union1 v2);
__device__ void f_59(unsigned char * v0, int v1, static_array<unsigned char,2l> v2);
__device__ void f_62(unsigned char * v0, static_array<unsigned char,5l> v1, char v2);
__device__ void f_61(unsigned char * v0, static_array<unsigned char,5l> v1, char v2);
__device__ void f_60(unsigned char * v0, int v1, static_array<Tuple0,2l> v2, int v3);
__device__ void f_50(unsigned char * v0, Union3 v1);
__device__ void f_63(unsigned char * v0, int v1);
__device__ void f_66(unsigned char * v0, static_array<unsigned char,2l> v1);
__device__ void f_67(unsigned char * v0, int v1);
__device__ void f_68(unsigned char * v0, static_array<unsigned char,3l> v1);
__device__ void f_69(unsigned char * v0, static_array<unsigned char,5l> v1);
__device__ void f_70(unsigned char * v0, static_array<unsigned char,4l> v1);
__device__ void f_65(unsigned char * v0, int v1, static_array<static_array<unsigned char,2l>,2l> v2, static_array<int,2l> v3, int v4, static_array<int,2l> v5, Union6 v6);
__device__ void f_72(unsigned char * v0, int v1);
__device__ void f_71(unsigned char * v0, int v1, static_array<static_array<unsigned char,2l>,2l> v2, static_array<int,2l> v3, int v4, static_array<int,2l> v5, Union6 v6, Union1 v7);
__device__ void f_64(unsigned char * v0, Union5 v1);
__device__ void f_73(unsigned char * v0, Union2 v1);
__device__ void f_74(unsigned char * v0, int v1);
__device__ void f_47(unsigned char * v0, unsigned long long v1, dynamic_array_list<Union3,128l> v2, Union4 v3, static_array<Union2,2l> v4, Union7 v5);
__device__ void f_76(unsigned char * v0, int v1);
__device__ void f_75(unsigned char * v0, dynamic_array_list<Union3,128l> v1, static_array<Union2,2l> v2, Union7 v3);
struct Union1_0 { // A_All_In
};
struct Union1_1 { // A_Call
};
struct Union1_2 { // A_Fold
};
struct Union1_3 { // A_Raise
    int v0;
    __device__ Union1_3(int t0) : v0(t0) {}
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
    int v0;
    int v1;
    __device__ Union3_1(int t0, int t1) : v0(t0), v1(t1) {}
    __device__ Union3_1() = delete;
};
struct Union3_2 { // PlayerAction
    Union1 v1;
    int v0;
    __device__ Union3_2(int t0, Union1 t1) : v0(t0), v1(t1) {}
    __device__ Union3_2() = delete;
};
struct Union3_3 { // PlayerGotCards
    static_array<unsigned char,2l> v1;
    int v0;
    __device__ Union3_3(int t0, static_array<unsigned char,2l> t1) : v0(t0), v1(t1) {}
    __device__ Union3_3() = delete;
};
struct Union3_4 { // Showdown
    static_array<Tuple0,2l> v1;
    int v0;
    int v2;
    __device__ Union3_4(int t0, static_array<Tuple0,2l> t1, int t2) : v0(t0), v1(t1), v2(t2) {}
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
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union6 v5;
    int v0;
    int v3;
    __device__ Union5_0(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union5_0() = delete;
};
struct Union5_1 { // G_Fold
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union6 v5;
    int v0;
    int v3;
    __device__ Union5_1(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union5_1() = delete;
};
struct Union5_2 { // G_Preflop
};
struct Union5_3 { // G_River
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union6 v5;
    int v0;
    int v3;
    __device__ Union5_3(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union5_3() = delete;
};
struct Union5_4 { // G_Round
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union6 v5;
    int v0;
    int v3;
    __device__ Union5_4(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union5_4() = delete;
};
struct Union5_5 { // G_Round'
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union6 v5;
    Union1 v6;
    int v0;
    int v3;
    __device__ Union5_5(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union6 t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union5_5() = delete;
};
struct Union5_6 { // G_Showdown
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union6 v5;
    int v0;
    int v3;
    __device__ Union5_6(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union5_6() = delete;
};
struct Union5_7 { // G_Turn
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union6 v5;
    int v0;
    int v3;
    __device__ Union5_7(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
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
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union6 v5;
    int v0;
    int v3;
    __device__ Union7_1(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union7_1() = delete;
};
struct Union7_2 { // WaitingForActionFromPlayerId
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union6 v5;
    int v0;
    int v3;
    __device__ Union7_2(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
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
    int v0;
    int v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(int t0, int t1) : v0(t0), v1(t1) {}
};
struct Tuple3 {
    Union1 v1;
    int v0;
    __device__ Tuple3() = default;
    __device__ Tuple3(int t0, Union1 t1) : v0(t0), v1(t1) {}
};
struct Tuple4 {
    static_array<unsigned char,2l> v1;
    int v0;
    __device__ Tuple4() = default;
    __device__ Tuple4(int t0, static_array<unsigned char,2l> t1) : v0(t0), v1(t1) {}
};
struct Tuple5 {
    static_array<Tuple0,2l> v1;
    int v0;
    int v2;
    __device__ Tuple5() = default;
    __device__ Tuple5(int t0, static_array<Tuple0,2l> t1, int t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple6 {
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union6 v5;
    int v0;
    int v3;
    __device__ Tuple6() = default;
    __device__ Tuple6(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union6 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Tuple7 {
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union6 v5;
    Union1 v6;
    int v0;
    int v3;
    __device__ Tuple7() = default;
    __device__ Tuple7(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union6 t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
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
    int v0;
    __device__ Tuple11() = default;
    __device__ Tuple11(int t0, unsigned long long t1) : v0(t0), v1(t1) {}
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
    int v1;
    bool v0;
    __device__ Tuple16() = default;
    __device__ Tuple16(bool t0, int t1) : v0(t0), v1(t1) {}
};
struct Tuple17 {
    int v0;
    int v1;
    int v2;
    __device__ Tuple17() = default;
    __device__ Tuple17(int t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
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
    int v0;
    int v1;
    unsigned char v2;
    __device__ Tuple18() = default;
    __device__ Tuple18(int t0, int t1, unsigned char t2) : v0(t0), v1(t1), v2(t2) {}
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
    int v0;
    __device__ Tuple19() = default;
    __device__ Tuple19(int t0, Union8 t1) : v0(t0), v1(t1) {}
};
struct Tuple20 {
    int v0;
    int v1;
    int v2;
    unsigned char v3;
    __device__ Tuple20() = default;
    __device__ Tuple20(int t0, int t1, int t2, unsigned char t3) : v0(t0), v1(t1), v2(t2), v3(t3) {}
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
        case 3: {
            int v8;
            v8 = f_1(v2);
            return Union1{Union1_3{v8}};
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
__device__ unsigned long long f_7(unsigned char * v0){
    unsigned long long * v1;
    v1 = (unsigned long long *)(v0+0ull);
    unsigned long long v3;
    v3 = v1[0l];
    return v3;
}
__device__ int f_8(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+8ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ inline bool while_method_1(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ unsigned char f_12(unsigned char * v0){
    unsigned char * v1;
    v1 = (unsigned char *)(v0+0ull);
    unsigned char v3;
    v3 = v1[0l];
    return v3;
}
__device__ unsigned char f_11(unsigned char * v0){
    unsigned char v1;
    v1 = f_12(v0);
    return v1;
}
__device__ static_array_list<unsigned char,5l> f_10(unsigned char * v0){
    static_array_list<unsigned char,5l> v1;
    v1 = static_array_list<unsigned char,5l>{};
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
        v8 = 4ull + v7;
        unsigned char * v9;
        v9 = (unsigned char *)(v0+v8);
        unsigned char v11;
        v11 = f_11(v9);
        v1[v5] = v11;
        v5 += 1l ;
    }
    return v1;
}
__device__ Tuple2 f_13(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    int * v4;
    v4 = (int *)(v0+4ull);
    int v6;
    v6 = v4[0l];
    return Tuple2{v3, v6};
}
__device__ int f_15(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+4ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ Tuple3 f_14(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    int v4;
    v4 = f_15(v0);
    unsigned char * v5;
    v5 = (unsigned char *)(v0+8ull);
    Union1 v13;
    switch (v4) {
        case 0: {
            f_3(v5);
            v13 = Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v5);
            v13 = Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v5);
            v13 = Union1{Union1_2{}};
            break;
        }
        case 3: {
            int v11;
            v11 = f_1(v5);
            v13 = Union1{Union1_3{v11}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple3{v3, v13};
}
__device__ Tuple4 f_16(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    static_array<unsigned char,2l> v4;
    int v6;
    v6 = 0l;
    while (while_method_0(v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = 4ull + v8;
        unsigned char * v10;
        v10 = (unsigned char *)(v0+v9);
        unsigned char v12;
        v12 = f_11(v10);
        v4[v6] = v12;
        v6 += 1l ;
    }
    return Tuple4{v3, v4};
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 5l;
    return v1;
}
__device__ Tuple0 f_19(unsigned char * v0){
    static_array<unsigned char,5l> v1;
    int v3;
    v3 = 0l;
    while (while_method_2(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = f_11(v6);
        v1[v3] = v8;
        v3 += 1l ;
    }
    char * v9;
    v9 = (char *)(v0+5ull);
    char v11;
    v11 = v9[0l];
    return Tuple0{v1, v11};
}
__device__ Tuple0 f_18(unsigned char * v0){
    static_array<unsigned char,5l> v1; char v2;
    Tuple0 tmp3 = f_19(v0);
    v1 = tmp3.v0; v2 = tmp3.v1;
    return Tuple0{v1, v2};
}
__device__ Tuple5 f_17(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    static_array<Tuple0,2l> v4;
    int v6;
    v6 = 0l;
    while (while_method_0(v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = v8 * 8ull;
        unsigned long long v10;
        v10 = 8ull + v9;
        unsigned char * v11;
        v11 = (unsigned char *)(v0+v10);
        static_array<unsigned char,5l> v13; char v14;
        Tuple0 tmp4 = f_18(v11);
        v13 = tmp4.v0; v14 = tmp4.v1;
        v4[v6] = Tuple0{v13, v14};
        v6 += 1l ;
    }
    int * v15;
    v15 = (int *)(v0+24ull);
    int v17;
    v17 = v15[0l];
    return Tuple5{v3, v4, v17};
}
__device__ Union3 f_9(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            static_array_list<unsigned char,5l> v5;
            v5 = f_10(v2);
            return Union3{Union3_0{v5}};
            break;
        }
        case 1: {
            int v7; int v8;
            Tuple2 tmp0 = f_13(v2);
            v7 = tmp0.v0; v8 = tmp0.v1;
            return Union3{Union3_1{v7, v8}};
            break;
        }
        case 2: {
            int v10; Union1 v11;
            Tuple3 tmp1 = f_14(v2);
            v10 = tmp1.v0; v11 = tmp1.v1;
            return Union3{Union3_2{v10, v11}};
            break;
        }
        case 3: {
            int v13; static_array<unsigned char,2l> v14;
            Tuple4 tmp2 = f_16(v2);
            v13 = tmp2.v0; v14 = tmp2.v1;
            return Union3{Union3_3{v13, v14}};
            break;
        }
        case 4: {
            int v16; static_array<Tuple0,2l> v17; int v18;
            Tuple5 tmp5 = f_17(v2);
            v16 = tmp5.v0; v17 = tmp5.v1; v18 = tmp5.v2;
            return Union3{Union3_4{v16, v17, v18}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ int f_20(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+6160ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ static_array<unsigned char,2l> f_23(unsigned char * v0){
    static_array<unsigned char,2l> v1;
    int v3;
    v3 = 0l;
    while (while_method_0(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = f_11(v6);
        v1[v3] = v8;
        v3 += 1l ;
    }
    return v1;
}
__device__ int f_24(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+28ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 3l;
    return v1;
}
__device__ static_array<unsigned char,3l> f_25(unsigned char * v0){
    static_array<unsigned char,3l> v1;
    int v3;
    v3 = 0l;
    while (while_method_3(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = f_11(v6);
        v1[v3] = v8;
        v3 += 1l ;
    }
    return v1;
}
__device__ static_array<unsigned char,5l> f_26(unsigned char * v0){
    static_array<unsigned char,5l> v1;
    int v3;
    v3 = 0l;
    while (while_method_2(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = f_11(v6);
        v1[v3] = v8;
        v3 += 1l ;
    }
    return v1;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ static_array<unsigned char,4l> f_27(unsigned char * v0){
    static_array<unsigned char,4l> v1;
    int v3;
    v3 = 0l;
    while (while_method_4(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = f_11(v6);
        v1[v3] = v8;
        v3 += 1l ;
    }
    return v1;
}
__device__ Tuple6 f_22(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    static_array<static_array<unsigned char,2l>,2l> v4;
    int v6;
    v6 = 0l;
    while (while_method_0(v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = v8 * 2ull;
        unsigned long long v10;
        v10 = 4ull + v9;
        unsigned char * v11;
        v11 = (unsigned char *)(v0+v10);
        static_array<unsigned char,2l> v13;
        v13 = f_23(v11);
        v4[v6] = v13;
        v6 += 1l ;
    }
    static_array<int,2l> v14;
    int v16;
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
        int v23;
        v23 = f_1(v21);
        v14[v16] = v23;
        v16 += 1l ;
    }
    int * v24;
    v24 = (int *)(v0+16ull);
    int v26;
    v26 = v24[0l];
    static_array<int,2l> v27;
    int v29;
    v29 = 0l;
    while (while_method_0(v29)){
        unsigned long long v31;
        v31 = (unsigned long long)v29;
        unsigned long long v32;
        v32 = v31 * 4ull;
        unsigned long long v33;
        v33 = 20ull + v32;
        unsigned char * v34;
        v34 = (unsigned char *)(v0+v33);
        int v36;
        v36 = f_1(v34);
        v27[v29] = v36;
        v29 += 1l ;
    }
    int v37;
    v37 = f_24(v0);
    unsigned char * v38;
    v38 = (unsigned char *)(v0+32ull);
    Union6 v48;
    switch (v37) {
        case 0: {
            static_array<unsigned char,3l> v41;
            v41 = f_25(v38);
            v48 = Union6{Union6_0{v41}};
            break;
        }
        case 1: {
            f_3(v38);
            v48 = Union6{Union6_1{}};
            break;
        }
        case 2: {
            static_array<unsigned char,5l> v44;
            v44 = f_26(v38);
            v48 = Union6{Union6_2{v44}};
            break;
        }
        case 3: {
            static_array<unsigned char,4l> v46;
            v46 = f_27(v38);
            v48 = Union6{Union6_3{v46}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple6{v3, v4, v14, v26, v27, v48};
}
__device__ int f_29(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+40ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ Tuple7 f_28(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    static_array<static_array<unsigned char,2l>,2l> v4;
    int v6;
    v6 = 0l;
    while (while_method_0(v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = v8 * 2ull;
        unsigned long long v10;
        v10 = 4ull + v9;
        unsigned char * v11;
        v11 = (unsigned char *)(v0+v10);
        static_array<unsigned char,2l> v13;
        v13 = f_23(v11);
        v4[v6] = v13;
        v6 += 1l ;
    }
    static_array<int,2l> v14;
    int v16;
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
        int v23;
        v23 = f_1(v21);
        v14[v16] = v23;
        v16 += 1l ;
    }
    int * v24;
    v24 = (int *)(v0+16ull);
    int v26;
    v26 = v24[0l];
    static_array<int,2l> v27;
    int v29;
    v29 = 0l;
    while (while_method_0(v29)){
        unsigned long long v31;
        v31 = (unsigned long long)v29;
        unsigned long long v32;
        v32 = v31 * 4ull;
        unsigned long long v33;
        v33 = 20ull + v32;
        unsigned char * v34;
        v34 = (unsigned char *)(v0+v33);
        int v36;
        v36 = f_1(v34);
        v27[v29] = v36;
        v29 += 1l ;
    }
    int v37;
    v37 = f_24(v0);
    unsigned char * v38;
    v38 = (unsigned char *)(v0+32ull);
    Union6 v48;
    switch (v37) {
        case 0: {
            static_array<unsigned char,3l> v41;
            v41 = f_25(v38);
            v48 = Union6{Union6_0{v41}};
            break;
        }
        case 1: {
            f_3(v38);
            v48 = Union6{Union6_1{}};
            break;
        }
        case 2: {
            static_array<unsigned char,5l> v44;
            v44 = f_26(v38);
            v48 = Union6{Union6_2{v44}};
            break;
        }
        case 3: {
            static_array<unsigned char,4l> v46;
            v46 = f_27(v38);
            v48 = Union6{Union6_3{v46}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    int v49;
    v49 = f_29(v0);
    unsigned char * v50;
    v50 = (unsigned char *)(v0+44ull);
    Union1 v58;
    switch (v49) {
        case 0: {
            f_3(v50);
            v58 = Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v50);
            v58 = Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v50);
            v58 = Union1{Union1_2{}};
            break;
        }
        case 3: {
            int v56;
            v56 = f_1(v50);
            v58 = Union1{Union1_3{v56}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple7{v3, v4, v14, v26, v27, v48, v58};
}
__device__ Union5 f_21(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            int v5; static_array<static_array<unsigned char,2l>,2l> v6; static_array<int,2l> v7; int v8; static_array<int,2l> v9; Union6 v10;
            Tuple6 tmp6 = f_22(v2);
            v5 = tmp6.v0; v6 = tmp6.v1; v7 = tmp6.v2; v8 = tmp6.v3; v9 = tmp6.v4; v10 = tmp6.v5;
            return Union5{Union5_0{v5, v6, v7, v8, v9, v10}};
            break;
        }
        case 1: {
            int v12; static_array<static_array<unsigned char,2l>,2l> v13; static_array<int,2l> v14; int v15; static_array<int,2l> v16; Union6 v17;
            Tuple6 tmp7 = f_22(v2);
            v12 = tmp7.v0; v13 = tmp7.v1; v14 = tmp7.v2; v15 = tmp7.v3; v16 = tmp7.v4; v17 = tmp7.v5;
            return Union5{Union5_1{v12, v13, v14, v15, v16, v17}};
            break;
        }
        case 2: {
            f_3(v2);
            return Union5{Union5_2{}};
            break;
        }
        case 3: {
            int v20; static_array<static_array<unsigned char,2l>,2l> v21; static_array<int,2l> v22; int v23; static_array<int,2l> v24; Union6 v25;
            Tuple6 tmp8 = f_22(v2);
            v20 = tmp8.v0; v21 = tmp8.v1; v22 = tmp8.v2; v23 = tmp8.v3; v24 = tmp8.v4; v25 = tmp8.v5;
            return Union5{Union5_3{v20, v21, v22, v23, v24, v25}};
            break;
        }
        case 4: {
            int v27; static_array<static_array<unsigned char,2l>,2l> v28; static_array<int,2l> v29; int v30; static_array<int,2l> v31; Union6 v32;
            Tuple6 tmp9 = f_22(v2);
            v27 = tmp9.v0; v28 = tmp9.v1; v29 = tmp9.v2; v30 = tmp9.v3; v31 = tmp9.v4; v32 = tmp9.v5;
            return Union5{Union5_4{v27, v28, v29, v30, v31, v32}};
            break;
        }
        case 5: {
            int v34; static_array<static_array<unsigned char,2l>,2l> v35; static_array<int,2l> v36; int v37; static_array<int,2l> v38; Union6 v39; Union1 v40;
            Tuple7 tmp10 = f_28(v2);
            v34 = tmp10.v0; v35 = tmp10.v1; v36 = tmp10.v2; v37 = tmp10.v3; v38 = tmp10.v4; v39 = tmp10.v5; v40 = tmp10.v6;
            return Union5{Union5_5{v34, v35, v36, v37, v38, v39, v40}};
            break;
        }
        case 6: {
            int v42; static_array<static_array<unsigned char,2l>,2l> v43; static_array<int,2l> v44; int v45; static_array<int,2l> v46; Union6 v47;
            Tuple6 tmp11 = f_22(v2);
            v42 = tmp11.v0; v43 = tmp11.v1; v44 = tmp11.v2; v45 = tmp11.v3; v46 = tmp11.v4; v47 = tmp11.v5;
            return Union5{Union5_6{v42, v43, v44, v45, v46, v47}};
            break;
        }
        case 7: {
            int v49; static_array<static_array<unsigned char,2l>,2l> v50; static_array<int,2l> v51; int v52; static_array<int,2l> v53; Union6 v54;
            Tuple6 tmp12 = f_22(v2);
            v49 = tmp12.v0; v50 = tmp12.v1; v51 = tmp12.v2; v52 = tmp12.v3; v53 = tmp12.v4; v54 = tmp12.v5;
            return Union5{Union5_7{v49, v50, v51, v52, v53, v54}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ int f_30(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+6248ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ Tuple1 f_6(unsigned char * v0){
    unsigned long long v1;
    v1 = f_7(v0);
    dynamic_array_list<Union3,128l> v2{0};
    int v4;
    v4 = f_8(v0);
    v2.unsafe_set_length(v4);
    int v5;
    v5 = v2.length_();
    int v6;
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
        Union3 v13;
        v13 = f_9(v11);
        v2[v6] = v13;
        v6 += 1l ;
    }
    int v14;
    v14 = f_20(v0);
    unsigned char * v15;
    v15 = (unsigned char *)(v0+6176ull);
    Union4 v21;
    switch (v14) {
        case 0: {
            f_3(v15);
            v21 = Union4{Union4_0{}};
            break;
        }
        case 1: {
            Union5 v19;
            v19 = f_21(v15);
            v21 = Union4{Union4_1{v19}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
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
        v28 = 6240ull + v27;
        unsigned char * v29;
        v29 = (unsigned char *)(v0+v28);
        Union2 v31;
        v31 = f_5(v29);
        v22[v24] = v31;
        v24 += 1l ;
    }
    int v32;
    v32 = f_30(v0);
    unsigned char * v33;
    v33 = (unsigned char *)(v0+6256ull);
    Union7 v51;
    switch (v32) {
        case 0: {
            f_3(v33);
            v51 = Union7{Union7_0{}};
            break;
        }
        case 1: {
            int v37; static_array<static_array<unsigned char,2l>,2l> v38; static_array<int,2l> v39; int v40; static_array<int,2l> v41; Union6 v42;
            Tuple6 tmp13 = f_22(v33);
            v37 = tmp13.v0; v38 = tmp13.v1; v39 = tmp13.v2; v40 = tmp13.v3; v41 = tmp13.v4; v42 = tmp13.v5;
            v51 = Union7{Union7_1{v37, v38, v39, v40, v41, v42}};
            break;
        }
        case 2: {
            int v44; static_array<static_array<unsigned char,2l>,2l> v45; static_array<int,2l> v46; int v47; static_array<int,2l> v48; Union6 v49;
            Tuple6 tmp14 = f_22(v33);
            v44 = tmp14.v0; v45 = tmp14.v1; v46 = tmp14.v2; v47 = tmp14.v3; v48 = tmp14.v4; v49 = tmp14.v5;
            v51 = Union7{Union7_2{v44, v45, v46, v47, v48, v49}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple1{v1, v2, v21, v22, v51};
}
__device__ inline bool while_method_5(bool v0, Union5 v1){
    return v0;
}
__device__ unsigned int loop_35(unsigned int v0, curandStatePhilox4_32_10_t & v1){
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
        return loop_35(v0, v1);
    }
}
__device__ Tuple12 draw_card_34(curandStatePhilox4_32_10_t & v0, unsigned long long v1){
    int v2;
    v2 = __popcll(v1);
    unsigned int v3;
    v3 = (unsigned int)v2;
    unsigned int v4;
    v4 = loop_35(v3, v0);
    int v5;
    v5 = (int)v4;
    unsigned int v6;
    v6 = (unsigned int)v1;
    unsigned long long v7;
    v7 = v1 >> 32l;
    unsigned int v8;
    v8 = (unsigned int)v7;
    int v9;
    v9 = __popc(v6);
    bool v10;
    v10 = v5 < v9;
    unsigned int v17;
    if (v10){
        int v11;
        v11 = v5 + 1l;
        unsigned int v12;
        v12 = __fns(v6,0ul,v11);
        v17 = v12;
    } else {
        int v13;
        v13 = v5 - v9;
        int v14;
        v14 = v13 + 1l;
        unsigned int v15;
        v15 = __fns(v8,0ul,v14);
        unsigned int v16;
        v16 = v15 + 32ul;
        v17 = v16;
    }
    unsigned char v18;
    v18 = (unsigned char)v17;
    int v19;
    v19 = (int)v17;
    unsigned long long v20;
    v20 = 1ull << v19;
    unsigned long long v21;
    v21 = v1 ^ v20;
    return Tuple12{v18, v21};
}
__device__ Tuple10 draw_cards_33(curandStatePhilox4_32_10_t & v0, unsigned long long v1){
    static_array<unsigned char,3l> v2;
    int v4; unsigned long long v5;
    Tuple11 tmp17 = Tuple11{0l, v1};
    v4 = tmp17.v0; v5 = tmp17.v1;
    while (while_method_3(v4)){
        unsigned char v7; unsigned long long v8;
        Tuple12 tmp18 = draw_card_34(v0, v5);
        v7 = tmp18.v0; v8 = tmp18.v1;
        v2[v4] = v7;
        v5 = v8;
        v4 += 1l ;
    }
    return Tuple10{v2, v5};
}
__device__ static_array_list<unsigned char,5l> get_community_cards_36(Union6 v0, static_array<unsigned char,3l> v1){
    static_array_list<unsigned char,5l> v2;
    v2 = static_array_list<unsigned char,5l>{};
    switch (v0.tag) {
        case 0: { // Flop
            static_array<unsigned char,3l> v4 = v0.case0.v0;
            int v5;
            v5 = 0l;
            while (while_method_3(v5)){
                unsigned char v7;
                v7 = v4[v5];
                v2.push(v7);
                v5 += 1l ;
            }
            break;
        }
        case 1: { // Preflop
            break;
        }
        case 2: { // River
            static_array<unsigned char,5l> v14 = v0.case2.v0;
            int v15;
            v15 = 0l;
            while (while_method_2(v15)){
                unsigned char v17;
                v17 = v14[v15];
                v2.push(v17);
                v15 += 1l ;
            }
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4l> v9 = v0.case3.v0;
            int v10;
            v10 = 0l;
            while (while_method_4(v10)){
                unsigned char v12;
                v12 = v9[v10];
                v2.push(v12);
                v10 += 1l ;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    int v19;
    v19 = 0l;
    while (while_method_3(v19)){
        unsigned char v21;
        v21 = v1[v19];
        v2.push(v21);
        v19 += 1l ;
    }
    return v2;
}
__device__ bool player_can_act_38(int v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<int,2l> v2, int v3, static_array<int,2l> v4, Union6 v5){
    int v6;
    v6 = v3 % 2l;
    int v7;
    v7 = v4[v6];
    bool v9;
    v9 = v7 > 0l;
    int v10;
    v10 = v2[v6];
    int v12;
    v12 = v2[0l];
    int v14; int v15;
    Tuple2 tmp20 = Tuple2{1l, v12};
    v14 = tmp20.v0; v15 = tmp20.v1;
    while (while_method_0(v14)){
        int v17;
        v17 = v2[v14];
        bool v19;
        v19 = v15 >= v17;
        int v20;
        if (v19){
            v20 = v15;
        } else {
            v20 = v17;
        }
        v15 = v20;
        v14 += 1l ;
    }
    bool v21;
    v21 = v10 < v15;
    int v22; int v23;
    Tuple2 tmp21 = Tuple2{0l, 0l};
    v22 = tmp21.v0; v23 = tmp21.v1;
    while (while_method_0(v22)){
        int v25;
        v25 = v4[v22];
        bool v27;
        v27 = 0l < v25;
        int v28;
        if (v27){
            v28 = 1l;
        } else {
            v28 = 0l;
        }
        int v29;
        v29 = v23 + v28;
        v23 = v29;
        v22 += 1l ;
    }
    if (v9){
        if (v21){
            return true;
        } else {
            bool v30;
            v30 = v3 < 2l;
            if (v30){
                bool v31;
                v31 = 0l < v23;
                return v31;
            } else {
                return false;
            }
        }
    } else {
        return false;
    }
}
__device__ Union5 go_next_street_39(int v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<int,2l> v2, int v3, static_array<int,2l> v4, Union6 v5){
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
__device__ Union5 try_round_37(int v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<int,2l> v2, int v3, static_array<int,2l> v4, Union6 v5){
    int v6;
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
    int v4; unsigned long long v5;
    Tuple11 tmp22 = Tuple11{0l, v1};
    v4 = tmp22.v0; v5 = tmp22.v1;
    while (while_method_0(v4)){
        unsigned char v7; unsigned long long v8;
        Tuple12 tmp23 = draw_card_34(v0, v5);
        v7 = tmp23.v0; v8 = tmp23.v1;
        v2[v4] = v7;
        v5 = v8;
        v4 += 1l ;
    }
    return Tuple13{v2, v5};
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ Tuple14 draw_cards_41(curandStatePhilox4_32_10_t & v0, unsigned long long v1){
    static_array<unsigned char,1l> v2;
    int v4; unsigned long long v5;
    Tuple11 tmp26 = Tuple11{0l, v1};
    v4 = tmp26.v0; v5 = tmp26.v1;
    while (while_method_6(v4)){
        unsigned char v7; unsigned long long v8;
        Tuple12 tmp27 = draw_card_34(v0, v5);
        v7 = tmp27.v0; v8 = tmp27.v1;
        v2[v4] = v7;
        v5 = v8;
        v4 += 1l ;
    }
    return Tuple14{v2, v5};
}
__device__ static_array_list<unsigned char,5l> get_community_cards_42(Union6 v0, static_array<unsigned char,1l> v1){
    static_array_list<unsigned char,5l> v2;
    v2 = static_array_list<unsigned char,5l>{};
    switch (v0.tag) {
        case 0: { // Flop
            static_array<unsigned char,3l> v4 = v0.case0.v0;
            int v5;
            v5 = 0l;
            while (while_method_3(v5)){
                unsigned char v7;
                v7 = v4[v5];
                v2.push(v7);
                v5 += 1l ;
            }
            break;
        }
        case 1: { // Preflop
            break;
        }
        case 2: { // River
            static_array<unsigned char,5l> v14 = v0.case2.v0;
            int v15;
            v15 = 0l;
            while (while_method_2(v15)){
                unsigned char v17;
                v17 = v14[v15];
                v2.push(v17);
                v15 += 1l ;
            }
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4l> v9 = v0.case3.v0;
            int v10;
            v10 = 0l;
            while (while_method_4(v10)){
                unsigned char v12;
                v12 = v9[v10];
                v2.push(v12);
                v10 += 1l ;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    int v19;
    v19 = 0l;
    while (while_method_6(v19)){
        unsigned char v21;
        v21 = v1[v19];
        v2.push(v21);
        v19 += 1l ;
    }
    return v2;
}
__device__ inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 6l;
    return v1;
}
__device__ inline bool while_method_8(static_array<float,6l> v0, int v1){
    bool v2;
    v2 = v1 < 6l;
    return v2;
}
__device__ inline bool while_method_9(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    return v2;
}
__device__ int loop_45(static_array<float,6l> v0, float v1, int v2){
    bool v3;
    v3 = v2 < 6l;
    if (v3){
        float v4;
        v4 = v0[v2];
        bool v6;
        v6 = v1 <= v4;
        if (v6){
            return v2;
        } else {
            int v7;
            v7 = v2 + 1l;
            return loop_45(v0, v1, v7);
        }
    } else {
        return 5l;
    }
}
__device__ int sample_discrete__44(static_array<float,6l> v0, curandStatePhilox4_32_10_t & v1){
    static_array<float,6l> v2;
    int v4;
    v4 = 0l;
    while (while_method_7(v4)){
        float v6;
        v6 = v0[v4];
        v2[v4] = v6;
        v4 += 1l ;
    }
    int v8;
    v8 = 1l;
    while (while_method_8(v2, v8)){
        int v10;
        v10 = 6l;
        while (while_method_9(v8, v10)){
            v10 -= 1l ;
            int v12;
            v12 = v10 - v8;
            float v13;
            v13 = v2[v12];
            float v15;
            v15 = v2[v10];
            float v17;
            v17 = v13 + v15;
            v2[v10] = v17;
        }
        int v18;
        v18 = v8 * 2l;
        v8 = v18;
    }
    float v19;
    v19 = v2[5l];
    float v21;
    v21 = curand_uniform(&v1);
    float v22;
    v22 = v21 * v19;
    int v23;
    v23 = 0l;
    return loop_45(v2, v22, v23);
}
__device__ Union1 sample_discrete_43(static_array<Tuple15,6l> v0, curandStatePhilox4_32_10_t & v1){
    static_array<float,6l> v2;
    int v4;
    v4 = 0l;
    while (while_method_7(v4)){
        Union1 v6; float v7;
        Tuple15 tmp31 = v0[v4];
        v6 = tmp31.v0; v7 = tmp31.v1;
        v2[v4] = v7;
        v4 += 1l ;
    }
    int v10;
    v10 = sample_discrete__44(v2, v1);
    Union1 v11; float v12;
    Tuple15 tmp32 = v0[v10];
    v11 = tmp32.v0; v12 = tmp32.v1;
    return v11;
}
__device__ inline bool while_method_10(int v0){
    bool v1;
    v1 = v0 < 7l;
    return v1;
}
__device__ inline bool while_method_11(static_array<unsigned char,7l> v0, bool v1, int v2){
    bool v3;
    v3 = v2 < 7l;
    return v3;
}
__device__ inline bool while_method_12(static_array<unsigned char,7l> v0, int v1){
    bool v2;
    v2 = v1 < 7l;
    return v2;
}
__device__ inline bool while_method_13(int v0, int v1, int v2, int v3){
    bool v4;
    v4 = v3 < v0;
    return v4;
}
__device__ Tuple0 score_46(static_array<unsigned char,7l> v0){
    static_array<unsigned char,7l> v1;
    int v3;
    v3 = 0l;
    while (while_method_10(v3)){
        unsigned char v5;
        v5 = v0[v3];
        v1[v3] = v5;
        v3 += 1l ;
    }
    static_array<unsigned char,7l> v7;
    bool v9; int v10;
    Tuple16 tmp39 = Tuple16{true, 1l};
    v9 = tmp39.v0; v10 = tmp39.v1;
    while (while_method_11(v1, v9, v10)){
        int v12;
        v12 = 0l;
        while (while_method_12(v1, v12)){
            int v14;
            v14 = v12 + v10;
            bool v15;
            v15 = v14 < 7l;
            int v16;
            if (v15){
                v16 = v14;
            } else {
                v16 = 7l;
            }
            int v17;
            v17 = v10 * 2l;
            int v18;
            v18 = v12 + v17;
            bool v19;
            v19 = v18 < 7l;
            int v20;
            if (v19){
                v20 = v18;
            } else {
                v20 = 7l;
            }
            int v21; int v22; int v23;
            Tuple17 tmp40 = Tuple17{v12, v16, v12};
            v21 = tmp40.v0; v22 = tmp40.v1; v23 = tmp40.v2;
            while (while_method_13(v20, v21, v22, v23)){
                bool v25;
                v25 = v21 < v16;
                bool v27;
                if (v25){
                    bool v26;
                    v26 = v22 < v20;
                    v27 = v26;
                } else {
                    v27 = false;
                }
                unsigned char v77; int v78; int v79;
                if (v27){
                    unsigned char v32;
                    if (v9){
                        unsigned char v28;
                        v28 = v1[v21];
                        v32 = v28;
                    } else {
                        unsigned char v30;
                        v30 = v7[v21];
                        v32 = v30;
                    }
                    unsigned char v37;
                    if (v9){
                        unsigned char v33;
                        v33 = v1[v22];
                        v37 = v33;
                    } else {
                        unsigned char v35;
                        v35 = v7[v22];
                        v37 = v35;
                    }
                    unsigned char v38;
                    v38 = v37 / 4u;
                    unsigned char v39;
                    v39 = v32 / 4u;
                    bool v40;
                    v40 = v38 < v39;
                    Union8 v46;
                    if (v40){
                        v46 = Union8{Union8_2{}};
                    } else {
                        bool v42;
                        v42 = v38 > v39;
                        if (v42){
                            v46 = Union8{Union8_1{}};
                        } else {
                            v46 = Union8{Union8_0{}};
                        }
                    }
                    Union8 v56;
                    switch (v46.tag) {
                        case 0: { // Eq
                            unsigned char v47;
                            v47 = v32 % 4u;
                            unsigned char v48;
                            v48 = v37 % 4u;
                            bool v49;
                            v49 = v47 < v48;
                            if (v49){
                                v56 = Union8{Union8_2{}};
                            } else {
                                bool v51;
                                v51 = v47 > v48;
                                if (v51){
                                    v56 = Union8{Union8_1{}};
                                } else {
                                    v56 = Union8{Union8_0{}};
                                }
                            }
                            break;
                        }
                        default: {
                            v56 = v46;
                        }
                    }
                    switch (v56.tag) {
                        case 1: { // Gt
                            int v57;
                            v57 = v22 + 1l;
                            v77 = v37; v78 = v21; v79 = v57;
                            break;
                        }
                        default: {
                            int v58;
                            v58 = v21 + 1l;
                            v77 = v32; v78 = v58; v79 = v22;
                        }
                    }
                } else {
                    if (v25){
                        unsigned char v66;
                        if (v9){
                            unsigned char v62;
                            v62 = v1[v21];
                            v66 = v62;
                        } else {
                            unsigned char v64;
                            v64 = v7[v21];
                            v66 = v64;
                        }
                        int v67;
                        v67 = v21 + 1l;
                        v77 = v66; v78 = v67; v79 = v22;
                    } else {
                        unsigned char v72;
                        if (v9){
                            unsigned char v68;
                            v68 = v1[v22];
                            v72 = v68;
                        } else {
                            unsigned char v70;
                            v70 = v7[v22];
                            v72 = v70;
                        }
                        int v73;
                        v73 = v22 + 1l;
                        v77 = v72; v78 = v21; v79 = v73;
                    }
                }
                if (v9){
                    v7[v23] = v77;
                } else {
                    v1[v23] = v77;
                }
                int v80;
                v80 = v23 + 1l;
                v21 = v78;
                v22 = v79;
                v23 = v80;
            }
            v12 = v18;
        }
        bool v81;
        v81 = v9 == false;
        int v82;
        v82 = v10 * 2l;
        v9 = v81;
        v10 = v82;
    }
    bool v83;
    v83 = v9 == false;
    static_array<unsigned char,7l> v84;
    if (v83){
        v84 = v7;
    } else {
        v84 = v1;
    }
    static_array<unsigned char,5l> v85;
    int v87; int v88; unsigned char v89;
    Tuple18 tmp41 = Tuple18{0l, 0l, 12u};
    v87 = tmp41.v0; v88 = tmp41.v1; v89 = tmp41.v2;
    while (while_method_10(v87)){
        unsigned char v91;
        v91 = v84[v87];
        bool v93;
        v93 = v88 < 5l;
        int v105; unsigned char v106;
        if (v93){
            unsigned char v94;
            v94 = v91 % 4u;
            bool v95;
            v95 = 0u == v94;
            if (v95){
                unsigned char v96;
                v96 = v91 / 4u;
                bool v97;
                v97 = v89 == v96;
                int v98;
                if (v97){
                    v98 = v88;
                } else {
                    v98 = 0l;
                }
                v85[v98] = v91;
                int v99;
                v99 = v98 + 1l;
                unsigned char v100;
                v100 = v96 - 1u;
                v105 = v99; v106 = v100;
            } else {
                v105 = v88; v106 = v89;
            }
        } else {
            break;
        }
        v88 = v105;
        v89 = v106;
        v87 += 1l ;
    }
    bool v107;
    v107 = v88 == 4l;
    bool v146;
    if (v107){
        unsigned char v108;
        v108 = v89 + 1u;
        bool v109;
        v109 = v108 == 0u;
        if (v109){
            unsigned char v110;
            v110 = v84[0l];
            unsigned char v112;
            v112 = v110 % 4u;
            bool v113;
            v113 = 0u == v112;
            bool v117;
            if (v113){
                unsigned char v114;
                v114 = v110 / 4u;
                bool v115;
                v115 = v114 == 12u;
                if (v115){
                    v85[4l] = v110;
                    v117 = true;
                } else {
                    v117 = false;
                }
            } else {
                v117 = false;
            }
            if (v117){
                v146 = true;
            } else {
                unsigned char v118;
                v118 = v84[1l];
                unsigned char v120;
                v120 = v118 % 4u;
                bool v121;
                v121 = 0u == v120;
                bool v125;
                if (v121){
                    unsigned char v122;
                    v122 = v118 / 4u;
                    bool v123;
                    v123 = v122 == 12u;
                    if (v123){
                        v85[4l] = v118;
                        v125 = true;
                    } else {
                        v125 = false;
                    }
                } else {
                    v125 = false;
                }
                if (v125){
                    v146 = true;
                } else {
                    unsigned char v126;
                    v126 = v84[2l];
                    unsigned char v128;
                    v128 = v126 % 4u;
                    bool v129;
                    v129 = 0u == v128;
                    bool v133;
                    if (v129){
                        unsigned char v130;
                        v130 = v126 / 4u;
                        bool v131;
                        v131 = v130 == 12u;
                        if (v131){
                            v85[4l] = v126;
                            v133 = true;
                        } else {
                            v133 = false;
                        }
                    } else {
                        v133 = false;
                    }
                    if (v133){
                        v146 = true;
                    } else {
                        unsigned char v134;
                        v134 = v84[3l];
                        unsigned char v136;
                        v136 = v134 % 4u;
                        bool v137;
                        v137 = 0u == v136;
                        if (v137){
                            unsigned char v138;
                            v138 = v134 / 4u;
                            bool v139;
                            v139 = v138 == 12u;
                            if (v139){
                                v85[4l] = v134;
                                v146 = true;
                            } else {
                                v146 = false;
                            }
                        } else {
                            v146 = false;
                        }
                    }
                }
            }
        } else {
            v146 = false;
        }
    } else {
        v146 = false;
    }
    Union9 v152;
    if (v146){
        v152 = Union9{Union9_1{v85}};
    } else {
        bool v148;
        v148 = v88 == 5l;
        if (v148){
            v152 = Union9{Union9_1{v85}};
        } else {
            v152 = Union9{Union9_0{}};
        }
    }
    static_array<unsigned char,5l> v153;
    int v155; int v156; unsigned char v157;
    Tuple18 tmp42 = Tuple18{0l, 0l, 12u};
    v155 = tmp42.v0; v156 = tmp42.v1; v157 = tmp42.v2;
    while (while_method_10(v155)){
        unsigned char v159;
        v159 = v84[v155];
        bool v161;
        v161 = v156 < 5l;
        int v173; unsigned char v174;
        if (v161){
            unsigned char v162;
            v162 = v159 % 4u;
            bool v163;
            v163 = 1u == v162;
            if (v163){
                unsigned char v164;
                v164 = v159 / 4u;
                bool v165;
                v165 = v157 == v164;
                int v166;
                if (v165){
                    v166 = v156;
                } else {
                    v166 = 0l;
                }
                v153[v166] = v159;
                int v167;
                v167 = v166 + 1l;
                unsigned char v168;
                v168 = v164 - 1u;
                v173 = v167; v174 = v168;
            } else {
                v173 = v156; v174 = v157;
            }
        } else {
            break;
        }
        v156 = v173;
        v157 = v174;
        v155 += 1l ;
    }
    bool v175;
    v175 = v156 == 4l;
    bool v214;
    if (v175){
        unsigned char v176;
        v176 = v157 + 1u;
        bool v177;
        v177 = v176 == 0u;
        if (v177){
            unsigned char v178;
            v178 = v84[0l];
            unsigned char v180;
            v180 = v178 % 4u;
            bool v181;
            v181 = 1u == v180;
            bool v185;
            if (v181){
                unsigned char v182;
                v182 = v178 / 4u;
                bool v183;
                v183 = v182 == 12u;
                if (v183){
                    v153[4l] = v178;
                    v185 = true;
                } else {
                    v185 = false;
                }
            } else {
                v185 = false;
            }
            if (v185){
                v214 = true;
            } else {
                unsigned char v186;
                v186 = v84[1l];
                unsigned char v188;
                v188 = v186 % 4u;
                bool v189;
                v189 = 1u == v188;
                bool v193;
                if (v189){
                    unsigned char v190;
                    v190 = v186 / 4u;
                    bool v191;
                    v191 = v190 == 12u;
                    if (v191){
                        v153[4l] = v186;
                        v193 = true;
                    } else {
                        v193 = false;
                    }
                } else {
                    v193 = false;
                }
                if (v193){
                    v214 = true;
                } else {
                    unsigned char v194;
                    v194 = v84[2l];
                    unsigned char v196;
                    v196 = v194 % 4u;
                    bool v197;
                    v197 = 1u == v196;
                    bool v201;
                    if (v197){
                        unsigned char v198;
                        v198 = v194 / 4u;
                        bool v199;
                        v199 = v198 == 12u;
                        if (v199){
                            v153[4l] = v194;
                            v201 = true;
                        } else {
                            v201 = false;
                        }
                    } else {
                        v201 = false;
                    }
                    if (v201){
                        v214 = true;
                    } else {
                        unsigned char v202;
                        v202 = v84[3l];
                        unsigned char v204;
                        v204 = v202 % 4u;
                        bool v205;
                        v205 = 1u == v204;
                        if (v205){
                            unsigned char v206;
                            v206 = v202 / 4u;
                            bool v207;
                            v207 = v206 == 12u;
                            if (v207){
                                v153[4l] = v202;
                                v214 = true;
                            } else {
                                v214 = false;
                            }
                        } else {
                            v214 = false;
                        }
                    }
                }
            }
        } else {
            v214 = false;
        }
    } else {
        v214 = false;
    }
    Union9 v220;
    if (v214){
        v220 = Union9{Union9_1{v153}};
    } else {
        bool v216;
        v216 = v156 == 5l;
        if (v216){
            v220 = Union9{Union9_1{v153}};
        } else {
            v220 = Union9{Union9_0{}};
        }
    }
    Union9 v248;
    switch (v152.tag) {
        case 0: { // None
            v248 = v220;
            break;
        }
        case 1: { // Some
            static_array<unsigned char,5l> v221 = v152.case1.v0;
            switch (v220.tag) {
                case 0: { // None
                    v248 = v152;
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5l> v222 = v220.case1.v0;
                    Union8 v223;
                    v223 = Union8{Union8_0{}};
                    int v224; Union8 v225;
                    Tuple19 tmp43 = Tuple19{0l, v223};
                    v224 = tmp43.v0; v225 = tmp43.v1;
                    while (while_method_2(v224)){
                        unsigned char v227;
                        v227 = v221[v224];
                        unsigned char v229;
                        v229 = v222[v224];
                        Union8 v241;
                        switch (v225.tag) {
                            case 0: { // Eq
                                unsigned char v231;
                                v231 = v227 / 4u;
                                unsigned char v232;
                                v232 = v229 / 4u;
                                bool v233;
                                v233 = v231 < v232;
                                if (v233){
                                    v241 = Union8{Union8_2{}};
                                } else {
                                    bool v235;
                                    v235 = v231 > v232;
                                    if (v235){
                                        v241 = Union8{Union8_1{}};
                                    } else {
                                        v241 = Union8{Union8_0{}};
                                    }
                                }
                                break;
                            }
                            default: {
                                break;
                            }
                        }
                        v225 = v241;
                        v224 += 1l ;
                    }
                    bool v242;
                    switch (v225.tag) {
                        case 1: { // Gt
                            v242 = true;
                            break;
                        }
                        default: {
                            v242 = false;
                        }
                    }
                    static_array<unsigned char,5l> v243;
                    if (v242){
                        v243 = v221;
                    } else {
                        v243 = v222;
                    }
                    v248 = Union9{Union9_1{v243}};
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
    static_array<unsigned char,5l> v249;
    int v251; int v252; unsigned char v253;
    Tuple18 tmp44 = Tuple18{0l, 0l, 12u};
    v251 = tmp44.v0; v252 = tmp44.v1; v253 = tmp44.v2;
    while (while_method_10(v251)){
        unsigned char v255;
        v255 = v84[v251];
        bool v257;
        v257 = v252 < 5l;
        int v269; unsigned char v270;
        if (v257){
            unsigned char v258;
            v258 = v255 % 4u;
            bool v259;
            v259 = 2u == v258;
            if (v259){
                unsigned char v260;
                v260 = v255 / 4u;
                bool v261;
                v261 = v253 == v260;
                int v262;
                if (v261){
                    v262 = v252;
                } else {
                    v262 = 0l;
                }
                v249[v262] = v255;
                int v263;
                v263 = v262 + 1l;
                unsigned char v264;
                v264 = v260 - 1u;
                v269 = v263; v270 = v264;
            } else {
                v269 = v252; v270 = v253;
            }
        } else {
            break;
        }
        v252 = v269;
        v253 = v270;
        v251 += 1l ;
    }
    bool v271;
    v271 = v252 == 4l;
    bool v310;
    if (v271){
        unsigned char v272;
        v272 = v253 + 1u;
        bool v273;
        v273 = v272 == 0u;
        if (v273){
            unsigned char v274;
            v274 = v84[0l];
            unsigned char v276;
            v276 = v274 % 4u;
            bool v277;
            v277 = 2u == v276;
            bool v281;
            if (v277){
                unsigned char v278;
                v278 = v274 / 4u;
                bool v279;
                v279 = v278 == 12u;
                if (v279){
                    v249[4l] = v274;
                    v281 = true;
                } else {
                    v281 = false;
                }
            } else {
                v281 = false;
            }
            if (v281){
                v310 = true;
            } else {
                unsigned char v282;
                v282 = v84[1l];
                unsigned char v284;
                v284 = v282 % 4u;
                bool v285;
                v285 = 2u == v284;
                bool v289;
                if (v285){
                    unsigned char v286;
                    v286 = v282 / 4u;
                    bool v287;
                    v287 = v286 == 12u;
                    if (v287){
                        v249[4l] = v282;
                        v289 = true;
                    } else {
                        v289 = false;
                    }
                } else {
                    v289 = false;
                }
                if (v289){
                    v310 = true;
                } else {
                    unsigned char v290;
                    v290 = v84[2l];
                    unsigned char v292;
                    v292 = v290 % 4u;
                    bool v293;
                    v293 = 2u == v292;
                    bool v297;
                    if (v293){
                        unsigned char v294;
                        v294 = v290 / 4u;
                        bool v295;
                        v295 = v294 == 12u;
                        if (v295){
                            v249[4l] = v290;
                            v297 = true;
                        } else {
                            v297 = false;
                        }
                    } else {
                        v297 = false;
                    }
                    if (v297){
                        v310 = true;
                    } else {
                        unsigned char v298;
                        v298 = v84[3l];
                        unsigned char v300;
                        v300 = v298 % 4u;
                        bool v301;
                        v301 = 2u == v300;
                        if (v301){
                            unsigned char v302;
                            v302 = v298 / 4u;
                            bool v303;
                            v303 = v302 == 12u;
                            if (v303){
                                v249[4l] = v298;
                                v310 = true;
                            } else {
                                v310 = false;
                            }
                        } else {
                            v310 = false;
                        }
                    }
                }
            }
        } else {
            v310 = false;
        }
    } else {
        v310 = false;
    }
    Union9 v316;
    if (v310){
        v316 = Union9{Union9_1{v249}};
    } else {
        bool v312;
        v312 = v252 == 5l;
        if (v312){
            v316 = Union9{Union9_1{v249}};
        } else {
            v316 = Union9{Union9_0{}};
        }
    }
    Union9 v344;
    switch (v248.tag) {
        case 0: { // None
            v344 = v316;
            break;
        }
        case 1: { // Some
            static_array<unsigned char,5l> v317 = v248.case1.v0;
            switch (v316.tag) {
                case 0: { // None
                    v344 = v248;
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5l> v318 = v316.case1.v0;
                    Union8 v319;
                    v319 = Union8{Union8_0{}};
                    int v320; Union8 v321;
                    Tuple19 tmp45 = Tuple19{0l, v319};
                    v320 = tmp45.v0; v321 = tmp45.v1;
                    while (while_method_2(v320)){
                        unsigned char v323;
                        v323 = v317[v320];
                        unsigned char v325;
                        v325 = v318[v320];
                        Union8 v337;
                        switch (v321.tag) {
                            case 0: { // Eq
                                unsigned char v327;
                                v327 = v323 / 4u;
                                unsigned char v328;
                                v328 = v325 / 4u;
                                bool v329;
                                v329 = v327 < v328;
                                if (v329){
                                    v337 = Union8{Union8_2{}};
                                } else {
                                    bool v331;
                                    v331 = v327 > v328;
                                    if (v331){
                                        v337 = Union8{Union8_1{}};
                                    } else {
                                        v337 = Union8{Union8_0{}};
                                    }
                                }
                                break;
                            }
                            default: {
                                break;
                            }
                        }
                        v321 = v337;
                        v320 += 1l ;
                    }
                    bool v338;
                    switch (v321.tag) {
                        case 1: { // Gt
                            v338 = true;
                            break;
                        }
                        default: {
                            v338 = false;
                        }
                    }
                    static_array<unsigned char,5l> v339;
                    if (v338){
                        v339 = v317;
                    } else {
                        v339 = v318;
                    }
                    v344 = Union9{Union9_1{v339}};
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
    static_array<unsigned char,5l> v345;
    int v347; int v348; unsigned char v349;
    Tuple18 tmp46 = Tuple18{0l, 0l, 12u};
    v347 = tmp46.v0; v348 = tmp46.v1; v349 = tmp46.v2;
    while (while_method_10(v347)){
        unsigned char v351;
        v351 = v84[v347];
        bool v353;
        v353 = v348 < 5l;
        int v365; unsigned char v366;
        if (v353){
            unsigned char v354;
            v354 = v351 % 4u;
            bool v355;
            v355 = 3u == v354;
            if (v355){
                unsigned char v356;
                v356 = v351 / 4u;
                bool v357;
                v357 = v349 == v356;
                int v358;
                if (v357){
                    v358 = v348;
                } else {
                    v358 = 0l;
                }
                v345[v358] = v351;
                int v359;
                v359 = v358 + 1l;
                unsigned char v360;
                v360 = v356 - 1u;
                v365 = v359; v366 = v360;
            } else {
                v365 = v348; v366 = v349;
            }
        } else {
            break;
        }
        v348 = v365;
        v349 = v366;
        v347 += 1l ;
    }
    bool v367;
    v367 = v348 == 4l;
    bool v406;
    if (v367){
        unsigned char v368;
        v368 = v349 + 1u;
        bool v369;
        v369 = v368 == 0u;
        if (v369){
            unsigned char v370;
            v370 = v84[0l];
            unsigned char v372;
            v372 = v370 % 4u;
            bool v373;
            v373 = 3u == v372;
            bool v377;
            if (v373){
                unsigned char v374;
                v374 = v370 / 4u;
                bool v375;
                v375 = v374 == 12u;
                if (v375){
                    v345[4l] = v370;
                    v377 = true;
                } else {
                    v377 = false;
                }
            } else {
                v377 = false;
            }
            if (v377){
                v406 = true;
            } else {
                unsigned char v378;
                v378 = v84[1l];
                unsigned char v380;
                v380 = v378 % 4u;
                bool v381;
                v381 = 3u == v380;
                bool v385;
                if (v381){
                    unsigned char v382;
                    v382 = v378 / 4u;
                    bool v383;
                    v383 = v382 == 12u;
                    if (v383){
                        v345[4l] = v378;
                        v385 = true;
                    } else {
                        v385 = false;
                    }
                } else {
                    v385 = false;
                }
                if (v385){
                    v406 = true;
                } else {
                    unsigned char v386;
                    v386 = v84[2l];
                    unsigned char v388;
                    v388 = v386 % 4u;
                    bool v389;
                    v389 = 3u == v388;
                    bool v393;
                    if (v389){
                        unsigned char v390;
                        v390 = v386 / 4u;
                        bool v391;
                        v391 = v390 == 12u;
                        if (v391){
                            v345[4l] = v386;
                            v393 = true;
                        } else {
                            v393 = false;
                        }
                    } else {
                        v393 = false;
                    }
                    if (v393){
                        v406 = true;
                    } else {
                        unsigned char v394;
                        v394 = v84[3l];
                        unsigned char v396;
                        v396 = v394 % 4u;
                        bool v397;
                        v397 = 3u == v396;
                        if (v397){
                            unsigned char v398;
                            v398 = v394 / 4u;
                            bool v399;
                            v399 = v398 == 12u;
                            if (v399){
                                v345[4l] = v394;
                                v406 = true;
                            } else {
                                v406 = false;
                            }
                        } else {
                            v406 = false;
                        }
                    }
                }
            }
        } else {
            v406 = false;
        }
    } else {
        v406 = false;
    }
    Union9 v412;
    if (v406){
        v412 = Union9{Union9_1{v345}};
    } else {
        bool v408;
        v408 = v348 == 5l;
        if (v408){
            v412 = Union9{Union9_1{v345}};
        } else {
            v412 = Union9{Union9_0{}};
        }
    }
    Union9 v440;
    switch (v344.tag) {
        case 0: { // None
            v440 = v412;
            break;
        }
        case 1: { // Some
            static_array<unsigned char,5l> v413 = v344.case1.v0;
            switch (v412.tag) {
                case 0: { // None
                    v440 = v344;
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5l> v414 = v412.case1.v0;
                    Union8 v415;
                    v415 = Union8{Union8_0{}};
                    int v416; Union8 v417;
                    Tuple19 tmp47 = Tuple19{0l, v415};
                    v416 = tmp47.v0; v417 = tmp47.v1;
                    while (while_method_2(v416)){
                        unsigned char v419;
                        v419 = v413[v416];
                        unsigned char v421;
                        v421 = v414[v416];
                        Union8 v433;
                        switch (v417.tag) {
                            case 0: { // Eq
                                unsigned char v423;
                                v423 = v419 / 4u;
                                unsigned char v424;
                                v424 = v421 / 4u;
                                bool v425;
                                v425 = v423 < v424;
                                if (v425){
                                    v433 = Union8{Union8_2{}};
                                } else {
                                    bool v427;
                                    v427 = v423 > v424;
                                    if (v427){
                                        v433 = Union8{Union8_1{}};
                                    } else {
                                        v433 = Union8{Union8_0{}};
                                    }
                                }
                                break;
                            }
                            default: {
                                break;
                            }
                        }
                        v417 = v433;
                        v416 += 1l ;
                    }
                    bool v434;
                    switch (v417.tag) {
                        case 1: { // Gt
                            v434 = true;
                            break;
                        }
                        default: {
                            v434 = false;
                        }
                    }
                    static_array<unsigned char,5l> v435;
                    if (v434){
                        v435 = v413;
                    } else {
                        v435 = v414;
                    }
                    v440 = Union9{Union9_1{v435}};
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
    static_array<unsigned char,5l> v1037; char v1038;
    switch (v440.tag) {
        case 0: { // None
            static_array<unsigned char,4l> v442;
            static_array<unsigned char,3l> v444;
            int v446; int v447; int v448; unsigned char v449;
            Tuple20 tmp48 = Tuple20{0l, 0l, 0l, 12u};
            v446 = tmp48.v0; v447 = tmp48.v1; v448 = tmp48.v2; v449 = tmp48.v3;
            while (while_method_10(v446)){
                unsigned char v451;
                v451 = v84[v446];
                bool v453;
                v453 = v448 < 4l;
                int v461; int v462; unsigned char v463;
                if (v453){
                    unsigned char v454;
                    v454 = v451 / 4u;
                    bool v455;
                    v455 = v449 == v454;
                    int v456;
                    if (v455){
                        v456 = v448;
                    } else {
                        v456 = 0l;
                    }
                    v442[v456] = v451;
                    int v457;
                    v457 = v456 + 1l;
                    v461 = v446; v462 = v457; v463 = v454;
                } else {
                    break;
                }
                v447 = v461;
                v448 = v462;
                v449 = v463;
                v446 += 1l ;
            }
            bool v464;
            v464 = v448 == 4l;
            Union10 v475;
            if (v464){
                int v465;
                v465 = 0l;
                while (while_method_3(v465)){
                    int v467;
                    v467 = v447 + -3l;
                    bool v468;
                    v468 = v465 < v467;
                    int v469;
                    if (v468){
                        v469 = 0l;
                    } else {
                        v469 = 4l;
                    }
                    int v470;
                    v470 = v469 + v465;
                    unsigned char v471;
                    v471 = v84[v470];
                    v444[v465] = v471;
                    v465 += 1l ;
                }
                v475 = Union10{Union10_1{v442, v444}};
            } else {
                v475 = Union10{Union10_0{}};
            }
            Union9 v498;
            switch (v475.tag) {
                case 0: { // None
                    v498 = Union9{Union9_0{}};
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,4l> v476 = v475.case1.v0; static_array<unsigned char,3l> v477 = v475.case1.v1;
                    static_array<unsigned char,1l> v478;
                    int v480;
                    v480 = 0l;
                    while (while_method_6(v480)){
                        unsigned char v482;
                        v482 = v477[v480];
                        v478[v480] = v482;
                        v480 += 1l ;
                    }
                    static_array<unsigned char,5l> v484;
                    int v486;
                    v486 = 0l;
                    while (while_method_4(v486)){
                        unsigned char v488;
                        v488 = v476[v486];
                        v484[v486] = v488;
                        v486 += 1l ;
                    }
                    int v490;
                    v490 = 0l;
                    while (while_method_6(v490)){
                        unsigned char v492;
                        v492 = v478[v490];
                        int v494;
                        v494 = 4l + v490;
                        v484[v494] = v492;
                        v490 += 1l ;
                    }
                    v498 = Union9{Union9_1{v484}};
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                }
            }
            switch (v498.tag) {
                case 0: { // None
                    static_array<unsigned char,3l> v500;
                    static_array<unsigned char,4l> v502;
                    int v504; int v505; int v506; unsigned char v507;
                    Tuple20 tmp49 = Tuple20{0l, 0l, 0l, 12u};
                    v504 = tmp49.v0; v505 = tmp49.v1; v506 = tmp49.v2; v507 = tmp49.v3;
                    while (while_method_10(v504)){
                        unsigned char v509;
                        v509 = v84[v504];
                        bool v511;
                        v511 = v506 < 3l;
                        int v519; int v520; unsigned char v521;
                        if (v511){
                            unsigned char v512;
                            v512 = v509 / 4u;
                            bool v513;
                            v513 = v507 == v512;
                            int v514;
                            if (v513){
                                v514 = v506;
                            } else {
                                v514 = 0l;
                            }
                            v500[v514] = v509;
                            int v515;
                            v515 = v514 + 1l;
                            v519 = v504; v520 = v515; v521 = v512;
                        } else {
                            break;
                        }
                        v505 = v519;
                        v506 = v520;
                        v507 = v521;
                        v504 += 1l ;
                    }
                    bool v522;
                    v522 = v506 == 3l;
                    Union11 v533;
                    if (v522){
                        int v523;
                        v523 = 0l;
                        while (while_method_4(v523)){
                            int v525;
                            v525 = v505 + -2l;
                            bool v526;
                            v526 = v523 < v525;
                            int v527;
                            if (v526){
                                v527 = 0l;
                            } else {
                                v527 = 3l;
                            }
                            int v528;
                            v528 = v527 + v523;
                            unsigned char v529;
                            v529 = v84[v528];
                            v502[v523] = v529;
                            v523 += 1l ;
                        }
                        v533 = Union11{Union11_1{v500, v502}};
                    } else {
                        v533 = Union11{Union11_0{}};
                    }
                    Union9 v589;
                    switch (v533.tag) {
                        case 0: { // None
                            v589 = Union9{Union9_0{}};
                            break;
                        }
                        case 1: { // Some
                            static_array<unsigned char,3l> v534 = v533.case1.v0; static_array<unsigned char,4l> v535 = v533.case1.v1;
                            static_array<unsigned char,2l> v536;
                            static_array<unsigned char,2l> v538;
                            int v540; int v541; int v542; unsigned char v543;
                            Tuple20 tmp50 = Tuple20{0l, 0l, 0l, 12u};
                            v540 = tmp50.v0; v541 = tmp50.v1; v542 = tmp50.v2; v543 = tmp50.v3;
                            while (while_method_4(v540)){
                                unsigned char v545;
                                v545 = v535[v540];
                                bool v547;
                                v547 = v542 < 2l;
                                int v555; int v556; unsigned char v557;
                                if (v547){
                                    unsigned char v548;
                                    v548 = v545 / 4u;
                                    bool v549;
                                    v549 = v543 == v548;
                                    int v550;
                                    if (v549){
                                        v550 = v542;
                                    } else {
                                        v550 = 0l;
                                    }
                                    v536[v550] = v545;
                                    int v551;
                                    v551 = v550 + 1l;
                                    v555 = v540; v556 = v551; v557 = v548;
                                } else {
                                    break;
                                }
                                v541 = v555;
                                v542 = v556;
                                v543 = v557;
                                v540 += 1l ;
                            }
                            bool v558;
                            v558 = v542 == 2l;
                            Union12 v569;
                            if (v558){
                                int v559;
                                v559 = 0l;
                                while (while_method_0(v559)){
                                    int v561;
                                    v561 = v541 + -1l;
                                    bool v562;
                                    v562 = v559 < v561;
                                    int v563;
                                    if (v562){
                                        v563 = 0l;
                                    } else {
                                        v563 = 2l;
                                    }
                                    int v564;
                                    v564 = v563 + v559;
                                    unsigned char v565;
                                    v565 = v535[v564];
                                    v538[v559] = v565;
                                    v559 += 1l ;
                                }
                                v569 = Union12{Union12_1{v536, v538}};
                            } else {
                                v569 = Union12{Union12_0{}};
                            }
                            switch (v569.tag) {
                                case 0: { // None
                                    v589 = Union9{Union9_0{}};
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,2l> v570 = v569.case1.v0; static_array<unsigned char,2l> v571 = v569.case1.v1;
                                    static_array<unsigned char,5l> v572;
                                    int v574;
                                    v574 = 0l;
                                    while (while_method_3(v574)){
                                        unsigned char v576;
                                        v576 = v534[v574];
                                        v572[v574] = v576;
                                        v574 += 1l ;
                                    }
                                    int v578;
                                    v578 = 0l;
                                    while (while_method_0(v578)){
                                        unsigned char v580;
                                        v580 = v570[v578];
                                        int v582;
                                        v582 = 3l + v578;
                                        v572[v582] = v580;
                                        v578 += 1l ;
                                    }
                                    v589 = Union9{Union9_1{v572}};
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
                    switch (v589.tag) {
                        case 0: { // None
                            static_array<unsigned char,5l> v591;
                            int v593; int v594;
                            Tuple2 tmp51 = Tuple2{0l, 0l};
                            v593 = tmp51.v0; v594 = tmp51.v1;
                            while (while_method_10(v593)){
                                unsigned char v596;
                                v596 = v84[v593];
                                unsigned char v598;
                                v598 = v596 % 4u;
                                bool v599;
                                v599 = v598 == 0u;
                                bool v601;
                                if (v599){
                                    bool v600;
                                    v600 = v594 < 5l;
                                    v601 = v600;
                                } else {
                                    v601 = false;
                                }
                                int v603;
                                if (v601){
                                    v591[v594] = v596;
                                    int v602;
                                    v602 = v594 + 1l;
                                    v603 = v602;
                                } else {
                                    v603 = v594;
                                }
                                v594 = v603;
                                v593 += 1l ;
                            }
                            bool v604;
                            v604 = v594 == 5l;
                            Union9 v607;
                            if (v604){
                                v607 = Union9{Union9_1{v591}};
                            } else {
                                v607 = Union9{Union9_0{}};
                            }
                            static_array<unsigned char,5l> v608;
                            int v610; int v611;
                            Tuple2 tmp52 = Tuple2{0l, 0l};
                            v610 = tmp52.v0; v611 = tmp52.v1;
                            while (while_method_10(v610)){
                                unsigned char v613;
                                v613 = v84[v610];
                                unsigned char v615;
                                v615 = v613 % 4u;
                                bool v616;
                                v616 = v615 == 1u;
                                bool v618;
                                if (v616){
                                    bool v617;
                                    v617 = v611 < 5l;
                                    v618 = v617;
                                } else {
                                    v618 = false;
                                }
                                int v620;
                                if (v618){
                                    v608[v611] = v613;
                                    int v619;
                                    v619 = v611 + 1l;
                                    v620 = v619;
                                } else {
                                    v620 = v611;
                                }
                                v611 = v620;
                                v610 += 1l ;
                            }
                            bool v621;
                            v621 = v611 == 5l;
                            Union9 v624;
                            if (v621){
                                v624 = Union9{Union9_1{v608}};
                            } else {
                                v624 = Union9{Union9_0{}};
                            }
                            Union9 v652;
                            switch (v607.tag) {
                                case 0: { // None
                                    v652 = v624;
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,5l> v625 = v607.case1.v0;
                                    switch (v624.tag) {
                                        case 0: { // None
                                            v652 = v607;
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5l> v626 = v624.case1.v0;
                                            Union8 v627;
                                            v627 = Union8{Union8_0{}};
                                            int v628; Union8 v629;
                                            Tuple19 tmp53 = Tuple19{0l, v627};
                                            v628 = tmp53.v0; v629 = tmp53.v1;
                                            while (while_method_2(v628)){
                                                unsigned char v631;
                                                v631 = v625[v628];
                                                unsigned char v633;
                                                v633 = v626[v628];
                                                Union8 v645;
                                                switch (v629.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v635;
                                                        v635 = v631 / 4u;
                                                        unsigned char v636;
                                                        v636 = v633 / 4u;
                                                        bool v637;
                                                        v637 = v635 < v636;
                                                        if (v637){
                                                            v645 = Union8{Union8_2{}};
                                                        } else {
                                                            bool v639;
                                                            v639 = v635 > v636;
                                                            if (v639){
                                                                v645 = Union8{Union8_1{}};
                                                            } else {
                                                                v645 = Union8{Union8_0{}};
                                                            }
                                                        }
                                                        break;
                                                    }
                                                    default: {
                                                        break;
                                                    }
                                                }
                                                v629 = v645;
                                                v628 += 1l ;
                                            }
                                            bool v646;
                                            switch (v629.tag) {
                                                case 1: { // Gt
                                                    v646 = true;
                                                    break;
                                                }
                                                default: {
                                                    v646 = false;
                                                }
                                            }
                                            static_array<unsigned char,5l> v647;
                                            if (v646){
                                                v647 = v625;
                                            } else {
                                                v647 = v626;
                                            }
                                            v652 = Union9{Union9_1{v647}};
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
                            static_array<unsigned char,5l> v653;
                            int v655; int v656;
                            Tuple2 tmp54 = Tuple2{0l, 0l};
                            v655 = tmp54.v0; v656 = tmp54.v1;
                            while (while_method_10(v655)){
                                unsigned char v658;
                                v658 = v84[v655];
                                unsigned char v660;
                                v660 = v658 % 4u;
                                bool v661;
                                v661 = v660 == 2u;
                                bool v663;
                                if (v661){
                                    bool v662;
                                    v662 = v656 < 5l;
                                    v663 = v662;
                                } else {
                                    v663 = false;
                                }
                                int v665;
                                if (v663){
                                    v653[v656] = v658;
                                    int v664;
                                    v664 = v656 + 1l;
                                    v665 = v664;
                                } else {
                                    v665 = v656;
                                }
                                v656 = v665;
                                v655 += 1l ;
                            }
                            bool v666;
                            v666 = v656 == 5l;
                            Union9 v669;
                            if (v666){
                                v669 = Union9{Union9_1{v653}};
                            } else {
                                v669 = Union9{Union9_0{}};
                            }
                            Union9 v697;
                            switch (v652.tag) {
                                case 0: { // None
                                    v697 = v669;
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,5l> v670 = v652.case1.v0;
                                    switch (v669.tag) {
                                        case 0: { // None
                                            v697 = v652;
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5l> v671 = v669.case1.v0;
                                            Union8 v672;
                                            v672 = Union8{Union8_0{}};
                                            int v673; Union8 v674;
                                            Tuple19 tmp55 = Tuple19{0l, v672};
                                            v673 = tmp55.v0; v674 = tmp55.v1;
                                            while (while_method_2(v673)){
                                                unsigned char v676;
                                                v676 = v670[v673];
                                                unsigned char v678;
                                                v678 = v671[v673];
                                                Union8 v690;
                                                switch (v674.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v680;
                                                        v680 = v676 / 4u;
                                                        unsigned char v681;
                                                        v681 = v678 / 4u;
                                                        bool v682;
                                                        v682 = v680 < v681;
                                                        if (v682){
                                                            v690 = Union8{Union8_2{}};
                                                        } else {
                                                            bool v684;
                                                            v684 = v680 > v681;
                                                            if (v684){
                                                                v690 = Union8{Union8_1{}};
                                                            } else {
                                                                v690 = Union8{Union8_0{}};
                                                            }
                                                        }
                                                        break;
                                                    }
                                                    default: {
                                                        break;
                                                    }
                                                }
                                                v674 = v690;
                                                v673 += 1l ;
                                            }
                                            bool v691;
                                            switch (v674.tag) {
                                                case 1: { // Gt
                                                    v691 = true;
                                                    break;
                                                }
                                                default: {
                                                    v691 = false;
                                                }
                                            }
                                            static_array<unsigned char,5l> v692;
                                            if (v691){
                                                v692 = v670;
                                            } else {
                                                v692 = v671;
                                            }
                                            v697 = Union9{Union9_1{v692}};
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
                            static_array<unsigned char,5l> v698;
                            int v700; int v701;
                            Tuple2 tmp56 = Tuple2{0l, 0l};
                            v700 = tmp56.v0; v701 = tmp56.v1;
                            while (while_method_10(v700)){
                                unsigned char v703;
                                v703 = v84[v700];
                                unsigned char v705;
                                v705 = v703 % 4u;
                                bool v706;
                                v706 = v705 == 3u;
                                bool v708;
                                if (v706){
                                    bool v707;
                                    v707 = v701 < 5l;
                                    v708 = v707;
                                } else {
                                    v708 = false;
                                }
                                int v710;
                                if (v708){
                                    v698[v701] = v703;
                                    int v709;
                                    v709 = v701 + 1l;
                                    v710 = v709;
                                } else {
                                    v710 = v701;
                                }
                                v701 = v710;
                                v700 += 1l ;
                            }
                            bool v711;
                            v711 = v701 == 5l;
                            Union9 v714;
                            if (v711){
                                v714 = Union9{Union9_1{v698}};
                            } else {
                                v714 = Union9{Union9_0{}};
                            }
                            Union9 v742;
                            switch (v697.tag) {
                                case 0: { // None
                                    v742 = v714;
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,5l> v715 = v697.case1.v0;
                                    switch (v714.tag) {
                                        case 0: { // None
                                            v742 = v697;
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5l> v716 = v714.case1.v0;
                                            Union8 v717;
                                            v717 = Union8{Union8_0{}};
                                            int v718; Union8 v719;
                                            Tuple19 tmp57 = Tuple19{0l, v717};
                                            v718 = tmp57.v0; v719 = tmp57.v1;
                                            while (while_method_2(v718)){
                                                unsigned char v721;
                                                v721 = v715[v718];
                                                unsigned char v723;
                                                v723 = v716[v718];
                                                Union8 v735;
                                                switch (v719.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v725;
                                                        v725 = v721 / 4u;
                                                        unsigned char v726;
                                                        v726 = v723 / 4u;
                                                        bool v727;
                                                        v727 = v725 < v726;
                                                        if (v727){
                                                            v735 = Union8{Union8_2{}};
                                                        } else {
                                                            bool v729;
                                                            v729 = v725 > v726;
                                                            if (v729){
                                                                v735 = Union8{Union8_1{}};
                                                            } else {
                                                                v735 = Union8{Union8_0{}};
                                                            }
                                                        }
                                                        break;
                                                    }
                                                    default: {
                                                        break;
                                                    }
                                                }
                                                v719 = v735;
                                                v718 += 1l ;
                                            }
                                            bool v736;
                                            switch (v719.tag) {
                                                case 1: { // Gt
                                                    v736 = true;
                                                    break;
                                                }
                                                default: {
                                                    v736 = false;
                                                }
                                            }
                                            static_array<unsigned char,5l> v737;
                                            if (v736){
                                                v737 = v715;
                                            } else {
                                                v737 = v716;
                                            }
                                            v742 = Union9{Union9_1{v737}};
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
                            switch (v742.tag) {
                                case 0: { // None
                                    static_array<unsigned char,5l> v744;
                                    int v746; int v747; unsigned char v748;
                                    Tuple18 tmp58 = Tuple18{0l, 0l, 12u};
                                    v746 = tmp58.v0; v747 = tmp58.v1; v748 = tmp58.v2;
                                    while (while_method_10(v746)){
                                        unsigned char v750;
                                        v750 = v84[v746];
                                        bool v752;
                                        v752 = v747 < 5l;
                                        int v764; unsigned char v765;
                                        if (v752){
                                            unsigned char v753;
                                            v753 = v750 / 4u;
                                            unsigned char v754;
                                            v754 = v753 - 1u;
                                            bool v755;
                                            v755 = v748 == v754;
                                            bool v756;
                                            v756 = v755 != true;
                                            if (v756){
                                                bool v757;
                                                v757 = v748 == v753;
                                                int v758;
                                                if (v757){
                                                    v758 = v747;
                                                } else {
                                                    v758 = 0l;
                                                }
                                                v744[v758] = v750;
                                                int v759;
                                                v759 = v758 + 1l;
                                                v764 = v759; v765 = v754;
                                            } else {
                                                v764 = v747; v765 = v748;
                                            }
                                        } else {
                                            break;
                                        }
                                        v747 = v764;
                                        v748 = v765;
                                        v746 += 1l ;
                                    }
                                    bool v766;
                                    v766 = v747 == 4l;
                                    bool v775;
                                    if (v766){
                                        unsigned char v767;
                                        v767 = v748 + 1u;
                                        bool v768;
                                        v768 = v767 == 0u;
                                        if (v768){
                                            unsigned char v769;
                                            v769 = v84[0l];
                                            unsigned char v771;
                                            v771 = v769 / 4u;
                                            bool v772;
                                            v772 = v771 == 12u;
                                            if (v772){
                                                v744[4l] = v769;
                                                v775 = true;
                                            } else {
                                                v775 = false;
                                            }
                                        } else {
                                            v775 = false;
                                        }
                                    } else {
                                        v775 = false;
                                    }
                                    Union9 v781;
                                    if (v775){
                                        v781 = Union9{Union9_1{v744}};
                                    } else {
                                        bool v777;
                                        v777 = v747 == 5l;
                                        if (v777){
                                            v781 = Union9{Union9_1{v744}};
                                        } else {
                                            v781 = Union9{Union9_0{}};
                                        }
                                    }
                                    switch (v781.tag) {
                                        case 0: { // None
                                            static_array<unsigned char,3l> v783;
                                            static_array<unsigned char,4l> v785;
                                            int v787; int v788; int v789; unsigned char v790;
                                            Tuple20 tmp59 = Tuple20{0l, 0l, 0l, 12u};
                                            v787 = tmp59.v0; v788 = tmp59.v1; v789 = tmp59.v2; v790 = tmp59.v3;
                                            while (while_method_10(v787)){
                                                unsigned char v792;
                                                v792 = v84[v787];
                                                bool v794;
                                                v794 = v789 < 3l;
                                                int v802; int v803; unsigned char v804;
                                                if (v794){
                                                    unsigned char v795;
                                                    v795 = v792 / 4u;
                                                    bool v796;
                                                    v796 = v790 == v795;
                                                    int v797;
                                                    if (v796){
                                                        v797 = v789;
                                                    } else {
                                                        v797 = 0l;
                                                    }
                                                    v783[v797] = v792;
                                                    int v798;
                                                    v798 = v797 + 1l;
                                                    v802 = v787; v803 = v798; v804 = v795;
                                                } else {
                                                    break;
                                                }
                                                v788 = v802;
                                                v789 = v803;
                                                v790 = v804;
                                                v787 += 1l ;
                                            }
                                            bool v805;
                                            v805 = v789 == 3l;
                                            Union11 v816;
                                            if (v805){
                                                int v806;
                                                v806 = 0l;
                                                while (while_method_4(v806)){
                                                    int v808;
                                                    v808 = v788 + -2l;
                                                    bool v809;
                                                    v809 = v806 < v808;
                                                    int v810;
                                                    if (v809){
                                                        v810 = 0l;
                                                    } else {
                                                        v810 = 3l;
                                                    }
                                                    int v811;
                                                    v811 = v810 + v806;
                                                    unsigned char v812;
                                                    v812 = v84[v811];
                                                    v785[v806] = v812;
                                                    v806 += 1l ;
                                                }
                                                v816 = Union11{Union11_1{v783, v785}};
                                            } else {
                                                v816 = Union11{Union11_0{}};
                                            }
                                            Union9 v839;
                                            switch (v816.tag) {
                                                case 0: { // None
                                                    v839 = Union9{Union9_0{}};
                                                    break;
                                                }
                                                case 1: { // Some
                                                    static_array<unsigned char,3l> v817 = v816.case1.v0; static_array<unsigned char,4l> v818 = v816.case1.v1;
                                                    static_array<unsigned char,2l> v819;
                                                    int v821;
                                                    v821 = 0l;
                                                    while (while_method_0(v821)){
                                                        unsigned char v823;
                                                        v823 = v818[v821];
                                                        v819[v821] = v823;
                                                        v821 += 1l ;
                                                    }
                                                    static_array<unsigned char,5l> v825;
                                                    int v827;
                                                    v827 = 0l;
                                                    while (while_method_3(v827)){
                                                        unsigned char v829;
                                                        v829 = v817[v827];
                                                        v825[v827] = v829;
                                                        v827 += 1l ;
                                                    }
                                                    int v831;
                                                    v831 = 0l;
                                                    while (while_method_0(v831)){
                                                        unsigned char v833;
                                                        v833 = v819[v831];
                                                        int v835;
                                                        v835 = 3l + v831;
                                                        v825[v835] = v833;
                                                        v831 += 1l ;
                                                    }
                                                    v839 = Union9{Union9_1{v825}};
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false);
                                                }
                                            }
                                            switch (v839.tag) {
                                                case 0: { // None
                                                    static_array<unsigned char,2l> v841;
                                                    static_array<unsigned char,5l> v843;
                                                    int v845; int v846; int v847; unsigned char v848;
                                                    Tuple20 tmp60 = Tuple20{0l, 0l, 0l, 12u};
                                                    v845 = tmp60.v0; v846 = tmp60.v1; v847 = tmp60.v2; v848 = tmp60.v3;
                                                    while (while_method_10(v845)){
                                                        unsigned char v850;
                                                        v850 = v84[v845];
                                                        bool v852;
                                                        v852 = v847 < 2l;
                                                        int v860; int v861; unsigned char v862;
                                                        if (v852){
                                                            unsigned char v853;
                                                            v853 = v850 / 4u;
                                                            bool v854;
                                                            v854 = v848 == v853;
                                                            int v855;
                                                            if (v854){
                                                                v855 = v847;
                                                            } else {
                                                                v855 = 0l;
                                                            }
                                                            v841[v855] = v850;
                                                            int v856;
                                                            v856 = v855 + 1l;
                                                            v860 = v845; v861 = v856; v862 = v853;
                                                        } else {
                                                            break;
                                                        }
                                                        v846 = v860;
                                                        v847 = v861;
                                                        v848 = v862;
                                                        v845 += 1l ;
                                                    }
                                                    bool v863;
                                                    v863 = v847 == 2l;
                                                    Union13 v874;
                                                    if (v863){
                                                        int v864;
                                                        v864 = 0l;
                                                        while (while_method_2(v864)){
                                                            int v866;
                                                            v866 = v846 + -1l;
                                                            bool v867;
                                                            v867 = v864 < v866;
                                                            int v868;
                                                            if (v867){
                                                                v868 = 0l;
                                                            } else {
                                                                v868 = 2l;
                                                            }
                                                            int v869;
                                                            v869 = v868 + v864;
                                                            unsigned char v870;
                                                            v870 = v84[v869];
                                                            v843[v864] = v870;
                                                            v864 += 1l ;
                                                        }
                                                        v874 = Union13{Union13_1{v841, v843}};
                                                    } else {
                                                        v874 = Union13{Union13_0{}};
                                                    }
                                                    Union9 v941;
                                                    switch (v874.tag) {
                                                        case 0: { // None
                                                            v941 = Union9{Union9_0{}};
                                                            break;
                                                        }
                                                        case 1: { // Some
                                                            static_array<unsigned char,2l> v875 = v874.case1.v0; static_array<unsigned char,5l> v876 = v874.case1.v1;
                                                            static_array<unsigned char,2l> v877;
                                                            static_array<unsigned char,3l> v879;
                                                            int v881; int v882; int v883; unsigned char v884;
                                                            Tuple20 tmp61 = Tuple20{0l, 0l, 0l, 12u};
                                                            v881 = tmp61.v0; v882 = tmp61.v1; v883 = tmp61.v2; v884 = tmp61.v3;
                                                            while (while_method_2(v881)){
                                                                unsigned char v886;
                                                                v886 = v876[v881];
                                                                bool v888;
                                                                v888 = v883 < 2l;
                                                                int v896; int v897; unsigned char v898;
                                                                if (v888){
                                                                    unsigned char v889;
                                                                    v889 = v886 / 4u;
                                                                    bool v890;
                                                                    v890 = v884 == v889;
                                                                    int v891;
                                                                    if (v890){
                                                                        v891 = v883;
                                                                    } else {
                                                                        v891 = 0l;
                                                                    }
                                                                    v877[v891] = v886;
                                                                    int v892;
                                                                    v892 = v891 + 1l;
                                                                    v896 = v881; v897 = v892; v898 = v889;
                                                                } else {
                                                                    break;
                                                                }
                                                                v882 = v896;
                                                                v883 = v897;
                                                                v884 = v898;
                                                                v881 += 1l ;
                                                            }
                                                            bool v899;
                                                            v899 = v883 == 2l;
                                                            Union14 v910;
                                                            if (v899){
                                                                int v900;
                                                                v900 = 0l;
                                                                while (while_method_3(v900)){
                                                                    int v902;
                                                                    v902 = v882 + -1l;
                                                                    bool v903;
                                                                    v903 = v900 < v902;
                                                                    int v904;
                                                                    if (v903){
                                                                        v904 = 0l;
                                                                    } else {
                                                                        v904 = 2l;
                                                                    }
                                                                    int v905;
                                                                    v905 = v904 + v900;
                                                                    unsigned char v906;
                                                                    v906 = v876[v905];
                                                                    v879[v900] = v906;
                                                                    v900 += 1l ;
                                                                }
                                                                v910 = Union14{Union14_1{v877, v879}};
                                                            } else {
                                                                v910 = Union14{Union14_0{}};
                                                            }
                                                            switch (v910.tag) {
                                                                case 0: { // None
                                                                    v941 = Union9{Union9_0{}};
                                                                    break;
                                                                }
                                                                case 1: { // Some
                                                                    static_array<unsigned char,2l> v911 = v910.case1.v0; static_array<unsigned char,3l> v912 = v910.case1.v1;
                                                                    static_array<unsigned char,1l> v913;
                                                                    int v915;
                                                                    v915 = 0l;
                                                                    while (while_method_6(v915)){
                                                                        unsigned char v917;
                                                                        v917 = v912[v915];
                                                                        v913[v915] = v917;
                                                                        v915 += 1l ;
                                                                    }
                                                                    static_array<unsigned char,5l> v919;
                                                                    int v921;
                                                                    v921 = 0l;
                                                                    while (while_method_0(v921)){
                                                                        unsigned char v923;
                                                                        v923 = v875[v921];
                                                                        v919[v921] = v923;
                                                                        v921 += 1l ;
                                                                    }
                                                                    int v925;
                                                                    v925 = 0l;
                                                                    while (while_method_0(v925)){
                                                                        unsigned char v927;
                                                                        v927 = v911[v925];
                                                                        int v929;
                                                                        v929 = 2l + v925;
                                                                        v919[v929] = v927;
                                                                        v925 += 1l ;
                                                                    }
                                                                    int v930;
                                                                    v930 = 0l;
                                                                    while (while_method_6(v930)){
                                                                        unsigned char v932;
                                                                        v932 = v913[v930];
                                                                        int v934;
                                                                        v934 = 4l + v930;
                                                                        v919[v934] = v932;
                                                                        v930 += 1l ;
                                                                    }
                                                                    v941 = Union9{Union9_1{v919}};
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
                                                    switch (v941.tag) {
                                                        case 0: { // None
                                                            static_array<unsigned char,2l> v943;
                                                            static_array<unsigned char,5l> v945;
                                                            int v947; int v948; int v949; unsigned char v950;
                                                            Tuple20 tmp62 = Tuple20{0l, 0l, 0l, 12u};
                                                            v947 = tmp62.v0; v948 = tmp62.v1; v949 = tmp62.v2; v950 = tmp62.v3;
                                                            while (while_method_10(v947)){
                                                                unsigned char v952;
                                                                v952 = v84[v947];
                                                                bool v954;
                                                                v954 = v949 < 2l;
                                                                int v962; int v963; unsigned char v964;
                                                                if (v954){
                                                                    unsigned char v955;
                                                                    v955 = v952 / 4u;
                                                                    bool v956;
                                                                    v956 = v950 == v955;
                                                                    int v957;
                                                                    if (v956){
                                                                        v957 = v949;
                                                                    } else {
                                                                        v957 = 0l;
                                                                    }
                                                                    v943[v957] = v952;
                                                                    int v958;
                                                                    v958 = v957 + 1l;
                                                                    v962 = v947; v963 = v958; v964 = v955;
                                                                } else {
                                                                    break;
                                                                }
                                                                v948 = v962;
                                                                v949 = v963;
                                                                v950 = v964;
                                                                v947 += 1l ;
                                                            }
                                                            bool v965;
                                                            v965 = v949 == 2l;
                                                            Union13 v976;
                                                            if (v965){
                                                                int v966;
                                                                v966 = 0l;
                                                                while (while_method_2(v966)){
                                                                    int v968;
                                                                    v968 = v948 + -1l;
                                                                    bool v969;
                                                                    v969 = v966 < v968;
                                                                    int v970;
                                                                    if (v969){
                                                                        v970 = 0l;
                                                                    } else {
                                                                        v970 = 2l;
                                                                    }
                                                                    int v971;
                                                                    v971 = v970 + v966;
                                                                    unsigned char v972;
                                                                    v972 = v84[v971];
                                                                    v945[v966] = v972;
                                                                    v966 += 1l ;
                                                                }
                                                                v976 = Union13{Union13_1{v943, v945}};
                                                            } else {
                                                                v976 = Union13{Union13_0{}};
                                                            }
                                                            Union9 v999;
                                                            switch (v976.tag) {
                                                                case 0: { // None
                                                                    v999 = Union9{Union9_0{}};
                                                                    break;
                                                                }
                                                                case 1: { // Some
                                                                    static_array<unsigned char,2l> v977 = v976.case1.v0; static_array<unsigned char,5l> v978 = v976.case1.v1;
                                                                    static_array<unsigned char,3l> v979;
                                                                    int v981;
                                                                    v981 = 0l;
                                                                    while (while_method_3(v981)){
                                                                        unsigned char v983;
                                                                        v983 = v978[v981];
                                                                        v979[v981] = v983;
                                                                        v981 += 1l ;
                                                                    }
                                                                    static_array<unsigned char,5l> v985;
                                                                    int v987;
                                                                    v987 = 0l;
                                                                    while (while_method_0(v987)){
                                                                        unsigned char v989;
                                                                        v989 = v977[v987];
                                                                        v985[v987] = v989;
                                                                        v987 += 1l ;
                                                                    }
                                                                    int v991;
                                                                    v991 = 0l;
                                                                    while (while_method_3(v991)){
                                                                        unsigned char v993;
                                                                        v993 = v979[v991];
                                                                        int v995;
                                                                        v995 = 2l + v991;
                                                                        v985[v995] = v993;
                                                                        v991 += 1l ;
                                                                    }
                                                                    v999 = Union9{Union9_1{v985}};
                                                                    break;
                                                                }
                                                                default: {
                                                                    assert("Invalid tag." && false);
                                                                }
                                                            }
                                                            switch (v999.tag) {
                                                                case 0: { // None
                                                                    static_array<unsigned char,5l> v1001;
                                                                    int v1003;
                                                                    v1003 = 0l;
                                                                    while (while_method_2(v1003)){
                                                                        unsigned char v1005;
                                                                        v1005 = v84[v1003];
                                                                        v1001[v1003] = v1005;
                                                                        v1003 += 1l ;
                                                                    }
                                                                    v1037 = v1001; v1038 = 0;
                                                                    break;
                                                                }
                                                                case 1: { // Some
                                                                    static_array<unsigned char,5l> v1000 = v999.case1.v0;
                                                                    v1037 = v1000; v1038 = 1;
                                                                    break;
                                                                }
                                                                default: {
                                                                    assert("Invalid tag." && false);
                                                                }
                                                            }
                                                            break;
                                                        }
                                                        case 1: { // Some
                                                            static_array<unsigned char,5l> v942 = v941.case1.v0;
                                                            v1037 = v942; v1038 = 2;
                                                            break;
                                                        }
                                                        default: {
                                                            assert("Invalid tag." && false);
                                                        }
                                                    }
                                                    break;
                                                }
                                                case 1: { // Some
                                                    static_array<unsigned char,5l> v840 = v839.case1.v0;
                                                    v1037 = v840; v1038 = 3;
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false);
                                                }
                                            }
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5l> v782 = v781.case1.v0;
                                            v1037 = v782; v1038 = 4;
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false);
                                        }
                                    }
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,5l> v743 = v742.case1.v0;
                                    v1037 = v743; v1038 = 5;
                                    break;
                                }
                                default: {
                                    assert("Invalid tag." && false);
                                }
                            }
                            break;
                        }
                        case 1: { // Some
                            static_array<unsigned char,5l> v590 = v589.case1.v0;
                            v1037 = v590; v1038 = 6;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                        }
                    }
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5l> v499 = v498.case1.v0;
                    v1037 = v499; v1038 = 7;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                }
            }
            break;
        }
        case 1: { // Some
            static_array<unsigned char,5l> v441 = v440.case1.v0;
            v1037 = v441; v1038 = 8;
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    return Tuple0{v1037, v1038};
}
__device__ Union5 play_loop_inner_32(unsigned long long & v0, dynamic_array_list<Union3,128l> & v1, curandStatePhilox4_32_10_t & v2, static_array<Union2,2l> v3, Union5 v4){
    dynamic_array_list<Union3,128l> & v5 = v1;
    unsigned long long & v6 = v0;
    bool v7; Union5 v8;
    Tuple9 tmp16 = Tuple9{true, v4};
    v7 = tmp16.v0; v8 = tmp16.v1;
    while (while_method_5(v7, v8)){
        bool v689; Union5 v690;
        switch (v8.tag) {
            case 0: { // G_Flop
                int v582 = v8.case0.v0; static_array<static_array<unsigned char,2l>,2l> v583 = v8.case0.v1; static_array<int,2l> v584 = v8.case0.v2; int v585 = v8.case0.v3; static_array<int,2l> v586 = v8.case0.v4; Union6 v587 = v8.case0.v5;
                static_array<unsigned char,3l> v588; unsigned long long v589;
                Tuple10 tmp19 = draw_cards_33(v2, v6);
                v588 = tmp19.v0; v589 = tmp19.v1;
                v0 = v589;
                static_array_list<unsigned char,5l> v590;
                v590 = get_community_cards_36(v587, v588);
                Union3 v591;
                v591 = Union3{Union3_0{v590}};
                v5.push(v591);
                Union6 v594;
                switch (v587.tag) {
                    case 1: { // Preflop
                        v594 = Union6{Union6_0{v588}};
                        break;
                    }
                    default: {
                        printf("%s\n", "Invalid street in flop.");
                        asm("exit;");
                    }
                }
                int v595;
                v595 = 2l;
                int v596;
                v596 = 0l;
                Union5 v597;
                v597 = try_round_37(v595, v583, v584, v596, v586, v594);
                v689 = true; v690 = v597;
                break;
            }
            case 1: { // G_Fold
                int v10 = v8.case1.v0; static_array<static_array<unsigned char,2l>,2l> v11 = v8.case1.v1; static_array<int,2l> v12 = v8.case1.v2; int v13 = v8.case1.v3; static_array<int,2l> v14 = v8.case1.v4; Union6 v15 = v8.case1.v5;
                int v16;
                v16 = v13 % 2l;
                int v17;
                v17 = v12[v16];
                int v19;
                v19 = v13 + 1l;
                int v20;
                v20 = v19 % 2l;
                Union3 v21;
                v21 = Union3{Union3_1{v17, v20}};
                v5.push(v21);
                v689 = false; v690 = v8;
                break;
            }
            case 2: { // G_Preflop
                static_array<unsigned char,2l> v654; unsigned long long v655;
                Tuple13 tmp24 = draw_cards_40(v2, v6);
                v654 = tmp24.v0; v655 = tmp24.v1;
                v0 = v655;
                static_array<unsigned char,2l> v656; unsigned long long v657;
                Tuple13 tmp25 = draw_cards_40(v2, v6);
                v656 = tmp25.v0; v657 = tmp25.v1;
                v0 = v657;
                Union3 v658;
                v658 = Union3{Union3_3{0l, v654}};
                v5.push(v658);
                Union3 v659;
                v659 = Union3{Union3_3{1l, v656}};
                v5.push(v659);
                static_array<static_array<unsigned char,2l>,2l> v660;
                v660[0l] = v654;
                v660[1l] = v656;
                static_array<int,2l> v662;
                v662[0l] = 2l;
                v662[1l] = 1l;
                static_array<int,2l> v664;
                int v666;
                v666 = 0l;
                while (while_method_0(v666)){
                    int v668;
                    v668 = v662[v666];
                    int v670;
                    v670 = 100l - v668;
                    v664[v666] = v670;
                    v666 += 1l ;
                }
                int v671;
                v671 = 2l;
                int v672;
                v672 = 0l;
                Union6 v673;
                v673 = Union6{Union6_1{}};
                Union5 v674;
                v674 = try_round_37(v671, v660, v662, v672, v664, v673);
                v689 = true; v690 = v674;
                break;
            }
            case 3: { // G_River
                int v626 = v8.case3.v0; static_array<static_array<unsigned char,2l>,2l> v627 = v8.case3.v1; static_array<int,2l> v628 = v8.case3.v2; int v629 = v8.case3.v3; static_array<int,2l> v630 = v8.case3.v4; Union6 v631 = v8.case3.v5;
                static_array<unsigned char,1l> v632; unsigned long long v633;
                Tuple14 tmp28 = draw_cards_41(v2, v6);
                v632 = tmp28.v0; v633 = tmp28.v1;
                v0 = v633;
                static_array_list<unsigned char,5l> v634;
                v634 = get_community_cards_42(v631, v632);
                Union3 v635;
                v635 = Union3{Union3_0{v634}};
                v5.push(v635);
                Union6 v650;
                switch (v631.tag) {
                    case 3: { // Turn
                        static_array<unsigned char,4l> v636 = v631.case3.v0;
                        static_array<unsigned char,5l> v637;
                        int v639;
                        v639 = 0l;
                        while (while_method_4(v639)){
                            unsigned char v641;
                            v641 = v636[v639];
                            v637[v639] = v641;
                            v639 += 1l ;
                        }
                        int v643;
                        v643 = 0l;
                        while (while_method_6(v643)){
                            unsigned char v645;
                            v645 = v632[v643];
                            int v647;
                            v647 = 4l + v643;
                            v637[v647] = v645;
                            v643 += 1l ;
                        }
                        v650 = Union6{Union6_2{v637}};
                        break;
                    }
                    default: {
                        printf("%s\n", "Invalid street in river.");
                        asm("exit;");
                    }
                }
                int v651;
                v651 = 2l;
                int v652;
                v652 = 0l;
                Union5 v653;
                v653 = try_round_37(v651, v627, v628, v652, v630, v650);
                v689 = true; v690 = v653;
                break;
            }
            case 4: { // G_Round
                int v97 = v8.case4.v0; static_array<static_array<unsigned char,2l>,2l> v98 = v8.case4.v1; static_array<int,2l> v99 = v8.case4.v2; int v100 = v8.case4.v3; static_array<int,2l> v101 = v8.case4.v4; Union6 v102 = v8.case4.v5;
                int v103;
                v103 = v100 % 2l;
                Union2 v104;
                v104 = v3[v103];
                switch (v104.tag) {
                    case 0: { // Computer
                        static_array<int,2l> v106;
                        int v108;
                        v108 = 0l;
                        while (while_method_0(v108)){
                            int v110;
                            v110 = v101[v108];
                            int v112;
                            v112 = v99[v108];
                            int v114;
                            v114 = v110 + v112;
                            v106[v108] = v114;
                            v108 += 1l ;
                        }
                        int v115;
                        v115 = v99[0l];
                        int v117; int v118;
                        Tuple2 tmp29 = Tuple2{1l, v115};
                        v117 = tmp29.v0; v118 = tmp29.v1;
                        while (while_method_0(v117)){
                            int v120;
                            v120 = v99[v117];
                            bool v122;
                            v122 = v118 >= v120;
                            int v123;
                            if (v122){
                                v123 = v118;
                            } else {
                                v123 = v120;
                            }
                            v118 = v123;
                            v117 += 1l ;
                        }
                        int v124;
                        v124 = v106[v103];
                        bool v126;
                        v126 = v118 < v124;
                        int v127;
                        if (v126){
                            v127 = v118;
                        } else {
                            v127 = v124;
                        }
                        static_array<int,2l> v128;
                        int v130;
                        v130 = 0l;
                        while (while_method_0(v130)){
                            int v132;
                            v132 = v99[v130];
                            bool v134;
                            v134 = v103 == v130;
                            int v135;
                            if (v134){
                                v135 = v127;
                            } else {
                                v135 = v132;
                            }
                            v128[v130] = v135;
                            v130 += 1l ;
                        }
                        static_array<int,2l> v136;
                        int v138;
                        v138 = 0l;
                        while (while_method_0(v138)){
                            int v140;
                            v140 = v106[v138];
                            int v142;
                            v142 = v128[v138];
                            int v144;
                            v144 = v140 - v142;
                            v136[v138] = v144;
                            v138 += 1l ;
                        }
                        int v145;
                        v145 = v128[0l];
                        int v147; int v148;
                        Tuple2 tmp30 = Tuple2{1l, v145};
                        v147 = tmp30.v0; v148 = tmp30.v1;
                        while (while_method_0(v147)){
                            int v150;
                            v150 = v128[v147];
                            int v152;
                            v152 = v148 + v150;
                            v148 = v152;
                            v147 += 1l ;
                        }
                        int v153;
                        v153 = v99[v103];
                        bool v155;
                        v155 = v153 < v118;
                        float v156;
                        if (v155){
                            v156 = 1.0f;
                        } else {
                            v156 = 0.0f;
                        }
                        int v157;
                        v157 = v148 / 3l;
                        bool v158;
                        v158 = v97 <= v157;
                        bool v162;
                        if (v158){
                            int v159;
                            v159 = v136[v103];
                            bool v161;
                            v161 = v157 < v159;
                            v162 = v161;
                        } else {
                            v162 = false;
                        }
                        float v163;
                        if (v162){
                            v163 = 1.0f;
                        } else {
                            v163 = 0.0f;
                        }
                        int v164;
                        v164 = v148 / 2l;
                        bool v165;
                        v165 = v97 <= v164;
                        bool v169;
                        if (v165){
                            int v166;
                            v166 = v136[v103];
                            bool v168;
                            v168 = v164 < v166;
                            v169 = v168;
                        } else {
                            v169 = false;
                        }
                        float v170;
                        if (v169){
                            v170 = 1.0f;
                        } else {
                            v170 = 0.0f;
                        }
                        bool v171;
                        v171 = v97 <= v148;
                        bool v175;
                        if (v171){
                            int v172;
                            v172 = v136[v103];
                            bool v174;
                            v174 = v148 < v172;
                            v175 = v174;
                        } else {
                            v175 = false;
                        }
                        float v176;
                        if (v175){
                            v176 = 1.0f;
                        } else {
                            v176 = 0.0f;
                        }
                        static_array<Tuple15,6l> v177;
                        Union1 v179;
                        v179 = Union1{Union1_2{}};
                        v177[0l] = Tuple15{v179, v156};
                        Union1 v181;
                        v181 = Union1{Union1_1{}};
                        v177[1l] = Tuple15{v181, 4.0f};
                        Union1 v183;
                        v183 = Union1{Union1_3{v157}};
                        v177[2l] = Tuple15{v183, v163};
                        Union1 v185;
                        v185 = Union1{Union1_3{v164}};
                        v177[3l] = Tuple15{v185, v170};
                        Union1 v187;
                        v187 = Union1{Union1_3{v148}};
                        v177[4l] = Tuple15{v187, v176};
                        Union1 v189;
                        v189 = Union1{Union1_0{}};
                        v177[5l] = Tuple15{v189, 1.0f};
                        Union1 v191;
                        v191 = sample_discrete_43(v177, v2);
                        Union3 v192;
                        v192 = Union3{Union3_2{v103, v191}};
                        v5.push(v192);
                        Union5 v380;
                        switch (v191.tag) {
                            case 0: { // A_All_In
                                static_array<int,2l> v310;
                                int v312;
                                v312 = 0l;
                                while (while_method_0(v312)){
                                    int v314;
                                    v314 = v101[v312];
                                    int v316;
                                    v316 = v99[v312];
                                    int v318;
                                    v318 = v314 + v316;
                                    v310[v312] = v318;
                                    v312 += 1l ;
                                }
                                int v319;
                                v319 = v99[0l];
                                int v321; int v322;
                                Tuple2 tmp33 = Tuple2{1l, v319};
                                v321 = tmp33.v0; v322 = tmp33.v1;
                                while (while_method_0(v321)){
                                    int v324;
                                    v324 = v99[v321];
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
                                int v328;
                                v328 = v310[v103];
                                bool v330;
                                v330 = v322 < v328;
                                int v331;
                                if (v330){
                                    v331 = v322;
                                } else {
                                    v331 = v328;
                                }
                                static_array<int,2l> v332;
                                int v334;
                                v334 = 0l;
                                while (while_method_0(v334)){
                                    int v336;
                                    v336 = v99[v334];
                                    bool v338;
                                    v338 = v103 == v334;
                                    int v339;
                                    if (v338){
                                        v339 = v331;
                                    } else {
                                        v339 = v336;
                                    }
                                    v332[v334] = v339;
                                    v334 += 1l ;
                                }
                                static_array<int,2l> v340;
                                int v342;
                                v342 = 0l;
                                while (while_method_0(v342)){
                                    int v344;
                                    v344 = v310[v342];
                                    int v346;
                                    v346 = v332[v342];
                                    int v348;
                                    v348 = v344 - v346;
                                    v340[v342] = v348;
                                    v342 += 1l ;
                                }
                                int v349;
                                v349 = v340[v103];
                                int v351;
                                v351 = v322 + v349;
                                int v352;
                                v352 = v310[v103];
                                bool v354;
                                v354 = v351 < v352;
                                int v355;
                                if (v354){
                                    v355 = v351;
                                } else {
                                    v355 = v352;
                                }
                                static_array<int,2l> v356;
                                int v358;
                                v358 = 0l;
                                while (while_method_0(v358)){
                                    int v360;
                                    v360 = v99[v358];
                                    bool v362;
                                    v362 = v103 == v358;
                                    int v363;
                                    if (v362){
                                        v363 = v355;
                                    } else {
                                        v363 = v360;
                                    }
                                    v356[v358] = v363;
                                    v358 += 1l ;
                                }
                                static_array<int,2l> v364;
                                int v366;
                                v366 = 0l;
                                while (while_method_0(v366)){
                                    int v368;
                                    v368 = v310[v366];
                                    int v370;
                                    v370 = v356[v366];
                                    int v372;
                                    v372 = v368 - v370;
                                    v364[v366] = v372;
                                    v366 += 1l ;
                                }
                                bool v373;
                                v373 = v349 >= v97;
                                int v374;
                                if (v373){
                                    v374 = v349;
                                } else {
                                    v374 = v97;
                                }
                                int v375;
                                v375 = v100 + 1l;
                                v380 = try_round_37(v374, v98, v356, v375, v364, v102);
                                break;
                            }
                            case 1: { // A_Call
                                static_array<int,2l> v194;
                                int v196;
                                v196 = 0l;
                                while (while_method_0(v196)){
                                    int v198;
                                    v198 = v101[v196];
                                    int v200;
                                    v200 = v99[v196];
                                    int v202;
                                    v202 = v198 + v200;
                                    v194[v196] = v202;
                                    v196 += 1l ;
                                }
                                int v203;
                                v203 = v99[0l];
                                int v205; int v206;
                                Tuple2 tmp34 = Tuple2{1l, v203};
                                v205 = tmp34.v0; v206 = tmp34.v1;
                                while (while_method_0(v205)){
                                    int v208;
                                    v208 = v99[v205];
                                    bool v210;
                                    v210 = v206 >= v208;
                                    int v211;
                                    if (v210){
                                        v211 = v206;
                                    } else {
                                        v211 = v208;
                                    }
                                    v206 = v211;
                                    v205 += 1l ;
                                }
                                int v212;
                                v212 = v194[v103];
                                bool v214;
                                v214 = v206 < v212;
                                int v215;
                                if (v214){
                                    v215 = v206;
                                } else {
                                    v215 = v212;
                                }
                                static_array<int,2l> v216;
                                int v218;
                                v218 = 0l;
                                while (while_method_0(v218)){
                                    int v220;
                                    v220 = v99[v218];
                                    bool v222;
                                    v222 = v103 == v218;
                                    int v223;
                                    if (v222){
                                        v223 = v215;
                                    } else {
                                        v223 = v220;
                                    }
                                    v216[v218] = v223;
                                    v218 += 1l ;
                                }
                                static_array<int,2l> v224;
                                int v226;
                                v226 = 0l;
                                while (while_method_0(v226)){
                                    int v228;
                                    v228 = v194[v226];
                                    int v230;
                                    v230 = v216[v226];
                                    int v232;
                                    v232 = v228 - v230;
                                    v224[v226] = v232;
                                    v226 += 1l ;
                                }
                                bool v233;
                                v233 = v103 < 2l;
                                if (v233){
                                    int v234;
                                    v234 = v100 + 1l;
                                    v380 = try_round_37(v97, v98, v216, v234, v224, v102);
                                } else {
                                    v380 = go_next_street_39(v97, v98, v216, v100, v224, v102);
                                }
                                break;
                            }
                            case 2: { // A_Fold
                                v380 = Union5{Union5_1{v97, v98, v99, v100, v101, v102}};
                                break;
                            }
                            case 3: { // A_Raise
                                int v238 = v191.case3.v0;
                                bool v239;
                                v239 = v97 <= v238;
                                bool v240;
                                v240 = v239 == false;
                                if (v240){
                                    assert("The raise amount must match the minimum." && v239);
                                } else {
                                }
                                static_array<int,2l> v242;
                                int v244;
                                v244 = 0l;
                                while (while_method_0(v244)){
                                    int v246;
                                    v246 = v101[v244];
                                    int v248;
                                    v248 = v99[v244];
                                    int v250;
                                    v250 = v246 + v248;
                                    v242[v244] = v250;
                                    v244 += 1l ;
                                }
                                int v251;
                                v251 = v99[0l];
                                int v253; int v254;
                                Tuple2 tmp35 = Tuple2{1l, v251};
                                v253 = tmp35.v0; v254 = tmp35.v1;
                                while (while_method_0(v253)){
                                    int v256;
                                    v256 = v99[v253];
                                    bool v258;
                                    v258 = v254 >= v256;
                                    int v259;
                                    if (v258){
                                        v259 = v254;
                                    } else {
                                        v259 = v256;
                                    }
                                    v254 = v259;
                                    v253 += 1l ;
                                }
                                int v260;
                                v260 = v242[v103];
                                bool v262;
                                v262 = v254 < v260;
                                int v263;
                                if (v262){
                                    v263 = v254;
                                } else {
                                    v263 = v260;
                                }
                                static_array<int,2l> v264;
                                int v266;
                                v266 = 0l;
                                while (while_method_0(v266)){
                                    int v268;
                                    v268 = v99[v266];
                                    bool v270;
                                    v270 = v103 == v266;
                                    int v271;
                                    if (v270){
                                        v271 = v263;
                                    } else {
                                        v271 = v268;
                                    }
                                    v264[v266] = v271;
                                    v266 += 1l ;
                                }
                                static_array<int,2l> v272;
                                int v274;
                                v274 = 0l;
                                while (while_method_0(v274)){
                                    int v276;
                                    v276 = v242[v274];
                                    int v278;
                                    v278 = v264[v274];
                                    int v280;
                                    v280 = v276 - v278;
                                    v272[v274] = v280;
                                    v274 += 1l ;
                                }
                                int v281;
                                v281 = v272[v103];
                                bool v283;
                                v283 = v238 < v281;
                                bool v284;
                                v284 = v283 == false;
                                if (v284){
                                    assert("The raise amount must be less than the stack size after calling." && v283);
                                } else {
                                }
                                int v286;
                                v286 = v254 + v238;
                                int v287;
                                v287 = v242[v103];
                                bool v289;
                                v289 = v286 < v287;
                                int v290;
                                if (v289){
                                    v290 = v286;
                                } else {
                                    v290 = v287;
                                }
                                static_array<int,2l> v291;
                                int v293;
                                v293 = 0l;
                                while (while_method_0(v293)){
                                    int v295;
                                    v295 = v99[v293];
                                    bool v297;
                                    v297 = v103 == v293;
                                    int v298;
                                    if (v297){
                                        v298 = v290;
                                    } else {
                                        v298 = v295;
                                    }
                                    v291[v293] = v298;
                                    v293 += 1l ;
                                }
                                static_array<int,2l> v299;
                                int v301;
                                v301 = 0l;
                                while (while_method_0(v301)){
                                    int v303;
                                    v303 = v242[v301];
                                    int v305;
                                    v305 = v291[v301];
                                    int v307;
                                    v307 = v303 - v305;
                                    v299[v301] = v307;
                                    v301 += 1l ;
                                }
                                int v308;
                                v308 = v100 + 1l;
                                v380 = try_round_37(v238, v98, v291, v308, v299, v102);
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                            }
                        }
                        v689 = true; v690 = v380;
                        break;
                    }
                    case 1: { // Human
                        v689 = false; v690 = v8;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                break;
            }
            case 5: { // G_Round'
                int v385 = v8.case5.v0; static_array<static_array<unsigned char,2l>,2l> v386 = v8.case5.v1; static_array<int,2l> v387 = v8.case5.v2; int v388 = v8.case5.v3; static_array<int,2l> v389 = v8.case5.v4; Union6 v390 = v8.case5.v5; Union1 v391 = v8.case5.v6;
                int v392;
                v392 = v388 % 2l;
                Union3 v393;
                v393 = Union3{Union3_2{v392, v391}};
                v5.push(v393);
                Union5 v581;
                switch (v391.tag) {
                    case 0: { // A_All_In
                        static_array<int,2l> v511;
                        int v513;
                        v513 = 0l;
                        while (while_method_0(v513)){
                            int v515;
                            v515 = v389[v513];
                            int v517;
                            v517 = v387[v513];
                            int v519;
                            v519 = v515 + v517;
                            v511[v513] = v519;
                            v513 += 1l ;
                        }
                        int v520;
                        v520 = v387[0l];
                        int v522; int v523;
                        Tuple2 tmp36 = Tuple2{1l, v520};
                        v522 = tmp36.v0; v523 = tmp36.v1;
                        while (while_method_0(v522)){
                            int v525;
                            v525 = v387[v522];
                            bool v527;
                            v527 = v523 >= v525;
                            int v528;
                            if (v527){
                                v528 = v523;
                            } else {
                                v528 = v525;
                            }
                            v523 = v528;
                            v522 += 1l ;
                        }
                        int v529;
                        v529 = v511[v392];
                        bool v531;
                        v531 = v523 < v529;
                        int v532;
                        if (v531){
                            v532 = v523;
                        } else {
                            v532 = v529;
                        }
                        static_array<int,2l> v533;
                        int v535;
                        v535 = 0l;
                        while (while_method_0(v535)){
                            int v537;
                            v537 = v387[v535];
                            bool v539;
                            v539 = v392 == v535;
                            int v540;
                            if (v539){
                                v540 = v532;
                            } else {
                                v540 = v537;
                            }
                            v533[v535] = v540;
                            v535 += 1l ;
                        }
                        static_array<int,2l> v541;
                        int v543;
                        v543 = 0l;
                        while (while_method_0(v543)){
                            int v545;
                            v545 = v511[v543];
                            int v547;
                            v547 = v533[v543];
                            int v549;
                            v549 = v545 - v547;
                            v541[v543] = v549;
                            v543 += 1l ;
                        }
                        int v550;
                        v550 = v541[v392];
                        int v552;
                        v552 = v523 + v550;
                        int v553;
                        v553 = v511[v392];
                        bool v555;
                        v555 = v552 < v553;
                        int v556;
                        if (v555){
                            v556 = v552;
                        } else {
                            v556 = v553;
                        }
                        static_array<int,2l> v557;
                        int v559;
                        v559 = 0l;
                        while (while_method_0(v559)){
                            int v561;
                            v561 = v387[v559];
                            bool v563;
                            v563 = v392 == v559;
                            int v564;
                            if (v563){
                                v564 = v556;
                            } else {
                                v564 = v561;
                            }
                            v557[v559] = v564;
                            v559 += 1l ;
                        }
                        static_array<int,2l> v565;
                        int v567;
                        v567 = 0l;
                        while (while_method_0(v567)){
                            int v569;
                            v569 = v511[v567];
                            int v571;
                            v571 = v557[v567];
                            int v573;
                            v573 = v569 - v571;
                            v565[v567] = v573;
                            v567 += 1l ;
                        }
                        bool v574;
                        v574 = v550 >= v385;
                        int v575;
                        if (v574){
                            v575 = v550;
                        } else {
                            v575 = v385;
                        }
                        int v576;
                        v576 = v388 + 1l;
                        v581 = try_round_37(v575, v386, v557, v576, v565, v390);
                        break;
                    }
                    case 1: { // A_Call
                        static_array<int,2l> v395;
                        int v397;
                        v397 = 0l;
                        while (while_method_0(v397)){
                            int v399;
                            v399 = v389[v397];
                            int v401;
                            v401 = v387[v397];
                            int v403;
                            v403 = v399 + v401;
                            v395[v397] = v403;
                            v397 += 1l ;
                        }
                        int v404;
                        v404 = v387[0l];
                        int v406; int v407;
                        Tuple2 tmp37 = Tuple2{1l, v404};
                        v406 = tmp37.v0; v407 = tmp37.v1;
                        while (while_method_0(v406)){
                            int v409;
                            v409 = v387[v406];
                            bool v411;
                            v411 = v407 >= v409;
                            int v412;
                            if (v411){
                                v412 = v407;
                            } else {
                                v412 = v409;
                            }
                            v407 = v412;
                            v406 += 1l ;
                        }
                        int v413;
                        v413 = v395[v392];
                        bool v415;
                        v415 = v407 < v413;
                        int v416;
                        if (v415){
                            v416 = v407;
                        } else {
                            v416 = v413;
                        }
                        static_array<int,2l> v417;
                        int v419;
                        v419 = 0l;
                        while (while_method_0(v419)){
                            int v421;
                            v421 = v387[v419];
                            bool v423;
                            v423 = v392 == v419;
                            int v424;
                            if (v423){
                                v424 = v416;
                            } else {
                                v424 = v421;
                            }
                            v417[v419] = v424;
                            v419 += 1l ;
                        }
                        static_array<int,2l> v425;
                        int v427;
                        v427 = 0l;
                        while (while_method_0(v427)){
                            int v429;
                            v429 = v395[v427];
                            int v431;
                            v431 = v417[v427];
                            int v433;
                            v433 = v429 - v431;
                            v425[v427] = v433;
                            v427 += 1l ;
                        }
                        bool v434;
                        v434 = v392 < 2l;
                        if (v434){
                            int v435;
                            v435 = v388 + 1l;
                            v581 = try_round_37(v385, v386, v417, v435, v425, v390);
                        } else {
                            v581 = go_next_street_39(v385, v386, v417, v388, v425, v390);
                        }
                        break;
                    }
                    case 2: { // A_Fold
                        v581 = Union5{Union5_1{v385, v386, v387, v388, v389, v390}};
                        break;
                    }
                    case 3: { // A_Raise
                        int v439 = v391.case3.v0;
                        bool v440;
                        v440 = v385 <= v439;
                        bool v441;
                        v441 = v440 == false;
                        if (v441){
                            assert("The raise amount must match the minimum." && v440);
                        } else {
                        }
                        static_array<int,2l> v443;
                        int v445;
                        v445 = 0l;
                        while (while_method_0(v445)){
                            int v447;
                            v447 = v389[v445];
                            int v449;
                            v449 = v387[v445];
                            int v451;
                            v451 = v447 + v449;
                            v443[v445] = v451;
                            v445 += 1l ;
                        }
                        int v452;
                        v452 = v387[0l];
                        int v454; int v455;
                        Tuple2 tmp38 = Tuple2{1l, v452};
                        v454 = tmp38.v0; v455 = tmp38.v1;
                        while (while_method_0(v454)){
                            int v457;
                            v457 = v387[v454];
                            bool v459;
                            v459 = v455 >= v457;
                            int v460;
                            if (v459){
                                v460 = v455;
                            } else {
                                v460 = v457;
                            }
                            v455 = v460;
                            v454 += 1l ;
                        }
                        int v461;
                        v461 = v443[v392];
                        bool v463;
                        v463 = v455 < v461;
                        int v464;
                        if (v463){
                            v464 = v455;
                        } else {
                            v464 = v461;
                        }
                        static_array<int,2l> v465;
                        int v467;
                        v467 = 0l;
                        while (while_method_0(v467)){
                            int v469;
                            v469 = v387[v467];
                            bool v471;
                            v471 = v392 == v467;
                            int v472;
                            if (v471){
                                v472 = v464;
                            } else {
                                v472 = v469;
                            }
                            v465[v467] = v472;
                            v467 += 1l ;
                        }
                        static_array<int,2l> v473;
                        int v475;
                        v475 = 0l;
                        while (while_method_0(v475)){
                            int v477;
                            v477 = v443[v475];
                            int v479;
                            v479 = v465[v475];
                            int v481;
                            v481 = v477 - v479;
                            v473[v475] = v481;
                            v475 += 1l ;
                        }
                        int v482;
                        v482 = v473[v392];
                        bool v484;
                        v484 = v439 < v482;
                        bool v485;
                        v485 = v484 == false;
                        if (v485){
                            assert("The raise amount must be less than the stack size after calling." && v484);
                        } else {
                        }
                        int v487;
                        v487 = v455 + v439;
                        int v488;
                        v488 = v443[v392];
                        bool v490;
                        v490 = v487 < v488;
                        int v491;
                        if (v490){
                            v491 = v487;
                        } else {
                            v491 = v488;
                        }
                        static_array<int,2l> v492;
                        int v494;
                        v494 = 0l;
                        while (while_method_0(v494)){
                            int v496;
                            v496 = v387[v494];
                            bool v498;
                            v498 = v392 == v494;
                            int v499;
                            if (v498){
                                v499 = v491;
                            } else {
                                v499 = v496;
                            }
                            v492[v494] = v499;
                            v494 += 1l ;
                        }
                        static_array<int,2l> v500;
                        int v502;
                        v502 = 0l;
                        while (while_method_0(v502)){
                            int v504;
                            v504 = v443[v502];
                            int v506;
                            v506 = v492[v502];
                            int v508;
                            v508 = v504 - v506;
                            v500[v502] = v508;
                            v502 += 1l ;
                        }
                        int v509;
                        v509 = v388 + 1l;
                        v581 = try_round_37(v439, v386, v492, v509, v500, v390);
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                v689 = true; v690 = v581;
                break;
            }
            case 6: { // G_Showdown
                int v22 = v8.case6.v0; static_array<static_array<unsigned char,2l>,2l> v23 = v8.case6.v1; static_array<int,2l> v24 = v8.case6.v2; int v25 = v8.case6.v3; static_array<int,2l> v26 = v8.case6.v4; Union6 v27 = v8.case6.v5;
                static_array<unsigned char,5l> v30;
                switch (v27.tag) {
                    case 2: { // River
                        static_array<unsigned char,5l> v28 = v27.case2.v0;
                        v30 = v28;
                        break;
                    }
                    default: {
                        printf("%s\n", "Invalid street in showdown.");
                        asm("exit;");
                    }
                }
                static_array<unsigned char,2l> v31;
                v31 = v23[0l];
                static_array<unsigned char,7l> v33;
                int v35;
                v35 = 0l;
                while (while_method_0(v35)){
                    unsigned char v37;
                    v37 = v31[v35];
                    v33[v35] = v37;
                    v35 += 1l ;
                }
                int v39;
                v39 = 0l;
                while (while_method_2(v39)){
                    unsigned char v41;
                    v41 = v30[v39];
                    int v43;
                    v43 = 2l + v39;
                    v33[v43] = v41;
                    v39 += 1l ;
                }
                static_array<unsigned char,5l> v44; char v45;
                Tuple0 tmp63 = score_46(v33);
                v44 = tmp63.v0; v45 = tmp63.v1;
                static_array<unsigned char,2l> v46;
                v46 = v23[1l];
                static_array<unsigned char,7l> v48;
                int v50;
                v50 = 0l;
                while (while_method_0(v50)){
                    unsigned char v52;
                    v52 = v46[v50];
                    v48[v50] = v52;
                    v50 += 1l ;
                }
                int v54;
                v54 = 0l;
                while (while_method_2(v54)){
                    unsigned char v56;
                    v56 = v30[v54];
                    int v58;
                    v58 = 2l + v54;
                    v48[v58] = v56;
                    v54 += 1l ;
                }
                static_array<unsigned char,5l> v59; char v60;
                Tuple0 tmp64 = score_46(v48);
                v59 = tmp64.v0; v60 = tmp64.v1;
                int v61;
                v61 = v25 % 2l;
                int v62;
                v62 = v24[v61];
                bool v64;
                v64 = v45 < v60;
                Union8 v70;
                if (v64){
                    v70 = Union8{Union8_2{}};
                } else {
                    bool v66;
                    v66 = v45 > v60;
                    if (v66){
                        v70 = Union8{Union8_1{}};
                    } else {
                        v70 = Union8{Union8_0{}};
                    }
                }
                Union8 v87;
                switch (v70.tag) {
                    case 0: { // Eq
                        Union8 v71;
                        v71 = Union8{Union8_0{}};
                        int v72;
                        v72 = 0l;
                        while (while_method_2(v72)){
                            unsigned char v74;
                            v74 = v44[v72];
                            unsigned char v76;
                            v76 = v59[v72];
                            bool v78;
                            v78 = v74 < v76;
                            Union8 v84;
                            if (v78){
                                v84 = Union8{Union8_2{}};
                            } else {
                                bool v80;
                                v80 = v74 > v76;
                                if (v80){
                                    v84 = Union8{Union8_1{}};
                                } else {
                                    v84 = Union8{Union8_0{}};
                                }
                            }
                            bool v85;
                            switch (v84.tag) {
                                case 0: { // Eq
                                    v85 = true;
                                    break;
                                }
                                default: {
                                    v85 = false;
                                }
                            }
                            bool v86;
                            v86 = v85 == false;
                            if (v86){
                                v71 = v84;
                                break;
                            } else {
                            }
                            v72 += 1l ;
                        }
                        v87 = v71;
                        break;
                    }
                    default: {
                        v87 = v70;
                    }
                }
                int v92; int v93;
                switch (v87.tag) {
                    case 0: { // Eq
                        v92 = 0l; v93 = -1l;
                        break;
                    }
                    case 1: { // Gt
                        v92 = v62; v93 = 0l;
                        break;
                    }
                    case 2: { // Lt
                        v92 = v62; v93 = 1l;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                static_array<Tuple0,2l> v94;
                v94[0l] = Tuple0{v44, v45};
                v94[1l] = Tuple0{v59, v60};
                Union3 v96;
                v96 = Union3{Union3_4{v92, v94, v93}};
                v5.push(v96);
                v689 = false; v690 = v8;
                break;
            }
            case 7: { // G_Turn
                int v598 = v8.case7.v0; static_array<static_array<unsigned char,2l>,2l> v599 = v8.case7.v1; static_array<int,2l> v600 = v8.case7.v2; int v601 = v8.case7.v3; static_array<int,2l> v602 = v8.case7.v4; Union6 v603 = v8.case7.v5;
                static_array<unsigned char,1l> v604; unsigned long long v605;
                Tuple14 tmp65 = draw_cards_41(v2, v6);
                v604 = tmp65.v0; v605 = tmp65.v1;
                v0 = v605;
                static_array_list<unsigned char,5l> v606;
                v606 = get_community_cards_42(v603, v604);
                Union3 v607;
                v607 = Union3{Union3_0{v606}};
                v5.push(v607);
                Union6 v622;
                switch (v603.tag) {
                    case 0: { // Flop
                        static_array<unsigned char,3l> v608 = v603.case0.v0;
                        static_array<unsigned char,4l> v609;
                        int v611;
                        v611 = 0l;
                        while (while_method_3(v611)){
                            unsigned char v613;
                            v613 = v608[v611];
                            v609[v611] = v613;
                            v611 += 1l ;
                        }
                        int v615;
                        v615 = 0l;
                        while (while_method_6(v615)){
                            unsigned char v617;
                            v617 = v604[v615];
                            int v619;
                            v619 = 3l + v615;
                            v609[v619] = v617;
                            v615 += 1l ;
                        }
                        v622 = Union6{Union6_3{v609}};
                        break;
                    }
                    default: {
                        printf("%s\n", "Invalid street in turn.");
                        asm("exit;");
                    }
                }
                int v623;
                v623 = 2l;
                int v624;
                v624 = 0l;
                Union5 v625;
                v625 = try_round_37(v623, v599, v600, v624, v602, v622);
                v689 = true; v690 = v625;
                break;
            }
            default: {
                assert("Invalid tag." && false);
            }
        }
        v7 = v689;
        v8 = v690;
    }
    return v8;
}
__device__ Tuple8 play_loop_31(Union4 v0, static_array<Union2,2l> v1, Union7 v2, unsigned long long & v3, dynamic_array_list<Union3,128l> & v4, curandStatePhilox4_32_10_t & v5, Union5 v6){
    Union5 v7;
    v7 = play_loop_inner_32(v3, v4, v5, v1, v6);
    switch (v7.tag) {
        case 1: { // G_Fold
            int v24 = v7.case1.v0; static_array<static_array<unsigned char,2l>,2l> v25 = v7.case1.v1; static_array<int,2l> v26 = v7.case1.v2; int v27 = v7.case1.v3; static_array<int,2l> v28 = v7.case1.v4; Union6 v29 = v7.case1.v5;
            Union4 v30;
            v30 = Union4{Union4_0{}};
            Union7 v31;
            v31 = Union7{Union7_1{v24, v25, v26, v27, v28, v29}};
            return Tuple8{v30, v1, v31};
            break;
        }
        case 4: { // G_Round
            int v8 = v7.case4.v0; static_array<static_array<unsigned char,2l>,2l> v9 = v7.case4.v1; static_array<int,2l> v10 = v7.case4.v2; int v11 = v7.case4.v3; static_array<int,2l> v12 = v7.case4.v4; Union6 v13 = v7.case4.v5;
            Union4 v14;
            v14 = Union4{Union4_1{v7}};
            Union7 v15;
            v15 = Union7{Union7_2{v8, v9, v10, v11, v12, v13}};
            return Tuple8{v14, v1, v15};
            break;
        }
        case 6: { // G_Showdown
            int v16 = v7.case6.v0; static_array<static_array<unsigned char,2l>,2l> v17 = v7.case6.v1; static_array<int,2l> v18 = v7.case6.v2; int v19 = v7.case6.v3; static_array<int,2l> v20 = v7.case6.v4; Union6 v21 = v7.case6.v5;
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
__device__ void f_49(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+8ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_51(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+0ull);
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
    int v2;
    v2 = v1.length;
    f_51(v0, v2);
    int v3;
    v3 = v1.length;
    int v4;
    v4 = 0l;
    while (while_method_1(v3, v4)){
        unsigned long long v6;
        v6 = (unsigned long long)v4;
        unsigned long long v7;
        v7 = 4ull + v6;
        unsigned char * v8;
        v8 = (unsigned char *)(v0+v7);
        unsigned char v10;
        v10 = v1[v4];
        f_53(v8, v10);
        v4 += 1l ;
    }
    return ;
}
__device__ void f_55(unsigned char * v0, int v1, int v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0l] = v1;
    int * v5;
    v5 = (int *)(v0+4ull);
    v5[0l] = v2;
    return ;
}
__device__ void f_57(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+4ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_58(unsigned char * v0){
    return ;
}
__device__ void f_56(unsigned char * v0, int v1, Union1 v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0l] = v1;
    int v5;
    v5 = v2.tag;
    f_57(v0, v5);
    unsigned char * v6;
    v6 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // A_All_In
            return f_58(v6);
            break;
        }
        case 1: { // A_Call
            return f_58(v6);
            break;
        }
        case 2: { // A_Fold
            return f_58(v6);
            break;
        }
        case 3: { // A_Raise
            int v8 = v2.case3.v0;
            return f_51(v6, v8);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_59(unsigned char * v0, int v1, static_array<unsigned char,2l> v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0l] = v1;
    int v5;
    v5 = 0l;
    while (while_method_0(v5)){
        unsigned long long v7;
        v7 = (unsigned long long)v5;
        unsigned long long v8;
        v8 = 4ull + v7;
        unsigned char * v9;
        v9 = (unsigned char *)(v0+v8);
        unsigned char v11;
        v11 = v2[v5];
        f_53(v9, v11);
        v5 += 1l ;
    }
    return ;
}
__device__ void f_62(unsigned char * v0, static_array<unsigned char,5l> v1, char v2){
    int v3;
    v3 = 0l;
    while (while_method_2(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = v1[v3];
        f_53(v6, v8);
        v3 += 1l ;
    }
    char * v10;
    v10 = (char *)(v0+5ull);
    v10[0l] = v2;
    return ;
}
__device__ void f_61(unsigned char * v0, static_array<unsigned char,5l> v1, char v2){
    return f_62(v0, v1, v2);
}
__device__ void f_60(unsigned char * v0, int v1, static_array<Tuple0,2l> v2, int v3){
    int * v4;
    v4 = (int *)(v0+0ull);
    v4[0l] = v1;
    int v6;
    v6 = 0l;
    while (while_method_0(v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = v8 * 8ull;
        unsigned long long v10;
        v10 = 8ull + v9;
        unsigned char * v11;
        v11 = (unsigned char *)(v0+v10);
        static_array<unsigned char,5l> v13; char v14;
        Tuple0 tmp68 = v2[v6];
        v13 = tmp68.v0; v14 = tmp68.v1;
        f_61(v11, v13, v14);
        v6 += 1l ;
    }
    int * v17;
    v17 = (int *)(v0+24ull);
    v17[0l] = v3;
    return ;
}
__device__ void f_50(unsigned char * v0, Union3 v1){
    int v2;
    v2 = v1.tag;
    f_51(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // CommunityCardsAre
            static_array_list<unsigned char,5l> v5 = v1.case0.v0;
            return f_52(v3, v5);
            break;
        }
        case 1: { // Fold
            int v6 = v1.case1.v0; int v7 = v1.case1.v1;
            return f_55(v3, v6, v7);
            break;
        }
        case 2: { // PlayerAction
            int v8 = v1.case2.v0; Union1 v9 = v1.case2.v1;
            return f_56(v3, v8, v9);
            break;
        }
        case 3: { // PlayerGotCards
            int v10 = v1.case3.v0; static_array<unsigned char,2l> v11 = v1.case3.v1;
            return f_59(v3, v10, v11);
            break;
        }
        case 4: { // Showdown
            int v12 = v1.case4.v0; static_array<Tuple0,2l> v13 = v1.case4.v1; int v14 = v1.case4.v2;
            return f_60(v3, v12, v13, v14);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_63(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+6160ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_66(unsigned char * v0, static_array<unsigned char,2l> v1){
    int v2;
    v2 = 0l;
    while (while_method_0(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v7;
        v7 = v1[v2];
        f_53(v5, v7);
        v2 += 1l ;
    }
    return ;
}
__device__ void f_67(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+28ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_68(unsigned char * v0, static_array<unsigned char,3l> v1){
    int v2;
    v2 = 0l;
    while (while_method_3(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v7;
        v7 = v1[v2];
        f_53(v5, v7);
        v2 += 1l ;
    }
    return ;
}
__device__ void f_69(unsigned char * v0, static_array<unsigned char,5l> v1){
    int v2;
    v2 = 0l;
    while (while_method_2(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v7;
        v7 = v1[v2];
        f_53(v5, v7);
        v2 += 1l ;
    }
    return ;
}
__device__ void f_70(unsigned char * v0, static_array<unsigned char,4l> v1){
    int v2;
    v2 = 0l;
    while (while_method_4(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v7;
        v7 = v1[v2];
        f_53(v5, v7);
        v2 += 1l ;
    }
    return ;
}
__device__ void f_65(unsigned char * v0, int v1, static_array<static_array<unsigned char,2l>,2l> v2, static_array<int,2l> v3, int v4, static_array<int,2l> v5, Union6 v6){
    int * v7;
    v7 = (int *)(v0+0ull);
    v7[0l] = v1;
    int v9;
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
        static_array<unsigned char,2l> v16;
        v16 = v2[v9];
        f_66(v14, v16);
        v9 += 1l ;
    }
    int v18;
    v18 = 0l;
    while (while_method_0(v18)){
        unsigned long long v20;
        v20 = (unsigned long long)v18;
        unsigned long long v21;
        v21 = v20 * 4ull;
        unsigned long long v22;
        v22 = 8ull + v21;
        unsigned char * v23;
        v23 = (unsigned char *)(v0+v22);
        int v25;
        v25 = v3[v18];
        f_51(v23, v25);
        v18 += 1l ;
    }
    int * v27;
    v27 = (int *)(v0+16ull);
    v27[0l] = v4;
    int v29;
    v29 = 0l;
    while (while_method_0(v29)){
        unsigned long long v31;
        v31 = (unsigned long long)v29;
        unsigned long long v32;
        v32 = v31 * 4ull;
        unsigned long long v33;
        v33 = 20ull + v32;
        unsigned char * v34;
        v34 = (unsigned char *)(v0+v33);
        int v36;
        v36 = v5[v29];
        f_51(v34, v36);
        v29 += 1l ;
    }
    int v38;
    v38 = v6.tag;
    f_67(v0, v38);
    unsigned char * v39;
    v39 = (unsigned char *)(v0+32ull);
    switch (v6.tag) {
        case 0: { // Flop
            static_array<unsigned char,3l> v41 = v6.case0.v0;
            return f_68(v39, v41);
            break;
        }
        case 1: { // Preflop
            return f_58(v39);
            break;
        }
        case 2: { // River
            static_array<unsigned char,5l> v42 = v6.case2.v0;
            return f_69(v39, v42);
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4l> v43 = v6.case3.v0;
            return f_70(v39, v43);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_72(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+40ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_71(unsigned char * v0, int v1, static_array<static_array<unsigned char,2l>,2l> v2, static_array<int,2l> v3, int v4, static_array<int,2l> v5, Union6 v6, Union1 v7){
    int * v8;
    v8 = (int *)(v0+0ull);
    v8[0l] = v1;
    int v10;
    v10 = 0l;
    while (while_method_0(v10)){
        unsigned long long v12;
        v12 = (unsigned long long)v10;
        unsigned long long v13;
        v13 = v12 * 2ull;
        unsigned long long v14;
        v14 = 4ull + v13;
        unsigned char * v15;
        v15 = (unsigned char *)(v0+v14);
        static_array<unsigned char,2l> v17;
        v17 = v2[v10];
        f_66(v15, v17);
        v10 += 1l ;
    }
    int v19;
    v19 = 0l;
    while (while_method_0(v19)){
        unsigned long long v21;
        v21 = (unsigned long long)v19;
        unsigned long long v22;
        v22 = v21 * 4ull;
        unsigned long long v23;
        v23 = 8ull + v22;
        unsigned char * v24;
        v24 = (unsigned char *)(v0+v23);
        int v26;
        v26 = v3[v19];
        f_51(v24, v26);
        v19 += 1l ;
    }
    int * v28;
    v28 = (int *)(v0+16ull);
    v28[0l] = v4;
    int v30;
    v30 = 0l;
    while (while_method_0(v30)){
        unsigned long long v32;
        v32 = (unsigned long long)v30;
        unsigned long long v33;
        v33 = v32 * 4ull;
        unsigned long long v34;
        v34 = 20ull + v33;
        unsigned char * v35;
        v35 = (unsigned char *)(v0+v34);
        int v37;
        v37 = v5[v30];
        f_51(v35, v37);
        v30 += 1l ;
    }
    int v39;
    v39 = v6.tag;
    f_67(v0, v39);
    unsigned char * v40;
    v40 = (unsigned char *)(v0+32ull);
    switch (v6.tag) {
        case 0: { // Flop
            static_array<unsigned char,3l> v42 = v6.case0.v0;
            f_68(v40, v42);
            break;
        }
        case 1: { // Preflop
            f_58(v40);
            break;
        }
        case 2: { // River
            static_array<unsigned char,5l> v43 = v6.case2.v0;
            f_69(v40, v43);
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4l> v44 = v6.case3.v0;
            f_70(v40, v44);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    int v45;
    v45 = v7.tag;
    f_72(v0, v45);
    unsigned char * v46;
    v46 = (unsigned char *)(v0+44ull);
    switch (v7.tag) {
        case 0: { // A_All_In
            return f_58(v46);
            break;
        }
        case 1: { // A_Call
            return f_58(v46);
            break;
        }
        case 2: { // A_Fold
            return f_58(v46);
            break;
        }
        case 3: { // A_Raise
            int v48 = v7.case3.v0;
            return f_51(v46, v48);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_64(unsigned char * v0, Union5 v1){
    int v2;
    v2 = v1.tag;
    f_51(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // G_Flop
            int v5 = v1.case0.v0; static_array<static_array<unsigned char,2l>,2l> v6 = v1.case0.v1; static_array<int,2l> v7 = v1.case0.v2; int v8 = v1.case0.v3; static_array<int,2l> v9 = v1.case0.v4; Union6 v10 = v1.case0.v5;
            return f_65(v3, v5, v6, v7, v8, v9, v10);
            break;
        }
        case 1: { // G_Fold
            int v11 = v1.case1.v0; static_array<static_array<unsigned char,2l>,2l> v12 = v1.case1.v1; static_array<int,2l> v13 = v1.case1.v2; int v14 = v1.case1.v3; static_array<int,2l> v15 = v1.case1.v4; Union6 v16 = v1.case1.v5;
            return f_65(v3, v11, v12, v13, v14, v15, v16);
            break;
        }
        case 2: { // G_Preflop
            return f_58(v3);
            break;
        }
        case 3: { // G_River
            int v17 = v1.case3.v0; static_array<static_array<unsigned char,2l>,2l> v18 = v1.case3.v1; static_array<int,2l> v19 = v1.case3.v2; int v20 = v1.case3.v3; static_array<int,2l> v21 = v1.case3.v4; Union6 v22 = v1.case3.v5;
            return f_65(v3, v17, v18, v19, v20, v21, v22);
            break;
        }
        case 4: { // G_Round
            int v23 = v1.case4.v0; static_array<static_array<unsigned char,2l>,2l> v24 = v1.case4.v1; static_array<int,2l> v25 = v1.case4.v2; int v26 = v1.case4.v3; static_array<int,2l> v27 = v1.case4.v4; Union6 v28 = v1.case4.v5;
            return f_65(v3, v23, v24, v25, v26, v27, v28);
            break;
        }
        case 5: { // G_Round'
            int v29 = v1.case5.v0; static_array<static_array<unsigned char,2l>,2l> v30 = v1.case5.v1; static_array<int,2l> v31 = v1.case5.v2; int v32 = v1.case5.v3; static_array<int,2l> v33 = v1.case5.v4; Union6 v34 = v1.case5.v5; Union1 v35 = v1.case5.v6;
            return f_71(v3, v29, v30, v31, v32, v33, v34, v35);
            break;
        }
        case 6: { // G_Showdown
            int v36 = v1.case6.v0; static_array<static_array<unsigned char,2l>,2l> v37 = v1.case6.v1; static_array<int,2l> v38 = v1.case6.v2; int v39 = v1.case6.v3; static_array<int,2l> v40 = v1.case6.v4; Union6 v41 = v1.case6.v5;
            return f_65(v3, v36, v37, v38, v39, v40, v41);
            break;
        }
        case 7: { // G_Turn
            int v42 = v1.case7.v0; static_array<static_array<unsigned char,2l>,2l> v43 = v1.case7.v1; static_array<int,2l> v44 = v1.case7.v2; int v45 = v1.case7.v3; static_array<int,2l> v46 = v1.case7.v4; Union6 v47 = v1.case7.v5;
            return f_65(v3, v42, v43, v44, v45, v46, v47);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_73(unsigned char * v0, Union2 v1){
    int v2;
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
__device__ void f_74(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+6248ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_47(unsigned char * v0, unsigned long long v1, dynamic_array_list<Union3,128l> v2, Union4 v3, static_array<Union2,2l> v4, Union7 v5){
    f_48(v0, v1);
    int v6;
    v6 = v2.length_();
    f_49(v0, v6);
    int v7;
    v7 = v2.length_();
    int v8;
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
        Union3 v15;
        v15 = v2[v8];
        f_50(v13, v15);
        v8 += 1l ;
    }
    int v17;
    v17 = v3.tag;
    f_63(v0, v17);
    unsigned char * v18;
    v18 = (unsigned char *)(v0+6176ull);
    switch (v3.tag) {
        case 0: { // None
            f_58(v18);
            break;
        }
        case 1: { // Some
            Union5 v20 = v3.case1.v0;
            f_64(v18, v20);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    int v21;
    v21 = 0l;
    while (while_method_0(v21)){
        unsigned long long v23;
        v23 = (unsigned long long)v21;
        unsigned long long v24;
        v24 = v23 * 4ull;
        unsigned long long v25;
        v25 = 6240ull + v24;
        unsigned char * v26;
        v26 = (unsigned char *)(v0+v25);
        Union2 v28;
        v28 = v4[v21];
        f_73(v26, v28);
        v21 += 1l ;
    }
    int v30;
    v30 = v5.tag;
    f_74(v0, v30);
    unsigned char * v31;
    v31 = (unsigned char *)(v0+6256ull);
    switch (v5.tag) {
        case 0: { // GameNotStarted
            return f_58(v31);
            break;
        }
        case 1: { // GameOver
            int v33 = v5.case1.v0; static_array<static_array<unsigned char,2l>,2l> v34 = v5.case1.v1; static_array<int,2l> v35 = v5.case1.v2; int v36 = v5.case1.v3; static_array<int,2l> v37 = v5.case1.v4; Union6 v38 = v5.case1.v5;
            return f_65(v31, v33, v34, v35, v36, v37, v38);
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            int v39 = v5.case2.v0; static_array<static_array<unsigned char,2l>,2l> v40 = v5.case2.v1; static_array<int,2l> v41 = v5.case2.v2; int v42 = v5.case2.v3; static_array<int,2l> v43 = v5.case2.v4; Union6 v44 = v5.case2.v5;
            return f_65(v31, v39, v40, v41, v42, v43, v44);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_76(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+6168ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_75(unsigned char * v0, dynamic_array_list<Union3,128l> v1, static_array<Union2,2l> v2, Union7 v3){
    int v4;
    v4 = v1.length_();
    f_51(v0, v4);
    int v5;
    v5 = v1.length_();
    int v6;
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
        Union3 v13;
        v13 = v1[v6];
        f_50(v11, v13);
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
        v19 = 6160ull + v18;
        unsigned char * v20;
        v20 = (unsigned char *)(v0+v19);
        Union2 v22;
        v22 = v2[v15];
        f_73(v20, v22);
        v15 += 1l ;
    }
    int v24;
    v24 = v3.tag;
    f_76(v0, v24);
    unsigned char * v25;
    v25 = (unsigned char *)(v0+6176ull);
    switch (v3.tag) {
        case 0: { // GameNotStarted
            return f_58(v25);
            break;
        }
        case 1: { // GameOver
            int v27 = v3.case1.v0; static_array<static_array<unsigned char,2l>,2l> v28 = v3.case1.v1; static_array<int,2l> v29 = v3.case1.v2; int v30 = v3.case1.v3; static_array<int,2l> v31 = v3.case1.v4; Union6 v32 = v3.case1.v5;
            return f_65(v25, v27, v28, v29, v30, v31, v32);
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            int v33 = v3.case2.v0; static_array<static_array<unsigned char,2l>,2l> v34 = v3.case2.v1; static_array<int,2l> v35 = v3.case2.v2; int v36 = v3.case2.v3; static_array<int,2l> v37 = v3.case2.v4; Union6 v38 = v3.case2.v5;
            return f_65(v25, v33, v34, v35, v36, v37, v38);
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
        unsigned long long v9; dynamic_array_list<Union3,128l> v10; Union4 v11; static_array<Union2,2l> v12; Union7 v13;
        Tuple1 tmp15 = f_6(v0);
        v9 = tmp15.v0; v10 = tmp15.v1; v11 = tmp15.v2; v12 = tmp15.v3; v13 = tmp15.v4;
        unsigned long long & v14 = v9;
        dynamic_array_list<Union3,128l> & v15 = v10;
        unsigned long long v16;
        v16 = clock64();
        int v17;
        v17 = threadIdx.x;
        int v18;
        v18 = blockIdx.x;
        int v19;
        v19 = v18 * 32l;
        int v20;
        v20 = v17 + v19;
        unsigned long long v21;
        v21 = (unsigned long long)v20;
        curandStatePhilox4_32_10_t v22;
        curand_init(v16,v21,0ull,&v22);
        Union4 v68; static_array<Union2,2l> v69; Union7 v70;
        switch (v8.tag) {
            case 0: { // ActionSelected
                Union1 v38 = v8.case0.v0;
                switch (v11.tag) {
                    case 0: { // None
                        v68 = v11; v69 = v12; v70 = v13;
                        break;
                    }
                    case 1: { // Some
                        Union5 v39 = v11.case1.v0;
                        switch (v39.tag) {
                            case 4: { // G_Round
                                int v40 = v39.case4.v0; static_array<static_array<unsigned char,2l>,2l> v41 = v39.case4.v1; static_array<int,2l> v42 = v39.case4.v2; int v43 = v39.case4.v3; static_array<int,2l> v44 = v39.case4.v4; Union6 v45 = v39.case4.v5;
                                Union5 v46;
                                v46 = Union5{Union5_5{v40, v41, v42, v43, v44, v45, v38}};
                                Tuple8 tmp66 = play_loop_31(v11, v12, v13, v14, v15, v22, v46);
                                v68 = tmp66.v0; v69 = tmp66.v1; v70 = tmp66.v2;
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
                static_array<Union2,2l> v37 = v8.case1.v0;
                v68 = v11; v69 = v37; v70 = v13;
                break;
            }
            case 2: { // StartGame
                static_array<Union2,2l> v23;
                Union2 v25;
                v25 = Union2{Union2_0{}};
                v23[0l] = v25;
                Union2 v27;
                v27 = Union2{Union2_1{}};
                v23[1l] = v27;
                dynamic_array_list<Union3,128l> v29{0};
                v14 = 4503599627370495ull;
                v15 = v29;
                Union4 v31;
                v31 = Union4{Union4_0{}};
                Union7 v32;
                v32 = Union7{Union7_0{}};
                Union5 v33;
                v33 = Union5{Union5_2{}};
                Tuple8 tmp67 = play_loop_31(v31, v23, v32, v14, v15, v22, v33);
                v68 = tmp67.v0; v69 = tmp67.v1; v70 = tmp67.v2;
                break;
            }
            default: {
                assert("Invalid tag." && false);
            }
        }
        f_47(v0, v9, v10, v68, v69, v70);
        return f_75(v2, v10, v69, v70);
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
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
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
        v1 = static_array(2)
        v3 = US2_0()
        v1[0] = v3
        del v3
        v5 = US2_1()
        v1[1] = v5
        del v5
        v7 = dynamic_array_list(128)
        v8 = 4503599627370495
        v9 = US3_0()
        v10 = US6_0()
        return method144(v8, v7, v9, v1, v10)
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
    v6 = static_array(2)
    v7 = 0
    while method6(v1, v7):
        v9 = v0[v7]
        v10 = method7(v9)
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
    v7 = static_array_list(5)
    v7.unsafe_set_length(v2)
    v8 = 0
    while method6(v2, v8):
        v10 = v0[v8]
        v11 = method16(v10)
        del v10
        v7[v8] = v11
        del v11
        v8 += 1 
    del v0, v2, v8
    return v7
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
    v6 = static_array(2)
    v7 = 0
    while method6(v1, v7):
        v9 = v0[v7]
        v10 = method16(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
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
    v6 = static_array(5)
    v7 = 0
    while method6(v1, v7):
        v9 = v0[v7]
        v10 = method16(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
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
    v6 = static_array(2)
    v7 = 0
    while method6(v1, v7):
        v9 = v0[v7]
        v10, v11 = method24(v9)
        del v9
        v6[v7] = (v10, v11)
        del v10, v11
        v7 += 1 
    del v0, v1, v7
    return v6
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
    v7 = dynamic_array_list(128)
    v7.unsafe_set_length(v2)
    v8 = 0
    while method6(v2, v8):
        v10 = v0[v8]
        v11 = method14(v10)
        del v10
        v7[v8] = v11
        del v11
        v8 += 1 
    del v0, v2, v8
    return v7
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
    v6 = static_array(2)
    v7 = 0
    while method6(v1, v7):
        v9 = v0[v7]
        v10 = method21(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
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
    v6 = static_array(2)
    v7 = 0
    while method6(v1, v7):
        v9 = v0[v7]
        v10 = method4(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
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
    v6 = static_array(3)
    v7 = 0
    while method6(v1, v7):
        v9 = v0[v7]
        v10 = method16(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
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
    v6 = static_array(4)
    v7 = 0
    while method6(v1, v7):
        v9 = v0[v7]
        v10 = method16(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
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
    v3 = v0[0:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method42(v0 : cp.ndarray) -> None:
    del v0
    return 
def method41(v0 : cp.ndarray, v1 : US1) -> None:
    v2 = v1.tag
    method40(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US1_0(): # A_All_In
            del v1
            return method42(v4)
        case US1_1(): # A_Call
            del v1
            return method42(v4)
        case US1_2(): # A_Fold
            del v1
            return method42(v4)
        case US1_3(v5): # A_Raise
            del v1
            return method40(v4, v5)
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
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US2_0(): # Computer
            del v1
            return method42(v4)
        case US2_1(): # Human
            del v1
            return method42(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method43(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method44(v2):
        v4 = u64(v2)
        v5 = v4 * 4
        del v4
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v9 = v1[v2]
        method45(v7, v9)
        del v7, v9
        v2 += 1 
    del v0, v1, v2
    return 
def method39(v0 : cp.ndarray, v1 : US0) -> None:
    v2 = v1.tag
    method40(v0, v2)
    del v2
    v4 = v0[8:].view(cp.uint8)
    del v0
    match v1:
        case US0_0(v5): # ActionSelected
            del v1
            return method41(v4, v5)
        case US0_1(v6): # PlayerChanged
            del v1
            return method43(v4, v6)
        case US0_2(): # StartGame
            del v1
            return method42(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method47(v0 : cp.ndarray, v1 : u64) -> None:
    v3 = v0[0:].view(cp.uint64)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method48(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[8:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method52(v0 : cp.ndarray, v1 : u8) -> None:
    v3 = v0[0:].view(cp.uint8)
    del v0
    v3[0] = v1
    del v1, v3
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
        v9 = v0[v7:].view(cp.uint8)
        del v7
        v11 = v1[v4]
        method51(v9, v11)
        del v9, v11
        v4 += 1 
    del v0, v1, v3, v4
    return 
def method53(v0 : cp.ndarray, v1 : i32, v2 : i32) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v6 = v0[4:].view(cp.int32)
    del v0
    v6[0] = v2
    del v2, v6
    return 
def method55(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[4:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method54(v0 : cp.ndarray, v1 : i32, v2 : US1) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = v2.tag
    method55(v0, v5)
    del v5
    v7 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US1_0(): # A_All_In
            del v2
            return method42(v7)
        case US1_1(): # A_Call
            del v2
            return method42(v7)
        case US1_2(): # A_Fold
            del v2
            return method42(v7)
        case US1_3(v8): # A_Raise
            del v2
            return method40(v7, v8)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method56(v0 : cp.ndarray, v1 : i32, v2 : static_array) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = 0
    while method44(v5):
        v7 = u64(v5)
        v8 = 4 + v7
        del v7
        v10 = v0[v8:].view(cp.uint8)
        del v8
        v12 = v2[v5]
        method51(v10, v12)
        del v10, v12
        v5 += 1 
    del v0, v2, v5
    return 
def method60(v0 : i32) -> bool:
    v1 = v0 < 5
    del v0
    return v1
def method59(v0 : cp.ndarray, v1 : static_array, v2 : i8) -> None:
    v3 = 0
    while method60(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v9 = v1[v3]
        method51(v7, v9)
        del v7, v9
        v3 += 1 
    del v1, v3
    v11 = v0[5:].view(cp.int8)
    del v0
    v11[0] = v2
    del v2, v11
    return 
def method58(v0 : cp.ndarray, v1 : static_array, v2 : i8) -> None:
    return method59(v0, v1, v2)
def method57(v0 : cp.ndarray, v1 : i32, v2 : static_array, v3 : i32) -> None:
    v5 = v0[0:].view(cp.int32)
    v5[0] = v1
    del v1, v5
    v6 = 0
    while method44(v6):
        v8 = u64(v6)
        v9 = v8 * 8
        del v8
        v10 = 8 + v9
        del v9
        v12 = v0[v10:].view(cp.uint8)
        del v10
        v15, v16 = v2[v6]
        method58(v12, v15, v16)
        del v12, v15, v16
        v6 += 1 
    del v2, v6
    v18 = v0[24:].view(cp.int32)
    del v0
    v18[0] = v3
    del v3, v18
    return 
def method49(v0 : cp.ndarray, v1 : US7) -> None:
    v2 = v1.tag
    method40(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US7_0(v5): # CommunityCardsAre
            del v1
            return method50(v4, v5)
        case US7_1(v6, v7): # Fold
            del v1
            return method53(v4, v6, v7)
        case US7_2(v8, v9): # PlayerAction
            del v1
            return method54(v4, v8, v9)
        case US7_3(v10, v11): # PlayerGotCards
            del v1
            return method56(v4, v10, v11)
        case US7_4(v12, v13, v14): # Showdown
            del v1
            return method57(v4, v12, v13, v14)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method61(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[6160:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method64(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method44(v2):
        v4 = u64(v2)
        v6 = v0[v4:].view(cp.uint8)
        del v4
        v8 = v1[v2]
        method51(v6, v8)
        del v6, v8
        v2 += 1 
    del v0, v1, v2
    return 
def method65(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[28:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method67(v0 : i32) -> bool:
    v1 = v0 < 3
    del v0
    return v1
def method66(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method67(v2):
        v4 = u64(v2)
        v6 = v0[v4:].view(cp.uint8)
        del v4
        v8 = v1[v2]
        method51(v6, v8)
        del v6, v8
        v2 += 1 
    del v0, v1, v2
    return 
def method68(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method60(v2):
        v4 = u64(v2)
        v6 = v0[v4:].view(cp.uint8)
        del v4
        v8 = v1[v2]
        method51(v6, v8)
        del v6, v8
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
        v6 = v0[v4:].view(cp.uint8)
        del v4
        v8 = v1[v2]
        method51(v6, v8)
        del v6, v8
        v2 += 1 
    del v0, v1, v2
    return 
def method63(v0 : cp.ndarray, v1 : i32, v2 : static_array, v3 : static_array, v4 : i32, v5 : static_array, v6 : US5) -> None:
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
        v15 = v0[v13:].view(cp.uint8)
        del v13
        v17 = v2[v9]
        method64(v15, v17)
        del v15, v17
        v9 += 1 
    del v2, v9
    v18 = 0
    while method44(v18):
        v20 = u64(v18)
        v21 = v20 * 4
        del v20
        v22 = 8 + v21
        del v21
        v24 = v0[v22:].view(cp.uint8)
        del v22
        v26 = v3[v18]
        method40(v24, v26)
        del v24, v26
        v18 += 1 
    del v3, v18
    v28 = v0[16:].view(cp.int32)
    v28[0] = v4
    del v4, v28
    v29 = 0
    while method44(v29):
        v31 = u64(v29)
        v32 = v31 * 4
        del v31
        v33 = 20 + v32
        del v32
        v35 = v0[v33:].view(cp.uint8)
        del v33
        v37 = v5[v29]
        method40(v35, v37)
        del v35, v37
        v29 += 1 
    del v5, v29
    v38 = v6.tag
    method65(v0, v38)
    del v38
    v40 = v0[32:].view(cp.uint8)
    del v0
    match v6:
        case US5_0(v41): # Flop
            del v6
            return method66(v40, v41)
        case US5_1(): # Preflop
            del v6
            return method42(v40)
        case US5_2(v42): # River
            del v6
            return method68(v40, v42)
        case US5_3(v43): # Turn
            del v6
            return method69(v40, v43)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method72(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[40:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method71(v0 : cp.ndarray, v1 : i32, v2 : static_array, v3 : static_array, v4 : i32, v5 : static_array, v6 : US5, v7 : US1) -> None:
    v9 = v0[0:].view(cp.int32)
    v9[0] = v1
    del v1, v9
    v10 = 0
    while method44(v10):
        v12 = u64(v10)
        v13 = v12 * 2
        del v12
        v14 = 4 + v13
        del v13
        v16 = v0[v14:].view(cp.uint8)
        del v14
        v18 = v2[v10]
        method64(v16, v18)
        del v16, v18
        v10 += 1 
    del v2, v10
    v19 = 0
    while method44(v19):
        v21 = u64(v19)
        v22 = v21 * 4
        del v21
        v23 = 8 + v22
        del v22
        v25 = v0[v23:].view(cp.uint8)
        del v23
        v27 = v3[v19]
        method40(v25, v27)
        del v25, v27
        v19 += 1 
    del v3, v19
    v29 = v0[16:].view(cp.int32)
    v29[0] = v4
    del v4, v29
    v30 = 0
    while method44(v30):
        v32 = u64(v30)
        v33 = v32 * 4
        del v32
        v34 = 20 + v33
        del v33
        v36 = v0[v34:].view(cp.uint8)
        del v34
        v38 = v5[v30]
        method40(v36, v38)
        del v36, v38
        v30 += 1 
    del v5, v30
    v39 = v6.tag
    method65(v0, v39)
    del v39
    v41 = v0[32:].view(cp.uint8)
    match v6:
        case US5_0(v42): # Flop
            method66(v41, v42)
        case US5_1(): # Preflop
            method42(v41)
        case US5_2(v43): # River
            method68(v41, v43)
        case US5_3(v44): # Turn
            method69(v41, v44)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v6, v41
    v45 = v7.tag
    method72(v0, v45)
    del v45
    v47 = v0[44:].view(cp.uint8)
    del v0
    match v7:
        case US1_0(): # A_All_In
            del v7
            return method42(v47)
        case US1_1(): # A_Call
            del v7
            return method42(v47)
        case US1_2(): # A_Fold
            del v7
            return method42(v47)
        case US1_3(v48): # A_Raise
            del v7
            return method40(v47, v48)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method62(v0 : cp.ndarray, v1 : US4) -> None:
    v2 = v1.tag
    method40(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US4_0(v5, v6, v7, v8, v9, v10): # G_Flop
            del v1
            return method63(v4, v5, v6, v7, v8, v9, v10)
        case US4_1(v11, v12, v13, v14, v15, v16): # G_Fold
            del v1
            return method63(v4, v11, v12, v13, v14, v15, v16)
        case US4_2(): # G_Preflop
            del v1
            return method42(v4)
        case US4_3(v17, v18, v19, v20, v21, v22): # G_River
            del v1
            return method63(v4, v17, v18, v19, v20, v21, v22)
        case US4_4(v23, v24, v25, v26, v27, v28): # G_Round
            del v1
            return method63(v4, v23, v24, v25, v26, v27, v28)
        case US4_5(v29, v30, v31, v32, v33, v34, v35): # G_Round'
            del v1
            return method71(v4, v29, v30, v31, v32, v33, v34, v35)
        case US4_6(v36, v37, v38, v39, v40, v41): # G_Showdown
            del v1
            return method63(v4, v36, v37, v38, v39, v40, v41)
        case US4_7(v42, v43, v44, v45, v46, v47): # G_Turn
            del v1
            return method63(v4, v42, v43, v44, v45, v46, v47)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method73(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[6248:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
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
        v14 = v0[v12:].view(cp.uint8)
        del v12
        v16 = v2[v8]
        method49(v14, v16)
        del v14, v16
        v8 += 1 
    del v2, v7, v8
    v17 = v3.tag
    method61(v0, v17)
    del v17
    v19 = v0[6176:].view(cp.uint8)
    match v3:
        case US3_0(): # None
            method42(v19)
        case US3_1(v20): # Some
            method62(v19, v20)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v3, v19
    v21 = 0
    while method44(v21):
        v23 = u64(v21)
        v24 = v23 * 4
        del v23
        v25 = 6240 + v24
        del v24
        v27 = v0[v25:].view(cp.uint8)
        del v25
        v29 = v4[v21]
        method45(v27, v29)
        del v27, v29
        v21 += 1 
    del v4, v21
    v30 = v5.tag
    method73(v0, v30)
    del v30
    v32 = v0[6256:].view(cp.uint8)
    del v0
    match v5:
        case US6_0(): # GameNotStarted
            del v5
            return method42(v32)
        case US6_1(v33, v34, v35, v36, v37, v38): # GameOver
            del v5
            return method63(v32, v33, v34, v35, v36, v37, v38)
        case US6_2(v39, v40, v41, v42, v43, v44): # WaitingForActionFromPlayerId
            del v5
            return method63(v32, v39, v40, v41, v42, v43, v44)
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
    v2 = v0[0:].view(cp.uint64)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method78(v0 : cp.ndarray) -> i32:
    v2 = v0[8:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method80(v0 : cp.ndarray) -> i32:
    v2 = v0[0:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method83(v0 : cp.ndarray) -> u8:
    v2 = v0[0:].view(cp.uint8)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method82(v0 : cp.ndarray) -> u8:
    v1 = method83(v0)
    del v0
    return v1
def method81(v0 : cp.ndarray) -> static_array_list:
    v2 = static_array_list(5)
    v3 = method80(v0)
    v2.unsafe_set_length(v3)
    del v3
    v4 = v2.length
    v5 = 0
    while method6(v4, v5):
        v7 = u64(v5)
        v8 = 4 + v7
        del v7
        v10 = v0[v8:].view(cp.uint8)
        del v8
        v11 = method82(v10)
        del v10
        v2[v5] = v11
        del v11
        v5 += 1 
    del v0, v4, v5
    return v2
def method84(v0 : cp.ndarray) -> Tuple[i32, i32]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v5 = v0[4:].view(cp.int32)
    del v0
    v6 = v5[0].item()
    del v5
    return v3, v6
def method86(v0 : cp.ndarray) -> i32:
    v2 = v0[4:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method87(v0 : cp.ndarray) -> None:
    del v0
    return 
def method85(v0 : cp.ndarray) -> Tuple[i32, US1]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method86(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method87(v6)
        v13 = US1_0()
    elif v4 == 1:
        method87(v6)
        v13 = US1_1()
    elif v4 == 2:
        method87(v6)
        v13 = US1_2()
    elif v4 == 3:
        v11 = method80(v6)
        v13 = US1_3(v11)
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v13
def method88(v0 : cp.ndarray) -> Tuple[i32, static_array]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v5 = static_array(2)
    v6 = 0
    while method44(v6):
        v8 = u64(v6)
        v9 = 4 + v8
        del v8
        v11 = v0[v9:].view(cp.uint8)
        del v9
        v12 = method82(v11)
        del v11
        v5[v6] = v12
        del v12
        v6 += 1 
    del v0, v6
    return v3, v5
def method91(v0 : cp.ndarray) -> Tuple[static_array, i8]:
    v2 = static_array(5)
    v3 = 0
    while method60(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method82(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v3
    v10 = v0[5:].view(cp.int8)
    del v0
    v11 = v10[0].item()
    del v10
    return v2, v11
def method90(v0 : cp.ndarray) -> Tuple[static_array, i8]:
    v1, v2 = method91(v0)
    del v0
    return v1, v2
def method89(v0 : cp.ndarray) -> Tuple[i32, static_array, i32]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v5 = static_array(2)
    v6 = 0
    while method44(v6):
        v8 = u64(v6)
        v9 = v8 * 8
        del v8
        v10 = 8 + v9
        del v9
        v12 = v0[v10:].view(cp.uint8)
        del v10
        v13, v14 = method90(v12)
        del v12
        v5[v6] = (v13, v14)
        del v13, v14
        v6 += 1 
    del v6
    v16 = v0[24:].view(cp.int32)
    del v0
    v17 = v16[0].item()
    del v16
    return v3, v5, v17
def method79(v0 : cp.ndarray) -> US7:
    v1 = method80(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5 = method81(v3)
        del v3
        return US7_0(v5)
    elif v1 == 1:
        del v1
        v7, v8 = method84(v3)
        del v3
        return US7_1(v7, v8)
    elif v1 == 2:
        del v1
        v10, v11 = method85(v3)
        del v3
        return US7_2(v10, v11)
    elif v1 == 3:
        del v1
        v13, v14 = method88(v3)
        del v3
        return US7_3(v13, v14)
    elif v1 == 4:
        del v1
        v16, v17, v18 = method89(v3)
        del v3
        return US7_4(v16, v17, v18)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method92(v0 : cp.ndarray) -> i32:
    v2 = v0[6160:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method95(v0 : cp.ndarray) -> static_array:
    v2 = static_array(2)
    v3 = 0
    while method44(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method82(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v0, v3
    return v2
def method96(v0 : cp.ndarray) -> i32:
    v2 = v0[28:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method97(v0 : cp.ndarray) -> static_array:
    v2 = static_array(3)
    v3 = 0
    while method67(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method82(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v0, v3
    return v2
def method98(v0 : cp.ndarray) -> static_array:
    v2 = static_array(5)
    v3 = 0
    while method60(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method82(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v0, v3
    return v2
def method99(v0 : cp.ndarray) -> static_array:
    v2 = static_array(4)
    v3 = 0
    while method70(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method82(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v0, v3
    return v2
def method94(v0 : cp.ndarray) -> Tuple[i32, static_array, static_array, i32, static_array, US5]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v5 = static_array(2)
    v6 = 0
    while method44(v6):
        v8 = u64(v6)
        v9 = v8 * 2
        del v8
        v10 = 4 + v9
        del v9
        v12 = v0[v10:].view(cp.uint8)
        del v10
        v13 = method95(v12)
        del v12
        v5[v6] = v13
        del v13
        v6 += 1 
    del v6
    v15 = static_array(2)
    v16 = 0
    while method44(v16):
        v18 = u64(v16)
        v19 = v18 * 4
        del v18
        v20 = 8 + v19
        del v19
        v22 = v0[v20:].view(cp.uint8)
        del v20
        v23 = method80(v22)
        del v22
        v15[v16] = v23
        del v23
        v16 += 1 
    del v16
    v25 = v0[16:].view(cp.int32)
    v26 = v25[0].item()
    del v25
    v28 = static_array(2)
    v29 = 0
    while method44(v29):
        v31 = u64(v29)
        v32 = v31 * 4
        del v31
        v33 = 20 + v32
        del v32
        v35 = v0[v33:].view(cp.uint8)
        del v33
        v36 = method80(v35)
        del v35
        v28[v29] = v36
        del v36
        v29 += 1 
    del v29
    v37 = method96(v0)
    v39 = v0[32:].view(cp.uint8)
    del v0
    if v37 == 0:
        v41 = method97(v39)
        v48 = US5_0(v41)
    elif v37 == 1:
        method87(v39)
        v48 = US5_1()
    elif v37 == 2:
        v44 = method98(v39)
        v48 = US5_2(v44)
    elif v37 == 3:
        v46 = method99(v39)
        v48 = US5_3(v46)
    else:
        raise Exception("Invalid tag.")
    del v37, v39
    return v3, v5, v15, v26, v28, v48
def method101(v0 : cp.ndarray) -> i32:
    v2 = v0[40:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method100(v0 : cp.ndarray) -> Tuple[i32, static_array, static_array, i32, static_array, US5, US1]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v5 = static_array(2)
    v6 = 0
    while method44(v6):
        v8 = u64(v6)
        v9 = v8 * 2
        del v8
        v10 = 4 + v9
        del v9
        v12 = v0[v10:].view(cp.uint8)
        del v10
        v13 = method95(v12)
        del v12
        v5[v6] = v13
        del v13
        v6 += 1 
    del v6
    v15 = static_array(2)
    v16 = 0
    while method44(v16):
        v18 = u64(v16)
        v19 = v18 * 4
        del v18
        v20 = 8 + v19
        del v19
        v22 = v0[v20:].view(cp.uint8)
        del v20
        v23 = method80(v22)
        del v22
        v15[v16] = v23
        del v23
        v16 += 1 
    del v16
    v25 = v0[16:].view(cp.int32)
    v26 = v25[0].item()
    del v25
    v28 = static_array(2)
    v29 = 0
    while method44(v29):
        v31 = u64(v29)
        v32 = v31 * 4
        del v31
        v33 = 20 + v32
        del v32
        v35 = v0[v33:].view(cp.uint8)
        del v33
        v36 = method80(v35)
        del v35
        v28[v29] = v36
        del v36
        v29 += 1 
    del v29
    v37 = method96(v0)
    v39 = v0[32:].view(cp.uint8)
    if v37 == 0:
        v41 = method97(v39)
        v48 = US5_0(v41)
    elif v37 == 1:
        method87(v39)
        v48 = US5_1()
    elif v37 == 2:
        v44 = method98(v39)
        v48 = US5_2(v44)
    elif v37 == 3:
        v46 = method99(v39)
        v48 = US5_3(v46)
    else:
        raise Exception("Invalid tag.")
    del v37, v39
    v49 = method101(v0)
    v51 = v0[44:].view(cp.uint8)
    del v0
    if v49 == 0:
        method87(v51)
        v58 = US1_0()
    elif v49 == 1:
        method87(v51)
        v58 = US1_1()
    elif v49 == 2:
        method87(v51)
        v58 = US1_2()
    elif v49 == 3:
        v56 = method80(v51)
        v58 = US1_3(v56)
    else:
        raise Exception("Invalid tag.")
    del v49, v51
    return v3, v5, v15, v26, v28, v48, v58
def method93(v0 : cp.ndarray) -> US4:
    v1 = method80(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5, v6, v7, v8, v9, v10 = method94(v3)
        del v3
        return US4_0(v5, v6, v7, v8, v9, v10)
    elif v1 == 1:
        del v1
        v12, v13, v14, v15, v16, v17 = method94(v3)
        del v3
        return US4_1(v12, v13, v14, v15, v16, v17)
    elif v1 == 2:
        del v1
        method87(v3)
        del v3
        return US4_2()
    elif v1 == 3:
        del v1
        v20, v21, v22, v23, v24, v25 = method94(v3)
        del v3
        return US4_3(v20, v21, v22, v23, v24, v25)
    elif v1 == 4:
        del v1
        v27, v28, v29, v30, v31, v32 = method94(v3)
        del v3
        return US4_4(v27, v28, v29, v30, v31, v32)
    elif v1 == 5:
        del v1
        v34, v35, v36, v37, v38, v39, v40 = method100(v3)
        del v3
        return US4_5(v34, v35, v36, v37, v38, v39, v40)
    elif v1 == 6:
        del v1
        v42, v43, v44, v45, v46, v47 = method94(v3)
        del v3
        return US4_6(v42, v43, v44, v45, v46, v47)
    elif v1 == 7:
        del v1
        v49, v50, v51, v52, v53, v54 = method94(v3)
        del v3
        return US4_7(v49, v50, v51, v52, v53, v54)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method102(v0 : cp.ndarray) -> US2:
    v1 = method80(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method87(v3)
        del v3
        return US2_0()
    elif v1 == 1:
        del v1
        method87(v3)
        del v3
        return US2_1()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method103(v0 : cp.ndarray) -> i32:
    v2 = v0[6248:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method76(v0 : cp.ndarray) -> Tuple[u64, dynamic_array_list, US3, static_array, US6]:
    v1 = method77(v0)
    v3 = dynamic_array_list(128)
    v4 = method78(v0)
    v3.unsafe_set_length(v4)
    del v4
    v5 = v3.length_()
    v6 = 0
    while method6(v5, v6):
        v8 = u64(v6)
        v9 = v8 * 48
        del v8
        v10 = 16 + v9
        del v9
        v12 = v0[v10:].view(cp.uint8)
        del v10
        v13 = method79(v12)
        del v12
        v3[v6] = v13
        del v13
        v6 += 1 
    del v5, v6
    v14 = method92(v0)
    v16 = v0[6176:].view(cp.uint8)
    if v14 == 0:
        method87(v16)
        v21 = US3_0()
    elif v14 == 1:
        v19 = method93(v16)
        v21 = US3_1(v19)
    else:
        raise Exception("Invalid tag.")
    del v14, v16
    v23 = static_array(2)
    v24 = 0
    while method44(v24):
        v26 = u64(v24)
        v27 = v26 * 4
        del v26
        v28 = 6240 + v27
        del v27
        v30 = v0[v28:].view(cp.uint8)
        del v28
        v31 = method102(v30)
        del v30
        v23[v24] = v31
        del v31
        v24 += 1 
    del v24
    v32 = method103(v0)
    v34 = v0[6256:].view(cp.uint8)
    del v0
    if v32 == 0:
        method87(v34)
        v51 = US6_0()
    elif v32 == 1:
        v37, v38, v39, v40, v41, v42 = method94(v34)
        v51 = US6_1(v37, v38, v39, v40, v41, v42)
    elif v32 == 2:
        v44, v45, v46, v47, v48, v49 = method94(v34)
        v51 = US6_2(v44, v45, v46, v47, v48, v49)
    else:
        raise Exception("Invalid tag.")
    del v32, v34
    return v1, v3, v21, v23, v51
def method105(v0 : cp.ndarray) -> i32:
    v2 = v0[6168:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method104(v0 : cp.ndarray) -> Tuple[dynamic_array_list, static_array, US6]:
    v2 = dynamic_array_list(128)
    v3 = method80(v0)
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
        v11 = v0[v9:].view(cp.uint8)
        del v9
        v12 = method79(v11)
        del v11
        v2[v5] = v12
        del v12
        v5 += 1 
    del v4, v5
    v14 = static_array(2)
    v15 = 0
    while method44(v15):
        v17 = u64(v15)
        v18 = v17 * 4
        del v17
        v19 = 6160 + v18
        del v18
        v21 = v0[v19:].view(cp.uint8)
        del v19
        v22 = method102(v21)
        del v21
        v14[v15] = v22
        del v22
        v15 += 1 
    del v15
    v23 = method105(v0)
    v25 = v0[6176:].view(cp.uint8)
    del v0
    if v23 == 0:
        method87(v25)
        v42 = US6_0()
    elif v23 == 1:
        v28, v29, v30, v31, v32, v33 = method94(v25)
        v42 = US6_1(v28, v29, v30, v31, v32, v33)
    elif v23 == 2:
        v35, v36, v37, v38, v39, v40 = method94(v25)
        v42 = US6_2(v35, v36, v37, v38, v39, v40)
    else:
        raise Exception("Invalid tag.")
    del v23, v25
    return v2, v14, v42
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
        v6 = v0[v3]
        v7 = method115(v6)
        del v6
        v1.append(v7)
        del v7
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
        v5 = v0[v2]
        v6 = method115(v5)
        del v5
        v1.append(v6)
        del v6
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
        v5 = v0[v2]
        v6 = method115(v5)
        del v5
        v1.append(v6)
        del v6
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
        v6, v7 = v0[v2]
        v8 = method126(v6, v7)
        del v6, v7
        v1.append(v8)
        del v8
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
        v6 = v0[v3]
        v7 = method113(v6)
        del v6
        v1.append(v7)
        del v7
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
        v5 = v0[v2]
        v6 = method123(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method135(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method44(v2):
        v5 = v0[v2]
        v6 = method118(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method137(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method67(v2):
        v5 = v0[v2]
        v6 = method115(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method138(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method70(v2):
        v5 = v0[v2]
        v6 = method115(v5)
        del v5
        v1.append(v6)
        del v6
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
        v5 = v0[v2]
        v6 = method141(v5)
        del v5
        v1.append(v6)
        del v6
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
