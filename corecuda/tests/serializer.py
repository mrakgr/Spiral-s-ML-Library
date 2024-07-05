kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
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
struct Union3;
struct Union4;
struct Union2;
struct Union1;
struct Tuple0;
__device__ void f_1(unsigned char * v0, int v1);
__device__ void f_3(unsigned char * v0, int v1);
__device__ void f_4(unsigned char * v0);
__device__ void f_5(unsigned char * v0, unsigned long long v1);
__device__ void f_7(unsigned char * v0, int v1);
__device__ void f_9(unsigned char * v0, float v1);
__device__ void f_8(unsigned char * v0, char v1, unsigned char v2, Union3 v3);
__device__ void f_11(unsigned char * v0, double v1);
__device__ void f_10(unsigned char * v0, unsigned short v1, Union4 v2);
__device__ void f_6(unsigned char * v0, Union2 v1);
__device__ void f_2(unsigned char * v0, int v1, Union0 v2, Union1 v3);
__device__ void f_0(unsigned char * v0, short v1, unsigned long long v2, static_array_list<Tuple0,14l> v3, unsigned short v4);
struct Tuple1;
__device__ int f_13(unsigned char * v0);
__device__ int f_15(unsigned char * v0);
__device__ void f_16(unsigned char * v0);
__device__ unsigned long long f_17(unsigned char * v0);
__device__ int f_19(unsigned char * v0);
struct Tuple2;
__device__ float f_21(unsigned char * v0);
__device__ Tuple2 f_20(unsigned char * v0);
struct Tuple3;
__device__ double f_23(unsigned char * v0);
__device__ Tuple3 f_22(unsigned char * v0);
__device__ Union2 f_18(unsigned char * v0);
__device__ Tuple0 f_14(unsigned char * v0);
__device__ Tuple1 f_12(unsigned char * v0);
__device__ void write_25(short v0);
__device__ void write_26(const char * v0);
__device__ void write_28(unsigned long long v0);
__device__ void write_32(int v0);
__device__ void write_35();
__device__ void write_36();
__device__ void write_34(Union0 v0);
__device__ void write_39();
__device__ void write_41(char v0);
__device__ void write_43(unsigned char v0);
__device__ void write_45(float v0);
__device__ void write_44(Union3 v0);
__device__ void write_42(unsigned char v0, Union3 v1);
__device__ void write_40(char v0, unsigned char v1, Union3 v2);
__device__ void write_46();
__device__ void write_48(unsigned short v0);
__device__ void write_50(double v0);
__device__ void write_49(Union4 v0);
__device__ void write_47(unsigned short v0, Union4 v1);
__device__ void write_38(Union2 v0);
__device__ void write_37(Union1 v0);
__device__ void write_33(Union0 v0, Union1 v1);
__device__ void write_31(int v0, Union0 v1, Union1 v2);
__device__ void write_30(static_array_list<Tuple0,14l> v0);
__device__ void write_29(static_array_list<Tuple0,14l> v0, unsigned short v1);
__device__ void write_27(unsigned long long v0, static_array_list<Tuple0,14l> v1, unsigned short v2);
__device__ void write_24(short v0, unsigned long long v1, static_array_list<Tuple0,14l> v2, unsigned short v3);
struct Union0_0 { // None
};
struct Union0_1 { // Some
    unsigned long long v0;
    __device__ Union0_1(unsigned long long t0) : v0(t0) {}
    __device__ Union0_1() = delete;
};
struct Union0 {
    union {
        Union0_0 case0; // None
        Union0_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // None
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // Some
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // None
            case 1: new (&this->case1) Union0_1(x.case1); break; // Some
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // None
            case 1: this->case1.~Union0_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union3_0 { // None
};
struct Union3_1 { // Some
    float v0;
    __device__ Union3_1(float t0) : v0(t0) {}
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
struct Union4_0 { // None
};
struct Union4_1 { // Some
    double v0;
    __device__ Union4_1(double t0) : v0(t0) {}
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
struct Union2_0 { // Q
    Union3 v2;
    unsigned char v1;
    char v0;
    __device__ Union2_0(char t0, unsigned char t1, Union3 t2) : v0(t0), v1(t1), v2(t2) {}
    __device__ Union2_0() = delete;
};
struct Union2_1 { // W
    Union4 v1;
    unsigned short v0;
    __device__ Union2_1(unsigned short t0, Union4 t1) : v0(t0), v1(t1) {}
    __device__ Union2_1() = delete;
};
struct Union2 {
    union {
        Union2_0 case0; // Q
        Union2_1 case1; // W
    };
    unsigned char tag{255};
    __device__ Union2() {}
    __device__ Union2(Union2_0 t) : tag(0), case0(t) {} // Q
    __device__ Union2(Union2_1 t) : tag(1), case1(t) {} // W
    __device__ Union2(Union2 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(x.case0); break; // Q
            case 1: new (&this->case1) Union2_1(x.case1); break; // W
        }
    }
    __device__ Union2(Union2 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(std::move(x.case0)); break; // Q
            case 1: new (&this->case1) Union2_1(std::move(x.case1)); break; // W
        }
    }
    __device__ Union2 & operator=(Union2 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Q
                case 1: this->case1 = x.case1; break; // W
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
                case 0: this->case0 = std::move(x.case0); break; // Q
                case 1: this->case1 = std::move(x.case1); break; // W
            }
        } else {
            this->~Union2();
            new (this) Union2{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union2() {
        switch(this->tag){
            case 0: this->case0.~Union2_0(); break; // Q
            case 1: this->case1.~Union2_1(); break; // W
        }
        this->tag = 255;
    }
};
struct Union1_0 { // None
};
struct Union1_1 { // Some
    Union2 v0;
    __device__ Union1_1(Union2 t0) : v0(t0) {}
    __device__ Union1_1() = delete;
};
struct Union1 {
    union {
        Union1_0 case0; // None
        Union1_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union1() {}
    __device__ Union1(Union1_0 t) : tag(0), case0(t) {} // None
    __device__ Union1(Union1_1 t) : tag(1), case1(t) {} // Some
    __device__ Union1(Union1 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(x.case0); break; // None
            case 1: new (&this->case1) Union1_1(x.case1); break; // Some
        }
    }
    __device__ Union1(Union1 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union1 & operator=(Union1 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union1();
            new (this) Union1{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union1() {
        switch(this->tag){
            case 0: this->case0.~Union1_0(); break; // None
            case 1: this->case1.~Union1_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Tuple0 {
    Union0 v1;
    Union1 v2;
    int v0;
    __device__ Tuple0() = default;
    __device__ Tuple0(int t0, Union0 t1, Union1 t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple1 {
    unsigned long long v1;
    static_array_list<Tuple0,14l> v2;
    short v0;
    unsigned short v3;
    __device__ Tuple1() = default;
    __device__ Tuple1(short t0, unsigned long long t1, static_array_list<Tuple0,14l> t2, unsigned short t3) : v0(t0), v1(t1), v2(t2), v3(t3) {}
};
struct Tuple2 {
    Union3 v2;
    unsigned char v1;
    char v0;
    __device__ Tuple2() = default;
    __device__ Tuple2(char t0, unsigned char t1, Union3 t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple3 {
    Union4 v1;
    unsigned short v0;
    __device__ Tuple3() = default;
    __device__ Tuple3(unsigned short t0, Union4 t1) : v0(t0), v1(t1) {}
};
__device__ void f_1(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+16ull);
    v2[0l] = v1;
    return ;
}
__device__ inline bool while_method_0(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ void f_3(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+4ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_4(unsigned char * v0){
    return ;
}
__device__ void f_5(unsigned char * v0, unsigned long long v1){
    unsigned long long * v2;
    v2 = (unsigned long long *)(v0+0ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_7(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+0ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_9(unsigned char * v0, float v1){
    float * v2;
    v2 = (float *)(v0+0ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_8(unsigned char * v0, char v1, unsigned char v2, Union3 v3){
    char * v4;
    v4 = (char *)(v0+0ull);
    v4[0l] = v1;
    unsigned char * v6;
    v6 = (unsigned char *)(v0+1ull);
    v6[0l] = v2;
    int v8;
    v8 = v3.tag;
    f_3(v0, v8);
    unsigned char * v9;
    v9 = (unsigned char *)(v0+8ull);
    switch (v3.tag) {
        case 0: { // None
            return f_4(v9);
            break;
        }
        case 1: { // Some
            float v11 = v3.case1.v0;
            return f_9(v9, v11);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_11(unsigned char * v0, double v1){
    double * v2;
    v2 = (double *)(v0+0ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_10(unsigned char * v0, unsigned short v1, Union4 v2){
    unsigned short * v3;
    v3 = (unsigned short *)(v0+0ull);
    v3[0l] = v1;
    int v5;
    v5 = v2.tag;
    f_3(v0, v5);
    unsigned char * v6;
    v6 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // None
            return f_4(v6);
            break;
        }
        case 1: { // Some
            double v8 = v2.case1.v0;
            return f_11(v6, v8);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_6(unsigned char * v0, Union2 v1){
    int v2;
    v2 = v1.tag;
    f_7(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // Q
            char v5 = v1.case0.v0; unsigned char v6 = v1.case0.v1; Union3 v7 = v1.case0.v2;
            return f_8(v3, v5, v6, v7);
            break;
        }
        case 1: { // W
            unsigned short v8 = v1.case1.v0; Union4 v9 = v1.case1.v1;
            return f_10(v3, v8, v9);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_2(unsigned char * v0, int v1, Union0 v2, Union1 v3){
    int * v4;
    v4 = (int *)(v0+0ull);
    v4[0l] = v1;
    int v6;
    v6 = v2.tag;
    f_3(v0, v6);
    unsigned char * v7;
    v7 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // None
            f_4(v7);
            break;
        }
        case 1: { // Some
            unsigned long long v9 = v2.case1.v0;
            f_5(v7, v9);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    int v10;
    v10 = v3.tag;
    f_1(v0, v10);
    unsigned char * v11;
    v11 = (unsigned char *)(v0+32ull);
    switch (v3.tag) {
        case 0: { // None
            return f_4(v11);
            break;
        }
        case 1: { // Some
            Union2 v13 = v3.case1.v0;
            return f_6(v11, v13);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_0(unsigned char * v0, short v1, unsigned long long v2, static_array_list<Tuple0,14l> v3, unsigned short v4){
    short * v5;
    v5 = (short *)(v0+0ull);
    v5[0l] = v1;
    unsigned long long * v7;
    v7 = (unsigned long long *)(v0+8ull);
    v7[0l] = v2;
    int v9;
    v9 = v3.length;
    f_1(v0, v9);
    int v10;
    v10 = v3.length;
    int v11;
    v11 = 0l;
    while (while_method_0(v10, v11)){
        unsigned long long v13;
        v13 = (unsigned long long)v11;
        unsigned long long v14;
        v14 = v13 * 64ull;
        unsigned long long v15;
        v15 = 32ull + v14;
        unsigned char * v16;
        v16 = (unsigned char *)(v0+v15);
        int v18; Union0 v19; Union1 v20;
        Tuple0 tmp0 = v3[v11];
        v18 = tmp0.v0; v19 = tmp0.v1; v20 = tmp0.v2;
        f_2(v16, v18, v19, v20);
        v11 += 1l ;
    }
    unsigned short * v24;
    v24 = (unsigned short *)(v0+928ull);
    v24[0l] = v4;
    return ;
}
__device__ int f_13(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+16ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ int f_15(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+4ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ void f_16(unsigned char * v0){
    return ;
}
__device__ unsigned long long f_17(unsigned char * v0){
    unsigned long long * v1;
    v1 = (unsigned long long *)(v0+0ull);
    unsigned long long v3;
    v3 = v1[0l];
    return v3;
}
__device__ int f_19(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ float f_21(unsigned char * v0){
    float * v1;
    v1 = (float *)(v0+0ull);
    float v3;
    v3 = v1[0l];
    return v3;
}
__device__ Tuple2 f_20(unsigned char * v0){
    char * v1;
    v1 = (char *)(v0+0ull);
    char v3;
    v3 = v1[0l];
    unsigned char * v4;
    v4 = (unsigned char *)(v0+1ull);
    unsigned char v6;
    v6 = v4[0l];
    int v7;
    v7 = f_15(v0);
    unsigned char * v8;
    v8 = (unsigned char *)(v0+8ull);
    Union3 v14;
    switch (v7) {
        case 0: {
            f_16(v8);
            v14 = Union3{Union3_0{}};
            break;
        }
        case 1: {
            float v12;
            v12 = f_21(v8);
            v14 = Union3{Union3_1{v12}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple2{v3, v6, v14};
}
__device__ double f_23(unsigned char * v0){
    double * v1;
    v1 = (double *)(v0+0ull);
    double v3;
    v3 = v1[0l];
    return v3;
}
__device__ Tuple3 f_22(unsigned char * v0){
    unsigned short * v1;
    v1 = (unsigned short *)(v0+0ull);
    unsigned short v3;
    v3 = v1[0l];
    int v4;
    v4 = f_15(v0);
    unsigned char * v5;
    v5 = (unsigned char *)(v0+8ull);
    Union4 v11;
    switch (v4) {
        case 0: {
            f_16(v5);
            v11 = Union4{Union4_0{}};
            break;
        }
        case 1: {
            double v9;
            v9 = f_23(v5);
            v11 = Union4{Union4_1{v9}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple3{v3, v11};
}
__device__ Union2 f_18(unsigned char * v0){
    int v1;
    v1 = f_19(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            char v5; unsigned char v6; Union3 v7;
            Tuple2 tmp1 = f_20(v2);
            v5 = tmp1.v0; v6 = tmp1.v1; v7 = tmp1.v2;
            return Union2{Union2_0{v5, v6, v7}};
            break;
        }
        case 1: {
            unsigned short v9; Union4 v10;
            Tuple3 tmp2 = f_22(v2);
            v9 = tmp2.v0; v10 = tmp2.v1;
            return Union2{Union2_1{v9, v10}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ Tuple0 f_14(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    int v4;
    v4 = f_15(v0);
    unsigned char * v5;
    v5 = (unsigned char *)(v0+8ull);
    Union0 v11;
    switch (v4) {
        case 0: {
            f_16(v5);
            v11 = Union0{Union0_0{}};
            break;
        }
        case 1: {
            unsigned long long v9;
            v9 = f_17(v5);
            v11 = Union0{Union0_1{v9}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    int v12;
    v12 = f_13(v0);
    unsigned char * v13;
    v13 = (unsigned char *)(v0+32ull);
    Union1 v19;
    switch (v12) {
        case 0: {
            f_16(v13);
            v19 = Union1{Union1_0{}};
            break;
        }
        case 1: {
            Union2 v17;
            v17 = f_18(v13);
            v19 = Union1{Union1_1{v17}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple0{v3, v11, v19};
}
__device__ Tuple1 f_12(unsigned char * v0){
    short * v1;
    v1 = (short *)(v0+0ull);
    short v3;
    v3 = v1[0l];
    unsigned long long * v4;
    v4 = (unsigned long long *)(v0+8ull);
    unsigned long long v6;
    v6 = v4[0l];
    static_array_list<Tuple0,14l> v7;
    v7 = static_array_list<Tuple0,14l>{};
    int v9;
    v9 = f_13(v0);
    v7.unsafe_set_length(v9);
    int v10;
    v10 = v7.length;
    int v11;
    v11 = 0l;
    while (while_method_0(v10, v11)){
        unsigned long long v13;
        v13 = (unsigned long long)v11;
        unsigned long long v14;
        v14 = v13 * 64ull;
        unsigned long long v15;
        v15 = 32ull + v14;
        unsigned char * v16;
        v16 = (unsigned char *)(v0+v15);
        int v18; Union0 v19; Union1 v20;
        Tuple0 tmp3 = f_14(v16);
        v18 = tmp3.v0; v19 = tmp3.v1; v20 = tmp3.v2;
        v7[v11] = Tuple0{v18, v19, v20};
        v11 += 1l ;
    }
    unsigned short * v21;
    v21 = (unsigned short *)(v0+928ull);
    unsigned short v23;
    v23 = v21[0l];
    return Tuple1{v3, v6, v7, v23};
}
__device__ void write_25(short v0){
    const char * v1;
    v1 = "%d";
    printf(v1,v0);
    return ;
}
__device__ void write_26(const char * v0){
    const char * v1;
    v1 = "%s";
    printf(v1,v0);
    return ;
}
__device__ void write_28(unsigned long long v0){
    const char * v1;
    v1 = "%u";
    printf(v1,v0);
    return ;
}
__device__ void write_32(int v0){
    const char * v1;
    v1 = "%d";
    printf(v1,v0);
    return ;
}
__device__ void write_35(){
    const char * v0;
    v0 = "None";
    return write_26(v0);
}
__device__ void write_36(){
    const char * v0;
    v0 = "Some";
    return write_26(v0);
}
__device__ void write_34(Union0 v0){
    switch (v0.tag) {
        case 0: { // None
            return write_35();
            break;
        }
        case 1: { // Some
            unsigned long long v1 = v0.case1.v0;
            write_36();
            const char * v2;
            v2 = "(";
            write_26(v2);
            write_28(v1);
            const char * v3;
            v3 = ")";
            return write_26(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void write_39(){
    const char * v0;
    v0 = "Q";
    return write_26(v0);
}
__device__ void write_41(char v0){
    const char * v1;
    v1 = "%d";
    printf(v1,v0);
    return ;
}
__device__ void write_43(unsigned char v0){
    const char * v1;
    v1 = "%u";
    printf(v1,v0);
    return ;
}
__device__ void write_45(float v0){
    const char * v1;
    v1 = "%f";
    printf(v1,v0);
    return ;
}
__device__ void write_44(Union3 v0){
    switch (v0.tag) {
        case 0: { // None
            return write_35();
            break;
        }
        case 1: { // Some
            float v1 = v0.case1.v0;
            write_36();
            const char * v2;
            v2 = "(";
            write_26(v2);
            write_45(v1);
            const char * v3;
            v3 = ")";
            return write_26(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void write_42(unsigned char v0, Union3 v1){
    write_43(v0);
    const char * v2;
    v2 = ", ";
    write_26(v2);
    return write_44(v1);
}
__device__ void write_40(char v0, unsigned char v1, Union3 v2){
    write_41(v0);
    const char * v3;
    v3 = ", ";
    write_26(v3);
    return write_42(v1, v2);
}
__device__ void write_46(){
    const char * v0;
    v0 = "W";
    return write_26(v0);
}
__device__ void write_48(unsigned short v0){
    const char * v1;
    v1 = "%u";
    printf(v1,v0);
    return ;
}
__device__ void write_50(double v0){
    const char * v1;
    v1 = "%f";
    printf(v1,v0);
    return ;
}
__device__ void write_49(Union4 v0){
    switch (v0.tag) {
        case 0: { // None
            return write_35();
            break;
        }
        case 1: { // Some
            double v1 = v0.case1.v0;
            write_36();
            const char * v2;
            v2 = "(";
            write_26(v2);
            write_50(v1);
            const char * v3;
            v3 = ")";
            return write_26(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void write_47(unsigned short v0, Union4 v1){
    write_48(v0);
    const char * v2;
    v2 = ", ";
    write_26(v2);
    return write_49(v1);
}
__device__ void write_38(Union2 v0){
    switch (v0.tag) {
        case 0: { // Q
            char v1 = v0.case0.v0; unsigned char v2 = v0.case0.v1; Union3 v3 = v0.case0.v2;
            write_39();
            const char * v4;
            v4 = "(";
            write_26(v4);
            write_40(v1, v2, v3);
            const char * v5;
            v5 = ")";
            return write_26(v5);
            break;
        }
        case 1: { // W
            unsigned short v6 = v0.case1.v0; Union4 v7 = v0.case1.v1;
            write_46();
            const char * v8;
            v8 = "(";
            write_26(v8);
            write_47(v6, v7);
            const char * v9;
            v9 = ")";
            return write_26(v9);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void write_37(Union1 v0){
    switch (v0.tag) {
        case 0: { // None
            return write_35();
            break;
        }
        case 1: { // Some
            Union2 v1 = v0.case1.v0;
            write_36();
            const char * v2;
            v2 = "(";
            write_26(v2);
            write_38(v1);
            const char * v3;
            v3 = ")";
            return write_26(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void write_33(Union0 v0, Union1 v1){
    write_34(v0);
    const char * v2;
    v2 = ", ";
    write_26(v2);
    return write_37(v1);
}
__device__ void write_31(int v0, Union0 v1, Union1 v2){
    write_32(v0);
    const char * v3;
    v3 = ", ";
    write_26(v3);
    return write_33(v1, v2);
}
__device__ void write_30(static_array_list<Tuple0,14l> v0){
    const char * v1;
    v1 = "[";
    write_26(v1);
    int v2;
    v2 = v0.length;
    bool v3;
    v3 = 100l < v2;
    int v4;
    if (v3){
        v4 = 100l;
    } else {
        v4 = v2;
    }
    int v5;
    v5 = 0l;
    while (while_method_0(v4, v5)){
        int v7; Union0 v8; Union1 v9;
        Tuple0 tmp5 = v0[v5];
        v7 = tmp5.v0; v8 = tmp5.v1; v9 = tmp5.v2;
        write_31(v7, v8, v9);
        int v13;
        v13 = v5 + 1l;
        int v14;
        v14 = v0.length;
        bool v15;
        v15 = v13 < v14;
        if (v15){
            const char * v16;
            v16 = "; ";
            write_26(v16);
        } else {
        }
        v5 += 1l ;
    }
    int v17;
    v17 = v0.length;
    bool v18;
    v18 = v17 > 100l;
    if (v18){
        const char * v19;
        v19 = "; ...";
        write_26(v19);
    } else {
    }
    const char * v20;
    v20 = "]";
    return write_26(v20);
}
__device__ void write_29(static_array_list<Tuple0,14l> v0, unsigned short v1){
    write_30(v0);
    const char * v2;
    v2 = ", ";
    write_26(v2);
    return write_48(v1);
}
__device__ void write_27(unsigned long long v0, static_array_list<Tuple0,14l> v1, unsigned short v2){
    write_28(v0);
    const char * v3;
    v3 = ", ";
    write_26(v3);
    return write_29(v1, v2);
}
__device__ void write_24(short v0, unsigned long long v1, static_array_list<Tuple0,14l> v2, unsigned short v3){
    write_25(v0);
    const char * v4;
    v4 = ", ";
    write_26(v4);
    return write_27(v1, v2, v3);
}
extern "C" __global__ void entry0(unsigned char * v0) {
    int v1;
    v1 = threadIdx.x;
    int v2;
    v2 = blockIdx.x;
    int v3;
    v3 = v2 * 32l;
    int v4;
    v4 = v1 + v3;
    bool v5;
    v5 = v4 == 0l;
    if (v5){
        static_array_list<Tuple0,14l> v6;
        v6 = static_array_list<Tuple0,14l>{};
        v6.unsafe_set_length(3l);
        Union0 v8;
        v8 = Union0{Union0_1{23ull}};
        Union3 v9;
        v9 = Union3{Union3_1{555.0f}};
        Union2 v10;
        v10 = Union2{Union2_0{5, 55u, v9}};
        Union1 v11;
        v11 = Union1{Union1_1{v10}};
        v6[0l] = Tuple0{1l, v8, v11};
        Union0 v16;
        v16 = Union0{Union0_1{34ull}};
        Union4 v17;
        v17 = Union4{Union4_1{222.222}};
        Union2 v18;
        v18 = Union2{Union2_1{2u, v17}};
        Union1 v19;
        v19 = Union1{Union1_1{v18}};
        v6[1l] = Tuple0{2l, v16, v19};
        Union0 v24;
        v24 = Union0{Union0_0{}};
        Union3 v25;
        v25 = Union3{Union3_1{890.876f}};
        Union2 v26;
        v26 = Union2{Union2_0{88, 80u, v25}};
        Union1 v27;
        v27 = Union1{Union1_1{v26}};
        v6[2l] = Tuple0{3l, v24, v27};
        short v32;
        v32 = -2;
        unsigned long long v33;
        v33 = 555555555ull;
        unsigned short v34;
        v34 = 3412u;
        f_0(v0, v32, v33, v6, v34);
        short v35; unsigned long long v36; static_array_list<Tuple0,14l> v37; unsigned short v38;
        Tuple1 tmp4 = f_12(v0);
        v35 = tmp4.v0; v36 = tmp4.v1; v37 = tmp4.v2; v38 = tmp4.v3;
        write_24(v35, v36, v37, v38);
        printf("\n");
        return ;
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
class US0_0(NamedTuple): # None
    tag = 0
class US0_1(NamedTuple): # Some
    v0 : u64
    tag = 1
US0 = Union[US0_0, US0_1]
class US3_0(NamedTuple): # None
    tag = 0
class US3_1(NamedTuple): # Some
    v0 : f32
    tag = 1
US3 = Union[US3_0, US3_1]
class US4_0(NamedTuple): # None
    tag = 0
class US4_1(NamedTuple): # Some
    v0 : f64
    tag = 1
US4 = Union[US4_0, US4_1]
class US2_0(NamedTuple): # Q
    v0 : i8
    v1 : u8
    v2 : US3
    tag = 0
class US2_1(NamedTuple): # W
    v0 : u16
    v1 : US4
    tag = 1
US2 = Union[US2_0, US2_1]
class US1_0(NamedTuple): # None
    tag = 0
class US1_1(NamedTuple): # Some
    v0 : US2
    tag = 1
US1 = Union[US1_0, US1_1]
def method1(v0 : cp.ndarray) -> i32:
    v2 = v0[16:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method2(v0 : i32, v1 : i32) -> bool:
    v2 = v1 < v0
    del v0, v1
    return v2
def method4(v0 : cp.ndarray) -> i32:
    v2 = v0[4:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method5(v0 : cp.ndarray) -> None:
    del v0
    return 
def method6(v0 : cp.ndarray) -> u64:
    v2 = v0[0:].view(cp.uint64)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method8(v0 : cp.ndarray) -> i32:
    v2 = v0[0:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method10(v0 : cp.ndarray) -> f32:
    v2 = v0[0:].view(cp.float32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method9(v0 : cp.ndarray) -> Tuple[i8, u8, US3]:
    v2 = v0[0:].view(cp.int8)
    v3 = v2[0].item()
    del v2
    v5 = v0[1:].view(cp.uint8)
    v6 = v5[0].item()
    del v5
    v7 = method4(v0)
    v9 = v0[8:].view(cp.uint8)
    del v0
    if v7 == 0:
        method5(v9)
        v14 = US3_0()
    elif v7 == 1:
        v12 = method10(v9)
        v14 = US3_1(v12)
    else:
        raise Exception("Invalid tag.")
    del v7, v9
    return v3, v6, v14
def method12(v0 : cp.ndarray) -> f64:
    v2 = v0[0:].view(cp.float64)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method11(v0 : cp.ndarray) -> Tuple[u16, US4]:
    v2 = v0[0:].view(cp.uint16)
    v3 = v2[0].item()
    del v2
    v4 = method4(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method5(v6)
        v11 = US4_0()
    elif v4 == 1:
        v9 = method12(v6)
        v11 = US4_1(v9)
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v11
def method7(v0 : cp.ndarray) -> US2:
    v1 = method8(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5, v6, v7 = method9(v3)
        del v3
        return US2_0(v5, v6, v7)
    elif v1 == 1:
        del v1
        v9, v10 = method11(v3)
        del v3
        return US2_1(v9, v10)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method3(v0 : cp.ndarray) -> Tuple[i32, US0, US1]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method4(v0)
    v6 = v0[8:].view(cp.uint8)
    if v4 == 0:
        method5(v6)
        v11 = US0_0()
    elif v4 == 1:
        v9 = method6(v6)
        v11 = US0_1(v9)
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    v12 = method1(v0)
    v14 = v0[32:].view(cp.uint8)
    del v0
    if v12 == 0:
        method5(v14)
        v19 = US1_0()
    elif v12 == 1:
        v17 = method7(v14)
        v19 = US1_1(v17)
    else:
        raise Exception("Invalid tag.")
    del v12, v14
    return v3, v11, v19
def method0(v0 : cp.ndarray) -> Tuple[i16, u64, static_array_list, u16]:
    v2 = v0[0:].view(cp.int16)
    v3 = v2[0].item()
    del v2
    v5 = v0[8:].view(cp.uint64)
    v6 = v5[0].item()
    del v5
    v8 = static_array_list(14)
    v9 = method1(v0)
    v8.unsafe_set_length(v9)
    del v9
    v10 = v8.length
    v11 = 0
    while method2(v10, v11):
        v13 = u64(v11)
        v14 = v13 * 64
        del v13
        v15 = 32 + v14
        del v14
        v17 = v0[v15:].view(cp.uint8)
        del v15
        v18, v19, v20 = method3(v17)
        del v17
        v8[v11] = (v18, v19, v20)
        del v18, v19, v20
        v11 += 1 
    del v10, v11
    v22 = v0[928:].view(cp.uint16)
    del v0
    v23 = v22[0].item()
    del v22
    return v3, v6, v8, v23
def method14(v0 : i16) -> None:
    print(v0, end="")
    del v0
    return 
def method15(v0 : string) -> None:
    print(v0, end="")
    del v0
    return 
def method17(v0 : u64) -> None:
    print(v0, end="")
    del v0
    return 
def method21(v0 : i32) -> None:
    print(v0, end="")
    del v0
    return 
def method24() -> None:
    v0 = "None"
    return method15(v0)
def method25() -> None:
    v0 = "Some"
    return method15(v0)
def method23(v0 : US0) -> None:
    match v0:
        case US0_0(): # None
            del v0
            return method24()
        case US0_1(v1): # Some
            del v0
            method25()
            v2 = "("
            method15(v2)
            del v2
            method17(v1)
            del v1
            v3 = ")"
            return method15(v3)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method28() -> None:
    v0 = "Q"
    return method15(v0)
def method30(v0 : i8) -> None:
    print(v0, end="")
    del v0
    return 
def method32(v0 : u8) -> None:
    print(v0, end="")
    del v0
    return 
def method34(v0 : f32) -> None:
    print("{:.6f}".format(v0), end="")
    del v0
    return 
def method33(v0 : US3) -> None:
    match v0:
        case US3_0(): # None
            del v0
            return method24()
        case US3_1(v1): # Some
            del v0
            method25()
            v2 = "("
            method15(v2)
            del v2
            method34(v1)
            del v1
            v3 = ")"
            return method15(v3)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method31(v0 : u8, v1 : US3) -> None:
    method32(v0)
    del v0
    v2 = ", "
    method15(v2)
    del v2
    return method33(v1)
def method29(v0 : i8, v1 : u8, v2 : US3) -> None:
    method30(v0)
    del v0
    v3 = ", "
    method15(v3)
    del v3
    return method31(v1, v2)
def method35() -> None:
    v0 = "W"
    return method15(v0)
def method37(v0 : u16) -> None:
    print(v0, end="")
    del v0
    return 
def method39(v0 : f64) -> None:
    print("{:.6f}".format(v0), end="")
    del v0
    return 
def method38(v0 : US4) -> None:
    match v0:
        case US4_0(): # None
            del v0
            return method24()
        case US4_1(v1): # Some
            del v0
            method25()
            v2 = "("
            method15(v2)
            del v2
            method39(v1)
            del v1
            v3 = ")"
            return method15(v3)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method36(v0 : u16, v1 : US4) -> None:
    method37(v0)
    del v0
    v2 = ", "
    method15(v2)
    del v2
    return method38(v1)
def method27(v0 : US2) -> None:
    match v0:
        case US2_0(v1, v2, v3): # Q
            del v0
            method28()
            v4 = "("
            method15(v4)
            del v4
            method29(v1, v2, v3)
            del v1, v2, v3
            v5 = ")"
            return method15(v5)
        case US2_1(v6, v7): # W
            del v0
            method35()
            v8 = "("
            method15(v8)
            del v8
            method36(v6, v7)
            del v6, v7
            v9 = ")"
            return method15(v9)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method26(v0 : US1) -> None:
    match v0:
        case US1_0(): # None
            del v0
            return method24()
        case US1_1(v1): # Some
            del v0
            method25()
            v2 = "("
            method15(v2)
            del v2
            method27(v1)
            del v1
            v3 = ")"
            return method15(v3)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method22(v0 : US0, v1 : US1) -> None:
    method23(v0)
    del v0
    v2 = ", "
    method15(v2)
    del v2
    return method26(v1)
def method20(v0 : i32, v1 : US0, v2 : US1) -> None:
    method21(v0)
    del v0
    v3 = ", "
    method15(v3)
    del v3
    return method22(v1, v2)
def method19(v0 : static_array_list) -> None:
    v1 = "["
    method15(v1)
    del v1
    v2 = v0.length
    v3 = 100 < v2
    if v3:
        v4 = 100
    else:
        v4 = v2
    del v2, v3
    v5 = 0
    while method2(v4, v5):
        v10, v11, v12 = v0[v5]
        method20(v10, v11, v12)
        del v10, v11, v12
        v13 = v5 + 1
        v14 = v0.length
        v15 = v13 < v14
        del v13, v14
        if v15:
            v16 = "; "
            method15(v16)
        else:
            pass
        del v15
        v5 += 1 
    del v4, v5
    v17 = v0.length
    del v0
    v18 = v17 > 100
    del v17
    if v18:
        v19 = "; ..."
        method15(v19)
    else:
        pass
    del v18
    v20 = "]"
    return method15(v20)
def method18(v0 : static_array_list, v1 : u16) -> None:
    method19(v0)
    del v0
    v2 = ", "
    method15(v2)
    del v2
    return method37(v1)
def method16(v0 : u64, v1 : static_array_list, v2 : u16) -> None:
    method17(v0)
    del v0
    v3 = ", "
    method15(v3)
    del v3
    return method18(v1, v2)
def method13(v0 : i16, v1 : u64, v2 : static_array_list, v3 : u16) -> None:
    method14(v0)
    del v0
    v4 = ", "
    method15(v4)
    del v4
    return method16(v1, v2, v3)
def main():
    v0 = cp.empty(944,dtype=cp.uint8)
    v1 = 0
    v2 = raw_module.get_function(f"entry{v1}")
    del v1
    v2.max_dynamic_shared_size_bytes = 0 
    v2((1,),(32,),v0,shared_mem=0)
    del v2
    v3, v4, v5, v6 = method0(v0)
    del v0
    method13(v3, v4, v5, v6)
    del v3, v4, v5, v6
    print()
    return 

if __name__ == '__main__': print(main())
