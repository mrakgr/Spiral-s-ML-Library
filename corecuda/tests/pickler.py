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
struct Union1;
struct Tuple0;
struct Tuple1;
__device__ Tuple0 method_0(float * v0, int v1, int v2);
struct Tuple2;
__device__ void write_2(char v0);
__device__ void write_3();
__device__ void write_4(const char * v0);
__device__ void write_7();
__device__ void write_8();
__device__ void write_9();
__device__ void write_6(Union0 v0);
__device__ void write_5(static_array_list<Union0,5l> v0);
__device__ void write_11();
__device__ void write_12();
__device__ void write_13();
__device__ void write_10(Union1 v0);
__device__ void write_14(int v0);
__device__ void write_1(static_array_list<Union0,5l> v0, Union1 v1, int v2, int v3);
struct Union0_0 { // Call
};
struct Union0_1 { // Fold
};
struct Union0_2 { // Raise
};
struct Union0 {
    union {
        Union0_0 case0; // Call
        Union0_1 case1; // Fold
        Union0_2 case2; // Raise
    };
    unsigned char tag{255};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // Call
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // Fold
    __device__ Union0(Union0_2 t) : tag(2), case2(t) {} // Raise
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // Call
            case 1: new (&this->case1) Union0_1(x.case1); break; // Fold
            case 2: new (&this->case2) Union0_2(x.case2); break; // Raise
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // Call
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // Fold
            case 2: new (&this->case2) Union0_2(std::move(x.case2)); break; // Raise
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Call
                case 1: this->case1 = x.case1; break; // Fold
                case 2: this->case2 = x.case2; break; // Raise
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
                case 0: this->case0 = std::move(x.case0); break; // Call
                case 1: this->case1 = std::move(x.case1); break; // Fold
                case 2: this->case2 = std::move(x.case2); break; // Raise
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // Call
            case 1: this->case1.~Union0_1(); break; // Fold
            case 2: this->case2.~Union0_2(); break; // Raise
        }
        this->tag = 255;
    }
};
struct Union1_0 { // Jack
};
struct Union1_1 { // King
};
struct Union1_2 { // Queen
};
struct Union1 {
    union {
        Union1_0 case0; // Jack
        Union1_1 case1; // King
        Union1_2 case2; // Queen
    };
    unsigned char tag{255};
    __device__ Union1() {}
    __device__ Union1(Union1_0 t) : tag(0), case0(t) {} // Jack
    __device__ Union1(Union1_1 t) : tag(1), case1(t) {} // King
    __device__ Union1(Union1_2 t) : tag(2), case2(t) {} // Queen
    __device__ Union1(Union1 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(x.case0); break; // Jack
            case 1: new (&this->case1) Union1_1(x.case1); break; // King
            case 2: new (&this->case2) Union1_2(x.case2); break; // Queen
        }
    }
    __device__ Union1(Union1 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // Jack
            case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // King
            case 2: new (&this->case2) Union1_2(std::move(x.case2)); break; // Queen
        }
    }
    __device__ Union1 & operator=(Union1 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Jack
                case 1: this->case1 = x.case1; break; // King
                case 2: this->case2 = x.case2; break; // Queen
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
                case 0: this->case0 = std::move(x.case0); break; // Jack
                case 1: this->case1 = std::move(x.case1); break; // King
                case 2: this->case2 = std::move(x.case2); break; // Queen
            }
        } else {
            this->~Union1();
            new (this) Union1{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union1() {
        switch(this->tag){
            case 0: this->case0.~Union1_0(); break; // Jack
            case 1: this->case1.~Union1_1(); break; // King
            case 2: this->case2.~Union1_2(); break; // Queen
        }
        this->tag = 255;
    }
};
struct Tuple0 {
    unsigned int v0;
    int v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(unsigned int t0, int t1) : v0(t0), v1(t1) {}
};
struct Tuple1 {
    int v0;
    int v1;
    int v2;
    __device__ Tuple1() = default;
    __device__ Tuple1(int t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple2 {
    int v0;
    int v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(int t0, int t1) : v0(t0), v1(t1) {}
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 39l;
    return v1;
}
__device__ inline bool while_method_1(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ Tuple0 method_0(float * v0, int v1, int v2){
    int v3; int v4; int v5;
    Tuple1 tmp0 = Tuple1{v2, 0l, 0l};
    v3 = tmp0.v0; v4 = tmp0.v1; v5 = tmp0.v2;
    while (while_method_1(v1, v3)){
        float v7;
        v7 = v0[v3];
        bool v8;
        v8 = v7 == 1.0f;
        bool v10;
        if (v8){
            v10 = true;
        } else {
            bool v9;
            v9 = v7 == 0.0f;
            v10 = v9;
        }
        bool v11;
        v11 = v10 == false;
        if (v11){
            assert("Unpickling failure. The int type should always be either be 1 or 0." && v10);
        } else {
        }
        bool v13;
        v13 = v7 == 0.0f;
        int v15; int v16;
        if (v13){
            v15 = v4; v16 = v5;
        } else {
            int v14;
            v14 = v5 + 1l;
            v15 = v3; v16 = v14;
        }
        v4 = v15;
        v5 = v16;
        v3 += 1l ;
    }
    bool v17;
    v17 = v5 == 0l;
    bool v19;
    if (v17){
        v19 = true;
    } else {
        bool v18;
        v18 = v5 == 1l;
        v19 = v18;
    }
    bool v20;
    v20 = v19 == false;
    if (v20){
        assert("Unpickling failure. Too many active indices in the one-hot vector." && v19);
    } else {
    }
    int v22;
    v22 = v4 - v2;
    unsigned int v23;
    v23 = (unsigned int)v22;
    return Tuple0{v23, v5};
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 5l;
    return v1;
}
__device__ void write_2(char v0){
    const char * v1;
    v1 = "%c";
    printf(v1,v0);
    return ;
}
__device__ void write_3(){
    return ;
}
__device__ void write_4(const char * v0){
    const char * v1;
    v1 = "%s";
    printf(v1,v0);
    return ;
}
__device__ void write_7(){
    const char * v0;
    v0 = "Call";
    return write_4(v0);
}
__device__ void write_8(){
    const char * v0;
    v0 = "Fold";
    return write_4(v0);
}
__device__ void write_9(){
    const char * v0;
    v0 = "Raise";
    return write_4(v0);
}
__device__ void write_6(Union0 v0){
    switch (v0.tag) {
        case 0: { // Call
            return write_7();
            break;
        }
        case 1: { // Fold
            return write_8();
            break;
        }
        case 2: { // Raise
            return write_9();
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void write_5(static_array_list<Union0,5l> v0){
    const char * v1;
    v1 = "[";
    write_4(v1);
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
    while (while_method_1(v4, v5)){
        Union0 v7;
        v7 = v0[v5];
        write_6(v7);
        int v9;
        v9 = v5 + 1l;
        int v10;
        v10 = v0.length;
        bool v11;
        v11 = v9 < v10;
        if (v11){
            const char * v12;
            v12 = "; ";
            write_4(v12);
        } else {
        }
        v5 += 1l ;
    }
    int v13;
    v13 = v0.length;
    bool v14;
    v14 = v13 > 100l;
    if (v14){
        const char * v15;
        v15 = "; ...";
        write_4(v15);
    } else {
    }
    const char * v16;
    v16 = "]";
    return write_4(v16);
}
__device__ void write_11(){
    const char * v0;
    v0 = "Jack";
    return write_4(v0);
}
__device__ void write_12(){
    const char * v0;
    v0 = "King";
    return write_4(v0);
}
__device__ void write_13(){
    const char * v0;
    v0 = "Queen";
    return write_4(v0);
}
__device__ void write_10(Union1 v0){
    switch (v0.tag) {
        case 0: { // Jack
            return write_11();
            break;
        }
        case 1: { // King
            return write_12();
            break;
        }
        case 2: { // Queen
            return write_13();
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void write_14(int v0){
    const char * v1;
    v1 = "%d";
    printf(v1,v0);
    return ;
}
__device__ void write_1(static_array_list<Union0,5l> v0, Union1 v1, int v2, int v3){
    char v4;
    v4 = '{';
    write_2(v4);
    write_3();
    const char * v5;
    v5 = "action_history";
    write_4(v5);
    const char * v6;
    v6 = " = ";
    write_4(v6);
    write_5(v0);
    const char * v7;
    v7 = "; ";
    write_4(v7);
    const char * v8;
    v8 = "card";
    write_4(v8);
    write_4(v6);
    write_10(v1);
    write_4(v7);
    const char * v9;
    v9 = "pot";
    write_4(v9);
    write_4(v6);
    write_14(v2);
    write_4(v7);
    const char * v10;
    v10 = "stack";
    write_4(v10);
    write_4(v6);
    write_14(v3);
    char v11;
    v11 = '}';
    return write_2(v11);
}
extern "C" __global__ void entry0() {
    int v0;
    v0 = threadIdx.x;
    int v1;
    v1 = blockIdx.x;
    int v2;
    v2 = v1 * 32l;
    int v3;
    v3 = v0 + v2;
    bool v4;
    v4 = v3 == 0l;
    if (v4){
        static_array_list<Union0,5l> v5;
        v5 = static_array_list<Union0,5l>{};
        v5.unsafe_set_length(3l);
        Union0 v7;
        v7 = Union0{Union0_2{}};
        v5[0l] = v7;
        Union0 v9;
        v9 = Union0{Union0_2{}};
        v5[1l] = v9;
        Union0 v11;
        v11 = Union0{Union0_0{}};
        v5[2l] = v11;
        Union1 v13;
        v13 = Union1{Union1_1{}};
        int v14;
        v14 = 8l;
        int v15;
        v15 = 5l;
        float v16[39l];
        int v17;
        v17 = 0l;
        while (while_method_0(v17)){
            assert("Tensor range check" && 0 <= v17 && v17 < 39l);
            v16[v17] = 0.0f;
            v17 += 1l ;
        }
        float * v19;
        v19 = v16+0l;
        unsigned int v21;
        v21 = (unsigned int)v15;
        int v22;
        v22 = (int)v21;
        bool v23;
        v23 = v22 < 10l;
        bool v24;
        v24 = v23 == false;
        if (v24){
            assert("Pickle failure. Int value out of bounds." && v23);
        } else {
        }
        v19[v22] = 1.0f;
        unsigned int v26;
        v26 = (unsigned int)v14;
        int v27;
        v27 = (int)v26;
        bool v28;
        v28 = v27 < 10l;
        bool v29;
        v29 = v28 == false;
        if (v29){
            assert("Pickle failure. Int value out of bounds." && v28);
        } else {
        }
        int v31;
        v31 = 10l + v27;
        v19[v31] = 1.0f;
        switch (v13.tag) {
            case 0: { // Jack
                v19[20l] = 1.0f;
                break;
            }
            case 1: { // King
                v19[21l] = 1.0f;
                break;
            }
            case 2: { // Queen
                v19[22l] = 1.0f;
                break;
            }
            default: {
                assert("Invalid tag." && false);
            }
        }
        int v32;
        v32 = v5.length;
        bool v33;
        v33 = v32 == 0l;
        if (v33){
            v19[23l] = 1.0f;
        } else {
        }
        int v34;
        v34 = v5.length;
        int v35;
        v35 = 0l;
        while (while_method_1(v34, v35)){
            Union0 v37;
            v37 = v5[v35];
            int v39;
            v39 = v35 * 3l;
            int v40;
            v40 = 24l + v39;
            switch (v37.tag) {
                case 0: { // Call
                    v19[v40] = 1.0f;
                    break;
                }
                case 1: { // Fold
                    int v41;
                    v41 = v40 + 1l;
                    v19[v41] = 1.0f;
                    break;
                }
                case 2: { // Raise
                    int v42;
                    v42 = v40 + 2l;
                    v19[v42] = 1.0f;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                }
            }
            v35 += 1l ;
        }
        int v43;
        v43 = 0l;
        int v44;
        v44 = 1l;
        int v45;
        v45 = 39l;
        float * v46;
        v46 = v16+v43;
        int v48;
        v48 = 0l;
        int v49;
        v49 = 10l;
        unsigned int v50; int v51;
        Tuple0 tmp1 = method_0(v46, v49, v48);
        v50 = tmp1.v0; v51 = tmp1.v1;
        int v52;
        v52 = (int)v50;
        int v53;
        v53 = 10l;
        int v54;
        v54 = 20l;
        unsigned int v55; int v56;
        Tuple0 tmp2 = method_0(v46, v54, v53);
        v55 = tmp2.v0; v56 = tmp2.v1;
        int v57;
        v57 = (int)v55;
        float v58;
        v58 = v46[20l];
        bool v59;
        v59 = v58 == 1.0f;
        bool v61;
        if (v59){
            v61 = true;
        } else {
            bool v60;
            v60 = v58 == 0.0f;
            v61 = v60;
        }
        bool v62;
        v62 = v61 == false;
        if (v62){
            assert("Unpickling failure. The unit type should always be either be 1 or 0." && v61);
        } else {
        }
        int v64;
        if (v59){
            v64 = 1l;
        } else {
            v64 = 0l;
        }
        float v65;
        v65 = v46[21l];
        bool v66;
        v66 = v65 == 1.0f;
        bool v68;
        if (v66){
            v68 = true;
        } else {
            bool v67;
            v67 = v65 == 0.0f;
            v68 = v67;
        }
        bool v69;
        v69 = v68 == false;
        if (v69){
            assert("Unpickling failure. The unit type should always be either be 1 or 0." && v68);
        } else {
        }
        int v71;
        if (v66){
            v71 = 1l;
        } else {
            v71 = 0l;
        }
        float v72;
        v72 = v46[22l];
        bool v73;
        v73 = v72 == 1.0f;
        bool v75;
        if (v73){
            v75 = true;
        } else {
            bool v74;
            v74 = v72 == 0.0f;
            v75 = v74;
        }
        bool v76;
        v76 = v75 == false;
        if (v76){
            assert("Unpickling failure. The unit type should always be either be 1 or 0." && v75);
        } else {
        }
        int v78;
        if (v73){
            v78 = 1l;
        } else {
            v78 = 0l;
        }
        bool v79;
        v79 = v71 == 1l;
        Union1 v82;
        if (v79){
            v82 = Union1{Union1_1{}};
        } else {
            v82 = Union1{Union1_0{}};
        }
        int v83;
        v83 = v64 + v71;
        bool v84;
        v84 = v78 == 1l;
        Union1 v86;
        if (v84){
            v86 = Union1{Union1_2{}};
        } else {
            v86 = v82;
        }
        int v87;
        v87 = v83 + v78;
        bool v88;
        v88 = v87 == 0l;
        bool v90;
        if (v88){
            v90 = true;
        } else {
            bool v89;
            v89 = v87 == 1l;
            v90 = v89;
        }
        bool v91;
        v91 = v90 == false;
        if (v91){
            assert("Unpickling failure. Only a single case of an union type should be active at most." && v90);
        } else {
        }
        int v93;
        v93 = 23l;
        static_array_list<Union0,5l> v94;
        v94 = static_array_list<Union0,5l>{};
        v94.unsafe_set_length(5l);
        float v96;
        v96 = v46[v93];
        bool v97;
        v97 = v96 == 0.0f;
        bool v99;
        if (v97){
            v99 = true;
        } else {
            bool v98;
            v98 = v96 == 1.0f;
            v99 = v98;
        }
        bool v100;
        v100 = v99 == false;
        if (v100){
            assert("Unpickle failure. The static array list emptiness flag should be 1 or 0." && v99);
        } else {
        }
        bool v102;
        v102 = v96 == 1.0f;
        int v103;
        v103 = v93 + 1l;
        int v104; int v105;
        Tuple2 tmp3 = Tuple2{0l, 0l};
        v104 = tmp3.v0; v105 = tmp3.v1;
        while (while_method_2(v104)){
            int v107;
            v107 = v104 * 3l;
            int v108;
            v108 = v103 + v107;
            float v109;
            v109 = v46[v108];
            bool v110;
            v110 = v109 == 1.0f;
            bool v112;
            if (v110){
                v112 = true;
            } else {
                bool v111;
                v111 = v109 == 0.0f;
                v112 = v111;
            }
            bool v113;
            v113 = v112 == false;
            if (v113){
                assert("Unpickling failure. The unit type should always be either be 1 or 0." && v112);
            } else {
            }
            int v115;
            if (v110){
                v115 = 1l;
            } else {
                v115 = 0l;
            }
            int v116;
            v116 = v108 + 1l;
            float v117;
            v117 = v46[v116];
            bool v118;
            v118 = v117 == 1.0f;
            bool v120;
            if (v118){
                v120 = true;
            } else {
                bool v119;
                v119 = v117 == 0.0f;
                v120 = v119;
            }
            bool v121;
            v121 = v120 == false;
            if (v121){
                assert("Unpickling failure. The unit type should always be either be 1 or 0." && v120);
            } else {
            }
            int v123;
            if (v118){
                v123 = 1l;
            } else {
                v123 = 0l;
            }
            int v124;
            v124 = v108 + 2l;
            float v125;
            v125 = v46[v124];
            bool v126;
            v126 = v125 == 1.0f;
            bool v128;
            if (v126){
                v128 = true;
            } else {
                bool v127;
                v127 = v125 == 0.0f;
                v128 = v127;
            }
            bool v129;
            v129 = v128 == false;
            if (v129){
                assert("Unpickling failure. The unit type should always be either be 1 or 0." && v128);
            } else {
            }
            int v131;
            if (v126){
                v131 = 1l;
            } else {
                v131 = 0l;
            }
            bool v132;
            v132 = v123 == 1l;
            Union0 v135;
            if (v132){
                v135 = Union0{Union0_1{}};
            } else {
                v135 = Union0{Union0_0{}};
            }
            int v136;
            v136 = v115 + v123;
            bool v137;
            v137 = v131 == 1l;
            Union0 v139;
            if (v137){
                v139 = Union0{Union0_2{}};
            } else {
                v139 = v135;
            }
            int v140;
            v140 = v136 + v131;
            bool v141;
            v141 = v140 == 0l;
            bool v143;
            if (v141){
                v143 = true;
            } else {
                bool v142;
                v142 = v140 == 1l;
                v143 = v142;
            }
            bool v144;
            v144 = v143 == false;
            if (v144){
                assert("Unpickling failure. Only a single case of an union type should be active at most." && v143);
            } else {
            }
            bool v146;
            v146 = v104 == v105;
            int v151;
            if (v146){
                bool v147;
                v147 = v140 == 1l;
                if (v147){
                    v94[v104] = v139;
                } else {
                }
                int v148;
                v148 = v105 + v140;
                v151 = v148;
            } else {
                bool v149;
                v149 = v141 == false;
                if (v149){
                    assert("Unpickle failure. Expected an inactive subsequence in the static array list unpickler." && v141);
                } else {
                }
                v151 = v105;
            }
            v105 = v151;
            v104 += 1l ;
        }
        if (v102){
            bool v152;
            v152 = v105 == 0l;
            bool v153;
            v153 = v152 == false;
            if (v153){
                assert("Unpickle failure. Empty static array lists should not have active elements." && v152);
            } else {
            }
        } else {
        }
        v94.unsafe_set_length(v105);
        int v155;
        if (v102){
            v155 = 1l;
        } else {
            v155 = 0l;
        }
        int v156;
        v156 = v155 + v105;
        bool v157;
        v157 = 1l < v156;
        int v158;
        if (v157){
            v158 = 1l;
        } else {
            v158 = v156;
        }
        int v159;
        v159 = v87 + v158;
        bool v160;
        v160 = v159 == 0l;
        bool v162;
        if (v160){
            v162 = true;
        } else {
            bool v161;
            v161 = v159 == 2l;
            v162 = v161;
        }
        bool v163;
        v163 = v162 == false;
        if (v163){
            assert("Unpickling failure. Two sides of a pair should either all be active or inactive." && v162);
        } else {
        }
        int v165;
        v165 = v159 / 2l;
        int v166;
        v166 = v56 + v165;
        bool v167;
        v167 = v166 == 0l;
        bool v169;
        if (v167){
            v169 = true;
        } else {
            bool v168;
            v168 = v166 == 2l;
            v169 = v168;
        }
        bool v170;
        v170 = v169 == false;
        if (v170){
            assert("Unpickling failure. Two sides of a pair should either all be active or inactive." && v169);
        } else {
        }
        int v172;
        v172 = v166 / 2l;
        int v173;
        v173 = v51 + v172;
        bool v174;
        v174 = v173 == 0l;
        bool v176;
        if (v174){
            v176 = true;
        } else {
            bool v175;
            v175 = v173 == 2l;
            v176 = v175;
        }
        bool v177;
        v177 = v176 == false;
        if (v177){
            assert("Unpickling failure. Two sides of a pair should either all be active or inactive." && v176);
        } else {
        }
        int v179;
        v179 = v173 / 2l;
        bool v180;
        v180 = v179 == 1l;
        bool v181;
        v181 = v180 == false;
        if (v181){
            assert("Invalid format detected during deserialization." && v180);
        } else {
        }
        write_1(v94, v86, v57, v52);
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
def main():
    v0 = 0
    v1 = raw_module.get_function(f"entry{v0}")
    del v0
    v1.max_dynamic_shared_size_bytes = 0 
    v1((1,),(32,),(),shared_mem=0)
    del v1
    return 

if __name__ == '__main__': print(main())
