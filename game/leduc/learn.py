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

struct Union1;
struct Union2;
struct Union0;
struct Tuple0;
__device__ void write_1(const char * v0);
__device__ void write_3();
__device__ void write_5();
__device__ void write_6();
__device__ void write_7();
__device__ void write_4(Union1 v0);
__device__ void write_8();
__device__ void write_10();
__device__ void write_11();
__device__ void write_12();
__device__ void write_9(Union2 v0);
__device__ void write_2(Union0 v0);
__device__ void write_0(static_array_list<Union0,10l> v0);
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
struct Union0_0 { // C1of2
    Union1 v0;
    __device__ Union0_0(Union1 t0) : v0(t0) {}
    __device__ Union0_0() = delete;
};
struct Union0_1 { // C2of2
    Union2 v0;
    __device__ Union0_1(Union2 t0) : v0(t0) {}
    __device__ Union0_1() = delete;
};
struct Union0 {
    union {
        Union0_0 case0; // C1of2
        Union0_1 case1; // C2of2
    };
    unsigned char tag{255};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // C1of2
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // C2of2
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // C1of2
            case 1: new (&this->case1) Union0_1(x.case1); break; // C2of2
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // C1of2
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // C2of2
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // C1of2
                case 1: this->case1 = x.case1; break; // C2of2
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
                case 0: this->case0 = std::move(x.case0); break; // C1of2
                case 1: this->case1 = std::move(x.case1); break; // C2of2
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // C1of2
            case 1: this->case1.~Union0_1(); break; // C2of2
        }
        this->tag = 255;
    }
};
struct Tuple0 {
    int v0;
    int v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(int t0, int t1) : v0(t0), v1(t1) {}
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 61l;
    return v1;
}
__device__ inline bool while_method_1(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 10l;
    return v1;
}
__device__ void write_1(const char * v0){
    const char * v1;
    v1 = "%s";
    printf(v1,v0);
    return ;
}
__device__ void write_3(){
    const char * v0;
    v0 = "C1of2";
    return write_1(v0);
}
__device__ void write_5(){
    const char * v0;
    v0 = "Call";
    return write_1(v0);
}
__device__ void write_6(){
    const char * v0;
    v0 = "Fold";
    return write_1(v0);
}
__device__ void write_7(){
    const char * v0;
    v0 = "Raise";
    return write_1(v0);
}
__device__ void write_4(Union1 v0){
    switch (v0.tag) {
        case 0: { // Call
            return write_5();
            break;
        }
        case 1: { // Fold
            return write_6();
            break;
        }
        case 2: { // Raise
            return write_7();
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void write_8(){
    const char * v0;
    v0 = "C2of2";
    return write_1(v0);
}
__device__ void write_10(){
    const char * v0;
    v0 = "Jack";
    return write_1(v0);
}
__device__ void write_11(){
    const char * v0;
    v0 = "King";
    return write_1(v0);
}
__device__ void write_12(){
    const char * v0;
    v0 = "Queen";
    return write_1(v0);
}
__device__ void write_9(Union2 v0){
    switch (v0.tag) {
        case 0: { // Jack
            return write_10();
            break;
        }
        case 1: { // King
            return write_11();
            break;
        }
        case 2: { // Queen
            return write_12();
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void write_2(Union0 v0){
    switch (v0.tag) {
        case 0: { // C1of2
            Union1 v1 = v0.case0.v0;
            write_3();
            const char * v2;
            v2 = "(";
            write_1(v2);
            write_4(v1);
            const char * v3;
            v3 = ")";
            return write_1(v3);
            break;
        }
        case 1: { // C2of2
            Union2 v4 = v0.case1.v0;
            write_8();
            const char * v5;
            v5 = "(";
            write_1(v5);
            write_9(v4);
            const char * v6;
            v6 = ")";
            return write_1(v6);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void write_0(static_array_list<Union0,10l> v0){
    const char * v1;
    v1 = "[";
    write_1(v1);
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
        write_2(v7);
        int v8;
        v8 = v5 + 1l;
        int v9;
        v9 = v0.length;
        bool v10;
        v10 = v8 < v9;
        if (v10){
            const char * v11;
            v11 = "; ";
            write_1(v11);
        } else {
        }
        v5 += 1l ;
    }
    int v12;
    v12 = v0.length;
    bool v13;
    v13 = v12 > 100l;
    if (v13){
        const char * v14;
        v14 = "; ...";
        write_1(v14);
    } else {
    }
    const char * v15;
    v15 = "]";
    return write_1(v15);
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
        static_array_list<Union0,10l> v5;
        v5 = static_array_list<Union0,10l>{};
        v5.unsafe_set_length(10l);
        Union2 v6;
        v6 = Union2{Union2_1{}};
        Union0 v7;
        v7 = Union0{Union0_1{v6}};
        v5[0l] = v7;
        Union1 v8;
        v8 = Union1{Union1_0{}};
        Union0 v9;
        v9 = Union0{Union0_0{v8}};
        v5[1l] = v9;
        Union1 v10;
        v10 = Union1{Union1_2{}};
        Union0 v11;
        v11 = Union0{Union0_0{v10}};
        v5[2l] = v11;
        Union1 v12;
        v12 = Union1{Union1_2{}};
        Union0 v13;
        v13 = Union0{Union0_0{v12}};
        v5[3l] = v13;
        Union1 v14;
        v14 = Union1{Union1_0{}};
        Union0 v15;
        v15 = Union0{Union0_0{v14}};
        v5[4l] = v15;
        Union2 v16;
        v16 = Union2{Union2_2{}};
        Union0 v17;
        v17 = Union0{Union0_1{v16}};
        v5[5l] = v17;
        Union1 v18;
        v18 = Union1{Union1_0{}};
        Union0 v19;
        v19 = Union0{Union0_0{v18}};
        v5[6l] = v19;
        Union1 v20;
        v20 = Union1{Union1_2{}};
        Union0 v21;
        v21 = Union0{Union0_0{v20}};
        v5[7l] = v21;
        Union1 v22;
        v22 = Union1{Union1_2{}};
        Union0 v23;
        v23 = Union0{Union0_0{v22}};
        v5[8l] = v23;
        Union1 v24;
        v24 = Union1{Union1_0{}};
        Union0 v25;
        v25 = Union0{Union0_0{v24}};
        v5[9l] = v25;
        float v26[61l];
        int v27;
        v27 = 0l;
        while (while_method_0(v27)){
            assert("Tensor range check" && 0 <= v27 && v27 < 61l);
            v26[v27] = 0.0f;
            v27 += 1l ;
        }
        float * v29;
        v29 = v26+0l;
        int v30;
        v30 = v5.length;
        bool v31;
        v31 = v30 == 0l;
        if (v31){
            v29[0l] = 1.0f;
        } else {
        }
        int v32;
        v32 = v5.length;
        int v33;
        v33 = 0l;
        while (while_method_1(v32, v33)){
            Union0 v35;
            v35 = v5[v33];
            int v36;
            v36 = v33 * 6l;
            int v37;
            v37 = 1l + v36;
            switch (v35.tag) {
                case 0: { // C1of2
                    Union1 v38 = v35.case0.v0;
                    switch (v38.tag) {
                        case 0: { // Call
                            v29[v37] = 1.0f;
                            break;
                        }
                        case 1: { // Fold
                            int v39;
                            v39 = v37 + 1l;
                            v29[v39] = 1.0f;
                            break;
                        }
                        case 2: { // Raise
                            int v40;
                            v40 = v37 + 2l;
                            v29[v40] = 1.0f;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                        }
                    }
                    break;
                }
                case 1: { // C2of2
                    Union2 v41 = v35.case1.v0;
                    int v42;
                    v42 = v37 + 3l;
                    switch (v41.tag) {
                        case 0: { // Jack
                            v29[v42] = 1.0f;
                            break;
                        }
                        case 1: { // King
                            int v43;
                            v43 = v42 + 1l;
                            v29[v43] = 1.0f;
                            break;
                        }
                        case 2: { // Queen
                            int v44;
                            v44 = v42 + 2l;
                            v29[v44] = 1.0f;
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
            v33 += 1l ;
        }
        int v45;
        v45 = 0l;
        int v46;
        v46 = 1l;
        int v47;
        v47 = 61l;
        float * v48;
        v48 = v26+v45;
        int v49;
        v49 = 0l;
        static_array_list<Union0,10l> v50;
        v50 = static_array_list<Union0,10l>{};
        v50.unsafe_set_length(10l);
        float v51;
        v51 = v48[v49];
        bool v52;
        v52 = v51 == 0.0f;
        bool v54;
        if (v52){
            v54 = true;
        } else {
            bool v53;
            v53 = v51 == 1.0f;
            v54 = v53;
        }
        bool v55;
        v55 = v54 == false;
        if (v55){
            assert("Unpickle failure. The static array list emptiness flag should be 1 or 0." && v54);
        } else {
        }
        bool v56;
        v56 = v51 == 1.0f;
        int v57;
        v57 = v49 + 1l;
        int v58; int v59;
        Tuple0 tmp0 = Tuple0{0l, 0l};
        v58 = tmp0.v0; v59 = tmp0.v1;
        while (while_method_2(v58)){
            int v61;
            v61 = v58 * 6l;
            int v62;
            v62 = v57 + v61;
            float v63;
            v63 = v48[v62];
            bool v64;
            v64 = v63 == 1.0f;
            bool v66;
            if (v64){
                v66 = true;
            } else {
                bool v65;
                v65 = v63 == 0.0f;
                v66 = v65;
            }
            bool v67;
            v67 = v66 == false;
            if (v67){
                assert("Unpickling failure. The unit type should always be either be 1 or 0." && v66);
            } else {
            }
            int v68;
            if (v64){
                v68 = 1l;
            } else {
                v68 = 0l;
            }
            int v69;
            v69 = v62 + 1l;
            float v70;
            v70 = v48[v69];
            bool v71;
            v71 = v70 == 1.0f;
            bool v73;
            if (v71){
                v73 = true;
            } else {
                bool v72;
                v72 = v70 == 0.0f;
                v73 = v72;
            }
            bool v74;
            v74 = v73 == false;
            if (v74){
                assert("Unpickling failure. The unit type should always be either be 1 or 0." && v73);
            } else {
            }
            int v75;
            if (v71){
                v75 = 1l;
            } else {
                v75 = 0l;
            }
            int v76;
            v76 = v62 + 2l;
            float v77;
            v77 = v48[v76];
            bool v78;
            v78 = v77 == 1.0f;
            bool v80;
            if (v78){
                v80 = true;
            } else {
                bool v79;
                v79 = v77 == 0.0f;
                v80 = v79;
            }
            bool v81;
            v81 = v80 == false;
            if (v81){
                assert("Unpickling failure. The unit type should always be either be 1 or 0." && v80);
            } else {
            }
            int v82;
            if (v78){
                v82 = 1l;
            } else {
                v82 = 0l;
            }
            bool v83;
            v83 = v75 == 1l;
            Union1 v86;
            if (v83){
                v86 = Union1{Union1_1{}};
            } else {
                v86 = Union1{Union1_0{}};
            }
            int v87;
            v87 = v68 + v75;
            bool v88;
            v88 = v82 == 1l;
            Union1 v90;
            if (v88){
                v90 = Union1{Union1_2{}};
            } else {
                v90 = v86;
            }
            int v91;
            v91 = v87 + v82;
            bool v92;
            v92 = v91 == 0l;
            bool v94;
            if (v92){
                v94 = true;
            } else {
                bool v93;
                v93 = v91 == 1l;
                v94 = v93;
            }
            bool v95;
            v95 = v94 == false;
            if (v95){
                assert("Unpickling failure. Only a single case of an union type should be active at most." && v94);
            } else {
            }
            int v96;
            v96 = v62 + 3l;
            float v97;
            v97 = v48[v96];
            bool v98;
            v98 = v97 == 1.0f;
            bool v100;
            if (v98){
                v100 = true;
            } else {
                bool v99;
                v99 = v97 == 0.0f;
                v100 = v99;
            }
            bool v101;
            v101 = v100 == false;
            if (v101){
                assert("Unpickling failure. The unit type should always be either be 1 or 0." && v100);
            } else {
            }
            int v102;
            if (v98){
                v102 = 1l;
            } else {
                v102 = 0l;
            }
            int v103;
            v103 = v96 + 1l;
            float v104;
            v104 = v48[v103];
            bool v105;
            v105 = v104 == 1.0f;
            bool v107;
            if (v105){
                v107 = true;
            } else {
                bool v106;
                v106 = v104 == 0.0f;
                v107 = v106;
            }
            bool v108;
            v108 = v107 == false;
            if (v108){
                assert("Unpickling failure. The unit type should always be either be 1 or 0." && v107);
            } else {
            }
            int v109;
            if (v105){
                v109 = 1l;
            } else {
                v109 = 0l;
            }
            int v110;
            v110 = v96 + 2l;
            float v111;
            v111 = v48[v110];
            bool v112;
            v112 = v111 == 1.0f;
            bool v114;
            if (v112){
                v114 = true;
            } else {
                bool v113;
                v113 = v111 == 0.0f;
                v114 = v113;
            }
            bool v115;
            v115 = v114 == false;
            if (v115){
                assert("Unpickling failure. The unit type should always be either be 1 or 0." && v114);
            } else {
            }
            int v116;
            if (v112){
                v116 = 1l;
            } else {
                v116 = 0l;
            }
            bool v117;
            v117 = v109 == 1l;
            Union2 v120;
            if (v117){
                v120 = Union2{Union2_1{}};
            } else {
                v120 = Union2{Union2_0{}};
            }
            int v121;
            v121 = v102 + v109;
            bool v122;
            v122 = v116 == 1l;
            Union2 v124;
            if (v122){
                v124 = Union2{Union2_2{}};
            } else {
                v124 = v120;
            }
            int v125;
            v125 = v121 + v116;
            bool v126;
            v126 = v125 == 0l;
            bool v128;
            if (v126){
                v128 = true;
            } else {
                bool v127;
                v127 = v125 == 1l;
                v128 = v127;
            }
            bool v129;
            v129 = v128 == false;
            if (v129){
                assert("Unpickling failure. Only a single case of an union type should be active at most." && v128);
            } else {
            }
            bool v130;
            v130 = v125 == 1l;
            Union0 v133;
            if (v130){
                v133 = Union0{Union0_1{v124}};
            } else {
                v133 = Union0{Union0_0{v90}};
            }
            int v134;
            v134 = v91 + v125;
            bool v135;
            v135 = v134 == 0l;
            bool v137;
            if (v135){
                v137 = true;
            } else {
                bool v136;
                v136 = v134 == 1l;
                v137 = v136;
            }
            bool v138;
            v138 = v137 == false;
            if (v138){
                assert("Unpickling failure. Only a single case of an union type should be active at most." && v137);
            } else {
            }
            bool v139;
            v139 = v58 == v59;
            int v144;
            if (v139){
                bool v140;
                v140 = v134 == 1l;
                if (v140){
                    v50[v58] = v133;
                } else {
                }
                int v141;
                v141 = v59 + v134;
                v144 = v141;
            } else {
                bool v142;
                v142 = v59 == 0l;
                bool v143;
                v143 = v142 == false;
                if (v143){
                    assert("Unpickle failure. Expected an inactive subsequence in the static array list unpickler." && v142);
                } else {
                }
                v144 = v59;
            }
            v59 = v144;
            v58 += 1l ;
        }
        if (v56){
            bool v145;
            v145 = v59 == 0l;
            bool v146;
            v146 = v145 == false;
            if (v146){
                assert("Unpickle failure. Empty static array lists should not have active elements." && v145);
            } else {
            }
        } else {
        }
        v50.unsafe_set_length(v59);
        int v147;
        if (v56){
            v147 = 1l;
        } else {
            v147 = 0l;
        }
        int v148;
        v148 = v147 + v59;
        bool v149;
        v149 = 1l < v148;
        int v150;
        if (v149){
            v150 = 1l;
        } else {
            v150 = v148;
        }
        bool v151;
        v151 = v150 == 1l;
        bool v152;
        v152 = v151 == false;
        if (v152){
            assert("Invalid format detected during deserialization." && v151);
        } else {
        }
        int v153;
        v153 = v5.length;
        int v154;
        v154 = v50.length;
        bool v155;
        v155 = v153 == v154;
        bool v171;
        if (v155){
            bool v156;
            v156 = true;
            int v157;
            v157 = v50.length;
            int v158;
            v158 = 0l;
            while (while_method_1(v157, v158)){
                Union0 v160;
                v160 = v5[v158];
                Union0 v161;
                v161 = v50[v158];
                bool v168;
                switch (v160.tag == v161.tag ? v160.tag : 255) {
                    case 0: { // C1of2
                        Union1 v162 = v160.case0.v0;
                        Union1 v163 = v161.case0.v0;
                        switch (v162.tag == v163.tag ? v162.tag : 255) {
                            case 0: { // Call
                                v168 = true;
                                break;
                            }
                            case 1: { // Fold
                                v168 = true;
                                break;
                            }
                            case 2: { // Raise
                                v168 = true;
                                break;
                            }
                            default: {
                                v168 = false;
                            }
                        }
                        break;
                    }
                    case 1: { // C2of2
                        Union2 v165 = v160.case1.v0;
                        Union2 v166 = v161.case1.v0;
                        switch (v165.tag == v166.tag ? v165.tag : 255) {
                            case 0: { // Jack
                                v168 = true;
                                break;
                            }
                            case 1: { // King
                                v168 = true;
                                break;
                            }
                            case 2: { // Queen
                                v168 = true;
                                break;
                            }
                            default: {
                                v168 = false;
                            }
                        }
                        break;
                    }
                    default: {
                        v168 = false;
                    }
                }
                bool v169;
                v169 = v168 != true;
                if (v169){
                    bool v170;
                    v170 = false;
                    v156 = v170;
                    break;
                } else {
                }
                v158 += 1l ;
            }
            v171 = v156;
        } else {
            v171 = false;
        }
        bool v172;
        v172 = v171 == false;
        if (v172){
            assert("The round trip has to preserve equality with the original input." && v171);
        } else {
        }
        write_0(v50);
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
