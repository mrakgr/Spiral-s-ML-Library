kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <mma.h>
using namespace nvcuda;
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
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
struct Union3;
struct Tuple0;
__device__ void method_1(float * v0, int v1, float * v2, int v3, float * v4, int v5);
__device__ void method_2(float * v0, float * v1);
__device__ void method_3(float * v0, float * v1);
struct Tuple1;
struct Tuple2;
struct Tuple3;
struct Tuple4;
__device__ void method_4(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5);
__device__ unsigned int loop_6(unsigned int v0, curandStatePhilox4_32_10_t & v1);
__device__ int int_range_5(int v0, int v1, curandStatePhilox4_32_10_t & v2);
__device__ Tuple0 noinline_run_0(unsigned char * v0, unsigned char * v1, Union3 v2);
__device__ void method_7(Union1 v0);
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
struct Union3_0 { // None
};
struct Union3_1 { // Some
    static_array_list<Union0,10l> v0;
    __device__ Union3_1(static_array_list<Union0,10l> t0) : v0(t0) {}
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
struct Tuple0 {
    Union1 v0;
    float * v1;
    int v2;
    int v3;
    int v4;
    float v5;
    __device__ Tuple0() = default;
    __device__ Tuple0(Union1 t0, float * t1, int t2, int t3, int t4, float t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Closure0 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Closure1 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple1 {
    int v0;
    float v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Closure2 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple2 {
    float v0;
    bool v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(float t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ Tuple2 operator()(Tuple2 tup0, Tuple2 tup1){
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
                return Tuple2{v5, true};
            } else {
                return Tuple2{v0, v1};
            }
        } else {
            if (v3){
                return Tuple2{v2, v3};
            } else {
                return Tuple2{v0, v1};
            }
        }
    }
};
struct Tuple3 {
    float v0;
    int v1;
    __device__ Tuple3() = default;
    __device__ Tuple3(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
    __device__ Tuple3 operator()(Tuple3 tup0, Tuple3 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple3{v0, v1};
        } else {
            return Tuple3{v2, v3};
        }
    }
};
struct Tuple4 {
    int v0;
    bool v1;
    __device__ Tuple4() = default;
    __device__ Tuple4(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure5 {
    __device__ Tuple4 operator()(Tuple4 tup0, Tuple4 tup1){
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
                return Tuple4{v5, true};
            } else {
                return Tuple4{v0, v1};
            }
        } else {
            if (v3){
                return Tuple4{v2, v3};
            } else {
                return Tuple4{v0, v1};
            }
        }
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 4096l;
    return v1;
}
__device__ inline bool while_method_1(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 8l;
    return v1;
}
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 16l;
    return v1;
}
__device__ void method_1(float * v0, int v1, float * v2, int v3, float * v4, int v5){
    unsigned int v6;
    v6 = 0ul;
    asm("mov.u32 %0, %dynamic_smem_size;" : "=r"(v6));
    unsigned long long v7;
    v7 = (unsigned long long)v6;
    bool v8;
    v8 = 1536ull <= v7;
    bool v9;
    v9 = v8 == false;
    if (v9){
        assert("The shared memory used in the matmult node is lower than the allocated amount." && v8);
    } else {
    }
    extern __shared__ unsigned char v11[];
    float * v12;
    v12 = reinterpret_cast<float *>(&v11[0ull]);
    float * v14;
    v14 = reinterpret_cast<float *>(&v11[768ull]);
    float * v16;
    v16 = reinterpret_cast<float *>(&v11[0ull]);
    int v18;
    v18 = threadIdx.x;
    int v19;
    v19 = v18 / 32l;
    bool v20;
    v20 = 0l <= v19;
    bool v21;
    v21 = v20 == false;
    if (v21){
        assert("The index needs to be zero or positive." && v20);
    } else {
    }
    int v23;
    v23 = v19 % 1l;
    bool v24;
    v24 = v19 < 1l;
    bool v25;
    v25 = v24 == false;
    if (v25){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v24);
    } else {
    }
    assert("Tensor range check" && 0 <= v19 && v19 < 1l);
    assert("Tensor range check" && 0 <= v23 && v23 < 1l);
    int v27;
    v27 = 16l * v23;
    int v28;
    v28 = 384l * v19;
    int v29;
    v29 = v28 + v27;
    float * v30;
    v30 = v16+v29;
    assert("Tensor range check" && 0 <= v19 && v19 < 1l);
    int v32;
    v32 = 192l * v19;
    int v33;
    v33 = threadIdx.x;
    int v34;
    v34 = v33 % 32l;
    bool v35;
    v35 = 0l <= v34;
    bool v36;
    v36 = v35 == false;
    if (v36){
        assert("The index needs to be zero or positive." && v35);
    } else {
    }
    int v38;
    v38 = v34 % 4l;
    int v39;
    v39 = v34 / 4l;
    bool v40;
    v40 = v39 < 8l;
    bool v41;
    v41 = v40 == false;
    if (v41){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v40);
    } else {
    }
    assert("Tensor range check" && 0 <= v39 && v39 < 8l);
    assert("Tensor range check" && 0 <= v38 && v38 < 4l);
    int v43;
    v43 = v38 + v32;
    int v44;
    v44 = 12l * v39;
    int v45;
    v45 = v44 + v43;
    float * v46;
    v46 = v12+v45;
    assert("Tensor range check" && 0 <= v23 && v23 < 1l);
    int v48;
    v48 = 192l * v23;
    int v49;
    v49 = threadIdx.x;
    int v50;
    v50 = v49 % 32l;
    bool v51;
    v51 = 0l <= v50;
    bool v52;
    v52 = v51 == false;
    if (v52){
        assert("The index needs to be zero or positive." && v51);
    } else {
    }
    int v54;
    v54 = v50 % 4l;
    int v55;
    v55 = v50 / 4l;
    bool v56;
    v56 = v55 < 8l;
    bool v57;
    v57 = v56 == false;
    if (v57){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v56);
    } else {
    }
    assert("Tensor range check" && 0 <= v55 && v55 < 8l);
    assert("Tensor range check" && 0 <= v54 && v54 < 4l);
    int v59;
    v59 = v54 + v48;
    int v60;
    v60 = 12l * v55;
    int v61;
    v61 = v60 + v59;
    float * v62;
    v62 = v14+v61;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v64[1l];
    int v65;
    v65 = 0l;
    while (while_method_3(v65)){
        int v67;
        v67 = 0l;
        while (while_method_4(v67)){
            assert("Tensor range check" && 0 <= v65 && v65 < 2l);
            assert("Tensor range check" && 0 <= v67 && v67 < 8l);
            int v69;
            v69 = 16l * v67;
            int v70;
            v70 = v69 + v3;
            int v71;
            v71 = 2048l * v65;
            int v72;
            v72 = v71 + v70;
            float * v73;
            v73 = v2+v72;
            // Pushing the loop unrolling to: 0
            int v75;
            v75 = 0l;
            #pragma unroll
            while (while_method_5(v75)){
                int v77;
                v77 = 0l;
                #pragma unroll
                while (while_method_5(v77)){
                    assert("Tensor range check" && 0 <= v75 && v75 < 1l);
                    assert("Tensor range check" && 0 <= v77 && v77 < 1l);
                    int v79;
                    v79 = v75 + v77;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v80 = v64[v79];
                    wmma::fill_fragment(v80, 0.0f);
                    v77 += 1l ;
                }
                v75 += 1l ;
            }
            int v81;
            v81 = 0l;
            #pragma unroll
            while (while_method_6(v81)){
                assert("Tensor range check" && 0 <= v65 && v65 < 2l);
                int v83;
                v83 = v71 + v5;
                assert("Tensor range check" && 0 <= v81 && v81 < 16l);
                int v84;
                v84 = 8l * v81;
                int v85;
                v85 = v84 + v83;
                float * v86;
                v86 = v4+v85;
                assert("Tensor range check" && 0 <= v67 && v67 < 8l);
                int v88;
                v88 = 2048l * v67;
                int v89;
                v89 = v88 + v1;
                assert("Tensor range check" && 0 <= v81 && v81 < 16l);
                int v90;
                v90 = v84 + v89;
                float * v91;
                v91 = v0+v90;
                int v93;
                v93 = threadIdx.x;
                bool v94;
                v94 = 0l <= v93;
                bool v95;
                v95 = v94 == false;
                if (v95){
                    assert("The index needs to be zero or positive." && v94);
                } else {
                }
                int v97;
                v97 = v93 % 2l;
                int v98;
                v98 = v93 / 2l;
                bool v99;
                v99 = v98 < 16l;
                bool v100;
                v100 = v99 == false;
                if (v100){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v99);
                } else {
                }
                assert("Tensor range check" && 0 <= v98 && v98 < 16l);
                assert("Tensor range check" && 0 <= v97 && v97 < 2l);
                int v102;
                v102 = 4l * v97;
                int v103;
                v103 = 12l * v98;
                int v104;
                v104 = v103 + v102;
                int v105;
                v105 = 128l * v98;
                int v106;
                v106 = v105 + v102;
                float * v107;
                v107 = v14+v104;
                float * v109;
                v109 = v91+v106;
                int v111;
                v111 = 0l;
                #pragma unroll
                while (while_method_5(v111)){
                    int v113;
                    v113 = 0l;
                    #pragma unroll
                    while (while_method_5(v113)){
                        assert("Tensor range check" && 0 <= v111 && v111 < 1l);
                        assert("Tensor range check" && 0 <= v113 && v113 < 1l);
                        int v115;
                        v115 = 8l * v113;
                        int v116;
                        v116 = 192l * v111;
                        int v117;
                        v117 = v116 + v115;
                        int v118;
                        v118 = 2048l * v111;
                        int v119;
                        v119 = v118 + v115;
                        float v120[4l];
                        int v121;
                        v121 = 0l;
                        #pragma unroll
                        while (while_method_2(v121)){
                            assert("Tensor range check" && 0 <= v121 && v121 < 4l);
                            int v123;
                            v123 = v121 + v119;
                            float v124;
                            v124 = v109[v123];
                            float v125;
                            v125 = wmma::__float_to_tf32(v124);
                            assert("Tensor range check" && 0 <= v121 && v121 < 4l);
                            v120[v121] = v125;
                            v121 += 1l ;
                        }
                        int4* v126;
                        v126 = reinterpret_cast<int4*>(v120 + 0l);
                        int4* v127;
                        v127 = reinterpret_cast<int4*>(v107 + v117);
                        assert("Pointer alignment check" && (unsigned long long)(v126) % 4l == 0 && (unsigned long long)(v127) % 4l == 0);
                        *v127 = *v126;
                        v113 += 1l ;
                    }
                    v111 += 1l ;
                }
                int v128;
                v128 = threadIdx.x;
                bool v129;
                v129 = 0l <= v128;
                bool v130;
                v130 = v129 == false;
                if (v130){
                    assert("The index needs to be zero or positive." && v129);
                } else {
                }
                int v132;
                v132 = v128 % 2l;
                int v133;
                v133 = v128 / 2l;
                bool v134;
                v134 = v133 < 16l;
                bool v135;
                v135 = v134 == false;
                if (v135){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v134);
                } else {
                }
                assert("Tensor range check" && 0 <= v133 && v133 < 16l);
                assert("Tensor range check" && 0 <= v132 && v132 < 2l);
                int v137;
                v137 = 4l * v132;
                int v138;
                v138 = 12l * v133;
                int v139;
                v139 = v138 + v137;
                int v140;
                v140 = 128l * v133;
                int v141;
                v141 = v140 + v137;
                float * v142;
                v142 = v12+v139;
                float * v144;
                v144 = v86+v141;
                int v146;
                v146 = 0l;
                #pragma unroll
                while (while_method_5(v146)){
                    int v148;
                    v148 = 0l;
                    #pragma unroll
                    while (while_method_5(v148)){
                        assert("Tensor range check" && 0 <= v146 && v146 < 1l);
                        assert("Tensor range check" && 0 <= v148 && v148 < 1l);
                        int v150;
                        v150 = 8l * v148;
                        int v151;
                        v151 = 192l * v146;
                        int v152;
                        v152 = v151 + v150;
                        int v153;
                        v153 = 2048l * v146;
                        int v154;
                        v154 = v153 + v150;
                        float v155[4l];
                        int v156;
                        v156 = 0l;
                        #pragma unroll
                        while (while_method_2(v156)){
                            assert("Tensor range check" && 0 <= v156 && v156 < 4l);
                            int v158;
                            v158 = v156 + v154;
                            float v159;
                            v159 = v144[v158];
                            float v160;
                            v160 = wmma::__float_to_tf32(v159);
                            assert("Tensor range check" && 0 <= v156 && v156 < 4l);
                            v155[v156] = v160;
                            v156 += 1l ;
                        }
                        int4* v161;
                        v161 = reinterpret_cast<int4*>(v155 + 0l);
                        int4* v162;
                        v162 = reinterpret_cast<int4*>(v142 + v152);
                        assert("Pointer alignment check" && (unsigned long long)(v161) % 4l == 0 && (unsigned long long)(v162) % 4l == 0);
                        *v162 = *v161;
                        v148 += 1l ;
                    }
                    v146 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v163[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v164[1l];
                int v165;
                v165 = 0l;
                #pragma unroll
                while (while_method_5(v165)){
                    int v167;
                    v167 = 0l;
                    #pragma unroll
                    while (while_method_5(v167)){
                        assert("Tensor range check" && 0 <= v165 && v165 < 1l);
                        assert("Tensor range check" && 0 <= v167 && v167 < 1l);
                        int v169;
                        v169 = v165 + v167;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v170 = v163[v169];
                        assert("Tensor range check" && 0 <= v165 && v165 < 1l);
                        int v171;
                        v171 = 192l * v165;
                        assert("Tensor range check" && 0 <= v167 && v167 < 1l);
                        int v172;
                        v172 = 8l * v167;
                        int v173;
                        v173 = v172 + v171;
                        int v174;
                        v174 = 0l;
                        #pragma unroll
                        while (while_method_3(v174)){
                            int v176;
                            v176 = 0l;
                            #pragma unroll
                            while (while_method_3(v176)){
                                assert("Tensor range check" && 0 <= v174 && v174 < 2l);
                                assert("Tensor range check" && 0 <= v176 && v176 < 2l);
                                int v178;
                                v178 = 96l * v176;
                                int v179;
                                v179 = v178 + v173;
                                int v180;
                                v180 = 4l * v174;
                                int v181;
                                v181 = v180 + v179;
                                float v182;
                                v182 = v46[v181];
                                bool v183;
                                v183 = 0l <= v176;
                                bool v185;
                                if (v183){
                                    bool v184;
                                    v184 = v176 < 2l;
                                    v185 = v184;
                                } else {
                                    v185 = false;
                                }
                                bool v186;
                                v186 = v185 == false;
                                if (v186){
                                    assert("The indices should be inside the range of the dimension." && v185);
                                } else {
                                }
                                bool v188;
                                v188 = 0l <= v174;
                                bool v190;
                                if (v188){
                                    bool v189;
                                    v189 = v174 < 2l;
                                    v190 = v189;
                                } else {
                                    v190 = false;
                                }
                                bool v191;
                                v191 = v190 == false;
                                if (v191){
                                    assert("The indices should be inside the range of the dimension." && v190);
                                } else {
                                }
                                int v193;
                                v193 = v174 * 2l;
                                int v194;
                                v194 = v176 + v193;
                                v170.x[v194] = v182;
                                v176 += 1l ;
                            }
                            v174 += 1l ;
                        }
                        v167 += 1l ;
                    }
                    v165 += 1l ;
                }
                int v195;
                v195 = 0l;
                #pragma unroll
                while (while_method_5(v195)){
                    int v197;
                    v197 = 0l;
                    #pragma unroll
                    while (while_method_5(v197)){
                        assert("Tensor range check" && 0 <= v195 && v195 < 1l);
                        assert("Tensor range check" && 0 <= v197 && v197 < 1l);
                        int v199;
                        v199 = v195 + v197;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v200 = v164[v199];
                        assert("Tensor range check" && 0 <= v195 && v195 < 1l);
                        int v201;
                        v201 = 192l * v195;
                        assert("Tensor range check" && 0 <= v197 && v197 < 1l);
                        int v202;
                        v202 = 8l * v197;
                        int v203;
                        v203 = v202 + v201;
                        int v204;
                        v204 = 0l;
                        #pragma unroll
                        while (while_method_3(v204)){
                            int v206;
                            v206 = 0l;
                            #pragma unroll
                            while (while_method_3(v206)){
                                assert("Tensor range check" && 0 <= v204 && v204 < 2l);
                                assert("Tensor range check" && 0 <= v206 && v206 < 2l);
                                int v208;
                                v208 = 4l * v206;
                                int v209;
                                v209 = v208 + v203;
                                int v210;
                                v210 = 96l * v204;
                                int v211;
                                v211 = v210 + v209;
                                float v212;
                                v212 = v62[v211];
                                bool v213;
                                v213 = 0l <= v206;
                                bool v215;
                                if (v213){
                                    bool v214;
                                    v214 = v206 < 2l;
                                    v215 = v214;
                                } else {
                                    v215 = false;
                                }
                                bool v216;
                                v216 = v215 == false;
                                if (v216){
                                    assert("The indices should be inside the range of the dimension." && v215);
                                } else {
                                }
                                bool v218;
                                v218 = 0l <= v204;
                                bool v220;
                                if (v218){
                                    bool v219;
                                    v219 = v204 < 2l;
                                    v220 = v219;
                                } else {
                                    v220 = false;
                                }
                                bool v221;
                                v221 = v220 == false;
                                if (v221){
                                    assert("The indices should be inside the range of the dimension." && v220);
                                } else {
                                }
                                int v223;
                                v223 = v204 * 2l;
                                int v224;
                                v224 = v206 + v223;
                                v200.x[v224] = v212;
                                v206 += 1l ;
                            }
                            v204 += 1l ;
                        }
                        v197 += 1l ;
                    }
                    v195 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                int v225;
                v225 = 0l;
                #pragma unroll
                while (while_method_5(v225)){
                    int v227;
                    v227 = 0l;
                    #pragma unroll
                    while (while_method_5(v227)){
                        int v229;
                        v229 = 0l;
                        #pragma unroll
                        while (while_method_5(v229)){
                            assert("Tensor range check" && 0 <= v225 && v225 < 1l);
                            assert("Tensor range check" && 0 <= v227 && v227 < 1l);
                            int v231;
                            v231 = v225 + v227;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v232 = v64[v231];
                            assert("Tensor range check" && 0 <= v225 && v225 < 1l);
                            assert("Tensor range check" && 0 <= v229 && v229 < 1l);
                            int v233;
                            v233 = v225 + v229;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v234 = v163[v233];
                            assert("Tensor range check" && 0 <= v227 && v227 < 1l);
                            assert("Tensor range check" && 0 <= v229 && v229 < 1l);
                            int v235;
                            v235 = v227 + v229;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v236 = v164[v235];
                            wmma::mma_sync(v232, v234, v236, v232);
                            v229 += 1l ;
                        }
                        v227 += 1l ;
                    }
                    v225 += 1l ;
                }
                v81 += 1l ;
            }
            int v237;
            v237 = 0l;
            #pragma unroll
            while (while_method_5(v237)){
                int v239;
                v239 = 0l;
                #pragma unroll
                while (while_method_5(v239)){
                    assert("Tensor range check" && 0 <= v237 && v237 < 1l);
                    assert("Tensor range check" && 0 <= v239 && v239 < 1l);
                    int v241;
                    v241 = v237 + v239;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v242 = v64[v241];
                    assert("Tensor range check" && 0 <= v237 && v237 < 1l);
                    assert("Tensor range check" && 0 <= v239 && v239 < 1l);
                    int v243;
                    v243 = 16l * v239;
                    int v244;
                    v244 = 384l * v237;
                    int v245;
                    v245 = v244 + v243;
                    float * v246;
                    v246 = v30+v245;
                    wmma::store_matrix_sync(v246, v242, 24l, wmma::mem_row_major);
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
            v252 = v248 % 4l;
            int v253;
            v253 = v248 / 4l;
            bool v254;
            v254 = v253 < 8l;
            bool v255;
            v255 = v254 == false;
            if (v255){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v254);
            } else {
            }
            assert("Tensor range check" && 0 <= v253 && v253 < 8l);
            assert("Tensor range check" && 0 <= v252 && v252 < 4l);
            int v257;
            v257 = 4l * v252;
            int v258;
            v258 = 128l * v253;
            int v259;
            v259 = v258 + v257;
            int v260;
            v260 = 24l * v253;
            int v261;
            v261 = v260 + v257;
            float * v262;
            v262 = v73+v259;
            float * v264;
            v264 = v16+v261;
            int v266;
            v266 = 0l;
            #pragma unroll
            while (while_method_3(v266)){
                int v268;
                v268 = 0l;
                #pragma unroll
                while (while_method_5(v268)){
                    assert("Tensor range check" && 0 <= v266 && v266 < 2l);
                    assert("Tensor range check" && 0 <= v268 && v268 < 1l);
                    int v270;
                    v270 = 16l * v268;
                    int v271;
                    v271 = 1024l * v266;
                    int v272;
                    v272 = v271 + v270;
                    int v273;
                    v273 = 192l * v266;
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
            v67 += 1l ;
        }
        v65 += 1l ;
    }
    return ;
}
__device__ inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 32l;
    return v1;
}
__device__ void method_2(float * v0, float * v1){
    int v2;
    v2 = blockIdx.x;
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    int v3;
    v3 = 4096l * v2;
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 1l);
    int v5;
    v5 = 4096l * v4;
    int v6;
    v6 = threadIdx.x;
    bool v7;
    v7 = 0l <= v6;
    bool v8;
    v8 = v7 == false;
    if (v8){
        assert("The index needs to be zero or positive." && v7);
    } else {
    }
    int v10;
    v10 = v6 % 32l;
    int v11;
    v11 = v6 / 32l;
    bool v12;
    v12 = v11 < 1l;
    bool v13;
    v13 = v12 == false;
    if (v13){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v12);
    } else {
    }
    assert("Tensor range check" && 0 <= v11 && v11 < 1l);
    assert("Tensor range check" && 0 <= v10 && v10 < 32l);
    int v15;
    v15 = 4l * v10;
    int v16;
    v16 = v15 + v3;
    int v17;
    v17 = 128l * v11;
    int v18;
    v18 = v17 + v16;
    assert("Tensor range check" && 0 <= v11 && v11 < 1l);
    assert("Tensor range check" && 0 <= v10 && v10 < 32l);
    int v19;
    v19 = v15 + v5;
    int v20;
    v20 = v17 + v19;
    int v21;
    v21 = 0l;
    while (while_method_7(v21)){
        assert("Tensor range check" && 0 <= v21 && v21 < 32l);
        int v23;
        v23 = 128l * v21;
        int v24;
        v24 = v23 + v18;
        float v25[4l];
        int v26[4l];
        int v27;
        v27 = 0l;
        while (while_method_5(v27)){
            assert("Tensor range check" && 0 <= v27 && v27 < 1l);
            int v29;
            v29 = 4l * v27;
            assert("Tensor range check" && 0 <= v27 && v27 < 1l);
            int v30;
            v30 = 128l * v27;
            int v31;
            v31 = v30 + v24;
            int4* v32;
            v32 = reinterpret_cast<int4*>(v1 + v31);
            int4* v33;
            v33 = reinterpret_cast<int4*>(v25 + v29);
            assert("Pointer alignment check" && (unsigned long long)(v32) % 4l == 0 && (unsigned long long)(v33) % 4l == 0);
            *v33 = *v32;
            v27 += 1l ;
        }
        int v34;
        v34 = 0l;
        while (while_method_5(v34)){
            int v36;
            v36 = 0l;
            while (while_method_2(v36)){
                bool v38;
                v38 = 0l <= v36;
                bool v40;
                if (v38){
                    bool v39;
                    v39 = v36 < 4l;
                    v40 = v39;
                } else {
                    v40 = false;
                }
                bool v41;
                v41 = v40 == false;
                if (v41){
                    assert("The indices should be inside the range of the dimension." && v40);
                } else {
                }
                bool v43;
                v43 = 0l <= v10;
                bool v45;
                if (v43){
                    bool v44;
                    v44 = v10 < 32l;
                    v45 = v44;
                } else {
                    v45 = false;
                }
                bool v46;
                v46 = v45 == false;
                if (v46){
                    assert("The indices should be inside the range of the dimension." && v45);
                } else {
                }
                int v48;
                v48 = v10 * 4l;
                int v49;
                v49 = v36 + v48;
                bool v50;
                v50 = 0l <= v34;
                bool v52;
                if (v50){
                    bool v51;
                    v51 = v34 < 1l;
                    v52 = v51;
                } else {
                    v52 = false;
                }
                bool v53;
                v53 = v52 == false;
                if (v53){
                    assert("The indices should be inside the range of the dimension." && v52);
                } else {
                }
                int v55;
                v55 = v34 * 128l;
                int v56;
                v56 = v49 + v55;
                assert("Tensor range check" && 0 <= v34 && v34 < 1l);
                assert("Tensor range check" && 0 <= v36 && v36 < 4l);
                int v57;
                v57 = 4l * v34;
                int v58;
                v58 = v57 + v36;
                v26[v58] = v56;
                v36 += 1l ;
            }
            v34 += 1l ;
        }
        bool v59;
        v59 = 0l <= v11;
        bool v60;
        v60 = v59 && v12;
        bool v61;
        v61 = v60 == false;
        if (v61){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v60);
        } else {
        }
        bool v63;
        v63 = 0l <= v21;
        bool v65;
        if (v63){
            bool v64;
            v64 = v21 < 32l;
            v65 = v64;
        } else {
            v65 = false;
        }
        bool v66;
        v66 = v65 == false;
        if (v66){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v65);
        } else {
        }
        int v68;
        v68 = v21 + v11;
        float v69[4l];
        int v70;
        v70 = 0l;
        while (while_method_5(v70)){
            int v72;
            v72 = 0l;
            while (while_method_2(v72)){
                assert("Tensor range check" && 0 <= v70 && v70 < 1l);
                assert("Tensor range check" && 0 <= v72 && v72 < 4l);
                int v74;
                v74 = 4l * v70;
                int v75;
                v75 = v74 + v72;
                float v76;
                v76 = v25[v75];
                float v77;
                v77 = v76 * v76;
                assert("Tensor range check" && 0 <= v70 && v70 < 1l);
                assert("Tensor range check" && 0 <= v72 && v72 < 4l);
                v69[v75] = v77;
                v72 += 1l ;
            }
            v70 += 1l ;
        }
        float v78;
        v78 = 0.0f;
        int v79;
        v79 = 0l;
        while (while_method_5(v79)){
            int v81;
            v81 = 0l;
            while (while_method_2(v81)){
                assert("Tensor range check" && 0 <= v79 && v79 < 1l);
                assert("Tensor range check" && 0 <= v81 && v81 < 4l);
                int v83;
                v83 = 4l * v79;
                int v84;
                v84 = v83 + v81;
                float v85;
                v85 = v69[v84];
                float v86;
                v86 = v78 + v85;
                v78 = v86;
                v81 += 1l ;
            }
            v79 += 1l ;
        }
        auto v87 = cooperative_groups::coalesced_threads();
        int v88;
        v88 = threadIdx.x;
        int v89;
        v89 = v88 / 32l;
        auto v90 = cooperative_groups::labeled_partition(v87,v89);
        Closure0 v91{};
        float v92;
        v92 = cooperative_groups::reduce(v90, v78, v91);
        float v93[4l];
        int v94;
        v94 = 0l;
        while (while_method_5(v94)){
            int v96;
            v96 = 0l;
            while (while_method_2(v96)){
                assert("Tensor range check" && 0 <= v94 && v94 < 1l);
                assert("Tensor range check" && 0 <= v96 && v96 < 4l);
                int v98;
                v98 = 4l * v94;
                int v99;
                v99 = v98 + v96;
                float v100;
                v100 = v25[v99];
                bool v101;
                v101 = v92 == 0.0f;
                bool v102;
                v102 = v101 != true;
                float v104;
                if (v102){
                    float v103;
                    v103 = v100 / v92;
                    v104 = v103;
                } else {
                    v104 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v94 && v94 < 1l);
                assert("Tensor range check" && 0 <= v96 && v96 < 4l);
                v93[v99] = v104;
                v96 += 1l ;
            }
            v94 += 1l ;
        }
        assert("Tensor range check" && 0 <= v21 && v21 < 32l);
        int v105;
        v105 = v23 + v20;
        int v106;
        v106 = 0l;
        while (while_method_5(v106)){
            assert("Tensor range check" && 0 <= v106 && v106 < 1l);
            int v108;
            v108 = 128l * v106;
            int v109;
            v109 = v108 + v105;
            assert("Tensor range check" && 0 <= v106 && v106 < 1l);
            int v110;
            v110 = 4l * v106;
            int4* v111;
            v111 = reinterpret_cast<int4*>(v93 + v110);
            int4* v112;
            v112 = reinterpret_cast<int4*>(v0 + v109);
            assert("Pointer alignment check" && (unsigned long long)(v111) % 4l == 0 && (unsigned long long)(v112) % 4l == 0);
            *v112 = *v111;
            v106 += 1l ;
        }
        v21 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return ;
}
__device__ inline bool while_method_8(int v0){
    bool v1;
    v1 = v0 < 1024l;
    return v1;
}
__device__ void method_3(float * v0, float * v1){
    int v2;
    v2 = blockIdx.x;
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    int v3;
    v3 = 4096l * v2;
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 1l);
    int v5;
    v5 = 4096l * v4;
    int v6;
    v6 = threadIdx.x;
    int v7;
    v7 = v6;
    while (while_method_8(v7)){
        bool v9;
        v9 = 0l <= v7;
        bool v10;
        v10 = v9 == false;
        if (v10){
            assert("The index needs to be zero or positive." && v9);
        } else {
        }
        int v12;
        v12 = v7 % 32l;
        int v13;
        v13 = v7 / 32l;
        bool v14;
        v14 = v13 < 32l;
        bool v15;
        v15 = v14 == false;
        if (v15){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v14);
        } else {
        }
        assert("Tensor range check" && 0 <= v13 && v13 < 32l);
        assert("Tensor range check" && 0 <= v12 && v12 < 32l);
        int v17;
        v17 = 4l * v12;
        int v18;
        v18 = v17 + v3;
        int v19;
        v19 = 128l * v13;
        int v20;
        v20 = v19 + v18;
        assert("Tensor range check" && 0 <= v13 && v13 < 32l);
        assert("Tensor range check" && 0 <= v12 && v12 < 32l);
        int v21;
        v21 = v17 + v5;
        int v22;
        v22 = v19 + v21;
        float v23[4l];
        float v24[4l];
        int4* v25;
        v25 = reinterpret_cast<int4*>(v1 + v20);
        int4* v26;
        v26 = reinterpret_cast<int4*>(v23 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v25) % 4l == 0 && (unsigned long long)(v26) % 4l == 0);
        *v26 = *v25;
        // Pushing the loop unrolling to: 0
        int v27;
        v27 = 0l;
        #pragma unroll
        while (while_method_2(v27)){
            assert("Tensor range check" && 0 <= v27 && v27 < 4l);
            float v29;
            v29 = v23[v27];
            bool v30;
            v30 = 0.0f >= v29;
            float v31;
            if (v30){
                v31 = 0.0f;
            } else {
                v31 = v29;
            }
            assert("Tensor range check" && 0 <= v27 && v27 < 4l);
            v24[v27] = v31;
            v27 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v32;
        v32 = reinterpret_cast<int4*>(v24 + 0l);
        int4* v33;
        v33 = reinterpret_cast<int4*>(v0 + v22);
        assert("Pointer alignment check" && (unsigned long long)(v32) % 4l == 0 && (unsigned long long)(v33) % 4l == 0);
        *v33 = *v32;
        v7 += 32l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return ;
}
__device__ void method_4(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5){
    int v6;
    v6 = blockIdx.x;
    assert("Tensor range check" && 0 <= v6 && v6 < 1l);
    int v7;
    v7 = 4096l * v6;
    int v8;
    v8 = blockIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 1l);
    int v9;
    v9 = 4096l * v8;
    int v10;
    v10 = v9 + v3;
    int v11;
    v11 = blockIdx.x;
    assert("Tensor range check" && 0 <= v11 && v11 < 1l);
    int v12;
    v12 = 32l * v11;
    int v13;
    v13 = v12 + v1;
    int v14;
    v14 = threadIdx.x;
    bool v15;
    v15 = 0l <= v14;
    bool v16;
    v16 = v15 == false;
    if (v16){
        assert("The index needs to be zero or positive." && v15);
    } else {
    }
    int v18;
    v18 = v14 % 32l;
    int v19;
    v19 = v14 / 32l;
    bool v20;
    v20 = v19 < 1l;
    bool v21;
    v21 = v20 == false;
    if (v21){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v20);
    } else {
    }
    assert("Tensor range check" && 0 <= v19 && v19 < 1l);
    assert("Tensor range check" && 0 <= v18 && v18 < 32l);
    int v23;
    v23 = 4l * v18;
    int v24;
    v24 = v23 + v7;
    int v25;
    v25 = 128l * v19;
    int v26;
    v26 = v25 + v24;
    assert("Tensor range check" && 0 <= v19 && v19 < 1l);
    assert("Tensor range check" && 0 <= v18 && v18 < 32l);
    int v27;
    v27 = v23 + v10;
    int v28;
    v28 = v25 + v27;
    assert("Tensor range check" && 0 <= v19 && v19 < 1l);
    int v29;
    v29 = v19 + v13;
    int v30;
    v30 = 0l;
    while (while_method_7(v30)){
        assert("Tensor range check" && 0 <= v30 && v30 < 32l);
        int v32;
        v32 = 128l * v30;
        int v33;
        v33 = v32 + v26;
        float v34[4l];
        int v35[4l];
        int v36;
        v36 = 0l;
        while (while_method_5(v36)){
            assert("Tensor range check" && 0 <= v36 && v36 < 1l);
            int v38;
            v38 = 4l * v36;
            assert("Tensor range check" && 0 <= v36 && v36 < 1l);
            int v39;
            v39 = 128l * v36;
            int v40;
            v40 = v39 + v33;
            int4* v41;
            v41 = reinterpret_cast<int4*>(v4 + v40);
            int4* v42;
            v42 = reinterpret_cast<int4*>(v34 + v38);
            assert("Pointer alignment check" && (unsigned long long)(v41) % 4l == 0 && (unsigned long long)(v42) % 4l == 0);
            *v42 = *v41;
            v36 += 1l ;
        }
        int v43;
        v43 = 0l;
        while (while_method_5(v43)){
            int v45;
            v45 = 0l;
            while (while_method_2(v45)){
                bool v47;
                v47 = 0l <= v45;
                bool v49;
                if (v47){
                    bool v48;
                    v48 = v45 < 4l;
                    v49 = v48;
                } else {
                    v49 = false;
                }
                bool v50;
                v50 = v49 == false;
                if (v50){
                    assert("The indices should be inside the range of the dimension." && v49);
                } else {
                }
                bool v52;
                v52 = 0l <= v18;
                bool v54;
                if (v52){
                    bool v53;
                    v53 = v18 < 32l;
                    v54 = v53;
                } else {
                    v54 = false;
                }
                bool v55;
                v55 = v54 == false;
                if (v55){
                    assert("The indices should be inside the range of the dimension." && v54);
                } else {
                }
                int v57;
                v57 = v18 * 4l;
                int v58;
                v58 = v45 + v57;
                bool v59;
                v59 = 0l <= v43;
                bool v61;
                if (v59){
                    bool v60;
                    v60 = v43 < 1l;
                    v61 = v60;
                } else {
                    v61 = false;
                }
                bool v62;
                v62 = v61 == false;
                if (v62){
                    assert("The indices should be inside the range of the dimension." && v61);
                } else {
                }
                int v64;
                v64 = v43 * 128l;
                int v65;
                v65 = v58 + v64;
                assert("Tensor range check" && 0 <= v43 && v43 < 1l);
                assert("Tensor range check" && 0 <= v45 && v45 < 4l);
                int v66;
                v66 = 4l * v43;
                int v67;
                v67 = v66 + v45;
                v35[v67] = v65;
                v45 += 1l ;
            }
            v43 += 1l ;
        }
        bool v68;
        v68 = 0l <= v19;
        bool v69;
        v69 = v68 && v20;
        bool v70;
        v70 = v69 == false;
        if (v70){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v69);
        } else {
        }
        bool v72;
        v72 = 0l <= v30;
        bool v74;
        if (v72){
            bool v73;
            v73 = v30 < 32l;
            v74 = v73;
        } else {
            v74 = false;
        }
        bool v75;
        v75 = v74 == false;
        if (v75){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v74);
        } else {
        }
        int v77;
        v77 = v30 + v19;
        bool v78[4l];
        int v79;
        v79 = 0l;
        while (while_method_5(v79)){
            int v81;
            v81 = 0l;
            while (while_method_2(v81)){
                assert("Tensor range check" && 0 <= v79 && v79 < 1l);
                assert("Tensor range check" && 0 <= v81 && v81 < 4l);
                int v83;
                v83 = 4l * v79;
                int v84;
                v84 = v83 + v81;
                float v85;
                v85 = v34[v84];
                int v86;
                v86 = v35[v84];
                bool v87;
                v87 = v86 < 3l;
                assert("Tensor range check" && 0 <= v79 && v79 < 1l);
                assert("Tensor range check" && 0 <= v81 && v81 < 4l);
                v78[v84] = v87;
                v81 += 1l ;
            }
            v79 += 1l ;
        }
        int v88[4l];
        int v89;
        v89 = 0l;
        while (while_method_5(v89)){
            int v91;
            v91 = 0l;
            while (while_method_2(v91)){
                assert("Tensor range check" && 0 <= v89 && v89 < 1l);
                assert("Tensor range check" && 0 <= v91 && v91 < 4l);
                int v93;
                v93 = 4l * v89;
                int v94;
                v94 = v93 + v91;
                bool v95;
                v95 = v78[v94];
                int v96;
                if (v95){
                    v96 = 1l;
                } else {
                    v96 = 0l;
                }
                assert("Tensor range check" && 0 <= v89 && v89 < 1l);
                assert("Tensor range check" && 0 <= v91 && v91 < 4l);
                v88[v94] = v96;
                v91 += 1l ;
            }
            v89 += 1l ;
        }
        int v97;
        v97 = 0l;
        int v98;
        v98 = 0l;
        while (while_method_5(v98)){
            int v100;
            v100 = 0l;
            while (while_method_2(v100)){
                assert("Tensor range check" && 0 <= v98 && v98 < 1l);
                assert("Tensor range check" && 0 <= v100 && v100 < 4l);
                int v102;
                v102 = 4l * v98;
                int v103;
                v103 = v102 + v100;
                int v104;
                v104 = v88[v103];
                int v105;
                v105 = v97 + v104;
                v97 = v105;
                v100 += 1l ;
            }
            v98 += 1l ;
        }
        auto v106 = cooperative_groups::coalesced_threads();
        int v107;
        v107 = threadIdx.x;
        int v108;
        v108 = v107 / 32l;
        auto v109 = cooperative_groups::labeled_partition(v106,v108);
        Closure1 v110{};
        int v111;
        v111 = cooperative_groups::reduce(v109, v97, v110);
        float v112[4l];
        int v113;
        v113 = 0l;
        while (while_method_5(v113)){
            int v115;
            v115 = 0l;
            while (while_method_2(v115)){
                assert("Tensor range check" && 0 <= v113 && v113 < 1l);
                assert("Tensor range check" && 0 <= v115 && v115 < 4l);
                int v117;
                v117 = 4l * v113;
                int v118;
                v118 = v117 + v115;
                float v119;
                v119 = v34[v118];
                bool v120;
                v120 = v78[v118];
                float v121;
                if (v120){
                    v121 = v119;
                } else {
                    v121 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v113 && v113 < 1l);
                assert("Tensor range check" && 0 <= v115 && v115 < 4l);
                v112[v118] = v121;
                v115 += 1l ;
            }
            v113 += 1l ;
        }
        float v122;
        v122 = 0.0f;
        int v123;
        v123 = 0l;
        while (while_method_5(v123)){
            int v125;
            v125 = 0l;
            while (while_method_2(v125)){
                assert("Tensor range check" && 0 <= v123 && v123 < 1l);
                assert("Tensor range check" && 0 <= v125 && v125 < 4l);
                int v127;
                v127 = 4l * v123;
                int v128;
                v128 = v127 + v125;
                float v129;
                v129 = v112[v128];
                float v130;
                v130 = v122 + v129;
                v122 = v130;
                v125 += 1l ;
            }
            v123 += 1l ;
        }
        auto v131 = cooperative_groups::coalesced_threads();
        int v132;
        v132 = threadIdx.x;
        int v133;
        v133 = v132 / 32l;
        auto v134 = cooperative_groups::labeled_partition(v131,v133);
        Closure0 v135{};
        float v136;
        v136 = cooperative_groups::reduce(v134, v122, v135);
        float v137;
        v137 = (float)v111;
        float v138;
        v138 = v136 / v137;
        float v139[4l];
        int v140;
        v140 = 0l;
        while (while_method_5(v140)){
            int v142;
            v142 = 0l;
            while (while_method_2(v142)){
                assert("Tensor range check" && 0 <= v140 && v140 < 1l);
                assert("Tensor range check" && 0 <= v142 && v142 < 4l);
                int v144;
                v144 = 4l * v140;
                int v145;
                v145 = v144 + v142;
                float v146;
                v146 = v34[v145];
                bool v147;
                v147 = v78[v145];
                float v148;
                if (v147){
                    v148 = v146;
                } else {
                    v148 = -1.0f / 0.0f;
                }
                float v149;
                v149 = v148 - v138;
                float v150;
                v150 = exp(v149);
                assert("Tensor range check" && 0 <= v140 && v140 < 1l);
                assert("Tensor range check" && 0 <= v142 && v142 < 4l);
                v139[v145] = v150;
                v142 += 1l ;
            }
            v140 += 1l ;
        }
        float v151;
        v151 = 0.0f;
        int v152;
        v152 = 0l;
        while (while_method_5(v152)){
            int v154;
            v154 = 0l;
            while (while_method_2(v154)){
                assert("Tensor range check" && 0 <= v152 && v152 < 1l);
                assert("Tensor range check" && 0 <= v154 && v154 < 4l);
                int v156;
                v156 = 4l * v152;
                int v157;
                v157 = v156 + v154;
                float v158;
                v158 = v139[v157];
                float v159;
                v159 = v151 + v158;
                v151 = v159;
                v154 += 1l ;
            }
            v152 += 1l ;
        }
        auto v160 = cooperative_groups::coalesced_threads();
        int v161;
        v161 = threadIdx.x;
        int v162;
        v162 = v161 / 32l;
        auto v163 = cooperative_groups::labeled_partition(v160,v162);
        float v164;
        v164 = cooperative_groups::reduce(v163, v151, v135);
        float v165[4l];
        int v166;
        v166 = 0l;
        while (while_method_5(v166)){
            int v168;
            v168 = 0l;
            while (while_method_2(v168)){
                assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                assert("Tensor range check" && 0 <= v168 && v168 < 4l);
                int v170;
                v170 = 4l * v166;
                int v171;
                v171 = v170 + v168;
                float v172;
                v172 = v139[v171];
                float v173;
                v173 = v172 / v164;
                assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                assert("Tensor range check" && 0 <= v168 && v168 < 4l);
                v165[v171] = v173;
                v168 += 1l ;
            }
            v166 += 1l ;
        }
        float v174[4l];
        float v175;
        v175 = 0.0f;
        int v176;
        v176 = 0l;
        while (while_method_5(v176)){
            assert("Tensor range check" && 0 <= v176 && v176 < 1l);
            int v178;
            v178 = 4l * v176;
            assert("Tensor range check" && 0 <= v176 && v176 < 1l);
            int v179; float v180;
            Tuple1 tmp0 = Tuple1{0l, 0.0f};
            v179 = tmp0.v0; v180 = tmp0.v1;
            while (while_method_2(v179)){
                assert("Tensor range check" && 0 <= v179 && v179 < 4l);
                int v182;
                v182 = v179 + v178;
                float v183;
                v183 = v165[v182];
                float v184;
                v184 = v180 + v183;
                v180 = v184;
                v179 += 1l ;
            }
            auto v185 = cooperative_groups::coalesced_threads();
            int v186;
            v186 = threadIdx.x;
            int v187;
            v187 = v186 / 32l;
            auto v188 = cooperative_groups::labeled_partition(v185,v187);
            Closure2 v189{};
            float v190;
            v190 = cooperative_groups::inclusive_scan(v188, v180, v189);
            float v191;
            v191 = v188.shfl_up(v190,1);
            bool v192;
            v192 = v188.thread_rank() == 0;
            float v193;
            if (v192){
                v193 = 0.0f;
            } else {
                v193 = v191;
            }
            float v194;
            v194 = v188.shfl(v190,v188.num_threads()-1);
            float v195;
            v195 = v175 + v193;
            int v196; float v197;
            Tuple1 tmp1 = Tuple1{0l, v195};
            v196 = tmp1.v0; v197 = tmp1.v1;
            while (while_method_2(v196)){
                assert("Tensor range check" && 0 <= v196 && v196 < 4l);
                int v199;
                v199 = v196 + v178;
                float v200;
                v200 = v165[v199];
                float v201;
                v201 = v197 + v200;
                assert("Tensor range check" && 0 <= v196 && v196 < 4l);
                v174[v199] = v201;
                v197 = v201;
                v196 += 1l ;
            }
            float v202;
            v202 = v175 + v194;
            v175 = v202;
            v176 += 1l ;
        }
        float v203[4l];
        bool v204[4l];
        int v205;
        v205 = 0l;
        while (while_method_5(v205)){
            int v207;
            v207 = 0l;
            while (while_method_2(v207)){
                assert("Tensor range check" && 0 <= v205 && v205 < 1l);
                assert("Tensor range check" && 0 <= v207 && v207 < 4l);
                int v209;
                v209 = 4l * v205;
                int v210;
                v210 = v209 + v207;
                float v211;
                v211 = v174[v210];
                float v212;
                v212 = v165[v210];
                bool v213;
                v213 = v212 > 0.0f;
                assert("Tensor range check" && 0 <= v205 && v205 < 1l);
                assert("Tensor range check" && 0 <= v207 && v207 < 4l);
                v203[v210] = v211;
                v204[v210] = v213;
                v207 += 1l ;
            }
            v205 += 1l ;
        }
        float v214; bool v215;
        Tuple2 tmp2 = Tuple2{-1.0f / 0.0f, false};
        v214 = tmp2.v0; v215 = tmp2.v1;
        int v216;
        v216 = 0l;
        while (while_method_5(v216)){
            int v218;
            v218 = 0l;
            while (while_method_2(v218)){
                assert("Tensor range check" && 0 <= v216 && v216 < 1l);
                assert("Tensor range check" && 0 <= v218 && v218 < 4l);
                int v220;
                v220 = 4l * v216;
                int v221;
                v221 = v220 + v218;
                float v222;
                v222 = v203[v221];
                bool v223;
                v223 = v204[v221];
                float v230; bool v231;
                if (v215){
                    if (v223){
                        bool v224;
                        v224 = v214 >= v222;
                        float v225;
                        if (v224){
                            v225 = v214;
                        } else {
                            v225 = v222;
                        }
                        v230 = v225; v231 = true;
                    } else {
                        v230 = v214; v231 = v215;
                    }
                } else {
                    if (v223){
                        v230 = v222; v231 = v223;
                    } else {
                        v230 = v214; v231 = v215;
                    }
                }
                v214 = v230;
                v215 = v231;
                v218 += 1l ;
            }
            v216 += 1l ;
        }
        auto v232 = cooperative_groups::coalesced_threads();
        int v233;
        v233 = threadIdx.x;
        int v234;
        v234 = v233 / 32l;
        auto v235 = cooperative_groups::labeled_partition(v232,v234);
        Closure3 v236{};
        float v237; bool v238;
        Tuple2 tmp3 = cooperative_groups::reduce(v235, Tuple2{v214, v215}, v236);
        v237 = tmp3.v0; v238 = tmp3.v1;
        bool v239;
        v239 = v238 == false;
        if (v239){
            assert("The local reduce must be true." && v238);
        } else {
        }
        float v241[4l];
        int v242[4l];
        int v243;
        v243 = 0l;
        while (while_method_5(v243)){
            int v245;
            v245 = 0l;
            while (while_method_2(v245)){
                assert("Tensor range check" && 0 <= v243 && v243 < 1l);
                assert("Tensor range check" && 0 <= v245 && v245 < 4l);
                int v247;
                v247 = 4l * v243;
                int v248;
                v248 = v247 + v245;
                int v249;
                v249 = v35[v248];
                float v250;
                v250 = curand_uniform(&v5);
                assert("Tensor range check" && 0 <= v243 && v243 < 1l);
                assert("Tensor range check" && 0 <= v245 && v245 < 4l);
                v241[v248] = v250;
                v242[v248] = v249;
                v245 += 1l ;
            }
            v243 += 1l ;
        }
        float v251; int v252;
        Tuple3 tmp4 = Tuple3{0.0f, 2147483647l};
        v251 = tmp4.v0; v252 = tmp4.v1;
        int v253;
        v253 = 0l;
        while (while_method_5(v253)){
            int v255;
            v255 = 0l;
            while (while_method_2(v255)){
                assert("Tensor range check" && 0 <= v253 && v253 < 1l);
                assert("Tensor range check" && 0 <= v255 && v255 < 4l);
                int v257;
                v257 = 4l * v253;
                int v258;
                v258 = v257 + v255;
                float v259;
                v259 = v241[v258];
                int v260;
                v260 = v242[v258];
                bool v261;
                v261 = v252 < v260;
                float v262; int v263;
                if (v261){
                    v262 = v251; v263 = v252;
                } else {
                    v262 = v259; v263 = v260;
                }
                v251 = v262;
                v252 = v263;
                v255 += 1l ;
            }
            v253 += 1l ;
        }
        auto v264 = cooperative_groups::coalesced_threads();
        int v265;
        v265 = threadIdx.x;
        int v266;
        v266 = v265 / 32l;
        auto v267 = cooperative_groups::labeled_partition(v264,v266);
        Closure4 v268{};
        float v269; int v270;
        Tuple3 tmp5 = cooperative_groups::reduce(v267, Tuple3{v251, v252}, v268);
        v269 = tmp5.v0; v270 = tmp5.v1;
        float v271;
        v271 = v237 * v269;
        int v272[4l];
        bool v273[4l];
        int v274;
        v274 = 0l;
        while (while_method_5(v274)){
            int v276;
            v276 = 0l;
            while (while_method_2(v276)){
                assert("Tensor range check" && 0 <= v274 && v274 < 1l);
                assert("Tensor range check" && 0 <= v276 && v276 < 4l);
                int v278;
                v278 = 4l * v274;
                int v279;
                v279 = v278 + v276;
                float v280;
                v280 = v203[v279];
                bool v281;
                v281 = v204[v279];
                int v282;
                v282 = v35[v279];
                int v285; bool v286;
                if (v281){
                    float v283;
                    v283 = v280 - v271;
                    bool v284;
                    v284 = v283 >= 0.0f;
                    v285 = v282; v286 = v284;
                } else {
                    v285 = 2147483647l; v286 = false;
                }
                assert("Tensor range check" && 0 <= v274 && v274 < 1l);
                assert("Tensor range check" && 0 <= v276 && v276 < 4l);
                v272[v279] = v285;
                v273[v279] = v286;
                v276 += 1l ;
            }
            v274 += 1l ;
        }
        int v287; bool v288;
        Tuple4 tmp6 = Tuple4{2147483647l, false};
        v287 = tmp6.v0; v288 = tmp6.v1;
        int v289;
        v289 = 0l;
        while (while_method_5(v289)){
            int v291;
            v291 = 0l;
            while (while_method_2(v291)){
                assert("Tensor range check" && 0 <= v289 && v289 < 1l);
                assert("Tensor range check" && 0 <= v291 && v291 < 4l);
                int v293;
                v293 = 4l * v289;
                int v294;
                v294 = v293 + v291;
                int v295;
                v295 = v272[v294];
                bool v296;
                v296 = v273[v294];
                int v303; bool v304;
                if (v288){
                    if (v296){
                        bool v297;
                        v297 = v287 < v295;
                        int v298;
                        if (v297){
                            v298 = v287;
                        } else {
                            v298 = v295;
                        }
                        v303 = v298; v304 = true;
                    } else {
                        v303 = v287; v304 = v288;
                    }
                } else {
                    if (v296){
                        v303 = v295; v304 = v296;
                    } else {
                        v303 = v287; v304 = v288;
                    }
                }
                v287 = v303;
                v288 = v304;
                v291 += 1l ;
            }
            v289 += 1l ;
        }
        auto v305 = cooperative_groups::coalesced_threads();
        int v306;
        v306 = threadIdx.x;
        int v307;
        v307 = v306 / 32l;
        auto v308 = cooperative_groups::labeled_partition(v305,v307);
        Closure5 v309{};
        int v310; bool v311;
        Tuple4 tmp7 = cooperative_groups::reduce(v308, Tuple4{v287, v288}, v309);
        v310 = tmp7.v0; v311 = tmp7.v1;
        bool v312;
        v312 = v311 == false;
        if (v312){
            assert("The local reduce must be true." && v311);
        } else {
        }
        bool v314;
        v314 = v310 < 3l;
        bool v315;
        v315 = v314 == false;
        if (v315){
            assert("The masking requirement is violated in masked_softmax_and_discrete_sample_." && v314);
        } else {
        }
        assert("Tensor range check" && 0 <= v30 && v30 < 32l);
        int v317;
        v317 = v32 + v28;
        int v318;
        v318 = 0l;
        while (while_method_5(v318)){
            assert("Tensor range check" && 0 <= v318 && v318 < 1l);
            int v320;
            v320 = 128l * v318;
            int v321;
            v321 = v320 + v317;
            assert("Tensor range check" && 0 <= v318 && v318 < 1l);
            int v322;
            v322 = 4l * v318;
            int4* v323;
            v323 = reinterpret_cast<int4*>(v165 + v322);
            int4* v324;
            v324 = reinterpret_cast<int4*>(v2 + v321);
            assert("Pointer alignment check" && (unsigned long long)(v323) % 4l == 0 && (unsigned long long)(v324) % 4l == 0);
            *v324 = *v323;
            v318 += 1l ;
        }
        assert("Tensor range check" && 0 <= v30 && v30 < 32l);
        int v325;
        v325 = v30 + v29;
        v0[v325] = v310;
        v30 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return ;
}
__device__ unsigned int loop_6(unsigned int v0, curandStatePhilox4_32_10_t & v1){
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
        return loop_6(v0, v1);
    }
}
__device__ int int_range_5(int v0, int v1, curandStatePhilox4_32_10_t & v2){
    int v3;
    v3 = v0 - v1;
    unsigned int v4;
    v4 = (unsigned int)v3;
    unsigned int v5;
    v5 = loop_6(v4, v2);
    unsigned int v6;
    v6 = (unsigned int)v1;
    unsigned int v7;
    v7 = v5 + v6;
    int v8;
    v8 = (int)v7;
    return v8;
}
__device__ __noinline__ Tuple0 noinline_run_0(unsigned char * v0, unsigned char * v1, Union3 v2){
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v3;
    v3 = reinterpret_cast<float *>(&v0[131072ull]);
    float * v5;
    v5 = reinterpret_cast<float *>(&v0[0ull]);
    int * v7;
    v7 = reinterpret_cast<int *>(&v0[196608ull]);
    unsigned long long v9;
    v9 = clock64();
    int v10;
    v10 = threadIdx.x;
    int v11;
    v11 = blockIdx.x;
    int v12;
    v12 = v11 * 32l;
    int v13;
    v13 = v10 + v12;
    unsigned long long v14;
    v14 = (unsigned long long)v13;
    curandStatePhilox4_32_10_t v15;
    curand_init(v9,v14,0ull,&v15);
    float * v16;
    v16 = reinterpret_cast<float *>(&v0[0ull]);
    int v18;
    v18 = blockIdx.x;
    assert("Tensor range check" && 0 <= v18 && v18 < 1l);
    int v19;
    v19 = 4096l * v18;
    unsigned long long v20;
    v20 = clock64();
    int v21;
    v21 = threadIdx.x;
    int v22;
    v22 = blockIdx.x;
    int v23;
    v23 = v22 * 32l;
    int v24;
    v24 = v21 + v23;
    unsigned long long v25;
    v25 = (unsigned long long)v24;
    curandStatePhilox4_32_10_t v26;
    curand_init(v20,v25,0ull,&v26);
    int v27;
    v27 = threadIdx.x;
    int v28;
    v28 = v27;
    while (while_method_0(v28)){
        bool v30;
        v30 = 0l <= v28;
        bool v31;
        v31 = v30 == false;
        if (v31){
            assert("The index needs to be zero or positive." && v30);
        } else {
        }
        int v33;
        v33 = v28 % 128l;
        int v34;
        v34 = v28 / 128l;
        bool v35;
        v35 = v34 < 32l;
        bool v36;
        v36 = v35 == false;
        if (v36){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v35);
        } else {
        }
        assert("Tensor range check" && 0 <= v34 && v34 < 32l);
        assert("Tensor range check" && 0 <= v33 && v33 < 128l);
        int v38;
        v38 = v33 + v19;
        int v39;
        v39 = 128l * v34;
        int v40;
        v40 = v39 + v38;
        v16[v40] = 0.0f;
        v28 += 32l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    switch (v2.tag) {
        case 0: { // None
            break;
        }
        case 1: { // Some
            static_array_list<Union0,10l> v41 = v2.case1.v0;
            int v42;
            v42 = threadIdx.x;
            assert("Tensor range check" && 0 <= v42 && v42 < 32l);
            int v43;
            v43 = 128l * v42;
            int v44;
            v44 = v43 + v19;
            float * v45;
            v45 = v16+v44;
            int v47;
            v47 = v41.length;
            bool v48;
            v48 = v47 == 0l;
            if (v48){
                v45[0l] = 1.0f;
            } else {
            }
            int v49;
            v49 = v41.length;
            int v50;
            v50 = 0l;
            while (while_method_1(v49, v50)){
                Union0 v52;
                v52 = v41[v50];
                int v54;
                v54 = v50 * 6l;
                int v55;
                v55 = 1l + v54;
                switch (v52.tag) {
                    case 0: { // C1of2
                        Union1 v56 = v52.case0.v0;
                        switch (v56.tag) {
                            case 0: { // Call
                                v45[v55] = 1.0f;
                                break;
                            }
                            case 1: { // Fold
                                int v57;
                                v57 = v55 + 1l;
                                v45[v57] = 1.0f;
                                break;
                            }
                            case 2: { // Raise
                                int v58;
                                v58 = v55 + 2l;
                                v45[v58] = 1.0f;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        break;
                    }
                    case 1: { // C2of2
                        Union2 v59 = v52.case1.v0;
                        int v60;
                        v60 = v55 + 3l;
                        switch (v59.tag) {
                            case 0: { // Jack
                                v45[v60] = 1.0f;
                                break;
                            }
                            case 1: { // King
                                int v61;
                                v61 = v60 + 1l;
                                v45[v61] = 1.0f;
                                break;
                            }
                            case 2: { // Queen
                                int v62;
                                v62 = v60 + 2l;
                                v45[v62] = 1.0f;
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
                v50 += 1l ;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v63;
    v63 = 0l;
    while (while_method_2(v63)){
        float * v65;
        v65 = reinterpret_cast<float *>(&v0[0ull]);
        float * v67;
        v67 = reinterpret_cast<float *>(&v1[0ull]);
        assert("Tensor range check" && 0 <= v63 && v63 < 4l);
        int v69;
        v69 = 16384l * v63;
        float * v70;
        v70 = reinterpret_cast<float *>(&v0[16384ull]);
        int v72;
        v72 = blockIdx.x;
        assert("Tensor range check" && 0 <= v72 && v72 < 1l);
        int v73;
        v73 = 4096l * v72;
        int v74;
        v74 = blockIdx.x;
        assert("Tensor range check" && 0 <= v74 && v74 < 1l);
        int v75;
        v75 = 4096l * v74;
        method_1(v67, v69, v70, v75, v65, v73);
        float * v76;
        v76 = reinterpret_cast<float *>(&v0[32768ull]);
        method_2(v76, v70);
        float * v78;
        v78 = reinterpret_cast<float *>(&v0[49152ull]);
        method_3(v78, v76);
        float * v80;
        v80 = reinterpret_cast<float *>(&v1[262144ull]);
        assert("Tensor range check" && 0 <= v63 && v63 < 4l);
        float * v82;
        v82 = reinterpret_cast<float *>(&v0[65536ull]);
        int v84;
        v84 = blockIdx.x;
        assert("Tensor range check" && 0 <= v84 && v84 < 1l);
        int v85;
        v85 = 4096l * v84;
        int v86;
        v86 = blockIdx.x;
        assert("Tensor range check" && 0 <= v86 && v86 < 1l);
        int v87;
        v87 = 4096l * v86;
        method_1(v80, v69, v82, v87, v78, v85);
        float * v88;
        v88 = reinterpret_cast<float *>(&v0[81920ull]);
        method_2(v88, v82);
        float * v90;
        v90 = reinterpret_cast<float *>(&v0[98304ull]);
        method_3(v90, v88);
        float * v92;
        v92 = reinterpret_cast<float *>(&v1[524288ull]);
        assert("Tensor range check" && 0 <= v63 && v63 < 4l);
        float * v94;
        v94 = reinterpret_cast<float *>(&v0[114688ull]);
        int v96;
        v96 = blockIdx.x;
        assert("Tensor range check" && 0 <= v96 && v96 < 1l);
        int v97;
        v97 = 4096l * v96;
        int v98;
        v98 = blockIdx.x;
        assert("Tensor range check" && 0 <= v98 && v98 < 1l);
        int v99;
        v99 = 4096l * v98;
        method_1(v92, v69, v94, v99, v90, v97);
        float * v100;
        v100 = reinterpret_cast<float *>(&v0[131072ull]);
        assert("Tensor range check" && 0 <= v63 && v63 < 4l);
        int v102;
        v102 = 4096l * v63;
        int * v103;
        v103 = reinterpret_cast<int *>(&v0[196608ull]);
        assert("Tensor range check" && 0 <= v63 && v63 < 4l);
        int v105;
        v105 = 32l * v63;
        method_4(v103, v105, v100, v102, v94, v26);
        v63 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v106;
    v106 = 0l;
    int v107;
    v107 = 4l;
    int v108;
    v108 = int_range_5(v107, v106, v26);
    float * v109;
    v109 = reinterpret_cast<float *>(&v0[131072ull]);
    int * v111;
    v111 = reinterpret_cast<int *>(&v0[196608ull]);
    assert("Tensor range check" && 0 <= v108 && v108 < 4l);
    int v113;
    v113 = 32l * v108;
    int v114;
    v114 = blockIdx.x;
    assert("Tensor range check" && 0 <= v114 && v114 < 1l);
    int v115;
    v115 = 32l * v114;
    int v116;
    v116 = v115 + v113;
    int v117;
    v117 = threadIdx.x;
    assert("Tensor range check" && 0 <= v117 && v117 < 32l);
    int v118;
    v118 = v117 + v116;
    int v119;
    v119 = v111[v118];
    bool v120;
    v120 = 0l == v119;
    Union1 v129;
    if (v120){
        v129 = Union1{Union1_1{}};
    } else {
        bool v122;
        v122 = 1l == v119;
        if (v122){
            v129 = Union1{Union1_0{}};
        } else {
            bool v124;
            v124 = 2l == v119;
            if (v124){
                v129 = Union1{Union1_2{}};
            } else {
                printf("%s\n", "Invalid output id in the Leduc model.");
                __trap();
            }
        }
    }
    int v130;
    v130 = blockIdx.x;
    int v131;
    v131 = threadIdx.x;
    assert("Tensor range check" && 0 <= v108 && v108 < 4l);
    assert("Tensor range check" && 0 <= v130 && v130 < 1l);
    assert("Tensor range check" && 0 <= v131 && v131 < 32l);
    assert("Tensor range check" && 0 <= v119 && v119 < 128l);
    int v132;
    v132 = 128l * v131;
    int v133;
    v133 = v132 + v119;
    int v134;
    v134 = 4096l * v130;
    int v135;
    v135 = v134 + v133;
    int v136;
    v136 = 4096l * v108;
    int v137;
    v137 = v136 + v135;
    float v138;
    v138 = v109[v137];
    int v139;
    v139 = blockIdx.x;
    assert("Tensor range check" && 0 <= v139 && v139 < 1l);
    int v140;
    v140 = 4096l * v139;
    int v141;
    v141 = threadIdx.x;
    assert("Tensor range check" && 0 <= v141 && v141 < 32l);
    int v142;
    v142 = 128l * v141;
    int v143;
    v143 = v142 + v140;
    assert("Tensor range check" && 0 <= v119 && v119 < 128l);
    int v144;
    v144 = v119 + v143;
    return Tuple0{v129, v109, v144, 4096l, 4l, v138};
}
__device__ void method_7(Union1 v0){
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
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1) {
    static_array_list<Union0,10l> v2;
    v2 = static_array_list<Union0,10l>{};
    v2.unsafe_set_length(10l);
    Union2 v4;
    v4 = Union2{Union2_1{}};
    Union0 v5;
    v5 = Union0{Union0_1{v4}};
    v2[0l] = v5;
    Union1 v8;
    v8 = Union1{Union1_0{}};
    Union0 v9;
    v9 = Union0{Union0_0{v8}};
    v2[1l] = v9;
    Union1 v12;
    v12 = Union1{Union1_2{}};
    Union0 v13;
    v13 = Union0{Union0_0{v12}};
    v2[2l] = v13;
    Union1 v16;
    v16 = Union1{Union1_2{}};
    Union0 v17;
    v17 = Union0{Union0_0{v16}};
    v2[3l] = v17;
    Union1 v20;
    v20 = Union1{Union1_0{}};
    Union0 v21;
    v21 = Union0{Union0_0{v20}};
    v2[4l] = v21;
    Union2 v24;
    v24 = Union2{Union2_2{}};
    Union0 v25;
    v25 = Union0{Union0_1{v24}};
    v2[5l] = v25;
    Union1 v28;
    v28 = Union1{Union1_0{}};
    Union0 v29;
    v29 = Union0{Union0_0{v28}};
    v2[6l] = v29;
    Union1 v32;
    v32 = Union1{Union1_2{}};
    Union0 v33;
    v33 = Union0{Union0_0{v32}};
    v2[7l] = v33;
    Union1 v36;
    v36 = Union1{Union1_2{}};
    Union0 v37;
    v37 = Union0{Union0_0{v36}};
    v2[8l] = v37;
    Union1 v40;
    v40 = Union1{Union1_0{}};
    Union0 v41;
    v41 = Union0{Union0_0{v40}};
    v2[9l] = v41;
    Union3 v44;
    v44 = Union3{Union3_1{v2}};
    Union1 v45; float * v46; int v47; int v48; int v49; float v50;
    Tuple0 tmp8 = noinline_run_0(v0, v1, v44);
    v45 = tmp8.v0; v46 = tmp8.v1; v47 = tmp8.v2; v48 = tmp8.v3; v49 = tmp8.v4; v50 = tmp8.v5;
    cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v51 = console_lock;
    auto v52 = cooperative_groups::coalesced_threads();
    v51.acquire();
    printf("{%s = ","action");
    method_7(v45);
    int v53;
    v53 = 0l;
    printf("; %s = %c","sampling_prob_ensemble", '[');
    int v54;
    v54 = 0l;
    while (while_method_1(v49, v54)){
        int v56;
        v56 = v53;
        bool v57;
        v57 = v56 >= 100l;
        if (v57){
            printf("%s"," ...");
            break;
        } else {
        }
        bool v58;
        v58 = v54 == 0l;
        bool v59;
        v59 = v58 != true;
        if (v59){
            printf("%s","; ");
        } else {
        }
        int v60;
        v60 = v53 + 1l;
        v53 = v60;
        int v61;
        v61 = v54 * v48;
        int v62;
        v62 = v47 + v61;
        float v63;
        v63 = v46[v62];
        printf("%f",v63);
        v54 += 1l ;
    }
    printf("%c",']');
    printf("; %s = %f}\n","sampling_prob_selected", v50);
    v51.release();
    v52.sync() ;
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
i8 = i16 = i32 = i64 = u8 = u16 = u32 = u64 = int; f32 = f64 = float; char = string = str

options = []
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39')
options.append('--restrict')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def main_body():
    v0 = cp.empty(786432,dtype=cp.uint8)
    v1 = cp.empty(197120,dtype=cp.uint8)
    v3 = v0[0:0+4*65536].view(cp.float32)
    v4 = cp.random.normal(0.0,0.00390625,65536,dtype=cp.float32) # type: ignore
    cp.copyto(v3[0:0+65536],v4[0:0+65536])
    del v3, v4
    v6 = v0[262144:262144+4*65536].view(cp.float32)
    v7 = cp.random.normal(0.0,0.00390625,65536,dtype=cp.float32) # type: ignore
    cp.copyto(v6[0:0+65536],v7[0:0+65536])
    del v6, v7
    v9 = v0[524288:524288+4*65536].view(cp.float32)
    v10 = cp.random.normal(0.0,0.00390625,65536,dtype=cp.float32) # type: ignore
    cp.copyto(v9[0:0+65536],v10[0:0+65536])
    del v9, v10
    v13 = "{}\n"
    v14 = "Running the kernel."
    print(v13.format(v14),end="")
    del v13, v14
    v15 = 0
    v16 = raw_module.get_function(f"entry{v15}")
    del v15
    v16.max_dynamic_shared_size_bytes = 1536 
    v16((1,),(32,),(v1, v0),shared_mem=1536)
    del v0, v1, v16
    cp.cuda.get_current_stream().synchronize()
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
