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
__device__ void method_0(float * v0, float * v1, int v2, float * v3, int v4);
__device__ void method_1(float * v0, float * v1);
__device__ void method_2(float * v0, float * v1);
__device__ void method_3(float * v0, float * v1, int v2, float * v3);
struct Tuple0;
struct Tuple1;
__device__ void method_4(int * v0, int v1, float * v2, curandStatePhilox4_32_10_t & v3);
__device__ void write_7(const char * v0);
__device__ void write_6();
__device__ void write_8();
__device__ void write_9();
__device__ void write_5(Union1 v0);
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
struct Closure0 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple0 {
    int v0;
    float v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Closure1 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple1 {
    float v0;
    int v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure2 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v0 >= 0.0f;
        bool v6;
        if (v4){
            bool v5;
            v5 = v2 >= 0.0f;
            v6 = v5;
        } else {
            v6 = false;
        }
        if (v6){
            bool v7;
            v7 = v0 <= v2;
            if (v7){
                return Tuple1{v0, v1};
            } else {
                return Tuple1{v2, v3};
            }
        } else {
            if (v4){
                return Tuple1{v0, v1};
            } else {
                bool v10;
                v10 = v2 >= 0.0f;
                if (v10){
                    return Tuple1{v2, v3};
                } else {
                    return Tuple1{v0, v1};
                }
            }
        }
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 2048l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 16l;
    return v1;
}
__device__ inline bool while_method_2(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 8l;
    return v1;
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
__device__ void method_0(float * v0, float * v1, int v2, float * v3, int v4){
    unsigned int v5;
    v5 = 0ul;
    asm("mov.u32 %0, %dynamic_smem_size;" : "=r"(v5));
    unsigned long long v6;
    v6 = (unsigned long long)v5;
    bool v7;
    v7 = 1536ull <= v6;
    bool v8;
    v8 = v7 == false;
    if (v8){
        assert("The shared memory used in the matmult node is lower than the allocated amount." && v7);
    } else {
    }
    extern __shared__ unsigned char v10[];
    float * v11;
    v11 = reinterpret_cast<float *>(&v10[0ull]);
    float * v13;
    v13 = reinterpret_cast<float *>(&v10[768ull]);
    float * v15;
    v15 = reinterpret_cast<float *>(&v10[0ull]);
    int v17;
    v17 = threadIdx.x;
    int v18;
    v18 = v17 / 32l;
    bool v19;
    v19 = 0l <= v18;
    bool v20;
    v20 = v19 == false;
    if (v20){
        assert("The index needs to be zero or positive." && v19);
    } else {
    }
    int v22;
    v22 = v18 % 1l;
    bool v23;
    v23 = v18 < 1l;
    bool v24;
    v24 = v23 == false;
    if (v24){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v23);
    } else {
    }
    assert("Tensor range check" && 0 <= v18 && v18 < 1l);
    assert("Tensor range check" && 0 <= v22 && v22 < 1l);
    int v26;
    v26 = 16l * v22;
    int v27;
    v27 = 384l * v18;
    int v28;
    v28 = v27 + v26;
    float * v29;
    v29 = v15+v28;
    assert("Tensor range check" && 0 <= v18 && v18 < 1l);
    int v31;
    v31 = 192l * v18;
    int v32;
    v32 = threadIdx.x;
    int v33;
    v33 = v32 % 32l;
    bool v34;
    v34 = 0l <= v33;
    bool v35;
    v35 = v34 == false;
    if (v35){
        assert("The index needs to be zero or positive." && v34);
    } else {
    }
    int v37;
    v37 = v33 % 4l;
    int v38;
    v38 = v33 / 4l;
    bool v39;
    v39 = v38 < 8l;
    bool v40;
    v40 = v39 == false;
    if (v40){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v39);
    } else {
    }
    assert("Tensor range check" && 0 <= v38 && v38 < 8l);
    assert("Tensor range check" && 0 <= v37 && v37 < 4l);
    int v42;
    v42 = v37 + v31;
    int v43;
    v43 = 12l * v38;
    int v44;
    v44 = v43 + v42;
    float * v45;
    v45 = v11+v44;
    assert("Tensor range check" && 0 <= v22 && v22 < 1l);
    int v47;
    v47 = 192l * v22;
    int v48;
    v48 = threadIdx.x;
    int v49;
    v49 = v48 % 32l;
    bool v50;
    v50 = 0l <= v49;
    bool v51;
    v51 = v50 == false;
    if (v51){
        assert("The index needs to be zero or positive." && v50);
    } else {
    }
    int v53;
    v53 = v49 % 4l;
    int v54;
    v54 = v49 / 4l;
    bool v55;
    v55 = v54 < 8l;
    bool v56;
    v56 = v55 == false;
    if (v56){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v55);
    } else {
    }
    assert("Tensor range check" && 0 <= v54 && v54 < 8l);
    assert("Tensor range check" && 0 <= v53 && v53 < 4l);
    int v58;
    v58 = v53 + v47;
    int v59;
    v59 = 12l * v54;
    int v60;
    v60 = v59 + v58;
    float * v61;
    v61 = v13+v60;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v63[1l];
    int v64;
    v64 = 0l;
    while (while_method_4(v64)){
        int v66;
        v66 = 0l;
        while (while_method_5(v66)){
            assert("Tensor range check" && 0 <= v64 && v64 < 1l);
            assert("Tensor range check" && 0 <= v66 && v66 < 8l);
            int v68;
            v68 = 16l * v66;
            int v69;
            v69 = 2048l * v64;
            int v70;
            v70 = v69 + v68;
            float * v71;
            v71 = v0+v70;
            // Pushing the loop unrolling to: 0
            int v73;
            v73 = 0l;
            #pragma unroll
            while (while_method_4(v73)){
                int v75;
                v75 = 0l;
                #pragma unroll
                while (while_method_4(v75)){
                    assert("Tensor range check" && 0 <= v73 && v73 < 1l);
                    assert("Tensor range check" && 0 <= v75 && v75 < 1l);
                    int v77;
                    v77 = v73 + v75;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v78 = v63[v77];
                    wmma::fill_fragment(v78, 0.0f);
                    v75 += 1l ;
                }
                v73 += 1l ;
            }
            int v79;
            v79 = 0l;
            #pragma unroll
            while (while_method_1(v79)){
                assert("Tensor range check" && 0 <= v64 && v64 < 1l);
                int v81;
                v81 = v69 + v4;
                assert("Tensor range check" && 0 <= v79 && v79 < 16l);
                int v82;
                v82 = 8l * v79;
                int v83;
                v83 = v82 + v81;
                float * v84;
                v84 = v3+v83;
                assert("Tensor range check" && 0 <= v66 && v66 < 8l);
                int v86;
                v86 = 2048l * v66;
                int v87;
                v87 = v86 + v2;
                assert("Tensor range check" && 0 <= v79 && v79 < 16l);
                int v88;
                v88 = v82 + v87;
                float * v89;
                v89 = v1+v88;
                int v91;
                v91 = threadIdx.x;
                bool v92;
                v92 = 0l <= v91;
                bool v93;
                v93 = v92 == false;
                if (v93){
                    assert("The index needs to be zero or positive." && v92);
                } else {
                }
                int v95;
                v95 = v91 % 2l;
                int v96;
                v96 = v91 / 2l;
                bool v97;
                v97 = v96 < 16l;
                bool v98;
                v98 = v97 == false;
                if (v98){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v97);
                } else {
                }
                assert("Tensor range check" && 0 <= v96 && v96 < 16l);
                assert("Tensor range check" && 0 <= v95 && v95 < 2l);
                int v100;
                v100 = 4l * v95;
                int v101;
                v101 = 12l * v96;
                int v102;
                v102 = v101 + v100;
                int v103;
                v103 = 128l * v96;
                int v104;
                v104 = v103 + v100;
                float * v105;
                v105 = v13+v102;
                float * v107;
                v107 = v89+v104;
                int v109;
                v109 = 0l;
                #pragma unroll
                while (while_method_4(v109)){
                    int v111;
                    v111 = 0l;
                    #pragma unroll
                    while (while_method_4(v111)){
                        assert("Tensor range check" && 0 <= v109 && v109 < 1l);
                        assert("Tensor range check" && 0 <= v111 && v111 < 1l);
                        int v113;
                        v113 = 8l * v111;
                        int v114;
                        v114 = 192l * v109;
                        int v115;
                        v115 = v114 + v113;
                        int v116;
                        v116 = 2048l * v109;
                        int v117;
                        v117 = v116 + v113;
                        float v118[4l];
                        int v119;
                        v119 = 0l;
                        #pragma unroll
                        while (while_method_3(v119)){
                            assert("Tensor range check" && 0 <= v119 && v119 < 4l);
                            int v121;
                            v121 = v119 + v117;
                            float v122;
                            v122 = v107[v121];
                            float v123;
                            v123 = wmma::__float_to_tf32(v122);
                            assert("Tensor range check" && 0 <= v119 && v119 < 4l);
                            v118[v119] = v123;
                            v119 += 1l ;
                        }
                        int4* v124;
                        v124 = reinterpret_cast<int4*>(v118 + 0l);
                        int4* v125;
                        v125 = reinterpret_cast<int4*>(v105 + v115);
                        assert("Pointer alignment check" && (unsigned long long)(v124) % 4l == 0 && (unsigned long long)(v125) % 4l == 0);
                        *v125 = *v124;
                        v111 += 1l ;
                    }
                    v109 += 1l ;
                }
                int v126;
                v126 = threadIdx.x;
                bool v127;
                v127 = 0l <= v126;
                bool v128;
                v128 = v127 == false;
                if (v128){
                    assert("The index needs to be zero or positive." && v127);
                } else {
                }
                int v130;
                v130 = v126 % 2l;
                int v131;
                v131 = v126 / 2l;
                bool v132;
                v132 = v131 < 16l;
                bool v133;
                v133 = v132 == false;
                if (v133){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v132);
                } else {
                }
                assert("Tensor range check" && 0 <= v131 && v131 < 16l);
                assert("Tensor range check" && 0 <= v130 && v130 < 2l);
                int v135;
                v135 = 4l * v130;
                int v136;
                v136 = 12l * v131;
                int v137;
                v137 = v136 + v135;
                int v138;
                v138 = 128l * v131;
                int v139;
                v139 = v138 + v135;
                float * v140;
                v140 = v11+v137;
                float * v142;
                v142 = v84+v139;
                int v144;
                v144 = 0l;
                #pragma unroll
                while (while_method_4(v144)){
                    int v146;
                    v146 = 0l;
                    #pragma unroll
                    while (while_method_4(v146)){
                        assert("Tensor range check" && 0 <= v144 && v144 < 1l);
                        assert("Tensor range check" && 0 <= v146 && v146 < 1l);
                        int v148;
                        v148 = 8l * v146;
                        int v149;
                        v149 = 192l * v144;
                        int v150;
                        v150 = v149 + v148;
                        int v151;
                        v151 = 2048l * v144;
                        int v152;
                        v152 = v151 + v148;
                        float v153[4l];
                        int v154;
                        v154 = 0l;
                        #pragma unroll
                        while (while_method_3(v154)){
                            assert("Tensor range check" && 0 <= v154 && v154 < 4l);
                            int v156;
                            v156 = v154 + v152;
                            float v157;
                            v157 = v142[v156];
                            float v158;
                            v158 = wmma::__float_to_tf32(v157);
                            assert("Tensor range check" && 0 <= v154 && v154 < 4l);
                            v153[v154] = v158;
                            v154 += 1l ;
                        }
                        int4* v159;
                        v159 = reinterpret_cast<int4*>(v153 + 0l);
                        int4* v160;
                        v160 = reinterpret_cast<int4*>(v140 + v150);
                        assert("Pointer alignment check" && (unsigned long long)(v159) % 4l == 0 && (unsigned long long)(v160) % 4l == 0);
                        *v160 = *v159;
                        v146 += 1l ;
                    }
                    v144 += 1l ;
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v161[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v162[1l];
                int v163;
                v163 = 0l;
                #pragma unroll
                while (while_method_4(v163)){
                    int v165;
                    v165 = 0l;
                    #pragma unroll
                    while (while_method_4(v165)){
                        assert("Tensor range check" && 0 <= v163 && v163 < 1l);
                        assert("Tensor range check" && 0 <= v165 && v165 < 1l);
                        int v167;
                        v167 = v163 + v165;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v168 = v161[v167];
                        assert("Tensor range check" && 0 <= v163 && v163 < 1l);
                        int v169;
                        v169 = 192l * v163;
                        assert("Tensor range check" && 0 <= v165 && v165 < 1l);
                        int v170;
                        v170 = 8l * v165;
                        int v171;
                        v171 = v170 + v169;
                        int v172;
                        v172 = 0l;
                        #pragma unroll
                        while (while_method_6(v172)){
                            int v174;
                            v174 = 0l;
                            #pragma unroll
                            while (while_method_6(v174)){
                                assert("Tensor range check" && 0 <= v172 && v172 < 2l);
                                assert("Tensor range check" && 0 <= v174 && v174 < 2l);
                                int v176;
                                v176 = 96l * v174;
                                int v177;
                                v177 = v176 + v171;
                                int v178;
                                v178 = 4l * v172;
                                int v179;
                                v179 = v178 + v177;
                                float v180;
                                v180 = v45[v179];
                                bool v181;
                                v181 = 0l <= v174;
                                bool v183;
                                if (v181){
                                    bool v182;
                                    v182 = v174 < 2l;
                                    v183 = v182;
                                } else {
                                    v183 = false;
                                }
                                bool v184;
                                v184 = v183 == false;
                                if (v184){
                                    assert("The indices should be inside the range of the dimension." && v183);
                                } else {
                                }
                                bool v186;
                                v186 = 0l <= v172;
                                bool v188;
                                if (v186){
                                    bool v187;
                                    v187 = v172 < 2l;
                                    v188 = v187;
                                } else {
                                    v188 = false;
                                }
                                bool v189;
                                v189 = v188 == false;
                                if (v189){
                                    assert("The indices should be inside the range of the dimension." && v188);
                                } else {
                                }
                                int v191;
                                v191 = v172 * 2l;
                                int v192;
                                v192 = v174 + v191;
                                v168.x[v192] = v180;
                                v174 += 1l ;
                            }
                            v172 += 1l ;
                        }
                        v165 += 1l ;
                    }
                    v163 += 1l ;
                }
                int v193;
                v193 = 0l;
                #pragma unroll
                while (while_method_4(v193)){
                    int v195;
                    v195 = 0l;
                    #pragma unroll
                    while (while_method_4(v195)){
                        assert("Tensor range check" && 0 <= v193 && v193 < 1l);
                        assert("Tensor range check" && 0 <= v195 && v195 < 1l);
                        int v197;
                        v197 = v193 + v195;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v198 = v162[v197];
                        assert("Tensor range check" && 0 <= v193 && v193 < 1l);
                        int v199;
                        v199 = 192l * v193;
                        assert("Tensor range check" && 0 <= v195 && v195 < 1l);
                        int v200;
                        v200 = 8l * v195;
                        int v201;
                        v201 = v200 + v199;
                        int v202;
                        v202 = 0l;
                        #pragma unroll
                        while (while_method_6(v202)){
                            int v204;
                            v204 = 0l;
                            #pragma unroll
                            while (while_method_6(v204)){
                                assert("Tensor range check" && 0 <= v202 && v202 < 2l);
                                assert("Tensor range check" && 0 <= v204 && v204 < 2l);
                                int v206;
                                v206 = 4l * v204;
                                int v207;
                                v207 = v206 + v201;
                                int v208;
                                v208 = 96l * v202;
                                int v209;
                                v209 = v208 + v207;
                                float v210;
                                v210 = v61[v209];
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
                        v195 += 1l ;
                    }
                    v193 += 1l ;
                }
                __syncthreads();
                int v223;
                v223 = 0l;
                #pragma unroll
                while (while_method_4(v223)){
                    int v225;
                    v225 = 0l;
                    #pragma unroll
                    while (while_method_4(v225)){
                        int v227;
                        v227 = 0l;
                        #pragma unroll
                        while (while_method_4(v227)){
                            assert("Tensor range check" && 0 <= v223 && v223 < 1l);
                            assert("Tensor range check" && 0 <= v225 && v225 < 1l);
                            int v229;
                            v229 = v223 + v225;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v230 = v63[v229];
                            assert("Tensor range check" && 0 <= v223 && v223 < 1l);
                            assert("Tensor range check" && 0 <= v227 && v227 < 1l);
                            int v231;
                            v231 = v223 + v227;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v232 = v161[v231];
                            assert("Tensor range check" && 0 <= v225 && v225 < 1l);
                            assert("Tensor range check" && 0 <= v227 && v227 < 1l);
                            int v233;
                            v233 = v225 + v227;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v234 = v162[v233];
                            wmma::mma_sync(v230, v232, v234, v230);
                            v227 += 1l ;
                        }
                        v225 += 1l ;
                    }
                    v223 += 1l ;
                }
                v79 += 1l ;
            }
            int v235;
            v235 = 0l;
            #pragma unroll
            while (while_method_4(v235)){
                int v237;
                v237 = 0l;
                #pragma unroll
                while (while_method_4(v237)){
                    assert("Tensor range check" && 0 <= v235 && v235 < 1l);
                    assert("Tensor range check" && 0 <= v237 && v237 < 1l);
                    int v239;
                    v239 = v235 + v237;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v240 = v63[v239];
                    assert("Tensor range check" && 0 <= v235 && v235 < 1l);
                    assert("Tensor range check" && 0 <= v237 && v237 < 1l);
                    int v241;
                    v241 = 16l * v237;
                    int v242;
                    v242 = 384l * v235;
                    int v243;
                    v243 = v242 + v241;
                    float * v244;
                    v244 = v29+v243;
                    wmma::store_matrix_sync(v244, v240, 24l, wmma::mem_row_major);
                    v237 += 1l ;
                }
                v235 += 1l ;
            }
            __syncthreads();
            int v246;
            v246 = threadIdx.x;
            bool v247;
            v247 = 0l <= v246;
            bool v248;
            v248 = v247 == false;
            if (v248){
                assert("The index needs to be zero or positive." && v247);
            } else {
            }
            int v250;
            v250 = v246 % 4l;
            int v251;
            v251 = v246 / 4l;
            bool v252;
            v252 = v251 < 8l;
            bool v253;
            v253 = v252 == false;
            if (v253){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v252);
            } else {
            }
            assert("Tensor range check" && 0 <= v251 && v251 < 8l);
            assert("Tensor range check" && 0 <= v250 && v250 < 4l);
            int v255;
            v255 = 4l * v250;
            int v256;
            v256 = 128l * v251;
            int v257;
            v257 = v256 + v255;
            int v258;
            v258 = 24l * v251;
            int v259;
            v259 = v258 + v255;
            float * v260;
            v260 = v71+v257;
            float * v262;
            v262 = v15+v259;
            int v264;
            v264 = 0l;
            #pragma unroll
            while (while_method_6(v264)){
                int v266;
                v266 = 0l;
                #pragma unroll
                while (while_method_4(v266)){
                    assert("Tensor range check" && 0 <= v264 && v264 < 2l);
                    assert("Tensor range check" && 0 <= v266 && v266 < 1l);
                    int v268;
                    v268 = 16l * v266;
                    int v269;
                    v269 = 1024l * v264;
                    int v270;
                    v270 = v269 + v268;
                    int v271;
                    v271 = 192l * v264;
                    int v272;
                    v272 = v271 + v268;
                    int4* v273;
                    v273 = reinterpret_cast<int4*>(v262 + v272);
                    int4* v274;
                    v274 = reinterpret_cast<int4*>(v260 + v270);
                    assert("Pointer alignment check" && (unsigned long long)(v273) % 4l == 0 && (unsigned long long)(v274) % 4l == 0);
                    *v274 = *v273;
                    v266 += 1l ;
                }
                v264 += 1l ;
            }
            __syncthreads();
            // Poping the loop unrolling to: 0
            v66 += 1l ;
        }
        v64 += 1l ;
    }
    return ;
}
__device__ void method_1(float * v0, float * v1){
    int v2;
    v2 = threadIdx.x;
    bool v3;
    v3 = 0l <= v2;
    bool v4;
    v4 = v3 == false;
    if (v4){
        assert("The index needs to be zero or positive." && v3);
    } else {
    }
    int v6;
    v6 = v2 % 32l;
    int v7;
    v7 = v2 / 32l;
    bool v8;
    v8 = v7 < 1l;
    bool v9;
    v9 = v8 == false;
    if (v9){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v8);
    } else {
    }
    assert("Tensor range check" && 0 <= v7 && v7 < 1l);
    assert("Tensor range check" && 0 <= v6 && v6 < 32l);
    int v11;
    v11 = 4l * v6;
    int v12;
    v12 = 128l * v7;
    int v13;
    v13 = v12 + v11;
    assert("Tensor range check" && 0 <= v7 && v7 < 1l);
    assert("Tensor range check" && 0 <= v6 && v6 < 32l);
    int v14;
    v14 = 0l;
    while (while_method_1(v14)){
        assert("Tensor range check" && 0 <= v14 && v14 < 16l);
        int v16;
        v16 = 128l * v14;
        int v17;
        v17 = v16 + v13;
        assert("Tensor range check" && 0 <= v14 && v14 < 16l);
        float v18[4l];
        int v19[4l];
        int v20;
        v20 = 0l;
        while (while_method_4(v20)){
            assert("Tensor range check" && 0 <= v20 && v20 < 1l);
            int v22;
            v22 = 4l * v20;
            assert("Tensor range check" && 0 <= v20 && v20 < 1l);
            int v23;
            v23 = 128l * v20;
            int v24;
            v24 = v23 + v17;
            int4* v25;
            v25 = reinterpret_cast<int4*>(v1 + v24);
            int4* v26;
            v26 = reinterpret_cast<int4*>(v18 + v22);
            assert("Pointer alignment check" && (unsigned long long)(v25) % 4l == 0 && (unsigned long long)(v26) % 4l == 0);
            *v26 = *v25;
            v20 += 1l ;
        }
        int v27;
        v27 = 0l;
        while (while_method_4(v27)){
            int v29;
            v29 = 0l;
            while (while_method_3(v29)){
                bool v31;
                v31 = 0l <= v29;
                bool v33;
                if (v31){
                    bool v32;
                    v32 = v29 < 4l;
                    v33 = v32;
                } else {
                    v33 = false;
                }
                bool v34;
                v34 = v33 == false;
                if (v34){
                    assert("The indices should be inside the range of the dimension." && v33);
                } else {
                }
                bool v36;
                v36 = 0l <= v6;
                bool v38;
                if (v36){
                    bool v37;
                    v37 = v6 < 32l;
                    v38 = v37;
                } else {
                    v38 = false;
                }
                bool v39;
                v39 = v38 == false;
                if (v39){
                    assert("The indices should be inside the range of the dimension." && v38);
                } else {
                }
                int v41;
                v41 = v6 * 4l;
                int v42;
                v42 = v29 + v41;
                bool v43;
                v43 = 0l <= v27;
                bool v45;
                if (v43){
                    bool v44;
                    v44 = v27 < 1l;
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
                v48 = v27 * 128l;
                int v49;
                v49 = v42 + v48;
                assert("Tensor range check" && 0 <= v27 && v27 < 1l);
                assert("Tensor range check" && 0 <= v29 && v29 < 4l);
                int v50;
                v50 = 4l * v27;
                int v51;
                v51 = v50 + v29;
                v19[v51] = v49;
                v29 += 1l ;
            }
            v27 += 1l ;
        }
        bool v52;
        v52 = 0l <= v7;
        bool v53;
        v53 = v52 && v8;
        bool v54;
        v54 = v53 == false;
        if (v54){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v53);
        } else {
        }
        bool v56;
        v56 = 0l <= v14;
        bool v58;
        if (v56){
            bool v57;
            v57 = v14 < 16l;
            v58 = v57;
        } else {
            v58 = false;
        }
        bool v59;
        v59 = v58 == false;
        if (v59){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v58);
        } else {
        }
        int v61;
        v61 = v14 + v7;
        float v62[4l];
        int v63;
        v63 = 0l;
        while (while_method_4(v63)){
            int v65;
            v65 = 0l;
            while (while_method_3(v65)){
                assert("Tensor range check" && 0 <= v63 && v63 < 1l);
                assert("Tensor range check" && 0 <= v65 && v65 < 4l);
                int v67;
                v67 = 4l * v63;
                int v68;
                v68 = v67 + v65;
                float v69;
                v69 = v18[v68];
                float v70;
                v70 = v69 * v69;
                assert("Tensor range check" && 0 <= v63 && v63 < 1l);
                assert("Tensor range check" && 0 <= v65 && v65 < 4l);
                v62[v68] = v70;
                v65 += 1l ;
            }
            v63 += 1l ;
        }
        float v71;
        v71 = 0.0f;
        int v72;
        v72 = 0l;
        while (while_method_4(v72)){
            int v74;
            v74 = 0l;
            while (while_method_3(v74)){
                assert("Tensor range check" && 0 <= v72 && v72 < 1l);
                assert("Tensor range check" && 0 <= v74 && v74 < 4l);
                int v76;
                v76 = 4l * v72;
                int v77;
                v77 = v76 + v74;
                float v78;
                v78 = v62[v77];
                float v79;
                v79 = v71 + v78;
                v71 = v79;
                v74 += 1l ;
            }
            v72 += 1l ;
        }
        auto v80 = cooperative_groups::coalesced_threads();
        int v81;
        v81 = threadIdx.x;
        int v82;
        v82 = v81 / 32l;
        auto v83 = cooperative_groups::labeled_partition(v80,v82);
        Closure0 v84{};
        float v85;
        v85 = cooperative_groups::reduce(v83, v71, v84);
        float v86[4l];
        int v87;
        v87 = 0l;
        while (while_method_4(v87)){
            int v89;
            v89 = 0l;
            while (while_method_3(v89)){
                assert("Tensor range check" && 0 <= v87 && v87 < 1l);
                assert("Tensor range check" && 0 <= v89 && v89 < 4l);
                int v91;
                v91 = 4l * v87;
                int v92;
                v92 = v91 + v89;
                float v93;
                v93 = v18[v92];
                bool v94;
                v94 = v85 == 0.0f;
                bool v95;
                v95 = v94 != true;
                float v97;
                if (v95){
                    float v96;
                    v96 = v93 / v85;
                    v97 = v96;
                } else {
                    v97 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v87 && v87 < 1l);
                assert("Tensor range check" && 0 <= v89 && v89 < 4l);
                v86[v92] = v97;
                v89 += 1l ;
            }
            v87 += 1l ;
        }
        int v98;
        v98 = 0l;
        while (while_method_4(v98)){
            assert("Tensor range check" && 0 <= v98 && v98 < 1l);
            int v100;
            v100 = 128l * v98;
            int v101;
            v101 = v100 + v17;
            assert("Tensor range check" && 0 <= v98 && v98 < 1l);
            int v102;
            v102 = 4l * v98;
            int4* v103;
            v103 = reinterpret_cast<int4*>(v86 + v102);
            int4* v104;
            v104 = reinterpret_cast<int4*>(v0 + v101);
            assert("Pointer alignment check" && (unsigned long long)(v103) % 4l == 0 && (unsigned long long)(v104) % 4l == 0);
            *v104 = *v103;
            v98 += 1l ;
        }
        v14 += 1l ;
    }
    __syncthreads();
    return ;
}
__device__ inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 512l;
    return v1;
}
__device__ void method_2(float * v0, float * v1){
    int v2;
    v2 = threadIdx.x;
    int v3;
    v3 = v2;
    while (while_method_7(v3)){
        bool v5;
        v5 = 0l <= v3;
        bool v6;
        v6 = v5 == false;
        if (v6){
            assert("The index needs to be zero or positive." && v5);
        } else {
        }
        int v8;
        v8 = v3 % 32l;
        int v9;
        v9 = v3 / 32l;
        bool v10;
        v10 = v9 < 16l;
        bool v11;
        v11 = v10 == false;
        if (v11){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v10);
        } else {
        }
        assert("Tensor range check" && 0 <= v9 && v9 < 16l);
        assert("Tensor range check" && 0 <= v8 && v8 < 32l);
        int v13;
        v13 = 4l * v8;
        int v14;
        v14 = 128l * v9;
        int v15;
        v15 = v14 + v13;
        assert("Tensor range check" && 0 <= v9 && v9 < 16l);
        assert("Tensor range check" && 0 <= v8 && v8 < 32l);
        float v16[4l];
        float v17[4l];
        int4* v18;
        v18 = reinterpret_cast<int4*>(v1 + v15);
        int4* v19;
        v19 = reinterpret_cast<int4*>(v16 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v18) % 4l == 0 && (unsigned long long)(v19) % 4l == 0);
        *v19 = *v18;
        // Pushing the loop unrolling to: 0
        int v20;
        v20 = 0l;
        #pragma unroll
        while (while_method_3(v20)){
            assert("Tensor range check" && 0 <= v20 && v20 < 4l);
            float v22;
            v22 = v16[v20];
            bool v23;
            v23 = 0.0f >= v22;
            float v24;
            if (v23){
                v24 = 0.0f;
            } else {
                v24 = v22;
            }
            assert("Tensor range check" && 0 <= v20 && v20 < 4l);
            v17[v20] = v24;
            v20 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v25;
        v25 = reinterpret_cast<int4*>(v17 + 0l);
        int4* v26;
        v26 = reinterpret_cast<int4*>(v0 + v15);
        assert("Pointer alignment check" && (unsigned long long)(v25) % 4l == 0 && (unsigned long long)(v26) % 4l == 0);
        *v26 = *v25;
        v3 += 32l ;
    }
    __syncthreads();
    return ;
}
__device__ void method_3(float * v0, float * v1, int v2, float * v3){
    unsigned int v4;
    v4 = 0ul;
    asm("mov.u32 %0, %dynamic_smem_size;" : "=r"(v4));
    unsigned long long v5;
    v5 = (unsigned long long)v4;
    bool v6;
    v6 = 1536ull <= v5;
    bool v7;
    v7 = v6 == false;
    if (v7){
        assert("The shared memory used in the matmult node is lower than the allocated amount." && v6);
    } else {
    }
    extern __shared__ unsigned char v9[];
    float * v10;
    v10 = reinterpret_cast<float *>(&v9[0ull]);
    float * v12;
    v12 = reinterpret_cast<float *>(&v9[768ull]);
    float * v14;
    v14 = reinterpret_cast<float *>(&v9[0ull]);
    int v16;
    v16 = threadIdx.x;
    int v17;
    v17 = v16 / 32l;
    bool v18;
    v18 = 0l <= v17;
    bool v19;
    v19 = v18 == false;
    if (v19){
        assert("The index needs to be zero or positive." && v18);
    } else {
    }
    int v21;
    v21 = v17 % 1l;
    bool v22;
    v22 = v17 < 1l;
    bool v23;
    v23 = v22 == false;
    if (v23){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v22);
    } else {
    }
    assert("Tensor range check" && 0 <= v17 && v17 < 1l);
    assert("Tensor range check" && 0 <= v21 && v21 < 1l);
    int v25;
    v25 = 16l * v21;
    int v26;
    v26 = 384l * v17;
    int v27;
    v27 = v26 + v25;
    float * v28;
    v28 = v14+v27;
    assert("Tensor range check" && 0 <= v17 && v17 < 1l);
    int v30;
    v30 = 192l * v17;
    int v31;
    v31 = threadIdx.x;
    int v32;
    v32 = v31 % 32l;
    bool v33;
    v33 = 0l <= v32;
    bool v34;
    v34 = v33 == false;
    if (v34){
        assert("The index needs to be zero or positive." && v33);
    } else {
    }
    int v36;
    v36 = v32 % 4l;
    int v37;
    v37 = v32 / 4l;
    bool v38;
    v38 = v37 < 8l;
    bool v39;
    v39 = v38 == false;
    if (v39){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v38);
    } else {
    }
    assert("Tensor range check" && 0 <= v37 && v37 < 8l);
    assert("Tensor range check" && 0 <= v36 && v36 < 4l);
    int v41;
    v41 = v36 + v30;
    int v42;
    v42 = 12l * v37;
    int v43;
    v43 = v42 + v41;
    float * v44;
    v44 = v10+v43;
    assert("Tensor range check" && 0 <= v21 && v21 < 1l);
    int v46;
    v46 = 192l * v21;
    int v47;
    v47 = threadIdx.x;
    int v48;
    v48 = v47 % 32l;
    bool v49;
    v49 = 0l <= v48;
    bool v50;
    v50 = v49 == false;
    if (v50){
        assert("The index needs to be zero or positive." && v49);
    } else {
    }
    int v52;
    v52 = v48 % 4l;
    int v53;
    v53 = v48 / 4l;
    bool v54;
    v54 = v53 < 8l;
    bool v55;
    v55 = v54 == false;
    if (v55){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v54);
    } else {
    }
    assert("Tensor range check" && 0 <= v53 && v53 < 8l);
    assert("Tensor range check" && 0 <= v52 && v52 < 4l);
    int v57;
    v57 = v52 + v46;
    int v58;
    v58 = 12l * v53;
    int v59;
    v59 = v58 + v57;
    float * v60;
    v60 = v12+v59;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v62[1l];
    int v63;
    v63 = 0l;
    while (while_method_4(v63)){
        int v65;
        v65 = 0l;
        while (while_method_5(v65)){
            assert("Tensor range check" && 0 <= v63 && v63 < 1l);
            assert("Tensor range check" && 0 <= v65 && v65 < 8l);
            int v67;
            v67 = 16l * v65;
            int v68;
            v68 = 2048l * v63;
            int v69;
            v69 = v68 + v67;
            float * v70;
            v70 = v0+v69;
            // Pushing the loop unrolling to: 0
            int v72;
            v72 = 0l;
            #pragma unroll
            while (while_method_4(v72)){
                int v74;
                v74 = 0l;
                #pragma unroll
                while (while_method_4(v74)){
                    assert("Tensor range check" && 0 <= v72 && v72 < 1l);
                    assert("Tensor range check" && 0 <= v74 && v74 < 1l);
                    int v76;
                    v76 = v72 + v74;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v77 = v62[v76];
                    wmma::fill_fragment(v77, 0.0f);
                    v74 += 1l ;
                }
                v72 += 1l ;
            }
            int v78;
            v78 = 0l;
            #pragma unroll
            while (while_method_1(v78)){
                assert("Tensor range check" && 0 <= v63 && v63 < 1l);
                assert("Tensor range check" && 0 <= v78 && v78 < 16l);
                int v80;
                v80 = 8l * v78;
                int v81;
                v81 = v80 + v68;
                float * v82;
                v82 = v3+v81;
                assert("Tensor range check" && 0 <= v65 && v65 < 8l);
                int v84;
                v84 = 2048l * v65;
                int v85;
                v85 = v84 + v2;
                assert("Tensor range check" && 0 <= v78 && v78 < 16l);
                int v86;
                v86 = v80 + v85;
                float * v87;
                v87 = v1+v86;
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
                v93 = v89 % 2l;
                int v94;
                v94 = v89 / 2l;
                bool v95;
                v95 = v94 < 16l;
                bool v96;
                v96 = v95 == false;
                if (v96){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v95);
                } else {
                }
                assert("Tensor range check" && 0 <= v94 && v94 < 16l);
                assert("Tensor range check" && 0 <= v93 && v93 < 2l);
                int v98;
                v98 = 4l * v93;
                int v99;
                v99 = 12l * v94;
                int v100;
                v100 = v99 + v98;
                int v101;
                v101 = 128l * v94;
                int v102;
                v102 = v101 + v98;
                float * v103;
                v103 = v12+v100;
                float * v105;
                v105 = v87+v102;
                int v107;
                v107 = 0l;
                #pragma unroll
                while (while_method_4(v107)){
                    int v109;
                    v109 = 0l;
                    #pragma unroll
                    while (while_method_4(v109)){
                        assert("Tensor range check" && 0 <= v107 && v107 < 1l);
                        assert("Tensor range check" && 0 <= v109 && v109 < 1l);
                        int v111;
                        v111 = 8l * v109;
                        int v112;
                        v112 = 192l * v107;
                        int v113;
                        v113 = v112 + v111;
                        int v114;
                        v114 = 2048l * v107;
                        int v115;
                        v115 = v114 + v111;
                        float v116[4l];
                        int v117;
                        v117 = 0l;
                        #pragma unroll
                        while (while_method_3(v117)){
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
                v128 = v124 % 2l;
                int v129;
                v129 = v124 / 2l;
                bool v130;
                v130 = v129 < 16l;
                bool v131;
                v131 = v130 == false;
                if (v131){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v130);
                } else {
                }
                assert("Tensor range check" && 0 <= v129 && v129 < 16l);
                assert("Tensor range check" && 0 <= v128 && v128 < 2l);
                int v133;
                v133 = 4l * v128;
                int v134;
                v134 = 12l * v129;
                int v135;
                v135 = v134 + v133;
                int v136;
                v136 = 128l * v129;
                int v137;
                v137 = v136 + v133;
                float * v138;
                v138 = v10+v135;
                float * v140;
                v140 = v82+v137;
                int v142;
                v142 = 0l;
                #pragma unroll
                while (while_method_4(v142)){
                    int v144;
                    v144 = 0l;
                    #pragma unroll
                    while (while_method_4(v144)){
                        assert("Tensor range check" && 0 <= v142 && v142 < 1l);
                        assert("Tensor range check" && 0 <= v144 && v144 < 1l);
                        int v146;
                        v146 = 8l * v144;
                        int v147;
                        v147 = 192l * v142;
                        int v148;
                        v148 = v147 + v146;
                        int v149;
                        v149 = 2048l * v142;
                        int v150;
                        v150 = v149 + v146;
                        float v151[4l];
                        int v152;
                        v152 = 0l;
                        #pragma unroll
                        while (while_method_3(v152)){
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
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v159[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v160[1l];
                int v161;
                v161 = 0l;
                #pragma unroll
                while (while_method_4(v161)){
                    int v163;
                    v163 = 0l;
                    #pragma unroll
                    while (while_method_4(v163)){
                        assert("Tensor range check" && 0 <= v161 && v161 < 1l);
                        assert("Tensor range check" && 0 <= v163 && v163 < 1l);
                        int v165;
                        v165 = v161 + v163;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v166 = v159[v165];
                        assert("Tensor range check" && 0 <= v161 && v161 < 1l);
                        int v167;
                        v167 = 192l * v161;
                        assert("Tensor range check" && 0 <= v163 && v163 < 1l);
                        int v168;
                        v168 = 8l * v163;
                        int v169;
                        v169 = v168 + v167;
                        int v170;
                        v170 = 0l;
                        #pragma unroll
                        while (while_method_6(v170)){
                            int v172;
                            v172 = 0l;
                            #pragma unroll
                            while (while_method_6(v172)){
                                assert("Tensor range check" && 0 <= v170 && v170 < 2l);
                                assert("Tensor range check" && 0 <= v172 && v172 < 2l);
                                int v174;
                                v174 = 96l * v172;
                                int v175;
                                v175 = v174 + v169;
                                int v176;
                                v176 = 4l * v170;
                                int v177;
                                v177 = v176 + v175;
                                float v178;
                                v178 = v44[v177];
                                bool v179;
                                v179 = 0l <= v172;
                                bool v181;
                                if (v179){
                                    bool v180;
                                    v180 = v172 < 2l;
                                    v181 = v180;
                                } else {
                                    v181 = false;
                                }
                                bool v182;
                                v182 = v181 == false;
                                if (v182){
                                    assert("The indices should be inside the range of the dimension." && v181);
                                } else {
                                }
                                bool v184;
                                v184 = 0l <= v170;
                                bool v186;
                                if (v184){
                                    bool v185;
                                    v185 = v170 < 2l;
                                    v186 = v185;
                                } else {
                                    v186 = false;
                                }
                                bool v187;
                                v187 = v186 == false;
                                if (v187){
                                    assert("The indices should be inside the range of the dimension." && v186);
                                } else {
                                }
                                int v189;
                                v189 = v170 * 2l;
                                int v190;
                                v190 = v172 + v189;
                                v166.x[v190] = v178;
                                v172 += 1l ;
                            }
                            v170 += 1l ;
                        }
                        v163 += 1l ;
                    }
                    v161 += 1l ;
                }
                int v191;
                v191 = 0l;
                #pragma unroll
                while (while_method_4(v191)){
                    int v193;
                    v193 = 0l;
                    #pragma unroll
                    while (while_method_4(v193)){
                        assert("Tensor range check" && 0 <= v191 && v191 < 1l);
                        assert("Tensor range check" && 0 <= v193 && v193 < 1l);
                        int v195;
                        v195 = v191 + v193;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v196 = v160[v195];
                        assert("Tensor range check" && 0 <= v191 && v191 < 1l);
                        int v197;
                        v197 = 192l * v191;
                        assert("Tensor range check" && 0 <= v193 && v193 < 1l);
                        int v198;
                        v198 = 8l * v193;
                        int v199;
                        v199 = v198 + v197;
                        int v200;
                        v200 = 0l;
                        #pragma unroll
                        while (while_method_6(v200)){
                            int v202;
                            v202 = 0l;
                            #pragma unroll
                            while (while_method_6(v202)){
                                assert("Tensor range check" && 0 <= v200 && v200 < 2l);
                                assert("Tensor range check" && 0 <= v202 && v202 < 2l);
                                int v204;
                                v204 = 4l * v202;
                                int v205;
                                v205 = v204 + v199;
                                int v206;
                                v206 = 96l * v200;
                                int v207;
                                v207 = v206 + v205;
                                float v208;
                                v208 = v60[v207];
                                bool v209;
                                v209 = 0l <= v202;
                                bool v211;
                                if (v209){
                                    bool v210;
                                    v210 = v202 < 2l;
                                    v211 = v210;
                                } else {
                                    v211 = false;
                                }
                                bool v212;
                                v212 = v211 == false;
                                if (v212){
                                    assert("The indices should be inside the range of the dimension." && v211);
                                } else {
                                }
                                bool v214;
                                v214 = 0l <= v200;
                                bool v216;
                                if (v214){
                                    bool v215;
                                    v215 = v200 < 2l;
                                    v216 = v215;
                                } else {
                                    v216 = false;
                                }
                                bool v217;
                                v217 = v216 == false;
                                if (v217){
                                    assert("The indices should be inside the range of the dimension." && v216);
                                } else {
                                }
                                int v219;
                                v219 = v200 * 2l;
                                int v220;
                                v220 = v202 + v219;
                                v196.x[v220] = v208;
                                v202 += 1l ;
                            }
                            v200 += 1l ;
                        }
                        v193 += 1l ;
                    }
                    v191 += 1l ;
                }
                __syncthreads();
                int v221;
                v221 = 0l;
                #pragma unroll
                while (while_method_4(v221)){
                    int v223;
                    v223 = 0l;
                    #pragma unroll
                    while (while_method_4(v223)){
                        int v225;
                        v225 = 0l;
                        #pragma unroll
                        while (while_method_4(v225)){
                            assert("Tensor range check" && 0 <= v221 && v221 < 1l);
                            assert("Tensor range check" && 0 <= v223 && v223 < 1l);
                            int v227;
                            v227 = v221 + v223;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v228 = v62[v227];
                            assert("Tensor range check" && 0 <= v221 && v221 < 1l);
                            assert("Tensor range check" && 0 <= v225 && v225 < 1l);
                            int v229;
                            v229 = v221 + v225;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v230 = v159[v229];
                            assert("Tensor range check" && 0 <= v223 && v223 < 1l);
                            assert("Tensor range check" && 0 <= v225 && v225 < 1l);
                            int v231;
                            v231 = v223 + v225;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v232 = v160[v231];
                            wmma::mma_sync(v228, v230, v232, v228);
                            v225 += 1l ;
                        }
                        v223 += 1l ;
                    }
                    v221 += 1l ;
                }
                v78 += 1l ;
            }
            int v233;
            v233 = 0l;
            #pragma unroll
            while (while_method_4(v233)){
                int v235;
                v235 = 0l;
                #pragma unroll
                while (while_method_4(v235)){
                    assert("Tensor range check" && 0 <= v233 && v233 < 1l);
                    assert("Tensor range check" && 0 <= v235 && v235 < 1l);
                    int v237;
                    v237 = v233 + v235;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v238 = v62[v237];
                    assert("Tensor range check" && 0 <= v233 && v233 < 1l);
                    assert("Tensor range check" && 0 <= v235 && v235 < 1l);
                    int v239;
                    v239 = 16l * v235;
                    int v240;
                    v240 = 384l * v233;
                    int v241;
                    v241 = v240 + v239;
                    float * v242;
                    v242 = v28+v241;
                    wmma::store_matrix_sync(v242, v238, 24l, wmma::mem_row_major);
                    v235 += 1l ;
                }
                v233 += 1l ;
            }
            __syncthreads();
            int v244;
            v244 = threadIdx.x;
            bool v245;
            v245 = 0l <= v244;
            bool v246;
            v246 = v245 == false;
            if (v246){
                assert("The index needs to be zero or positive." && v245);
            } else {
            }
            int v248;
            v248 = v244 % 4l;
            int v249;
            v249 = v244 / 4l;
            bool v250;
            v250 = v249 < 8l;
            bool v251;
            v251 = v250 == false;
            if (v251){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v250);
            } else {
            }
            assert("Tensor range check" && 0 <= v249 && v249 < 8l);
            assert("Tensor range check" && 0 <= v248 && v248 < 4l);
            int v253;
            v253 = 4l * v248;
            int v254;
            v254 = 128l * v249;
            int v255;
            v255 = v254 + v253;
            int v256;
            v256 = 24l * v249;
            int v257;
            v257 = v256 + v253;
            float * v258;
            v258 = v70+v255;
            float * v260;
            v260 = v14+v257;
            int v262;
            v262 = 0l;
            #pragma unroll
            while (while_method_6(v262)){
                int v264;
                v264 = 0l;
                #pragma unroll
                while (while_method_4(v264)){
                    assert("Tensor range check" && 0 <= v262 && v262 < 2l);
                    assert("Tensor range check" && 0 <= v264 && v264 < 1l);
                    int v266;
                    v266 = 16l * v264;
                    int v267;
                    v267 = 1024l * v262;
                    int v268;
                    v268 = v267 + v266;
                    int v269;
                    v269 = 192l * v262;
                    int v270;
                    v270 = v269 + v266;
                    int4* v271;
                    v271 = reinterpret_cast<int4*>(v260 + v270);
                    int4* v272;
                    v272 = reinterpret_cast<int4*>(v258 + v268);
                    assert("Pointer alignment check" && (unsigned long long)(v271) % 4l == 0 && (unsigned long long)(v272) % 4l == 0);
                    *v272 = *v271;
                    v264 += 1l ;
                }
                v262 += 1l ;
            }
            __syncthreads();
            // Poping the loop unrolling to: 0
            v65 += 1l ;
        }
        v63 += 1l ;
    }
    return ;
}
__device__ void method_4(int * v0, int v1, float * v2, curandStatePhilox4_32_10_t & v3){
    int v4;
    v4 = threadIdx.x;
    bool v5;
    v5 = 0l <= v4;
    bool v6;
    v6 = v5 == false;
    if (v6){
        assert("The index needs to be zero or positive." && v5);
    } else {
    }
    int v8;
    v8 = v4 % 32l;
    int v9;
    v9 = v4 / 32l;
    bool v10;
    v10 = v9 < 1l;
    bool v11;
    v11 = v10 == false;
    if (v11){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v10);
    } else {
    }
    assert("Tensor range check" && 0 <= v9 && v9 < 1l);
    assert("Tensor range check" && 0 <= v8 && v8 < 32l);
    int v13;
    v13 = 4l * v8;
    int v14;
    v14 = 128l * v9;
    int v15;
    v15 = v14 + v13;
    assert("Tensor range check" && 0 <= v9 && v9 < 1l);
    int v16;
    v16 = v9 + v1;
    int v17;
    v17 = 0l;
    while (while_method_1(v17)){
        assert("Tensor range check" && 0 <= v17 && v17 < 16l);
        int v19;
        v19 = 128l * v17;
        int v20;
        v20 = v19 + v15;
        float v21[4l];
        int v22[4l];
        int v23;
        v23 = 0l;
        while (while_method_4(v23)){
            assert("Tensor range check" && 0 <= v23 && v23 < 1l);
            int v25;
            v25 = 4l * v23;
            assert("Tensor range check" && 0 <= v23 && v23 < 1l);
            int v26;
            v26 = 128l * v23;
            int v27;
            v27 = v26 + v20;
            int4* v28;
            v28 = reinterpret_cast<int4*>(v2 + v27);
            int4* v29;
            v29 = reinterpret_cast<int4*>(v21 + v25);
            assert("Pointer alignment check" && (unsigned long long)(v28) % 4l == 0 && (unsigned long long)(v29) % 4l == 0);
            *v29 = *v28;
            v23 += 1l ;
        }
        int v30;
        v30 = 0l;
        while (while_method_4(v30)){
            int v32;
            v32 = 0l;
            while (while_method_3(v32)){
                bool v34;
                v34 = 0l <= v32;
                bool v36;
                if (v34){
                    bool v35;
                    v35 = v32 < 4l;
                    v36 = v35;
                } else {
                    v36 = false;
                }
                bool v37;
                v37 = v36 == false;
                if (v37){
                    assert("The indices should be inside the range of the dimension." && v36);
                } else {
                }
                bool v39;
                v39 = 0l <= v8;
                bool v41;
                if (v39){
                    bool v40;
                    v40 = v8 < 32l;
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
                int v44;
                v44 = v8 * 4l;
                int v45;
                v45 = v32 + v44;
                bool v46;
                v46 = 0l <= v30;
                bool v48;
                if (v46){
                    bool v47;
                    v47 = v30 < 1l;
                    v48 = v47;
                } else {
                    v48 = false;
                }
                bool v49;
                v49 = v48 == false;
                if (v49){
                    assert("The indices should be inside the range of the dimension." && v48);
                } else {
                }
                int v51;
                v51 = v30 * 128l;
                int v52;
                v52 = v45 + v51;
                assert("Tensor range check" && 0 <= v30 && v30 < 1l);
                assert("Tensor range check" && 0 <= v32 && v32 < 4l);
                int v53;
                v53 = 4l * v30;
                int v54;
                v54 = v53 + v32;
                v22[v54] = v52;
                v32 += 1l ;
            }
            v30 += 1l ;
        }
        bool v55;
        v55 = 0l <= v9;
        bool v56;
        v56 = v55 && v10;
        bool v57;
        v57 = v56 == false;
        if (v57){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v56);
        } else {
        }
        bool v59;
        v59 = 0l <= v17;
        bool v61;
        if (v59){
            bool v60;
            v60 = v17 < 16l;
            v61 = v60;
        } else {
            v61 = false;
        }
        bool v62;
        v62 = v61 == false;
        if (v62){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v61);
        } else {
        }
        int v64;
        v64 = v17 + v9;
        float v65;
        v65 = 0.0f;
        int v66;
        v66 = 0l;
        while (while_method_4(v66)){
            int v68;
            v68 = 0l;
            while (while_method_3(v68)){
                assert("Tensor range check" && 0 <= v66 && v66 < 1l);
                assert("Tensor range check" && 0 <= v68 && v68 < 4l);
                int v70;
                v70 = 4l * v66;
                int v71;
                v71 = v70 + v68;
                float v72;
                v72 = v21[v71];
                float v73;
                v73 = v65 + v72;
                v65 = v73;
                v68 += 1l ;
            }
            v66 += 1l ;
        }
        auto v74 = cooperative_groups::coalesced_threads();
        int v75;
        v75 = threadIdx.x;
        int v76;
        v76 = v75 / 32l;
        auto v77 = cooperative_groups::labeled_partition(v74,v76);
        Closure0 v78{};
        float v79;
        v79 = cooperative_groups::reduce(v77, v65, v78);
        float v80;
        v80 = v79 / 128.0f;
        float v81[4l];
        int v82;
        v82 = 0l;
        while (while_method_4(v82)){
            int v84;
            v84 = 0l;
            while (while_method_3(v84)){
                assert("Tensor range check" && 0 <= v82 && v82 < 1l);
                assert("Tensor range check" && 0 <= v84 && v84 < 4l);
                int v86;
                v86 = 4l * v82;
                int v87;
                v87 = v86 + v84;
                float v88;
                v88 = v21[v87];
                float v89;
                v89 = v88 - v80;
                float v90;
                v90 = exp(v89);
                assert("Tensor range check" && 0 <= v82 && v82 < 1l);
                assert("Tensor range check" && 0 <= v84 && v84 < 4l);
                v81[v87] = v90;
                v84 += 1l ;
            }
            v82 += 1l ;
        }
        float v91;
        v91 = 0.0f;
        int v92;
        v92 = 0l;
        while (while_method_4(v92)){
            int v94;
            v94 = 0l;
            while (while_method_3(v94)){
                assert("Tensor range check" && 0 <= v92 && v92 < 1l);
                assert("Tensor range check" && 0 <= v94 && v94 < 4l);
                int v96;
                v96 = 4l * v92;
                int v97;
                v97 = v96 + v94;
                float v98;
                v98 = v81[v97];
                float v99;
                v99 = v91 + v98;
                v91 = v99;
                v94 += 1l ;
            }
            v92 += 1l ;
        }
        auto v100 = cooperative_groups::coalesced_threads();
        int v101;
        v101 = threadIdx.x;
        int v102;
        v102 = v101 / 32l;
        auto v103 = cooperative_groups::labeled_partition(v100,v102);
        float v104;
        v104 = cooperative_groups::reduce(v103, v91, v78);
        float v105[4l];
        int v106;
        v106 = 0l;
        while (while_method_4(v106)){
            int v108;
            v108 = 0l;
            while (while_method_3(v108)){
                assert("Tensor range check" && 0 <= v106 && v106 < 1l);
                assert("Tensor range check" && 0 <= v108 && v108 < 4l);
                int v110;
                v110 = 4l * v106;
                int v111;
                v111 = v110 + v108;
                float v112;
                v112 = v81[v111];
                bool v113;
                v113 = v104 == 0.0f;
                bool v114;
                v114 = v113 != true;
                float v116;
                if (v114){
                    float v115;
                    v115 = v112 / v104;
                    v116 = v115;
                } else {
                    v116 = 0.0078125f;
                }
                assert("Tensor range check" && 0 <= v106 && v106 < 1l);
                assert("Tensor range check" && 0 <= v108 && v108 < 4l);
                v105[v111] = v116;
                v108 += 1l ;
            }
            v106 += 1l ;
        }
        float v117[4l];
        float v118;
        v118 = 0.0f;
        int v119;
        v119 = 0l;
        while (while_method_4(v119)){
            assert("Tensor range check" && 0 <= v119 && v119 < 1l);
            int v121;
            v121 = 4l * v119;
            assert("Tensor range check" && 0 <= v119 && v119 < 1l);
            int v122; float v123;
            Tuple0 tmp0 = Tuple0{0l, 0.0f};
            v122 = tmp0.v0; v123 = tmp0.v1;
            while (while_method_3(v122)){
                assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                int v125;
                v125 = v122 + v121;
                float v126;
                v126 = v105[v125];
                float v127;
                v127 = v123 + v126;
                v123 = v127;
                v122 += 1l ;
            }
            auto v128 = cooperative_groups::coalesced_threads();
            int v129;
            v129 = threadIdx.x;
            int v130;
            v130 = v129 / 32l;
            auto v131 = cooperative_groups::labeled_partition(v128,v130);
            Closure1 v132{};
            float v133;
            v133 = cooperative_groups::inclusive_scan(v131, v123, v132);
            float v134;
            v134 = v131.shfl_up(v133,1);
            bool v135;
            v135 = v131.thread_rank() == 0;
            float v136;
            if (v135){
                v136 = 0.0f;
            } else {
                v136 = v134;
            }
            float v137;
            v137 = v131.shfl(v133,v131.num_threads()-1);
            float v138;
            v138 = v118 + v136;
            int v139; float v140;
            Tuple0 tmp1 = Tuple0{0l, v138};
            v139 = tmp1.v0; v140 = tmp1.v1;
            while (while_method_3(v139)){
                assert("Tensor range check" && 0 <= v139 && v139 < 4l);
                int v142;
                v142 = v139 + v121;
                float v143;
                v143 = v105[v142];
                float v144;
                v144 = v140 + v143;
                assert("Tensor range check" && 0 <= v139 && v139 < 4l);
                v117[v142] = v144;
                v140 = v144;
                v139 += 1l ;
            }
            float v145;
            v145 = v118 + v137;
            v118 = v145;
            v119 += 1l ;
        }
        float v146;
        v146 = curand_uniform(&v3);
        float v147[4l];
        int v148;
        v148 = 0l;
        while (while_method_4(v148)){
            int v150;
            v150 = 0l;
            while (while_method_3(v150)){
                assert("Tensor range check" && 0 <= v148 && v148 < 1l);
                assert("Tensor range check" && 0 <= v150 && v150 < 4l);
                int v152;
                v152 = 4l * v148;
                int v153;
                v153 = v152 + v150;
                float v154;
                v154 = v117[v153];
                float v155;
                v155 = v154 - v146;
                assert("Tensor range check" && 0 <= v148 && v148 < 1l);
                assert("Tensor range check" && 0 <= v150 && v150 < 4l);
                v147[v153] = v155;
                v150 += 1l ;
            }
            v148 += 1l ;
        }
        float v156; int v157;
        Tuple1 tmp2 = Tuple1{-1.0f / 0.0f, 0l};
        v156 = tmp2.v0; v157 = tmp2.v1;
        int v158;
        v158 = 0l;
        while (while_method_4(v158)){
            int v160;
            v160 = 0l;
            while (while_method_3(v160)){
                assert("Tensor range check" && 0 <= v158 && v158 < 1l);
                assert("Tensor range check" && 0 <= v160 && v160 < 4l);
                int v162;
                v162 = 4l * v158;
                int v163;
                v163 = v162 + v160;
                float v164;
                v164 = v147[v163];
                int v165;
                v165 = v22[v163];
                bool v166;
                v166 = v156 >= 0.0f;
                bool v168;
                if (v166){
                    bool v167;
                    v167 = v164 >= 0.0f;
                    v168 = v167;
                } else {
                    v168 = false;
                }
                float v177; int v178;
                if (v168){
                    bool v169;
                    v169 = v156 <= v164;
                    if (v169){
                        v177 = v156; v178 = v157;
                    } else {
                        v177 = v164; v178 = v165;
                    }
                } else {
                    if (v166){
                        v177 = v156; v178 = v157;
                    } else {
                        bool v172;
                        v172 = v164 >= 0.0f;
                        if (v172){
                            v177 = v164; v178 = v165;
                        } else {
                            v177 = v156; v178 = v157;
                        }
                    }
                }
                v156 = v177;
                v157 = v178;
                v160 += 1l ;
            }
            v158 += 1l ;
        }
        auto v179 = cooperative_groups::coalesced_threads();
        int v180;
        v180 = threadIdx.x;
        int v181;
        v181 = v180 / 32l;
        auto v182 = cooperative_groups::labeled_partition(v179,v181);
        Closure2 v183{};
        float v184; int v185;
        Tuple1 tmp3 = cooperative_groups::reduce(v182, Tuple1{v156, v157}, v183);
        v184 = tmp3.v0; v185 = tmp3.v1;
        assert("Tensor range check" && 0 <= v17 && v17 < 16l);
        int v186;
        v186 = v17 + v16;
        v0[v186] = v185;
        v17 += 1l ;
    }
    __syncthreads();
    return ;
}
__device__ inline bool while_method_8(int v0){
    bool v1;
    v1 = v0 < 64l;
    return v1;
}
__device__ void write_7(const char * v0){
    const char * v1;
    v1 = "%s";
    printf(v1,v0);
    return ;
}
__device__ void write_6(){
    const char * v0;
    v0 = "Call";
    return write_7(v0);
}
__device__ void write_8(){
    const char * v0;
    v0 = "Fold";
    return write_7(v0);
}
__device__ void write_9(){
    const char * v0;
    v0 = "Raise";
    return write_7(v0);
}
__device__ void write_5(Union1 v0){
    switch (v0.tag) {
        case 0: { // Call
            return write_6();
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
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1) {
    float * v2;
    v2 = reinterpret_cast<float *>(&v0[0ull]);
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 1l);
    int v5;
    v5 = 2048l * v4;
    int v6;
    v6 = threadIdx.x;
    int v7;
    v7 = v6;
    while (while_method_0(v7)){
        bool v9;
        v9 = 0l <= v7;
        bool v10;
        v10 = v9 == false;
        if (v10){
            assert("The index needs to be zero or positive." && v9);
        } else {
        }
        int v12;
        v12 = v7 % 128l;
        int v13;
        v13 = v7 / 128l;
        bool v14;
        v14 = v13 < 16l;
        bool v15;
        v15 = v14 == false;
        if (v15){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v14);
        } else {
        }
        assert("Tensor range check" && 0 <= v13 && v13 < 16l);
        assert("Tensor range check" && 0 <= v12 && v12 < 128l);
        int v17;
        v17 = v12 + v5;
        int v18;
        v18 = 128l * v13;
        int v19;
        v19 = v18 + v17;
        v2[v19] = 0.0f;
        v7 += 32l ;
    }
    __syncthreads();
    static_array_list<Union0,10l> v20;
    v20 = static_array_list<Union0,10l>{};
    v20.unsafe_set_length(10l);
    Union2 v22;
    v22 = Union2{Union2_1{}};
    Union0 v23;
    v23 = Union0{Union0_1{v22}};
    v20[0l] = v23;
    Union1 v26;
    v26 = Union1{Union1_0{}};
    Union0 v27;
    v27 = Union0{Union0_0{v26}};
    v20[1l] = v27;
    Union1 v30;
    v30 = Union1{Union1_2{}};
    Union0 v31;
    v31 = Union0{Union0_0{v30}};
    v20[2l] = v31;
    Union1 v34;
    v34 = Union1{Union1_2{}};
    Union0 v35;
    v35 = Union0{Union0_0{v34}};
    v20[3l] = v35;
    Union1 v38;
    v38 = Union1{Union1_0{}};
    Union0 v39;
    v39 = Union0{Union0_0{v38}};
    v20[4l] = v39;
    Union2 v42;
    v42 = Union2{Union2_2{}};
    Union0 v43;
    v43 = Union0{Union0_1{v42}};
    v20[5l] = v43;
    Union1 v46;
    v46 = Union1{Union1_0{}};
    Union0 v47;
    v47 = Union0{Union0_0{v46}};
    v20[6l] = v47;
    Union1 v50;
    v50 = Union1{Union1_2{}};
    Union0 v51;
    v51 = Union0{Union0_0{v50}};
    v20[7l] = v51;
    Union1 v54;
    v54 = Union1{Union1_2{}};
    Union0 v55;
    v55 = Union0{Union0_0{v54}};
    v20[8l] = v55;
    Union1 v58;
    v58 = Union1{Union1_0{}};
    Union0 v59;
    v59 = Union0{Union0_0{v58}};
    v20[9l] = v59;
    int v62;
    v62 = threadIdx.x;
    int v63;
    v63 = v62;
    while (while_method_1(v63)){
        bool v65;
        v65 = 0l <= v63;
        bool v66;
        v66 = v65 == false;
        if (v66){
            assert("The index needs to be zero or positive." && v65);
        } else {
        }
        bool v68;
        v68 = v63 < 16l;
        bool v69;
        v69 = v68 == false;
        if (v69){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v68);
        } else {
        }
        assert("Tensor range check" && 0 <= v63 && v63 < 16l);
        int v71;
        v71 = 128l * v63;
        int v72;
        v72 = v71 + v5;
        float * v73;
        v73 = v2+v72;
        int v75;
        v75 = v20.length;
        bool v76;
        v76 = v75 == 0l;
        if (v76){
            v73[0l] = 1.0f;
        } else {
        }
        int v77;
        v77 = v20.length;
        int v78;
        v78 = 0l;
        while (while_method_2(v77, v78)){
            Union0 v80;
            v80 = v20[v78];
            int v82;
            v82 = v78 * 6l;
            int v83;
            v83 = 1l + v82;
            switch (v80.tag) {
                case 0: { // C1of2
                    Union1 v84 = v80.case0.v0;
                    switch (v84.tag) {
                        case 0: { // Call
                            v73[v83] = 1.0f;
                            break;
                        }
                        case 1: { // Fold
                            int v85;
                            v85 = v83 + 1l;
                            v73[v85] = 1.0f;
                            break;
                        }
                        case 2: { // Raise
                            int v86;
                            v86 = v83 + 2l;
                            v73[v86] = 1.0f;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                        }
                    }
                    break;
                }
                case 1: { // C2of2
                    Union2 v87 = v80.case1.v0;
                    int v88;
                    v88 = v83 + 3l;
                    switch (v87.tag) {
                        case 0: { // Jack
                            v73[v88] = 1.0f;
                            break;
                        }
                        case 1: { // King
                            int v89;
                            v89 = v88 + 1l;
                            v73[v89] = 1.0f;
                            break;
                        }
                        case 2: { // Queen
                            int v90;
                            v90 = v88 + 2l;
                            v73[v90] = 1.0f;
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
            v78 += 1l ;
        }
        v63 += 32l ;
    }
    __syncthreads();
    int v91;
    v91 = 0l;
    while (while_method_3(v91)){
        unsigned long long v93;
        v93 = clock64();
        int v94;
        v94 = threadIdx.x;
        int v95;
        v95 = blockIdx.x;
        int v96;
        v96 = v95 * 32l;
        int v97;
        v97 = v94 + v96;
        unsigned long long v98;
        v98 = (unsigned long long)v97;
        curandStatePhilox4_32_10_t v99;
        curand_init(v93,v98,0ull,&v99);
        int v100;
        v100 = blockIdx.x;
        float * v101;
        v101 = reinterpret_cast<float *>(&v0[0ull]);
        assert("Tensor range check" && 0 <= v100 && v100 < 1l);
        int v103;
        v103 = 2048l * v100;
        float * v104;
        v104 = reinterpret_cast<float *>(&v1[0ull]);
        assert("Tensor range check" && 0 <= v91 && v91 < 4l);
        int v106;
        v106 = 16384l * v91;
        float * v107;
        v107 = reinterpret_cast<float *>(&v0[8192ull]);
        method_0(v107, v104, v106, v101, v103);
        float * v109;
        v109 = reinterpret_cast<float *>(&v0[16384ull]);
        method_1(v109, v107);
        float * v111;
        v111 = reinterpret_cast<float *>(&v0[24576ull]);
        method_2(v111, v109);
        float * v113;
        v113 = reinterpret_cast<float *>(&v1[262144ull]);
        assert("Tensor range check" && 0 <= v91 && v91 < 4l);
        float * v115;
        v115 = reinterpret_cast<float *>(&v0[32768ull]);
        method_3(v115, v113, v106, v111);
        float * v117;
        v117 = reinterpret_cast<float *>(&v0[40960ull]);
        method_1(v117, v115);
        float * v119;
        v119 = reinterpret_cast<float *>(&v0[49152ull]);
        method_2(v119, v117);
        float * v121;
        v121 = reinterpret_cast<float *>(&v1[524288ull]);
        assert("Tensor range check" && 0 <= v91 && v91 < 4l);
        float * v123;
        v123 = reinterpret_cast<float *>(&v0[57344ull]);
        method_3(v123, v121, v106, v119);
        int * v125;
        v125 = reinterpret_cast<int *>(&v0[65536ull]);
        assert("Tensor range check" && 0 <= v100 && v100 < 1l);
        int v127;
        v127 = 64l * v100;
        assert("Tensor range check" && 0 <= v91 && v91 < 4l);
        int v128;
        v128 = 16l * v91;
        int v129;
        v129 = v128 + v127;
        method_4(v125, v129, v123, v99);
        v91 += 1l ;
    }
    __syncthreads();
    int * v130;
    v130 = reinterpret_cast<int *>(&v0[65536ull]);
    int v132;
    v132 = blockIdx.x;
    assert("Tensor range check" && 0 <= v132 && v132 < 1l);
    int v133;
    v133 = 64l * v132;
    int v134;
    v134 = threadIdx.x;
    int v135;
    v135 = v134;
    while (while_method_8(v135)){
        bool v137;
        v137 = 0l <= v135;
        bool v138;
        v138 = v137 == false;
        if (v138){
            assert("The index needs to be zero or positive." && v137);
        } else {
        }
        int v140;
        v140 = v135 % 16l;
        int v141;
        v141 = v135 / 16l;
        bool v142;
        v142 = v141 < 4l;
        bool v143;
        v143 = v142 == false;
        if (v143){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v142);
        } else {
        }
        assert("Tensor range check" && 0 <= v141 && v141 < 4l);
        assert("Tensor range check" && 0 <= v140 && v140 < 16l);
        int v145;
        v145 = v140 + v133;
        int v146;
        v146 = 16l * v141;
        int v147;
        v147 = v146 + v145;
        int v148;
        v148 = v130[v147];
        int v149;
        v149 = v148 % 3l;
        bool v150;
        v150 = 0l == v149;
        Union1 v156;
        if (v150){
            v156 = Union1{Union1_1{}};
        } else {
            bool v152;
            v152 = 1l == v149;
            if (v152){
                v156 = Union1{Union1_0{}};
            } else {
                v156 = Union1{Union1_2{}};
            }
        }
        static cuda::binary_semaphore<cuda::thread_scope_system> v157(1l);
        v157.acquire();
        write_5(v156);
        printf("\n");
        v157.release();
        v135 += 32l ;
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
i8 = i16 = i32 = i64 = u8 = u16 = u32 = u64 = int; f32 = f64 = float; char = string = str

options = []
options.append('--diag-suppress=550,20012,68')
options.append('--dopt=on')
options.append('--restrict')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def method0(v0 : string) -> None:
    print(v0, end="")
    del v0
    return 
def method2(v0 : char) -> None:
    print(v0, end="")
    del v0
    return 
def method3(v0 : i32, v1 : i32) -> bool:
    v2 = v1 < v0
    del v0, v1
    return v2
def method4(v0 : i32) -> None:
    print(v0, end="")
    del v0
    return 
def method1(v0 : cp.ndarray, v1 : i32, v2 : i32, v3 : i32, v4 : i32, v5 : i32, v6 : i32, v7 : i32) -> None:
    v8 = 0
    v9 = '['
    method2(v9)
    del v9
    v10 = 0
    while method3(v5, v10):
        v12 = v8
        v13 = v12 >= 100
        del v12
        if v13:
            v14 = " ..."
            method0(v14)
            del v14
            break
        else:
            pass
        del v13
        v15 = v10 == 0
        v16 = v15 != True
        del v15
        if v16:
            v17 = "; "
            method0(v17)
        else:
            pass
        del v16
        v18 = '['
        method2(v18)
        del v18
        v19 = 0
        while method3(v6, v19):
            v21 = v8
            v22 = v21 >= 100
            del v21
            if v22:
                v23 = " ..."
                method0(v23)
                del v23
                break
            else:
                pass
            del v22
            v24 = v19 == 0
            v25 = v24 != True
            del v24
            if v25:
                v26 = "; "
                method0(v26)
            else:
                pass
            del v25
            v27 = '['
            method2(v27)
            del v27
            v28 = 0
            while method3(v7, v28):
                v30 = v8
                v31 = v30 >= 100
                del v30
                if v31:
                    v32 = " ..."
                    method0(v32)
                    del v32
                    break
                else:
                    pass
                del v31
                v33 = v28 == 0
                v34 = v33 != True
                del v33
                if v34:
                    v35 = "; "
                    method0(v35)
                else:
                    pass
                del v34
                v36 = v8 + 1
                v8 = v36
                del v36
                v37 = v10 * v2
                v38 = v1 + v37
                del v37
                v39 = v19 * v3
                v40 = v38 + v39
                del v38, v39
                v41 = v28 * v4
                v42 = v40 + v41
                del v40, v41
                v43 = v0[v42].item()
                del v42
                method4(v43)
                del v43
                v28 += 1 
            del v28
            v44 = ']'
            method2(v44)
            del v44
            v19 += 1 
        del v19
        v45 = ']'
        method2(v45)
        del v45
        v10 += 1 
    del v0, v1, v2, v3, v4, v5, v6, v7, v8, v10
    v46 = ']'
    return method2(v46)
def main():
    v0 = cp.empty(786432,dtype=cp.uint8)
    v1 = cp.empty(65792,dtype=cp.uint8)
    v3 = v0[0:0+4*65536].view(cp.float32)
    v4 = cp.random.normal(0.0,1.0,65536,dtype=cp.float32) # type: ignore
    cp.copyto(v3[0:0+65536],v4[0:0+65536])
    del v3, v4
    v6 = v0[262144:262144+4*65536].view(cp.float32)
    v7 = cp.random.normal(0.0,1.0,65536,dtype=cp.float32) # type: ignore
    cp.copyto(v6[0:0+65536],v7[0:0+65536])
    del v6, v7
    v9 = v0[524288:524288+4*65536].view(cp.float32)
    v10 = cp.random.normal(0.0,1.0,65536,dtype=cp.float32) # type: ignore
    cp.copyto(v9[0:0+65536],v10[0:0+65536])
    del v9, v10
    v11 = "Running the kernel."
    method0(v11)
    del v11
    print()
    v12 = 0
    v13 = raw_module.get_function(f"entry{v12}")
    del v12
    v13.max_dynamic_shared_size_bytes = 1536 
    v13((1,),(32,),(v1, v0),shared_mem=1536)
    del v0, v13
    cp.cuda.get_current_stream().synchronize()
    v14 = "The output tensor is:"
    method0(v14)
    del v14
    print()
    v16 = v1[65536:65536+4*64].view(cp.int32)
    del v1
    v17 = 0
    v18 = 64
    v19 = 16
    v20 = 1
    v21 = 1
    v22 = 4
    v23 = 16
    method1(v16, v17, v18, v19, v20, v21, v22, v23)
    del v16, v17, v18, v19, v20, v21, v22, v23
    print()
    v24 = "==="
    method0(v24)
    del v24
    print()
    return 

if __name__ == '__main__': print(main())
