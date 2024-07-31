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
__device__ void method_0(float * v0, int v1, float * v2, int v3, float * v4, int v5);
__device__ void method_1(float * v0, float * v1);
__device__ void method_2(float * v0, float * v1);
struct Tuple0;
struct Tuple1;
__device__ void method_3(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5);
__device__ unsigned int loop_5(unsigned int v0, curandStatePhilox4_32_10_t & v1);
__device__ int int_range_4(int v0, int v1, curandStatePhilox4_32_10_t & v2);
__device__ void write_7(char v0);
__device__ void write_8();
__device__ void write_9(const char * v0);
__device__ void write_10(int v0);
__device__ void write_12(float v0);
__device__ void write_11(float * v0, int v1, int v2, int v3);
__device__ void write_6(int v0, float * v1, int v2, int v3, int v4, float v5);
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
        v2 = v1 + v0;
        return v2;
    }
};
struct Closure1 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v1 + v0;
        return v2;
    }
};
struct Tuple0 {
    int v0;
    float v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Closure2 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v1 + v0;
        return v2;
    }
};
struct Closure3 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        return v0;
    }
};
struct Tuple1 {
    float v0;
    int v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v2 >= 0.0f;
        bool v6;
        if (v4){
            bool v5;
            v5 = v0 >= 0.0f;
            v6 = v5;
        } else {
            v6 = false;
        }
        if (v6){
            bool v7;
            v7 = v2 <= v0;
            if (v7){
                return Tuple1{v2, v3};
            } else {
                return Tuple1{v0, v1};
            }
        } else {
            if (v4){
                return Tuple1{v2, v3};
            } else {
                bool v10;
                v10 = v0 >= 0.0f;
                if (v10){
                    return Tuple1{v0, v1};
                } else {
                    return Tuple1{v2, v3};
                }
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
__device__ void method_0(float * v0, int v1, float * v2, int v3, float * v4, int v5){
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
                __syncthreads();
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
                __syncthreads();
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
            __syncthreads();
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
            __syncthreads();
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
__device__ void method_1(float * v0, float * v1){
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
    __syncthreads();
    return ;
}
__device__ inline bool while_method_8(int v0){
    bool v1;
    v1 = v0 < 1024l;
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
    __syncthreads();
    return ;
}
__device__ void method_3(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5){
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
                bool v173;
                v173 = v164 == 0.0f;
                bool v174;
                v174 = v173 != true;
                float v176;
                if (v174){
                    float v175;
                    v175 = v172 / v164;
                    v176 = v175;
                } else {
                    v176 = 0.0078125f;
                }
                assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                assert("Tensor range check" && 0 <= v168 && v168 < 4l);
                v165[v171] = v176;
                v168 += 1l ;
            }
            v166 += 1l ;
        }
        float v177[4l];
        float v178;
        v178 = 0.0f;
        int v179;
        v179 = 0l;
        while (while_method_5(v179)){
            assert("Tensor range check" && 0 <= v179 && v179 < 1l);
            int v181;
            v181 = 4l * v179;
            assert("Tensor range check" && 0 <= v179 && v179 < 1l);
            int v182; float v183;
            Tuple0 tmp0 = Tuple0{0l, 0.0f};
            v182 = tmp0.v0; v183 = tmp0.v1;
            while (while_method_2(v182)){
                assert("Tensor range check" && 0 <= v182 && v182 < 4l);
                int v185;
                v185 = v182 + v181;
                float v186;
                v186 = v165[v185];
                float v187;
                v187 = v183 + v186;
                v183 = v187;
                v182 += 1l ;
            }
            auto v188 = cooperative_groups::coalesced_threads();
            int v189;
            v189 = threadIdx.x;
            int v190;
            v190 = v189 / 32l;
            auto v191 = cooperative_groups::labeled_partition(v188,v190);
            Closure2 v192{};
            float v193;
            v193 = cooperative_groups::inclusive_scan(v191, v183, v192);
            float v194;
            v194 = v191.shfl_up(v193,1);
            bool v195;
            v195 = v191.thread_rank() == 0;
            float v196;
            if (v195){
                v196 = 0.0f;
            } else {
                v196 = v194;
            }
            float v197;
            v197 = v191.shfl(v193,v191.num_threads()-1);
            float v198;
            v198 = v178 + v196;
            int v199; float v200;
            Tuple0 tmp1 = Tuple0{0l, v198};
            v199 = tmp1.v0; v200 = tmp1.v1;
            while (while_method_2(v199)){
                assert("Tensor range check" && 0 <= v199 && v199 < 4l);
                int v202;
                v202 = v199 + v181;
                float v203;
                v203 = v165[v202];
                float v204;
                v204 = v200 + v203;
                assert("Tensor range check" && 0 <= v199 && v199 < 4l);
                v177[v202] = v204;
                v200 = v204;
                v199 += 1l ;
            }
            float v205;
            v205 = v178 + v197;
            v178 = v205;
            v179 += 1l ;
        }
        float v206;
        v206 = curand_uniform(&v5);
        float v207[4l];
        int v208;
        v208 = 0l;
        while (while_method_5(v208)){
            int v210;
            v210 = 0l;
            while (while_method_2(v210)){
                assert("Tensor range check" && 0 <= v208 && v208 < 1l);
                assert("Tensor range check" && 0 <= v210 && v210 < 4l);
                int v212;
                v212 = 4l * v208;
                int v213;
                v213 = v212 + v210;
                int v214;
                v214 = v35[v213];
                assert("Tensor range check" && 0 <= v208 && v208 < 1l);
                assert("Tensor range check" && 0 <= v210 && v210 < 4l);
                v207[v213] = v206;
                v210 += 1l ;
            }
            v208 += 1l ;
        }
        float v215;
        v215 = 0.0f;
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
                v222 = v207[v221];
                v215 = v222;
                v218 += 1l ;
            }
            v216 += 1l ;
        }
        auto v223 = cooperative_groups::coalesced_threads();
        int v224;
        v224 = threadIdx.x;
        int v225;
        v225 = v224 / 32l;
        auto v226 = cooperative_groups::labeled_partition(v223,v225);
        Closure3 v227{};
        float v228;
        v228 = cooperative_groups::reduce(v226, v215, v227);
        float v229[4l];
        int v230;
        v230 = 0l;
        while (while_method_5(v230)){
            int v232;
            v232 = 0l;
            while (while_method_2(v232)){
                assert("Tensor range check" && 0 <= v230 && v230 < 1l);
                assert("Tensor range check" && 0 <= v232 && v232 < 4l);
                int v234;
                v234 = 4l * v230;
                int v235;
                v235 = v234 + v232;
                float v236;
                v236 = v177[v235];
                float v237;
                v237 = v236 - v228;
                assert("Tensor range check" && 0 <= v230 && v230 < 1l);
                assert("Tensor range check" && 0 <= v232 && v232 < 4l);
                v229[v235] = v237;
                v232 += 1l ;
            }
            v230 += 1l ;
        }
        float v238; int v239;
        Tuple1 tmp2 = Tuple1{-1.0f / 0.0f, 0l};
        v238 = tmp2.v0; v239 = tmp2.v1;
        int v240;
        v240 = 0l;
        while (while_method_5(v240)){
            int v242;
            v242 = 0l;
            while (while_method_2(v242)){
                assert("Tensor range check" && 0 <= v240 && v240 < 1l);
                assert("Tensor range check" && 0 <= v242 && v242 < 4l);
                int v244;
                v244 = 4l * v240;
                int v245;
                v245 = v244 + v242;
                float v246;
                v246 = v229[v245];
                int v247;
                v247 = v35[v245];
                bool v248;
                v248 = v238 >= 0.0f;
                bool v250;
                if (v248){
                    bool v249;
                    v249 = v246 >= 0.0f;
                    v250 = v249;
                } else {
                    v250 = false;
                }
                float v259; int v260;
                if (v250){
                    bool v251;
                    v251 = v238 <= v246;
                    if (v251){
                        v259 = v238; v260 = v239;
                    } else {
                        v259 = v246; v260 = v247;
                    }
                } else {
                    if (v248){
                        v259 = v238; v260 = v239;
                    } else {
                        bool v254;
                        v254 = v246 >= 0.0f;
                        if (v254){
                            v259 = v246; v260 = v247;
                        } else {
                            v259 = v238; v260 = v239;
                        }
                    }
                }
                v238 = v259;
                v239 = v260;
                v242 += 1l ;
            }
            v240 += 1l ;
        }
        auto v261 = cooperative_groups::coalesced_threads();
        int v262;
        v262 = threadIdx.x;
        int v263;
        v263 = v262 / 32l;
        auto v264 = cooperative_groups::labeled_partition(v261,v263);
        Closure4 v265{};
        float v266; int v267;
        Tuple1 tmp3 = cooperative_groups::reduce(v264, Tuple1{v238, v239}, v265);
        v266 = tmp3.v0; v267 = tmp3.v1;
        assert("Tensor range check" && 0 <= v30 && v30 < 32l);
        int v268;
        v268 = v32 + v28;
        int v269;
        v269 = 0l;
        while (while_method_5(v269)){
            assert("Tensor range check" && 0 <= v269 && v269 < 1l);
            int v271;
            v271 = 128l * v269;
            int v272;
            v272 = v271 + v268;
            assert("Tensor range check" && 0 <= v269 && v269 < 1l);
            int v273;
            v273 = 4l * v269;
            int4* v274;
            v274 = reinterpret_cast<int4*>(v165 + v273);
            int4* v275;
            v275 = reinterpret_cast<int4*>(v2 + v272);
            assert("Pointer alignment check" && (unsigned long long)(v274) % 4l == 0 && (unsigned long long)(v275) % 4l == 0);
            *v275 = *v274;
            v269 += 1l ;
        }
        assert("Tensor range check" && 0 <= v30 && v30 < 32l);
        int v276;
        v276 = v30 + v29;
        v0[v276] = v267;
        v30 += 1l ;
    }
    __syncthreads();
    return ;
}
__device__ unsigned int loop_5(unsigned int v0, curandStatePhilox4_32_10_t & v1){
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
        return loop_5(v0, v1);
    }
}
__device__ int int_range_4(int v0, int v1, curandStatePhilox4_32_10_t & v2){
    int v3;
    v3 = v0 - v1;
    unsigned int v4;
    v4 = (unsigned int)v3;
    unsigned int v5;
    v5 = loop_5(v4, v2);
    unsigned int v6;
    v6 = (unsigned int)v1;
    unsigned int v7;
    v7 = v5 + v6;
    int v8;
    v8 = (int)v7;
    return v8;
}
__device__ void write_7(char v0){
    const char * v1;
    v1 = "%c";
    printf(v1,v0);
    return ;
}
__device__ void write_8(){
    return ;
}
__device__ void write_9(const char * v0){
    const char * v1;
    v1 = "%s";
    printf(v1,v0);
    return ;
}
__device__ void write_10(int v0){
    const char * v1;
    v1 = "%d";
    printf(v1,v0);
    return ;
}
__device__ void write_12(float v0){
    const char * v1;
    v1 = "%f";
    printf(v1,v0);
    return ;
}
__device__ void write_11(float * v0, int v1, int v2, int v3){
    int v4;
    v4 = 0l;
    char v5;
    v5 = '[';
    write_7(v5);
    int v6;
    v6 = 0l;
    while (while_method_1(v3, v6)){
        int v8;
        v8 = v4;
        bool v9;
        v9 = v8 >= 100l;
        if (v9){
            const char * v10;
            v10 = " ...";
            write_9(v10);
            break;
        } else {
        }
        bool v11;
        v11 = v6 == 0l;
        bool v12;
        v12 = v11 != true;
        if (v12){
            const char * v13;
            v13 = "; ";
            write_9(v13);
        } else {
        }
        int v14;
        v14 = v4 + 1l;
        v4 = v14;
        int v15;
        v15 = v6 * v2;
        int v16;
        v16 = v1 + v15;
        float v17;
        v17 = v0[v16];
        write_12(v17);
        v6 += 1l ;
    }
    char v18;
    v18 = ']';
    return write_7(v18);
}
__device__ void write_6(int v0, float * v1, int v2, int v3, int v4, float v5){
    char v6;
    v6 = '{';
    write_7(v6);
    write_8();
    const char * v7;
    v7 = "action";
    write_9(v7);
    const char * v8;
    v8 = " = ";
    write_9(v8);
    write_10(v0);
    const char * v9;
    v9 = "; ";
    write_9(v9);
    const char * v10;
    v10 = "sampling_prob_ensemble";
    write_9(v10);
    write_9(v8);
    write_11(v1, v2, v3, v4);
    write_9(v9);
    const char * v11;
    v11 = "sampling_prob_selected";
    write_9(v11);
    write_9(v8);
    write_12(v5);
    char v12;
    v12 = '}';
    return write_7(v12);
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
    float * v44;
    v44 = reinterpret_cast<float *>(&v0[131072ull]);
    float * v46;
    v46 = reinterpret_cast<float *>(&v0[0ull]);
    int * v48;
    v48 = reinterpret_cast<int *>(&v0[196608ull]);
    unsigned long long v50;
    v50 = clock64();
    int v51;
    v51 = threadIdx.x;
    int v52;
    v52 = blockIdx.x;
    int v53;
    v53 = v52 * 32l;
    int v54;
    v54 = v51 + v53;
    unsigned long long v55;
    v55 = (unsigned long long)v54;
    curandStatePhilox4_32_10_t v56;
    curand_init(v50,v55,0ull,&v56);
    float * v57;
    v57 = reinterpret_cast<float *>(&v0[0ull]);
    int v59;
    v59 = blockIdx.x;
    assert("Tensor range check" && 0 <= v59 && v59 < 1l);
    int v60;
    v60 = 4096l * v59;
    unsigned long long v61;
    v61 = clock64();
    int v62;
    v62 = threadIdx.x;
    int v63;
    v63 = blockIdx.x;
    int v64;
    v64 = v63 * 32l;
    int v65;
    v65 = v62 + v64;
    unsigned long long v66;
    v66 = (unsigned long long)v65;
    curandStatePhilox4_32_10_t v67;
    curand_init(v61,v66,0ull,&v67);
    int v68;
    v68 = threadIdx.x;
    int v69;
    v69 = v68;
    while (while_method_0(v69)){
        bool v71;
        v71 = 0l <= v69;
        bool v72;
        v72 = v71 == false;
        if (v72){
            assert("The index needs to be zero or positive." && v71);
        } else {
        }
        int v74;
        v74 = v69 % 128l;
        int v75;
        v75 = v69 / 128l;
        bool v76;
        v76 = v75 < 32l;
        bool v77;
        v77 = v76 == false;
        if (v77){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v76);
        } else {
        }
        assert("Tensor range check" && 0 <= v75 && v75 < 32l);
        assert("Tensor range check" && 0 <= v74 && v74 < 128l);
        int v79;
        v79 = v74 + v60;
        int v80;
        v80 = 128l * v75;
        int v81;
        v81 = v80 + v79;
        v57[v81] = 0.0f;
        v69 += 32l ;
    }
    __syncthreads();
    int v82;
    v82 = threadIdx.x;
    assert("Tensor range check" && 0 <= v82 && v82 < 32l);
    int v83;
    v83 = 128l * v82;
    int v84;
    v84 = v83 + v60;
    float * v85;
    v85 = v57+v84;
    int v87;
    v87 = v2.length;
    bool v88;
    v88 = v87 == 0l;
    if (v88){
        v85[0l] = 1.0f;
    } else {
    }
    int v89;
    v89 = v2.length;
    int v90;
    v90 = 0l;
    while (while_method_1(v89, v90)){
        Union0 v92;
        v92 = v2[v90];
        int v94;
        v94 = v90 * 6l;
        int v95;
        v95 = 1l + v94;
        switch (v92.tag) {
            case 0: { // C1of2
                Union1 v96 = v92.case0.v0;
                switch (v96.tag) {
                    case 0: { // Call
                        v85[v95] = 1.0f;
                        break;
                    }
                    case 1: { // Fold
                        int v97;
                        v97 = v95 + 1l;
                        v85[v97] = 1.0f;
                        break;
                    }
                    case 2: { // Raise
                        int v98;
                        v98 = v95 + 2l;
                        v85[v98] = 1.0f;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                break;
            }
            case 1: { // C2of2
                Union2 v99 = v92.case1.v0;
                int v100;
                v100 = v95 + 3l;
                switch (v99.tag) {
                    case 0: { // Jack
                        v85[v100] = 1.0f;
                        break;
                    }
                    case 1: { // King
                        int v101;
                        v101 = v100 + 1l;
                        v85[v101] = 1.0f;
                        break;
                    }
                    case 2: { // Queen
                        int v102;
                        v102 = v100 + 2l;
                        v85[v102] = 1.0f;
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
        v90 += 1l ;
    }
    __syncthreads();
    int v103;
    v103 = 0l;
    while (while_method_2(v103)){
        float * v105;
        v105 = reinterpret_cast<float *>(&v0[0ull]);
        float * v107;
        v107 = reinterpret_cast<float *>(&v1[0ull]);
        assert("Tensor range check" && 0 <= v103 && v103 < 4l);
        int v109;
        v109 = 16384l * v103;
        float * v110;
        v110 = reinterpret_cast<float *>(&v0[16384ull]);
        int v112;
        v112 = blockIdx.x;
        assert("Tensor range check" && 0 <= v112 && v112 < 1l);
        int v113;
        v113 = 4096l * v112;
        int v114;
        v114 = blockIdx.x;
        assert("Tensor range check" && 0 <= v114 && v114 < 1l);
        int v115;
        v115 = 4096l * v114;
        method_0(v107, v109, v110, v115, v105, v113);
        float * v116;
        v116 = reinterpret_cast<float *>(&v0[32768ull]);
        method_1(v116, v110);
        float * v118;
        v118 = reinterpret_cast<float *>(&v0[49152ull]);
        method_2(v118, v116);
        float * v120;
        v120 = reinterpret_cast<float *>(&v1[262144ull]);
        assert("Tensor range check" && 0 <= v103 && v103 < 4l);
        float * v122;
        v122 = reinterpret_cast<float *>(&v0[65536ull]);
        int v124;
        v124 = blockIdx.x;
        assert("Tensor range check" && 0 <= v124 && v124 < 1l);
        int v125;
        v125 = 4096l * v124;
        int v126;
        v126 = blockIdx.x;
        assert("Tensor range check" && 0 <= v126 && v126 < 1l);
        int v127;
        v127 = 4096l * v126;
        method_0(v120, v109, v122, v127, v118, v125);
        float * v128;
        v128 = reinterpret_cast<float *>(&v0[81920ull]);
        method_1(v128, v122);
        float * v130;
        v130 = reinterpret_cast<float *>(&v0[98304ull]);
        method_2(v130, v128);
        float * v132;
        v132 = reinterpret_cast<float *>(&v1[524288ull]);
        assert("Tensor range check" && 0 <= v103 && v103 < 4l);
        float * v134;
        v134 = reinterpret_cast<float *>(&v0[114688ull]);
        int v136;
        v136 = blockIdx.x;
        assert("Tensor range check" && 0 <= v136 && v136 < 1l);
        int v137;
        v137 = 4096l * v136;
        int v138;
        v138 = blockIdx.x;
        assert("Tensor range check" && 0 <= v138 && v138 < 1l);
        int v139;
        v139 = 4096l * v138;
        method_0(v132, v109, v134, v139, v130, v137);
        float * v140;
        v140 = reinterpret_cast<float *>(&v0[131072ull]);
        assert("Tensor range check" && 0 <= v103 && v103 < 4l);
        int v142;
        v142 = 4096l * v103;
        int * v143;
        v143 = reinterpret_cast<int *>(&v0[196608ull]);
        assert("Tensor range check" && 0 <= v103 && v103 < 4l);
        int v145;
        v145 = 32l * v103;
        method_3(v143, v145, v140, v142, v134, v67);
        v103 += 1l ;
    }
    __syncthreads();
    int v146;
    v146 = 0l;
    int v147;
    v147 = 4l;
    int v148;
    v148 = int_range_4(v147, v146, v67);
    float * v149;
    v149 = reinterpret_cast<float *>(&v0[131072ull]);
    int * v151;
    v151 = reinterpret_cast<int *>(&v0[196608ull]);
    assert("Tensor range check" && 0 <= v148 && v148 < 4l);
    int v153;
    v153 = 32l * v148;
    int v154;
    v154 = blockIdx.x;
    assert("Tensor range check" && 0 <= v154 && v154 < 1l);
    int v155;
    v155 = 32l * v154;
    int v156;
    v156 = v155 + v153;
    int v157;
    v157 = threadIdx.x;
    assert("Tensor range check" && 0 <= v157 && v157 < 32l);
    int v158;
    v158 = v157 + v156;
    int v159;
    v159 = v151[v158];
    int v160;
    v160 = blockIdx.x;
    int v161;
    v161 = threadIdx.x;
    assert("Tensor range check" && 0 <= v148 && v148 < 4l);
    assert("Tensor range check" && 0 <= v160 && v160 < 1l);
    assert("Tensor range check" && 0 <= v161 && v161 < 32l);
    assert("Tensor range check" && 0 <= v159 && v159 < 128l);
    int v162;
    v162 = 128l * v161;
    int v163;
    v163 = v162 + v159;
    int v164;
    v164 = 4096l * v160;
    int v165;
    v165 = v164 + v163;
    int v166;
    v166 = 4096l * v148;
    int v167;
    v167 = v166 + v165;
    float v168;
    v168 = v149[v167];
    int v169;
    v169 = blockIdx.x;
    assert("Tensor range check" && 0 <= v169 && v169 < 1l);
    int v170;
    v170 = 4096l * v169;
    int v171;
    v171 = threadIdx.x;
    assert("Tensor range check" && 0 <= v171 && v171 < 32l);
    int v172;
    v172 = 128l * v171;
    int v173;
    v173 = v172 + v170;
    assert("Tensor range check" && 0 <= v159 && v159 < 128l);
    int v174;
    v174 = v159 + v173;
    static cuda::binary_semaphore<cuda::thread_scope_system> v175(1l);
    v175.acquire();
    int v176;
    v176 = 4096l;
    int v177;
    v177 = 4l;
    write_6(v159, v149, v174, v176, v177, v168);
    printf("\n");
    v175.release();
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
def method1(v0 : char) -> None:
    print(v0, end="")
    del v0
    return 
def method2(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method3(v0 : i32) -> bool:
    v1 = v0 < 1
    del v0
    return v1
def method4(v0 : i32) -> bool:
    v1 = v0 < 32
    del v0
    return v1
def method5(v0 : i32) -> bool:
    v1 = v0 < 128
    del v0
    return v1
def method6(v0 : f32) -> None:
    print("{:.6f}".format(v0), end="")
    del v0
    return 
def method7() -> None:
    return 
def method8(v0 : i32) -> None:
    print(v0, end="")
    del v0
    return 
def main():
    v0 = cp.empty(786432,dtype=cp.uint8)
    v1 = cp.empty(197120,dtype=cp.uint8)
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
    v16 = v1[131072:131072+4*16384].view(cp.float32)
    v17 = 0
    v18 = '['
    method1(v18)
    del v18
    v19 = 0
    while method2(v19):
        v21 = v17
        v22 = v21 >= 2147483647
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
        method1(v27)
        del v27
        v28 = 0
        while method3(v28):
            v30 = v17
            v31 = v30 >= 2147483647
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
            v36 = '['
            method1(v36)
            del v36
            v37 = 0
            while method4(v37):
                v39 = v17
                v40 = v39 >= 2147483647
                del v39
                if v40:
                    v41 = " ..."
                    method0(v41)
                    del v41
                    break
                else:
                    pass
                del v40
                v42 = v37 == 0
                v43 = v42 != True
                del v42
                if v43:
                    v44 = "; "
                    method0(v44)
                else:
                    pass
                del v43
                v45 = '['
                method1(v45)
                del v45
                v46 = 0
                while method5(v46):
                    v48 = v17
                    v49 = v48 >= 2147483647
                    del v48
                    if v49:
                        v50 = " ..."
                        method0(v50)
                        del v50
                        break
                    else:
                        pass
                    del v49
                    v51 = v46 == 0
                    v52 = v51 != True
                    del v51
                    if v52:
                        v53 = "; "
                        method0(v53)
                    else:
                        pass
                    del v52
                    v54 = v17 + 1
                    v17 = v54
                    del v54
                    v55 = v19 * 4096
                    v56 = v28 * 4096
                    v57 = v55 + v56
                    del v55, v56
                    v58 = v37 * 128
                    v59 = v57 + v58
                    del v57, v58
                    v60 = v59 + v46
                    del v59
                    v61 = v16[v60].item()
                    del v60
                    method6(v61)
                    del v61
                    v46 += 1 
                del v46
                v62 = ']'
                method1(v62)
                del v62
                v37 += 1 
            del v37
            v63 = ']'
            method1(v63)
            del v63
            v28 += 1 
        del v28
        v64 = ']'
        method1(v64)
        del v64
        v19 += 1 
    del v16, v17, v19
    v65 = ']'
    method1(v65)
    del v65
    method7()
    print()
    v67 = v1[196608:196608+4*128].view(cp.int32)
    del v1
    v68 = 0
    v69 = '['
    method1(v69)
    del v69
    v70 = 0
    while method2(v70):
        v72 = v68
        v73 = v72 >= 2147483647
        del v72
        if v73:
            v74 = " ..."
            method0(v74)
            del v74
            break
        else:
            pass
        del v73
        v75 = v70 == 0
        v76 = v75 != True
        del v75
        if v76:
            v77 = "; "
            method0(v77)
        else:
            pass
        del v76
        v78 = '['
        method1(v78)
        del v78
        v79 = 0
        while method3(v79):
            v81 = v68
            v82 = v81 >= 2147483647
            del v81
            if v82:
                v83 = " ..."
                method0(v83)
                del v83
                break
            else:
                pass
            del v82
            v84 = v79 == 0
            v85 = v84 != True
            del v84
            if v85:
                v86 = "; "
                method0(v86)
            else:
                pass
            del v85
            v87 = '['
            method1(v87)
            del v87
            v88 = 0
            while method4(v88):
                v90 = v68
                v91 = v90 >= 2147483647
                del v90
                if v91:
                    v92 = " ..."
                    method0(v92)
                    del v92
                    break
                else:
                    pass
                del v91
                v93 = v88 == 0
                v94 = v93 != True
                del v93
                if v94:
                    v95 = "; "
                    method0(v95)
                else:
                    pass
                del v94
                v96 = v68 + 1
                v68 = v96
                del v96
                v97 = v70 * 32
                v98 = v79 * 32
                v99 = v97 + v98
                del v97, v98
                v100 = v99 + v88
                del v99
                v101 = v67[v100].item()
                del v100
                method8(v101)
                del v101
                v88 += 1 
            del v88
            v102 = ']'
            method1(v102)
            del v102
            v79 += 1 
        del v79
        v103 = ']'
        method1(v103)
        del v103
        v70 += 1 
    del v67, v68, v70
    v104 = ']'
    method1(v104)
    del v104
    method7()
    print()
    return 

if __name__ == '__main__': print(main())
