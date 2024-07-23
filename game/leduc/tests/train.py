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
__device__ void method_3(float * v0, int v1, float * v2);
struct Tuple0;
struct Tuple1;
__device__ void method_4(int * v0, int v1, float * v2, int v3, curandStatePhilox4_32_10_t & v4);
__device__ unsigned int loop_6(unsigned int v0, curandStatePhilox4_32_10_t & v1);
__device__ int int_range_5(int v0, int v1, curandStatePhilox4_32_10_t & v2);
__device__ void write_8(char v0);
__device__ void write_9();
__device__ void write_10(const char * v0);
__device__ void write_12();
__device__ void write_13();
__device__ void write_14();
__device__ void write_11(Union1 v0);
__device__ void write_16(float v0);
__device__ void write_15(float * v0, int v1, int v2, int v3);
__device__ void write_7(Union1 v0, float * v1, int v2, int v3, int v4, float v5);
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
        assert("Tensor range check" && 0 <= v21 && v21 < 32l);
        int v25;
        v25 = v23 + v20;
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
            v32 = v31 + v24;
            int4* v33;
            v33 = reinterpret_cast<int4*>(v1 + v32);
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
            while (while_method_2(v37)){
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
                v44 = 0l <= v10;
                bool v46;
                if (v44){
                    bool v45;
                    v45 = v10 < 32l;
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
                v49 = v10 * 4l;
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
        v60 = 0l <= v11;
        bool v61;
        v61 = v60 && v12;
        bool v62;
        v62 = v61 == false;
        if (v62){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v61);
        } else {
        }
        bool v64;
        v64 = 0l <= v21;
        bool v66;
        if (v64){
            bool v65;
            v65 = v21 < 32l;
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
        v69 = v21 + v11;
        float v70[4l];
        int v71;
        v71 = 0l;
        while (while_method_5(v71)){
            int v73;
            v73 = 0l;
            while (while_method_2(v73)){
                assert("Tensor range check" && 0 <= v71 && v71 < 1l);
                assert("Tensor range check" && 0 <= v73 && v73 < 4l);
                int v75;
                v75 = 4l * v71;
                int v76;
                v76 = v75 + v73;
                float v77;
                v77 = v26[v76];
                float v78;
                v78 = v77 * v77;
                assert("Tensor range check" && 0 <= v71 && v71 < 1l);
                assert("Tensor range check" && 0 <= v73 && v73 < 4l);
                v70[v76] = v78;
                v73 += 1l ;
            }
            v71 += 1l ;
        }
        float v79;
        v79 = 0.0f;
        int v80;
        v80 = 0l;
        while (while_method_5(v80)){
            int v82;
            v82 = 0l;
            while (while_method_2(v82)){
                assert("Tensor range check" && 0 <= v80 && v80 < 1l);
                assert("Tensor range check" && 0 <= v82 && v82 < 4l);
                int v84;
                v84 = 4l * v80;
                int v85;
                v85 = v84 + v82;
                float v86;
                v86 = v70[v85];
                float v87;
                v87 = v79 + v86;
                v79 = v87;
                v82 += 1l ;
            }
            v80 += 1l ;
        }
        auto v88 = cooperative_groups::coalesced_threads();
        int v89;
        v89 = threadIdx.x;
        int v90;
        v90 = v89 / 32l;
        auto v91 = cooperative_groups::labeled_partition(v88,v90);
        Closure0 v92{};
        float v93;
        v93 = cooperative_groups::reduce(v91, v79, v92);
        float v94[4l];
        int v95;
        v95 = 0l;
        while (while_method_5(v95)){
            int v97;
            v97 = 0l;
            while (while_method_2(v97)){
                assert("Tensor range check" && 0 <= v95 && v95 < 1l);
                assert("Tensor range check" && 0 <= v97 && v97 < 4l);
                int v99;
                v99 = 4l * v95;
                int v100;
                v100 = v99 + v97;
                float v101;
                v101 = v26[v100];
                bool v102;
                v102 = v93 == 0.0f;
                bool v103;
                v103 = v102 != true;
                float v105;
                if (v103){
                    float v104;
                    v104 = v101 / v93;
                    v105 = v104;
                } else {
                    v105 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v95 && v95 < 1l);
                assert("Tensor range check" && 0 <= v97 && v97 < 4l);
                v94[v100] = v105;
                v97 += 1l ;
            }
            v95 += 1l ;
        }
        int v106;
        v106 = 0l;
        while (while_method_5(v106)){
            assert("Tensor range check" && 0 <= v106 && v106 < 1l);
            int v108;
            v108 = 128l * v106;
            int v109;
            v109 = v108 + v25;
            assert("Tensor range check" && 0 <= v106 && v106 < 1l);
            int v110;
            v110 = 4l * v106;
            int4* v111;
            v111 = reinterpret_cast<int4*>(v94 + v110);
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
__device__ void method_3(float * v0, int v1, float * v2){
    int v3;
    v3 = blockIdx.x;
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v4;
    v4 = 4096l * v3;
    int v5;
    v5 = blockIdx.x;
    assert("Tensor range check" && 0 <= v5 && v5 < 1l);
    int v6;
    v6 = 4096l * v5;
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
    v14 = v13 < 1l;
    bool v15;
    v15 = v14 == false;
    if (v15){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v14);
    } else {
    }
    assert("Tensor range check" && 0 <= v13 && v13 < 1l);
    assert("Tensor range check" && 0 <= v12 && v12 < 32l);
    int v17;
    v17 = 4l * v12;
    int v18;
    v18 = v17 + v4;
    int v19;
    v19 = 128l * v13;
    int v20;
    v20 = v19 + v18;
    assert("Tensor range check" && 0 <= v13 && v13 < 1l);
    assert("Tensor range check" && 0 <= v12 && v12 < 32l);
    int v21;
    v21 = v17 + v7;
    int v22;
    v22 = v19 + v21;
    int v23;
    v23 = 0l;
    while (while_method_7(v23)){
        assert("Tensor range check" && 0 <= v23 && v23 < 32l);
        int v25;
        v25 = 128l * v23;
        int v26;
        v26 = v25 + v20;
        assert("Tensor range check" && 0 <= v23 && v23 < 32l);
        int v27;
        v27 = v25 + v22;
        float v28[4l];
        int v29[4l];
        int v30;
        v30 = 0l;
        while (while_method_5(v30)){
            assert("Tensor range check" && 0 <= v30 && v30 < 1l);
            int v32;
            v32 = 4l * v30;
            assert("Tensor range check" && 0 <= v30 && v30 < 1l);
            int v33;
            v33 = 128l * v30;
            int v34;
            v34 = v33 + v26;
            int4* v35;
            v35 = reinterpret_cast<int4*>(v2 + v34);
            int4* v36;
            v36 = reinterpret_cast<int4*>(v28 + v32);
            assert("Pointer alignment check" && (unsigned long long)(v35) % 4l == 0 && (unsigned long long)(v36) % 4l == 0);
            *v36 = *v35;
            v30 += 1l ;
        }
        int v37;
        v37 = 0l;
        while (while_method_5(v37)){
            int v39;
            v39 = 0l;
            while (while_method_2(v39)){
                bool v41;
                v41 = 0l <= v39;
                bool v43;
                if (v41){
                    bool v42;
                    v42 = v39 < 4l;
                    v43 = v42;
                } else {
                    v43 = false;
                }
                bool v44;
                v44 = v43 == false;
                if (v44){
                    assert("The indices should be inside the range of the dimension." && v43);
                } else {
                }
                bool v46;
                v46 = 0l <= v12;
                bool v48;
                if (v46){
                    bool v47;
                    v47 = v12 < 32l;
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
                v51 = v12 * 4l;
                int v52;
                v52 = v39 + v51;
                bool v53;
                v53 = 0l <= v37;
                bool v55;
                if (v53){
                    bool v54;
                    v54 = v37 < 1l;
                    v55 = v54;
                } else {
                    v55 = false;
                }
                bool v56;
                v56 = v55 == false;
                if (v56){
                    assert("The indices should be inside the range of the dimension." && v55);
                } else {
                }
                int v58;
                v58 = v37 * 128l;
                int v59;
                v59 = v52 + v58;
                assert("Tensor range check" && 0 <= v37 && v37 < 1l);
                assert("Tensor range check" && 0 <= v39 && v39 < 4l);
                int v60;
                v60 = 4l * v37;
                int v61;
                v61 = v60 + v39;
                v29[v61] = v59;
                v39 += 1l ;
            }
            v37 += 1l ;
        }
        bool v62;
        v62 = 0l <= v13;
        bool v63;
        v63 = v62 && v14;
        bool v64;
        v64 = v63 == false;
        if (v64){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v63);
        } else {
        }
        bool v66;
        v66 = 0l <= v23;
        bool v68;
        if (v66){
            bool v67;
            v67 = v23 < 32l;
            v68 = v67;
        } else {
            v68 = false;
        }
        bool v69;
        v69 = v68 == false;
        if (v69){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v68);
        } else {
        }
        int v71;
        v71 = v23 + v13;
        float v72;
        v72 = 0.0f;
        int v73;
        v73 = 0l;
        while (while_method_5(v73)){
            int v75;
            v75 = 0l;
            while (while_method_2(v75)){
                assert("Tensor range check" && 0 <= v73 && v73 < 1l);
                assert("Tensor range check" && 0 <= v75 && v75 < 4l);
                int v77;
                v77 = 4l * v73;
                int v78;
                v78 = v77 + v75;
                float v79;
                v79 = v28[v78];
                float v80;
                v80 = v72 + v79;
                v72 = v80;
                v75 += 1l ;
            }
            v73 += 1l ;
        }
        auto v81 = cooperative_groups::coalesced_threads();
        int v82;
        v82 = threadIdx.x;
        int v83;
        v83 = v82 / 32l;
        auto v84 = cooperative_groups::labeled_partition(v81,v83);
        Closure0 v85{};
        float v86;
        v86 = cooperative_groups::reduce(v84, v72, v85);
        float v87;
        v87 = v86 / 128.0f;
        float v88[4l];
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
                float v95;
                v95 = v28[v94];
                float v96;
                v96 = v95 - v87;
                float v97;
                v97 = exp(v96);
                assert("Tensor range check" && 0 <= v89 && v89 < 1l);
                assert("Tensor range check" && 0 <= v91 && v91 < 4l);
                v88[v94] = v97;
                v91 += 1l ;
            }
            v89 += 1l ;
        }
        float v98;
        v98 = 0.0f;
        int v99;
        v99 = 0l;
        while (while_method_5(v99)){
            int v101;
            v101 = 0l;
            while (while_method_2(v101)){
                assert("Tensor range check" && 0 <= v99 && v99 < 1l);
                assert("Tensor range check" && 0 <= v101 && v101 < 4l);
                int v103;
                v103 = 4l * v99;
                int v104;
                v104 = v103 + v101;
                float v105;
                v105 = v88[v104];
                float v106;
                v106 = v98 + v105;
                v98 = v106;
                v101 += 1l ;
            }
            v99 += 1l ;
        }
        auto v107 = cooperative_groups::coalesced_threads();
        int v108;
        v108 = threadIdx.x;
        int v109;
        v109 = v108 / 32l;
        auto v110 = cooperative_groups::labeled_partition(v107,v109);
        float v111;
        v111 = cooperative_groups::reduce(v110, v98, v85);
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
                v119 = v88[v118];
                bool v120;
                v120 = v111 == 0.0f;
                bool v121;
                v121 = v120 != true;
                float v123;
                if (v121){
                    float v122;
                    v122 = v119 / v111;
                    v123 = v122;
                } else {
                    v123 = 0.0078125f;
                }
                assert("Tensor range check" && 0 <= v113 && v113 < 1l);
                assert("Tensor range check" && 0 <= v115 && v115 < 4l);
                v112[v118] = v123;
                v115 += 1l ;
            }
            v113 += 1l ;
        }
        int v124;
        v124 = 0l;
        while (while_method_5(v124)){
            assert("Tensor range check" && 0 <= v124 && v124 < 1l);
            int v126;
            v126 = 128l * v124;
            int v127;
            v127 = v126 + v27;
            assert("Tensor range check" && 0 <= v124 && v124 < 1l);
            int v128;
            v128 = 4l * v124;
            int4* v129;
            v129 = reinterpret_cast<int4*>(v112 + v128);
            int4* v130;
            v130 = reinterpret_cast<int4*>(v0 + v127);
            assert("Pointer alignment check" && (unsigned long long)(v129) % 4l == 0 && (unsigned long long)(v130) % 4l == 0);
            *v130 = *v129;
            v124 += 1l ;
        }
        v23 += 1l ;
    }
    __syncthreads();
    return ;
}
__device__ void method_4(int * v0, int v1, float * v2, int v3, curandStatePhilox4_32_10_t & v4){
    int v5;
    v5 = blockIdx.x;
    assert("Tensor range check" && 0 <= v5 && v5 < 1l);
    int v6;
    v6 = 4096l * v5;
    int v7;
    v7 = v6 + v3;
    int v8;
    v8 = blockIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 1l);
    int v9;
    v9 = 32l * v8;
    int v10;
    v10 = v9 + v1;
    int v11;
    v11 = threadIdx.x;
    bool v12;
    v12 = 0l <= v11;
    bool v13;
    v13 = v12 == false;
    if (v13){
        assert("The index needs to be zero or positive." && v12);
    } else {
    }
    int v15;
    v15 = v11 % 32l;
    int v16;
    v16 = v11 / 32l;
    bool v17;
    v17 = v16 < 1l;
    bool v18;
    v18 = v17 == false;
    if (v18){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v17);
    } else {
    }
    assert("Tensor range check" && 0 <= v16 && v16 < 1l);
    assert("Tensor range check" && 0 <= v15 && v15 < 32l);
    int v20;
    v20 = 4l * v15;
    int v21;
    v21 = v20 + v7;
    int v22;
    v22 = 128l * v16;
    int v23;
    v23 = v22 + v21;
    assert("Tensor range check" && 0 <= v16 && v16 < 1l);
    int v24;
    v24 = v16 + v10;
    int v25;
    v25 = 0l;
    while (while_method_7(v25)){
        assert("Tensor range check" && 0 <= v25 && v25 < 32l);
        int v27;
        v27 = 128l * v25;
        int v28;
        v28 = v27 + v23;
        float v29[4l];
        int v30[4l];
        int v31;
        v31 = 0l;
        while (while_method_5(v31)){
            assert("Tensor range check" && 0 <= v31 && v31 < 1l);
            int v33;
            v33 = 4l * v31;
            assert("Tensor range check" && 0 <= v31 && v31 < 1l);
            int v34;
            v34 = 128l * v31;
            int v35;
            v35 = v34 + v28;
            int4* v36;
            v36 = reinterpret_cast<int4*>(v2 + v35);
            int4* v37;
            v37 = reinterpret_cast<int4*>(v29 + v33);
            assert("Pointer alignment check" && (unsigned long long)(v36) % 4l == 0 && (unsigned long long)(v37) % 4l == 0);
            *v37 = *v36;
            v31 += 1l ;
        }
        int v38;
        v38 = 0l;
        while (while_method_5(v38)){
            int v40;
            v40 = 0l;
            while (while_method_2(v40)){
                bool v42;
                v42 = 0l <= v40;
                bool v44;
                if (v42){
                    bool v43;
                    v43 = v40 < 4l;
                    v44 = v43;
                } else {
                    v44 = false;
                }
                bool v45;
                v45 = v44 == false;
                if (v45){
                    assert("The indices should be inside the range of the dimension." && v44);
                } else {
                }
                bool v47;
                v47 = 0l <= v15;
                bool v49;
                if (v47){
                    bool v48;
                    v48 = v15 < 32l;
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
                int v52;
                v52 = v15 * 4l;
                int v53;
                v53 = v40 + v52;
                bool v54;
                v54 = 0l <= v38;
                bool v56;
                if (v54){
                    bool v55;
                    v55 = v38 < 1l;
                    v56 = v55;
                } else {
                    v56 = false;
                }
                bool v57;
                v57 = v56 == false;
                if (v57){
                    assert("The indices should be inside the range of the dimension." && v56);
                } else {
                }
                int v59;
                v59 = v38 * 128l;
                int v60;
                v60 = v53 + v59;
                assert("Tensor range check" && 0 <= v38 && v38 < 1l);
                assert("Tensor range check" && 0 <= v40 && v40 < 4l);
                int v61;
                v61 = 4l * v38;
                int v62;
                v62 = v61 + v40;
                v30[v62] = v60;
                v40 += 1l ;
            }
            v38 += 1l ;
        }
        bool v63;
        v63 = 0l <= v16;
        bool v64;
        v64 = v63 && v17;
        bool v65;
        v65 = v64 == false;
        if (v65){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v64);
        } else {
        }
        bool v67;
        v67 = 0l <= v25;
        bool v69;
        if (v67){
            bool v68;
            v68 = v25 < 32l;
            v69 = v68;
        } else {
            v69 = false;
        }
        bool v70;
        v70 = v69 == false;
        if (v70){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v69);
        } else {
        }
        int v72;
        v72 = v25 + v16;
        float v73[4l];
        float v74;
        v74 = 0.0f;
        int v75;
        v75 = 0l;
        while (while_method_5(v75)){
            assert("Tensor range check" && 0 <= v75 && v75 < 1l);
            int v77;
            v77 = 4l * v75;
            assert("Tensor range check" && 0 <= v75 && v75 < 1l);
            int v78; float v79;
            Tuple0 tmp0 = Tuple0{0l, 0.0f};
            v78 = tmp0.v0; v79 = tmp0.v1;
            while (while_method_2(v78)){
                assert("Tensor range check" && 0 <= v78 && v78 < 4l);
                int v81;
                v81 = v78 + v77;
                float v82;
                v82 = v29[v81];
                float v83;
                v83 = v79 + v82;
                v79 = v83;
                v78 += 1l ;
            }
            auto v84 = cooperative_groups::coalesced_threads();
            int v85;
            v85 = threadIdx.x;
            int v86;
            v86 = v85 / 32l;
            auto v87 = cooperative_groups::labeled_partition(v84,v86);
            Closure1 v88{};
            float v89;
            v89 = cooperative_groups::inclusive_scan(v87, v79, v88);
            float v90;
            v90 = v87.shfl_up(v89,1);
            bool v91;
            v91 = v87.thread_rank() == 0;
            float v92;
            if (v91){
                v92 = 0.0f;
            } else {
                v92 = v90;
            }
            float v93;
            v93 = v87.shfl(v89,v87.num_threads()-1);
            float v94;
            v94 = v74 + v92;
            int v95; float v96;
            Tuple0 tmp1 = Tuple0{0l, v94};
            v95 = tmp1.v0; v96 = tmp1.v1;
            while (while_method_2(v95)){
                assert("Tensor range check" && 0 <= v95 && v95 < 4l);
                int v98;
                v98 = v95 + v77;
                float v99;
                v99 = v29[v98];
                float v100;
                v100 = v96 + v99;
                assert("Tensor range check" && 0 <= v95 && v95 < 4l);
                v73[v98] = v100;
                v96 = v100;
                v95 += 1l ;
            }
            float v101;
            v101 = v74 + v93;
            v74 = v101;
            v75 += 1l ;
        }
        float v102;
        v102 = curand_uniform(&v4);
        float v103[4l];
        int v104;
        v104 = 0l;
        while (while_method_5(v104)){
            int v106;
            v106 = 0l;
            while (while_method_2(v106)){
                assert("Tensor range check" && 0 <= v104 && v104 < 1l);
                assert("Tensor range check" && 0 <= v106 && v106 < 4l);
                int v108;
                v108 = 4l * v104;
                int v109;
                v109 = v108 + v106;
                float v110;
                v110 = v73[v109];
                float v111;
                v111 = v110 - v102;
                assert("Tensor range check" && 0 <= v104 && v104 < 1l);
                assert("Tensor range check" && 0 <= v106 && v106 < 4l);
                v103[v109] = v111;
                v106 += 1l ;
            }
            v104 += 1l ;
        }
        float v112; int v113;
        Tuple1 tmp2 = Tuple1{-1.0f / 0.0f, 0l};
        v112 = tmp2.v0; v113 = tmp2.v1;
        int v114;
        v114 = 0l;
        while (while_method_5(v114)){
            int v116;
            v116 = 0l;
            while (while_method_2(v116)){
                assert("Tensor range check" && 0 <= v114 && v114 < 1l);
                assert("Tensor range check" && 0 <= v116 && v116 < 4l);
                int v118;
                v118 = 4l * v114;
                int v119;
                v119 = v118 + v116;
                float v120;
                v120 = v103[v119];
                int v121;
                v121 = v30[v119];
                bool v122;
                v122 = v112 >= 0.0f;
                bool v124;
                if (v122){
                    bool v123;
                    v123 = v120 >= 0.0f;
                    v124 = v123;
                } else {
                    v124 = false;
                }
                float v133; int v134;
                if (v124){
                    bool v125;
                    v125 = v112 <= v120;
                    if (v125){
                        v133 = v112; v134 = v113;
                    } else {
                        v133 = v120; v134 = v121;
                    }
                } else {
                    if (v122){
                        v133 = v112; v134 = v113;
                    } else {
                        bool v128;
                        v128 = v120 >= 0.0f;
                        if (v128){
                            v133 = v120; v134 = v121;
                        } else {
                            v133 = v112; v134 = v113;
                        }
                    }
                }
                v112 = v133;
                v113 = v134;
                v116 += 1l ;
            }
            v114 += 1l ;
        }
        auto v135 = cooperative_groups::coalesced_threads();
        int v136;
        v136 = threadIdx.x;
        int v137;
        v137 = v136 / 32l;
        auto v138 = cooperative_groups::labeled_partition(v135,v137);
        Closure2 v139{};
        float v140; int v141;
        Tuple1 tmp3 = cooperative_groups::reduce(v138, Tuple1{v112, v113}, v139);
        v140 = tmp3.v0; v141 = tmp3.v1;
        assert("Tensor range check" && 0 <= v25 && v25 < 32l);
        int v142;
        v142 = v25 + v24;
        v0[v142] = v141;
        v25 += 1l ;
    }
    __syncthreads();
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
__device__ void write_8(char v0){
    const char * v1;
    v1 = "%c";
    printf(v1,v0);
    return ;
}
__device__ void write_9(){
    return ;
}
__device__ void write_10(const char * v0){
    const char * v1;
    v1 = "%s";
    printf(v1,v0);
    return ;
}
__device__ void write_12(){
    const char * v0;
    v0 = "Call";
    return write_10(v0);
}
__device__ void write_13(){
    const char * v0;
    v0 = "Fold";
    return write_10(v0);
}
__device__ void write_14(){
    const char * v0;
    v0 = "Raise";
    return write_10(v0);
}
__device__ void write_11(Union1 v0){
    switch (v0.tag) {
        case 0: { // Call
            return write_12();
            break;
        }
        case 1: { // Fold
            return write_13();
            break;
        }
        case 2: { // Raise
            return write_14();
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void write_16(float v0){
    const char * v1;
    v1 = "%f";
    printf(v1,v0);
    return ;
}
__device__ void write_15(float * v0, int v1, int v2, int v3){
    int v4;
    v4 = 0l;
    char v5;
    v5 = '[';
    write_8(v5);
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
            write_10(v10);
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
            write_10(v13);
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
        write_16(v17);
        v6 += 1l ;
    }
    char v18;
    v18 = ']';
    return write_8(v18);
}
__device__ void write_7(Union1 v0, float * v1, int v2, int v3, int v4, float v5){
    char v6;
    v6 = '{';
    write_8(v6);
    write_9();
    const char * v7;
    v7 = "action";
    write_10(v7);
    const char * v8;
    v8 = " = ";
    write_10(v8);
    write_11(v0);
    const char * v9;
    v9 = "; ";
    write_10(v9);
    const char * v10;
    v10 = "sampling_prob_ensemble";
    write_10(v10);
    write_10(v8);
    write_15(v1, v2, v3, v4);
    write_10(v9);
    const char * v11;
    v11 = "sampling_prob_selected";
    write_10(v11);
    write_10(v8);
    write_16(v5);
    char v12;
    v12 = '}';
    return write_8(v12);
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
        method_3(v140, v142, v134);
        int * v143;
        v143 = reinterpret_cast<int *>(&v0[196608ull]);
        assert("Tensor range check" && 0 <= v103 && v103 < 4l);
        int v145;
        v145 = 32l * v103;
        method_4(v143, v145, v140, v142, v67);
        v103 += 1l ;
    }
    __syncthreads();
    int v146;
    v146 = 0l;
    int v147;
    v147 = 4l;
    int v148;
    v148 = int_range_5(v147, v146, v67);
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
    v160 = v159 % 3l;
    bool v161;
    v161 = 0l == v160;
    Union1 v167;
    if (v161){
        v167 = Union1{Union1_1{}};
    } else {
        bool v163;
        v163 = 1l == v160;
        if (v163){
            v167 = Union1{Union1_0{}};
        } else {
            v167 = Union1{Union1_2{}};
        }
    }
    int v168;
    v168 = blockIdx.x;
    assert("Tensor range check" && 0 <= v168 && v168 < 1l);
    int v169;
    v169 = 4096l * v168;
    int v170;
    v170 = threadIdx.x;
    assert("Tensor range check" && 0 <= v170 && v170 < 32l);
    int v171;
    v171 = 128l * v170;
    int v172;
    v172 = v171 + v169;
    assert("Tensor range check" && 0 <= v159 && v159 < 128l);
    int v173;
    v173 = v159 + v172;
    assert("Tensor range check" && 0 <= v148 && v148 < 4l);
    int v174;
    v174 = 4096l * v148;
    int v175;
    v175 = blockIdx.x;
    assert("Tensor range check" && 0 <= v175 && v175 < 1l);
    int v176;
    v176 = 4096l * v175;
    int v177;
    v177 = v176 + v174;
    int v178;
    v178 = threadIdx.x;
    assert("Tensor range check" && 0 <= v178 && v178 < 32l);
    int v179;
    v179 = 128l * v178;
    int v180;
    v180 = v179 + v177;
    assert("Tensor range check" && 0 <= v159 && v159 < 128l);
    int v181;
    v181 = v159 + v180;
    float v182;
    v182 = v149[v181];
    static cuda::binary_semaphore<cuda::thread_scope_system> v183(1l);
    v183.acquire();
    int v184;
    v184 = 4096l;
    int v185;
    v185 = 4l;
    write_7(v167, v149, v173, v184, v185, v182);
    printf("\n");
    v183.release();
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
    del v0, v1, v13
    cp.cuda.get_current_stream().synchronize()
    return 

if __name__ == '__main__': print(main())
