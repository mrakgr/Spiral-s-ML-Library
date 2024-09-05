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

__device__ void method_0(float * v0, float * v1, int v2, float * v3, int v4);
__device__ void method_1(float * v0, float * v1);
__device__ void method_2(float * v0, float * v1);
__device__ void method_3(float * v0, float * v1, int v2, float * v3, int v4);
struct Tuple0;
struct Tuple1;
struct Tuple2;
struct Tuple3;
__device__ void method_4(int * v0, float * v1, float * v2, curandStatePhilox4_32_10_t & v3);
__device__ void method_5(float * v0, int v1, float * v2, int v3, float * v4, int v5);
__device__ void method_6(float * v0, int v1, float * v2, int v3, float * v4, int v5);
__device__ void method_7(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5);
__device__ void method_8(float * v0, int v1, float * v2, int v3, float * v4, int v5);
__device__ void method_9(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5);
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
    bool v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(float t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure2 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
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
                return Tuple1{v5, true};
            } else {
                return Tuple1{v0, v1};
            }
        } else {
            if (v3){
                return Tuple1{v2, v3};
            } else {
                return Tuple1{v0, v1};
            }
        }
    }
};
struct Tuple2 {
    float v0;
    int v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ Tuple2 operator()(Tuple2 tup0, Tuple2 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple2{v0, v1};
        } else {
            return Tuple2{v2, v3};
        }
    }
};
struct Tuple3 {
    int v0;
    bool v1;
    __device__ Tuple3() = default;
    __device__ Tuple3(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
    __device__ Tuple3 operator()(Tuple3 tup0, Tuple3 tup1){
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
                return Tuple3{v5, true};
            } else {
                return Tuple3{v0, v1};
            }
        } else {
            if (v3){
                return Tuple3{v2, v3};
            } else {
                return Tuple3{v0, v1};
            }
        }
    }
};
struct Closure5 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v0 + v1;
        return v2;
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
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
    while (while_method_0(v64)){
        int v66;
        v66 = 0l;
        while (while_method_0(v66)){
            assert("Tensor range check" && 0 <= v64 && v64 < 1l);
            assert("Tensor range check" && 0 <= v66 && v66 < 1l);
            int v68;
            v68 = 16l * v66;
            int v69;
            v69 = v68 + v2;
            int v70;
            v70 = 256l * v64;
            int v71;
            v71 = v70 + v69;
            float * v72;
            v72 = v1+v71;
            // Pushing the loop unrolling to: 0
            int v74;
            v74 = 0l;
            #pragma unroll
            while (while_method_0(v74)){
                int v76;
                v76 = 0l;
                #pragma unroll
                while (while_method_0(v76)){
                    assert("Tensor range check" && 0 <= v74 && v74 < 1l);
                    assert("Tensor range check" && 0 <= v76 && v76 < 1l);
                    int v78;
                    v78 = v74 + v76;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v79 = v63[v78];
                    wmma::fill_fragment(v79, 0.0f);
                    v76 += 1l ;
                }
                v74 += 1l ;
            }
            int v80;
            v80 = 0l;
            #pragma unroll
            while (while_method_0(v80)){
                assert("Tensor range check" && 0 <= v64 && v64 < 1l);
                int v82;
                v82 = 128l * v64;
                int v83;
                v83 = v82 + v4;
                assert("Tensor range check" && 0 <= v80 && v80 < 1l);
                int v84;
                v84 = 8l * v80;
                int v85;
                v85 = v84 + v83;
                float * v86;
                v86 = v3+v85;
                assert("Tensor range check" && 0 <= v66 && v66 < 1l);
                int v88;
                v88 = 128l * v66;
                assert("Tensor range check" && 0 <= v80 && v80 < 1l);
                int v89;
                v89 = v84 + v88;
                float * v90;
                v90 = v0+v89;
                int v92;
                v92 = threadIdx.x;
                bool v93;
                v93 = 0l <= v92;
                bool v94;
                v94 = v93 == false;
                if (v94){
                    assert("The index needs to be zero or positive." && v93);
                } else {
                }
                int v96;
                v96 = v92 % 2l;
                int v97;
                v97 = v92 / 2l;
                bool v98;
                v98 = v97 < 16l;
                bool v99;
                v99 = v98 == false;
                if (v99){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v98);
                } else {
                }
                assert("Tensor range check" && 0 <= v97 && v97 < 16l);
                assert("Tensor range check" && 0 <= v96 && v96 < 2l);
                int v101;
                v101 = 4l * v96;
                int v102;
                v102 = 12l * v97;
                int v103;
                v103 = v102 + v101;
                int v104;
                v104 = 8l * v97;
                int v105;
                v105 = v104 + v101;
                float * v106;
                v106 = v13+v103;
                float * v108;
                v108 = v90+v105;
                int v110;
                v110 = 0l;
                #pragma unroll
                while (while_method_0(v110)){
                    int v112;
                    v112 = 0l;
                    #pragma unroll
                    while (while_method_0(v112)){
                        assert("Tensor range check" && 0 <= v110 && v110 < 1l);
                        assert("Tensor range check" && 0 <= v112 && v112 < 1l);
                        int v114;
                        v114 = 8l * v112;
                        int v115;
                        v115 = 192l * v110;
                        int v116;
                        v116 = v115 + v114;
                        int v117;
                        v117 = 128l * v110;
                        int v118;
                        v118 = v117 + v114;
                        float v119[4l];
                        int v120;
                        v120 = 0l;
                        #pragma unroll
                        while (while_method_1(v120)){
                            assert("Tensor range check" && 0 <= v120 && v120 < 4l);
                            int v122;
                            v122 = v120 + v118;
                            float v123;
                            v123 = v108[v122];
                            float v124;
                            v124 = wmma::__float_to_tf32(v123);
                            assert("Tensor range check" && 0 <= v120 && v120 < 4l);
                            v119[v120] = v124;
                            v120 += 1l ;
                        }
                        int4* v125;
                        v125 = reinterpret_cast<int4*>(v119 + 0l);
                        int4* v126;
                        v126 = reinterpret_cast<int4*>(v106 + v116);
                        assert("Pointer alignment check" && (unsigned long long)(v125) % 4l == 0 && (unsigned long long)(v126) % 4l == 0);
                        *v126 = *v125;
                        v112 += 1l ;
                    }
                    v110 += 1l ;
                }
                int v127;
                v127 = threadIdx.x;
                bool v128;
                v128 = 0l <= v127;
                bool v129;
                v129 = v128 == false;
                if (v129){
                    assert("The index needs to be zero or positive." && v128);
                } else {
                }
                int v131;
                v131 = v127 % 2l;
                int v132;
                v132 = v127 / 2l;
                bool v133;
                v133 = v132 < 16l;
                bool v134;
                v134 = v133 == false;
                if (v134){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v133);
                } else {
                }
                assert("Tensor range check" && 0 <= v132 && v132 < 16l);
                assert("Tensor range check" && 0 <= v131 && v131 < 2l);
                int v136;
                v136 = 4l * v131;
                int v137;
                v137 = 12l * v132;
                int v138;
                v138 = v137 + v136;
                int v139;
                v139 = 8l * v132;
                int v140;
                v140 = v139 + v136;
                float * v141;
                v141 = v11+v138;
                float * v143;
                v143 = v86+v140;
                int v145;
                v145 = 0l;
                #pragma unroll
                while (while_method_0(v145)){
                    int v147;
                    v147 = 0l;
                    #pragma unroll
                    while (while_method_0(v147)){
                        assert("Tensor range check" && 0 <= v145 && v145 < 1l);
                        assert("Tensor range check" && 0 <= v147 && v147 < 1l);
                        int v149;
                        v149 = 8l * v147;
                        int v150;
                        v150 = 192l * v145;
                        int v151;
                        v151 = v150 + v149;
                        int v152;
                        v152 = 128l * v145;
                        int v153;
                        v153 = v152 + v149;
                        float v154[4l];
                        int v155;
                        v155 = 0l;
                        #pragma unroll
                        while (while_method_1(v155)){
                            assert("Tensor range check" && 0 <= v155 && v155 < 4l);
                            int v157;
                            v157 = v155 + v153;
                            float v158;
                            v158 = v143[v157];
                            float v159;
                            v159 = wmma::__float_to_tf32(v158);
                            assert("Tensor range check" && 0 <= v155 && v155 < 4l);
                            v154[v155] = v159;
                            v155 += 1l ;
                        }
                        int4* v160;
                        v160 = reinterpret_cast<int4*>(v154 + 0l);
                        int4* v161;
                        v161 = reinterpret_cast<int4*>(v141 + v151);
                        assert("Pointer alignment check" && (unsigned long long)(v160) % 4l == 0 && (unsigned long long)(v161) % 4l == 0);
                        *v161 = *v160;
                        v147 += 1l ;
                    }
                    v145 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v162[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v163[1l];
                int v164;
                v164 = 0l;
                #pragma unroll
                while (while_method_0(v164)){
                    int v166;
                    v166 = 0l;
                    #pragma unroll
                    while (while_method_0(v166)){
                        assert("Tensor range check" && 0 <= v164 && v164 < 1l);
                        assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                        int v168;
                        v168 = v164 + v166;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v169 = v162[v168];
                        assert("Tensor range check" && 0 <= v164 && v164 < 1l);
                        int v170;
                        v170 = 192l * v164;
                        assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                        int v171;
                        v171 = 8l * v166;
                        int v172;
                        v172 = v171 + v170;
                        int v173;
                        v173 = 0l;
                        #pragma unroll
                        while (while_method_2(v173)){
                            int v175;
                            v175 = 0l;
                            #pragma unroll
                            while (while_method_2(v175)){
                                assert("Tensor range check" && 0 <= v173 && v173 < 2l);
                                assert("Tensor range check" && 0 <= v175 && v175 < 2l);
                                int v177;
                                v177 = 96l * v175;
                                int v178;
                                v178 = v177 + v172;
                                int v179;
                                v179 = 4l * v173;
                                int v180;
                                v180 = v179 + v178;
                                float v181;
                                v181 = v45[v180];
                                bool v182;
                                v182 = 0l <= v175;
                                bool v184;
                                if (v182){
                                    bool v183;
                                    v183 = v175 < 2l;
                                    v184 = v183;
                                } else {
                                    v184 = false;
                                }
                                bool v185;
                                v185 = v184 == false;
                                if (v185){
                                    assert("The indices should be inside the range of the dimension." && v184);
                                } else {
                                }
                                bool v187;
                                v187 = 0l <= v173;
                                bool v189;
                                if (v187){
                                    bool v188;
                                    v188 = v173 < 2l;
                                    v189 = v188;
                                } else {
                                    v189 = false;
                                }
                                bool v190;
                                v190 = v189 == false;
                                if (v190){
                                    assert("The indices should be inside the range of the dimension." && v189);
                                } else {
                                }
                                int v192;
                                v192 = v173 * 2l;
                                int v193;
                                v193 = v175 + v192;
                                v169.x[v193] = v181;
                                v175 += 1l ;
                            }
                            v173 += 1l ;
                        }
                        v166 += 1l ;
                    }
                    v164 += 1l ;
                }
                int v194;
                v194 = 0l;
                #pragma unroll
                while (while_method_0(v194)){
                    int v196;
                    v196 = 0l;
                    #pragma unroll
                    while (while_method_0(v196)){
                        assert("Tensor range check" && 0 <= v194 && v194 < 1l);
                        assert("Tensor range check" && 0 <= v196 && v196 < 1l);
                        int v198;
                        v198 = v194 + v196;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v199 = v163[v198];
                        assert("Tensor range check" && 0 <= v194 && v194 < 1l);
                        int v200;
                        v200 = 192l * v194;
                        assert("Tensor range check" && 0 <= v196 && v196 < 1l);
                        int v201;
                        v201 = 8l * v196;
                        int v202;
                        v202 = v201 + v200;
                        int v203;
                        v203 = 0l;
                        #pragma unroll
                        while (while_method_2(v203)){
                            int v205;
                            v205 = 0l;
                            #pragma unroll
                            while (while_method_2(v205)){
                                assert("Tensor range check" && 0 <= v203 && v203 < 2l);
                                assert("Tensor range check" && 0 <= v205 && v205 < 2l);
                                int v207;
                                v207 = 4l * v205;
                                int v208;
                                v208 = v207 + v202;
                                int v209;
                                v209 = 96l * v203;
                                int v210;
                                v210 = v209 + v208;
                                float v211;
                                v211 = v61[v210];
                                bool v212;
                                v212 = 0l <= v205;
                                bool v214;
                                if (v212){
                                    bool v213;
                                    v213 = v205 < 2l;
                                    v214 = v213;
                                } else {
                                    v214 = false;
                                }
                                bool v215;
                                v215 = v214 == false;
                                if (v215){
                                    assert("The indices should be inside the range of the dimension." && v214);
                                } else {
                                }
                                bool v217;
                                v217 = 0l <= v203;
                                bool v219;
                                if (v217){
                                    bool v218;
                                    v218 = v203 < 2l;
                                    v219 = v218;
                                } else {
                                    v219 = false;
                                }
                                bool v220;
                                v220 = v219 == false;
                                if (v220){
                                    assert("The indices should be inside the range of the dimension." && v219);
                                } else {
                                }
                                int v222;
                                v222 = v203 * 2l;
                                int v223;
                                v223 = v205 + v222;
                                v199.x[v223] = v211;
                                v205 += 1l ;
                            }
                            v203 += 1l ;
                        }
                        v196 += 1l ;
                    }
                    v194 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                int v224;
                v224 = 0l;
                #pragma unroll
                while (while_method_0(v224)){
                    int v226;
                    v226 = 0l;
                    #pragma unroll
                    while (while_method_0(v226)){
                        int v228;
                        v228 = 0l;
                        #pragma unroll
                        while (while_method_0(v228)){
                            assert("Tensor range check" && 0 <= v224 && v224 < 1l);
                            assert("Tensor range check" && 0 <= v226 && v226 < 1l);
                            int v230;
                            v230 = v224 + v226;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v231 = v63[v230];
                            assert("Tensor range check" && 0 <= v224 && v224 < 1l);
                            assert("Tensor range check" && 0 <= v228 && v228 < 1l);
                            int v232;
                            v232 = v224 + v228;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v233 = v162[v232];
                            assert("Tensor range check" && 0 <= v226 && v226 < 1l);
                            assert("Tensor range check" && 0 <= v228 && v228 < 1l);
                            int v234;
                            v234 = v226 + v228;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v235 = v163[v234];
                            wmma::mma_sync(v231, v233, v235, v231);
                            v228 += 1l ;
                        }
                        v226 += 1l ;
                    }
                    v224 += 1l ;
                }
                v80 += 1l ;
            }
            int v236;
            v236 = 0l;
            #pragma unroll
            while (while_method_0(v236)){
                int v238;
                v238 = 0l;
                #pragma unroll
                while (while_method_0(v238)){
                    assert("Tensor range check" && 0 <= v236 && v236 < 1l);
                    assert("Tensor range check" && 0 <= v238 && v238 < 1l);
                    int v240;
                    v240 = v236 + v238;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v241 = v63[v240];
                    assert("Tensor range check" && 0 <= v236 && v236 < 1l);
                    assert("Tensor range check" && 0 <= v238 && v238 < 1l);
                    int v242;
                    v242 = 16l * v238;
                    int v243;
                    v243 = 384l * v236;
                    int v244;
                    v244 = v243 + v242;
                    float * v245;
                    v245 = v29+v244;
                    wmma::store_matrix_sync(v245, v241, 24l, wmma::mem_row_major);
                    v238 += 1l ;
                }
                v236 += 1l ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0l));
            int v247;
            v247 = threadIdx.x;
            bool v248;
            v248 = 0l <= v247;
            bool v249;
            v249 = v248 == false;
            if (v249){
                assert("The index needs to be zero or positive." && v248);
            } else {
            }
            int v251;
            v251 = v247 % 4l;
            int v252;
            v252 = v247 / 4l;
            bool v253;
            v253 = v252 < 8l;
            bool v254;
            v254 = v253 == false;
            if (v254){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v253);
            } else {
            }
            assert("Tensor range check" && 0 <= v252 && v252 < 8l);
            assert("Tensor range check" && 0 <= v251 && v251 < 4l);
            int v256;
            v256 = 4l * v251;
            int v257;
            v257 = 16l * v252;
            int v258;
            v258 = v257 + v256;
            int v259;
            v259 = 24l * v252;
            int v260;
            v260 = v259 + v256;
            float * v261;
            v261 = v72+v258;
            float * v263;
            v263 = v15+v260;
            int v265;
            v265 = 0l;
            #pragma unroll
            while (while_method_2(v265)){
                int v267;
                v267 = 0l;
                #pragma unroll
                while (while_method_0(v267)){
                    assert("Tensor range check" && 0 <= v265 && v265 < 2l);
                    assert("Tensor range check" && 0 <= v267 && v267 < 1l);
                    int v269;
                    v269 = 16l * v267;
                    int v270;
                    v270 = 128l * v265;
                    int v271;
                    v271 = v270 + v269;
                    int v272;
                    v272 = 192l * v265;
                    int v273;
                    v273 = v272 + v269;
                    int4* v274;
                    v274 = reinterpret_cast<int4*>(v263 + v273);
                    int4* v275;
                    v275 = reinterpret_cast<int4*>(v261 + v271);
                    assert("Pointer alignment check" && (unsigned long long)(v274) % 4l == 0 && (unsigned long long)(v275) % 4l == 0);
                    *v275 = *v274;
                    v267 += 1l ;
                }
                v265 += 1l ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0l));
            // Poping the loop unrolling to: 0
            v66 += 1l ;
        }
        v64 += 1l ;
    }
    return ;
}
__device__ void method_1(float * v0, float * v1){
    int v2;
    v2 = blockIdx.x;
    assert("Tensor range check" && 0 <= v2 && v2 < 24l);
    int v3;
    v3 = 256l * v2;
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 24l);
    int v5;
    v5 = 256l * v4;
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
    v10 = v6 % 4l;
    int v11;
    v11 = v6 / 4l;
    bool v12;
    v12 = v11 < 8l;
    bool v13;
    v13 = v12 == false;
    if (v13){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v12);
    } else {
    }
    assert("Tensor range check" && 0 <= v11 && v11 < 8l);
    assert("Tensor range check" && 0 <= v10 && v10 < 4l);
    int v15;
    v15 = 4l * v10;
    int v16;
    v16 = v15 + v3;
    int v17;
    v17 = 16l * v11;
    int v18;
    v18 = v17 + v16;
    assert("Tensor range check" && 0 <= v11 && v11 < 8l);
    assert("Tensor range check" && 0 <= v10 && v10 < 4l);
    int v19;
    v19 = v15 + v5;
    int v20;
    v20 = v17 + v19;
    int v21;
    v21 = 0l;
    while (while_method_2(v21)){
        assert("Tensor range check" && 0 <= v21 && v21 < 2l);
        int v23;
        v23 = 128l * v21;
        int v24;
        v24 = v23 + v18;
        float v25[4l];
        int v26[4l];
        int v27;
        v27 = 0l;
        while (while_method_0(v27)){
            assert("Tensor range check" && 0 <= v27 && v27 < 1l);
            int v29;
            v29 = 4l * v27;
            assert("Tensor range check" && 0 <= v27 && v27 < 1l);
            int v30;
            v30 = 16l * v27;
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
        while (while_method_0(v34)){
            int v36;
            v36 = 0l;
            while (while_method_1(v36)){
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
                    v44 = v10 < 4l;
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
                v55 = v34 * 16l;
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
            v64 = v21 < 2l;
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
        v68 = v21 * 8l;
        int v69;
        v69 = v68 + v11;
        float v70[4l];
        int v71;
        v71 = 0l;
        while (while_method_0(v71)){
            int v73;
            v73 = 0l;
            while (while_method_1(v73)){
                assert("Tensor range check" && 0 <= v71 && v71 < 1l);
                assert("Tensor range check" && 0 <= v73 && v73 < 4l);
                int v75;
                v75 = 4l * v71;
                int v76;
                v76 = v75 + v73;
                float v77;
                v77 = v25[v76];
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
        while (while_method_0(v80)){
            int v82;
            v82 = 0l;
            while (while_method_1(v82)){
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
        v90 = v89 / 4l;
        auto v91 = cooperative_groups::labeled_partition(v88,v90);
        Closure0 v92{};
        float v93;
        v93 = cooperative_groups::reduce(v91, v79, v92);
        float v94[4l];
        int v95;
        v95 = 0l;
        while (while_method_0(v95)){
            int v97;
            v97 = 0l;
            while (while_method_1(v97)){
                assert("Tensor range check" && 0 <= v95 && v95 < 1l);
                assert("Tensor range check" && 0 <= v97 && v97 < 4l);
                int v99;
                v99 = 4l * v95;
                int v100;
                v100 = v99 + v97;
                float v101;
                v101 = v25[v100];
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
        assert("Tensor range check" && 0 <= v21 && v21 < 2l);
        int v106;
        v106 = v23 + v20;
        int v107;
        v107 = 0l;
        while (while_method_0(v107)){
            assert("Tensor range check" && 0 <= v107 && v107 < 1l);
            int v109;
            v109 = 16l * v107;
            int v110;
            v110 = v109 + v106;
            assert("Tensor range check" && 0 <= v107 && v107 < 1l);
            int v111;
            v111 = 4l * v107;
            int4* v112;
            v112 = reinterpret_cast<int4*>(v94 + v111);
            int4* v113;
            v113 = reinterpret_cast<int4*>(v0 + v110);
            assert("Pointer alignment check" && (unsigned long long)(v112) % 4l == 0 && (unsigned long long)(v113) % 4l == 0);
            *v113 = *v112;
            v107 += 1l ;
        }
        v21 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return ;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 64l;
    return v1;
}
__device__ void method_2(float * v0, float * v1){
    int v2;
    v2 = blockIdx.x;
    assert("Tensor range check" && 0 <= v2 && v2 < 24l);
    int v3;
    v3 = 256l * v2;
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 24l);
    int v5;
    v5 = 256l * v4;
    int v6;
    v6 = threadIdx.x;
    int v7;
    v7 = v6;
    while (while_method_3(v7)){
        bool v9;
        v9 = 0l <= v7;
        bool v10;
        v10 = v9 == false;
        if (v10){
            assert("The index needs to be zero or positive." && v9);
        } else {
        }
        int v12;
        v12 = v7 % 4l;
        int v13;
        v13 = v7 / 4l;
        bool v14;
        v14 = v13 < 16l;
        bool v15;
        v15 = v14 == false;
        if (v15){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v14);
        } else {
        }
        assert("Tensor range check" && 0 <= v13 && v13 < 16l);
        assert("Tensor range check" && 0 <= v12 && v12 < 4l);
        int v17;
        v17 = 4l * v12;
        int v18;
        v18 = v17 + v3;
        int v19;
        v19 = 16l * v13;
        int v20;
        v20 = v19 + v18;
        assert("Tensor range check" && 0 <= v13 && v13 < 16l);
        assert("Tensor range check" && 0 <= v12 && v12 < 4l);
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
        while (while_method_1(v27)){
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
__device__ void method_3(float * v0, float * v1, int v2, float * v3, int v4){
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
    while (while_method_0(v64)){
        int v66;
        v66 = 0l;
        while (while_method_0(v66)){
            assert("Tensor range check" && 0 <= v64 && v64 < 1l);
            assert("Tensor range check" && 0 <= v66 && v66 < 1l);
            int v68;
            v68 = 16l * v66;
            int v69;
            v69 = v68 + v2;
            int v70;
            v70 = 256l * v64;
            int v71;
            v71 = v70 + v69;
            float * v72;
            v72 = v1+v71;
            // Pushing the loop unrolling to: 0
            int v74;
            v74 = 0l;
            #pragma unroll
            while (while_method_0(v74)){
                int v76;
                v76 = 0l;
                #pragma unroll
                while (while_method_0(v76)){
                    assert("Tensor range check" && 0 <= v74 && v74 < 1l);
                    assert("Tensor range check" && 0 <= v76 && v76 < 1l);
                    int v78;
                    v78 = v74 + v76;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v79 = v63[v78];
                    wmma::fill_fragment(v79, 0.0f);
                    v76 += 1l ;
                }
                v74 += 1l ;
            }
            int v80;
            v80 = 0l;
            #pragma unroll
            while (while_method_2(v80)){
                assert("Tensor range check" && 0 <= v64 && v64 < 1l);
                int v82;
                v82 = v70 + v4;
                assert("Tensor range check" && 0 <= v80 && v80 < 2l);
                int v83;
                v83 = 8l * v80;
                int v84;
                v84 = v83 + v82;
                float * v85;
                v85 = v3+v84;
                assert("Tensor range check" && 0 <= v66 && v66 < 1l);
                int v87;
                v87 = 256l * v66;
                assert("Tensor range check" && 0 <= v80 && v80 < 2l);
                int v88;
                v88 = v83 + v87;
                float * v89;
                v89 = v0+v88;
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
                v103 = 16l * v96;
                int v104;
                v104 = v103 + v100;
                float * v105;
                v105 = v13+v102;
                float * v107;
                v107 = v89+v104;
                int v109;
                v109 = 0l;
                #pragma unroll
                while (while_method_0(v109)){
                    int v111;
                    v111 = 0l;
                    #pragma unroll
                    while (while_method_0(v111)){
                        assert("Tensor range check" && 0 <= v109 && v109 < 1l);
                        assert("Tensor range check" && 0 <= v111 && v111 < 1l);
                        int v113;
                        v113 = 8l * v111;
                        int v114;
                        v114 = 192l * v109;
                        int v115;
                        v115 = v114 + v113;
                        int v116;
                        v116 = 256l * v109;
                        int v117;
                        v117 = v116 + v113;
                        float v118[4l];
                        int v119;
                        v119 = 0l;
                        #pragma unroll
                        while (while_method_1(v119)){
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
                v138 = 16l * v131;
                int v139;
                v139 = v138 + v135;
                float * v140;
                v140 = v11+v137;
                float * v142;
                v142 = v85+v139;
                int v144;
                v144 = 0l;
                #pragma unroll
                while (while_method_0(v144)){
                    int v146;
                    v146 = 0l;
                    #pragma unroll
                    while (while_method_0(v146)){
                        assert("Tensor range check" && 0 <= v144 && v144 < 1l);
                        assert("Tensor range check" && 0 <= v146 && v146 < 1l);
                        int v148;
                        v148 = 8l * v146;
                        int v149;
                        v149 = 192l * v144;
                        int v150;
                        v150 = v149 + v148;
                        int v151;
                        v151 = 256l * v144;
                        int v152;
                        v152 = v151 + v148;
                        float v153[4l];
                        int v154;
                        v154 = 0l;
                        #pragma unroll
                        while (while_method_1(v154)){
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
                asm("barrier.cta.sync %0;" :: "r"(0l));
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v161[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v162[1l];
                int v163;
                v163 = 0l;
                #pragma unroll
                while (while_method_0(v163)){
                    int v165;
                    v165 = 0l;
                    #pragma unroll
                    while (while_method_0(v165)){
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
                        while (while_method_2(v172)){
                            int v174;
                            v174 = 0l;
                            #pragma unroll
                            while (while_method_2(v174)){
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
                while (while_method_0(v193)){
                    int v195;
                    v195 = 0l;
                    #pragma unroll
                    while (while_method_0(v195)){
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
                        while (while_method_2(v202)){
                            int v204;
                            v204 = 0l;
                            #pragma unroll
                            while (while_method_2(v204)){
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
                asm("barrier.cta.sync %0;" :: "r"(0l));
                int v223;
                v223 = 0l;
                #pragma unroll
                while (while_method_0(v223)){
                    int v225;
                    v225 = 0l;
                    #pragma unroll
                    while (while_method_0(v225)){
                        int v227;
                        v227 = 0l;
                        #pragma unroll
                        while (while_method_0(v227)){
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
                v80 += 1l ;
            }
            int v235;
            v235 = 0l;
            #pragma unroll
            while (while_method_0(v235)){
                int v237;
                v237 = 0l;
                #pragma unroll
                while (while_method_0(v237)){
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
            asm("barrier.cta.sync %0;" :: "r"(0l));
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
            v256 = 16l * v251;
            int v257;
            v257 = v256 + v255;
            int v258;
            v258 = 24l * v251;
            int v259;
            v259 = v258 + v255;
            float * v260;
            v260 = v72+v257;
            float * v262;
            v262 = v15+v259;
            int v264;
            v264 = 0l;
            #pragma unroll
            while (while_method_2(v264)){
                int v266;
                v266 = 0l;
                #pragma unroll
                while (while_method_0(v266)){
                    assert("Tensor range check" && 0 <= v264 && v264 < 2l);
                    assert("Tensor range check" && 0 <= v266 && v266 < 1l);
                    int v268;
                    v268 = 16l * v266;
                    int v269;
                    v269 = 128l * v264;
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
            asm("barrier.cta.sync %0;" :: "r"(0l));
            // Poping the loop unrolling to: 0
            v66 += 1l ;
        }
        v64 += 1l ;
    }
    return ;
}
__device__ void method_4(int * v0, float * v1, float * v2, curandStatePhilox4_32_10_t & v3){
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 24l);
    int v5;
    v5 = 256l * v4;
    int v6;
    v6 = blockIdx.x;
    assert("Tensor range check" && 0 <= v6 && v6 < 24l);
    int v7;
    v7 = 256l * v6;
    int v8;
    v8 = blockIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 24l);
    int v9;
    v9 = 16l * v8;
    int v10;
    v10 = threadIdx.x;
    bool v11;
    v11 = 0l <= v10;
    bool v12;
    v12 = v11 == false;
    if (v12){
        assert("The index needs to be zero or positive." && v11);
    } else {
    }
    int v14;
    v14 = v10 % 4l;
    int v15;
    v15 = v10 / 4l;
    bool v16;
    v16 = v15 < 8l;
    bool v17;
    v17 = v16 == false;
    if (v17){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v16);
    } else {
    }
    assert("Tensor range check" && 0 <= v15 && v15 < 8l);
    assert("Tensor range check" && 0 <= v14 && v14 < 4l);
    int v19;
    v19 = 4l * v14;
    int v20;
    v20 = v19 + v5;
    int v21;
    v21 = 16l * v15;
    int v22;
    v22 = v21 + v20;
    assert("Tensor range check" && 0 <= v15 && v15 < 8l);
    assert("Tensor range check" && 0 <= v14 && v14 < 4l);
    int v23;
    v23 = v19 + v7;
    int v24;
    v24 = v21 + v23;
    assert("Tensor range check" && 0 <= v15 && v15 < 8l);
    int v25;
    v25 = v15 + v9;
    int v26;
    v26 = 0l;
    while (while_method_2(v26)){
        assert("Tensor range check" && 0 <= v26 && v26 < 2l);
        int v28;
        v28 = 128l * v26;
        int v29;
        v29 = v28 + v22;
        float v30[4l];
        int v31[4l];
        int v32;
        v32 = 0l;
        while (while_method_0(v32)){
            assert("Tensor range check" && 0 <= v32 && v32 < 1l);
            int v34;
            v34 = 4l * v32;
            assert("Tensor range check" && 0 <= v32 && v32 < 1l);
            int v35;
            v35 = 16l * v32;
            int v36;
            v36 = v35 + v29;
            int4* v37;
            v37 = reinterpret_cast<int4*>(v2 + v36);
            int4* v38;
            v38 = reinterpret_cast<int4*>(v30 + v34);
            assert("Pointer alignment check" && (unsigned long long)(v37) % 4l == 0 && (unsigned long long)(v38) % 4l == 0);
            *v38 = *v37;
            v32 += 1l ;
        }
        int v39;
        v39 = 0l;
        while (while_method_0(v39)){
            int v41;
            v41 = 0l;
            while (while_method_1(v41)){
                bool v43;
                v43 = 0l <= v41;
                bool v45;
                if (v43){
                    bool v44;
                    v44 = v41 < 4l;
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
                bool v48;
                v48 = 0l <= v14;
                bool v50;
                if (v48){
                    bool v49;
                    v49 = v14 < 4l;
                    v50 = v49;
                } else {
                    v50 = false;
                }
                bool v51;
                v51 = v50 == false;
                if (v51){
                    assert("The indices should be inside the range of the dimension." && v50);
                } else {
                }
                int v53;
                v53 = v14 * 4l;
                int v54;
                v54 = v41 + v53;
                bool v55;
                v55 = 0l <= v39;
                bool v57;
                if (v55){
                    bool v56;
                    v56 = v39 < 1l;
                    v57 = v56;
                } else {
                    v57 = false;
                }
                bool v58;
                v58 = v57 == false;
                if (v58){
                    assert("The indices should be inside the range of the dimension." && v57);
                } else {
                }
                int v60;
                v60 = v39 * 16l;
                int v61;
                v61 = v54 + v60;
                assert("Tensor range check" && 0 <= v39 && v39 < 1l);
                assert("Tensor range check" && 0 <= v41 && v41 < 4l);
                int v62;
                v62 = 4l * v39;
                int v63;
                v63 = v62 + v41;
                v31[v63] = v61;
                v41 += 1l ;
            }
            v39 += 1l ;
        }
        bool v64;
        v64 = 0l <= v15;
        bool v65;
        v65 = v64 && v16;
        bool v66;
        v66 = v65 == false;
        if (v66){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v65);
        } else {
        }
        bool v68;
        v68 = 0l <= v26;
        bool v70;
        if (v68){
            bool v69;
            v69 = v26 < 2l;
            v70 = v69;
        } else {
            v70 = false;
        }
        bool v71;
        v71 = v70 == false;
        if (v71){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v70);
        } else {
        }
        int v73;
        v73 = v26 * 8l;
        int v74;
        v74 = v73 + v15;
        float v75;
        v75 = 0.0f;
        int v76;
        v76 = 0l;
        while (while_method_0(v76)){
            int v78;
            v78 = 0l;
            while (while_method_1(v78)){
                assert("Tensor range check" && 0 <= v76 && v76 < 1l);
                assert("Tensor range check" && 0 <= v78 && v78 < 4l);
                int v80;
                v80 = 4l * v76;
                int v81;
                v81 = v80 + v78;
                float v82;
                v82 = v30[v81];
                float v83;
                v83 = v75 + v82;
                v75 = v83;
                v78 += 1l ;
            }
            v76 += 1l ;
        }
        auto v84 = cooperative_groups::coalesced_threads();
        int v85;
        v85 = threadIdx.x;
        int v86;
        v86 = v85 / 4l;
        auto v87 = cooperative_groups::labeled_partition(v84,v86);
        Closure0 v88{};
        float v89;
        v89 = cooperative_groups::reduce(v87, v75, v88);
        float v90;
        v90 = v89 / 16.0f;
        float v91[4l];
        int v92;
        v92 = 0l;
        while (while_method_0(v92)){
            int v94;
            v94 = 0l;
            while (while_method_1(v94)){
                assert("Tensor range check" && 0 <= v92 && v92 < 1l);
                assert("Tensor range check" && 0 <= v94 && v94 < 4l);
                int v96;
                v96 = 4l * v92;
                int v97;
                v97 = v96 + v94;
                float v98;
                v98 = v30[v97];
                float v99;
                v99 = v98 - v90;
                float v100;
                v100 = exp(v99);
                assert("Tensor range check" && 0 <= v92 && v92 < 1l);
                assert("Tensor range check" && 0 <= v94 && v94 < 4l);
                v91[v97] = v100;
                v94 += 1l ;
            }
            v92 += 1l ;
        }
        float v101;
        v101 = 0.0f;
        int v102;
        v102 = 0l;
        while (while_method_0(v102)){
            int v104;
            v104 = 0l;
            while (while_method_1(v104)){
                assert("Tensor range check" && 0 <= v102 && v102 < 1l);
                assert("Tensor range check" && 0 <= v104 && v104 < 4l);
                int v106;
                v106 = 4l * v102;
                int v107;
                v107 = v106 + v104;
                float v108;
                v108 = v91[v107];
                float v109;
                v109 = v101 + v108;
                v101 = v109;
                v104 += 1l ;
            }
            v102 += 1l ;
        }
        auto v110 = cooperative_groups::coalesced_threads();
        int v111;
        v111 = threadIdx.x;
        int v112;
        v112 = v111 / 4l;
        auto v113 = cooperative_groups::labeled_partition(v110,v112);
        float v114;
        v114 = cooperative_groups::reduce(v113, v101, v88);
        float v115[4l];
        int v116;
        v116 = 0l;
        while (while_method_0(v116)){
            int v118;
            v118 = 0l;
            while (while_method_1(v118)){
                assert("Tensor range check" && 0 <= v116 && v116 < 1l);
                assert("Tensor range check" && 0 <= v118 && v118 < 4l);
                int v120;
                v120 = 4l * v116;
                int v121;
                v121 = v120 + v118;
                float v122;
                v122 = v91[v121];
                float v123;
                v123 = v122 / v114;
                assert("Tensor range check" && 0 <= v116 && v116 < 1l);
                assert("Tensor range check" && 0 <= v118 && v118 < 4l);
                v115[v121] = v123;
                v118 += 1l ;
            }
            v116 += 1l ;
        }
        float v124[4l];
        float v125;
        v125 = 0.0f;
        int v126;
        v126 = 0l;
        while (while_method_0(v126)){
            assert("Tensor range check" && 0 <= v126 && v126 < 1l);
            int v128;
            v128 = 4l * v126;
            assert("Tensor range check" && 0 <= v126 && v126 < 1l);
            int v129; float v130;
            Tuple0 tmp0 = Tuple0{0l, 0.0f};
            v129 = tmp0.v0; v130 = tmp0.v1;
            while (while_method_1(v129)){
                assert("Tensor range check" && 0 <= v129 && v129 < 4l);
                int v132;
                v132 = v129 + v128;
                float v133;
                v133 = v115[v132];
                float v134;
                v134 = v130 + v133;
                v130 = v134;
                v129 += 1l ;
            }
            auto v135 = cooperative_groups::coalesced_threads();
            int v136;
            v136 = threadIdx.x;
            int v137;
            v137 = v136 / 4l;
            auto v138 = cooperative_groups::labeled_partition(v135,v137);
            Closure1 v139{};
            float v140;
            v140 = cooperative_groups::inclusive_scan(v138, v130, v139);
            float v141;
            v141 = v138.shfl_up(v140,1);
            bool v142;
            v142 = v138.thread_rank() == 0;
            float v143;
            if (v142){
                v143 = 0.0f;
            } else {
                v143 = v141;
            }
            float v144;
            v144 = v138.shfl(v140,v138.num_threads()-1);
            float v145;
            v145 = v125 + v143;
            int v146; float v147;
            Tuple0 tmp1 = Tuple0{0l, v145};
            v146 = tmp1.v0; v147 = tmp1.v1;
            while (while_method_1(v146)){
                assert("Tensor range check" && 0 <= v146 && v146 < 4l);
                int v149;
                v149 = v146 + v128;
                float v150;
                v150 = v115[v149];
                float v151;
                v151 = v147 + v150;
                assert("Tensor range check" && 0 <= v146 && v146 < 4l);
                v124[v149] = v151;
                v147 = v151;
                v146 += 1l ;
            }
            float v152;
            v152 = v125 + v144;
            v125 = v152;
            v126 += 1l ;
        }
        float v153[4l];
        bool v154[4l];
        int v155;
        v155 = 0l;
        while (while_method_0(v155)){
            int v157;
            v157 = 0l;
            while (while_method_1(v157)){
                assert("Tensor range check" && 0 <= v155 && v155 < 1l);
                assert("Tensor range check" && 0 <= v157 && v157 < 4l);
                int v159;
                v159 = 4l * v155;
                int v160;
                v160 = v159 + v157;
                float v161;
                v161 = v124[v160];
                float v162;
                v162 = v115[v160];
                bool v163;
                v163 = v162 > 0.0f;
                assert("Tensor range check" && 0 <= v155 && v155 < 1l);
                assert("Tensor range check" && 0 <= v157 && v157 < 4l);
                v153[v160] = v161;
                v154[v160] = v163;
                v157 += 1l ;
            }
            v155 += 1l ;
        }
        float v164; bool v165;
        Tuple1 tmp2 = Tuple1{-1.0f / 0.0f, false};
        v164 = tmp2.v0; v165 = tmp2.v1;
        int v166;
        v166 = 0l;
        while (while_method_0(v166)){
            int v168;
            v168 = 0l;
            while (while_method_1(v168)){
                assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                assert("Tensor range check" && 0 <= v168 && v168 < 4l);
                int v170;
                v170 = 4l * v166;
                int v171;
                v171 = v170 + v168;
                float v172;
                v172 = v153[v171];
                bool v173;
                v173 = v154[v171];
                float v180; bool v181;
                if (v165){
                    if (v173){
                        bool v174;
                        v174 = v164 >= v172;
                        float v175;
                        if (v174){
                            v175 = v164;
                        } else {
                            v175 = v172;
                        }
                        v180 = v175; v181 = true;
                    } else {
                        v180 = v164; v181 = v165;
                    }
                } else {
                    if (v173){
                        v180 = v172; v181 = v173;
                    } else {
                        v180 = v164; v181 = v165;
                    }
                }
                v164 = v180;
                v165 = v181;
                v168 += 1l ;
            }
            v166 += 1l ;
        }
        auto v182 = cooperative_groups::coalesced_threads();
        int v183;
        v183 = threadIdx.x;
        int v184;
        v184 = v183 / 4l;
        auto v185 = cooperative_groups::labeled_partition(v182,v184);
        Closure2 v186{};
        float v187; bool v188;
        Tuple1 tmp3 = cooperative_groups::reduce(v185, Tuple1{v164, v165}, v186);
        v187 = tmp3.v0; v188 = tmp3.v1;
        bool v189;
        v189 = v188 == false;
        if (v189){
            assert("The local reduce must be true." && v188);
        } else {
        }
        float v191[4l];
        int v192[4l];
        int v193;
        v193 = 0l;
        while (while_method_0(v193)){
            int v195;
            v195 = 0l;
            while (while_method_1(v195)){
                assert("Tensor range check" && 0 <= v193 && v193 < 1l);
                assert("Tensor range check" && 0 <= v195 && v195 < 4l);
                int v197;
                v197 = 4l * v193;
                int v198;
                v198 = v197 + v195;
                int v199;
                v199 = v31[v198];
                float v200;
                v200 = curand_uniform(&v3);
                assert("Tensor range check" && 0 <= v193 && v193 < 1l);
                assert("Tensor range check" && 0 <= v195 && v195 < 4l);
                v191[v198] = v200;
                v192[v198] = v199;
                v195 += 1l ;
            }
            v193 += 1l ;
        }
        float v201; int v202;
        Tuple2 tmp4 = Tuple2{0.0f, 2147483647l};
        v201 = tmp4.v0; v202 = tmp4.v1;
        int v203;
        v203 = 0l;
        while (while_method_0(v203)){
            int v205;
            v205 = 0l;
            while (while_method_1(v205)){
                assert("Tensor range check" && 0 <= v203 && v203 < 1l);
                assert("Tensor range check" && 0 <= v205 && v205 < 4l);
                int v207;
                v207 = 4l * v203;
                int v208;
                v208 = v207 + v205;
                float v209;
                v209 = v191[v208];
                int v210;
                v210 = v192[v208];
                bool v211;
                v211 = v202 < v210;
                float v212; int v213;
                if (v211){
                    v212 = v201; v213 = v202;
                } else {
                    v212 = v209; v213 = v210;
                }
                v201 = v212;
                v202 = v213;
                v205 += 1l ;
            }
            v203 += 1l ;
        }
        auto v214 = cooperative_groups::coalesced_threads();
        int v215;
        v215 = threadIdx.x;
        int v216;
        v216 = v215 / 4l;
        auto v217 = cooperative_groups::labeled_partition(v214,v216);
        Closure3 v218{};
        float v219; int v220;
        Tuple2 tmp5 = cooperative_groups::reduce(v217, Tuple2{v201, v202}, v218);
        v219 = tmp5.v0; v220 = tmp5.v1;
        float v221;
        v221 = v187 * v219;
        int v222[4l];
        bool v223[4l];
        int v224;
        v224 = 0l;
        while (while_method_0(v224)){
            int v226;
            v226 = 0l;
            while (while_method_1(v226)){
                assert("Tensor range check" && 0 <= v224 && v224 < 1l);
                assert("Tensor range check" && 0 <= v226 && v226 < 4l);
                int v228;
                v228 = 4l * v224;
                int v229;
                v229 = v228 + v226;
                float v230;
                v230 = v153[v229];
                bool v231;
                v231 = v154[v229];
                int v232;
                v232 = v31[v229];
                int v235; bool v236;
                if (v231){
                    float v233;
                    v233 = v230 - v221;
                    bool v234;
                    v234 = v233 >= 0.0f;
                    v235 = v232; v236 = v234;
                } else {
                    v235 = 2147483647l; v236 = false;
                }
                assert("Tensor range check" && 0 <= v224 && v224 < 1l);
                assert("Tensor range check" && 0 <= v226 && v226 < 4l);
                v222[v229] = v235;
                v223[v229] = v236;
                v226 += 1l ;
            }
            v224 += 1l ;
        }
        int v237; bool v238;
        Tuple3 tmp6 = Tuple3{2147483647l, false};
        v237 = tmp6.v0; v238 = tmp6.v1;
        int v239;
        v239 = 0l;
        while (while_method_0(v239)){
            int v241;
            v241 = 0l;
            while (while_method_1(v241)){
                assert("Tensor range check" && 0 <= v239 && v239 < 1l);
                assert("Tensor range check" && 0 <= v241 && v241 < 4l);
                int v243;
                v243 = 4l * v239;
                int v244;
                v244 = v243 + v241;
                int v245;
                v245 = v222[v244];
                bool v246;
                v246 = v223[v244];
                int v253; bool v254;
                if (v238){
                    if (v246){
                        bool v247;
                        v247 = v237 < v245;
                        int v248;
                        if (v247){
                            v248 = v237;
                        } else {
                            v248 = v245;
                        }
                        v253 = v248; v254 = true;
                    } else {
                        v253 = v237; v254 = v238;
                    }
                } else {
                    if (v246){
                        v253 = v245; v254 = v246;
                    } else {
                        v253 = v237; v254 = v238;
                    }
                }
                v237 = v253;
                v238 = v254;
                v241 += 1l ;
            }
            v239 += 1l ;
        }
        auto v255 = cooperative_groups::coalesced_threads();
        int v256;
        v256 = threadIdx.x;
        int v257;
        v257 = v256 / 4l;
        auto v258 = cooperative_groups::labeled_partition(v255,v257);
        Closure4 v259{};
        int v260; bool v261;
        Tuple3 tmp7 = cooperative_groups::reduce(v258, Tuple3{v237, v238}, v259);
        v260 = tmp7.v0; v261 = tmp7.v1;
        bool v262;
        v262 = v261 == false;
        if (v262){
            assert("The local reduce must be true." && v261);
        } else {
        }
        assert("Tensor range check" && 0 <= v26 && v26 < 2l);
        int v264;
        v264 = v28 + v24;
        int v265;
        v265 = 0l;
        while (while_method_0(v265)){
            assert("Tensor range check" && 0 <= v265 && v265 < 1l);
            int v267;
            v267 = 16l * v265;
            int v268;
            v268 = v267 + v264;
            assert("Tensor range check" && 0 <= v265 && v265 < 1l);
            int v269;
            v269 = 4l * v265;
            int4* v270;
            v270 = reinterpret_cast<int4*>(v115 + v269);
            int4* v271;
            v271 = reinterpret_cast<int4*>(v1 + v268);
            assert("Pointer alignment check" && (unsigned long long)(v270) % 4l == 0 && (unsigned long long)(v271) % 4l == 0);
            *v271 = *v270;
            v265 += 1l ;
        }
        assert("Tensor range check" && 0 <= v26 && v26 < 2l);
        int v272;
        v272 = 8l * v26;
        int v273;
        v273 = v272 + v25;
        v0[v273] = v260;
        v26 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return ;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 16l;
    return v1;
}
__device__ void method_5(float * v0, int v1, float * v2, int v3, float * v4, int v5){
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
    while (while_method_0(v65)){
        int v67;
        v67 = 0l;
        while (while_method_0(v67)){
            assert("Tensor range check" && 0 <= v65 && v65 < 1l);
            assert("Tensor range check" && 0 <= v67 && v67 < 1l);
            int v69;
            v69 = 16l * v67;
            int v70;
            v70 = v69 + v3;
            int v71;
            v71 = 256l * v65;
            int v72;
            v72 = v71 + v70;
            float * v73;
            v73 = v2+v72;
            // Pushing the loop unrolling to: 0
            int v75;
            v75 = 0l;
            #pragma unroll
            while (while_method_0(v75)){
                int v77;
                v77 = 0l;
                #pragma unroll
                while (while_method_0(v77)){
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
            while (while_method_0(v81)){
                assert("Tensor range check" && 0 <= v65 && v65 < 1l);
                int v83;
                v83 = 128l * v65;
                int v84;
                v84 = v83 + v5;
                assert("Tensor range check" && 0 <= v81 && v81 < 1l);
                int v85;
                v85 = 8l * v81;
                int v86;
                v86 = v85 + v84;
                float * v87;
                v87 = v4+v86;
                assert("Tensor range check" && 0 <= v67 && v67 < 1l);
                int v89;
                v89 = 128l * v67;
                int v90;
                v90 = v89 + v1;
                assert("Tensor range check" && 0 <= v81 && v81 < 1l);
                int v91;
                v91 = v85 + v90;
                float * v92;
                v92 = v0+v91;
                int v94;
                v94 = threadIdx.x;
                bool v95;
                v95 = 0l <= v94;
                bool v96;
                v96 = v95 == false;
                if (v96){
                    assert("The index needs to be zero or positive." && v95);
                } else {
                }
                int v98;
                v98 = v94 % 2l;
                int v99;
                v99 = v94 / 2l;
                bool v100;
                v100 = v99 < 16l;
                bool v101;
                v101 = v100 == false;
                if (v101){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v100);
                } else {
                }
                assert("Tensor range check" && 0 <= v99 && v99 < 16l);
                assert("Tensor range check" && 0 <= v98 && v98 < 2l);
                int v103;
                v103 = 4l * v98;
                int v104;
                v104 = 12l * v99;
                int v105;
                v105 = v104 + v103;
                int v106;
                v106 = 8l * v99;
                int v107;
                v107 = v106 + v103;
                float * v108;
                v108 = v14+v105;
                float * v110;
                v110 = v92+v107;
                int v112;
                v112 = 0l;
                #pragma unroll
                while (while_method_0(v112)){
                    int v114;
                    v114 = 0l;
                    #pragma unroll
                    while (while_method_0(v114)){
                        assert("Tensor range check" && 0 <= v112 && v112 < 1l);
                        assert("Tensor range check" && 0 <= v114 && v114 < 1l);
                        int v116;
                        v116 = 8l * v114;
                        int v117;
                        v117 = 192l * v112;
                        int v118;
                        v118 = v117 + v116;
                        int v119;
                        v119 = 128l * v112;
                        int v120;
                        v120 = v119 + v116;
                        float v121[4l];
                        int v122;
                        v122 = 0l;
                        #pragma unroll
                        while (while_method_1(v122)){
                            assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                            int v124;
                            v124 = v122 + v120;
                            float v125;
                            v125 = v110[v124];
                            float v126;
                            v126 = wmma::__float_to_tf32(v125);
                            assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                            v121[v122] = v126;
                            v122 += 1l ;
                        }
                        int4* v127;
                        v127 = reinterpret_cast<int4*>(v121 + 0l);
                        int4* v128;
                        v128 = reinterpret_cast<int4*>(v108 + v118);
                        assert("Pointer alignment check" && (unsigned long long)(v127) % 4l == 0 && (unsigned long long)(v128) % 4l == 0);
                        *v128 = *v127;
                        v114 += 1l ;
                    }
                    v112 += 1l ;
                }
                int v129;
                v129 = threadIdx.x;
                bool v130;
                v130 = 0l <= v129;
                bool v131;
                v131 = v130 == false;
                if (v131){
                    assert("The index needs to be zero or positive." && v130);
                } else {
                }
                int v133;
                v133 = v129 % 2l;
                int v134;
                v134 = v129 / 2l;
                bool v135;
                v135 = v134 < 16l;
                bool v136;
                v136 = v135 == false;
                if (v136){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v135);
                } else {
                }
                assert("Tensor range check" && 0 <= v134 && v134 < 16l);
                assert("Tensor range check" && 0 <= v133 && v133 < 2l);
                int v138;
                v138 = 4l * v133;
                int v139;
                v139 = 12l * v134;
                int v140;
                v140 = v139 + v138;
                int v141;
                v141 = 8l * v134;
                int v142;
                v142 = v141 + v138;
                float * v143;
                v143 = v12+v140;
                float * v145;
                v145 = v87+v142;
                int v147;
                v147 = 0l;
                #pragma unroll
                while (while_method_0(v147)){
                    int v149;
                    v149 = 0l;
                    #pragma unroll
                    while (while_method_0(v149)){
                        assert("Tensor range check" && 0 <= v147 && v147 < 1l);
                        assert("Tensor range check" && 0 <= v149 && v149 < 1l);
                        int v151;
                        v151 = 8l * v149;
                        int v152;
                        v152 = 192l * v147;
                        int v153;
                        v153 = v152 + v151;
                        int v154;
                        v154 = 128l * v147;
                        int v155;
                        v155 = v154 + v151;
                        float v156[4l];
                        int v157;
                        v157 = 0l;
                        #pragma unroll
                        while (while_method_1(v157)){
                            assert("Tensor range check" && 0 <= v157 && v157 < 4l);
                            int v159;
                            v159 = v157 + v155;
                            float v160;
                            v160 = v145[v159];
                            float v161;
                            v161 = wmma::__float_to_tf32(v160);
                            assert("Tensor range check" && 0 <= v157 && v157 < 4l);
                            v156[v157] = v161;
                            v157 += 1l ;
                        }
                        int4* v162;
                        v162 = reinterpret_cast<int4*>(v156 + 0l);
                        int4* v163;
                        v163 = reinterpret_cast<int4*>(v143 + v153);
                        assert("Pointer alignment check" && (unsigned long long)(v162) % 4l == 0 && (unsigned long long)(v163) % 4l == 0);
                        *v163 = *v162;
                        v149 += 1l ;
                    }
                    v147 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v164[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v165[1l];
                int v166;
                v166 = 0l;
                #pragma unroll
                while (while_method_0(v166)){
                    int v168;
                    v168 = 0l;
                    #pragma unroll
                    while (while_method_0(v168)){
                        assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                        assert("Tensor range check" && 0 <= v168 && v168 < 1l);
                        int v170;
                        v170 = v166 + v168;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v171 = v164[v170];
                        assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                        int v172;
                        v172 = 192l * v166;
                        assert("Tensor range check" && 0 <= v168 && v168 < 1l);
                        int v173;
                        v173 = 8l * v168;
                        int v174;
                        v174 = v173 + v172;
                        int v175;
                        v175 = 0l;
                        #pragma unroll
                        while (while_method_2(v175)){
                            int v177;
                            v177 = 0l;
                            #pragma unroll
                            while (while_method_2(v177)){
                                assert("Tensor range check" && 0 <= v175 && v175 < 2l);
                                assert("Tensor range check" && 0 <= v177 && v177 < 2l);
                                int v179;
                                v179 = 96l * v177;
                                int v180;
                                v180 = v179 + v174;
                                int v181;
                                v181 = 4l * v175;
                                int v182;
                                v182 = v181 + v180;
                                float v183;
                                v183 = v46[v182];
                                bool v184;
                                v184 = 0l <= v177;
                                bool v186;
                                if (v184){
                                    bool v185;
                                    v185 = v177 < 2l;
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
                                bool v189;
                                v189 = 0l <= v175;
                                bool v191;
                                if (v189){
                                    bool v190;
                                    v190 = v175 < 2l;
                                    v191 = v190;
                                } else {
                                    v191 = false;
                                }
                                bool v192;
                                v192 = v191 == false;
                                if (v192){
                                    assert("The indices should be inside the range of the dimension." && v191);
                                } else {
                                }
                                int v194;
                                v194 = v175 * 2l;
                                int v195;
                                v195 = v177 + v194;
                                v171.x[v195] = v183;
                                v177 += 1l ;
                            }
                            v175 += 1l ;
                        }
                        v168 += 1l ;
                    }
                    v166 += 1l ;
                }
                int v196;
                v196 = 0l;
                #pragma unroll
                while (while_method_0(v196)){
                    int v198;
                    v198 = 0l;
                    #pragma unroll
                    while (while_method_0(v198)){
                        assert("Tensor range check" && 0 <= v196 && v196 < 1l);
                        assert("Tensor range check" && 0 <= v198 && v198 < 1l);
                        int v200;
                        v200 = v196 + v198;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v201 = v165[v200];
                        assert("Tensor range check" && 0 <= v196 && v196 < 1l);
                        int v202;
                        v202 = 192l * v196;
                        assert("Tensor range check" && 0 <= v198 && v198 < 1l);
                        int v203;
                        v203 = 8l * v198;
                        int v204;
                        v204 = v203 + v202;
                        int v205;
                        v205 = 0l;
                        #pragma unroll
                        while (while_method_2(v205)){
                            int v207;
                            v207 = 0l;
                            #pragma unroll
                            while (while_method_2(v207)){
                                assert("Tensor range check" && 0 <= v205 && v205 < 2l);
                                assert("Tensor range check" && 0 <= v207 && v207 < 2l);
                                int v209;
                                v209 = 4l * v207;
                                int v210;
                                v210 = v209 + v204;
                                int v211;
                                v211 = 96l * v205;
                                int v212;
                                v212 = v211 + v210;
                                float v213;
                                v213 = v62[v212];
                                bool v214;
                                v214 = 0l <= v207;
                                bool v216;
                                if (v214){
                                    bool v215;
                                    v215 = v207 < 2l;
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
                                bool v219;
                                v219 = 0l <= v205;
                                bool v221;
                                if (v219){
                                    bool v220;
                                    v220 = v205 < 2l;
                                    v221 = v220;
                                } else {
                                    v221 = false;
                                }
                                bool v222;
                                v222 = v221 == false;
                                if (v222){
                                    assert("The indices should be inside the range of the dimension." && v221);
                                } else {
                                }
                                int v224;
                                v224 = v205 * 2l;
                                int v225;
                                v225 = v207 + v224;
                                v201.x[v225] = v213;
                                v207 += 1l ;
                            }
                            v205 += 1l ;
                        }
                        v198 += 1l ;
                    }
                    v196 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                int v226;
                v226 = 0l;
                #pragma unroll
                while (while_method_0(v226)){
                    int v228;
                    v228 = 0l;
                    #pragma unroll
                    while (while_method_0(v228)){
                        int v230;
                        v230 = 0l;
                        #pragma unroll
                        while (while_method_0(v230)){
                            assert("Tensor range check" && 0 <= v226 && v226 < 1l);
                            assert("Tensor range check" && 0 <= v228 && v228 < 1l);
                            int v232;
                            v232 = v226 + v228;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v233 = v64[v232];
                            assert("Tensor range check" && 0 <= v226 && v226 < 1l);
                            assert("Tensor range check" && 0 <= v230 && v230 < 1l);
                            int v234;
                            v234 = v226 + v230;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v235 = v164[v234];
                            assert("Tensor range check" && 0 <= v228 && v228 < 1l);
                            assert("Tensor range check" && 0 <= v230 && v230 < 1l);
                            int v236;
                            v236 = v228 + v230;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v237 = v165[v236];
                            wmma::mma_sync(v233, v235, v237, v233);
                            v230 += 1l ;
                        }
                        v228 += 1l ;
                    }
                    v226 += 1l ;
                }
                v81 += 1l ;
            }
            int v238;
            v238 = 0l;
            #pragma unroll
            while (while_method_0(v238)){
                int v240;
                v240 = 0l;
                #pragma unroll
                while (while_method_0(v240)){
                    assert("Tensor range check" && 0 <= v238 && v238 < 1l);
                    assert("Tensor range check" && 0 <= v240 && v240 < 1l);
                    int v242;
                    v242 = v238 + v240;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v243 = v64[v242];
                    assert("Tensor range check" && 0 <= v238 && v238 < 1l);
                    assert("Tensor range check" && 0 <= v240 && v240 < 1l);
                    int v244;
                    v244 = 16l * v240;
                    int v245;
                    v245 = 384l * v238;
                    int v246;
                    v246 = v245 + v244;
                    float * v247;
                    v247 = v30+v246;
                    wmma::store_matrix_sync(v247, v243, 24l, wmma::mem_row_major);
                    v240 += 1l ;
                }
                v238 += 1l ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0l));
            int v249;
            v249 = threadIdx.x;
            bool v250;
            v250 = 0l <= v249;
            bool v251;
            v251 = v250 == false;
            if (v251){
                assert("The index needs to be zero or positive." && v250);
            } else {
            }
            int v253;
            v253 = v249 % 4l;
            int v254;
            v254 = v249 / 4l;
            bool v255;
            v255 = v254 < 8l;
            bool v256;
            v256 = v255 == false;
            if (v256){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v255);
            } else {
            }
            assert("Tensor range check" && 0 <= v254 && v254 < 8l);
            assert("Tensor range check" && 0 <= v253 && v253 < 4l);
            int v258;
            v258 = 4l * v253;
            int v259;
            v259 = 16l * v254;
            int v260;
            v260 = v259 + v258;
            int v261;
            v261 = 24l * v254;
            int v262;
            v262 = v261 + v258;
            float * v263;
            v263 = v73+v260;
            float * v265;
            v265 = v16+v262;
            int v267;
            v267 = 0l;
            #pragma unroll
            while (while_method_2(v267)){
                int v269;
                v269 = 0l;
                #pragma unroll
                while (while_method_0(v269)){
                    assert("Tensor range check" && 0 <= v267 && v267 < 2l);
                    assert("Tensor range check" && 0 <= v269 && v269 < 1l);
                    int v271;
                    v271 = 16l * v269;
                    int v272;
                    v272 = 128l * v267;
                    int v273;
                    v273 = v272 + v271;
                    int v274;
                    v274 = 192l * v267;
                    int v275;
                    v275 = v274 + v271;
                    int4* v276;
                    v276 = reinterpret_cast<int4*>(v265 + v275);
                    int4* v277;
                    v277 = reinterpret_cast<int4*>(v263 + v273);
                    assert("Pointer alignment check" && (unsigned long long)(v276) % 4l == 0 && (unsigned long long)(v277) % 4l == 0);
                    *v277 = *v276;
                    v269 += 1l ;
                }
                v267 += 1l ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0l));
            // Poping the loop unrolling to: 0
            v67 += 1l ;
        }
        v65 += 1l ;
    }
    return ;
}
__device__ void method_6(float * v0, int v1, float * v2, int v3, float * v4, int v5){
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
    while (while_method_0(v65)){
        int v67;
        v67 = 0l;
        while (while_method_0(v67)){
            assert("Tensor range check" && 0 <= v65 && v65 < 1l);
            assert("Tensor range check" && 0 <= v67 && v67 < 1l);
            int v69;
            v69 = 16l * v67;
            int v70;
            v70 = v69 + v3;
            int v71;
            v71 = 256l * v65;
            int v72;
            v72 = v71 + v70;
            float * v73;
            v73 = v2+v72;
            // Pushing the loop unrolling to: 0
            int v75;
            v75 = 0l;
            #pragma unroll
            while (while_method_0(v75)){
                int v77;
                v77 = 0l;
                #pragma unroll
                while (while_method_0(v77)){
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
            while (while_method_2(v81)){
                assert("Tensor range check" && 0 <= v65 && v65 < 1l);
                int v83;
                v83 = v71 + v5;
                assert("Tensor range check" && 0 <= v81 && v81 < 2l);
                int v84;
                v84 = 8l * v81;
                int v85;
                v85 = v84 + v83;
                float * v86;
                v86 = v4+v85;
                assert("Tensor range check" && 0 <= v67 && v67 < 1l);
                int v88;
                v88 = 256l * v67;
                int v89;
                v89 = v88 + v1;
                assert("Tensor range check" && 0 <= v81 && v81 < 2l);
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
                v105 = 16l * v98;
                int v106;
                v106 = v105 + v102;
                float * v107;
                v107 = v14+v104;
                float * v109;
                v109 = v91+v106;
                int v111;
                v111 = 0l;
                #pragma unroll
                while (while_method_0(v111)){
                    int v113;
                    v113 = 0l;
                    #pragma unroll
                    while (while_method_0(v113)){
                        assert("Tensor range check" && 0 <= v111 && v111 < 1l);
                        assert("Tensor range check" && 0 <= v113 && v113 < 1l);
                        int v115;
                        v115 = 8l * v113;
                        int v116;
                        v116 = 192l * v111;
                        int v117;
                        v117 = v116 + v115;
                        int v118;
                        v118 = 256l * v111;
                        int v119;
                        v119 = v118 + v115;
                        float v120[4l];
                        int v121;
                        v121 = 0l;
                        #pragma unroll
                        while (while_method_1(v121)){
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
                v140 = 16l * v133;
                int v141;
                v141 = v140 + v137;
                float * v142;
                v142 = v12+v139;
                float * v144;
                v144 = v86+v141;
                int v146;
                v146 = 0l;
                #pragma unroll
                while (while_method_0(v146)){
                    int v148;
                    v148 = 0l;
                    #pragma unroll
                    while (while_method_0(v148)){
                        assert("Tensor range check" && 0 <= v146 && v146 < 1l);
                        assert("Tensor range check" && 0 <= v148 && v148 < 1l);
                        int v150;
                        v150 = 8l * v148;
                        int v151;
                        v151 = 192l * v146;
                        int v152;
                        v152 = v151 + v150;
                        int v153;
                        v153 = 256l * v146;
                        int v154;
                        v154 = v153 + v150;
                        float v155[4l];
                        int v156;
                        v156 = 0l;
                        #pragma unroll
                        while (while_method_1(v156)){
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
                while (while_method_0(v165)){
                    int v167;
                    v167 = 0l;
                    #pragma unroll
                    while (while_method_0(v167)){
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
                        while (while_method_2(v174)){
                            int v176;
                            v176 = 0l;
                            #pragma unroll
                            while (while_method_2(v176)){
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
                while (while_method_0(v195)){
                    int v197;
                    v197 = 0l;
                    #pragma unroll
                    while (while_method_0(v197)){
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
                        while (while_method_2(v204)){
                            int v206;
                            v206 = 0l;
                            #pragma unroll
                            while (while_method_2(v206)){
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
                while (while_method_0(v225)){
                    int v227;
                    v227 = 0l;
                    #pragma unroll
                    while (while_method_0(v227)){
                        int v229;
                        v229 = 0l;
                        #pragma unroll
                        while (while_method_0(v229)){
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
            while (while_method_0(v237)){
                int v239;
                v239 = 0l;
                #pragma unroll
                while (while_method_0(v239)){
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
            v258 = 16l * v253;
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
            while (while_method_2(v266)){
                int v268;
                v268 = 0l;
                #pragma unroll
                while (while_method_0(v268)){
                    assert("Tensor range check" && 0 <= v266 && v266 < 2l);
                    assert("Tensor range check" && 0 <= v268 && v268 < 1l);
                    int v270;
                    v270 = 16l * v268;
                    int v271;
                    v271 = 128l * v266;
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
__device__ void method_7(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5){
    int v6;
    v6 = blockIdx.x;
    assert("Tensor range check" && 0 <= v6 && v6 < 24l);
    int v7;
    v7 = 256l * v6;
    int v8;
    v8 = blockIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 24l);
    int v9;
    v9 = 256l * v8;
    int v10;
    v10 = v9 + v3;
    int v11;
    v11 = blockIdx.x;
    assert("Tensor range check" && 0 <= v11 && v11 < 24l);
    int v12;
    v12 = 16l * v11;
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
    v18 = v14 % 4l;
    int v19;
    v19 = v14 / 4l;
    bool v20;
    v20 = v19 < 8l;
    bool v21;
    v21 = v20 == false;
    if (v21){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v20);
    } else {
    }
    assert("Tensor range check" && 0 <= v19 && v19 < 8l);
    assert("Tensor range check" && 0 <= v18 && v18 < 4l);
    int v23;
    v23 = 4l * v18;
    int v24;
    v24 = v23 + v7;
    int v25;
    v25 = 16l * v19;
    int v26;
    v26 = v25 + v24;
    assert("Tensor range check" && 0 <= v19 && v19 < 8l);
    assert("Tensor range check" && 0 <= v18 && v18 < 4l);
    int v27;
    v27 = v23 + v10;
    int v28;
    v28 = v25 + v27;
    assert("Tensor range check" && 0 <= v19 && v19 < 8l);
    int v29;
    v29 = v19 + v13;
    int v30;
    v30 = 0l;
    while (while_method_2(v30)){
        assert("Tensor range check" && 0 <= v30 && v30 < 2l);
        int v32;
        v32 = 128l * v30;
        int v33;
        v33 = v32 + v26;
        float v34[4l];
        int v35[4l];
        int v36;
        v36 = 0l;
        while (while_method_0(v36)){
            assert("Tensor range check" && 0 <= v36 && v36 < 1l);
            int v38;
            v38 = 4l * v36;
            assert("Tensor range check" && 0 <= v36 && v36 < 1l);
            int v39;
            v39 = 16l * v36;
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
        while (while_method_0(v43)){
            int v45;
            v45 = 0l;
            while (while_method_1(v45)){
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
                    v53 = v18 < 4l;
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
                v64 = v43 * 16l;
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
            v73 = v30 < 2l;
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
        v77 = v30 * 8l;
        int v78;
        v78 = v77 + v19;
        float v79;
        v79 = 0.0f;
        int v80;
        v80 = 0l;
        while (while_method_0(v80)){
            int v82;
            v82 = 0l;
            while (while_method_1(v82)){
                assert("Tensor range check" && 0 <= v80 && v80 < 1l);
                assert("Tensor range check" && 0 <= v82 && v82 < 4l);
                int v84;
                v84 = 4l * v80;
                int v85;
                v85 = v84 + v82;
                float v86;
                v86 = v34[v85];
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
        v90 = v89 / 4l;
        auto v91 = cooperative_groups::labeled_partition(v88,v90);
        Closure0 v92{};
        float v93;
        v93 = cooperative_groups::reduce(v91, v79, v92);
        float v94;
        v94 = v93 / 16.0f;
        float v95[4l];
        int v96;
        v96 = 0l;
        while (while_method_0(v96)){
            int v98;
            v98 = 0l;
            while (while_method_1(v98)){
                assert("Tensor range check" && 0 <= v96 && v96 < 1l);
                assert("Tensor range check" && 0 <= v98 && v98 < 4l);
                int v100;
                v100 = 4l * v96;
                int v101;
                v101 = v100 + v98;
                float v102;
                v102 = v34[v101];
                float v103;
                v103 = v102 - v94;
                float v104;
                v104 = exp(v103);
                assert("Tensor range check" && 0 <= v96 && v96 < 1l);
                assert("Tensor range check" && 0 <= v98 && v98 < 4l);
                v95[v101] = v104;
                v98 += 1l ;
            }
            v96 += 1l ;
        }
        float v105;
        v105 = 0.0f;
        int v106;
        v106 = 0l;
        while (while_method_0(v106)){
            int v108;
            v108 = 0l;
            while (while_method_1(v108)){
                assert("Tensor range check" && 0 <= v106 && v106 < 1l);
                assert("Tensor range check" && 0 <= v108 && v108 < 4l);
                int v110;
                v110 = 4l * v106;
                int v111;
                v111 = v110 + v108;
                float v112;
                v112 = v95[v111];
                float v113;
                v113 = v105 + v112;
                v105 = v113;
                v108 += 1l ;
            }
            v106 += 1l ;
        }
        auto v114 = cooperative_groups::coalesced_threads();
        int v115;
        v115 = threadIdx.x;
        int v116;
        v116 = v115 / 4l;
        auto v117 = cooperative_groups::labeled_partition(v114,v116);
        float v118;
        v118 = cooperative_groups::reduce(v117, v105, v92);
        float v119[4l];
        int v120;
        v120 = 0l;
        while (while_method_0(v120)){
            int v122;
            v122 = 0l;
            while (while_method_1(v122)){
                assert("Tensor range check" && 0 <= v120 && v120 < 1l);
                assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                int v124;
                v124 = 4l * v120;
                int v125;
                v125 = v124 + v122;
                float v126;
                v126 = v95[v125];
                float v127;
                v127 = v126 / v118;
                assert("Tensor range check" && 0 <= v120 && v120 < 1l);
                assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                v119[v125] = v127;
                v122 += 1l ;
            }
            v120 += 1l ;
        }
        float v128[4l];
        float v129;
        v129 = 0.0f;
        int v130;
        v130 = 0l;
        while (while_method_0(v130)){
            assert("Tensor range check" && 0 <= v130 && v130 < 1l);
            int v132;
            v132 = 4l * v130;
            assert("Tensor range check" && 0 <= v130 && v130 < 1l);
            int v133; float v134;
            Tuple0 tmp8 = Tuple0{0l, 0.0f};
            v133 = tmp8.v0; v134 = tmp8.v1;
            while (while_method_1(v133)){
                assert("Tensor range check" && 0 <= v133 && v133 < 4l);
                int v136;
                v136 = v133 + v132;
                float v137;
                v137 = v119[v136];
                float v138;
                v138 = v134 + v137;
                v134 = v138;
                v133 += 1l ;
            }
            auto v139 = cooperative_groups::coalesced_threads();
            int v140;
            v140 = threadIdx.x;
            int v141;
            v141 = v140 / 4l;
            auto v142 = cooperative_groups::labeled_partition(v139,v141);
            Closure1 v143{};
            float v144;
            v144 = cooperative_groups::inclusive_scan(v142, v134, v143);
            float v145;
            v145 = v142.shfl_up(v144,1);
            bool v146;
            v146 = v142.thread_rank() == 0;
            float v147;
            if (v146){
                v147 = 0.0f;
            } else {
                v147 = v145;
            }
            float v148;
            v148 = v142.shfl(v144,v142.num_threads()-1);
            float v149;
            v149 = v129 + v147;
            int v150; float v151;
            Tuple0 tmp9 = Tuple0{0l, v149};
            v150 = tmp9.v0; v151 = tmp9.v1;
            while (while_method_1(v150)){
                assert("Tensor range check" && 0 <= v150 && v150 < 4l);
                int v153;
                v153 = v150 + v132;
                float v154;
                v154 = v119[v153];
                float v155;
                v155 = v151 + v154;
                assert("Tensor range check" && 0 <= v150 && v150 < 4l);
                v128[v153] = v155;
                v151 = v155;
                v150 += 1l ;
            }
            float v156;
            v156 = v129 + v148;
            v129 = v156;
            v130 += 1l ;
        }
        float v157[4l];
        bool v158[4l];
        int v159;
        v159 = 0l;
        while (while_method_0(v159)){
            int v161;
            v161 = 0l;
            while (while_method_1(v161)){
                assert("Tensor range check" && 0 <= v159 && v159 < 1l);
                assert("Tensor range check" && 0 <= v161 && v161 < 4l);
                int v163;
                v163 = 4l * v159;
                int v164;
                v164 = v163 + v161;
                float v165;
                v165 = v128[v164];
                float v166;
                v166 = v119[v164];
                bool v167;
                v167 = v166 > 0.0f;
                assert("Tensor range check" && 0 <= v159 && v159 < 1l);
                assert("Tensor range check" && 0 <= v161 && v161 < 4l);
                v157[v164] = v165;
                v158[v164] = v167;
                v161 += 1l ;
            }
            v159 += 1l ;
        }
        float v168; bool v169;
        Tuple1 tmp10 = Tuple1{-1.0f / 0.0f, false};
        v168 = tmp10.v0; v169 = tmp10.v1;
        int v170;
        v170 = 0l;
        while (while_method_0(v170)){
            int v172;
            v172 = 0l;
            while (while_method_1(v172)){
                assert("Tensor range check" && 0 <= v170 && v170 < 1l);
                assert("Tensor range check" && 0 <= v172 && v172 < 4l);
                int v174;
                v174 = 4l * v170;
                int v175;
                v175 = v174 + v172;
                float v176;
                v176 = v157[v175];
                bool v177;
                v177 = v158[v175];
                float v184; bool v185;
                if (v169){
                    if (v177){
                        bool v178;
                        v178 = v168 >= v176;
                        float v179;
                        if (v178){
                            v179 = v168;
                        } else {
                            v179 = v176;
                        }
                        v184 = v179; v185 = true;
                    } else {
                        v184 = v168; v185 = v169;
                    }
                } else {
                    if (v177){
                        v184 = v176; v185 = v177;
                    } else {
                        v184 = v168; v185 = v169;
                    }
                }
                v168 = v184;
                v169 = v185;
                v172 += 1l ;
            }
            v170 += 1l ;
        }
        auto v186 = cooperative_groups::coalesced_threads();
        int v187;
        v187 = threadIdx.x;
        int v188;
        v188 = v187 / 4l;
        auto v189 = cooperative_groups::labeled_partition(v186,v188);
        Closure2 v190{};
        float v191; bool v192;
        Tuple1 tmp11 = cooperative_groups::reduce(v189, Tuple1{v168, v169}, v190);
        v191 = tmp11.v0; v192 = tmp11.v1;
        bool v193;
        v193 = v192 == false;
        if (v193){
            assert("The local reduce must be true." && v192);
        } else {
        }
        float v195[4l];
        int v196[4l];
        int v197;
        v197 = 0l;
        while (while_method_0(v197)){
            int v199;
            v199 = 0l;
            while (while_method_1(v199)){
                assert("Tensor range check" && 0 <= v197 && v197 < 1l);
                assert("Tensor range check" && 0 <= v199 && v199 < 4l);
                int v201;
                v201 = 4l * v197;
                int v202;
                v202 = v201 + v199;
                int v203;
                v203 = v35[v202];
                float v204;
                v204 = curand_uniform(&v5);
                assert("Tensor range check" && 0 <= v197 && v197 < 1l);
                assert("Tensor range check" && 0 <= v199 && v199 < 4l);
                v195[v202] = v204;
                v196[v202] = v203;
                v199 += 1l ;
            }
            v197 += 1l ;
        }
        float v205; int v206;
        Tuple2 tmp12 = Tuple2{0.0f, 2147483647l};
        v205 = tmp12.v0; v206 = tmp12.v1;
        int v207;
        v207 = 0l;
        while (while_method_0(v207)){
            int v209;
            v209 = 0l;
            while (while_method_1(v209)){
                assert("Tensor range check" && 0 <= v207 && v207 < 1l);
                assert("Tensor range check" && 0 <= v209 && v209 < 4l);
                int v211;
                v211 = 4l * v207;
                int v212;
                v212 = v211 + v209;
                float v213;
                v213 = v195[v212];
                int v214;
                v214 = v196[v212];
                bool v215;
                v215 = v206 < v214;
                float v216; int v217;
                if (v215){
                    v216 = v205; v217 = v206;
                } else {
                    v216 = v213; v217 = v214;
                }
                v205 = v216;
                v206 = v217;
                v209 += 1l ;
            }
            v207 += 1l ;
        }
        auto v218 = cooperative_groups::coalesced_threads();
        int v219;
        v219 = threadIdx.x;
        int v220;
        v220 = v219 / 4l;
        auto v221 = cooperative_groups::labeled_partition(v218,v220);
        Closure3 v222{};
        float v223; int v224;
        Tuple2 tmp13 = cooperative_groups::reduce(v221, Tuple2{v205, v206}, v222);
        v223 = tmp13.v0; v224 = tmp13.v1;
        float v225;
        v225 = v191 * v223;
        int v226[4l];
        bool v227[4l];
        int v228;
        v228 = 0l;
        while (while_method_0(v228)){
            int v230;
            v230 = 0l;
            while (while_method_1(v230)){
                assert("Tensor range check" && 0 <= v228 && v228 < 1l);
                assert("Tensor range check" && 0 <= v230 && v230 < 4l);
                int v232;
                v232 = 4l * v228;
                int v233;
                v233 = v232 + v230;
                float v234;
                v234 = v157[v233];
                bool v235;
                v235 = v158[v233];
                int v236;
                v236 = v35[v233];
                int v239; bool v240;
                if (v235){
                    float v237;
                    v237 = v234 - v225;
                    bool v238;
                    v238 = v237 >= 0.0f;
                    v239 = v236; v240 = v238;
                } else {
                    v239 = 2147483647l; v240 = false;
                }
                assert("Tensor range check" && 0 <= v228 && v228 < 1l);
                assert("Tensor range check" && 0 <= v230 && v230 < 4l);
                v226[v233] = v239;
                v227[v233] = v240;
                v230 += 1l ;
            }
            v228 += 1l ;
        }
        int v241; bool v242;
        Tuple3 tmp14 = Tuple3{2147483647l, false};
        v241 = tmp14.v0; v242 = tmp14.v1;
        int v243;
        v243 = 0l;
        while (while_method_0(v243)){
            int v245;
            v245 = 0l;
            while (while_method_1(v245)){
                assert("Tensor range check" && 0 <= v243 && v243 < 1l);
                assert("Tensor range check" && 0 <= v245 && v245 < 4l);
                int v247;
                v247 = 4l * v243;
                int v248;
                v248 = v247 + v245;
                int v249;
                v249 = v226[v248];
                bool v250;
                v250 = v227[v248];
                int v257; bool v258;
                if (v242){
                    if (v250){
                        bool v251;
                        v251 = v241 < v249;
                        int v252;
                        if (v251){
                            v252 = v241;
                        } else {
                            v252 = v249;
                        }
                        v257 = v252; v258 = true;
                    } else {
                        v257 = v241; v258 = v242;
                    }
                } else {
                    if (v250){
                        v257 = v249; v258 = v250;
                    } else {
                        v257 = v241; v258 = v242;
                    }
                }
                v241 = v257;
                v242 = v258;
                v245 += 1l ;
            }
            v243 += 1l ;
        }
        auto v259 = cooperative_groups::coalesced_threads();
        int v260;
        v260 = threadIdx.x;
        int v261;
        v261 = v260 / 4l;
        auto v262 = cooperative_groups::labeled_partition(v259,v261);
        Closure4 v263{};
        int v264; bool v265;
        Tuple3 tmp15 = cooperative_groups::reduce(v262, Tuple3{v241, v242}, v263);
        v264 = tmp15.v0; v265 = tmp15.v1;
        bool v266;
        v266 = v265 == false;
        if (v266){
            assert("The local reduce must be true." && v265);
        } else {
        }
        assert("Tensor range check" && 0 <= v30 && v30 < 2l);
        int v268;
        v268 = v32 + v28;
        int v269;
        v269 = 0l;
        while (while_method_0(v269)){
            assert("Tensor range check" && 0 <= v269 && v269 < 1l);
            int v271;
            v271 = 16l * v269;
            int v272;
            v272 = v271 + v268;
            assert("Tensor range check" && 0 <= v269 && v269 < 1l);
            int v273;
            v273 = 4l * v269;
            int4* v274;
            v274 = reinterpret_cast<int4*>(v119 + v273);
            int4* v275;
            v275 = reinterpret_cast<int4*>(v2 + v272);
            assert("Pointer alignment check" && (unsigned long long)(v274) % 4l == 0 && (unsigned long long)(v275) % 4l == 0);
            *v275 = *v274;
            v269 += 1l ;
        }
        assert("Tensor range check" && 0 <= v30 && v30 < 2l);
        int v276;
        v276 = 8l * v30;
        int v277;
        v277 = v276 + v29;
        v0[v277] = v264;
        v30 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return ;
}
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 512l;
    return v1;
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 1024l;
    return v1;
}
__device__ void method_8(float * v0, int v1, float * v2, int v3, float * v4, int v5){
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
    while (while_method_2(v65)){
        int v67;
        v67 = 0l;
        while (while_method_5(v67)){
            assert("Tensor range check" && 0 <= v65 && v65 < 2l);
            assert("Tensor range check" && 0 <= v67 && v67 < 512l);
            int v69;
            v69 = 16l * v67;
            int v70;
            v70 = v69 + v3;
            int v71;
            v71 = 131072l * v65;
            int v72;
            v72 = v71 + v70;
            float * v73;
            v73 = v2+v72;
            // Pushing the loop unrolling to: 0
            int v75;
            v75 = 0l;
            #pragma unroll
            while (while_method_0(v75)){
                int v77;
                v77 = 0l;
                #pragma unroll
                while (while_method_0(v77)){
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
                assert("Tensor range check" && 0 <= v81 && v81 < 1024l);
                int v84;
                v84 = 8l * v81;
                int v85;
                v85 = v84 + v83;
                float * v86;
                v86 = v4+v85;
                assert("Tensor range check" && 0 <= v67 && v67 < 512l);
                int v88;
                v88 = 131072l * v67;
                int v89;
                v89 = v88 + v1;
                assert("Tensor range check" && 0 <= v81 && v81 < 1024l);
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
                v105 = 8192l * v98;
                int v106;
                v106 = v105 + v102;
                float * v107;
                v107 = v14+v104;
                float * v109;
                v109 = v91+v106;
                int v111;
                v111 = 0l;
                #pragma unroll
                while (while_method_0(v111)){
                    int v113;
                    v113 = 0l;
                    #pragma unroll
                    while (while_method_0(v113)){
                        assert("Tensor range check" && 0 <= v111 && v111 < 1l);
                        assert("Tensor range check" && 0 <= v113 && v113 < 1l);
                        int v115;
                        v115 = 8l * v113;
                        int v116;
                        v116 = 192l * v111;
                        int v117;
                        v117 = v116 + v115;
                        int v118;
                        v118 = 131072l * v111;
                        int v119;
                        v119 = v118 + v115;
                        float v120[4l];
                        int v121;
                        v121 = 0l;
                        #pragma unroll
                        while (while_method_1(v121)){
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
                v140 = 8192l * v133;
                int v141;
                v141 = v140 + v137;
                float * v142;
                v142 = v12+v139;
                float * v144;
                v144 = v86+v141;
                int v146;
                v146 = 0l;
                #pragma unroll
                while (while_method_0(v146)){
                    int v148;
                    v148 = 0l;
                    #pragma unroll
                    while (while_method_0(v148)){
                        assert("Tensor range check" && 0 <= v146 && v146 < 1l);
                        assert("Tensor range check" && 0 <= v148 && v148 < 1l);
                        int v150;
                        v150 = 8l * v148;
                        int v151;
                        v151 = 192l * v146;
                        int v152;
                        v152 = v151 + v150;
                        int v153;
                        v153 = 131072l * v146;
                        int v154;
                        v154 = v153 + v150;
                        float v155[4l];
                        int v156;
                        v156 = 0l;
                        #pragma unroll
                        while (while_method_1(v156)){
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
                while (while_method_0(v165)){
                    int v167;
                    v167 = 0l;
                    #pragma unroll
                    while (while_method_0(v167)){
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
                        while (while_method_2(v174)){
                            int v176;
                            v176 = 0l;
                            #pragma unroll
                            while (while_method_2(v176)){
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
                while (while_method_0(v195)){
                    int v197;
                    v197 = 0l;
                    #pragma unroll
                    while (while_method_0(v197)){
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
                        while (while_method_2(v204)){
                            int v206;
                            v206 = 0l;
                            #pragma unroll
                            while (while_method_2(v206)){
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
                while (while_method_0(v225)){
                    int v227;
                    v227 = 0l;
                    #pragma unroll
                    while (while_method_0(v227)){
                        int v229;
                        v229 = 0l;
                        #pragma unroll
                        while (while_method_0(v229)){
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
            while (while_method_0(v237)){
                int v239;
                v239 = 0l;
                #pragma unroll
                while (while_method_0(v239)){
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
            v258 = 8192l * v253;
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
            while (while_method_2(v266)){
                int v268;
                v268 = 0l;
                #pragma unroll
                while (while_method_0(v268)){
                    assert("Tensor range check" && 0 <= v266 && v266 < 2l);
                    assert("Tensor range check" && 0 <= v268 && v268 < 1l);
                    int v270;
                    v270 = 16l * v268;
                    int v271;
                    v271 = 65536l * v266;
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
__device__ void method_9(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5){
    int v6;
    v6 = blockIdx.x;
    assert("Tensor range check" && 0 <= v6 && v6 < 24l);
    int v7;
    v7 = 262144l * v6;
    int v8;
    v8 = blockIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 24l);
    int v9;
    v9 = 262144l * v8;
    int v10;
    v10 = v9 + v3;
    int v11;
    v11 = blockIdx.x;
    assert("Tensor range check" && 0 <= v11 && v11 < 24l);
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
    v25 = 8192l * v19;
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
        v32 = 8192l * v30;
        int v33;
        v33 = v32 + v26;
        float v34[256l];
        int v35[256l];
        int v36;
        v36 = 0l;
        while (while_method_3(v36)){
            assert("Tensor range check" && 0 <= v36 && v36 < 64l);
            int v38;
            v38 = 4l * v36;
            assert("Tensor range check" && 0 <= v36 && v36 < 64l);
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
        while (while_method_3(v43)){
            int v45;
            v45 = 0l;
            while (while_method_1(v45)){
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
                    v60 = v43 < 64l;
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
                assert("Tensor range check" && 0 <= v43 && v43 < 64l);
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
        bool v78[256l];
        int v79;
        v79 = 0l;
        while (while_method_3(v79)){
            int v81;
            v81 = 0l;
            while (while_method_1(v81)){
                assert("Tensor range check" && 0 <= v79 && v79 < 64l);
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
                v87 = v86 < 11l;
                assert("Tensor range check" && 0 <= v79 && v79 < 64l);
                assert("Tensor range check" && 0 <= v81 && v81 < 4l);
                v78[v84] = v87;
                v81 += 1l ;
            }
            v79 += 1l ;
        }
        int v88[256l];
        int v89;
        v89 = 0l;
        while (while_method_3(v89)){
            int v91;
            v91 = 0l;
            while (while_method_1(v91)){
                assert("Tensor range check" && 0 <= v89 && v89 < 64l);
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
                assert("Tensor range check" && 0 <= v89 && v89 < 64l);
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
        while (while_method_3(v98)){
            int v100;
            v100 = 0l;
            while (while_method_1(v100)){
                assert("Tensor range check" && 0 <= v98 && v98 < 64l);
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
        Closure5 v110{};
        int v111;
        v111 = cooperative_groups::reduce(v109, v97, v110);
        float v112[256l];
        int v113;
        v113 = 0l;
        while (while_method_3(v113)){
            int v115;
            v115 = 0l;
            while (while_method_1(v115)){
                assert("Tensor range check" && 0 <= v113 && v113 < 64l);
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
                assert("Tensor range check" && 0 <= v113 && v113 < 64l);
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
        while (while_method_3(v123)){
            int v125;
            v125 = 0l;
            while (while_method_1(v125)){
                assert("Tensor range check" && 0 <= v123 && v123 < 64l);
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
        float v139[256l];
        int v140;
        v140 = 0l;
        while (while_method_3(v140)){
            int v142;
            v142 = 0l;
            while (while_method_1(v142)){
                assert("Tensor range check" && 0 <= v140 && v140 < 64l);
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
                assert("Tensor range check" && 0 <= v140 && v140 < 64l);
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
        while (while_method_3(v152)){
            int v154;
            v154 = 0l;
            while (while_method_1(v154)){
                assert("Tensor range check" && 0 <= v152 && v152 < 64l);
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
        float v165[256l];
        int v166;
        v166 = 0l;
        while (while_method_3(v166)){
            int v168;
            v168 = 0l;
            while (while_method_1(v168)){
                assert("Tensor range check" && 0 <= v166 && v166 < 64l);
                assert("Tensor range check" && 0 <= v168 && v168 < 4l);
                int v170;
                v170 = 4l * v166;
                int v171;
                v171 = v170 + v168;
                float v172;
                v172 = v139[v171];
                float v173;
                v173 = v172 / v164;
                assert("Tensor range check" && 0 <= v166 && v166 < 64l);
                assert("Tensor range check" && 0 <= v168 && v168 < 4l);
                v165[v171] = v173;
                v168 += 1l ;
            }
            v166 += 1l ;
        }
        float v174[256l];
        float v175;
        v175 = 0.0f;
        int v176;
        v176 = 0l;
        while (while_method_3(v176)){
            assert("Tensor range check" && 0 <= v176 && v176 < 64l);
            int v178;
            v178 = 4l * v176;
            assert("Tensor range check" && 0 <= v176 && v176 < 64l);
            int v179; float v180;
            Tuple0 tmp16 = Tuple0{0l, 0.0f};
            v179 = tmp16.v0; v180 = tmp16.v1;
            while (while_method_1(v179)){
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
            Closure1 v189{};
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
            Tuple0 tmp17 = Tuple0{0l, v195};
            v196 = tmp17.v0; v197 = tmp17.v1;
            while (while_method_1(v196)){
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
        float v203[256l];
        bool v204[256l];
        int v205;
        v205 = 0l;
        while (while_method_3(v205)){
            int v207;
            v207 = 0l;
            while (while_method_1(v207)){
                assert("Tensor range check" && 0 <= v205 && v205 < 64l);
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
                assert("Tensor range check" && 0 <= v205 && v205 < 64l);
                assert("Tensor range check" && 0 <= v207 && v207 < 4l);
                v203[v210] = v211;
                v204[v210] = v213;
                v207 += 1l ;
            }
            v205 += 1l ;
        }
        float v214; bool v215;
        Tuple1 tmp18 = Tuple1{-1.0f / 0.0f, false};
        v214 = tmp18.v0; v215 = tmp18.v1;
        int v216;
        v216 = 0l;
        while (while_method_3(v216)){
            int v218;
            v218 = 0l;
            while (while_method_1(v218)){
                assert("Tensor range check" && 0 <= v216 && v216 < 64l);
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
        Closure2 v236{};
        float v237; bool v238;
        Tuple1 tmp19 = cooperative_groups::reduce(v235, Tuple1{v214, v215}, v236);
        v237 = tmp19.v0; v238 = tmp19.v1;
        bool v239;
        v239 = v238 == false;
        if (v239){
            assert("The local reduce must be true." && v238);
        } else {
        }
        float v241[256l];
        int v242[256l];
        int v243;
        v243 = 0l;
        while (while_method_3(v243)){
            int v245;
            v245 = 0l;
            while (while_method_1(v245)){
                assert("Tensor range check" && 0 <= v243 && v243 < 64l);
                assert("Tensor range check" && 0 <= v245 && v245 < 4l);
                int v247;
                v247 = 4l * v243;
                int v248;
                v248 = v247 + v245;
                int v249;
                v249 = v35[v248];
                float v250;
                v250 = curand_uniform(&v5);
                assert("Tensor range check" && 0 <= v243 && v243 < 64l);
                assert("Tensor range check" && 0 <= v245 && v245 < 4l);
                v241[v248] = v250;
                v242[v248] = v249;
                v245 += 1l ;
            }
            v243 += 1l ;
        }
        float v251; int v252;
        Tuple2 tmp20 = Tuple2{0.0f, 2147483647l};
        v251 = tmp20.v0; v252 = tmp20.v1;
        int v253;
        v253 = 0l;
        while (while_method_3(v253)){
            int v255;
            v255 = 0l;
            while (while_method_1(v255)){
                assert("Tensor range check" && 0 <= v253 && v253 < 64l);
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
        Closure3 v268{};
        float v269; int v270;
        Tuple2 tmp21 = cooperative_groups::reduce(v267, Tuple2{v251, v252}, v268);
        v269 = tmp21.v0; v270 = tmp21.v1;
        float v271;
        v271 = v237 * v269;
        int v272[256l];
        bool v273[256l];
        int v274;
        v274 = 0l;
        while (while_method_3(v274)){
            int v276;
            v276 = 0l;
            while (while_method_1(v276)){
                assert("Tensor range check" && 0 <= v274 && v274 < 64l);
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
                assert("Tensor range check" && 0 <= v274 && v274 < 64l);
                assert("Tensor range check" && 0 <= v276 && v276 < 4l);
                v272[v279] = v285;
                v273[v279] = v286;
                v276 += 1l ;
            }
            v274 += 1l ;
        }
        int v287; bool v288;
        Tuple3 tmp22 = Tuple3{2147483647l, false};
        v287 = tmp22.v0; v288 = tmp22.v1;
        int v289;
        v289 = 0l;
        while (while_method_3(v289)){
            int v291;
            v291 = 0l;
            while (while_method_1(v291)){
                assert("Tensor range check" && 0 <= v289 && v289 < 64l);
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
        Closure4 v309{};
        int v310; bool v311;
        Tuple3 tmp23 = cooperative_groups::reduce(v308, Tuple3{v287, v288}, v309);
        v310 = tmp23.v0; v311 = tmp23.v1;
        bool v312;
        v312 = v311 == false;
        if (v312){
            assert("The local reduce must be true." && v311);
        } else {
        }
        bool v314;
        v314 = v310 < 11l;
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
        while (while_method_3(v318)){
            assert("Tensor range check" && 0 <= v318 && v318 < 64l);
            int v320;
            v320 = 128l * v318;
            int v321;
            v321 = v320 + v317;
            assert("Tensor range check" && 0 <= v318 && v318 < 64l);
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
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1) {
    int v2;
    v2 = threadIdx.x;
    int v3;
    v3 = blockIdx.x;
    int v4;
    v4 = v3 * 32l;
    int v5;
    v5 = v2 + v4;
    unsigned long long v6;
    v6 = (unsigned long long)v5;
    curandStatePhilox4_32_10_t v7;
    curand_init(12344321ull,v6,0ull,&v7);
    float * v8;
    v8 = reinterpret_cast<float *>(&v0[0ull]);
    float * v10;
    v10 = reinterpret_cast<float *>(&v1[0ull]);
    float * v12;
    v12 = reinterpret_cast<float *>(&v0[12288ull]);
    int v14;
    v14 = blockIdx.x;
    assert("Tensor range check" && 0 <= v14 && v14 < 24l);
    int v15;
    v15 = 128l * v14;
    int v16;
    v16 = blockIdx.x;
    assert("Tensor range check" && 0 <= v16 && v16 < 24l);
    int v17;
    v17 = 256l * v16;
    method_0(v10, v12, v17, v8, v15);
    float * v18;
    v18 = reinterpret_cast<float *>(&v0[36864ull]);
    method_1(v18, v12);
    float * v20;
    v20 = reinterpret_cast<float *>(&v0[61440ull]);
    method_2(v20, v18);
    float * v22;
    v22 = reinterpret_cast<float *>(&v1[512ull]);
    float * v24;
    v24 = reinterpret_cast<float *>(&v0[86016ull]);
    int v26;
    v26 = blockIdx.x;
    assert("Tensor range check" && 0 <= v26 && v26 < 24l);
    int v27;
    v27 = 256l * v26;
    int v28;
    v28 = blockIdx.x;
    assert("Tensor range check" && 0 <= v28 && v28 < 24l);
    int v29;
    v29 = 256l * v28;
    method_3(v22, v24, v29, v20, v27);
    float * v30;
    v30 = reinterpret_cast<float *>(&v0[110592ull]);
    method_1(v30, v24);
    float * v32;
    v32 = reinterpret_cast<float *>(&v0[135168ull]);
    method_2(v32, v30);
    float * v34;
    v34 = reinterpret_cast<float *>(&v1[1536ull]);
    float * v36;
    v36 = reinterpret_cast<float *>(&v0[159744ull]);
    int v38;
    v38 = blockIdx.x;
    assert("Tensor range check" && 0 <= v38 && v38 < 24l);
    int v39;
    v39 = 256l * v38;
    int v40;
    v40 = blockIdx.x;
    assert("Tensor range check" && 0 <= v40 && v40 < 24l);
    int v41;
    v41 = 256l * v40;
    method_3(v34, v36, v41, v32, v39);
    float * v42;
    v42 = reinterpret_cast<float *>(&v0[184320ull]);
    int * v44;
    v44 = reinterpret_cast<int *>(&v0[208896ull]);
    return method_4(v44, v42, v36, v7);
}
extern "C" __global__ void entry1(unsigned char * v0, unsigned char * v1) {
    int v2;
    v2 = threadIdx.x;
    int v3;
    v3 = blockIdx.x;
    int v4;
    v4 = v3 * 32l;
    int v5;
    v5 = v2 + v4;
    unsigned long long v6;
    v6 = (unsigned long long)v5;
    curandStatePhilox4_32_10_t v7;
    curand_init(12344321ull,v6,0ull,&v7);
    int v8;
    v8 = 0l;
    while (while_method_4(v8)){
        float * v10;
        v10 = reinterpret_cast<float *>(&v0[0ull]);
        float * v12;
        v12 = reinterpret_cast<float *>(&v1[0ull]);
        assert("Tensor range check" && 0 <= v8 && v8 < 16l);
        int v14;
        v14 = 128l * v8;
        float * v15;
        v15 = reinterpret_cast<float *>(&v0[12288ull]);
        int v17;
        v17 = blockIdx.x;
        assert("Tensor range check" && 0 <= v17 && v17 < 24l);
        int v18;
        v18 = 128l * v17;
        int v19;
        v19 = blockIdx.x;
        assert("Tensor range check" && 0 <= v19 && v19 < 24l);
        int v20;
        v20 = 256l * v19;
        method_5(v12, v14, v15, v20, v10, v18);
        float * v21;
        v21 = reinterpret_cast<float *>(&v0[36864ull]);
        method_1(v21, v15);
        float * v23;
        v23 = reinterpret_cast<float *>(&v0[61440ull]);
        method_2(v23, v21);
        float * v25;
        v25 = reinterpret_cast<float *>(&v1[8192ull]);
        assert("Tensor range check" && 0 <= v8 && v8 < 16l);
        int v27;
        v27 = 256l * v8;
        float * v28;
        v28 = reinterpret_cast<float *>(&v0[86016ull]);
        int v30;
        v30 = blockIdx.x;
        assert("Tensor range check" && 0 <= v30 && v30 < 24l);
        int v31;
        v31 = 256l * v30;
        int v32;
        v32 = blockIdx.x;
        assert("Tensor range check" && 0 <= v32 && v32 < 24l);
        int v33;
        v33 = 256l * v32;
        method_6(v25, v27, v28, v33, v23, v31);
        float * v34;
        v34 = reinterpret_cast<float *>(&v0[110592ull]);
        method_1(v34, v28);
        float * v36;
        v36 = reinterpret_cast<float *>(&v0[135168ull]);
        method_2(v36, v34);
        float * v38;
        v38 = reinterpret_cast<float *>(&v1[24576ull]);
        assert("Tensor range check" && 0 <= v8 && v8 < 16l);
        float * v40;
        v40 = reinterpret_cast<float *>(&v0[159744ull]);
        int v42;
        v42 = blockIdx.x;
        assert("Tensor range check" && 0 <= v42 && v42 < 24l);
        int v43;
        v43 = 256l * v42;
        int v44;
        v44 = blockIdx.x;
        assert("Tensor range check" && 0 <= v44 && v44 < 24l);
        int v45;
        v45 = 256l * v44;
        method_6(v38, v27, v40, v45, v36, v43);
        float * v46;
        v46 = reinterpret_cast<float *>(&v0[184320ull]);
        assert("Tensor range check" && 0 <= v8 && v8 < 16l);
        int v48;
        v48 = 6144l * v8;
        int * v49;
        v49 = reinterpret_cast<int *>(&v0[577536ull]);
        assert("Tensor range check" && 0 <= v8 && v8 < 16l);
        int v51;
        v51 = 384l * v8;
        method_7(v49, v51, v46, v48, v40, v7);
        v8 += 1l ;
    }
    return ;
}
extern "C" __global__ void entry2(unsigned char * v0, unsigned char * v1) {
    int v2;
    v2 = threadIdx.x;
    int v3;
    v3 = blockIdx.x;
    int v4;
    v4 = v3 * 32l;
    int v5;
    v5 = v2 + v4;
    unsigned long long v6;
    v6 = (unsigned long long)v5;
    curandStatePhilox4_32_10_t v7;
    curand_init(12344321ull,v6,0ull,&v7);
    int v8;
    v8 = 0l;
    while (while_method_1(v8)){
        float * v10;
        v10 = reinterpret_cast<float *>(&v0[0ull]);
        float * v12;
        v12 = reinterpret_cast<float *>(&v1[0ull]);
        assert("Tensor range check" && 0 <= v8 && v8 < 4l);
        int v14;
        v14 = 67108864l * v8;
        float * v15;
        v15 = reinterpret_cast<float *>(&v0[25165824ull]);
        int v17;
        v17 = blockIdx.x;
        assert("Tensor range check" && 0 <= v17 && v17 < 24l);
        int v18;
        v18 = 262144l * v17;
        int v19;
        v19 = blockIdx.x;
        assert("Tensor range check" && 0 <= v19 && v19 < 24l);
        int v20;
        v20 = 262144l * v19;
        method_8(v12, v14, v15, v20, v10, v18);
        float * v21;
        v21 = reinterpret_cast<float *>(&v0[50331648ull]);
        assert("Tensor range check" && 0 <= v8 && v8 < 4l);
        int v23;
        v23 = 6291456l * v8;
        int * v24;
        v24 = reinterpret_cast<int *>(&v0[150994944ull]);
        assert("Tensor range check" && 0 <= v8 && v8 < 4l);
        int v26;
        v26 = 768l * v8;
        method_9(v24, v26, v21, v23, v15, v7);
        v8 += 1l ;
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

import sys
import pathlib
options = []
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39')
options.append('--restrict')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def method1(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method2(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method0() -> None:
    v0 = "test_text_outputs/layers/"
    v1 = "test1"
    v2 = "layers.txt"
    v3 = pathlib.Path(v0,v1,v2)
    del v0, v1, v2
    v3.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v3),'w')
    del v3
    v4 = cp.empty(160,dtype=cp.uint8)
    v5 = cp.empty(2304,dtype=cp.uint8)
    del v5
    v8 = "{}\n"
    v9 = "---"
    print(v8.format(v9),end="")
    del v9
    v11 = v4[0:0+4*16].view(cp.float32)
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method1(v33):
        v35 = v31
        v36 = v35 >= 100
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method1(v41):
            v43 = v31
            v44 = v43 >= 100
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 4
            v51 = v50 + v41
            del v50
            v52 = v11[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v11, v31, v33
    print(v32.format(']'),end="")
    v54 = "\n"
    print(v54,end="")
    v56 = v4[64:64+4*16].view(cp.float32)
    v76 = 0
    print(v32.format('['),end="")
    v77 = 0
    while method1(v77):
        v79 = v76
        v80 = v79 >= 100
        del v79
        if v80:
            v81 = " ..."
            print(v32.format(v81),end="")
            del v81
            break
        else:
            pass
        del v80
        v82 = v77 == 0
        v83 = v82 != True
        del v82
        if v83:
            v84 = "; "
            print(v32.format(v84),end="")
            del v84
        else:
            pass
        del v83
        print(v32.format('['),end="")
        v85 = 0
        while method1(v85):
            v87 = v76
            v88 = v87 >= 100
            del v87
            if v88:
                v89 = " ..."
                print(v32.format(v89),end="")
                del v89
                break
            else:
                pass
            del v88
            v90 = v85 == 0
            v91 = v90 != True
            del v90
            if v91:
                v92 = "; "
                print(v32.format(v92),end="")
                del v92
            else:
                pass
            del v91
            v93 = v76 + 1
            v76 = v93
            del v93
            v94 = v77 * 4
            v95 = v94 + v85
            del v94
            v96 = v56[v95].item()
            del v95
            v97 = "{:.6f}"
            print(v97.format(v96),end="")
            del v96, v97
            v85 += 1 
        del v85
        print(v32.format(']'),end="")
        v77 += 1 
    del v56, v76, v77
    print(v32.format(']'),end="")
    print(v54,end="")
    v99 = v4[128:128+4*8].view(cp.float32)
    v119 = 0
    print(v32.format('['),end="")
    v120 = 0
    while method2(v120):
        v122 = v119
        v123 = v122 >= 100
        del v122
        if v123:
            v124 = " ..."
            print(v32.format(v124),end="")
            del v124
            break
        else:
            pass
        del v123
        v125 = v120 == 0
        v126 = v125 != True
        del v125
        if v126:
            v127 = "; "
            print(v32.format(v127),end="")
            del v127
        else:
            pass
        del v126
        print(v32.format('['),end="")
        v128 = 0
        while method1(v128):
            v130 = v119
            v131 = v130 >= 100
            del v130
            if v131:
                v132 = " ..."
                print(v32.format(v132),end="")
                del v132
                break
            else:
                pass
            del v131
            v133 = v128 == 0
            v134 = v133 != True
            del v133
            if v134:
                v135 = "; "
                print(v32.format(v135),end="")
                del v135
            else:
                pass
            del v134
            v136 = v119 + 1
            v119 = v136
            del v136
            v137 = v120 * 4
            v138 = v137 + v128
            del v137
            v139 = v99[v138].item()
            del v138
            v140 = "{:.6f}"
            print(v140.format(v139),end="")
            del v139, v140
            v128 += 1 
        del v128
        print(v32.format(']'),end="")
        v120 += 1 
    del v99, v119, v120
    print(v32.format(']'),end="")
    print(v54,end="")
    v142 = v4[0:0+4*16].view(cp.float32)
    v143 = cp.random.normal(0.0,0.25,16,dtype=cp.float32) # type: ignore
    cp.copyto(v142[0:0+16],v143[0:0+16])
    del v142, v143
    v145 = v4[64:64+4*16].view(cp.float32)
    v146 = cp.random.normal(0.0,0.25,16,dtype=cp.float32) # type: ignore
    cp.copyto(v145[0:0+16],v146[0:0+16])
    del v145, v146
    v148 = v4[128:128+4*8].view(cp.float32)
    v149 = cp.random.normal(0.0,0.35355338,8,dtype=cp.float32) # type: ignore
    cp.copyto(v148[0:0+8],v149[0:0+8])
    del v148, v149
    v152 = "Done initing."
    print(v8.format(v152),end="")
    del v8, v152
    v154 = v4[0:0+4*16].view(cp.float32)
    v174 = 0
    print(v32.format('['),end="")
    v175 = 0
    while method1(v175):
        v177 = v174
        v178 = v177 >= 100
        del v177
        if v178:
            v179 = " ..."
            print(v32.format(v179),end="")
            del v179
            break
        else:
            pass
        del v178
        v180 = v175 == 0
        v181 = v180 != True
        del v180
        if v181:
            v182 = "; "
            print(v32.format(v182),end="")
            del v182
        else:
            pass
        del v181
        print(v32.format('['),end="")
        v183 = 0
        while method1(v183):
            v185 = v174
            v186 = v185 >= 100
            del v185
            if v186:
                v187 = " ..."
                print(v32.format(v187),end="")
                del v187
                break
            else:
                pass
            del v186
            v188 = v183 == 0
            v189 = v188 != True
            del v188
            if v189:
                v190 = "; "
                print(v32.format(v190),end="")
                del v190
            else:
                pass
            del v189
            v191 = v174 + 1
            v174 = v191
            del v191
            v192 = v175 * 4
            v193 = v192 + v183
            del v192
            v194 = v154[v193].item()
            del v193
            v195 = "{:.6f}"
            print(v195.format(v194),end="")
            del v194, v195
            v183 += 1 
        del v183
        print(v32.format(']'),end="")
        v175 += 1 
    del v154, v174, v175
    print(v32.format(']'),end="")
    print(v54,end="")
    v197 = v4[64:64+4*16].view(cp.float32)
    v217 = 0
    print(v32.format('['),end="")
    v218 = 0
    while method1(v218):
        v220 = v217
        v221 = v220 >= 100
        del v220
        if v221:
            v222 = " ..."
            print(v32.format(v222),end="")
            del v222
            break
        else:
            pass
        del v221
        v223 = v218 == 0
        v224 = v223 != True
        del v223
        if v224:
            v225 = "; "
            print(v32.format(v225),end="")
            del v225
        else:
            pass
        del v224
        print(v32.format('['),end="")
        v226 = 0
        while method1(v226):
            v228 = v217
            v229 = v228 >= 100
            del v228
            if v229:
                v230 = " ..."
                print(v32.format(v230),end="")
                del v230
                break
            else:
                pass
            del v229
            v231 = v226 == 0
            v232 = v231 != True
            del v231
            if v232:
                v233 = "; "
                print(v32.format(v233),end="")
                del v233
            else:
                pass
            del v232
            v234 = v217 + 1
            v217 = v234
            del v234
            v235 = v218 * 4
            v236 = v235 + v226
            del v235
            v237 = v197[v236].item()
            del v236
            v238 = "{:.6f}"
            print(v238.format(v237),end="")
            del v237, v238
            v226 += 1 
        del v226
        print(v32.format(']'),end="")
        v218 += 1 
    del v197, v217, v218
    print(v32.format(']'),end="")
    print(v54,end="")
    v240 = v4[128:128+4*8].view(cp.float32)
    del v4
    v260 = 0
    print(v32.format('['),end="")
    v261 = 0
    while method2(v261):
        v263 = v260
        v264 = v263 >= 100
        del v263
        if v264:
            v265 = " ..."
            print(v32.format(v265),end="")
            del v265
            break
        else:
            pass
        del v264
        v266 = v261 == 0
        v267 = v266 != True
        del v266
        if v267:
            v268 = "; "
            print(v32.format(v268),end="")
            del v268
        else:
            pass
        del v267
        print(v32.format('['),end="")
        v269 = 0
        while method1(v269):
            v271 = v260
            v272 = v271 >= 100
            del v271
            if v272:
                v273 = " ..."
                print(v32.format(v273),end="")
                del v273
                break
            else:
                pass
            del v272
            v274 = v269 == 0
            v275 = v274 != True
            del v274
            if v275:
                v276 = "; "
                print(v32.format(v276),end="")
                del v276
            else:
                pass
            del v275
            v277 = v260 + 1
            v260 = v277
            del v277
            v278 = v261 * 4
            v279 = v278 + v269
            del v278
            v280 = v240[v279].item()
            del v279
            v281 = "{:.6f}"
            print(v281.format(v280),end="")
            del v280, v281
            v269 += 1 
        del v269
        print(v32.format(']'),end="")
        v261 += 1 
    del v240, v260, v261
    print(v32.format(']'),end="")
    del v32
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method4(v0 : i32) -> bool:
    v1 = v0 < 24
    del v0
    return v1
def method5(v0 : i32) -> bool:
    v1 = v0 < 1
    del v0
    return v1
def method3() -> None:
    v0 = "test_text_outputs/layers/"
    v1 = "test2"
    v2 = "layers.txt"
    v3 = pathlib.Path(v0,v1,v2)
    del v0, v1, v2
    v3.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v3),'w')
    del v3
    v4 = cp.empty(128,dtype=cp.uint8)
    v5 = cp.empty(2112,dtype=cp.uint8)
    v8 = "{}\n"
    v9 = "---"
    print(v8.format(v9),end="")
    del v9
    v11 = v4[0:0+4*8].view(cp.float32)
    v12 = cp.random.normal(0.0,0.35355338,8,dtype=cp.float32) # type: ignore
    cp.copyto(v11[0:0+8],v12[0:0+8])
    del v11, v12
    v14 = v4[32:32+4*16].view(cp.float32)
    v15 = cp.random.normal(0.0,0.25,16,dtype=cp.float32) # type: ignore
    cp.copyto(v14[0:0+16],v15[0:0+16])
    del v14, v15
    v17 = v4[96:96+4*8].view(cp.float32)
    v18 = cp.random.normal(0.0,0.35355338,8,dtype=cp.float32) # type: ignore
    cp.copyto(v17[0:0+8],v18[0:0+8])
    del v17, v18
    v21 = "Here are the weight matrices."
    print(v8.format(v21),end="")
    del v21
    v23 = v4[0:0+4*8].view(cp.float32)
    v43 = 0
    v44 = "{}"
    print(v44.format('['),end="")
    v45 = 0
    while method1(v45):
        v47 = v43
        v48 = v47 >= 100
        del v47
        if v48:
            v49 = " ..."
            print(v44.format(v49),end="")
            del v49
            break
        else:
            pass
        del v48
        v50 = v45 == 0
        v51 = v50 != True
        del v50
        if v51:
            v52 = "; "
            print(v44.format(v52),end="")
            del v52
        else:
            pass
        del v51
        print(v44.format('['),end="")
        v53 = 0
        while method2(v53):
            v55 = v43
            v56 = v55 >= 100
            del v55
            if v56:
                v57 = " ..."
                print(v44.format(v57),end="")
                del v57
                break
            else:
                pass
            del v56
            v58 = v53 == 0
            v59 = v58 != True
            del v58
            if v59:
                v60 = "; "
                print(v44.format(v60),end="")
                del v60
            else:
                pass
            del v59
            v61 = v43 + 1
            v43 = v61
            del v61
            v62 = v45 * 2
            v63 = v62 + v53
            del v62
            v64 = v23[v63].item()
            del v63
            v65 = "{:.6f}"
            print(v65.format(v64),end="")
            del v64, v65
            v53 += 1 
        del v53
        print(v44.format(']'),end="")
        v45 += 1 
    del v23, v43, v45
    print(v44.format(']'),end="")
    v66 = "\n"
    print(v66,end="")
    v68 = v4[32:32+4*16].view(cp.float32)
    v88 = 0
    print(v44.format('['),end="")
    v89 = 0
    while method1(v89):
        v91 = v88
        v92 = v91 >= 100
        del v91
        if v92:
            v93 = " ..."
            print(v44.format(v93),end="")
            del v93
            break
        else:
            pass
        del v92
        v94 = v89 == 0
        v95 = v94 != True
        del v94
        if v95:
            v96 = "; "
            print(v44.format(v96),end="")
            del v96
        else:
            pass
        del v95
        print(v44.format('['),end="")
        v97 = 0
        while method1(v97):
            v99 = v88
            v100 = v99 >= 100
            del v99
            if v100:
                v101 = " ..."
                print(v44.format(v101),end="")
                del v101
                break
            else:
                pass
            del v100
            v102 = v97 == 0
            v103 = v102 != True
            del v102
            if v103:
                v104 = "; "
                print(v44.format(v104),end="")
                del v104
            else:
                pass
            del v103
            v105 = v88 + 1
            v88 = v105
            del v105
            v106 = v89 * 4
            v107 = v106 + v97
            del v106
            v108 = v68[v107].item()
            del v107
            v109 = "{:.6f}"
            print(v109.format(v108),end="")
            del v108, v109
            v97 += 1 
        del v97
        print(v44.format(']'),end="")
        v89 += 1 
    del v68, v88, v89
    print(v44.format(']'),end="")
    print(v66,end="")
    v111 = v4[96:96+4*8].view(cp.float32)
    del v4
    v131 = 0
    print(v44.format('['),end="")
    v132 = 0
    while method2(v132):
        v134 = v131
        v135 = v134 >= 100
        del v134
        if v135:
            v136 = " ..."
            print(v44.format(v136),end="")
            del v136
            break
        else:
            pass
        del v135
        v137 = v132 == 0
        v138 = v137 != True
        del v137
        if v138:
            v139 = "; "
            print(v44.format(v139),end="")
            del v139
        else:
            pass
        del v138
        print(v44.format('['),end="")
        v140 = 0
        while method1(v140):
            v142 = v131
            v143 = v142 >= 100
            del v142
            if v143:
                v144 = " ..."
                print(v44.format(v144),end="")
                del v144
                break
            else:
                pass
            del v143
            v145 = v140 == 0
            v146 = v145 != True
            del v145
            if v146:
                v147 = "; "
                print(v44.format(v147),end="")
                del v147
            else:
                pass
            del v146
            v148 = v131 + 1
            v131 = v148
            del v148
            v149 = v132 * 4
            v150 = v149 + v140
            del v149
            v151 = v111[v150].item()
            del v150
            v152 = "{:.6f}"
            print(v152.format(v151),end="")
            del v151, v152
            v140 += 1 
        del v140
        print(v44.format(']'),end="")
        v132 += 1 
    del v111, v131, v132
    print(v44.format(']'),end="")
    print(v66,end="")
    v155 = "Here is the input tensor."
    print(v8.format(v155),end="")
    del v8, v155
    v157 = v5[0:0+4*48].view(cp.float32)
    del v5
    v158 = cp.random.normal(0.0,1.0,48,dtype=cp.float32) # type: ignore
    cp.copyto(v157[0:0+48],v158[0:0+48])
    del v158
    v186 = 0
    print(v44.format('['),end="")
    v187 = 0
    while method4(v187):
        v189 = v186
        v190 = v189 >= 100
        del v189
        if v190:
            v191 = " ..."
            print(v44.format(v191),end="")
            del v191
            break
        else:
            pass
        del v190
        v192 = v187 == 0
        v193 = v192 != True
        del v192
        if v193:
            v194 = "; "
            print(v44.format(v194),end="")
            del v194
        else:
            pass
        del v193
        print(v44.format('['),end="")
        v195 = 0
        while method5(v195):
            v197 = v186
            v198 = v197 >= 100
            del v197
            if v198:
                v199 = " ..."
                print(v44.format(v199),end="")
                del v199
                break
            else:
                pass
            del v198
            v200 = v195 == 0
            v201 = v200 != True
            del v200
            if v201:
                v202 = "; "
                print(v44.format(v202),end="")
                del v202
            else:
                pass
            del v201
            print(v44.format('['),end="")
            v203 = 0
            while method2(v203):
                v205 = v186
                v206 = v205 >= 100
                del v205
                if v206:
                    v207 = " ..."
                    print(v44.format(v207),end="")
                    del v207
                    break
                else:
                    pass
                del v206
                v208 = v203 == 0
                v209 = v208 != True
                del v208
                if v209:
                    v210 = "; "
                    print(v44.format(v210),end="")
                    del v210
                else:
                    pass
                del v209
                v211 = v186 + 1
                v186 = v211
                del v211
                v212 = v187 * 2
                v213 = v195 * 2
                v214 = v212 + v213
                del v212, v213
                v215 = v214 + v203
                del v214
                v216 = v157[v215].item()
                del v215
                v217 = "{:.6f}"
                print(v217.format(v216),end="")
                del v216, v217
                v203 += 1 
            del v203
            print(v44.format(']'),end="")
            v195 += 1 
        del v195
        print(v44.format(']'),end="")
        v187 += 1 
    del v157, v186, v187
    print(v44.format(']'),end="")
    del v44
    print(v66,end="")
    del v66
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method7(v0 : i32) -> bool:
    v1 = v0 < 16
    del v0
    return v1
def method8(v0 : i32) -> bool:
    v1 = v0 < 8
    del v0
    return v1
def method6() -> None:
    v0 = "test_text_outputs/layers/"
    v1 = "test3"
    v2 = "layers.txt"
    v3 = pathlib.Path(v0,v1,v2)
    del v0, v1, v2
    v3.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v3),'w')
    del v3
    v4 = cp.empty(2560,dtype=cp.uint8)
    v5 = cp.empty(210432,dtype=cp.uint8)
    v7 = v4[0:0+4*128].view(cp.float32)
    v8 = cp.random.normal(0.0,0.088388346,128,dtype=cp.float32) # type: ignore
    cp.copyto(v7[0:0+128],v8[0:0+128])
    del v7, v8
    v10 = v4[512:512+4*256].view(cp.float32)
    v11 = cp.random.normal(0.0,0.0625,256,dtype=cp.float32) # type: ignore
    cp.copyto(v10[0:0+256],v11[0:0+256])
    del v10, v11
    v13 = v4[1536:1536+4*256].view(cp.float32)
    v14 = cp.random.normal(0.0,0.0625,256,dtype=cp.float32) # type: ignore
    cp.copyto(v13[0:0+256],v14[0:0+256])
    del v13, v14
    v17 = "{}\n"
    v18 = "Here are the weight matrices."
    print(v17.format(v18),end="")
    del v18
    v20 = v4[0:0+4*128].view(cp.float32)
    v40 = 0
    v41 = "{}"
    print(v41.format('['),end="")
    v42 = 0
    while method7(v42):
        v44 = v40
        v45 = v44 >= 100
        del v44
        if v45:
            v46 = " ..."
            print(v41.format(v46),end="")
            del v46
            break
        else:
            pass
        del v45
        v47 = v42 == 0
        v48 = v47 != True
        del v47
        if v48:
            v49 = "; "
            print(v41.format(v49),end="")
            del v49
        else:
            pass
        del v48
        print(v41.format('['),end="")
        v50 = 0
        while method8(v50):
            v52 = v40
            v53 = v52 >= 100
            del v52
            if v53:
                v54 = " ..."
                print(v41.format(v54),end="")
                del v54
                break
            else:
                pass
            del v53
            v55 = v50 == 0
            v56 = v55 != True
            del v55
            if v56:
                v57 = "; "
                print(v41.format(v57),end="")
                del v57
            else:
                pass
            del v56
            v58 = v40 + 1
            v40 = v58
            del v58
            v59 = v42 * 8
            v60 = v59 + v50
            del v59
            v61 = v20[v60].item()
            del v60
            v62 = "{:.6f}"
            print(v62.format(v61),end="")
            del v61, v62
            v50 += 1 
        del v50
        print(v41.format(']'),end="")
        v42 += 1 
    del v20, v40, v42
    print(v41.format(']'),end="")
    v63 = "\n"
    print(v63,end="")
    v65 = v4[512:512+4*256].view(cp.float32)
    v85 = 0
    print(v41.format('['),end="")
    v86 = 0
    while method7(v86):
        v88 = v85
        v89 = v88 >= 100
        del v88
        if v89:
            v90 = " ..."
            print(v41.format(v90),end="")
            del v90
            break
        else:
            pass
        del v89
        v91 = v86 == 0
        v92 = v91 != True
        del v91
        if v92:
            v93 = "; "
            print(v41.format(v93),end="")
            del v93
        else:
            pass
        del v92
        print(v41.format('['),end="")
        v94 = 0
        while method7(v94):
            v96 = v85
            v97 = v96 >= 100
            del v96
            if v97:
                v98 = " ..."
                print(v41.format(v98),end="")
                del v98
                break
            else:
                pass
            del v97
            v99 = v94 == 0
            v100 = v99 != True
            del v99
            if v100:
                v101 = "; "
                print(v41.format(v101),end="")
                del v101
            else:
                pass
            del v100
            v102 = v85 + 1
            v85 = v102
            del v102
            v103 = v86 * 16
            v104 = v103 + v94
            del v103
            v105 = v65[v104].item()
            del v104
            v106 = "{:.6f}"
            print(v106.format(v105),end="")
            del v105, v106
            v94 += 1 
        del v94
        print(v41.format(']'),end="")
        v86 += 1 
    del v65, v85, v86
    print(v41.format(']'),end="")
    print(v63,end="")
    v108 = v4[1536:1536+4*256].view(cp.float32)
    v128 = 0
    print(v41.format('['),end="")
    v129 = 0
    while method7(v129):
        v131 = v128
        v132 = v131 >= 100
        del v131
        if v132:
            v133 = " ..."
            print(v41.format(v133),end="")
            del v133
            break
        else:
            pass
        del v132
        v134 = v129 == 0
        v135 = v134 != True
        del v134
        if v135:
            v136 = "; "
            print(v41.format(v136),end="")
            del v136
        else:
            pass
        del v135
        print(v41.format('['),end="")
        v137 = 0
        while method7(v137):
            v139 = v128
            v140 = v139 >= 100
            del v139
            if v140:
                v141 = " ..."
                print(v41.format(v141),end="")
                del v141
                break
            else:
                pass
            del v140
            v142 = v137 == 0
            v143 = v142 != True
            del v142
            if v143:
                v144 = "; "
                print(v41.format(v144),end="")
                del v144
            else:
                pass
            del v143
            v145 = v128 + 1
            v128 = v145
            del v145
            v146 = v129 * 16
            v147 = v146 + v137
            del v146
            v148 = v108[v147].item()
            del v147
            v149 = "{:.6f}"
            print(v149.format(v148),end="")
            del v148, v149
            v137 += 1 
        del v137
        print(v41.format(']'),end="")
        v129 += 1 
    del v108, v128, v129
    print(v41.format(']'),end="")
    print(v63,end="")
    v151 = v5[0:0+4*3072].view(cp.float32)
    v152 = cp.random.normal(0.0,1.0,3072,dtype=cp.float32) # type: ignore
    cp.copyto(v151[0:0+3072],v152[0:0+3072])
    del v152
    v180 = 0
    print(v41.format('['),end="")
    v181 = 0
    while method4(v181):
        v183 = v180
        v184 = v183 >= 100
        del v183
        if v184:
            v185 = " ..."
            print(v41.format(v185),end="")
            del v185
            break
        else:
            pass
        del v184
        v186 = v181 == 0
        v187 = v186 != True
        del v186
        if v187:
            v188 = "; "
            print(v41.format(v188),end="")
            del v188
        else:
            pass
        del v187
        print(v41.format('['),end="")
        v189 = 0
        while method7(v189):
            v191 = v180
            v192 = v191 >= 100
            del v191
            if v192:
                v193 = " ..."
                print(v41.format(v193),end="")
                del v193
                break
            else:
                pass
            del v192
            v194 = v189 == 0
            v195 = v194 != True
            del v194
            if v195:
                v196 = "; "
                print(v41.format(v196),end="")
                del v196
            else:
                pass
            del v195
            print(v41.format('['),end="")
            v197 = 0
            while method8(v197):
                v199 = v180
                v200 = v199 >= 100
                del v199
                if v200:
                    v201 = " ..."
                    print(v41.format(v201),end="")
                    del v201
                    break
                else:
                    pass
                del v200
                v202 = v197 == 0
                v203 = v202 != True
                del v202
                if v203:
                    v204 = "; "
                    print(v41.format(v204),end="")
                    del v204
                else:
                    pass
                del v203
                v205 = v180 + 1
                v180 = v205
                del v205
                v206 = v181 * 128
                v207 = v189 * 8
                v208 = v206 + v207
                del v206, v207
                v209 = v208 + v197
                del v208
                v210 = v151[v209].item()
                del v209
                v211 = "{:.6f}"
                print(v211.format(v210),end="")
                del v210, v211
                v197 += 1 
            del v197
            print(v41.format(']'),end="")
            v189 += 1 
        del v189
        print(v41.format(']'),end="")
        v181 += 1 
    del v151, v180, v181
    print(v41.format(']'),end="")
    print(v63,end="")
    v214 = "Here is the output tensor."
    print(v17.format(v214),end="")
    del v214
    v215 = cp.cuda.Device().attributes['MultiProcessorCount']
    v216 = v215 == 24
    del v215
    v217 = v216 == False
    if v217:
        v218 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v216, v218
        del v218
    else:
        pass
    del v216, v217
    v219 = 0
    v220 = raw_module.get_function(f"entry{v219}")
    del v219
    v220.max_dynamic_shared_size_bytes = 1536 
    v220((24,),(32,),(v5, v4),shared_mem=1536)
    del v4, v220
    v222 = v5[184320:184320+4*6144].view(cp.float32)
    v224 = v5[208896:208896+4*384].view(cp.int32)
    del v5
    v269 = 0
    print(v41.format('['),end="")
    v270 = 0
    while method4(v270):
        v272 = v269
        v273 = v272 >= 100
        del v272
        if v273:
            v274 = " ..."
            print(v41.format(v274),end="")
            del v274
            break
        else:
            pass
        del v273
        v275 = v270 == 0
        v276 = v275 != True
        del v275
        if v276:
            v277 = "; "
            print(v41.format(v277),end="")
            del v277
        else:
            pass
        del v276
        print(v41.format('['),end="")
        v278 = 0
        while method7(v278):
            v280 = v269
            v281 = v280 >= 100
            del v280
            if v281:
                v282 = " ..."
                print(v41.format(v282),end="")
                del v282
                break
            else:
                pass
            del v281
            v283 = v278 == 0
            v284 = v283 != True
            del v283
            if v284:
                v285 = "; "
                print(v41.format(v285),end="")
                del v285
            else:
                pass
            del v284
            print(v41.format('['),end="")
            v286 = 0
            while method7(v286):
                v288 = v269
                v289 = v288 >= 100
                del v288
                if v289:
                    v290 = " ..."
                    print(v41.format(v290),end="")
                    del v290
                    break
                else:
                    pass
                del v289
                v291 = v286 == 0
                v292 = v291 != True
                del v291
                if v292:
                    v293 = "; "
                    print(v41.format(v293),end="")
                    del v293
                else:
                    pass
                del v292
                v294 = v269 + 1
                v269 = v294
                del v294
                v295 = v270 * 256
                v296 = v278 * 16
                v297 = v295 + v296
                del v295, v296
                v298 = v297 + v286
                del v297
                v299 = v222[v298].item()
                del v298
                v300 = "{:.6f}"
                print(v300.format(v299),end="")
                del v299, v300
                v286 += 1 
            del v286
            print(v41.format(']'),end="")
            v278 += 1 
        del v278
        print(v41.format(']'),end="")
        v270 += 1 
    del v222, v269, v270
    print(v41.format(']'),end="")
    v301 = 0
    v302 = ", {}"
    print(v302.format('['),end="")
    del v302
    v303 = 0
    while method4(v303):
        v305 = v301
        v306 = v305 >= 100
        del v305
        if v306:
            v307 = " ..."
            print(v41.format(v307),end="")
            del v307
            break
        else:
            pass
        del v306
        v308 = v303 == 0
        v309 = v308 != True
        del v308
        if v309:
            v310 = "; "
            print(v41.format(v310),end="")
            del v310
        else:
            pass
        del v309
        print(v41.format('['),end="")
        v311 = 0
        while method7(v311):
            v313 = v301
            v314 = v313 >= 100
            del v313
            if v314:
                v315 = " ..."
                print(v41.format(v315),end="")
                del v315
                break
            else:
                pass
            del v314
            v316 = v311 == 0
            v317 = v316 != True
            del v316
            if v317:
                v318 = "; "
                print(v41.format(v318),end="")
                del v318
            else:
                pass
            del v317
            v319 = v301 + 1
            v301 = v319
            del v319
            v320 = v303 * 16
            v321 = v320 + v311
            del v320
            v322 = v224[v321].item()
            del v321
            print(v41.format(v322),end="")
            del v322
            v311 += 1 
        del v311
        print(v41.format(']'),end="")
        v303 += 1 
    del v224, v301, v303
    print(v41.format(']'),end="")
    del v41
    print(v63,end="")
    del v63
    v325 = "===="
    print(v17.format(v325),end="")
    del v17, v325
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method9() -> None:
    v0 = "test_text_outputs/layers/"
    v1 = "test4"
    v2 = "layers.txt"
    v3 = pathlib.Path(v0,v1,v2)
    del v0, v1, v2
    v3.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v3),'w')
    del v3
    v4 = cp.empty(40960,dtype=cp.uint8)
    v5 = cp.empty(602112,dtype=cp.uint8)
    v7 = v4[0:0+4*2048].view(cp.float32)
    v8 = cp.random.normal(0.0,0.022097087,2048,dtype=cp.float32) # type: ignore
    cp.copyto(v7[0:0+2048],v8[0:0+2048])
    del v7, v8
    v10 = v4[8192:8192+4*4096].view(cp.float32)
    v11 = cp.random.normal(0.0,0.015625,4096,dtype=cp.float32) # type: ignore
    cp.copyto(v10[0:0+4096],v11[0:0+4096])
    del v10, v11
    v13 = v4[24576:24576+4*4096].view(cp.float32)
    v14 = cp.random.normal(0.0,0.015625,4096,dtype=cp.float32) # type: ignore
    cp.copyto(v13[0:0+4096],v14[0:0+4096])
    del v13, v14
    v17 = "{}\n"
    v18 = "Here are the weight matrices."
    print(v17.format(v18),end="")
    del v18
    v20 = v4[0:0+4*2048].view(cp.float32)
    v48 = 0
    v49 = "{}"
    print(v49.format('['),end="")
    v50 = 0
    while method7(v50):
        v52 = v48
        v53 = v52 >= 100
        del v52
        if v53:
            v54 = " ..."
            print(v49.format(v54),end="")
            del v54
            break
        else:
            pass
        del v53
        v55 = v50 == 0
        v56 = v55 != True
        del v55
        if v56:
            v57 = "; "
            print(v49.format(v57),end="")
            del v57
        else:
            pass
        del v56
        print(v49.format('['),end="")
        v58 = 0
        while method7(v58):
            v60 = v48
            v61 = v60 >= 100
            del v60
            if v61:
                v62 = " ..."
                print(v49.format(v62),end="")
                del v62
                break
            else:
                pass
            del v61
            v63 = v58 == 0
            v64 = v63 != True
            del v63
            if v64:
                v65 = "; "
                print(v49.format(v65),end="")
                del v65
            else:
                pass
            del v64
            print(v49.format('['),end="")
            v66 = 0
            while method8(v66):
                v68 = v48
                v69 = v68 >= 100
                del v68
                if v69:
                    v70 = " ..."
                    print(v49.format(v70),end="")
                    del v70
                    break
                else:
                    pass
                del v69
                v71 = v66 == 0
                v72 = v71 != True
                del v71
                if v72:
                    v73 = "; "
                    print(v49.format(v73),end="")
                    del v73
                else:
                    pass
                del v72
                v74 = v48 + 1
                v48 = v74
                del v74
                v75 = v50 * 128
                v76 = v58 * 8
                v77 = v75 + v76
                del v75, v76
                v78 = v77 + v66
                del v77
                v79 = v20[v78].item()
                del v78
                v80 = "{:.6f}"
                print(v80.format(v79),end="")
                del v79, v80
                v66 += 1 
            del v66
            print(v49.format(']'),end="")
            v58 += 1 
        del v58
        print(v49.format(']'),end="")
        v50 += 1 
    del v20, v48, v50
    print(v49.format(']'),end="")
    v81 = "\n"
    print(v81,end="")
    v83 = v4[8192:8192+4*4096].view(cp.float32)
    v111 = 0
    print(v49.format('['),end="")
    v112 = 0
    while method7(v112):
        v114 = v111
        v115 = v114 >= 100
        del v114
        if v115:
            v116 = " ..."
            print(v49.format(v116),end="")
            del v116
            break
        else:
            pass
        del v115
        v117 = v112 == 0
        v118 = v117 != True
        del v117
        if v118:
            v119 = "; "
            print(v49.format(v119),end="")
            del v119
        else:
            pass
        del v118
        print(v49.format('['),end="")
        v120 = 0
        while method7(v120):
            v122 = v111
            v123 = v122 >= 100
            del v122
            if v123:
                v124 = " ..."
                print(v49.format(v124),end="")
                del v124
                break
            else:
                pass
            del v123
            v125 = v120 == 0
            v126 = v125 != True
            del v125
            if v126:
                v127 = "; "
                print(v49.format(v127),end="")
                del v127
            else:
                pass
            del v126
            print(v49.format('['),end="")
            v128 = 0
            while method7(v128):
                v130 = v111
                v131 = v130 >= 100
                del v130
                if v131:
                    v132 = " ..."
                    print(v49.format(v132),end="")
                    del v132
                    break
                else:
                    pass
                del v131
                v133 = v128 == 0
                v134 = v133 != True
                del v133
                if v134:
                    v135 = "; "
                    print(v49.format(v135),end="")
                    del v135
                else:
                    pass
                del v134
                v136 = v111 + 1
                v111 = v136
                del v136
                v137 = v112 * 256
                v138 = v120 * 16
                v139 = v137 + v138
                del v137, v138
                v140 = v139 + v128
                del v139
                v141 = v83[v140].item()
                del v140
                v142 = "{:.6f}"
                print(v142.format(v141),end="")
                del v141, v142
                v128 += 1 
            del v128
            print(v49.format(']'),end="")
            v120 += 1 
        del v120
        print(v49.format(']'),end="")
        v112 += 1 
    del v83, v111, v112
    print(v49.format(']'),end="")
    print(v81,end="")
    v144 = v4[24576:24576+4*4096].view(cp.float32)
    v172 = 0
    print(v49.format('['),end="")
    v173 = 0
    while method7(v173):
        v175 = v172
        v176 = v175 >= 100
        del v175
        if v176:
            v177 = " ..."
            print(v49.format(v177),end="")
            del v177
            break
        else:
            pass
        del v176
        v178 = v173 == 0
        v179 = v178 != True
        del v178
        if v179:
            v180 = "; "
            print(v49.format(v180),end="")
            del v180
        else:
            pass
        del v179
        print(v49.format('['),end="")
        v181 = 0
        while method7(v181):
            v183 = v172
            v184 = v183 >= 100
            del v183
            if v184:
                v185 = " ..."
                print(v49.format(v185),end="")
                del v185
                break
            else:
                pass
            del v184
            v186 = v181 == 0
            v187 = v186 != True
            del v186
            if v187:
                v188 = "; "
                print(v49.format(v188),end="")
                del v188
            else:
                pass
            del v187
            print(v49.format('['),end="")
            v189 = 0
            while method7(v189):
                v191 = v172
                v192 = v191 >= 100
                del v191
                if v192:
                    v193 = " ..."
                    print(v49.format(v193),end="")
                    del v193
                    break
                else:
                    pass
                del v192
                v194 = v189 == 0
                v195 = v194 != True
                del v194
                if v195:
                    v196 = "; "
                    print(v49.format(v196),end="")
                    del v196
                else:
                    pass
                del v195
                v197 = v172 + 1
                v172 = v197
                del v197
                v198 = v173 * 256
                v199 = v181 * 16
                v200 = v198 + v199
                del v198, v199
                v201 = v200 + v189
                del v200
                v202 = v144[v201].item()
                del v201
                v203 = "{:.6f}"
                print(v203.format(v202),end="")
                del v202, v203
                v189 += 1 
            del v189
            print(v49.format(']'),end="")
            v181 += 1 
        del v181
        print(v49.format(']'),end="")
        v173 += 1 
    del v144, v172, v173
    print(v49.format(']'),end="")
    print(v81,end="")
    v205 = v5[0:0+4*3072].view(cp.float32)
    v206 = cp.random.normal(0.0,1.0,3072,dtype=cp.float32) # type: ignore
    cp.copyto(v205[0:0+3072],v206[0:0+3072])
    del v206
    v234 = 0
    print(v49.format('['),end="")
    v235 = 0
    while method4(v235):
        v237 = v234
        v238 = v237 >= 100
        del v237
        if v238:
            v239 = " ..."
            print(v49.format(v239),end="")
            del v239
            break
        else:
            pass
        del v238
        v240 = v235 == 0
        v241 = v240 != True
        del v240
        if v241:
            v242 = "; "
            print(v49.format(v242),end="")
            del v242
        else:
            pass
        del v241
        print(v49.format('['),end="")
        v243 = 0
        while method7(v243):
            v245 = v234
            v246 = v245 >= 100
            del v245
            if v246:
                v247 = " ..."
                print(v49.format(v247),end="")
                del v247
                break
            else:
                pass
            del v246
            v248 = v243 == 0
            v249 = v248 != True
            del v248
            if v249:
                v250 = "; "
                print(v49.format(v250),end="")
                del v250
            else:
                pass
            del v249
            print(v49.format('['),end="")
            v251 = 0
            while method8(v251):
                v253 = v234
                v254 = v253 >= 100
                del v253
                if v254:
                    v255 = " ..."
                    print(v49.format(v255),end="")
                    del v255
                    break
                else:
                    pass
                del v254
                v256 = v251 == 0
                v257 = v256 != True
                del v256
                if v257:
                    v258 = "; "
                    print(v49.format(v258),end="")
                    del v258
                else:
                    pass
                del v257
                v259 = v234 + 1
                v234 = v259
                del v259
                v260 = v235 * 128
                v261 = v243 * 8
                v262 = v260 + v261
                del v260, v261
                v263 = v262 + v251
                del v262
                v264 = v205[v263].item()
                del v263
                v265 = "{:.6f}"
                print(v265.format(v264),end="")
                del v264, v265
                v251 += 1 
            del v251
            print(v49.format(']'),end="")
            v243 += 1 
        del v243
        print(v49.format(']'),end="")
        v235 += 1 
    del v205, v234, v235
    print(v49.format(']'),end="")
    print(v81,end="")
    v268 = "Here is the output tensor."
    print(v17.format(v268),end="")
    del v17, v268
    v269 = cp.cuda.Device().attributes['MultiProcessorCount']
    v270 = v269 == 24
    del v269
    v271 = v270 == False
    if v271:
        v272 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v270, v272
        del v272
    else:
        pass
    del v270, v271
    v273 = 1
    v274 = raw_module.get_function(f"entry{v273}")
    del v273
    v274.max_dynamic_shared_size_bytes = 1536 
    v274((24,),(32,),(v5, v4),shared_mem=1536)
    del v4, v274
    v276 = v5[577536:577536+4*6144].view(cp.int32)
    del v5
    v310 = 0
    print(v49.format('['),end="")
    v311 = 0
    while method7(v311):
        v313 = v310
        v314 = v313 >= 2147483647
        del v313
        if v314:
            v315 = " ..."
            print(v49.format(v315),end="")
            del v315
            break
        else:
            pass
        del v314
        v316 = v311 == 0
        v317 = v316 != True
        del v316
        if v317:
            v318 = "; "
            print(v49.format(v318),end="")
            del v318
        else:
            pass
        del v317
        print(v49.format('['),end="")
        v319 = 0
        while method4(v319):
            v321 = v310
            v322 = v321 >= 2147483647
            del v321
            if v322:
                v323 = " ..."
                print(v49.format(v323),end="")
                del v323
                break
            else:
                pass
            del v322
            v324 = v319 == 0
            v325 = v324 != True
            del v324
            if v325:
                v326 = "; "
                print(v49.format(v326),end="")
                del v326
            else:
                pass
            del v325
            print(v49.format('['),end="")
            v327 = 0
            while method7(v327):
                v329 = v310
                v330 = v329 >= 2147483647
                del v329
                if v330:
                    v331 = " ..."
                    print(v49.format(v331),end="")
                    del v331
                    break
                else:
                    pass
                del v330
                v332 = v327 == 0
                v333 = v332 != True
                del v332
                if v333:
                    v334 = "; "
                    print(v49.format(v334),end="")
                    del v334
                else:
                    pass
                del v333
                v335 = v310 + 1
                v310 = v335
                del v335
                v336 = v311 * 384
                v337 = v319 * 16
                v338 = v336 + v337
                del v336, v337
                v339 = v338 + v327
                del v338
                v340 = v276[v339].item()
                del v339
                print(v49.format(v340),end="")
                del v340
                v327 += 1 
            del v327
            print(v49.format(']'),end="")
            v319 += 1 
        del v319
        print(v49.format(']'),end="")
        v311 += 1 
    del v276, v310, v311
    print(v49.format(']'),end="")
    del v49
    print(v81,end="")
    del v81
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method11(v0 : i32) -> bool:
    v1 = v0 < 32
    del v0
    return v1
def method10() -> None:
    v0 = "test_text_outputs/layers/"
    v1 = "test5"
    v2 = "layers.txt"
    v3 = pathlib.Path(v0,v1,v2)
    del v0, v1, v2
    v3.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v3),'w')
    del v3
    v4 = cp.empty(1073741824,dtype=cp.uint8)
    v5 = cp.empty(151007232,dtype=cp.uint8)
    v7 = v4[0:0+4*268435456].view(cp.float32)
    v8 = cp.random.normal(0.0,6.1035156E-05,268435456,dtype=cp.float32) # type: ignore
    cp.copyto(v7[0:0+268435456],v8[0:0+268435456])
    del v7, v8
    v10 = v5[0:0+4*6291456].view(cp.float32)
    v11 = cp.random.normal(0.0,1.0,6291456,dtype=cp.float32) # type: ignore
    cp.copyto(v10[0:0+6291456],v11[0:0+6291456])
    del v10, v11
    v12 = cp.cuda.Device().attributes['MultiProcessorCount']
    v13 = v12 == 24
    del v12
    v14 = v13 == False
    if v14:
        v15 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v13, v15
        del v15
    else:
        pass
    del v13, v14
    v16 = 2
    v17 = raw_module.get_function(f"entry{v16}")
    del v16
    v17.max_dynamic_shared_size_bytes = 1536 
    v17((24,),(32,),(v5, v4),shared_mem=1536)
    del v4, v17
    v19 = v5[150994944:150994944+4*3072].view(cp.int32)
    del v5
    v55 = 0
    v56 = "{}"
    print(v56.format('['),end="")
    v57 = 0
    while method1(v57):
        v59 = v55
        v60 = v59 >= 2147483647
        del v59
        if v60:
            v61 = " ..."
            print(v56.format(v61),end="")
            del v61
            break
        else:
            pass
        del v60
        v62 = v57 == 0
        v63 = v62 != True
        del v62
        if v63:
            v64 = "; "
            print(v56.format(v64),end="")
            del v64
        else:
            pass
        del v63
        print(v56.format('['),end="")
        v65 = 0
        while method4(v65):
            v67 = v55
            v68 = v67 >= 2147483647
            del v67
            if v68:
                v69 = " ..."
                print(v56.format(v69),end="")
                del v69
                break
            else:
                pass
            del v68
            v70 = v65 == 0
            v71 = v70 != True
            del v70
            if v71:
                v72 = "; "
                print(v56.format(v72),end="")
                del v72
            else:
                pass
            del v71
            print(v56.format('['),end="")
            v73 = 0
            while method11(v73):
                v75 = v55
                v76 = v75 >= 2147483647
                del v75
                if v76:
                    v77 = " ..."
                    print(v56.format(v77),end="")
                    del v77
                    break
                else:
                    pass
                del v76
                v78 = v73 == 0
                v79 = v78 != True
                del v78
                if v79:
                    v80 = "; "
                    print(v56.format(v80),end="")
                    del v80
                else:
                    pass
                del v79
                v81 = v55 + 1
                v55 = v81
                del v81
                v82 = v57 * 768
                v83 = v65 * 32
                v84 = v82 + v83
                del v82, v83
                v85 = v84 + v73
                del v84
                v86 = v19[v85].item()
                del v85
                print(v56.format(v86),end="")
                del v86
                v73 += 1 
            del v73
            print(v56.format(']'),end="")
            v65 += 1 
        del v65
        print(v56.format(']'),end="")
        v57 += 1 
    del v19, v55, v57
    print(v56.format(']'),end="")
    del v56
    v87 = "\n"
    print(v87,end="")
    del v87
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def main_body():
    cp.random.seed(12344321)
    method0()
    cp.random.seed(12344321)
    method3()
    cp.random.seed(12344321)
    method6()
    cp.random.seed(12344321)
    method9()
    cp.random.seed(12344321)
    return method10()

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
