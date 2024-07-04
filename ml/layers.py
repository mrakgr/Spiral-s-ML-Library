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

__device__ void method_1(float * v0, int v1, float * v2, int v3, float * v4, int v5);
__device__ void method_0(unsigned char * v0, unsigned char * v1, int v2, int v3);
__device__ void method_3(float * v0, int v1, float * v2);
__device__ void method_2(unsigned char * v0, unsigned char * v1, int v2, int v3);
__device__ void method_5(float * v0, int v1, float * v2);
__device__ void method_4(unsigned char * v0, unsigned char * v1, int v2, int v3);
__device__ void method_7(float * v0, int v1, float * v2, int v3, float * v4);
__device__ void method_6(unsigned char * v0, unsigned char * v1, int v2, int v3);
__device__ void method_8(unsigned char * v0, unsigned char * v1, int v2, int v3);
__device__ void method_9(unsigned char * v0, unsigned char * v1, int v2, int v3);
__device__ void method_10(unsigned char * v0, unsigned char * v1, int v2, int v3);
struct Tuple0;
struct Tuple1;
__device__ void method_12(curandStatePhilox4_32_10_t & v0, int * v1, int v2, float * v3, int v4);
__device__ void method_11(unsigned char * v0, unsigned char * v1, int v2, int v3, curandStatePhilox4_32_10_t & v4);
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
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 2l;
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
    while (while_method_1(v65)){
        int v67;
        v67 = 0l;
        while (while_method_1(v67)){
            assert("Tensor range check" && 0 <= v65 && v65 < 1l);
            assert("Tensor range check" && 0 <= v67 && v67 < 1l);
            int v69;
            v69 = 16l * v67;
            int v70;
            v70 = v69 + v1;
            int v71;
            v71 = 256l * v65;
            int v72;
            v72 = v71 + v70;
            float * v73;
            v73 = v0+v72;
            // Pushing the loop unrolling to: 0
            int v75;
            v75 = 0l;
            #pragma unroll
            while (while_method_1(v75)){
                int v77;
                v77 = 0l;
                #pragma unroll
                while (while_method_1(v77)){
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
            while (while_method_1(v81)){
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
                v90 = v89 + v3;
                assert("Tensor range check" && 0 <= v81 && v81 < 1l);
                int v91;
                v91 = v85 + v90;
                float * v92;
                v92 = v2+v91;
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
                while (while_method_1(v112)){
                    int v114;
                    v114 = 0l;
                    #pragma unroll
                    while (while_method_1(v114)){
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
                        while (while_method_0(v122)){
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
                while (while_method_1(v147)){
                    int v149;
                    v149 = 0l;
                    #pragma unroll
                    while (while_method_1(v149)){
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
                        while (while_method_0(v157)){
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
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v164[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v165[1l];
                int v166;
                v166 = 0l;
                #pragma unroll
                while (while_method_1(v166)){
                    int v168;
                    v168 = 0l;
                    #pragma unroll
                    while (while_method_1(v168)){
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
                while (while_method_1(v196)){
                    int v198;
                    v198 = 0l;
                    #pragma unroll
                    while (while_method_1(v198)){
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
                __syncthreads();
                int v226;
                v226 = 0l;
                #pragma unroll
                while (while_method_1(v226)){
                    int v228;
                    v228 = 0l;
                    #pragma unroll
                    while (while_method_1(v228)){
                        int v230;
                        v230 = 0l;
                        #pragma unroll
                        while (while_method_1(v230)){
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
            while (while_method_1(v238)){
                int v240;
                v240 = 0l;
                #pragma unroll
                while (while_method_1(v240)){
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
            __syncthreads();
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
                while (while_method_1(v269)){
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
            __syncthreads();
            // Poping the loop unrolling to: 0
            v67 += 1l ;
        }
        v65 += 1l ;
    }
    return ;
}
__device__ void method_0(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[0ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v6;
    v6 = 128l * v3;
    float * v7;
    v7 = reinterpret_cast<float *>(&v1[0ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 4l);
    int v9;
    v9 = 128l * v2;
    float * v10;
    v10 = reinterpret_cast<float *>(&v0[512ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v12;
    v12 = 256l * v3;
    return method_1(v10, v12, v7, v9, v4, v6);
}
__device__ void method_3(float * v0, int v1, float * v2){
    int v3;
    v3 = threadIdx.x;
    bool v4;
    v4 = 0l <= v3;
    bool v5;
    v5 = v4 == false;
    if (v5){
        assert("The index needs to be zero or positive." && v4);
    } else {
    }
    int v7;
    v7 = v3 % 4l;
    int v8;
    v8 = v3 / 4l;
    bool v9;
    v9 = v8 < 8l;
    bool v10;
    v10 = v9 == false;
    if (v10){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v9);
    } else {
    }
    assert("Tensor range check" && 0 <= v8 && v8 < 8l);
    assert("Tensor range check" && 0 <= v7 && v7 < 4l);
    int v12;
    v12 = 4l * v7;
    int v13;
    v13 = v12 + v1;
    int v14;
    v14 = 16l * v8;
    int v15;
    v15 = v14 + v13;
    assert("Tensor range check" && 0 <= v8 && v8 < 8l);
    assert("Tensor range check" && 0 <= v7 && v7 < 4l);
    int v16;
    v16 = 0l;
    while (while_method_2(v16)){
        assert("Tensor range check" && 0 <= v16 && v16 < 2l);
        int v18;
        v18 = 128l * v16;
        int v19;
        v19 = v18 + v15;
        assert("Tensor range check" && 0 <= v16 && v16 < 2l);
        float v20[4l];
        int v21[4l];
        int v22;
        v22 = 0l;
        while (while_method_1(v22)){
            assert("Tensor range check" && 0 <= v22 && v22 < 1l);
            int v24;
            v24 = 4l * v22;
            assert("Tensor range check" && 0 <= v22 && v22 < 1l);
            int v25;
            v25 = 16l * v22;
            int v26;
            v26 = v25 + v19;
            int4* v27;
            v27 = reinterpret_cast<int4*>(v2 + v26);
            int4* v28;
            v28 = reinterpret_cast<int4*>(v20 + v24);
            assert("Pointer alignment check" && (unsigned long long)(v27) % 4l == 0 && (unsigned long long)(v28) % 4l == 0);
            *v28 = *v27;
            v22 += 1l ;
        }
        int v29;
        v29 = 0l;
        while (while_method_1(v29)){
            int v31;
            v31 = 0l;
            while (while_method_0(v31)){
                bool v33;
                v33 = 0l <= v31;
                bool v35;
                if (v33){
                    bool v34;
                    v34 = v31 < 4l;
                    v35 = v34;
                } else {
                    v35 = false;
                }
                bool v36;
                v36 = v35 == false;
                if (v36){
                    assert("The indices should be inside the range of the dimension." && v35);
                } else {
                }
                bool v38;
                v38 = 0l <= v7;
                bool v40;
                if (v38){
                    bool v39;
                    v39 = v7 < 4l;
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
                int v43;
                v43 = v7 * 4l;
                int v44;
                v44 = v31 + v43;
                bool v45;
                v45 = 0l <= v29;
                bool v47;
                if (v45){
                    bool v46;
                    v46 = v29 < 1l;
                    v47 = v46;
                } else {
                    v47 = false;
                }
                bool v48;
                v48 = v47 == false;
                if (v48){
                    assert("The indices should be inside the range of the dimension." && v47);
                } else {
                }
                int v50;
                v50 = v29 * 16l;
                int v51;
                v51 = v44 + v50;
                assert("Tensor range check" && 0 <= v29 && v29 < 1l);
                assert("Tensor range check" && 0 <= v31 && v31 < 4l);
                int v52;
                v52 = 4l * v29;
                int v53;
                v53 = v52 + v31;
                v21[v53] = v51;
                v31 += 1l ;
            }
            v29 += 1l ;
        }
        bool v54;
        v54 = 0l <= v8;
        bool v55;
        v55 = v54 && v9;
        bool v56;
        v56 = v55 == false;
        if (v56){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v55);
        } else {
        }
        bool v58;
        v58 = 0l <= v16;
        bool v60;
        if (v58){
            bool v59;
            v59 = v16 < 2l;
            v60 = v59;
        } else {
            v60 = false;
        }
        bool v61;
        v61 = v60 == false;
        if (v61){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v60);
        } else {
        }
        int v63;
        v63 = v16 * 8l;
        int v64;
        v64 = v63 + v8;
        float v65[4l];
        int v66;
        v66 = 0l;
        while (while_method_1(v66)){
            int v68;
            v68 = 0l;
            while (while_method_0(v68)){
                assert("Tensor range check" && 0 <= v66 && v66 < 1l);
                assert("Tensor range check" && 0 <= v68 && v68 < 4l);
                int v70;
                v70 = 4l * v66;
                int v71;
                v71 = v70 + v68;
                float v72;
                v72 = v20[v71];
                float v73;
                v73 = v72 * v72;
                assert("Tensor range check" && 0 <= v66 && v66 < 1l);
                assert("Tensor range check" && 0 <= v68 && v68 < 4l);
                v65[v71] = v73;
                v68 += 1l ;
            }
            v66 += 1l ;
        }
        float v74;
        v74 = 0.0f;
        int v75;
        v75 = 0l;
        while (while_method_1(v75)){
            int v77;
            v77 = 0l;
            while (while_method_0(v77)){
                assert("Tensor range check" && 0 <= v75 && v75 < 1l);
                assert("Tensor range check" && 0 <= v77 && v77 < 4l);
                int v79;
                v79 = 4l * v75;
                int v80;
                v80 = v79 + v77;
                float v81;
                v81 = v65[v80];
                float v82;
                v82 = v74 + v81;
                v74 = v82;
                v77 += 1l ;
            }
            v75 += 1l ;
        }
        auto v83 = cooperative_groups::coalesced_threads();
        int v84;
        v84 = threadIdx.x;
        int v85;
        v85 = v84 / 4l;
        auto v86 = cooperative_groups::labeled_partition(v83,v85);
        Closure0 v87{};
        float v88;
        v88 = cooperative_groups::reduce(v86, v74, v87);
        float v89[4l];
        int v90;
        v90 = 0l;
        while (while_method_1(v90)){
            int v92;
            v92 = 0l;
            while (while_method_0(v92)){
                assert("Tensor range check" && 0 <= v90 && v90 < 1l);
                assert("Tensor range check" && 0 <= v92 && v92 < 4l);
                int v94;
                v94 = 4l * v90;
                int v95;
                v95 = v94 + v92;
                float v96;
                v96 = v20[v95];
                bool v97;
                v97 = v88 == 0.0f;
                bool v98;
                v98 = v97 != true;
                float v100;
                if (v98){
                    float v99;
                    v99 = v96 / v88;
                    v100 = v99;
                } else {
                    v100 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v90 && v90 < 1l);
                assert("Tensor range check" && 0 <= v92 && v92 < 4l);
                v89[v95] = v100;
                v92 += 1l ;
            }
            v90 += 1l ;
        }
        int v101;
        v101 = 0l;
        while (while_method_1(v101)){
            assert("Tensor range check" && 0 <= v101 && v101 < 1l);
            int v103;
            v103 = 16l * v101;
            int v104;
            v104 = v103 + v19;
            assert("Tensor range check" && 0 <= v101 && v101 < 1l);
            int v105;
            v105 = 4l * v101;
            int4* v106;
            v106 = reinterpret_cast<int4*>(v89 + v105);
            int4* v107;
            v107 = reinterpret_cast<int4*>(v0 + v104);
            assert("Pointer alignment check" && (unsigned long long)(v106) % 4l == 0 && (unsigned long long)(v107) % 4l == 0);
            *v107 = *v106;
            v101 += 1l ;
        }
        v16 += 1l ;
    }
    __syncthreads();
    return ;
}
__device__ void method_2(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[512ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v6;
    v6 = 256l * v3;
    float * v7;
    v7 = reinterpret_cast<float *>(&v0[1536ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    return method_3(v7, v6, v4);
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 64l;
    return v1;
}
__device__ void method_5(float * v0, int v1, float * v2){
    int v3;
    v3 = threadIdx.x;
    int v4;
    v4 = v3;
    while (while_method_3(v4)){
        bool v6;
        v6 = 0l <= v4;
        bool v7;
        v7 = v6 == false;
        if (v7){
            assert("The index needs to be zero or positive." && v6);
        } else {
        }
        int v9;
        v9 = v4 % 4l;
        int v10;
        v10 = v4 / 4l;
        bool v11;
        v11 = v10 < 16l;
        bool v12;
        v12 = v11 == false;
        if (v12){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v11);
        } else {
        }
        assert("Tensor range check" && 0 <= v10 && v10 < 16l);
        assert("Tensor range check" && 0 <= v9 && v9 < 4l);
        int v14;
        v14 = 4l * v9;
        int v15;
        v15 = v14 + v1;
        int v16;
        v16 = 16l * v10;
        int v17;
        v17 = v16 + v15;
        assert("Tensor range check" && 0 <= v10 && v10 < 16l);
        assert("Tensor range check" && 0 <= v9 && v9 < 4l);
        float v18[4l];
        float v19[4l];
        int4* v20;
        v20 = reinterpret_cast<int4*>(v2 + v17);
        int4* v21;
        v21 = reinterpret_cast<int4*>(v18 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v20) % 4l == 0 && (unsigned long long)(v21) % 4l == 0);
        *v21 = *v20;
        // Pushing the loop unrolling to: 0
        int v22;
        v22 = 0l;
        #pragma unroll
        while (while_method_0(v22)){
            assert("Tensor range check" && 0 <= v22 && v22 < 4l);
            float v24;
            v24 = v18[v22];
            bool v25;
            v25 = 0.0f >= v24;
            float v26;
            if (v25){
                v26 = 0.0f;
            } else {
                v26 = v24;
            }
            assert("Tensor range check" && 0 <= v22 && v22 < 4l);
            v19[v22] = v26;
            v22 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v27;
        v27 = reinterpret_cast<int4*>(v19 + 0l);
        int4* v28;
        v28 = reinterpret_cast<int4*>(v0 + v17);
        assert("Pointer alignment check" && (unsigned long long)(v27) % 4l == 0 && (unsigned long long)(v28) % 4l == 0);
        *v28 = *v27;
        v4 += 32l ;
    }
    __syncthreads();
    return ;
}
__device__ void method_4(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[1536ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v6;
    v6 = 256l * v3;
    float * v7;
    v7 = reinterpret_cast<float *>(&v0[2560ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    return method_5(v7, v6, v4);
}
__device__ void method_7(float * v0, int v1, float * v2, int v3, float * v4){
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
    while (while_method_1(v64)){
        int v66;
        v66 = 0l;
        while (while_method_1(v66)){
            assert("Tensor range check" && 0 <= v64 && v64 < 1l);
            assert("Tensor range check" && 0 <= v66 && v66 < 1l);
            int v68;
            v68 = 16l * v66;
            int v69;
            v69 = v68 + v1;
            int v70;
            v70 = 256l * v64;
            int v71;
            v71 = v70 + v69;
            float * v72;
            v72 = v0+v71;
            // Pushing the loop unrolling to: 0
            int v74;
            v74 = 0l;
            #pragma unroll
            while (while_method_1(v74)){
                int v76;
                v76 = 0l;
                #pragma unroll
                while (while_method_1(v76)){
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
                v82 = v70 + v1;
                assert("Tensor range check" && 0 <= v80 && v80 < 2l);
                int v83;
                v83 = 8l * v80;
                int v84;
                v84 = v83 + v82;
                float * v85;
                v85 = v4+v84;
                assert("Tensor range check" && 0 <= v66 && v66 < 1l);
                int v87;
                v87 = 256l * v66;
                int v88;
                v88 = v87 + v3;
                assert("Tensor range check" && 0 <= v80 && v80 < 2l);
                int v89;
                v89 = v83 + v88;
                float * v90;
                v90 = v2+v89;
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
                v104 = 16l * v97;
                int v105;
                v105 = v104 + v101;
                float * v106;
                v106 = v13+v103;
                float * v108;
                v108 = v90+v105;
                int v110;
                v110 = 0l;
                #pragma unroll
                while (while_method_1(v110)){
                    int v112;
                    v112 = 0l;
                    #pragma unroll
                    while (while_method_1(v112)){
                        assert("Tensor range check" && 0 <= v110 && v110 < 1l);
                        assert("Tensor range check" && 0 <= v112 && v112 < 1l);
                        int v114;
                        v114 = 8l * v112;
                        int v115;
                        v115 = 192l * v110;
                        int v116;
                        v116 = v115 + v114;
                        int v117;
                        v117 = 256l * v110;
                        int v118;
                        v118 = v117 + v114;
                        float v119[4l];
                        int v120;
                        v120 = 0l;
                        #pragma unroll
                        while (while_method_0(v120)){
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
                v139 = 16l * v132;
                int v140;
                v140 = v139 + v136;
                float * v141;
                v141 = v11+v138;
                float * v143;
                v143 = v85+v140;
                int v145;
                v145 = 0l;
                #pragma unroll
                while (while_method_1(v145)){
                    int v147;
                    v147 = 0l;
                    #pragma unroll
                    while (while_method_1(v147)){
                        assert("Tensor range check" && 0 <= v145 && v145 < 1l);
                        assert("Tensor range check" && 0 <= v147 && v147 < 1l);
                        int v149;
                        v149 = 8l * v147;
                        int v150;
                        v150 = 192l * v145;
                        int v151;
                        v151 = v150 + v149;
                        int v152;
                        v152 = 256l * v145;
                        int v153;
                        v153 = v152 + v149;
                        float v154[4l];
                        int v155;
                        v155 = 0l;
                        #pragma unroll
                        while (while_method_0(v155)){
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
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v162[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v163[1l];
                int v164;
                v164 = 0l;
                #pragma unroll
                while (while_method_1(v164)){
                    int v166;
                    v166 = 0l;
                    #pragma unroll
                    while (while_method_1(v166)){
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
                while (while_method_1(v194)){
                    int v196;
                    v196 = 0l;
                    #pragma unroll
                    while (while_method_1(v196)){
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
                __syncthreads();
                int v224;
                v224 = 0l;
                #pragma unroll
                while (while_method_1(v224)){
                    int v226;
                    v226 = 0l;
                    #pragma unroll
                    while (while_method_1(v226)){
                        int v228;
                        v228 = 0l;
                        #pragma unroll
                        while (while_method_1(v228)){
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
            while (while_method_1(v236)){
                int v238;
                v238 = 0l;
                #pragma unroll
                while (while_method_1(v238)){
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
            __syncthreads();
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
                while (while_method_1(v267)){
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
            __syncthreads();
            // Poping the loop unrolling to: 0
            v66 += 1l ;
        }
        v64 += 1l ;
    }
    return ;
}
__device__ void method_6(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[2560ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v6;
    v6 = 256l * v3;
    float * v7;
    v7 = reinterpret_cast<float *>(&v1[2048ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 4l);
    int v9;
    v9 = 256l * v2;
    float * v10;
    v10 = reinterpret_cast<float *>(&v0[3584ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    return method_7(v10, v6, v7, v9, v4);
}
__device__ void method_8(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[3584ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v6;
    v6 = 256l * v3;
    float * v7;
    v7 = reinterpret_cast<float *>(&v0[4608ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    return method_3(v7, v6, v4);
}
__device__ void method_9(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[4608ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v6;
    v6 = 256l * v3;
    float * v7;
    v7 = reinterpret_cast<float *>(&v0[5632ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    return method_5(v7, v6, v4);
}
__device__ void method_10(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[5632ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v6;
    v6 = 256l * v3;
    float * v7;
    v7 = reinterpret_cast<float *>(&v1[6144ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 4l);
    int v9;
    v9 = 256l * v2;
    float * v10;
    v10 = reinterpret_cast<float *>(&v0[6656ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    return method_7(v10, v6, v7, v9, v4);
}
__device__ void method_12(curandStatePhilox4_32_10_t & v0, int * v1, int v2, float * v3, int v4){
    int v5;
    v5 = threadIdx.x;
    bool v6;
    v6 = 0l <= v5;
    bool v7;
    v7 = v6 == false;
    if (v7){
        assert("The index needs to be zero or positive." && v6);
    } else {
    }
    int v9;
    v9 = v5 % 4l;
    int v10;
    v10 = v5 / 4l;
    bool v11;
    v11 = v10 < 8l;
    bool v12;
    v12 = v11 == false;
    if (v12){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v11);
    } else {
    }
    assert("Tensor range check" && 0 <= v10 && v10 < 8l);
    assert("Tensor range check" && 0 <= v9 && v9 < 4l);
    int v14;
    v14 = 4l * v9;
    int v15;
    v15 = v14 + v4;
    int v16;
    v16 = 16l * v10;
    int v17;
    v17 = v16 + v15;
    assert("Tensor range check" && 0 <= v10 && v10 < 8l);
    int v18;
    v18 = v10 + v2;
    int v19;
    v19 = 0l;
    while (while_method_2(v19)){
        assert("Tensor range check" && 0 <= v19 && v19 < 2l);
        int v21;
        v21 = 128l * v19;
        int v22;
        v22 = v21 + v17;
        float v23[4l];
        int v24[4l];
        int v25;
        v25 = 0l;
        while (while_method_1(v25)){
            assert("Tensor range check" && 0 <= v25 && v25 < 1l);
            int v27;
            v27 = 4l * v25;
            assert("Tensor range check" && 0 <= v25 && v25 < 1l);
            int v28;
            v28 = 16l * v25;
            int v29;
            v29 = v28 + v22;
            int4* v30;
            v30 = reinterpret_cast<int4*>(v3 + v29);
            int4* v31;
            v31 = reinterpret_cast<int4*>(v23 + v27);
            assert("Pointer alignment check" && (unsigned long long)(v30) % 4l == 0 && (unsigned long long)(v31) % 4l == 0);
            *v31 = *v30;
            v25 += 1l ;
        }
        int v32;
        v32 = 0l;
        while (while_method_1(v32)){
            int v34;
            v34 = 0l;
            while (while_method_0(v34)){
                bool v36;
                v36 = 0l <= v34;
                bool v38;
                if (v36){
                    bool v37;
                    v37 = v34 < 4l;
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
                bool v41;
                v41 = 0l <= v9;
                bool v43;
                if (v41){
                    bool v42;
                    v42 = v9 < 4l;
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
                int v46;
                v46 = v9 * 4l;
                int v47;
                v47 = v34 + v46;
                bool v48;
                v48 = 0l <= v32;
                bool v50;
                if (v48){
                    bool v49;
                    v49 = v32 < 1l;
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
                v53 = v32 * 16l;
                int v54;
                v54 = v47 + v53;
                assert("Tensor range check" && 0 <= v32 && v32 < 1l);
                assert("Tensor range check" && 0 <= v34 && v34 < 4l);
                int v55;
                v55 = 4l * v32;
                int v56;
                v56 = v55 + v34;
                v24[v56] = v54;
                v34 += 1l ;
            }
            v32 += 1l ;
        }
        bool v57;
        v57 = 0l <= v10;
        bool v58;
        v58 = v57 && v11;
        bool v59;
        v59 = v58 == false;
        if (v59){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v58);
        } else {
        }
        bool v61;
        v61 = 0l <= v19;
        bool v63;
        if (v61){
            bool v62;
            v62 = v19 < 2l;
            v63 = v62;
        } else {
            v63 = false;
        }
        bool v64;
        v64 = v63 == false;
        if (v64){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v63);
        } else {
        }
        int v66;
        v66 = v19 * 8l;
        int v67;
        v67 = v66 + v10;
        float v68;
        v68 = 0.0f;
        int v69;
        v69 = 0l;
        while (while_method_1(v69)){
            int v71;
            v71 = 0l;
            while (while_method_0(v71)){
                assert("Tensor range check" && 0 <= v69 && v69 < 1l);
                assert("Tensor range check" && 0 <= v71 && v71 < 4l);
                int v73;
                v73 = 4l * v69;
                int v74;
                v74 = v73 + v71;
                float v75;
                v75 = v23[v74];
                float v76;
                v76 = v68 + v75;
                v68 = v76;
                v71 += 1l ;
            }
            v69 += 1l ;
        }
        auto v77 = cooperative_groups::coalesced_threads();
        int v78;
        v78 = threadIdx.x;
        int v79;
        v79 = v78 / 4l;
        auto v80 = cooperative_groups::labeled_partition(v77,v79);
        Closure0 v81{};
        float v82;
        v82 = cooperative_groups::reduce(v80, v68, v81);
        float v83;
        v83 = v82 / 16.0f;
        float v84[4l];
        int v85;
        v85 = 0l;
        while (while_method_1(v85)){
            int v87;
            v87 = 0l;
            while (while_method_0(v87)){
                assert("Tensor range check" && 0 <= v85 && v85 < 1l);
                assert("Tensor range check" && 0 <= v87 && v87 < 4l);
                int v89;
                v89 = 4l * v85;
                int v90;
                v90 = v89 + v87;
                float v91;
                v91 = v23[v90];
                float v92;
                v92 = v91 - v83;
                float v93;
                v93 = exp(v92);
                assert("Tensor range check" && 0 <= v85 && v85 < 1l);
                assert("Tensor range check" && 0 <= v87 && v87 < 4l);
                v84[v90] = v93;
                v87 += 1l ;
            }
            v85 += 1l ;
        }
        float v94;
        v94 = 0.0f;
        int v95;
        v95 = 0l;
        while (while_method_1(v95)){
            int v97;
            v97 = 0l;
            while (while_method_0(v97)){
                assert("Tensor range check" && 0 <= v95 && v95 < 1l);
                assert("Tensor range check" && 0 <= v97 && v97 < 4l);
                int v99;
                v99 = 4l * v95;
                int v100;
                v100 = v99 + v97;
                float v101;
                v101 = v84[v100];
                float v102;
                v102 = v94 + v101;
                v94 = v102;
                v97 += 1l ;
            }
            v95 += 1l ;
        }
        auto v103 = cooperative_groups::coalesced_threads();
        int v104;
        v104 = threadIdx.x;
        int v105;
        v105 = v104 / 4l;
        auto v106 = cooperative_groups::labeled_partition(v103,v105);
        float v107;
        v107 = cooperative_groups::reduce(v106, v94, v81);
        float v108[4l];
        int v109;
        v109 = 0l;
        while (while_method_1(v109)){
            int v111;
            v111 = 0l;
            while (while_method_0(v111)){
                assert("Tensor range check" && 0 <= v109 && v109 < 1l);
                assert("Tensor range check" && 0 <= v111 && v111 < 4l);
                int v113;
                v113 = 4l * v109;
                int v114;
                v114 = v113 + v111;
                float v115;
                v115 = v84[v114];
                bool v116;
                v116 = v107 == 0.0f;
                bool v117;
                v117 = v116 != true;
                float v119;
                if (v117){
                    float v118;
                    v118 = v115 / v107;
                    v119 = v118;
                } else {
                    v119 = 0.0625f;
                }
                assert("Tensor range check" && 0 <= v109 && v109 < 1l);
                assert("Tensor range check" && 0 <= v111 && v111 < 4l);
                v108[v114] = v119;
                v111 += 1l ;
            }
            v109 += 1l ;
        }
        float v120[4l];
        float v121;
        v121 = 0.0f;
        int v122;
        v122 = 0l;
        while (while_method_1(v122)){
            assert("Tensor range check" && 0 <= v122 && v122 < 1l);
            int v124;
            v124 = 4l * v122;
            assert("Tensor range check" && 0 <= v122 && v122 < 1l);
            int v125; float v126;
            Tuple0 tmp0 = Tuple0{0l, 0.0f};
            v125 = tmp0.v0; v126 = tmp0.v1;
            while (while_method_0(v125)){
                assert("Tensor range check" && 0 <= v125 && v125 < 4l);
                int v128;
                v128 = v125 + v124;
                float v129;
                v129 = v108[v128];
                float v130;
                v130 = v126 + v129;
                v126 = v130;
                v125 += 1l ;
            }
            auto v131 = cooperative_groups::coalesced_threads();
            int v132;
            v132 = threadIdx.x;
            int v133;
            v133 = v132 / 4l;
            auto v134 = cooperative_groups::labeled_partition(v131,v133);
            Closure1 v135{};
            float v136;
            v136 = cooperative_groups::inclusive_scan(v134, v126, v135);
            float v137;
            v137 = v134.shfl_up(v136,1);
            bool v138;
            v138 = v134.thread_rank() == 0;
            float v139;
            if (v138){
                v139 = 0.0f;
            } else {
                v139 = v137;
            }
            float v140;
            v140 = v134.shfl(v136,v134.num_threads()-1);
            float v141;
            v141 = v121 + v139;
            int v142; float v143;
            Tuple0 tmp1 = Tuple0{0l, v141};
            v142 = tmp1.v0; v143 = tmp1.v1;
            while (while_method_0(v142)){
                assert("Tensor range check" && 0 <= v142 && v142 < 4l);
                int v145;
                v145 = v142 + v124;
                float v146;
                v146 = v108[v145];
                float v147;
                v147 = v143 + v146;
                assert("Tensor range check" && 0 <= v142 && v142 < 4l);
                v120[v145] = v147;
                v143 = v147;
                v142 += 1l ;
            }
            float v148;
            v148 = v121 + v140;
            v121 = v148;
            v122 += 1l ;
        }
        float v149;
        v149 = curand_uniform(&v0);
        float v150[4l];
        int v151;
        v151 = 0l;
        while (while_method_1(v151)){
            int v153;
            v153 = 0l;
            while (while_method_0(v153)){
                assert("Tensor range check" && 0 <= v151 && v151 < 1l);
                assert("Tensor range check" && 0 <= v153 && v153 < 4l);
                int v155;
                v155 = 4l * v151;
                int v156;
                v156 = v155 + v153;
                float v157;
                v157 = v120[v156];
                float v158;
                v158 = v157 - v149;
                assert("Tensor range check" && 0 <= v151 && v151 < 1l);
                assert("Tensor range check" && 0 <= v153 && v153 < 4l);
                v150[v156] = v158;
                v153 += 1l ;
            }
            v151 += 1l ;
        }
        float v159; int v160;
        Tuple1 tmp2 = Tuple1{-1.0f / 0.0f, 0l};
        v159 = tmp2.v0; v160 = tmp2.v1;
        int v161;
        v161 = 0l;
        while (while_method_1(v161)){
            int v163;
            v163 = 0l;
            while (while_method_0(v163)){
                assert("Tensor range check" && 0 <= v161 && v161 < 1l);
                assert("Tensor range check" && 0 <= v163 && v163 < 4l);
                int v165;
                v165 = 4l * v161;
                int v166;
                v166 = v165 + v163;
                float v167;
                v167 = v150[v166];
                int v168;
                v168 = v24[v166];
                bool v169;
                v169 = v159 >= 0.0f;
                bool v171;
                if (v169){
                    bool v170;
                    v170 = v167 >= 0.0f;
                    v171 = v170;
                } else {
                    v171 = false;
                }
                float v180; int v181;
                if (v171){
                    bool v172;
                    v172 = v159 <= v167;
                    if (v172){
                        v180 = v159; v181 = v160;
                    } else {
                        v180 = v167; v181 = v168;
                    }
                } else {
                    if (v169){
                        v180 = v159; v181 = v160;
                    } else {
                        bool v175;
                        v175 = v167 >= 0.0f;
                        if (v175){
                            v180 = v167; v181 = v168;
                        } else {
                            v180 = v159; v181 = v160;
                        }
                    }
                }
                v159 = v180;
                v160 = v181;
                v163 += 1l ;
            }
            v161 += 1l ;
        }
        auto v182 = cooperative_groups::coalesced_threads();
        int v183;
        v183 = threadIdx.x;
        int v184;
        v184 = v183 / 4l;
        auto v185 = cooperative_groups::labeled_partition(v182,v184);
        Closure2 v186{};
        float v187; int v188;
        Tuple1 tmp3 = cooperative_groups::reduce(v185, Tuple1{v159, v160}, v186);
        v187 = tmp3.v0; v188 = tmp3.v1;
        assert("Tensor range check" && 0 <= v19 && v19 < 2l);
        int v189;
        v189 = 8l * v19;
        int v190;
        v190 = v189 + v18;
        v1[v190] = v188;
        v19 += 1l ;
    }
    __syncthreads();
    return ;
}
__device__ void method_11(unsigned char * v0, unsigned char * v1, int v2, int v3, curandStatePhilox4_32_10_t & v4){
    float * v5;
    v5 = reinterpret_cast<float *>(&v0[6656ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v7;
    v7 = 256l * v3;
    int * v8;
    v8 = reinterpret_cast<int *>(&v0[7680ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 4l);
    int v10;
    v10 = 16l * v2;
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v11;
    v11 = 16l * v3;
    int v12;
    v12 = v11 + v10;
    return method_12(v4, v8, v12, v5, v7);
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1) {
    unsigned long long v2;
    v2 = clock64();
    int v3;
    v3 = threadIdx.x;
    int v4;
    v4 = blockIdx.x;
    int v5;
    v5 = v4 * 32l;
    int v6;
    v6 = v3 + v5;
    unsigned long long v7;
    v7 = (unsigned long long)v6;
    curandStatePhilox4_32_10_t v8;
    curand_init(v2,v7,0ull,&v8);
    int v9;
    v9 = 0l;
    while (while_method_0(v9)){
        int v11;
        v11 = blockIdx.x;
        int v12;
        v12 = v11;
        while (while_method_1(v12)){
            bool v14;
            v14 = 0l <= v12;
            bool v15;
            v15 = v14 == false;
            if (v15){
                assert("The index needs to be zero or positive." && v14);
            } else {
            }
            bool v17;
            v17 = v12 < 1l;
            bool v18;
            v18 = v17 == false;
            if (v18){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v17);
            } else {
            }
            method_0(v0, v1, v9, v12);
            method_2(v0, v1, v9, v12);
            method_4(v0, v1, v9, v12);
            method_6(v0, v1, v9, v12);
            method_8(v0, v1, v9, v12);
            method_9(v0, v1, v9, v12);
            method_10(v0, v1, v9, v12);
            method_11(v0, v1, v9, v12, v8);
            v12 += 1l ;
        }
        v9 += 1l ;
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
def method4(v0 : f32) -> None:
    print("{:.6f}".format(v0), end="")
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
def method5(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method7(v0 : i32) -> None:
    print(v0, end="")
    del v0
    return 
def method6(v0 : cp.ndarray, v1 : i32, v2 : i32, v3 : i32, v4 : i32, v5 : i32) -> None:
    v6 = 0
    v7 = '['
    method2(v7)
    del v7
    v8 = 0
    while method3(v4, v8):
        v10 = v6
        v11 = v10 >= 100
        del v10
        if v11:
            v12 = " ..."
            method0(v12)
            del v12
            break
        else:
            pass
        del v11
        v13 = v8 == 0
        v14 = v13 != True
        del v13
        if v14:
            v15 = "; "
            method0(v15)
        else:
            pass
        del v14
        v16 = '['
        method2(v16)
        del v16
        v17 = 0
        while method3(v5, v17):
            v19 = v6
            v20 = v19 >= 100
            del v19
            if v20:
                v21 = " ..."
                method0(v21)
                del v21
                break
            else:
                pass
            del v20
            v22 = v17 == 0
            v23 = v22 != True
            del v22
            if v23:
                v24 = "; "
                method0(v24)
            else:
                pass
            del v23
            v25 = v6 + 1
            v6 = v25
            del v25
            v26 = v8 * v2
            v27 = v1 + v26
            del v26
            v28 = v17 * v3
            v29 = v27 + v28
            del v27, v28
            v30 = v0[v29].item()
            del v29
            method7(v30)
            del v30
            v17 += 1 
        del v17
        v31 = ']'
        method2(v31)
        del v31
        v8 += 1 
    del v0, v1, v2, v3, v4, v5, v6, v8
    v32 = ']'
    return method2(v32)
def main():
    v0 = cp.empty(10240,dtype=cp.uint8)
    v1 = cp.empty(7936,dtype=cp.uint8)
    v3 = v0[0:0+4*512].view(cp.float32)
    v4 = cp.random.normal(0.0,1.0,512,dtype=cp.float32) # type: ignore
    cp.copyto(v3[0:0+512],v4[0:0+512])
    del v3, v4
    v6 = v0[2048:2048+4*1024].view(cp.float32)
    v7 = cp.random.normal(0.0,1.0,1024,dtype=cp.float32) # type: ignore
    cp.copyto(v6[0:0+1024],v7[0:0+1024])
    del v6, v7
    v9 = v0[6144:6144+4*1024].view(cp.float32)
    v10 = cp.random.normal(0.0,1.0,1024,dtype=cp.float32) # type: ignore
    cp.copyto(v9[0:0+1024],v10[0:0+1024])
    del v9, v10
    v11 = "Here are the weight matrices."
    method0(v11)
    del v11
    print()
    v13 = v0[0:0+4*512].view(cp.float32)
    v14 = 0
    v15 = 128
    v16 = 8
    v17 = 1
    v18 = 4
    v19 = 16
    v20 = 8
    method1(v13, v14, v15, v16, v17, v18, v19, v20)
    del v13, v14, v15, v16, v17, v18, v19, v20
    print()
    v22 = v0[2048:2048+4*1024].view(cp.float32)
    v23 = 0
    v24 = 256
    v25 = 16
    v26 = 1
    v27 = 4
    v28 = 16
    v29 = 16
    method1(v22, v23, v24, v25, v26, v27, v28, v29)
    del v22, v23, v24, v25, v26, v27, v28, v29
    print()
    v31 = v0[6144:6144+4*1024].view(cp.float32)
    v32 = 0
    v33 = 256
    v34 = 16
    v35 = 1
    v36 = 4
    v37 = 16
    v38 = 16
    method1(v31, v32, v33, v34, v35, v36, v37, v38)
    del v31, v32, v33, v34, v35, v36, v37, v38
    print()
    v40 = v1[0:0+4*128].view(cp.float32)
    v41 = cp.random.normal(0.0,1.0,128,dtype=cp.float32) # type: ignore
    cp.copyto(v40[0:0+128],v41[0:0+128])
    del v41
    v42 = 0
    v43 = 128
    v44 = 8
    v45 = 1
    v46 = 1
    v47 = 16
    v48 = 8
    method1(v40, v42, v43, v44, v45, v46, v47, v48)
    del v40, v42, v43, v44, v45, v46, v47, v48
    print()
    v49 = "Here is the output tensor."
    method0(v49)
    del v49
    print()
    v50 = 0
    v51 = raw_module.get_function(f"entry{v50}")
    del v50
    v51.max_dynamic_shared_size_bytes = 1536 
    v51((1,),(32,),(v1, v0),shared_mem=1536)
    del v0, v51
    v52 = 0
    while method5(v52):
        v55 = v1[7680:7680+4*64].view(cp.int32)
        assert 0 <= v52 < 4, 'Tensor range check'
        v56 = 16 * v52
        v57 = 1
        v58 = 1
        v59 = 16
        v60 = 1
        method6(v55, v56, v57, v58, v59, v60)
        del v55, v56, v57, v58, v59, v60
        print()
        v61 = "==="
        method0(v61)
        del v61
        print()
        v52 += 1 
    del v1, v52
    return 

if __name__ == '__main__': print(main())
