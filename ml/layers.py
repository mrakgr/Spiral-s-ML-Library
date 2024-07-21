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

__device__ void method_0(float * v0, int v1, float * v2, int v3, float * v4, int v5);
__device__ void method_1(float * v0, float * v1);
__device__ void method_2(float * v0, float * v1);
__device__ void method_3(float * v0, int v1, float * v2, int v3, float * v4, int v5);
__device__ void method_4(float * v0, int v1, float * v2);
struct Tuple0;
struct Tuple1;
__device__ void method_5(int * v0, int v1, float * v2, int v3, curandStatePhilox4_32_10_t & v4);
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
    v1 = v0 < 16l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
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
    while (while_method_1(v65)){
        int v67;
        v67 = 0l;
        while (while_method_1(v67)){
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
                        while (while_method_2(v122)){
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
                        while (while_method_2(v157)){
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
                        while (while_method_3(v175)){
                            int v177;
                            v177 = 0l;
                            #pragma unroll
                            while (while_method_3(v177)){
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
                        while (while_method_3(v205)){
                            int v207;
                            v207 = 0l;
                            #pragma unroll
                            while (while_method_3(v207)){
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
            while (while_method_3(v267)){
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
__device__ void method_1(float * v0, float * v1){
    int v2;
    v2 = blockIdx.x;
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    int v3;
    v3 = 256l * v2;
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 1l);
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
    while (while_method_3(v21)){
        assert("Tensor range check" && 0 <= v21 && v21 < 2l);
        int v23;
        v23 = 128l * v21;
        int v24;
        v24 = v23 + v18;
        assert("Tensor range check" && 0 <= v21 && v21 < 2l);
        int v25;
        v25 = v23 + v20;
        float v26[4l];
        int v27[4l];
        int v28;
        v28 = 0l;
        while (while_method_1(v28)){
            assert("Tensor range check" && 0 <= v28 && v28 < 1l);
            int v30;
            v30 = 4l * v28;
            assert("Tensor range check" && 0 <= v28 && v28 < 1l);
            int v31;
            v31 = 16l * v28;
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
        while (while_method_1(v35)){
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
                    v45 = v10 < 4l;
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
                v56 = v35 * 16l;
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
            v65 = v21 < 2l;
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
        v69 = v21 * 8l;
        int v70;
        v70 = v69 + v11;
        float v71[4l];
        int v72;
        v72 = 0l;
        while (while_method_1(v72)){
            int v74;
            v74 = 0l;
            while (while_method_2(v74)){
                assert("Tensor range check" && 0 <= v72 && v72 < 1l);
                assert("Tensor range check" && 0 <= v74 && v74 < 4l);
                int v76;
                v76 = 4l * v72;
                int v77;
                v77 = v76 + v74;
                float v78;
                v78 = v26[v77];
                float v79;
                v79 = v78 * v78;
                assert("Tensor range check" && 0 <= v72 && v72 < 1l);
                assert("Tensor range check" && 0 <= v74 && v74 < 4l);
                v71[v77] = v79;
                v74 += 1l ;
            }
            v72 += 1l ;
        }
        float v80;
        v80 = 0.0f;
        int v81;
        v81 = 0l;
        while (while_method_1(v81)){
            int v83;
            v83 = 0l;
            while (while_method_2(v83)){
                assert("Tensor range check" && 0 <= v81 && v81 < 1l);
                assert("Tensor range check" && 0 <= v83 && v83 < 4l);
                int v85;
                v85 = 4l * v81;
                int v86;
                v86 = v85 + v83;
                float v87;
                v87 = v71[v86];
                float v88;
                v88 = v80 + v87;
                v80 = v88;
                v83 += 1l ;
            }
            v81 += 1l ;
        }
        auto v89 = cooperative_groups::coalesced_threads();
        int v90;
        v90 = threadIdx.x;
        int v91;
        v91 = v90 / 4l;
        auto v92 = cooperative_groups::labeled_partition(v89,v91);
        Closure0 v93{};
        float v94;
        v94 = cooperative_groups::reduce(v92, v80, v93);
        float v95[4l];
        int v96;
        v96 = 0l;
        while (while_method_1(v96)){
            int v98;
            v98 = 0l;
            while (while_method_2(v98)){
                assert("Tensor range check" && 0 <= v96 && v96 < 1l);
                assert("Tensor range check" && 0 <= v98 && v98 < 4l);
                int v100;
                v100 = 4l * v96;
                int v101;
                v101 = v100 + v98;
                float v102;
                v102 = v26[v101];
                bool v103;
                v103 = v94 == 0.0f;
                bool v104;
                v104 = v103 != true;
                float v106;
                if (v104){
                    float v105;
                    v105 = v102 / v94;
                    v106 = v105;
                } else {
                    v106 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v96 && v96 < 1l);
                assert("Tensor range check" && 0 <= v98 && v98 < 4l);
                v95[v101] = v106;
                v98 += 1l ;
            }
            v96 += 1l ;
        }
        int v107;
        v107 = 0l;
        while (while_method_1(v107)){
            assert("Tensor range check" && 0 <= v107 && v107 < 1l);
            int v109;
            v109 = 16l * v107;
            int v110;
            v110 = v109 + v25;
            assert("Tensor range check" && 0 <= v107 && v107 < 1l);
            int v111;
            v111 = 4l * v107;
            int4* v112;
            v112 = reinterpret_cast<int4*>(v95 + v111);
            int4* v113;
            v113 = reinterpret_cast<int4*>(v0 + v110);
            assert("Pointer alignment check" && (unsigned long long)(v112) % 4l == 0 && (unsigned long long)(v113) % 4l == 0);
            *v113 = *v112;
            v107 += 1l ;
        }
        v21 += 1l ;
    }
    __syncthreads();
    return ;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 64l;
    return v1;
}
__device__ void method_2(float * v0, float * v1){
    int v2;
    v2 = blockIdx.x;
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    int v3;
    v3 = 256l * v2;
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 1l);
    int v5;
    v5 = 256l * v4;
    int v6;
    v6 = threadIdx.x;
    int v7;
    v7 = v6;
    while (while_method_4(v7)){
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
__device__ void method_3(float * v0, int v1, float * v2, int v3, float * v4, int v5){
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
            while (while_method_3(v81)){
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
                while (while_method_1(v111)){
                    int v113;
                    v113 = 0l;
                    #pragma unroll
                    while (while_method_1(v113)){
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
                while (while_method_1(v146)){
                    int v148;
                    v148 = 0l;
                    #pragma unroll
                    while (while_method_1(v148)){
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
                while (while_method_1(v165)){
                    int v167;
                    v167 = 0l;
                    #pragma unroll
                    while (while_method_1(v167)){
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
                while (while_method_1(v195)){
                    int v197;
                    v197 = 0l;
                    #pragma unroll
                    while (while_method_1(v197)){
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
                while (while_method_1(v225)){
                    int v227;
                    v227 = 0l;
                    #pragma unroll
                    while (while_method_1(v227)){
                        int v229;
                        v229 = 0l;
                        #pragma unroll
                        while (while_method_1(v229)){
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
            while (while_method_1(v237)){
                int v239;
                v239 = 0l;
                #pragma unroll
                while (while_method_1(v239)){
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
            while (while_method_3(v266)){
                int v268;
                v268 = 0l;
                #pragma unroll
                while (while_method_1(v268)){
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
            __syncthreads();
            // Poping the loop unrolling to: 0
            v67 += 1l ;
        }
        v65 += 1l ;
    }
    return ;
}
__device__ void method_4(float * v0, int v1, float * v2){
    int v3;
    v3 = blockIdx.x;
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v4;
    v4 = 256l * v3;
    int v5;
    v5 = blockIdx.x;
    assert("Tensor range check" && 0 <= v5 && v5 < 1l);
    int v6;
    v6 = 256l * v5;
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
    v12 = v8 % 4l;
    int v13;
    v13 = v8 / 4l;
    bool v14;
    v14 = v13 < 8l;
    bool v15;
    v15 = v14 == false;
    if (v15){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v14);
    } else {
    }
    assert("Tensor range check" && 0 <= v13 && v13 < 8l);
    assert("Tensor range check" && 0 <= v12 && v12 < 4l);
    int v17;
    v17 = 4l * v12;
    int v18;
    v18 = v17 + v4;
    int v19;
    v19 = 16l * v13;
    int v20;
    v20 = v19 + v18;
    assert("Tensor range check" && 0 <= v13 && v13 < 8l);
    assert("Tensor range check" && 0 <= v12 && v12 < 4l);
    int v21;
    v21 = v17 + v7;
    int v22;
    v22 = v19 + v21;
    int v23;
    v23 = 0l;
    while (while_method_3(v23)){
        assert("Tensor range check" && 0 <= v23 && v23 < 2l);
        int v25;
        v25 = 128l * v23;
        int v26;
        v26 = v25 + v20;
        assert("Tensor range check" && 0 <= v23 && v23 < 2l);
        int v27;
        v27 = v25 + v22;
        float v28[4l];
        int v29[4l];
        int v30;
        v30 = 0l;
        while (while_method_1(v30)){
            assert("Tensor range check" && 0 <= v30 && v30 < 1l);
            int v32;
            v32 = 4l * v30;
            assert("Tensor range check" && 0 <= v30 && v30 < 1l);
            int v33;
            v33 = 16l * v30;
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
        while (while_method_1(v37)){
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
                    v47 = v12 < 4l;
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
                v58 = v37 * 16l;
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
            v67 = v23 < 2l;
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
        v71 = v23 * 8l;
        int v72;
        v72 = v71 + v13;
        float v73;
        v73 = 0.0f;
        int v74;
        v74 = 0l;
        while (while_method_1(v74)){
            int v76;
            v76 = 0l;
            while (while_method_2(v76)){
                assert("Tensor range check" && 0 <= v74 && v74 < 1l);
                assert("Tensor range check" && 0 <= v76 && v76 < 4l);
                int v78;
                v78 = 4l * v74;
                int v79;
                v79 = v78 + v76;
                float v80;
                v80 = v28[v79];
                float v81;
                v81 = v73 + v80;
                v73 = v81;
                v76 += 1l ;
            }
            v74 += 1l ;
        }
        auto v82 = cooperative_groups::coalesced_threads();
        int v83;
        v83 = threadIdx.x;
        int v84;
        v84 = v83 / 4l;
        auto v85 = cooperative_groups::labeled_partition(v82,v84);
        Closure0 v86{};
        float v87;
        v87 = cooperative_groups::reduce(v85, v73, v86);
        float v88;
        v88 = v87 / 16.0f;
        float v89[4l];
        int v90;
        v90 = 0l;
        while (while_method_1(v90)){
            int v92;
            v92 = 0l;
            while (while_method_2(v92)){
                assert("Tensor range check" && 0 <= v90 && v90 < 1l);
                assert("Tensor range check" && 0 <= v92 && v92 < 4l);
                int v94;
                v94 = 4l * v90;
                int v95;
                v95 = v94 + v92;
                float v96;
                v96 = v28[v95];
                float v97;
                v97 = v96 - v88;
                float v98;
                v98 = exp(v97);
                assert("Tensor range check" && 0 <= v90 && v90 < 1l);
                assert("Tensor range check" && 0 <= v92 && v92 < 4l);
                v89[v95] = v98;
                v92 += 1l ;
            }
            v90 += 1l ;
        }
        float v99;
        v99 = 0.0f;
        int v100;
        v100 = 0l;
        while (while_method_1(v100)){
            int v102;
            v102 = 0l;
            while (while_method_2(v102)){
                assert("Tensor range check" && 0 <= v100 && v100 < 1l);
                assert("Tensor range check" && 0 <= v102 && v102 < 4l);
                int v104;
                v104 = 4l * v100;
                int v105;
                v105 = v104 + v102;
                float v106;
                v106 = v89[v105];
                float v107;
                v107 = v99 + v106;
                v99 = v107;
                v102 += 1l ;
            }
            v100 += 1l ;
        }
        auto v108 = cooperative_groups::coalesced_threads();
        int v109;
        v109 = threadIdx.x;
        int v110;
        v110 = v109 / 4l;
        auto v111 = cooperative_groups::labeled_partition(v108,v110);
        float v112;
        v112 = cooperative_groups::reduce(v111, v99, v86);
        float v113[4l];
        int v114;
        v114 = 0l;
        while (while_method_1(v114)){
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
                v120 = v89[v119];
                bool v121;
                v121 = v112 == 0.0f;
                bool v122;
                v122 = v121 != true;
                float v124;
                if (v122){
                    float v123;
                    v123 = v120 / v112;
                    v124 = v123;
                } else {
                    v124 = 0.0625f;
                }
                assert("Tensor range check" && 0 <= v114 && v114 < 1l);
                assert("Tensor range check" && 0 <= v116 && v116 < 4l);
                v113[v119] = v124;
                v116 += 1l ;
            }
            v114 += 1l ;
        }
        int v125;
        v125 = 0l;
        while (while_method_1(v125)){
            assert("Tensor range check" && 0 <= v125 && v125 < 1l);
            int v127;
            v127 = 16l * v125;
            int v128;
            v128 = v127 + v27;
            assert("Tensor range check" && 0 <= v125 && v125 < 1l);
            int v129;
            v129 = 4l * v125;
            int4* v130;
            v130 = reinterpret_cast<int4*>(v113 + v129);
            int4* v131;
            v131 = reinterpret_cast<int4*>(v0 + v128);
            assert("Pointer alignment check" && (unsigned long long)(v130) % 4l == 0 && (unsigned long long)(v131) % 4l == 0);
            *v131 = *v130;
            v125 += 1l ;
        }
        v23 += 1l ;
    }
    __syncthreads();
    return ;
}
__device__ void method_5(int * v0, int v1, float * v2, int v3, curandStatePhilox4_32_10_t & v4){
    int v5;
    v5 = blockIdx.x;
    assert("Tensor range check" && 0 <= v5 && v5 < 1l);
    int v6;
    v6 = 256l * v5;
    int v7;
    v7 = v6 + v3;
    int v8;
    v8 = blockIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 1l);
    int v9;
    v9 = 16l * v8;
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
    v15 = v11 % 4l;
    int v16;
    v16 = v11 / 4l;
    bool v17;
    v17 = v16 < 8l;
    bool v18;
    v18 = v17 == false;
    if (v18){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v17);
    } else {
    }
    assert("Tensor range check" && 0 <= v16 && v16 < 8l);
    assert("Tensor range check" && 0 <= v15 && v15 < 4l);
    int v20;
    v20 = 4l * v15;
    int v21;
    v21 = v20 + v7;
    int v22;
    v22 = 16l * v16;
    int v23;
    v23 = v22 + v21;
    assert("Tensor range check" && 0 <= v16 && v16 < 8l);
    int v24;
    v24 = v16 + v10;
    int v25;
    v25 = 0l;
    while (while_method_3(v25)){
        assert("Tensor range check" && 0 <= v25 && v25 < 2l);
        int v27;
        v27 = 128l * v25;
        int v28;
        v28 = v27 + v23;
        float v29[4l];
        int v30[4l];
        int v31;
        v31 = 0l;
        while (while_method_1(v31)){
            assert("Tensor range check" && 0 <= v31 && v31 < 1l);
            int v33;
            v33 = 4l * v31;
            assert("Tensor range check" && 0 <= v31 && v31 < 1l);
            int v34;
            v34 = 16l * v31;
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
        while (while_method_1(v38)){
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
                    v48 = v15 < 4l;
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
                v59 = v38 * 16l;
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
            v68 = v25 < 2l;
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
        v72 = v25 * 8l;
        int v73;
        v73 = v72 + v16;
        float v74[4l];
        float v75;
        v75 = 0.0f;
        int v76;
        v76 = 0l;
        while (while_method_1(v76)){
            assert("Tensor range check" && 0 <= v76 && v76 < 1l);
            int v78;
            v78 = 4l * v76;
            assert("Tensor range check" && 0 <= v76 && v76 < 1l);
            int v79; float v80;
            Tuple0 tmp0 = Tuple0{0l, 0.0f};
            v79 = tmp0.v0; v80 = tmp0.v1;
            while (while_method_2(v79)){
                assert("Tensor range check" && 0 <= v79 && v79 < 4l);
                int v82;
                v82 = v79 + v78;
                float v83;
                v83 = v29[v82];
                float v84;
                v84 = v80 + v83;
                v80 = v84;
                v79 += 1l ;
            }
            auto v85 = cooperative_groups::coalesced_threads();
            int v86;
            v86 = threadIdx.x;
            int v87;
            v87 = v86 / 4l;
            auto v88 = cooperative_groups::labeled_partition(v85,v87);
            Closure1 v89{};
            float v90;
            v90 = cooperative_groups::inclusive_scan(v88, v80, v89);
            float v91;
            v91 = v88.shfl_up(v90,1);
            bool v92;
            v92 = v88.thread_rank() == 0;
            float v93;
            if (v92){
                v93 = 0.0f;
            } else {
                v93 = v91;
            }
            float v94;
            v94 = v88.shfl(v90,v88.num_threads()-1);
            float v95;
            v95 = v75 + v93;
            int v96; float v97;
            Tuple0 tmp1 = Tuple0{0l, v95};
            v96 = tmp1.v0; v97 = tmp1.v1;
            while (while_method_2(v96)){
                assert("Tensor range check" && 0 <= v96 && v96 < 4l);
                int v99;
                v99 = v96 + v78;
                float v100;
                v100 = v29[v99];
                float v101;
                v101 = v97 + v100;
                assert("Tensor range check" && 0 <= v96 && v96 < 4l);
                v74[v99] = v101;
                v97 = v101;
                v96 += 1l ;
            }
            float v102;
            v102 = v75 + v94;
            v75 = v102;
            v76 += 1l ;
        }
        float v103;
        v103 = curand_uniform(&v4);
        float v104[4l];
        int v105;
        v105 = 0l;
        while (while_method_1(v105)){
            int v107;
            v107 = 0l;
            while (while_method_2(v107)){
                assert("Tensor range check" && 0 <= v105 && v105 < 1l);
                assert("Tensor range check" && 0 <= v107 && v107 < 4l);
                int v109;
                v109 = 4l * v105;
                int v110;
                v110 = v109 + v107;
                float v111;
                v111 = v74[v110];
                float v112;
                v112 = v111 - v103;
                assert("Tensor range check" && 0 <= v105 && v105 < 1l);
                assert("Tensor range check" && 0 <= v107 && v107 < 4l);
                v104[v110] = v112;
                v107 += 1l ;
            }
            v105 += 1l ;
        }
        float v113; int v114;
        Tuple1 tmp2 = Tuple1{-1.0f / 0.0f, 0l};
        v113 = tmp2.v0; v114 = tmp2.v1;
        int v115;
        v115 = 0l;
        while (while_method_1(v115)){
            int v117;
            v117 = 0l;
            while (while_method_2(v117)){
                assert("Tensor range check" && 0 <= v115 && v115 < 1l);
                assert("Tensor range check" && 0 <= v117 && v117 < 4l);
                int v119;
                v119 = 4l * v115;
                int v120;
                v120 = v119 + v117;
                float v121;
                v121 = v104[v120];
                int v122;
                v122 = v30[v120];
                bool v123;
                v123 = v113 >= 0.0f;
                bool v125;
                if (v123){
                    bool v124;
                    v124 = v121 >= 0.0f;
                    v125 = v124;
                } else {
                    v125 = false;
                }
                float v134; int v135;
                if (v125){
                    bool v126;
                    v126 = v113 <= v121;
                    if (v126){
                        v134 = v113; v135 = v114;
                    } else {
                        v134 = v121; v135 = v122;
                    }
                } else {
                    if (v123){
                        v134 = v113; v135 = v114;
                    } else {
                        bool v129;
                        v129 = v121 >= 0.0f;
                        if (v129){
                            v134 = v121; v135 = v122;
                        } else {
                            v134 = v113; v135 = v114;
                        }
                    }
                }
                v113 = v134;
                v114 = v135;
                v117 += 1l ;
            }
            v115 += 1l ;
        }
        auto v136 = cooperative_groups::coalesced_threads();
        int v137;
        v137 = threadIdx.x;
        int v138;
        v138 = v137 / 4l;
        auto v139 = cooperative_groups::labeled_partition(v136,v138);
        Closure2 v140{};
        float v141; int v142;
        Tuple1 tmp3 = cooperative_groups::reduce(v139, Tuple1{v113, v114}, v140);
        v141 = tmp3.v0; v142 = tmp3.v1;
        assert("Tensor range check" && 0 <= v25 && v25 < 2l);
        int v143;
        v143 = 8l * v25;
        int v144;
        v144 = v143 + v24;
        v0[v144] = v142;
        v25 += 1l ;
    }
    __syncthreads();
    return ;
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1) {
    int v2;
    v2 = 0l;
    while (while_method_0(v2)){
        unsigned long long v4;
        v4 = clock64();
        int v5;
        v5 = threadIdx.x;
        int v6;
        v6 = blockIdx.x;
        int v7;
        v7 = v6 * 32l;
        int v8;
        v8 = v5 + v7;
        unsigned long long v9;
        v9 = (unsigned long long)v8;
        curandStatePhilox4_32_10_t v10;
        curand_init(v4,v9,0ull,&v10);
        float * v11;
        v11 = reinterpret_cast<float *>(&v0[0ull]);
        float * v13;
        v13 = reinterpret_cast<float *>(&v1[0ull]);
        assert("Tensor range check" && 0 <= v2 && v2 < 16l);
        int v15;
        v15 = 128l * v2;
        float * v16;
        v16 = reinterpret_cast<float *>(&v0[512ull]);
        int v18;
        v18 = blockIdx.x;
        assert("Tensor range check" && 0 <= v18 && v18 < 1l);
        int v19;
        v19 = 128l * v18;
        int v20;
        v20 = blockIdx.x;
        assert("Tensor range check" && 0 <= v20 && v20 < 1l);
        int v21;
        v21 = 256l * v20;
        method_0(v13, v15, v16, v21, v11, v19);
        float * v22;
        v22 = reinterpret_cast<float *>(&v0[1536ull]);
        method_1(v22, v16);
        float * v24;
        v24 = reinterpret_cast<float *>(&v0[2560ull]);
        method_2(v24, v22);
        float * v26;
        v26 = reinterpret_cast<float *>(&v1[8192ull]);
        assert("Tensor range check" && 0 <= v2 && v2 < 16l);
        int v28;
        v28 = 256l * v2;
        float * v29;
        v29 = reinterpret_cast<float *>(&v0[3584ull]);
        int v31;
        v31 = blockIdx.x;
        assert("Tensor range check" && 0 <= v31 && v31 < 1l);
        int v32;
        v32 = 256l * v31;
        int v33;
        v33 = blockIdx.x;
        assert("Tensor range check" && 0 <= v33 && v33 < 1l);
        int v34;
        v34 = 256l * v33;
        method_3(v26, v28, v29, v34, v24, v32);
        float * v35;
        v35 = reinterpret_cast<float *>(&v0[4608ull]);
        method_1(v35, v29);
        float * v37;
        v37 = reinterpret_cast<float *>(&v0[5632ull]);
        method_2(v37, v35);
        float * v39;
        v39 = reinterpret_cast<float *>(&v1[24576ull]);
        assert("Tensor range check" && 0 <= v2 && v2 < 16l);
        float * v41;
        v41 = reinterpret_cast<float *>(&v0[6656ull]);
        int v43;
        v43 = blockIdx.x;
        assert("Tensor range check" && 0 <= v43 && v43 < 1l);
        int v44;
        v44 = 256l * v43;
        int v45;
        v45 = blockIdx.x;
        assert("Tensor range check" && 0 <= v45 && v45 < 1l);
        int v46;
        v46 = 256l * v45;
        method_3(v39, v28, v41, v46, v37, v44);
        float * v47;
        v47 = reinterpret_cast<float *>(&v0[7680ull]);
        assert("Tensor range check" && 0 <= v2 && v2 < 16l);
        method_4(v47, v28, v41);
        int * v49;
        v49 = reinterpret_cast<int *>(&v0[24064ull]);
        assert("Tensor range check" && 0 <= v2 && v2 < 16l);
        int v51;
        v51 = 16l * v2;
        method_5(v49, v51, v47, v28, v10);
        v2 += 1l ;
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
    v1 = v0 < 16
    del v0
    return v1
def method6(v0 : i32) -> bool:
    v1 = v0 < 1
    del v0
    return v1
def method7() -> None:
    return 
def method8(v0 : i32) -> None:
    print(v0, end="")
    del v0
    return 
def main():
    v0 = cp.empty(40960,dtype=cp.uint8)
    v1 = cp.empty(25088,dtype=cp.uint8)
    v3 = v0[0:0+4*2048].view(cp.float32)
    v4 = cp.random.normal(0.0,1.0,2048,dtype=cp.float32) # type: ignore
    cp.copyto(v3[0:0+2048],v4[0:0+2048])
    del v3, v4
    v6 = v0[8192:8192+4*4096].view(cp.float32)
    v7 = cp.random.normal(0.0,1.0,4096,dtype=cp.float32) # type: ignore
    cp.copyto(v6[0:0+4096],v7[0:0+4096])
    del v6, v7
    v9 = v0[24576:24576+4*4096].view(cp.float32)
    v10 = cp.random.normal(0.0,1.0,4096,dtype=cp.float32) # type: ignore
    cp.copyto(v9[0:0+4096],v10[0:0+4096])
    del v9, v10
    v11 = "Here are the weight matrices."
    method0(v11)
    del v11
    print()
    v13 = v0[0:0+4*2048].view(cp.float32)
    v14 = 0
    v15 = 128
    v16 = 8
    v17 = 1
    v18 = 16
    v19 = 16
    v20 = 8
    method1(v13, v14, v15, v16, v17, v18, v19, v20)
    del v13, v14, v15, v16, v17, v18, v19, v20
    print()
    v22 = v0[8192:8192+4*4096].view(cp.float32)
    v23 = 0
    v24 = 256
    v25 = 16
    v26 = 1
    v27 = 16
    v28 = 16
    v29 = 16
    method1(v22, v23, v24, v25, v26, v27, v28, v29)
    del v22, v23, v24, v25, v26, v27, v28, v29
    print()
    v31 = v0[24576:24576+4*4096].view(cp.float32)
    v32 = 0
    v33 = 256
    v34 = 16
    v35 = 1
    v36 = 16
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
    v43 = raw_module.get_function(f"entry{v42}")
    del v42
    v43.max_dynamic_shared_size_bytes = 1536 
    v43((1,),(32,),(v1, v0),shared_mem=1536)
    del v0, v43
    v44 = "Here is the input tensor."
    method0(v44)
    del v44
    print()
    v45 = 0
    v46 = 128
    v47 = 8
    v48 = 1
    v49 = 1
    v50 = 16
    v51 = 8
    method1(v40, v45, v46, v47, v48, v49, v50, v51)
    del v40, v45, v46, v47, v48, v49, v50, v51
    print()
    v52 = "Here is the output tensor."
    method0(v52)
    del v52
    print()
    v54 = v1[7680:7680+4*4096].view(cp.float32)
    v55 = 0
    v56 = '['
    method2(v56)
    del v56
    v57 = 0
    while method5(v57):
        v59 = v55
        v60 = v59 >= 512
        del v59
        if v60:
            v61 = " ..."
            method0(v61)
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
            method0(v64)
        else:
            pass
        del v63
        v65 = '['
        method2(v65)
        del v65
        v66 = 0
        while method6(v66):
            v68 = v55
            v69 = v68 >= 512
            del v68
            if v69:
                v70 = " ..."
                method0(v70)
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
                method0(v73)
            else:
                pass
            del v72
            v74 = '['
            method2(v74)
            del v74
            v75 = 0
            while method5(v75):
                v77 = v55
                v78 = v77 >= 512
                del v77
                if v78:
                    v79 = " ..."
                    method0(v79)
                    del v79
                    break
                else:
                    pass
                del v78
                v80 = v75 == 0
                v81 = v80 != True
                del v80
                if v81:
                    v82 = "; "
                    method0(v82)
                else:
                    pass
                del v81
                v83 = '['
                method2(v83)
                del v83
                v84 = 0
                while method5(v84):
                    v86 = v55
                    v87 = v86 >= 512
                    del v86
                    if v87:
                        v88 = " ..."
                        method0(v88)
                        del v88
                        break
                    else:
                        pass
                    del v87
                    v89 = v84 == 0
                    v90 = v89 != True
                    del v89
                    if v90:
                        v91 = "; "
                        method0(v91)
                    else:
                        pass
                    del v90
                    v92 = v55 + 1
                    v55 = v92
                    del v92
                    v93 = v57 * 256
                    v94 = v66 * 256
                    v95 = v93 + v94
                    del v93, v94
                    v96 = v75 * 16
                    v97 = v95 + v96
                    del v95, v96
                    v98 = v97 + v84
                    del v97
                    v99 = v54[v98].item()
                    del v98
                    method4(v99)
                    del v99
                    v84 += 1 
                del v84
                v100 = ']'
                method2(v100)
                del v100
                v75 += 1 
            del v75
            v101 = ']'
            method2(v101)
            del v101
            v66 += 1 
        del v66
        v102 = ']'
        method2(v102)
        del v102
        v57 += 1 
    del v54, v55, v57
    v103 = ']'
    method2(v103)
    del v103
    method7()
    print()
    v105 = v1[24064:24064+4*256].view(cp.int32)
    del v1
    v106 = 0
    v107 = '['
    method2(v107)
    del v107
    v108 = 0
    while method5(v108):
        v110 = v106
        v111 = v110 >= 2147483647
        del v110
        if v111:
            v112 = " ..."
            method0(v112)
            del v112
            break
        else:
            pass
        del v111
        v113 = v108 == 0
        v114 = v113 != True
        del v113
        if v114:
            v115 = "; "
            method0(v115)
        else:
            pass
        del v114
        v116 = '['
        method2(v116)
        del v116
        v117 = 0
        while method6(v117):
            v119 = v106
            v120 = v119 >= 2147483647
            del v119
            if v120:
                v121 = " ..."
                method0(v121)
                del v121
                break
            else:
                pass
            del v120
            v122 = v117 == 0
            v123 = v122 != True
            del v122
            if v123:
                v124 = "; "
                method0(v124)
            else:
                pass
            del v123
            v125 = '['
            method2(v125)
            del v125
            v126 = 0
            while method5(v126):
                v128 = v106
                v129 = v128 >= 2147483647
                del v128
                if v129:
                    v130 = " ..."
                    method0(v130)
                    del v130
                    break
                else:
                    pass
                del v129
                v131 = v126 == 0
                v132 = v131 != True
                del v131
                if v132:
                    v133 = "; "
                    method0(v133)
                else:
                    pass
                del v132
                v134 = v106 + 1
                v106 = v134
                del v134
                v135 = v108 * 16
                v136 = v117 * 16
                v137 = v135 + v136
                del v135, v136
                v138 = v137 + v126
                del v137
                v139 = v105[v138].item()
                del v138
                method8(v139)
                del v139
                v126 += 1 
            del v126
            v140 = ']'
            method2(v140)
            del v140
            v117 += 1 
        del v117
        v141 = ']'
        method2(v141)
        del v141
        v108 += 1 
    del v105, v106, v108
    v142 = ']'
    method2(v142)
    del v142
    method7()
    print()
    return 

if __name__ == '__main__': print(main())
