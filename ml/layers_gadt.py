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

__device__ void method_0(float * v0, float * v1, float * v2);
__device__ void method_1(float * v0, float * v1);
__device__ void method_2(float * v0, float * v1);
__device__ void method_3(float * v0, float * v1, float * v2);
struct Tuple0;
struct Tuple1;
__device__ void method_4(int * v0, float * v1, curandStatePhilox4_32_10_t & v2);
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
__device__ void method_0(float * v0, float * v1, float * v2){
    unsigned int v3;
    v3 = 0ul;
    asm("mov.u32 %0, %dynamic_smem_size;" : "=r"(v3));
    unsigned long long v4;
    v4 = (unsigned long long)v3;
    bool v5;
    v5 = 1536ull <= v4;
    bool v6;
    v6 = v5 == false;
    if (v6){
        assert("The shared memory used in the matmult node is lower than the allocated amount." && v5);
    } else {
    }
    extern __shared__ unsigned char v8[];
    float * v9;
    v9 = reinterpret_cast<float *>(&v8[0ull]);
    float * v11;
    v11 = reinterpret_cast<float *>(&v8[768ull]);
    float * v13;
    v13 = reinterpret_cast<float *>(&v8[0ull]);
    int v15;
    v15 = threadIdx.x;
    int v16;
    v16 = v15 / 32l;
    bool v17;
    v17 = 0l <= v16;
    bool v18;
    v18 = v17 == false;
    if (v18){
        assert("The index needs to be zero or positive." && v17);
    } else {
    }
    int v20;
    v20 = v16 % 1l;
    bool v21;
    v21 = v16 < 1l;
    bool v22;
    v22 = v21 == false;
    if (v22){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v21);
    } else {
    }
    assert("Tensor range check" && 0 <= v16 && v16 < 1l);
    assert("Tensor range check" && 0 <= v20 && v20 < 1l);
    int v24;
    v24 = 16l * v20;
    int v25;
    v25 = 384l * v16;
    int v26;
    v26 = v25 + v24;
    float * v27;
    v27 = v13+v26;
    assert("Tensor range check" && 0 <= v16 && v16 < 1l);
    int v29;
    v29 = 192l * v16;
    int v30;
    v30 = threadIdx.x;
    int v31;
    v31 = v30 % 32l;
    bool v32;
    v32 = 0l <= v31;
    bool v33;
    v33 = v32 == false;
    if (v33){
        assert("The index needs to be zero or positive." && v32);
    } else {
    }
    int v35;
    v35 = v31 % 4l;
    int v36;
    v36 = v31 / 4l;
    bool v37;
    v37 = v36 < 8l;
    bool v38;
    v38 = v37 == false;
    if (v38){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v37);
    } else {
    }
    assert("Tensor range check" && 0 <= v36 && v36 < 8l);
    assert("Tensor range check" && 0 <= v35 && v35 < 4l);
    int v40;
    v40 = v35 + v29;
    int v41;
    v41 = 12l * v36;
    int v42;
    v42 = v41 + v40;
    float * v43;
    v43 = v9+v42;
    assert("Tensor range check" && 0 <= v20 && v20 < 1l);
    int v45;
    v45 = 192l * v20;
    int v46;
    v46 = threadIdx.x;
    int v47;
    v47 = v46 % 32l;
    bool v48;
    v48 = 0l <= v47;
    bool v49;
    v49 = v48 == false;
    if (v49){
        assert("The index needs to be zero or positive." && v48);
    } else {
    }
    int v51;
    v51 = v47 % 4l;
    int v52;
    v52 = v47 / 4l;
    bool v53;
    v53 = v52 < 8l;
    bool v54;
    v54 = v53 == false;
    if (v54){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v53);
    } else {
    }
    assert("Tensor range check" && 0 <= v52 && v52 < 8l);
    assert("Tensor range check" && 0 <= v51 && v51 < 4l);
    int v56;
    v56 = v51 + v45;
    int v57;
    v57 = 12l * v52;
    int v58;
    v58 = v57 + v56;
    float * v59;
    v59 = v11+v58;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v61[1l];
    int v62;
    v62 = 0l;
    while (while_method_0(v62)){
        int v64;
        v64 = 0l;
        while (while_method_0(v64)){
            assert("Tensor range check" && 0 <= v62 && v62 < 1l);
            assert("Tensor range check" && 0 <= v64 && v64 < 1l);
            int v66;
            v66 = 16l * v64;
            int v67;
            v67 = 256l * v62;
            int v68;
            v68 = v67 + v66;
            float * v69;
            v69 = v0+v68;
            // Pushing the loop unrolling to: 0
            int v71;
            v71 = 0l;
            #pragma unroll
            while (while_method_0(v71)){
                int v73;
                v73 = 0l;
                #pragma unroll
                while (while_method_0(v73)){
                    assert("Tensor range check" && 0 <= v71 && v71 < 1l);
                    assert("Tensor range check" && 0 <= v73 && v73 < 1l);
                    int v75;
                    v75 = v71 + v73;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v76 = v61[v75];
                    wmma::fill_fragment(v76, 0.0f);
                    v73 += 1l ;
                }
                v71 += 1l ;
            }
            int v77;
            v77 = 0l;
            #pragma unroll
            while (while_method_0(v77)){
                assert("Tensor range check" && 0 <= v62 && v62 < 1l);
                int v79;
                v79 = 128l * v62;
                assert("Tensor range check" && 0 <= v77 && v77 < 1l);
                int v80;
                v80 = 8l * v77;
                int v81;
                v81 = v80 + v79;
                float * v82;
                v82 = v2+v81;
                assert("Tensor range check" && 0 <= v64 && v64 < 1l);
                int v84;
                v84 = 128l * v64;
                assert("Tensor range check" && 0 <= v77 && v77 < 1l);
                int v85;
                v85 = v80 + v84;
                float * v86;
                v86 = v1+v85;
                int v88;
                v88 = threadIdx.x;
                bool v89;
                v89 = 0l <= v88;
                bool v90;
                v90 = v89 == false;
                if (v90){
                    assert("The index needs to be zero or positive." && v89);
                } else {
                }
                int v92;
                v92 = v88 % 2l;
                int v93;
                v93 = v88 / 2l;
                bool v94;
                v94 = v93 < 16l;
                bool v95;
                v95 = v94 == false;
                if (v95){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v94);
                } else {
                }
                assert("Tensor range check" && 0 <= v93 && v93 < 16l);
                assert("Tensor range check" && 0 <= v92 && v92 < 2l);
                int v97;
                v97 = 4l * v92;
                int v98;
                v98 = 12l * v93;
                int v99;
                v99 = v98 + v97;
                int v100;
                v100 = 8l * v93;
                int v101;
                v101 = v100 + v97;
                float * v102;
                v102 = v11+v99;
                float * v104;
                v104 = v86+v101;
                int v106;
                v106 = 0l;
                #pragma unroll
                while (while_method_0(v106)){
                    int v108;
                    v108 = 0l;
                    #pragma unroll
                    while (while_method_0(v108)){
                        assert("Tensor range check" && 0 <= v106 && v106 < 1l);
                        assert("Tensor range check" && 0 <= v108 && v108 < 1l);
                        int v110;
                        v110 = 8l * v108;
                        int v111;
                        v111 = 192l * v106;
                        int v112;
                        v112 = v111 + v110;
                        int v113;
                        v113 = 128l * v106;
                        int v114;
                        v114 = v113 + v110;
                        float v115[4l];
                        int v116;
                        v116 = 0l;
                        #pragma unroll
                        while (while_method_1(v116)){
                            assert("Tensor range check" && 0 <= v116 && v116 < 4l);
                            int v118;
                            v118 = v116 + v114;
                            float v119;
                            v119 = v104[v118];
                            float v120;
                            v120 = wmma::__float_to_tf32(v119);
                            assert("Tensor range check" && 0 <= v116 && v116 < 4l);
                            v115[v116] = v120;
                            v116 += 1l ;
                        }
                        int4* v121;
                        v121 = reinterpret_cast<int4*>(v115 + 0l);
                        int4* v122;
                        v122 = reinterpret_cast<int4*>(v102 + v112);
                        assert("Pointer alignment check" && (unsigned long long)(v121) % 4l == 0 && (unsigned long long)(v122) % 4l == 0);
                        *v122 = *v121;
                        v108 += 1l ;
                    }
                    v106 += 1l ;
                }
                int v123;
                v123 = threadIdx.x;
                bool v124;
                v124 = 0l <= v123;
                bool v125;
                v125 = v124 == false;
                if (v125){
                    assert("The index needs to be zero or positive." && v124);
                } else {
                }
                int v127;
                v127 = v123 % 2l;
                int v128;
                v128 = v123 / 2l;
                bool v129;
                v129 = v128 < 16l;
                bool v130;
                v130 = v129 == false;
                if (v130){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v129);
                } else {
                }
                assert("Tensor range check" && 0 <= v128 && v128 < 16l);
                assert("Tensor range check" && 0 <= v127 && v127 < 2l);
                int v132;
                v132 = 4l * v127;
                int v133;
                v133 = 12l * v128;
                int v134;
                v134 = v133 + v132;
                int v135;
                v135 = 8l * v128;
                int v136;
                v136 = v135 + v132;
                float * v137;
                v137 = v9+v134;
                float * v139;
                v139 = v82+v136;
                int v141;
                v141 = 0l;
                #pragma unroll
                while (while_method_0(v141)){
                    int v143;
                    v143 = 0l;
                    #pragma unroll
                    while (while_method_0(v143)){
                        assert("Tensor range check" && 0 <= v141 && v141 < 1l);
                        assert("Tensor range check" && 0 <= v143 && v143 < 1l);
                        int v145;
                        v145 = 8l * v143;
                        int v146;
                        v146 = 192l * v141;
                        int v147;
                        v147 = v146 + v145;
                        int v148;
                        v148 = 128l * v141;
                        int v149;
                        v149 = v148 + v145;
                        float v150[4l];
                        int v151;
                        v151 = 0l;
                        #pragma unroll
                        while (while_method_1(v151)){
                            assert("Tensor range check" && 0 <= v151 && v151 < 4l);
                            int v153;
                            v153 = v151 + v149;
                            float v154;
                            v154 = v139[v153];
                            float v155;
                            v155 = wmma::__float_to_tf32(v154);
                            assert("Tensor range check" && 0 <= v151 && v151 < 4l);
                            v150[v151] = v155;
                            v151 += 1l ;
                        }
                        int4* v156;
                        v156 = reinterpret_cast<int4*>(v150 + 0l);
                        int4* v157;
                        v157 = reinterpret_cast<int4*>(v137 + v147);
                        assert("Pointer alignment check" && (unsigned long long)(v156) % 4l == 0 && (unsigned long long)(v157) % 4l == 0);
                        *v157 = *v156;
                        v143 += 1l ;
                    }
                    v141 += 1l ;
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v158[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v159[1l];
                int v160;
                v160 = 0l;
                #pragma unroll
                while (while_method_0(v160)){
                    int v162;
                    v162 = 0l;
                    #pragma unroll
                    while (while_method_0(v162)){
                        assert("Tensor range check" && 0 <= v160 && v160 < 1l);
                        assert("Tensor range check" && 0 <= v162 && v162 < 1l);
                        int v164;
                        v164 = v160 + v162;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v165 = v158[v164];
                        assert("Tensor range check" && 0 <= v160 && v160 < 1l);
                        int v166;
                        v166 = 192l * v160;
                        assert("Tensor range check" && 0 <= v162 && v162 < 1l);
                        int v167;
                        v167 = 8l * v162;
                        int v168;
                        v168 = v167 + v166;
                        int v169;
                        v169 = 0l;
                        #pragma unroll
                        while (while_method_2(v169)){
                            int v171;
                            v171 = 0l;
                            #pragma unroll
                            while (while_method_2(v171)){
                                assert("Tensor range check" && 0 <= v169 && v169 < 2l);
                                assert("Tensor range check" && 0 <= v171 && v171 < 2l);
                                int v173;
                                v173 = 96l * v171;
                                int v174;
                                v174 = v173 + v168;
                                int v175;
                                v175 = 4l * v169;
                                int v176;
                                v176 = v175 + v174;
                                float v177;
                                v177 = v43[v176];
                                bool v178;
                                v178 = 0l <= v171;
                                bool v180;
                                if (v178){
                                    bool v179;
                                    v179 = v171 < 2l;
                                    v180 = v179;
                                } else {
                                    v180 = false;
                                }
                                bool v181;
                                v181 = v180 == false;
                                if (v181){
                                    assert("The indices should be inside the range of the dimension." && v180);
                                } else {
                                }
                                bool v183;
                                v183 = 0l <= v169;
                                bool v185;
                                if (v183){
                                    bool v184;
                                    v184 = v169 < 2l;
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
                                int v188;
                                v188 = v169 * 2l;
                                int v189;
                                v189 = v171 + v188;
                                v165.x[v189] = v177;
                                v171 += 1l ;
                            }
                            v169 += 1l ;
                        }
                        v162 += 1l ;
                    }
                    v160 += 1l ;
                }
                int v190;
                v190 = 0l;
                #pragma unroll
                while (while_method_0(v190)){
                    int v192;
                    v192 = 0l;
                    #pragma unroll
                    while (while_method_0(v192)){
                        assert("Tensor range check" && 0 <= v190 && v190 < 1l);
                        assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                        int v194;
                        v194 = v190 + v192;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v195 = v159[v194];
                        assert("Tensor range check" && 0 <= v190 && v190 < 1l);
                        int v196;
                        v196 = 192l * v190;
                        assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                        int v197;
                        v197 = 8l * v192;
                        int v198;
                        v198 = v197 + v196;
                        int v199;
                        v199 = 0l;
                        #pragma unroll
                        while (while_method_2(v199)){
                            int v201;
                            v201 = 0l;
                            #pragma unroll
                            while (while_method_2(v201)){
                                assert("Tensor range check" && 0 <= v199 && v199 < 2l);
                                assert("Tensor range check" && 0 <= v201 && v201 < 2l);
                                int v203;
                                v203 = 4l * v201;
                                int v204;
                                v204 = v203 + v198;
                                int v205;
                                v205 = 96l * v199;
                                int v206;
                                v206 = v205 + v204;
                                float v207;
                                v207 = v59[v206];
                                bool v208;
                                v208 = 0l <= v201;
                                bool v210;
                                if (v208){
                                    bool v209;
                                    v209 = v201 < 2l;
                                    v210 = v209;
                                } else {
                                    v210 = false;
                                }
                                bool v211;
                                v211 = v210 == false;
                                if (v211){
                                    assert("The indices should be inside the range of the dimension." && v210);
                                } else {
                                }
                                bool v213;
                                v213 = 0l <= v199;
                                bool v215;
                                if (v213){
                                    bool v214;
                                    v214 = v199 < 2l;
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
                                int v218;
                                v218 = v199 * 2l;
                                int v219;
                                v219 = v201 + v218;
                                v195.x[v219] = v207;
                                v201 += 1l ;
                            }
                            v199 += 1l ;
                        }
                        v192 += 1l ;
                    }
                    v190 += 1l ;
                }
                __syncthreads();
                int v220;
                v220 = 0l;
                #pragma unroll
                while (while_method_0(v220)){
                    int v222;
                    v222 = 0l;
                    #pragma unroll
                    while (while_method_0(v222)){
                        int v224;
                        v224 = 0l;
                        #pragma unroll
                        while (while_method_0(v224)){
                            assert("Tensor range check" && 0 <= v220 && v220 < 1l);
                            assert("Tensor range check" && 0 <= v222 && v222 < 1l);
                            int v226;
                            v226 = v220 + v222;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v227 = v61[v226];
                            assert("Tensor range check" && 0 <= v220 && v220 < 1l);
                            assert("Tensor range check" && 0 <= v224 && v224 < 1l);
                            int v228;
                            v228 = v220 + v224;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v229 = v158[v228];
                            assert("Tensor range check" && 0 <= v222 && v222 < 1l);
                            assert("Tensor range check" && 0 <= v224 && v224 < 1l);
                            int v230;
                            v230 = v222 + v224;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v231 = v159[v230];
                            wmma::mma_sync(v227, v229, v231, v227);
                            v224 += 1l ;
                        }
                        v222 += 1l ;
                    }
                    v220 += 1l ;
                }
                v77 += 1l ;
            }
            int v232;
            v232 = 0l;
            #pragma unroll
            while (while_method_0(v232)){
                int v234;
                v234 = 0l;
                #pragma unroll
                while (while_method_0(v234)){
                    assert("Tensor range check" && 0 <= v232 && v232 < 1l);
                    assert("Tensor range check" && 0 <= v234 && v234 < 1l);
                    int v236;
                    v236 = v232 + v234;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v237 = v61[v236];
                    assert("Tensor range check" && 0 <= v232 && v232 < 1l);
                    assert("Tensor range check" && 0 <= v234 && v234 < 1l);
                    int v238;
                    v238 = 16l * v234;
                    int v239;
                    v239 = 384l * v232;
                    int v240;
                    v240 = v239 + v238;
                    float * v241;
                    v241 = v27+v240;
                    wmma::store_matrix_sync(v241, v237, 24l, wmma::mem_row_major);
                    v234 += 1l ;
                }
                v232 += 1l ;
            }
            __syncthreads();
            int v243;
            v243 = threadIdx.x;
            bool v244;
            v244 = 0l <= v243;
            bool v245;
            v245 = v244 == false;
            if (v245){
                assert("The index needs to be zero or positive." && v244);
            } else {
            }
            int v247;
            v247 = v243 % 4l;
            int v248;
            v248 = v243 / 4l;
            bool v249;
            v249 = v248 < 8l;
            bool v250;
            v250 = v249 == false;
            if (v250){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v249);
            } else {
            }
            assert("Tensor range check" && 0 <= v248 && v248 < 8l);
            assert("Tensor range check" && 0 <= v247 && v247 < 4l);
            int v252;
            v252 = 4l * v247;
            int v253;
            v253 = 16l * v248;
            int v254;
            v254 = v253 + v252;
            int v255;
            v255 = 24l * v248;
            int v256;
            v256 = v255 + v252;
            float * v257;
            v257 = v69+v254;
            float * v259;
            v259 = v13+v256;
            int v261;
            v261 = 0l;
            #pragma unroll
            while (while_method_2(v261)){
                int v263;
                v263 = 0l;
                #pragma unroll
                while (while_method_0(v263)){
                    assert("Tensor range check" && 0 <= v261 && v261 < 2l);
                    assert("Tensor range check" && 0 <= v263 && v263 < 1l);
                    int v265;
                    v265 = 16l * v263;
                    int v266;
                    v266 = 128l * v261;
                    int v267;
                    v267 = v266 + v265;
                    int v268;
                    v268 = 192l * v261;
                    int v269;
                    v269 = v268 + v265;
                    int4* v270;
                    v270 = reinterpret_cast<int4*>(v259 + v269);
                    int4* v271;
                    v271 = reinterpret_cast<int4*>(v257 + v267);
                    assert("Pointer alignment check" && (unsigned long long)(v270) % 4l == 0 && (unsigned long long)(v271) % 4l == 0);
                    *v271 = *v270;
                    v263 += 1l ;
                }
                v261 += 1l ;
            }
            __syncthreads();
            // Poping the loop unrolling to: 0
            v64 += 1l ;
        }
        v62 += 1l ;
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
    v6 = v2 % 4l;
    int v7;
    v7 = v2 / 4l;
    bool v8;
    v8 = v7 < 8l;
    bool v9;
    v9 = v8 == false;
    if (v9){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v8);
    } else {
    }
    assert("Tensor range check" && 0 <= v7 && v7 < 8l);
    assert("Tensor range check" && 0 <= v6 && v6 < 4l);
    int v11;
    v11 = 4l * v6;
    int v12;
    v12 = 16l * v7;
    int v13;
    v13 = v12 + v11;
    assert("Tensor range check" && 0 <= v7 && v7 < 8l);
    assert("Tensor range check" && 0 <= v6 && v6 < 4l);
    int v14;
    v14 = 0l;
    while (while_method_2(v14)){
        assert("Tensor range check" && 0 <= v14 && v14 < 2l);
        int v16;
        v16 = 128l * v14;
        int v17;
        v17 = v16 + v13;
        assert("Tensor range check" && 0 <= v14 && v14 < 2l);
        float v18[4l];
        int v19[4l];
        int v20;
        v20 = 0l;
        while (while_method_0(v20)){
            assert("Tensor range check" && 0 <= v20 && v20 < 1l);
            int v22;
            v22 = 4l * v20;
            assert("Tensor range check" && 0 <= v20 && v20 < 1l);
            int v23;
            v23 = 16l * v20;
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
        while (while_method_0(v27)){
            int v29;
            v29 = 0l;
            while (while_method_1(v29)){
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
                    v37 = v6 < 4l;
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
                v48 = v27 * 16l;
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
            v57 = v14 < 2l;
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
        v61 = v14 * 8l;
        int v62;
        v62 = v61 + v7;
        float v63[4l];
        int v64;
        v64 = 0l;
        while (while_method_0(v64)){
            int v66;
            v66 = 0l;
            while (while_method_1(v66)){
                assert("Tensor range check" && 0 <= v64 && v64 < 1l);
                assert("Tensor range check" && 0 <= v66 && v66 < 4l);
                int v68;
                v68 = 4l * v64;
                int v69;
                v69 = v68 + v66;
                float v70;
                v70 = v18[v69];
                float v71;
                v71 = v70 * v70;
                assert("Tensor range check" && 0 <= v64 && v64 < 1l);
                assert("Tensor range check" && 0 <= v66 && v66 < 4l);
                v63[v69] = v71;
                v66 += 1l ;
            }
            v64 += 1l ;
        }
        float v72;
        v72 = 0.0f;
        int v73;
        v73 = 0l;
        while (while_method_0(v73)){
            int v75;
            v75 = 0l;
            while (while_method_1(v75)){
                assert("Tensor range check" && 0 <= v73 && v73 < 1l);
                assert("Tensor range check" && 0 <= v75 && v75 < 4l);
                int v77;
                v77 = 4l * v73;
                int v78;
                v78 = v77 + v75;
                float v79;
                v79 = v63[v78];
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
        v83 = v82 / 4l;
        auto v84 = cooperative_groups::labeled_partition(v81,v83);
        Closure0 v85{};
        float v86;
        v86 = cooperative_groups::reduce(v84, v72, v85);
        float v87[4l];
        int v88;
        v88 = 0l;
        while (while_method_0(v88)){
            int v90;
            v90 = 0l;
            while (while_method_1(v90)){
                assert("Tensor range check" && 0 <= v88 && v88 < 1l);
                assert("Tensor range check" && 0 <= v90 && v90 < 4l);
                int v92;
                v92 = 4l * v88;
                int v93;
                v93 = v92 + v90;
                float v94;
                v94 = v18[v93];
                bool v95;
                v95 = v86 == 0.0f;
                bool v96;
                v96 = v95 != true;
                float v98;
                if (v96){
                    float v97;
                    v97 = v94 / v86;
                    v98 = v97;
                } else {
                    v98 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v88 && v88 < 1l);
                assert("Tensor range check" && 0 <= v90 && v90 < 4l);
                v87[v93] = v98;
                v90 += 1l ;
            }
            v88 += 1l ;
        }
        int v99;
        v99 = 0l;
        while (while_method_0(v99)){
            assert("Tensor range check" && 0 <= v99 && v99 < 1l);
            int v101;
            v101 = 16l * v99;
            int v102;
            v102 = v101 + v17;
            assert("Tensor range check" && 0 <= v99 && v99 < 1l);
            int v103;
            v103 = 4l * v99;
            int4* v104;
            v104 = reinterpret_cast<int4*>(v87 + v103);
            int4* v105;
            v105 = reinterpret_cast<int4*>(v0 + v102);
            assert("Pointer alignment check" && (unsigned long long)(v104) % 4l == 0 && (unsigned long long)(v105) % 4l == 0);
            *v105 = *v104;
            v99 += 1l ;
        }
        v14 += 1l ;
    }
    __syncthreads();
    return ;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 64l;
    return v1;
}
__device__ void method_2(float * v0, float * v1){
    int v2;
    v2 = threadIdx.x;
    int v3;
    v3 = v2;
    while (while_method_3(v3)){
        bool v5;
        v5 = 0l <= v3;
        bool v6;
        v6 = v5 == false;
        if (v6){
            assert("The index needs to be zero or positive." && v5);
        } else {
        }
        int v8;
        v8 = v3 % 4l;
        int v9;
        v9 = v3 / 4l;
        bool v10;
        v10 = v9 < 16l;
        bool v11;
        v11 = v10 == false;
        if (v11){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v10);
        } else {
        }
        assert("Tensor range check" && 0 <= v9 && v9 < 16l);
        assert("Tensor range check" && 0 <= v8 && v8 < 4l);
        int v13;
        v13 = 4l * v8;
        int v14;
        v14 = 16l * v9;
        int v15;
        v15 = v14 + v13;
        assert("Tensor range check" && 0 <= v9 && v9 < 16l);
        assert("Tensor range check" && 0 <= v8 && v8 < 4l);
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
        while (while_method_1(v20)){
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
__device__ void method_3(float * v0, float * v1, float * v2){
    unsigned int v3;
    v3 = 0ul;
    asm("mov.u32 %0, %dynamic_smem_size;" : "=r"(v3));
    unsigned long long v4;
    v4 = (unsigned long long)v3;
    bool v5;
    v5 = 1536ull <= v4;
    bool v6;
    v6 = v5 == false;
    if (v6){
        assert("The shared memory used in the matmult node is lower than the allocated amount." && v5);
    } else {
    }
    extern __shared__ unsigned char v8[];
    float * v9;
    v9 = reinterpret_cast<float *>(&v8[0ull]);
    float * v11;
    v11 = reinterpret_cast<float *>(&v8[768ull]);
    float * v13;
    v13 = reinterpret_cast<float *>(&v8[0ull]);
    int v15;
    v15 = threadIdx.x;
    int v16;
    v16 = v15 / 32l;
    bool v17;
    v17 = 0l <= v16;
    bool v18;
    v18 = v17 == false;
    if (v18){
        assert("The index needs to be zero or positive." && v17);
    } else {
    }
    int v20;
    v20 = v16 % 1l;
    bool v21;
    v21 = v16 < 1l;
    bool v22;
    v22 = v21 == false;
    if (v22){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v21);
    } else {
    }
    assert("Tensor range check" && 0 <= v16 && v16 < 1l);
    assert("Tensor range check" && 0 <= v20 && v20 < 1l);
    int v24;
    v24 = 16l * v20;
    int v25;
    v25 = 384l * v16;
    int v26;
    v26 = v25 + v24;
    float * v27;
    v27 = v13+v26;
    assert("Tensor range check" && 0 <= v16 && v16 < 1l);
    int v29;
    v29 = 192l * v16;
    int v30;
    v30 = threadIdx.x;
    int v31;
    v31 = v30 % 32l;
    bool v32;
    v32 = 0l <= v31;
    bool v33;
    v33 = v32 == false;
    if (v33){
        assert("The index needs to be zero or positive." && v32);
    } else {
    }
    int v35;
    v35 = v31 % 4l;
    int v36;
    v36 = v31 / 4l;
    bool v37;
    v37 = v36 < 8l;
    bool v38;
    v38 = v37 == false;
    if (v38){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v37);
    } else {
    }
    assert("Tensor range check" && 0 <= v36 && v36 < 8l);
    assert("Tensor range check" && 0 <= v35 && v35 < 4l);
    int v40;
    v40 = v35 + v29;
    int v41;
    v41 = 12l * v36;
    int v42;
    v42 = v41 + v40;
    float * v43;
    v43 = v9+v42;
    assert("Tensor range check" && 0 <= v20 && v20 < 1l);
    int v45;
    v45 = 192l * v20;
    int v46;
    v46 = threadIdx.x;
    int v47;
    v47 = v46 % 32l;
    bool v48;
    v48 = 0l <= v47;
    bool v49;
    v49 = v48 == false;
    if (v49){
        assert("The index needs to be zero or positive." && v48);
    } else {
    }
    int v51;
    v51 = v47 % 4l;
    int v52;
    v52 = v47 / 4l;
    bool v53;
    v53 = v52 < 8l;
    bool v54;
    v54 = v53 == false;
    if (v54){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v53);
    } else {
    }
    assert("Tensor range check" && 0 <= v52 && v52 < 8l);
    assert("Tensor range check" && 0 <= v51 && v51 < 4l);
    int v56;
    v56 = v51 + v45;
    int v57;
    v57 = 12l * v52;
    int v58;
    v58 = v57 + v56;
    float * v59;
    v59 = v11+v58;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v61[1l];
    int v62;
    v62 = 0l;
    while (while_method_0(v62)){
        int v64;
        v64 = 0l;
        while (while_method_0(v64)){
            assert("Tensor range check" && 0 <= v62 && v62 < 1l);
            assert("Tensor range check" && 0 <= v64 && v64 < 1l);
            int v66;
            v66 = 16l * v64;
            int v67;
            v67 = 256l * v62;
            int v68;
            v68 = v67 + v66;
            float * v69;
            v69 = v0+v68;
            // Pushing the loop unrolling to: 0
            int v71;
            v71 = 0l;
            #pragma unroll
            while (while_method_0(v71)){
                int v73;
                v73 = 0l;
                #pragma unroll
                while (while_method_0(v73)){
                    assert("Tensor range check" && 0 <= v71 && v71 < 1l);
                    assert("Tensor range check" && 0 <= v73 && v73 < 1l);
                    int v75;
                    v75 = v71 + v73;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v76 = v61[v75];
                    wmma::fill_fragment(v76, 0.0f);
                    v73 += 1l ;
                }
                v71 += 1l ;
            }
            int v77;
            v77 = 0l;
            #pragma unroll
            while (while_method_2(v77)){
                assert("Tensor range check" && 0 <= v62 && v62 < 1l);
                assert("Tensor range check" && 0 <= v77 && v77 < 2l);
                int v79;
                v79 = 8l * v77;
                int v80;
                v80 = v79 + v67;
                float * v81;
                v81 = v2+v80;
                assert("Tensor range check" && 0 <= v64 && v64 < 1l);
                int v83;
                v83 = 256l * v64;
                assert("Tensor range check" && 0 <= v77 && v77 < 2l);
                int v84;
                v84 = v79 + v83;
                float * v85;
                v85 = v1+v84;
                int v87;
                v87 = threadIdx.x;
                bool v88;
                v88 = 0l <= v87;
                bool v89;
                v89 = v88 == false;
                if (v89){
                    assert("The index needs to be zero or positive." && v88);
                } else {
                }
                int v91;
                v91 = v87 % 2l;
                int v92;
                v92 = v87 / 2l;
                bool v93;
                v93 = v92 < 16l;
                bool v94;
                v94 = v93 == false;
                if (v94){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v93);
                } else {
                }
                assert("Tensor range check" && 0 <= v92 && v92 < 16l);
                assert("Tensor range check" && 0 <= v91 && v91 < 2l);
                int v96;
                v96 = 4l * v91;
                int v97;
                v97 = 12l * v92;
                int v98;
                v98 = v97 + v96;
                int v99;
                v99 = 16l * v92;
                int v100;
                v100 = v99 + v96;
                float * v101;
                v101 = v11+v98;
                float * v103;
                v103 = v85+v100;
                int v105;
                v105 = 0l;
                #pragma unroll
                while (while_method_0(v105)){
                    int v107;
                    v107 = 0l;
                    #pragma unroll
                    while (while_method_0(v107)){
                        assert("Tensor range check" && 0 <= v105 && v105 < 1l);
                        assert("Tensor range check" && 0 <= v107 && v107 < 1l);
                        int v109;
                        v109 = 8l * v107;
                        int v110;
                        v110 = 192l * v105;
                        int v111;
                        v111 = v110 + v109;
                        int v112;
                        v112 = 256l * v105;
                        int v113;
                        v113 = v112 + v109;
                        float v114[4l];
                        int v115;
                        v115 = 0l;
                        #pragma unroll
                        while (while_method_1(v115)){
                            assert("Tensor range check" && 0 <= v115 && v115 < 4l);
                            int v117;
                            v117 = v115 + v113;
                            float v118;
                            v118 = v103[v117];
                            float v119;
                            v119 = wmma::__float_to_tf32(v118);
                            assert("Tensor range check" && 0 <= v115 && v115 < 4l);
                            v114[v115] = v119;
                            v115 += 1l ;
                        }
                        int4* v120;
                        v120 = reinterpret_cast<int4*>(v114 + 0l);
                        int4* v121;
                        v121 = reinterpret_cast<int4*>(v101 + v111);
                        assert("Pointer alignment check" && (unsigned long long)(v120) % 4l == 0 && (unsigned long long)(v121) % 4l == 0);
                        *v121 = *v120;
                        v107 += 1l ;
                    }
                    v105 += 1l ;
                }
                int v122;
                v122 = threadIdx.x;
                bool v123;
                v123 = 0l <= v122;
                bool v124;
                v124 = v123 == false;
                if (v124){
                    assert("The index needs to be zero or positive." && v123);
                } else {
                }
                int v126;
                v126 = v122 % 2l;
                int v127;
                v127 = v122 / 2l;
                bool v128;
                v128 = v127 < 16l;
                bool v129;
                v129 = v128 == false;
                if (v129){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v128);
                } else {
                }
                assert("Tensor range check" && 0 <= v127 && v127 < 16l);
                assert("Tensor range check" && 0 <= v126 && v126 < 2l);
                int v131;
                v131 = 4l * v126;
                int v132;
                v132 = 12l * v127;
                int v133;
                v133 = v132 + v131;
                int v134;
                v134 = 16l * v127;
                int v135;
                v135 = v134 + v131;
                float * v136;
                v136 = v9+v133;
                float * v138;
                v138 = v81+v135;
                int v140;
                v140 = 0l;
                #pragma unroll
                while (while_method_0(v140)){
                    int v142;
                    v142 = 0l;
                    #pragma unroll
                    while (while_method_0(v142)){
                        assert("Tensor range check" && 0 <= v140 && v140 < 1l);
                        assert("Tensor range check" && 0 <= v142 && v142 < 1l);
                        int v144;
                        v144 = 8l * v142;
                        int v145;
                        v145 = 192l * v140;
                        int v146;
                        v146 = v145 + v144;
                        int v147;
                        v147 = 256l * v140;
                        int v148;
                        v148 = v147 + v144;
                        float v149[4l];
                        int v150;
                        v150 = 0l;
                        #pragma unroll
                        while (while_method_1(v150)){
                            assert("Tensor range check" && 0 <= v150 && v150 < 4l);
                            int v152;
                            v152 = v150 + v148;
                            float v153;
                            v153 = v138[v152];
                            float v154;
                            v154 = wmma::__float_to_tf32(v153);
                            assert("Tensor range check" && 0 <= v150 && v150 < 4l);
                            v149[v150] = v154;
                            v150 += 1l ;
                        }
                        int4* v155;
                        v155 = reinterpret_cast<int4*>(v149 + 0l);
                        int4* v156;
                        v156 = reinterpret_cast<int4*>(v136 + v146);
                        assert("Pointer alignment check" && (unsigned long long)(v155) % 4l == 0 && (unsigned long long)(v156) % 4l == 0);
                        *v156 = *v155;
                        v142 += 1l ;
                    }
                    v140 += 1l ;
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v157[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v158[1l];
                int v159;
                v159 = 0l;
                #pragma unroll
                while (while_method_0(v159)){
                    int v161;
                    v161 = 0l;
                    #pragma unroll
                    while (while_method_0(v161)){
                        assert("Tensor range check" && 0 <= v159 && v159 < 1l);
                        assert("Tensor range check" && 0 <= v161 && v161 < 1l);
                        int v163;
                        v163 = v159 + v161;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v164 = v157[v163];
                        assert("Tensor range check" && 0 <= v159 && v159 < 1l);
                        int v165;
                        v165 = 192l * v159;
                        assert("Tensor range check" && 0 <= v161 && v161 < 1l);
                        int v166;
                        v166 = 8l * v161;
                        int v167;
                        v167 = v166 + v165;
                        int v168;
                        v168 = 0l;
                        #pragma unroll
                        while (while_method_2(v168)){
                            int v170;
                            v170 = 0l;
                            #pragma unroll
                            while (while_method_2(v170)){
                                assert("Tensor range check" && 0 <= v168 && v168 < 2l);
                                assert("Tensor range check" && 0 <= v170 && v170 < 2l);
                                int v172;
                                v172 = 96l * v170;
                                int v173;
                                v173 = v172 + v167;
                                int v174;
                                v174 = 4l * v168;
                                int v175;
                                v175 = v174 + v173;
                                float v176;
                                v176 = v43[v175];
                                bool v177;
                                v177 = 0l <= v170;
                                bool v179;
                                if (v177){
                                    bool v178;
                                    v178 = v170 < 2l;
                                    v179 = v178;
                                } else {
                                    v179 = false;
                                }
                                bool v180;
                                v180 = v179 == false;
                                if (v180){
                                    assert("The indices should be inside the range of the dimension." && v179);
                                } else {
                                }
                                bool v182;
                                v182 = 0l <= v168;
                                bool v184;
                                if (v182){
                                    bool v183;
                                    v183 = v168 < 2l;
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
                                int v187;
                                v187 = v168 * 2l;
                                int v188;
                                v188 = v170 + v187;
                                v164.x[v188] = v176;
                                v170 += 1l ;
                            }
                            v168 += 1l ;
                        }
                        v161 += 1l ;
                    }
                    v159 += 1l ;
                }
                int v189;
                v189 = 0l;
                #pragma unroll
                while (while_method_0(v189)){
                    int v191;
                    v191 = 0l;
                    #pragma unroll
                    while (while_method_0(v191)){
                        assert("Tensor range check" && 0 <= v189 && v189 < 1l);
                        assert("Tensor range check" && 0 <= v191 && v191 < 1l);
                        int v193;
                        v193 = v189 + v191;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v194 = v158[v193];
                        assert("Tensor range check" && 0 <= v189 && v189 < 1l);
                        int v195;
                        v195 = 192l * v189;
                        assert("Tensor range check" && 0 <= v191 && v191 < 1l);
                        int v196;
                        v196 = 8l * v191;
                        int v197;
                        v197 = v196 + v195;
                        int v198;
                        v198 = 0l;
                        #pragma unroll
                        while (while_method_2(v198)){
                            int v200;
                            v200 = 0l;
                            #pragma unroll
                            while (while_method_2(v200)){
                                assert("Tensor range check" && 0 <= v198 && v198 < 2l);
                                assert("Tensor range check" && 0 <= v200 && v200 < 2l);
                                int v202;
                                v202 = 4l * v200;
                                int v203;
                                v203 = v202 + v197;
                                int v204;
                                v204 = 96l * v198;
                                int v205;
                                v205 = v204 + v203;
                                float v206;
                                v206 = v59[v205];
                                bool v207;
                                v207 = 0l <= v200;
                                bool v209;
                                if (v207){
                                    bool v208;
                                    v208 = v200 < 2l;
                                    v209 = v208;
                                } else {
                                    v209 = false;
                                }
                                bool v210;
                                v210 = v209 == false;
                                if (v210){
                                    assert("The indices should be inside the range of the dimension." && v209);
                                } else {
                                }
                                bool v212;
                                v212 = 0l <= v198;
                                bool v214;
                                if (v212){
                                    bool v213;
                                    v213 = v198 < 2l;
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
                                int v217;
                                v217 = v198 * 2l;
                                int v218;
                                v218 = v200 + v217;
                                v194.x[v218] = v206;
                                v200 += 1l ;
                            }
                            v198 += 1l ;
                        }
                        v191 += 1l ;
                    }
                    v189 += 1l ;
                }
                __syncthreads();
                int v219;
                v219 = 0l;
                #pragma unroll
                while (while_method_0(v219)){
                    int v221;
                    v221 = 0l;
                    #pragma unroll
                    while (while_method_0(v221)){
                        int v223;
                        v223 = 0l;
                        #pragma unroll
                        while (while_method_0(v223)){
                            assert("Tensor range check" && 0 <= v219 && v219 < 1l);
                            assert("Tensor range check" && 0 <= v221 && v221 < 1l);
                            int v225;
                            v225 = v219 + v221;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v226 = v61[v225];
                            assert("Tensor range check" && 0 <= v219 && v219 < 1l);
                            assert("Tensor range check" && 0 <= v223 && v223 < 1l);
                            int v227;
                            v227 = v219 + v223;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v228 = v157[v227];
                            assert("Tensor range check" && 0 <= v221 && v221 < 1l);
                            assert("Tensor range check" && 0 <= v223 && v223 < 1l);
                            int v229;
                            v229 = v221 + v223;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v230 = v158[v229];
                            wmma::mma_sync(v226, v228, v230, v226);
                            v223 += 1l ;
                        }
                        v221 += 1l ;
                    }
                    v219 += 1l ;
                }
                v77 += 1l ;
            }
            int v231;
            v231 = 0l;
            #pragma unroll
            while (while_method_0(v231)){
                int v233;
                v233 = 0l;
                #pragma unroll
                while (while_method_0(v233)){
                    assert("Tensor range check" && 0 <= v231 && v231 < 1l);
                    assert("Tensor range check" && 0 <= v233 && v233 < 1l);
                    int v235;
                    v235 = v231 + v233;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v236 = v61[v235];
                    assert("Tensor range check" && 0 <= v231 && v231 < 1l);
                    assert("Tensor range check" && 0 <= v233 && v233 < 1l);
                    int v237;
                    v237 = 16l * v233;
                    int v238;
                    v238 = 384l * v231;
                    int v239;
                    v239 = v238 + v237;
                    float * v240;
                    v240 = v27+v239;
                    wmma::store_matrix_sync(v240, v236, 24l, wmma::mem_row_major);
                    v233 += 1l ;
                }
                v231 += 1l ;
            }
            __syncthreads();
            int v242;
            v242 = threadIdx.x;
            bool v243;
            v243 = 0l <= v242;
            bool v244;
            v244 = v243 == false;
            if (v244){
                assert("The index needs to be zero or positive." && v243);
            } else {
            }
            int v246;
            v246 = v242 % 4l;
            int v247;
            v247 = v242 / 4l;
            bool v248;
            v248 = v247 < 8l;
            bool v249;
            v249 = v248 == false;
            if (v249){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v248);
            } else {
            }
            assert("Tensor range check" && 0 <= v247 && v247 < 8l);
            assert("Tensor range check" && 0 <= v246 && v246 < 4l);
            int v251;
            v251 = 4l * v246;
            int v252;
            v252 = 16l * v247;
            int v253;
            v253 = v252 + v251;
            int v254;
            v254 = 24l * v247;
            int v255;
            v255 = v254 + v251;
            float * v256;
            v256 = v69+v253;
            float * v258;
            v258 = v13+v255;
            int v260;
            v260 = 0l;
            #pragma unroll
            while (while_method_2(v260)){
                int v262;
                v262 = 0l;
                #pragma unroll
                while (while_method_0(v262)){
                    assert("Tensor range check" && 0 <= v260 && v260 < 2l);
                    assert("Tensor range check" && 0 <= v262 && v262 < 1l);
                    int v264;
                    v264 = 16l * v262;
                    int v265;
                    v265 = 128l * v260;
                    int v266;
                    v266 = v265 + v264;
                    int v267;
                    v267 = 192l * v260;
                    int v268;
                    v268 = v267 + v264;
                    int4* v269;
                    v269 = reinterpret_cast<int4*>(v258 + v268);
                    int4* v270;
                    v270 = reinterpret_cast<int4*>(v256 + v266);
                    assert("Pointer alignment check" && (unsigned long long)(v269) % 4l == 0 && (unsigned long long)(v270) % 4l == 0);
                    *v270 = *v269;
                    v262 += 1l ;
                }
                v260 += 1l ;
            }
            __syncthreads();
            // Poping the loop unrolling to: 0
            v64 += 1l ;
        }
        v62 += 1l ;
    }
    return ;
}
__device__ void method_4(int * v0, float * v1, curandStatePhilox4_32_10_t & v2){
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
    v13 = 16l * v8;
    int v14;
    v14 = v13 + v12;
    assert("Tensor range check" && 0 <= v8 && v8 < 8l);
    int v15;
    v15 = 0l;
    while (while_method_2(v15)){
        assert("Tensor range check" && 0 <= v15 && v15 < 2l);
        int v17;
        v17 = 128l * v15;
        int v18;
        v18 = v17 + v14;
        float v19[4l];
        int v20[4l];
        int v21;
        v21 = 0l;
        while (while_method_0(v21)){
            assert("Tensor range check" && 0 <= v21 && v21 < 1l);
            int v23;
            v23 = 4l * v21;
            assert("Tensor range check" && 0 <= v21 && v21 < 1l);
            int v24;
            v24 = 16l * v21;
            int v25;
            v25 = v24 + v18;
            int4* v26;
            v26 = reinterpret_cast<int4*>(v1 + v25);
            int4* v27;
            v27 = reinterpret_cast<int4*>(v19 + v23);
            assert("Pointer alignment check" && (unsigned long long)(v26) % 4l == 0 && (unsigned long long)(v27) % 4l == 0);
            *v27 = *v26;
            v21 += 1l ;
        }
        int v28;
        v28 = 0l;
        while (while_method_0(v28)){
            int v30;
            v30 = 0l;
            while (while_method_1(v30)){
                bool v32;
                v32 = 0l <= v30;
                bool v34;
                if (v32){
                    bool v33;
                    v33 = v30 < 4l;
                    v34 = v33;
                } else {
                    v34 = false;
                }
                bool v35;
                v35 = v34 == false;
                if (v35){
                    assert("The indices should be inside the range of the dimension." && v34);
                } else {
                }
                bool v37;
                v37 = 0l <= v7;
                bool v39;
                if (v37){
                    bool v38;
                    v38 = v7 < 4l;
                    v39 = v38;
                } else {
                    v39 = false;
                }
                bool v40;
                v40 = v39 == false;
                if (v40){
                    assert("The indices should be inside the range of the dimension." && v39);
                } else {
                }
                int v42;
                v42 = v7 * 4l;
                int v43;
                v43 = v30 + v42;
                bool v44;
                v44 = 0l <= v28;
                bool v46;
                if (v44){
                    bool v45;
                    v45 = v28 < 1l;
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
                v49 = v28 * 16l;
                int v50;
                v50 = v43 + v49;
                assert("Tensor range check" && 0 <= v28 && v28 < 1l);
                assert("Tensor range check" && 0 <= v30 && v30 < 4l);
                int v51;
                v51 = 4l * v28;
                int v52;
                v52 = v51 + v30;
                v20[v52] = v50;
                v30 += 1l ;
            }
            v28 += 1l ;
        }
        bool v53;
        v53 = 0l <= v8;
        bool v54;
        v54 = v53 && v9;
        bool v55;
        v55 = v54 == false;
        if (v55){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v54);
        } else {
        }
        bool v57;
        v57 = 0l <= v15;
        bool v59;
        if (v57){
            bool v58;
            v58 = v15 < 2l;
            v59 = v58;
        } else {
            v59 = false;
        }
        bool v60;
        v60 = v59 == false;
        if (v60){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v59);
        } else {
        }
        int v62;
        v62 = v15 * 8l;
        int v63;
        v63 = v62 + v8;
        float v64;
        v64 = 0.0f;
        int v65;
        v65 = 0l;
        while (while_method_0(v65)){
            int v67;
            v67 = 0l;
            while (while_method_1(v67)){
                assert("Tensor range check" && 0 <= v65 && v65 < 1l);
                assert("Tensor range check" && 0 <= v67 && v67 < 4l);
                int v69;
                v69 = 4l * v65;
                int v70;
                v70 = v69 + v67;
                float v71;
                v71 = v19[v70];
                float v72;
                v72 = v64 + v71;
                v64 = v72;
                v67 += 1l ;
            }
            v65 += 1l ;
        }
        auto v73 = cooperative_groups::coalesced_threads();
        int v74;
        v74 = threadIdx.x;
        int v75;
        v75 = v74 / 4l;
        auto v76 = cooperative_groups::labeled_partition(v73,v75);
        Closure0 v77{};
        float v78;
        v78 = cooperative_groups::reduce(v76, v64, v77);
        float v79;
        v79 = v78 / 16.0f;
        float v80[4l];
        int v81;
        v81 = 0l;
        while (while_method_0(v81)){
            int v83;
            v83 = 0l;
            while (while_method_1(v83)){
                assert("Tensor range check" && 0 <= v81 && v81 < 1l);
                assert("Tensor range check" && 0 <= v83 && v83 < 4l);
                int v85;
                v85 = 4l * v81;
                int v86;
                v86 = v85 + v83;
                float v87;
                v87 = v19[v86];
                float v88;
                v88 = v87 - v79;
                float v89;
                v89 = exp(v88);
                assert("Tensor range check" && 0 <= v81 && v81 < 1l);
                assert("Tensor range check" && 0 <= v83 && v83 < 4l);
                v80[v86] = v89;
                v83 += 1l ;
            }
            v81 += 1l ;
        }
        float v90;
        v90 = 0.0f;
        int v91;
        v91 = 0l;
        while (while_method_0(v91)){
            int v93;
            v93 = 0l;
            while (while_method_1(v93)){
                assert("Tensor range check" && 0 <= v91 && v91 < 1l);
                assert("Tensor range check" && 0 <= v93 && v93 < 4l);
                int v95;
                v95 = 4l * v91;
                int v96;
                v96 = v95 + v93;
                float v97;
                v97 = v80[v96];
                float v98;
                v98 = v90 + v97;
                v90 = v98;
                v93 += 1l ;
            }
            v91 += 1l ;
        }
        auto v99 = cooperative_groups::coalesced_threads();
        int v100;
        v100 = threadIdx.x;
        int v101;
        v101 = v100 / 4l;
        auto v102 = cooperative_groups::labeled_partition(v99,v101);
        float v103;
        v103 = cooperative_groups::reduce(v102, v90, v77);
        float v104[4l];
        int v105;
        v105 = 0l;
        while (while_method_0(v105)){
            int v107;
            v107 = 0l;
            while (while_method_1(v107)){
                assert("Tensor range check" && 0 <= v105 && v105 < 1l);
                assert("Tensor range check" && 0 <= v107 && v107 < 4l);
                int v109;
                v109 = 4l * v105;
                int v110;
                v110 = v109 + v107;
                float v111;
                v111 = v80[v110];
                bool v112;
                v112 = v103 == 0.0f;
                bool v113;
                v113 = v112 != true;
                float v115;
                if (v113){
                    float v114;
                    v114 = v111 / v103;
                    v115 = v114;
                } else {
                    v115 = 0.0625f;
                }
                assert("Tensor range check" && 0 <= v105 && v105 < 1l);
                assert("Tensor range check" && 0 <= v107 && v107 < 4l);
                v104[v110] = v115;
                v107 += 1l ;
            }
            v105 += 1l ;
        }
        float v116[4l];
        float v117;
        v117 = 0.0f;
        int v118;
        v118 = 0l;
        while (while_method_0(v118)){
            assert("Tensor range check" && 0 <= v118 && v118 < 1l);
            int v120;
            v120 = 4l * v118;
            assert("Tensor range check" && 0 <= v118 && v118 < 1l);
            int v121; float v122;
            Tuple0 tmp0 = Tuple0{0l, 0.0f};
            v121 = tmp0.v0; v122 = tmp0.v1;
            while (while_method_1(v121)){
                assert("Tensor range check" && 0 <= v121 && v121 < 4l);
                int v124;
                v124 = v121 + v120;
                float v125;
                v125 = v104[v124];
                float v126;
                v126 = v122 + v125;
                v122 = v126;
                v121 += 1l ;
            }
            auto v127 = cooperative_groups::coalesced_threads();
            int v128;
            v128 = threadIdx.x;
            int v129;
            v129 = v128 / 4l;
            auto v130 = cooperative_groups::labeled_partition(v127,v129);
            Closure1 v131{};
            float v132;
            v132 = cooperative_groups::inclusive_scan(v130, v122, v131);
            float v133;
            v133 = v130.shfl_up(v132,1);
            bool v134;
            v134 = v130.thread_rank() == 0;
            float v135;
            if (v134){
                v135 = 0.0f;
            } else {
                v135 = v133;
            }
            float v136;
            v136 = v130.shfl(v132,v130.num_threads()-1);
            float v137;
            v137 = v117 + v135;
            int v138; float v139;
            Tuple0 tmp1 = Tuple0{0l, v137};
            v138 = tmp1.v0; v139 = tmp1.v1;
            while (while_method_1(v138)){
                assert("Tensor range check" && 0 <= v138 && v138 < 4l);
                int v141;
                v141 = v138 + v120;
                float v142;
                v142 = v104[v141];
                float v143;
                v143 = v139 + v142;
                assert("Tensor range check" && 0 <= v138 && v138 < 4l);
                v116[v141] = v143;
                v139 = v143;
                v138 += 1l ;
            }
            float v144;
            v144 = v117 + v136;
            v117 = v144;
            v118 += 1l ;
        }
        float v145;
        v145 = curand_uniform(&v2);
        float v146[4l];
        int v147;
        v147 = 0l;
        while (while_method_0(v147)){
            int v149;
            v149 = 0l;
            while (while_method_1(v149)){
                assert("Tensor range check" && 0 <= v147 && v147 < 1l);
                assert("Tensor range check" && 0 <= v149 && v149 < 4l);
                int v151;
                v151 = 4l * v147;
                int v152;
                v152 = v151 + v149;
                float v153;
                v153 = v116[v152];
                float v154;
                v154 = v153 - v145;
                assert("Tensor range check" && 0 <= v147 && v147 < 1l);
                assert("Tensor range check" && 0 <= v149 && v149 < 4l);
                v146[v152] = v154;
                v149 += 1l ;
            }
            v147 += 1l ;
        }
        float v155; int v156;
        Tuple1 tmp2 = Tuple1{-1.0f / 0.0f, 0l};
        v155 = tmp2.v0; v156 = tmp2.v1;
        int v157;
        v157 = 0l;
        while (while_method_0(v157)){
            int v159;
            v159 = 0l;
            while (while_method_1(v159)){
                assert("Tensor range check" && 0 <= v157 && v157 < 1l);
                assert("Tensor range check" && 0 <= v159 && v159 < 4l);
                int v161;
                v161 = 4l * v157;
                int v162;
                v162 = v161 + v159;
                float v163;
                v163 = v146[v162];
                int v164;
                v164 = v20[v162];
                bool v165;
                v165 = v155 >= 0.0f;
                bool v167;
                if (v165){
                    bool v166;
                    v166 = v163 >= 0.0f;
                    v167 = v166;
                } else {
                    v167 = false;
                }
                float v176; int v177;
                if (v167){
                    bool v168;
                    v168 = v155 <= v163;
                    if (v168){
                        v176 = v155; v177 = v156;
                    } else {
                        v176 = v163; v177 = v164;
                    }
                } else {
                    if (v165){
                        v176 = v155; v177 = v156;
                    } else {
                        bool v171;
                        v171 = v163 >= 0.0f;
                        if (v171){
                            v176 = v163; v177 = v164;
                        } else {
                            v176 = v155; v177 = v156;
                        }
                    }
                }
                v155 = v176;
                v156 = v177;
                v159 += 1l ;
            }
            v157 += 1l ;
        }
        auto v178 = cooperative_groups::coalesced_threads();
        int v179;
        v179 = threadIdx.x;
        int v180;
        v180 = v179 / 4l;
        auto v181 = cooperative_groups::labeled_partition(v178,v180);
        Closure2 v182{};
        float v183; int v184;
        Tuple1 tmp3 = cooperative_groups::reduce(v181, Tuple1{v155, v156}, v182);
        v183 = tmp3.v0; v184 = tmp3.v1;
        assert("Tensor range check" && 0 <= v15 && v15 < 2l);
        int v185;
        v185 = 8l * v15;
        int v186;
        v186 = v185 + v8;
        v0[v186] = v184;
        v15 += 1l ;
    }
    __syncthreads();
    return ;
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
    float * v9;
    v9 = reinterpret_cast<float *>(&v0[0ull]);
    float * v11;
    v11 = reinterpret_cast<float *>(&v1[0ull]);
    float * v13;
    v13 = reinterpret_cast<float *>(&v0[512ull]);
    method_0(v13, v11, v9);
    float * v15;
    v15 = reinterpret_cast<float *>(&v0[1536ull]);
    method_1(v15, v13);
    float * v17;
    v17 = reinterpret_cast<float *>(&v0[2560ull]);
    method_2(v17, v15);
    float * v19;
    v19 = reinterpret_cast<float *>(&v1[512ull]);
    float * v21;
    v21 = reinterpret_cast<float *>(&v0[3584ull]);
    method_3(v21, v19, v17);
    float * v23;
    v23 = reinterpret_cast<float *>(&v0[4608ull]);
    method_1(v23, v21);
    float * v25;
    v25 = reinterpret_cast<float *>(&v0[5632ull]);
    method_2(v25, v23);
    float * v27;
    v27 = reinterpret_cast<float *>(&v1[1536ull]);
    float * v29;
    v29 = reinterpret_cast<float *>(&v0[6656ull]);
    method_3(v29, v27, v25);
    int * v31;
    v31 = reinterpret_cast<int *>(&v0[7680ull]);
    return method_4(v31, v29, v8);
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
def method1(v0 : cp.ndarray, v1 : i32, v2 : i32, v3 : i32, v4 : i32, v5 : i32) -> None:
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
            method4(v30)
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
def method6(v0 : i32) -> None:
    print(v0, end="")
    del v0
    return 
def method5(v0 : cp.ndarray, v1 : i32, v2 : i32, v3 : i32) -> None:
    v4 = 0
    v5 = '['
    method2(v5)
    del v5
    v6 = 0
    while method3(v3, v6):
        v8 = v4
        v9 = v8 >= 100
        del v8
        if v9:
            v10 = " ..."
            method0(v10)
            del v10
            break
        else:
            pass
        del v9
        v11 = v6 == 0
        v12 = v11 != True
        del v11
        if v12:
            v13 = "; "
            method0(v13)
        else:
            pass
        del v12
        v14 = v4 + 1
        v4 = v14
        del v14
        v15 = v6 * v2
        v16 = v1 + v15
        del v15
        v17 = v0[v16].item()
        del v16
        method6(v17)
        del v17
        v6 += 1 
    del v0, v1, v2, v3, v4, v6
    v18 = ']'
    return method2(v18)
def main():
    v0 = cp.empty(2560,dtype=cp.uint8)
    v1 = cp.empty(7744,dtype=cp.uint8)
    v3 = v0[0:0+4*128].view(cp.float32)
    v4 = cp.random.normal(0.0,1.0,128,dtype=cp.float32) # type: ignore
    cp.copyto(v3[0:0+128],v4[0:0+128])
    del v3, v4
    v6 = v0[512:512+4*256].view(cp.float32)
    v7 = cp.random.normal(0.0,1.0,256,dtype=cp.float32) # type: ignore
    cp.copyto(v6[0:0+256],v7[0:0+256])
    del v6, v7
    v9 = v0[1536:1536+4*256].view(cp.float32)
    v10 = cp.random.normal(0.0,1.0,256,dtype=cp.float32) # type: ignore
    cp.copyto(v9[0:0+256],v10[0:0+256])
    del v9, v10
    v11 = "Here are the weight matrices."
    method0(v11)
    del v11
    print()
    v13 = v0[0:0+4*128].view(cp.float32)
    v14 = 0
    v15 = 8
    v16 = 1
    v17 = 16
    v18 = 8
    method1(v13, v14, v15, v16, v17, v18)
    del v13, v14, v15, v16, v17, v18
    print()
    v20 = v0[512:512+4*256].view(cp.float32)
    v21 = 0
    v22 = 16
    v23 = 1
    v24 = 16
    v25 = 16
    method1(v20, v21, v22, v23, v24, v25)
    del v20, v21, v22, v23, v24, v25
    print()
    v27 = v0[1536:1536+4*256].view(cp.float32)
    v28 = 0
    v29 = 16
    v30 = 1
    v31 = 16
    v32 = 16
    method1(v27, v28, v29, v30, v31, v32)
    del v27, v28, v29, v30, v31, v32
    print()
    v34 = v1[0:0+4*128].view(cp.float32)
    v35 = cp.random.normal(0.0,1.0,128,dtype=cp.float32) # type: ignore
    cp.copyto(v34[0:0+128],v35[0:0+128])
    del v35
    v36 = 0
    v37 = 8
    v38 = 1
    v39 = 16
    v40 = 8
    method1(v34, v36, v37, v38, v39, v40)
    del v34, v36, v37, v38, v39, v40
    print()
    v41 = "Here is the output tensor."
    method0(v41)
    del v41
    print()
    v42 = 0
    v43 = raw_module.get_function(f"entry{v42}")
    del v42
    v43.max_dynamic_shared_size_bytes = 1536 
    v43((1,),(32,),(v1, v0),shared_mem=1536)
    del v0, v43
    v45 = v1[7680:7680+4*16].view(cp.int32)
    del v1
    v46 = 0
    v47 = 1
    v48 = 16
    method5(v45, v46, v47, v48)
    del v45, v46, v47, v48
    print()
    v49 = "===="
    method0(v49)
    del v49
    print()
    return 

if __name__ == '__main__': print(main())
