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

__device__ void method_0(float * v0, float * v1, int v2, float * v3);
__device__ void method_1(float * v0, float * v1);
__device__ void method_2(float * v0, float * v1);
__device__ void method_3(float * v0, float * v1, int v2, float * v3);
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
__device__ void method_0(float * v0, float * v1, int v2, float * v3){
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
    while (while_method_1(v63)){
        int v65;
        v65 = 0l;
        while (while_method_1(v65)){
            assert("Tensor range check" && 0 <= v63 && v63 < 1l);
            assert("Tensor range check" && 0 <= v65 && v65 < 1l);
            int v67;
            v67 = 16l * v65;
            int v68;
            v68 = 256l * v63;
            int v69;
            v69 = v68 + v67;
            float * v70;
            v70 = v0+v69;
            // Pushing the loop unrolling to: 0
            int v72;
            v72 = 0l;
            #pragma unroll
            while (while_method_1(v72)){
                int v74;
                v74 = 0l;
                #pragma unroll
                while (while_method_1(v74)){
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
                int v80;
                v80 = 128l * v63;
                assert("Tensor range check" && 0 <= v78 && v78 < 1l);
                int v81;
                v81 = 8l * v78;
                int v82;
                v82 = v81 + v80;
                float * v83;
                v83 = v3+v82;
                assert("Tensor range check" && 0 <= v65 && v65 < 1l);
                int v85;
                v85 = 128l * v65;
                int v86;
                v86 = v85 + v2;
                assert("Tensor range check" && 0 <= v78 && v78 < 1l);
                int v87;
                v87 = v81 + v86;
                float * v88;
                v88 = v1+v87;
                int v90;
                v90 = threadIdx.x;
                bool v91;
                v91 = 0l <= v90;
                bool v92;
                v92 = v91 == false;
                if (v92){
                    assert("The index needs to be zero or positive." && v91);
                } else {
                }
                int v94;
                v94 = v90 % 2l;
                int v95;
                v95 = v90 / 2l;
                bool v96;
                v96 = v95 < 16l;
                bool v97;
                v97 = v96 == false;
                if (v97){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v96);
                } else {
                }
                assert("Tensor range check" && 0 <= v95 && v95 < 16l);
                assert("Tensor range check" && 0 <= v94 && v94 < 2l);
                int v99;
                v99 = 4l * v94;
                int v100;
                v100 = 12l * v95;
                int v101;
                v101 = v100 + v99;
                int v102;
                v102 = 8l * v95;
                int v103;
                v103 = v102 + v99;
                float * v104;
                v104 = v12+v101;
                float * v106;
                v106 = v88+v103;
                int v108;
                v108 = 0l;
                #pragma unroll
                while (while_method_1(v108)){
                    int v110;
                    v110 = 0l;
                    #pragma unroll
                    while (while_method_1(v110)){
                        assert("Tensor range check" && 0 <= v108 && v108 < 1l);
                        assert("Tensor range check" && 0 <= v110 && v110 < 1l);
                        int v112;
                        v112 = 8l * v110;
                        int v113;
                        v113 = 192l * v108;
                        int v114;
                        v114 = v113 + v112;
                        int v115;
                        v115 = 128l * v108;
                        int v116;
                        v116 = v115 + v112;
                        float v117[4l];
                        int v118;
                        v118 = 0l;
                        #pragma unroll
                        while (while_method_2(v118)){
                            assert("Tensor range check" && 0 <= v118 && v118 < 4l);
                            int v120;
                            v120 = v118 + v116;
                            float v121;
                            v121 = v106[v120];
                            float v122;
                            v122 = wmma::__float_to_tf32(v121);
                            assert("Tensor range check" && 0 <= v118 && v118 < 4l);
                            v117[v118] = v122;
                            v118 += 1l ;
                        }
                        int4* v123;
                        v123 = reinterpret_cast<int4*>(v117 + 0l);
                        int4* v124;
                        v124 = reinterpret_cast<int4*>(v104 + v114);
                        assert("Pointer alignment check" && (unsigned long long)(v123) % 4l == 0 && (unsigned long long)(v124) % 4l == 0);
                        *v124 = *v123;
                        v110 += 1l ;
                    }
                    v108 += 1l ;
                }
                int v125;
                v125 = threadIdx.x;
                bool v126;
                v126 = 0l <= v125;
                bool v127;
                v127 = v126 == false;
                if (v127){
                    assert("The index needs to be zero or positive." && v126);
                } else {
                }
                int v129;
                v129 = v125 % 2l;
                int v130;
                v130 = v125 / 2l;
                bool v131;
                v131 = v130 < 16l;
                bool v132;
                v132 = v131 == false;
                if (v132){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v131);
                } else {
                }
                assert("Tensor range check" && 0 <= v130 && v130 < 16l);
                assert("Tensor range check" && 0 <= v129 && v129 < 2l);
                int v134;
                v134 = 4l * v129;
                int v135;
                v135 = 12l * v130;
                int v136;
                v136 = v135 + v134;
                int v137;
                v137 = 8l * v130;
                int v138;
                v138 = v137 + v134;
                float * v139;
                v139 = v10+v136;
                float * v141;
                v141 = v83+v138;
                int v143;
                v143 = 0l;
                #pragma unroll
                while (while_method_1(v143)){
                    int v145;
                    v145 = 0l;
                    #pragma unroll
                    while (while_method_1(v145)){
                        assert("Tensor range check" && 0 <= v143 && v143 < 1l);
                        assert("Tensor range check" && 0 <= v145 && v145 < 1l);
                        int v147;
                        v147 = 8l * v145;
                        int v148;
                        v148 = 192l * v143;
                        int v149;
                        v149 = v148 + v147;
                        int v150;
                        v150 = 128l * v143;
                        int v151;
                        v151 = v150 + v147;
                        float v152[4l];
                        int v153;
                        v153 = 0l;
                        #pragma unroll
                        while (while_method_2(v153)){
                            assert("Tensor range check" && 0 <= v153 && v153 < 4l);
                            int v155;
                            v155 = v153 + v151;
                            float v156;
                            v156 = v141[v155];
                            float v157;
                            v157 = wmma::__float_to_tf32(v156);
                            assert("Tensor range check" && 0 <= v153 && v153 < 4l);
                            v152[v153] = v157;
                            v153 += 1l ;
                        }
                        int4* v158;
                        v158 = reinterpret_cast<int4*>(v152 + 0l);
                        int4* v159;
                        v159 = reinterpret_cast<int4*>(v139 + v149);
                        assert("Pointer alignment check" && (unsigned long long)(v158) % 4l == 0 && (unsigned long long)(v159) % 4l == 0);
                        *v159 = *v158;
                        v145 += 1l ;
                    }
                    v143 += 1l ;
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v160[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v161[1l];
                int v162;
                v162 = 0l;
                #pragma unroll
                while (while_method_1(v162)){
                    int v164;
                    v164 = 0l;
                    #pragma unroll
                    while (while_method_1(v164)){
                        assert("Tensor range check" && 0 <= v162 && v162 < 1l);
                        assert("Tensor range check" && 0 <= v164 && v164 < 1l);
                        int v166;
                        v166 = v162 + v164;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v167 = v160[v166];
                        assert("Tensor range check" && 0 <= v162 && v162 < 1l);
                        int v168;
                        v168 = 192l * v162;
                        assert("Tensor range check" && 0 <= v164 && v164 < 1l);
                        int v169;
                        v169 = 8l * v164;
                        int v170;
                        v170 = v169 + v168;
                        int v171;
                        v171 = 0l;
                        #pragma unroll
                        while (while_method_3(v171)){
                            int v173;
                            v173 = 0l;
                            #pragma unroll
                            while (while_method_3(v173)){
                                assert("Tensor range check" && 0 <= v171 && v171 < 2l);
                                assert("Tensor range check" && 0 <= v173 && v173 < 2l);
                                int v175;
                                v175 = 96l * v173;
                                int v176;
                                v176 = v175 + v170;
                                int v177;
                                v177 = 4l * v171;
                                int v178;
                                v178 = v177 + v176;
                                float v179;
                                v179 = v44[v178];
                                bool v180;
                                v180 = 0l <= v173;
                                bool v182;
                                if (v180){
                                    bool v181;
                                    v181 = v173 < 2l;
                                    v182 = v181;
                                } else {
                                    v182 = false;
                                }
                                bool v183;
                                v183 = v182 == false;
                                if (v183){
                                    assert("The indices should be inside the range of the dimension." && v182);
                                } else {
                                }
                                bool v185;
                                v185 = 0l <= v171;
                                bool v187;
                                if (v185){
                                    bool v186;
                                    v186 = v171 < 2l;
                                    v187 = v186;
                                } else {
                                    v187 = false;
                                }
                                bool v188;
                                v188 = v187 == false;
                                if (v188){
                                    assert("The indices should be inside the range of the dimension." && v187);
                                } else {
                                }
                                int v190;
                                v190 = v171 * 2l;
                                int v191;
                                v191 = v173 + v190;
                                v167.x[v191] = v179;
                                v173 += 1l ;
                            }
                            v171 += 1l ;
                        }
                        v164 += 1l ;
                    }
                    v162 += 1l ;
                }
                int v192;
                v192 = 0l;
                #pragma unroll
                while (while_method_1(v192)){
                    int v194;
                    v194 = 0l;
                    #pragma unroll
                    while (while_method_1(v194)){
                        assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                        assert("Tensor range check" && 0 <= v194 && v194 < 1l);
                        int v196;
                        v196 = v192 + v194;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v197 = v161[v196];
                        assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                        int v198;
                        v198 = 192l * v192;
                        assert("Tensor range check" && 0 <= v194 && v194 < 1l);
                        int v199;
                        v199 = 8l * v194;
                        int v200;
                        v200 = v199 + v198;
                        int v201;
                        v201 = 0l;
                        #pragma unroll
                        while (while_method_3(v201)){
                            int v203;
                            v203 = 0l;
                            #pragma unroll
                            while (while_method_3(v203)){
                                assert("Tensor range check" && 0 <= v201 && v201 < 2l);
                                assert("Tensor range check" && 0 <= v203 && v203 < 2l);
                                int v205;
                                v205 = 4l * v203;
                                int v206;
                                v206 = v205 + v200;
                                int v207;
                                v207 = 96l * v201;
                                int v208;
                                v208 = v207 + v206;
                                float v209;
                                v209 = v60[v208];
                                bool v210;
                                v210 = 0l <= v203;
                                bool v212;
                                if (v210){
                                    bool v211;
                                    v211 = v203 < 2l;
                                    v212 = v211;
                                } else {
                                    v212 = false;
                                }
                                bool v213;
                                v213 = v212 == false;
                                if (v213){
                                    assert("The indices should be inside the range of the dimension." && v212);
                                } else {
                                }
                                bool v215;
                                v215 = 0l <= v201;
                                bool v217;
                                if (v215){
                                    bool v216;
                                    v216 = v201 < 2l;
                                    v217 = v216;
                                } else {
                                    v217 = false;
                                }
                                bool v218;
                                v218 = v217 == false;
                                if (v218){
                                    assert("The indices should be inside the range of the dimension." && v217);
                                } else {
                                }
                                int v220;
                                v220 = v201 * 2l;
                                int v221;
                                v221 = v203 + v220;
                                v197.x[v221] = v209;
                                v203 += 1l ;
                            }
                            v201 += 1l ;
                        }
                        v194 += 1l ;
                    }
                    v192 += 1l ;
                }
                __syncthreads();
                int v222;
                v222 = 0l;
                #pragma unroll
                while (while_method_1(v222)){
                    int v224;
                    v224 = 0l;
                    #pragma unroll
                    while (while_method_1(v224)){
                        int v226;
                        v226 = 0l;
                        #pragma unroll
                        while (while_method_1(v226)){
                            assert("Tensor range check" && 0 <= v222 && v222 < 1l);
                            assert("Tensor range check" && 0 <= v224 && v224 < 1l);
                            int v228;
                            v228 = v222 + v224;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v229 = v62[v228];
                            assert("Tensor range check" && 0 <= v222 && v222 < 1l);
                            assert("Tensor range check" && 0 <= v226 && v226 < 1l);
                            int v230;
                            v230 = v222 + v226;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v231 = v160[v230];
                            assert("Tensor range check" && 0 <= v224 && v224 < 1l);
                            assert("Tensor range check" && 0 <= v226 && v226 < 1l);
                            int v232;
                            v232 = v224 + v226;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v233 = v161[v232];
                            wmma::mma_sync(v229, v231, v233, v229);
                            v226 += 1l ;
                        }
                        v224 += 1l ;
                    }
                    v222 += 1l ;
                }
                v78 += 1l ;
            }
            int v234;
            v234 = 0l;
            #pragma unroll
            while (while_method_1(v234)){
                int v236;
                v236 = 0l;
                #pragma unroll
                while (while_method_1(v236)){
                    assert("Tensor range check" && 0 <= v234 && v234 < 1l);
                    assert("Tensor range check" && 0 <= v236 && v236 < 1l);
                    int v238;
                    v238 = v234 + v236;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v239 = v62[v238];
                    assert("Tensor range check" && 0 <= v234 && v234 < 1l);
                    assert("Tensor range check" && 0 <= v236 && v236 < 1l);
                    int v240;
                    v240 = 16l * v236;
                    int v241;
                    v241 = 384l * v234;
                    int v242;
                    v242 = v241 + v240;
                    float * v243;
                    v243 = v28+v242;
                    wmma::store_matrix_sync(v243, v239, 24l, wmma::mem_row_major);
                    v236 += 1l ;
                }
                v234 += 1l ;
            }
            __syncthreads();
            int v245;
            v245 = threadIdx.x;
            bool v246;
            v246 = 0l <= v245;
            bool v247;
            v247 = v246 == false;
            if (v247){
                assert("The index needs to be zero or positive." && v246);
            } else {
            }
            int v249;
            v249 = v245 % 4l;
            int v250;
            v250 = v245 / 4l;
            bool v251;
            v251 = v250 < 8l;
            bool v252;
            v252 = v251 == false;
            if (v252){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v251);
            } else {
            }
            assert("Tensor range check" && 0 <= v250 && v250 < 8l);
            assert("Tensor range check" && 0 <= v249 && v249 < 4l);
            int v254;
            v254 = 4l * v249;
            int v255;
            v255 = 16l * v250;
            int v256;
            v256 = v255 + v254;
            int v257;
            v257 = 24l * v250;
            int v258;
            v258 = v257 + v254;
            float * v259;
            v259 = v70+v256;
            float * v261;
            v261 = v14+v258;
            int v263;
            v263 = 0l;
            #pragma unroll
            while (while_method_3(v263)){
                int v265;
                v265 = 0l;
                #pragma unroll
                while (while_method_1(v265)){
                    assert("Tensor range check" && 0 <= v263 && v263 < 2l);
                    assert("Tensor range check" && 0 <= v265 && v265 < 1l);
                    int v267;
                    v267 = 16l * v265;
                    int v268;
                    v268 = 128l * v263;
                    int v269;
                    v269 = v268 + v267;
                    int v270;
                    v270 = 192l * v263;
                    int v271;
                    v271 = v270 + v267;
                    int4* v272;
                    v272 = reinterpret_cast<int4*>(v261 + v271);
                    int4* v273;
                    v273 = reinterpret_cast<int4*>(v259 + v269);
                    assert("Pointer alignment check" && (unsigned long long)(v272) % 4l == 0 && (unsigned long long)(v273) % 4l == 0);
                    *v273 = *v272;
                    v265 += 1l ;
                }
                v263 += 1l ;
            }
            __syncthreads();
            // Poping the loop unrolling to: 0
            v65 += 1l ;
        }
        v63 += 1l ;
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
    while (while_method_3(v14)){
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
        while (while_method_1(v20)){
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
        while (while_method_1(v27)){
            int v29;
            v29 = 0l;
            while (while_method_2(v29)){
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
        while (while_method_1(v64)){
            int v66;
            v66 = 0l;
            while (while_method_2(v66)){
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
        while (while_method_1(v73)){
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
        while (while_method_1(v88)){
            int v90;
            v90 = 0l;
            while (while_method_2(v90)){
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
        while (while_method_1(v99)){
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
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 64l;
    return v1;
}
__device__ void method_2(float * v0, float * v1){
    int v2;
    v2 = threadIdx.x;
    int v3;
    v3 = v2;
    while (while_method_4(v3)){
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
        while (while_method_2(v20)){
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
    while (while_method_1(v63)){
        int v65;
        v65 = 0l;
        while (while_method_1(v65)){
            assert("Tensor range check" && 0 <= v63 && v63 < 1l);
            assert("Tensor range check" && 0 <= v65 && v65 < 1l);
            int v67;
            v67 = 16l * v65;
            int v68;
            v68 = 256l * v63;
            int v69;
            v69 = v68 + v67;
            float * v70;
            v70 = v0+v69;
            // Pushing the loop unrolling to: 0
            int v72;
            v72 = 0l;
            #pragma unroll
            while (while_method_1(v72)){
                int v74;
                v74 = 0l;
                #pragma unroll
                while (while_method_1(v74)){
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
            while (while_method_3(v78)){
                assert("Tensor range check" && 0 <= v63 && v63 < 1l);
                assert("Tensor range check" && 0 <= v78 && v78 < 2l);
                int v80;
                v80 = 8l * v78;
                int v81;
                v81 = v80 + v68;
                float * v82;
                v82 = v3+v81;
                assert("Tensor range check" && 0 <= v65 && v65 < 1l);
                int v84;
                v84 = 256l * v65;
                int v85;
                v85 = v84 + v2;
                assert("Tensor range check" && 0 <= v78 && v78 < 2l);
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
                v101 = 16l * v94;
                int v102;
                v102 = v101 + v98;
                float * v103;
                v103 = v12+v100;
                float * v105;
                v105 = v87+v102;
                int v107;
                v107 = 0l;
                #pragma unroll
                while (while_method_1(v107)){
                    int v109;
                    v109 = 0l;
                    #pragma unroll
                    while (while_method_1(v109)){
                        assert("Tensor range check" && 0 <= v107 && v107 < 1l);
                        assert("Tensor range check" && 0 <= v109 && v109 < 1l);
                        int v111;
                        v111 = 8l * v109;
                        int v112;
                        v112 = 192l * v107;
                        int v113;
                        v113 = v112 + v111;
                        int v114;
                        v114 = 256l * v107;
                        int v115;
                        v115 = v114 + v111;
                        float v116[4l];
                        int v117;
                        v117 = 0l;
                        #pragma unroll
                        while (while_method_2(v117)){
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
                v136 = 16l * v129;
                int v137;
                v137 = v136 + v133;
                float * v138;
                v138 = v10+v135;
                float * v140;
                v140 = v82+v137;
                int v142;
                v142 = 0l;
                #pragma unroll
                while (while_method_1(v142)){
                    int v144;
                    v144 = 0l;
                    #pragma unroll
                    while (while_method_1(v144)){
                        assert("Tensor range check" && 0 <= v142 && v142 < 1l);
                        assert("Tensor range check" && 0 <= v144 && v144 < 1l);
                        int v146;
                        v146 = 8l * v144;
                        int v147;
                        v147 = 192l * v142;
                        int v148;
                        v148 = v147 + v146;
                        int v149;
                        v149 = 256l * v142;
                        int v150;
                        v150 = v149 + v146;
                        float v151[4l];
                        int v152;
                        v152 = 0l;
                        #pragma unroll
                        while (while_method_2(v152)){
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
                while (while_method_1(v161)){
                    int v163;
                    v163 = 0l;
                    #pragma unroll
                    while (while_method_1(v163)){
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
                        while (while_method_3(v170)){
                            int v172;
                            v172 = 0l;
                            #pragma unroll
                            while (while_method_3(v172)){
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
                while (while_method_1(v191)){
                    int v193;
                    v193 = 0l;
                    #pragma unroll
                    while (while_method_1(v193)){
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
                        while (while_method_3(v200)){
                            int v202;
                            v202 = 0l;
                            #pragma unroll
                            while (while_method_3(v202)){
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
                while (while_method_1(v221)){
                    int v223;
                    v223 = 0l;
                    #pragma unroll
                    while (while_method_1(v223)){
                        int v225;
                        v225 = 0l;
                        #pragma unroll
                        while (while_method_1(v225)){
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
            while (while_method_1(v233)){
                int v235;
                v235 = 0l;
                #pragma unroll
                while (while_method_1(v235)){
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
            v254 = 16l * v249;
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
            while (while_method_3(v262)){
                int v264;
                v264 = 0l;
                #pragma unroll
                while (while_method_1(v264)){
                    assert("Tensor range check" && 0 <= v262 && v262 < 2l);
                    assert("Tensor range check" && 0 <= v264 && v264 < 1l);
                    int v266;
                    v266 = 16l * v264;
                    int v267;
                    v267 = 128l * v262;
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
__device__ void method_4(float * v0, int v1, float * v2){
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
    assert("Tensor range check" && 0 <= v7 && v7 < 4l);
    int v15;
    v15 = v12 + v1;
    int v16;
    v16 = v13 + v15;
    int v17;
    v17 = 0l;
    while (while_method_3(v17)){
        assert("Tensor range check" && 0 <= v17 && v17 < 2l);
        int v19;
        v19 = 128l * v17;
        int v20;
        v20 = v19 + v14;
        assert("Tensor range check" && 0 <= v17 && v17 < 2l);
        int v21;
        v21 = v19 + v16;
        float v22[4l];
        int v23[4l];
        int v24;
        v24 = 0l;
        while (while_method_1(v24)){
            assert("Tensor range check" && 0 <= v24 && v24 < 1l);
            int v26;
            v26 = 4l * v24;
            assert("Tensor range check" && 0 <= v24 && v24 < 1l);
            int v27;
            v27 = 16l * v24;
            int v28;
            v28 = v27 + v20;
            int4* v29;
            v29 = reinterpret_cast<int4*>(v2 + v28);
            int4* v30;
            v30 = reinterpret_cast<int4*>(v22 + v26);
            assert("Pointer alignment check" && (unsigned long long)(v29) % 4l == 0 && (unsigned long long)(v30) % 4l == 0);
            *v30 = *v29;
            v24 += 1l ;
        }
        int v31;
        v31 = 0l;
        while (while_method_1(v31)){
            int v33;
            v33 = 0l;
            while (while_method_2(v33)){
                bool v35;
                v35 = 0l <= v33;
                bool v37;
                if (v35){
                    bool v36;
                    v36 = v33 < 4l;
                    v37 = v36;
                } else {
                    v37 = false;
                }
                bool v38;
                v38 = v37 == false;
                if (v38){
                    assert("The indices should be inside the range of the dimension." && v37);
                } else {
                }
                bool v40;
                v40 = 0l <= v7;
                bool v42;
                if (v40){
                    bool v41;
                    v41 = v7 < 4l;
                    v42 = v41;
                } else {
                    v42 = false;
                }
                bool v43;
                v43 = v42 == false;
                if (v43){
                    assert("The indices should be inside the range of the dimension." && v42);
                } else {
                }
                int v45;
                v45 = v7 * 4l;
                int v46;
                v46 = v33 + v45;
                bool v47;
                v47 = 0l <= v31;
                bool v49;
                if (v47){
                    bool v48;
                    v48 = v31 < 1l;
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
                v52 = v31 * 16l;
                int v53;
                v53 = v46 + v52;
                assert("Tensor range check" && 0 <= v31 && v31 < 1l);
                assert("Tensor range check" && 0 <= v33 && v33 < 4l);
                int v54;
                v54 = 4l * v31;
                int v55;
                v55 = v54 + v33;
                v23[v55] = v53;
                v33 += 1l ;
            }
            v31 += 1l ;
        }
        bool v56;
        v56 = 0l <= v8;
        bool v57;
        v57 = v56 && v9;
        bool v58;
        v58 = v57 == false;
        if (v58){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v57);
        } else {
        }
        bool v60;
        v60 = 0l <= v17;
        bool v62;
        if (v60){
            bool v61;
            v61 = v17 < 2l;
            v62 = v61;
        } else {
            v62 = false;
        }
        bool v63;
        v63 = v62 == false;
        if (v63){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v62);
        } else {
        }
        int v65;
        v65 = v17 * 8l;
        int v66;
        v66 = v65 + v8;
        float v67;
        v67 = 0.0f;
        int v68;
        v68 = 0l;
        while (while_method_1(v68)){
            int v70;
            v70 = 0l;
            while (while_method_2(v70)){
                assert("Tensor range check" && 0 <= v68 && v68 < 1l);
                assert("Tensor range check" && 0 <= v70 && v70 < 4l);
                int v72;
                v72 = 4l * v68;
                int v73;
                v73 = v72 + v70;
                float v74;
                v74 = v22[v73];
                float v75;
                v75 = v67 + v74;
                v67 = v75;
                v70 += 1l ;
            }
            v68 += 1l ;
        }
        auto v76 = cooperative_groups::coalesced_threads();
        int v77;
        v77 = threadIdx.x;
        int v78;
        v78 = v77 / 4l;
        auto v79 = cooperative_groups::labeled_partition(v76,v78);
        Closure0 v80{};
        float v81;
        v81 = cooperative_groups::reduce(v79, v67, v80);
        float v82;
        v82 = v81 / 16.0f;
        float v83[4l];
        int v84;
        v84 = 0l;
        while (while_method_1(v84)){
            int v86;
            v86 = 0l;
            while (while_method_2(v86)){
                assert("Tensor range check" && 0 <= v84 && v84 < 1l);
                assert("Tensor range check" && 0 <= v86 && v86 < 4l);
                int v88;
                v88 = 4l * v84;
                int v89;
                v89 = v88 + v86;
                float v90;
                v90 = v22[v89];
                float v91;
                v91 = v90 - v82;
                float v92;
                v92 = exp(v91);
                assert("Tensor range check" && 0 <= v84 && v84 < 1l);
                assert("Tensor range check" && 0 <= v86 && v86 < 4l);
                v83[v89] = v92;
                v86 += 1l ;
            }
            v84 += 1l ;
        }
        float v93;
        v93 = 0.0f;
        int v94;
        v94 = 0l;
        while (while_method_1(v94)){
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
                v100 = v83[v99];
                float v101;
                v101 = v93 + v100;
                v93 = v101;
                v96 += 1l ;
            }
            v94 += 1l ;
        }
        auto v102 = cooperative_groups::coalesced_threads();
        int v103;
        v103 = threadIdx.x;
        int v104;
        v104 = v103 / 4l;
        auto v105 = cooperative_groups::labeled_partition(v102,v104);
        float v106;
        v106 = cooperative_groups::reduce(v105, v93, v80);
        float v107[4l];
        int v108;
        v108 = 0l;
        while (while_method_1(v108)){
            int v110;
            v110 = 0l;
            while (while_method_2(v110)){
                assert("Tensor range check" && 0 <= v108 && v108 < 1l);
                assert("Tensor range check" && 0 <= v110 && v110 < 4l);
                int v112;
                v112 = 4l * v108;
                int v113;
                v113 = v112 + v110;
                float v114;
                v114 = v83[v113];
                bool v115;
                v115 = v106 == 0.0f;
                bool v116;
                v116 = v115 != true;
                float v118;
                if (v116){
                    float v117;
                    v117 = v114 / v106;
                    v118 = v117;
                } else {
                    v118 = 0.0625f;
                }
                assert("Tensor range check" && 0 <= v108 && v108 < 1l);
                assert("Tensor range check" && 0 <= v110 && v110 < 4l);
                v107[v113] = v118;
                v110 += 1l ;
            }
            v108 += 1l ;
        }
        int v119;
        v119 = 0l;
        while (while_method_1(v119)){
            assert("Tensor range check" && 0 <= v119 && v119 < 1l);
            int v121;
            v121 = 16l * v119;
            int v122;
            v122 = v121 + v21;
            assert("Tensor range check" && 0 <= v119 && v119 < 1l);
            int v123;
            v123 = 4l * v119;
            int4* v124;
            v124 = reinterpret_cast<int4*>(v107 + v123);
            int4* v125;
            v125 = reinterpret_cast<int4*>(v0 + v122);
            assert("Pointer alignment check" && (unsigned long long)(v124) % 4l == 0 && (unsigned long long)(v125) % 4l == 0);
            *v125 = *v124;
            v119 += 1l ;
        }
        v17 += 1l ;
    }
    __syncthreads();
    return ;
}
__device__ void method_5(int * v0, int v1, float * v2, int v3, curandStatePhilox4_32_10_t & v4){
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
    v15 = v14 + v3;
    int v16;
    v16 = 16l * v10;
    int v17;
    v17 = v16 + v15;
    assert("Tensor range check" && 0 <= v10 && v10 < 8l);
    int v18;
    v18 = v10 + v1;
    int v19;
    v19 = 0l;
    while (while_method_3(v19)){
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
            v30 = reinterpret_cast<int4*>(v2 + v29);
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
            while (while_method_2(v34)){
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
        float v68[4l];
        float v69;
        v69 = 0.0f;
        int v70;
        v70 = 0l;
        while (while_method_1(v70)){
            assert("Tensor range check" && 0 <= v70 && v70 < 1l);
            int v72;
            v72 = 4l * v70;
            assert("Tensor range check" && 0 <= v70 && v70 < 1l);
            int v73; float v74;
            Tuple0 tmp0 = Tuple0{0l, 0.0f};
            v73 = tmp0.v0; v74 = tmp0.v1;
            while (while_method_2(v73)){
                assert("Tensor range check" && 0 <= v73 && v73 < 4l);
                int v76;
                v76 = v73 + v72;
                float v77;
                v77 = v23[v76];
                float v78;
                v78 = v74 + v77;
                v74 = v78;
                v73 += 1l ;
            }
            auto v79 = cooperative_groups::coalesced_threads();
            int v80;
            v80 = threadIdx.x;
            int v81;
            v81 = v80 / 4l;
            auto v82 = cooperative_groups::labeled_partition(v79,v81);
            Closure1 v83{};
            float v84;
            v84 = cooperative_groups::inclusive_scan(v82, v74, v83);
            float v85;
            v85 = v82.shfl_up(v84,1);
            bool v86;
            v86 = v82.thread_rank() == 0;
            float v87;
            if (v86){
                v87 = 0.0f;
            } else {
                v87 = v85;
            }
            float v88;
            v88 = v82.shfl(v84,v82.num_threads()-1);
            float v89;
            v89 = v69 + v87;
            int v90; float v91;
            Tuple0 tmp1 = Tuple0{0l, v89};
            v90 = tmp1.v0; v91 = tmp1.v1;
            while (while_method_2(v90)){
                assert("Tensor range check" && 0 <= v90 && v90 < 4l);
                int v93;
                v93 = v90 + v72;
                float v94;
                v94 = v23[v93];
                float v95;
                v95 = v91 + v94;
                assert("Tensor range check" && 0 <= v90 && v90 < 4l);
                v68[v93] = v95;
                v91 = v95;
                v90 += 1l ;
            }
            float v96;
            v96 = v69 + v88;
            v69 = v96;
            v70 += 1l ;
        }
        float v97;
        v97 = curand_uniform(&v4);
        float v98[4l];
        int v99;
        v99 = 0l;
        while (while_method_1(v99)){
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
                v105 = v68[v104];
                float v106;
                v106 = v105 - v97;
                assert("Tensor range check" && 0 <= v99 && v99 < 1l);
                assert("Tensor range check" && 0 <= v101 && v101 < 4l);
                v98[v104] = v106;
                v101 += 1l ;
            }
            v99 += 1l ;
        }
        float v107; int v108;
        Tuple1 tmp2 = Tuple1{-1.0f / 0.0f, 0l};
        v107 = tmp2.v0; v108 = tmp2.v1;
        int v109;
        v109 = 0l;
        while (while_method_1(v109)){
            int v111;
            v111 = 0l;
            while (while_method_2(v111)){
                assert("Tensor range check" && 0 <= v109 && v109 < 1l);
                assert("Tensor range check" && 0 <= v111 && v111 < 4l);
                int v113;
                v113 = 4l * v109;
                int v114;
                v114 = v113 + v111;
                float v115;
                v115 = v98[v114];
                int v116;
                v116 = v24[v114];
                bool v117;
                v117 = v107 >= 0.0f;
                bool v119;
                if (v117){
                    bool v118;
                    v118 = v115 >= 0.0f;
                    v119 = v118;
                } else {
                    v119 = false;
                }
                float v128; int v129;
                if (v119){
                    bool v120;
                    v120 = v107 <= v115;
                    if (v120){
                        v128 = v107; v129 = v108;
                    } else {
                        v128 = v115; v129 = v116;
                    }
                } else {
                    if (v117){
                        v128 = v107; v129 = v108;
                    } else {
                        bool v123;
                        v123 = v115 >= 0.0f;
                        if (v123){
                            v128 = v115; v129 = v116;
                        } else {
                            v128 = v107; v129 = v108;
                        }
                    }
                }
                v107 = v128;
                v108 = v129;
                v111 += 1l ;
            }
            v109 += 1l ;
        }
        auto v130 = cooperative_groups::coalesced_threads();
        int v131;
        v131 = threadIdx.x;
        int v132;
        v132 = v131 / 4l;
        auto v133 = cooperative_groups::labeled_partition(v130,v132);
        Closure2 v134{};
        float v135; int v136;
        Tuple1 tmp3 = cooperative_groups::reduce(v133, Tuple1{v107, v108}, v134);
        v135 = tmp3.v0; v136 = tmp3.v1;
        assert("Tensor range check" && 0 <= v19 && v19 < 2l);
        int v137;
        v137 = 8l * v19;
        int v138;
        v138 = v137 + v18;
        v0[v138] = v136;
        v19 += 1l ;
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
        method_0(v16, v13, v15, v11);
        float * v18;
        v18 = reinterpret_cast<float *>(&v0[1536ull]);
        method_1(v18, v16);
        float * v20;
        v20 = reinterpret_cast<float *>(&v0[2560ull]);
        method_2(v20, v18);
        float * v22;
        v22 = reinterpret_cast<float *>(&v1[8192ull]);
        assert("Tensor range check" && 0 <= v2 && v2 < 16l);
        int v24;
        v24 = 256l * v2;
        float * v25;
        v25 = reinterpret_cast<float *>(&v0[3584ull]);
        method_3(v25, v22, v24, v20);
        float * v27;
        v27 = reinterpret_cast<float *>(&v0[4608ull]);
        method_1(v27, v25);
        float * v29;
        v29 = reinterpret_cast<float *>(&v0[5632ull]);
        method_2(v29, v27);
        float * v31;
        v31 = reinterpret_cast<float *>(&v1[24576ull]);
        assert("Tensor range check" && 0 <= v2 && v2 < 16l);
        float * v33;
        v33 = reinterpret_cast<float *>(&v0[6656ull]);
        method_3(v33, v31, v24, v29);
        float * v35;
        v35 = reinterpret_cast<float *>(&v0[7680ull]);
        assert("Tensor range check" && 0 <= v2 && v2 < 16l);
        method_4(v35, v24, v33);
        int * v37;
        v37 = reinterpret_cast<int *>(&v0[24064ull]);
        assert("Tensor range check" && 0 <= v2 && v2 < 16l);
        int v39;
        v39 = 16l * v2;
        method_5(v37, v39, v35, v24, v10);
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
def method5(v0 : cp.ndarray, v1 : i32, v2 : i32, v3 : i32, v4 : i32, v5 : i32) -> None:
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
def method6(v0 : i32) -> bool:
    v1 = v0 < 16
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
    v46 = 8
    v47 = 1
    v48 = 16
    v49 = 8
    method5(v40, v45, v46, v47, v48, v49)
    del v40, v45, v46, v47, v48, v49
    print()
    v50 = "Here is the output tensor."
    method0(v50)
    del v50
    print()
    v52 = v1[7680:7680+4*4096].view(cp.float32)
    v53 = 0
    v54 = '['
    method2(v54)
    del v54
    v55 = 0
    while method6(v55):
        v57 = v53
        v58 = v57 >= 512
        del v57
        if v58:
            v59 = " ..."
            method0(v59)
            del v59
            break
        else:
            pass
        del v58
        v60 = v55 == 0
        v61 = v60 != True
        del v60
        if v61:
            v62 = "; "
            method0(v62)
        else:
            pass
        del v61
        v63 = '['
        method2(v63)
        del v63
        v64 = 0
        while method6(v64):
            v66 = v53
            v67 = v66 >= 512
            del v66
            if v67:
                v68 = " ..."
                method0(v68)
                del v68
                break
            else:
                pass
            del v67
            v69 = v64 == 0
            v70 = v69 != True
            del v69
            if v70:
                v71 = "; "
                method0(v71)
            else:
                pass
            del v70
            v72 = '['
            method2(v72)
            del v72
            v73 = 0
            while method6(v73):
                v75 = v53
                v76 = v75 >= 512
                del v75
                if v76:
                    v77 = " ..."
                    method0(v77)
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
                    method0(v80)
                else:
                    pass
                del v79
                v81 = v53 + 1
                v53 = v81
                del v81
                v82 = v55 * 256
                v83 = v64 * 16
                v84 = v82 + v83
                del v82, v83
                v85 = v84 + v73
                del v84
                v86 = v52[v85].item()
                del v85
                method4(v86)
                del v86
                v73 += 1 
            del v73
            v87 = ']'
            method2(v87)
            del v87
            v64 += 1 
        del v64
        v88 = ']'
        method2(v88)
        del v88
        v55 += 1 
    del v52, v53, v55
    v89 = ']'
    method2(v89)
    del v89
    method7()
    print()
    v91 = v1[24064:24064+4*256].view(cp.int32)
    del v1
    v92 = 0
    v93 = '['
    method2(v93)
    del v93
    v94 = 0
    while method6(v94):
        v96 = v92
        v97 = v96 >= 2147483647
        del v96
        if v97:
            v98 = " ..."
            method0(v98)
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
            method0(v101)
        else:
            pass
        del v100
        v102 = '['
        method2(v102)
        del v102
        v103 = 0
        while method6(v103):
            v105 = v92
            v106 = v105 >= 2147483647
            del v105
            if v106:
                v107 = " ..."
                method0(v107)
                del v107
                break
            else:
                pass
            del v106
            v108 = v103 == 0
            v109 = v108 != True
            del v108
            if v109:
                v110 = "; "
                method0(v110)
            else:
                pass
            del v109
            v111 = v92 + 1
            v92 = v111
            del v111
            v112 = v94 * 16
            v113 = v112 + v103
            del v112
            v114 = v91[v113].item()
            del v113
            method8(v114)
            del v114
            v103 += 1 
        del v103
        v115 = ']'
        method2(v115)
        del v115
        v94 += 1 
    del v91, v92, v94
    v116 = ']'
    method2(v116)
    del v116
    method7()
    print()
    return 

if __name__ == '__main__': print(main())
