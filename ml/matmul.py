kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <mma.h>
using namespace nvcuda;
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

__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 8l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
extern "C" __global__ void entry0(float * v0, float * v1, float * v2) {
    unsigned int v3;
    v3 = 0ul;
    asm("mov.u32 %0, %dynamic_smem_size;" : "=r"(v3));
    unsigned long long v4;
    v4 = (unsigned long long)v3;
    bool v5;
    v5 = 69632ull <= v4;
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
    v11 = reinterpret_cast<float *>(&v8[34816ull]);
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
    v20 = v16 % 8l;
    int v21;
    v21 = v16 / 8l;
    bool v22;
    v22 = v21 < 2l;
    bool v23;
    v23 = v22 == false;
    if (v23){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v22);
    } else {
    }
    assert("Tensor range check" && 0 <= v21 && v21 < 2l);
    assert("Tensor range check" && 0 <= v20 && v20 < 8l);
    int v25;
    v25 = 16l * v20;
    int v26;
    v26 = 8704l * v21;
    int v27;
    v27 = v26 + v25;
    float * v28;
    v28 = v13+v27;
    assert("Tensor range check" && 0 <= v21 && v21 < 2l);
    int v30;
    v30 = 4352l * v21;
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
    v42 = 68l * v37;
    int v43;
    v43 = v42 + v41;
    float * v44;
    v44 = v9+v43;
    assert("Tensor range check" && 0 <= v20 && v20 < 8l);
    int v46;
    v46 = 1088l * v20;
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
    v58 = 68l * v53;
    int v59;
    v59 = v58 + v57;
    float * v60;
    v60 = v11+v59;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v62[4l];
    int v63;
    v63 = 0l;
    while (while_method_0(v63)){
        int v65;
        v65 = 0l;
        while (while_method_0(v65)){
            assert("Tensor range check" && 0 <= v63 && v63 < 4l);
            assert("Tensor range check" && 0 <= v65 && v65 < 4l);
            int v67;
            v67 = 128l * v65;
            int v68;
            v68 = 65536l * v63;
            int v69;
            v69 = v68 + v67;
            float * v70;
            v70 = v2+v69;
            // Pushing the loop unrolling to: 0
            int v72;
            v72 = threadIdx.x;
            bool v73;
            v73 = 0l <= v72;
            bool v74;
            v74 = v73 == false;
            if (v74){
                assert("The index needs to be zero or positive." && v73);
            } else {
            }
            int v76;
            v76 = v72 % 32l;
            int v77;
            v77 = v72 / 32l;
            bool v78;
            v78 = v77 < 16l;
            bool v79;
            v79 = v78 == false;
            if (v79){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v78);
            } else {
            }
            assert("Tensor range check" && 0 <= v77 && v77 < 16l);
            assert("Tensor range check" && 0 <= v76 && v76 < 32l);
            int v81;
            v81 = 4l * v76;
            int v82;
            v82 = 136l * v77;
            int v83;
            v83 = v82 + v81;
            int v84;
            v84 = 512l * v77;
            int v85;
            v85 = v84 + v81;
            float * v86;
            v86 = v13+v83;
            float * v88;
            v88 = v70+v85;
            int v90;
            v90 = 0l;
            #pragma unroll
            while (while_method_1(v90)){
                int v92;
                v92 = 0l;
                #pragma unroll
                while (while_method_2(v92)){
                    assert("Tensor range check" && 0 <= v90 && v90 < 8l);
                    assert("Tensor range check" && 0 <= v92 && v92 < 1l);
                    int v94;
                    v94 = 128l * v92;
                    int v95;
                    v95 = 2176l * v90;
                    int v96;
                    v96 = v95 + v94;
                    int v97;
                    v97 = 8192l * v90;
                    int v98;
                    v98 = v97 + v94;
                    int4* v99;
                    v99 = reinterpret_cast<int4*>(v88 + v98);
                    int4* v100;
                    v100 = reinterpret_cast<int4*>(v86 + v96);
                    assert("Pointer alignment check" && (unsigned long long)(v99) % 4l == 0 && (unsigned long long)(v100) % 4l == 0);
                    *v100 = *v99;
                    v92 += 1l ;
                }
                v90 += 1l ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0l));
            int v101;
            v101 = 0l;
            #pragma unroll
            while (while_method_0(v101)){
                int v103;
                v103 = 0l;
                #pragma unroll
                while (while_method_2(v103)){
                    assert("Tensor range check" && 0 <= v101 && v101 < 4l);
                    assert("Tensor range check" && 0 <= v103 && v103 < 1l);
                    int v105;
                    v105 = v101 + v103;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v106 = v62[v105];
                    assert("Tensor range check" && 0 <= v101 && v101 < 4l);
                    assert("Tensor range check" && 0 <= v103 && v103 < 1l);
                    int v107;
                    v107 = 16l * v103;
                    int v108;
                    v108 = 2176l * v101;
                    int v109;
                    v109 = v108 + v107;
                    float * v110;
                    v110 = v28+v109;
                    wmma::load_matrix_sync(v106, v110, 136l, wmma::mem_row_major);
                    v103 += 1l ;
                }
                v101 += 1l ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0l));
            int v112;
            v112 = 0l;
            #pragma unroll
            while (while_method_1(v112)){
                assert("Tensor range check" && 0 <= v63 && v63 < 4l);
                assert("Tensor range check" && 0 <= v112 && v112 < 8l);
                int v114;
                v114 = 64l * v112;
                int v115;
                v115 = v114 + v68;
                float * v116;
                v116 = v0+v115;
                assert("Tensor range check" && 0 <= v65 && v65 < 4l);
                int v118;
                v118 = 65536l * v65;
                assert("Tensor range check" && 0 <= v112 && v112 < 8l);
                int v119;
                v119 = v114 + v118;
                float * v120;
                v120 = v1+v119;
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
                v126 = v122 % 16l;
                int v127;
                v127 = v122 / 16l;
                bool v128;
                v128 = v127 < 32l;
                bool v129;
                v129 = v128 == false;
                if (v129){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v128);
                } else {
                }
                assert("Tensor range check" && 0 <= v127 && v127 < 32l);
                assert("Tensor range check" && 0 <= v126 && v126 < 16l);
                int v131;
                v131 = 4l * v126;
                int v132;
                v132 = 68l * v127;
                int v133;
                v133 = v132 + v131;
                int v134;
                v134 = 512l * v127;
                int v135;
                v135 = v134 + v131;
                float * v136;
                v136 = v11+v133;
                float * v138;
                v138 = v120+v135;
                int v140;
                v140 = 0l;
                #pragma unroll
                while (while_method_0(v140)){
                    int v142;
                    v142 = 0l;
                    #pragma unroll
                    while (while_method_2(v142)){
                        assert("Tensor range check" && 0 <= v140 && v140 < 4l);
                        assert("Tensor range check" && 0 <= v142 && v142 < 1l);
                        int v144;
                        v144 = 64l * v142;
                        int v145;
                        v145 = 2176l * v140;
                        int v146;
                        v146 = v145 + v144;
                        int v147;
                        v147 = 16384l * v140;
                        int v148;
                        v148 = v147 + v144;
                        float v149[4l];
                        int v150;
                        v150 = 0l;
                        #pragma unroll
                        while (while_method_0(v150)){
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
                int v157;
                v157 = threadIdx.x;
                bool v158;
                v158 = 0l <= v157;
                bool v159;
                v159 = v158 == false;
                if (v159){
                    assert("The index needs to be zero or positive." && v158);
                } else {
                }
                int v161;
                v161 = v157 % 16l;
                int v162;
                v162 = v157 / 16l;
                bool v163;
                v163 = v162 < 32l;
                bool v164;
                v164 = v163 == false;
                if (v164){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v163);
                } else {
                }
                assert("Tensor range check" && 0 <= v162 && v162 < 32l);
                assert("Tensor range check" && 0 <= v161 && v161 < 16l);
                int v166;
                v166 = 4l * v161;
                int v167;
                v167 = 68l * v162;
                int v168;
                v168 = v167 + v166;
                int v169;
                v169 = 512l * v162;
                int v170;
                v170 = v169 + v166;
                float * v171;
                v171 = v9+v168;
                float * v173;
                v173 = v116+v170;
                int v175;
                v175 = 0l;
                #pragma unroll
                while (while_method_0(v175)){
                    int v177;
                    v177 = 0l;
                    #pragma unroll
                    while (while_method_2(v177)){
                        assert("Tensor range check" && 0 <= v175 && v175 < 4l);
                        assert("Tensor range check" && 0 <= v177 && v177 < 1l);
                        int v179;
                        v179 = 64l * v177;
                        int v180;
                        v180 = 2176l * v175;
                        int v181;
                        v181 = v180 + v179;
                        int v182;
                        v182 = 16384l * v175;
                        int v183;
                        v183 = v182 + v179;
                        float v184[4l];
                        int v185;
                        v185 = 0l;
                        #pragma unroll
                        while (while_method_0(v185)){
                            assert("Tensor range check" && 0 <= v185 && v185 < 4l);
                            int v187;
                            v187 = v185 + v183;
                            float v188;
                            v188 = v173[v187];
                            float v189;
                            v189 = wmma::__float_to_tf32(v188);
                            assert("Tensor range check" && 0 <= v185 && v185 < 4l);
                            v184[v185] = v189;
                            v185 += 1l ;
                        }
                        int4* v190;
                        v190 = reinterpret_cast<int4*>(v184 + 0l);
                        int4* v191;
                        v191 = reinterpret_cast<int4*>(v171 + v181);
                        assert("Pointer alignment check" && (unsigned long long)(v190) % 4l == 0 && (unsigned long long)(v191) % 4l == 0);
                        *v191 = *v190;
                        v177 += 1l ;
                    }
                    v175 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v192[32l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v193[8l];
                int v194;
                v194 = 0l;
                #pragma unroll
                while (while_method_0(v194)){
                    int v196;
                    v196 = 0l;
                    #pragma unroll
                    while (while_method_1(v196)){
                        assert("Tensor range check" && 0 <= v194 && v194 < 4l);
                        assert("Tensor range check" && 0 <= v196 && v196 < 8l);
                        int v198;
                        v198 = 8l * v194;
                        int v199;
                        v199 = v198 + v196;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v200 = v192[v199];
                        assert("Tensor range check" && 0 <= v194 && v194 < 4l);
                        int v201;
                        v201 = 1088l * v194;
                        assert("Tensor range check" && 0 <= v196 && v196 < 8l);
                        int v202;
                        v202 = 8l * v196;
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
                                v208 = 544l * v206;
                                int v209;
                                v209 = v208 + v203;
                                int v210;
                                v210 = 4l * v204;
                                int v211;
                                v211 = v210 + v209;
                                float v212;
                                v212 = v44[v211];
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
                        v196 += 1l ;
                    }
                    v194 += 1l ;
                }
                int v225;
                v225 = 0l;
                #pragma unroll
                while (while_method_2(v225)){
                    int v227;
                    v227 = 0l;
                    #pragma unroll
                    while (while_method_1(v227)){
                        assert("Tensor range check" && 0 <= v225 && v225 < 1l);
                        assert("Tensor range check" && 0 <= v227 && v227 < 8l);
                        int v229;
                        v229 = 8l * v225;
                        int v230;
                        v230 = v229 + v227;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v231 = v193[v230];
                        assert("Tensor range check" && 0 <= v225 && v225 < 1l);
                        int v232;
                        v232 = 1088l * v225;
                        assert("Tensor range check" && 0 <= v227 && v227 < 8l);
                        int v233;
                        v233 = 8l * v227;
                        int v234;
                        v234 = v233 + v232;
                        int v235;
                        v235 = 0l;
                        #pragma unroll
                        while (while_method_3(v235)){
                            int v237;
                            v237 = 0l;
                            #pragma unroll
                            while (while_method_3(v237)){
                                assert("Tensor range check" && 0 <= v235 && v235 < 2l);
                                assert("Tensor range check" && 0 <= v237 && v237 < 2l);
                                int v239;
                                v239 = 4l * v237;
                                int v240;
                                v240 = v239 + v234;
                                int v241;
                                v241 = 544l * v235;
                                int v242;
                                v242 = v241 + v240;
                                float v243;
                                v243 = v60[v242];
                                bool v244;
                                v244 = 0l <= v237;
                                bool v246;
                                if (v244){
                                    bool v245;
                                    v245 = v237 < 2l;
                                    v246 = v245;
                                } else {
                                    v246 = false;
                                }
                                bool v247;
                                v247 = v246 == false;
                                if (v247){
                                    assert("The indices should be inside the range of the dimension." && v246);
                                } else {
                                }
                                bool v249;
                                v249 = 0l <= v235;
                                bool v251;
                                if (v249){
                                    bool v250;
                                    v250 = v235 < 2l;
                                    v251 = v250;
                                } else {
                                    v251 = false;
                                }
                                bool v252;
                                v252 = v251 == false;
                                if (v252){
                                    assert("The indices should be inside the range of the dimension." && v251);
                                } else {
                                }
                                int v254;
                                v254 = v235 * 2l;
                                int v255;
                                v255 = v237 + v254;
                                v231.x[v255] = v243;
                                v237 += 1l ;
                            }
                            v235 += 1l ;
                        }
                        v227 += 1l ;
                    }
                    v225 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                int v256;
                v256 = 0l;
                #pragma unroll
                while (while_method_0(v256)){
                    int v258;
                    v258 = 0l;
                    #pragma unroll
                    while (while_method_2(v258)){
                        int v260;
                        v260 = 0l;
                        #pragma unroll
                        while (while_method_1(v260)){
                            assert("Tensor range check" && 0 <= v256 && v256 < 4l);
                            assert("Tensor range check" && 0 <= v258 && v258 < 1l);
                            int v262;
                            v262 = v256 + v258;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v263 = v62[v262];
                            assert("Tensor range check" && 0 <= v256 && v256 < 4l);
                            assert("Tensor range check" && 0 <= v260 && v260 < 8l);
                            int v264;
                            v264 = 8l * v256;
                            int v265;
                            v265 = v264 + v260;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v266 = v192[v265];
                            assert("Tensor range check" && 0 <= v258 && v258 < 1l);
                            assert("Tensor range check" && 0 <= v260 && v260 < 8l);
                            int v267;
                            v267 = 8l * v258;
                            int v268;
                            v268 = v267 + v260;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v269 = v193[v268];
                            wmma::mma_sync(v263, v266, v269, v263);
                            v260 += 1l ;
                        }
                        v258 += 1l ;
                    }
                    v256 += 1l ;
                }
                v112 += 1l ;
            }
            int v270;
            v270 = 0l;
            #pragma unroll
            while (while_method_0(v270)){
                int v272;
                v272 = 0l;
                #pragma unroll
                while (while_method_2(v272)){
                    assert("Tensor range check" && 0 <= v270 && v270 < 4l);
                    assert("Tensor range check" && 0 <= v272 && v272 < 1l);
                    int v274;
                    v274 = v270 + v272;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v275 = v62[v274];
                    assert("Tensor range check" && 0 <= v270 && v270 < 4l);
                    assert("Tensor range check" && 0 <= v272 && v272 < 1l);
                    int v276;
                    v276 = 16l * v272;
                    int v277;
                    v277 = 2176l * v270;
                    int v278;
                    v278 = v277 + v276;
                    float * v279;
                    v279 = v28+v278;
                    wmma::store_matrix_sync(v279, v275, 136l, wmma::mem_row_major);
                    v272 += 1l ;
                }
                v270 += 1l ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0l));
            int v281;
            v281 = threadIdx.x;
            bool v282;
            v282 = 0l <= v281;
            bool v283;
            v283 = v282 == false;
            if (v283){
                assert("The index needs to be zero or positive." && v282);
            } else {
            }
            int v285;
            v285 = v281 % 32l;
            int v286;
            v286 = v281 / 32l;
            bool v287;
            v287 = v286 < 16l;
            bool v288;
            v288 = v287 == false;
            if (v288){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v287);
            } else {
            }
            assert("Tensor range check" && 0 <= v286 && v286 < 16l);
            assert("Tensor range check" && 0 <= v285 && v285 < 32l);
            int v290;
            v290 = 4l * v285;
            int v291;
            v291 = 512l * v286;
            int v292;
            v292 = v291 + v290;
            int v293;
            v293 = 136l * v286;
            int v294;
            v294 = v293 + v290;
            float * v295;
            v295 = v70+v292;
            float * v297;
            v297 = v13+v294;
            int v299;
            v299 = 0l;
            #pragma unroll
            while (while_method_1(v299)){
                int v301;
                v301 = 0l;
                #pragma unroll
                while (while_method_2(v301)){
                    assert("Tensor range check" && 0 <= v299 && v299 < 8l);
                    assert("Tensor range check" && 0 <= v301 && v301 < 1l);
                    int v303;
                    v303 = 128l * v301;
                    int v304;
                    v304 = 8192l * v299;
                    int v305;
                    v305 = v304 + v303;
                    int v306;
                    v306 = 2176l * v299;
                    int v307;
                    v307 = v306 + v303;
                    int4* v308;
                    v308 = reinterpret_cast<int4*>(v297 + v307);
                    int4* v309;
                    v309 = reinterpret_cast<int4*>(v295 + v305);
                    assert("Pointer alignment check" && (unsigned long long)(v308) % 4l == 0 && (unsigned long long)(v309) % 4l == 0);
                    *v309 = *v308;
                    v301 += 1l ;
                }
                v299 += 1l ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0l));
            // Poping the loop unrolling to: 0
            v65 += 1l ;
        }
        v63 += 1l ;
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

from max_blocks_per_sm import max_blocks_per_sm
options = []
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39')
options.append('--restrict')
options.append('--maxrregcount=128')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def main_body():
    v0 = cp.random.normal(0.0,1.0,262144,dtype=cp.float32) # type: ignore
    v1 = cp.random.normal(0.0,1.0,262144,dtype=cp.float32) # type: ignore
    v2 = cp.random.normal(0.0,1.0,262144,dtype=cp.float32) # type: ignore
    v3 = v2.reshape((512, 512))
    v4 = v1.reshape((512, 512))
    v5 = cp.transpose(v4)
    del v4
    v6 = v0.reshape((512, 512))
    v7 = (cp.matmul(v3,v5) + v6).flatten()
    del v3, v5, v6
    v8 = v7.size
    v9 = 262144 == v8
    del v8
    v10 = v9 == False
    if v10:
        v11 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v9, v11
        del v11
    else:
        pass
    del v9, v10
    max_blocks_per_sm(cp.cuda.Device(),raw_module.get_function('entry0'),512,is_print=True)
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
    v16 = 0
    v17 = raw_module.get_function(f"entry{v16}")
    del v16
    v17.max_dynamic_shared_size_bytes = 69632
    v17((24,),(512,),(v2, v1, v0),shared_mem=69632)
    del v1, v2, v17
    v18 = cp.max(cp.abs(v0-v7))
    del v0, v7
    return v18

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
