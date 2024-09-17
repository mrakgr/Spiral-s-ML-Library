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
    v1 = v0 < 4096l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 32l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 64l;
    return v1;
}
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 16l;
    return v1;
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 8l;
    return v1;
}
__device__ inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
extern "C" __global__ void entry0(float * v0, float * v1, float * v2) {
    extern __shared__ unsigned char v3[];
    float * v4;
    v4 = reinterpret_cast<float *>(&v3[0ull]);
    float * v6;
    v6 = reinterpret_cast<float *>(&v3[34816ull]);
    float * v8;
    v8 = reinterpret_cast<float *>(&v3[0ull]);
    int v10;
    v10 = threadIdx.x;
    int v11;
    v11 = v10 / 32l;
    bool v12;
    v12 = 0l <= v11;
    bool v13;
    v13 = v12 == false;
    if (v13){
        assert("The index needs to be zero or positive." && v12);
    } else {
    }
    int v15;
    v15 = v11 % 8l;
    int v16;
    v16 = v11 / 8l;
    bool v17;
    v17 = v16 < 2l;
    bool v18;
    v18 = v17 == false;
    if (v18){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v17);
    } else {
    }
    assert("Tensor range check" && 0 <= v16 && v16 < 2l);
    assert("Tensor range check" && 0 <= v15 && v15 < 8l);
    int v20;
    v20 = 16l * v15;
    int v21;
    v21 = 8704l * v16;
    int v22;
    v22 = v21 + v20;
    float * v23;
    v23 = v8+v22;
    assert("Tensor range check" && 0 <= v16 && v16 < 2l);
    int v25;
    v25 = 4352l * v16;
    int v26;
    v26 = threadIdx.x;
    int v27;
    v27 = v26 % 32l;
    bool v28;
    v28 = 0l <= v27;
    bool v29;
    v29 = v28 == false;
    if (v29){
        assert("The index needs to be zero or positive." && v28);
    } else {
    }
    int v31;
    v31 = v27 % 4l;
    int v32;
    v32 = v27 / 4l;
    bool v33;
    v33 = v32 < 8l;
    bool v34;
    v34 = v33 == false;
    if (v34){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v33);
    } else {
    }
    assert("Tensor range check" && 0 <= v32 && v32 < 8l);
    assert("Tensor range check" && 0 <= v31 && v31 < 4l);
    int v36;
    v36 = v31 + v25;
    int v37;
    v37 = 68l * v32;
    int v38;
    v38 = v37 + v36;
    float * v39;
    v39 = v4+v38;
    assert("Tensor range check" && 0 <= v15 && v15 < 8l);
    int v41;
    v41 = 1088l * v15;
    int v42;
    v42 = threadIdx.x;
    int v43;
    v43 = v42 % 32l;
    bool v44;
    v44 = 0l <= v43;
    bool v45;
    v45 = v44 == false;
    if (v45){
        assert("The index needs to be zero or positive." && v44);
    } else {
    }
    int v47;
    v47 = v43 % 4l;
    int v48;
    v48 = v43 / 4l;
    bool v49;
    v49 = v48 < 8l;
    bool v50;
    v50 = v49 == false;
    if (v50){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v49);
    } else {
    }
    assert("Tensor range check" && 0 <= v48 && v48 < 8l);
    assert("Tensor range check" && 0 <= v47 && v47 < 4l);
    int v52;
    v52 = v47 + v41;
    int v53;
    v53 = 68l * v48;
    int v54;
    v54 = v53 + v52;
    float * v55;
    v55 = v6+v54;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v57[4l];
    int v58;
    v58 = blockIdx.x;
    int v59;
    v59 = v58;
    while (while_method_0(v59)){
        bool v61;
        v61 = 0l <= v59;
        bool v62;
        v62 = v61 == false;
        if (v62){
            assert("The index needs to be zero or positive." && v61);
        } else {
        }
        int v64;
        v64 = v59 % 64l;
        int v65;
        v65 = v59 / 64l;
        bool v66;
        v66 = v65 < 64l;
        bool v67;
        v67 = v66 == false;
        if (v67){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v66);
        } else {
        }
        assert("Tensor range check" && 0 <= v65 && v65 < 64l);
        assert("Tensor range check" && 0 <= v64 && v64 < 64l);
        int v69;
        v69 = 128l * v64;
        int v70;
        v70 = 1048576l * v65;
        int v71;
        v71 = v70 + v69;
        float * v72;
        v72 = v2+v71;
        // Pushing the loop unrolling to: 0
        int v74;
        v74 = threadIdx.x;
        bool v75;
        v75 = 0l <= v74;
        bool v76;
        v76 = v75 == false;
        if (v76){
            assert("The index needs to be zero or positive." && v75);
        } else {
        }
        int v78;
        v78 = v74 % 128l;
        int v79;
        v79 = v74 / 128l;
        bool v80;
        v80 = v79 < 4l;
        bool v81;
        v81 = v80 == false;
        if (v81){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v80);
        } else {
        }
        assert("Tensor range check" && 0 <= v79 && v79 < 4l);
        assert("Tensor range check" && 0 <= v78 && v78 < 128l);
        int v83;
        v83 = 136l * v79;
        int v84;
        v84 = v83 + v78;
        int v85;
        v85 = 8192l * v79;
        int v86;
        v86 = v85 + v78;
        float * v87;
        v87 = v8+v84;
        float * v89;
        v89 = v72+v86;
        int v91;
        v91 = 0l;
        #pragma unroll
        while (while_method_1(v91)){
            int v93;
            v93 = 0l;
            #pragma unroll
            while (while_method_2(v93)){
                assert("Tensor range check" && 0 <= v91 && v91 < 32l);
                assert("Tensor range check" && 0 <= v93 && v93 < 1l);
                int v95;
                v95 = 128l * v93;
                int v96;
                v96 = 544l * v91;
                int v97;
                v97 = v96 + v95;
                int v98;
                v98 = 32768l * v91;
                int v99;
                v99 = v98 + v95;
                int* v100;
                v100 = reinterpret_cast<int*>(v89 + v99);
                int* v101;
                v101 = reinterpret_cast<int*>(v87 + v97);
                assert("Pointer alignment check" && (unsigned long long)(v100) % 1l == 0 && (unsigned long long)(v101) % 1l == 0);
                *v101 = *v100;
                v93 += 1l ;
            }
            v91 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v102;
        v102 = 0l;
        #pragma unroll
        while (while_method_3(v102)){
            int v104;
            v104 = 0l;
            #pragma unroll
            while (while_method_2(v104)){
                assert("Tensor range check" && 0 <= v102 && v102 < 4l);
                assert("Tensor range check" && 0 <= v104 && v104 < 1l);
                int v106;
                v106 = v102 + v104;
                wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v107 = v57[v106];
                assert("Tensor range check" && 0 <= v102 && v102 < 4l);
                assert("Tensor range check" && 0 <= v104 && v104 < 1l);
                int v108;
                v108 = 16l * v104;
                int v109;
                v109 = 2176l * v102;
                int v110;
                v110 = v109 + v108;
                float * v111;
                v111 = v23+v110;
                wmma::load_matrix_sync(v107, v111, 136l, wmma::mem_row_major);
                v104 += 1l ;
            }
            v102 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        // Poping the loop unrolling to: 0
        int v113;
        v113 = 0l;
        while (while_method_4(v113)){
            assert("Tensor range check" && 0 <= v65 && v65 < 64l);
            int v115;
            v115 = 524288l * v65;
            assert("Tensor range check" && 0 <= v113 && v113 < 64l);
            int v116;
            v116 = 64l * v113;
            int v117;
            v117 = v116 + v115;
            float * v118;
            v118 = v0+v117;
            assert("Tensor range check" && 0 <= v64 && v64 < 64l);
            int v120;
            v120 = 524288l * v64;
            assert("Tensor range check" && 0 <= v113 && v113 < 64l);
            int v121;
            v121 = v116 + v120;
            float * v122;
            v122 = v1+v121;
            // Pushing the loop unrolling to: 0
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
            v128 = v124 % 64l;
            int v129;
            v129 = v124 / 64l;
            bool v130;
            v130 = v129 < 8l;
            bool v131;
            v131 = v130 == false;
            if (v131){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v130);
            } else {
            }
            assert("Tensor range check" && 0 <= v129 && v129 < 8l);
            assert("Tensor range check" && 0 <= v128 && v128 < 64l);
            int v133;
            v133 = 68l * v129;
            int v134;
            v134 = v133 + v128;
            int v135;
            v135 = 4096l * v129;
            int v136;
            v136 = v135 + v128;
            float * v137;
            v137 = v6+v134;
            float * v139;
            v139 = v122+v136;
            int v141;
            v141 = 0l;
            #pragma unroll
            while (while_method_5(v141)){
                int v143;
                v143 = 0l;
                #pragma unroll
                while (while_method_2(v143)){
                    assert("Tensor range check" && 0 <= v141 && v141 < 16l);
                    assert("Tensor range check" && 0 <= v143 && v143 < 1l);
                    int v145;
                    v145 = 64l * v143;
                    int v146;
                    v146 = 544l * v141;
                    int v147;
                    v147 = v146 + v145;
                    int v148;
                    v148 = 32768l * v141;
                    int v149;
                    v149 = v148 + v145;
                    float v150[1l];
                    int v151;
                    v151 = 0l;
                    #pragma unroll
                    while (while_method_2(v151)){
                        assert("Tensor range check" && 0 <= v151 && v151 < 1l);
                        int v153;
                        v153 = v151 + v149;
                        float v154;
                        v154 = v139[v153];
                        float v155;
                        v155 = wmma::__float_to_tf32(v154);
                        assert("Tensor range check" && 0 <= v151 && v151 < 1l);
                        v150[v151] = v155;
                        v151 += 1l ;
                    }
                    int* v156;
                    v156 = reinterpret_cast<int*>(v150 + 0l);
                    int* v157;
                    v157 = reinterpret_cast<int*>(v137 + v147);
                    assert("Pointer alignment check" && (unsigned long long)(v156) % 1l == 0 && (unsigned long long)(v157) % 1l == 0);
                    *v157 = *v156;
                    v143 += 1l ;
                }
                v141 += 1l ;
            }
            int v158;
            v158 = threadIdx.x;
            bool v159;
            v159 = 0l <= v158;
            bool v160;
            v160 = v159 == false;
            if (v160){
                assert("The index needs to be zero or positive." && v159);
            } else {
            }
            int v162;
            v162 = v158 % 64l;
            int v163;
            v163 = v158 / 64l;
            bool v164;
            v164 = v163 < 8l;
            bool v165;
            v165 = v164 == false;
            if (v165){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v164);
            } else {
            }
            assert("Tensor range check" && 0 <= v163 && v163 < 8l);
            assert("Tensor range check" && 0 <= v162 && v162 < 64l);
            int v167;
            v167 = 68l * v163;
            int v168;
            v168 = v167 + v162;
            int v169;
            v169 = 4096l * v163;
            int v170;
            v170 = v169 + v162;
            float * v171;
            v171 = v4+v168;
            float * v173;
            v173 = v118+v170;
            int v175;
            v175 = 0l;
            #pragma unroll
            while (while_method_5(v175)){
                int v177;
                v177 = 0l;
                #pragma unroll
                while (while_method_2(v177)){
                    assert("Tensor range check" && 0 <= v175 && v175 < 16l);
                    assert("Tensor range check" && 0 <= v177 && v177 < 1l);
                    int v179;
                    v179 = 64l * v177;
                    int v180;
                    v180 = 544l * v175;
                    int v181;
                    v181 = v180 + v179;
                    int v182;
                    v182 = 32768l * v175;
                    int v183;
                    v183 = v182 + v179;
                    float v184[1l];
                    int v185;
                    v185 = 0l;
                    #pragma unroll
                    while (while_method_2(v185)){
                        assert("Tensor range check" && 0 <= v185 && v185 < 1l);
                        int v187;
                        v187 = v185 + v183;
                        float v188;
                        v188 = v173[v187];
                        float v189;
                        v189 = wmma::__float_to_tf32(v188);
                        assert("Tensor range check" && 0 <= v185 && v185 < 1l);
                        v184[v185] = v189;
                        v185 += 1l ;
                    }
                    int* v190;
                    v190 = reinterpret_cast<int*>(v184 + 0l);
                    int* v191;
                    v191 = reinterpret_cast<int*>(v171 + v181);
                    assert("Pointer alignment check" && (unsigned long long)(v190) % 1l == 0 && (unsigned long long)(v191) % 1l == 0);
                    *v191 = *v190;
                    v177 += 1l ;
                }
                v175 += 1l ;
            }
            // Poping the loop unrolling to: 0
            asm("barrier.cta.sync %0;" :: "r"(0l));
            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v192[1l];
            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v193[8l];
            // Pushing the loop unrolling to: 0
            int v194;
            v194 = 0l;
            #pragma unroll
            while (while_method_2(v194)){
                int v196;
                v196 = 0l;
                #pragma unroll
                while (while_method_6(v196)){
                    assert("Tensor range check" && 0 <= v194 && v194 < 1l);
                    assert("Tensor range check" && 0 <= v196 && v196 < 8l);
                    int v198;
                    v198 = 8l * v194;
                    int v199;
                    v199 = v198 + v196;
                    wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v200 = v193[v199];
                    assert("Tensor range check" && 0 <= v194 && v194 < 1l);
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
                    while (while_method_7(v204)){
                        int v206;
                        v206 = 0l;
                        #pragma unroll
                        while (while_method_7(v206)){
                            assert("Tensor range check" && 0 <= v204 && v204 < 2l);
                            assert("Tensor range check" && 0 <= v206 && v206 < 2l);
                            int v208;
                            v208 = 4l * v206;
                            int v209;
                            v209 = v208 + v203;
                            int v210;
                            v210 = 544l * v204;
                            int v211;
                            v211 = v210 + v209;
                            float v212;
                            v212 = v55[v211];
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
            // Poping the loop unrolling to: 0
            // Pushing the loop unrolling to: 0
            int v225;
            v225 = 0l;
            #pragma unroll
            while (while_method_3(v225)){
                int v227;
                v227 = 0l;
                #pragma unroll
                while (while_method_6(v227)){
                    wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v229 = v192[0l];
                    assert("Tensor range check" && 0 <= v225 && v225 < 4l);
                    int v230;
                    v230 = 1088l * v225;
                    assert("Tensor range check" && 0 <= v227 && v227 < 8l);
                    int v231;
                    v231 = 8l * v227;
                    int v232;
                    v232 = v231 + v230;
                    int v233;
                    v233 = 0l;
                    #pragma unroll
                    while (while_method_7(v233)){
                        int v235;
                        v235 = 0l;
                        #pragma unroll
                        while (while_method_7(v235)){
                            assert("Tensor range check" && 0 <= v233 && v233 < 2l);
                            assert("Tensor range check" && 0 <= v235 && v235 < 2l);
                            int v237;
                            v237 = 544l * v235;
                            int v238;
                            v238 = v237 + v232;
                            int v239;
                            v239 = 4l * v233;
                            int v240;
                            v240 = v239 + v238;
                            float v241;
                            v241 = v39[v240];
                            bool v242;
                            v242 = 0l <= v235;
                            bool v244;
                            if (v242){
                                bool v243;
                                v243 = v235 < 2l;
                                v244 = v243;
                            } else {
                                v244 = false;
                            }
                            bool v245;
                            v245 = v244 == false;
                            if (v245){
                                assert("The indices should be inside the range of the dimension." && v244);
                            } else {
                            }
                            bool v247;
                            v247 = 0l <= v233;
                            bool v249;
                            if (v247){
                                bool v248;
                                v248 = v233 < 2l;
                                v249 = v248;
                            } else {
                                v249 = false;
                            }
                            bool v250;
                            v250 = v249 == false;
                            if (v250){
                                assert("The indices should be inside the range of the dimension." && v249);
                            } else {
                            }
                            int v252;
                            v252 = v233 * 2l;
                            int v253;
                            v253 = v235 + v252;
                            v229.x[v253] = v241;
                            v235 += 1l ;
                        }
                        v233 += 1l ;
                    }
                    int v254;
                    v254 = 0l;
                    #pragma unroll
                    while (while_method_2(v254)){
                        assert("Tensor range check" && 0 <= v225 && v225 < 4l);
                        assert("Tensor range check" && 0 <= v254 && v254 < 1l);
                        int v256;
                        v256 = v225 + v254;
                        wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v257 = v57[v256];
                        assert("Tensor range check" && 0 <= v254 && v254 < 1l);
                        assert("Tensor range check" && 0 <= v227 && v227 < 8l);
                        int v258;
                        v258 = 8l * v254;
                        int v259;
                        v259 = v258 + v227;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v260 = v193[v259];
                        wmma::mma_sync(v257, v229, v260, v257);
                        v254 += 1l ;
                    }
                    v227 += 1l ;
                }
                v225 += 1l ;
            }
            // Poping the loop unrolling to: 0
            asm("barrier.cta.sync %0;" :: "r"(0l));
            v113 += 1l ;
        }
        // Pushing the loop unrolling to: 0
        int v261;
        v261 = 0l;
        #pragma unroll
        while (while_method_3(v261)){
            int v263;
            v263 = 0l;
            #pragma unroll
            while (while_method_2(v263)){
                assert("Tensor range check" && 0 <= v261 && v261 < 4l);
                assert("Tensor range check" && 0 <= v263 && v263 < 1l);
                int v265;
                v265 = v261 + v263;
                wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v266 = v57[v265];
                assert("Tensor range check" && 0 <= v261 && v261 < 4l);
                assert("Tensor range check" && 0 <= v263 && v263 < 1l);
                int v267;
                v267 = 16l * v263;
                int v268;
                v268 = 2176l * v261;
                int v269;
                v269 = v268 + v267;
                float * v270;
                v270 = v23+v269;
                wmma::store_matrix_sync(v270, v266, 136l, wmma::mem_row_major);
                v263 += 1l ;
            }
            v261 += 1l ;
        }
        // Poping the loop unrolling to: 0
        asm("barrier.cta.sync %0;" :: "r"(0l));
        // Pushing the loop unrolling to: 0
        int v272;
        v272 = threadIdx.x;
        bool v273;
        v273 = 0l <= v272;
        bool v274;
        v274 = v273 == false;
        if (v274){
            assert("The index needs to be zero or positive." && v273);
        } else {
        }
        int v276;
        v276 = v272 % 128l;
        int v277;
        v277 = v272 / 128l;
        bool v278;
        v278 = v277 < 4l;
        bool v279;
        v279 = v278 == false;
        if (v279){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v278);
        } else {
        }
        assert("Tensor range check" && 0 <= v277 && v277 < 4l);
        assert("Tensor range check" && 0 <= v276 && v276 < 128l);
        int v281;
        v281 = 8192l * v277;
        int v282;
        v282 = v281 + v276;
        int v283;
        v283 = 136l * v277;
        int v284;
        v284 = v283 + v276;
        float * v285;
        v285 = v72+v282;
        float * v287;
        v287 = v8+v284;
        int v289;
        v289 = 0l;
        #pragma unroll
        while (while_method_1(v289)){
            int v291;
            v291 = 0l;
            #pragma unroll
            while (while_method_2(v291)){
                assert("Tensor range check" && 0 <= v289 && v289 < 32l);
                assert("Tensor range check" && 0 <= v291 && v291 < 1l);
                int v293;
                v293 = 128l * v291;
                int v294;
                v294 = 32768l * v289;
                int v295;
                v295 = v294 + v293;
                int v296;
                v296 = 544l * v289;
                int v297;
                v297 = v296 + v293;
                int* v298;
                v298 = reinterpret_cast<int*>(v287 + v297);
                int* v299;
                v299 = reinterpret_cast<int*>(v285 + v295);
                assert("Pointer alignment check" && (unsigned long long)(v298) % 1l == 0 && (unsigned long long)(v299) % 1l == 0);
                *v299 = *v298;
                v291 += 1l ;
            }
            v289 += 1l ;
        }
        // Poping the loop unrolling to: 0
        asm("barrier.cta.sync %0;" :: "r"(0l));
        v59 += 24l ;
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
i8 = int; i16 = int; i32 = int; i64 = int; u8 = int; u16 = int; u32 = int; u64 = int; f32 = float; f64 = float; char = str; string = str

options = []
options.append('--define-macro=NDEBUG')
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39,177')
options.append('--restrict')
options.append('--maxrregcount=128')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def main_body():
    v0 = cp.random.normal(0.0,1.0,67108864,dtype=cp.float32) # type: ignore
    v1 = cp.random.normal(0.0,1.0,33554432,dtype=cp.float32) # type: ignore
    v2 = cp.random.normal(0.0,1.0,33554432,dtype=cp.float32) # type: ignore
    v3 = v2.reshape((8192, 4096))
    v4 = v1.reshape((8192, 4096))
    v5 = cp.transpose(v4)
    del v4
    v6 = v0.reshape((8192, 8192))
    v7 = (cp.matmul(v3,v5) + v6).flatten()
    del v3, v5, v6
    v8 = v7.size
    v9 = 67108864 == v8
    del v8
    v10 = v9 == False
    if v10:
        v11 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v9, v11
        del v11
    else:
        pass
    del v9, v10
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
    v17.max_dynamic_shared_size_bytes = 98304 
    print(f'Threads per block, blocks per grid: {512}, {24}')
    v17((24,),(512,),(v2, v1, v0),shared_mem=98304)
    del v1, v2, v17
    v18 = cp.max(cp.abs(v0-v7))
    del v0, v7
    return v18

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
