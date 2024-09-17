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
    v1 = v0 < 4096;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 8;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 1;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 4;
    return v1;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 64;
    return v1;
}
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 2;
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
    v11 = v10 / 32;
    bool v12;
    v12 = 0 <= v11;
    bool v13;
    v13 = v12 == false;
    if (v13){
        assert("The index needs to be zero or positive." && v12);
    } else {
    }
    int v15;
    v15 = v11 % 8;
    int v16;
    v16 = v11 / 8;
    bool v17;
    v17 = v16 < 2;
    bool v18;
    v18 = v17 == false;
    if (v18){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v17);
    } else {
    }
    assert("Tensor range check" && 0 <= v16 && v16 < 2);
    assert("Tensor range check" && 0 <= v15 && v15 < 8);
    int v20;
    v20 = 16 * v15;
    int v21;
    v21 = 8704 * v16;
    int v22;
    v22 = v21 + v20;
    float * v23;
    v23 = v8+v22;
    assert("Tensor range check" && 0 <= v16 && v16 < 2);
    int v25;
    v25 = 4352 * v16;
    int v26;
    v26 = threadIdx.x;
    int v27;
    v27 = v26 % 32;
    bool v28;
    v28 = 0 <= v27;
    bool v29;
    v29 = v28 == false;
    if (v29){
        assert("The index needs to be zero or positive." && v28);
    } else {
    }
    int v31;
    v31 = v27 % 4;
    int v32;
    v32 = v27 / 4;
    bool v33;
    v33 = v32 < 8;
    bool v34;
    v34 = v33 == false;
    if (v34){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v33);
    } else {
    }
    assert("Tensor range check" && 0 <= v32 && v32 < 8);
    assert("Tensor range check" && 0 <= v31 && v31 < 4);
    int v36;
    v36 = v31 + v25;
    int v37;
    v37 = 68 * v32;
    int v38;
    v38 = v37 + v36;
    float * v39;
    v39 = v4+v38;
    assert("Tensor range check" && 0 <= v15 && v15 < 8);
    int v41;
    v41 = 1088 * v15;
    int v42;
    v42 = threadIdx.x;
    int v43;
    v43 = v42 % 32;
    bool v44;
    v44 = 0 <= v43;
    bool v45;
    v45 = v44 == false;
    if (v45){
        assert("The index needs to be zero or positive." && v44);
    } else {
    }
    int v47;
    v47 = v43 % 4;
    int v48;
    v48 = v43 / 4;
    bool v49;
    v49 = v48 < 8;
    bool v50;
    v50 = v49 == false;
    if (v50){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v49);
    } else {
    }
    assert("Tensor range check" && 0 <= v48 && v48 < 8);
    assert("Tensor range check" && 0 <= v47 && v47 < 4);
    int v52;
    v52 = v47 + v41;
    int v53;
    v53 = 68 * v48;
    int v54;
    v54 = v53 + v52;
    float * v55;
    v55 = v6+v54;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> v57[4];
    int v58;
    v58 = blockIdx.x;
    int v59;
    v59 = v58;
    while (while_method_0(v59)){
        bool v61;
        v61 = 0 <= v59;
        bool v62;
        v62 = v61 == false;
        if (v62){
            assert("The index needs to be zero or positive." && v61);
        } else {
        }
        int v64;
        v64 = v59 % 64;
        int v65;
        v65 = v59 / 64;
        bool v66;
        v66 = v65 < 64;
        bool v67;
        v67 = v66 == false;
        if (v67){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v66);
        } else {
        }
        assert("Tensor range check" && 0 <= v65 && v65 < 64);
        assert("Tensor range check" && 0 <= v64 && v64 < 64);
        int v69;
        v69 = 128 * v64;
        int v70;
        v70 = 1048576 * v65;
        int v71;
        v71 = v70 + v69;
        float * v72;
        v72 = v2+v71;
        // Pushing the loop unrolling to: 0
        int v74;
        v74 = threadIdx.x;
        bool v75;
        v75 = 0 <= v74;
        bool v76;
        v76 = v75 == false;
        if (v76){
            assert("The index needs to be zero or positive." && v75);
        } else {
        }
        int v78;
        v78 = v74 % 32;
        int v79;
        v79 = v74 / 32;
        bool v80;
        v80 = v79 < 16;
        bool v81;
        v81 = v80 == false;
        if (v81){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v80);
        } else {
        }
        assert("Tensor range check" && 0 <= v79 && v79 < 16);
        assert("Tensor range check" && 0 <= v78 && v78 < 32);
        int v83;
        v83 = 4 * v78;
        int v84;
        v84 = 136 * v79;
        int v85;
        v85 = v84 + v83;
        int v86;
        v86 = 8192 * v79;
        int v87;
        v87 = v86 + v83;
        float * v88;
        v88 = v8+v85;
        float * v90;
        v90 = v72+v87;
        int v92;
        v92 = 0;
        #pragma unroll
        while (while_method_1(v92)){
            int v94;
            v94 = 0;
            #pragma unroll
            while (while_method_2(v94)){
                assert("Tensor range check" && 0 <= v92 && v92 < 8);
                assert("Tensor range check" && 0 <= v94 && v94 < 1);
                int v96;
                v96 = 128 * v94;
                int v97;
                v97 = 2176 * v92;
                int v98;
                v98 = v97 + v96;
                int v99;
                v99 = 131072 * v92;
                int v100;
                v100 = v99 + v96;
                int4* v101;
                v101 = reinterpret_cast<int4*>(v90 + v100);
                int4* v102;
                v102 = reinterpret_cast<int4*>(v88 + v98);
                assert("Pointer alignment check" && (unsigned long long)(v101) % 4 == 0 && (unsigned long long)(v102) % 4 == 0);
                *v102 = *v101;
                v94 += 1 ;
            }
            v92 += 1 ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0));
        int v103;
        v103 = 0;
        #pragma unroll
        while (while_method_3(v103)){
            int v105;
            v105 = 0;
            #pragma unroll
            while (while_method_2(v105)){
                assert("Tensor range check" && 0 <= v103 && v103 < 4);
                assert("Tensor range check" && 0 <= v105 && v105 < 1);
                int v107;
                v107 = v103 + v105;
                wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v108 = v57[v107];
                assert("Tensor range check" && 0 <= v103 && v103 < 4);
                assert("Tensor range check" && 0 <= v105 && v105 < 1);
                int v109;
                v109 = 16 * v105;
                int v110;
                v110 = 2176 * v103;
                int v111;
                v111 = v110 + v109;
                float * v112;
                v112 = v23+v111;
                wmma::load_matrix_sync(v108, v112, 136, wmma::mem_row_major);
                v105 += 1 ;
            }
            v103 += 1 ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0));
        // Poping the loop unrolling to: 0
        int v114;
        v114 = 0;
        while (while_method_4(v114)){
            assert("Tensor range check" && 0 <= v65 && v65 < 64);
            int v116;
            v116 = 524288 * v65;
            assert("Tensor range check" && 0 <= v114 && v114 < 64);
            int v117;
            v117 = 64 * v114;
            int v118;
            v118 = v117 + v116;
            float * v119;
            v119 = v0+v118;
            assert("Tensor range check" && 0 <= v64 && v64 < 64);
            int v121;
            v121 = 524288 * v64;
            assert("Tensor range check" && 0 <= v114 && v114 < 64);
            int v122;
            v122 = v117 + v121;
            float * v123;
            v123 = v1+v122;
            // Pushing the loop unrolling to: 0
            int v125;
            v125 = threadIdx.x;
            bool v126;
            v126 = 0 <= v125;
            bool v127;
            v127 = v126 == false;
            if (v127){
                assert("The index needs to be zero or positive." && v126);
            } else {
            }
            int v129;
            v129 = v125 % 16;
            int v130;
            v130 = v125 / 16;
            bool v131;
            v131 = v130 < 32;
            bool v132;
            v132 = v131 == false;
            if (v132){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v131);
            } else {
            }
            assert("Tensor range check" && 0 <= v130 && v130 < 32);
            assert("Tensor range check" && 0 <= v129 && v129 < 16);
            int v134;
            v134 = 4 * v129;
            int v135;
            v135 = 68 * v130;
            int v136;
            v136 = v135 + v134;
            int v137;
            v137 = 4096 * v130;
            int v138;
            v138 = v137 + v134;
            float * v139;
            v139 = v6+v136;
            float * v141;
            v141 = v123+v138;
            int v143;
            v143 = 0;
            #pragma unroll
            while (while_method_3(v143)){
                int v145;
                v145 = 0;
                #pragma unroll
                while (while_method_2(v145)){
                    assert("Tensor range check" && 0 <= v143 && v143 < 4);
                    assert("Tensor range check" && 0 <= v145 && v145 < 1);
                    int v147;
                    v147 = 64 * v145;
                    int v148;
                    v148 = 2176 * v143;
                    int v149;
                    v149 = v148 + v147;
                    int v150;
                    v150 = 131072 * v143;
                    int v151;
                    v151 = v150 + v147;
                    float v152[4];
                    int4* v153;
                    v153 = reinterpret_cast<int4*>(v141 + v151);
                    int4* v154;
                    v154 = reinterpret_cast<int4*>(v152 + 0);
                    assert("Pointer alignment check" && (unsigned long long)(v153) % 4 == 0 && (unsigned long long)(v154) % 4 == 0);
                    *v154 = *v153;
                    int v155;
                    v155 = 0;
                    #pragma unroll
                    while (while_method_3(v155)){
                        assert("Tensor range check" && 0 <= v155 && v155 < 4);
                        float v157;
                        v157 = v152[v155];
                        float v158;
                        v158 = wmma::__float_to_tf32(v157);
                        assert("Tensor range check" && 0 <= v155 && v155 < 4);
                        v152[v155] = v158;
                        v155 += 1 ;
                    }
                    int4* v159;
                    v159 = reinterpret_cast<int4*>(v152 + 0);
                    int4* v160;
                    v160 = reinterpret_cast<int4*>(v139 + v149);
                    assert("Pointer alignment check" && (unsigned long long)(v159) % 4 == 0 && (unsigned long long)(v160) % 4 == 0);
                    *v160 = *v159;
                    v145 += 1 ;
                }
                v143 += 1 ;
            }
            int v161;
            v161 = threadIdx.x;
            bool v162;
            v162 = 0 <= v161;
            bool v163;
            v163 = v162 == false;
            if (v163){
                assert("The index needs to be zero or positive." && v162);
            } else {
            }
            int v165;
            v165 = v161 % 16;
            int v166;
            v166 = v161 / 16;
            bool v167;
            v167 = v166 < 32;
            bool v168;
            v168 = v167 == false;
            if (v168){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v167);
            } else {
            }
            assert("Tensor range check" && 0 <= v166 && v166 < 32);
            assert("Tensor range check" && 0 <= v165 && v165 < 16);
            int v170;
            v170 = 4 * v165;
            int v171;
            v171 = 68 * v166;
            int v172;
            v172 = v171 + v170;
            int v173;
            v173 = 4096 * v166;
            int v174;
            v174 = v173 + v170;
            float * v175;
            v175 = v4+v172;
            float * v177;
            v177 = v119+v174;
            int v179;
            v179 = 0;
            #pragma unroll
            while (while_method_3(v179)){
                int v181;
                v181 = 0;
                #pragma unroll
                while (while_method_2(v181)){
                    assert("Tensor range check" && 0 <= v179 && v179 < 4);
                    assert("Tensor range check" && 0 <= v181 && v181 < 1);
                    int v183;
                    v183 = 64 * v181;
                    int v184;
                    v184 = 2176 * v179;
                    int v185;
                    v185 = v184 + v183;
                    int v186;
                    v186 = 131072 * v179;
                    int v187;
                    v187 = v186 + v183;
                    float v188[4];
                    int4* v189;
                    v189 = reinterpret_cast<int4*>(v177 + v187);
                    int4* v190;
                    v190 = reinterpret_cast<int4*>(v188 + 0);
                    assert("Pointer alignment check" && (unsigned long long)(v189) % 4 == 0 && (unsigned long long)(v190) % 4 == 0);
                    *v190 = *v189;
                    int v191;
                    v191 = 0;
                    #pragma unroll
                    while (while_method_3(v191)){
                        assert("Tensor range check" && 0 <= v191 && v191 < 4);
                        float v193;
                        v193 = v188[v191];
                        float v194;
                        v194 = wmma::__float_to_tf32(v193);
                        assert("Tensor range check" && 0 <= v191 && v191 < 4);
                        v188[v191] = v194;
                        v191 += 1 ;
                    }
                    int4* v195;
                    v195 = reinterpret_cast<int4*>(v188 + 0);
                    int4* v196;
                    v196 = reinterpret_cast<int4*>(v175 + v185);
                    assert("Pointer alignment check" && (unsigned long long)(v195) % 4 == 0 && (unsigned long long)(v196) % 4 == 0);
                    *v196 = *v195;
                    v181 += 1 ;
                }
                v179 += 1 ;
            }
            // Poping the loop unrolling to: 0
            asm("barrier.cta.sync %0;" :: "r"(0));
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v197[1];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v198[8];
            // Pushing the loop unrolling to: 0
            int v199;
            v199 = 0;
            #pragma unroll
            while (while_method_2(v199)){
                int v201;
                v201 = 0;
                #pragma unroll
                while (while_method_1(v201)){
                    assert("Tensor range check" && 0 <= v199 && v199 < 1);
                    assert("Tensor range check" && 0 <= v201 && v201 < 8);
                    int v203;
                    v203 = 8 * v199;
                    int v204;
                    v204 = v203 + v201;
                    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v205 = v198[v204];
                    assert("Tensor range check" && 0 <= v199 && v199 < 1);
                    int v206;
                    v206 = 1088 * v199;
                    assert("Tensor range check" && 0 <= v201 && v201 < 8);
                    int v207;
                    v207 = 8 * v201;
                    int v208;
                    v208 = v207 + v206;
                    int v209;
                    v209 = 0;
                    #pragma unroll
                    while (while_method_5(v209)){
                        int v211;
                        v211 = 0;
                        #pragma unroll
                        while (while_method_5(v211)){
                            assert("Tensor range check" && 0 <= v209 && v209 < 2);
                            assert("Tensor range check" && 0 <= v211 && v211 < 2);
                            int v213;
                            v213 = 4 * v211;
                            int v214;
                            v214 = v213 + v208;
                            int v215;
                            v215 = 544 * v209;
                            int v216;
                            v216 = v215 + v214;
                            float v217;
                            v217 = v55[v216];
                            bool v218;
                            v218 = 0 <= v211;
                            bool v220;
                            if (v218){
                                bool v219;
                                v219 = v211 < 2;
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
                            bool v223;
                            v223 = 0 <= v209;
                            bool v225;
                            if (v223){
                                bool v224;
                                v224 = v209 < 2;
                                v225 = v224;
                            } else {
                                v225 = false;
                            }
                            bool v226;
                            v226 = v225 == false;
                            if (v226){
                                assert("The indices should be inside the range of the dimension." && v225);
                            } else {
                            }
                            int v228;
                            v228 = v209 * 2;
                            int v229;
                            v229 = v211 + v228;
                            v205.x[v229] = v217;
                            v211 += 1 ;
                        }
                        v209 += 1 ;
                    }
                    v201 += 1 ;
                }
                v199 += 1 ;
            }
            // Poping the loop unrolling to: 0
            // Pushing the loop unrolling to: 0
            int v230;
            v230 = 0;
            #pragma unroll
            while (while_method_3(v230)){
                int v232;
                v232 = 0;
                #pragma unroll
                while (while_method_1(v232)){
                    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v234 = v197[0];
                    assert("Tensor range check" && 0 <= v230 && v230 < 4);
                    int v235;
                    v235 = 1088 * v230;
                    assert("Tensor range check" && 0 <= v232 && v232 < 8);
                    int v236;
                    v236 = 8 * v232;
                    int v237;
                    v237 = v236 + v235;
                    int v238;
                    v238 = 0;
                    #pragma unroll
                    while (while_method_5(v238)){
                        int v240;
                        v240 = 0;
                        #pragma unroll
                        while (while_method_5(v240)){
                            assert("Tensor range check" && 0 <= v238 && v238 < 2);
                            assert("Tensor range check" && 0 <= v240 && v240 < 2);
                            int v242;
                            v242 = 544 * v240;
                            int v243;
                            v243 = v242 + v237;
                            int v244;
                            v244 = 4 * v238;
                            int v245;
                            v245 = v244 + v243;
                            float v246;
                            v246 = v39[v245];
                            bool v247;
                            v247 = 0 <= v240;
                            bool v249;
                            if (v247){
                                bool v248;
                                v248 = v240 < 2;
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
                            bool v252;
                            v252 = 0 <= v238;
                            bool v254;
                            if (v252){
                                bool v253;
                                v253 = v238 < 2;
                                v254 = v253;
                            } else {
                                v254 = false;
                            }
                            bool v255;
                            v255 = v254 == false;
                            if (v255){
                                assert("The indices should be inside the range of the dimension." && v254);
                            } else {
                            }
                            int v257;
                            v257 = v238 * 2;
                            int v258;
                            v258 = v240 + v257;
                            v234.x[v258] = v246;
                            v240 += 1 ;
                        }
                        v238 += 1 ;
                    }
                    int v259;
                    v259 = 0;
                    #pragma unroll
                    while (while_method_2(v259)){
                        assert("Tensor range check" && 0 <= v230 && v230 < 4);
                        assert("Tensor range check" && 0 <= v259 && v259 < 1);
                        int v261;
                        v261 = v230 + v259;
                        wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v262 = v57[v261];
                        assert("Tensor range check" && 0 <= v259 && v259 < 1);
                        assert("Tensor range check" && 0 <= v232 && v232 < 8);
                        int v263;
                        v263 = 8 * v259;
                        int v264;
                        v264 = v263 + v232;
                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v265 = v198[v264];
                        wmma::mma_sync(v262, v234, v265, v262);
                        v259 += 1 ;
                    }
                    v232 += 1 ;
                }
                v230 += 1 ;
            }
            // Poping the loop unrolling to: 0
            asm("barrier.cta.sync %0;" :: "r"(0));
            v114 += 1 ;
        }
        // Pushing the loop unrolling to: 0
        int v266;
        v266 = 0;
        #pragma unroll
        while (while_method_3(v266)){
            int v268;
            v268 = 0;
            #pragma unroll
            while (while_method_2(v268)){
                assert("Tensor range check" && 0 <= v266 && v266 < 4);
                assert("Tensor range check" && 0 <= v268 && v268 < 1);
                int v270;
                v270 = v266 + v268;
                wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v271 = v57[v270];
                assert("Tensor range check" && 0 <= v266 && v266 < 4);
                assert("Tensor range check" && 0 <= v268 && v268 < 1);
                int v272;
                v272 = 16 * v268;
                int v273;
                v273 = 2176 * v266;
                int v274;
                v274 = v273 + v272;
                float * v275;
                v275 = v23+v274;
                wmma::store_matrix_sync(v275, v271, 136, wmma::mem_row_major);
                v268 += 1 ;
            }
            v266 += 1 ;
        }
        // Poping the loop unrolling to: 0
        asm("barrier.cta.sync %0;" :: "r"(0));
        // Pushing the loop unrolling to: 0
        int v277;
        v277 = threadIdx.x;
        bool v278;
        v278 = 0 <= v277;
        bool v279;
        v279 = v278 == false;
        if (v279){
            assert("The index needs to be zero or positive." && v278);
        } else {
        }
        int v281;
        v281 = v277 % 32;
        int v282;
        v282 = v277 / 32;
        bool v283;
        v283 = v282 < 16;
        bool v284;
        v284 = v283 == false;
        if (v284){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v283);
        } else {
        }
        assert("Tensor range check" && 0 <= v282 && v282 < 16);
        assert("Tensor range check" && 0 <= v281 && v281 < 32);
        int v286;
        v286 = 4 * v281;
        int v287;
        v287 = 8192 * v282;
        int v288;
        v288 = v287 + v286;
        int v289;
        v289 = 136 * v282;
        int v290;
        v290 = v289 + v286;
        float * v291;
        v291 = v72+v288;
        float * v293;
        v293 = v8+v290;
        int v295;
        v295 = 0;
        #pragma unroll
        while (while_method_1(v295)){
            int v297;
            v297 = 0;
            #pragma unroll
            while (while_method_2(v297)){
                assert("Tensor range check" && 0 <= v295 && v295 < 8);
                assert("Tensor range check" && 0 <= v297 && v297 < 1);
                int v299;
                v299 = 128 * v297;
                int v300;
                v300 = 131072 * v295;
                int v301;
                v301 = v300 + v299;
                int v302;
                v302 = 2176 * v295;
                int v303;
                v303 = v302 + v299;
                int4* v304;
                v304 = reinterpret_cast<int4*>(v293 + v303);
                int4* v305;
                v305 = reinterpret_cast<int4*>(v291 + v301);
                assert("Pointer alignment check" && (unsigned long long)(v304) % 4 == 0 && (unsigned long long)(v305) % 4 == 0);
                *v305 = *v304;
                v297 += 1 ;
            }
            v295 += 1 ;
        }
        // Poping the loop unrolling to: 0
        asm("barrier.cta.sync %0;" :: "r"(0));
        v59 += 24 ;
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
