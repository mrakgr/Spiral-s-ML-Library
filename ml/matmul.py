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
    v1 = v0 < 4;
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
    v58 = 0;
    while (while_method_0(v58)){
        int v60;
        v60 = 0;
        while (while_method_0(v60)){
            assert("Tensor range check" && 0 <= v58 && v58 < 4);
            assert("Tensor range check" && 0 <= v60 && v60 < 4);
            int v62;
            v62 = 128 * v60;
            int v63;
            v63 = 65536 * v58;
            int v64;
            v64 = v63 + v62;
            float * v65;
            v65 = v2+v64;
            // Pushing the loop unrolling to: 0
            int v67;
            v67 = threadIdx.x;
            bool v68;
            v68 = 0 <= v67;
            bool v69;
            v69 = v68 == false;
            if (v69){
                assert("The index needs to be zero or positive." && v68);
            } else {
            }
            int v71;
            v71 = v67 % 32;
            int v72;
            v72 = v67 / 32;
            bool v73;
            v73 = v72 < 16;
            bool v74;
            v74 = v73 == false;
            if (v74){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v73);
            } else {
            }
            assert("Tensor range check" && 0 <= v72 && v72 < 16);
            assert("Tensor range check" && 0 <= v71 && v71 < 32);
            int v76;
            v76 = 4 * v71;
            int v77;
            v77 = 136 * v72;
            int v78;
            v78 = v77 + v76;
            int v79;
            v79 = 512 * v72;
            int v80;
            v80 = v79 + v76;
            float * v81;
            v81 = v8+v78;
            float * v83;
            v83 = v65+v80;
            int v85;
            v85 = 0;
            #pragma unroll
            while (while_method_1(v85)){
                int v87;
                v87 = 0;
                #pragma unroll
                while (while_method_2(v87)){
                    assert("Tensor range check" && 0 <= v85 && v85 < 8);
                    assert("Tensor range check" && 0 <= v87 && v87 < 1);
                    int v89;
                    v89 = 128 * v87;
                    int v90;
                    v90 = 2176 * v85;
                    int v91;
                    v91 = v90 + v89;
                    int v92;
                    v92 = 8192 * v85;
                    int v93;
                    v93 = v92 + v89;
                    int4* v94;
                    v94 = reinterpret_cast<int4*>(v83 + v93);
                    int4* v95;
                    v95 = reinterpret_cast<int4*>(v81 + v91);
                    assert("Pointer alignment check" && (unsigned long long)(v94) % 4 == 0 && (unsigned long long)(v95) % 4 == 0);
                    *v95 = *v94;
                    v87 += 1 ;
                }
                v85 += 1 ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0));
            int v96;
            v96 = 0;
            #pragma unroll
            while (while_method_0(v96)){
                int v98;
                v98 = 0;
                #pragma unroll
                while (while_method_2(v98)){
                    assert("Tensor range check" && 0 <= v96 && v96 < 4);
                    assert("Tensor range check" && 0 <= v98 && v98 < 1);
                    int v100;
                    v100 = v96 + v98;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v101 = v57[v100];
                    assert("Tensor range check" && 0 <= v96 && v96 < 4);
                    assert("Tensor range check" && 0 <= v98 && v98 < 1);
                    int v102;
                    v102 = 16 * v98;
                    int v103;
                    v103 = 2176 * v96;
                    int v104;
                    v104 = v103 + v102;
                    float * v105;
                    v105 = v23+v104;
                    wmma::load_matrix_sync(v101, v105, 136, wmma::mem_row_major);
                    v98 += 1 ;
                }
                v96 += 1 ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0));
            int v107;
            v107 = 0;
            #pragma unroll
            while (while_method_1(v107)){
                assert("Tensor range check" && 0 <= v58 && v58 < 4);
                assert("Tensor range check" && 0 <= v107 && v107 < 8);
                int v109;
                v109 = 64 * v107;
                int v110;
                v110 = v109 + v63;
                float * v111;
                v111 = v0+v110;
                assert("Tensor range check" && 0 <= v60 && v60 < 4);
                int v113;
                v113 = 65536 * v60;
                assert("Tensor range check" && 0 <= v107 && v107 < 8);
                int v114;
                v114 = v109 + v113;
                float * v115;
                v115 = v1+v114;
                int v117;
                v117 = threadIdx.x;
                bool v118;
                v118 = 0 <= v117;
                bool v119;
                v119 = v118 == false;
                if (v119){
                    assert("The index needs to be zero or positive." && v118);
                } else {
                }
                int v121;
                v121 = v117 % 16;
                int v122;
                v122 = v117 / 16;
                bool v123;
                v123 = v122 < 32;
                bool v124;
                v124 = v123 == false;
                if (v124){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v123);
                } else {
                }
                assert("Tensor range check" && 0 <= v122 && v122 < 32);
                assert("Tensor range check" && 0 <= v121 && v121 < 16);
                int v126;
                v126 = 4 * v121;
                int v127;
                v127 = 68 * v122;
                int v128;
                v128 = v127 + v126;
                int v129;
                v129 = 512 * v122;
                int v130;
                v130 = v129 + v126;
                float * v131;
                v131 = v6+v128;
                float * v133;
                v133 = v115+v130;
                int v135;
                v135 = 0;
                #pragma unroll
                while (while_method_0(v135)){
                    int v137;
                    v137 = 0;
                    #pragma unroll
                    while (while_method_2(v137)){
                        assert("Tensor range check" && 0 <= v135 && v135 < 4);
                        assert("Tensor range check" && 0 <= v137 && v137 < 1);
                        int v139;
                        v139 = 64 * v137;
                        int v140;
                        v140 = 2176 * v135;
                        int v141;
                        v141 = v140 + v139;
                        int v142;
                        v142 = 16384 * v135;
                        int v143;
                        v143 = v142 + v139;
                        float v144[4];
                        int4* v145;
                        v145 = reinterpret_cast<int4*>(v133 + v143);
                        int4* v146;
                        v146 = reinterpret_cast<int4*>(v144 + 0);
                        assert("Pointer alignment check" && (unsigned long long)(v145) % 4 == 0 && (unsigned long long)(v146) % 4 == 0);
                        *v146 = *v145;
                        int v147;
                        v147 = 0;
                        #pragma unroll
                        while (while_method_0(v147)){
                            assert("Tensor range check" && 0 <= v147 && v147 < 4);
                            float v149;
                            v149 = v144[v147];
                            float v150;
                            v150 = wmma::__float_to_tf32(v149);
                            assert("Tensor range check" && 0 <= v147 && v147 < 4);
                            v144[v147] = v150;
                            v147 += 1 ;
                        }
                        int4* v151;
                        v151 = reinterpret_cast<int4*>(v144 + 0);
                        int4* v152;
                        v152 = reinterpret_cast<int4*>(v131 + v141);
                        assert("Pointer alignment check" && (unsigned long long)(v151) % 4 == 0 && (unsigned long long)(v152) % 4 == 0);
                        *v152 = *v151;
                        v137 += 1 ;
                    }
                    v135 += 1 ;
                }
                int v153;
                v153 = threadIdx.x;
                bool v154;
                v154 = 0 <= v153;
                bool v155;
                v155 = v154 == false;
                if (v155){
                    assert("The index needs to be zero or positive." && v154);
                } else {
                }
                int v157;
                v157 = v153 % 16;
                int v158;
                v158 = v153 / 16;
                bool v159;
                v159 = v158 < 32;
                bool v160;
                v160 = v159 == false;
                if (v160){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v159);
                } else {
                }
                assert("Tensor range check" && 0 <= v158 && v158 < 32);
                assert("Tensor range check" && 0 <= v157 && v157 < 16);
                int v162;
                v162 = 4 * v157;
                int v163;
                v163 = 68 * v158;
                int v164;
                v164 = v163 + v162;
                int v165;
                v165 = 512 * v158;
                int v166;
                v166 = v165 + v162;
                float * v167;
                v167 = v4+v164;
                float * v169;
                v169 = v111+v166;
                int v171;
                v171 = 0;
                #pragma unroll
                while (while_method_0(v171)){
                    int v173;
                    v173 = 0;
                    #pragma unroll
                    while (while_method_2(v173)){
                        assert("Tensor range check" && 0 <= v171 && v171 < 4);
                        assert("Tensor range check" && 0 <= v173 && v173 < 1);
                        int v175;
                        v175 = 64 * v173;
                        int v176;
                        v176 = 2176 * v171;
                        int v177;
                        v177 = v176 + v175;
                        int v178;
                        v178 = 16384 * v171;
                        int v179;
                        v179 = v178 + v175;
                        float v180[4];
                        int4* v181;
                        v181 = reinterpret_cast<int4*>(v169 + v179);
                        int4* v182;
                        v182 = reinterpret_cast<int4*>(v180 + 0);
                        assert("Pointer alignment check" && (unsigned long long)(v181) % 4 == 0 && (unsigned long long)(v182) % 4 == 0);
                        *v182 = *v181;
                        int v183;
                        v183 = 0;
                        #pragma unroll
                        while (while_method_0(v183)){
                            assert("Tensor range check" && 0 <= v183 && v183 < 4);
                            float v185;
                            v185 = v180[v183];
                            float v186;
                            v186 = wmma::__float_to_tf32(v185);
                            assert("Tensor range check" && 0 <= v183 && v183 < 4);
                            v180[v183] = v186;
                            v183 += 1 ;
                        }
                        int4* v187;
                        v187 = reinterpret_cast<int4*>(v180 + 0);
                        int4* v188;
                        v188 = reinterpret_cast<int4*>(v167 + v177);
                        assert("Pointer alignment check" && (unsigned long long)(v187) % 4 == 0 && (unsigned long long)(v188) % 4 == 0);
                        *v188 = *v187;
                        v173 += 1 ;
                    }
                    v171 += 1 ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0));
                wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v189[32];
                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v190[8];
                int v191;
                v191 = 0;
                #pragma unroll
                while (while_method_0(v191)){
                    int v193;
                    v193 = 0;
                    #pragma unroll
                    while (while_method_1(v193)){
                        assert("Tensor range check" && 0 <= v191 && v191 < 4);
                        assert("Tensor range check" && 0 <= v193 && v193 < 8);
                        int v195;
                        v195 = 8 * v191;
                        int v196;
                        v196 = v195 + v193;
                        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v197 = v189[v196];
                        assert("Tensor range check" && 0 <= v191 && v191 < 4);
                        int v198;
                        v198 = 1088 * v191;
                        assert("Tensor range check" && 0 <= v193 && v193 < 8);
                        int v199;
                        v199 = 8 * v193;
                        int v200;
                        v200 = v199 + v198;
                        int v201;
                        v201 = 0;
                        #pragma unroll
                        while (while_method_3(v201)){
                            int v203;
                            v203 = 0;
                            #pragma unroll
                            while (while_method_3(v203)){
                                assert("Tensor range check" && 0 <= v201 && v201 < 2);
                                assert("Tensor range check" && 0 <= v203 && v203 < 2);
                                int v205;
                                v205 = 544 * v203;
                                int v206;
                                v206 = v205 + v200;
                                int v207;
                                v207 = 4 * v201;
                                int v208;
                                v208 = v207 + v206;
                                float v209;
                                v209 = v39[v208];
                                bool v210;
                                v210 = 0 <= v203;
                                bool v212;
                                if (v210){
                                    bool v211;
                                    v211 = v203 < 2;
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
                                v215 = 0 <= v201;
                                bool v217;
                                if (v215){
                                    bool v216;
                                    v216 = v201 < 2;
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
                                v220 = v201 * 2;
                                int v221;
                                v221 = v203 + v220;
                                v197.x[v221] = v209;
                                v203 += 1 ;
                            }
                            v201 += 1 ;
                        }
                        v193 += 1 ;
                    }
                    v191 += 1 ;
                }
                int v222;
                v222 = 0;
                #pragma unroll
                while (while_method_2(v222)){
                    int v224;
                    v224 = 0;
                    #pragma unroll
                    while (while_method_1(v224)){
                        assert("Tensor range check" && 0 <= v222 && v222 < 1);
                        assert("Tensor range check" && 0 <= v224 && v224 < 8);
                        int v226;
                        v226 = 8 * v222;
                        int v227;
                        v227 = v226 + v224;
                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v228 = v190[v227];
                        assert("Tensor range check" && 0 <= v222 && v222 < 1);
                        int v229;
                        v229 = 1088 * v222;
                        assert("Tensor range check" && 0 <= v224 && v224 < 8);
                        int v230;
                        v230 = 8 * v224;
                        int v231;
                        v231 = v230 + v229;
                        int v232;
                        v232 = 0;
                        #pragma unroll
                        while (while_method_3(v232)){
                            int v234;
                            v234 = 0;
                            #pragma unroll
                            while (while_method_3(v234)){
                                assert("Tensor range check" && 0 <= v232 && v232 < 2);
                                assert("Tensor range check" && 0 <= v234 && v234 < 2);
                                int v236;
                                v236 = 4 * v234;
                                int v237;
                                v237 = v236 + v231;
                                int v238;
                                v238 = 544 * v232;
                                int v239;
                                v239 = v238 + v237;
                                float v240;
                                v240 = v55[v239];
                                bool v241;
                                v241 = 0 <= v234;
                                bool v243;
                                if (v241){
                                    bool v242;
                                    v242 = v234 < 2;
                                    v243 = v242;
                                } else {
                                    v243 = false;
                                }
                                bool v244;
                                v244 = v243 == false;
                                if (v244){
                                    assert("The indices should be inside the range of the dimension." && v243);
                                } else {
                                }
                                bool v246;
                                v246 = 0 <= v232;
                                bool v248;
                                if (v246){
                                    bool v247;
                                    v247 = v232 < 2;
                                    v248 = v247;
                                } else {
                                    v248 = false;
                                }
                                bool v249;
                                v249 = v248 == false;
                                if (v249){
                                    assert("The indices should be inside the range of the dimension." && v248);
                                } else {
                                }
                                int v251;
                                v251 = v232 * 2;
                                int v252;
                                v252 = v234 + v251;
                                v228.x[v252] = v240;
                                v234 += 1 ;
                            }
                            v232 += 1 ;
                        }
                        v224 += 1 ;
                    }
                    v222 += 1 ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0));
                int v253;
                v253 = 0;
                #pragma unroll
                while (while_method_0(v253)){
                    int v255;
                    v255 = 0;
                    #pragma unroll
                    while (while_method_2(v255)){
                        int v257;
                        v257 = 0;
                        #pragma unroll
                        while (while_method_1(v257)){
                            assert("Tensor range check" && 0 <= v253 && v253 < 4);
                            assert("Tensor range check" && 0 <= v255 && v255 < 1);
                            int v259;
                            v259 = v253 + v255;
                            wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v260 = v57[v259];
                            assert("Tensor range check" && 0 <= v253 && v253 < 4);
                            assert("Tensor range check" && 0 <= v257 && v257 < 8);
                            int v261;
                            v261 = 8 * v253;
                            int v262;
                            v262 = v261 + v257;
                            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v263 = v189[v262];
                            assert("Tensor range check" && 0 <= v255 && v255 < 1);
                            assert("Tensor range check" && 0 <= v257 && v257 < 8);
                            int v264;
                            v264 = 8 * v255;
                            int v265;
                            v265 = v264 + v257;
                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v266 = v190[v265];
                            wmma::mma_sync(v260, v263, v266, v260);
                            v257 += 1 ;
                        }
                        v255 += 1 ;
                    }
                    v253 += 1 ;
                }
                v107 += 1 ;
            }
            int v267;
            v267 = 0;
            #pragma unroll
            while (while_method_0(v267)){
                int v269;
                v269 = 0;
                #pragma unroll
                while (while_method_2(v269)){
                    assert("Tensor range check" && 0 <= v267 && v267 < 4);
                    assert("Tensor range check" && 0 <= v269 && v269 < 1);
                    int v271;
                    v271 = v267 + v269;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v272 = v57[v271];
                    assert("Tensor range check" && 0 <= v267 && v267 < 4);
                    assert("Tensor range check" && 0 <= v269 && v269 < 1);
                    int v273;
                    v273 = 16 * v269;
                    int v274;
                    v274 = 2176 * v267;
                    int v275;
                    v275 = v274 + v273;
                    float * v276;
                    v276 = v23+v275;
                    wmma::store_matrix_sync(v276, v272, 136, wmma::mem_row_major);
                    v269 += 1 ;
                }
                v267 += 1 ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0));
            int v278;
            v278 = threadIdx.x;
            bool v279;
            v279 = 0 <= v278;
            bool v280;
            v280 = v279 == false;
            if (v280){
                assert("The index needs to be zero or positive." && v279);
            } else {
            }
            int v282;
            v282 = v278 % 32;
            int v283;
            v283 = v278 / 32;
            bool v284;
            v284 = v283 < 16;
            bool v285;
            v285 = v284 == false;
            if (v285){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v284);
            } else {
            }
            assert("Tensor range check" && 0 <= v283 && v283 < 16);
            assert("Tensor range check" && 0 <= v282 && v282 < 32);
            int v287;
            v287 = 4 * v282;
            int v288;
            v288 = 512 * v283;
            int v289;
            v289 = v288 + v287;
            int v290;
            v290 = 136 * v283;
            int v291;
            v291 = v290 + v287;
            float * v292;
            v292 = v65+v289;
            float * v294;
            v294 = v8+v291;
            int v296;
            v296 = 0;
            #pragma unroll
            while (while_method_1(v296)){
                int v298;
                v298 = 0;
                #pragma unroll
                while (while_method_2(v298)){
                    assert("Tensor range check" && 0 <= v296 && v296 < 8);
                    assert("Tensor range check" && 0 <= v298 && v298 < 1);
                    int v300;
                    v300 = 128 * v298;
                    int v301;
                    v301 = 8192 * v296;
                    int v302;
                    v302 = v301 + v300;
                    int v303;
                    v303 = 2176 * v296;
                    int v304;
                    v304 = v303 + v300;
                    int4* v305;
                    v305 = reinterpret_cast<int4*>(v294 + v304);
                    int4* v306;
                    v306 = reinterpret_cast<int4*>(v292 + v302);
                    assert("Pointer alignment check" && (unsigned long long)(v305) % 4 == 0 && (unsigned long long)(v306) % 4 == 0);
                    *v306 = *v305;
                    v298 += 1 ;
                }
                v296 += 1 ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0));
            // Poping the loop unrolling to: 0
            v60 += 1 ;
        }
        v58 += 1 ;
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

from max_blocks_per_sm import max_blocks_per_sm
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
