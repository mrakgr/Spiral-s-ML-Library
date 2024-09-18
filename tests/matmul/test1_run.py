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
    v1 = v0 < 8192;
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
    v6 = reinterpret_cast<float *>(&v3[17408ull]);
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
    v17 = v16 < 1;
    bool v18;
    v18 = v17 == false;
    if (v18){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v17);
    } else {
    }
    assert("Tensor range check" && 0 <= v16 && v16 < 1);
    assert("Tensor range check" && 0 <= v15 && v15 < 8);
    int v20;
    v20 = 16 * v15;
    int v21;
    v21 = 8704 * v16;
    int v22;
    v22 = v21 + v20;
    float * v23;
    v23 = v8+v22;
    assert("Tensor range check" && 0 <= v16 && v16 < 1);
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
        v66 = v65 < 128;
        bool v67;
        v67 = v66 == false;
        if (v67){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v66);
        } else {
        }
        assert("Tensor range check" && 0 <= v65 && v65 < 128);
        assert("Tensor range check" && 0 <= v64 && v64 < 64);
        int v69;
        v69 = 128 * v64;
        int v70;
        v70 = 524288 * v65;
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
        v80 = v79 < 8;
        bool v81;
        v81 = v80 == false;
        if (v81){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v80);
        } else {
        }
        assert("Tensor range check" && 0 <= v79 && v79 < 8);
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
                v97 = 1088 * v92;
                int v98;
                v98 = v97 + v96;
                int v99;
                v99 = 65536 * v92;
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
            assert("Tensor range check" && 0 <= v65 && v65 < 128);
            int v116;
            v116 = 262144 * v65;
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
            v131 = v130 < 16;
            bool v132;
            v132 = v131 == false;
            if (v132){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v131);
            } else {
            }
            assert("Tensor range check" && 0 <= v130 && v130 < 16);
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
            while (while_method_1(v143)){
                int v145;
                v145 = 0;
                #pragma unroll
                while (while_method_2(v145)){
                    assert("Tensor range check" && 0 <= v143 && v143 < 8);
                    assert("Tensor range check" && 0 <= v145 && v145 < 1);
                    int v147;
                    v147 = 64 * v145;
                    int v148;
                    v148 = 1088 * v143;
                    int v149;
                    v149 = v148 + v147;
                    int v150;
                    v150 = 65536 * v143;
                    int v151;
                    v151 = v150 + v147;
                    int4* v152;
                    v152 = reinterpret_cast<int4*>(v141 + v151);
                    int4* v153;
                    v153 = reinterpret_cast<int4*>(v139 + v149);
                    assert("Pointer alignment check" && (unsigned long long)(v152) % 4 == 0 && (unsigned long long)(v153) % 4 == 0);
                    *v153 = *v152;
                    v145 += 1 ;
                }
                v143 += 1 ;
            }
            int v154;
            v154 = threadIdx.x;
            bool v155;
            v155 = 0 <= v154;
            bool v156;
            v156 = v155 == false;
            if (v156){
                assert("The index needs to be zero or positive." && v155);
            } else {
            }
            int v158;
            v158 = v154 % 16;
            int v159;
            v159 = v154 / 16;
            bool v160;
            v160 = v159 < 16;
            bool v161;
            v161 = v160 == false;
            if (v161){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v160);
            } else {
            }
            assert("Tensor range check" && 0 <= v159 && v159 < 16);
            assert("Tensor range check" && 0 <= v158 && v158 < 16);
            int v163;
            v163 = 4 * v158;
            int v164;
            v164 = 68 * v159;
            int v165;
            v165 = v164 + v163;
            int v166;
            v166 = 4096 * v159;
            int v167;
            v167 = v166 + v163;
            float * v168;
            v168 = v4+v165;
            float * v170;
            v170 = v119+v167;
            int v172;
            v172 = 0;
            #pragma unroll
            while (while_method_3(v172)){
                int v174;
                v174 = 0;
                #pragma unroll
                while (while_method_2(v174)){
                    assert("Tensor range check" && 0 <= v172 && v172 < 4);
                    assert("Tensor range check" && 0 <= v174 && v174 < 1);
                    int v176;
                    v176 = 64 * v174;
                    int v177;
                    v177 = 1088 * v172;
                    int v178;
                    v178 = v177 + v176;
                    int v179;
                    v179 = 65536 * v172;
                    int v180;
                    v180 = v179 + v176;
                    int4* v181;
                    v181 = reinterpret_cast<int4*>(v170 + v180);
                    int4* v182;
                    v182 = reinterpret_cast<int4*>(v168 + v178);
                    assert("Pointer alignment check" && (unsigned long long)(v181) % 4 == 0 && (unsigned long long)(v182) % 4 == 0);
                    *v182 = *v181;
                    v174 += 1 ;
                }
                v172 += 1 ;
            }
            // Poping the loop unrolling to: 0
            asm("barrier.cta.sync %0;" :: "r"(0));
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v183[1];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v184[8];
            // Pushing the loop unrolling to: 0
            int v185;
            v185 = 0;
            #pragma unroll
            while (while_method_2(v185)){
                int v187;
                v187 = 0;
                #pragma unroll
                while (while_method_1(v187)){
                    assert("Tensor range check" && 0 <= v185 && v185 < 1);
                    assert("Tensor range check" && 0 <= v187 && v187 < 8);
                    int v189;
                    v189 = 8 * v185;
                    int v190;
                    v190 = v189 + v187;
                    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v191 = v184[v190];
                    assert("Tensor range check" && 0 <= v185 && v185 < 1);
                    int v192;
                    v192 = 1088 * v185;
                    assert("Tensor range check" && 0 <= v187 && v187 < 8);
                    int v193;
                    v193 = 8 * v187;
                    int v194;
                    v194 = v193 + v192;
                    int v195;
                    v195 = 0;
                    #pragma unroll
                    while (while_method_5(v195)){
                        int v197;
                        v197 = 0;
                        #pragma unroll
                        while (while_method_5(v197)){
                            assert("Tensor range check" && 0 <= v195 && v195 < 2);
                            assert("Tensor range check" && 0 <= v197 && v197 < 2);
                            int v199;
                            v199 = 4 * v197;
                            int v200;
                            v200 = v199 + v194;
                            int v201;
                            v201 = 544 * v195;
                            int v202;
                            v202 = v201 + v200;
                            float v203;
                            v203 = v55[v202];
                            bool v204;
                            v204 = 0 <= v197;
                            bool v206;
                            if (v204){
                                bool v205;
                                v205 = v197 < 2;
                                v206 = v205;
                            } else {
                                v206 = false;
                            }
                            bool v207;
                            v207 = v206 == false;
                            if (v207){
                                assert("The indices should be inside the range of the dimension." && v206);
                            } else {
                            }
                            bool v209;
                            v209 = 0 <= v195;
                            bool v211;
                            if (v209){
                                bool v210;
                                v210 = v195 < 2;
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
                            int v214;
                            v214 = v195 * 2;
                            int v215;
                            v215 = v197 + v214;
                            v191.x[v215] = wmma::__float_to_tf32(v203);
                            v197 += 1 ;
                        }
                        v195 += 1 ;
                    }
                    v187 += 1 ;
                }
                v185 += 1 ;
            }
            // Poping the loop unrolling to: 0
            // Pushing the loop unrolling to: 0
            int v216;
            v216 = 0;
            #pragma unroll
            while (while_method_3(v216)){
                int v218;
                v218 = 0;
                #pragma unroll
                while (while_method_1(v218)){
                    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v220 = v183[0];
                    assert("Tensor range check" && 0 <= v216 && v216 < 4);
                    int v221;
                    v221 = 1088 * v216;
                    assert("Tensor range check" && 0 <= v218 && v218 < 8);
                    int v222;
                    v222 = 8 * v218;
                    int v223;
                    v223 = v222 + v221;
                    int v224;
                    v224 = 0;
                    #pragma unroll
                    while (while_method_5(v224)){
                        int v226;
                        v226 = 0;
                        #pragma unroll
                        while (while_method_5(v226)){
                            assert("Tensor range check" && 0 <= v224 && v224 < 2);
                            assert("Tensor range check" && 0 <= v226 && v226 < 2);
                            int v228;
                            v228 = 544 * v226;
                            int v229;
                            v229 = v228 + v223;
                            int v230;
                            v230 = 4 * v224;
                            int v231;
                            v231 = v230 + v229;
                            float v232;
                            v232 = v39[v231];
                            bool v233;
                            v233 = 0 <= v226;
                            bool v235;
                            if (v233){
                                bool v234;
                                v234 = v226 < 2;
                                v235 = v234;
                            } else {
                                v235 = false;
                            }
                            bool v236;
                            v236 = v235 == false;
                            if (v236){
                                assert("The indices should be inside the range of the dimension." && v235);
                            } else {
                            }
                            bool v238;
                            v238 = 0 <= v224;
                            bool v240;
                            if (v238){
                                bool v239;
                                v239 = v224 < 2;
                                v240 = v239;
                            } else {
                                v240 = false;
                            }
                            bool v241;
                            v241 = v240 == false;
                            if (v241){
                                assert("The indices should be inside the range of the dimension." && v240);
                            } else {
                            }
                            int v243;
                            v243 = v224 * 2;
                            int v244;
                            v244 = v226 + v243;
                            v220.x[v244] = wmma::__float_to_tf32(v232);
                            v226 += 1 ;
                        }
                        v224 += 1 ;
                    }
                    int v245;
                    v245 = 0;
                    #pragma unroll
                    while (while_method_2(v245)){
                        assert("Tensor range check" && 0 <= v216 && v216 < 4);
                        assert("Tensor range check" && 0 <= v245 && v245 < 1);
                        int v247;
                        v247 = v216 + v245;
                        wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v248 = v57[v247];
                        assert("Tensor range check" && 0 <= v245 && v245 < 1);
                        assert("Tensor range check" && 0 <= v218 && v218 < 8);
                        int v249;
                        v249 = 8 * v245;
                        int v250;
                        v250 = v249 + v218;
                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v251 = v184[v250];
                        wmma::mma_sync(v248, v220, v251, v248);
                        v245 += 1 ;
                    }
                    v218 += 1 ;
                }
                v216 += 1 ;
            }
            // Poping the loop unrolling to: 0
            asm("barrier.cta.sync %0;" :: "r"(0));
            v114 += 1 ;
        }
        // Pushing the loop unrolling to: 0
        int v252;
        v252 = 0;
        #pragma unroll
        while (while_method_3(v252)){
            int v254;
            v254 = 0;
            #pragma unroll
            while (while_method_2(v254)){
                assert("Tensor range check" && 0 <= v252 && v252 < 4);
                assert("Tensor range check" && 0 <= v254 && v254 < 1);
                int v256;
                v256 = v252 + v254;
                wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v257 = v57[v256];
                assert("Tensor range check" && 0 <= v252 && v252 < 4);
                assert("Tensor range check" && 0 <= v254 && v254 < 1);
                int v258;
                v258 = 16 * v254;
                int v259;
                v259 = 2176 * v252;
                int v260;
                v260 = v259 + v258;
                float * v261;
                v261 = v23+v260;
                wmma::store_matrix_sync(v261, v257, 136, wmma::mem_row_major);
                v254 += 1 ;
            }
            v252 += 1 ;
        }
        // Poping the loop unrolling to: 0
        asm("barrier.cta.sync %0;" :: "r"(0));
        // Pushing the loop unrolling to: 0
        int v263;
        v263 = threadIdx.x;
        bool v264;
        v264 = 0 <= v263;
        bool v265;
        v265 = v264 == false;
        if (v265){
            assert("The index needs to be zero or positive." && v264);
        } else {
        }
        int v267;
        v267 = v263 % 32;
        int v268;
        v268 = v263 / 32;
        bool v269;
        v269 = v268 < 8;
        bool v270;
        v270 = v269 == false;
        if (v270){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v269);
        } else {
        }
        assert("Tensor range check" && 0 <= v268 && v268 < 8);
        assert("Tensor range check" && 0 <= v267 && v267 < 32);
        int v272;
        v272 = 4 * v267;
        int v273;
        v273 = 8192 * v268;
        int v274;
        v274 = v273 + v272;
        int v275;
        v275 = 136 * v268;
        int v276;
        v276 = v275 + v272;
        float * v277;
        v277 = v72+v274;
        float * v279;
        v279 = v8+v276;
        int v281;
        v281 = 0;
        #pragma unroll
        while (while_method_1(v281)){
            int v283;
            v283 = 0;
            #pragma unroll
            while (while_method_2(v283)){
                assert("Tensor range check" && 0 <= v281 && v281 < 8);
                assert("Tensor range check" && 0 <= v283 && v283 < 1);
                int v285;
                v285 = 128 * v283;
                int v286;
                v286 = 65536 * v281;
                int v287;
                v287 = v286 + v285;
                int v288;
                v288 = 1088 * v281;
                int v289;
                v289 = v288 + v285;
                int4* v290;
                v290 = reinterpret_cast<int4*>(v279 + v289);
                int4* v291;
                v291 = reinterpret_cast<int4*>(v277 + v287);
                assert("Pointer alignment check" && (unsigned long long)(v290) % 4 == 0 && (unsigned long long)(v291) % 4 == 0);
                *v291 = *v290;
                v283 += 1 ;
            }
            v281 += 1 ;
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
options.append('--maxrregcount=255')
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
    print(f'Threads per block, blocks per grid: {256}, {24}')
    v17((24,),(256,),(v2, v1, v0),shared_mem=98304)
    del v1, v2, v17
    v18 = cp.max(cp.abs(v0-v7))
    del v0, v7
    return v18

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
