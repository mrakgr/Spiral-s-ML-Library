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
    v1 = v0 < 16l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 8l;
    return v1;
}
__device__ inline bool while_method_4(int v0){
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
    v17 = v16 < 1l;
    bool v18;
    v18 = v17 == false;
    if (v18){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v17);
    } else {
    }
    assert("Tensor range check" && 0 <= v16 && v16 < 1l);
    assert("Tensor range check" && 0 <= v15 && v15 < 8l);
    int v20;
    v20 = 16l * v15;
    int v21;
    v21 = 17408l * v16;
    int v22;
    v22 = v21 + v20;
    float * v23;
    v23 = v8+v22;
    assert("Tensor range check" && 0 <= v16 && v16 < 1l);
    int v25;
    v25 = 8704l * v16;
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
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v57[8l];
    int v58;
    v58 = 0l;
    while (while_method_0(v58)){
        int v60;
        v60 = 0l;
        while (while_method_0(v60)){
            assert("Tensor range check" && 0 <= v58 && v58 < 4l);
            assert("Tensor range check" && 0 <= v60 && v60 < 4l);
            int v62;
            v62 = 128l * v60;
            int v63;
            v63 = 65536l * v58;
            int v64;
            v64 = v63 + v62;
            float * v65;
            v65 = v2+v64;
            // Pushing the loop unrolling to: 0
            int v67;
            v67 = threadIdx.x;
            bool v68;
            v68 = 0l <= v67;
            bool v69;
            v69 = v68 == false;
            if (v69){
                assert("The index needs to be zero or positive." && v68);
            } else {
            }
            int v71;
            v71 = v67 % 32l;
            int v72;
            v72 = v67 / 32l;
            bool v73;
            v73 = v72 < 8l;
            bool v74;
            v74 = v73 == false;
            if (v74){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v73);
            } else {
            }
            assert("Tensor range check" && 0 <= v72 && v72 < 8l);
            assert("Tensor range check" && 0 <= v71 && v71 < 32l);
            int v76;
            v76 = 4l * v71;
            int v77;
            v77 = 136l * v72;
            int v78;
            v78 = v77 + v76;
            int v79;
            v79 = 512l * v72;
            int v80;
            v80 = v79 + v76;
            float * v81;
            v81 = v8+v78;
            float * v83;
            v83 = v65+v80;
            int v85;
            v85 = 0l;
            #pragma unroll
            while (while_method_1(v85)){
                int v87;
                v87 = 0l;
                #pragma unroll
                while (while_method_2(v87)){
                    assert("Tensor range check" && 0 <= v85 && v85 < 16l);
                    assert("Tensor range check" && 0 <= v87 && v87 < 1l);
                    int v89;
                    v89 = 128l * v87;
                    int v90;
                    v90 = 1088l * v85;
                    int v91;
                    v91 = v90 + v89;
                    int v92;
                    v92 = 4096l * v85;
                    int v93;
                    v93 = v92 + v89;
                    int4* v94;
                    v94 = reinterpret_cast<int4*>(v83 + v93);
                    int4* v95;
                    v95 = reinterpret_cast<int4*>(v81 + v91);
                    assert("Pointer alignment check" && (unsigned long long)(v94) % 4l == 0 && (unsigned long long)(v95) % 4l == 0);
                    *v95 = *v94;
                    v87 += 1l ;
                }
                v85 += 1l ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0l));
            int v96;
            v96 = 0l;
            #pragma unroll
            while (while_method_3(v96)){
                int v98;
                v98 = 0l;
                #pragma unroll
                while (while_method_2(v98)){
                    assert("Tensor range check" && 0 <= v96 && v96 < 8l);
                    assert("Tensor range check" && 0 <= v98 && v98 < 1l);
                    int v100;
                    v100 = v96 + v98;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v101 = v57[v100];
                    assert("Tensor range check" && 0 <= v96 && v96 < 8l);
                    assert("Tensor range check" && 0 <= v98 && v98 < 1l);
                    int v102;
                    v102 = 16l * v98;
                    int v103;
                    v103 = 2176l * v96;
                    int v104;
                    v104 = v103 + v102;
                    float * v105;
                    v105 = v23+v104;
                    wmma::load_matrix_sync(v101, v105, 136l, wmma::mem_row_major);
                    v98 += 1l ;
                }
                v96 += 1l ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0l));
            int v107;
            v107 = 0l;
            #pragma unroll
            while (while_method_3(v107)){
                assert("Tensor range check" && 0 <= v58 && v58 < 4l);
                assert("Tensor range check" && 0 <= v107 && v107 < 8l);
                int v109;
                v109 = 64l * v107;
                int v110;
                v110 = v109 + v63;
                float * v111;
                v111 = v0+v110;
                assert("Tensor range check" && 0 <= v60 && v60 < 4l);
                int v113;
                v113 = 65536l * v60;
                assert("Tensor range check" && 0 <= v107 && v107 < 8l);
                int v114;
                v114 = v109 + v113;
                float * v115;
                v115 = v1+v114;
                int v117;
                v117 = threadIdx.x;
                bool v118;
                v118 = 0l <= v117;
                bool v119;
                v119 = v118 == false;
                if (v119){
                    assert("The index needs to be zero or positive." && v118);
                } else {
                }
                int v121;
                v121 = v117 % 16l;
                int v122;
                v122 = v117 / 16l;
                bool v123;
                v123 = v122 < 16l;
                bool v124;
                v124 = v123 == false;
                if (v124){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v123);
                } else {
                }
                assert("Tensor range check" && 0 <= v122 && v122 < 16l);
                assert("Tensor range check" && 0 <= v121 && v121 < 16l);
                int v126;
                v126 = 4l * v121;
                int v127;
                v127 = 68l * v122;
                int v128;
                v128 = v127 + v126;
                int v129;
                v129 = 512l * v122;
                int v130;
                v130 = v129 + v126;
                float * v131;
                v131 = v6+v128;
                float * v133;
                v133 = v115+v130;
                int v135;
                v135 = 0l;
                #pragma unroll
                while (while_method_3(v135)){
                    int v137;
                    v137 = 0l;
                    #pragma unroll
                    while (while_method_2(v137)){
                        assert("Tensor range check" && 0 <= v135 && v135 < 8l);
                        assert("Tensor range check" && 0 <= v137 && v137 < 1l);
                        int v139;
                        v139 = 64l * v137;
                        int v140;
                        v140 = 1088l * v135;
                        int v141;
                        v141 = v140 + v139;
                        int v142;
                        v142 = 8192l * v135;
                        int v143;
                        v143 = v142 + v139;
                        float v144[4l];
                        int v145;
                        v145 = 0l;
                        #pragma unroll
                        while (while_method_0(v145)){
                            assert("Tensor range check" && 0 <= v145 && v145 < 4l);
                            int v147;
                            v147 = v145 + v143;
                            float v148;
                            v148 = v133[v147];
                            float v149;
                            v149 = wmma::__float_to_tf32(v148);
                            assert("Tensor range check" && 0 <= v145 && v145 < 4l);
                            v144[v145] = v149;
                            v145 += 1l ;
                        }
                        int4* v150;
                        v150 = reinterpret_cast<int4*>(v144 + 0l);
                        int4* v151;
                        v151 = reinterpret_cast<int4*>(v131 + v141);
                        assert("Pointer alignment check" && (unsigned long long)(v150) % 4l == 0 && (unsigned long long)(v151) % 4l == 0);
                        *v151 = *v150;
                        v137 += 1l ;
                    }
                    v135 += 1l ;
                }
                int v152;
                v152 = threadIdx.x;
                bool v153;
                v153 = 0l <= v152;
                bool v154;
                v154 = v153 == false;
                if (v154){
                    assert("The index needs to be zero or positive." && v153);
                } else {
                }
                int v156;
                v156 = v152 % 16l;
                int v157;
                v157 = v152 / 16l;
                bool v158;
                v158 = v157 < 16l;
                bool v159;
                v159 = v158 == false;
                if (v159){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v158);
                } else {
                }
                assert("Tensor range check" && 0 <= v157 && v157 < 16l);
                assert("Tensor range check" && 0 <= v156 && v156 < 16l);
                int v161;
                v161 = 4l * v156;
                int v162;
                v162 = 68l * v157;
                int v163;
                v163 = v162 + v161;
                int v164;
                v164 = 512l * v157;
                int v165;
                v165 = v164 + v161;
                float * v166;
                v166 = v4+v163;
                float * v168;
                v168 = v111+v165;
                int v170;
                v170 = 0l;
                #pragma unroll
                while (while_method_3(v170)){
                    int v172;
                    v172 = 0l;
                    #pragma unroll
                    while (while_method_2(v172)){
                        assert("Tensor range check" && 0 <= v170 && v170 < 8l);
                        assert("Tensor range check" && 0 <= v172 && v172 < 1l);
                        int v174;
                        v174 = 64l * v172;
                        int v175;
                        v175 = 1088l * v170;
                        int v176;
                        v176 = v175 + v174;
                        int v177;
                        v177 = 8192l * v170;
                        int v178;
                        v178 = v177 + v174;
                        float v179[4l];
                        int v180;
                        v180 = 0l;
                        #pragma unroll
                        while (while_method_0(v180)){
                            assert("Tensor range check" && 0 <= v180 && v180 < 4l);
                            int v182;
                            v182 = v180 + v178;
                            float v183;
                            v183 = v168[v182];
                            float v184;
                            v184 = wmma::__float_to_tf32(v183);
                            assert("Tensor range check" && 0 <= v180 && v180 < 4l);
                            v179[v180] = v184;
                            v180 += 1l ;
                        }
                        int4* v185;
                        v185 = reinterpret_cast<int4*>(v179 + 0l);
                        int4* v186;
                        v186 = reinterpret_cast<int4*>(v166 + v176);
                        assert("Pointer alignment check" && (unsigned long long)(v185) % 4l == 0 && (unsigned long long)(v186) % 4l == 0);
                        *v186 = *v185;
                        v172 += 1l ;
                    }
                    v170 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v187[64l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v188[8l];
                int v189;
                v189 = 0l;
                #pragma unroll
                while (while_method_3(v189)){
                    int v191;
                    v191 = 0l;
                    #pragma unroll
                    while (while_method_3(v191)){
                        assert("Tensor range check" && 0 <= v189 && v189 < 8l);
                        assert("Tensor range check" && 0 <= v191 && v191 < 8l);
                        int v193;
                        v193 = 8l * v189;
                        int v194;
                        v194 = v193 + v191;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v195 = v187[v194];
                        assert("Tensor range check" && 0 <= v189 && v189 < 8l);
                        int v196;
                        v196 = 1088l * v189;
                        assert("Tensor range check" && 0 <= v191 && v191 < 8l);
                        int v197;
                        v197 = 8l * v191;
                        int v198;
                        v198 = v197 + v196;
                        int v199;
                        v199 = 0l;
                        #pragma unroll
                        while (while_method_4(v199)){
                            int v201;
                            v201 = 0l;
                            #pragma unroll
                            while (while_method_4(v201)){
                                assert("Tensor range check" && 0 <= v199 && v199 < 2l);
                                assert("Tensor range check" && 0 <= v201 && v201 < 2l);
                                int v203;
                                v203 = 544l * v201;
                                int v204;
                                v204 = v203 + v198;
                                int v205;
                                v205 = 4l * v199;
                                int v206;
                                v206 = v205 + v204;
                                float v207;
                                v207 = v39[v206];
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
                        v191 += 1l ;
                    }
                    v189 += 1l ;
                }
                int v220;
                v220 = 0l;
                #pragma unroll
                while (while_method_2(v220)){
                    int v222;
                    v222 = 0l;
                    #pragma unroll
                    while (while_method_3(v222)){
                        assert("Tensor range check" && 0 <= v220 && v220 < 1l);
                        assert("Tensor range check" && 0 <= v222 && v222 < 8l);
                        int v224;
                        v224 = 8l * v220;
                        int v225;
                        v225 = v224 + v222;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v226 = v188[v225];
                        assert("Tensor range check" && 0 <= v220 && v220 < 1l);
                        int v227;
                        v227 = 1088l * v220;
                        assert("Tensor range check" && 0 <= v222 && v222 < 8l);
                        int v228;
                        v228 = 8l * v222;
                        int v229;
                        v229 = v228 + v227;
                        int v230;
                        v230 = 0l;
                        #pragma unroll
                        while (while_method_4(v230)){
                            int v232;
                            v232 = 0l;
                            #pragma unroll
                            while (while_method_4(v232)){
                                assert("Tensor range check" && 0 <= v230 && v230 < 2l);
                                assert("Tensor range check" && 0 <= v232 && v232 < 2l);
                                int v234;
                                v234 = 4l * v232;
                                int v235;
                                v235 = v234 + v229;
                                int v236;
                                v236 = 544l * v230;
                                int v237;
                                v237 = v236 + v235;
                                float v238;
                                v238 = v55[v237];
                                bool v239;
                                v239 = 0l <= v232;
                                bool v241;
                                if (v239){
                                    bool v240;
                                    v240 = v232 < 2l;
                                    v241 = v240;
                                } else {
                                    v241 = false;
                                }
                                bool v242;
                                v242 = v241 == false;
                                if (v242){
                                    assert("The indices should be inside the range of the dimension." && v241);
                                } else {
                                }
                                bool v244;
                                v244 = 0l <= v230;
                                bool v246;
                                if (v244){
                                    bool v245;
                                    v245 = v230 < 2l;
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
                                int v249;
                                v249 = v230 * 2l;
                                int v250;
                                v250 = v232 + v249;
                                v226.x[v250] = v238;
                                v232 += 1l ;
                            }
                            v230 += 1l ;
                        }
                        v222 += 1l ;
                    }
                    v220 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                int v251;
                v251 = 0l;
                #pragma unroll
                while (while_method_3(v251)){
                    int v253;
                    v253 = 0l;
                    #pragma unroll
                    while (while_method_2(v253)){
                        int v255;
                        v255 = 0l;
                        #pragma unroll
                        while (while_method_3(v255)){
                            assert("Tensor range check" && 0 <= v251 && v251 < 8l);
                            assert("Tensor range check" && 0 <= v253 && v253 < 1l);
                            int v257;
                            v257 = v251 + v253;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v258 = v57[v257];
                            assert("Tensor range check" && 0 <= v251 && v251 < 8l);
                            assert("Tensor range check" && 0 <= v255 && v255 < 8l);
                            int v259;
                            v259 = 8l * v251;
                            int v260;
                            v260 = v259 + v255;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v261 = v187[v260];
                            assert("Tensor range check" && 0 <= v253 && v253 < 1l);
                            assert("Tensor range check" && 0 <= v255 && v255 < 8l);
                            int v262;
                            v262 = 8l * v253;
                            int v263;
                            v263 = v262 + v255;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v264 = v188[v263];
                            wmma::mma_sync(v258, v261, v264, v258);
                            v255 += 1l ;
                        }
                        v253 += 1l ;
                    }
                    v251 += 1l ;
                }
                v107 += 1l ;
            }
            int v265;
            v265 = 0l;
            #pragma unroll
            while (while_method_3(v265)){
                int v267;
                v267 = 0l;
                #pragma unroll
                while (while_method_2(v267)){
                    assert("Tensor range check" && 0 <= v265 && v265 < 8l);
                    assert("Tensor range check" && 0 <= v267 && v267 < 1l);
                    int v269;
                    v269 = v265 + v267;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v270 = v57[v269];
                    assert("Tensor range check" && 0 <= v265 && v265 < 8l);
                    assert("Tensor range check" && 0 <= v267 && v267 < 1l);
                    int v271;
                    v271 = 16l * v267;
                    int v272;
                    v272 = 2176l * v265;
                    int v273;
                    v273 = v272 + v271;
                    float * v274;
                    v274 = v23+v273;
                    wmma::store_matrix_sync(v274, v270, 136l, wmma::mem_row_major);
                    v267 += 1l ;
                }
                v265 += 1l ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0l));
            int v276;
            v276 = threadIdx.x;
            bool v277;
            v277 = 0l <= v276;
            bool v278;
            v278 = v277 == false;
            if (v278){
                assert("The index needs to be zero or positive." && v277);
            } else {
            }
            int v280;
            v280 = v276 % 32l;
            int v281;
            v281 = v276 / 32l;
            bool v282;
            v282 = v281 < 8l;
            bool v283;
            v283 = v282 == false;
            if (v283){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v282);
            } else {
            }
            assert("Tensor range check" && 0 <= v281 && v281 < 8l);
            assert("Tensor range check" && 0 <= v280 && v280 < 32l);
            int v285;
            v285 = 4l * v280;
            int v286;
            v286 = 512l * v281;
            int v287;
            v287 = v286 + v285;
            int v288;
            v288 = 136l * v281;
            int v289;
            v289 = v288 + v285;
            float * v290;
            v290 = v65+v287;
            float * v292;
            v292 = v8+v289;
            int v294;
            v294 = 0l;
            #pragma unroll
            while (while_method_1(v294)){
                int v296;
                v296 = 0l;
                #pragma unroll
                while (while_method_2(v296)){
                    assert("Tensor range check" && 0 <= v294 && v294 < 16l);
                    assert("Tensor range check" && 0 <= v296 && v296 < 1l);
                    int v298;
                    v298 = 128l * v296;
                    int v299;
                    v299 = 4096l * v294;
                    int v300;
                    v300 = v299 + v298;
                    int v301;
                    v301 = 1088l * v294;
                    int v302;
                    v302 = v301 + v298;
                    int4* v303;
                    v303 = reinterpret_cast<int4*>(v292 + v302);
                    int4* v304;
                    v304 = reinterpret_cast<int4*>(v290 + v300);
                    assert("Pointer alignment check" && (unsigned long long)(v303) % 4l == 0 && (unsigned long long)(v304) % 4l == 0);
                    *v304 = *v303;
                    v296 += 1l ;
                }
                v294 += 1l ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0l));
            // Poping the loop unrolling to: 0
            v60 += 1l ;
        }
        v58 += 1l ;
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
options.append('--maxrregcount=256')
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
    max_blocks_per_sm(cp.cuda.Device(),raw_module.get_function('entry0'),256,is_print=True)
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
    v17.max_dynamic_shared_size_bytes = 81920 
    v17((24,),(256,),(v2, v1, v0),shared_mem=81920)
    del v1, v2, v17
    v18 = cp.max(cp.abs(v0-v7))
    del v0, v7
    return v18

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
