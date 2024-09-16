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
        v78 = v74 % 32l;
        int v79;
        v79 = v74 / 32l;
        bool v80;
        v80 = v79 < 16l;
        bool v81;
        v81 = v80 == false;
        if (v81){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v80);
        } else {
        }
        assert("Tensor range check" && 0 <= v79 && v79 < 16l);
        assert("Tensor range check" && 0 <= v78 && v78 < 32l);
        int v83;
        v83 = 4l * v78;
        int v84;
        v84 = 136l * v79;
        int v85;
        v85 = v84 + v83;
        int v86;
        v86 = 8192l * v79;
        int v87;
        v87 = v86 + v83;
        float * v88;
        v88 = v8+v85;
        float * v90;
        v90 = v72+v87;
        int v92;
        v92 = 0l;
        #pragma unroll
        while (while_method_1(v92)){
            int v94;
            v94 = 0l;
            #pragma unroll
            while (while_method_2(v94)){
                assert("Tensor range check" && 0 <= v92 && v92 < 8l);
                assert("Tensor range check" && 0 <= v94 && v94 < 1l);
                int v96;
                v96 = 128l * v94;
                int v97;
                v97 = 2176l * v92;
                int v98;
                v98 = v97 + v96;
                int v99;
                v99 = 131072l * v92;
                int v100;
                v100 = v99 + v96;
                int4* v101;
                v101 = reinterpret_cast<int4*>(v90 + v100);
                int4* v102;
                v102 = reinterpret_cast<int4*>(v88 + v98);
                assert("Pointer alignment check" && (unsigned long long)(v101) % 4l == 0 && (unsigned long long)(v102) % 4l == 0);
                *v102 = *v101;
                v94 += 1l ;
            }
            v92 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v103;
        v103 = 0l;
        #pragma unroll
        while (while_method_3(v103)){
            int v105;
            v105 = 0l;
            #pragma unroll
            while (while_method_2(v105)){
                assert("Tensor range check" && 0 <= v103 && v103 < 4l);
                assert("Tensor range check" && 0 <= v105 && v105 < 1l);
                int v107;
                v107 = v103 + v105;
                wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v108 = v57[v107];
                assert("Tensor range check" && 0 <= v103 && v103 < 4l);
                assert("Tensor range check" && 0 <= v105 && v105 < 1l);
                int v109;
                v109 = 16l * v105;
                int v110;
                v110 = 2176l * v103;
                int v111;
                v111 = v110 + v109;
                float * v112;
                v112 = v23+v111;
                wmma::load_matrix_sync(v108, v112, 136l, wmma::mem_row_major);
                v105 += 1l ;
            }
            v103 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        // Poping the loop unrolling to: 0
        int v114;
        v114 = 0l;
        while (while_method_4(v114)){
            assert("Tensor range check" && 0 <= v65 && v65 < 64l);
            int v116;
            v116 = 524288l * v65;
            assert("Tensor range check" && 0 <= v114 && v114 < 64l);
            int v117;
            v117 = 64l * v114;
            int v118;
            v118 = v117 + v116;
            float * v119;
            v119 = v0+v118;
            assert("Tensor range check" && 0 <= v64 && v64 < 64l);
            int v121;
            v121 = 524288l * v64;
            assert("Tensor range check" && 0 <= v114 && v114 < 64l);
            int v122;
            v122 = v117 + v121;
            float * v123;
            v123 = v1+v122;
            // Pushing the loop unrolling to: 0
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
            v129 = v125 % 16l;
            int v130;
            v130 = v125 / 16l;
            bool v131;
            v131 = v130 < 32l;
            bool v132;
            v132 = v131 == false;
            if (v132){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v131);
            } else {
            }
            assert("Tensor range check" && 0 <= v130 && v130 < 32l);
            assert("Tensor range check" && 0 <= v129 && v129 < 16l);
            int v134;
            v134 = 4l * v129;
            int v135;
            v135 = 68l * v130;
            int v136;
            v136 = v135 + v134;
            int v137;
            v137 = 4096l * v130;
            int v138;
            v138 = v137 + v134;
            float * v139;
            v139 = v6+v136;
            float * v141;
            v141 = v123+v138;
            int v143;
            v143 = 0l;
            #pragma unroll
            while (while_method_3(v143)){
                int v145;
                v145 = 0l;
                #pragma unroll
                while (while_method_2(v145)){
                    assert("Tensor range check" && 0 <= v143 && v143 < 4l);
                    assert("Tensor range check" && 0 <= v145 && v145 < 1l);
                    int v147;
                    v147 = 64l * v145;
                    int v148;
                    v148 = 2176l * v143;
                    int v149;
                    v149 = v148 + v147;
                    int v150;
                    v150 = 131072l * v143;
                    int v151;
                    v151 = v150 + v147;
                    float v152[4l];
                    int v153;
                    v153 = 0l;
                    #pragma unroll
                    while (while_method_3(v153)){
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
            int v160;
            v160 = threadIdx.x;
            bool v161;
            v161 = 0l <= v160;
            bool v162;
            v162 = v161 == false;
            if (v162){
                assert("The index needs to be zero or positive." && v161);
            } else {
            }
            int v164;
            v164 = v160 % 16l;
            int v165;
            v165 = v160 / 16l;
            bool v166;
            v166 = v165 < 32l;
            bool v167;
            v167 = v166 == false;
            if (v167){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v166);
            } else {
            }
            assert("Tensor range check" && 0 <= v165 && v165 < 32l);
            assert("Tensor range check" && 0 <= v164 && v164 < 16l);
            int v169;
            v169 = 4l * v164;
            int v170;
            v170 = 68l * v165;
            int v171;
            v171 = v170 + v169;
            int v172;
            v172 = 4096l * v165;
            int v173;
            v173 = v172 + v169;
            float * v174;
            v174 = v4+v171;
            float * v176;
            v176 = v119+v173;
            int v178;
            v178 = 0l;
            #pragma unroll
            while (while_method_3(v178)){
                int v180;
                v180 = 0l;
                #pragma unroll
                while (while_method_2(v180)){
                    assert("Tensor range check" && 0 <= v178 && v178 < 4l);
                    assert("Tensor range check" && 0 <= v180 && v180 < 1l);
                    int v182;
                    v182 = 64l * v180;
                    int v183;
                    v183 = 2176l * v178;
                    int v184;
                    v184 = v183 + v182;
                    int v185;
                    v185 = 131072l * v178;
                    int v186;
                    v186 = v185 + v182;
                    float v187[4l];
                    int v188;
                    v188 = 0l;
                    #pragma unroll
                    while (while_method_3(v188)){
                        assert("Tensor range check" && 0 <= v188 && v188 < 4l);
                        int v190;
                        v190 = v188 + v186;
                        float v191;
                        v191 = v176[v190];
                        float v192;
                        v192 = wmma::__float_to_tf32(v191);
                        assert("Tensor range check" && 0 <= v188 && v188 < 4l);
                        v187[v188] = v192;
                        v188 += 1l ;
                    }
                    int4* v193;
                    v193 = reinterpret_cast<int4*>(v187 + 0l);
                    int4* v194;
                    v194 = reinterpret_cast<int4*>(v174 + v184);
                    assert("Pointer alignment check" && (unsigned long long)(v193) % 4l == 0 && (unsigned long long)(v194) % 4l == 0);
                    *v194 = *v193;
                    v180 += 1l ;
                }
                v178 += 1l ;
            }
            // Poping the loop unrolling to: 0
            asm("barrier.cta.sync %0;" :: "r"(0l));
            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v195[1l];
            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v196[8l];
            // Pushing the loop unrolling to: 0
            int v197;
            v197 = 0l;
            #pragma unroll
            while (while_method_2(v197)){
                int v199;
                v199 = 0l;
                #pragma unroll
                while (while_method_1(v199)){
                    assert("Tensor range check" && 0 <= v197 && v197 < 1l);
                    assert("Tensor range check" && 0 <= v199 && v199 < 8l);
                    int v201;
                    v201 = 8l * v197;
                    int v202;
                    v202 = v201 + v199;
                    wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v203 = v196[v202];
                    assert("Tensor range check" && 0 <= v197 && v197 < 1l);
                    int v204;
                    v204 = 1088l * v197;
                    assert("Tensor range check" && 0 <= v199 && v199 < 8l);
                    int v205;
                    v205 = 8l * v199;
                    int v206;
                    v206 = v205 + v204;
                    int v207;
                    v207 = 0l;
                    #pragma unroll
                    while (while_method_5(v207)){
                        int v209;
                        v209 = 0l;
                        #pragma unroll
                        while (while_method_5(v209)){
                            assert("Tensor range check" && 0 <= v207 && v207 < 2l);
                            assert("Tensor range check" && 0 <= v209 && v209 < 2l);
                            int v211;
                            v211 = 4l * v209;
                            int v212;
                            v212 = v211 + v206;
                            int v213;
                            v213 = 544l * v207;
                            int v214;
                            v214 = v213 + v212;
                            float v215;
                            v215 = v55[v214];
                            bool v216;
                            v216 = 0l <= v209;
                            bool v218;
                            if (v216){
                                bool v217;
                                v217 = v209 < 2l;
                                v218 = v217;
                            } else {
                                v218 = false;
                            }
                            bool v219;
                            v219 = v218 == false;
                            if (v219){
                                assert("The indices should be inside the range of the dimension." && v218);
                            } else {
                            }
                            bool v221;
                            v221 = 0l <= v207;
                            bool v223;
                            if (v221){
                                bool v222;
                                v222 = v207 < 2l;
                                v223 = v222;
                            } else {
                                v223 = false;
                            }
                            bool v224;
                            v224 = v223 == false;
                            if (v224){
                                assert("The indices should be inside the range of the dimension." && v223);
                            } else {
                            }
                            int v226;
                            v226 = v207 * 2l;
                            int v227;
                            v227 = v209 + v226;
                            v203.x[v227] = v215;
                            v209 += 1l ;
                        }
                        v207 += 1l ;
                    }
                    v199 += 1l ;
                }
                v197 += 1l ;
            }
            // Poping the loop unrolling to: 0
            // Pushing the loop unrolling to: 0
            int v228;
            v228 = 0l;
            #pragma unroll
            while (while_method_3(v228)){
                int v230;
                v230 = 0l;
                #pragma unroll
                while (while_method_1(v230)){
                    wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v232 = v195[0l];
                    assert("Tensor range check" && 0 <= v228 && v228 < 4l);
                    int v233;
                    v233 = 1088l * v228;
                    assert("Tensor range check" && 0 <= v230 && v230 < 8l);
                    int v234;
                    v234 = 8l * v230;
                    int v235;
                    v235 = v234 + v233;
                    int v236;
                    v236 = 0l;
                    #pragma unroll
                    while (while_method_5(v236)){
                        int v238;
                        v238 = 0l;
                        #pragma unroll
                        while (while_method_5(v238)){
                            assert("Tensor range check" && 0 <= v236 && v236 < 2l);
                            assert("Tensor range check" && 0 <= v238 && v238 < 2l);
                            int v240;
                            v240 = 544l * v238;
                            int v241;
                            v241 = v240 + v235;
                            int v242;
                            v242 = 4l * v236;
                            int v243;
                            v243 = v242 + v241;
                            float v244;
                            v244 = v39[v243];
                            bool v245;
                            v245 = 0l <= v238;
                            bool v247;
                            if (v245){
                                bool v246;
                                v246 = v238 < 2l;
                                v247 = v246;
                            } else {
                                v247 = false;
                            }
                            bool v248;
                            v248 = v247 == false;
                            if (v248){
                                assert("The indices should be inside the range of the dimension." && v247);
                            } else {
                            }
                            bool v250;
                            v250 = 0l <= v236;
                            bool v252;
                            if (v250){
                                bool v251;
                                v251 = v236 < 2l;
                                v252 = v251;
                            } else {
                                v252 = false;
                            }
                            bool v253;
                            v253 = v252 == false;
                            if (v253){
                                assert("The indices should be inside the range of the dimension." && v252);
                            } else {
                            }
                            int v255;
                            v255 = v236 * 2l;
                            int v256;
                            v256 = v238 + v255;
                            v232.x[v256] = v244;
                            v238 += 1l ;
                        }
                        v236 += 1l ;
                    }
                    int v257;
                    v257 = 0l;
                    #pragma unroll
                    while (while_method_2(v257)){
                        assert("Tensor range check" && 0 <= v228 && v228 < 4l);
                        assert("Tensor range check" && 0 <= v257 && v257 < 1l);
                        int v259;
                        v259 = v228 + v257;
                        wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v260 = v57[v259];
                        assert("Tensor range check" && 0 <= v257 && v257 < 1l);
                        assert("Tensor range check" && 0 <= v230 && v230 < 8l);
                        int v261;
                        v261 = 8l * v257;
                        int v262;
                        v262 = v261 + v230;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v263 = v196[v262];
                        wmma::mma_sync(v260, v232, v263, v260);
                        v257 += 1l ;
                    }
                    v230 += 1l ;
                }
                v228 += 1l ;
            }
            // Poping the loop unrolling to: 0
            asm("barrier.cta.sync %0;" :: "r"(0l));
            v114 += 1l ;
        }
        // Pushing the loop unrolling to: 0
        int v264;
        v264 = 0l;
        #pragma unroll
        while (while_method_3(v264)){
            int v266;
            v266 = 0l;
            #pragma unroll
            while (while_method_2(v266)){
                assert("Tensor range check" && 0 <= v264 && v264 < 4l);
                assert("Tensor range check" && 0 <= v266 && v266 < 1l);
                int v268;
                v268 = v264 + v266;
                wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v269 = v57[v268];
                assert("Tensor range check" && 0 <= v264 && v264 < 4l);
                assert("Tensor range check" && 0 <= v266 && v266 < 1l);
                int v270;
                v270 = 16l * v266;
                int v271;
                v271 = 2176l * v264;
                int v272;
                v272 = v271 + v270;
                float * v273;
                v273 = v23+v272;
                wmma::store_matrix_sync(v273, v269, 136l, wmma::mem_row_major);
                v266 += 1l ;
            }
            v264 += 1l ;
        }
        // Poping the loop unrolling to: 0
        asm("barrier.cta.sync %0;" :: "r"(0l));
        // Pushing the loop unrolling to: 0
        int v275;
        v275 = threadIdx.x;
        bool v276;
        v276 = 0l <= v275;
        bool v277;
        v277 = v276 == false;
        if (v277){
            assert("The index needs to be zero or positive." && v276);
        } else {
        }
        int v279;
        v279 = v275 % 32l;
        int v280;
        v280 = v275 / 32l;
        bool v281;
        v281 = v280 < 16l;
        bool v282;
        v282 = v281 == false;
        if (v282){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v281);
        } else {
        }
        assert("Tensor range check" && 0 <= v280 && v280 < 16l);
        assert("Tensor range check" && 0 <= v279 && v279 < 32l);
        int v284;
        v284 = 4l * v279;
        int v285;
        v285 = 8192l * v280;
        int v286;
        v286 = v285 + v284;
        int v287;
        v287 = 136l * v280;
        int v288;
        v288 = v287 + v284;
        float * v289;
        v289 = v72+v286;
        float * v291;
        v291 = v8+v288;
        int v293;
        v293 = 0l;
        #pragma unroll
        while (while_method_1(v293)){
            int v295;
            v295 = 0l;
            #pragma unroll
            while (while_method_2(v295)){
                assert("Tensor range check" && 0 <= v293 && v293 < 8l);
                assert("Tensor range check" && 0 <= v295 && v295 < 1l);
                int v297;
                v297 = 128l * v295;
                int v298;
                v298 = 131072l * v293;
                int v299;
                v299 = v298 + v297;
                int v300;
                v300 = 2176l * v293;
                int v301;
                v301 = v300 + v297;
                int4* v302;
                v302 = reinterpret_cast<int4*>(v291 + v301);
                int4* v303;
                v303 = reinterpret_cast<int4*>(v289 + v299);
                assert("Pointer alignment check" && (unsigned long long)(v302) % 4l == 0 && (unsigned long long)(v303) % 4l == 0);
                *v303 = *v302;
                v295 += 1l ;
            }
            v293 += 1l ;
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
