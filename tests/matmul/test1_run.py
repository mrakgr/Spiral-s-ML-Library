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
    v1 = v0 < 16l;
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
        int v61;
        v61 = v59 % 2l;
        bool v62;
        v62 = v61 == 1l;
        bool v63;
        v63 = 0l <= v59;
        bool v64;
        v64 = v63 == false;
        if (v64){
            assert("The index needs to be zero or positive." && v63);
        } else {
        }
        int v66;
        v66 = v59 % 4l;
        int v67;
        v67 = v59 / 4l;
        bool v68;
        v68 = v67 < 4l;
        bool v69;
        v69 = v68 == false;
        if (v69){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v68);
        } else {
        }
        assert("Tensor range check" && 0 <= v67 && v67 < 4l);
        assert("Tensor range check" && 0 <= v66 && v66 < 4l);
        int v71;
        v71 = 128l * v66;
        int v72;
        v72 = 65536l * v67;
        int v73;
        v73 = v72 + v71;
        float * v74;
        v74 = v2+v73;
        // Pushing the loop unrolling to: 0
        int v76;
        v76 = threadIdx.x;
        bool v77;
        v77 = 0l <= v76;
        bool v78;
        v78 = v77 == false;
        if (v78){
            assert("The index needs to be zero or positive." && v77);
        } else {
        }
        int v80;
        v80 = v76 % 32l;
        int v81;
        v81 = v76 / 32l;
        bool v82;
        v82 = v81 < 16l;
        bool v83;
        v83 = v82 == false;
        if (v83){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v82);
        } else {
        }
        assert("Tensor range check" && 0 <= v81 && v81 < 16l);
        assert("Tensor range check" && 0 <= v80 && v80 < 32l);
        int v85;
        v85 = 4l * v80;
        int v86;
        v86 = 136l * v81;
        int v87;
        v87 = v86 + v85;
        int v88;
        v88 = 512l * v81;
        int v89;
        v89 = v88 + v85;
        float * v90;
        v90 = v8+v87;
        float * v92;
        v92 = v74+v89;
        int v94;
        v94 = 0l;
        #pragma unroll
        while (while_method_1(v94)){
            int v96;
            v96 = 0l;
            #pragma unroll
            while (while_method_2(v96)){
                assert("Tensor range check" && 0 <= v94 && v94 < 8l);
                assert("Tensor range check" && 0 <= v96 && v96 < 1l);
                int v98;
                v98 = 128l * v96;
                int v99;
                v99 = 2176l * v94;
                int v100;
                v100 = v99 + v98;
                int v101;
                v101 = 8192l * v94;
                int v102;
                v102 = v101 + v98;
                int4* v103;
                v103 = reinterpret_cast<int4*>(v92 + v102);
                int4* v104;
                v104 = reinterpret_cast<int4*>(v90 + v100);
                assert("Pointer alignment check" && (unsigned long long)(v103) % 4l == 0 && (unsigned long long)(v104) % 4l == 0);
                *v104 = *v103;
                v96 += 1l ;
            }
            v94 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v105;
        v105 = 0l;
        #pragma unroll
        while (while_method_3(v105)){
            int v107;
            v107 = 0l;
            #pragma unroll
            while (while_method_2(v107)){
                assert("Tensor range check" && 0 <= v105 && v105 < 4l);
                assert("Tensor range check" && 0 <= v107 && v107 < 1l);
                int v109;
                v109 = v105 + v107;
                wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v110 = v57[v109];
                assert("Tensor range check" && 0 <= v105 && v105 < 4l);
                assert("Tensor range check" && 0 <= v107 && v107 < 1l);
                int v111;
                v111 = 16l * v107;
                int v112;
                v112 = 2176l * v105;
                int v113;
                v113 = v112 + v111;
                float * v114;
                v114 = v23+v113;
                wmma::load_matrix_sync(v110, v114, 136l, wmma::mem_row_major);
                v107 += 1l ;
            }
            v105 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        // Poping the loop unrolling to: 0
        int v116;
        v116 = 0l;
        while (while_method_1(v116)){
            int v120;
            if (v62){
                int v118;
                v118 = 8l - v116;
                int v119;
                v119 = v118 - 1l;
                v120 = v119;
            } else {
                v120 = v116;
            }
            assert("Tensor range check" && 0 <= v67 && v67 < 4l);
            assert("Tensor range check" && 0 <= v120 && v120 < 8l);
            int v121;
            v121 = 64l * v120;
            int v122;
            v122 = v121 + v72;
            float * v123;
            v123 = v0+v122;
            assert("Tensor range check" && 0 <= v66 && v66 < 4l);
            int v125;
            v125 = 65536l * v66;
            assert("Tensor range check" && 0 <= v120 && v120 < 8l);
            int v126;
            v126 = v121 + v125;
            float * v127;
            v127 = v1+v126;
            // Pushing the loop unrolling to: 0
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
            v133 = v129 % 16l;
            int v134;
            v134 = v129 / 16l;
            bool v135;
            v135 = v134 < 32l;
            bool v136;
            v136 = v135 == false;
            if (v136){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v135);
            } else {
            }
            assert("Tensor range check" && 0 <= v134 && v134 < 32l);
            assert("Tensor range check" && 0 <= v133 && v133 < 16l);
            int v138;
            v138 = 4l * v133;
            int v139;
            v139 = 68l * v134;
            int v140;
            v140 = v139 + v138;
            int v141;
            v141 = 512l * v134;
            int v142;
            v142 = v141 + v138;
            float * v143;
            v143 = v6+v140;
            float * v145;
            v145 = v127+v142;
            int v147;
            v147 = 0l;
            #pragma unroll
            while (while_method_3(v147)){
                int v149;
                v149 = 0l;
                #pragma unroll
                while (while_method_2(v149)){
                    assert("Tensor range check" && 0 <= v147 && v147 < 4l);
                    assert("Tensor range check" && 0 <= v149 && v149 < 1l);
                    int v151;
                    v151 = 64l * v149;
                    int v152;
                    v152 = 2176l * v147;
                    int v153;
                    v153 = v152 + v151;
                    int v154;
                    v154 = 16384l * v147;
                    int v155;
                    v155 = v154 + v151;
                    float v156[4l];
                    int v157;
                    v157 = 0l;
                    #pragma unroll
                    while (while_method_3(v157)){
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
            int v164;
            v164 = threadIdx.x;
            bool v165;
            v165 = 0l <= v164;
            bool v166;
            v166 = v165 == false;
            if (v166){
                assert("The index needs to be zero or positive." && v165);
            } else {
            }
            int v168;
            v168 = v164 % 16l;
            int v169;
            v169 = v164 / 16l;
            bool v170;
            v170 = v169 < 32l;
            bool v171;
            v171 = v170 == false;
            if (v171){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v170);
            } else {
            }
            assert("Tensor range check" && 0 <= v169 && v169 < 32l);
            assert("Tensor range check" && 0 <= v168 && v168 < 16l);
            int v173;
            v173 = 4l * v168;
            int v174;
            v174 = 68l * v169;
            int v175;
            v175 = v174 + v173;
            int v176;
            v176 = 512l * v169;
            int v177;
            v177 = v176 + v173;
            float * v178;
            v178 = v4+v175;
            float * v180;
            v180 = v123+v177;
            int v182;
            v182 = 0l;
            #pragma unroll
            while (while_method_3(v182)){
                int v184;
                v184 = 0l;
                #pragma unroll
                while (while_method_2(v184)){
                    assert("Tensor range check" && 0 <= v182 && v182 < 4l);
                    assert("Tensor range check" && 0 <= v184 && v184 < 1l);
                    int v186;
                    v186 = 64l * v184;
                    int v187;
                    v187 = 2176l * v182;
                    int v188;
                    v188 = v187 + v186;
                    int v189;
                    v189 = 16384l * v182;
                    int v190;
                    v190 = v189 + v186;
                    float v191[4l];
                    int v192;
                    v192 = 0l;
                    #pragma unroll
                    while (while_method_3(v192)){
                        assert("Tensor range check" && 0 <= v192 && v192 < 4l);
                        int v194;
                        v194 = v192 + v190;
                        float v195;
                        v195 = v180[v194];
                        float v196;
                        v196 = wmma::__float_to_tf32(v195);
                        assert("Tensor range check" && 0 <= v192 && v192 < 4l);
                        v191[v192] = v196;
                        v192 += 1l ;
                    }
                    int4* v197;
                    v197 = reinterpret_cast<int4*>(v191 + 0l);
                    int4* v198;
                    v198 = reinterpret_cast<int4*>(v178 + v188);
                    assert("Pointer alignment check" && (unsigned long long)(v197) % 4l == 0 && (unsigned long long)(v198) % 4l == 0);
                    *v198 = *v197;
                    v184 += 1l ;
                }
                v182 += 1l ;
            }
            // Poping the loop unrolling to: 0
            asm("barrier.cta.sync %0;" :: "r"(0l));
            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v199[1l];
            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v200[8l];
            // Pushing the loop unrolling to: 0
            int v201;
            v201 = 0l;
            #pragma unroll
            while (while_method_2(v201)){
                int v203;
                v203 = 0l;
                #pragma unroll
                while (while_method_1(v203)){
                    assert("Tensor range check" && 0 <= v201 && v201 < 1l);
                    assert("Tensor range check" && 0 <= v203 && v203 < 8l);
                    int v205;
                    v205 = 8l * v201;
                    int v206;
                    v206 = v205 + v203;
                    wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v207 = v200[v206];
                    assert("Tensor range check" && 0 <= v201 && v201 < 1l);
                    int v208;
                    v208 = 1088l * v201;
                    assert("Tensor range check" && 0 <= v203 && v203 < 8l);
                    int v209;
                    v209 = 8l * v203;
                    int v210;
                    v210 = v209 + v208;
                    int v211;
                    v211 = 0l;
                    #pragma unroll
                    while (while_method_4(v211)){
                        int v213;
                        v213 = 0l;
                        #pragma unroll
                        while (while_method_4(v213)){
                            assert("Tensor range check" && 0 <= v211 && v211 < 2l);
                            assert("Tensor range check" && 0 <= v213 && v213 < 2l);
                            int v215;
                            v215 = 4l * v213;
                            int v216;
                            v216 = v215 + v210;
                            int v217;
                            v217 = 544l * v211;
                            int v218;
                            v218 = v217 + v216;
                            float v219;
                            v219 = v55[v218];
                            bool v220;
                            v220 = 0l <= v213;
                            bool v222;
                            if (v220){
                                bool v221;
                                v221 = v213 < 2l;
                                v222 = v221;
                            } else {
                                v222 = false;
                            }
                            bool v223;
                            v223 = v222 == false;
                            if (v223){
                                assert("The indices should be inside the range of the dimension." && v222);
                            } else {
                            }
                            bool v225;
                            v225 = 0l <= v211;
                            bool v227;
                            if (v225){
                                bool v226;
                                v226 = v211 < 2l;
                                v227 = v226;
                            } else {
                                v227 = false;
                            }
                            bool v228;
                            v228 = v227 == false;
                            if (v228){
                                assert("The indices should be inside the range of the dimension." && v227);
                            } else {
                            }
                            int v230;
                            v230 = v211 * 2l;
                            int v231;
                            v231 = v213 + v230;
                            v207.x[v231] = v219;
                            v213 += 1l ;
                        }
                        v211 += 1l ;
                    }
                    v203 += 1l ;
                }
                v201 += 1l ;
            }
            // Poping the loop unrolling to: 0
            // Pushing the loop unrolling to: 0
            int v232;
            v232 = 0l;
            #pragma unroll
            while (while_method_3(v232)){
                int v234;
                v234 = 0l;
                #pragma unroll
                while (while_method_1(v234)){
                    wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v236 = v199[0l];
                    assert("Tensor range check" && 0 <= v232 && v232 < 4l);
                    int v237;
                    v237 = 1088l * v232;
                    assert("Tensor range check" && 0 <= v234 && v234 < 8l);
                    int v238;
                    v238 = 8l * v234;
                    int v239;
                    v239 = v238 + v237;
                    int v240;
                    v240 = 0l;
                    #pragma unroll
                    while (while_method_4(v240)){
                        int v242;
                        v242 = 0l;
                        #pragma unroll
                        while (while_method_4(v242)){
                            assert("Tensor range check" && 0 <= v240 && v240 < 2l);
                            assert("Tensor range check" && 0 <= v242 && v242 < 2l);
                            int v244;
                            v244 = 544l * v242;
                            int v245;
                            v245 = v244 + v239;
                            int v246;
                            v246 = 4l * v240;
                            int v247;
                            v247 = v246 + v245;
                            float v248;
                            v248 = v39[v247];
                            bool v249;
                            v249 = 0l <= v242;
                            bool v251;
                            if (v249){
                                bool v250;
                                v250 = v242 < 2l;
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
                            bool v254;
                            v254 = 0l <= v240;
                            bool v256;
                            if (v254){
                                bool v255;
                                v255 = v240 < 2l;
                                v256 = v255;
                            } else {
                                v256 = false;
                            }
                            bool v257;
                            v257 = v256 == false;
                            if (v257){
                                assert("The indices should be inside the range of the dimension." && v256);
                            } else {
                            }
                            int v259;
                            v259 = v240 * 2l;
                            int v260;
                            v260 = v242 + v259;
                            v236.x[v260] = v248;
                            v242 += 1l ;
                        }
                        v240 += 1l ;
                    }
                    int v261;
                    v261 = 0l;
                    #pragma unroll
                    while (while_method_2(v261)){
                        assert("Tensor range check" && 0 <= v232 && v232 < 4l);
                        assert("Tensor range check" && 0 <= v261 && v261 < 1l);
                        int v263;
                        v263 = v232 + v261;
                        wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v264 = v57[v263];
                        assert("Tensor range check" && 0 <= v261 && v261 < 1l);
                        assert("Tensor range check" && 0 <= v234 && v234 < 8l);
                        int v265;
                        v265 = 8l * v261;
                        int v266;
                        v266 = v265 + v234;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v267 = v200[v266];
                        wmma::mma_sync(v264, v236, v267, v264);
                        v261 += 1l ;
                    }
                    v234 += 1l ;
                }
                v232 += 1l ;
            }
            // Poping the loop unrolling to: 0
            asm("barrier.cta.sync %0;" :: "r"(0l));
            v116 += 1l ;
        }
        // Pushing the loop unrolling to: 0
        int v268;
        v268 = 0l;
        #pragma unroll
        while (while_method_3(v268)){
            int v270;
            v270 = 0l;
            #pragma unroll
            while (while_method_2(v270)){
                assert("Tensor range check" && 0 <= v268 && v268 < 4l);
                assert("Tensor range check" && 0 <= v270 && v270 < 1l);
                int v272;
                v272 = v268 + v270;
                wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v273 = v57[v272];
                assert("Tensor range check" && 0 <= v268 && v268 < 4l);
                assert("Tensor range check" && 0 <= v270 && v270 < 1l);
                int v274;
                v274 = 16l * v270;
                int v275;
                v275 = 2176l * v268;
                int v276;
                v276 = v275 + v274;
                float * v277;
                v277 = v23+v276;
                wmma::store_matrix_sync(v277, v273, 136l, wmma::mem_row_major);
                v270 += 1l ;
            }
            v268 += 1l ;
        }
        // Poping the loop unrolling to: 0
        asm("barrier.cta.sync %0;" :: "r"(0l));
        // Pushing the loop unrolling to: 0
        int v279;
        v279 = threadIdx.x;
        bool v280;
        v280 = 0l <= v279;
        bool v281;
        v281 = v280 == false;
        if (v281){
            assert("The index needs to be zero or positive." && v280);
        } else {
        }
        int v283;
        v283 = v279 % 32l;
        int v284;
        v284 = v279 / 32l;
        bool v285;
        v285 = v284 < 16l;
        bool v286;
        v286 = v285 == false;
        if (v286){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v285);
        } else {
        }
        assert("Tensor range check" && 0 <= v284 && v284 < 16l);
        assert("Tensor range check" && 0 <= v283 && v283 < 32l);
        int v288;
        v288 = 4l * v283;
        int v289;
        v289 = 512l * v284;
        int v290;
        v290 = v289 + v288;
        int v291;
        v291 = 136l * v284;
        int v292;
        v292 = v291 + v288;
        float * v293;
        v293 = v74+v290;
        float * v295;
        v295 = v8+v292;
        int v297;
        v297 = 0l;
        #pragma unroll
        while (while_method_1(v297)){
            int v299;
            v299 = 0l;
            #pragma unroll
            while (while_method_2(v299)){
                assert("Tensor range check" && 0 <= v297 && v297 < 8l);
                assert("Tensor range check" && 0 <= v299 && v299 < 1l);
                int v301;
                v301 = 128l * v299;
                int v302;
                v302 = 8192l * v297;
                int v303;
                v303 = v302 + v301;
                int v304;
                v304 = 2176l * v297;
                int v305;
                v305 = v304 + v301;
                int4* v306;
                v306 = reinterpret_cast<int4*>(v295 + v305);
                int4* v307;
                v307 = reinterpret_cast<int4*>(v293 + v303);
                assert("Pointer alignment check" && (unsigned long long)(v306) % 4l == 0 && (unsigned long long)(v307) % 4l == 0);
                *v307 = *v306;
                v299 += 1l ;
            }
            v297 += 1l ;
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
