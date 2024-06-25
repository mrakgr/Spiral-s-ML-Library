kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
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

__device__ void method_1(float * v0, int v1, float * v2, int v3, float * v4, int v5);
__device__ void method_0(unsigned char * v0, unsigned char * v1, int v2, int v3);
__device__ void method_3(float * v0, int v1, float * v2);
__device__ void method_2(unsigned char * v0, unsigned char * v1, int v2, int v3);
__device__ void method_5(float * v0, int v1, float * v2, int v3, float * v4);
__device__ void method_4(unsigned char * v0, unsigned char * v1, int v2, int v3);
__device__ void method_6(unsigned char * v0, unsigned char * v1, int v2, int v3);
__device__ void method_7(unsigned char * v0, unsigned char * v1, int v2, int v3);
__device__ void method_9(float * v0, int v1, float * v2, int v3);
__device__ void method_8(unsigned char * v0, unsigned char * v1, int v2, int v3);
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
__device__ void method_1(float * v0, int v1, float * v2, int v3, float * v4, int v5){
    extern __shared__ unsigned char v6[];
    float * v7;
    v7 = reinterpret_cast<float *>(&v6[0ull]);
    float * v8;
    v8 = reinterpret_cast<float *>(&v6[768ull]);
    float * v9;
    v9 = reinterpret_cast<float *>(&v6[0ull]);
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
    int v14;
    v14 = v11 % 1l;
    bool v15;
    v15 = v11 < 1l;
    bool v16;
    v16 = v15 == false;
    if (v16){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v15);
    } else {
    }
    assert("Tensor range check" && 0 <= v11 && v11 < 1l);
    assert("Tensor range check" && 0 <= v14 && v14 < 1l);
    int v17;
    v17 = 16l * v14;
    int v18;
    v18 = 384l * v11;
    int v19;
    v19 = v18 + v17;
    float * v20;
    v20 = v9+v19;
    assert("Tensor range check" && 0 <= v11 && v11 < 1l);
    int v21;
    v21 = 192l * v11;
    int v22;
    v22 = threadIdx.x;
    int v23;
    v23 = v22 % 32l;
    bool v24;
    v24 = 0l <= v23;
    bool v25;
    v25 = v24 == false;
    if (v25){
        assert("The index needs to be zero or positive." && v24);
    } else {
    }
    int v26;
    v26 = v23 % 4l;
    int v27;
    v27 = v23 / 4l;
    bool v28;
    v28 = v27 < 8l;
    bool v29;
    v29 = v28 == false;
    if (v29){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v28);
    } else {
    }
    assert("Tensor range check" && 0 <= v27 && v27 < 8l);
    assert("Tensor range check" && 0 <= v26 && v26 < 4l);
    int v30;
    v30 = v26 + v21;
    int v31;
    v31 = 12l * v27;
    int v32;
    v32 = v31 + v30;
    float * v33;
    v33 = v7+v32;
    assert("Tensor range check" && 0 <= v14 && v14 < 1l);
    int v34;
    v34 = 192l * v14;
    int v35;
    v35 = threadIdx.x;
    int v36;
    v36 = v35 % 32l;
    bool v37;
    v37 = 0l <= v36;
    bool v38;
    v38 = v37 == false;
    if (v38){
        assert("The index needs to be zero or positive." && v37);
    } else {
    }
    int v39;
    v39 = v36 % 4l;
    int v40;
    v40 = v36 / 4l;
    bool v41;
    v41 = v40 < 8l;
    bool v42;
    v42 = v41 == false;
    if (v42){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v41);
    } else {
    }
    assert("Tensor range check" && 0 <= v40 && v40 < 8l);
    assert("Tensor range check" && 0 <= v39 && v39 < 4l);
    int v43;
    v43 = v39 + v34;
    int v44;
    v44 = 12l * v40;
    int v45;
    v45 = v44 + v43;
    float * v46;
    v46 = v8+v45;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v47[1l];
    int v48;
    v48 = 0l;
    while (while_method_0(v48)){
        int v50;
        v50 = 0l;
        while (while_method_0(v50)){
            assert("Tensor range check" && 0 <= v48 && v48 < 1l);
            assert("Tensor range check" && 0 <= v50 && v50 < 1l);
            int v52;
            v52 = 16l * v50;
            int v53;
            v53 = v52 + v1;
            int v54;
            v54 = 256l * v48;
            int v55;
            v55 = v54 + v53;
            float * v56;
            v56 = v0+v55;
            // Pushing the loop unrolling to: 0
            int v57;
            v57 = 0l;
            #pragma unroll
            while (while_method_0(v57)){
                int v59;
                v59 = 0l;
                #pragma unroll
                while (while_method_0(v59)){
                    assert("Tensor range check" && 0 <= v57 && v57 < 1l);
                    assert("Tensor range check" && 0 <= v59 && v59 < 1l);
                    int v61;
                    v61 = v57 + v59;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v62 = v47[v61];
                    wmma::fill_fragment(v62, 0.0f);
                    v59 += 1l ;
                }
                v57 += 1l ;
            }
            int v63;
            v63 = 0l;
            #pragma unroll
            while (while_method_0(v63)){
                assert("Tensor range check" && 0 <= v48 && v48 < 1l);
                int v65;
                v65 = 128l * v48;
                int v66;
                v66 = v65 + v5;
                assert("Tensor range check" && 0 <= v63 && v63 < 1l);
                int v67;
                v67 = 8l * v63;
                int v68;
                v68 = v67 + v66;
                float * v69;
                v69 = v4+v68;
                assert("Tensor range check" && 0 <= v50 && v50 < 1l);
                int v70;
                v70 = 128l * v50;
                int v71;
                v71 = v70 + v3;
                assert("Tensor range check" && 0 <= v63 && v63 < 1l);
                int v72;
                v72 = v67 + v71;
                float * v73;
                v73 = v2+v72;
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
                int v77;
                v77 = v74 % 2l;
                int v78;
                v78 = v74 / 2l;
                bool v79;
                v79 = v78 < 16l;
                bool v80;
                v80 = v79 == false;
                if (v80){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v79);
                } else {
                }
                assert("Tensor range check" && 0 <= v78 && v78 < 16l);
                assert("Tensor range check" && 0 <= v77 && v77 < 2l);
                int v81;
                v81 = 4l * v77;
                int v82;
                v82 = 12l * v78;
                int v83;
                v83 = v82 + v81;
                int v84;
                v84 = 8l * v78;
                int v85;
                v85 = v84 + v81;
                float * v86;
                v86 = v8+v83;
                float * v87;
                v87 = v73+v85;
                int v88;
                v88 = 0l;
                #pragma unroll
                while (while_method_0(v88)){
                    int v90;
                    v90 = 0l;
                    #pragma unroll
                    while (while_method_0(v90)){
                        assert("Tensor range check" && 0 <= v88 && v88 < 1l);
                        assert("Tensor range check" && 0 <= v90 && v90 < 1l);
                        int v92;
                        v92 = 8l * v90;
                        int v93;
                        v93 = 192l * v88;
                        int v94;
                        v94 = v93 + v92;
                        int v95;
                        v95 = 128l * v88;
                        int v96;
                        v96 = v95 + v92;
                        float v97[4l];
                        int v98;
                        v98 = 0l;
                        #pragma unroll
                        while (while_method_1(v98)){
                            assert("Tensor range check" && 0 <= v98 && v98 < 4l);
                            int v100;
                            v100 = v98 + v96;
                            float v101;
                            v101 = v87[v100];
                            float v102;
                            v102 = wmma::__float_to_tf32(v101);
                            assert("Tensor range check" && 0 <= v98 && v98 < 4l);
                            v97[v98] = v102;
                            v98 += 1l ;
                        }
                        int4* v103;
                        v103 = reinterpret_cast<int4*>(v97 + 0l);
                        int4* v104;
                        v104 = reinterpret_cast<int4*>(v86 + v94);
                        assert("Pointer alignment check" && (unsigned long long)(v103) % 4l == 0 && (unsigned long long)(v104) % 4l == 0);
                        *v104 = *v103;
                        v90 += 1l ;
                    }
                    v88 += 1l ;
                }
                int v105;
                v105 = threadIdx.x;
                bool v106;
                v106 = 0l <= v105;
                bool v107;
                v107 = v106 == false;
                if (v107){
                    assert("The index needs to be zero or positive." && v106);
                } else {
                }
                int v108;
                v108 = v105 % 2l;
                int v109;
                v109 = v105 / 2l;
                bool v110;
                v110 = v109 < 16l;
                bool v111;
                v111 = v110 == false;
                if (v111){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v110);
                } else {
                }
                assert("Tensor range check" && 0 <= v109 && v109 < 16l);
                assert("Tensor range check" && 0 <= v108 && v108 < 2l);
                int v112;
                v112 = 4l * v108;
                int v113;
                v113 = 12l * v109;
                int v114;
                v114 = v113 + v112;
                int v115;
                v115 = 8l * v109;
                int v116;
                v116 = v115 + v112;
                float * v117;
                v117 = v7+v114;
                float * v118;
                v118 = v69+v116;
                int v119;
                v119 = 0l;
                #pragma unroll
                while (while_method_0(v119)){
                    int v121;
                    v121 = 0l;
                    #pragma unroll
                    while (while_method_0(v121)){
                        assert("Tensor range check" && 0 <= v119 && v119 < 1l);
                        assert("Tensor range check" && 0 <= v121 && v121 < 1l);
                        int v123;
                        v123 = 8l * v121;
                        int v124;
                        v124 = 192l * v119;
                        int v125;
                        v125 = v124 + v123;
                        int v126;
                        v126 = 128l * v119;
                        int v127;
                        v127 = v126 + v123;
                        float v128[4l];
                        int v129;
                        v129 = 0l;
                        #pragma unroll
                        while (while_method_1(v129)){
                            assert("Tensor range check" && 0 <= v129 && v129 < 4l);
                            int v131;
                            v131 = v129 + v127;
                            float v132;
                            v132 = v118[v131];
                            float v133;
                            v133 = wmma::__float_to_tf32(v132);
                            assert("Tensor range check" && 0 <= v129 && v129 < 4l);
                            v128[v129] = v133;
                            v129 += 1l ;
                        }
                        int4* v134;
                        v134 = reinterpret_cast<int4*>(v128 + 0l);
                        int4* v135;
                        v135 = reinterpret_cast<int4*>(v117 + v125);
                        assert("Pointer alignment check" && (unsigned long long)(v134) % 4l == 0 && (unsigned long long)(v135) % 4l == 0);
                        *v135 = *v134;
                        v121 += 1l ;
                    }
                    v119 += 1l ;
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v136[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v137[1l];
                int v138;
                v138 = 0l;
                #pragma unroll
                while (while_method_0(v138)){
                    int v140;
                    v140 = 0l;
                    #pragma unroll
                    while (while_method_0(v140)){
                        assert("Tensor range check" && 0 <= v138 && v138 < 1l);
                        assert("Tensor range check" && 0 <= v140 && v140 < 1l);
                        int v142;
                        v142 = v138 + v140;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v143 = v136[v142];
                        assert("Tensor range check" && 0 <= v138 && v138 < 1l);
                        int v144;
                        v144 = 192l * v138;
                        assert("Tensor range check" && 0 <= v140 && v140 < 1l);
                        int v145;
                        v145 = 8l * v140;
                        int v146;
                        v146 = v145 + v144;
                        int v147;
                        v147 = 0l;
                        #pragma unroll
                        while (while_method_2(v147)){
                            int v149;
                            v149 = 0l;
                            #pragma unroll
                            while (while_method_2(v149)){
                                assert("Tensor range check" && 0 <= v147 && v147 < 2l);
                                assert("Tensor range check" && 0 <= v149 && v149 < 2l);
                                int v151;
                                v151 = 96l * v149;
                                int v152;
                                v152 = v151 + v146;
                                int v153;
                                v153 = 4l * v147;
                                int v154;
                                v154 = v153 + v152;
                                float v155;
                                v155 = v33[v154];
                                bool v156;
                                v156 = 0l <= v149;
                                bool v158;
                                if (v156){
                                    bool v157;
                                    v157 = v149 < 2l;
                                    v158 = v157;
                                } else {
                                    v158 = false;
                                }
                                bool v159;
                                v159 = v158 == false;
                                if (v159){
                                    assert("The indices should be inside the range of the dimension." && v158);
                                } else {
                                }
                                bool v160;
                                v160 = 0l <= v147;
                                bool v162;
                                if (v160){
                                    bool v161;
                                    v161 = v147 < 2l;
                                    v162 = v161;
                                } else {
                                    v162 = false;
                                }
                                bool v163;
                                v163 = v162 == false;
                                if (v163){
                                    assert("The indices should be inside the range of the dimension." && v162);
                                } else {
                                }
                                int v164;
                                v164 = v147 * 2l;
                                int v165;
                                v165 = v149 + v164;
                                v143.x[v165] = v155;
                                v149 += 1l ;
                            }
                            v147 += 1l ;
                        }
                        v140 += 1l ;
                    }
                    v138 += 1l ;
                }
                int v166;
                v166 = 0l;
                #pragma unroll
                while (while_method_0(v166)){
                    int v168;
                    v168 = 0l;
                    #pragma unroll
                    while (while_method_0(v168)){
                        assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                        assert("Tensor range check" && 0 <= v168 && v168 < 1l);
                        int v170;
                        v170 = v166 + v168;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v171 = v137[v170];
                        assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                        int v172;
                        v172 = 192l * v166;
                        assert("Tensor range check" && 0 <= v168 && v168 < 1l);
                        int v173;
                        v173 = 8l * v168;
                        int v174;
                        v174 = v173 + v172;
                        int v175;
                        v175 = 0l;
                        #pragma unroll
                        while (while_method_2(v175)){
                            int v177;
                            v177 = 0l;
                            #pragma unroll
                            while (while_method_2(v177)){
                                assert("Tensor range check" && 0 <= v175 && v175 < 2l);
                                assert("Tensor range check" && 0 <= v177 && v177 < 2l);
                                int v179;
                                v179 = 4l * v177;
                                int v180;
                                v180 = v179 + v174;
                                int v181;
                                v181 = 96l * v175;
                                int v182;
                                v182 = v181 + v180;
                                float v183;
                                v183 = v46[v182];
                                bool v184;
                                v184 = 0l <= v177;
                                bool v186;
                                if (v184){
                                    bool v185;
                                    v185 = v177 < 2l;
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
                                bool v188;
                                v188 = 0l <= v175;
                                bool v190;
                                if (v188){
                                    bool v189;
                                    v189 = v175 < 2l;
                                    v190 = v189;
                                } else {
                                    v190 = false;
                                }
                                bool v191;
                                v191 = v190 == false;
                                if (v191){
                                    assert("The indices should be inside the range of the dimension." && v190);
                                } else {
                                }
                                int v192;
                                v192 = v175 * 2l;
                                int v193;
                                v193 = v177 + v192;
                                v171.x[v193] = v183;
                                v177 += 1l ;
                            }
                            v175 += 1l ;
                        }
                        v168 += 1l ;
                    }
                    v166 += 1l ;
                }
                __syncthreads();
                int v194;
                v194 = 0l;
                #pragma unroll
                while (while_method_0(v194)){
                    int v196;
                    v196 = 0l;
                    #pragma unroll
                    while (while_method_0(v196)){
                        int v198;
                        v198 = 0l;
                        #pragma unroll
                        while (while_method_0(v198)){
                            assert("Tensor range check" && 0 <= v194 && v194 < 1l);
                            assert("Tensor range check" && 0 <= v196 && v196 < 1l);
                            int v200;
                            v200 = v194 + v196;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v201 = v47[v200];
                            assert("Tensor range check" && 0 <= v194 && v194 < 1l);
                            assert("Tensor range check" && 0 <= v198 && v198 < 1l);
                            int v202;
                            v202 = v194 + v198;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v203 = v136[v202];
                            assert("Tensor range check" && 0 <= v196 && v196 < 1l);
                            assert("Tensor range check" && 0 <= v198 && v198 < 1l);
                            int v204;
                            v204 = v196 + v198;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v205 = v137[v204];
                            wmma::mma_sync(v201, v203, v205, v201);
                            v198 += 1l ;
                        }
                        v196 += 1l ;
                    }
                    v194 += 1l ;
                }
                v63 += 1l ;
            }
            int v206;
            v206 = 0l;
            #pragma unroll
            while (while_method_0(v206)){
                int v208;
                v208 = 0l;
                #pragma unroll
                while (while_method_0(v208)){
                    assert("Tensor range check" && 0 <= v206 && v206 < 1l);
                    assert("Tensor range check" && 0 <= v208 && v208 < 1l);
                    int v210;
                    v210 = v206 + v208;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v211 = v47[v210];
                    assert("Tensor range check" && 0 <= v206 && v206 < 1l);
                    assert("Tensor range check" && 0 <= v208 && v208 < 1l);
                    int v212;
                    v212 = 16l * v208;
                    int v213;
                    v213 = 384l * v206;
                    int v214;
                    v214 = v213 + v212;
                    float * v215;
                    v215 = v20+v214;
                    wmma::store_matrix_sync(v215, v211, 24l, wmma::mem_row_major);
                    v208 += 1l ;
                }
                v206 += 1l ;
            }
            __syncthreads();
            int v216;
            v216 = threadIdx.x;
            bool v217;
            v217 = 0l <= v216;
            bool v218;
            v218 = v217 == false;
            if (v218){
                assert("The index needs to be zero or positive." && v217);
            } else {
            }
            int v219;
            v219 = v216 % 4l;
            int v220;
            v220 = v216 / 4l;
            bool v221;
            v221 = v220 < 8l;
            bool v222;
            v222 = v221 == false;
            if (v222){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v221);
            } else {
            }
            assert("Tensor range check" && 0 <= v220 && v220 < 8l);
            assert("Tensor range check" && 0 <= v219 && v219 < 4l);
            int v223;
            v223 = 4l * v219;
            int v224;
            v224 = 16l * v220;
            int v225;
            v225 = v224 + v223;
            int v226;
            v226 = 24l * v220;
            int v227;
            v227 = v226 + v223;
            float * v228;
            v228 = v56+v225;
            float * v229;
            v229 = v9+v227;
            int v230;
            v230 = 0l;
            #pragma unroll
            while (while_method_2(v230)){
                int v232;
                v232 = 0l;
                #pragma unroll
                while (while_method_0(v232)){
                    assert("Tensor range check" && 0 <= v230 && v230 < 2l);
                    assert("Tensor range check" && 0 <= v232 && v232 < 1l);
                    int v234;
                    v234 = 16l * v232;
                    int v235;
                    v235 = 128l * v230;
                    int v236;
                    v236 = v235 + v234;
                    int v237;
                    v237 = 192l * v230;
                    int v238;
                    v238 = v237 + v234;
                    int4* v239;
                    v239 = reinterpret_cast<int4*>(v229 + v238);
                    int4* v240;
                    v240 = reinterpret_cast<int4*>(v228 + v236);
                    assert("Pointer alignment check" && (unsigned long long)(v239) % 4l == 0 && (unsigned long long)(v240) % 4l == 0);
                    *v240 = *v239;
                    v232 += 1l ;
                }
                v230 += 1l ;
            }
            __syncthreads();
            // Poping the loop unrolling to: 0
            v50 += 1l ;
        }
        v48 += 1l ;
    }
    return ;
}
__device__ void method_0(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[0ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v5;
    v5 = 128l * v3;
    float * v6;
    v6 = reinterpret_cast<float *>(&v0[0ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 4l);
    int v7;
    v7 = 128l * v2;
    float * v8;
    v8 = reinterpret_cast<float *>(&v0[512ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v9;
    v9 = 256l * v3;
    return method_1(v8, v9, v6, v7, v4, v5);
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 64l;
    return v1;
}
__device__ void method_3(float * v0, int v1, float * v2){
    int v3;
    v3 = threadIdx.x;
    int v4;
    v4 = v3;
    while (while_method_3(v4)){
        bool v6;
        v6 = 0l <= v4;
        bool v7;
        v7 = v6 == false;
        if (v7){
            assert("The index needs to be zero or positive." && v6);
        } else {
        }
        int v8;
        v8 = v4 % 4l;
        int v9;
        v9 = v4 / 4l;
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
        int v12;
        v12 = 4l * v8;
        int v13;
        v13 = v12 + v1;
        int v14;
        v14 = 16l * v9;
        int v15;
        v15 = v14 + v13;
        assert("Tensor range check" && 0 <= v9 && v9 < 16l);
        assert("Tensor range check" && 0 <= v8 && v8 < 4l);
        float v16[4l];
        float v17[4l];
        int4* v18;
        v18 = reinterpret_cast<int4*>(v2 + v15);
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
            float v23;
            v23 = tanh(v22);
            assert("Tensor range check" && 0 <= v20 && v20 < 4l);
            v17[v20] = v23;
            v20 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v24;
        v24 = reinterpret_cast<int4*>(v17 + 0l);
        int4* v25;
        v25 = reinterpret_cast<int4*>(v0 + v15);
        assert("Pointer alignment check" && (unsigned long long)(v24) % 4l == 0 && (unsigned long long)(v25) % 4l == 0);
        *v25 = *v24;
        v4 += 32l ;
    }
    __syncthreads();
    return ;
}
__device__ void method_2(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[512ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v5;
    v5 = 256l * v3;
    float * v6;
    v6 = reinterpret_cast<float *>(&v0[1536ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    return method_3(v6, v5, v4);
}
__device__ void method_5(float * v0, int v1, float * v2, int v3, float * v4){
    extern __shared__ unsigned char v5[];
    float * v6;
    v6 = reinterpret_cast<float *>(&v5[0ull]);
    float * v7;
    v7 = reinterpret_cast<float *>(&v5[768ull]);
    float * v8;
    v8 = reinterpret_cast<float *>(&v5[0ull]);
    int v9;
    v9 = threadIdx.x;
    int v10;
    v10 = v9 / 32l;
    bool v11;
    v11 = 0l <= v10;
    bool v12;
    v12 = v11 == false;
    if (v12){
        assert("The index needs to be zero or positive." && v11);
    } else {
    }
    int v13;
    v13 = v10 % 1l;
    bool v14;
    v14 = v10 < 1l;
    bool v15;
    v15 = v14 == false;
    if (v15){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v14);
    } else {
    }
    assert("Tensor range check" && 0 <= v10 && v10 < 1l);
    assert("Tensor range check" && 0 <= v13 && v13 < 1l);
    int v16;
    v16 = 16l * v13;
    int v17;
    v17 = 384l * v10;
    int v18;
    v18 = v17 + v16;
    float * v19;
    v19 = v8+v18;
    assert("Tensor range check" && 0 <= v10 && v10 < 1l);
    int v20;
    v20 = 192l * v10;
    int v21;
    v21 = threadIdx.x;
    int v22;
    v22 = v21 % 32l;
    bool v23;
    v23 = 0l <= v22;
    bool v24;
    v24 = v23 == false;
    if (v24){
        assert("The index needs to be zero or positive." && v23);
    } else {
    }
    int v25;
    v25 = v22 % 4l;
    int v26;
    v26 = v22 / 4l;
    bool v27;
    v27 = v26 < 8l;
    bool v28;
    v28 = v27 == false;
    if (v28){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v27);
    } else {
    }
    assert("Tensor range check" && 0 <= v26 && v26 < 8l);
    assert("Tensor range check" && 0 <= v25 && v25 < 4l);
    int v29;
    v29 = v25 + v20;
    int v30;
    v30 = 12l * v26;
    int v31;
    v31 = v30 + v29;
    float * v32;
    v32 = v6+v31;
    assert("Tensor range check" && 0 <= v13 && v13 < 1l);
    int v33;
    v33 = 192l * v13;
    int v34;
    v34 = threadIdx.x;
    int v35;
    v35 = v34 % 32l;
    bool v36;
    v36 = 0l <= v35;
    bool v37;
    v37 = v36 == false;
    if (v37){
        assert("The index needs to be zero or positive." && v36);
    } else {
    }
    int v38;
    v38 = v35 % 4l;
    int v39;
    v39 = v35 / 4l;
    bool v40;
    v40 = v39 < 8l;
    bool v41;
    v41 = v40 == false;
    if (v41){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v40);
    } else {
    }
    assert("Tensor range check" && 0 <= v39 && v39 < 8l);
    assert("Tensor range check" && 0 <= v38 && v38 < 4l);
    int v42;
    v42 = v38 + v33;
    int v43;
    v43 = 12l * v39;
    int v44;
    v44 = v43 + v42;
    float * v45;
    v45 = v7+v44;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v46[1l];
    int v47;
    v47 = 0l;
    while (while_method_0(v47)){
        int v49;
        v49 = 0l;
        while (while_method_0(v49)){
            assert("Tensor range check" && 0 <= v47 && v47 < 1l);
            assert("Tensor range check" && 0 <= v49 && v49 < 1l);
            int v51;
            v51 = 16l * v49;
            int v52;
            v52 = v51 + v1;
            int v53;
            v53 = 256l * v47;
            int v54;
            v54 = v53 + v52;
            float * v55;
            v55 = v0+v54;
            // Pushing the loop unrolling to: 0
            int v56;
            v56 = 0l;
            #pragma unroll
            while (while_method_0(v56)){
                int v58;
                v58 = 0l;
                #pragma unroll
                while (while_method_0(v58)){
                    assert("Tensor range check" && 0 <= v56 && v56 < 1l);
                    assert("Tensor range check" && 0 <= v58 && v58 < 1l);
                    int v60;
                    v60 = v56 + v58;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v61 = v46[v60];
                    wmma::fill_fragment(v61, 0.0f);
                    v58 += 1l ;
                }
                v56 += 1l ;
            }
            int v62;
            v62 = 0l;
            #pragma unroll
            while (while_method_2(v62)){
                assert("Tensor range check" && 0 <= v47 && v47 < 1l);
                int v64;
                v64 = v53 + v1;
                assert("Tensor range check" && 0 <= v62 && v62 < 2l);
                int v65;
                v65 = 8l * v62;
                int v66;
                v66 = v65 + v64;
                float * v67;
                v67 = v4+v66;
                assert("Tensor range check" && 0 <= v49 && v49 < 1l);
                int v68;
                v68 = 256l * v49;
                int v69;
                v69 = v68 + v3;
                assert("Tensor range check" && 0 <= v62 && v62 < 2l);
                int v70;
                v70 = v65 + v69;
                float * v71;
                v71 = v2+v70;
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
                int v75;
                v75 = v72 % 2l;
                int v76;
                v76 = v72 / 2l;
                bool v77;
                v77 = v76 < 16l;
                bool v78;
                v78 = v77 == false;
                if (v78){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v77);
                } else {
                }
                assert("Tensor range check" && 0 <= v76 && v76 < 16l);
                assert("Tensor range check" && 0 <= v75 && v75 < 2l);
                int v79;
                v79 = 4l * v75;
                int v80;
                v80 = 12l * v76;
                int v81;
                v81 = v80 + v79;
                int v82;
                v82 = 16l * v76;
                int v83;
                v83 = v82 + v79;
                float * v84;
                v84 = v7+v81;
                float * v85;
                v85 = v71+v83;
                int v86;
                v86 = 0l;
                #pragma unroll
                while (while_method_0(v86)){
                    int v88;
                    v88 = 0l;
                    #pragma unroll
                    while (while_method_0(v88)){
                        assert("Tensor range check" && 0 <= v86 && v86 < 1l);
                        assert("Tensor range check" && 0 <= v88 && v88 < 1l);
                        int v90;
                        v90 = 8l * v88;
                        int v91;
                        v91 = 192l * v86;
                        int v92;
                        v92 = v91 + v90;
                        int v93;
                        v93 = 256l * v86;
                        int v94;
                        v94 = v93 + v90;
                        float v95[4l];
                        int v96;
                        v96 = 0l;
                        #pragma unroll
                        while (while_method_1(v96)){
                            assert("Tensor range check" && 0 <= v96 && v96 < 4l);
                            int v98;
                            v98 = v96 + v94;
                            float v99;
                            v99 = v85[v98];
                            float v100;
                            v100 = wmma::__float_to_tf32(v99);
                            assert("Tensor range check" && 0 <= v96 && v96 < 4l);
                            v95[v96] = v100;
                            v96 += 1l ;
                        }
                        int4* v101;
                        v101 = reinterpret_cast<int4*>(v95 + 0l);
                        int4* v102;
                        v102 = reinterpret_cast<int4*>(v84 + v92);
                        assert("Pointer alignment check" && (unsigned long long)(v101) % 4l == 0 && (unsigned long long)(v102) % 4l == 0);
                        *v102 = *v101;
                        v88 += 1l ;
                    }
                    v86 += 1l ;
                }
                int v103;
                v103 = threadIdx.x;
                bool v104;
                v104 = 0l <= v103;
                bool v105;
                v105 = v104 == false;
                if (v105){
                    assert("The index needs to be zero or positive." && v104);
                } else {
                }
                int v106;
                v106 = v103 % 2l;
                int v107;
                v107 = v103 / 2l;
                bool v108;
                v108 = v107 < 16l;
                bool v109;
                v109 = v108 == false;
                if (v109){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v108);
                } else {
                }
                assert("Tensor range check" && 0 <= v107 && v107 < 16l);
                assert("Tensor range check" && 0 <= v106 && v106 < 2l);
                int v110;
                v110 = 4l * v106;
                int v111;
                v111 = 12l * v107;
                int v112;
                v112 = v111 + v110;
                int v113;
                v113 = 16l * v107;
                int v114;
                v114 = v113 + v110;
                float * v115;
                v115 = v6+v112;
                float * v116;
                v116 = v67+v114;
                int v117;
                v117 = 0l;
                #pragma unroll
                while (while_method_0(v117)){
                    int v119;
                    v119 = 0l;
                    #pragma unroll
                    while (while_method_0(v119)){
                        assert("Tensor range check" && 0 <= v117 && v117 < 1l);
                        assert("Tensor range check" && 0 <= v119 && v119 < 1l);
                        int v121;
                        v121 = 8l * v119;
                        int v122;
                        v122 = 192l * v117;
                        int v123;
                        v123 = v122 + v121;
                        int v124;
                        v124 = 256l * v117;
                        int v125;
                        v125 = v124 + v121;
                        float v126[4l];
                        int v127;
                        v127 = 0l;
                        #pragma unroll
                        while (while_method_1(v127)){
                            assert("Tensor range check" && 0 <= v127 && v127 < 4l);
                            int v129;
                            v129 = v127 + v125;
                            float v130;
                            v130 = v116[v129];
                            float v131;
                            v131 = wmma::__float_to_tf32(v130);
                            assert("Tensor range check" && 0 <= v127 && v127 < 4l);
                            v126[v127] = v131;
                            v127 += 1l ;
                        }
                        int4* v132;
                        v132 = reinterpret_cast<int4*>(v126 + 0l);
                        int4* v133;
                        v133 = reinterpret_cast<int4*>(v115 + v123);
                        assert("Pointer alignment check" && (unsigned long long)(v132) % 4l == 0 && (unsigned long long)(v133) % 4l == 0);
                        *v133 = *v132;
                        v119 += 1l ;
                    }
                    v117 += 1l ;
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v134[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v135[1l];
                int v136;
                v136 = 0l;
                #pragma unroll
                while (while_method_0(v136)){
                    int v138;
                    v138 = 0l;
                    #pragma unroll
                    while (while_method_0(v138)){
                        assert("Tensor range check" && 0 <= v136 && v136 < 1l);
                        assert("Tensor range check" && 0 <= v138 && v138 < 1l);
                        int v140;
                        v140 = v136 + v138;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v141 = v134[v140];
                        assert("Tensor range check" && 0 <= v136 && v136 < 1l);
                        int v142;
                        v142 = 192l * v136;
                        assert("Tensor range check" && 0 <= v138 && v138 < 1l);
                        int v143;
                        v143 = 8l * v138;
                        int v144;
                        v144 = v143 + v142;
                        int v145;
                        v145 = 0l;
                        #pragma unroll
                        while (while_method_2(v145)){
                            int v147;
                            v147 = 0l;
                            #pragma unroll
                            while (while_method_2(v147)){
                                assert("Tensor range check" && 0 <= v145 && v145 < 2l);
                                assert("Tensor range check" && 0 <= v147 && v147 < 2l);
                                int v149;
                                v149 = 96l * v147;
                                int v150;
                                v150 = v149 + v144;
                                int v151;
                                v151 = 4l * v145;
                                int v152;
                                v152 = v151 + v150;
                                float v153;
                                v153 = v32[v152];
                                bool v154;
                                v154 = 0l <= v147;
                                bool v156;
                                if (v154){
                                    bool v155;
                                    v155 = v147 < 2l;
                                    v156 = v155;
                                } else {
                                    v156 = false;
                                }
                                bool v157;
                                v157 = v156 == false;
                                if (v157){
                                    assert("The indices should be inside the range of the dimension." && v156);
                                } else {
                                }
                                bool v158;
                                v158 = 0l <= v145;
                                bool v160;
                                if (v158){
                                    bool v159;
                                    v159 = v145 < 2l;
                                    v160 = v159;
                                } else {
                                    v160 = false;
                                }
                                bool v161;
                                v161 = v160 == false;
                                if (v161){
                                    assert("The indices should be inside the range of the dimension." && v160);
                                } else {
                                }
                                int v162;
                                v162 = v145 * 2l;
                                int v163;
                                v163 = v147 + v162;
                                v141.x[v163] = v153;
                                v147 += 1l ;
                            }
                            v145 += 1l ;
                        }
                        v138 += 1l ;
                    }
                    v136 += 1l ;
                }
                int v164;
                v164 = 0l;
                #pragma unroll
                while (while_method_0(v164)){
                    int v166;
                    v166 = 0l;
                    #pragma unroll
                    while (while_method_0(v166)){
                        assert("Tensor range check" && 0 <= v164 && v164 < 1l);
                        assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                        int v168;
                        v168 = v164 + v166;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v169 = v135[v168];
                        assert("Tensor range check" && 0 <= v164 && v164 < 1l);
                        int v170;
                        v170 = 192l * v164;
                        assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                        int v171;
                        v171 = 8l * v166;
                        int v172;
                        v172 = v171 + v170;
                        int v173;
                        v173 = 0l;
                        #pragma unroll
                        while (while_method_2(v173)){
                            int v175;
                            v175 = 0l;
                            #pragma unroll
                            while (while_method_2(v175)){
                                assert("Tensor range check" && 0 <= v173 && v173 < 2l);
                                assert("Tensor range check" && 0 <= v175 && v175 < 2l);
                                int v177;
                                v177 = 4l * v175;
                                int v178;
                                v178 = v177 + v172;
                                int v179;
                                v179 = 96l * v173;
                                int v180;
                                v180 = v179 + v178;
                                float v181;
                                v181 = v45[v180];
                                bool v182;
                                v182 = 0l <= v175;
                                bool v184;
                                if (v182){
                                    bool v183;
                                    v183 = v175 < 2l;
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
                                bool v186;
                                v186 = 0l <= v173;
                                bool v188;
                                if (v186){
                                    bool v187;
                                    v187 = v173 < 2l;
                                    v188 = v187;
                                } else {
                                    v188 = false;
                                }
                                bool v189;
                                v189 = v188 == false;
                                if (v189){
                                    assert("The indices should be inside the range of the dimension." && v188);
                                } else {
                                }
                                int v190;
                                v190 = v173 * 2l;
                                int v191;
                                v191 = v175 + v190;
                                v169.x[v191] = v181;
                                v175 += 1l ;
                            }
                            v173 += 1l ;
                        }
                        v166 += 1l ;
                    }
                    v164 += 1l ;
                }
                __syncthreads();
                int v192;
                v192 = 0l;
                #pragma unroll
                while (while_method_0(v192)){
                    int v194;
                    v194 = 0l;
                    #pragma unroll
                    while (while_method_0(v194)){
                        int v196;
                        v196 = 0l;
                        #pragma unroll
                        while (while_method_0(v196)){
                            assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                            assert("Tensor range check" && 0 <= v194 && v194 < 1l);
                            int v198;
                            v198 = v192 + v194;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v199 = v46[v198];
                            assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                            assert("Tensor range check" && 0 <= v196 && v196 < 1l);
                            int v200;
                            v200 = v192 + v196;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v201 = v134[v200];
                            assert("Tensor range check" && 0 <= v194 && v194 < 1l);
                            assert("Tensor range check" && 0 <= v196 && v196 < 1l);
                            int v202;
                            v202 = v194 + v196;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v203 = v135[v202];
                            wmma::mma_sync(v199, v201, v203, v199);
                            v196 += 1l ;
                        }
                        v194 += 1l ;
                    }
                    v192 += 1l ;
                }
                v62 += 1l ;
            }
            int v204;
            v204 = 0l;
            #pragma unroll
            while (while_method_0(v204)){
                int v206;
                v206 = 0l;
                #pragma unroll
                while (while_method_0(v206)){
                    assert("Tensor range check" && 0 <= v204 && v204 < 1l);
                    assert("Tensor range check" && 0 <= v206 && v206 < 1l);
                    int v208;
                    v208 = v204 + v206;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v209 = v46[v208];
                    assert("Tensor range check" && 0 <= v204 && v204 < 1l);
                    assert("Tensor range check" && 0 <= v206 && v206 < 1l);
                    int v210;
                    v210 = 16l * v206;
                    int v211;
                    v211 = 384l * v204;
                    int v212;
                    v212 = v211 + v210;
                    float * v213;
                    v213 = v19+v212;
                    wmma::store_matrix_sync(v213, v209, 24l, wmma::mem_row_major);
                    v206 += 1l ;
                }
                v204 += 1l ;
            }
            __syncthreads();
            int v214;
            v214 = threadIdx.x;
            bool v215;
            v215 = 0l <= v214;
            bool v216;
            v216 = v215 == false;
            if (v216){
                assert("The index needs to be zero or positive." && v215);
            } else {
            }
            int v217;
            v217 = v214 % 4l;
            int v218;
            v218 = v214 / 4l;
            bool v219;
            v219 = v218 < 8l;
            bool v220;
            v220 = v219 == false;
            if (v220){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v219);
            } else {
            }
            assert("Tensor range check" && 0 <= v218 && v218 < 8l);
            assert("Tensor range check" && 0 <= v217 && v217 < 4l);
            int v221;
            v221 = 4l * v217;
            int v222;
            v222 = 16l * v218;
            int v223;
            v223 = v222 + v221;
            int v224;
            v224 = 24l * v218;
            int v225;
            v225 = v224 + v221;
            float * v226;
            v226 = v55+v223;
            float * v227;
            v227 = v8+v225;
            int v228;
            v228 = 0l;
            #pragma unroll
            while (while_method_2(v228)){
                int v230;
                v230 = 0l;
                #pragma unroll
                while (while_method_0(v230)){
                    assert("Tensor range check" && 0 <= v228 && v228 < 2l);
                    assert("Tensor range check" && 0 <= v230 && v230 < 1l);
                    int v232;
                    v232 = 16l * v230;
                    int v233;
                    v233 = 128l * v228;
                    int v234;
                    v234 = v233 + v232;
                    int v235;
                    v235 = 192l * v228;
                    int v236;
                    v236 = v235 + v232;
                    int4* v237;
                    v237 = reinterpret_cast<int4*>(v227 + v236);
                    int4* v238;
                    v238 = reinterpret_cast<int4*>(v226 + v234);
                    assert("Pointer alignment check" && (unsigned long long)(v237) % 4l == 0 && (unsigned long long)(v238) % 4l == 0);
                    *v238 = *v237;
                    v230 += 1l ;
                }
                v228 += 1l ;
            }
            __syncthreads();
            // Poping the loop unrolling to: 0
            v49 += 1l ;
        }
        v47 += 1l ;
    }
    return ;
}
__device__ void method_4(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[1536ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v5;
    v5 = 256l * v3;
    float * v6;
    v6 = reinterpret_cast<float *>(&v0[2048ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 4l);
    int v7;
    v7 = 256l * v2;
    float * v8;
    v8 = reinterpret_cast<float *>(&v0[2560ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    return method_5(v8, v5, v6, v7, v4);
}
__device__ void method_6(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[2560ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v5;
    v5 = 256l * v3;
    float * v6;
    v6 = reinterpret_cast<float *>(&v0[3584ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    return method_3(v6, v5, v4);
}
__device__ void method_7(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[3584ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v5;
    v5 = 256l * v3;
    float * v6;
    v6 = reinterpret_cast<float *>(&v0[6144ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 4l);
    int v7;
    v7 = 256l * v2;
    float * v8;
    v8 = reinterpret_cast<float *>(&v0[4608ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    return method_5(v8, v5, v6, v7, v4);
}
__device__ void method_9(float * v0, int v1, float * v2, int v3){
    int v4;
    v4 = threadIdx.x;
    int v5;
    v5 = v4;
    while (while_method_3(v5)){
        bool v7;
        v7 = 0l <= v5;
        bool v8;
        v8 = v7 == false;
        if (v8){
            assert("The index needs to be zero or positive." && v7);
        } else {
        }
        int v9;
        v9 = v5 % 4l;
        int v10;
        v10 = v5 / 4l;
        bool v11;
        v11 = v10 < 16l;
        bool v12;
        v12 = v11 == false;
        if (v12){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v11);
        } else {
        }
        assert("Tensor range check" && 0 <= v10 && v10 < 16l);
        assert("Tensor range check" && 0 <= v9 && v9 < 4l);
        int v13;
        v13 = 4l * v9;
        int v14;
        v14 = v13 + v3;
        int v15;
        v15 = 16l * v10;
        int v16;
        v16 = v15 + v14;
        assert("Tensor range check" && 0 <= v10 && v10 < 16l);
        assert("Tensor range check" && 0 <= v9 && v9 < 4l);
        int v17;
        v17 = v13 + v1;
        int v18;
        v18 = v15 + v17;
        float v19[4l];
        float v20[4l];
        int4* v21;
        v21 = reinterpret_cast<int4*>(v2 + v16);
        int4* v22;
        v22 = reinterpret_cast<int4*>(v19 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v21) % 4l == 0 && (unsigned long long)(v22) % 4l == 0);
        *v22 = *v21;
        // Pushing the loop unrolling to: 0
        int v23;
        v23 = 0l;
        #pragma unroll
        while (while_method_1(v23)){
            assert("Tensor range check" && 0 <= v23 && v23 < 4l);
            float v25;
            v25 = v19[v23];
            float v26;
            v26 = tanh(v25);
            assert("Tensor range check" && 0 <= v23 && v23 < 4l);
            v20[v23] = v26;
            v23 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v27;
        v27 = reinterpret_cast<int4*>(v20 + 0l);
        int4* v28;
        v28 = reinterpret_cast<int4*>(v0 + v18);
        assert("Pointer alignment check" && (unsigned long long)(v27) % 4l == 0 && (unsigned long long)(v28) % 4l == 0);
        *v28 = *v27;
        v5 += 32l ;
    }
    __syncthreads();
    return ;
}
__device__ void method_8(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[4608ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v5;
    v5 = 256l * v3;
    float * v6;
    v6 = reinterpret_cast<float *>(&v0[5632ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 4l);
    int v7;
    v7 = 256l * v2;
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v8;
    v8 = v5 + v7;
    return method_9(v6, v8, v4, v5);
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1, int v2) {
    unsigned long long v3;
    v3 = clock64();
    curandStatePhilox4_32_10_t v4;
    curand_init(v3,0ull,0ull,&v4);
    int v5;
    v5 = blockIdx.x;
    int v6;
    v6 = v5;
    while (while_method_0(v6)){
        bool v8;
        v8 = 0l <= v6;
        bool v9;
        v9 = v8 == false;
        if (v9){
            assert("The index needs to be zero or positive." && v8);
        } else {
        }
        bool v10;
        v10 = v6 < 1l;
        bool v11;
        v11 = v10 == false;
        if (v11){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v10);
        } else {
        }
        method_0(v0, v1, v2, v6);
        method_2(v0, v1, v2, v6);
        method_4(v0, v1, v2, v6);
        method_6(v0, v1, v2, v6);
        method_7(v0, v1, v2, v6);
        method_8(v0, v1, v2, v6);
        v6 += 1l ;
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
def method5(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method6(v0 : cp.ndarray, v1 : i32, v2 : i32, v3 : i32, v4 : i32, v5 : i32) -> None:
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
def main():
    v0 = cp.empty(10240,dtype=cp.uint8)
    v1 = cp.empty(9728,dtype=cp.uint8)
    v2 = v0[0:0+4*512].view(cp.float32)
    v3 = cp.random.normal(0.0,1.0,512,dtype=cp.float32) # type: ignore
    cp.copyto(v2[0:0+512],v3[0:0+512])
    del v2, v3
    v4 = v0[2048:2048+4*1024].view(cp.float32)
    v5 = cp.random.normal(0.0,1.0,1024,dtype=cp.float32) # type: ignore
    cp.copyto(v4[0:0+1024],v5[0:0+1024])
    del v4, v5
    v6 = v0[6144:6144+4*1024].view(cp.float32)
    v7 = cp.random.normal(0.0,1.0,1024,dtype=cp.float32) # type: ignore
    cp.copyto(v6[0:0+1024],v7[0:0+1024])
    del v6, v7
    v8 = "Here are the weight matrices."
    method0(v8)
    del v8
    print()
    v9 = v0[0:0+4*512].view(cp.float32)
    v10 = 0
    v11 = 128
    v12 = 8
    v13 = 1
    v14 = 4
    v15 = 16
    v16 = 8
    method1(v9, v10, v11, v12, v13, v14, v15, v16)
    del v9, v10, v11, v12, v13, v14, v15, v16
    print()
    v17 = v0[2048:2048+4*1024].view(cp.float32)
    v18 = 0
    v19 = 256
    v20 = 16
    v21 = 1
    v22 = 4
    v23 = 16
    v24 = 16
    method1(v17, v18, v19, v20, v21, v22, v23, v24)
    del v17, v18, v19, v20, v21, v22, v23, v24
    print()
    v25 = v0[6144:6144+4*1024].view(cp.float32)
    v26 = 0
    v27 = 256
    v28 = 16
    v29 = 1
    v30 = 4
    v31 = 16
    v32 = 16
    method1(v25, v26, v27, v28, v29, v30, v31, v32)
    del v25, v26, v27, v28, v29, v30, v31, v32
    print()
    v33 = v1[0:0+4*128].view(cp.float32)
    v34 = cp.random.normal(0.0,1.0,128,dtype=cp.float32) # type: ignore
    cp.copyto(v33[0:0+128],v34[0:0+128])
    del v34
    v35 = 0
    v36 = 128
    v37 = 8
    v38 = 1
    v39 = 1
    v40 = 16
    v41 = 8
    method1(v33, v35, v36, v37, v38, v39, v40, v41)
    del v33, v35, v36, v37, v38, v39, v40, v41
    print()
    v42 = "Here is the output tensor."
    method0(v42)
    del v42
    print()
    v43 = 0
    while method5(v43):
        v45 = 0
        v46 = raw_module.get_function(f"entry{v45}")
        del v45
        v46.max_dynamic_shared_size_bytes = 1536 
        v46((1,),(32,),(v0, v1, v43),shared_mem=1536)
        del v46
        v47 = v0[5632:5632+4*1024].view(cp.float32)
        assert 0 <= v43 < 4, 'Tensor range check'
        v48 = 256 * v43
        v49 = 16
        v50 = 1
        v51 = 16
        v52 = 16
        method6(v47, v48, v49, v50, v51, v52)
        del v47, v48, v49, v50, v51, v52
        print()
        v53 = "==="
        method0(v53)
        del v53
        print()
        v43 += 1 
    del v0, v1, v43
    return 

if __name__ == '__main__': print(main())
