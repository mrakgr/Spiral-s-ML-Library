kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <mma.h>
using namespace nvcuda;
using default_int = long;
using default_uint = unsigned long;
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

__device__ void method_1(float * v0, long v1, float * v2, float * v3, long v4);
__device__ void method_0(unsigned char * v0, unsigned char * v1, long v2);
__device__ void method_3(float * v0, long v1, float * v2);
__device__ void method_2(unsigned char * v0, unsigned char * v1, long v2);
__device__ void method_5(float * v0, long v1, float * v2, float * v3);
__device__ void method_4(unsigned char * v0, unsigned char * v1, long v2);
__device__ void method_6(unsigned char * v0, unsigned char * v1, long v2);
__device__ void method_7(unsigned char * v0, unsigned char * v1, long v2);
__device__ void method_8(unsigned char * v0, unsigned char * v1, long v2);
__device__ inline bool while_method_0(long v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_1(long v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_2(long v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
__device__ void method_1(float * v0, long v1, float * v2, float * v3, long v4){
    extern __shared__ unsigned char v5[];
    float * v6;
    v6 = reinterpret_cast<float *>(&v5[0ull]);
    float * v7;
    v7 = reinterpret_cast<float *>(&v5[768ull]);
    float * v8;
    v8 = reinterpret_cast<float *>(&v5[0ull]);
    long v9;
    v9 = threadIdx.x;
    long v10;
    v10 = v9 / 32l;
    bool v11;
    v11 = 0l <= v10;
    bool v12;
    v12 = v11 == false;
    if (v12){
        assert("The index needs to be zero or positive." && v11);
    } else {
    }
    long v13;
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
    long v16;
    v16 = 16l * v13;
    long v17;
    v17 = 384l * v10;
    long v18;
    v18 = v17 + v16;
    float * v19;
    v19 = v8+v18;
    assert("Tensor range check" && 0 <= v10 && v10 < 1l);
    long v20;
    v20 = 192l * v10;
    long v21;
    v21 = threadIdx.x;
    long v22;
    v22 = v21 % 32l;
    bool v23;
    v23 = 0l <= v22;
    bool v24;
    v24 = v23 == false;
    if (v24){
        assert("The index needs to be zero or positive." && v23);
    } else {
    }
    long v25;
    v25 = v22 % 4l;
    long v26;
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
    long v29;
    v29 = v25 + v20;
    long v30;
    v30 = 12l * v26;
    long v31;
    v31 = v30 + v29;
    float * v32;
    v32 = v6+v31;
    assert("Tensor range check" && 0 <= v13 && v13 < 1l);
    long v33;
    v33 = 192l * v13;
    long v34;
    v34 = threadIdx.x;
    long v35;
    v35 = v34 % 32l;
    bool v36;
    v36 = 0l <= v35;
    bool v37;
    v37 = v36 == false;
    if (v37){
        assert("The index needs to be zero or positive." && v36);
    } else {
    }
    long v38;
    v38 = v35 % 4l;
    long v39;
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
    long v42;
    v42 = v38 + v33;
    long v43;
    v43 = 12l * v39;
    long v44;
    v44 = v43 + v42;
    float * v45;
    v45 = v7+v44;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v46[1l];
    long v47;
    v47 = 0l;
    while (while_method_0(v47)){
        long v49;
        v49 = 0l;
        while (while_method_0(v49)){
            assert("Tensor range check" && 0 <= v47 && v47 < 1l);
            assert("Tensor range check" && 0 <= v49 && v49 < 1l);
            long v51;
            v51 = 16l * v49;
            long v52;
            v52 = v51 + v1;
            long v53;
            v53 = 256l * v47;
            long v54;
            v54 = v53 + v52;
            float * v55;
            v55 = v0+v54;
            // Pushing the loop unrolling to: 0
            long v56;
            v56 = 0l;
            #pragma unroll
            while (while_method_0(v56)){
                long v58;
                v58 = 0l;
                #pragma unroll
                while (while_method_0(v58)){
                    assert("Tensor range check" && 0 <= v56 && v56 < 1l);
                    assert("Tensor range check" && 0 <= v58 && v58 < 1l);
                    long v60;
                    v60 = v56 + v58;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v61 = v46[v60];
                    wmma::fill_fragment(v61, 0.0f);
                    v58 += 1l ;
                }
                v56 += 1l ;
            }
            long v62;
            v62 = 0l;
            #pragma unroll
            while (while_method_0(v62)){
                assert("Tensor range check" && 0 <= v47 && v47 < 1l);
                long v64;
                v64 = 128l * v47;
                long v65;
                v65 = v64 + v4;
                assert("Tensor range check" && 0 <= v62 && v62 < 1l);
                long v66;
                v66 = 8l * v62;
                long v67;
                v67 = v66 + v65;
                float * v68;
                v68 = v3+v67;
                assert("Tensor range check" && 0 <= v49 && v49 < 1l);
                long v69;
                v69 = 128l * v49;
                assert("Tensor range check" && 0 <= v62 && v62 < 1l);
                long v70;
                v70 = v66 + v69;
                float * v71;
                v71 = v2+v70;
                long v72;
                v72 = threadIdx.x;
                bool v73;
                v73 = 0l <= v72;
                bool v74;
                v74 = v73 == false;
                if (v74){
                    assert("The index needs to be zero or positive." && v73);
                } else {
                }
                long v75;
                v75 = v72 % 2l;
                long v76;
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
                long v79;
                v79 = 4l * v75;
                long v80;
                v80 = 12l * v76;
                long v81;
                v81 = v80 + v79;
                long v82;
                v82 = 8l * v76;
                long v83;
                v83 = v82 + v79;
                float * v84;
                v84 = v7+v81;
                float * v85;
                v85 = v71+v83;
                long v86;
                v86 = 0l;
                #pragma unroll
                while (while_method_0(v86)){
                    long v88;
                    v88 = 0l;
                    #pragma unroll
                    while (while_method_0(v88)){
                        assert("Tensor range check" && 0 <= v86 && v86 < 1l);
                        assert("Tensor range check" && 0 <= v88 && v88 < 1l);
                        long v90;
                        v90 = 8l * v88;
                        long v91;
                        v91 = 192l * v86;
                        long v92;
                        v92 = v91 + v90;
                        long v93;
                        v93 = 128l * v86;
                        long v94;
                        v94 = v93 + v90;
                        float v95[4l];
                        long v96;
                        v96 = 0l;
                        #pragma unroll
                        while (while_method_1(v96)){
                            assert("Tensor range check" && 0 <= v96 && v96 < 4l);
                            long v98;
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
                long v103;
                v103 = threadIdx.x;
                bool v104;
                v104 = 0l <= v103;
                bool v105;
                v105 = v104 == false;
                if (v105){
                    assert("The index needs to be zero or positive." && v104);
                } else {
                }
                long v106;
                v106 = v103 % 2l;
                long v107;
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
                long v110;
                v110 = 4l * v106;
                long v111;
                v111 = 12l * v107;
                long v112;
                v112 = v111 + v110;
                long v113;
                v113 = 8l * v107;
                long v114;
                v114 = v113 + v110;
                float * v115;
                v115 = v6+v112;
                float * v116;
                v116 = v68+v114;
                long v117;
                v117 = 0l;
                #pragma unroll
                while (while_method_0(v117)){
                    long v119;
                    v119 = 0l;
                    #pragma unroll
                    while (while_method_0(v119)){
                        assert("Tensor range check" && 0 <= v117 && v117 < 1l);
                        assert("Tensor range check" && 0 <= v119 && v119 < 1l);
                        long v121;
                        v121 = 8l * v119;
                        long v122;
                        v122 = 192l * v117;
                        long v123;
                        v123 = v122 + v121;
                        long v124;
                        v124 = 128l * v117;
                        long v125;
                        v125 = v124 + v121;
                        float v126[4l];
                        long v127;
                        v127 = 0l;
                        #pragma unroll
                        while (while_method_1(v127)){
                            assert("Tensor range check" && 0 <= v127 && v127 < 4l);
                            long v129;
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
                long v136;
                v136 = 0l;
                #pragma unroll
                while (while_method_0(v136)){
                    long v138;
                    v138 = 0l;
                    #pragma unroll
                    while (while_method_0(v138)){
                        assert("Tensor range check" && 0 <= v136 && v136 < 1l);
                        assert("Tensor range check" && 0 <= v138 && v138 < 1l);
                        long v140;
                        v140 = v136 + v138;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v141 = v134[v140];
                        assert("Tensor range check" && 0 <= v136 && v136 < 1l);
                        long v142;
                        v142 = 192l * v136;
                        assert("Tensor range check" && 0 <= v138 && v138 < 1l);
                        long v143;
                        v143 = 8l * v138;
                        long v144;
                        v144 = v143 + v142;
                        long v145;
                        v145 = 0l;
                        #pragma unroll
                        while (while_method_2(v145)){
                            long v147;
                            v147 = 0l;
                            #pragma unroll
                            while (while_method_2(v147)){
                                assert("Tensor range check" && 0 <= v145 && v145 < 2l);
                                assert("Tensor range check" && 0 <= v147 && v147 < 2l);
                                long v149;
                                v149 = 96l * v147;
                                long v150;
                                v150 = v149 + v144;
                                long v151;
                                v151 = 4l * v145;
                                long v152;
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
                                long v162;
                                v162 = v145 * 2l;
                                long v163;
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
                long v164;
                v164 = 0l;
                #pragma unroll
                while (while_method_0(v164)){
                    long v166;
                    v166 = 0l;
                    #pragma unroll
                    while (while_method_0(v166)){
                        assert("Tensor range check" && 0 <= v164 && v164 < 1l);
                        assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                        long v168;
                        v168 = v164 + v166;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v169 = v135[v168];
                        assert("Tensor range check" && 0 <= v164 && v164 < 1l);
                        long v170;
                        v170 = 192l * v164;
                        assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                        long v171;
                        v171 = 8l * v166;
                        long v172;
                        v172 = v171 + v170;
                        long v173;
                        v173 = 0l;
                        #pragma unroll
                        while (while_method_2(v173)){
                            long v175;
                            v175 = 0l;
                            #pragma unroll
                            while (while_method_2(v175)){
                                assert("Tensor range check" && 0 <= v173 && v173 < 2l);
                                assert("Tensor range check" && 0 <= v175 && v175 < 2l);
                                long v177;
                                v177 = 4l * v175;
                                long v178;
                                v178 = v177 + v172;
                                long v179;
                                v179 = 96l * v173;
                                long v180;
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
                                long v190;
                                v190 = v173 * 2l;
                                long v191;
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
                long v192;
                v192 = 0l;
                #pragma unroll
                while (while_method_0(v192)){
                    long v194;
                    v194 = 0l;
                    #pragma unroll
                    while (while_method_0(v194)){
                        long v196;
                        v196 = 0l;
                        #pragma unroll
                        while (while_method_0(v196)){
                            assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                            assert("Tensor range check" && 0 <= v194 && v194 < 1l);
                            long v198;
                            v198 = v192 + v194;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v199 = v46[v198];
                            assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                            assert("Tensor range check" && 0 <= v196 && v196 < 1l);
                            long v200;
                            v200 = v192 + v196;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v201 = v134[v200];
                            assert("Tensor range check" && 0 <= v194 && v194 < 1l);
                            assert("Tensor range check" && 0 <= v196 && v196 < 1l);
                            long v202;
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
            long v204;
            v204 = 0l;
            #pragma unroll
            while (while_method_0(v204)){
                long v206;
                v206 = 0l;
                #pragma unroll
                while (while_method_0(v206)){
                    assert("Tensor range check" && 0 <= v204 && v204 < 1l);
                    assert("Tensor range check" && 0 <= v206 && v206 < 1l);
                    long v208;
                    v208 = v204 + v206;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v209 = v46[v208];
                    assert("Tensor range check" && 0 <= v204 && v204 < 1l);
                    assert("Tensor range check" && 0 <= v206 && v206 < 1l);
                    long v210;
                    v210 = 16l * v206;
                    long v211;
                    v211 = 384l * v204;
                    long v212;
                    v212 = v211 + v210;
                    float * v213;
                    v213 = v19+v212;
                    wmma::store_matrix_sync(v213, v209, 24l, wmma::mem_row_major);
                    v206 += 1l ;
                }
                v204 += 1l ;
            }
            __syncthreads();
            long v214;
            v214 = threadIdx.x;
            bool v215;
            v215 = 0l <= v214;
            bool v216;
            v216 = v215 == false;
            if (v216){
                assert("The index needs to be zero or positive." && v215);
            } else {
            }
            long v217;
            v217 = v214 % 4l;
            long v218;
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
            long v221;
            v221 = 4l * v217;
            long v222;
            v222 = 16l * v218;
            long v223;
            v223 = v222 + v221;
            long v224;
            v224 = 24l * v218;
            long v225;
            v225 = v224 + v221;
            float * v226;
            v226 = v55+v223;
            float * v227;
            v227 = v8+v225;
            long v228;
            v228 = 0l;
            #pragma unroll
            while (while_method_2(v228)){
                long v230;
                v230 = 0l;
                #pragma unroll
                while (while_method_0(v230)){
                    assert("Tensor range check" && 0 <= v228 && v228 < 2l);
                    assert("Tensor range check" && 0 <= v230 && v230 < 1l);
                    long v232;
                    v232 = 16l * v230;
                    long v233;
                    v233 = 128l * v228;
                    long v234;
                    v234 = v233 + v232;
                    long v235;
                    v235 = 192l * v228;
                    long v236;
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
__device__ void method_0(unsigned char * v0, unsigned char * v1, long v2){
    float * v3;
    v3 = reinterpret_cast<float *>(&v1[0ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    long v4;
    v4 = 128l * v2;
    float * v5;
    v5 = reinterpret_cast<float *>(&v0[0ull]);
    float * v6;
    v6 = reinterpret_cast<float *>(&v1[512ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    long v7;
    v7 = 256l * v2;
    return method_1(v6, v7, v5, v3, v4);
}
__device__ inline bool while_method_3(long v0){
    bool v1;
    v1 = v0 < 64l;
    return v1;
}
__device__ void method_3(float * v0, long v1, float * v2){
    long v3;
    v3 = threadIdx.x;
    long v4;
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
        long v8;
        v8 = v4 % 4l;
        long v9;
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
        long v12;
        v12 = 4l * v8;
        long v13;
        v13 = v12 + v1;
        long v14;
        v14 = 16l * v9;
        long v15;
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
        long v20;
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
__device__ void method_2(unsigned char * v0, unsigned char * v1, long v2){
    float * v3;
    v3 = reinterpret_cast<float *>(&v1[512ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    long v4;
    v4 = 256l * v2;
    float * v5;
    v5 = reinterpret_cast<float *>(&v1[1536ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    return method_3(v5, v4, v3);
}
__device__ void method_5(float * v0, long v1, float * v2, float * v3){
    extern __shared__ unsigned char v4[];
    float * v5;
    v5 = reinterpret_cast<float *>(&v4[0ull]);
    float * v6;
    v6 = reinterpret_cast<float *>(&v4[768ull]);
    float * v7;
    v7 = reinterpret_cast<float *>(&v4[0ull]);
    long v8;
    v8 = threadIdx.x;
    long v9;
    v9 = v8 / 32l;
    bool v10;
    v10 = 0l <= v9;
    bool v11;
    v11 = v10 == false;
    if (v11){
        assert("The index needs to be zero or positive." && v10);
    } else {
    }
    long v12;
    v12 = v9 % 1l;
    bool v13;
    v13 = v9 < 1l;
    bool v14;
    v14 = v13 == false;
    if (v14){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v13);
    } else {
    }
    assert("Tensor range check" && 0 <= v9 && v9 < 1l);
    assert("Tensor range check" && 0 <= v12 && v12 < 1l);
    long v15;
    v15 = 16l * v12;
    long v16;
    v16 = 384l * v9;
    long v17;
    v17 = v16 + v15;
    float * v18;
    v18 = v7+v17;
    assert("Tensor range check" && 0 <= v9 && v9 < 1l);
    long v19;
    v19 = 192l * v9;
    long v20;
    v20 = threadIdx.x;
    long v21;
    v21 = v20 % 32l;
    bool v22;
    v22 = 0l <= v21;
    bool v23;
    v23 = v22 == false;
    if (v23){
        assert("The index needs to be zero or positive." && v22);
    } else {
    }
    long v24;
    v24 = v21 % 4l;
    long v25;
    v25 = v21 / 4l;
    bool v26;
    v26 = v25 < 8l;
    bool v27;
    v27 = v26 == false;
    if (v27){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v26);
    } else {
    }
    assert("Tensor range check" && 0 <= v25 && v25 < 8l);
    assert("Tensor range check" && 0 <= v24 && v24 < 4l);
    long v28;
    v28 = v24 + v19;
    long v29;
    v29 = 12l * v25;
    long v30;
    v30 = v29 + v28;
    float * v31;
    v31 = v5+v30;
    assert("Tensor range check" && 0 <= v12 && v12 < 1l);
    long v32;
    v32 = 192l * v12;
    long v33;
    v33 = threadIdx.x;
    long v34;
    v34 = v33 % 32l;
    bool v35;
    v35 = 0l <= v34;
    bool v36;
    v36 = v35 == false;
    if (v36){
        assert("The index needs to be zero or positive." && v35);
    } else {
    }
    long v37;
    v37 = v34 % 4l;
    long v38;
    v38 = v34 / 4l;
    bool v39;
    v39 = v38 < 8l;
    bool v40;
    v40 = v39 == false;
    if (v40){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v39);
    } else {
    }
    assert("Tensor range check" && 0 <= v38 && v38 < 8l);
    assert("Tensor range check" && 0 <= v37 && v37 < 4l);
    long v41;
    v41 = v37 + v32;
    long v42;
    v42 = 12l * v38;
    long v43;
    v43 = v42 + v41;
    float * v44;
    v44 = v6+v43;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v45[1l];
    long v46;
    v46 = 0l;
    while (while_method_0(v46)){
        long v48;
        v48 = 0l;
        while (while_method_0(v48)){
            assert("Tensor range check" && 0 <= v46 && v46 < 1l);
            assert("Tensor range check" && 0 <= v48 && v48 < 1l);
            long v50;
            v50 = 16l * v48;
            long v51;
            v51 = v50 + v1;
            long v52;
            v52 = 256l * v46;
            long v53;
            v53 = v52 + v51;
            float * v54;
            v54 = v0+v53;
            // Pushing the loop unrolling to: 0
            long v55;
            v55 = 0l;
            #pragma unroll
            while (while_method_0(v55)){
                long v57;
                v57 = 0l;
                #pragma unroll
                while (while_method_0(v57)){
                    assert("Tensor range check" && 0 <= v55 && v55 < 1l);
                    assert("Tensor range check" && 0 <= v57 && v57 < 1l);
                    long v59;
                    v59 = v55 + v57;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v60 = v45[v59];
                    wmma::fill_fragment(v60, 0.0f);
                    v57 += 1l ;
                }
                v55 += 1l ;
            }
            long v61;
            v61 = 0l;
            #pragma unroll
            while (while_method_2(v61)){
                assert("Tensor range check" && 0 <= v46 && v46 < 1l);
                long v63;
                v63 = v52 + v1;
                assert("Tensor range check" && 0 <= v61 && v61 < 2l);
                long v64;
                v64 = 8l * v61;
                long v65;
                v65 = v64 + v63;
                float * v66;
                v66 = v3+v65;
                assert("Tensor range check" && 0 <= v48 && v48 < 1l);
                long v67;
                v67 = 256l * v48;
                assert("Tensor range check" && 0 <= v61 && v61 < 2l);
                long v68;
                v68 = v64 + v67;
                float * v69;
                v69 = v2+v68;
                long v70;
                v70 = threadIdx.x;
                bool v71;
                v71 = 0l <= v70;
                bool v72;
                v72 = v71 == false;
                if (v72){
                    assert("The index needs to be zero or positive." && v71);
                } else {
                }
                long v73;
                v73 = v70 % 2l;
                long v74;
                v74 = v70 / 2l;
                bool v75;
                v75 = v74 < 16l;
                bool v76;
                v76 = v75 == false;
                if (v76){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v75);
                } else {
                }
                assert("Tensor range check" && 0 <= v74 && v74 < 16l);
                assert("Tensor range check" && 0 <= v73 && v73 < 2l);
                long v77;
                v77 = 4l * v73;
                long v78;
                v78 = 12l * v74;
                long v79;
                v79 = v78 + v77;
                long v80;
                v80 = 16l * v74;
                long v81;
                v81 = v80 + v77;
                float * v82;
                v82 = v6+v79;
                float * v83;
                v83 = v69+v81;
                long v84;
                v84 = 0l;
                #pragma unroll
                while (while_method_0(v84)){
                    long v86;
                    v86 = 0l;
                    #pragma unroll
                    while (while_method_0(v86)){
                        assert("Tensor range check" && 0 <= v84 && v84 < 1l);
                        assert("Tensor range check" && 0 <= v86 && v86 < 1l);
                        long v88;
                        v88 = 8l * v86;
                        long v89;
                        v89 = 192l * v84;
                        long v90;
                        v90 = v89 + v88;
                        long v91;
                        v91 = 256l * v84;
                        long v92;
                        v92 = v91 + v88;
                        float v93[4l];
                        long v94;
                        v94 = 0l;
                        #pragma unroll
                        while (while_method_1(v94)){
                            assert("Tensor range check" && 0 <= v94 && v94 < 4l);
                            long v96;
                            v96 = v94 + v92;
                            float v97;
                            v97 = v83[v96];
                            float v98;
                            v98 = wmma::__float_to_tf32(v97);
                            assert("Tensor range check" && 0 <= v94 && v94 < 4l);
                            v93[v94] = v98;
                            v94 += 1l ;
                        }
                        int4* v99;
                        v99 = reinterpret_cast<int4*>(v93 + 0l);
                        int4* v100;
                        v100 = reinterpret_cast<int4*>(v82 + v90);
                        assert("Pointer alignment check" && (unsigned long long)(v99) % 4l == 0 && (unsigned long long)(v100) % 4l == 0);
                        *v100 = *v99;
                        v86 += 1l ;
                    }
                    v84 += 1l ;
                }
                long v101;
                v101 = threadIdx.x;
                bool v102;
                v102 = 0l <= v101;
                bool v103;
                v103 = v102 == false;
                if (v103){
                    assert("The index needs to be zero or positive." && v102);
                } else {
                }
                long v104;
                v104 = v101 % 2l;
                long v105;
                v105 = v101 / 2l;
                bool v106;
                v106 = v105 < 16l;
                bool v107;
                v107 = v106 == false;
                if (v107){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v106);
                } else {
                }
                assert("Tensor range check" && 0 <= v105 && v105 < 16l);
                assert("Tensor range check" && 0 <= v104 && v104 < 2l);
                long v108;
                v108 = 4l * v104;
                long v109;
                v109 = 12l * v105;
                long v110;
                v110 = v109 + v108;
                long v111;
                v111 = 16l * v105;
                long v112;
                v112 = v111 + v108;
                float * v113;
                v113 = v5+v110;
                float * v114;
                v114 = v66+v112;
                long v115;
                v115 = 0l;
                #pragma unroll
                while (while_method_0(v115)){
                    long v117;
                    v117 = 0l;
                    #pragma unroll
                    while (while_method_0(v117)){
                        assert("Tensor range check" && 0 <= v115 && v115 < 1l);
                        assert("Tensor range check" && 0 <= v117 && v117 < 1l);
                        long v119;
                        v119 = 8l * v117;
                        long v120;
                        v120 = 192l * v115;
                        long v121;
                        v121 = v120 + v119;
                        long v122;
                        v122 = 256l * v115;
                        long v123;
                        v123 = v122 + v119;
                        float v124[4l];
                        long v125;
                        v125 = 0l;
                        #pragma unroll
                        while (while_method_1(v125)){
                            assert("Tensor range check" && 0 <= v125 && v125 < 4l);
                            long v127;
                            v127 = v125 + v123;
                            float v128;
                            v128 = v114[v127];
                            float v129;
                            v129 = wmma::__float_to_tf32(v128);
                            assert("Tensor range check" && 0 <= v125 && v125 < 4l);
                            v124[v125] = v129;
                            v125 += 1l ;
                        }
                        int4* v130;
                        v130 = reinterpret_cast<int4*>(v124 + 0l);
                        int4* v131;
                        v131 = reinterpret_cast<int4*>(v113 + v121);
                        assert("Pointer alignment check" && (unsigned long long)(v130) % 4l == 0 && (unsigned long long)(v131) % 4l == 0);
                        *v131 = *v130;
                        v117 += 1l ;
                    }
                    v115 += 1l ;
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v132[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v133[1l];
                long v134;
                v134 = 0l;
                #pragma unroll
                while (while_method_0(v134)){
                    long v136;
                    v136 = 0l;
                    #pragma unroll
                    while (while_method_0(v136)){
                        assert("Tensor range check" && 0 <= v134 && v134 < 1l);
                        assert("Tensor range check" && 0 <= v136 && v136 < 1l);
                        long v138;
                        v138 = v134 + v136;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v139 = v132[v138];
                        assert("Tensor range check" && 0 <= v134 && v134 < 1l);
                        long v140;
                        v140 = 192l * v134;
                        assert("Tensor range check" && 0 <= v136 && v136 < 1l);
                        long v141;
                        v141 = 8l * v136;
                        long v142;
                        v142 = v141 + v140;
                        long v143;
                        v143 = 0l;
                        #pragma unroll
                        while (while_method_2(v143)){
                            long v145;
                            v145 = 0l;
                            #pragma unroll
                            while (while_method_2(v145)){
                                assert("Tensor range check" && 0 <= v143 && v143 < 2l);
                                assert("Tensor range check" && 0 <= v145 && v145 < 2l);
                                long v147;
                                v147 = 96l * v145;
                                long v148;
                                v148 = v147 + v142;
                                long v149;
                                v149 = 4l * v143;
                                long v150;
                                v150 = v149 + v148;
                                float v151;
                                v151 = v31[v150];
                                bool v152;
                                v152 = 0l <= v145;
                                bool v154;
                                if (v152){
                                    bool v153;
                                    v153 = v145 < 2l;
                                    v154 = v153;
                                } else {
                                    v154 = false;
                                }
                                bool v155;
                                v155 = v154 == false;
                                if (v155){
                                    assert("The indices should be inside the range of the dimension." && v154);
                                } else {
                                }
                                bool v156;
                                v156 = 0l <= v143;
                                bool v158;
                                if (v156){
                                    bool v157;
                                    v157 = v143 < 2l;
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
                                long v160;
                                v160 = v143 * 2l;
                                long v161;
                                v161 = v145 + v160;
                                v139.x[v161] = v151;
                                v145 += 1l ;
                            }
                            v143 += 1l ;
                        }
                        v136 += 1l ;
                    }
                    v134 += 1l ;
                }
                long v162;
                v162 = 0l;
                #pragma unroll
                while (while_method_0(v162)){
                    long v164;
                    v164 = 0l;
                    #pragma unroll
                    while (while_method_0(v164)){
                        assert("Tensor range check" && 0 <= v162 && v162 < 1l);
                        assert("Tensor range check" && 0 <= v164 && v164 < 1l);
                        long v166;
                        v166 = v162 + v164;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v167 = v133[v166];
                        assert("Tensor range check" && 0 <= v162 && v162 < 1l);
                        long v168;
                        v168 = 192l * v162;
                        assert("Tensor range check" && 0 <= v164 && v164 < 1l);
                        long v169;
                        v169 = 8l * v164;
                        long v170;
                        v170 = v169 + v168;
                        long v171;
                        v171 = 0l;
                        #pragma unroll
                        while (while_method_2(v171)){
                            long v173;
                            v173 = 0l;
                            #pragma unroll
                            while (while_method_2(v173)){
                                assert("Tensor range check" && 0 <= v171 && v171 < 2l);
                                assert("Tensor range check" && 0 <= v173 && v173 < 2l);
                                long v175;
                                v175 = 4l * v173;
                                long v176;
                                v176 = v175 + v170;
                                long v177;
                                v177 = 96l * v171;
                                long v178;
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
                                bool v184;
                                v184 = 0l <= v171;
                                bool v186;
                                if (v184){
                                    bool v185;
                                    v185 = v171 < 2l;
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
                                long v188;
                                v188 = v171 * 2l;
                                long v189;
                                v189 = v173 + v188;
                                v167.x[v189] = v179;
                                v173 += 1l ;
                            }
                            v171 += 1l ;
                        }
                        v164 += 1l ;
                    }
                    v162 += 1l ;
                }
                __syncthreads();
                long v190;
                v190 = 0l;
                #pragma unroll
                while (while_method_0(v190)){
                    long v192;
                    v192 = 0l;
                    #pragma unroll
                    while (while_method_0(v192)){
                        long v194;
                        v194 = 0l;
                        #pragma unroll
                        while (while_method_0(v194)){
                            assert("Tensor range check" && 0 <= v190 && v190 < 1l);
                            assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                            long v196;
                            v196 = v190 + v192;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v197 = v45[v196];
                            assert("Tensor range check" && 0 <= v190 && v190 < 1l);
                            assert("Tensor range check" && 0 <= v194 && v194 < 1l);
                            long v198;
                            v198 = v190 + v194;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v199 = v132[v198];
                            assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                            assert("Tensor range check" && 0 <= v194 && v194 < 1l);
                            long v200;
                            v200 = v192 + v194;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v201 = v133[v200];
                            wmma::mma_sync(v197, v199, v201, v197);
                            v194 += 1l ;
                        }
                        v192 += 1l ;
                    }
                    v190 += 1l ;
                }
                v61 += 1l ;
            }
            long v202;
            v202 = 0l;
            #pragma unroll
            while (while_method_0(v202)){
                long v204;
                v204 = 0l;
                #pragma unroll
                while (while_method_0(v204)){
                    assert("Tensor range check" && 0 <= v202 && v202 < 1l);
                    assert("Tensor range check" && 0 <= v204 && v204 < 1l);
                    long v206;
                    v206 = v202 + v204;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v207 = v45[v206];
                    assert("Tensor range check" && 0 <= v202 && v202 < 1l);
                    assert("Tensor range check" && 0 <= v204 && v204 < 1l);
                    long v208;
                    v208 = 16l * v204;
                    long v209;
                    v209 = 384l * v202;
                    long v210;
                    v210 = v209 + v208;
                    float * v211;
                    v211 = v18+v210;
                    wmma::store_matrix_sync(v211, v207, 24l, wmma::mem_row_major);
                    v204 += 1l ;
                }
                v202 += 1l ;
            }
            __syncthreads();
            long v212;
            v212 = threadIdx.x;
            bool v213;
            v213 = 0l <= v212;
            bool v214;
            v214 = v213 == false;
            if (v214){
                assert("The index needs to be zero or positive." && v213);
            } else {
            }
            long v215;
            v215 = v212 % 4l;
            long v216;
            v216 = v212 / 4l;
            bool v217;
            v217 = v216 < 8l;
            bool v218;
            v218 = v217 == false;
            if (v218){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v217);
            } else {
            }
            assert("Tensor range check" && 0 <= v216 && v216 < 8l);
            assert("Tensor range check" && 0 <= v215 && v215 < 4l);
            long v219;
            v219 = 4l * v215;
            long v220;
            v220 = 16l * v216;
            long v221;
            v221 = v220 + v219;
            long v222;
            v222 = 24l * v216;
            long v223;
            v223 = v222 + v219;
            float * v224;
            v224 = v54+v221;
            float * v225;
            v225 = v7+v223;
            long v226;
            v226 = 0l;
            #pragma unroll
            while (while_method_2(v226)){
                long v228;
                v228 = 0l;
                #pragma unroll
                while (while_method_0(v228)){
                    assert("Tensor range check" && 0 <= v226 && v226 < 2l);
                    assert("Tensor range check" && 0 <= v228 && v228 < 1l);
                    long v230;
                    v230 = 16l * v228;
                    long v231;
                    v231 = 128l * v226;
                    long v232;
                    v232 = v231 + v230;
                    long v233;
                    v233 = 192l * v226;
                    long v234;
                    v234 = v233 + v230;
                    int4* v235;
                    v235 = reinterpret_cast<int4*>(v225 + v234);
                    int4* v236;
                    v236 = reinterpret_cast<int4*>(v224 + v232);
                    assert("Pointer alignment check" && (unsigned long long)(v235) % 4l == 0 && (unsigned long long)(v236) % 4l == 0);
                    *v236 = *v235;
                    v228 += 1l ;
                }
                v226 += 1l ;
            }
            __syncthreads();
            // Poping the loop unrolling to: 0
            v48 += 1l ;
        }
        v46 += 1l ;
    }
    return ;
}
__device__ void method_4(unsigned char * v0, unsigned char * v1, long v2){
    float * v3;
    v3 = reinterpret_cast<float *>(&v1[1536ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    long v4;
    v4 = 256l * v2;
    float * v5;
    v5 = reinterpret_cast<float *>(&v0[2048ull]);
    float * v6;
    v6 = reinterpret_cast<float *>(&v1[2560ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    return method_5(v6, v4, v5, v3);
}
__device__ void method_6(unsigned char * v0, unsigned char * v1, long v2){
    float * v3;
    v3 = reinterpret_cast<float *>(&v1[2560ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    long v4;
    v4 = 256l * v2;
    float * v5;
    v5 = reinterpret_cast<float *>(&v1[3584ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    return method_3(v5, v4, v3);
}
__device__ void method_7(unsigned char * v0, unsigned char * v1, long v2){
    float * v3;
    v3 = reinterpret_cast<float *>(&v1[3584ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    long v4;
    v4 = 256l * v2;
    float * v5;
    v5 = reinterpret_cast<float *>(&v0[6144ull]);
    float * v6;
    v6 = reinterpret_cast<float *>(&v1[4608ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    return method_5(v6, v4, v5, v3);
}
__device__ void method_8(unsigned char * v0, unsigned char * v1, long v2){
    float * v3;
    v3 = reinterpret_cast<float *>(&v1[4608ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    long v4;
    v4 = 256l * v2;
    float * v5;
    v5 = reinterpret_cast<float *>(&v1[5632ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    return method_3(v5, v4, v3);
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1) {
    long v2;
    v2 = blockIdx.x;
    long v3;
    v3 = v2;
    while (while_method_0(v3)){
        bool v5;
        v5 = 0l <= v3;
        bool v6;
        v6 = v5 == false;
        if (v6){
            assert("The index needs to be zero or positive." && v5);
        } else {
        }
        bool v7;
        v7 = v3 < 1l;
        bool v8;
        v8 = v7 == false;
        if (v8){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v7);
        } else {
        }
        method_0(v0, v1, v3);
        method_2(v0, v1, v3);
        method_4(v0, v1, v3);
        method_6(v0, v1, v3);
        method_7(v0, v1, v3);
        method_8(v0, v1, v3);
        v3 += 1l ;
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
def main():
    v0 = cp.empty(10240,dtype=cp.uint8)
    v1 = cp.empty(6656,dtype=cp.uint8)
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
    v44 = raw_module.get_function(f"entry{v43}")
    del v43
    v44.max_dynamic_shared_size_bytes = 1536 
    v44((1,),(32,),(v0, v1),shared_mem=1536)
    del v0, v44
    v45 = v1[5632:5632+4*256].view(cp.float32)
    del v1
    v46 = 0
    v47 = 256
    v48 = 16
    v49 = 1
    v50 = 1
    v51 = 16
    v52 = 16
    method1(v45, v46, v47, v48, v49, v50, v51, v52)
    del v45, v46, v47, v48, v49, v50, v51, v52
    print()
    return 

if __name__ == '__main__': print(main())
