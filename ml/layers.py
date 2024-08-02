kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <mma.h>
using namespace nvcuda;
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
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

__device__ void method_0(float * v0, int v1, float * v2, int v3, float * v4, int v5);
__device__ void method_1(float * v0, float * v1);
__device__ void method_2(float * v0, float * v1);
struct Tuple0;
struct Tuple1;
__device__ Tuple1 method_4(float v0, int v1, float v2, int v3);
__device__ void method_3(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5);
struct Closure0 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Closure1 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple0 {
    int v0;
    float v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Closure2 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple1 {
    float v0;
    int v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple1{v0, v1};
        } else {
            return Tuple1{v2, v3};
        }
    }
};
struct Closure4 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        return method_4(v0, v1, v2, v3);
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 16l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 8l;
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
__device__ void method_0(float * v0, int v1, float * v2, int v3, float * v4, int v5){
    unsigned int v6;
    v6 = 0ul;
    asm("mov.u32 %0, %dynamic_smem_size;" : "=r"(v6));
    unsigned long long v7;
    v7 = (unsigned long long)v6;
    bool v8;
    v8 = 1536ull <= v7;
    bool v9;
    v9 = v8 == false;
    if (v9){
        assert("The shared memory used in the matmult node is lower than the allocated amount." && v8);
    } else {
    }
    extern __shared__ unsigned char v11[];
    float * v12;
    v12 = reinterpret_cast<float *>(&v11[0ull]);
    float * v14;
    v14 = reinterpret_cast<float *>(&v11[768ull]);
    float * v16;
    v16 = reinterpret_cast<float *>(&v11[0ull]);
    int v18;
    v18 = threadIdx.x;
    int v19;
    v19 = v18 / 32l;
    bool v20;
    v20 = 0l <= v19;
    bool v21;
    v21 = v20 == false;
    if (v21){
        assert("The index needs to be zero or positive." && v20);
    } else {
    }
    int v23;
    v23 = v19 % 1l;
    bool v24;
    v24 = v19 < 1l;
    bool v25;
    v25 = v24 == false;
    if (v25){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v24);
    } else {
    }
    assert("Tensor range check" && 0 <= v19 && v19 < 1l);
    assert("Tensor range check" && 0 <= v23 && v23 < 1l);
    int v27;
    v27 = 16l * v23;
    int v28;
    v28 = 384l * v19;
    int v29;
    v29 = v28 + v27;
    float * v30;
    v30 = v16+v29;
    assert("Tensor range check" && 0 <= v19 && v19 < 1l);
    int v32;
    v32 = 192l * v19;
    int v33;
    v33 = threadIdx.x;
    int v34;
    v34 = v33 % 32l;
    bool v35;
    v35 = 0l <= v34;
    bool v36;
    v36 = v35 == false;
    if (v36){
        assert("The index needs to be zero or positive." && v35);
    } else {
    }
    int v38;
    v38 = v34 % 4l;
    int v39;
    v39 = v34 / 4l;
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
    int v43;
    v43 = v38 + v32;
    int v44;
    v44 = 12l * v39;
    int v45;
    v45 = v44 + v43;
    float * v46;
    v46 = v12+v45;
    assert("Tensor range check" && 0 <= v23 && v23 < 1l);
    int v48;
    v48 = 192l * v23;
    int v49;
    v49 = threadIdx.x;
    int v50;
    v50 = v49 % 32l;
    bool v51;
    v51 = 0l <= v50;
    bool v52;
    v52 = v51 == false;
    if (v52){
        assert("The index needs to be zero or positive." && v51);
    } else {
    }
    int v54;
    v54 = v50 % 4l;
    int v55;
    v55 = v50 / 4l;
    bool v56;
    v56 = v55 < 8l;
    bool v57;
    v57 = v56 == false;
    if (v57){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v56);
    } else {
    }
    assert("Tensor range check" && 0 <= v55 && v55 < 8l);
    assert("Tensor range check" && 0 <= v54 && v54 < 4l);
    int v59;
    v59 = v54 + v48;
    int v60;
    v60 = 12l * v55;
    int v61;
    v61 = v60 + v59;
    float * v62;
    v62 = v14+v61;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v64[1l];
    int v65;
    v65 = 0l;
    while (while_method_1(v65)){
        int v67;
        v67 = 0l;
        while (while_method_2(v67)){
            assert("Tensor range check" && 0 <= v65 && v65 < 1l);
            assert("Tensor range check" && 0 <= v67 && v67 < 8l);
            int v69;
            v69 = 16l * v67;
            int v70;
            v70 = v69 + v3;
            int v71;
            v71 = 2048l * v65;
            int v72;
            v72 = v71 + v70;
            float * v73;
            v73 = v2+v72;
            // Pushing the loop unrolling to: 0
            int v75;
            v75 = 0l;
            #pragma unroll
            while (while_method_1(v75)){
                int v77;
                v77 = 0l;
                #pragma unroll
                while (while_method_1(v77)){
                    assert("Tensor range check" && 0 <= v75 && v75 < 1l);
                    assert("Tensor range check" && 0 <= v77 && v77 < 1l);
                    int v79;
                    v79 = v75 + v77;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v80 = v64[v79];
                    wmma::fill_fragment(v80, 0.0f);
                    v77 += 1l ;
                }
                v75 += 1l ;
            }
            int v81;
            v81 = 0l;
            #pragma unroll
            while (while_method_0(v81)){
                assert("Tensor range check" && 0 <= v65 && v65 < 1l);
                int v83;
                v83 = v71 + v5;
                assert("Tensor range check" && 0 <= v81 && v81 < 16l);
                int v84;
                v84 = 8l * v81;
                int v85;
                v85 = v84 + v83;
                float * v86;
                v86 = v4+v85;
                assert("Tensor range check" && 0 <= v67 && v67 < 8l);
                int v88;
                v88 = 2048l * v67;
                int v89;
                v89 = v88 + v1;
                assert("Tensor range check" && 0 <= v81 && v81 < 16l);
                int v90;
                v90 = v84 + v89;
                float * v91;
                v91 = v0+v90;
                int v93;
                v93 = threadIdx.x;
                bool v94;
                v94 = 0l <= v93;
                bool v95;
                v95 = v94 == false;
                if (v95){
                    assert("The index needs to be zero or positive." && v94);
                } else {
                }
                int v97;
                v97 = v93 % 2l;
                int v98;
                v98 = v93 / 2l;
                bool v99;
                v99 = v98 < 16l;
                bool v100;
                v100 = v99 == false;
                if (v100){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v99);
                } else {
                }
                assert("Tensor range check" && 0 <= v98 && v98 < 16l);
                assert("Tensor range check" && 0 <= v97 && v97 < 2l);
                int v102;
                v102 = 4l * v97;
                int v103;
                v103 = 12l * v98;
                int v104;
                v104 = v103 + v102;
                int v105;
                v105 = 128l * v98;
                int v106;
                v106 = v105 + v102;
                float * v107;
                v107 = v14+v104;
                float * v109;
                v109 = v91+v106;
                int v111;
                v111 = 0l;
                #pragma unroll
                while (while_method_1(v111)){
                    int v113;
                    v113 = 0l;
                    #pragma unroll
                    while (while_method_1(v113)){
                        assert("Tensor range check" && 0 <= v111 && v111 < 1l);
                        assert("Tensor range check" && 0 <= v113 && v113 < 1l);
                        int v115;
                        v115 = 8l * v113;
                        int v116;
                        v116 = 192l * v111;
                        int v117;
                        v117 = v116 + v115;
                        int v118;
                        v118 = 2048l * v111;
                        int v119;
                        v119 = v118 + v115;
                        float v120[4l];
                        int v121;
                        v121 = 0l;
                        #pragma unroll
                        while (while_method_3(v121)){
                            assert("Tensor range check" && 0 <= v121 && v121 < 4l);
                            int v123;
                            v123 = v121 + v119;
                            float v124;
                            v124 = v109[v123];
                            float v125;
                            v125 = wmma::__float_to_tf32(v124);
                            assert("Tensor range check" && 0 <= v121 && v121 < 4l);
                            v120[v121] = v125;
                            v121 += 1l ;
                        }
                        int4* v126;
                        v126 = reinterpret_cast<int4*>(v120 + 0l);
                        int4* v127;
                        v127 = reinterpret_cast<int4*>(v107 + v117);
                        assert("Pointer alignment check" && (unsigned long long)(v126) % 4l == 0 && (unsigned long long)(v127) % 4l == 0);
                        *v127 = *v126;
                        v113 += 1l ;
                    }
                    v111 += 1l ;
                }
                int v128;
                v128 = threadIdx.x;
                bool v129;
                v129 = 0l <= v128;
                bool v130;
                v130 = v129 == false;
                if (v130){
                    assert("The index needs to be zero or positive." && v129);
                } else {
                }
                int v132;
                v132 = v128 % 2l;
                int v133;
                v133 = v128 / 2l;
                bool v134;
                v134 = v133 < 16l;
                bool v135;
                v135 = v134 == false;
                if (v135){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v134);
                } else {
                }
                assert("Tensor range check" && 0 <= v133 && v133 < 16l);
                assert("Tensor range check" && 0 <= v132 && v132 < 2l);
                int v137;
                v137 = 4l * v132;
                int v138;
                v138 = 12l * v133;
                int v139;
                v139 = v138 + v137;
                int v140;
                v140 = 128l * v133;
                int v141;
                v141 = v140 + v137;
                float * v142;
                v142 = v12+v139;
                float * v144;
                v144 = v86+v141;
                int v146;
                v146 = 0l;
                #pragma unroll
                while (while_method_1(v146)){
                    int v148;
                    v148 = 0l;
                    #pragma unroll
                    while (while_method_1(v148)){
                        assert("Tensor range check" && 0 <= v146 && v146 < 1l);
                        assert("Tensor range check" && 0 <= v148 && v148 < 1l);
                        int v150;
                        v150 = 8l * v148;
                        int v151;
                        v151 = 192l * v146;
                        int v152;
                        v152 = v151 + v150;
                        int v153;
                        v153 = 2048l * v146;
                        int v154;
                        v154 = v153 + v150;
                        float v155[4l];
                        int v156;
                        v156 = 0l;
                        #pragma unroll
                        while (while_method_3(v156)){
                            assert("Tensor range check" && 0 <= v156 && v156 < 4l);
                            int v158;
                            v158 = v156 + v154;
                            float v159;
                            v159 = v144[v158];
                            float v160;
                            v160 = wmma::__float_to_tf32(v159);
                            assert("Tensor range check" && 0 <= v156 && v156 < 4l);
                            v155[v156] = v160;
                            v156 += 1l ;
                        }
                        int4* v161;
                        v161 = reinterpret_cast<int4*>(v155 + 0l);
                        int4* v162;
                        v162 = reinterpret_cast<int4*>(v142 + v152);
                        assert("Pointer alignment check" && (unsigned long long)(v161) % 4l == 0 && (unsigned long long)(v162) % 4l == 0);
                        *v162 = *v161;
                        v148 += 1l ;
                    }
                    v146 += 1l ;
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v163[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v164[1l];
                int v165;
                v165 = 0l;
                #pragma unroll
                while (while_method_1(v165)){
                    int v167;
                    v167 = 0l;
                    #pragma unroll
                    while (while_method_1(v167)){
                        assert("Tensor range check" && 0 <= v165 && v165 < 1l);
                        assert("Tensor range check" && 0 <= v167 && v167 < 1l);
                        int v169;
                        v169 = v165 + v167;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v170 = v163[v169];
                        assert("Tensor range check" && 0 <= v165 && v165 < 1l);
                        int v171;
                        v171 = 192l * v165;
                        assert("Tensor range check" && 0 <= v167 && v167 < 1l);
                        int v172;
                        v172 = 8l * v167;
                        int v173;
                        v173 = v172 + v171;
                        int v174;
                        v174 = 0l;
                        #pragma unroll
                        while (while_method_4(v174)){
                            int v176;
                            v176 = 0l;
                            #pragma unroll
                            while (while_method_4(v176)){
                                assert("Tensor range check" && 0 <= v174 && v174 < 2l);
                                assert("Tensor range check" && 0 <= v176 && v176 < 2l);
                                int v178;
                                v178 = 96l * v176;
                                int v179;
                                v179 = v178 + v173;
                                int v180;
                                v180 = 4l * v174;
                                int v181;
                                v181 = v180 + v179;
                                float v182;
                                v182 = v46[v181];
                                bool v183;
                                v183 = 0l <= v176;
                                bool v185;
                                if (v183){
                                    bool v184;
                                    v184 = v176 < 2l;
                                    v185 = v184;
                                } else {
                                    v185 = false;
                                }
                                bool v186;
                                v186 = v185 == false;
                                if (v186){
                                    assert("The indices should be inside the range of the dimension." && v185);
                                } else {
                                }
                                bool v188;
                                v188 = 0l <= v174;
                                bool v190;
                                if (v188){
                                    bool v189;
                                    v189 = v174 < 2l;
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
                                int v193;
                                v193 = v174 * 2l;
                                int v194;
                                v194 = v176 + v193;
                                v170.x[v194] = v182;
                                v176 += 1l ;
                            }
                            v174 += 1l ;
                        }
                        v167 += 1l ;
                    }
                    v165 += 1l ;
                }
                int v195;
                v195 = 0l;
                #pragma unroll
                while (while_method_1(v195)){
                    int v197;
                    v197 = 0l;
                    #pragma unroll
                    while (while_method_1(v197)){
                        assert("Tensor range check" && 0 <= v195 && v195 < 1l);
                        assert("Tensor range check" && 0 <= v197 && v197 < 1l);
                        int v199;
                        v199 = v195 + v197;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v200 = v164[v199];
                        assert("Tensor range check" && 0 <= v195 && v195 < 1l);
                        int v201;
                        v201 = 192l * v195;
                        assert("Tensor range check" && 0 <= v197 && v197 < 1l);
                        int v202;
                        v202 = 8l * v197;
                        int v203;
                        v203 = v202 + v201;
                        int v204;
                        v204 = 0l;
                        #pragma unroll
                        while (while_method_4(v204)){
                            int v206;
                            v206 = 0l;
                            #pragma unroll
                            while (while_method_4(v206)){
                                assert("Tensor range check" && 0 <= v204 && v204 < 2l);
                                assert("Tensor range check" && 0 <= v206 && v206 < 2l);
                                int v208;
                                v208 = 4l * v206;
                                int v209;
                                v209 = v208 + v203;
                                int v210;
                                v210 = 96l * v204;
                                int v211;
                                v211 = v210 + v209;
                                float v212;
                                v212 = v62[v211];
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
                        v197 += 1l ;
                    }
                    v195 += 1l ;
                }
                __syncthreads();
                int v225;
                v225 = 0l;
                #pragma unroll
                while (while_method_1(v225)){
                    int v227;
                    v227 = 0l;
                    #pragma unroll
                    while (while_method_1(v227)){
                        int v229;
                        v229 = 0l;
                        #pragma unroll
                        while (while_method_1(v229)){
                            assert("Tensor range check" && 0 <= v225 && v225 < 1l);
                            assert("Tensor range check" && 0 <= v227 && v227 < 1l);
                            int v231;
                            v231 = v225 + v227;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v232 = v64[v231];
                            assert("Tensor range check" && 0 <= v225 && v225 < 1l);
                            assert("Tensor range check" && 0 <= v229 && v229 < 1l);
                            int v233;
                            v233 = v225 + v229;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v234 = v163[v233];
                            assert("Tensor range check" && 0 <= v227 && v227 < 1l);
                            assert("Tensor range check" && 0 <= v229 && v229 < 1l);
                            int v235;
                            v235 = v227 + v229;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v236 = v164[v235];
                            wmma::mma_sync(v232, v234, v236, v232);
                            v229 += 1l ;
                        }
                        v227 += 1l ;
                    }
                    v225 += 1l ;
                }
                v81 += 1l ;
            }
            int v237;
            v237 = 0l;
            #pragma unroll
            while (while_method_1(v237)){
                int v239;
                v239 = 0l;
                #pragma unroll
                while (while_method_1(v239)){
                    assert("Tensor range check" && 0 <= v237 && v237 < 1l);
                    assert("Tensor range check" && 0 <= v239 && v239 < 1l);
                    int v241;
                    v241 = v237 + v239;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v242 = v64[v241];
                    assert("Tensor range check" && 0 <= v237 && v237 < 1l);
                    assert("Tensor range check" && 0 <= v239 && v239 < 1l);
                    int v243;
                    v243 = 16l * v239;
                    int v244;
                    v244 = 384l * v237;
                    int v245;
                    v245 = v244 + v243;
                    float * v246;
                    v246 = v30+v245;
                    wmma::store_matrix_sync(v246, v242, 24l, wmma::mem_row_major);
                    v239 += 1l ;
                }
                v237 += 1l ;
            }
            __syncthreads();
            int v248;
            v248 = threadIdx.x;
            bool v249;
            v249 = 0l <= v248;
            bool v250;
            v250 = v249 == false;
            if (v250){
                assert("The index needs to be zero or positive." && v249);
            } else {
            }
            int v252;
            v252 = v248 % 4l;
            int v253;
            v253 = v248 / 4l;
            bool v254;
            v254 = v253 < 8l;
            bool v255;
            v255 = v254 == false;
            if (v255){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v254);
            } else {
            }
            assert("Tensor range check" && 0 <= v253 && v253 < 8l);
            assert("Tensor range check" && 0 <= v252 && v252 < 4l);
            int v257;
            v257 = 4l * v252;
            int v258;
            v258 = 128l * v253;
            int v259;
            v259 = v258 + v257;
            int v260;
            v260 = 24l * v253;
            int v261;
            v261 = v260 + v257;
            float * v262;
            v262 = v73+v259;
            float * v264;
            v264 = v16+v261;
            int v266;
            v266 = 0l;
            #pragma unroll
            while (while_method_4(v266)){
                int v268;
                v268 = 0l;
                #pragma unroll
                while (while_method_1(v268)){
                    assert("Tensor range check" && 0 <= v266 && v266 < 2l);
                    assert("Tensor range check" && 0 <= v268 && v268 < 1l);
                    int v270;
                    v270 = 16l * v268;
                    int v271;
                    v271 = 1024l * v266;
                    int v272;
                    v272 = v271 + v270;
                    int v273;
                    v273 = 192l * v266;
                    int v274;
                    v274 = v273 + v270;
                    int4* v275;
                    v275 = reinterpret_cast<int4*>(v264 + v274);
                    int4* v276;
                    v276 = reinterpret_cast<int4*>(v262 + v272);
                    assert("Pointer alignment check" && (unsigned long long)(v275) % 4l == 0 && (unsigned long long)(v276) % 4l == 0);
                    *v276 = *v275;
                    v268 += 1l ;
                }
                v266 += 1l ;
            }
            __syncthreads();
            // Poping the loop unrolling to: 0
            v67 += 1l ;
        }
        v65 += 1l ;
    }
    return ;
}
__device__ void method_1(float * v0, float * v1){
    int v2;
    v2 = blockIdx.x;
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    int v3;
    v3 = 2048l * v2;
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 1l);
    int v5;
    v5 = 2048l * v4;
    int v6;
    v6 = threadIdx.x;
    bool v7;
    v7 = 0l <= v6;
    bool v8;
    v8 = v7 == false;
    if (v8){
        assert("The index needs to be zero or positive." && v7);
    } else {
    }
    int v10;
    v10 = v6 % 32l;
    int v11;
    v11 = v6 / 32l;
    bool v12;
    v12 = v11 < 1l;
    bool v13;
    v13 = v12 == false;
    if (v13){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v12);
    } else {
    }
    assert("Tensor range check" && 0 <= v11 && v11 < 1l);
    assert("Tensor range check" && 0 <= v10 && v10 < 32l);
    int v15;
    v15 = 4l * v10;
    int v16;
    v16 = v15 + v3;
    int v17;
    v17 = 128l * v11;
    int v18;
    v18 = v17 + v16;
    assert("Tensor range check" && 0 <= v11 && v11 < 1l);
    assert("Tensor range check" && 0 <= v10 && v10 < 32l);
    int v19;
    v19 = v15 + v5;
    int v20;
    v20 = v17 + v19;
    int v21;
    v21 = 0l;
    while (while_method_0(v21)){
        assert("Tensor range check" && 0 <= v21 && v21 < 16l);
        int v23;
        v23 = 128l * v21;
        int v24;
        v24 = v23 + v18;
        float v25[4l];
        int v26[4l];
        int v27;
        v27 = 0l;
        while (while_method_1(v27)){
            assert("Tensor range check" && 0 <= v27 && v27 < 1l);
            int v29;
            v29 = 4l * v27;
            assert("Tensor range check" && 0 <= v27 && v27 < 1l);
            int v30;
            v30 = 128l * v27;
            int v31;
            v31 = v30 + v24;
            int4* v32;
            v32 = reinterpret_cast<int4*>(v1 + v31);
            int4* v33;
            v33 = reinterpret_cast<int4*>(v25 + v29);
            assert("Pointer alignment check" && (unsigned long long)(v32) % 4l == 0 && (unsigned long long)(v33) % 4l == 0);
            *v33 = *v32;
            v27 += 1l ;
        }
        int v34;
        v34 = 0l;
        while (while_method_1(v34)){
            int v36;
            v36 = 0l;
            while (while_method_3(v36)){
                bool v38;
                v38 = 0l <= v36;
                bool v40;
                if (v38){
                    bool v39;
                    v39 = v36 < 4l;
                    v40 = v39;
                } else {
                    v40 = false;
                }
                bool v41;
                v41 = v40 == false;
                if (v41){
                    assert("The indices should be inside the range of the dimension." && v40);
                } else {
                }
                bool v43;
                v43 = 0l <= v10;
                bool v45;
                if (v43){
                    bool v44;
                    v44 = v10 < 32l;
                    v45 = v44;
                } else {
                    v45 = false;
                }
                bool v46;
                v46 = v45 == false;
                if (v46){
                    assert("The indices should be inside the range of the dimension." && v45);
                } else {
                }
                int v48;
                v48 = v10 * 4l;
                int v49;
                v49 = v36 + v48;
                bool v50;
                v50 = 0l <= v34;
                bool v52;
                if (v50){
                    bool v51;
                    v51 = v34 < 1l;
                    v52 = v51;
                } else {
                    v52 = false;
                }
                bool v53;
                v53 = v52 == false;
                if (v53){
                    assert("The indices should be inside the range of the dimension." && v52);
                } else {
                }
                int v55;
                v55 = v34 * 128l;
                int v56;
                v56 = v49 + v55;
                assert("Tensor range check" && 0 <= v34 && v34 < 1l);
                assert("Tensor range check" && 0 <= v36 && v36 < 4l);
                int v57;
                v57 = 4l * v34;
                int v58;
                v58 = v57 + v36;
                v26[v58] = v56;
                v36 += 1l ;
            }
            v34 += 1l ;
        }
        bool v59;
        v59 = 0l <= v11;
        bool v60;
        v60 = v59 && v12;
        bool v61;
        v61 = v60 == false;
        if (v61){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v60);
        } else {
        }
        bool v63;
        v63 = 0l <= v21;
        bool v65;
        if (v63){
            bool v64;
            v64 = v21 < 16l;
            v65 = v64;
        } else {
            v65 = false;
        }
        bool v66;
        v66 = v65 == false;
        if (v66){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v65);
        } else {
        }
        int v68;
        v68 = v21 + v11;
        float v69[4l];
        int v70;
        v70 = 0l;
        while (while_method_1(v70)){
            int v72;
            v72 = 0l;
            while (while_method_3(v72)){
                assert("Tensor range check" && 0 <= v70 && v70 < 1l);
                assert("Tensor range check" && 0 <= v72 && v72 < 4l);
                int v74;
                v74 = 4l * v70;
                int v75;
                v75 = v74 + v72;
                float v76;
                v76 = v25[v75];
                float v77;
                v77 = v76 * v76;
                assert("Tensor range check" && 0 <= v70 && v70 < 1l);
                assert("Tensor range check" && 0 <= v72 && v72 < 4l);
                v69[v75] = v77;
                v72 += 1l ;
            }
            v70 += 1l ;
        }
        float v78;
        v78 = 0.0f;
        int v79;
        v79 = 0l;
        while (while_method_1(v79)){
            int v81;
            v81 = 0l;
            while (while_method_3(v81)){
                assert("Tensor range check" && 0 <= v79 && v79 < 1l);
                assert("Tensor range check" && 0 <= v81 && v81 < 4l);
                int v83;
                v83 = 4l * v79;
                int v84;
                v84 = v83 + v81;
                float v85;
                v85 = v69[v84];
                float v86;
                v86 = v78 + v85;
                v78 = v86;
                v81 += 1l ;
            }
            v79 += 1l ;
        }
        auto v87 = cooperative_groups::coalesced_threads();
        int v88;
        v88 = threadIdx.x;
        int v89;
        v89 = v88 / 32l;
        auto v90 = cooperative_groups::labeled_partition(v87,v89);
        Closure0 v91{};
        float v92;
        v92 = cooperative_groups::reduce(v90, v78, v91);
        float v93[4l];
        int v94;
        v94 = 0l;
        while (while_method_1(v94)){
            int v96;
            v96 = 0l;
            while (while_method_3(v96)){
                assert("Tensor range check" && 0 <= v94 && v94 < 1l);
                assert("Tensor range check" && 0 <= v96 && v96 < 4l);
                int v98;
                v98 = 4l * v94;
                int v99;
                v99 = v98 + v96;
                float v100;
                v100 = v25[v99];
                bool v101;
                v101 = v92 == 0.0f;
                bool v102;
                v102 = v101 != true;
                float v104;
                if (v102){
                    float v103;
                    v103 = v100 / v92;
                    v104 = v103;
                } else {
                    v104 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v94 && v94 < 1l);
                assert("Tensor range check" && 0 <= v96 && v96 < 4l);
                v93[v99] = v104;
                v96 += 1l ;
            }
            v94 += 1l ;
        }
        assert("Tensor range check" && 0 <= v21 && v21 < 16l);
        int v105;
        v105 = v23 + v20;
        int v106;
        v106 = 0l;
        while (while_method_1(v106)){
            assert("Tensor range check" && 0 <= v106 && v106 < 1l);
            int v108;
            v108 = 128l * v106;
            int v109;
            v109 = v108 + v105;
            assert("Tensor range check" && 0 <= v106 && v106 < 1l);
            int v110;
            v110 = 4l * v106;
            int4* v111;
            v111 = reinterpret_cast<int4*>(v93 + v110);
            int4* v112;
            v112 = reinterpret_cast<int4*>(v0 + v109);
            assert("Pointer alignment check" && (unsigned long long)(v111) % 4l == 0 && (unsigned long long)(v112) % 4l == 0);
            *v112 = *v111;
            v106 += 1l ;
        }
        v21 += 1l ;
    }
    __syncthreads();
    return ;
}
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 512l;
    return v1;
}
__device__ void method_2(float * v0, float * v1){
    int v2;
    v2 = blockIdx.x;
    assert("Tensor range check" && 0 <= v2 && v2 < 1l);
    int v3;
    v3 = 2048l * v2;
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 1l);
    int v5;
    v5 = 2048l * v4;
    int v6;
    v6 = threadIdx.x;
    int v7;
    v7 = v6;
    while (while_method_5(v7)){
        bool v9;
        v9 = 0l <= v7;
        bool v10;
        v10 = v9 == false;
        if (v10){
            assert("The index needs to be zero or positive." && v9);
        } else {
        }
        int v12;
        v12 = v7 % 32l;
        int v13;
        v13 = v7 / 32l;
        bool v14;
        v14 = v13 < 16l;
        bool v15;
        v15 = v14 == false;
        if (v15){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v14);
        } else {
        }
        assert("Tensor range check" && 0 <= v13 && v13 < 16l);
        assert("Tensor range check" && 0 <= v12 && v12 < 32l);
        int v17;
        v17 = 4l * v12;
        int v18;
        v18 = v17 + v3;
        int v19;
        v19 = 128l * v13;
        int v20;
        v20 = v19 + v18;
        assert("Tensor range check" && 0 <= v13 && v13 < 16l);
        assert("Tensor range check" && 0 <= v12 && v12 < 32l);
        int v21;
        v21 = v17 + v5;
        int v22;
        v22 = v19 + v21;
        float v23[4l];
        float v24[4l];
        int4* v25;
        v25 = reinterpret_cast<int4*>(v1 + v20);
        int4* v26;
        v26 = reinterpret_cast<int4*>(v23 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v25) % 4l == 0 && (unsigned long long)(v26) % 4l == 0);
        *v26 = *v25;
        // Pushing the loop unrolling to: 0
        int v27;
        v27 = 0l;
        #pragma unroll
        while (while_method_3(v27)){
            assert("Tensor range check" && 0 <= v27 && v27 < 4l);
            float v29;
            v29 = v23[v27];
            bool v30;
            v30 = 0.0f >= v29;
            float v31;
            if (v30){
                v31 = 0.0f;
            } else {
                v31 = v29;
            }
            assert("Tensor range check" && 0 <= v27 && v27 < 4l);
            v24[v27] = v31;
            v27 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v32;
        v32 = reinterpret_cast<int4*>(v24 + 0l);
        int4* v33;
        v33 = reinterpret_cast<int4*>(v0 + v22);
        assert("Pointer alignment check" && (unsigned long long)(v32) % 4l == 0 && (unsigned long long)(v33) % 4l == 0);
        *v33 = *v32;
        v7 += 32l ;
    }
    __syncthreads();
    return ;
}
__device__ Tuple1 method_4(float v0, int v1, float v2, int v3){
    bool v4;
    v4 = v1 < v3;
    float v5; int v6; float v7; int v8;
    if (v4){
        v5 = v0; v6 = v1; v7 = v2; v8 = v3;
    } else {
        v5 = v2; v6 = v3; v7 = v0; v8 = v1;
    }
    bool v9;
    v9 = v5 >= 0.0f;
    bool v11;
    if (v9){
        bool v10;
        v10 = v7 >= 0.0f;
        v11 = v10;
    } else {
        v11 = false;
    }
    if (v11){
        bool v12;
        v12 = v5 <= v7;
        if (v12){
            return Tuple1{v5, v6};
        } else {
            return Tuple1{v7, v8};
        }
    } else {
        if (v9){
            return Tuple1{v5, v6};
        } else {
            bool v15;
            v15 = v7 >= 0.0f;
            if (v15){
                return Tuple1{v7, v8};
            } else {
                return Tuple1{v5, v6};
            }
        }
    }
}
__device__ void method_3(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5){
    int v6;
    v6 = blockIdx.x;
    assert("Tensor range check" && 0 <= v6 && v6 < 1l);
    int v7;
    v7 = 2048l * v6;
    int v8;
    v8 = blockIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 1l);
    int v9;
    v9 = 2048l * v8;
    int v10;
    v10 = v9 + v3;
    int v11;
    v11 = blockIdx.x;
    assert("Tensor range check" && 0 <= v11 && v11 < 1l);
    int v12;
    v12 = 16l * v11;
    int v13;
    v13 = v12 + v1;
    int v14;
    v14 = threadIdx.x;
    bool v15;
    v15 = 0l <= v14;
    bool v16;
    v16 = v15 == false;
    if (v16){
        assert("The index needs to be zero or positive." && v15);
    } else {
    }
    int v18;
    v18 = v14 % 32l;
    int v19;
    v19 = v14 / 32l;
    bool v20;
    v20 = v19 < 1l;
    bool v21;
    v21 = v20 == false;
    if (v21){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v20);
    } else {
    }
    assert("Tensor range check" && 0 <= v19 && v19 < 1l);
    assert("Tensor range check" && 0 <= v18 && v18 < 32l);
    int v23;
    v23 = 4l * v18;
    int v24;
    v24 = v23 + v7;
    int v25;
    v25 = 128l * v19;
    int v26;
    v26 = v25 + v24;
    assert("Tensor range check" && 0 <= v19 && v19 < 1l);
    assert("Tensor range check" && 0 <= v18 && v18 < 32l);
    int v27;
    v27 = v23 + v10;
    int v28;
    v28 = v25 + v27;
    assert("Tensor range check" && 0 <= v19 && v19 < 1l);
    int v29;
    v29 = v19 + v13;
    int v30;
    v30 = 0l;
    while (while_method_0(v30)){
        assert("Tensor range check" && 0 <= v30 && v30 < 16l);
        int v32;
        v32 = 128l * v30;
        int v33;
        v33 = v32 + v26;
        float v34[4l];
        int v35[4l];
        int v36;
        v36 = 0l;
        while (while_method_1(v36)){
            assert("Tensor range check" && 0 <= v36 && v36 < 1l);
            int v38;
            v38 = 4l * v36;
            assert("Tensor range check" && 0 <= v36 && v36 < 1l);
            int v39;
            v39 = 128l * v36;
            int v40;
            v40 = v39 + v33;
            int4* v41;
            v41 = reinterpret_cast<int4*>(v4 + v40);
            int4* v42;
            v42 = reinterpret_cast<int4*>(v34 + v38);
            assert("Pointer alignment check" && (unsigned long long)(v41) % 4l == 0 && (unsigned long long)(v42) % 4l == 0);
            *v42 = *v41;
            v36 += 1l ;
        }
        int v43;
        v43 = 0l;
        while (while_method_1(v43)){
            int v45;
            v45 = 0l;
            while (while_method_3(v45)){
                bool v47;
                v47 = 0l <= v45;
                bool v49;
                if (v47){
                    bool v48;
                    v48 = v45 < 4l;
                    v49 = v48;
                } else {
                    v49 = false;
                }
                bool v50;
                v50 = v49 == false;
                if (v50){
                    assert("The indices should be inside the range of the dimension." && v49);
                } else {
                }
                bool v52;
                v52 = 0l <= v18;
                bool v54;
                if (v52){
                    bool v53;
                    v53 = v18 < 32l;
                    v54 = v53;
                } else {
                    v54 = false;
                }
                bool v55;
                v55 = v54 == false;
                if (v55){
                    assert("The indices should be inside the range of the dimension." && v54);
                } else {
                }
                int v57;
                v57 = v18 * 4l;
                int v58;
                v58 = v45 + v57;
                bool v59;
                v59 = 0l <= v43;
                bool v61;
                if (v59){
                    bool v60;
                    v60 = v43 < 1l;
                    v61 = v60;
                } else {
                    v61 = false;
                }
                bool v62;
                v62 = v61 == false;
                if (v62){
                    assert("The indices should be inside the range of the dimension." && v61);
                } else {
                }
                int v64;
                v64 = v43 * 128l;
                int v65;
                v65 = v58 + v64;
                assert("Tensor range check" && 0 <= v43 && v43 < 1l);
                assert("Tensor range check" && 0 <= v45 && v45 < 4l);
                int v66;
                v66 = 4l * v43;
                int v67;
                v67 = v66 + v45;
                v35[v67] = v65;
                v45 += 1l ;
            }
            v43 += 1l ;
        }
        bool v68;
        v68 = 0l <= v19;
        bool v69;
        v69 = v68 && v20;
        bool v70;
        v70 = v69 == false;
        if (v70){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v69);
        } else {
        }
        bool v72;
        v72 = 0l <= v30;
        bool v74;
        if (v72){
            bool v73;
            v73 = v30 < 16l;
            v74 = v73;
        } else {
            v74 = false;
        }
        bool v75;
        v75 = v74 == false;
        if (v75){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v74);
        } else {
        }
        int v77;
        v77 = v30 + v19;
        bool v78[4l];
        int v79;
        v79 = 0l;
        while (while_method_1(v79)){
            int v81;
            v81 = 0l;
            while (while_method_3(v81)){
                assert("Tensor range check" && 0 <= v79 && v79 < 1l);
                assert("Tensor range check" && 0 <= v81 && v81 < 4l);
                int v83;
                v83 = 4l * v79;
                int v84;
                v84 = v83 + v81;
                float v85;
                v85 = v34[v84];
                int v86;
                v86 = v35[v84];
                bool v87;
                v87 = v86 < 3l;
                assert("Tensor range check" && 0 <= v79 && v79 < 1l);
                assert("Tensor range check" && 0 <= v81 && v81 < 4l);
                v78[v84] = v87;
                v81 += 1l ;
            }
            v79 += 1l ;
        }
        int v88[4l];
        int v89;
        v89 = 0l;
        while (while_method_1(v89)){
            int v91;
            v91 = 0l;
            while (while_method_3(v91)){
                assert("Tensor range check" && 0 <= v89 && v89 < 1l);
                assert("Tensor range check" && 0 <= v91 && v91 < 4l);
                int v93;
                v93 = 4l * v89;
                int v94;
                v94 = v93 + v91;
                bool v95;
                v95 = v78[v94];
                int v96;
                if (v95){
                    v96 = 1l;
                } else {
                    v96 = 0l;
                }
                assert("Tensor range check" && 0 <= v89 && v89 < 1l);
                assert("Tensor range check" && 0 <= v91 && v91 < 4l);
                v88[v94] = v96;
                v91 += 1l ;
            }
            v89 += 1l ;
        }
        int v97;
        v97 = 0l;
        int v98;
        v98 = 0l;
        while (while_method_1(v98)){
            int v100;
            v100 = 0l;
            while (while_method_3(v100)){
                assert("Tensor range check" && 0 <= v98 && v98 < 1l);
                assert("Tensor range check" && 0 <= v100 && v100 < 4l);
                int v102;
                v102 = 4l * v98;
                int v103;
                v103 = v102 + v100;
                int v104;
                v104 = v88[v103];
                int v105;
                v105 = v97 + v104;
                v97 = v105;
                v100 += 1l ;
            }
            v98 += 1l ;
        }
        auto v106 = cooperative_groups::coalesced_threads();
        int v107;
        v107 = threadIdx.x;
        int v108;
        v108 = v107 / 32l;
        auto v109 = cooperative_groups::labeled_partition(v106,v108);
        Closure1 v110{};
        int v111;
        v111 = cooperative_groups::reduce(v109, v97, v110);
        float v112[4l];
        int v113;
        v113 = 0l;
        while (while_method_1(v113)){
            int v115;
            v115 = 0l;
            while (while_method_3(v115)){
                assert("Tensor range check" && 0 <= v113 && v113 < 1l);
                assert("Tensor range check" && 0 <= v115 && v115 < 4l);
                int v117;
                v117 = 4l * v113;
                int v118;
                v118 = v117 + v115;
                float v119;
                v119 = v34[v118];
                bool v120;
                v120 = v78[v118];
                float v121;
                if (v120){
                    v121 = v119;
                } else {
                    v121 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v113 && v113 < 1l);
                assert("Tensor range check" && 0 <= v115 && v115 < 4l);
                v112[v118] = v121;
                v115 += 1l ;
            }
            v113 += 1l ;
        }
        float v122;
        v122 = 0.0f;
        int v123;
        v123 = 0l;
        while (while_method_1(v123)){
            int v125;
            v125 = 0l;
            while (while_method_3(v125)){
                assert("Tensor range check" && 0 <= v123 && v123 < 1l);
                assert("Tensor range check" && 0 <= v125 && v125 < 4l);
                int v127;
                v127 = 4l * v123;
                int v128;
                v128 = v127 + v125;
                float v129;
                v129 = v112[v128];
                float v130;
                v130 = v122 + v129;
                v122 = v130;
                v125 += 1l ;
            }
            v123 += 1l ;
        }
        auto v131 = cooperative_groups::coalesced_threads();
        int v132;
        v132 = threadIdx.x;
        int v133;
        v133 = v132 / 32l;
        auto v134 = cooperative_groups::labeled_partition(v131,v133);
        Closure0 v135{};
        float v136;
        v136 = cooperative_groups::reduce(v134, v122, v135);
        float v137;
        v137 = (float)v111;
        float v138;
        v138 = v136 / v137;
        float v139[4l];
        int v140;
        v140 = 0l;
        while (while_method_1(v140)){
            int v142;
            v142 = 0l;
            while (while_method_3(v142)){
                assert("Tensor range check" && 0 <= v140 && v140 < 1l);
                assert("Tensor range check" && 0 <= v142 && v142 < 4l);
                int v144;
                v144 = 4l * v140;
                int v145;
                v145 = v144 + v142;
                float v146;
                v146 = v34[v145];
                bool v147;
                v147 = v78[v145];
                float v148;
                if (v147){
                    v148 = v146;
                } else {
                    v148 = -1.0f / 0.0f;
                }
                float v149;
                v149 = v148 - v138;
                float v150;
                v150 = exp(v149);
                assert("Tensor range check" && 0 <= v140 && v140 < 1l);
                assert("Tensor range check" && 0 <= v142 && v142 < 4l);
                v139[v145] = v150;
                v142 += 1l ;
            }
            v140 += 1l ;
        }
        float v151;
        v151 = 0.0f;
        int v152;
        v152 = 0l;
        while (while_method_1(v152)){
            int v154;
            v154 = 0l;
            while (while_method_3(v154)){
                assert("Tensor range check" && 0 <= v152 && v152 < 1l);
                assert("Tensor range check" && 0 <= v154 && v154 < 4l);
                int v156;
                v156 = 4l * v152;
                int v157;
                v157 = v156 + v154;
                float v158;
                v158 = v139[v157];
                float v159;
                v159 = v151 + v158;
                v151 = v159;
                v154 += 1l ;
            }
            v152 += 1l ;
        }
        auto v160 = cooperative_groups::coalesced_threads();
        int v161;
        v161 = threadIdx.x;
        int v162;
        v162 = v161 / 32l;
        auto v163 = cooperative_groups::labeled_partition(v160,v162);
        float v164;
        v164 = cooperative_groups::reduce(v163, v151, v135);
        float v165[4l];
        int v166;
        v166 = 0l;
        while (while_method_1(v166)){
            int v168;
            v168 = 0l;
            while (while_method_3(v168)){
                assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                assert("Tensor range check" && 0 <= v168 && v168 < 4l);
                int v170;
                v170 = 4l * v166;
                int v171;
                v171 = v170 + v168;
                float v172;
                v172 = v139[v171];
                bool v173;
                v173 = v164 == 0.0f;
                bool v174;
                v174 = v173 != true;
                float v176;
                if (v174){
                    float v175;
                    v175 = v172 / v164;
                    v176 = v175;
                } else {
                    v176 = 0.0078125f;
                }
                assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                assert("Tensor range check" && 0 <= v168 && v168 < 4l);
                v165[v171] = v176;
                v168 += 1l ;
            }
            v166 += 1l ;
        }
        float v177[4l];
        float v178;
        v178 = 0.0f;
        int v179;
        v179 = 0l;
        while (while_method_1(v179)){
            assert("Tensor range check" && 0 <= v179 && v179 < 1l);
            int v181;
            v181 = 4l * v179;
            assert("Tensor range check" && 0 <= v179 && v179 < 1l);
            int v182; float v183;
            Tuple0 tmp0 = Tuple0{0l, 0.0f};
            v182 = tmp0.v0; v183 = tmp0.v1;
            while (while_method_3(v182)){
                assert("Tensor range check" && 0 <= v182 && v182 < 4l);
                int v185;
                v185 = v182 + v181;
                float v186;
                v186 = v165[v185];
                float v187;
                v187 = v183 + v186;
                v183 = v187;
                v182 += 1l ;
            }
            auto v188 = cooperative_groups::coalesced_threads();
            int v189;
            v189 = threadIdx.x;
            int v190;
            v190 = v189 / 32l;
            auto v191 = cooperative_groups::labeled_partition(v188,v190);
            Closure2 v192{};
            float v193;
            v193 = cooperative_groups::inclusive_scan(v191, v183, v192);
            float v194;
            v194 = v191.shfl_up(v193,1);
            bool v195;
            v195 = v191.thread_rank() == 0;
            float v196;
            if (v195){
                v196 = 0.0f;
            } else {
                v196 = v194;
            }
            float v197;
            v197 = v191.shfl(v193,v191.num_threads()-1);
            float v198;
            v198 = v178 + v196;
            int v199; float v200;
            Tuple0 tmp1 = Tuple0{0l, v198};
            v199 = tmp1.v0; v200 = tmp1.v1;
            while (while_method_3(v199)){
                assert("Tensor range check" && 0 <= v199 && v199 < 4l);
                int v202;
                v202 = v199 + v181;
                float v203;
                v203 = v165[v202];
                float v204;
                v204 = v200 + v203;
                assert("Tensor range check" && 0 <= v199 && v199 < 4l);
                v177[v202] = v204;
                v200 = v204;
                v199 += 1l ;
            }
            float v205;
            v205 = v178 + v197;
            v178 = v205;
            v179 += 1l ;
        }
        float v206[4l];
        int v207[4l];
        int v208;
        v208 = 0l;
        while (while_method_1(v208)){
            int v210;
            v210 = 0l;
            while (while_method_3(v210)){
                assert("Tensor range check" && 0 <= v208 && v208 < 1l);
                assert("Tensor range check" && 0 <= v210 && v210 < 4l);
                int v212;
                v212 = 4l * v208;
                int v213;
                v213 = v212 + v210;
                int v214;
                v214 = v35[v213];
                float v215;
                v215 = curand_uniform(&v5);
                assert("Tensor range check" && 0 <= v208 && v208 < 1l);
                assert("Tensor range check" && 0 <= v210 && v210 < 4l);
                v206[v213] = v215;
                v207[v213] = v214;
                v210 += 1l ;
            }
            v208 += 1l ;
        }
        float v216; int v217;
        Tuple1 tmp2 = Tuple1{0.0f, 2147483647l};
        v216 = tmp2.v0; v217 = tmp2.v1;
        int v218;
        v218 = 0l;
        while (while_method_1(v218)){
            int v220;
            v220 = 0l;
            while (while_method_3(v220)){
                assert("Tensor range check" && 0 <= v218 && v218 < 1l);
                assert("Tensor range check" && 0 <= v220 && v220 < 4l);
                int v222;
                v222 = 4l * v218;
                int v223;
                v223 = v222 + v220;
                float v224;
                v224 = v206[v223];
                int v225;
                v225 = v207[v223];
                bool v226;
                v226 = v217 < v225;
                float v227; int v228;
                if (v226){
                    v227 = v216; v228 = v217;
                } else {
                    v227 = v224; v228 = v225;
                }
                v216 = v227;
                v217 = v228;
                v220 += 1l ;
            }
            v218 += 1l ;
        }
        auto v229 = cooperative_groups::coalesced_threads();
        int v230;
        v230 = threadIdx.x;
        int v231;
        v231 = v230 / 32l;
        auto v232 = cooperative_groups::labeled_partition(v229,v231);
        Closure3 v233{};
        float v234; int v235;
        Tuple1 tmp3 = cooperative_groups::reduce(v232, Tuple1{v216, v217}, v233);
        v234 = tmp3.v0; v235 = tmp3.v1;
        float v236[4l];
        int v237;
        v237 = 0l;
        while (while_method_1(v237)){
            int v239;
            v239 = 0l;
            while (while_method_3(v239)){
                assert("Tensor range check" && 0 <= v237 && v237 < 1l);
                assert("Tensor range check" && 0 <= v239 && v239 < 4l);
                int v241;
                v241 = 4l * v237;
                int v242;
                v242 = v241 + v239;
                float v243;
                v243 = v177[v242];
                float v244;
                v244 = v243 - v234;
                assert("Tensor range check" && 0 <= v237 && v237 < 1l);
                assert("Tensor range check" && 0 <= v239 && v239 < 4l);
                v236[v242] = v244;
                v239 += 1l ;
            }
            v237 += 1l ;
        }
        float v245; int v246;
        Tuple1 tmp4 = Tuple1{-1.0f / 0.0f, 2147483647l};
        v245 = tmp4.v0; v246 = tmp4.v1;
        int v247;
        v247 = 0l;
        while (while_method_1(v247)){
            int v249;
            v249 = 0l;
            while (while_method_3(v249)){
                assert("Tensor range check" && 0 <= v247 && v247 < 1l);
                assert("Tensor range check" && 0 <= v249 && v249 < 4l);
                int v251;
                v251 = 4l * v247;
                int v252;
                v252 = v251 + v249;
                float v253;
                v253 = v236[v252];
                int v254;
                v254 = v35[v252];
                float v255; int v256;
                Tuple1 tmp5 = method_4(v245, v246, v253, v254);
                v255 = tmp5.v0; v256 = tmp5.v1;
                v245 = v255;
                v246 = v256;
                v249 += 1l ;
            }
            v247 += 1l ;
        }
        auto v257 = cooperative_groups::coalesced_threads();
        int v258;
        v258 = threadIdx.x;
        int v259;
        v259 = v258 / 32l;
        auto v260 = cooperative_groups::labeled_partition(v257,v259);
        Closure4 v261{};
        float v262; int v263;
        Tuple1 tmp6 = cooperative_groups::reduce(v260, Tuple1{v245, v246}, v261);
        v262 = tmp6.v0; v263 = tmp6.v1;
        assert("Tensor range check" && 0 <= v30 && v30 < 16l);
        int v264;
        v264 = v32 + v28;
        int v265;
        v265 = 0l;
        while (while_method_1(v265)){
            assert("Tensor range check" && 0 <= v265 && v265 < 1l);
            int v267;
            v267 = 128l * v265;
            int v268;
            v268 = v267 + v264;
            assert("Tensor range check" && 0 <= v265 && v265 < 1l);
            int v269;
            v269 = 4l * v265;
            int4* v270;
            v270 = reinterpret_cast<int4*>(v165 + v269);
            int4* v271;
            v271 = reinterpret_cast<int4*>(v2 + v268);
            assert("Pointer alignment check" && (unsigned long long)(v270) % 4l == 0 && (unsigned long long)(v271) % 4l == 0);
            *v271 = *v270;
            v265 += 1l ;
        }
        assert("Tensor range check" && 0 <= v30 && v30 < 16l);
        int v272;
        v272 = v30 + v29;
        v0[v272] = v263;
        v30 += 1l ;
    }
    __syncthreads();
    return ;
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1) {
    unsigned long long v2;
    v2 = clock64();
    int v3;
    v3 = threadIdx.x;
    int v4;
    v4 = blockIdx.x;
    int v5;
    v5 = v4 * 32l;
    int v6;
    v6 = v3 + v5;
    unsigned long long v7;
    v7 = (unsigned long long)v6;
    curandStatePhilox4_32_10_t v8;
    curand_init(v2,v7,0ull,&v8);
    int v9;
    v9 = 0l;
    while (while_method_0(v9)){
        float * v11;
        v11 = reinterpret_cast<float *>(&v0[0ull]);
        float * v13;
        v13 = reinterpret_cast<float *>(&v1[0ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 16l);
        int v15;
        v15 = 16384l * v9;
        float * v16;
        v16 = reinterpret_cast<float *>(&v0[8192ull]);
        int v18;
        v18 = blockIdx.x;
        assert("Tensor range check" && 0 <= v18 && v18 < 1l);
        int v19;
        v19 = 2048l * v18;
        int v20;
        v20 = blockIdx.x;
        assert("Tensor range check" && 0 <= v20 && v20 < 1l);
        int v21;
        v21 = 2048l * v20;
        method_0(v13, v15, v16, v21, v11, v19);
        float * v22;
        v22 = reinterpret_cast<float *>(&v0[16384ull]);
        method_1(v22, v16);
        float * v24;
        v24 = reinterpret_cast<float *>(&v0[24576ull]);
        method_2(v24, v22);
        float * v26;
        v26 = reinterpret_cast<float *>(&v1[1048576ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 16l);
        float * v28;
        v28 = reinterpret_cast<float *>(&v0[32768ull]);
        int v30;
        v30 = blockIdx.x;
        assert("Tensor range check" && 0 <= v30 && v30 < 1l);
        int v31;
        v31 = 2048l * v30;
        int v32;
        v32 = blockIdx.x;
        assert("Tensor range check" && 0 <= v32 && v32 < 1l);
        int v33;
        v33 = 2048l * v32;
        method_0(v26, v15, v28, v33, v24, v31);
        float * v34;
        v34 = reinterpret_cast<float *>(&v0[40960ull]);
        method_1(v34, v28);
        float * v36;
        v36 = reinterpret_cast<float *>(&v0[49152ull]);
        method_2(v36, v34);
        float * v38;
        v38 = reinterpret_cast<float *>(&v1[2097152ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 16l);
        float * v40;
        v40 = reinterpret_cast<float *>(&v0[57344ull]);
        int v42;
        v42 = blockIdx.x;
        assert("Tensor range check" && 0 <= v42 && v42 < 1l);
        int v43;
        v43 = 2048l * v42;
        int v44;
        v44 = blockIdx.x;
        assert("Tensor range check" && 0 <= v44 && v44 < 1l);
        int v45;
        v45 = 2048l * v44;
        method_0(v38, v15, v40, v45, v36, v43);
        float * v46;
        v46 = reinterpret_cast<float *>(&v0[65536ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 16l);
        int v48;
        v48 = 2048l * v9;
        int * v49;
        v49 = reinterpret_cast<int *>(&v0[196608ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 16l);
        int v51;
        v51 = 16l * v9;
        method_3(v49, v51, v46, v48, v40, v8);
        v9 += 1l ;
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
def method0(v0 : i32) -> bool:
    v1 = v0 < 16
    del v0
    return v1
def method1(v0 : i32) -> bool:
    v1 = v0 < 1
    del v0
    return v1
def method2(v0 : i32) -> bool:
    v1 = v0 < 128
    del v0
    return v1
def main():
    v0 = cp.empty(3145728,dtype=cp.uint8)
    v1 = cp.empty(197632,dtype=cp.uint8)
    v3 = v0[0:0+4*262144].view(cp.float32)
    v4 = cp.random.normal(0.0,1.0,262144,dtype=cp.float32) # type: ignore
    cp.copyto(v3[0:0+262144],v4[0:0+262144])
    del v3, v4
    v6 = v0[1048576:1048576+4*262144].view(cp.float32)
    v7 = cp.random.normal(0.0,1.0,262144,dtype=cp.float32) # type: ignore
    cp.copyto(v6[0:0+262144],v7[0:0+262144])
    del v6, v7
    v9 = v0[2097152:2097152+4*262144].view(cp.float32)
    v10 = cp.random.normal(0.0,1.0,262144,dtype=cp.float32) # type: ignore
    cp.copyto(v9[0:0+262144],v10[0:0+262144])
    del v9, v10
    v12 = v1[0:0+4*2048].view(cp.float32)
    v13 = cp.random.normal(0.0,1.0,2048,dtype=cp.float32) # type: ignore
    cp.copyto(v12[0:0+2048],v13[0:0+2048])
    del v12, v13
    v14 = 0
    v15 = raw_module.get_function(f"entry{v14}")
    del v14
    v15.max_dynamic_shared_size_bytes = 1536 
    v15((1,),(32,),(v1, v0),shared_mem=1536)
    del v0, v15
    v18 = "{}\n"
    v19 = "Here is the output tensor."
    print(v18.format(v19),end="")
    del v18, v19
    v21 = v1[65536:65536+4*32768].view(cp.float32)
    v70 = 0
    v71 = "{}"
    print(v71.format('['),end="")
    v72 = 0
    while method0(v72):
        v74 = v70
        v75 = v74 >= 2147483647
        del v74
        if v75:
            v76 = " ..."
            print(v71.format(v76),end="")
            del v76
            break
        else:
            pass
        del v75
        v77 = v72 == 0
        v78 = v77 != True
        del v77
        if v78:
            v79 = "; "
            print(v71.format(v79),end="")
            del v79
        else:
            pass
        del v78
        print(v71.format('['),end="")
        v80 = 0
        while method1(v80):
            v82 = v70
            v83 = v82 >= 2147483647
            del v82
            if v83:
                v84 = " ..."
                print(v71.format(v84),end="")
                del v84
                break
            else:
                pass
            del v83
            v85 = v80 == 0
            v86 = v85 != True
            del v85
            if v86:
                v87 = "; "
                print(v71.format(v87),end="")
                del v87
            else:
                pass
            del v86
            print(v71.format('['),end="")
            v88 = 0
            while method0(v88):
                v90 = v70
                v91 = v90 >= 2147483647
                del v90
                if v91:
                    v92 = " ..."
                    print(v71.format(v92),end="")
                    del v92
                    break
                else:
                    pass
                del v91
                v93 = v88 == 0
                v94 = v93 != True
                del v93
                if v94:
                    v95 = "; "
                    print(v71.format(v95),end="")
                    del v95
                else:
                    pass
                del v94
                print(v71.format('['),end="")
                v96 = 0
                while method2(v96):
                    v98 = v70
                    v99 = v98 >= 2147483647
                    del v98
                    if v99:
                        v100 = " ..."
                        print(v71.format(v100),end="")
                        del v100
                        break
                    else:
                        pass
                    del v99
                    v101 = v96 == 0
                    v102 = v101 != True
                    del v101
                    if v102:
                        v103 = "; "
                        print(v71.format(v103),end="")
                        del v103
                    else:
                        pass
                    del v102
                    v104 = v70 + 1
                    v70 = v104
                    del v104
                    v105 = v72 * 2048
                    v106 = v80 * 2048
                    v107 = v105 + v106
                    del v105, v106
                    v108 = v88 * 128
                    v109 = v107 + v108
                    del v107, v108
                    v110 = v109 + v96
                    del v109
                    v111 = v21[v110].item()
                    del v110
                    v112 = "{:.6f}"
                    print(v112.format(v111),end="")
                    del v111, v112
                    v96 += 1 
                del v96
                print(v71.format(']'),end="")
                v88 += 1 
            del v88
            print(v71.format(']'),end="")
            v80 += 1 
        del v80
        print(v71.format(']'),end="")
        v72 += 1 
    del v21, v70, v72
    print(v71.format(']'),end="")
    v115 = "\n"
    print(v115,end="")
    v117 = v1[196608:196608+4*256].view(cp.int32)
    del v1
    v153 = 0
    print(v71.format('['),end="")
    v154 = 0
    while method0(v154):
        v156 = v153
        v157 = v156 >= 2147483647
        del v156
        if v157:
            v158 = " ..."
            print(v71.format(v158),end="")
            del v158
            break
        else:
            pass
        del v157
        v159 = v154 == 0
        v160 = v159 != True
        del v159
        if v160:
            v161 = "; "
            print(v71.format(v161),end="")
            del v161
        else:
            pass
        del v160
        print(v71.format('['),end="")
        v162 = 0
        while method1(v162):
            v164 = v153
            v165 = v164 >= 2147483647
            del v164
            if v165:
                v166 = " ..."
                print(v71.format(v166),end="")
                del v166
                break
            else:
                pass
            del v165
            v167 = v162 == 0
            v168 = v167 != True
            del v167
            if v168:
                v169 = "; "
                print(v71.format(v169),end="")
                del v169
            else:
                pass
            del v168
            print(v71.format('['),end="")
            v170 = 0
            while method0(v170):
                v172 = v153
                v173 = v172 >= 2147483647
                del v172
                if v173:
                    v174 = " ..."
                    print(v71.format(v174),end="")
                    del v174
                    break
                else:
                    pass
                del v173
                v175 = v170 == 0
                v176 = v175 != True
                del v175
                if v176:
                    v177 = "; "
                    print(v71.format(v177),end="")
                    del v177
                else:
                    pass
                del v176
                v178 = v153 + 1
                v153 = v178
                del v178
                v179 = v154 * 16
                v180 = v162 * 16
                v181 = v179 + v180
                del v179, v180
                v182 = v181 + v170
                del v181
                v183 = v117[v182].item()
                del v182
                print(v71.format(v183),end="")
                del v183
                v170 += 1 
            del v170
            print(v71.format(']'),end="")
            v162 += 1 
        del v162
        print(v71.format(']'),end="")
        v154 += 1 
    del v117, v153, v154
    print(v71.format(']'),end="")
    del v71
    print(v115,end="")
    del v115
    return 

if __name__ == '__main__': print(main())
