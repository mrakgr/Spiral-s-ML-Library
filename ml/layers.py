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

__device__ void method_1(float * v0, int v1, float * v2, int v3, float * v4, int v5);
__device__ void method_0(unsigned char * v0, unsigned char * v1, int v2, int v3);
__device__ void method_3(float * v0, int v1, float * v2);
__device__ void method_2(unsigned char * v0, unsigned char * v1, int v2, int v3);
__device__ void method_5(float * v0, int v1, float * v2);
__device__ void method_4(unsigned char * v0, unsigned char * v1, int v2, int v3);
__device__ void method_7(float * v0, int v1, float * v2, int v3, float * v4);
__device__ void method_6(unsigned char * v0, unsigned char * v1, int v2, int v3);
__device__ void method_8(unsigned char * v0, unsigned char * v1, int v2, int v3);
__device__ void method_9(unsigned char * v0, unsigned char * v1, int v2, int v3);
__device__ void method_10(unsigned char * v0, unsigned char * v1, int v2, int v3);
struct Tuple0;
struct Tuple1;
__device__ void method_12(curandStatePhilox4_32_10_t & v0, int * v1, int v2, float * v3, int v4);
__device__ void method_11(unsigned char * v0, unsigned char * v1, int v2, int v3, curandStatePhilox4_32_10_t & v4);
struct Closure0 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
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
struct Closure1 {
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
struct Closure2 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v0 >= 0.0f;
        bool v6;
        if (v4){
            bool v5;
            v5 = v2 >= 0.0f;
            v6 = v5;
        } else {
            v6 = false;
        }
        if (v6){
            bool v7;
            v7 = v0 <= v2;
            if (v7){
                return Tuple1{v0, v1};
            } else {
                return Tuple1{v2, v3};
            }
        } else {
            if (v4){
                return Tuple1{v0, v1};
            } else {
                bool v10;
                v10 = v2 >= 0.0f;
                if (v10){
                    return Tuple1{v2, v3};
                } else {
                    return Tuple1{v0, v1};
                }
            }
        }
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
__device__ void method_1(float * v0, int v1, float * v2, int v3, float * v4, int v5){
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
    extern __shared__ unsigned char v10[];
    float * v11;
    v11 = reinterpret_cast<float *>(&v10[0ull]);
    float * v12;
    v12 = reinterpret_cast<float *>(&v10[768ull]);
    float * v13;
    v13 = reinterpret_cast<float *>(&v10[0ull]);
    int v14;
    v14 = threadIdx.x;
    int v15;
    v15 = v14 / 32l;
    bool v16;
    v16 = 0l <= v15;
    bool v17;
    v17 = v16 == false;
    if (v17){
        assert("The index needs to be zero or positive." && v16);
    } else {
    }
    int v18;
    v18 = v15 % 1l;
    bool v19;
    v19 = v15 < 1l;
    bool v20;
    v20 = v19 == false;
    if (v20){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v19);
    } else {
    }
    assert("Tensor range check" && 0 <= v15 && v15 < 1l);
    assert("Tensor range check" && 0 <= v18 && v18 < 1l);
    int v21;
    v21 = 16l * v18;
    int v22;
    v22 = 384l * v15;
    int v23;
    v23 = v22 + v21;
    float * v24;
    v24 = v13+v23;
    assert("Tensor range check" && 0 <= v15 && v15 < 1l);
    int v25;
    v25 = 192l * v15;
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
    int v30;
    v30 = v27 % 4l;
    int v31;
    v31 = v27 / 4l;
    bool v32;
    v32 = v31 < 8l;
    bool v33;
    v33 = v32 == false;
    if (v33){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v32);
    } else {
    }
    assert("Tensor range check" && 0 <= v31 && v31 < 8l);
    assert("Tensor range check" && 0 <= v30 && v30 < 4l);
    int v34;
    v34 = v30 + v25;
    int v35;
    v35 = 12l * v31;
    int v36;
    v36 = v35 + v34;
    float * v37;
    v37 = v11+v36;
    assert("Tensor range check" && 0 <= v18 && v18 < 1l);
    int v38;
    v38 = 192l * v18;
    int v39;
    v39 = threadIdx.x;
    int v40;
    v40 = v39 % 32l;
    bool v41;
    v41 = 0l <= v40;
    bool v42;
    v42 = v41 == false;
    if (v42){
        assert("The index needs to be zero or positive." && v41);
    } else {
    }
    int v43;
    v43 = v40 % 4l;
    int v44;
    v44 = v40 / 4l;
    bool v45;
    v45 = v44 < 8l;
    bool v46;
    v46 = v45 == false;
    if (v46){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v45);
    } else {
    }
    assert("Tensor range check" && 0 <= v44 && v44 < 8l);
    assert("Tensor range check" && 0 <= v43 && v43 < 4l);
    int v47;
    v47 = v43 + v38;
    int v48;
    v48 = 12l * v44;
    int v49;
    v49 = v48 + v47;
    float * v50;
    v50 = v12+v49;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v51[1l];
    int v52;
    v52 = 0l;
    while (while_method_1(v52)){
        int v54;
        v54 = 0l;
        while (while_method_1(v54)){
            assert("Tensor range check" && 0 <= v52 && v52 < 1l);
            assert("Tensor range check" && 0 <= v54 && v54 < 1l);
            int v56;
            v56 = 16l * v54;
            int v57;
            v57 = v56 + v1;
            int v58;
            v58 = 256l * v52;
            int v59;
            v59 = v58 + v57;
            float * v60;
            v60 = v0+v59;
            // Pushing the loop unrolling to: 0
            int v61;
            v61 = 0l;
            #pragma unroll
            while (while_method_1(v61)){
                int v63;
                v63 = 0l;
                #pragma unroll
                while (while_method_1(v63)){
                    assert("Tensor range check" && 0 <= v61 && v61 < 1l);
                    assert("Tensor range check" && 0 <= v63 && v63 < 1l);
                    int v65;
                    v65 = v61 + v63;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v66 = v51[v65];
                    wmma::fill_fragment(v66, 0.0f);
                    v63 += 1l ;
                }
                v61 += 1l ;
            }
            int v67;
            v67 = 0l;
            #pragma unroll
            while (while_method_1(v67)){
                assert("Tensor range check" && 0 <= v52 && v52 < 1l);
                int v69;
                v69 = 128l * v52;
                int v70;
                v70 = v69 + v5;
                assert("Tensor range check" && 0 <= v67 && v67 < 1l);
                int v71;
                v71 = 8l * v67;
                int v72;
                v72 = v71 + v70;
                float * v73;
                v73 = v4+v72;
                assert("Tensor range check" && 0 <= v54 && v54 < 1l);
                int v74;
                v74 = 128l * v54;
                int v75;
                v75 = v74 + v3;
                assert("Tensor range check" && 0 <= v67 && v67 < 1l);
                int v76;
                v76 = v71 + v75;
                float * v77;
                v77 = v2+v76;
                int v78;
                v78 = threadIdx.x;
                bool v79;
                v79 = 0l <= v78;
                bool v80;
                v80 = v79 == false;
                if (v80){
                    assert("The index needs to be zero or positive." && v79);
                } else {
                }
                int v81;
                v81 = v78 % 2l;
                int v82;
                v82 = v78 / 2l;
                bool v83;
                v83 = v82 < 16l;
                bool v84;
                v84 = v83 == false;
                if (v84){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v83);
                } else {
                }
                assert("Tensor range check" && 0 <= v82 && v82 < 16l);
                assert("Tensor range check" && 0 <= v81 && v81 < 2l);
                int v85;
                v85 = 4l * v81;
                int v86;
                v86 = 12l * v82;
                int v87;
                v87 = v86 + v85;
                int v88;
                v88 = 8l * v82;
                int v89;
                v89 = v88 + v85;
                float * v90;
                v90 = v12+v87;
                float * v91;
                v91 = v77+v89;
                int v92;
                v92 = 0l;
                #pragma unroll
                while (while_method_1(v92)){
                    int v94;
                    v94 = 0l;
                    #pragma unroll
                    while (while_method_1(v94)){
                        assert("Tensor range check" && 0 <= v92 && v92 < 1l);
                        assert("Tensor range check" && 0 <= v94 && v94 < 1l);
                        int v96;
                        v96 = 8l * v94;
                        int v97;
                        v97 = 192l * v92;
                        int v98;
                        v98 = v97 + v96;
                        int v99;
                        v99 = 128l * v92;
                        int v100;
                        v100 = v99 + v96;
                        float v101[4l];
                        int v102;
                        v102 = 0l;
                        #pragma unroll
                        while (while_method_0(v102)){
                            assert("Tensor range check" && 0 <= v102 && v102 < 4l);
                            int v104;
                            v104 = v102 + v100;
                            float v105;
                            v105 = v91[v104];
                            float v106;
                            v106 = wmma::__float_to_tf32(v105);
                            assert("Tensor range check" && 0 <= v102 && v102 < 4l);
                            v101[v102] = v106;
                            v102 += 1l ;
                        }
                        int4* v107;
                        v107 = reinterpret_cast<int4*>(v101 + 0l);
                        int4* v108;
                        v108 = reinterpret_cast<int4*>(v90 + v98);
                        assert("Pointer alignment check" && (unsigned long long)(v107) % 4l == 0 && (unsigned long long)(v108) % 4l == 0);
                        *v108 = *v107;
                        v94 += 1l ;
                    }
                    v92 += 1l ;
                }
                int v109;
                v109 = threadIdx.x;
                bool v110;
                v110 = 0l <= v109;
                bool v111;
                v111 = v110 == false;
                if (v111){
                    assert("The index needs to be zero or positive." && v110);
                } else {
                }
                int v112;
                v112 = v109 % 2l;
                int v113;
                v113 = v109 / 2l;
                bool v114;
                v114 = v113 < 16l;
                bool v115;
                v115 = v114 == false;
                if (v115){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v114);
                } else {
                }
                assert("Tensor range check" && 0 <= v113 && v113 < 16l);
                assert("Tensor range check" && 0 <= v112 && v112 < 2l);
                int v116;
                v116 = 4l * v112;
                int v117;
                v117 = 12l * v113;
                int v118;
                v118 = v117 + v116;
                int v119;
                v119 = 8l * v113;
                int v120;
                v120 = v119 + v116;
                float * v121;
                v121 = v11+v118;
                float * v122;
                v122 = v73+v120;
                int v123;
                v123 = 0l;
                #pragma unroll
                while (while_method_1(v123)){
                    int v125;
                    v125 = 0l;
                    #pragma unroll
                    while (while_method_1(v125)){
                        assert("Tensor range check" && 0 <= v123 && v123 < 1l);
                        assert("Tensor range check" && 0 <= v125 && v125 < 1l);
                        int v127;
                        v127 = 8l * v125;
                        int v128;
                        v128 = 192l * v123;
                        int v129;
                        v129 = v128 + v127;
                        int v130;
                        v130 = 128l * v123;
                        int v131;
                        v131 = v130 + v127;
                        float v132[4l];
                        int v133;
                        v133 = 0l;
                        #pragma unroll
                        while (while_method_0(v133)){
                            assert("Tensor range check" && 0 <= v133 && v133 < 4l);
                            int v135;
                            v135 = v133 + v131;
                            float v136;
                            v136 = v122[v135];
                            float v137;
                            v137 = wmma::__float_to_tf32(v136);
                            assert("Tensor range check" && 0 <= v133 && v133 < 4l);
                            v132[v133] = v137;
                            v133 += 1l ;
                        }
                        int4* v138;
                        v138 = reinterpret_cast<int4*>(v132 + 0l);
                        int4* v139;
                        v139 = reinterpret_cast<int4*>(v121 + v129);
                        assert("Pointer alignment check" && (unsigned long long)(v138) % 4l == 0 && (unsigned long long)(v139) % 4l == 0);
                        *v139 = *v138;
                        v125 += 1l ;
                    }
                    v123 += 1l ;
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v140[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v141[1l];
                int v142;
                v142 = 0l;
                #pragma unroll
                while (while_method_1(v142)){
                    int v144;
                    v144 = 0l;
                    #pragma unroll
                    while (while_method_1(v144)){
                        assert("Tensor range check" && 0 <= v142 && v142 < 1l);
                        assert("Tensor range check" && 0 <= v144 && v144 < 1l);
                        int v146;
                        v146 = v142 + v144;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v147 = v140[v146];
                        assert("Tensor range check" && 0 <= v142 && v142 < 1l);
                        int v148;
                        v148 = 192l * v142;
                        assert("Tensor range check" && 0 <= v144 && v144 < 1l);
                        int v149;
                        v149 = 8l * v144;
                        int v150;
                        v150 = v149 + v148;
                        int v151;
                        v151 = 0l;
                        #pragma unroll
                        while (while_method_2(v151)){
                            int v153;
                            v153 = 0l;
                            #pragma unroll
                            while (while_method_2(v153)){
                                assert("Tensor range check" && 0 <= v151 && v151 < 2l);
                                assert("Tensor range check" && 0 <= v153 && v153 < 2l);
                                int v155;
                                v155 = 96l * v153;
                                int v156;
                                v156 = v155 + v150;
                                int v157;
                                v157 = 4l * v151;
                                int v158;
                                v158 = v157 + v156;
                                float v159;
                                v159 = v37[v158];
                                bool v160;
                                v160 = 0l <= v153;
                                bool v162;
                                if (v160){
                                    bool v161;
                                    v161 = v153 < 2l;
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
                                bool v164;
                                v164 = 0l <= v151;
                                bool v166;
                                if (v164){
                                    bool v165;
                                    v165 = v151 < 2l;
                                    v166 = v165;
                                } else {
                                    v166 = false;
                                }
                                bool v167;
                                v167 = v166 == false;
                                if (v167){
                                    assert("The indices should be inside the range of the dimension." && v166);
                                } else {
                                }
                                int v168;
                                v168 = v151 * 2l;
                                int v169;
                                v169 = v153 + v168;
                                v147.x[v169] = v159;
                                v153 += 1l ;
                            }
                            v151 += 1l ;
                        }
                        v144 += 1l ;
                    }
                    v142 += 1l ;
                }
                int v170;
                v170 = 0l;
                #pragma unroll
                while (while_method_1(v170)){
                    int v172;
                    v172 = 0l;
                    #pragma unroll
                    while (while_method_1(v172)){
                        assert("Tensor range check" && 0 <= v170 && v170 < 1l);
                        assert("Tensor range check" && 0 <= v172 && v172 < 1l);
                        int v174;
                        v174 = v170 + v172;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v175 = v141[v174];
                        assert("Tensor range check" && 0 <= v170 && v170 < 1l);
                        int v176;
                        v176 = 192l * v170;
                        assert("Tensor range check" && 0 <= v172 && v172 < 1l);
                        int v177;
                        v177 = 8l * v172;
                        int v178;
                        v178 = v177 + v176;
                        int v179;
                        v179 = 0l;
                        #pragma unroll
                        while (while_method_2(v179)){
                            int v181;
                            v181 = 0l;
                            #pragma unroll
                            while (while_method_2(v181)){
                                assert("Tensor range check" && 0 <= v179 && v179 < 2l);
                                assert("Tensor range check" && 0 <= v181 && v181 < 2l);
                                int v183;
                                v183 = 4l * v181;
                                int v184;
                                v184 = v183 + v178;
                                int v185;
                                v185 = 96l * v179;
                                int v186;
                                v186 = v185 + v184;
                                float v187;
                                v187 = v50[v186];
                                bool v188;
                                v188 = 0l <= v181;
                                bool v190;
                                if (v188){
                                    bool v189;
                                    v189 = v181 < 2l;
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
                                bool v192;
                                v192 = 0l <= v179;
                                bool v194;
                                if (v192){
                                    bool v193;
                                    v193 = v179 < 2l;
                                    v194 = v193;
                                } else {
                                    v194 = false;
                                }
                                bool v195;
                                v195 = v194 == false;
                                if (v195){
                                    assert("The indices should be inside the range of the dimension." && v194);
                                } else {
                                }
                                int v196;
                                v196 = v179 * 2l;
                                int v197;
                                v197 = v181 + v196;
                                v175.x[v197] = v187;
                                v181 += 1l ;
                            }
                            v179 += 1l ;
                        }
                        v172 += 1l ;
                    }
                    v170 += 1l ;
                }
                __syncthreads();
                int v198;
                v198 = 0l;
                #pragma unroll
                while (while_method_1(v198)){
                    int v200;
                    v200 = 0l;
                    #pragma unroll
                    while (while_method_1(v200)){
                        int v202;
                        v202 = 0l;
                        #pragma unroll
                        while (while_method_1(v202)){
                            assert("Tensor range check" && 0 <= v198 && v198 < 1l);
                            assert("Tensor range check" && 0 <= v200 && v200 < 1l);
                            int v204;
                            v204 = v198 + v200;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v205 = v51[v204];
                            assert("Tensor range check" && 0 <= v198 && v198 < 1l);
                            assert("Tensor range check" && 0 <= v202 && v202 < 1l);
                            int v206;
                            v206 = v198 + v202;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v207 = v140[v206];
                            assert("Tensor range check" && 0 <= v200 && v200 < 1l);
                            assert("Tensor range check" && 0 <= v202 && v202 < 1l);
                            int v208;
                            v208 = v200 + v202;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v209 = v141[v208];
                            wmma::mma_sync(v205, v207, v209, v205);
                            v202 += 1l ;
                        }
                        v200 += 1l ;
                    }
                    v198 += 1l ;
                }
                v67 += 1l ;
            }
            int v210;
            v210 = 0l;
            #pragma unroll
            while (while_method_1(v210)){
                int v212;
                v212 = 0l;
                #pragma unroll
                while (while_method_1(v212)){
                    assert("Tensor range check" && 0 <= v210 && v210 < 1l);
                    assert("Tensor range check" && 0 <= v212 && v212 < 1l);
                    int v214;
                    v214 = v210 + v212;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v215 = v51[v214];
                    assert("Tensor range check" && 0 <= v210 && v210 < 1l);
                    assert("Tensor range check" && 0 <= v212 && v212 < 1l);
                    int v216;
                    v216 = 16l * v212;
                    int v217;
                    v217 = 384l * v210;
                    int v218;
                    v218 = v217 + v216;
                    float * v219;
                    v219 = v24+v218;
                    wmma::store_matrix_sync(v219, v215, 24l, wmma::mem_row_major);
                    v212 += 1l ;
                }
                v210 += 1l ;
            }
            __syncthreads();
            int v220;
            v220 = threadIdx.x;
            bool v221;
            v221 = 0l <= v220;
            bool v222;
            v222 = v221 == false;
            if (v222){
                assert("The index needs to be zero or positive." && v221);
            } else {
            }
            int v223;
            v223 = v220 % 4l;
            int v224;
            v224 = v220 / 4l;
            bool v225;
            v225 = v224 < 8l;
            bool v226;
            v226 = v225 == false;
            if (v226){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v225);
            } else {
            }
            assert("Tensor range check" && 0 <= v224 && v224 < 8l);
            assert("Tensor range check" && 0 <= v223 && v223 < 4l);
            int v227;
            v227 = 4l * v223;
            int v228;
            v228 = 16l * v224;
            int v229;
            v229 = v228 + v227;
            int v230;
            v230 = 24l * v224;
            int v231;
            v231 = v230 + v227;
            float * v232;
            v232 = v60+v229;
            float * v233;
            v233 = v13+v231;
            int v234;
            v234 = 0l;
            #pragma unroll
            while (while_method_2(v234)){
                int v236;
                v236 = 0l;
                #pragma unroll
                while (while_method_1(v236)){
                    assert("Tensor range check" && 0 <= v234 && v234 < 2l);
                    assert("Tensor range check" && 0 <= v236 && v236 < 1l);
                    int v238;
                    v238 = 16l * v236;
                    int v239;
                    v239 = 128l * v234;
                    int v240;
                    v240 = v239 + v238;
                    int v241;
                    v241 = 192l * v234;
                    int v242;
                    v242 = v241 + v238;
                    int4* v243;
                    v243 = reinterpret_cast<int4*>(v233 + v242);
                    int4* v244;
                    v244 = reinterpret_cast<int4*>(v232 + v240);
                    assert("Pointer alignment check" && (unsigned long long)(v243) % 4l == 0 && (unsigned long long)(v244) % 4l == 0);
                    *v244 = *v243;
                    v236 += 1l ;
                }
                v234 += 1l ;
            }
            __syncthreads();
            // Poping the loop unrolling to: 0
            v54 += 1l ;
        }
        v52 += 1l ;
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
    v6 = reinterpret_cast<float *>(&v1[0ull]);
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
__device__ void method_3(float * v0, int v1, float * v2){
    int v3;
    v3 = threadIdx.x;
    bool v4;
    v4 = 0l <= v3;
    bool v5;
    v5 = v4 == false;
    if (v5){
        assert("The index needs to be zero or positive." && v4);
    } else {
    }
    int v6;
    v6 = v3 % 4l;
    int v7;
    v7 = v3 / 4l;
    bool v8;
    v8 = v7 < 8l;
    bool v9;
    v9 = v8 == false;
    if (v9){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v8);
    } else {
    }
    assert("Tensor range check" && 0 <= v7 && v7 < 8l);
    assert("Tensor range check" && 0 <= v6 && v6 < 4l);
    int v10;
    v10 = 4l * v6;
    int v11;
    v11 = v10 + v1;
    int v12;
    v12 = 16l * v7;
    int v13;
    v13 = v12 + v11;
    assert("Tensor range check" && 0 <= v7 && v7 < 8l);
    assert("Tensor range check" && 0 <= v6 && v6 < 4l);
    int v14;
    v14 = 0l;
    while (while_method_2(v14)){
        assert("Tensor range check" && 0 <= v14 && v14 < 2l);
        int v16;
        v16 = 128l * v14;
        int v17;
        v17 = v16 + v13;
        assert("Tensor range check" && 0 <= v14 && v14 < 2l);
        float v18[4l];
        int v19[4l];
        int v20;
        v20 = 0l;
        while (while_method_1(v20)){
            assert("Tensor range check" && 0 <= v20 && v20 < 1l);
            int v22;
            v22 = 4l * v20;
            assert("Tensor range check" && 0 <= v20 && v20 < 1l);
            int v23;
            v23 = 16l * v20;
            int v24;
            v24 = v23 + v17;
            int4* v25;
            v25 = reinterpret_cast<int4*>(v2 + v24);
            int4* v26;
            v26 = reinterpret_cast<int4*>(v18 + v22);
            assert("Pointer alignment check" && (unsigned long long)(v25) % 4l == 0 && (unsigned long long)(v26) % 4l == 0);
            *v26 = *v25;
            v20 += 1l ;
        }
        int v27;
        v27 = 0l;
        while (while_method_1(v27)){
            int v29;
            v29 = 0l;
            while (while_method_0(v29)){
                bool v31;
                v31 = 0l <= v29;
                bool v33;
                if (v31){
                    bool v32;
                    v32 = v29 < 4l;
                    v33 = v32;
                } else {
                    v33 = false;
                }
                bool v34;
                v34 = v33 == false;
                if (v34){
                    assert("The indices should be inside the range of the dimension." && v33);
                } else {
                }
                bool v35;
                v35 = 0l <= v6;
                bool v37;
                if (v35){
                    bool v36;
                    v36 = v6 < 4l;
                    v37 = v36;
                } else {
                    v37 = false;
                }
                bool v38;
                v38 = v37 == false;
                if (v38){
                    assert("The indices should be inside the range of the dimension." && v37);
                } else {
                }
                int v39;
                v39 = v6 * 4l;
                int v40;
                v40 = v29 + v39;
                bool v41;
                v41 = 0l <= v27;
                bool v43;
                if (v41){
                    bool v42;
                    v42 = v27 < 1l;
                    v43 = v42;
                } else {
                    v43 = false;
                }
                bool v44;
                v44 = v43 == false;
                if (v44){
                    assert("The indices should be inside the range of the dimension." && v43);
                } else {
                }
                int v45;
                v45 = v27 * 16l;
                int v46;
                v46 = v40 + v45;
                assert("Tensor range check" && 0 <= v27 && v27 < 1l);
                assert("Tensor range check" && 0 <= v29 && v29 < 4l);
                int v47;
                v47 = 4l * v27;
                int v48;
                v48 = v47 + v29;
                v19[v48] = v46;
                v29 += 1l ;
            }
            v27 += 1l ;
        }
        bool v49;
        v49 = 0l <= v7;
        bool v50;
        v50 = v49 && v8;
        bool v51;
        v51 = v50 == false;
        if (v51){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v50);
        } else {
        }
        bool v52;
        v52 = 0l <= v14;
        bool v54;
        if (v52){
            bool v53;
            v53 = v14 < 2l;
            v54 = v53;
        } else {
            v54 = false;
        }
        bool v55;
        v55 = v54 == false;
        if (v55){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v54);
        } else {
        }
        int v56;
        v56 = v14 * 8l;
        int v57;
        v57 = v56 + v7;
        float v58[4l];
        int v59;
        v59 = 0l;
        while (while_method_1(v59)){
            int v61;
            v61 = 0l;
            while (while_method_0(v61)){
                assert("Tensor range check" && 0 <= v59 && v59 < 1l);
                assert("Tensor range check" && 0 <= v61 && v61 < 4l);
                int v63;
                v63 = 4l * v59;
                int v64;
                v64 = v63 + v61;
                float v65;
                v65 = v18[v64];
                float v66;
                v66 = v65 * v65;
                assert("Tensor range check" && 0 <= v59 && v59 < 1l);
                assert("Tensor range check" && 0 <= v61 && v61 < 4l);
                v58[v64] = v66;
                v61 += 1l ;
            }
            v59 += 1l ;
        }
        float v67;
        v67 = 0.0f;
        int v68;
        v68 = 0l;
        while (while_method_1(v68)){
            int v70;
            v70 = 0l;
            while (while_method_0(v70)){
                assert("Tensor range check" && 0 <= v68 && v68 < 1l);
                assert("Tensor range check" && 0 <= v70 && v70 < 4l);
                int v72;
                v72 = 4l * v68;
                int v73;
                v73 = v72 + v70;
                float v74;
                v74 = v58[v73];
                float v75;
                v75 = v67 + v74;
                v67 = v75;
                v70 += 1l ;
            }
            v68 += 1l ;
        }
        auto v76 = cooperative_groups::coalesced_threads();
        int v77;
        v77 = threadIdx.x;
        int v78;
        v78 = v77 / 4l;
        auto v79 = cooperative_groups::labeled_partition(v76,v78);
        Closure0 v80{};
        float v81;
        v81 = cooperative_groups::reduce(v79, v67, v80);
        float v82[4l];
        int v83;
        v83 = 0l;
        while (while_method_1(v83)){
            int v85;
            v85 = 0l;
            while (while_method_0(v85)){
                assert("Tensor range check" && 0 <= v83 && v83 < 1l);
                assert("Tensor range check" && 0 <= v85 && v85 < 4l);
                int v87;
                v87 = 4l * v83;
                int v88;
                v88 = v87 + v85;
                float v89;
                v89 = v18[v88];
                bool v90;
                v90 = v81 == 0.0f;
                bool v91;
                v91 = v90 != true;
                float v93;
                if (v91){
                    float v92;
                    v92 = v89 / v81;
                    v93 = v92;
                } else {
                    v93 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v83 && v83 < 1l);
                assert("Tensor range check" && 0 <= v85 && v85 < 4l);
                v82[v88] = v93;
                v85 += 1l ;
            }
            v83 += 1l ;
        }
        int v94;
        v94 = 0l;
        while (while_method_1(v94)){
            assert("Tensor range check" && 0 <= v94 && v94 < 1l);
            int v96;
            v96 = 16l * v94;
            int v97;
            v97 = v96 + v17;
            assert("Tensor range check" && 0 <= v94 && v94 < 1l);
            int v98;
            v98 = 4l * v94;
            int4* v99;
            v99 = reinterpret_cast<int4*>(v82 + v98);
            int4* v100;
            v100 = reinterpret_cast<int4*>(v0 + v97);
            assert("Pointer alignment check" && (unsigned long long)(v99) % 4l == 0 && (unsigned long long)(v100) % 4l == 0);
            *v100 = *v99;
            v94 += 1l ;
        }
        v14 += 1l ;
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
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 64l;
    return v1;
}
__device__ void method_5(float * v0, int v1, float * v2){
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
        while (while_method_0(v20)){
            assert("Tensor range check" && 0 <= v20 && v20 < 4l);
            float v22;
            v22 = v16[v20];
            bool v23;
            v23 = 0.0f >= v22;
            float v24;
            if (v23){
                v24 = 0.0f;
            } else {
                v24 = v22;
            }
            assert("Tensor range check" && 0 <= v20 && v20 < 4l);
            v17[v20] = v24;
            v20 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v25;
        v25 = reinterpret_cast<int4*>(v17 + 0l);
        int4* v26;
        v26 = reinterpret_cast<int4*>(v0 + v15);
        assert("Pointer alignment check" && (unsigned long long)(v25) % 4l == 0 && (unsigned long long)(v26) % 4l == 0);
        *v26 = *v25;
        v4 += 32l ;
    }
    __syncthreads();
    return ;
}
__device__ void method_4(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[1536ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v5;
    v5 = 256l * v3;
    float * v6;
    v6 = reinterpret_cast<float *>(&v0[2560ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    return method_5(v6, v5, v4);
}
__device__ void method_7(float * v0, int v1, float * v2, int v3, float * v4){
    unsigned int v5;
    v5 = 0ul;
    asm("mov.u32 %0, %dynamic_smem_size;" : "=r"(v5));
    unsigned long long v6;
    v6 = (unsigned long long)v5;
    bool v7;
    v7 = 1536ull <= v6;
    bool v8;
    v8 = v7 == false;
    if (v8){
        assert("The shared memory used in the matmult node is lower than the allocated amount." && v7);
    } else {
    }
    extern __shared__ unsigned char v9[];
    float * v10;
    v10 = reinterpret_cast<float *>(&v9[0ull]);
    float * v11;
    v11 = reinterpret_cast<float *>(&v9[768ull]);
    float * v12;
    v12 = reinterpret_cast<float *>(&v9[0ull]);
    int v13;
    v13 = threadIdx.x;
    int v14;
    v14 = v13 / 32l;
    bool v15;
    v15 = 0l <= v14;
    bool v16;
    v16 = v15 == false;
    if (v16){
        assert("The index needs to be zero or positive." && v15);
    } else {
    }
    int v17;
    v17 = v14 % 1l;
    bool v18;
    v18 = v14 < 1l;
    bool v19;
    v19 = v18 == false;
    if (v19){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v18);
    } else {
    }
    assert("Tensor range check" && 0 <= v14 && v14 < 1l);
    assert("Tensor range check" && 0 <= v17 && v17 < 1l);
    int v20;
    v20 = 16l * v17;
    int v21;
    v21 = 384l * v14;
    int v22;
    v22 = v21 + v20;
    float * v23;
    v23 = v12+v22;
    assert("Tensor range check" && 0 <= v14 && v14 < 1l);
    int v24;
    v24 = 192l * v14;
    int v25;
    v25 = threadIdx.x;
    int v26;
    v26 = v25 % 32l;
    bool v27;
    v27 = 0l <= v26;
    bool v28;
    v28 = v27 == false;
    if (v28){
        assert("The index needs to be zero or positive." && v27);
    } else {
    }
    int v29;
    v29 = v26 % 4l;
    int v30;
    v30 = v26 / 4l;
    bool v31;
    v31 = v30 < 8l;
    bool v32;
    v32 = v31 == false;
    if (v32){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v31);
    } else {
    }
    assert("Tensor range check" && 0 <= v30 && v30 < 8l);
    assert("Tensor range check" && 0 <= v29 && v29 < 4l);
    int v33;
    v33 = v29 + v24;
    int v34;
    v34 = 12l * v30;
    int v35;
    v35 = v34 + v33;
    float * v36;
    v36 = v10+v35;
    assert("Tensor range check" && 0 <= v17 && v17 < 1l);
    int v37;
    v37 = 192l * v17;
    int v38;
    v38 = threadIdx.x;
    int v39;
    v39 = v38 % 32l;
    bool v40;
    v40 = 0l <= v39;
    bool v41;
    v41 = v40 == false;
    if (v41){
        assert("The index needs to be zero or positive." && v40);
    } else {
    }
    int v42;
    v42 = v39 % 4l;
    int v43;
    v43 = v39 / 4l;
    bool v44;
    v44 = v43 < 8l;
    bool v45;
    v45 = v44 == false;
    if (v45){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v44);
    } else {
    }
    assert("Tensor range check" && 0 <= v43 && v43 < 8l);
    assert("Tensor range check" && 0 <= v42 && v42 < 4l);
    int v46;
    v46 = v42 + v37;
    int v47;
    v47 = 12l * v43;
    int v48;
    v48 = v47 + v46;
    float * v49;
    v49 = v11+v48;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v50[1l];
    int v51;
    v51 = 0l;
    while (while_method_1(v51)){
        int v53;
        v53 = 0l;
        while (while_method_1(v53)){
            assert("Tensor range check" && 0 <= v51 && v51 < 1l);
            assert("Tensor range check" && 0 <= v53 && v53 < 1l);
            int v55;
            v55 = 16l * v53;
            int v56;
            v56 = v55 + v1;
            int v57;
            v57 = 256l * v51;
            int v58;
            v58 = v57 + v56;
            float * v59;
            v59 = v0+v58;
            // Pushing the loop unrolling to: 0
            int v60;
            v60 = 0l;
            #pragma unroll
            while (while_method_1(v60)){
                int v62;
                v62 = 0l;
                #pragma unroll
                while (while_method_1(v62)){
                    assert("Tensor range check" && 0 <= v60 && v60 < 1l);
                    assert("Tensor range check" && 0 <= v62 && v62 < 1l);
                    int v64;
                    v64 = v60 + v62;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v65 = v50[v64];
                    wmma::fill_fragment(v65, 0.0f);
                    v62 += 1l ;
                }
                v60 += 1l ;
            }
            int v66;
            v66 = 0l;
            #pragma unroll
            while (while_method_2(v66)){
                assert("Tensor range check" && 0 <= v51 && v51 < 1l);
                int v68;
                v68 = v57 + v1;
                assert("Tensor range check" && 0 <= v66 && v66 < 2l);
                int v69;
                v69 = 8l * v66;
                int v70;
                v70 = v69 + v68;
                float * v71;
                v71 = v4+v70;
                assert("Tensor range check" && 0 <= v53 && v53 < 1l);
                int v72;
                v72 = 256l * v53;
                int v73;
                v73 = v72 + v3;
                assert("Tensor range check" && 0 <= v66 && v66 < 2l);
                int v74;
                v74 = v69 + v73;
                float * v75;
                v75 = v2+v74;
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
                int v79;
                v79 = v76 % 2l;
                int v80;
                v80 = v76 / 2l;
                bool v81;
                v81 = v80 < 16l;
                bool v82;
                v82 = v81 == false;
                if (v82){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v81);
                } else {
                }
                assert("Tensor range check" && 0 <= v80 && v80 < 16l);
                assert("Tensor range check" && 0 <= v79 && v79 < 2l);
                int v83;
                v83 = 4l * v79;
                int v84;
                v84 = 12l * v80;
                int v85;
                v85 = v84 + v83;
                int v86;
                v86 = 16l * v80;
                int v87;
                v87 = v86 + v83;
                float * v88;
                v88 = v11+v85;
                float * v89;
                v89 = v75+v87;
                int v90;
                v90 = 0l;
                #pragma unroll
                while (while_method_1(v90)){
                    int v92;
                    v92 = 0l;
                    #pragma unroll
                    while (while_method_1(v92)){
                        assert("Tensor range check" && 0 <= v90 && v90 < 1l);
                        assert("Tensor range check" && 0 <= v92 && v92 < 1l);
                        int v94;
                        v94 = 8l * v92;
                        int v95;
                        v95 = 192l * v90;
                        int v96;
                        v96 = v95 + v94;
                        int v97;
                        v97 = 256l * v90;
                        int v98;
                        v98 = v97 + v94;
                        float v99[4l];
                        int v100;
                        v100 = 0l;
                        #pragma unroll
                        while (while_method_0(v100)){
                            assert("Tensor range check" && 0 <= v100 && v100 < 4l);
                            int v102;
                            v102 = v100 + v98;
                            float v103;
                            v103 = v89[v102];
                            float v104;
                            v104 = wmma::__float_to_tf32(v103);
                            assert("Tensor range check" && 0 <= v100 && v100 < 4l);
                            v99[v100] = v104;
                            v100 += 1l ;
                        }
                        int4* v105;
                        v105 = reinterpret_cast<int4*>(v99 + 0l);
                        int4* v106;
                        v106 = reinterpret_cast<int4*>(v88 + v96);
                        assert("Pointer alignment check" && (unsigned long long)(v105) % 4l == 0 && (unsigned long long)(v106) % 4l == 0);
                        *v106 = *v105;
                        v92 += 1l ;
                    }
                    v90 += 1l ;
                }
                int v107;
                v107 = threadIdx.x;
                bool v108;
                v108 = 0l <= v107;
                bool v109;
                v109 = v108 == false;
                if (v109){
                    assert("The index needs to be zero or positive." && v108);
                } else {
                }
                int v110;
                v110 = v107 % 2l;
                int v111;
                v111 = v107 / 2l;
                bool v112;
                v112 = v111 < 16l;
                bool v113;
                v113 = v112 == false;
                if (v113){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v112);
                } else {
                }
                assert("Tensor range check" && 0 <= v111 && v111 < 16l);
                assert("Tensor range check" && 0 <= v110 && v110 < 2l);
                int v114;
                v114 = 4l * v110;
                int v115;
                v115 = 12l * v111;
                int v116;
                v116 = v115 + v114;
                int v117;
                v117 = 16l * v111;
                int v118;
                v118 = v117 + v114;
                float * v119;
                v119 = v10+v116;
                float * v120;
                v120 = v71+v118;
                int v121;
                v121 = 0l;
                #pragma unroll
                while (while_method_1(v121)){
                    int v123;
                    v123 = 0l;
                    #pragma unroll
                    while (while_method_1(v123)){
                        assert("Tensor range check" && 0 <= v121 && v121 < 1l);
                        assert("Tensor range check" && 0 <= v123 && v123 < 1l);
                        int v125;
                        v125 = 8l * v123;
                        int v126;
                        v126 = 192l * v121;
                        int v127;
                        v127 = v126 + v125;
                        int v128;
                        v128 = 256l * v121;
                        int v129;
                        v129 = v128 + v125;
                        float v130[4l];
                        int v131;
                        v131 = 0l;
                        #pragma unroll
                        while (while_method_0(v131)){
                            assert("Tensor range check" && 0 <= v131 && v131 < 4l);
                            int v133;
                            v133 = v131 + v129;
                            float v134;
                            v134 = v120[v133];
                            float v135;
                            v135 = wmma::__float_to_tf32(v134);
                            assert("Tensor range check" && 0 <= v131 && v131 < 4l);
                            v130[v131] = v135;
                            v131 += 1l ;
                        }
                        int4* v136;
                        v136 = reinterpret_cast<int4*>(v130 + 0l);
                        int4* v137;
                        v137 = reinterpret_cast<int4*>(v119 + v127);
                        assert("Pointer alignment check" && (unsigned long long)(v136) % 4l == 0 && (unsigned long long)(v137) % 4l == 0);
                        *v137 = *v136;
                        v123 += 1l ;
                    }
                    v121 += 1l ;
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v138[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v139[1l];
                int v140;
                v140 = 0l;
                #pragma unroll
                while (while_method_1(v140)){
                    int v142;
                    v142 = 0l;
                    #pragma unroll
                    while (while_method_1(v142)){
                        assert("Tensor range check" && 0 <= v140 && v140 < 1l);
                        assert("Tensor range check" && 0 <= v142 && v142 < 1l);
                        int v144;
                        v144 = v140 + v142;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v145 = v138[v144];
                        assert("Tensor range check" && 0 <= v140 && v140 < 1l);
                        int v146;
                        v146 = 192l * v140;
                        assert("Tensor range check" && 0 <= v142 && v142 < 1l);
                        int v147;
                        v147 = 8l * v142;
                        int v148;
                        v148 = v147 + v146;
                        int v149;
                        v149 = 0l;
                        #pragma unroll
                        while (while_method_2(v149)){
                            int v151;
                            v151 = 0l;
                            #pragma unroll
                            while (while_method_2(v151)){
                                assert("Tensor range check" && 0 <= v149 && v149 < 2l);
                                assert("Tensor range check" && 0 <= v151 && v151 < 2l);
                                int v153;
                                v153 = 96l * v151;
                                int v154;
                                v154 = v153 + v148;
                                int v155;
                                v155 = 4l * v149;
                                int v156;
                                v156 = v155 + v154;
                                float v157;
                                v157 = v36[v156];
                                bool v158;
                                v158 = 0l <= v151;
                                bool v160;
                                if (v158){
                                    bool v159;
                                    v159 = v151 < 2l;
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
                                bool v162;
                                v162 = 0l <= v149;
                                bool v164;
                                if (v162){
                                    bool v163;
                                    v163 = v149 < 2l;
                                    v164 = v163;
                                } else {
                                    v164 = false;
                                }
                                bool v165;
                                v165 = v164 == false;
                                if (v165){
                                    assert("The indices should be inside the range of the dimension." && v164);
                                } else {
                                }
                                int v166;
                                v166 = v149 * 2l;
                                int v167;
                                v167 = v151 + v166;
                                v145.x[v167] = v157;
                                v151 += 1l ;
                            }
                            v149 += 1l ;
                        }
                        v142 += 1l ;
                    }
                    v140 += 1l ;
                }
                int v168;
                v168 = 0l;
                #pragma unroll
                while (while_method_1(v168)){
                    int v170;
                    v170 = 0l;
                    #pragma unroll
                    while (while_method_1(v170)){
                        assert("Tensor range check" && 0 <= v168 && v168 < 1l);
                        assert("Tensor range check" && 0 <= v170 && v170 < 1l);
                        int v172;
                        v172 = v168 + v170;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v173 = v139[v172];
                        assert("Tensor range check" && 0 <= v168 && v168 < 1l);
                        int v174;
                        v174 = 192l * v168;
                        assert("Tensor range check" && 0 <= v170 && v170 < 1l);
                        int v175;
                        v175 = 8l * v170;
                        int v176;
                        v176 = v175 + v174;
                        int v177;
                        v177 = 0l;
                        #pragma unroll
                        while (while_method_2(v177)){
                            int v179;
                            v179 = 0l;
                            #pragma unroll
                            while (while_method_2(v179)){
                                assert("Tensor range check" && 0 <= v177 && v177 < 2l);
                                assert("Tensor range check" && 0 <= v179 && v179 < 2l);
                                int v181;
                                v181 = 4l * v179;
                                int v182;
                                v182 = v181 + v176;
                                int v183;
                                v183 = 96l * v177;
                                int v184;
                                v184 = v183 + v182;
                                float v185;
                                v185 = v49[v184];
                                bool v186;
                                v186 = 0l <= v179;
                                bool v188;
                                if (v186){
                                    bool v187;
                                    v187 = v179 < 2l;
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
                                bool v190;
                                v190 = 0l <= v177;
                                bool v192;
                                if (v190){
                                    bool v191;
                                    v191 = v177 < 2l;
                                    v192 = v191;
                                } else {
                                    v192 = false;
                                }
                                bool v193;
                                v193 = v192 == false;
                                if (v193){
                                    assert("The indices should be inside the range of the dimension." && v192);
                                } else {
                                }
                                int v194;
                                v194 = v177 * 2l;
                                int v195;
                                v195 = v179 + v194;
                                v173.x[v195] = v185;
                                v179 += 1l ;
                            }
                            v177 += 1l ;
                        }
                        v170 += 1l ;
                    }
                    v168 += 1l ;
                }
                __syncthreads();
                int v196;
                v196 = 0l;
                #pragma unroll
                while (while_method_1(v196)){
                    int v198;
                    v198 = 0l;
                    #pragma unroll
                    while (while_method_1(v198)){
                        int v200;
                        v200 = 0l;
                        #pragma unroll
                        while (while_method_1(v200)){
                            assert("Tensor range check" && 0 <= v196 && v196 < 1l);
                            assert("Tensor range check" && 0 <= v198 && v198 < 1l);
                            int v202;
                            v202 = v196 + v198;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v203 = v50[v202];
                            assert("Tensor range check" && 0 <= v196 && v196 < 1l);
                            assert("Tensor range check" && 0 <= v200 && v200 < 1l);
                            int v204;
                            v204 = v196 + v200;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v205 = v138[v204];
                            assert("Tensor range check" && 0 <= v198 && v198 < 1l);
                            assert("Tensor range check" && 0 <= v200 && v200 < 1l);
                            int v206;
                            v206 = v198 + v200;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v207 = v139[v206];
                            wmma::mma_sync(v203, v205, v207, v203);
                            v200 += 1l ;
                        }
                        v198 += 1l ;
                    }
                    v196 += 1l ;
                }
                v66 += 1l ;
            }
            int v208;
            v208 = 0l;
            #pragma unroll
            while (while_method_1(v208)){
                int v210;
                v210 = 0l;
                #pragma unroll
                while (while_method_1(v210)){
                    assert("Tensor range check" && 0 <= v208 && v208 < 1l);
                    assert("Tensor range check" && 0 <= v210 && v210 < 1l);
                    int v212;
                    v212 = v208 + v210;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v213 = v50[v212];
                    assert("Tensor range check" && 0 <= v208 && v208 < 1l);
                    assert("Tensor range check" && 0 <= v210 && v210 < 1l);
                    int v214;
                    v214 = 16l * v210;
                    int v215;
                    v215 = 384l * v208;
                    int v216;
                    v216 = v215 + v214;
                    float * v217;
                    v217 = v23+v216;
                    wmma::store_matrix_sync(v217, v213, 24l, wmma::mem_row_major);
                    v210 += 1l ;
                }
                v208 += 1l ;
            }
            __syncthreads();
            int v218;
            v218 = threadIdx.x;
            bool v219;
            v219 = 0l <= v218;
            bool v220;
            v220 = v219 == false;
            if (v220){
                assert("The index needs to be zero or positive." && v219);
            } else {
            }
            int v221;
            v221 = v218 % 4l;
            int v222;
            v222 = v218 / 4l;
            bool v223;
            v223 = v222 < 8l;
            bool v224;
            v224 = v223 == false;
            if (v224){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v223);
            } else {
            }
            assert("Tensor range check" && 0 <= v222 && v222 < 8l);
            assert("Tensor range check" && 0 <= v221 && v221 < 4l);
            int v225;
            v225 = 4l * v221;
            int v226;
            v226 = 16l * v222;
            int v227;
            v227 = v226 + v225;
            int v228;
            v228 = 24l * v222;
            int v229;
            v229 = v228 + v225;
            float * v230;
            v230 = v59+v227;
            float * v231;
            v231 = v12+v229;
            int v232;
            v232 = 0l;
            #pragma unroll
            while (while_method_2(v232)){
                int v234;
                v234 = 0l;
                #pragma unroll
                while (while_method_1(v234)){
                    assert("Tensor range check" && 0 <= v232 && v232 < 2l);
                    assert("Tensor range check" && 0 <= v234 && v234 < 1l);
                    int v236;
                    v236 = 16l * v234;
                    int v237;
                    v237 = 128l * v232;
                    int v238;
                    v238 = v237 + v236;
                    int v239;
                    v239 = 192l * v232;
                    int v240;
                    v240 = v239 + v236;
                    int4* v241;
                    v241 = reinterpret_cast<int4*>(v231 + v240);
                    int4* v242;
                    v242 = reinterpret_cast<int4*>(v230 + v238);
                    assert("Pointer alignment check" && (unsigned long long)(v241) % 4l == 0 && (unsigned long long)(v242) % 4l == 0);
                    *v242 = *v241;
                    v234 += 1l ;
                }
                v232 += 1l ;
            }
            __syncthreads();
            // Poping the loop unrolling to: 0
            v53 += 1l ;
        }
        v51 += 1l ;
    }
    return ;
}
__device__ void method_6(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[2560ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v5;
    v5 = 256l * v3;
    float * v6;
    v6 = reinterpret_cast<float *>(&v1[2048ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 4l);
    int v7;
    v7 = 256l * v2;
    float * v8;
    v8 = reinterpret_cast<float *>(&v0[3584ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    return method_7(v8, v5, v6, v7, v4);
}
__device__ void method_8(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[3584ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v5;
    v5 = 256l * v3;
    float * v6;
    v6 = reinterpret_cast<float *>(&v0[4608ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    return method_3(v6, v5, v4);
}
__device__ void method_9(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[4608ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v5;
    v5 = 256l * v3;
    float * v6;
    v6 = reinterpret_cast<float *>(&v0[5632ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    return method_5(v6, v5, v4);
}
__device__ void method_10(unsigned char * v0, unsigned char * v1, int v2, int v3){
    float * v4;
    v4 = reinterpret_cast<float *>(&v0[5632ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v5;
    v5 = 256l * v3;
    float * v6;
    v6 = reinterpret_cast<float *>(&v1[6144ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 4l);
    int v7;
    v7 = 256l * v2;
    float * v8;
    v8 = reinterpret_cast<float *>(&v0[6656ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    return method_7(v8, v5, v6, v7, v4);
}
__device__ void method_12(curandStatePhilox4_32_10_t & v0, int * v1, int v2, float * v3, int v4){
    int v5;
    v5 = threadIdx.x;
    bool v6;
    v6 = 0l <= v5;
    bool v7;
    v7 = v6 == false;
    if (v7){
        assert("The index needs to be zero or positive." && v6);
    } else {
    }
    int v8;
    v8 = v5 % 4l;
    int v9;
    v9 = v5 / 4l;
    bool v10;
    v10 = v9 < 8l;
    bool v11;
    v11 = v10 == false;
    if (v11){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v10);
    } else {
    }
    assert("Tensor range check" && 0 <= v9 && v9 < 8l);
    assert("Tensor range check" && 0 <= v8 && v8 < 4l);
    int v12;
    v12 = 4l * v8;
    int v13;
    v13 = v12 + v4;
    int v14;
    v14 = 16l * v9;
    int v15;
    v15 = v14 + v13;
    assert("Tensor range check" && 0 <= v9 && v9 < 8l);
    int v16;
    v16 = v9 + v2;
    int v17;
    v17 = 0l;
    while (while_method_2(v17)){
        assert("Tensor range check" && 0 <= v17 && v17 < 2l);
        int v19;
        v19 = 128l * v17;
        int v20;
        v20 = v19 + v15;
        float v21[4l];
        int v22[4l];
        int v23;
        v23 = 0l;
        while (while_method_1(v23)){
            assert("Tensor range check" && 0 <= v23 && v23 < 1l);
            int v25;
            v25 = 4l * v23;
            assert("Tensor range check" && 0 <= v23 && v23 < 1l);
            int v26;
            v26 = 16l * v23;
            int v27;
            v27 = v26 + v20;
            int4* v28;
            v28 = reinterpret_cast<int4*>(v3 + v27);
            int4* v29;
            v29 = reinterpret_cast<int4*>(v21 + v25);
            assert("Pointer alignment check" && (unsigned long long)(v28) % 4l == 0 && (unsigned long long)(v29) % 4l == 0);
            *v29 = *v28;
            v23 += 1l ;
        }
        int v30;
        v30 = 0l;
        while (while_method_1(v30)){
            int v32;
            v32 = 0l;
            while (while_method_0(v32)){
                bool v34;
                v34 = 0l <= v32;
                bool v36;
                if (v34){
                    bool v35;
                    v35 = v32 < 4l;
                    v36 = v35;
                } else {
                    v36 = false;
                }
                bool v37;
                v37 = v36 == false;
                if (v37){
                    assert("The indices should be inside the range of the dimension." && v36);
                } else {
                }
                bool v38;
                v38 = 0l <= v8;
                bool v40;
                if (v38){
                    bool v39;
                    v39 = v8 < 4l;
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
                int v42;
                v42 = v8 * 4l;
                int v43;
                v43 = v32 + v42;
                bool v44;
                v44 = 0l <= v30;
                bool v46;
                if (v44){
                    bool v45;
                    v45 = v30 < 1l;
                    v46 = v45;
                } else {
                    v46 = false;
                }
                bool v47;
                v47 = v46 == false;
                if (v47){
                    assert("The indices should be inside the range of the dimension." && v46);
                } else {
                }
                int v48;
                v48 = v30 * 16l;
                int v49;
                v49 = v43 + v48;
                assert("Tensor range check" && 0 <= v30 && v30 < 1l);
                assert("Tensor range check" && 0 <= v32 && v32 < 4l);
                int v50;
                v50 = 4l * v30;
                int v51;
                v51 = v50 + v32;
                v22[v51] = v49;
                v32 += 1l ;
            }
            v30 += 1l ;
        }
        bool v52;
        v52 = 0l <= v9;
        bool v53;
        v53 = v52 && v10;
        bool v54;
        v54 = v53 == false;
        if (v54){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v53);
        } else {
        }
        bool v55;
        v55 = 0l <= v17;
        bool v57;
        if (v55){
            bool v56;
            v56 = v17 < 2l;
            v57 = v56;
        } else {
            v57 = false;
        }
        bool v58;
        v58 = v57 == false;
        if (v58){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v57);
        } else {
        }
        int v59;
        v59 = v17 * 8l;
        int v60;
        v60 = v59 + v9;
        float v61;
        v61 = 0.0f;
        int v62;
        v62 = 0l;
        while (while_method_1(v62)){
            int v64;
            v64 = 0l;
            while (while_method_0(v64)){
                assert("Tensor range check" && 0 <= v62 && v62 < 1l);
                assert("Tensor range check" && 0 <= v64 && v64 < 4l);
                int v66;
                v66 = 4l * v62;
                int v67;
                v67 = v66 + v64;
                float v68;
                v68 = v21[v67];
                float v69;
                v69 = v61 + v68;
                v61 = v69;
                v64 += 1l ;
            }
            v62 += 1l ;
        }
        auto v70 = cooperative_groups::coalesced_threads();
        int v71;
        v71 = threadIdx.x;
        int v72;
        v72 = v71 / 4l;
        auto v73 = cooperative_groups::labeled_partition(v70,v72);
        Closure0 v74{};
        float v75;
        v75 = cooperative_groups::reduce(v73, v61, v74);
        float v76;
        v76 = v75 / 16.0f;
        float v77[4l];
        int v78;
        v78 = 0l;
        while (while_method_1(v78)){
            int v80;
            v80 = 0l;
            while (while_method_0(v80)){
                assert("Tensor range check" && 0 <= v78 && v78 < 1l);
                assert("Tensor range check" && 0 <= v80 && v80 < 4l);
                int v82;
                v82 = 4l * v78;
                int v83;
                v83 = v82 + v80;
                float v84;
                v84 = v21[v83];
                float v85;
                v85 = v84 - v76;
                float v86;
                v86 = exp(v85);
                assert("Tensor range check" && 0 <= v78 && v78 < 1l);
                assert("Tensor range check" && 0 <= v80 && v80 < 4l);
                v77[v83] = v86;
                v80 += 1l ;
            }
            v78 += 1l ;
        }
        float v87;
        v87 = 0.0f;
        int v88;
        v88 = 0l;
        while (while_method_1(v88)){
            int v90;
            v90 = 0l;
            while (while_method_0(v90)){
                assert("Tensor range check" && 0 <= v88 && v88 < 1l);
                assert("Tensor range check" && 0 <= v90 && v90 < 4l);
                int v92;
                v92 = 4l * v88;
                int v93;
                v93 = v92 + v90;
                float v94;
                v94 = v77[v93];
                float v95;
                v95 = v87 + v94;
                v87 = v95;
                v90 += 1l ;
            }
            v88 += 1l ;
        }
        auto v96 = cooperative_groups::coalesced_threads();
        int v97;
        v97 = threadIdx.x;
        int v98;
        v98 = v97 / 4l;
        auto v99 = cooperative_groups::labeled_partition(v96,v98);
        float v100;
        v100 = cooperative_groups::reduce(v99, v87, v74);
        float v101[4l];
        int v102;
        v102 = 0l;
        while (while_method_1(v102)){
            int v104;
            v104 = 0l;
            while (while_method_0(v104)){
                assert("Tensor range check" && 0 <= v102 && v102 < 1l);
                assert("Tensor range check" && 0 <= v104 && v104 < 4l);
                int v106;
                v106 = 4l * v102;
                int v107;
                v107 = v106 + v104;
                float v108;
                v108 = v77[v107];
                bool v109;
                v109 = v100 == 0.0f;
                bool v110;
                v110 = v109 != true;
                float v112;
                if (v110){
                    float v111;
                    v111 = v108 / v100;
                    v112 = v111;
                } else {
                    v112 = 0.0625f;
                }
                assert("Tensor range check" && 0 <= v102 && v102 < 1l);
                assert("Tensor range check" && 0 <= v104 && v104 < 4l);
                v101[v107] = v112;
                v104 += 1l ;
            }
            v102 += 1l ;
        }
        float v113[4l];
        float v114;
        v114 = 0.0f;
        int v115;
        v115 = 0l;
        while (while_method_1(v115)){
            assert("Tensor range check" && 0 <= v115 && v115 < 1l);
            int v117;
            v117 = 4l * v115;
            assert("Tensor range check" && 0 <= v115 && v115 < 1l);
            int v118; float v119;
            Tuple0 tmp0 = Tuple0{0l, 0.0f};
            v118 = tmp0.v0; v119 = tmp0.v1;
            while (while_method_0(v118)){
                assert("Tensor range check" && 0 <= v118 && v118 < 4l);
                int v121;
                v121 = v118 + v117;
                float v122;
                v122 = v101[v121];
                float v123;
                v123 = v119 + v122;
                v119 = v123;
                v118 += 1l ;
            }
            auto v124 = cooperative_groups::coalesced_threads();
            int v125;
            v125 = threadIdx.x;
            int v126;
            v126 = v125 / 4l;
            auto v127 = cooperative_groups::labeled_partition(v124,v126);
            Closure1 v128{};
            float v129;
            v129 = cooperative_groups::inclusive_scan(v127, v119, v128);
            float v130;
            v130 = v127.shfl_up(v129,1);
            bool v131;
            v131 = v127.thread_rank() == 0;
            float v132;
            if (v131){
                v132 = 0.0f;
            } else {
                v132 = v130;
            }
            float v133;
            v133 = v127.shfl(v129,v127.num_threads()-1);
            float v134;
            v134 = v114 + v132;
            int v135; float v136;
            Tuple0 tmp1 = Tuple0{0l, v134};
            v135 = tmp1.v0; v136 = tmp1.v1;
            while (while_method_0(v135)){
                assert("Tensor range check" && 0 <= v135 && v135 < 4l);
                int v138;
                v138 = v135 + v117;
                float v139;
                v139 = v101[v138];
                float v140;
                v140 = v136 + v139;
                assert("Tensor range check" && 0 <= v135 && v135 < 4l);
                v113[v138] = v140;
                v136 = v140;
                v135 += 1l ;
            }
            float v141;
            v141 = v114 + v133;
            v114 = v141;
            v115 += 1l ;
        }
        float v142;
        v142 = curand_uniform(&v0);
        float v143[4l];
        int v144;
        v144 = 0l;
        while (while_method_1(v144)){
            int v146;
            v146 = 0l;
            while (while_method_0(v146)){
                assert("Tensor range check" && 0 <= v144 && v144 < 1l);
                assert("Tensor range check" && 0 <= v146 && v146 < 4l);
                int v148;
                v148 = 4l * v144;
                int v149;
                v149 = v148 + v146;
                float v150;
                v150 = v113[v149];
                float v151;
                v151 = v150 - v142;
                assert("Tensor range check" && 0 <= v144 && v144 < 1l);
                assert("Tensor range check" && 0 <= v146 && v146 < 4l);
                v143[v149] = v151;
                v146 += 1l ;
            }
            v144 += 1l ;
        }
        float v152; int v153;
        Tuple1 tmp2 = Tuple1{-1.0f / 0.0f, 0l};
        v152 = tmp2.v0; v153 = tmp2.v1;
        int v154;
        v154 = 0l;
        while (while_method_1(v154)){
            int v156;
            v156 = 0l;
            while (while_method_0(v156)){
                assert("Tensor range check" && 0 <= v154 && v154 < 1l);
                assert("Tensor range check" && 0 <= v156 && v156 < 4l);
                int v158;
                v158 = 4l * v154;
                int v159;
                v159 = v158 + v156;
                float v160;
                v160 = v143[v159];
                int v161;
                v161 = v22[v159];
                bool v162;
                v162 = v152 >= 0.0f;
                bool v164;
                if (v162){
                    bool v163;
                    v163 = v160 >= 0.0f;
                    v164 = v163;
                } else {
                    v164 = false;
                }
                float v173; int v174;
                if (v164){
                    bool v165;
                    v165 = v152 <= v160;
                    if (v165){
                        v173 = v152; v174 = v153;
                    } else {
                        v173 = v160; v174 = v161;
                    }
                } else {
                    if (v162){
                        v173 = v152; v174 = v153;
                    } else {
                        bool v168;
                        v168 = v160 >= 0.0f;
                        if (v168){
                            v173 = v160; v174 = v161;
                        } else {
                            v173 = v152; v174 = v153;
                        }
                    }
                }
                v152 = v173;
                v153 = v174;
                v156 += 1l ;
            }
            v154 += 1l ;
        }
        auto v175 = cooperative_groups::coalesced_threads();
        int v176;
        v176 = threadIdx.x;
        int v177;
        v177 = v176 / 4l;
        auto v178 = cooperative_groups::labeled_partition(v175,v177);
        Closure2 v179{};
        float v180; int v181;
        Tuple1 tmp3 = cooperative_groups::reduce(v178, Tuple1{v152, v153}, v179);
        v180 = tmp3.v0; v181 = tmp3.v1;
        assert("Tensor range check" && 0 <= v17 && v17 < 2l);
        int v182;
        v182 = 8l * v17;
        int v183;
        v183 = v182 + v16;
        v1[v183] = v181;
        v17 += 1l ;
    }
    __syncthreads();
    return ;
}
__device__ void method_11(unsigned char * v0, unsigned char * v1, int v2, int v3, curandStatePhilox4_32_10_t & v4){
    float * v5;
    v5 = reinterpret_cast<float *>(&v0[6656ull]);
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v6;
    v6 = 256l * v3;
    int * v7;
    v7 = reinterpret_cast<int *>(&v0[7680ull]);
    assert("Tensor range check" && 0 <= v2 && v2 < 4l);
    int v8;
    v8 = 16l * v2;
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v9;
    v9 = 16l * v3;
    int v10;
    v10 = v9 + v8;
    return method_12(v4, v7, v10, v5, v6);
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
        int v11;
        v11 = blockIdx.x;
        int v12;
        v12 = v11;
        while (while_method_1(v12)){
            bool v14;
            v14 = 0l <= v12;
            bool v15;
            v15 = v14 == false;
            if (v15){
                assert("The index needs to be zero or positive." && v14);
            } else {
            }
            bool v16;
            v16 = v12 < 1l;
            bool v17;
            v17 = v16 == false;
            if (v17){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v16);
            } else {
            }
            method_0(v0, v1, v9, v12);
            method_2(v0, v1, v9, v12);
            method_4(v0, v1, v9, v12);
            method_6(v0, v1, v9, v12);
            method_8(v0, v1, v9, v12);
            method_9(v0, v1, v9, v12);
            method_10(v0, v1, v9, v12);
            method_11(v0, v1, v9, v12, v8);
            v12 += 1l ;
        }
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
def method7(v0 : i32) -> None:
    print(v0, end="")
    del v0
    return 
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
            method7(v30)
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
    v1 = cp.empty(7936,dtype=cp.uint8)
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
    v44((1,),(32,),(v1, v0),shared_mem=1536)
    del v0, v44
    v45 = 0
    while method5(v45):
        v47 = v1[7680:7680+4*64].view(cp.int32)
        assert 0 <= v45 < 4, 'Tensor range check'
        v48 = 16 * v45
        v49 = 1
        v50 = 1
        v51 = 16
        v52 = 1
        method6(v47, v48, v49, v50, v51, v52)
        del v47, v48, v49, v50, v51, v52
        print()
        v53 = "==="
        method0(v53)
        del v53
        print()
        v45 += 1 
    del v1, v45
    return 

if __name__ == '__main__': print(main())
