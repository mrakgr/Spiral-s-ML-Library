kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
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

struct Tuple0;
struct Tuple1;
struct Tuple2;
struct Tuple0 {
    int v0;
    float v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Closure0 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v1 + v0;
        return v2;
    }
};
struct Tuple1 {
    float v0;
    int v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure1 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v2 > v0;
        if (v4){
            return Tuple1{v2, v3};
        } else {
            return Tuple1{v0, v1};
        }
    }
};
struct Closure2 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v1 + v0;
        return v2;
    }
};
struct Tuple2 {
    int v0;
    int v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(int t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v1 + v0;
        return v2;
    }
};
struct Closure4 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v1 + v0;
        return v2;
    }
};
struct Closure5 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        return v0;
    }
};
struct Closure6 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v2 >= 0.0f;
        bool v6;
        if (v4){
            bool v5;
            v5 = v0 >= 0.0f;
            v6 = v5;
        } else {
            v6 = false;
        }
        if (v6){
            bool v7;
            v7 = v2 <= v0;
            if (v7){
                return Tuple1{v2, v3};
            } else {
                return Tuple1{v0, v1};
            }
        } else {
            if (v4){
                return Tuple1{v2, v3};
            } else {
                bool v10;
                v10 = v0 >= 0.0f;
                if (v10){
                    return Tuple1{v0, v1};
                } else {
                    return Tuple1{v2, v3};
                }
            }
        }
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 1024l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 32l;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
extern "C" __global__ void entry0(int * v0, float * v1, float * v2, int * v3, float * v4, float * v5, float * v6, float * v7, float * v8, float * v9, int * v10, int * v11, int * v12, int * v13, int * v14, int * v15) {
    unsigned long long v16;
    v16 = clock64();
    curandStatePhilox4_32_10_t v17;
    curand_init(v16,0ull,0ull,&v17);
    int v18;
    v18 = threadIdx.x;
    int v19;
    v19 = v18;
    while (while_method_0(v19)){
        bool v21;
        v21 = 0l <= v19;
        bool v22;
        v22 = v21 == false;
        if (v22){
            assert("The index needs to be zero or positive." && v21);
        } else {
        }
        int v24;
        v24 = v19 % 4l;
        int v25;
        v25 = v19 / 4l;
        bool v26;
        v26 = v25 < 256l;
        bool v27;
        v27 = v26 == false;
        if (v27){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v26);
        } else {
        }
        assert("Tensor range check" && 0 <= v25 && v25 < 256l);
        assert("Tensor range check" && 0 <= v24 && v24 < 4l);
        int v29;
        v29 = 4l * v24;
        int v30;
        v30 = 16l * v25;
        int v31;
        v31 = v30 + v29;
        assert("Tensor range check" && 0 <= v25 && v25 < 256l);
        assert("Tensor range check" && 0 <= v24 && v24 < 4l);
        float v32[4l];
        float v33[4l];
        int4* v34;
        v34 = reinterpret_cast<int4*>(v1 + v31);
        int4* v35;
        v35 = reinterpret_cast<int4*>(v32 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v34) % 4l == 0 && (unsigned long long)(v35) % 4l == 0);
        *v35 = *v34;
        // Pushing the loop unrolling to: 0
        int v36;
        v36 = 0l;
        #pragma unroll
        while (while_method_1(v36)){
            assert("Tensor range check" && 0 <= v36 && v36 < 4l);
            float v38;
            v38 = v32[v36];
            float v39;
            v39 = 1.0f + v38;
            assert("Tensor range check" && 0 <= v36 && v36 < 4l);
            v33[v36] = v39;
            v36 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v40;
        v40 = reinterpret_cast<int4*>(v33 + 0l);
        int4* v41;
        v41 = reinterpret_cast<int4*>(v1 + v31);
        assert("Pointer alignment check" && (unsigned long long)(v40) % 4l == 0 && (unsigned long long)(v41) % 4l == 0);
        *v41 = *v40;
        v19 += 32l ;
    }
    __syncthreads();
    float v42;
    v42 = 0.0f;
    int v43;
    v43 = threadIdx.x;
    int v44;
    v44 = v43;
    while (while_method_0(v44)){
        bool v46;
        v46 = 0l <= v44;
        bool v47;
        v47 = v46 == false;
        if (v47){
            assert("The index needs to be zero or positive." && v46);
        } else {
        }
        int v49;
        v49 = v44 % 4l;
        int v50;
        v50 = v44 / 4l;
        bool v51;
        v51 = v50 < 256l;
        bool v52;
        v52 = v51 == false;
        if (v52){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v51);
        } else {
        }
        assert("Tensor range check" && 0 <= v50 && v50 < 256l);
        assert("Tensor range check" && 0 <= v49 && v49 < 4l);
        int v54;
        v54 = 4l * v49;
        int v55;
        v55 = 16l * v50;
        int v56;
        v56 = v55 + v54;
        float v57[4l];
        int4* v58;
        v58 = reinterpret_cast<int4*>(v1 + v56);
        int4* v59;
        v59 = reinterpret_cast<int4*>(v57 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v58) % 4l == 0 && (unsigned long long)(v59) % 4l == 0);
        *v59 = *v58;
        int v60; float v61;
        Tuple0 tmp0 = Tuple0{0l, v42};
        v60 = tmp0.v0; v61 = tmp0.v1;
        while (while_method_1(v60)){
            assert("Tensor range check" && 0 <= v60 && v60 < 4l);
            float v63;
            v63 = v57[v60];
            float v64;
            v64 = v61 + v63;
            v61 = v64;
            v60 += 1l ;
        }
        v42 = v61;
        v44 += 32l ;
    }
    auto v65 = cooperative_groups::coalesced_threads();
    Closure0 v66{};
    float v67;
    v67 = cooperative_groups::reduce(v65, v42, v66);
    int v68;
    v68 = threadIdx.x;
    int v69;
    v69 = v68 / 32l;
    __shared__ float v70[1l];
    assert("Tensor range check" && 0 <= v69 && v69 < 1l);
    v70[v69] = v67;
    __syncthreads();
    int v71;
    v71 = threadIdx.x;
    int v72;
    v72 = v71 % 32l;
    bool v73;
    v73 = v69 == 0l;
    bool v75;
    if (v73){
        bool v74;
        v74 = v72 < 1l;
        v75 = v74;
    } else {
        v75 = false;
    }
    if (v75){
        auto v76 = cooperative_groups::coalesced_threads();
        assert("Tensor range check" && 0 <= v72 && v72 < 1l);
        float v77;
        v77 = v70[v72];
        float v78;
        v78 = cooperative_groups::reduce(v76, v77, v66);
        v2[0l] = v78;
    } else {
    }
    __syncthreads();
    int v79;
    v79 = threadIdx.x;
    bool v80;
    v80 = 0l <= v79;
    bool v81;
    v81 = v80 == false;
    if (v81){
        assert("The index needs to be zero or positive." && v80);
    } else {
    }
    int v83;
    v83 = v79 % 4l;
    int v84;
    v84 = v79 / 4l;
    bool v85;
    v85 = v84 < 8l;
    bool v86;
    v86 = v85 == false;
    if (v86){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v85);
    } else {
    }
    assert("Tensor range check" && 0 <= v84 && v84 < 8l);
    assert("Tensor range check" && 0 <= v83 && v83 < 4l);
    int v88;
    v88 = 4l * v83;
    int v89;
    v89 = 16l * v84;
    int v90;
    v90 = v89 + v88;
    assert("Tensor range check" && 0 <= v84 && v84 < 8l);
    assert("Tensor range check" && 0 <= v83 && v83 < 4l);
    int v91;
    v91 = 0l;
    while (while_method_2(v91)){
        assert("Tensor range check" && 0 <= v91 && v91 < 32l);
        int v93;
        v93 = 128l * v91;
        int v94;
        v94 = v93 + v90;
        int v95[4l];
        int v96[4l];
        int v97;
        v97 = 0l;
        while (while_method_3(v97)){
            assert("Tensor range check" && 0 <= v97 && v97 < 1l);
            int v99;
            v99 = 4l * v97;
            assert("Tensor range check" && 0 <= v97 && v97 < 1l);
            int v100;
            v100 = 16l * v97;
            int v101;
            v101 = v100 + v94;
            int4* v102;
            v102 = reinterpret_cast<int4*>(v0 + v101);
            int4* v103;
            v103 = reinterpret_cast<int4*>(v95 + v99);
            assert("Pointer alignment check" && (unsigned long long)(v102) % 4l == 0 && (unsigned long long)(v103) % 4l == 0);
            *v103 = *v102;
            v97 += 1l ;
        }
        int v104;
        v104 = 0l;
        while (while_method_3(v104)){
            int v106;
            v106 = 0l;
            while (while_method_1(v106)){
                bool v108;
                v108 = 0l <= v106;
                bool v110;
                if (v108){
                    bool v109;
                    v109 = v106 < 4l;
                    v110 = v109;
                } else {
                    v110 = false;
                }
                bool v111;
                v111 = v110 == false;
                if (v111){
                    assert("The indices should be inside the range of the dimension." && v110);
                } else {
                }
                bool v113;
                v113 = 0l <= v83;
                bool v115;
                if (v113){
                    bool v114;
                    v114 = v83 < 4l;
                    v115 = v114;
                } else {
                    v115 = false;
                }
                bool v116;
                v116 = v115 == false;
                if (v116){
                    assert("The indices should be inside the range of the dimension." && v115);
                } else {
                }
                int v118;
                v118 = v83 * 4l;
                int v119;
                v119 = v106 + v118;
                bool v120;
                v120 = 0l <= v104;
                bool v122;
                if (v120){
                    bool v121;
                    v121 = v104 < 1l;
                    v122 = v121;
                } else {
                    v122 = false;
                }
                bool v123;
                v123 = v122 == false;
                if (v123){
                    assert("The indices should be inside the range of the dimension." && v122);
                } else {
                }
                int v125;
                v125 = v104 * 16l;
                int v126;
                v126 = v119 + v125;
                assert("Tensor range check" && 0 <= v104 && v104 < 1l);
                assert("Tensor range check" && 0 <= v106 && v106 < 4l);
                int v127;
                v127 = 4l * v104;
                int v128;
                v128 = v127 + v106;
                v96[v128] = v126;
                v106 += 1l ;
            }
            v104 += 1l ;
        }
        bool v129;
        v129 = 0l <= v84;
        bool v130;
        v130 = v129 && v85;
        bool v131;
        v131 = v130 == false;
        if (v131){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v130);
        } else {
        }
        bool v133;
        v133 = 0l <= v91;
        bool v135;
        if (v133){
            bool v134;
            v134 = v91 < 32l;
            v135 = v134;
        } else {
            v135 = false;
        }
        bool v136;
        v136 = v135 == false;
        if (v136){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v135);
        } else {
        }
        int v138;
        v138 = v91 * 8l;
        int v139;
        v139 = v138 + v84;
        assert("Tensor range check" && 0 <= v91 && v91 < 32l);
        int v140;
        v140 = 0l;
        while (while_method_3(v140)){
            assert("Tensor range check" && 0 <= v140 && v140 < 1l);
            int v142;
            v142 = 16l * v140;
            int v143;
            v143 = v142 + v94;
            assert("Tensor range check" && 0 <= v140 && v140 < 1l);
            int v144;
            v144 = 4l * v140;
            int4* v145;
            v145 = reinterpret_cast<int4*>(v95 + v144);
            int4* v146;
            v146 = reinterpret_cast<int4*>(v3 + v143);
            assert("Pointer alignment check" && (unsigned long long)(v145) % 4l == 0 && (unsigned long long)(v146) % 4l == 0);
            *v146 = *v145;
            v140 += 1l ;
        }
        v91 += 1l ;
    }
    __syncthreads();
    int v147;
    v147 = threadIdx.x;
    bool v148;
    v148 = 0l <= v147;
    bool v149;
    v149 = v148 == false;
    if (v149){
        assert("The index needs to be zero or positive." && v148);
    } else {
    }
    int v151;
    v151 = v147 % 4l;
    int v152;
    v152 = v147 / 4l;
    bool v153;
    v153 = v152 < 8l;
    bool v154;
    v154 = v153 == false;
    if (v154){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v153);
    } else {
    }
    assert("Tensor range check" && 0 <= v152 && v152 < 8l);
    assert("Tensor range check" && 0 <= v151 && v151 < 4l);
    int v156;
    v156 = 4l * v151;
    int v157;
    v157 = 16l * v152;
    int v158;
    v158 = v157 + v156;
    assert("Tensor range check" && 0 <= v152 && v152 < 8l);
    assert("Tensor range check" && 0 <= v151 && v151 < 4l);
    int v159;
    v159 = 0l;
    while (while_method_2(v159)){
        assert("Tensor range check" && 0 <= v159 && v159 < 32l);
        int v161;
        v161 = 128l * v159;
        int v162;
        v162 = v161 + v158;
        float v163[4l];
        int v164[4l];
        int v165;
        v165 = 0l;
        while (while_method_3(v165)){
            assert("Tensor range check" && 0 <= v165 && v165 < 1l);
            int v167;
            v167 = 4l * v165;
            assert("Tensor range check" && 0 <= v165 && v165 < 1l);
            int v168;
            v168 = 16l * v165;
            int v169;
            v169 = v168 + v162;
            int4* v170;
            v170 = reinterpret_cast<int4*>(v1 + v169);
            int4* v171;
            v171 = reinterpret_cast<int4*>(v163 + v167);
            assert("Pointer alignment check" && (unsigned long long)(v170) % 4l == 0 && (unsigned long long)(v171) % 4l == 0);
            *v171 = *v170;
            v165 += 1l ;
        }
        int v172;
        v172 = 0l;
        while (while_method_3(v172)){
            int v174;
            v174 = 0l;
            while (while_method_1(v174)){
                bool v176;
                v176 = 0l <= v174;
                bool v178;
                if (v176){
                    bool v177;
                    v177 = v174 < 4l;
                    v178 = v177;
                } else {
                    v178 = false;
                }
                bool v179;
                v179 = v178 == false;
                if (v179){
                    assert("The indices should be inside the range of the dimension." && v178);
                } else {
                }
                bool v181;
                v181 = 0l <= v151;
                bool v183;
                if (v181){
                    bool v182;
                    v182 = v151 < 4l;
                    v183 = v182;
                } else {
                    v183 = false;
                }
                bool v184;
                v184 = v183 == false;
                if (v184){
                    assert("The indices should be inside the range of the dimension." && v183);
                } else {
                }
                int v186;
                v186 = v151 * 4l;
                int v187;
                v187 = v174 + v186;
                bool v188;
                v188 = 0l <= v172;
                bool v190;
                if (v188){
                    bool v189;
                    v189 = v172 < 1l;
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
                v193 = v172 * 16l;
                int v194;
                v194 = v187 + v193;
                assert("Tensor range check" && 0 <= v172 && v172 < 1l);
                assert("Tensor range check" && 0 <= v174 && v174 < 4l);
                int v195;
                v195 = 4l * v172;
                int v196;
                v196 = v195 + v174;
                v164[v196] = v194;
                v174 += 1l ;
            }
            v172 += 1l ;
        }
        bool v197;
        v197 = 0l <= v152;
        bool v198;
        v198 = v197 && v153;
        bool v199;
        v199 = v198 == false;
        if (v199){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v198);
        } else {
        }
        bool v201;
        v201 = 0l <= v159;
        bool v203;
        if (v201){
            bool v202;
            v202 = v159 < 32l;
            v203 = v202;
        } else {
            v203 = false;
        }
        bool v204;
        v204 = v203 == false;
        if (v204){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v203);
        } else {
        }
        int v206;
        v206 = v159 * 8l;
        int v207;
        v207 = v206 + v152;
        int v208[4l];
        int v209[4l];
        int v210;
        v210 = 0l;
        while (while_method_3(v210)){
            int v212;
            v212 = 0l;
            while (while_method_1(v212)){
                assert("Tensor range check" && 0 <= v210 && v210 < 1l);
                assert("Tensor range check" && 0 <= v212 && v212 < 4l);
                int v214;
                v214 = 4l * v210;
                int v215;
                v215 = v214 + v212;
                int v216;
                v216 = v164[v215];
                assert("Tensor range check" && 0 <= v210 && v210 < 1l);
                assert("Tensor range check" && 0 <= v212 && v212 < 4l);
                v208[v215] = v207;
                v209[v215] = v216;
                v212 += 1l ;
            }
            v210 += 1l ;
        }
        assert("Tensor range check" && 0 <= v159 && v159 < 32l);
        int v217;
        v217 = 0l;
        while (while_method_3(v217)){
            assert("Tensor range check" && 0 <= v217 && v217 < 1l);
            int v219;
            v219 = 16l * v217;
            int v220;
            v220 = v219 + v162;
            assert("Tensor range check" && 0 <= v217 && v217 < 1l);
            int v221;
            v221 = 4l * v217;
            int4* v222;
            v222 = reinterpret_cast<int4*>(v208 + v221);
            int4* v223;
            v223 = reinterpret_cast<int4*>(v12 + v220);
            assert("Pointer alignment check" && (unsigned long long)(v222) % 4l == 0 && (unsigned long long)(v223) % 4l == 0);
            *v223 = *v222;
            int4* v224;
            v224 = reinterpret_cast<int4*>(v209 + v221);
            int4* v225;
            v225 = reinterpret_cast<int4*>(v13 + v220);
            assert("Pointer alignment check" && (unsigned long long)(v224) % 4l == 0 && (unsigned long long)(v225) % 4l == 0);
            *v225 = *v224;
            v217 += 1l ;
        }
        v159 += 1l ;
    }
    __syncthreads();
    int v226;
    v226 = threadIdx.x;
    bool v227;
    v227 = 0l <= v226;
    bool v228;
    v228 = v227 == false;
    if (v228){
        assert("The index needs to be zero or positive." && v227);
    } else {
    }
    int v230;
    v230 = v226 % 4l;
    int v231;
    v231 = v226 / 4l;
    bool v232;
    v232 = v231 < 8l;
    bool v233;
    v233 = v232 == false;
    if (v233){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v232);
    } else {
    }
    assert("Tensor range check" && 0 <= v231 && v231 < 8l);
    assert("Tensor range check" && 0 <= v230 && v230 < 4l);
    int v235;
    v235 = 4l * v230;
    int v236;
    v236 = 16l * v231;
    int v237;
    v237 = v236 + v235;
    assert("Tensor range check" && 0 <= v231 && v231 < 8l);
    int v238;
    v238 = 0l;
    while (while_method_2(v238)){
        assert("Tensor range check" && 0 <= v238 && v238 < 32l);
        int v240;
        v240 = 128l * v238;
        int v241;
        v241 = v240 + v237;
        float v242[4l];
        int v243[4l];
        int v244;
        v244 = 0l;
        while (while_method_3(v244)){
            assert("Tensor range check" && 0 <= v244 && v244 < 1l);
            int v246;
            v246 = 4l * v244;
            assert("Tensor range check" && 0 <= v244 && v244 < 1l);
            int v247;
            v247 = 16l * v244;
            int v248;
            v248 = v247 + v241;
            int4* v249;
            v249 = reinterpret_cast<int4*>(v1 + v248);
            int4* v250;
            v250 = reinterpret_cast<int4*>(v242 + v246);
            assert("Pointer alignment check" && (unsigned long long)(v249) % 4l == 0 && (unsigned long long)(v250) % 4l == 0);
            *v250 = *v249;
            v244 += 1l ;
        }
        int v251;
        v251 = 0l;
        while (while_method_3(v251)){
            int v253;
            v253 = 0l;
            while (while_method_1(v253)){
                bool v255;
                v255 = 0l <= v253;
                bool v257;
                if (v255){
                    bool v256;
                    v256 = v253 < 4l;
                    v257 = v256;
                } else {
                    v257 = false;
                }
                bool v258;
                v258 = v257 == false;
                if (v258){
                    assert("The indices should be inside the range of the dimension." && v257);
                } else {
                }
                bool v260;
                v260 = 0l <= v230;
                bool v262;
                if (v260){
                    bool v261;
                    v261 = v230 < 4l;
                    v262 = v261;
                } else {
                    v262 = false;
                }
                bool v263;
                v263 = v262 == false;
                if (v263){
                    assert("The indices should be inside the range of the dimension." && v262);
                } else {
                }
                int v265;
                v265 = v230 * 4l;
                int v266;
                v266 = v253 + v265;
                bool v267;
                v267 = 0l <= v251;
                bool v269;
                if (v267){
                    bool v268;
                    v268 = v251 < 1l;
                    v269 = v268;
                } else {
                    v269 = false;
                }
                bool v270;
                v270 = v269 == false;
                if (v270){
                    assert("The indices should be inside the range of the dimension." && v269);
                } else {
                }
                int v272;
                v272 = v251 * 16l;
                int v273;
                v273 = v266 + v272;
                assert("Tensor range check" && 0 <= v251 && v251 < 1l);
                assert("Tensor range check" && 0 <= v253 && v253 < 4l);
                int v274;
                v274 = 4l * v251;
                int v275;
                v275 = v274 + v253;
                v243[v275] = v273;
                v253 += 1l ;
            }
            v251 += 1l ;
        }
        bool v276;
        v276 = 0l <= v231;
        bool v277;
        v277 = v276 && v232;
        bool v278;
        v278 = v277 == false;
        if (v278){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v277);
        } else {
        }
        bool v280;
        v280 = 0l <= v238;
        bool v282;
        if (v280){
            bool v281;
            v281 = v238 < 32l;
            v282 = v281;
        } else {
            v282 = false;
        }
        bool v283;
        v283 = v282 == false;
        if (v283){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v282);
        } else {
        }
        int v285;
        v285 = v238 * 8l;
        int v286;
        v286 = v285 + v231;
        assert("Tensor range check" && 0 <= v238 && v238 < 32l);
        int v287;
        v287 = 8l * v238;
        int v288;
        v288 = v287 + v231;
        v14[v288] = v286;
        v238 += 1l ;
    }
    __syncthreads();
    int v289;
    v289 = threadIdx.x;
    bool v290;
    v290 = 0l <= v289;
    bool v291;
    v291 = v290 == false;
    if (v291){
        assert("The index needs to be zero or positive." && v290);
    } else {
    }
    int v293;
    v293 = v289 % 4l;
    int v294;
    v294 = v289 / 4l;
    bool v295;
    v295 = v294 < 8l;
    bool v296;
    v296 = v295 == false;
    if (v296){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v295);
    } else {
    }
    assert("Tensor range check" && 0 <= v294 && v294 < 8l);
    assert("Tensor range check" && 0 <= v293 && v293 < 4l);
    int v298;
    v298 = 4l * v293;
    int v299;
    v299 = 16l * v294;
    int v300;
    v300 = v299 + v298;
    assert("Tensor range check" && 0 <= v294 && v294 < 8l);
    assert("Tensor range check" && 0 <= v293 && v293 < 4l);
    int v301;
    v301 = 0l;
    while (while_method_2(v301)){
        assert("Tensor range check" && 0 <= v301 && v301 < 32l);
        int v303;
        v303 = 128l * v301;
        int v304;
        v304 = v303 + v300;
        float v305[4l];
        int v306[4l];
        int v307;
        v307 = 0l;
        while (while_method_3(v307)){
            assert("Tensor range check" && 0 <= v307 && v307 < 1l);
            int v309;
            v309 = 4l * v307;
            assert("Tensor range check" && 0 <= v307 && v307 < 1l);
            int v310;
            v310 = 16l * v307;
            int v311;
            v311 = v310 + v304;
            int4* v312;
            v312 = reinterpret_cast<int4*>(v1 + v311);
            int4* v313;
            v313 = reinterpret_cast<int4*>(v305 + v309);
            assert("Pointer alignment check" && (unsigned long long)(v312) % 4l == 0 && (unsigned long long)(v313) % 4l == 0);
            *v313 = *v312;
            v307 += 1l ;
        }
        int v314;
        v314 = 0l;
        while (while_method_3(v314)){
            int v316;
            v316 = 0l;
            while (while_method_1(v316)){
                bool v318;
                v318 = 0l <= v316;
                bool v320;
                if (v318){
                    bool v319;
                    v319 = v316 < 4l;
                    v320 = v319;
                } else {
                    v320 = false;
                }
                bool v321;
                v321 = v320 == false;
                if (v321){
                    assert("The indices should be inside the range of the dimension." && v320);
                } else {
                }
                bool v323;
                v323 = 0l <= v293;
                bool v325;
                if (v323){
                    bool v324;
                    v324 = v293 < 4l;
                    v325 = v324;
                } else {
                    v325 = false;
                }
                bool v326;
                v326 = v325 == false;
                if (v326){
                    assert("The indices should be inside the range of the dimension." && v325);
                } else {
                }
                int v328;
                v328 = v293 * 4l;
                int v329;
                v329 = v316 + v328;
                bool v330;
                v330 = 0l <= v314;
                bool v332;
                if (v330){
                    bool v331;
                    v331 = v314 < 1l;
                    v332 = v331;
                } else {
                    v332 = false;
                }
                bool v333;
                v333 = v332 == false;
                if (v333){
                    assert("The indices should be inside the range of the dimension." && v332);
                } else {
                }
                int v335;
                v335 = v314 * 16l;
                int v336;
                v336 = v329 + v335;
                assert("Tensor range check" && 0 <= v314 && v314 < 1l);
                assert("Tensor range check" && 0 <= v316 && v316 < 4l);
                int v337;
                v337 = 4l * v314;
                int v338;
                v338 = v337 + v316;
                v306[v338] = v336;
                v316 += 1l ;
            }
            v314 += 1l ;
        }
        bool v339;
        v339 = 0l <= v294;
        bool v340;
        v340 = v339 && v295;
        bool v341;
        v341 = v340 == false;
        if (v341){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v340);
        } else {
        }
        bool v343;
        v343 = 0l <= v301;
        bool v345;
        if (v343){
            bool v344;
            v344 = v301 < 32l;
            v345 = v344;
        } else {
            v345 = false;
        }
        bool v346;
        v346 = v345 == false;
        if (v346){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v345);
        } else {
        }
        int v348;
        v348 = v301 * 8l;
        int v349;
        v349 = v348 + v294;
        float v350;
        v350 = 0.0f;
        int v351;
        v351 = 0l;
        while (while_method_3(v351)){
            int v353;
            v353 = 0l;
            while (while_method_1(v353)){
                assert("Tensor range check" && 0 <= v351 && v351 < 1l);
                assert("Tensor range check" && 0 <= v353 && v353 < 4l);
                int v355;
                v355 = 4l * v351;
                int v356;
                v356 = v355 + v353;
                float v357;
                v357 = v305[v356];
                float v358;
                v358 = v350 + v357;
                v350 = v358;
                v353 += 1l ;
            }
            v351 += 1l ;
        }
        auto v359 = cooperative_groups::coalesced_threads();
        int v360;
        v360 = threadIdx.x;
        int v361;
        v361 = v360 / 4l;
        auto v362 = cooperative_groups::labeled_partition(v359,v361);
        float v363;
        v363 = cooperative_groups::reduce(v362, v350, v66);
        float v364;
        v364 = v363 / 16.0f;
        float v365[4l];
        int v366;
        v366 = 0l;
        while (while_method_3(v366)){
            int v368;
            v368 = 0l;
            while (while_method_1(v368)){
                assert("Tensor range check" && 0 <= v366 && v366 < 1l);
                assert("Tensor range check" && 0 <= v368 && v368 < 4l);
                int v370;
                v370 = 4l * v366;
                int v371;
                v371 = v370 + v368;
                float v372;
                v372 = v305[v371];
                float v373;
                v373 = v372 - v364;
                float v374;
                v374 = exp(v373);
                assert("Tensor range check" && 0 <= v366 && v366 < 1l);
                assert("Tensor range check" && 0 <= v368 && v368 < 4l);
                v365[v371] = v374;
                v368 += 1l ;
            }
            v366 += 1l ;
        }
        float v375;
        v375 = 0.0f;
        int v376;
        v376 = 0l;
        while (while_method_3(v376)){
            int v378;
            v378 = 0l;
            while (while_method_1(v378)){
                assert("Tensor range check" && 0 <= v376 && v376 < 1l);
                assert("Tensor range check" && 0 <= v378 && v378 < 4l);
                int v380;
                v380 = 4l * v376;
                int v381;
                v381 = v380 + v378;
                float v382;
                v382 = v365[v381];
                float v383;
                v383 = v375 + v382;
                v375 = v383;
                v378 += 1l ;
            }
            v376 += 1l ;
        }
        auto v384 = cooperative_groups::coalesced_threads();
        int v385;
        v385 = threadIdx.x;
        int v386;
        v386 = v385 / 4l;
        auto v387 = cooperative_groups::labeled_partition(v384,v386);
        float v388;
        v388 = cooperative_groups::reduce(v387, v375, v66);
        float v389[4l];
        int v390;
        v390 = 0l;
        while (while_method_3(v390)){
            int v392;
            v392 = 0l;
            while (while_method_1(v392)){
                assert("Tensor range check" && 0 <= v390 && v390 < 1l);
                assert("Tensor range check" && 0 <= v392 && v392 < 4l);
                int v394;
                v394 = 4l * v390;
                int v395;
                v395 = v394 + v392;
                float v396;
                v396 = v365[v395];
                bool v397;
                v397 = v388 == 0.0f;
                bool v398;
                v398 = v397 != true;
                float v400;
                if (v398){
                    float v399;
                    v399 = v396 / v388;
                    v400 = v399;
                } else {
                    v400 = 0.0625f;
                }
                assert("Tensor range check" && 0 <= v390 && v390 < 1l);
                assert("Tensor range check" && 0 <= v392 && v392 < 4l);
                v389[v395] = v400;
                v392 += 1l ;
            }
            v390 += 1l ;
        }
        assert("Tensor range check" && 0 <= v301 && v301 < 32l);
        int v401;
        v401 = 0l;
        while (while_method_3(v401)){
            assert("Tensor range check" && 0 <= v401 && v401 < 1l);
            int v403;
            v403 = 16l * v401;
            int v404;
            v404 = v403 + v304;
            assert("Tensor range check" && 0 <= v401 && v401 < 1l);
            int v405;
            v405 = 4l * v401;
            int4* v406;
            v406 = reinterpret_cast<int4*>(v389 + v405);
            int4* v407;
            v407 = reinterpret_cast<int4*>(v4 + v404);
            assert("Pointer alignment check" && (unsigned long long)(v406) % 4l == 0 && (unsigned long long)(v407) % 4l == 0);
            *v407 = *v406;
            v401 += 1l ;
        }
        v301 += 1l ;
    }
    __syncthreads();
    int v408;
    v408 = threadIdx.x;
    bool v409;
    v409 = 0l <= v408;
    bool v410;
    v410 = v409 == false;
    if (v410){
        assert("The index needs to be zero or positive." && v409);
    } else {
    }
    int v412;
    v412 = v408 % 4l;
    int v413;
    v413 = v408 / 4l;
    bool v414;
    v414 = v413 < 8l;
    bool v415;
    v415 = v414 == false;
    if (v415){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v414);
    } else {
    }
    assert("Tensor range check" && 0 <= v413 && v413 < 8l);
    assert("Tensor range check" && 0 <= v412 && v412 < 4l);
    int v417;
    v417 = 4l * v412;
    int v418;
    v418 = 16l * v413;
    int v419;
    v419 = v418 + v417;
    assert("Tensor range check" && 0 <= v413 && v413 < 8l);
    assert("Tensor range check" && 0 <= v412 && v412 < 4l);
    int v420;
    v420 = 0l;
    while (while_method_2(v420)){
        assert("Tensor range check" && 0 <= v420 && v420 < 32l);
        int v422;
        v422 = 128l * v420;
        int v423;
        v423 = v422 + v419;
        float v424[4l];
        int v425[4l];
        int v426;
        v426 = 0l;
        while (while_method_3(v426)){
            assert("Tensor range check" && 0 <= v426 && v426 < 1l);
            int v428;
            v428 = 4l * v426;
            assert("Tensor range check" && 0 <= v426 && v426 < 1l);
            int v429;
            v429 = 16l * v426;
            int v430;
            v430 = v429 + v423;
            int4* v431;
            v431 = reinterpret_cast<int4*>(v1 + v430);
            int4* v432;
            v432 = reinterpret_cast<int4*>(v424 + v428);
            assert("Pointer alignment check" && (unsigned long long)(v431) % 4l == 0 && (unsigned long long)(v432) % 4l == 0);
            *v432 = *v431;
            v426 += 1l ;
        }
        int v433;
        v433 = 0l;
        while (while_method_3(v433)){
            int v435;
            v435 = 0l;
            while (while_method_1(v435)){
                bool v437;
                v437 = 0l <= v435;
                bool v439;
                if (v437){
                    bool v438;
                    v438 = v435 < 4l;
                    v439 = v438;
                } else {
                    v439 = false;
                }
                bool v440;
                v440 = v439 == false;
                if (v440){
                    assert("The indices should be inside the range of the dimension." && v439);
                } else {
                }
                bool v442;
                v442 = 0l <= v412;
                bool v444;
                if (v442){
                    bool v443;
                    v443 = v412 < 4l;
                    v444 = v443;
                } else {
                    v444 = false;
                }
                bool v445;
                v445 = v444 == false;
                if (v445){
                    assert("The indices should be inside the range of the dimension." && v444);
                } else {
                }
                int v447;
                v447 = v412 * 4l;
                int v448;
                v448 = v435 + v447;
                bool v449;
                v449 = 0l <= v433;
                bool v451;
                if (v449){
                    bool v450;
                    v450 = v433 < 1l;
                    v451 = v450;
                } else {
                    v451 = false;
                }
                bool v452;
                v452 = v451 == false;
                if (v452){
                    assert("The indices should be inside the range of the dimension." && v451);
                } else {
                }
                int v454;
                v454 = v433 * 16l;
                int v455;
                v455 = v448 + v454;
                assert("Tensor range check" && 0 <= v433 && v433 < 1l);
                assert("Tensor range check" && 0 <= v435 && v435 < 4l);
                int v456;
                v456 = 4l * v433;
                int v457;
                v457 = v456 + v435;
                v425[v457] = v455;
                v435 += 1l ;
            }
            v433 += 1l ;
        }
        bool v458;
        v458 = 0l <= v413;
        bool v459;
        v459 = v458 && v414;
        bool v460;
        v460 = v459 == false;
        if (v460){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v459);
        } else {
        }
        bool v462;
        v462 = 0l <= v420;
        bool v464;
        if (v462){
            bool v463;
            v463 = v420 < 32l;
            v464 = v463;
        } else {
            v464 = false;
        }
        bool v465;
        v465 = v464 == false;
        if (v465){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v464);
        } else {
        }
        int v467;
        v467 = v420 * 8l;
        int v468;
        v468 = v467 + v413;
        float v469[4l];
        int v470;
        v470 = 0l;
        while (while_method_3(v470)){
            int v472;
            v472 = 0l;
            while (while_method_1(v472)){
                assert("Tensor range check" && 0 <= v470 && v470 < 1l);
                assert("Tensor range check" && 0 <= v472 && v472 < 4l);
                int v474;
                v474 = 4l * v470;
                int v475;
                v475 = v474 + v472;
                float v476;
                v476 = v424[v475];
                float v477;
                v477 = v476 * v476;
                assert("Tensor range check" && 0 <= v470 && v470 < 1l);
                assert("Tensor range check" && 0 <= v472 && v472 < 4l);
                v469[v475] = v477;
                v472 += 1l ;
            }
            v470 += 1l ;
        }
        float v478;
        v478 = 0.0f;
        int v479;
        v479 = 0l;
        while (while_method_3(v479)){
            int v481;
            v481 = 0l;
            while (while_method_1(v481)){
                assert("Tensor range check" && 0 <= v479 && v479 < 1l);
                assert("Tensor range check" && 0 <= v481 && v481 < 4l);
                int v483;
                v483 = 4l * v479;
                int v484;
                v484 = v483 + v481;
                float v485;
                v485 = v469[v484];
                float v486;
                v486 = v478 + v485;
                v478 = v486;
                v481 += 1l ;
            }
            v479 += 1l ;
        }
        auto v487 = cooperative_groups::coalesced_threads();
        int v488;
        v488 = threadIdx.x;
        int v489;
        v489 = v488 / 4l;
        auto v490 = cooperative_groups::labeled_partition(v487,v489);
        float v491;
        v491 = cooperative_groups::reduce(v490, v478, v66);
        float v492[4l];
        int v493;
        v493 = 0l;
        while (while_method_3(v493)){
            int v495;
            v495 = 0l;
            while (while_method_1(v495)){
                assert("Tensor range check" && 0 <= v493 && v493 < 1l);
                assert("Tensor range check" && 0 <= v495 && v495 < 4l);
                int v497;
                v497 = 4l * v493;
                int v498;
                v498 = v497 + v495;
                float v499;
                v499 = v424[v498];
                bool v500;
                v500 = v491 == 0.0f;
                bool v501;
                v501 = v500 != true;
                float v503;
                if (v501){
                    float v502;
                    v502 = v499 / v491;
                    v503 = v502;
                } else {
                    v503 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v493 && v493 < 1l);
                assert("Tensor range check" && 0 <= v495 && v495 < 4l);
                v492[v498] = v503;
                v495 += 1l ;
            }
            v493 += 1l ;
        }
        assert("Tensor range check" && 0 <= v420 && v420 < 32l);
        int v504;
        v504 = 0l;
        while (while_method_3(v504)){
            assert("Tensor range check" && 0 <= v504 && v504 < 1l);
            int v506;
            v506 = 16l * v504;
            int v507;
            v507 = v506 + v423;
            assert("Tensor range check" && 0 <= v504 && v504 < 1l);
            int v508;
            v508 = 4l * v504;
            int4* v509;
            v509 = reinterpret_cast<int4*>(v492 + v508);
            int4* v510;
            v510 = reinterpret_cast<int4*>(v9 + v507);
            assert("Pointer alignment check" && (unsigned long long)(v509) % 4l == 0 && (unsigned long long)(v510) % 4l == 0);
            *v510 = *v509;
            v504 += 1l ;
        }
        v420 += 1l ;
    }
    __syncthreads();
    int v511;
    v511 = threadIdx.x;
    bool v512;
    v512 = 0l <= v511;
    bool v513;
    v513 = v512 == false;
    if (v513){
        assert("The index needs to be zero or positive." && v512);
    } else {
    }
    int v515;
    v515 = v511 % 4l;
    int v516;
    v516 = v511 / 4l;
    bool v517;
    v517 = v516 < 8l;
    bool v518;
    v518 = v517 == false;
    if (v518){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v517);
    } else {
    }
    assert("Tensor range check" && 0 <= v516 && v516 < 8l);
    assert("Tensor range check" && 0 <= v515 && v515 < 4l);
    int v520;
    v520 = 4l * v515;
    int v521;
    v521 = 16l * v516;
    int v522;
    v522 = v521 + v520;
    assert("Tensor range check" && 0 <= v516 && v516 < 8l);
    int v523;
    v523 = 0l;
    while (while_method_2(v523)){
        assert("Tensor range check" && 0 <= v523 && v523 < 32l);
        int v525;
        v525 = 128l * v523;
        int v526;
        v526 = v525 + v522;
        float v527[4l];
        int v528[4l];
        int v529;
        v529 = 0l;
        while (while_method_3(v529)){
            assert("Tensor range check" && 0 <= v529 && v529 < 1l);
            int v531;
            v531 = 4l * v529;
            assert("Tensor range check" && 0 <= v529 && v529 < 1l);
            int v532;
            v532 = 16l * v529;
            int v533;
            v533 = v532 + v526;
            int4* v534;
            v534 = reinterpret_cast<int4*>(v1 + v533);
            int4* v535;
            v535 = reinterpret_cast<int4*>(v527 + v531);
            assert("Pointer alignment check" && (unsigned long long)(v534) % 4l == 0 && (unsigned long long)(v535) % 4l == 0);
            *v535 = *v534;
            v529 += 1l ;
        }
        int v536;
        v536 = 0l;
        while (while_method_3(v536)){
            int v538;
            v538 = 0l;
            while (while_method_1(v538)){
                bool v540;
                v540 = 0l <= v538;
                bool v542;
                if (v540){
                    bool v541;
                    v541 = v538 < 4l;
                    v542 = v541;
                } else {
                    v542 = false;
                }
                bool v543;
                v543 = v542 == false;
                if (v543){
                    assert("The indices should be inside the range of the dimension." && v542);
                } else {
                }
                bool v545;
                v545 = 0l <= v515;
                bool v547;
                if (v545){
                    bool v546;
                    v546 = v515 < 4l;
                    v547 = v546;
                } else {
                    v547 = false;
                }
                bool v548;
                v548 = v547 == false;
                if (v548){
                    assert("The indices should be inside the range of the dimension." && v547);
                } else {
                }
                int v550;
                v550 = v515 * 4l;
                int v551;
                v551 = v538 + v550;
                bool v552;
                v552 = 0l <= v536;
                bool v554;
                if (v552){
                    bool v553;
                    v553 = v536 < 1l;
                    v554 = v553;
                } else {
                    v554 = false;
                }
                bool v555;
                v555 = v554 == false;
                if (v555){
                    assert("The indices should be inside the range of the dimension." && v554);
                } else {
                }
                int v557;
                v557 = v536 * 16l;
                int v558;
                v558 = v551 + v557;
                assert("Tensor range check" && 0 <= v536 && v536 < 1l);
                assert("Tensor range check" && 0 <= v538 && v538 < 4l);
                int v559;
                v559 = 4l * v536;
                int v560;
                v560 = v559 + v538;
                v528[v560] = v558;
                v538 += 1l ;
            }
            v536 += 1l ;
        }
        bool v561;
        v561 = 0l <= v516;
        bool v562;
        v562 = v561 && v517;
        bool v563;
        v563 = v562 == false;
        if (v563){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v562);
        } else {
        }
        bool v565;
        v565 = 0l <= v523;
        bool v567;
        if (v565){
            bool v566;
            v566 = v523 < 32l;
            v567 = v566;
        } else {
            v567 = false;
        }
        bool v568;
        v568 = v567 == false;
        if (v568){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v567);
        } else {
        }
        int v570;
        v570 = v523 * 8l;
        int v571;
        v571 = v570 + v516;
        float v572; int v573;
        Tuple1 tmp1 = Tuple1{-1.0f / 0.0f, 0l};
        v572 = tmp1.v0; v573 = tmp1.v1;
        int v574;
        v574 = 0l;
        while (while_method_3(v574)){
            int v576;
            v576 = 0l;
            while (while_method_1(v576)){
                assert("Tensor range check" && 0 <= v574 && v574 < 1l);
                assert("Tensor range check" && 0 <= v576 && v576 < 4l);
                int v578;
                v578 = 4l * v574;
                int v579;
                v579 = v578 + v576;
                float v580;
                v580 = v527[v579];
                int v581;
                v581 = v528[v579];
                bool v582;
                v582 = v572 > v580;
                float v583; int v584;
                if (v582){
                    v583 = v572; v584 = v573;
                } else {
                    v583 = v580; v584 = v581;
                }
                v572 = v583;
                v573 = v584;
                v576 += 1l ;
            }
            v574 += 1l ;
        }
        auto v585 = cooperative_groups::coalesced_threads();
        int v586;
        v586 = threadIdx.x;
        int v587;
        v587 = v586 / 4l;
        auto v588 = cooperative_groups::labeled_partition(v585,v587);
        Closure1 v589{};
        float v590; int v591;
        Tuple1 tmp2 = cooperative_groups::reduce(v588, Tuple1{v572, v573}, v589);
        v590 = tmp2.v0; v591 = tmp2.v1;
        assert("Tensor range check" && 0 <= v523 && v523 < 32l);
        int v592;
        v592 = 8l * v523;
        int v593;
        v593 = v592 + v516;
        v10[v593] = v591;
        v523 += 1l ;
    }
    __syncthreads();
    int v594;
    v594 = threadIdx.x;
    bool v595;
    v595 = 0l <= v594;
    bool v596;
    v596 = v595 == false;
    if (v596){
        assert("The index needs to be zero or positive." && v595);
    } else {
    }
    int v598;
    v598 = v594 % 4l;
    int v599;
    v599 = v594 / 4l;
    bool v600;
    v600 = v599 < 8l;
    bool v601;
    v601 = v600 == false;
    if (v601){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v600);
    } else {
    }
    assert("Tensor range check" && 0 <= v599 && v599 < 8l);
    assert("Tensor range check" && 0 <= v598 && v598 < 4l);
    int v603;
    v603 = 4l * v598;
    int v604;
    v604 = 16l * v599;
    int v605;
    v605 = v604 + v603;
    assert("Tensor range check" && 0 <= v599 && v599 < 8l);
    assert("Tensor range check" && 0 <= v598 && v598 < 4l);
    int v606;
    v606 = 0l;
    while (while_method_2(v606)){
        assert("Tensor range check" && 0 <= v606 && v606 < 32l);
        int v608;
        v608 = 128l * v606;
        int v609;
        v609 = v608 + v605;
        float v610[4l];
        int v611[4l];
        int v612;
        v612 = 0l;
        while (while_method_3(v612)){
            assert("Tensor range check" && 0 <= v612 && v612 < 1l);
            int v614;
            v614 = 4l * v612;
            assert("Tensor range check" && 0 <= v612 && v612 < 1l);
            int v615;
            v615 = 16l * v612;
            int v616;
            v616 = v615 + v609;
            int4* v617;
            v617 = reinterpret_cast<int4*>(v1 + v616);
            int4* v618;
            v618 = reinterpret_cast<int4*>(v610 + v614);
            assert("Pointer alignment check" && (unsigned long long)(v617) % 4l == 0 && (unsigned long long)(v618) % 4l == 0);
            *v618 = *v617;
            v612 += 1l ;
        }
        int v619;
        v619 = 0l;
        while (while_method_3(v619)){
            int v621;
            v621 = 0l;
            while (while_method_1(v621)){
                bool v623;
                v623 = 0l <= v621;
                bool v625;
                if (v623){
                    bool v624;
                    v624 = v621 < 4l;
                    v625 = v624;
                } else {
                    v625 = false;
                }
                bool v626;
                v626 = v625 == false;
                if (v626){
                    assert("The indices should be inside the range of the dimension." && v625);
                } else {
                }
                bool v628;
                v628 = 0l <= v598;
                bool v630;
                if (v628){
                    bool v629;
                    v629 = v598 < 4l;
                    v630 = v629;
                } else {
                    v630 = false;
                }
                bool v631;
                v631 = v630 == false;
                if (v631){
                    assert("The indices should be inside the range of the dimension." && v630);
                } else {
                }
                int v633;
                v633 = v598 * 4l;
                int v634;
                v634 = v621 + v633;
                bool v635;
                v635 = 0l <= v619;
                bool v637;
                if (v635){
                    bool v636;
                    v636 = v619 < 1l;
                    v637 = v636;
                } else {
                    v637 = false;
                }
                bool v638;
                v638 = v637 == false;
                if (v638){
                    assert("The indices should be inside the range of the dimension." && v637);
                } else {
                }
                int v640;
                v640 = v619 * 16l;
                int v641;
                v641 = v634 + v640;
                assert("Tensor range check" && 0 <= v619 && v619 < 1l);
                assert("Tensor range check" && 0 <= v621 && v621 < 4l);
                int v642;
                v642 = 4l * v619;
                int v643;
                v643 = v642 + v621;
                v611[v643] = v641;
                v621 += 1l ;
            }
            v619 += 1l ;
        }
        bool v644;
        v644 = 0l <= v599;
        bool v645;
        v645 = v644 && v600;
        bool v646;
        v646 = v645 == false;
        if (v646){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v645);
        } else {
        }
        bool v648;
        v648 = 0l <= v606;
        bool v650;
        if (v648){
            bool v649;
            v649 = v606 < 32l;
            v650 = v649;
        } else {
            v650 = false;
        }
        bool v651;
        v651 = v650 == false;
        if (v651){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v650);
        } else {
        }
        int v653;
        v653 = v606 * 8l;
        int v654;
        v654 = v653 + v599;
        float v655;
        v655 = 0.0f;
        int v656;
        v656 = 0l;
        while (while_method_3(v656)){
            int v658;
            v658 = 0l;
            while (while_method_1(v658)){
                assert("Tensor range check" && 0 <= v656 && v656 < 1l);
                assert("Tensor range check" && 0 <= v658 && v658 < 4l);
                int v660;
                v660 = 4l * v656;
                int v661;
                v661 = v660 + v658;
                float v662;
                v662 = v610[v661];
                float v663;
                v663 = v655 + v662;
                v655 = v663;
                v658 += 1l ;
            }
            v656 += 1l ;
        }
        auto v664 = cooperative_groups::coalesced_threads();
        int v665;
        v665 = threadIdx.x;
        int v666;
        v666 = v665 / 4l;
        auto v667 = cooperative_groups::labeled_partition(v664,v666);
        float v668;
        v668 = cooperative_groups::reduce(v667, v655, v66);
        float v669;
        v669 = v668 / 16.0f;
        float v670[4l];
        int v671;
        v671 = 0l;
        while (while_method_3(v671)){
            int v673;
            v673 = 0l;
            while (while_method_1(v673)){
                assert("Tensor range check" && 0 <= v671 && v671 < 1l);
                assert("Tensor range check" && 0 <= v673 && v673 < 4l);
                int v675;
                v675 = 4l * v671;
                int v676;
                v676 = v675 + v673;
                float v677;
                v677 = v610[v676];
                float v678;
                v678 = v677 - v669;
                float v679;
                v679 = exp(v678);
                assert("Tensor range check" && 0 <= v671 && v671 < 1l);
                assert("Tensor range check" && 0 <= v673 && v673 < 4l);
                v670[v676] = v679;
                v673 += 1l ;
            }
            v671 += 1l ;
        }
        float v680;
        v680 = 0.0f;
        int v681;
        v681 = 0l;
        while (while_method_3(v681)){
            int v683;
            v683 = 0l;
            while (while_method_1(v683)){
                assert("Tensor range check" && 0 <= v681 && v681 < 1l);
                assert("Tensor range check" && 0 <= v683 && v683 < 4l);
                int v685;
                v685 = 4l * v681;
                int v686;
                v686 = v685 + v683;
                float v687;
                v687 = v670[v686];
                float v688;
                v688 = v680 + v687;
                v680 = v688;
                v683 += 1l ;
            }
            v681 += 1l ;
        }
        auto v689 = cooperative_groups::coalesced_threads();
        int v690;
        v690 = threadIdx.x;
        int v691;
        v691 = v690 / 4l;
        auto v692 = cooperative_groups::labeled_partition(v689,v691);
        float v693;
        v693 = cooperative_groups::reduce(v692, v680, v66);
        float v694[4l];
        int v695;
        v695 = 0l;
        while (while_method_3(v695)){
            int v697;
            v697 = 0l;
            while (while_method_1(v697)){
                assert("Tensor range check" && 0 <= v695 && v695 < 1l);
                assert("Tensor range check" && 0 <= v697 && v697 < 4l);
                int v699;
                v699 = 4l * v695;
                int v700;
                v700 = v699 + v697;
                float v701;
                v701 = v670[v700];
                bool v702;
                v702 = v693 == 0.0f;
                bool v703;
                v703 = v702 != true;
                float v705;
                if (v703){
                    float v704;
                    v704 = v701 / v693;
                    v705 = v704;
                } else {
                    v705 = 0.0625f;
                }
                assert("Tensor range check" && 0 <= v695 && v695 < 1l);
                assert("Tensor range check" && 0 <= v697 && v697 < 4l);
                v694[v700] = v705;
                v697 += 1l ;
            }
            v695 += 1l ;
        }
        float v706[4l];
        float v707;
        v707 = 0.0f;
        int v708;
        v708 = 0l;
        while (while_method_3(v708)){
            assert("Tensor range check" && 0 <= v708 && v708 < 1l);
            int v710;
            v710 = 4l * v708;
            assert("Tensor range check" && 0 <= v708 && v708 < 1l);
            int v711; float v712;
            Tuple0 tmp3 = Tuple0{0l, 0.0f};
            v711 = tmp3.v0; v712 = tmp3.v1;
            while (while_method_1(v711)){
                assert("Tensor range check" && 0 <= v711 && v711 < 4l);
                int v714;
                v714 = v711 + v710;
                float v715;
                v715 = v694[v714];
                float v716;
                v716 = v712 + v715;
                v712 = v716;
                v711 += 1l ;
            }
            auto v717 = cooperative_groups::coalesced_threads();
            int v718;
            v718 = threadIdx.x;
            int v719;
            v719 = v718 / 4l;
            auto v720 = cooperative_groups::labeled_partition(v717,v719);
            Closure2 v721{};
            float v722;
            v722 = cooperative_groups::inclusive_scan(v720, v712, v721);
            float v723;
            v723 = v720.shfl_up(v722,1);
            bool v724;
            v724 = v720.thread_rank() == 0;
            float v725;
            if (v724){
                v725 = 0.0f;
            } else {
                v725 = v723;
            }
            float v726;
            v726 = v720.shfl(v722,v720.num_threads()-1);
            float v727;
            v727 = v707 + v725;
            int v728; float v729;
            Tuple0 tmp4 = Tuple0{0l, v727};
            v728 = tmp4.v0; v729 = tmp4.v1;
            while (while_method_1(v728)){
                assert("Tensor range check" && 0 <= v728 && v728 < 4l);
                int v731;
                v731 = v728 + v710;
                float v732;
                v732 = v694[v731];
                float v733;
                v733 = v729 + v732;
                assert("Tensor range check" && 0 <= v728 && v728 < 4l);
                v706[v731] = v733;
                v729 = v733;
                v728 += 1l ;
            }
            float v734;
            v734 = v707 + v726;
            v707 = v734;
            v708 += 1l ;
        }
        assert("Tensor range check" && 0 <= v606 && v606 < 32l);
        int v735;
        v735 = 0l;
        while (while_method_3(v735)){
            assert("Tensor range check" && 0 <= v735 && v735 < 1l);
            int v737;
            v737 = 16l * v735;
            int v738;
            v738 = v737 + v609;
            assert("Tensor range check" && 0 <= v735 && v735 < 1l);
            int v739;
            v739 = 4l * v735;
            int4* v740;
            v740 = reinterpret_cast<int4*>(v694 + v739);
            int4* v741;
            v741 = reinterpret_cast<int4*>(v7 + v738);
            assert("Pointer alignment check" && (unsigned long long)(v740) % 4l == 0 && (unsigned long long)(v741) % 4l == 0);
            *v741 = *v740;
            int4* v742;
            v742 = reinterpret_cast<int4*>(v706 + v739);
            int4* v743;
            v743 = reinterpret_cast<int4*>(v8 + v738);
            assert("Pointer alignment check" && (unsigned long long)(v742) % 4l == 0 && (unsigned long long)(v743) % 4l == 0);
            *v743 = *v742;
            v735 += 1l ;
        }
        v606 += 1l ;
    }
    __syncthreads();
    int v744;
    v744 = threadIdx.x;
    bool v745;
    v745 = 0l <= v744;
    bool v746;
    v746 = v745 == false;
    if (v746){
        assert("The index needs to be zero or positive." && v745);
    } else {
    }
    int v748;
    v748 = v744 % 4l;
    int v749;
    v749 = v744 / 4l;
    bool v750;
    v750 = v749 < 8l;
    bool v751;
    v751 = v750 == false;
    if (v751){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v750);
    } else {
    }
    assert("Tensor range check" && 0 <= v749 && v749 < 8l);
    assert("Tensor range check" && 0 <= v748 && v748 < 4l);
    int v753;
    v753 = 4l * v748;
    int v754;
    v754 = 16l * v749;
    int v755;
    v755 = v754 + v753;
    assert("Tensor range check" && 0 <= v749 && v749 < 8l);
    assert("Tensor range check" && 0 <= v748 && v748 < 4l);
    int v756;
    v756 = 0l;
    while (while_method_2(v756)){
        assert("Tensor range check" && 0 <= v756 && v756 < 32l);
        int v758;
        v758 = 128l * v756;
        int v759;
        v759 = v758 + v755;
        int v760[4l];
        int v761[4l];
        int v762;
        v762 = 0l;
        while (while_method_3(v762)){
            assert("Tensor range check" && 0 <= v762 && v762 < 1l);
            int v764;
            v764 = 4l * v762;
            assert("Tensor range check" && 0 <= v762 && v762 < 1l);
            int v765;
            v765 = 16l * v762;
            int v766;
            v766 = v765 + v759;
            int4* v767;
            v767 = reinterpret_cast<int4*>(v0 + v766);
            int4* v768;
            v768 = reinterpret_cast<int4*>(v760 + v764);
            assert("Pointer alignment check" && (unsigned long long)(v767) % 4l == 0 && (unsigned long long)(v768) % 4l == 0);
            *v768 = *v767;
            v762 += 1l ;
        }
        int v769;
        v769 = 0l;
        while (while_method_3(v769)){
            int v771;
            v771 = 0l;
            while (while_method_1(v771)){
                bool v773;
                v773 = 0l <= v771;
                bool v775;
                if (v773){
                    bool v774;
                    v774 = v771 < 4l;
                    v775 = v774;
                } else {
                    v775 = false;
                }
                bool v776;
                v776 = v775 == false;
                if (v776){
                    assert("The indices should be inside the range of the dimension." && v775);
                } else {
                }
                bool v778;
                v778 = 0l <= v748;
                bool v780;
                if (v778){
                    bool v779;
                    v779 = v748 < 4l;
                    v780 = v779;
                } else {
                    v780 = false;
                }
                bool v781;
                v781 = v780 == false;
                if (v781){
                    assert("The indices should be inside the range of the dimension." && v780);
                } else {
                }
                int v783;
                v783 = v748 * 4l;
                int v784;
                v784 = v771 + v783;
                bool v785;
                v785 = 0l <= v769;
                bool v787;
                if (v785){
                    bool v786;
                    v786 = v769 < 1l;
                    v787 = v786;
                } else {
                    v787 = false;
                }
                bool v788;
                v788 = v787 == false;
                if (v788){
                    assert("The indices should be inside the range of the dimension." && v787);
                } else {
                }
                int v790;
                v790 = v769 * 16l;
                int v791;
                v791 = v784 + v790;
                assert("Tensor range check" && 0 <= v769 && v769 < 1l);
                assert("Tensor range check" && 0 <= v771 && v771 < 4l);
                int v792;
                v792 = 4l * v769;
                int v793;
                v793 = v792 + v771;
                v761[v793] = v791;
                v771 += 1l ;
            }
            v769 += 1l ;
        }
        bool v794;
        v794 = 0l <= v749;
        bool v795;
        v795 = v794 && v750;
        bool v796;
        v796 = v795 == false;
        if (v796){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v795);
        } else {
        }
        bool v798;
        v798 = 0l <= v756;
        bool v800;
        if (v798){
            bool v799;
            v799 = v756 < 32l;
            v800 = v799;
        } else {
            v800 = false;
        }
        bool v801;
        v801 = v800 == false;
        if (v801){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v800);
        } else {
        }
        int v803;
        v803 = v756 * 8l;
        int v804;
        v804 = v803 + v749;
        int v805[4l];
        int v806;
        v806 = 0l;
        int v807;
        v807 = 0l;
        while (while_method_3(v807)){
            assert("Tensor range check" && 0 <= v807 && v807 < 1l);
            int v809;
            v809 = 4l * v807;
            assert("Tensor range check" && 0 <= v807 && v807 < 1l);
            int v810; int v811;
            Tuple2 tmp5 = Tuple2{0l, 0l};
            v810 = tmp5.v0; v811 = tmp5.v1;
            while (while_method_1(v810)){
                assert("Tensor range check" && 0 <= v810 && v810 < 4l);
                int v813;
                v813 = v810 + v809;
                int v814;
                v814 = v760[v813];
                int v815;
                v815 = v811 + v814;
                v811 = v815;
                v810 += 1l ;
            }
            auto v816 = cooperative_groups::coalesced_threads();
            int v817;
            v817 = threadIdx.x;
            int v818;
            v818 = v817 / 4l;
            auto v819 = cooperative_groups::labeled_partition(v816,v818);
            Closure3 v820{};
            int v821;
            v821 = cooperative_groups::inclusive_scan(v819, v811, v820);
            int v822;
            v822 = v819.shfl_up(v821,1);
            bool v823;
            v823 = v819.thread_rank() == 0;
            int v824;
            if (v823){
                v824 = 0l;
            } else {
                v824 = v822;
            }
            int v825;
            v825 = v819.shfl(v821,v819.num_threads()-1);
            int v826;
            v826 = v806 + v824;
            int v827; int v828;
            Tuple2 tmp6 = Tuple2{0l, v826};
            v827 = tmp6.v0; v828 = tmp6.v1;
            while (while_method_1(v827)){
                assert("Tensor range check" && 0 <= v827 && v827 < 4l);
                int v830;
                v830 = v827 + v809;
                int v831;
                v831 = v760[v830];
                assert("Tensor range check" && 0 <= v827 && v827 < 4l);
                v805[v830] = v828;
                int v832;
                v832 = v828 + v831;
                v828 = v832;
                v827 += 1l ;
            }
            int v833;
            v833 = v806 + v825;
            v806 = v833;
            v807 += 1l ;
        }
        assert("Tensor range check" && 0 <= v756 && v756 < 32l);
        int v834;
        v834 = 0l;
        while (while_method_3(v834)){
            assert("Tensor range check" && 0 <= v834 && v834 < 1l);
            int v836;
            v836 = 16l * v834;
            int v837;
            v837 = v836 + v759;
            assert("Tensor range check" && 0 <= v834 && v834 < 1l);
            int v838;
            v838 = 4l * v834;
            int4* v839;
            v839 = reinterpret_cast<int4*>(v805 + v838);
            int4* v840;
            v840 = reinterpret_cast<int4*>(v15 + v837);
            assert("Pointer alignment check" && (unsigned long long)(v839) % 4l == 0 && (unsigned long long)(v840) % 4l == 0);
            *v840 = *v839;
            v834 += 1l ;
        }
        v756 += 1l ;
    }
    __syncthreads();
    int v841;
    v841 = threadIdx.x;
    bool v842;
    v842 = 0l <= v841;
    bool v843;
    v843 = v842 == false;
    if (v843){
        assert("The index needs to be zero or positive." && v842);
    } else {
    }
    int v845;
    v845 = v841 % 4l;
    int v846;
    v846 = v841 / 4l;
    bool v847;
    v847 = v846 < 8l;
    bool v848;
    v848 = v847 == false;
    if (v848){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v847);
    } else {
    }
    assert("Tensor range check" && 0 <= v846 && v846 < 8l);
    assert("Tensor range check" && 0 <= v845 && v845 < 4l);
    int v850;
    v850 = 4l * v845;
    int v851;
    v851 = 16l * v846;
    int v852;
    v852 = v851 + v850;
    assert("Tensor range check" && 0 <= v846 && v846 < 8l);
    assert("Tensor range check" && 0 <= v845 && v845 < 4l);
    int v853;
    v853 = 0l;
    while (while_method_2(v853)){
        assert("Tensor range check" && 0 <= v853 && v853 < 32l);
        int v855;
        v855 = 128l * v853;
        int v856;
        v856 = v855 + v852;
        float v857[4l];
        int v858[4l];
        int v859;
        v859 = 0l;
        while (while_method_3(v859)){
            assert("Tensor range check" && 0 <= v859 && v859 < 1l);
            int v861;
            v861 = 4l * v859;
            assert("Tensor range check" && 0 <= v859 && v859 < 1l);
            int v862;
            v862 = 16l * v859;
            int v863;
            v863 = v862 + v856;
            int4* v864;
            v864 = reinterpret_cast<int4*>(v1 + v863);
            int4* v865;
            v865 = reinterpret_cast<int4*>(v857 + v861);
            assert("Pointer alignment check" && (unsigned long long)(v864) % 4l == 0 && (unsigned long long)(v865) % 4l == 0);
            *v865 = *v864;
            v859 += 1l ;
        }
        int v866;
        v866 = 0l;
        while (while_method_3(v866)){
            int v868;
            v868 = 0l;
            while (while_method_1(v868)){
                bool v870;
                v870 = 0l <= v868;
                bool v872;
                if (v870){
                    bool v871;
                    v871 = v868 < 4l;
                    v872 = v871;
                } else {
                    v872 = false;
                }
                bool v873;
                v873 = v872 == false;
                if (v873){
                    assert("The indices should be inside the range of the dimension." && v872);
                } else {
                }
                bool v875;
                v875 = 0l <= v845;
                bool v877;
                if (v875){
                    bool v876;
                    v876 = v845 < 4l;
                    v877 = v876;
                } else {
                    v877 = false;
                }
                bool v878;
                v878 = v877 == false;
                if (v878){
                    assert("The indices should be inside the range of the dimension." && v877);
                } else {
                }
                int v880;
                v880 = v845 * 4l;
                int v881;
                v881 = v868 + v880;
                bool v882;
                v882 = 0l <= v866;
                bool v884;
                if (v882){
                    bool v883;
                    v883 = v866 < 1l;
                    v884 = v883;
                } else {
                    v884 = false;
                }
                bool v885;
                v885 = v884 == false;
                if (v885){
                    assert("The indices should be inside the range of the dimension." && v884);
                } else {
                }
                int v887;
                v887 = v866 * 16l;
                int v888;
                v888 = v881 + v887;
                assert("Tensor range check" && 0 <= v866 && v866 < 1l);
                assert("Tensor range check" && 0 <= v868 && v868 < 4l);
                int v889;
                v889 = 4l * v866;
                int v890;
                v890 = v889 + v868;
                v858[v890] = v888;
                v868 += 1l ;
            }
            v866 += 1l ;
        }
        bool v891;
        v891 = 0l <= v846;
        bool v892;
        v892 = v891 && v847;
        bool v893;
        v893 = v892 == false;
        if (v893){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v892);
        } else {
        }
        bool v895;
        v895 = 0l <= v853;
        bool v897;
        if (v895){
            bool v896;
            v896 = v853 < 32l;
            v897 = v896;
        } else {
            v897 = false;
        }
        bool v898;
        v898 = v897 == false;
        if (v898){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v897);
        } else {
        }
        int v900;
        v900 = v853 * 8l;
        int v901;
        v901 = v900 + v846;
        bool v902[4l];
        int v903;
        v903 = 0l;
        while (while_method_3(v903)){
            int v905;
            v905 = 0l;
            while (while_method_1(v905)){
                assert("Tensor range check" && 0 <= v903 && v903 < 1l);
                assert("Tensor range check" && 0 <= v905 && v905 < 4l);
                int v907;
                v907 = 4l * v903;
                int v908;
                v908 = v907 + v905;
                float v909;
                v909 = v857[v908];
                int v910;
                v910 = v858[v908];
                bool v911;
                v911 = v910 < 4l;
                assert("Tensor range check" && 0 <= v903 && v903 < 1l);
                assert("Tensor range check" && 0 <= v905 && v905 < 4l);
                v902[v908] = v911;
                v905 += 1l ;
            }
            v903 += 1l ;
        }
        int v912[4l];
        int v913;
        v913 = 0l;
        while (while_method_3(v913)){
            int v915;
            v915 = 0l;
            while (while_method_1(v915)){
                assert("Tensor range check" && 0 <= v913 && v913 < 1l);
                assert("Tensor range check" && 0 <= v915 && v915 < 4l);
                int v917;
                v917 = 4l * v913;
                int v918;
                v918 = v917 + v915;
                bool v919;
                v919 = v902[v918];
                int v920;
                if (v919){
                    v920 = 1l;
                } else {
                    v920 = 0l;
                }
                assert("Tensor range check" && 0 <= v913 && v913 < 1l);
                assert("Tensor range check" && 0 <= v915 && v915 < 4l);
                v912[v918] = v920;
                v915 += 1l ;
            }
            v913 += 1l ;
        }
        int v921;
        v921 = 0l;
        int v922;
        v922 = 0l;
        while (while_method_3(v922)){
            int v924;
            v924 = 0l;
            while (while_method_1(v924)){
                assert("Tensor range check" && 0 <= v922 && v922 < 1l);
                assert("Tensor range check" && 0 <= v924 && v924 < 4l);
                int v926;
                v926 = 4l * v922;
                int v927;
                v927 = v926 + v924;
                int v928;
                v928 = v912[v927];
                int v929;
                v929 = v921 + v928;
                v921 = v929;
                v924 += 1l ;
            }
            v922 += 1l ;
        }
        auto v930 = cooperative_groups::coalesced_threads();
        int v931;
        v931 = threadIdx.x;
        int v932;
        v932 = v931 / 4l;
        auto v933 = cooperative_groups::labeled_partition(v930,v932);
        Closure4 v934{};
        int v935;
        v935 = cooperative_groups::reduce(v933, v921, v934);
        float v936[4l];
        int v937;
        v937 = 0l;
        while (while_method_3(v937)){
            int v939;
            v939 = 0l;
            while (while_method_1(v939)){
                assert("Tensor range check" && 0 <= v937 && v937 < 1l);
                assert("Tensor range check" && 0 <= v939 && v939 < 4l);
                int v941;
                v941 = 4l * v937;
                int v942;
                v942 = v941 + v939;
                float v943;
                v943 = v857[v942];
                bool v944;
                v944 = v902[v942];
                float v945;
                if (v944){
                    v945 = v943;
                } else {
                    v945 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v937 && v937 < 1l);
                assert("Tensor range check" && 0 <= v939 && v939 < 4l);
                v936[v942] = v945;
                v939 += 1l ;
            }
            v937 += 1l ;
        }
        float v946;
        v946 = 0.0f;
        int v947;
        v947 = 0l;
        while (while_method_3(v947)){
            int v949;
            v949 = 0l;
            while (while_method_1(v949)){
                assert("Tensor range check" && 0 <= v947 && v947 < 1l);
                assert("Tensor range check" && 0 <= v949 && v949 < 4l);
                int v951;
                v951 = 4l * v947;
                int v952;
                v952 = v951 + v949;
                float v953;
                v953 = v936[v952];
                float v954;
                v954 = v946 + v953;
                v946 = v954;
                v949 += 1l ;
            }
            v947 += 1l ;
        }
        auto v955 = cooperative_groups::coalesced_threads();
        int v956;
        v956 = threadIdx.x;
        int v957;
        v957 = v956 / 4l;
        auto v958 = cooperative_groups::labeled_partition(v955,v957);
        float v959;
        v959 = cooperative_groups::reduce(v958, v946, v66);
        float v960;
        v960 = (float)v935;
        float v961;
        v961 = v959 / v960;
        float v962[4l];
        int v963;
        v963 = 0l;
        while (while_method_3(v963)){
            int v965;
            v965 = 0l;
            while (while_method_1(v965)){
                assert("Tensor range check" && 0 <= v963 && v963 < 1l);
                assert("Tensor range check" && 0 <= v965 && v965 < 4l);
                int v967;
                v967 = 4l * v963;
                int v968;
                v968 = v967 + v965;
                float v969;
                v969 = v857[v968];
                bool v970;
                v970 = v902[v968];
                float v971;
                if (v970){
                    v971 = v969;
                } else {
                    v971 = -1.0f / 0.0f;
                }
                float v972;
                v972 = v971 - v961;
                float v973;
                v973 = exp(v972);
                assert("Tensor range check" && 0 <= v963 && v963 < 1l);
                assert("Tensor range check" && 0 <= v965 && v965 < 4l);
                v962[v968] = v973;
                v965 += 1l ;
            }
            v963 += 1l ;
        }
        float v974;
        v974 = 0.0f;
        int v975;
        v975 = 0l;
        while (while_method_3(v975)){
            int v977;
            v977 = 0l;
            while (while_method_1(v977)){
                assert("Tensor range check" && 0 <= v975 && v975 < 1l);
                assert("Tensor range check" && 0 <= v977 && v977 < 4l);
                int v979;
                v979 = 4l * v975;
                int v980;
                v980 = v979 + v977;
                float v981;
                v981 = v962[v980];
                float v982;
                v982 = v974 + v981;
                v974 = v982;
                v977 += 1l ;
            }
            v975 += 1l ;
        }
        auto v983 = cooperative_groups::coalesced_threads();
        int v984;
        v984 = threadIdx.x;
        int v985;
        v985 = v984 / 4l;
        auto v986 = cooperative_groups::labeled_partition(v983,v985);
        float v987;
        v987 = cooperative_groups::reduce(v986, v974, v66);
        float v988[4l];
        int v989;
        v989 = 0l;
        while (while_method_3(v989)){
            int v991;
            v991 = 0l;
            while (while_method_1(v991)){
                assert("Tensor range check" && 0 <= v989 && v989 < 1l);
                assert("Tensor range check" && 0 <= v991 && v991 < 4l);
                int v993;
                v993 = 4l * v989;
                int v994;
                v994 = v993 + v991;
                float v995;
                v995 = v962[v994];
                bool v996;
                v996 = v987 == 0.0f;
                bool v997;
                v997 = v996 != true;
                float v999;
                if (v997){
                    float v998;
                    v998 = v995 / v987;
                    v999 = v998;
                } else {
                    v999 = 0.0625f;
                }
                assert("Tensor range check" && 0 <= v989 && v989 < 1l);
                assert("Tensor range check" && 0 <= v991 && v991 < 4l);
                v988[v994] = v999;
                v991 += 1l ;
            }
            v989 += 1l ;
        }
        assert("Tensor range check" && 0 <= v853 && v853 < 32l);
        int v1000;
        v1000 = 0l;
        while (while_method_3(v1000)){
            assert("Tensor range check" && 0 <= v1000 && v1000 < 1l);
            int v1002;
            v1002 = 16l * v1000;
            int v1003;
            v1003 = v1002 + v856;
            assert("Tensor range check" && 0 <= v1000 && v1000 < 1l);
            int v1004;
            v1004 = 4l * v1000;
            int4* v1005;
            v1005 = reinterpret_cast<int4*>(v988 + v1004);
            int4* v1006;
            v1006 = reinterpret_cast<int4*>(v6 + v1003);
            assert("Pointer alignment check" && (unsigned long long)(v1005) % 4l == 0 && (unsigned long long)(v1006) % 4l == 0);
            *v1006 = *v1005;
            v1000 += 1l ;
        }
        v853 += 1l ;
    }
    __syncthreads();
    int v1007;
    v1007 = threadIdx.x;
    bool v1008;
    v1008 = 0l <= v1007;
    bool v1009;
    v1009 = v1008 == false;
    if (v1009){
        assert("The index needs to be zero or positive." && v1008);
    } else {
    }
    int v1011;
    v1011 = v1007 % 4l;
    int v1012;
    v1012 = v1007 / 4l;
    bool v1013;
    v1013 = v1012 < 8l;
    bool v1014;
    v1014 = v1013 == false;
    if (v1014){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1013);
    } else {
    }
    assert("Tensor range check" && 0 <= v1012 && v1012 < 8l);
    assert("Tensor range check" && 0 <= v1011 && v1011 < 4l);
    int v1016;
    v1016 = 4l * v1011;
    int v1017;
    v1017 = 16l * v1012;
    int v1018;
    v1018 = v1017 + v1016;
    assert("Tensor range check" && 0 <= v1012 && v1012 < 8l);
    assert("Tensor range check" && 0 <= v1011 && v1011 < 4l);
    assert("Tensor range check" && 0 <= v1012 && v1012 < 8l);
    int v1019;
    v1019 = 0l;
    while (while_method_2(v1019)){
        assert("Tensor range check" && 0 <= v1019 && v1019 < 32l);
        int v1021;
        v1021 = 128l * v1019;
        int v1022;
        v1022 = v1021 + v1018;
        float v1023[4l];
        int v1024[4l];
        int v1025;
        v1025 = 0l;
        while (while_method_3(v1025)){
            assert("Tensor range check" && 0 <= v1025 && v1025 < 1l);
            int v1027;
            v1027 = 4l * v1025;
            assert("Tensor range check" && 0 <= v1025 && v1025 < 1l);
            int v1028;
            v1028 = 16l * v1025;
            int v1029;
            v1029 = v1028 + v1022;
            int4* v1030;
            v1030 = reinterpret_cast<int4*>(v1 + v1029);
            int4* v1031;
            v1031 = reinterpret_cast<int4*>(v1023 + v1027);
            assert("Pointer alignment check" && (unsigned long long)(v1030) % 4l == 0 && (unsigned long long)(v1031) % 4l == 0);
            *v1031 = *v1030;
            v1025 += 1l ;
        }
        int v1032;
        v1032 = 0l;
        while (while_method_3(v1032)){
            int v1034;
            v1034 = 0l;
            while (while_method_1(v1034)){
                bool v1036;
                v1036 = 0l <= v1034;
                bool v1038;
                if (v1036){
                    bool v1037;
                    v1037 = v1034 < 4l;
                    v1038 = v1037;
                } else {
                    v1038 = false;
                }
                bool v1039;
                v1039 = v1038 == false;
                if (v1039){
                    assert("The indices should be inside the range of the dimension." && v1038);
                } else {
                }
                bool v1041;
                v1041 = 0l <= v1011;
                bool v1043;
                if (v1041){
                    bool v1042;
                    v1042 = v1011 < 4l;
                    v1043 = v1042;
                } else {
                    v1043 = false;
                }
                bool v1044;
                v1044 = v1043 == false;
                if (v1044){
                    assert("The indices should be inside the range of the dimension." && v1043);
                } else {
                }
                int v1046;
                v1046 = v1011 * 4l;
                int v1047;
                v1047 = v1034 + v1046;
                bool v1048;
                v1048 = 0l <= v1032;
                bool v1050;
                if (v1048){
                    bool v1049;
                    v1049 = v1032 < 1l;
                    v1050 = v1049;
                } else {
                    v1050 = false;
                }
                bool v1051;
                v1051 = v1050 == false;
                if (v1051){
                    assert("The indices should be inside the range of the dimension." && v1050);
                } else {
                }
                int v1053;
                v1053 = v1032 * 16l;
                int v1054;
                v1054 = v1047 + v1053;
                assert("Tensor range check" && 0 <= v1032 && v1032 < 1l);
                assert("Tensor range check" && 0 <= v1034 && v1034 < 4l);
                int v1055;
                v1055 = 4l * v1032;
                int v1056;
                v1056 = v1055 + v1034;
                v1024[v1056] = v1054;
                v1034 += 1l ;
            }
            v1032 += 1l ;
        }
        bool v1057;
        v1057 = 0l <= v1012;
        bool v1058;
        v1058 = v1057 && v1013;
        bool v1059;
        v1059 = v1058 == false;
        if (v1059){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1058);
        } else {
        }
        bool v1061;
        v1061 = 0l <= v1019;
        bool v1063;
        if (v1061){
            bool v1062;
            v1062 = v1019 < 32l;
            v1063 = v1062;
        } else {
            v1063 = false;
        }
        bool v1064;
        v1064 = v1063 == false;
        if (v1064){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1063);
        } else {
        }
        int v1066;
        v1066 = v1019 * 8l;
        int v1067;
        v1067 = v1066 + v1012;
        bool v1068[4l];
        int v1069;
        v1069 = 0l;
        while (while_method_3(v1069)){
            int v1071;
            v1071 = 0l;
            while (while_method_1(v1071)){
                assert("Tensor range check" && 0 <= v1069 && v1069 < 1l);
                assert("Tensor range check" && 0 <= v1071 && v1071 < 4l);
                int v1073;
                v1073 = 4l * v1069;
                int v1074;
                v1074 = v1073 + v1071;
                float v1075;
                v1075 = v1023[v1074];
                int v1076;
                v1076 = v1024[v1074];
                bool v1077;
                v1077 = v1076 < 4l;
                assert("Tensor range check" && 0 <= v1069 && v1069 < 1l);
                assert("Tensor range check" && 0 <= v1071 && v1071 < 4l);
                v1068[v1074] = v1077;
                v1071 += 1l ;
            }
            v1069 += 1l ;
        }
        int v1078[4l];
        int v1079;
        v1079 = 0l;
        while (while_method_3(v1079)){
            int v1081;
            v1081 = 0l;
            while (while_method_1(v1081)){
                assert("Tensor range check" && 0 <= v1079 && v1079 < 1l);
                assert("Tensor range check" && 0 <= v1081 && v1081 < 4l);
                int v1083;
                v1083 = 4l * v1079;
                int v1084;
                v1084 = v1083 + v1081;
                bool v1085;
                v1085 = v1068[v1084];
                int v1086;
                if (v1085){
                    v1086 = 1l;
                } else {
                    v1086 = 0l;
                }
                assert("Tensor range check" && 0 <= v1079 && v1079 < 1l);
                assert("Tensor range check" && 0 <= v1081 && v1081 < 4l);
                v1078[v1084] = v1086;
                v1081 += 1l ;
            }
            v1079 += 1l ;
        }
        int v1087;
        v1087 = 0l;
        int v1088;
        v1088 = 0l;
        while (while_method_3(v1088)){
            int v1090;
            v1090 = 0l;
            while (while_method_1(v1090)){
                assert("Tensor range check" && 0 <= v1088 && v1088 < 1l);
                assert("Tensor range check" && 0 <= v1090 && v1090 < 4l);
                int v1092;
                v1092 = 4l * v1088;
                int v1093;
                v1093 = v1092 + v1090;
                int v1094;
                v1094 = v1078[v1093];
                int v1095;
                v1095 = v1087 + v1094;
                v1087 = v1095;
                v1090 += 1l ;
            }
            v1088 += 1l ;
        }
        auto v1096 = cooperative_groups::coalesced_threads();
        int v1097;
        v1097 = threadIdx.x;
        int v1098;
        v1098 = v1097 / 4l;
        auto v1099 = cooperative_groups::labeled_partition(v1096,v1098);
        Closure4 v1100{};
        int v1101;
        v1101 = cooperative_groups::reduce(v1099, v1087, v1100);
        float v1102[4l];
        int v1103;
        v1103 = 0l;
        while (while_method_3(v1103)){
            int v1105;
            v1105 = 0l;
            while (while_method_1(v1105)){
                assert("Tensor range check" && 0 <= v1103 && v1103 < 1l);
                assert("Tensor range check" && 0 <= v1105 && v1105 < 4l);
                int v1107;
                v1107 = 4l * v1103;
                int v1108;
                v1108 = v1107 + v1105;
                float v1109;
                v1109 = v1023[v1108];
                bool v1110;
                v1110 = v1068[v1108];
                float v1111;
                if (v1110){
                    v1111 = v1109;
                } else {
                    v1111 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1103 && v1103 < 1l);
                assert("Tensor range check" && 0 <= v1105 && v1105 < 4l);
                v1102[v1108] = v1111;
                v1105 += 1l ;
            }
            v1103 += 1l ;
        }
        float v1112;
        v1112 = 0.0f;
        int v1113;
        v1113 = 0l;
        while (while_method_3(v1113)){
            int v1115;
            v1115 = 0l;
            while (while_method_1(v1115)){
                assert("Tensor range check" && 0 <= v1113 && v1113 < 1l);
                assert("Tensor range check" && 0 <= v1115 && v1115 < 4l);
                int v1117;
                v1117 = 4l * v1113;
                int v1118;
                v1118 = v1117 + v1115;
                float v1119;
                v1119 = v1102[v1118];
                float v1120;
                v1120 = v1112 + v1119;
                v1112 = v1120;
                v1115 += 1l ;
            }
            v1113 += 1l ;
        }
        auto v1121 = cooperative_groups::coalesced_threads();
        int v1122;
        v1122 = threadIdx.x;
        int v1123;
        v1123 = v1122 / 4l;
        auto v1124 = cooperative_groups::labeled_partition(v1121,v1123);
        float v1125;
        v1125 = cooperative_groups::reduce(v1124, v1112, v66);
        float v1126;
        v1126 = (float)v1101;
        float v1127;
        v1127 = v1125 / v1126;
        float v1128[4l];
        int v1129;
        v1129 = 0l;
        while (while_method_3(v1129)){
            int v1131;
            v1131 = 0l;
            while (while_method_1(v1131)){
                assert("Tensor range check" && 0 <= v1129 && v1129 < 1l);
                assert("Tensor range check" && 0 <= v1131 && v1131 < 4l);
                int v1133;
                v1133 = 4l * v1129;
                int v1134;
                v1134 = v1133 + v1131;
                float v1135;
                v1135 = v1023[v1134];
                bool v1136;
                v1136 = v1068[v1134];
                float v1137;
                if (v1136){
                    v1137 = v1135;
                } else {
                    v1137 = -1.0f / 0.0f;
                }
                float v1138;
                v1138 = v1137 - v1127;
                float v1139;
                v1139 = exp(v1138);
                assert("Tensor range check" && 0 <= v1129 && v1129 < 1l);
                assert("Tensor range check" && 0 <= v1131 && v1131 < 4l);
                v1128[v1134] = v1139;
                v1131 += 1l ;
            }
            v1129 += 1l ;
        }
        float v1140;
        v1140 = 0.0f;
        int v1141;
        v1141 = 0l;
        while (while_method_3(v1141)){
            int v1143;
            v1143 = 0l;
            while (while_method_1(v1143)){
                assert("Tensor range check" && 0 <= v1141 && v1141 < 1l);
                assert("Tensor range check" && 0 <= v1143 && v1143 < 4l);
                int v1145;
                v1145 = 4l * v1141;
                int v1146;
                v1146 = v1145 + v1143;
                float v1147;
                v1147 = v1128[v1146];
                float v1148;
                v1148 = v1140 + v1147;
                v1140 = v1148;
                v1143 += 1l ;
            }
            v1141 += 1l ;
        }
        auto v1149 = cooperative_groups::coalesced_threads();
        int v1150;
        v1150 = threadIdx.x;
        int v1151;
        v1151 = v1150 / 4l;
        auto v1152 = cooperative_groups::labeled_partition(v1149,v1151);
        float v1153;
        v1153 = cooperative_groups::reduce(v1152, v1140, v66);
        float v1154[4l];
        int v1155;
        v1155 = 0l;
        while (while_method_3(v1155)){
            int v1157;
            v1157 = 0l;
            while (while_method_1(v1157)){
                assert("Tensor range check" && 0 <= v1155 && v1155 < 1l);
                assert("Tensor range check" && 0 <= v1157 && v1157 < 4l);
                int v1159;
                v1159 = 4l * v1155;
                int v1160;
                v1160 = v1159 + v1157;
                float v1161;
                v1161 = v1128[v1160];
                bool v1162;
                v1162 = v1153 == 0.0f;
                bool v1163;
                v1163 = v1162 != true;
                float v1165;
                if (v1163){
                    float v1164;
                    v1164 = v1161 / v1153;
                    v1165 = v1164;
                } else {
                    v1165 = 0.0625f;
                }
                assert("Tensor range check" && 0 <= v1155 && v1155 < 1l);
                assert("Tensor range check" && 0 <= v1157 && v1157 < 4l);
                v1154[v1160] = v1165;
                v1157 += 1l ;
            }
            v1155 += 1l ;
        }
        float v1166[4l];
        float v1167;
        v1167 = 0.0f;
        int v1168;
        v1168 = 0l;
        while (while_method_3(v1168)){
            assert("Tensor range check" && 0 <= v1168 && v1168 < 1l);
            int v1170;
            v1170 = 4l * v1168;
            assert("Tensor range check" && 0 <= v1168 && v1168 < 1l);
            int v1171; float v1172;
            Tuple0 tmp7 = Tuple0{0l, 0.0f};
            v1171 = tmp7.v0; v1172 = tmp7.v1;
            while (while_method_1(v1171)){
                assert("Tensor range check" && 0 <= v1171 && v1171 < 4l);
                int v1174;
                v1174 = v1171 + v1170;
                float v1175;
                v1175 = v1154[v1174];
                float v1176;
                v1176 = v1172 + v1175;
                v1172 = v1176;
                v1171 += 1l ;
            }
            auto v1177 = cooperative_groups::coalesced_threads();
            int v1178;
            v1178 = threadIdx.x;
            int v1179;
            v1179 = v1178 / 4l;
            auto v1180 = cooperative_groups::labeled_partition(v1177,v1179);
            Closure2 v1181{};
            float v1182;
            v1182 = cooperative_groups::inclusive_scan(v1180, v1172, v1181);
            float v1183;
            v1183 = v1180.shfl_up(v1182,1);
            bool v1184;
            v1184 = v1180.thread_rank() == 0;
            float v1185;
            if (v1184){
                v1185 = 0.0f;
            } else {
                v1185 = v1183;
            }
            float v1186;
            v1186 = v1180.shfl(v1182,v1180.num_threads()-1);
            float v1187;
            v1187 = v1167 + v1185;
            int v1188; float v1189;
            Tuple0 tmp8 = Tuple0{0l, v1187};
            v1188 = tmp8.v0; v1189 = tmp8.v1;
            while (while_method_1(v1188)){
                assert("Tensor range check" && 0 <= v1188 && v1188 < 4l);
                int v1191;
                v1191 = v1188 + v1170;
                float v1192;
                v1192 = v1154[v1191];
                float v1193;
                v1193 = v1189 + v1192;
                assert("Tensor range check" && 0 <= v1188 && v1188 < 4l);
                v1166[v1191] = v1193;
                v1189 = v1193;
                v1188 += 1l ;
            }
            float v1194;
            v1194 = v1167 + v1186;
            v1167 = v1194;
            v1168 += 1l ;
        }
        float v1195;
        v1195 = curand_uniform(&v17);
        float v1196[4l];
        int v1197;
        v1197 = 0l;
        while (while_method_3(v1197)){
            int v1199;
            v1199 = 0l;
            while (while_method_1(v1199)){
                assert("Tensor range check" && 0 <= v1197 && v1197 < 1l);
                assert("Tensor range check" && 0 <= v1199 && v1199 < 4l);
                int v1201;
                v1201 = 4l * v1197;
                int v1202;
                v1202 = v1201 + v1199;
                int v1203;
                v1203 = v1024[v1202];
                assert("Tensor range check" && 0 <= v1197 && v1197 < 1l);
                assert("Tensor range check" && 0 <= v1199 && v1199 < 4l);
                v1196[v1202] = v1195;
                v1199 += 1l ;
            }
            v1197 += 1l ;
        }
        float v1204;
        v1204 = 0.0f;
        int v1205;
        v1205 = 0l;
        while (while_method_3(v1205)){
            int v1207;
            v1207 = 0l;
            while (while_method_1(v1207)){
                assert("Tensor range check" && 0 <= v1205 && v1205 < 1l);
                assert("Tensor range check" && 0 <= v1207 && v1207 < 4l);
                int v1209;
                v1209 = 4l * v1205;
                int v1210;
                v1210 = v1209 + v1207;
                float v1211;
                v1211 = v1196[v1210];
                v1204 = v1211;
                v1207 += 1l ;
            }
            v1205 += 1l ;
        }
        auto v1212 = cooperative_groups::coalesced_threads();
        int v1213;
        v1213 = threadIdx.x;
        int v1214;
        v1214 = v1213 / 4l;
        auto v1215 = cooperative_groups::labeled_partition(v1212,v1214);
        Closure5 v1216{};
        float v1217;
        v1217 = cooperative_groups::reduce(v1215, v1204, v1216);
        float v1218[4l];
        int v1219;
        v1219 = 0l;
        while (while_method_3(v1219)){
            int v1221;
            v1221 = 0l;
            while (while_method_1(v1221)){
                assert("Tensor range check" && 0 <= v1219 && v1219 < 1l);
                assert("Tensor range check" && 0 <= v1221 && v1221 < 4l);
                int v1223;
                v1223 = 4l * v1219;
                int v1224;
                v1224 = v1223 + v1221;
                float v1225;
                v1225 = v1166[v1224];
                float v1226;
                v1226 = v1225 - v1217;
                assert("Tensor range check" && 0 <= v1219 && v1219 < 1l);
                assert("Tensor range check" && 0 <= v1221 && v1221 < 4l);
                v1218[v1224] = v1226;
                v1221 += 1l ;
            }
            v1219 += 1l ;
        }
        float v1227; int v1228;
        Tuple1 tmp9 = Tuple1{-1.0f / 0.0f, 0l};
        v1227 = tmp9.v0; v1228 = tmp9.v1;
        int v1229;
        v1229 = 0l;
        while (while_method_3(v1229)){
            int v1231;
            v1231 = 0l;
            while (while_method_1(v1231)){
                assert("Tensor range check" && 0 <= v1229 && v1229 < 1l);
                assert("Tensor range check" && 0 <= v1231 && v1231 < 4l);
                int v1233;
                v1233 = 4l * v1229;
                int v1234;
                v1234 = v1233 + v1231;
                float v1235;
                v1235 = v1218[v1234];
                int v1236;
                v1236 = v1024[v1234];
                bool v1237;
                v1237 = v1227 >= 0.0f;
                bool v1239;
                if (v1237){
                    bool v1238;
                    v1238 = v1235 >= 0.0f;
                    v1239 = v1238;
                } else {
                    v1239 = false;
                }
                float v1248; int v1249;
                if (v1239){
                    bool v1240;
                    v1240 = v1227 <= v1235;
                    if (v1240){
                        v1248 = v1227; v1249 = v1228;
                    } else {
                        v1248 = v1235; v1249 = v1236;
                    }
                } else {
                    if (v1237){
                        v1248 = v1227; v1249 = v1228;
                    } else {
                        bool v1243;
                        v1243 = v1235 >= 0.0f;
                        if (v1243){
                            v1248 = v1235; v1249 = v1236;
                        } else {
                            v1248 = v1227; v1249 = v1228;
                        }
                    }
                }
                v1227 = v1248;
                v1228 = v1249;
                v1231 += 1l ;
            }
            v1229 += 1l ;
        }
        auto v1250 = cooperative_groups::coalesced_threads();
        int v1251;
        v1251 = threadIdx.x;
        int v1252;
        v1252 = v1251 / 4l;
        auto v1253 = cooperative_groups::labeled_partition(v1250,v1252);
        Closure6 v1254{};
        float v1255; int v1256;
        Tuple1 tmp10 = cooperative_groups::reduce(v1253, Tuple1{v1227, v1228}, v1254);
        v1255 = tmp10.v0; v1256 = tmp10.v1;
        assert("Tensor range check" && 0 <= v1019 && v1019 < 32l);
        int v1257;
        v1257 = 0l;
        while (while_method_3(v1257)){
            assert("Tensor range check" && 0 <= v1257 && v1257 < 1l);
            int v1259;
            v1259 = 16l * v1257;
            int v1260;
            v1260 = v1259 + v1022;
            assert("Tensor range check" && 0 <= v1257 && v1257 < 1l);
            int v1261;
            v1261 = 4l * v1257;
            int4* v1262;
            v1262 = reinterpret_cast<int4*>(v1154 + v1261);
            int4* v1263;
            v1263 = reinterpret_cast<int4*>(v5 + v1260);
            assert("Pointer alignment check" && (unsigned long long)(v1262) % 4l == 0 && (unsigned long long)(v1263) % 4l == 0);
            *v1263 = *v1262;
            v1257 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1019 && v1019 < 32l);
        int v1264;
        v1264 = 8l * v1019;
        int v1265;
        v1265 = v1264 + v1012;
        v11[v1265] = v1256;
        v1019 += 1l ;
    }
    __syncthreads();
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
def method0(v0 : char) -> None:
    print(v0, end="")
    del v0
    return 
def method1(v0 : i32) -> bool:
    v1 = v0 < 256
    del v0
    return v1
def method2(v0 : string) -> None:
    print(v0, end="")
    del v0
    return 
def method3(v0 : i32) -> bool:
    v1 = v0 < 16
    del v0
    return v1
def method4(v0 : f32) -> None:
    print("{:.6f}".format(v0), end="")
    del v0
    return 
def method5() -> None:
    return 
def method6(v0 : i32) -> None:
    print(v0, end="")
    del v0
    return 
def main():
    v0 = cp.arange(0,4096,1,dtype=cp.int32) # type: ignore
    v1 = v0.size
    v2 = 4096 == v1
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = cp.random.normal(0.0,1.0,4096,dtype=cp.float32) # type: ignore
    v6 = cp.random.uniform(size=256,dtype=cp.float32) # type: ignore
    del v6
    v7 = cp.empty(1,dtype=cp.float32)
    v8 = cp.empty(4096,dtype=cp.int32)
    v9 = cp.empty(4096,dtype=cp.float32)
    v10 = cp.empty(4096,dtype=cp.float32)
    v11 = cp.empty(4096,dtype=cp.float32)
    v12 = cp.empty(4096,dtype=cp.float32)
    v13 = cp.empty(4096,dtype=cp.float32)
    v14 = cp.empty(4096,dtype=cp.float32)
    v15 = cp.empty(256,dtype=cp.int32)
    v16 = cp.empty(256,dtype=cp.int32)
    v17 = cp.empty(4096,dtype=cp.int32)
    v18 = cp.empty(4096,dtype=cp.int32)
    v19 = cp.empty(256,dtype=cp.int32)
    v20 = cp.empty(4096,dtype=cp.int32)
    v21 = 0
    v22 = raw_module.get_function(f"entry{v21}")
    del v21
    v22.max_dynamic_shared_size_bytes = 0 
    v22((1,),(32,),(v0, v5, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20),shared_mem=0)
    del v0, v5, v7, v8, v9, v11, v12, v13, v14, v15, v17, v18, v19, v20, v22
    v23 = 0
    v24 = '['
    method0(v24)
    del v24
    v25 = 0
    while method1(v25):
        v27 = v23
        v28 = v27 >= 1024
        del v27
        if v28:
            v29 = " ..."
            method2(v29)
            del v29
            break
        else:
            pass
        del v28
        v30 = v25 == 0
        v31 = v30 != True
        del v30
        if v31:
            v32 = "; "
            method2(v32)
        else:
            pass
        del v31
        v33 = '['
        method0(v33)
        del v33
        v34 = 0
        while method3(v34):
            v36 = v23
            v37 = v36 >= 1024
            del v36
            if v37:
                v38 = " ..."
                method2(v38)
                del v38
                break
            else:
                pass
            del v37
            v39 = v34 == 0
            v40 = v39 != True
            del v39
            if v40:
                v41 = "; "
                method2(v41)
            else:
                pass
            del v40
            v42 = v23 + 1
            v23 = v42
            del v42
            v43 = v25 * 16
            v44 = v43 + v34
            del v43
            v45 = v10[v44].item()
            del v44
            method4(v45)
            del v45
            v34 += 1 
        del v34
        v46 = ']'
        method0(v46)
        del v46
        v25 += 1 
    del v10, v23, v25
    v47 = ']'
    method0(v47)
    del v47
    method5()
    print()
    v48 = 0
    v49 = '['
    method0(v49)
    del v49
    v50 = 0
    while method1(v50):
        v52 = v48
        v53 = v52 >= 1024
        del v52
        if v53:
            v54 = " ..."
            method2(v54)
            del v54
            break
        else:
            pass
        del v53
        v55 = v50 == 0
        v56 = v55 != True
        del v55
        if v56:
            v57 = "; "
            method2(v57)
        else:
            pass
        del v56
        v58 = v48 + 1
        v48 = v58
        del v58
        v59 = v16[v50].item()
        method6(v59)
        del v59
        v50 += 1 
    del v16, v48, v50
    v60 = ']'
    method0(v60)
    del v60
    method5()
    print()
    return 

if __name__ == '__main__': print(main())
