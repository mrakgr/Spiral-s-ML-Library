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
struct Closure1 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v0 > v2;
        if (v4){
            return Tuple1{v0, v1};
        } else {
            return Tuple1{v2, v3};
        }
    }
};
struct Closure2 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
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
        v2 = v0 + v1;
        return v2;
    }
};
struct Closure4 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Closure5 {
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
    v1 = v0 < 16l;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 2l;
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
        v24 = v19 % 64l;
        int v25;
        v25 = v19 / 64l;
        bool v26;
        v26 = v25 < 16l;
        bool v27;
        v27 = v26 == false;
        if (v27){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v26);
        } else {
        }
        assert("Tensor range check" && 0 <= v25 && v25 < 16l);
        assert("Tensor range check" && 0 <= v24 && v24 < 64l);
        int v29;
        v29 = 4l * v24;
        int v30;
        v30 = 256l * v25;
        int v31;
        v31 = v30 + v29;
        assert("Tensor range check" && 0 <= v25 && v25 < 16l);
        assert("Tensor range check" && 0 <= v24 && v24 < 64l);
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
        v49 = v44 % 64l;
        int v50;
        v50 = v44 / 64l;
        bool v51;
        v51 = v50 < 16l;
        bool v52;
        v52 = v51 == false;
        if (v52){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v51);
        } else {
        }
        assert("Tensor range check" && 0 <= v50 && v50 < 16l);
        assert("Tensor range check" && 0 <= v49 && v49 < 64l);
        int v54;
        v54 = 4l * v49;
        int v55;
        v55 = 256l * v50;
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
    v83 = v79 % 32l;
    int v84;
    v84 = v79 / 32l;
    bool v85;
    v85 = v84 < 1l;
    bool v86;
    v86 = v85 == false;
    if (v86){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v85);
    } else {
    }
    assert("Tensor range check" && 0 <= v84 && v84 < 1l);
    assert("Tensor range check" && 0 <= v83 && v83 < 32l);
    int v88;
    v88 = 4l * v83;
    int v89;
    v89 = 256l * v84;
    int v90;
    v90 = v89 + v88;
    assert("Tensor range check" && 0 <= v84 && v84 < 1l);
    assert("Tensor range check" && 0 <= v83 && v83 < 32l);
    int v91;
    v91 = 0l;
    while (while_method_2(v91)){
        assert("Tensor range check" && 0 <= v91 && v91 < 16l);
        int v93;
        v93 = 256l * v91;
        int v94;
        v94 = v93 + v90;
        int v95[8l];
        int v96[8l];
        int v97;
        v97 = 0l;
        while (while_method_3(v97)){
            assert("Tensor range check" && 0 <= v97 && v97 < 2l);
            int v99;
            v99 = 4l * v97;
            assert("Tensor range check" && 0 <= v97 && v97 < 2l);
            int v100;
            v100 = 128l * v97;
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
                    v114 = v83 < 32l;
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
                    v121 = v104 < 2l;
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
                v125 = v104 * 128l;
                int v126;
                v126 = v119 + v125;
                assert("Tensor range check" && 0 <= v104 && v104 < 2l);
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
            v134 = v91 < 16l;
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
        v138 = v91 + v84;
        assert("Tensor range check" && 0 <= v91 && v91 < 16l);
        int v139;
        v139 = 0l;
        while (while_method_3(v139)){
            assert("Tensor range check" && 0 <= v139 && v139 < 2l);
            int v141;
            v141 = 128l * v139;
            int v142;
            v142 = v141 + v94;
            assert("Tensor range check" && 0 <= v139 && v139 < 2l);
            int v143;
            v143 = 4l * v139;
            int4* v144;
            v144 = reinterpret_cast<int4*>(v95 + v143);
            int4* v145;
            v145 = reinterpret_cast<int4*>(v3 + v142);
            assert("Pointer alignment check" && (unsigned long long)(v144) % 4l == 0 && (unsigned long long)(v145) % 4l == 0);
            *v145 = *v144;
            v139 += 1l ;
        }
        v91 += 1l ;
    }
    __syncthreads();
    int v146;
    v146 = threadIdx.x;
    bool v147;
    v147 = 0l <= v146;
    bool v148;
    v148 = v147 == false;
    if (v148){
        assert("The index needs to be zero or positive." && v147);
    } else {
    }
    int v150;
    v150 = v146 % 32l;
    int v151;
    v151 = v146 / 32l;
    bool v152;
    v152 = v151 < 1l;
    bool v153;
    v153 = v152 == false;
    if (v153){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v152);
    } else {
    }
    assert("Tensor range check" && 0 <= v151 && v151 < 1l);
    assert("Tensor range check" && 0 <= v150 && v150 < 32l);
    int v155;
    v155 = 4l * v150;
    int v156;
    v156 = 256l * v151;
    int v157;
    v157 = v156 + v155;
    assert("Tensor range check" && 0 <= v151 && v151 < 1l);
    assert("Tensor range check" && 0 <= v150 && v150 < 32l);
    int v158;
    v158 = 0l;
    while (while_method_2(v158)){
        assert("Tensor range check" && 0 <= v158 && v158 < 16l);
        int v160;
        v160 = 256l * v158;
        int v161;
        v161 = v160 + v157;
        float v162[8l];
        int v163[8l];
        int v164;
        v164 = 0l;
        while (while_method_3(v164)){
            assert("Tensor range check" && 0 <= v164 && v164 < 2l);
            int v166;
            v166 = 4l * v164;
            assert("Tensor range check" && 0 <= v164 && v164 < 2l);
            int v167;
            v167 = 128l * v164;
            int v168;
            v168 = v167 + v161;
            int4* v169;
            v169 = reinterpret_cast<int4*>(v1 + v168);
            int4* v170;
            v170 = reinterpret_cast<int4*>(v162 + v166);
            assert("Pointer alignment check" && (unsigned long long)(v169) % 4l == 0 && (unsigned long long)(v170) % 4l == 0);
            *v170 = *v169;
            v164 += 1l ;
        }
        int v171;
        v171 = 0l;
        while (while_method_3(v171)){
            int v173;
            v173 = 0l;
            while (while_method_1(v173)){
                bool v175;
                v175 = 0l <= v173;
                bool v177;
                if (v175){
                    bool v176;
                    v176 = v173 < 4l;
                    v177 = v176;
                } else {
                    v177 = false;
                }
                bool v178;
                v178 = v177 == false;
                if (v178){
                    assert("The indices should be inside the range of the dimension." && v177);
                } else {
                }
                bool v180;
                v180 = 0l <= v150;
                bool v182;
                if (v180){
                    bool v181;
                    v181 = v150 < 32l;
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
                int v185;
                v185 = v150 * 4l;
                int v186;
                v186 = v173 + v185;
                bool v187;
                v187 = 0l <= v171;
                bool v189;
                if (v187){
                    bool v188;
                    v188 = v171 < 2l;
                    v189 = v188;
                } else {
                    v189 = false;
                }
                bool v190;
                v190 = v189 == false;
                if (v190){
                    assert("The indices should be inside the range of the dimension." && v189);
                } else {
                }
                int v192;
                v192 = v171 * 128l;
                int v193;
                v193 = v186 + v192;
                assert("Tensor range check" && 0 <= v171 && v171 < 2l);
                assert("Tensor range check" && 0 <= v173 && v173 < 4l);
                int v194;
                v194 = 4l * v171;
                int v195;
                v195 = v194 + v173;
                v163[v195] = v193;
                v173 += 1l ;
            }
            v171 += 1l ;
        }
        bool v196;
        v196 = 0l <= v151;
        bool v197;
        v197 = v196 && v152;
        bool v198;
        v198 = v197 == false;
        if (v198){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v197);
        } else {
        }
        bool v200;
        v200 = 0l <= v158;
        bool v202;
        if (v200){
            bool v201;
            v201 = v158 < 16l;
            v202 = v201;
        } else {
            v202 = false;
        }
        bool v203;
        v203 = v202 == false;
        if (v203){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v202);
        } else {
        }
        int v205;
        v205 = v158 + v151;
        int v206[8l];
        int v207[8l];
        int v208;
        v208 = 0l;
        while (while_method_3(v208)){
            int v210;
            v210 = 0l;
            while (while_method_1(v210)){
                assert("Tensor range check" && 0 <= v208 && v208 < 2l);
                assert("Tensor range check" && 0 <= v210 && v210 < 4l);
                int v212;
                v212 = 4l * v208;
                int v213;
                v213 = v212 + v210;
                int v214;
                v214 = v163[v213];
                assert("Tensor range check" && 0 <= v208 && v208 < 2l);
                assert("Tensor range check" && 0 <= v210 && v210 < 4l);
                v206[v213] = v205;
                v207[v213] = v214;
                v210 += 1l ;
            }
            v208 += 1l ;
        }
        assert("Tensor range check" && 0 <= v158 && v158 < 16l);
        int v215;
        v215 = 0l;
        while (while_method_3(v215)){
            assert("Tensor range check" && 0 <= v215 && v215 < 2l);
            int v217;
            v217 = 128l * v215;
            int v218;
            v218 = v217 + v161;
            assert("Tensor range check" && 0 <= v215 && v215 < 2l);
            int v219;
            v219 = 4l * v215;
            int4* v220;
            v220 = reinterpret_cast<int4*>(v206 + v219);
            int4* v221;
            v221 = reinterpret_cast<int4*>(v12 + v218);
            assert("Pointer alignment check" && (unsigned long long)(v220) % 4l == 0 && (unsigned long long)(v221) % 4l == 0);
            *v221 = *v220;
            int4* v222;
            v222 = reinterpret_cast<int4*>(v207 + v219);
            int4* v223;
            v223 = reinterpret_cast<int4*>(v13 + v218);
            assert("Pointer alignment check" && (unsigned long long)(v222) % 4l == 0 && (unsigned long long)(v223) % 4l == 0);
            *v223 = *v222;
            v215 += 1l ;
        }
        v158 += 1l ;
    }
    __syncthreads();
    int v224;
    v224 = threadIdx.x;
    bool v225;
    v225 = 0l <= v224;
    bool v226;
    v226 = v225 == false;
    if (v226){
        assert("The index needs to be zero or positive." && v225);
    } else {
    }
    int v228;
    v228 = v224 % 32l;
    int v229;
    v229 = v224 / 32l;
    bool v230;
    v230 = v229 < 1l;
    bool v231;
    v231 = v230 == false;
    if (v231){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v230);
    } else {
    }
    assert("Tensor range check" && 0 <= v229 && v229 < 1l);
    assert("Tensor range check" && 0 <= v228 && v228 < 32l);
    int v233;
    v233 = 4l * v228;
    int v234;
    v234 = 256l * v229;
    int v235;
    v235 = v234 + v233;
    assert("Tensor range check" && 0 <= v229 && v229 < 1l);
    int v236;
    v236 = 0l;
    while (while_method_2(v236)){
        assert("Tensor range check" && 0 <= v236 && v236 < 16l);
        int v238;
        v238 = 256l * v236;
        int v239;
        v239 = v238 + v235;
        float v240[8l];
        int v241[8l];
        int v242;
        v242 = 0l;
        while (while_method_3(v242)){
            assert("Tensor range check" && 0 <= v242 && v242 < 2l);
            int v244;
            v244 = 4l * v242;
            assert("Tensor range check" && 0 <= v242 && v242 < 2l);
            int v245;
            v245 = 128l * v242;
            int v246;
            v246 = v245 + v239;
            int4* v247;
            v247 = reinterpret_cast<int4*>(v1 + v246);
            int4* v248;
            v248 = reinterpret_cast<int4*>(v240 + v244);
            assert("Pointer alignment check" && (unsigned long long)(v247) % 4l == 0 && (unsigned long long)(v248) % 4l == 0);
            *v248 = *v247;
            v242 += 1l ;
        }
        int v249;
        v249 = 0l;
        while (while_method_3(v249)){
            int v251;
            v251 = 0l;
            while (while_method_1(v251)){
                bool v253;
                v253 = 0l <= v251;
                bool v255;
                if (v253){
                    bool v254;
                    v254 = v251 < 4l;
                    v255 = v254;
                } else {
                    v255 = false;
                }
                bool v256;
                v256 = v255 == false;
                if (v256){
                    assert("The indices should be inside the range of the dimension." && v255);
                } else {
                }
                bool v258;
                v258 = 0l <= v228;
                bool v260;
                if (v258){
                    bool v259;
                    v259 = v228 < 32l;
                    v260 = v259;
                } else {
                    v260 = false;
                }
                bool v261;
                v261 = v260 == false;
                if (v261){
                    assert("The indices should be inside the range of the dimension." && v260);
                } else {
                }
                int v263;
                v263 = v228 * 4l;
                int v264;
                v264 = v251 + v263;
                bool v265;
                v265 = 0l <= v249;
                bool v267;
                if (v265){
                    bool v266;
                    v266 = v249 < 2l;
                    v267 = v266;
                } else {
                    v267 = false;
                }
                bool v268;
                v268 = v267 == false;
                if (v268){
                    assert("The indices should be inside the range of the dimension." && v267);
                } else {
                }
                int v270;
                v270 = v249 * 128l;
                int v271;
                v271 = v264 + v270;
                assert("Tensor range check" && 0 <= v249 && v249 < 2l);
                assert("Tensor range check" && 0 <= v251 && v251 < 4l);
                int v272;
                v272 = 4l * v249;
                int v273;
                v273 = v272 + v251;
                v241[v273] = v271;
                v251 += 1l ;
            }
            v249 += 1l ;
        }
        bool v274;
        v274 = 0l <= v229;
        bool v275;
        v275 = v274 && v230;
        bool v276;
        v276 = v275 == false;
        if (v276){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v275);
        } else {
        }
        bool v278;
        v278 = 0l <= v236;
        bool v280;
        if (v278){
            bool v279;
            v279 = v236 < 16l;
            v280 = v279;
        } else {
            v280 = false;
        }
        bool v281;
        v281 = v280 == false;
        if (v281){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v280);
        } else {
        }
        int v283;
        v283 = v236 + v229;
        assert("Tensor range check" && 0 <= v236 && v236 < 16l);
        v14[v283] = v283;
        v236 += 1l ;
    }
    __syncthreads();
    int v284;
    v284 = threadIdx.x;
    bool v285;
    v285 = 0l <= v284;
    bool v286;
    v286 = v285 == false;
    if (v286){
        assert("The index needs to be zero or positive." && v285);
    } else {
    }
    int v288;
    v288 = v284 % 32l;
    int v289;
    v289 = v284 / 32l;
    bool v290;
    v290 = v289 < 1l;
    bool v291;
    v291 = v290 == false;
    if (v291){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v290);
    } else {
    }
    assert("Tensor range check" && 0 <= v289 && v289 < 1l);
    assert("Tensor range check" && 0 <= v288 && v288 < 32l);
    int v293;
    v293 = 4l * v288;
    int v294;
    v294 = 256l * v289;
    int v295;
    v295 = v294 + v293;
    assert("Tensor range check" && 0 <= v289 && v289 < 1l);
    assert("Tensor range check" && 0 <= v288 && v288 < 32l);
    int v296;
    v296 = 0l;
    while (while_method_2(v296)){
        assert("Tensor range check" && 0 <= v296 && v296 < 16l);
        int v298;
        v298 = 256l * v296;
        int v299;
        v299 = v298 + v295;
        float v300[8l];
        int v301[8l];
        int v302;
        v302 = 0l;
        while (while_method_3(v302)){
            assert("Tensor range check" && 0 <= v302 && v302 < 2l);
            int v304;
            v304 = 4l * v302;
            assert("Tensor range check" && 0 <= v302 && v302 < 2l);
            int v305;
            v305 = 128l * v302;
            int v306;
            v306 = v305 + v299;
            int4* v307;
            v307 = reinterpret_cast<int4*>(v1 + v306);
            int4* v308;
            v308 = reinterpret_cast<int4*>(v300 + v304);
            assert("Pointer alignment check" && (unsigned long long)(v307) % 4l == 0 && (unsigned long long)(v308) % 4l == 0);
            *v308 = *v307;
            v302 += 1l ;
        }
        int v309;
        v309 = 0l;
        while (while_method_3(v309)){
            int v311;
            v311 = 0l;
            while (while_method_1(v311)){
                bool v313;
                v313 = 0l <= v311;
                bool v315;
                if (v313){
                    bool v314;
                    v314 = v311 < 4l;
                    v315 = v314;
                } else {
                    v315 = false;
                }
                bool v316;
                v316 = v315 == false;
                if (v316){
                    assert("The indices should be inside the range of the dimension." && v315);
                } else {
                }
                bool v318;
                v318 = 0l <= v288;
                bool v320;
                if (v318){
                    bool v319;
                    v319 = v288 < 32l;
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
                int v323;
                v323 = v288 * 4l;
                int v324;
                v324 = v311 + v323;
                bool v325;
                v325 = 0l <= v309;
                bool v327;
                if (v325){
                    bool v326;
                    v326 = v309 < 2l;
                    v327 = v326;
                } else {
                    v327 = false;
                }
                bool v328;
                v328 = v327 == false;
                if (v328){
                    assert("The indices should be inside the range of the dimension." && v327);
                } else {
                }
                int v330;
                v330 = v309 * 128l;
                int v331;
                v331 = v324 + v330;
                assert("Tensor range check" && 0 <= v309 && v309 < 2l);
                assert("Tensor range check" && 0 <= v311 && v311 < 4l);
                int v332;
                v332 = 4l * v309;
                int v333;
                v333 = v332 + v311;
                v301[v333] = v331;
                v311 += 1l ;
            }
            v309 += 1l ;
        }
        bool v334;
        v334 = 0l <= v289;
        bool v335;
        v335 = v334 && v290;
        bool v336;
        v336 = v335 == false;
        if (v336){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v335);
        } else {
        }
        bool v338;
        v338 = 0l <= v296;
        bool v340;
        if (v338){
            bool v339;
            v339 = v296 < 16l;
            v340 = v339;
        } else {
            v340 = false;
        }
        bool v341;
        v341 = v340 == false;
        if (v341){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v340);
        } else {
        }
        int v343;
        v343 = v296 + v289;
        float v344;
        v344 = 0.0f;
        int v345;
        v345 = 0l;
        while (while_method_3(v345)){
            int v347;
            v347 = 0l;
            while (while_method_1(v347)){
                assert("Tensor range check" && 0 <= v345 && v345 < 2l);
                assert("Tensor range check" && 0 <= v347 && v347 < 4l);
                int v349;
                v349 = 4l * v345;
                int v350;
                v350 = v349 + v347;
                float v351;
                v351 = v300[v350];
                float v352;
                v352 = v344 + v351;
                v344 = v352;
                v347 += 1l ;
            }
            v345 += 1l ;
        }
        auto v353 = cooperative_groups::coalesced_threads();
        int v354;
        v354 = threadIdx.x;
        int v355;
        v355 = v354 / 32l;
        auto v356 = cooperative_groups::labeled_partition(v353,v355);
        float v357;
        v357 = cooperative_groups::reduce(v356, v344, v66);
        float v358;
        v358 = v357 / 256.0f;
        float v359[8l];
        int v360;
        v360 = 0l;
        while (while_method_3(v360)){
            int v362;
            v362 = 0l;
            while (while_method_1(v362)){
                assert("Tensor range check" && 0 <= v360 && v360 < 2l);
                assert("Tensor range check" && 0 <= v362 && v362 < 4l);
                int v364;
                v364 = 4l * v360;
                int v365;
                v365 = v364 + v362;
                float v366;
                v366 = v300[v365];
                float v367;
                v367 = v366 - v358;
                float v368;
                v368 = exp(v367);
                assert("Tensor range check" && 0 <= v360 && v360 < 2l);
                assert("Tensor range check" && 0 <= v362 && v362 < 4l);
                v359[v365] = v368;
                v362 += 1l ;
            }
            v360 += 1l ;
        }
        float v369;
        v369 = 0.0f;
        int v370;
        v370 = 0l;
        while (while_method_3(v370)){
            int v372;
            v372 = 0l;
            while (while_method_1(v372)){
                assert("Tensor range check" && 0 <= v370 && v370 < 2l);
                assert("Tensor range check" && 0 <= v372 && v372 < 4l);
                int v374;
                v374 = 4l * v370;
                int v375;
                v375 = v374 + v372;
                float v376;
                v376 = v359[v375];
                float v377;
                v377 = v369 + v376;
                v369 = v377;
                v372 += 1l ;
            }
            v370 += 1l ;
        }
        auto v378 = cooperative_groups::coalesced_threads();
        int v379;
        v379 = threadIdx.x;
        int v380;
        v380 = v379 / 32l;
        auto v381 = cooperative_groups::labeled_partition(v378,v380);
        float v382;
        v382 = cooperative_groups::reduce(v381, v369, v66);
        float v383[8l];
        int v384;
        v384 = 0l;
        while (while_method_3(v384)){
            int v386;
            v386 = 0l;
            while (while_method_1(v386)){
                assert("Tensor range check" && 0 <= v384 && v384 < 2l);
                assert("Tensor range check" && 0 <= v386 && v386 < 4l);
                int v388;
                v388 = 4l * v384;
                int v389;
                v389 = v388 + v386;
                float v390;
                v390 = v359[v389];
                bool v391;
                v391 = v382 == 0.0f;
                bool v392;
                v392 = v391 != true;
                float v394;
                if (v392){
                    float v393;
                    v393 = v390 / v382;
                    v394 = v393;
                } else {
                    v394 = 0.00390625f;
                }
                assert("Tensor range check" && 0 <= v384 && v384 < 2l);
                assert("Tensor range check" && 0 <= v386 && v386 < 4l);
                v383[v389] = v394;
                v386 += 1l ;
            }
            v384 += 1l ;
        }
        assert("Tensor range check" && 0 <= v296 && v296 < 16l);
        int v395;
        v395 = 0l;
        while (while_method_3(v395)){
            assert("Tensor range check" && 0 <= v395 && v395 < 2l);
            int v397;
            v397 = 128l * v395;
            int v398;
            v398 = v397 + v299;
            assert("Tensor range check" && 0 <= v395 && v395 < 2l);
            int v399;
            v399 = 4l * v395;
            int4* v400;
            v400 = reinterpret_cast<int4*>(v383 + v399);
            int4* v401;
            v401 = reinterpret_cast<int4*>(v4 + v398);
            assert("Pointer alignment check" && (unsigned long long)(v400) % 4l == 0 && (unsigned long long)(v401) % 4l == 0);
            *v401 = *v400;
            v395 += 1l ;
        }
        v296 += 1l ;
    }
    __syncthreads();
    int v402;
    v402 = threadIdx.x;
    bool v403;
    v403 = 0l <= v402;
    bool v404;
    v404 = v403 == false;
    if (v404){
        assert("The index needs to be zero or positive." && v403);
    } else {
    }
    int v406;
    v406 = v402 % 32l;
    int v407;
    v407 = v402 / 32l;
    bool v408;
    v408 = v407 < 1l;
    bool v409;
    v409 = v408 == false;
    if (v409){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v408);
    } else {
    }
    assert("Tensor range check" && 0 <= v407 && v407 < 1l);
    assert("Tensor range check" && 0 <= v406 && v406 < 32l);
    int v411;
    v411 = 4l * v406;
    int v412;
    v412 = 256l * v407;
    int v413;
    v413 = v412 + v411;
    assert("Tensor range check" && 0 <= v407 && v407 < 1l);
    assert("Tensor range check" && 0 <= v406 && v406 < 32l);
    int v414;
    v414 = 0l;
    while (while_method_2(v414)){
        assert("Tensor range check" && 0 <= v414 && v414 < 16l);
        int v416;
        v416 = 256l * v414;
        int v417;
        v417 = v416 + v413;
        float v418[8l];
        int v419[8l];
        int v420;
        v420 = 0l;
        while (while_method_3(v420)){
            assert("Tensor range check" && 0 <= v420 && v420 < 2l);
            int v422;
            v422 = 4l * v420;
            assert("Tensor range check" && 0 <= v420 && v420 < 2l);
            int v423;
            v423 = 128l * v420;
            int v424;
            v424 = v423 + v417;
            int4* v425;
            v425 = reinterpret_cast<int4*>(v1 + v424);
            int4* v426;
            v426 = reinterpret_cast<int4*>(v418 + v422);
            assert("Pointer alignment check" && (unsigned long long)(v425) % 4l == 0 && (unsigned long long)(v426) % 4l == 0);
            *v426 = *v425;
            v420 += 1l ;
        }
        int v427;
        v427 = 0l;
        while (while_method_3(v427)){
            int v429;
            v429 = 0l;
            while (while_method_1(v429)){
                bool v431;
                v431 = 0l <= v429;
                bool v433;
                if (v431){
                    bool v432;
                    v432 = v429 < 4l;
                    v433 = v432;
                } else {
                    v433 = false;
                }
                bool v434;
                v434 = v433 == false;
                if (v434){
                    assert("The indices should be inside the range of the dimension." && v433);
                } else {
                }
                bool v436;
                v436 = 0l <= v406;
                bool v438;
                if (v436){
                    bool v437;
                    v437 = v406 < 32l;
                    v438 = v437;
                } else {
                    v438 = false;
                }
                bool v439;
                v439 = v438 == false;
                if (v439){
                    assert("The indices should be inside the range of the dimension." && v438);
                } else {
                }
                int v441;
                v441 = v406 * 4l;
                int v442;
                v442 = v429 + v441;
                bool v443;
                v443 = 0l <= v427;
                bool v445;
                if (v443){
                    bool v444;
                    v444 = v427 < 2l;
                    v445 = v444;
                } else {
                    v445 = false;
                }
                bool v446;
                v446 = v445 == false;
                if (v446){
                    assert("The indices should be inside the range of the dimension." && v445);
                } else {
                }
                int v448;
                v448 = v427 * 128l;
                int v449;
                v449 = v442 + v448;
                assert("Tensor range check" && 0 <= v427 && v427 < 2l);
                assert("Tensor range check" && 0 <= v429 && v429 < 4l);
                int v450;
                v450 = 4l * v427;
                int v451;
                v451 = v450 + v429;
                v419[v451] = v449;
                v429 += 1l ;
            }
            v427 += 1l ;
        }
        bool v452;
        v452 = 0l <= v407;
        bool v453;
        v453 = v452 && v408;
        bool v454;
        v454 = v453 == false;
        if (v454){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v453);
        } else {
        }
        bool v456;
        v456 = 0l <= v414;
        bool v458;
        if (v456){
            bool v457;
            v457 = v414 < 16l;
            v458 = v457;
        } else {
            v458 = false;
        }
        bool v459;
        v459 = v458 == false;
        if (v459){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v458);
        } else {
        }
        int v461;
        v461 = v414 + v407;
        float v462[8l];
        int v463;
        v463 = 0l;
        while (while_method_3(v463)){
            int v465;
            v465 = 0l;
            while (while_method_1(v465)){
                assert("Tensor range check" && 0 <= v463 && v463 < 2l);
                assert("Tensor range check" && 0 <= v465 && v465 < 4l);
                int v467;
                v467 = 4l * v463;
                int v468;
                v468 = v467 + v465;
                float v469;
                v469 = v418[v468];
                float v470;
                v470 = v469 * v469;
                assert("Tensor range check" && 0 <= v463 && v463 < 2l);
                assert("Tensor range check" && 0 <= v465 && v465 < 4l);
                v462[v468] = v470;
                v465 += 1l ;
            }
            v463 += 1l ;
        }
        float v471;
        v471 = 0.0f;
        int v472;
        v472 = 0l;
        while (while_method_3(v472)){
            int v474;
            v474 = 0l;
            while (while_method_1(v474)){
                assert("Tensor range check" && 0 <= v472 && v472 < 2l);
                assert("Tensor range check" && 0 <= v474 && v474 < 4l);
                int v476;
                v476 = 4l * v472;
                int v477;
                v477 = v476 + v474;
                float v478;
                v478 = v462[v477];
                float v479;
                v479 = v471 + v478;
                v471 = v479;
                v474 += 1l ;
            }
            v472 += 1l ;
        }
        auto v480 = cooperative_groups::coalesced_threads();
        int v481;
        v481 = threadIdx.x;
        int v482;
        v482 = v481 / 32l;
        auto v483 = cooperative_groups::labeled_partition(v480,v482);
        float v484;
        v484 = cooperative_groups::reduce(v483, v471, v66);
        float v485[8l];
        int v486;
        v486 = 0l;
        while (while_method_3(v486)){
            int v488;
            v488 = 0l;
            while (while_method_1(v488)){
                assert("Tensor range check" && 0 <= v486 && v486 < 2l);
                assert("Tensor range check" && 0 <= v488 && v488 < 4l);
                int v490;
                v490 = 4l * v486;
                int v491;
                v491 = v490 + v488;
                float v492;
                v492 = v418[v491];
                bool v493;
                v493 = v484 == 0.0f;
                bool v494;
                v494 = v493 != true;
                float v496;
                if (v494){
                    float v495;
                    v495 = v492 / v484;
                    v496 = v495;
                } else {
                    v496 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v486 && v486 < 2l);
                assert("Tensor range check" && 0 <= v488 && v488 < 4l);
                v485[v491] = v496;
                v488 += 1l ;
            }
            v486 += 1l ;
        }
        assert("Tensor range check" && 0 <= v414 && v414 < 16l);
        int v497;
        v497 = 0l;
        while (while_method_3(v497)){
            assert("Tensor range check" && 0 <= v497 && v497 < 2l);
            int v499;
            v499 = 128l * v497;
            int v500;
            v500 = v499 + v417;
            assert("Tensor range check" && 0 <= v497 && v497 < 2l);
            int v501;
            v501 = 4l * v497;
            int4* v502;
            v502 = reinterpret_cast<int4*>(v485 + v501);
            int4* v503;
            v503 = reinterpret_cast<int4*>(v9 + v500);
            assert("Pointer alignment check" && (unsigned long long)(v502) % 4l == 0 && (unsigned long long)(v503) % 4l == 0);
            *v503 = *v502;
            v497 += 1l ;
        }
        v414 += 1l ;
    }
    __syncthreads();
    int v504;
    v504 = threadIdx.x;
    bool v505;
    v505 = 0l <= v504;
    bool v506;
    v506 = v505 == false;
    if (v506){
        assert("The index needs to be zero or positive." && v505);
    } else {
    }
    int v508;
    v508 = v504 % 32l;
    int v509;
    v509 = v504 / 32l;
    bool v510;
    v510 = v509 < 1l;
    bool v511;
    v511 = v510 == false;
    if (v511){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v510);
    } else {
    }
    assert("Tensor range check" && 0 <= v509 && v509 < 1l);
    assert("Tensor range check" && 0 <= v508 && v508 < 32l);
    int v513;
    v513 = 4l * v508;
    int v514;
    v514 = 256l * v509;
    int v515;
    v515 = v514 + v513;
    assert("Tensor range check" && 0 <= v509 && v509 < 1l);
    int v516;
    v516 = 0l;
    while (while_method_2(v516)){
        assert("Tensor range check" && 0 <= v516 && v516 < 16l);
        int v518;
        v518 = 256l * v516;
        int v519;
        v519 = v518 + v515;
        float v520[8l];
        int v521[8l];
        int v522;
        v522 = 0l;
        while (while_method_3(v522)){
            assert("Tensor range check" && 0 <= v522 && v522 < 2l);
            int v524;
            v524 = 4l * v522;
            assert("Tensor range check" && 0 <= v522 && v522 < 2l);
            int v525;
            v525 = 128l * v522;
            int v526;
            v526 = v525 + v519;
            int4* v527;
            v527 = reinterpret_cast<int4*>(v1 + v526);
            int4* v528;
            v528 = reinterpret_cast<int4*>(v520 + v524);
            assert("Pointer alignment check" && (unsigned long long)(v527) % 4l == 0 && (unsigned long long)(v528) % 4l == 0);
            *v528 = *v527;
            v522 += 1l ;
        }
        int v529;
        v529 = 0l;
        while (while_method_3(v529)){
            int v531;
            v531 = 0l;
            while (while_method_1(v531)){
                bool v533;
                v533 = 0l <= v531;
                bool v535;
                if (v533){
                    bool v534;
                    v534 = v531 < 4l;
                    v535 = v534;
                } else {
                    v535 = false;
                }
                bool v536;
                v536 = v535 == false;
                if (v536){
                    assert("The indices should be inside the range of the dimension." && v535);
                } else {
                }
                bool v538;
                v538 = 0l <= v508;
                bool v540;
                if (v538){
                    bool v539;
                    v539 = v508 < 32l;
                    v540 = v539;
                } else {
                    v540 = false;
                }
                bool v541;
                v541 = v540 == false;
                if (v541){
                    assert("The indices should be inside the range of the dimension." && v540);
                } else {
                }
                int v543;
                v543 = v508 * 4l;
                int v544;
                v544 = v531 + v543;
                bool v545;
                v545 = 0l <= v529;
                bool v547;
                if (v545){
                    bool v546;
                    v546 = v529 < 2l;
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
                v550 = v529 * 128l;
                int v551;
                v551 = v544 + v550;
                assert("Tensor range check" && 0 <= v529 && v529 < 2l);
                assert("Tensor range check" && 0 <= v531 && v531 < 4l);
                int v552;
                v552 = 4l * v529;
                int v553;
                v553 = v552 + v531;
                v521[v553] = v551;
                v531 += 1l ;
            }
            v529 += 1l ;
        }
        bool v554;
        v554 = 0l <= v509;
        bool v555;
        v555 = v554 && v510;
        bool v556;
        v556 = v555 == false;
        if (v556){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v555);
        } else {
        }
        bool v558;
        v558 = 0l <= v516;
        bool v560;
        if (v558){
            bool v559;
            v559 = v516 < 16l;
            v560 = v559;
        } else {
            v560 = false;
        }
        bool v561;
        v561 = v560 == false;
        if (v561){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v560);
        } else {
        }
        int v563;
        v563 = v516 + v509;
        float v564; int v565;
        Tuple1 tmp1 = Tuple1{-1.0f / 0.0f, 0l};
        v564 = tmp1.v0; v565 = tmp1.v1;
        int v566;
        v566 = 0l;
        while (while_method_3(v566)){
            int v568;
            v568 = 0l;
            while (while_method_1(v568)){
                assert("Tensor range check" && 0 <= v566 && v566 < 2l);
                assert("Tensor range check" && 0 <= v568 && v568 < 4l);
                int v570;
                v570 = 4l * v566;
                int v571;
                v571 = v570 + v568;
                float v572;
                v572 = v520[v571];
                int v573;
                v573 = v521[v571];
                bool v574;
                v574 = v564 > v572;
                float v575; int v576;
                if (v574){
                    v575 = v564; v576 = v565;
                } else {
                    v575 = v572; v576 = v573;
                }
                v564 = v575;
                v565 = v576;
                v568 += 1l ;
            }
            v566 += 1l ;
        }
        auto v577 = cooperative_groups::coalesced_threads();
        int v578;
        v578 = threadIdx.x;
        int v579;
        v579 = v578 / 32l;
        auto v580 = cooperative_groups::labeled_partition(v577,v579);
        Closure1 v581{};
        float v582; int v583;
        Tuple1 tmp2 = cooperative_groups::reduce(v580, Tuple1{v564, v565}, v581);
        v582 = tmp2.v0; v583 = tmp2.v1;
        assert("Tensor range check" && 0 <= v516 && v516 < 16l);
        v10[v563] = v583;
        v516 += 1l ;
    }
    __syncthreads();
    int v584;
    v584 = threadIdx.x;
    bool v585;
    v585 = 0l <= v584;
    bool v586;
    v586 = v585 == false;
    if (v586){
        assert("The index needs to be zero or positive." && v585);
    } else {
    }
    int v588;
    v588 = v584 % 32l;
    int v589;
    v589 = v584 / 32l;
    bool v590;
    v590 = v589 < 1l;
    bool v591;
    v591 = v590 == false;
    if (v591){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v590);
    } else {
    }
    assert("Tensor range check" && 0 <= v589 && v589 < 1l);
    assert("Tensor range check" && 0 <= v588 && v588 < 32l);
    int v593;
    v593 = 4l * v588;
    int v594;
    v594 = 256l * v589;
    int v595;
    v595 = v594 + v593;
    assert("Tensor range check" && 0 <= v589 && v589 < 1l);
    assert("Tensor range check" && 0 <= v588 && v588 < 32l);
    int v596;
    v596 = 0l;
    while (while_method_2(v596)){
        assert("Tensor range check" && 0 <= v596 && v596 < 16l);
        int v598;
        v598 = 256l * v596;
        int v599;
        v599 = v598 + v595;
        float v600[8l];
        int v601[8l];
        int v602;
        v602 = 0l;
        while (while_method_3(v602)){
            assert("Tensor range check" && 0 <= v602 && v602 < 2l);
            int v604;
            v604 = 4l * v602;
            assert("Tensor range check" && 0 <= v602 && v602 < 2l);
            int v605;
            v605 = 128l * v602;
            int v606;
            v606 = v605 + v599;
            int4* v607;
            v607 = reinterpret_cast<int4*>(v1 + v606);
            int4* v608;
            v608 = reinterpret_cast<int4*>(v600 + v604);
            assert("Pointer alignment check" && (unsigned long long)(v607) % 4l == 0 && (unsigned long long)(v608) % 4l == 0);
            *v608 = *v607;
            v602 += 1l ;
        }
        int v609;
        v609 = 0l;
        while (while_method_3(v609)){
            int v611;
            v611 = 0l;
            while (while_method_1(v611)){
                bool v613;
                v613 = 0l <= v611;
                bool v615;
                if (v613){
                    bool v614;
                    v614 = v611 < 4l;
                    v615 = v614;
                } else {
                    v615 = false;
                }
                bool v616;
                v616 = v615 == false;
                if (v616){
                    assert("The indices should be inside the range of the dimension." && v615);
                } else {
                }
                bool v618;
                v618 = 0l <= v588;
                bool v620;
                if (v618){
                    bool v619;
                    v619 = v588 < 32l;
                    v620 = v619;
                } else {
                    v620 = false;
                }
                bool v621;
                v621 = v620 == false;
                if (v621){
                    assert("The indices should be inside the range of the dimension." && v620);
                } else {
                }
                int v623;
                v623 = v588 * 4l;
                int v624;
                v624 = v611 + v623;
                bool v625;
                v625 = 0l <= v609;
                bool v627;
                if (v625){
                    bool v626;
                    v626 = v609 < 2l;
                    v627 = v626;
                } else {
                    v627 = false;
                }
                bool v628;
                v628 = v627 == false;
                if (v628){
                    assert("The indices should be inside the range of the dimension." && v627);
                } else {
                }
                int v630;
                v630 = v609 * 128l;
                int v631;
                v631 = v624 + v630;
                assert("Tensor range check" && 0 <= v609 && v609 < 2l);
                assert("Tensor range check" && 0 <= v611 && v611 < 4l);
                int v632;
                v632 = 4l * v609;
                int v633;
                v633 = v632 + v611;
                v601[v633] = v631;
                v611 += 1l ;
            }
            v609 += 1l ;
        }
        bool v634;
        v634 = 0l <= v589;
        bool v635;
        v635 = v634 && v590;
        bool v636;
        v636 = v635 == false;
        if (v636){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v635);
        } else {
        }
        bool v638;
        v638 = 0l <= v596;
        bool v640;
        if (v638){
            bool v639;
            v639 = v596 < 16l;
            v640 = v639;
        } else {
            v640 = false;
        }
        bool v641;
        v641 = v640 == false;
        if (v641){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v640);
        } else {
        }
        int v643;
        v643 = v596 + v589;
        float v644;
        v644 = 0.0f;
        int v645;
        v645 = 0l;
        while (while_method_3(v645)){
            int v647;
            v647 = 0l;
            while (while_method_1(v647)){
                assert("Tensor range check" && 0 <= v645 && v645 < 2l);
                assert("Tensor range check" && 0 <= v647 && v647 < 4l);
                int v649;
                v649 = 4l * v645;
                int v650;
                v650 = v649 + v647;
                float v651;
                v651 = v600[v650];
                float v652;
                v652 = v644 + v651;
                v644 = v652;
                v647 += 1l ;
            }
            v645 += 1l ;
        }
        auto v653 = cooperative_groups::coalesced_threads();
        int v654;
        v654 = threadIdx.x;
        int v655;
        v655 = v654 / 32l;
        auto v656 = cooperative_groups::labeled_partition(v653,v655);
        float v657;
        v657 = cooperative_groups::reduce(v656, v644, v66);
        float v658;
        v658 = v657 / 256.0f;
        float v659[8l];
        int v660;
        v660 = 0l;
        while (while_method_3(v660)){
            int v662;
            v662 = 0l;
            while (while_method_1(v662)){
                assert("Tensor range check" && 0 <= v660 && v660 < 2l);
                assert("Tensor range check" && 0 <= v662 && v662 < 4l);
                int v664;
                v664 = 4l * v660;
                int v665;
                v665 = v664 + v662;
                float v666;
                v666 = v600[v665];
                float v667;
                v667 = v666 - v658;
                float v668;
                v668 = exp(v667);
                assert("Tensor range check" && 0 <= v660 && v660 < 2l);
                assert("Tensor range check" && 0 <= v662 && v662 < 4l);
                v659[v665] = v668;
                v662 += 1l ;
            }
            v660 += 1l ;
        }
        float v669;
        v669 = 0.0f;
        int v670;
        v670 = 0l;
        while (while_method_3(v670)){
            int v672;
            v672 = 0l;
            while (while_method_1(v672)){
                assert("Tensor range check" && 0 <= v670 && v670 < 2l);
                assert("Tensor range check" && 0 <= v672 && v672 < 4l);
                int v674;
                v674 = 4l * v670;
                int v675;
                v675 = v674 + v672;
                float v676;
                v676 = v659[v675];
                float v677;
                v677 = v669 + v676;
                v669 = v677;
                v672 += 1l ;
            }
            v670 += 1l ;
        }
        auto v678 = cooperative_groups::coalesced_threads();
        int v679;
        v679 = threadIdx.x;
        int v680;
        v680 = v679 / 32l;
        auto v681 = cooperative_groups::labeled_partition(v678,v680);
        float v682;
        v682 = cooperative_groups::reduce(v681, v669, v66);
        float v683[8l];
        int v684;
        v684 = 0l;
        while (while_method_3(v684)){
            int v686;
            v686 = 0l;
            while (while_method_1(v686)){
                assert("Tensor range check" && 0 <= v684 && v684 < 2l);
                assert("Tensor range check" && 0 <= v686 && v686 < 4l);
                int v688;
                v688 = 4l * v684;
                int v689;
                v689 = v688 + v686;
                float v690;
                v690 = v659[v689];
                bool v691;
                v691 = v682 == 0.0f;
                bool v692;
                v692 = v691 != true;
                float v694;
                if (v692){
                    float v693;
                    v693 = v690 / v682;
                    v694 = v693;
                } else {
                    v694 = 0.00390625f;
                }
                assert("Tensor range check" && 0 <= v684 && v684 < 2l);
                assert("Tensor range check" && 0 <= v686 && v686 < 4l);
                v683[v689] = v694;
                v686 += 1l ;
            }
            v684 += 1l ;
        }
        float v695[8l];
        float v696;
        v696 = 0.0f;
        int v697;
        v697 = 0l;
        while (while_method_3(v697)){
            assert("Tensor range check" && 0 <= v697 && v697 < 2l);
            int v699;
            v699 = 4l * v697;
            assert("Tensor range check" && 0 <= v697 && v697 < 2l);
            int v700; float v701;
            Tuple0 tmp3 = Tuple0{0l, 0.0f};
            v700 = tmp3.v0; v701 = tmp3.v1;
            while (while_method_1(v700)){
                assert("Tensor range check" && 0 <= v700 && v700 < 4l);
                int v703;
                v703 = v700 + v699;
                float v704;
                v704 = v683[v703];
                float v705;
                v705 = v701 + v704;
                v701 = v705;
                v700 += 1l ;
            }
            auto v706 = cooperative_groups::coalesced_threads();
            int v707;
            v707 = threadIdx.x;
            int v708;
            v708 = v707 / 32l;
            auto v709 = cooperative_groups::labeled_partition(v706,v708);
            Closure2 v710{};
            float v711;
            v711 = cooperative_groups::inclusive_scan(v709, v701, v710);
            float v712;
            v712 = v709.shfl_up(v711,1);
            bool v713;
            v713 = v709.thread_rank() == 0;
            float v714;
            if (v713){
                v714 = 0.0f;
            } else {
                v714 = v712;
            }
            float v715;
            v715 = v709.shfl(v711,v709.num_threads()-1);
            float v716;
            v716 = v696 + v714;
            int v717; float v718;
            Tuple0 tmp4 = Tuple0{0l, v716};
            v717 = tmp4.v0; v718 = tmp4.v1;
            while (while_method_1(v717)){
                assert("Tensor range check" && 0 <= v717 && v717 < 4l);
                int v720;
                v720 = v717 + v699;
                float v721;
                v721 = v683[v720];
                float v722;
                v722 = v718 + v721;
                assert("Tensor range check" && 0 <= v717 && v717 < 4l);
                v695[v720] = v722;
                v718 = v722;
                v717 += 1l ;
            }
            float v723;
            v723 = v696 + v715;
            v696 = v723;
            v697 += 1l ;
        }
        assert("Tensor range check" && 0 <= v596 && v596 < 16l);
        int v724;
        v724 = 0l;
        while (while_method_3(v724)){
            assert("Tensor range check" && 0 <= v724 && v724 < 2l);
            int v726;
            v726 = 128l * v724;
            int v727;
            v727 = v726 + v599;
            assert("Tensor range check" && 0 <= v724 && v724 < 2l);
            int v728;
            v728 = 4l * v724;
            int4* v729;
            v729 = reinterpret_cast<int4*>(v683 + v728);
            int4* v730;
            v730 = reinterpret_cast<int4*>(v7 + v727);
            assert("Pointer alignment check" && (unsigned long long)(v729) % 4l == 0 && (unsigned long long)(v730) % 4l == 0);
            *v730 = *v729;
            int4* v731;
            v731 = reinterpret_cast<int4*>(v695 + v728);
            int4* v732;
            v732 = reinterpret_cast<int4*>(v8 + v727);
            assert("Pointer alignment check" && (unsigned long long)(v731) % 4l == 0 && (unsigned long long)(v732) % 4l == 0);
            *v732 = *v731;
            v724 += 1l ;
        }
        v596 += 1l ;
    }
    __syncthreads();
    int v733;
    v733 = threadIdx.x;
    bool v734;
    v734 = 0l <= v733;
    bool v735;
    v735 = v734 == false;
    if (v735){
        assert("The index needs to be zero or positive." && v734);
    } else {
    }
    int v737;
    v737 = v733 % 32l;
    int v738;
    v738 = v733 / 32l;
    bool v739;
    v739 = v738 < 1l;
    bool v740;
    v740 = v739 == false;
    if (v740){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v739);
    } else {
    }
    assert("Tensor range check" && 0 <= v738 && v738 < 1l);
    assert("Tensor range check" && 0 <= v737 && v737 < 32l);
    int v742;
    v742 = 4l * v737;
    int v743;
    v743 = 256l * v738;
    int v744;
    v744 = v743 + v742;
    assert("Tensor range check" && 0 <= v738 && v738 < 1l);
    assert("Tensor range check" && 0 <= v737 && v737 < 32l);
    int v745;
    v745 = 0l;
    while (while_method_2(v745)){
        assert("Tensor range check" && 0 <= v745 && v745 < 16l);
        int v747;
        v747 = 256l * v745;
        int v748;
        v748 = v747 + v744;
        int v749[8l];
        int v750[8l];
        int v751;
        v751 = 0l;
        while (while_method_3(v751)){
            assert("Tensor range check" && 0 <= v751 && v751 < 2l);
            int v753;
            v753 = 4l * v751;
            assert("Tensor range check" && 0 <= v751 && v751 < 2l);
            int v754;
            v754 = 128l * v751;
            int v755;
            v755 = v754 + v748;
            int4* v756;
            v756 = reinterpret_cast<int4*>(v0 + v755);
            int4* v757;
            v757 = reinterpret_cast<int4*>(v749 + v753);
            assert("Pointer alignment check" && (unsigned long long)(v756) % 4l == 0 && (unsigned long long)(v757) % 4l == 0);
            *v757 = *v756;
            v751 += 1l ;
        }
        int v758;
        v758 = 0l;
        while (while_method_3(v758)){
            int v760;
            v760 = 0l;
            while (while_method_1(v760)){
                bool v762;
                v762 = 0l <= v760;
                bool v764;
                if (v762){
                    bool v763;
                    v763 = v760 < 4l;
                    v764 = v763;
                } else {
                    v764 = false;
                }
                bool v765;
                v765 = v764 == false;
                if (v765){
                    assert("The indices should be inside the range of the dimension." && v764);
                } else {
                }
                bool v767;
                v767 = 0l <= v737;
                bool v769;
                if (v767){
                    bool v768;
                    v768 = v737 < 32l;
                    v769 = v768;
                } else {
                    v769 = false;
                }
                bool v770;
                v770 = v769 == false;
                if (v770){
                    assert("The indices should be inside the range of the dimension." && v769);
                } else {
                }
                int v772;
                v772 = v737 * 4l;
                int v773;
                v773 = v760 + v772;
                bool v774;
                v774 = 0l <= v758;
                bool v776;
                if (v774){
                    bool v775;
                    v775 = v758 < 2l;
                    v776 = v775;
                } else {
                    v776 = false;
                }
                bool v777;
                v777 = v776 == false;
                if (v777){
                    assert("The indices should be inside the range of the dimension." && v776);
                } else {
                }
                int v779;
                v779 = v758 * 128l;
                int v780;
                v780 = v773 + v779;
                assert("Tensor range check" && 0 <= v758 && v758 < 2l);
                assert("Tensor range check" && 0 <= v760 && v760 < 4l);
                int v781;
                v781 = 4l * v758;
                int v782;
                v782 = v781 + v760;
                v750[v782] = v780;
                v760 += 1l ;
            }
            v758 += 1l ;
        }
        bool v783;
        v783 = 0l <= v738;
        bool v784;
        v784 = v783 && v739;
        bool v785;
        v785 = v784 == false;
        if (v785){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v784);
        } else {
        }
        bool v787;
        v787 = 0l <= v745;
        bool v789;
        if (v787){
            bool v788;
            v788 = v745 < 16l;
            v789 = v788;
        } else {
            v789 = false;
        }
        bool v790;
        v790 = v789 == false;
        if (v790){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v789);
        } else {
        }
        int v792;
        v792 = v745 + v738;
        int v793[8l];
        int v794;
        v794 = 0l;
        int v795;
        v795 = 0l;
        while (while_method_3(v795)){
            assert("Tensor range check" && 0 <= v795 && v795 < 2l);
            int v797;
            v797 = 4l * v795;
            assert("Tensor range check" && 0 <= v795 && v795 < 2l);
            int v798; int v799;
            Tuple2 tmp5 = Tuple2{0l, 0l};
            v798 = tmp5.v0; v799 = tmp5.v1;
            while (while_method_1(v798)){
                assert("Tensor range check" && 0 <= v798 && v798 < 4l);
                int v801;
                v801 = v798 + v797;
                int v802;
                v802 = v749[v801];
                int v803;
                v803 = v799 + v802;
                v799 = v803;
                v798 += 1l ;
            }
            auto v804 = cooperative_groups::coalesced_threads();
            int v805;
            v805 = threadIdx.x;
            int v806;
            v806 = v805 / 32l;
            auto v807 = cooperative_groups::labeled_partition(v804,v806);
            Closure3 v808{};
            int v809;
            v809 = cooperative_groups::inclusive_scan(v807, v799, v808);
            int v810;
            v810 = v807.shfl_up(v809,1);
            bool v811;
            v811 = v807.thread_rank() == 0;
            int v812;
            if (v811){
                v812 = 0l;
            } else {
                v812 = v810;
            }
            int v813;
            v813 = v807.shfl(v809,v807.num_threads()-1);
            int v814;
            v814 = v794 + v812;
            int v815; int v816;
            Tuple2 tmp6 = Tuple2{0l, v814};
            v815 = tmp6.v0; v816 = tmp6.v1;
            while (while_method_1(v815)){
                assert("Tensor range check" && 0 <= v815 && v815 < 4l);
                int v818;
                v818 = v815 + v797;
                int v819;
                v819 = v749[v818];
                assert("Tensor range check" && 0 <= v815 && v815 < 4l);
                v793[v818] = v816;
                int v820;
                v820 = v816 + v819;
                v816 = v820;
                v815 += 1l ;
            }
            int v821;
            v821 = v794 + v813;
            v794 = v821;
            v795 += 1l ;
        }
        assert("Tensor range check" && 0 <= v745 && v745 < 16l);
        int v822;
        v822 = 0l;
        while (while_method_3(v822)){
            assert("Tensor range check" && 0 <= v822 && v822 < 2l);
            int v824;
            v824 = 128l * v822;
            int v825;
            v825 = v824 + v748;
            assert("Tensor range check" && 0 <= v822 && v822 < 2l);
            int v826;
            v826 = 4l * v822;
            int4* v827;
            v827 = reinterpret_cast<int4*>(v793 + v826);
            int4* v828;
            v828 = reinterpret_cast<int4*>(v15 + v825);
            assert("Pointer alignment check" && (unsigned long long)(v827) % 4l == 0 && (unsigned long long)(v828) % 4l == 0);
            *v828 = *v827;
            v822 += 1l ;
        }
        v745 += 1l ;
    }
    __syncthreads();
    int v829;
    v829 = threadIdx.x;
    bool v830;
    v830 = 0l <= v829;
    bool v831;
    v831 = v830 == false;
    if (v831){
        assert("The index needs to be zero or positive." && v830);
    } else {
    }
    int v833;
    v833 = v829 % 32l;
    int v834;
    v834 = v829 / 32l;
    bool v835;
    v835 = v834 < 1l;
    bool v836;
    v836 = v835 == false;
    if (v836){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v835);
    } else {
    }
    assert("Tensor range check" && 0 <= v834 && v834 < 1l);
    assert("Tensor range check" && 0 <= v833 && v833 < 32l);
    int v838;
    v838 = 4l * v833;
    int v839;
    v839 = 256l * v834;
    int v840;
    v840 = v839 + v838;
    assert("Tensor range check" && 0 <= v834 && v834 < 1l);
    assert("Tensor range check" && 0 <= v833 && v833 < 32l);
    int v841;
    v841 = 0l;
    while (while_method_2(v841)){
        assert("Tensor range check" && 0 <= v841 && v841 < 16l);
        int v843;
        v843 = 256l * v841;
        int v844;
        v844 = v843 + v840;
        float v845[8l];
        int v846[8l];
        int v847;
        v847 = 0l;
        while (while_method_3(v847)){
            assert("Tensor range check" && 0 <= v847 && v847 < 2l);
            int v849;
            v849 = 4l * v847;
            assert("Tensor range check" && 0 <= v847 && v847 < 2l);
            int v850;
            v850 = 128l * v847;
            int v851;
            v851 = v850 + v844;
            int4* v852;
            v852 = reinterpret_cast<int4*>(v1 + v851);
            int4* v853;
            v853 = reinterpret_cast<int4*>(v845 + v849);
            assert("Pointer alignment check" && (unsigned long long)(v852) % 4l == 0 && (unsigned long long)(v853) % 4l == 0);
            *v853 = *v852;
            v847 += 1l ;
        }
        int v854;
        v854 = 0l;
        while (while_method_3(v854)){
            int v856;
            v856 = 0l;
            while (while_method_1(v856)){
                bool v858;
                v858 = 0l <= v856;
                bool v860;
                if (v858){
                    bool v859;
                    v859 = v856 < 4l;
                    v860 = v859;
                } else {
                    v860 = false;
                }
                bool v861;
                v861 = v860 == false;
                if (v861){
                    assert("The indices should be inside the range of the dimension." && v860);
                } else {
                }
                bool v863;
                v863 = 0l <= v833;
                bool v865;
                if (v863){
                    bool v864;
                    v864 = v833 < 32l;
                    v865 = v864;
                } else {
                    v865 = false;
                }
                bool v866;
                v866 = v865 == false;
                if (v866){
                    assert("The indices should be inside the range of the dimension." && v865);
                } else {
                }
                int v868;
                v868 = v833 * 4l;
                int v869;
                v869 = v856 + v868;
                bool v870;
                v870 = 0l <= v854;
                bool v872;
                if (v870){
                    bool v871;
                    v871 = v854 < 2l;
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
                int v875;
                v875 = v854 * 128l;
                int v876;
                v876 = v869 + v875;
                assert("Tensor range check" && 0 <= v854 && v854 < 2l);
                assert("Tensor range check" && 0 <= v856 && v856 < 4l);
                int v877;
                v877 = 4l * v854;
                int v878;
                v878 = v877 + v856;
                v846[v878] = v876;
                v856 += 1l ;
            }
            v854 += 1l ;
        }
        bool v879;
        v879 = 0l <= v834;
        bool v880;
        v880 = v879 && v835;
        bool v881;
        v881 = v880 == false;
        if (v881){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v880);
        } else {
        }
        bool v883;
        v883 = 0l <= v841;
        bool v885;
        if (v883){
            bool v884;
            v884 = v841 < 16l;
            v885 = v884;
        } else {
            v885 = false;
        }
        bool v886;
        v886 = v885 == false;
        if (v886){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v885);
        } else {
        }
        int v888;
        v888 = v841 + v834;
        bool v889[8l];
        int v890;
        v890 = 0l;
        while (while_method_3(v890)){
            int v892;
            v892 = 0l;
            while (while_method_1(v892)){
                assert("Tensor range check" && 0 <= v890 && v890 < 2l);
                assert("Tensor range check" && 0 <= v892 && v892 < 4l);
                int v894;
                v894 = 4l * v890;
                int v895;
                v895 = v894 + v892;
                float v896;
                v896 = v845[v895];
                int v897;
                v897 = v846[v895];
                bool v898;
                v898 = v897 < 4l;
                assert("Tensor range check" && 0 <= v890 && v890 < 2l);
                assert("Tensor range check" && 0 <= v892 && v892 < 4l);
                v889[v895] = v898;
                v892 += 1l ;
            }
            v890 += 1l ;
        }
        int v899[8l];
        int v900;
        v900 = 0l;
        while (while_method_3(v900)){
            int v902;
            v902 = 0l;
            while (while_method_1(v902)){
                assert("Tensor range check" && 0 <= v900 && v900 < 2l);
                assert("Tensor range check" && 0 <= v902 && v902 < 4l);
                int v904;
                v904 = 4l * v900;
                int v905;
                v905 = v904 + v902;
                bool v906;
                v906 = v889[v905];
                int v907;
                if (v906){
                    v907 = 1l;
                } else {
                    v907 = 0l;
                }
                assert("Tensor range check" && 0 <= v900 && v900 < 2l);
                assert("Tensor range check" && 0 <= v902 && v902 < 4l);
                v899[v905] = v907;
                v902 += 1l ;
            }
            v900 += 1l ;
        }
        int v908;
        v908 = 0l;
        int v909;
        v909 = 0l;
        while (while_method_3(v909)){
            int v911;
            v911 = 0l;
            while (while_method_1(v911)){
                assert("Tensor range check" && 0 <= v909 && v909 < 2l);
                assert("Tensor range check" && 0 <= v911 && v911 < 4l);
                int v913;
                v913 = 4l * v909;
                int v914;
                v914 = v913 + v911;
                int v915;
                v915 = v899[v914];
                int v916;
                v916 = v908 + v915;
                v908 = v916;
                v911 += 1l ;
            }
            v909 += 1l ;
        }
        auto v917 = cooperative_groups::coalesced_threads();
        int v918;
        v918 = threadIdx.x;
        int v919;
        v919 = v918 / 32l;
        auto v920 = cooperative_groups::labeled_partition(v917,v919);
        Closure4 v921{};
        int v922;
        v922 = cooperative_groups::reduce(v920, v908, v921);
        float v923[8l];
        int v924;
        v924 = 0l;
        while (while_method_3(v924)){
            int v926;
            v926 = 0l;
            while (while_method_1(v926)){
                assert("Tensor range check" && 0 <= v924 && v924 < 2l);
                assert("Tensor range check" && 0 <= v926 && v926 < 4l);
                int v928;
                v928 = 4l * v924;
                int v929;
                v929 = v928 + v926;
                float v930;
                v930 = v845[v929];
                bool v931;
                v931 = v889[v929];
                float v932;
                if (v931){
                    v932 = v930;
                } else {
                    v932 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v924 && v924 < 2l);
                assert("Tensor range check" && 0 <= v926 && v926 < 4l);
                v923[v929] = v932;
                v926 += 1l ;
            }
            v924 += 1l ;
        }
        float v933;
        v933 = 0.0f;
        int v934;
        v934 = 0l;
        while (while_method_3(v934)){
            int v936;
            v936 = 0l;
            while (while_method_1(v936)){
                assert("Tensor range check" && 0 <= v934 && v934 < 2l);
                assert("Tensor range check" && 0 <= v936 && v936 < 4l);
                int v938;
                v938 = 4l * v934;
                int v939;
                v939 = v938 + v936;
                float v940;
                v940 = v923[v939];
                float v941;
                v941 = v933 + v940;
                v933 = v941;
                v936 += 1l ;
            }
            v934 += 1l ;
        }
        auto v942 = cooperative_groups::coalesced_threads();
        int v943;
        v943 = threadIdx.x;
        int v944;
        v944 = v943 / 32l;
        auto v945 = cooperative_groups::labeled_partition(v942,v944);
        float v946;
        v946 = cooperative_groups::reduce(v945, v933, v66);
        float v947;
        v947 = (float)v922;
        float v948;
        v948 = v946 / v947;
        float v949[8l];
        int v950;
        v950 = 0l;
        while (while_method_3(v950)){
            int v952;
            v952 = 0l;
            while (while_method_1(v952)){
                assert("Tensor range check" && 0 <= v950 && v950 < 2l);
                assert("Tensor range check" && 0 <= v952 && v952 < 4l);
                int v954;
                v954 = 4l * v950;
                int v955;
                v955 = v954 + v952;
                float v956;
                v956 = v845[v955];
                bool v957;
                v957 = v889[v955];
                float v958;
                if (v957){
                    v958 = v956;
                } else {
                    v958 = -1.0f / 0.0f;
                }
                float v959;
                v959 = v958 - v948;
                float v960;
                v960 = exp(v959);
                assert("Tensor range check" && 0 <= v950 && v950 < 2l);
                assert("Tensor range check" && 0 <= v952 && v952 < 4l);
                v949[v955] = v960;
                v952 += 1l ;
            }
            v950 += 1l ;
        }
        float v961;
        v961 = 0.0f;
        int v962;
        v962 = 0l;
        while (while_method_3(v962)){
            int v964;
            v964 = 0l;
            while (while_method_1(v964)){
                assert("Tensor range check" && 0 <= v962 && v962 < 2l);
                assert("Tensor range check" && 0 <= v964 && v964 < 4l);
                int v966;
                v966 = 4l * v962;
                int v967;
                v967 = v966 + v964;
                float v968;
                v968 = v949[v967];
                float v969;
                v969 = v961 + v968;
                v961 = v969;
                v964 += 1l ;
            }
            v962 += 1l ;
        }
        auto v970 = cooperative_groups::coalesced_threads();
        int v971;
        v971 = threadIdx.x;
        int v972;
        v972 = v971 / 32l;
        auto v973 = cooperative_groups::labeled_partition(v970,v972);
        float v974;
        v974 = cooperative_groups::reduce(v973, v961, v66);
        float v975[8l];
        int v976;
        v976 = 0l;
        while (while_method_3(v976)){
            int v978;
            v978 = 0l;
            while (while_method_1(v978)){
                assert("Tensor range check" && 0 <= v976 && v976 < 2l);
                assert("Tensor range check" && 0 <= v978 && v978 < 4l);
                int v980;
                v980 = 4l * v976;
                int v981;
                v981 = v980 + v978;
                float v982;
                v982 = v949[v981];
                bool v983;
                v983 = v974 == 0.0f;
                bool v984;
                v984 = v983 != true;
                float v986;
                if (v984){
                    float v985;
                    v985 = v982 / v974;
                    v986 = v985;
                } else {
                    v986 = 0.00390625f;
                }
                assert("Tensor range check" && 0 <= v976 && v976 < 2l);
                assert("Tensor range check" && 0 <= v978 && v978 < 4l);
                v975[v981] = v986;
                v978 += 1l ;
            }
            v976 += 1l ;
        }
        assert("Tensor range check" && 0 <= v841 && v841 < 16l);
        int v987;
        v987 = 0l;
        while (while_method_3(v987)){
            assert("Tensor range check" && 0 <= v987 && v987 < 2l);
            int v989;
            v989 = 128l * v987;
            int v990;
            v990 = v989 + v844;
            assert("Tensor range check" && 0 <= v987 && v987 < 2l);
            int v991;
            v991 = 4l * v987;
            int4* v992;
            v992 = reinterpret_cast<int4*>(v975 + v991);
            int4* v993;
            v993 = reinterpret_cast<int4*>(v6 + v990);
            assert("Pointer alignment check" && (unsigned long long)(v992) % 4l == 0 && (unsigned long long)(v993) % 4l == 0);
            *v993 = *v992;
            v987 += 1l ;
        }
        v841 += 1l ;
    }
    __syncthreads();
    int v994;
    v994 = threadIdx.x;
    bool v995;
    v995 = 0l <= v994;
    bool v996;
    v996 = v995 == false;
    if (v996){
        assert("The index needs to be zero or positive." && v995);
    } else {
    }
    int v998;
    v998 = v994 % 32l;
    int v999;
    v999 = v994 / 32l;
    bool v1000;
    v1000 = v999 < 1l;
    bool v1001;
    v1001 = v1000 == false;
    if (v1001){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1000);
    } else {
    }
    assert("Tensor range check" && 0 <= v999 && v999 < 1l);
    assert("Tensor range check" && 0 <= v998 && v998 < 32l);
    int v1003;
    v1003 = 4l * v998;
    int v1004;
    v1004 = 256l * v999;
    int v1005;
    v1005 = v1004 + v1003;
    assert("Tensor range check" && 0 <= v999 && v999 < 1l);
    assert("Tensor range check" && 0 <= v998 && v998 < 32l);
    assert("Tensor range check" && 0 <= v999 && v999 < 1l);
    int v1006;
    v1006 = 0l;
    while (while_method_2(v1006)){
        assert("Tensor range check" && 0 <= v1006 && v1006 < 16l);
        int v1008;
        v1008 = 256l * v1006;
        int v1009;
        v1009 = v1008 + v1005;
        float v1010[8l];
        int v1011[8l];
        int v1012;
        v1012 = 0l;
        while (while_method_3(v1012)){
            assert("Tensor range check" && 0 <= v1012 && v1012 < 2l);
            int v1014;
            v1014 = 4l * v1012;
            assert("Tensor range check" && 0 <= v1012 && v1012 < 2l);
            int v1015;
            v1015 = 128l * v1012;
            int v1016;
            v1016 = v1015 + v1009;
            int4* v1017;
            v1017 = reinterpret_cast<int4*>(v1 + v1016);
            int4* v1018;
            v1018 = reinterpret_cast<int4*>(v1010 + v1014);
            assert("Pointer alignment check" && (unsigned long long)(v1017) % 4l == 0 && (unsigned long long)(v1018) % 4l == 0);
            *v1018 = *v1017;
            v1012 += 1l ;
        }
        int v1019;
        v1019 = 0l;
        while (while_method_3(v1019)){
            int v1021;
            v1021 = 0l;
            while (while_method_1(v1021)){
                bool v1023;
                v1023 = 0l <= v1021;
                bool v1025;
                if (v1023){
                    bool v1024;
                    v1024 = v1021 < 4l;
                    v1025 = v1024;
                } else {
                    v1025 = false;
                }
                bool v1026;
                v1026 = v1025 == false;
                if (v1026){
                    assert("The indices should be inside the range of the dimension." && v1025);
                } else {
                }
                bool v1028;
                v1028 = 0l <= v998;
                bool v1030;
                if (v1028){
                    bool v1029;
                    v1029 = v998 < 32l;
                    v1030 = v1029;
                } else {
                    v1030 = false;
                }
                bool v1031;
                v1031 = v1030 == false;
                if (v1031){
                    assert("The indices should be inside the range of the dimension." && v1030);
                } else {
                }
                int v1033;
                v1033 = v998 * 4l;
                int v1034;
                v1034 = v1021 + v1033;
                bool v1035;
                v1035 = 0l <= v1019;
                bool v1037;
                if (v1035){
                    bool v1036;
                    v1036 = v1019 < 2l;
                    v1037 = v1036;
                } else {
                    v1037 = false;
                }
                bool v1038;
                v1038 = v1037 == false;
                if (v1038){
                    assert("The indices should be inside the range of the dimension." && v1037);
                } else {
                }
                int v1040;
                v1040 = v1019 * 128l;
                int v1041;
                v1041 = v1034 + v1040;
                assert("Tensor range check" && 0 <= v1019 && v1019 < 2l);
                assert("Tensor range check" && 0 <= v1021 && v1021 < 4l);
                int v1042;
                v1042 = 4l * v1019;
                int v1043;
                v1043 = v1042 + v1021;
                v1011[v1043] = v1041;
                v1021 += 1l ;
            }
            v1019 += 1l ;
        }
        bool v1044;
        v1044 = 0l <= v999;
        bool v1045;
        v1045 = v1044 && v1000;
        bool v1046;
        v1046 = v1045 == false;
        if (v1046){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1045);
        } else {
        }
        bool v1048;
        v1048 = 0l <= v1006;
        bool v1050;
        if (v1048){
            bool v1049;
            v1049 = v1006 < 16l;
            v1050 = v1049;
        } else {
            v1050 = false;
        }
        bool v1051;
        v1051 = v1050 == false;
        if (v1051){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1050);
        } else {
        }
        int v1053;
        v1053 = v1006 + v999;
        bool v1054[8l];
        int v1055;
        v1055 = 0l;
        while (while_method_3(v1055)){
            int v1057;
            v1057 = 0l;
            while (while_method_1(v1057)){
                assert("Tensor range check" && 0 <= v1055 && v1055 < 2l);
                assert("Tensor range check" && 0 <= v1057 && v1057 < 4l);
                int v1059;
                v1059 = 4l * v1055;
                int v1060;
                v1060 = v1059 + v1057;
                float v1061;
                v1061 = v1010[v1060];
                int v1062;
                v1062 = v1011[v1060];
                bool v1063;
                v1063 = v1062 < 4l;
                assert("Tensor range check" && 0 <= v1055 && v1055 < 2l);
                assert("Tensor range check" && 0 <= v1057 && v1057 < 4l);
                v1054[v1060] = v1063;
                v1057 += 1l ;
            }
            v1055 += 1l ;
        }
        int v1064[8l];
        int v1065;
        v1065 = 0l;
        while (while_method_3(v1065)){
            int v1067;
            v1067 = 0l;
            while (while_method_1(v1067)){
                assert("Tensor range check" && 0 <= v1065 && v1065 < 2l);
                assert("Tensor range check" && 0 <= v1067 && v1067 < 4l);
                int v1069;
                v1069 = 4l * v1065;
                int v1070;
                v1070 = v1069 + v1067;
                bool v1071;
                v1071 = v1054[v1070];
                int v1072;
                if (v1071){
                    v1072 = 1l;
                } else {
                    v1072 = 0l;
                }
                assert("Tensor range check" && 0 <= v1065 && v1065 < 2l);
                assert("Tensor range check" && 0 <= v1067 && v1067 < 4l);
                v1064[v1070] = v1072;
                v1067 += 1l ;
            }
            v1065 += 1l ;
        }
        int v1073;
        v1073 = 0l;
        int v1074;
        v1074 = 0l;
        while (while_method_3(v1074)){
            int v1076;
            v1076 = 0l;
            while (while_method_1(v1076)){
                assert("Tensor range check" && 0 <= v1074 && v1074 < 2l);
                assert("Tensor range check" && 0 <= v1076 && v1076 < 4l);
                int v1078;
                v1078 = 4l * v1074;
                int v1079;
                v1079 = v1078 + v1076;
                int v1080;
                v1080 = v1064[v1079];
                int v1081;
                v1081 = v1073 + v1080;
                v1073 = v1081;
                v1076 += 1l ;
            }
            v1074 += 1l ;
        }
        auto v1082 = cooperative_groups::coalesced_threads();
        int v1083;
        v1083 = threadIdx.x;
        int v1084;
        v1084 = v1083 / 32l;
        auto v1085 = cooperative_groups::labeled_partition(v1082,v1084);
        Closure4 v1086{};
        int v1087;
        v1087 = cooperative_groups::reduce(v1085, v1073, v1086);
        float v1088[8l];
        int v1089;
        v1089 = 0l;
        while (while_method_3(v1089)){
            int v1091;
            v1091 = 0l;
            while (while_method_1(v1091)){
                assert("Tensor range check" && 0 <= v1089 && v1089 < 2l);
                assert("Tensor range check" && 0 <= v1091 && v1091 < 4l);
                int v1093;
                v1093 = 4l * v1089;
                int v1094;
                v1094 = v1093 + v1091;
                float v1095;
                v1095 = v1010[v1094];
                bool v1096;
                v1096 = v1054[v1094];
                float v1097;
                if (v1096){
                    v1097 = v1095;
                } else {
                    v1097 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1089 && v1089 < 2l);
                assert("Tensor range check" && 0 <= v1091 && v1091 < 4l);
                v1088[v1094] = v1097;
                v1091 += 1l ;
            }
            v1089 += 1l ;
        }
        float v1098;
        v1098 = 0.0f;
        int v1099;
        v1099 = 0l;
        while (while_method_3(v1099)){
            int v1101;
            v1101 = 0l;
            while (while_method_1(v1101)){
                assert("Tensor range check" && 0 <= v1099 && v1099 < 2l);
                assert("Tensor range check" && 0 <= v1101 && v1101 < 4l);
                int v1103;
                v1103 = 4l * v1099;
                int v1104;
                v1104 = v1103 + v1101;
                float v1105;
                v1105 = v1088[v1104];
                float v1106;
                v1106 = v1098 + v1105;
                v1098 = v1106;
                v1101 += 1l ;
            }
            v1099 += 1l ;
        }
        auto v1107 = cooperative_groups::coalesced_threads();
        int v1108;
        v1108 = threadIdx.x;
        int v1109;
        v1109 = v1108 / 32l;
        auto v1110 = cooperative_groups::labeled_partition(v1107,v1109);
        float v1111;
        v1111 = cooperative_groups::reduce(v1110, v1098, v66);
        float v1112;
        v1112 = (float)v1087;
        float v1113;
        v1113 = v1111 / v1112;
        float v1114[8l];
        int v1115;
        v1115 = 0l;
        while (while_method_3(v1115)){
            int v1117;
            v1117 = 0l;
            while (while_method_1(v1117)){
                assert("Tensor range check" && 0 <= v1115 && v1115 < 2l);
                assert("Tensor range check" && 0 <= v1117 && v1117 < 4l);
                int v1119;
                v1119 = 4l * v1115;
                int v1120;
                v1120 = v1119 + v1117;
                float v1121;
                v1121 = v1010[v1120];
                bool v1122;
                v1122 = v1054[v1120];
                float v1123;
                if (v1122){
                    v1123 = v1121;
                } else {
                    v1123 = -1.0f / 0.0f;
                }
                float v1124;
                v1124 = v1123 - v1113;
                float v1125;
                v1125 = exp(v1124);
                assert("Tensor range check" && 0 <= v1115 && v1115 < 2l);
                assert("Tensor range check" && 0 <= v1117 && v1117 < 4l);
                v1114[v1120] = v1125;
                v1117 += 1l ;
            }
            v1115 += 1l ;
        }
        float v1126;
        v1126 = 0.0f;
        int v1127;
        v1127 = 0l;
        while (while_method_3(v1127)){
            int v1129;
            v1129 = 0l;
            while (while_method_1(v1129)){
                assert("Tensor range check" && 0 <= v1127 && v1127 < 2l);
                assert("Tensor range check" && 0 <= v1129 && v1129 < 4l);
                int v1131;
                v1131 = 4l * v1127;
                int v1132;
                v1132 = v1131 + v1129;
                float v1133;
                v1133 = v1114[v1132];
                float v1134;
                v1134 = v1126 + v1133;
                v1126 = v1134;
                v1129 += 1l ;
            }
            v1127 += 1l ;
        }
        auto v1135 = cooperative_groups::coalesced_threads();
        int v1136;
        v1136 = threadIdx.x;
        int v1137;
        v1137 = v1136 / 32l;
        auto v1138 = cooperative_groups::labeled_partition(v1135,v1137);
        float v1139;
        v1139 = cooperative_groups::reduce(v1138, v1126, v66);
        float v1140[8l];
        int v1141;
        v1141 = 0l;
        while (while_method_3(v1141)){
            int v1143;
            v1143 = 0l;
            while (while_method_1(v1143)){
                assert("Tensor range check" && 0 <= v1141 && v1141 < 2l);
                assert("Tensor range check" && 0 <= v1143 && v1143 < 4l);
                int v1145;
                v1145 = 4l * v1141;
                int v1146;
                v1146 = v1145 + v1143;
                float v1147;
                v1147 = v1114[v1146];
                bool v1148;
                v1148 = v1139 == 0.0f;
                bool v1149;
                v1149 = v1148 != true;
                float v1151;
                if (v1149){
                    float v1150;
                    v1150 = v1147 / v1139;
                    v1151 = v1150;
                } else {
                    v1151 = 0.00390625f;
                }
                assert("Tensor range check" && 0 <= v1141 && v1141 < 2l);
                assert("Tensor range check" && 0 <= v1143 && v1143 < 4l);
                v1140[v1146] = v1151;
                v1143 += 1l ;
            }
            v1141 += 1l ;
        }
        float v1152[8l];
        float v1153;
        v1153 = 0.0f;
        int v1154;
        v1154 = 0l;
        while (while_method_3(v1154)){
            assert("Tensor range check" && 0 <= v1154 && v1154 < 2l);
            int v1156;
            v1156 = 4l * v1154;
            assert("Tensor range check" && 0 <= v1154 && v1154 < 2l);
            int v1157; float v1158;
            Tuple0 tmp7 = Tuple0{0l, 0.0f};
            v1157 = tmp7.v0; v1158 = tmp7.v1;
            while (while_method_1(v1157)){
                assert("Tensor range check" && 0 <= v1157 && v1157 < 4l);
                int v1160;
                v1160 = v1157 + v1156;
                float v1161;
                v1161 = v1140[v1160];
                float v1162;
                v1162 = v1158 + v1161;
                v1158 = v1162;
                v1157 += 1l ;
            }
            auto v1163 = cooperative_groups::coalesced_threads();
            int v1164;
            v1164 = threadIdx.x;
            int v1165;
            v1165 = v1164 / 32l;
            auto v1166 = cooperative_groups::labeled_partition(v1163,v1165);
            Closure2 v1167{};
            float v1168;
            v1168 = cooperative_groups::inclusive_scan(v1166, v1158, v1167);
            float v1169;
            v1169 = v1166.shfl_up(v1168,1);
            bool v1170;
            v1170 = v1166.thread_rank() == 0;
            float v1171;
            if (v1170){
                v1171 = 0.0f;
            } else {
                v1171 = v1169;
            }
            float v1172;
            v1172 = v1166.shfl(v1168,v1166.num_threads()-1);
            float v1173;
            v1173 = v1153 + v1171;
            int v1174; float v1175;
            Tuple0 tmp8 = Tuple0{0l, v1173};
            v1174 = tmp8.v0; v1175 = tmp8.v1;
            while (while_method_1(v1174)){
                assert("Tensor range check" && 0 <= v1174 && v1174 < 4l);
                int v1177;
                v1177 = v1174 + v1156;
                float v1178;
                v1178 = v1140[v1177];
                float v1179;
                v1179 = v1175 + v1178;
                assert("Tensor range check" && 0 <= v1174 && v1174 < 4l);
                v1152[v1177] = v1179;
                v1175 = v1179;
                v1174 += 1l ;
            }
            float v1180;
            v1180 = v1153 + v1172;
            v1153 = v1180;
            v1154 += 1l ;
        }
        float v1181;
        v1181 = curand_uniform(&v17);
        float v1182[8l];
        int v1183;
        v1183 = 0l;
        while (while_method_3(v1183)){
            int v1185;
            v1185 = 0l;
            while (while_method_1(v1185)){
                assert("Tensor range check" && 0 <= v1183 && v1183 < 2l);
                assert("Tensor range check" && 0 <= v1185 && v1185 < 4l);
                int v1187;
                v1187 = 4l * v1183;
                int v1188;
                v1188 = v1187 + v1185;
                float v1189;
                v1189 = v1152[v1188];
                float v1190;
                v1190 = v1189 - v1181;
                assert("Tensor range check" && 0 <= v1183 && v1183 < 2l);
                assert("Tensor range check" && 0 <= v1185 && v1185 < 4l);
                v1182[v1188] = v1190;
                v1185 += 1l ;
            }
            v1183 += 1l ;
        }
        float v1191; int v1192;
        Tuple1 tmp9 = Tuple1{-1.0f / 0.0f, 0l};
        v1191 = tmp9.v0; v1192 = tmp9.v1;
        int v1193;
        v1193 = 0l;
        while (while_method_3(v1193)){
            int v1195;
            v1195 = 0l;
            while (while_method_1(v1195)){
                assert("Tensor range check" && 0 <= v1193 && v1193 < 2l);
                assert("Tensor range check" && 0 <= v1195 && v1195 < 4l);
                int v1197;
                v1197 = 4l * v1193;
                int v1198;
                v1198 = v1197 + v1195;
                float v1199;
                v1199 = v1182[v1198];
                int v1200;
                v1200 = v1011[v1198];
                bool v1201;
                v1201 = v1191 >= 0.0f;
                bool v1203;
                if (v1201){
                    bool v1202;
                    v1202 = v1199 >= 0.0f;
                    v1203 = v1202;
                } else {
                    v1203 = false;
                }
                float v1212; int v1213;
                if (v1203){
                    bool v1204;
                    v1204 = v1191 <= v1199;
                    if (v1204){
                        v1212 = v1191; v1213 = v1192;
                    } else {
                        v1212 = v1199; v1213 = v1200;
                    }
                } else {
                    if (v1201){
                        v1212 = v1191; v1213 = v1192;
                    } else {
                        bool v1207;
                        v1207 = v1199 >= 0.0f;
                        if (v1207){
                            v1212 = v1199; v1213 = v1200;
                        } else {
                            v1212 = v1191; v1213 = v1192;
                        }
                    }
                }
                v1191 = v1212;
                v1192 = v1213;
                v1195 += 1l ;
            }
            v1193 += 1l ;
        }
        auto v1214 = cooperative_groups::coalesced_threads();
        int v1215;
        v1215 = threadIdx.x;
        int v1216;
        v1216 = v1215 / 32l;
        auto v1217 = cooperative_groups::labeled_partition(v1214,v1216);
        Closure5 v1218{};
        float v1219; int v1220;
        Tuple1 tmp10 = cooperative_groups::reduce(v1217, Tuple1{v1191, v1192}, v1218);
        v1219 = tmp10.v0; v1220 = tmp10.v1;
        assert("Tensor range check" && 0 <= v1006 && v1006 < 16l);
        int v1221;
        v1221 = 0l;
        while (while_method_3(v1221)){
            assert("Tensor range check" && 0 <= v1221 && v1221 < 2l);
            int v1223;
            v1223 = 128l * v1221;
            int v1224;
            v1224 = v1223 + v1009;
            assert("Tensor range check" && 0 <= v1221 && v1221 < 2l);
            int v1225;
            v1225 = 4l * v1221;
            int4* v1226;
            v1226 = reinterpret_cast<int4*>(v1140 + v1225);
            int4* v1227;
            v1227 = reinterpret_cast<int4*>(v5 + v1224);
            assert("Pointer alignment check" && (unsigned long long)(v1226) % 4l == 0 && (unsigned long long)(v1227) % 4l == 0);
            *v1227 = *v1226;
            v1221 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1006 && v1006 < 16l);
        v11[v1053] = v1220;
        v1006 += 1l ;
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
    v1 = v0 < 16
    del v0
    return v1
def method2(v0 : string) -> None:
    print(v0, end="")
    del v0
    return 
def method3(v0 : i32) -> bool:
    v1 = v0 < 256
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
    v6 = cp.random.uniform(size=16,dtype=cp.float32) # type: ignore
    del v6
    v7 = cp.empty(1,dtype=cp.float32)
    v8 = cp.empty(4096,dtype=cp.int32)
    v9 = cp.empty(4096,dtype=cp.float32)
    v10 = cp.empty(4096,dtype=cp.float32)
    v11 = cp.empty(4096,dtype=cp.float32)
    v12 = cp.empty(4096,dtype=cp.float32)
    v13 = cp.empty(4096,dtype=cp.float32)
    v14 = cp.empty(4096,dtype=cp.float32)
    v15 = cp.empty(16,dtype=cp.int32)
    v16 = cp.empty(16,dtype=cp.int32)
    v17 = cp.empty(4096,dtype=cp.int32)
    v18 = cp.empty(4096,dtype=cp.int32)
    v19 = cp.empty(16,dtype=cp.int32)
    v20 = cp.empty(4096,dtype=cp.int32)
    v21 = 0
    v22 = raw_module.get_function(f"entry{v21}")
    del v21
    v22.max_dynamic_shared_size_bytes = 0 
    v22((1,),(32,),(v0, v5, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20),shared_mem=0)
    del v0, v5, v7, v8, v9, v12, v13, v14, v15, v17, v18, v19, v20, v22
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
            v43 = v25 * 256
            v44 = v43 + v34
            del v43
            v45 = v11[v44].item()
            del v44
            method4(v45)
            del v45
            v34 += 1 
        del v34
        v46 = ']'
        method0(v46)
        del v46
        v25 += 1 
    del v11, v23, v25
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
        v58 = '['
        method0(v58)
        del v58
        v59 = 0
        while method3(v59):
            v61 = v48
            v62 = v61 >= 1024
            del v61
            if v62:
                v63 = " ..."
                method2(v63)
                del v63
                break
            else:
                pass
            del v62
            v64 = v59 == 0
            v65 = v64 != True
            del v64
            if v65:
                v66 = "; "
                method2(v66)
            else:
                pass
            del v65
            v67 = v48 + 1
            v48 = v67
            del v67
            v68 = v50 * 256
            v69 = v68 + v59
            del v68
            v70 = v10[v69].item()
            del v69
            method4(v70)
            del v70
            v59 += 1 
        del v59
        v71 = ']'
        method0(v71)
        del v71
        v50 += 1 
    del v10, v48, v50
    v72 = ']'
    method0(v72)
    del v72
    method5()
    print()
    v73 = 0
    v74 = '['
    method0(v74)
    del v74
    v75 = 0
    while method1(v75):
        v77 = v73
        v78 = v77 >= 1024
        del v77
        if v78:
            v79 = " ..."
            method2(v79)
            del v79
            break
        else:
            pass
        del v78
        v80 = v75 == 0
        v81 = v80 != True
        del v80
        if v81:
            v82 = "; "
            method2(v82)
        else:
            pass
        del v81
        v83 = v73 + 1
        v73 = v83
        del v83
        v84 = v16[v75].item()
        method6(v84)
        del v84
        v75 += 1 
    del v16, v73, v75
    v85 = ']'
    method0(v85)
    del v85
    method5()
    print()
    return 

if __name__ == '__main__': print(main())
