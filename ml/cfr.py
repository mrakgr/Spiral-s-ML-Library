kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
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
struct Tuple3;
struct Tuple4;
__device__ Tuple0 method_1(float * v0, float * v1, float * v2, float * v3, float * v4, int v5);
__device__ void push_0(float * v0, float * v1, float * v2, float * v3, float * v4, int * v5, float * v6, int * v7, int * v8, double * v9, double * v10, float * v11, float * v12, float * v13, int v14, int & v15, double * v16, double * v17, int v18, int v19);
struct Tuple0 {
    float v0;
    float v1;
    int v2;
    __device__ Tuple0() = default;
    __device__ Tuple0(float t0, float t1, int t2) : v0(t0), v1(t1), v2(t2) {}
};
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
struct Tuple1 {
    int v0;
    float v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(int t0, float t1) : v0(t0), v1(t1) {}
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
    float v0;
    bool v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(float t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ Tuple2 operator()(Tuple2 tup0, Tuple2 tup1){
        float v0 = tup0.v0; bool v1 = tup0.v1; float v2 = tup1.v0; bool v3 = tup1.v1;
        if (v1){
            if (v3){
                bool v4;
                v4 = v0 >= v2;
                float v5;
                if (v4){
                    v5 = v0;
                } else {
                    v5 = v2;
                }
                return Tuple2{v5, true};
            } else {
                return Tuple2{v0, v1};
            }
        } else {
            if (v3){
                return Tuple2{v2, v3};
            } else {
                return Tuple2{v0, v1};
            }
        }
    }
};
struct Tuple3 {
    float v0;
    int v1;
    __device__ Tuple3() = default;
    __device__ Tuple3(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
    __device__ Tuple3 operator()(Tuple3 tup0, Tuple3 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple3{v0, v1};
        } else {
            return Tuple3{v2, v3};
        }
    }
};
struct Tuple4 {
    int v0;
    bool v1;
    __device__ Tuple4() = default;
    __device__ Tuple4(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure5 {
    __device__ Tuple4 operator()(Tuple4 tup0, Tuple4 tup1){
        int v0 = tup0.v0; bool v1 = tup0.v1; int v2 = tup1.v0; bool v3 = tup1.v1;
        if (v1){
            if (v3){
                bool v4;
                v4 = v0 < v2;
                int v5;
                if (v4){
                    v5 = v0;
                } else {
                    v5 = v2;
                }
                return Tuple4{v5, true};
            } else {
                return Tuple4{v0, v1};
            }
        } else {
            if (v3){
                return Tuple4{v2, v3};
            } else {
                return Tuple4{v0, v1};
            }
        }
    }
};
struct Closure6 {
    int v0;
    __device__ Tuple3 operator()(Tuple3 tup0, Tuple3 tup1){
        int & v0 = this->v0;
        float v1 = tup0.v0; int v2 = tup0.v1; float v3 = tup1.v0; int v4 = tup1.v1;
        bool v5;
        v5 = v2 == v0;
        if (v5){
            return Tuple3{v1, v2};
        } else {
            bool v6;
            v6 = v4 == v0;
            if (v6){
                return Tuple3{v3, v4};
            } else {
                return Tuple3{v1, v2};
            }
        }
    }
    __device__ Closure6(int _v0) : v0(_v0) { }
};
struct Closure7 {
    __device__ bool operator()(bool tup0, bool tup1){
        bool v0 = tup0; bool v1 = tup1;
        bool v2;
        v2 = v0 || v1;
        return v2;
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ Tuple0 method_1(float * v0, float * v1, float * v2, float * v3, float * v4, int v5){
    assert("Tensor range check" && 0 <= v5 && v5 < 4096l);
    int v6;
    v6 = 4l * v5;
    float * v7;
    v7 = v0+v6;
    float * v9;
    v9 = v1+v6;
    __shared__ float * v11[32l];
    __shared__ float * v12[32l];
    /* void shared array create v13 */;
    __shared__ float v14[32l];
    __shared__ float v15[32l];
    __shared__ int v16[32l];
    int v17;
    v17 = threadIdx.x;
    assert("Tensor range check" && 0 <= v17 && v17 < 32l);
    v11[v17] = v7;
    v12[v17] = v9;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v18;
    v18 = threadIdx.x;
    bool v19;
    v19 = 0l <= v18;
    bool v20;
    v20 = v19 == false;
    if (v20){
        assert("The index needs to be zero or positive." && v19);
    } else {
    }
    int v22;
    v22 = v18 % 1l;
    bool v23;
    v23 = v18 < 32l;
    bool v24;
    v24 = v23 == false;
    if (v24){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v23);
    } else {
    }
    assert("Tensor range check" && 0 <= v18 && v18 < 32l);
    int v26;
    v26 = 0l;
    while (while_method_1(v26)){
        bool v28;
        v28 = v19 && v23;
        bool v29;
        v29 = v28 == false;
        if (v29){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v28);
        } else {
        }
        bool v31;
        v31 = 0l <= v26;
        bool v33;
        if (v31){
            bool v32;
            v32 = v26 < 1l;
            v33 = v32;
        } else {
            v33 = false;
        }
        bool v34;
        v34 = v33 == false;
        if (v34){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v33);
        } else {
        }
        int v36;
        v36 = v26 * 32l;
        int v37;
        v37 = v36 + v18;
        assert("Tensor range check" && 0 <= v26 && v26 < 1l);
        int v38;
        v38 = 32l * v26;
        int v39;
        v39 = v38 + v18;
        float * v40;
        v40 = v11[v39];
        float * v41;
        v41 = v12[v39];
        /* void array index */;
        assert("Tensor range check" && 0 <= v22 && v22 < 1l);
        int v42;
        v42 = 4l * v22;
        float v43[4l];
        float v44[4l];
        int v45[4l];
        int v46;
        v46 = 0l;
        while (while_method_1(v46)){
            assert("Tensor range check" && 0 <= v46 && v46 < 1l);
            int v48;
            v48 = 4l * v46;
            assert("Tensor range check" && 0 <= v46 && v46 < 1l);
            int v49;
            v49 = v48 + v42;
            int4* v50;
            v50 = reinterpret_cast<int4*>(v40 + v49);
            int4* v51;
            v51 = reinterpret_cast<int4*>(v43 + v48);
            assert("Pointer alignment check" && (unsigned long long)(v50) % 4l == 0 && (unsigned long long)(v51) % 4l == 0);
            *v51 = *v50;
            int4* v52;
            v52 = reinterpret_cast<int4*>(v41 + v49);
            int4* v53;
            v53 = reinterpret_cast<int4*>(v44 + v48);
            assert("Pointer alignment check" && (unsigned long long)(v52) % 4l == 0 && (unsigned long long)(v53) % 4l == 0);
            *v53 = *v52;
            v46 += 1l ;
        }
        int v54;
        v54 = 0l;
        while (while_method_1(v54)){
            int v56;
            v56 = 0l;
            while (while_method_2(v56)){
                bool v58;
                v58 = 0l <= v56;
                bool v60;
                if (v58){
                    bool v59;
                    v59 = v56 < 4l;
                    v60 = v59;
                } else {
                    v60 = false;
                }
                bool v61;
                v61 = v60 == false;
                if (v61){
                    assert("The indices should be inside the range of the dimension." && v60);
                } else {
                }
                bool v63;
                v63 = 0l <= v22;
                bool v65;
                if (v63){
                    bool v64;
                    v64 = v22 < 1l;
                    v65 = v64;
                } else {
                    v65 = false;
                }
                bool v66;
                v66 = v65 == false;
                if (v66){
                    assert("The indices should be inside the range of the dimension." && v65);
                } else {
                }
                int v68;
                v68 = v22 * 4l;
                int v69;
                v69 = v56 + v68;
                bool v70;
                v70 = 0l <= v54;
                bool v72;
                if (v70){
                    bool v71;
                    v71 = v54 < 1l;
                    v72 = v71;
                } else {
                    v72 = false;
                }
                bool v73;
                v73 = v72 == false;
                if (v73){
                    assert("The indices should be inside the range of the dimension." && v72);
                } else {
                }
                int v75;
                v75 = v54 * 4l;
                int v76;
                v76 = v69 + v75;
                assert("Tensor range check" && 0 <= v54 && v54 < 1l);
                assert("Tensor range check" && 0 <= v56 && v56 < 4l);
                int v77;
                v77 = 4l * v54;
                int v78;
                v78 = v77 + v56;
                v45[v78] = v76;
                v56 += 1l ;
            }
            v54 += 1l ;
        }
        unsigned long long v79;
        v79 = clock64();
        int v80;
        v80 = threadIdx.x;
        unsigned long long v81;
        v81 = (unsigned long long)v80;
        curandStatePhilox4_32_10_t v82;
        curand_init(v79,v81,0ull,&v82);
        bool v83[4l];
        int v84;
        v84 = 0l;
        while (while_method_1(v84)){
            int v86;
            v86 = 0l;
            while (while_method_2(v86)){
                assert("Tensor range check" && 0 <= v84 && v84 < 1l);
                assert("Tensor range check" && 0 <= v86 && v86 < 4l);
                int v88;
                v88 = 4l * v84;
                int v89;
                v89 = v88 + v86;
                float v90;
                v90 = v43[v89];
                int v91;
                v91 = v45[v89];
                bool v92;
                v92 = v91 < 3l;
                assert("Tensor range check" && 0 <= v84 && v84 < 1l);
                assert("Tensor range check" && 0 <= v86 && v86 < 4l);
                v83[v89] = v92;
                v86 += 1l ;
            }
            v84 += 1l ;
        }
        float v93[4l];
        int v94;
        v94 = 0l;
        while (while_method_1(v94)){
            int v96;
            v96 = 0l;
            while (while_method_2(v96)){
                assert("Tensor range check" && 0 <= v94 && v94 < 1l);
                assert("Tensor range check" && 0 <= v96 && v96 < 4l);
                int v98;
                v98 = 4l * v94;
                int v99;
                v99 = v98 + v96;
                float v100;
                v100 = v43[v99];
                bool v101;
                v101 = v83[v99];
                float v104;
                if (v101){
                    bool v102;
                    v102 = 0.0f >= v100;
                    if (v102){
                        v104 = 0.0f;
                    } else {
                        v104 = v100;
                    }
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
        float v105;
        v105 = 0.0f;
        int v106;
        v106 = 0l;
        while (while_method_1(v106)){
            int v108;
            v108 = 0l;
            while (while_method_2(v108)){
                assert("Tensor range check" && 0 <= v106 && v106 < 1l);
                assert("Tensor range check" && 0 <= v108 && v108 < 4l);
                int v110;
                v110 = 4l * v106;
                int v111;
                v111 = v110 + v108;
                float v112;
                v112 = v93[v111];
                float v113;
                v113 = v105 + v112;
                v105 = v113;
                v108 += 1l ;
            }
            v106 += 1l ;
        }
        auto v114 = cooperative_groups::coalesced_threads();
        int v115;
        v115 = threadIdx.x;
        auto v116 = cooperative_groups::labeled_partition(v114,v115);
        Closure0 v117{};
        float v118;
        v118 = cooperative_groups::reduce(v116, v105, v117);
        int v119[4l];
        int v120;
        v120 = 0l;
        while (while_method_1(v120)){
            int v122;
            v122 = 0l;
            while (while_method_2(v122)){
                assert("Tensor range check" && 0 <= v120 && v120 < 1l);
                assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                int v124;
                v124 = 4l * v120;
                int v125;
                v125 = v124 + v122;
                bool v126;
                v126 = v83[v125];
                int v127;
                if (v126){
                    v127 = 1l;
                } else {
                    v127 = 0l;
                }
                assert("Tensor range check" && 0 <= v120 && v120 < 1l);
                assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                v119[v125] = v127;
                v122 += 1l ;
            }
            v120 += 1l ;
        }
        int v128;
        v128 = 0l;
        int v129;
        v129 = 0l;
        while (while_method_1(v129)){
            int v131;
            v131 = 0l;
            while (while_method_2(v131)){
                assert("Tensor range check" && 0 <= v129 && v129 < 1l);
                assert("Tensor range check" && 0 <= v131 && v131 < 4l);
                int v133;
                v133 = 4l * v129;
                int v134;
                v134 = v133 + v131;
                int v135;
                v135 = v119[v134];
                int v136;
                v136 = v128 + v135;
                v128 = v136;
                v131 += 1l ;
            }
            v129 += 1l ;
        }
        auto v137 = cooperative_groups::coalesced_threads();
        int v138;
        v138 = threadIdx.x;
        auto v139 = cooperative_groups::labeled_partition(v137,v138);
        Closure1 v140{};
        int v141;
        v141 = cooperative_groups::reduce(v139, v128, v140);
        float v142;
        v142 = (float)v141;
        float v143;
        v143 = 1.0f / v142;
        float v144[4l];
        int v145;
        v145 = 0l;
        while (while_method_1(v145)){
            int v147;
            v147 = 0l;
            while (while_method_2(v147)){
                assert("Tensor range check" && 0 <= v145 && v145 < 1l);
                assert("Tensor range check" && 0 <= v147 && v147 < 4l);
                int v149;
                v149 = 4l * v145;
                int v150;
                v150 = v149 + v147;
                float v151;
                v151 = v93[v150];
                bool v152;
                v152 = v83[v150];
                bool v153;
                v153 = v152 == false;
                float v158;
                if (v153){
                    v158 = 0.0f;
                } else {
                    bool v154;
                    v154 = v118 == 0.0f;
                    bool v155;
                    v155 = v154 != true;
                    if (v155){
                        float v156;
                        v156 = v151 / v118;
                        v158 = v156;
                    } else {
                        v158 = v143;
                    }
                }
                assert("Tensor range check" && 0 <= v145 && v145 < 1l);
                assert("Tensor range check" && 0 <= v147 && v147 < 4l);
                v144[v150] = v158;
                v147 += 1l ;
            }
            v145 += 1l ;
        }
        float v159[4l];
        float v160;
        v160 = 0.0f;
        int v161;
        v161 = 0l;
        while (while_method_1(v161)){
            assert("Tensor range check" && 0 <= v161 && v161 < 1l);
            int v163;
            v163 = 4l * v161;
            assert("Tensor range check" && 0 <= v161 && v161 < 1l);
            int v164; float v165;
            Tuple1 tmp0 = Tuple1{0l, 0.0f};
            v164 = tmp0.v0; v165 = tmp0.v1;
            while (while_method_2(v164)){
                assert("Tensor range check" && 0 <= v164 && v164 < 4l);
                int v167;
                v167 = v164 + v163;
                float v168;
                v168 = v144[v167];
                float v169;
                v169 = v165 + v168;
                v165 = v169;
                v164 += 1l ;
            }
            auto v170 = cooperative_groups::coalesced_threads();
            int v171;
            v171 = threadIdx.x;
            auto v172 = cooperative_groups::labeled_partition(v170,v171);
            Closure2 v173{};
            float v174;
            v174 = cooperative_groups::inclusive_scan(v172, v165, v173);
            float v175;
            v175 = v172.shfl_up(v174,1);
            bool v176;
            v176 = v172.thread_rank() == 0;
            float v177;
            if (v176){
                v177 = 0.0f;
            } else {
                v177 = v175;
            }
            float v178;
            v178 = v172.shfl(v174,v172.num_threads()-1);
            float v179;
            v179 = v160 + v177;
            int v180; float v181;
            Tuple1 tmp1 = Tuple1{0l, v179};
            v180 = tmp1.v0; v181 = tmp1.v1;
            while (while_method_2(v180)){
                assert("Tensor range check" && 0 <= v180 && v180 < 4l);
                int v183;
                v183 = v180 + v163;
                float v184;
                v184 = v144[v183];
                float v185;
                v185 = v181 + v184;
                assert("Tensor range check" && 0 <= v180 && v180 < 4l);
                v159[v183] = v185;
                v181 = v185;
                v180 += 1l ;
            }
            float v186;
            v186 = v160 + v178;
            v160 = v186;
            v161 += 1l ;
        }
        float v187[4l];
        bool v188[4l];
        int v189;
        v189 = 0l;
        while (while_method_1(v189)){
            int v191;
            v191 = 0l;
            while (while_method_2(v191)){
                assert("Tensor range check" && 0 <= v189 && v189 < 1l);
                assert("Tensor range check" && 0 <= v191 && v191 < 4l);
                int v193;
                v193 = 4l * v189;
                int v194;
                v194 = v193 + v191;
                float v195;
                v195 = v159[v194];
                float v196;
                v196 = v144[v194];
                bool v197;
                v197 = v196 > 0.0f;
                assert("Tensor range check" && 0 <= v189 && v189 < 1l);
                assert("Tensor range check" && 0 <= v191 && v191 < 4l);
                v187[v194] = v195;
                v188[v194] = v197;
                v191 += 1l ;
            }
            v189 += 1l ;
        }
        float v198; bool v199;
        Tuple2 tmp2 = Tuple2{-1.0f / 0.0f, false};
        v198 = tmp2.v0; v199 = tmp2.v1;
        int v200;
        v200 = 0l;
        while (while_method_1(v200)){
            int v202;
            v202 = 0l;
            while (while_method_2(v202)){
                assert("Tensor range check" && 0 <= v200 && v200 < 1l);
                assert("Tensor range check" && 0 <= v202 && v202 < 4l);
                int v204;
                v204 = 4l * v200;
                int v205;
                v205 = v204 + v202;
                float v206;
                v206 = v187[v205];
                bool v207;
                v207 = v188[v205];
                float v214; bool v215;
                if (v199){
                    if (v207){
                        bool v208;
                        v208 = v198 >= v206;
                        float v209;
                        if (v208){
                            v209 = v198;
                        } else {
                            v209 = v206;
                        }
                        v214 = v209; v215 = true;
                    } else {
                        v214 = v198; v215 = v199;
                    }
                } else {
                    if (v207){
                        v214 = v206; v215 = v207;
                    } else {
                        v214 = v198; v215 = v199;
                    }
                }
                v198 = v214;
                v199 = v215;
                v202 += 1l ;
            }
            v200 += 1l ;
        }
        auto v216 = cooperative_groups::coalesced_threads();
        int v217;
        v217 = threadIdx.x;
        auto v218 = cooperative_groups::labeled_partition(v216,v217);
        Closure3 v219{};
        float v220; bool v221;
        Tuple2 tmp3 = cooperative_groups::reduce(v218, Tuple2{v198, v199}, v219);
        v220 = tmp3.v0; v221 = tmp3.v1;
        bool v222;
        v222 = v221 == false;
        if (v222){
            assert("The local reduce must be true." && v221);
        } else {
        }
        float v224[4l];
        int v225[4l];
        int v226;
        v226 = 0l;
        while (while_method_1(v226)){
            int v228;
            v228 = 0l;
            while (while_method_2(v228)){
                assert("Tensor range check" && 0 <= v226 && v226 < 1l);
                assert("Tensor range check" && 0 <= v228 && v228 < 4l);
                int v230;
                v230 = 4l * v226;
                int v231;
                v231 = v230 + v228;
                int v232;
                v232 = v45[v231];
                float v233;
                v233 = curand_uniform(&v82);
                assert("Tensor range check" && 0 <= v226 && v226 < 1l);
                assert("Tensor range check" && 0 <= v228 && v228 < 4l);
                v224[v231] = v233;
                v225[v231] = v232;
                v228 += 1l ;
            }
            v226 += 1l ;
        }
        float v234; int v235;
        Tuple3 tmp4 = Tuple3{0.0f, 2147483647l};
        v234 = tmp4.v0; v235 = tmp4.v1;
        int v236;
        v236 = 0l;
        while (while_method_1(v236)){
            int v238;
            v238 = 0l;
            while (while_method_2(v238)){
                assert("Tensor range check" && 0 <= v236 && v236 < 1l);
                assert("Tensor range check" && 0 <= v238 && v238 < 4l);
                int v240;
                v240 = 4l * v236;
                int v241;
                v241 = v240 + v238;
                float v242;
                v242 = v224[v241];
                int v243;
                v243 = v225[v241];
                bool v244;
                v244 = v235 < v243;
                float v245; int v246;
                if (v244){
                    v245 = v234; v246 = v235;
                } else {
                    v245 = v242; v246 = v243;
                }
                v234 = v245;
                v235 = v246;
                v238 += 1l ;
            }
            v236 += 1l ;
        }
        auto v247 = cooperative_groups::coalesced_threads();
        int v248;
        v248 = threadIdx.x;
        auto v249 = cooperative_groups::labeled_partition(v247,v248);
        Closure4 v250{};
        float v251; int v252;
        Tuple3 tmp5 = cooperative_groups::reduce(v249, Tuple3{v234, v235}, v250);
        v251 = tmp5.v0; v252 = tmp5.v1;
        float v253;
        v253 = v220 * v251;
        int v254[4l];
        bool v255[4l];
        int v256;
        v256 = 0l;
        while (while_method_1(v256)){
            int v258;
            v258 = 0l;
            while (while_method_2(v258)){
                assert("Tensor range check" && 0 <= v256 && v256 < 1l);
                assert("Tensor range check" && 0 <= v258 && v258 < 4l);
                int v260;
                v260 = 4l * v256;
                int v261;
                v261 = v260 + v258;
                float v262;
                v262 = v187[v261];
                bool v263;
                v263 = v188[v261];
                int v264;
                v264 = v45[v261];
                int v267; bool v268;
                if (v263){
                    float v265;
                    v265 = v262 - v253;
                    bool v266;
                    v266 = v265 >= 0.0f;
                    v267 = v264; v268 = v266;
                } else {
                    v267 = 2147483647l; v268 = false;
                }
                assert("Tensor range check" && 0 <= v256 && v256 < 1l);
                assert("Tensor range check" && 0 <= v258 && v258 < 4l);
                v254[v261] = v267;
                v255[v261] = v268;
                v258 += 1l ;
            }
            v256 += 1l ;
        }
        int v269; bool v270;
        Tuple4 tmp6 = Tuple4{2147483647l, false};
        v269 = tmp6.v0; v270 = tmp6.v1;
        int v271;
        v271 = 0l;
        while (while_method_1(v271)){
            int v273;
            v273 = 0l;
            while (while_method_2(v273)){
                assert("Tensor range check" && 0 <= v271 && v271 < 1l);
                assert("Tensor range check" && 0 <= v273 && v273 < 4l);
                int v275;
                v275 = 4l * v271;
                int v276;
                v276 = v275 + v273;
                int v277;
                v277 = v254[v276];
                bool v278;
                v278 = v255[v276];
                int v285; bool v286;
                if (v270){
                    if (v278){
                        bool v279;
                        v279 = v269 < v277;
                        int v280;
                        if (v279){
                            v280 = v269;
                        } else {
                            v280 = v277;
                        }
                        v285 = v280; v286 = true;
                    } else {
                        v285 = v269; v286 = v270;
                    }
                } else {
                    if (v278){
                        v285 = v277; v286 = v278;
                    } else {
                        v285 = v269; v286 = v270;
                    }
                }
                v269 = v285;
                v270 = v286;
                v273 += 1l ;
            }
            v271 += 1l ;
        }
        auto v287 = cooperative_groups::coalesced_threads();
        int v288;
        v288 = threadIdx.x;
        auto v289 = cooperative_groups::labeled_partition(v287,v288);
        Closure5 v290{};
        int v291; bool v292;
        Tuple4 tmp7 = cooperative_groups::reduce(v289, Tuple4{v269, v270}, v290);
        v291 = tmp7.v0; v292 = tmp7.v1;
        bool v293;
        v293 = v292 == false;
        if (v293){
            assert("The local reduce must be true." && v292);
        } else {
        }
        bool v295[4l];
        int v296;
        v296 = 0l;
        while (while_method_1(v296)){
            int v298;
            v298 = 0l;
            while (while_method_2(v298)){
                assert("Tensor range check" && 0 <= v296 && v296 < 1l);
                assert("Tensor range check" && 0 <= v298 && v298 < 4l);
                int v300;
                v300 = 4l * v296;
                int v301;
                v301 = v300 + v298;
                float v302;
                v302 = v44[v301];
                int v303;
                v303 = v45[v301];
                bool v304;
                v304 = v303 < 3l;
                assert("Tensor range check" && 0 <= v296 && v296 < 1l);
                assert("Tensor range check" && 0 <= v298 && v298 < 4l);
                v295[v301] = v304;
                v298 += 1l ;
            }
            v296 += 1l ;
        }
        float v305[4l];
        int v306;
        v306 = 0l;
        while (while_method_1(v306)){
            int v308;
            v308 = 0l;
            while (while_method_2(v308)){
                assert("Tensor range check" && 0 <= v306 && v306 < 1l);
                assert("Tensor range check" && 0 <= v308 && v308 < 4l);
                int v310;
                v310 = 4l * v306;
                int v311;
                v311 = v310 + v308;
                float v312;
                v312 = v44[v311];
                bool v313;
                v313 = v295[v311];
                float v316;
                if (v313){
                    bool v314;
                    v314 = 0.0f >= v312;
                    if (v314){
                        v316 = 0.0f;
                    } else {
                        v316 = v312;
                    }
                } else {
                    v316 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v306 && v306 < 1l);
                assert("Tensor range check" && 0 <= v308 && v308 < 4l);
                v305[v311] = v316;
                v308 += 1l ;
            }
            v306 += 1l ;
        }
        float v317;
        v317 = 0.0f;
        int v318;
        v318 = 0l;
        while (while_method_1(v318)){
            int v320;
            v320 = 0l;
            while (while_method_2(v320)){
                assert("Tensor range check" && 0 <= v318 && v318 < 1l);
                assert("Tensor range check" && 0 <= v320 && v320 < 4l);
                int v322;
                v322 = 4l * v318;
                int v323;
                v323 = v322 + v320;
                float v324;
                v324 = v305[v323];
                float v325;
                v325 = v317 + v324;
                v317 = v325;
                v320 += 1l ;
            }
            v318 += 1l ;
        }
        auto v326 = cooperative_groups::coalesced_threads();
        int v327;
        v327 = threadIdx.x;
        auto v328 = cooperative_groups::labeled_partition(v326,v327);
        float v329;
        v329 = cooperative_groups::reduce(v328, v317, v117);
        int v330[4l];
        int v331;
        v331 = 0l;
        while (while_method_1(v331)){
            int v333;
            v333 = 0l;
            while (while_method_2(v333)){
                assert("Tensor range check" && 0 <= v331 && v331 < 1l);
                assert("Tensor range check" && 0 <= v333 && v333 < 4l);
                int v335;
                v335 = 4l * v331;
                int v336;
                v336 = v335 + v333;
                bool v337;
                v337 = v295[v336];
                int v338;
                if (v337){
                    v338 = 1l;
                } else {
                    v338 = 0l;
                }
                assert("Tensor range check" && 0 <= v331 && v331 < 1l);
                assert("Tensor range check" && 0 <= v333 && v333 < 4l);
                v330[v336] = v338;
                v333 += 1l ;
            }
            v331 += 1l ;
        }
        int v339;
        v339 = 0l;
        int v340;
        v340 = 0l;
        while (while_method_1(v340)){
            int v342;
            v342 = 0l;
            while (while_method_2(v342)){
                assert("Tensor range check" && 0 <= v340 && v340 < 1l);
                assert("Tensor range check" && 0 <= v342 && v342 < 4l);
                int v344;
                v344 = 4l * v340;
                int v345;
                v345 = v344 + v342;
                int v346;
                v346 = v330[v345];
                int v347;
                v347 = v339 + v346;
                v339 = v347;
                v342 += 1l ;
            }
            v340 += 1l ;
        }
        auto v348 = cooperative_groups::coalesced_threads();
        int v349;
        v349 = threadIdx.x;
        auto v350 = cooperative_groups::labeled_partition(v348,v349);
        int v351;
        v351 = cooperative_groups::reduce(v350, v339, v140);
        float v352;
        v352 = (float)v351;
        float v353;
        v353 = 1.0f / v352;
        float v354[4l];
        int v355;
        v355 = 0l;
        while (while_method_1(v355)){
            int v357;
            v357 = 0l;
            while (while_method_2(v357)){
                assert("Tensor range check" && 0 <= v355 && v355 < 1l);
                assert("Tensor range check" && 0 <= v357 && v357 < 4l);
                int v359;
                v359 = 4l * v355;
                int v360;
                v360 = v359 + v357;
                float v361;
                v361 = v305[v360];
                bool v362;
                v362 = v295[v360];
                bool v363;
                v363 = v362 == false;
                float v368;
                if (v363){
                    v368 = 0.0f;
                } else {
                    bool v364;
                    v364 = v329 == 0.0f;
                    bool v365;
                    v365 = v364 != true;
                    if (v365){
                        float v366;
                        v366 = v361 / v329;
                        v368 = v366;
                    } else {
                        v368 = v353;
                    }
                }
                assert("Tensor range check" && 0 <= v355 && v355 < 1l);
                assert("Tensor range check" && 0 <= v357 && v357 < 4l);
                v354[v360] = v368;
                v357 += 1l ;
            }
            v355 += 1l ;
        }
        float v369; int v370;
        Tuple3 tmp8 = Tuple3{0.0f, 2147483647l};
        v369 = tmp8.v0; v370 = tmp8.v1;
        int v371;
        v371 = 0l;
        while (while_method_1(v371)){
            int v373;
            v373 = 0l;
            while (while_method_2(v373)){
                assert("Tensor range check" && 0 <= v371 && v371 < 1l);
                assert("Tensor range check" && 0 <= v373 && v373 < 4l);
                int v375;
                v375 = 4l * v371;
                int v376;
                v376 = v375 + v373;
                float v377;
                v377 = v144[v376];
                int v378;
                v378 = v45[v376];
                bool v379;
                v379 = v370 == v291;
                float v383; int v384;
                if (v379){
                    v383 = v369; v384 = v370;
                } else {
                    bool v380;
                    v380 = v378 == v291;
                    if (v380){
                        v383 = v377; v384 = v378;
                    } else {
                        v383 = v369; v384 = v370;
                    }
                }
                v369 = v383;
                v370 = v384;
                v373 += 1l ;
            }
            v371 += 1l ;
        }
        auto v385 = cooperative_groups::coalesced_threads();
        int v386;
        v386 = threadIdx.x;
        auto v387 = cooperative_groups::labeled_partition(v385,v386);
        Closure6 v388{v291};
        float v389; int v390;
        Tuple3 tmp9 = cooperative_groups::reduce(v387, Tuple3{v369, v370}, v388);
        v389 = tmp9.v0; v390 = tmp9.v1;
        bool v391;
        v391 = v390 == 2147483647l;
        bool v392;
        v392 = v391 != true;
        bool v393;
        v393 = v392 == false;
        if (v393){
            assert("Expected a valid action id in get_action." && v392);
        } else {
        }
        float v395; int v396;
        Tuple3 tmp10 = Tuple3{0.0f, 2147483647l};
        v395 = tmp10.v0; v396 = tmp10.v1;
        int v397;
        v397 = 0l;
        while (while_method_1(v397)){
            int v399;
            v399 = 0l;
            while (while_method_2(v399)){
                assert("Tensor range check" && 0 <= v397 && v397 < 1l);
                assert("Tensor range check" && 0 <= v399 && v399 < 4l);
                int v401;
                v401 = 4l * v397;
                int v402;
                v402 = v401 + v399;
                float v403;
                v403 = v354[v402];
                int v404;
                v404 = v45[v402];
                bool v405;
                v405 = v396 == v291;
                float v409; int v410;
                if (v405){
                    v409 = v395; v410 = v396;
                } else {
                    bool v406;
                    v406 = v404 == v291;
                    if (v406){
                        v409 = v403; v410 = v404;
                    } else {
                        v409 = v395; v410 = v396;
                    }
                }
                v395 = v409;
                v396 = v410;
                v399 += 1l ;
            }
            v397 += 1l ;
        }
        auto v411 = cooperative_groups::coalesced_threads();
        int v412;
        v412 = threadIdx.x;
        auto v413 = cooperative_groups::labeled_partition(v411,v412);
        float v414; int v415;
        Tuple3 tmp11 = cooperative_groups::reduce(v413, Tuple3{v395, v396}, v388);
        v414 = tmp11.v0; v415 = tmp11.v1;
        bool v416;
        v416 = v415 == 2147483647l;
        bool v417;
        v417 = v416 != true;
        bool v418;
        v418 = v417 == false;
        if (v418){
            assert("Expected a valid action id in get_action." && v417);
        } else {
        }
        int v420;
        v420 = 0l;
        while (while_method_1(v420)){
            assert("Tensor range check" && 0 <= v420 && v420 < 1l);
            assert("Tensor range check" && 0 <= v420 && v420 < 1l);
            v420 += 1l ;
        }
        assert("Tensor range check" && 0 <= v37 && v37 < 32l);
        v14[v37] = v414;
        v15[v37] = v389;
        v16[v37] = v291;
        v26 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v422;
    v422 = threadIdx.x;
    assert("Tensor range check" && 0 <= v422 && v422 < 32l);
    float v423;
    v423 = v14[v422];
    float v424;
    v424 = v15[v422];
    int v425;
    v425 = v16[v422];
    return Tuple0{v423, v424, v425};
}
__device__ void push_0(float * v0, float * v1, float * v2, float * v3, float * v4, int * v5, float * v6, int * v7, int * v8, double * v9, double * v10, float * v11, float * v12, float * v13, int v14, int & v15, double * v16, double * v17, int v18, int v19){
    float v20; float v21; int v22;
    Tuple0 tmp12 = method_1(v0, v1, v2, v3, v4, v19);
    v20 = tmp12.v0; v21 = tmp12.v1; v22 = tmp12.v2;
    int v23 = v15;
    int v24;
    v24 = v23 + 1l;
    v15 = v24;
    assert("Tensor range check" && 0 <= v23 && v23 < 16l);
    assert("Tensor range check" && 0 <= v14 && v14 < 32l);
    int v25;
    v25 = 32l * v23;
    int v26;
    v26 = v25 + v14;
    v5[v26] = v22;
    v6[v26] = v21;
    v7[v26] = v18;
    v8[v26] = v19;
    double v27;
    v27 = (double)v21;
    double v28;
    v28 = log(v27);
    double v29;
    v29 = (double)v20;
    double v30;
    v30 = log(v29);
    assert("Tensor range check" && 0 <= v18 && v18 < 2l);
    double v31;
    v31 = v16[v18];
    double v32;
    v32 = v17[v18];
    double v33;
    v33 = v30 + v31;
    double v34;
    v34 = v28 + v32;
    assert("Tensor range check" && 0 <= v18 && v18 < 2l);
    v16[v18] = v33;
    v17[v18] = v34;
    assert("Tensor range check" && 0 <= v23 && v23 < 16l);
    int v35;
    v35 = 64l * v23;
    assert("Tensor range check" && 0 <= v14 && v14 < 32l);
    int v36;
    v36 = 2l * v14;
    int v37;
    v37 = v36 + v35;
    int v38;
    v38 = 0l;
    while (while_method_0(v38)){
        assert("Tensor range check" && 0 <= v38 && v38 < 2l);
        double v40;
        v40 = v16[v38];
        double v41;
        v41 = v17[v38];
        assert("Tensor range check" && 0 <= v38 && v38 < 2l);
        int v42;
        v42 = v38 + v37;
        v9[v42] = v40;
        v10[v42] = v41;
        v38 += 1l ;
    }
    return ;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 > 0l;
    return v1;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 128l;
    return v1;
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1) {
    float * v2;
    v2 = reinterpret_cast<float *>(&v1[0ull]);
    float * v4;
    v4 = reinterpret_cast<float *>(&v1[65536ull]);
    float * v6;
    v6 = reinterpret_cast<float *>(&v1[131072ull]);
    float * v8;
    v8 = reinterpret_cast<float *>(&v1[196608ull]);
    float * v10;
    v10 = reinterpret_cast<float *>(&v1[262144ull]);
    int * v12;
    v12 = reinterpret_cast<int *>(&v1[327680ull]);
    float * v14;
    v14 = reinterpret_cast<float *>(&v1[329728ull]);
    int * v16;
    v16 = reinterpret_cast<int *>(&v1[331776ull]);
    int * v18;
    v18 = reinterpret_cast<int *>(&v1[333824ull]);
    double * v20;
    v20 = reinterpret_cast<double *>(&v1[335872ull]);
    double * v22;
    v22 = reinterpret_cast<double *>(&v1[344064ull]);
    float * v24;
    v24 = reinterpret_cast<float *>(&v1[352256ull]);
    float * v26;
    v26 = reinterpret_cast<float *>(&v1[360448ull]);
    float * v28;
    v28 = reinterpret_cast<float *>(&v1[362496ull]);
    int v30;
    v30 = threadIdx.x;
    int v31 = 0l;
    double v32[2l];
    double v33[2l];
    int v34;
    v34 = 0l;
    while (while_method_0(v34)){
        assert("Tensor range check" && 0 <= v34 && v34 < 2l);
        v32[v34] = 0.0;
        v33[v34] = 0.0;
        v34 += 1l ;
    }
    int v36;
    v36 = 235l;
    int v37;
    v37 = 0l;
    push_0(v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, v31, v32, v33, v37, v36);
    int v38;
    v38 = 212l;
    int v39;
    v39 = 1l;
    push_0(v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, v31, v32, v33, v39, v38);
    int v40;
    v40 = 790l;
    int v41;
    v41 = 0l;
    push_0(v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, v31, v32, v33, v41, v40);
    int v42;
    v42 = 343l;
    int v43;
    v43 = 1l;
    push_0(v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, v31, v32, v33, v43, v42);
    int v44;
    v44 = 457l;
    int v45;
    v45 = 0l;
    push_0(v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, v31, v32, v33, v45, v44);
    int v46;
    v46 = 3447l;
    int v47;
    v47 = 1l;
    push_0(v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, v31, v32, v33, v47, v46);
    int v48 = v31;
    int v49;
    v49 = threadIdx.x;
    float v50[2l];
    v50[0l] = 13.0f;
    v50[1l] = -13.0f;
    int v51;
    v51 = v48;
    while (while_method_3(v51)){
        v51 -= 1l ;
        assert("Tensor range check" && 0 <= v51 && v51 < 16l);
        assert("Tensor range check" && 0 <= v49 && v49 < 32l);
        int v53;
        v53 = 32l * v51;
        int v54;
        v54 = v53 + v49;
        int v55;
        v55 = v12[v54];
        float v56;
        v56 = v14[v54];
        int v57;
        v57 = v16[v54];
        int v58;
        v58 = v18[v54];
        assert("Tensor range check" && 0 <= v57 && v57 < 2l);
        float v59;
        v59 = v50[v57];
        assert("Tensor range check" && 0 <= v58 && v58 < 4096l);
        int v60;
        v60 = 4l * v58;
        assert("Tensor range check" && 0 <= v51 && v51 < 16l);
        int v61;
        v61 = 128l * v51;
        assert("Tensor range check" && 0 <= v49 && v49 < 32l);
        int v62;
        v62 = 4l * v49;
        int v63;
        v63 = v62 + v61;
        assert("Tensor range check" && 0 <= v51 && v51 < 16l);
        int v64;
        v64 = 64l * v51;
        double * v65;
        v65 = v20+v64;
        double * v67;
        v67 = v22+v64;
        assert("Tensor range check" && 0 <= v49 && v49 < 32l);
        int v69;
        v69 = 2l * v49;
        double v70[2l];
        int v71;
        v71 = 0l;
        while (while_method_0(v71)){
            assert("Tensor range check" && 0 <= v71 && v71 < 2l);
            int v73;
            v73 = v71 + v69;
            double v74;
            v74 = v65[v73];
            bool v75;
            v75 = v57 == v71;
            double v76;
            if (v75){
                v76 = 0.0;
            } else {
                v76 = v74;
            }
            assert("Tensor range check" && 0 <= v71 && v71 < 2l);
            v70[v71] = v76;
            v71 += 1l ;
        }
        double v77;
        v77 = 0.0;
        int v78;
        v78 = 0l;
        while (while_method_0(v78)){
            assert("Tensor range check" && 0 <= v78 && v78 < 2l);
            double v80;
            v80 = v70[v78];
            double v81;
            v81 = v77 + v80;
            v77 = v81;
            v78 += 1l ;
        }
        double v82;
        v82 = 0.0;
        int v83;
        v83 = 0l;
        while (while_method_0(v83)){
            assert("Tensor range check" && 0 <= v83 && v83 < 2l);
            int v85;
            v85 = v83 + v69;
            double v86;
            v86 = v67[v85];
            double v87;
            v87 = v82 + v86;
            v82 = v87;
            v83 += 1l ;
        }
        double v88;
        v88 = v77 - v82;
        double v89;
        v89 = exp(v88);
        float v90;
        v90 = (float)v89;
        float v91;
        v91 = v59 * v90;
        assert("Tensor range check" && 0 <= v51 && v51 < 16l);
        assert("Tensor range check" && 0 <= v49 && v49 < 32l);
        v26[v54] = v91;
        v28[v54] = v90;
        float * v92;
        v92 = v4+v60;
        float * v94;
        v94 = v8+v60;
        float * v96;
        v96 = v10+v60;
        float * v98;
        v98 = v24+v63;
        __shared__ float v100[32l];
        __shared__ int v101[32l];
        __shared__ float v102[32l];
        __shared__ int v103[32l];
        __shared__ double * v104[32l];
        __shared__ double * v105[32l];
        __shared__ float * v106[32l];
        __shared__ float * v107[32l];
        __shared__ float * v108[32l];
        __shared__ float * v109[32l];
        /* void shared array create v110 */;
        __shared__ float v111[32l];
        int v112;
        v112 = threadIdx.x;
        assert("Tensor range check" && 0 <= v112 && v112 < 32l);
        v100[v112] = v56;
        v101[v112] = v55;
        v102[v112] = v59;
        v103[v112] = v57;
        v104[v112] = v65;
        v105[v112] = v67;
        v106[v112] = v92;
        v107[v112] = v94;
        v108[v112] = v96;
        v109[v112] = v98;
        /* void array set */;
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v113;
        v113 = threadIdx.x;
        bool v114;
        v114 = 0l <= v113;
        bool v115;
        v115 = v114 == false;
        if (v115){
            assert("The index needs to be zero or positive." && v114);
        } else {
        }
        int v117;
        v117 = v113 % 1l;
        bool v118;
        v118 = v113 < 32l;
        bool v119;
        v119 = v118 == false;
        if (v119){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v118);
        } else {
        }
        assert("Tensor range check" && 0 <= v113 && v113 < 32l);
        int v121;
        v121 = 0l;
        while (while_method_1(v121)){
            bool v123;
            v123 = v114 && v118;
            bool v124;
            v124 = v123 == false;
            if (v124){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v123);
            } else {
            }
            bool v126;
            v126 = 0l <= v121;
            bool v128;
            if (v126){
                bool v127;
                v127 = v121 < 1l;
                v128 = v127;
            } else {
                v128 = false;
            }
            bool v129;
            v129 = v128 == false;
            if (v129){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v128);
            } else {
            }
            int v131;
            v131 = v121 * 32l;
            int v132;
            v132 = v131 + v113;
            assert("Tensor range check" && 0 <= v121 && v121 < 1l);
            int v133;
            v133 = 32l * v121;
            int v134;
            v134 = v133 + v113;
            float v135;
            v135 = v100[v134];
            int v136;
            v136 = v101[v134];
            float v137;
            v137 = v102[v134];
            int v138;
            v138 = v103[v134];
            double * v139;
            v139 = v104[v134];
            double * v140;
            v140 = v105[v134];
            float * v141;
            v141 = v106[v134];
            float * v142;
            v142 = v107[v134];
            float * v143;
            v143 = v108[v134];
            float * v144;
            v144 = v109[v134];
            /* void array index */;
            assert("Tensor range check" && 0 <= v117 && v117 < 1l);
            int v145;
            v145 = 4l * v117;
            float v146[4l];
            float v147[4l];
            float v148[4l];
            int v149[4l];
            int v150;
            v150 = 0l;
            while (while_method_1(v150)){
                assert("Tensor range check" && 0 <= v150 && v150 < 1l);
                int v152;
                v152 = 4l * v150;
                assert("Tensor range check" && 0 <= v150 && v150 < 1l);
                int v153;
                v153 = v152 + v145;
                int4* v154;
                v154 = reinterpret_cast<int4*>(v141 + v153);
                int4* v155;
                v155 = reinterpret_cast<int4*>(v146 + v152);
                assert("Pointer alignment check" && (unsigned long long)(v154) % 4l == 0 && (unsigned long long)(v155) % 4l == 0);
                *v155 = *v154;
                int4* v156;
                v156 = reinterpret_cast<int4*>(v142 + v153);
                int4* v157;
                v157 = reinterpret_cast<int4*>(v147 + v152);
                assert("Pointer alignment check" && (unsigned long long)(v156) % 4l == 0 && (unsigned long long)(v157) % 4l == 0);
                *v157 = *v156;
                int4* v158;
                v158 = reinterpret_cast<int4*>(v143 + v153);
                int4* v159;
                v159 = reinterpret_cast<int4*>(v148 + v152);
                assert("Pointer alignment check" && (unsigned long long)(v158) % 4l == 0 && (unsigned long long)(v159) % 4l == 0);
                *v159 = *v158;
                v150 += 1l ;
            }
            int v160;
            v160 = 0l;
            while (while_method_1(v160)){
                int v162;
                v162 = 0l;
                while (while_method_2(v162)){
                    bool v164;
                    v164 = 0l <= v162;
                    bool v166;
                    if (v164){
                        bool v165;
                        v165 = v162 < 4l;
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
                    bool v169;
                    v169 = 0l <= v117;
                    bool v171;
                    if (v169){
                        bool v170;
                        v170 = v117 < 1l;
                        v171 = v170;
                    } else {
                        v171 = false;
                    }
                    bool v172;
                    v172 = v171 == false;
                    if (v172){
                        assert("The indices should be inside the range of the dimension." && v171);
                    } else {
                    }
                    int v174;
                    v174 = v117 * 4l;
                    int v175;
                    v175 = v162 + v174;
                    bool v176;
                    v176 = 0l <= v160;
                    bool v178;
                    if (v176){
                        bool v177;
                        v177 = v160 < 1l;
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
                    int v181;
                    v181 = v160 * 4l;
                    int v182;
                    v182 = v175 + v181;
                    assert("Tensor range check" && 0 <= v160 && v160 < 1l);
                    assert("Tensor range check" && 0 <= v162 && v162 < 4l);
                    int v183;
                    v183 = 4l * v160;
                    int v184;
                    v184 = v183 + v162;
                    v149[v184] = v182;
                    v162 += 1l ;
                }
                v160 += 1l ;
            }
            float v185[4l];
            int v186;
            v186 = 0l;
            while (while_method_1(v186)){
                int v188;
                v188 = 0l;
                while (while_method_2(v188)){
                    assert("Tensor range check" && 0 <= v186 && v186 < 1l);
                    assert("Tensor range check" && 0 <= v188 && v188 < 4l);
                    int v190;
                    v190 = 4l * v186;
                    int v191;
                    v191 = v190 + v188;
                    float v192;
                    v192 = v147[v191];
                    float v193;
                    v193 = v148[v191];
                    bool v194;
                    v194 = v193 == 0.0f;
                    bool v195;
                    v195 = v194 != true;
                    float v197;
                    if (v195){
                        float v196;
                        v196 = v192 / v193;
                        v197 = v196;
                    } else {
                        v197 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v186 && v186 < 1l);
                    assert("Tensor range check" && 0 <= v188 && v188 < 4l);
                    v185[v191] = v197;
                    v188 += 1l ;
                }
                v186 += 1l ;
            }
            bool v198[4l];
            int v199;
            v199 = 0l;
            while (while_method_1(v199)){
                int v201;
                v201 = 0l;
                while (while_method_2(v201)){
                    assert("Tensor range check" && 0 <= v199 && v199 < 1l);
                    assert("Tensor range check" && 0 <= v201 && v201 < 4l);
                    int v203;
                    v203 = 4l * v199;
                    int v204;
                    v204 = v203 + v201;
                    float v205;
                    v205 = v146[v204];
                    int v206;
                    v206 = v149[v204];
                    bool v207;
                    v207 = v206 < 3l;
                    assert("Tensor range check" && 0 <= v199 && v199 < 1l);
                    assert("Tensor range check" && 0 <= v201 && v201 < 4l);
                    v198[v204] = v207;
                    v201 += 1l ;
                }
                v199 += 1l ;
            }
            float v208[4l];
            int v209;
            v209 = 0l;
            while (while_method_1(v209)){
                int v211;
                v211 = 0l;
                while (while_method_2(v211)){
                    assert("Tensor range check" && 0 <= v209 && v209 < 1l);
                    assert("Tensor range check" && 0 <= v211 && v211 < 4l);
                    int v213;
                    v213 = 4l * v209;
                    int v214;
                    v214 = v213 + v211;
                    float v215;
                    v215 = v146[v214];
                    bool v216;
                    v216 = v198[v214];
                    float v219;
                    if (v216){
                        bool v217;
                        v217 = 0.0f >= v215;
                        if (v217){
                            v219 = 0.0f;
                        } else {
                            v219 = v215;
                        }
                    } else {
                        v219 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v209 && v209 < 1l);
                    assert("Tensor range check" && 0 <= v211 && v211 < 4l);
                    v208[v214] = v219;
                    v211 += 1l ;
                }
                v209 += 1l ;
            }
            float v220;
            v220 = 0.0f;
            int v221;
            v221 = 0l;
            while (while_method_1(v221)){
                int v223;
                v223 = 0l;
                while (while_method_2(v223)){
                    assert("Tensor range check" && 0 <= v221 && v221 < 1l);
                    assert("Tensor range check" && 0 <= v223 && v223 < 4l);
                    int v225;
                    v225 = 4l * v221;
                    int v226;
                    v226 = v225 + v223;
                    float v227;
                    v227 = v208[v226];
                    float v228;
                    v228 = v220 + v227;
                    v220 = v228;
                    v223 += 1l ;
                }
                v221 += 1l ;
            }
            auto v229 = cooperative_groups::coalesced_threads();
            int v230;
            v230 = threadIdx.x;
            auto v231 = cooperative_groups::labeled_partition(v229,v230);
            Closure0 v232{};
            float v233;
            v233 = cooperative_groups::reduce(v231, v220, v232);
            int v234[4l];
            int v235;
            v235 = 0l;
            while (while_method_1(v235)){
                int v237;
                v237 = 0l;
                while (while_method_2(v237)){
                    assert("Tensor range check" && 0 <= v235 && v235 < 1l);
                    assert("Tensor range check" && 0 <= v237 && v237 < 4l);
                    int v239;
                    v239 = 4l * v235;
                    int v240;
                    v240 = v239 + v237;
                    bool v241;
                    v241 = v198[v240];
                    int v242;
                    if (v241){
                        v242 = 1l;
                    } else {
                        v242 = 0l;
                    }
                    assert("Tensor range check" && 0 <= v235 && v235 < 1l);
                    assert("Tensor range check" && 0 <= v237 && v237 < 4l);
                    v234[v240] = v242;
                    v237 += 1l ;
                }
                v235 += 1l ;
            }
            int v243;
            v243 = 0l;
            int v244;
            v244 = 0l;
            while (while_method_1(v244)){
                int v246;
                v246 = 0l;
                while (while_method_2(v246)){
                    assert("Tensor range check" && 0 <= v244 && v244 < 1l);
                    assert("Tensor range check" && 0 <= v246 && v246 < 4l);
                    int v248;
                    v248 = 4l * v244;
                    int v249;
                    v249 = v248 + v246;
                    int v250;
                    v250 = v234[v249];
                    int v251;
                    v251 = v243 + v250;
                    v243 = v251;
                    v246 += 1l ;
                }
                v244 += 1l ;
            }
            auto v252 = cooperative_groups::coalesced_threads();
            int v253;
            v253 = threadIdx.x;
            auto v254 = cooperative_groups::labeled_partition(v252,v253);
            Closure1 v255{};
            int v256;
            v256 = cooperative_groups::reduce(v254, v243, v255);
            float v257;
            v257 = (float)v256;
            float v258;
            v258 = 1.0f / v257;
            float v259[4l];
            int v260;
            v260 = 0l;
            while (while_method_1(v260)){
                int v262;
                v262 = 0l;
                while (while_method_2(v262)){
                    assert("Tensor range check" && 0 <= v260 && v260 < 1l);
                    assert("Tensor range check" && 0 <= v262 && v262 < 4l);
                    int v264;
                    v264 = 4l * v260;
                    int v265;
                    v265 = v264 + v262;
                    float v266;
                    v266 = v208[v265];
                    bool v267;
                    v267 = v198[v265];
                    bool v268;
                    v268 = v267 == false;
                    float v273;
                    if (v268){
                        v273 = 0.0f;
                    } else {
                        bool v269;
                        v269 = v233 == 0.0f;
                        bool v270;
                        v270 = v269 != true;
                        if (v270){
                            float v271;
                            v271 = v266 / v233;
                            v273 = v271;
                        } else {
                            v273 = v258;
                        }
                    }
                    assert("Tensor range check" && 0 <= v260 && v260 < 1l);
                    assert("Tensor range check" && 0 <= v262 && v262 < 4l);
                    v259[v265] = v273;
                    v262 += 1l ;
                }
                v260 += 1l ;
            }
            float v274[4l];
            int v275;
            v275 = 0l;
            while (while_method_1(v275)){
                int v277;
                v277 = 0l;
                while (while_method_2(v277)){
                    assert("Tensor range check" && 0 <= v275 && v275 < 1l);
                    assert("Tensor range check" && 0 <= v277 && v277 < 4l);
                    int v279;
                    v279 = 4l * v275;
                    int v280;
                    v280 = v279 + v277;
                    float v281;
                    v281 = v185[v280];
                    int v282;
                    v282 = v149[v280];
                    bool v283;
                    v283 = v136 == v282;
                    float v286;
                    if (v283){
                        float v284;
                        v284 = v137 - v281;
                        float v285;
                        v285 = v284 / v135;
                        v286 = v285;
                    } else {
                        v286 = 0.0f;
                    }
                    float v287;
                    v287 = v286 + v281;
                    assert("Tensor range check" && 0 <= v275 && v275 < 1l);
                    assert("Tensor range check" && 0 <= v277 && v277 < 4l);
                    v274[v280] = v287;
                    v277 += 1l ;
                }
                v275 += 1l ;
            }
            float v288[4l];
            int v289;
            v289 = 0l;
            while (while_method_1(v289)){
                int v291;
                v291 = 0l;
                while (while_method_2(v291)){
                    assert("Tensor range check" && 0 <= v289 && v289 < 1l);
                    assert("Tensor range check" && 0 <= v291 && v291 < 4l);
                    int v293;
                    v293 = 4l * v289;
                    int v294;
                    v294 = v293 + v291;
                    float v295;
                    v295 = v259[v294];
                    float v296;
                    v296 = v274[v294];
                    float v297;
                    v297 = v295 * v296;
                    assert("Tensor range check" && 0 <= v289 && v289 < 1l);
                    assert("Tensor range check" && 0 <= v291 && v291 < 4l);
                    v288[v294] = v297;
                    v291 += 1l ;
                }
                v289 += 1l ;
            }
            float v298;
            v298 = 0.0f;
            int v299;
            v299 = 0l;
            while (while_method_1(v299)){
                int v301;
                v301 = 0l;
                while (while_method_2(v301)){
                    assert("Tensor range check" && 0 <= v299 && v299 < 1l);
                    assert("Tensor range check" && 0 <= v301 && v301 < 4l);
                    int v303;
                    v303 = 4l * v299;
                    int v304;
                    v304 = v303 + v301;
                    float v305;
                    v305 = v288[v304];
                    float v306;
                    v306 = v298 + v305;
                    v298 = v306;
                    v301 += 1l ;
                }
                v299 += 1l ;
            }
            auto v307 = cooperative_groups::coalesced_threads();
            int v308;
            v308 = threadIdx.x;
            auto v309 = cooperative_groups::labeled_partition(v307,v308);
            float v310;
            v310 = cooperative_groups::reduce(v309, v298, v232);
            assert("Tensor range check" && 0 <= v132 && v132 < 32l);
            int v311;
            v311 = 2l * v132;
            double v312[2l];
            int v313;
            v313 = 0l;
            while (while_method_0(v313)){
                assert("Tensor range check" && 0 <= v313 && v313 < 2l);
                int v315;
                v315 = v313 + v311;
                double v316;
                v316 = v139[v315];
                bool v317;
                v317 = v138 == v313;
                double v318;
                if (v317){
                    v318 = 0.0;
                } else {
                    v318 = v316;
                }
                assert("Tensor range check" && 0 <= v313 && v313 < 2l);
                v312[v313] = v318;
                v313 += 1l ;
            }
            double v319;
            v319 = 0.0;
            int v320;
            v320 = 0l;
            while (while_method_0(v320)){
                assert("Tensor range check" && 0 <= v320 && v320 < 2l);
                double v322;
                v322 = v312[v320];
                double v323;
                v323 = v319 + v322;
                v319 = v323;
                v320 += 1l ;
            }
            double v324;
            v324 = 0.0;
            int v325;
            v325 = 0l;
            while (while_method_0(v325)){
                assert("Tensor range check" && 0 <= v325 && v325 < 2l);
                int v327;
                v327 = v325 + v311;
                double v328;
                v328 = v140[v327];
                double v329;
                v329 = v324 + v328;
                v324 = v329;
                v325 += 1l ;
            }
            double v330;
            v330 = v319 - v324;
            double v331;
            v331 = exp(v330);
            float v332;
            v332 = (float)v331;
            float v333[4l];
            int v334;
            v334 = 0l;
            while (while_method_1(v334)){
                int v336;
                v336 = 0l;
                while (while_method_2(v336)){
                    assert("Tensor range check" && 0 <= v334 && v334 < 1l);
                    assert("Tensor range check" && 0 <= v336 && v336 < 4l);
                    int v338;
                    v338 = 4l * v334;
                    int v339;
                    v339 = v338 + v336;
                    float v340;
                    v340 = v274[v339];
                    float v341;
                    v341 = v340 - v310;
                    float v342;
                    v342 = v332 * v341;
                    assert("Tensor range check" && 0 <= v334 && v334 < 1l);
                    assert("Tensor range check" && 0 <= v336 && v336 < 4l);
                    v333[v339] = v342;
                    v336 += 1l ;
                }
                v334 += 1l ;
            }
            int v343;
            v343 = 0l;
            while (while_method_1(v343)){
                assert("Tensor range check" && 0 <= v343 && v343 < 1l);
                int v345;
                v345 = 4l * v343;
                int v346;
                v346 = v345 + v145;
                assert("Tensor range check" && 0 <= v343 && v343 < 1l);
                int4* v347;
                v347 = reinterpret_cast<int4*>(v333 + v345);
                int4* v348;
                v348 = reinterpret_cast<int4*>(v144 + v346);
                assert("Pointer alignment check" && (unsigned long long)(v347) % 4l == 0 && (unsigned long long)(v348) % 4l == 0);
                *v348 = *v347;
                v343 += 1l ;
            }
            assert("Tensor range check" && 0 <= v132 && v132 < 32l);
            v111[v132] = v310;
            v121 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v349;
        v349 = threadIdx.x;
        assert("Tensor range check" && 0 <= v349 && v349 < 32l);
        float v350;
        v350 = v111[v349];
        assert("Tensor range check" && 0 <= v57 && v57 < 2l);
        v50[v57] = v350;
        cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v351 = console_lock;
        auto v352 = cooperative_groups::coalesced_threads();
        v351.acquire();
        int v353;
        v353 = 0l;
        printf("{%s = %c","rewards", '[');
        int v354;
        v354 = 0l;
        while (while_method_0(v354)){
            int v356;
            v356 = v353;
            bool v357;
            v357 = v356 >= 100l;
            if (v357){
                printf("%s"," ...");
                break;
            } else {
            }
            bool v358;
            v358 = v354 == 0l;
            bool v359;
            v359 = v358 != true;
            if (v359){
                printf("%s","; ");
            } else {
            }
            int v360;
            v360 = v353 + 1l;
            v353 = v360;
            float v361;
            v361 = v50[v354];
            printf("%f",v361);
            v354 += 1l ;
        }
        printf("%c",']');
        printf("}\n");
        v351.release();
        v352.sync() ;
    }
    int v380 = v31;
    int v381;
    v381 = threadIdx.x;
    int v382;
    v382 = v380;
    while (while_method_3(v382)){
        v382 -= 1l ;
        assert("Tensor range check" && 0 <= v382 && v382 < 16l);
        assert("Tensor range check" && 0 <= v381 && v381 < 32l);
        int v384;
        v384 = 32l * v382;
        int v385;
        v385 = v384 + v381;
        int v386;
        v386 = v12[v385];
        float v387;
        v387 = v14[v385];
        int v388;
        v388 = v16[v385];
        int v389;
        v389 = v18[v385];
        assert("Tensor range check" && 0 <= v382 && v382 < 16l);
        assert("Tensor range check" && 0 <= v381 && v381 < 32l);
        float v390;
        v390 = v26[v385];
        float v391;
        v391 = v28[v385];
        assert("Tensor range check" && 0 <= v389 && v389 < 4096l);
        int v392;
        v392 = 4l * v389;
        float * v393;
        v393 = v2+v392;
        float * v395;
        v395 = v4+v392;
        float * v397;
        v397 = v6+v392;
        float * v399;
        v399 = v8+v392;
        float * v401;
        v401 = v10+v392;
        assert("Tensor range check" && 0 <= v382 && v382 < 16l);
        int v403;
        v403 = 128l * v382;
        assert("Tensor range check" && 0 <= v381 && v381 < 32l);
        int v404;
        v404 = 4l * v381;
        int v405;
        v405 = v404 + v403;
        assert("Tensor range check" && 0 <= v386 && v386 < 4l);
        float * v406;
        v406 = v399+v386;
        float * v408;
        v408 = v401+v386;
        float v410;
        v410 = atomicAdd(v406,v390);
        float v411;
        v411 = atomicAdd(v408,v391);
        float * v412;
        v412 = v24+v405;
        __shared__ float * v414[32l];
        __shared__ float * v415[32l];
        /* void shared array create v416 */;
        /* void shared array create v417 */;
        int v418;
        v418 = threadIdx.x;
        assert("Tensor range check" && 0 <= v418 && v418 < 32l);
        v414[v418] = v397;
        v415[v418] = v412;
        /* void array set */;
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v419;
        v419 = threadIdx.x;
        bool v420;
        v420 = 0l <= v419;
        bool v421;
        v421 = v420 == false;
        if (v421){
            assert("The index needs to be zero or positive." && v420);
        } else {
        }
        int v423;
        v423 = v419 % 1l;
        bool v424;
        v424 = v419 < 32l;
        bool v425;
        v425 = v424 == false;
        if (v425){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v424);
        } else {
        }
        assert("Tensor range check" && 0 <= v419 && v419 < 32l);
        int v427;
        v427 = 0l;
        while (while_method_1(v427)){
            bool v429;
            v429 = v420 && v424;
            bool v430;
            v430 = v429 == false;
            if (v430){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v429);
            } else {
            }
            bool v432;
            v432 = 0l <= v427;
            bool v434;
            if (v432){
                bool v433;
                v433 = v427 < 1l;
                v434 = v433;
            } else {
                v434 = false;
            }
            bool v435;
            v435 = v434 == false;
            if (v435){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v434);
            } else {
            }
            int v437;
            v437 = v427 * 32l;
            int v438;
            v438 = v437 + v419;
            assert("Tensor range check" && 0 <= v427 && v427 < 1l);
            int v439;
            v439 = 32l * v427;
            int v440;
            v440 = v439 + v419;
            float * v441;
            v441 = v414[v440];
            float * v442;
            v442 = v415[v440];
            /* void array index */;
            assert("Tensor range check" && 0 <= v423 && v423 < 1l);
            int v443;
            v443 = 4l * v423;
            float v444[4l];
            int v445[4l];
            int v446;
            v446 = 0l;
            while (while_method_1(v446)){
                assert("Tensor range check" && 0 <= v446 && v446 < 1l);
                int v448;
                v448 = 4l * v446;
                assert("Tensor range check" && 0 <= v446 && v446 < 1l);
                int v449;
                v449 = v448 + v443;
                int4* v450;
                v450 = reinterpret_cast<int4*>(v442 + v449);
                int4* v451;
                v451 = reinterpret_cast<int4*>(v444 + v448);
                assert("Pointer alignment check" && (unsigned long long)(v450) % 4l == 0 && (unsigned long long)(v451) % 4l == 0);
                *v451 = *v450;
                v446 += 1l ;
            }
            int v452;
            v452 = 0l;
            while (while_method_1(v452)){
                int v454;
                v454 = 0l;
                while (while_method_2(v454)){
                    bool v456;
                    v456 = 0l <= v454;
                    bool v458;
                    if (v456){
                        bool v457;
                        v457 = v454 < 4l;
                        v458 = v457;
                    } else {
                        v458 = false;
                    }
                    bool v459;
                    v459 = v458 == false;
                    if (v459){
                        assert("The indices should be inside the range of the dimension." && v458);
                    } else {
                    }
                    bool v461;
                    v461 = 0l <= v423;
                    bool v463;
                    if (v461){
                        bool v462;
                        v462 = v423 < 1l;
                        v463 = v462;
                    } else {
                        v463 = false;
                    }
                    bool v464;
                    v464 = v463 == false;
                    if (v464){
                        assert("The indices should be inside the range of the dimension." && v463);
                    } else {
                    }
                    int v466;
                    v466 = v423 * 4l;
                    int v467;
                    v467 = v454 + v466;
                    bool v468;
                    v468 = 0l <= v452;
                    bool v470;
                    if (v468){
                        bool v469;
                        v469 = v452 < 1l;
                        v470 = v469;
                    } else {
                        v470 = false;
                    }
                    bool v471;
                    v471 = v470 == false;
                    if (v471){
                        assert("The indices should be inside the range of the dimension." && v470);
                    } else {
                    }
                    int v473;
                    v473 = v452 * 4l;
                    int v474;
                    v474 = v467 + v473;
                    assert("Tensor range check" && 0 <= v452 && v452 < 1l);
                    assert("Tensor range check" && 0 <= v454 && v454 < 4l);
                    int v475;
                    v475 = 4l * v452;
                    int v476;
                    v476 = v475 + v454;
                    v445[v476] = v474;
                    v454 += 1l ;
                }
                v452 += 1l ;
            }
            int v477;
            v477 = 0l;
            while (while_method_1(v477)){
                int v479;
                v479 = 0l;
                while (while_method_2(v479)){
                    assert("Tensor range check" && 0 <= v477 && v477 < 1l);
                    assert("Tensor range check" && 0 <= v479 && v479 < 4l);
                    int v481;
                    v481 = 4l * v477;
                    int v482;
                    v482 = v481 + v479;
                    float v483;
                    v483 = v444[v482];
                    int v484;
                    v484 = v445[v482];
                    assert("Tensor range check" && 0 <= v484 && v484 < 4l);
                    float * v485;
                    v485 = v441+v484;
                    float v487;
                    v487 = atomicAdd(v485,v483);
                    v479 += 1l ;
                }
                v477 += 1l ;
            }
            int v488;
            v488 = 0l;
            while (while_method_1(v488)){
                assert("Tensor range check" && 0 <= v488 && v488 < 1l);
                assert("Tensor range check" && 0 <= v488 && v488 < 1l);
                v488 += 1l ;
            }
            assert("Tensor range check" && 0 <= v438 && v438 < 32l);
            /* void array set */;
            v427 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v490;
        v490 = threadIdx.x;
        assert("Tensor range check" && 0 <= v490 && v490 < 32l);
        /* void array index */;
    }
    int v491;
    v491 = threadIdx.x;
    bool v492;
    v492 = 0l <= v491;
    bool v493;
    v493 = v492 == false;
    if (v493){
        assert("The index needs to be zero or positive." && v492);
    } else {
    }
    int v495;
    v495 = v491 % 1l;
    bool v496;
    v496 = v491 < 32l;
    bool v497;
    v497 = v496 == false;
    if (v497){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v496);
    } else {
    }
    assert("Tensor range check" && 0 <= v491 && v491 < 32l);
    assert("Tensor range check" && 0 <= v495 && v495 < 1l);
    int v499;
    v499 = 4l * v495;
    int v500;
    v500 = 4l * v491;
    int v501;
    v501 = v500 + v499;
    assert("Tensor range check" && 0 <= v491 && v491 < 32l);
    assert("Tensor range check" && 0 <= v495 && v495 < 1l);
    int v502;
    v502 = 0l;
    while (while_method_4(v502)){
        assert("Tensor range check" && 0 <= v502 && v502 < 128l);
        int v504;
        v504 = 128l * v502;
        int v505;
        v505 = v504 + v501;
        float v506[4l];
        float v507[4l];
        float v508[4l];
        float v509[4l];
        float v510[4l];
        int v511[4l];
        int v512;
        v512 = 0l;
        while (while_method_1(v512)){
            assert("Tensor range check" && 0 <= v512 && v512 < 1l);
            int v514;
            v514 = 4l * v512;
            assert("Tensor range check" && 0 <= v512 && v512 < 1l);
            int v515;
            v515 = v514 + v505;
            int4* v516;
            v516 = reinterpret_cast<int4*>(v2 + v515);
            int4* v517;
            v517 = reinterpret_cast<int4*>(v506 + v514);
            assert("Pointer alignment check" && (unsigned long long)(v516) % 4l == 0 && (unsigned long long)(v517) % 4l == 0);
            *v517 = *v516;
            int4* v518;
            v518 = reinterpret_cast<int4*>(v4 + v515);
            int4* v519;
            v519 = reinterpret_cast<int4*>(v507 + v514);
            assert("Pointer alignment check" && (unsigned long long)(v518) % 4l == 0 && (unsigned long long)(v519) % 4l == 0);
            *v519 = *v518;
            int4* v520;
            v520 = reinterpret_cast<int4*>(v6 + v515);
            int4* v521;
            v521 = reinterpret_cast<int4*>(v508 + v514);
            assert("Pointer alignment check" && (unsigned long long)(v520) % 4l == 0 && (unsigned long long)(v521) % 4l == 0);
            *v521 = *v520;
            int4* v522;
            v522 = reinterpret_cast<int4*>(v8 + v515);
            int4* v523;
            v523 = reinterpret_cast<int4*>(v509 + v514);
            assert("Pointer alignment check" && (unsigned long long)(v522) % 4l == 0 && (unsigned long long)(v523) % 4l == 0);
            *v523 = *v522;
            int4* v524;
            v524 = reinterpret_cast<int4*>(v10 + v515);
            int4* v525;
            v525 = reinterpret_cast<int4*>(v510 + v514);
            assert("Pointer alignment check" && (unsigned long long)(v524) % 4l == 0 && (unsigned long long)(v525) % 4l == 0);
            *v525 = *v524;
            v512 += 1l ;
        }
        int v526;
        v526 = 0l;
        while (while_method_1(v526)){
            int v528;
            v528 = 0l;
            while (while_method_2(v528)){
                bool v530;
                v530 = 0l <= v528;
                bool v532;
                if (v530){
                    bool v531;
                    v531 = v528 < 4l;
                    v532 = v531;
                } else {
                    v532 = false;
                }
                bool v533;
                v533 = v532 == false;
                if (v533){
                    assert("The indices should be inside the range of the dimension." && v532);
                } else {
                }
                bool v535;
                v535 = 0l <= v495;
                bool v537;
                if (v535){
                    bool v536;
                    v536 = v495 < 1l;
                    v537 = v536;
                } else {
                    v537 = false;
                }
                bool v538;
                v538 = v537 == false;
                if (v538){
                    assert("The indices should be inside the range of the dimension." && v537);
                } else {
                }
                int v540;
                v540 = v495 * 4l;
                int v541;
                v541 = v528 + v540;
                bool v542;
                v542 = 0l <= v526;
                bool v544;
                if (v542){
                    bool v543;
                    v543 = v526 < 1l;
                    v544 = v543;
                } else {
                    v544 = false;
                }
                bool v545;
                v545 = v544 == false;
                if (v545){
                    assert("The indices should be inside the range of the dimension." && v544);
                } else {
                }
                int v547;
                v547 = v526 * 4l;
                int v548;
                v548 = v541 + v547;
                assert("Tensor range check" && 0 <= v526 && v526 < 1l);
                assert("Tensor range check" && 0 <= v528 && v528 < 4l);
                int v549;
                v549 = 4l * v526;
                int v550;
                v550 = v549 + v528;
                v511[v550] = v548;
                v528 += 1l ;
            }
            v526 += 1l ;
        }
        bool v551;
        v551 = v492 && v496;
        bool v552;
        v552 = v551 == false;
        if (v552){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v551);
        } else {
        }
        bool v554;
        v554 = 0l <= v502;
        bool v556;
        if (v554){
            bool v555;
            v555 = v502 < 128l;
            v556 = v555;
        } else {
            v556 = false;
        }
        bool v557;
        v557 = v556 == false;
        if (v557){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v556);
        } else {
        }
        int v559;
        v559 = v502 * 32l;
        int v560;
        v560 = v559 + v491;
        bool v561[4l];
        int v562;
        v562 = 0l;
        while (while_method_1(v562)){
            int v564;
            v564 = 0l;
            while (while_method_2(v564)){
                assert("Tensor range check" && 0 <= v562 && v562 < 1l);
                assert("Tensor range check" && 0 <= v564 && v564 < 4l);
                int v566;
                v566 = 4l * v562;
                int v567;
                v567 = v566 + v564;
                float v568;
                v568 = v508[v567];
                bool v569;
                v569 = v568 == 0.0f;
                bool v570;
                v570 = v569 != true;
                assert("Tensor range check" && 0 <= v562 && v562 < 1l);
                assert("Tensor range check" && 0 <= v564 && v564 < 4l);
                v561[v567] = v570;
                v564 += 1l ;
            }
            v562 += 1l ;
        }
        bool v571;
        v571 = false;
        int v572;
        v572 = 0l;
        while (while_method_1(v572)){
            int v574;
            v574 = 0l;
            while (while_method_2(v574)){
                assert("Tensor range check" && 0 <= v572 && v572 < 1l);
                assert("Tensor range check" && 0 <= v574 && v574 < 4l);
                int v576;
                v576 = 4l * v572;
                int v577;
                v577 = v576 + v574;
                bool v578;
                v578 = v561[v577];
                bool v579;
                v579 = v571 || v578;
                v571 = v579;
                v574 += 1l ;
            }
            v572 += 1l ;
        }
        auto v580 = cooperative_groups::coalesced_threads();
        int v581;
        v581 = threadIdx.x;
        auto v582 = cooperative_groups::labeled_partition(v580,v581);
        Closure7 v583{};
        bool v584;
        v584 = cooperative_groups::reduce(v582, v571, v583);
        if (v584){
            float v585[4l];
            int v586;
            v586 = 0l;
            while (while_method_1(v586)){
                int v588;
                v588 = 0l;
                while (while_method_2(v588)){
                    assert("Tensor range check" && 0 <= v586 && v586 < 1l);
                    assert("Tensor range check" && 0 <= v588 && v588 < 4l);
                    int v590;
                    v590 = 4l * v586;
                    int v591;
                    v591 = v590 + v588;
                    float v592;
                    v592 = v507[v591];
                    float v593;
                    v593 = v508[v591];
                    float v594;
                    v594 = v592 + v593;
                    bool v595;
                    v595 = 0.0f >= v594;
                    float v596;
                    if (v595){
                        v596 = 0.0f;
                    } else {
                        v596 = v594;
                    }
                    assert("Tensor range check" && 0 <= v586 && v586 < 1l);
                    assert("Tensor range check" && 0 <= v588 && v588 < 4l);
                    v585[v591] = v596;
                    v588 += 1l ;
                }
                v586 += 1l ;
            }
            float v597[4l];
            int v598;
            v598 = 0l;
            while (while_method_1(v598)){
                int v600;
                v600 = 0l;
                while (while_method_2(v600)){
                    assert("Tensor range check" && 0 <= v598 && v598 < 1l);
                    assert("Tensor range check" && 0 <= v600 && v600 < 4l);
                    int v602;
                    v602 = 4l * v598;
                    int v603;
                    v603 = v602 + v600;
                    float v604;
                    v604 = v585[v603];
                    bool v605;
                    v605 = 0.0f >= v604;
                    float v606;
                    if (v605){
                        v606 = 0.0f;
                    } else {
                        v606 = v604;
                    }
                    assert("Tensor range check" && 0 <= v598 && v598 < 1l);
                    assert("Tensor range check" && 0 <= v600 && v600 < 4l);
                    v597[v603] = v606;
                    v600 += 1l ;
                }
                v598 += 1l ;
            }
            float v607;
            v607 = 0.0f;
            int v608;
            v608 = 0l;
            while (while_method_1(v608)){
                int v610;
                v610 = 0l;
                while (while_method_2(v610)){
                    assert("Tensor range check" && 0 <= v608 && v608 < 1l);
                    assert("Tensor range check" && 0 <= v610 && v610 < 4l);
                    int v612;
                    v612 = 4l * v608;
                    int v613;
                    v613 = v612 + v610;
                    float v614;
                    v614 = v597[v613];
                    float v615;
                    v615 = v607 + v614;
                    v607 = v615;
                    v610 += 1l ;
                }
                v608 += 1l ;
            }
            auto v616 = cooperative_groups::coalesced_threads();
            int v617;
            v617 = threadIdx.x;
            auto v618 = cooperative_groups::labeled_partition(v616,v617);
            Closure0 v619{};
            float v620;
            v620 = cooperative_groups::reduce(v618, v607, v619);
            float v621[4l];
            int v622;
            v622 = 0l;
            while (while_method_1(v622)){
                int v624;
                v624 = 0l;
                while (while_method_2(v624)){
                    assert("Tensor range check" && 0 <= v622 && v622 < 1l);
                    assert("Tensor range check" && 0 <= v624 && v624 < 4l);
                    int v626;
                    v626 = 4l * v622;
                    int v627;
                    v627 = v626 + v624;
                    float v628;
                    v628 = v597[v627];
                    bool v629;
                    v629 = v620 == 0.0f;
                    bool v630;
                    v630 = v629 != true;
                    float v632;
                    if (v630){
                        float v631;
                        v631 = v628 / v620;
                        v632 = v631;
                    } else {
                        v632 = 0.25f;
                    }
                    assert("Tensor range check" && 0 <= v622 && v622 < 1l);
                    assert("Tensor range check" && 0 <= v624 && v624 < 4l);
                    v621[v627] = v632;
                    v624 += 1l ;
                }
                v622 += 1l ;
            }
            float v633[4l];
            int v634;
            v634 = 0l;
            while (while_method_1(v634)){
                int v636;
                v636 = 0l;
                while (while_method_2(v636)){
                    assert("Tensor range check" && 0 <= v634 && v634 < 1l);
                    assert("Tensor range check" && 0 <= v636 && v636 < 4l);
                    int v638;
                    v638 = 4l * v634;
                    int v639;
                    v639 = v638 + v636;
                    float v640;
                    v640 = v506[v639];
                    float v641;
                    v641 = v621[v639];
                    float v642;
                    v642 = v640 + v641;
                    assert("Tensor range check" && 0 <= v634 && v634 < 1l);
                    assert("Tensor range check" && 0 <= v636 && v636 < 4l);
                    v633[v639] = v642;
                    v636 += 1l ;
                }
                v634 += 1l ;
            }
            float v643[4l];
            int v644;
            v644 = 0l;
            while (while_method_1(v644)){
                int v646;
                v646 = 0l;
                while (while_method_2(v646)){
                    assert("Tensor range check" && 0 <= v644 && v644 < 1l);
                    assert("Tensor range check" && 0 <= v646 && v646 < 4l);
                    int v648;
                    v648 = 4l * v644;
                    int v649;
                    v649 = v648 + v646;
                    float v650;
                    v650 = v633[v649];
                    float v651;
                    v651 = -v650;
                    bool v652;
                    v652 = v650 >= v651;
                    float v653;
                    if (v652){
                        v653 = v650;
                    } else {
                        v653 = v651;
                    }
                    assert("Tensor range check" && 0 <= v644 && v644 < 1l);
                    assert("Tensor range check" && 0 <= v646 && v646 < 4l);
                    v643[v649] = v653;
                    v646 += 1l ;
                }
                v644 += 1l ;
            }
            float v654;
            v654 = 0.0f;
            int v655;
            v655 = 0l;
            while (while_method_1(v655)){
                int v657;
                v657 = 0l;
                while (while_method_2(v657)){
                    assert("Tensor range check" && 0 <= v655 && v655 < 1l);
                    assert("Tensor range check" && 0 <= v657 && v657 < 4l);
                    int v659;
                    v659 = 4l * v655;
                    int v660;
                    v660 = v659 + v657;
                    float v661;
                    v661 = v643[v660];
                    float v662;
                    v662 = v654 + v661;
                    v654 = v662;
                    v657 += 1l ;
                }
                v655 += 1l ;
            }
            auto v663 = cooperative_groups::coalesced_threads();
            int v664;
            v664 = threadIdx.x;
            auto v665 = cooperative_groups::labeled_partition(v663,v664);
            float v666;
            v666 = cooperative_groups::reduce(v665, v654, v619);
            bool v667;
            v667 = v666 > 100.0f;
            float v669;
            if (v667){
                float v668;
                v668 = 100.0f / v666;
                v669 = v668;
            } else {
                v669 = 1.0f;
            }
            float v670[4l];
            int v671;
            v671 = 0l;
            while (while_method_1(v671)){
                int v673;
                v673 = 0l;
                while (while_method_2(v673)){
                    assert("Tensor range check" && 0 <= v671 && v671 < 1l);
                    assert("Tensor range check" && 0 <= v673 && v673 < 4l);
                    int v675;
                    v675 = 4l * v671;
                    int v676;
                    v676 = v675 + v673;
                    float v677;
                    v677 = v643[v676];
                    float v678;
                    v678 = v669 * v677;
                    assert("Tensor range check" && 0 <= v671 && v671 < 1l);
                    assert("Tensor range check" && 0 <= v673 && v673 < 4l);
                    v670[v676] = v678;
                    v673 += 1l ;
                }
                v671 += 1l ;
            }
            int v679;
            v679 = 0l;
            while (while_method_1(v679)){
                int v681;
                v681 = 0l;
                while (while_method_2(v681)){
                    assert("Tensor range check" && 0 <= v679 && v679 < 1l);
                    assert("Tensor range check" && 0 <= v681 && v681 < 4l);
                    int v683;
                    v683 = 4l * v679;
                    int v684;
                    v684 = v683 + v681;
                    float v685;
                    v685 = v585[v684];
                    float v686;
                    v686 = v670[v684];
                    float v687;
                    v687 = v509[v684];
                    float v688;
                    v688 = v510[v684];
                    assert("Tensor range check" && 0 <= v679 && v679 < 1l);
                    assert("Tensor range check" && 0 <= v681 && v681 < 4l);
                    v506[v684] = v686;
                    v507[v684] = v685;
                    v508[v684] = 0.0f;
                    v509[v684] = v687;
                    v510[v684] = v688;
                    v681 += 1l ;
                }
                v679 += 1l ;
            }
        } else {
        }
        assert("Tensor range check" && 0 <= v502 && v502 < 128l);
        int v689;
        v689 = 0l;
        while (while_method_1(v689)){
            assert("Tensor range check" && 0 <= v689 && v689 < 1l);
            int v691;
            v691 = 4l * v689;
            int v692;
            v692 = v691 + v505;
            assert("Tensor range check" && 0 <= v689 && v689 < 1l);
            int4* v693;
            v693 = reinterpret_cast<int4*>(v506 + v691);
            int4* v694;
            v694 = reinterpret_cast<int4*>(v2 + v692);
            assert("Pointer alignment check" && (unsigned long long)(v693) % 4l == 0 && (unsigned long long)(v694) % 4l == 0);
            *v694 = *v693;
            int4* v695;
            v695 = reinterpret_cast<int4*>(v507 + v691);
            int4* v696;
            v696 = reinterpret_cast<int4*>(v4 + v692);
            assert("Pointer alignment check" && (unsigned long long)(v695) % 4l == 0 && (unsigned long long)(v696) % 4l == 0);
            *v696 = *v695;
            int4* v697;
            v697 = reinterpret_cast<int4*>(v508 + v691);
            int4* v698;
            v698 = reinterpret_cast<int4*>(v6 + v692);
            assert("Pointer alignment check" && (unsigned long long)(v697) % 4l == 0 && (unsigned long long)(v698) % 4l == 0);
            *v698 = *v697;
            int4* v699;
            v699 = reinterpret_cast<int4*>(v509 + v691);
            int4* v700;
            v700 = reinterpret_cast<int4*>(v8 + v692);
            assert("Pointer alignment check" && (unsigned long long)(v699) % 4l == 0 && (unsigned long long)(v700) % 4l == 0);
            *v700 = *v699;
            int4* v701;
            v701 = reinterpret_cast<int4*>(v510 + v691);
            int4* v702;
            v702 = reinterpret_cast<int4*>(v10 + v692);
            assert("Pointer alignment check" && (unsigned long long)(v701) % 4l == 0 && (unsigned long long)(v702) % 4l == 0);
            *v702 = *v701;
            v689 += 1l ;
        }
        v502 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
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
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39')
options.append('--restrict')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def main_body():
    v0 = cp.empty(364544,dtype=cp.uint8)
    v1 = cp.empty(0,dtype=cp.uint8)
    v3 = v0[0:0+4*16384].view(cp.float32)
    v5 = v0[65536:65536+4*16384].view(cp.float32)
    v7 = v0[131072:131072+4*16384].view(cp.float32)
    v9 = v0[196608:196608+4*16384].view(cp.float32)
    v11 = v0[262144:262144+4*16384].view(cp.float32)
    v13 = v0[327680:327680+4*512].view(cp.int32)
    v15 = v0[329728:329728+4*512].view(cp.float32)
    v17 = v0[331776:331776+4*512].view(cp.int32)
    v19 = v0[333824:333824+4*512].view(cp.int32)
    v21 = v0[335872:335872+8*1024].view(cp.float64)
    v23 = v0[344064:344064+8*1024].view(cp.float64)
    v25 = v0[352256:352256+4*2048].view(cp.float32)
    v27 = v0[360448:360448+4*512].view(cp.float32)
    v29 = v0[362496:362496+4*512].view(cp.float32)
    v3[:] = 0
    del v3
    v5[:] = 0
    del v5
    v7[:] = 0
    del v7
    v9[:] = 0
    del v9
    v11[:] = 0
    del v11
    v13[:] = 0
    del v13
    v15[:] = 0
    del v15
    v17[:] = 0
    del v17
    v19[:] = 0
    del v19
    v21[:] = 0
    del v21
    v23[:] = 0
    del v23
    v25[:] = 0
    del v25
    v27[:] = 0
    del v27
    v29[:] = 0
    del v29
    v30 = 0
    v31 = raw_module.get_function(f"entry{v30}")
    del v30
    v31.max_dynamic_shared_size_bytes = 0 
    v31((1,),(32,),(v1, v0),shared_mem=0)
    del v0, v1, v31
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
