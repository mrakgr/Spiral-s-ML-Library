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
__device__ Tuple0 get_action_1(float * v0, float * v1, float * v2, float * v3, float * v4, int v5);
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
__device__ Tuple0 get_action_1(float * v0, float * v1, float * v2, float * v3, float * v4, int v5){
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
    Tuple0 tmp12 = get_action_1(v0, v1, v2, v3, v4, v19);
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
extern "C" __global__ void entry0(int * v0, float * v1, int * v2, int * v3, double * v4, double * v5, float * v6, float * v7, float * v8, float * v9, float * v10, float * v11, float * v12, float * v13) {
    int v14;
    v14 = threadIdx.x;
    int v15 = 0l;
    double v16[2l];
    double v17[2l];
    int v18;
    v18 = 0l;
    while (while_method_0(v18)){
        assert("Tensor range check" && 0 <= v18 && v18 < 2l);
        v16[v18] = 0.0;
        v17[v18] = 0.0;
        v18 += 1l ;
    }
    int v20;
    v20 = 235l;
    int v21;
    v21 = 0l;
    push_0(v9, v10, v11, v12, v13, v0, v1, v2, v3, v4, v5, v6, v7, v8, v14, v15, v16, v17, v21, v20);
    int v22;
    v22 = 212l;
    int v23;
    v23 = 1l;
    push_0(v9, v10, v11, v12, v13, v0, v1, v2, v3, v4, v5, v6, v7, v8, v14, v15, v16, v17, v23, v22);
    int v24;
    v24 = 790l;
    int v25;
    v25 = 0l;
    push_0(v9, v10, v11, v12, v13, v0, v1, v2, v3, v4, v5, v6, v7, v8, v14, v15, v16, v17, v25, v24);
    int v26;
    v26 = 343l;
    int v27;
    v27 = 1l;
    push_0(v9, v10, v11, v12, v13, v0, v1, v2, v3, v4, v5, v6, v7, v8, v14, v15, v16, v17, v27, v26);
    int v28;
    v28 = 457l;
    int v29;
    v29 = 0l;
    push_0(v9, v10, v11, v12, v13, v0, v1, v2, v3, v4, v5, v6, v7, v8, v14, v15, v16, v17, v29, v28);
    int v30;
    v30 = 3447l;
    int v31;
    v31 = 1l;
    push_0(v9, v10, v11, v12, v13, v0, v1, v2, v3, v4, v5, v6, v7, v8, v14, v15, v16, v17, v31, v30);
    int v32 = v15;
    int v33; float v34;
    Tuple1 tmp13 = Tuple1{v32, -13.0f};
    v33 = tmp13.v0; v34 = tmp13.v1;
    while (while_method_3(v33)){
        v33 -= 1l ;
        assert("Tensor range check" && 0 <= v33 && v33 < 16l);
        assert("Tensor range check" && 0 <= v14 && v14 < 32l);
        int v36;
        v36 = 32l * v33;
        int v37;
        v37 = v36 + v14;
        int v38;
        v38 = v0[v37];
        float v39;
        v39 = v1[v37];
        int v40;
        v40 = v2[v37];
        int v41;
        v41 = v3[v37];
        assert("Tensor range check" && 0 <= v41 && v41 < 4096l);
        int v42;
        v42 = 4l * v41;
        assert("Tensor range check" && 0 <= v33 && v33 < 16l);
        int v43;
        v43 = 128l * v33;
        assert("Tensor range check" && 0 <= v14 && v14 < 32l);
        int v44;
        v44 = 4l * v14;
        int v45;
        v45 = v44 + v43;
        assert("Tensor range check" && 0 <= v33 && v33 < 16l);
        int v46;
        v46 = 64l * v33;
        double * v47;
        v47 = v4+v46;
        double * v49;
        v49 = v5+v46;
        assert("Tensor range check" && 0 <= v14 && v14 < 32l);
        int v51;
        v51 = 2l * v14;
        double v52[2l];
        int v53;
        v53 = 0l;
        while (while_method_0(v53)){
            assert("Tensor range check" && 0 <= v53 && v53 < 2l);
            int v55;
            v55 = v53 + v51;
            double v56;
            v56 = v47[v55];
            bool v57;
            v57 = v40 == v53;
            double v58;
            if (v57){
                v58 = 0.0;
            } else {
                v58 = v56;
            }
            assert("Tensor range check" && 0 <= v53 && v53 < 2l);
            v52[v53] = v58;
            v53 += 1l ;
        }
        double v59;
        v59 = 0.0;
        int v60;
        v60 = 0l;
        while (while_method_0(v60)){
            assert("Tensor range check" && 0 <= v60 && v60 < 2l);
            double v62;
            v62 = v52[v60];
            double v63;
            v63 = v59 + v62;
            v59 = v63;
            v60 += 1l ;
        }
        double v64;
        v64 = 0.0;
        int v65;
        v65 = 0l;
        while (while_method_0(v65)){
            assert("Tensor range check" && 0 <= v65 && v65 < 2l);
            int v67;
            v67 = v65 + v51;
            double v68;
            v68 = v49[v67];
            double v69;
            v69 = v64 + v68;
            v64 = v69;
            v65 += 1l ;
        }
        double v70;
        v70 = v59 - v64;
        double v71;
        v71 = exp(v70);
        float v72;
        v72 = (float)v71;
        float v73;
        v73 = v34 * v72;
        assert("Tensor range check" && 0 <= v33 && v33 < 16l);
        assert("Tensor range check" && 0 <= v14 && v14 < 32l);
        v7[v37] = v73;
        v8[v37] = v72;
        float * v74;
        v74 = v10+v42;
        float * v76;
        v76 = v12+v42;
        float * v78;
        v78 = v13+v42;
        float * v80;
        v80 = v6+v45;
        __shared__ float v82[32l];
        __shared__ float v83[32l];
        __shared__ int v84[32l];
        __shared__ int v85[32l];
        __shared__ double * v86[32l];
        __shared__ double * v87[32l];
        __shared__ float * v88[32l];
        __shared__ float * v89[32l];
        __shared__ float * v90[32l];
        __shared__ float * v91[32l];
        /* void shared array create v92 */;
        __shared__ float v93[32l];
        int v94;
        v94 = threadIdx.x;
        assert("Tensor range check" && 0 <= v94 && v94 < 32l);
        v82[v94] = v34;
        v83[v94] = v39;
        v84[v94] = v38;
        v85[v94] = v40;
        v86[v94] = v47;
        v87[v94] = v49;
        v88[v94] = v74;
        v89[v94] = v76;
        v90[v94] = v78;
        v91[v94] = v80;
        /* void array set */;
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v95;
        v95 = threadIdx.x;
        bool v96;
        v96 = 0l <= v95;
        bool v97;
        v97 = v96 == false;
        if (v97){
            assert("The index needs to be zero or positive." && v96);
        } else {
        }
        int v99;
        v99 = v95 % 1l;
        bool v100;
        v100 = v95 < 32l;
        bool v101;
        v101 = v100 == false;
        if (v101){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v100);
        } else {
        }
        assert("Tensor range check" && 0 <= v95 && v95 < 32l);
        int v103;
        v103 = 0l;
        while (while_method_1(v103)){
            bool v105;
            v105 = v96 && v100;
            bool v106;
            v106 = v105 == false;
            if (v106){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v105);
            } else {
            }
            bool v108;
            v108 = 0l <= v103;
            bool v110;
            if (v108){
                bool v109;
                v109 = v103 < 1l;
                v110 = v109;
            } else {
                v110 = false;
            }
            bool v111;
            v111 = v110 == false;
            if (v111){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v110);
            } else {
            }
            int v113;
            v113 = v103 * 32l;
            int v114;
            v114 = v113 + v95;
            assert("Tensor range check" && 0 <= v103 && v103 < 1l);
            int v115;
            v115 = 32l * v103;
            int v116;
            v116 = v115 + v95;
            float v117;
            v117 = v82[v116];
            float v118;
            v118 = v83[v116];
            int v119;
            v119 = v84[v116];
            int v120;
            v120 = v85[v116];
            double * v121;
            v121 = v86[v116];
            double * v122;
            v122 = v87[v116];
            float * v123;
            v123 = v88[v116];
            float * v124;
            v124 = v89[v116];
            float * v125;
            v125 = v90[v116];
            float * v126;
            v126 = v91[v116];
            /* void array index */;
            assert("Tensor range check" && 0 <= v99 && v99 < 1l);
            int v127;
            v127 = 4l * v99;
            float v128[4l];
            float v129[4l];
            float v130[4l];
            int v131[4l];
            int v132;
            v132 = 0l;
            while (while_method_1(v132)){
                assert("Tensor range check" && 0 <= v132 && v132 < 1l);
                int v134;
                v134 = 4l * v132;
                assert("Tensor range check" && 0 <= v132 && v132 < 1l);
                int v135;
                v135 = v134 + v127;
                int4* v136;
                v136 = reinterpret_cast<int4*>(v123 + v135);
                int4* v137;
                v137 = reinterpret_cast<int4*>(v128 + v134);
                assert("Pointer alignment check" && (unsigned long long)(v136) % 4l == 0 && (unsigned long long)(v137) % 4l == 0);
                *v137 = *v136;
                int4* v138;
                v138 = reinterpret_cast<int4*>(v124 + v135);
                int4* v139;
                v139 = reinterpret_cast<int4*>(v129 + v134);
                assert("Pointer alignment check" && (unsigned long long)(v138) % 4l == 0 && (unsigned long long)(v139) % 4l == 0);
                *v139 = *v138;
                int4* v140;
                v140 = reinterpret_cast<int4*>(v125 + v135);
                int4* v141;
                v141 = reinterpret_cast<int4*>(v130 + v134);
                assert("Pointer alignment check" && (unsigned long long)(v140) % 4l == 0 && (unsigned long long)(v141) % 4l == 0);
                *v141 = *v140;
                v132 += 1l ;
            }
            int v142;
            v142 = 0l;
            while (while_method_1(v142)){
                int v144;
                v144 = 0l;
                while (while_method_2(v144)){
                    bool v146;
                    v146 = 0l <= v144;
                    bool v148;
                    if (v146){
                        bool v147;
                        v147 = v144 < 4l;
                        v148 = v147;
                    } else {
                        v148 = false;
                    }
                    bool v149;
                    v149 = v148 == false;
                    if (v149){
                        assert("The indices should be inside the range of the dimension." && v148);
                    } else {
                    }
                    bool v151;
                    v151 = 0l <= v99;
                    bool v153;
                    if (v151){
                        bool v152;
                        v152 = v99 < 1l;
                        v153 = v152;
                    } else {
                        v153 = false;
                    }
                    bool v154;
                    v154 = v153 == false;
                    if (v154){
                        assert("The indices should be inside the range of the dimension." && v153);
                    } else {
                    }
                    int v156;
                    v156 = v99 * 4l;
                    int v157;
                    v157 = v144 + v156;
                    bool v158;
                    v158 = 0l <= v142;
                    bool v160;
                    if (v158){
                        bool v159;
                        v159 = v142 < 1l;
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
                    int v163;
                    v163 = v142 * 4l;
                    int v164;
                    v164 = v157 + v163;
                    assert("Tensor range check" && 0 <= v142 && v142 < 1l);
                    assert("Tensor range check" && 0 <= v144 && v144 < 4l);
                    int v165;
                    v165 = 4l * v142;
                    int v166;
                    v166 = v165 + v144;
                    v131[v166] = v164;
                    v144 += 1l ;
                }
                v142 += 1l ;
            }
            float v167[4l];
            int v168;
            v168 = 0l;
            while (while_method_1(v168)){
                int v170;
                v170 = 0l;
                while (while_method_2(v170)){
                    assert("Tensor range check" && 0 <= v168 && v168 < 1l);
                    assert("Tensor range check" && 0 <= v170 && v170 < 4l);
                    int v172;
                    v172 = 4l * v168;
                    int v173;
                    v173 = v172 + v170;
                    float v174;
                    v174 = v129[v173];
                    float v175;
                    v175 = v130[v173];
                    bool v176;
                    v176 = v175 == 0.0f;
                    bool v177;
                    v177 = v176 != true;
                    float v179;
                    if (v177){
                        float v178;
                        v178 = v174 / v175;
                        v179 = v178;
                    } else {
                        v179 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v168 && v168 < 1l);
                    assert("Tensor range check" && 0 <= v170 && v170 < 4l);
                    v167[v173] = v179;
                    v170 += 1l ;
                }
                v168 += 1l ;
            }
            bool v180[4l];
            int v181;
            v181 = 0l;
            while (while_method_1(v181)){
                int v183;
                v183 = 0l;
                while (while_method_2(v183)){
                    assert("Tensor range check" && 0 <= v181 && v181 < 1l);
                    assert("Tensor range check" && 0 <= v183 && v183 < 4l);
                    int v185;
                    v185 = 4l * v181;
                    int v186;
                    v186 = v185 + v183;
                    float v187;
                    v187 = v128[v186];
                    int v188;
                    v188 = v131[v186];
                    bool v189;
                    v189 = v188 < 3l;
                    assert("Tensor range check" && 0 <= v181 && v181 < 1l);
                    assert("Tensor range check" && 0 <= v183 && v183 < 4l);
                    v180[v186] = v189;
                    v183 += 1l ;
                }
                v181 += 1l ;
            }
            float v190[4l];
            int v191;
            v191 = 0l;
            while (while_method_1(v191)){
                int v193;
                v193 = 0l;
                while (while_method_2(v193)){
                    assert("Tensor range check" && 0 <= v191 && v191 < 1l);
                    assert("Tensor range check" && 0 <= v193 && v193 < 4l);
                    int v195;
                    v195 = 4l * v191;
                    int v196;
                    v196 = v195 + v193;
                    float v197;
                    v197 = v128[v196];
                    bool v198;
                    v198 = v180[v196];
                    float v201;
                    if (v198){
                        bool v199;
                        v199 = 0.0f >= v197;
                        if (v199){
                            v201 = 0.0f;
                        } else {
                            v201 = v197;
                        }
                    } else {
                        v201 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v191 && v191 < 1l);
                    assert("Tensor range check" && 0 <= v193 && v193 < 4l);
                    v190[v196] = v201;
                    v193 += 1l ;
                }
                v191 += 1l ;
            }
            float v202;
            v202 = 0.0f;
            int v203;
            v203 = 0l;
            while (while_method_1(v203)){
                int v205;
                v205 = 0l;
                while (while_method_2(v205)){
                    assert("Tensor range check" && 0 <= v203 && v203 < 1l);
                    assert("Tensor range check" && 0 <= v205 && v205 < 4l);
                    int v207;
                    v207 = 4l * v203;
                    int v208;
                    v208 = v207 + v205;
                    float v209;
                    v209 = v190[v208];
                    float v210;
                    v210 = v202 + v209;
                    v202 = v210;
                    v205 += 1l ;
                }
                v203 += 1l ;
            }
            auto v211 = cooperative_groups::coalesced_threads();
            int v212;
            v212 = threadIdx.x;
            auto v213 = cooperative_groups::labeled_partition(v211,v212);
            Closure0 v214{};
            float v215;
            v215 = cooperative_groups::reduce(v213, v202, v214);
            int v216[4l];
            int v217;
            v217 = 0l;
            while (while_method_1(v217)){
                int v219;
                v219 = 0l;
                while (while_method_2(v219)){
                    assert("Tensor range check" && 0 <= v217 && v217 < 1l);
                    assert("Tensor range check" && 0 <= v219 && v219 < 4l);
                    int v221;
                    v221 = 4l * v217;
                    int v222;
                    v222 = v221 + v219;
                    bool v223;
                    v223 = v180[v222];
                    int v224;
                    if (v223){
                        v224 = 1l;
                    } else {
                        v224 = 0l;
                    }
                    assert("Tensor range check" && 0 <= v217 && v217 < 1l);
                    assert("Tensor range check" && 0 <= v219 && v219 < 4l);
                    v216[v222] = v224;
                    v219 += 1l ;
                }
                v217 += 1l ;
            }
            int v225;
            v225 = 0l;
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
                    v232 = v216[v231];
                    int v233;
                    v233 = v225 + v232;
                    v225 = v233;
                    v228 += 1l ;
                }
                v226 += 1l ;
            }
            auto v234 = cooperative_groups::coalesced_threads();
            int v235;
            v235 = threadIdx.x;
            auto v236 = cooperative_groups::labeled_partition(v234,v235);
            Closure1 v237{};
            int v238;
            v238 = cooperative_groups::reduce(v236, v225, v237);
            float v239;
            v239 = (float)v238;
            float v240;
            v240 = 1.0f / v239;
            float v241[4l];
            int v242;
            v242 = 0l;
            while (while_method_1(v242)){
                int v244;
                v244 = 0l;
                while (while_method_2(v244)){
                    assert("Tensor range check" && 0 <= v242 && v242 < 1l);
                    assert("Tensor range check" && 0 <= v244 && v244 < 4l);
                    int v246;
                    v246 = 4l * v242;
                    int v247;
                    v247 = v246 + v244;
                    float v248;
                    v248 = v190[v247];
                    bool v249;
                    v249 = v180[v247];
                    bool v250;
                    v250 = v249 == false;
                    float v255;
                    if (v250){
                        v255 = 0.0f;
                    } else {
                        bool v251;
                        v251 = v215 == 0.0f;
                        bool v252;
                        v252 = v251 != true;
                        if (v252){
                            float v253;
                            v253 = v248 / v215;
                            v255 = v253;
                        } else {
                            v255 = v240;
                        }
                    }
                    assert("Tensor range check" && 0 <= v242 && v242 < 1l);
                    assert("Tensor range check" && 0 <= v244 && v244 < 4l);
                    v241[v247] = v255;
                    v244 += 1l ;
                }
                v242 += 1l ;
            }
            float v256[4l];
            int v257;
            v257 = 0l;
            while (while_method_1(v257)){
                int v259;
                v259 = 0l;
                while (while_method_2(v259)){
                    assert("Tensor range check" && 0 <= v257 && v257 < 1l);
                    assert("Tensor range check" && 0 <= v259 && v259 < 4l);
                    int v261;
                    v261 = 4l * v257;
                    int v262;
                    v262 = v261 + v259;
                    float v263;
                    v263 = v167[v262];
                    int v264;
                    v264 = v131[v262];
                    bool v265;
                    v265 = v119 == v264;
                    float v268;
                    if (v265){
                        float v266;
                        v266 = v117 - v263;
                        float v267;
                        v267 = v266 / v118;
                        v268 = v267;
                    } else {
                        v268 = 0.0f;
                    }
                    float v269;
                    v269 = v268 + v263;
                    assert("Tensor range check" && 0 <= v257 && v257 < 1l);
                    assert("Tensor range check" && 0 <= v259 && v259 < 4l);
                    v256[v262] = v269;
                    v259 += 1l ;
                }
                v257 += 1l ;
            }
            float v270[4l];
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
                    float v277;
                    v277 = v241[v276];
                    float v278;
                    v278 = v256[v276];
                    float v279;
                    v279 = v277 * v278;
                    assert("Tensor range check" && 0 <= v271 && v271 < 1l);
                    assert("Tensor range check" && 0 <= v273 && v273 < 4l);
                    v270[v276] = v279;
                    v273 += 1l ;
                }
                v271 += 1l ;
            }
            float v280;
            v280 = 0.0f;
            int v281;
            v281 = 0l;
            while (while_method_1(v281)){
                int v283;
                v283 = 0l;
                while (while_method_2(v283)){
                    assert("Tensor range check" && 0 <= v281 && v281 < 1l);
                    assert("Tensor range check" && 0 <= v283 && v283 < 4l);
                    int v285;
                    v285 = 4l * v281;
                    int v286;
                    v286 = v285 + v283;
                    float v287;
                    v287 = v270[v286];
                    float v288;
                    v288 = v280 + v287;
                    v280 = v288;
                    v283 += 1l ;
                }
                v281 += 1l ;
            }
            auto v289 = cooperative_groups::coalesced_threads();
            int v290;
            v290 = threadIdx.x;
            auto v291 = cooperative_groups::labeled_partition(v289,v290);
            float v292;
            v292 = cooperative_groups::reduce(v291, v280, v214);
            assert("Tensor range check" && 0 <= v114 && v114 < 32l);
            int v293;
            v293 = 2l * v114;
            double v294[2l];
            int v295;
            v295 = 0l;
            while (while_method_0(v295)){
                assert("Tensor range check" && 0 <= v295 && v295 < 2l);
                int v297;
                v297 = v295 + v293;
                double v298;
                v298 = v121[v297];
                bool v299;
                v299 = v120 == v295;
                double v300;
                if (v299){
                    v300 = 0.0;
                } else {
                    v300 = v298;
                }
                assert("Tensor range check" && 0 <= v295 && v295 < 2l);
                v294[v295] = v300;
                v295 += 1l ;
            }
            double v301;
            v301 = 0.0;
            int v302;
            v302 = 0l;
            while (while_method_0(v302)){
                assert("Tensor range check" && 0 <= v302 && v302 < 2l);
                double v304;
                v304 = v294[v302];
                double v305;
                v305 = v301 + v304;
                v301 = v305;
                v302 += 1l ;
            }
            double v306;
            v306 = 0.0;
            int v307;
            v307 = 0l;
            while (while_method_0(v307)){
                assert("Tensor range check" && 0 <= v307 && v307 < 2l);
                int v309;
                v309 = v307 + v293;
                double v310;
                v310 = v122[v309];
                double v311;
                v311 = v306 + v310;
                v306 = v311;
                v307 += 1l ;
            }
            double v312;
            v312 = v301 - v306;
            double v313;
            v313 = exp(v312);
            float v314;
            v314 = (float)v313;
            float v315[4l];
            int v316;
            v316 = 0l;
            while (while_method_1(v316)){
                int v318;
                v318 = 0l;
                while (while_method_2(v318)){
                    assert("Tensor range check" && 0 <= v316 && v316 < 1l);
                    assert("Tensor range check" && 0 <= v318 && v318 < 4l);
                    int v320;
                    v320 = 4l * v316;
                    int v321;
                    v321 = v320 + v318;
                    float v322;
                    v322 = v256[v321];
                    float v323;
                    v323 = v322 - v292;
                    float v324;
                    v324 = v314 * v323;
                    assert("Tensor range check" && 0 <= v316 && v316 < 1l);
                    assert("Tensor range check" && 0 <= v318 && v318 < 4l);
                    v315[v321] = v324;
                    v318 += 1l ;
                }
                v316 += 1l ;
            }
            int v325;
            v325 = 0l;
            while (while_method_1(v325)){
                assert("Tensor range check" && 0 <= v325 && v325 < 1l);
                int v327;
                v327 = 4l * v325;
                int v328;
                v328 = v327 + v127;
                assert("Tensor range check" && 0 <= v325 && v325 < 1l);
                int4* v329;
                v329 = reinterpret_cast<int4*>(v315 + v327);
                int4* v330;
                v330 = reinterpret_cast<int4*>(v126 + v328);
                assert("Pointer alignment check" && (unsigned long long)(v329) % 4l == 0 && (unsigned long long)(v330) % 4l == 0);
                *v330 = *v329;
                v325 += 1l ;
            }
            assert("Tensor range check" && 0 <= v114 && v114 < 32l);
            v93[v114] = v292;
            v103 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v331;
        v331 = threadIdx.x;
        assert("Tensor range check" && 0 <= v331 && v331 < 32l);
        float v332;
        v332 = v93[v331];
        v34 = v332;
    }
    cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v333 = console_lock;
    auto v334 = cooperative_groups::coalesced_threads();
    v333.acquire();
    printf("{%s = %f}\n","fin_reward", v34);
    v333.release();
    v334.sync() ;
    int v337 = v15;
    int v338;
    v338 = v337;
    while (while_method_3(v338)){
        v338 -= 1l ;
        assert("Tensor range check" && 0 <= v338 && v338 < 16l);
        assert("Tensor range check" && 0 <= v14 && v14 < 32l);
        int v340;
        v340 = 32l * v338;
        int v341;
        v341 = v340 + v14;
        int v342;
        v342 = v0[v341];
        float v343;
        v343 = v1[v341];
        int v344;
        v344 = v2[v341];
        int v345;
        v345 = v3[v341];
        assert("Tensor range check" && 0 <= v338 && v338 < 16l);
        assert("Tensor range check" && 0 <= v14 && v14 < 32l);
        float v346;
        v346 = v7[v341];
        float v347;
        v347 = v8[v341];
        assert("Tensor range check" && 0 <= v345 && v345 < 4096l);
        int v348;
        v348 = 4l * v345;
        float * v349;
        v349 = v9+v348;
        float * v351;
        v351 = v10+v348;
        float * v353;
        v353 = v11+v348;
        float * v355;
        v355 = v12+v348;
        float * v357;
        v357 = v13+v348;
        assert("Tensor range check" && 0 <= v338 && v338 < 16l);
        int v359;
        v359 = 128l * v338;
        assert("Tensor range check" && 0 <= v14 && v14 < 32l);
        int v360;
        v360 = 4l * v14;
        int v361;
        v361 = v360 + v359;
        assert("Tensor range check" && 0 <= v342 && v342 < 4l);
        float * v362;
        v362 = v355+v342;
        float * v364;
        v364 = v357+v342;
        float v366;
        v366 = atomicAdd(v362,v346);
        float v367;
        v367 = atomicAdd(v364,v347);
        float * v368;
        v368 = v6+v361;
        __shared__ float * v370[32l];
        __shared__ float * v371[32l];
        /* void shared array create v372 */;
        /* void shared array create v373 */;
        int v374;
        v374 = threadIdx.x;
        assert("Tensor range check" && 0 <= v374 && v374 < 32l);
        v370[v374] = v353;
        v371[v374] = v368;
        /* void array set */;
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v375;
        v375 = threadIdx.x;
        bool v376;
        v376 = 0l <= v375;
        bool v377;
        v377 = v376 == false;
        if (v377){
            assert("The index needs to be zero or positive." && v376);
        } else {
        }
        int v379;
        v379 = v375 % 1l;
        bool v380;
        v380 = v375 < 32l;
        bool v381;
        v381 = v380 == false;
        if (v381){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v380);
        } else {
        }
        assert("Tensor range check" && 0 <= v375 && v375 < 32l);
        int v383;
        v383 = 0l;
        while (while_method_1(v383)){
            bool v385;
            v385 = v376 && v380;
            bool v386;
            v386 = v385 == false;
            if (v386){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v385);
            } else {
            }
            bool v388;
            v388 = 0l <= v383;
            bool v390;
            if (v388){
                bool v389;
                v389 = v383 < 1l;
                v390 = v389;
            } else {
                v390 = false;
            }
            bool v391;
            v391 = v390 == false;
            if (v391){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v390);
            } else {
            }
            int v393;
            v393 = v383 * 32l;
            int v394;
            v394 = v393 + v375;
            assert("Tensor range check" && 0 <= v383 && v383 < 1l);
            int v395;
            v395 = 32l * v383;
            int v396;
            v396 = v395 + v375;
            float * v397;
            v397 = v370[v396];
            float * v398;
            v398 = v371[v396];
            /* void array index */;
            assert("Tensor range check" && 0 <= v379 && v379 < 1l);
            int v399;
            v399 = 4l * v379;
            float v400[4l];
            int v401[4l];
            int v402;
            v402 = 0l;
            while (while_method_1(v402)){
                assert("Tensor range check" && 0 <= v402 && v402 < 1l);
                int v404;
                v404 = 4l * v402;
                assert("Tensor range check" && 0 <= v402 && v402 < 1l);
                int v405;
                v405 = v404 + v399;
                int4* v406;
                v406 = reinterpret_cast<int4*>(v398 + v405);
                int4* v407;
                v407 = reinterpret_cast<int4*>(v400 + v404);
                assert("Pointer alignment check" && (unsigned long long)(v406) % 4l == 0 && (unsigned long long)(v407) % 4l == 0);
                *v407 = *v406;
                v402 += 1l ;
            }
            int v408;
            v408 = 0l;
            while (while_method_1(v408)){
                int v410;
                v410 = 0l;
                while (while_method_2(v410)){
                    bool v412;
                    v412 = 0l <= v410;
                    bool v414;
                    if (v412){
                        bool v413;
                        v413 = v410 < 4l;
                        v414 = v413;
                    } else {
                        v414 = false;
                    }
                    bool v415;
                    v415 = v414 == false;
                    if (v415){
                        assert("The indices should be inside the range of the dimension." && v414);
                    } else {
                    }
                    bool v417;
                    v417 = 0l <= v379;
                    bool v419;
                    if (v417){
                        bool v418;
                        v418 = v379 < 1l;
                        v419 = v418;
                    } else {
                        v419 = false;
                    }
                    bool v420;
                    v420 = v419 == false;
                    if (v420){
                        assert("The indices should be inside the range of the dimension." && v419);
                    } else {
                    }
                    int v422;
                    v422 = v379 * 4l;
                    int v423;
                    v423 = v410 + v422;
                    bool v424;
                    v424 = 0l <= v408;
                    bool v426;
                    if (v424){
                        bool v425;
                        v425 = v408 < 1l;
                        v426 = v425;
                    } else {
                        v426 = false;
                    }
                    bool v427;
                    v427 = v426 == false;
                    if (v427){
                        assert("The indices should be inside the range of the dimension." && v426);
                    } else {
                    }
                    int v429;
                    v429 = v408 * 4l;
                    int v430;
                    v430 = v423 + v429;
                    assert("Tensor range check" && 0 <= v408 && v408 < 1l);
                    assert("Tensor range check" && 0 <= v410 && v410 < 4l);
                    int v431;
                    v431 = 4l * v408;
                    int v432;
                    v432 = v431 + v410;
                    v401[v432] = v430;
                    v410 += 1l ;
                }
                v408 += 1l ;
            }
            int v433;
            v433 = 0l;
            while (while_method_1(v433)){
                int v435;
                v435 = 0l;
                while (while_method_2(v435)){
                    assert("Tensor range check" && 0 <= v433 && v433 < 1l);
                    assert("Tensor range check" && 0 <= v435 && v435 < 4l);
                    int v437;
                    v437 = 4l * v433;
                    int v438;
                    v438 = v437 + v435;
                    float v439;
                    v439 = v400[v438];
                    int v440;
                    v440 = v401[v438];
                    assert("Tensor range check" && 0 <= v440 && v440 < 4l);
                    float * v441;
                    v441 = v397+v440;
                    float v443;
                    v443 = atomicAdd(v441,v439);
                    v435 += 1l ;
                }
                v433 += 1l ;
            }
            int v444;
            v444 = 0l;
            while (while_method_1(v444)){
                assert("Tensor range check" && 0 <= v444 && v444 < 1l);
                assert("Tensor range check" && 0 <= v444 && v444 < 1l);
                v444 += 1l ;
            }
            assert("Tensor range check" && 0 <= v394 && v394 < 32l);
            /* void array set */;
            v383 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v446;
        v446 = threadIdx.x;
        assert("Tensor range check" && 0 <= v446 && v446 < 32l);
        /* void array index */;
    }
    int v447;
    v447 = threadIdx.x;
    bool v448;
    v448 = 0l <= v447;
    bool v449;
    v449 = v448 == false;
    if (v449){
        assert("The index needs to be zero or positive." && v448);
    } else {
    }
    int v451;
    v451 = v447 % 1l;
    bool v452;
    v452 = v447 < 32l;
    bool v453;
    v453 = v452 == false;
    if (v453){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v452);
    } else {
    }
    assert("Tensor range check" && 0 <= v447 && v447 < 32l);
    assert("Tensor range check" && 0 <= v451 && v451 < 1l);
    int v455;
    v455 = 4l * v451;
    int v456;
    v456 = 4l * v447;
    int v457;
    v457 = v456 + v455;
    assert("Tensor range check" && 0 <= v447 && v447 < 32l);
    assert("Tensor range check" && 0 <= v451 && v451 < 1l);
    int v458;
    v458 = 0l;
    while (while_method_4(v458)){
        assert("Tensor range check" && 0 <= v458 && v458 < 128l);
        int v460;
        v460 = 128l * v458;
        int v461;
        v461 = v460 + v457;
        float v462[4l];
        float v463[4l];
        float v464[4l];
        float v465[4l];
        float v466[4l];
        int v467[4l];
        int v468;
        v468 = 0l;
        while (while_method_1(v468)){
            assert("Tensor range check" && 0 <= v468 && v468 < 1l);
            int v470;
            v470 = 4l * v468;
            assert("Tensor range check" && 0 <= v468 && v468 < 1l);
            int v471;
            v471 = v470 + v461;
            int4* v472;
            v472 = reinterpret_cast<int4*>(v9 + v471);
            int4* v473;
            v473 = reinterpret_cast<int4*>(v462 + v470);
            assert("Pointer alignment check" && (unsigned long long)(v472) % 4l == 0 && (unsigned long long)(v473) % 4l == 0);
            *v473 = *v472;
            int4* v474;
            v474 = reinterpret_cast<int4*>(v10 + v471);
            int4* v475;
            v475 = reinterpret_cast<int4*>(v463 + v470);
            assert("Pointer alignment check" && (unsigned long long)(v474) % 4l == 0 && (unsigned long long)(v475) % 4l == 0);
            *v475 = *v474;
            int4* v476;
            v476 = reinterpret_cast<int4*>(v11 + v471);
            int4* v477;
            v477 = reinterpret_cast<int4*>(v464 + v470);
            assert("Pointer alignment check" && (unsigned long long)(v476) % 4l == 0 && (unsigned long long)(v477) % 4l == 0);
            *v477 = *v476;
            int4* v478;
            v478 = reinterpret_cast<int4*>(v12 + v471);
            int4* v479;
            v479 = reinterpret_cast<int4*>(v465 + v470);
            assert("Pointer alignment check" && (unsigned long long)(v478) % 4l == 0 && (unsigned long long)(v479) % 4l == 0);
            *v479 = *v478;
            int4* v480;
            v480 = reinterpret_cast<int4*>(v13 + v471);
            int4* v481;
            v481 = reinterpret_cast<int4*>(v466 + v470);
            assert("Pointer alignment check" && (unsigned long long)(v480) % 4l == 0 && (unsigned long long)(v481) % 4l == 0);
            *v481 = *v480;
            v468 += 1l ;
        }
        int v482;
        v482 = 0l;
        while (while_method_1(v482)){
            int v484;
            v484 = 0l;
            while (while_method_2(v484)){
                bool v486;
                v486 = 0l <= v484;
                bool v488;
                if (v486){
                    bool v487;
                    v487 = v484 < 4l;
                    v488 = v487;
                } else {
                    v488 = false;
                }
                bool v489;
                v489 = v488 == false;
                if (v489){
                    assert("The indices should be inside the range of the dimension." && v488);
                } else {
                }
                bool v491;
                v491 = 0l <= v451;
                bool v493;
                if (v491){
                    bool v492;
                    v492 = v451 < 1l;
                    v493 = v492;
                } else {
                    v493 = false;
                }
                bool v494;
                v494 = v493 == false;
                if (v494){
                    assert("The indices should be inside the range of the dimension." && v493);
                } else {
                }
                int v496;
                v496 = v451 * 4l;
                int v497;
                v497 = v484 + v496;
                bool v498;
                v498 = 0l <= v482;
                bool v500;
                if (v498){
                    bool v499;
                    v499 = v482 < 1l;
                    v500 = v499;
                } else {
                    v500 = false;
                }
                bool v501;
                v501 = v500 == false;
                if (v501){
                    assert("The indices should be inside the range of the dimension." && v500);
                } else {
                }
                int v503;
                v503 = v482 * 4l;
                int v504;
                v504 = v497 + v503;
                assert("Tensor range check" && 0 <= v482 && v482 < 1l);
                assert("Tensor range check" && 0 <= v484 && v484 < 4l);
                int v505;
                v505 = 4l * v482;
                int v506;
                v506 = v505 + v484;
                v467[v506] = v504;
                v484 += 1l ;
            }
            v482 += 1l ;
        }
        bool v507;
        v507 = v448 && v452;
        bool v508;
        v508 = v507 == false;
        if (v508){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v507);
        } else {
        }
        bool v510;
        v510 = 0l <= v458;
        bool v512;
        if (v510){
            bool v511;
            v511 = v458 < 128l;
            v512 = v511;
        } else {
            v512 = false;
        }
        bool v513;
        v513 = v512 == false;
        if (v513){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v512);
        } else {
        }
        int v515;
        v515 = v458 * 32l;
        int v516;
        v516 = v515 + v447;
        bool v517[4l];
        int v518;
        v518 = 0l;
        while (while_method_1(v518)){
            int v520;
            v520 = 0l;
            while (while_method_2(v520)){
                assert("Tensor range check" && 0 <= v518 && v518 < 1l);
                assert("Tensor range check" && 0 <= v520 && v520 < 4l);
                int v522;
                v522 = 4l * v518;
                int v523;
                v523 = v522 + v520;
                float v524;
                v524 = v464[v523];
                bool v525;
                v525 = v524 == 0.0f;
                bool v526;
                v526 = v525 != true;
                assert("Tensor range check" && 0 <= v518 && v518 < 1l);
                assert("Tensor range check" && 0 <= v520 && v520 < 4l);
                v517[v523] = v526;
                v520 += 1l ;
            }
            v518 += 1l ;
        }
        bool v527;
        v527 = false;
        int v528;
        v528 = 0l;
        while (while_method_1(v528)){
            int v530;
            v530 = 0l;
            while (while_method_2(v530)){
                assert("Tensor range check" && 0 <= v528 && v528 < 1l);
                assert("Tensor range check" && 0 <= v530 && v530 < 4l);
                int v532;
                v532 = 4l * v528;
                int v533;
                v533 = v532 + v530;
                bool v534;
                v534 = v517[v533];
                bool v535;
                v535 = v527 || v534;
                v527 = v535;
                v530 += 1l ;
            }
            v528 += 1l ;
        }
        auto v536 = cooperative_groups::coalesced_threads();
        int v537;
        v537 = threadIdx.x;
        auto v538 = cooperative_groups::labeled_partition(v536,v537);
        Closure7 v539{};
        bool v540;
        v540 = cooperative_groups::reduce(v538, v527, v539);
        if (v540){
            float v541[4l];
            int v542;
            v542 = 0l;
            while (while_method_1(v542)){
                int v544;
                v544 = 0l;
                while (while_method_2(v544)){
                    assert("Tensor range check" && 0 <= v542 && v542 < 1l);
                    assert("Tensor range check" && 0 <= v544 && v544 < 4l);
                    int v546;
                    v546 = 4l * v542;
                    int v547;
                    v547 = v546 + v544;
                    float v548;
                    v548 = v463[v547];
                    float v549;
                    v549 = v464[v547];
                    float v550;
                    v550 = v548 + v549;
                    bool v551;
                    v551 = 0.0f >= v550;
                    float v552;
                    if (v551){
                        v552 = 0.0f;
                    } else {
                        v552 = v550;
                    }
                    assert("Tensor range check" && 0 <= v542 && v542 < 1l);
                    assert("Tensor range check" && 0 <= v544 && v544 < 4l);
                    v541[v547] = v552;
                    v544 += 1l ;
                }
                v542 += 1l ;
            }
            float v553[4l];
            int v554;
            v554 = 0l;
            while (while_method_1(v554)){
                int v556;
                v556 = 0l;
                while (while_method_2(v556)){
                    assert("Tensor range check" && 0 <= v554 && v554 < 1l);
                    assert("Tensor range check" && 0 <= v556 && v556 < 4l);
                    int v558;
                    v558 = 4l * v554;
                    int v559;
                    v559 = v558 + v556;
                    float v560;
                    v560 = v541[v559];
                    bool v561;
                    v561 = 0.0f >= v560;
                    float v562;
                    if (v561){
                        v562 = 0.0f;
                    } else {
                        v562 = v560;
                    }
                    assert("Tensor range check" && 0 <= v554 && v554 < 1l);
                    assert("Tensor range check" && 0 <= v556 && v556 < 4l);
                    v553[v559] = v562;
                    v556 += 1l ;
                }
                v554 += 1l ;
            }
            float v563;
            v563 = 0.0f;
            int v564;
            v564 = 0l;
            while (while_method_1(v564)){
                int v566;
                v566 = 0l;
                while (while_method_2(v566)){
                    assert("Tensor range check" && 0 <= v564 && v564 < 1l);
                    assert("Tensor range check" && 0 <= v566 && v566 < 4l);
                    int v568;
                    v568 = 4l * v564;
                    int v569;
                    v569 = v568 + v566;
                    float v570;
                    v570 = v553[v569];
                    float v571;
                    v571 = v563 + v570;
                    v563 = v571;
                    v566 += 1l ;
                }
                v564 += 1l ;
            }
            auto v572 = cooperative_groups::coalesced_threads();
            int v573;
            v573 = threadIdx.x;
            auto v574 = cooperative_groups::labeled_partition(v572,v573);
            Closure0 v575{};
            float v576;
            v576 = cooperative_groups::reduce(v574, v563, v575);
            float v577[4l];
            int v578;
            v578 = 0l;
            while (while_method_1(v578)){
                int v580;
                v580 = 0l;
                while (while_method_2(v580)){
                    assert("Tensor range check" && 0 <= v578 && v578 < 1l);
                    assert("Tensor range check" && 0 <= v580 && v580 < 4l);
                    int v582;
                    v582 = 4l * v578;
                    int v583;
                    v583 = v582 + v580;
                    float v584;
                    v584 = v553[v583];
                    bool v585;
                    v585 = v576 == 0.0f;
                    bool v586;
                    v586 = v585 != true;
                    float v588;
                    if (v586){
                        float v587;
                        v587 = v584 / v576;
                        v588 = v587;
                    } else {
                        v588 = 0.25f;
                    }
                    assert("Tensor range check" && 0 <= v578 && v578 < 1l);
                    assert("Tensor range check" && 0 <= v580 && v580 < 4l);
                    v577[v583] = v588;
                    v580 += 1l ;
                }
                v578 += 1l ;
            }
            float v589[4l];
            int v590;
            v590 = 0l;
            while (while_method_1(v590)){
                int v592;
                v592 = 0l;
                while (while_method_2(v592)){
                    assert("Tensor range check" && 0 <= v590 && v590 < 1l);
                    assert("Tensor range check" && 0 <= v592 && v592 < 4l);
                    int v594;
                    v594 = 4l * v590;
                    int v595;
                    v595 = v594 + v592;
                    float v596;
                    v596 = v462[v595];
                    float v597;
                    v597 = v577[v595];
                    float v598;
                    v598 = v596 + v597;
                    assert("Tensor range check" && 0 <= v590 && v590 < 1l);
                    assert("Tensor range check" && 0 <= v592 && v592 < 4l);
                    v589[v595] = v598;
                    v592 += 1l ;
                }
                v590 += 1l ;
            }
            float v599[4l];
            int v600;
            v600 = 0l;
            while (while_method_1(v600)){
                int v602;
                v602 = 0l;
                while (while_method_2(v602)){
                    assert("Tensor range check" && 0 <= v600 && v600 < 1l);
                    assert("Tensor range check" && 0 <= v602 && v602 < 4l);
                    int v604;
                    v604 = 4l * v600;
                    int v605;
                    v605 = v604 + v602;
                    float v606;
                    v606 = v589[v605];
                    float v607;
                    v607 = -v606;
                    bool v608;
                    v608 = v606 >= v607;
                    float v609;
                    if (v608){
                        v609 = v606;
                    } else {
                        v609 = v607;
                    }
                    assert("Tensor range check" && 0 <= v600 && v600 < 1l);
                    assert("Tensor range check" && 0 <= v602 && v602 < 4l);
                    v599[v605] = v609;
                    v602 += 1l ;
                }
                v600 += 1l ;
            }
            float v610;
            v610 = 0.0f;
            int v611;
            v611 = 0l;
            while (while_method_1(v611)){
                int v613;
                v613 = 0l;
                while (while_method_2(v613)){
                    assert("Tensor range check" && 0 <= v611 && v611 < 1l);
                    assert("Tensor range check" && 0 <= v613 && v613 < 4l);
                    int v615;
                    v615 = 4l * v611;
                    int v616;
                    v616 = v615 + v613;
                    float v617;
                    v617 = v599[v616];
                    float v618;
                    v618 = v610 + v617;
                    v610 = v618;
                    v613 += 1l ;
                }
                v611 += 1l ;
            }
            auto v619 = cooperative_groups::coalesced_threads();
            int v620;
            v620 = threadIdx.x;
            auto v621 = cooperative_groups::labeled_partition(v619,v620);
            float v622;
            v622 = cooperative_groups::reduce(v621, v610, v575);
            bool v623;
            v623 = v622 > 100.0f;
            float v625;
            if (v623){
                float v624;
                v624 = 100.0f / v622;
                v625 = v624;
            } else {
                v625 = 1.0f;
            }
            float v626[4l];
            int v627;
            v627 = 0l;
            while (while_method_1(v627)){
                int v629;
                v629 = 0l;
                while (while_method_2(v629)){
                    assert("Tensor range check" && 0 <= v627 && v627 < 1l);
                    assert("Tensor range check" && 0 <= v629 && v629 < 4l);
                    int v631;
                    v631 = 4l * v627;
                    int v632;
                    v632 = v631 + v629;
                    float v633;
                    v633 = v599[v632];
                    float v634;
                    v634 = v625 * v633;
                    assert("Tensor range check" && 0 <= v627 && v627 < 1l);
                    assert("Tensor range check" && 0 <= v629 && v629 < 4l);
                    v626[v632] = v634;
                    v629 += 1l ;
                }
                v627 += 1l ;
            }
            float v635[4l];
            float v636[4l];
            int v637;
            v637 = 0l;
            while (while_method_1(v637)){
                int v639;
                v639 = 0l;
                while (while_method_2(v639)){
                    assert("Tensor range check" && 0 <= v637 && v637 < 1l);
                    assert("Tensor range check" && 0 <= v639 && v639 < 4l);
                    int v641;
                    v641 = 4l * v637;
                    int v642;
                    v642 = v641 + v639;
                    float v643;
                    v643 = v465[v642];
                    float v644;
                    v644 = v466[v642];
                    bool v645;
                    v645 = v644 > 100.0f;
                    float v647;
                    if (v645){
                        float v646;
                        v646 = 100.0f / v644;
                        v647 = v646;
                    } else {
                        v647 = 1.0f;
                    }
                    float v648;
                    v648 = v643 * v647;
                    float v649;
                    v649 = v644 * v647;
                    assert("Tensor range check" && 0 <= v637 && v637 < 1l);
                    assert("Tensor range check" && 0 <= v639 && v639 < 4l);
                    v635[v642] = v648;
                    v636[v642] = v649;
                    v639 += 1l ;
                }
                v637 += 1l ;
            }
            cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v650 = console_lock;
            auto v651 = cooperative_groups::coalesced_threads();
            v650.acquire();
            int v652;
            v652 = 0l;
            printf("{%s = %c","average", '[');
            int v653;
            v653 = 0l;
            while (while_method_1(v653)){
                int v655;
                v655 = v652;
                bool v656;
                v656 = v655 >= 100l;
                if (v656){
                    printf("%s"," ...");
                    break;
                } else {
                }
                bool v657;
                v657 = v653 == 0l;
                bool v658;
                v658 = v657 != true;
                if (v658){
                    printf("%s","; ");
                } else {
                }
                printf("%c",'[');
                int v659;
                v659 = 0l;
                while (while_method_2(v659)){
                    int v661;
                    v661 = v652;
                    bool v662;
                    v662 = v661 >= 100l;
                    if (v662){
                        printf("%s"," ...");
                        break;
                    } else {
                    }
                    bool v663;
                    v663 = v659 == 0l;
                    bool v664;
                    v664 = v663 != true;
                    if (v664){
                        printf("%s","; ");
                    } else {
                    }
                    int v665;
                    v665 = v652 + 1l;
                    v652 = v665;
                    int v666;
                    v666 = v653 * 4l;
                    int v667;
                    v667 = v666 + v659;
                    float v668;
                    v668 = v626[v667];
                    printf("%f",v668);
                    v659 += 1l ;
                }
                printf("%c",']');
                v653 += 1l ;
            }
            printf("%c",']');
            int v669;
            v669 = 0l;
            printf("; %s = %d; %s = %c","i", v516, "value", '[');
            int v670;
            v670 = 0l;
            while (while_method_1(v670)){
                int v672;
                v672 = v669;
                bool v673;
                v673 = v672 >= 100l;
                if (v673){
                    printf("%s"," ...");
                    break;
                } else {
                }
                bool v674;
                v674 = v670 == 0l;
                bool v675;
                v675 = v674 != true;
                if (v675){
                    printf("%s","; ");
                } else {
                }
                printf("%c",'[');
                int v676;
                v676 = 0l;
                while (while_method_2(v676)){
                    int v678;
                    v678 = v669;
                    bool v679;
                    v679 = v678 >= 100l;
                    if (v679){
                        printf("%s"," ...");
                        break;
                    } else {
                    }
                    bool v680;
                    v680 = v676 == 0l;
                    bool v681;
                    v681 = v680 != true;
                    if (v681){
                        printf("%s","; ");
                    } else {
                    }
                    int v682;
                    v682 = v669 + 1l;
                    v669 = v682;
                    int v683;
                    v683 = v670 * 4l;
                    int v684;
                    v684 = v683 + v676;
                    float v685;
                    v685 = v635[v684];
                    float v686;
                    v686 = v636[v684];
                    printf("%f, %f",v685, v686);
                    v676 += 1l ;
                }
                printf("%c",']');
                v670 += 1l ;
            }
            printf("%c",']');
            printf("}\n");
            v650.release();
            v651.sync() ;
            int v742;
            v742 = 0l;
            while (while_method_1(v742)){
                int v744;
                v744 = 0l;
                while (while_method_2(v744)){
                    assert("Tensor range check" && 0 <= v742 && v742 < 1l);
                    assert("Tensor range check" && 0 <= v744 && v744 < 4l);
                    int v746;
                    v746 = 4l * v742;
                    int v747;
                    v747 = v746 + v744;
                    float v748;
                    v748 = v541[v747];
                    float v749;
                    v749 = v626[v747];
                    float v750;
                    v750 = v635[v747];
                    float v751;
                    v751 = v636[v747];
                    assert("Tensor range check" && 0 <= v742 && v742 < 1l);
                    assert("Tensor range check" && 0 <= v744 && v744 < 4l);
                    v462[v747] = v749;
                    v463[v747] = v748;
                    v464[v747] = 0.0f;
                    v465[v747] = v750;
                    v466[v747] = v751;
                    v744 += 1l ;
                }
                v742 += 1l ;
            }
        } else {
        }
        assert("Tensor range check" && 0 <= v458 && v458 < 128l);
        int v752;
        v752 = 0l;
        while (while_method_1(v752)){
            assert("Tensor range check" && 0 <= v752 && v752 < 1l);
            int v754;
            v754 = 4l * v752;
            int v755;
            v755 = v754 + v461;
            assert("Tensor range check" && 0 <= v752 && v752 < 1l);
            int4* v756;
            v756 = reinterpret_cast<int4*>(v462 + v754);
            int4* v757;
            v757 = reinterpret_cast<int4*>(v9 + v755);
            assert("Pointer alignment check" && (unsigned long long)(v756) % 4l == 0 && (unsigned long long)(v757) % 4l == 0);
            *v757 = *v756;
            int4* v758;
            v758 = reinterpret_cast<int4*>(v463 + v754);
            int4* v759;
            v759 = reinterpret_cast<int4*>(v10 + v755);
            assert("Pointer alignment check" && (unsigned long long)(v758) % 4l == 0 && (unsigned long long)(v759) % 4l == 0);
            *v759 = *v758;
            int4* v760;
            v760 = reinterpret_cast<int4*>(v464 + v754);
            int4* v761;
            v761 = reinterpret_cast<int4*>(v11 + v755);
            assert("Pointer alignment check" && (unsigned long long)(v760) % 4l == 0 && (unsigned long long)(v761) % 4l == 0);
            *v761 = *v760;
            int4* v762;
            v762 = reinterpret_cast<int4*>(v465 + v754);
            int4* v763;
            v763 = reinterpret_cast<int4*>(v12 + v755);
            assert("Pointer alignment check" && (unsigned long long)(v762) % 4l == 0 && (unsigned long long)(v763) % 4l == 0);
            *v763 = *v762;
            int4* v764;
            v764 = reinterpret_cast<int4*>(v466 + v754);
            int4* v765;
            v765 = reinterpret_cast<int4*>(v13 + v755);
            assert("Pointer alignment check" && (unsigned long long)(v764) % 4l == 0 && (unsigned long long)(v765) % 4l == 0);
            *v765 = *v764;
            v752 += 1l ;
        }
        v458 += 1l ;
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
    v0 = cp.empty(512,dtype=cp.int32)
    v1 = cp.empty(512,dtype=cp.float32)
    v2 = cp.empty(512,dtype=cp.int32)
    v3 = cp.empty(512,dtype=cp.int32)
    v4 = cp.empty(1024,dtype=cp.float64)
    v5 = cp.empty(1024,dtype=cp.float64)
    v6 = cp.empty(2048,dtype=cp.float32)
    v7 = cp.empty(512,dtype=cp.float32)
    v8 = cp.empty(512,dtype=cp.float32)
    v9 = cp.empty(16384,dtype=cp.float32)
    v10 = cp.empty(16384,dtype=cp.float32)
    v11 = cp.empty(16384,dtype=cp.float32)
    v12 = cp.empty(16384,dtype=cp.float32)
    v13 = cp.empty(16384,dtype=cp.float32)
    v14 = 0
    v15 = raw_module.get_function(f"entry{v14}")
    del v14
    v15.max_dynamic_shared_size_bytes = 0 
    v15((1,),(32,),(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13),shared_mem=0)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v15
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
