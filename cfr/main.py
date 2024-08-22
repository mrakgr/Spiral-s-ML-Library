kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cooperative_groups.h>
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
__device__ void push_0(float * v0, float * v1, float * v2, float * v3, float * v4, int * v5, float * v6, int * v7, int * v8, double * v9, double * v10, float * v11, float * v12, float * v13, float * v14, int v15, int & v16, double * v17, double * v18, int v19, int v20);
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
__device__ void push_0(float * v0, float * v1, float * v2, float * v3, float * v4, int * v5, float * v6, int * v7, int * v8, double * v9, double * v10, float * v11, float * v12, float * v13, float * v14, int v15, int & v16, double * v17, double * v18, int v19, int v20){
    float v21; float v22; int v23;
    Tuple0 tmp12 = get_action_1(v0, v1, v2, v3, v4, v20);
    v21 = tmp12.v0; v22 = tmp12.v1; v23 = tmp12.v2;
    int v24 = v16;
    int v25;
    v25 = v24 + 1l;
    v16 = v25;
    assert("Tensor range check" && 0 <= v24 && v24 < 16l);
    assert("Tensor range check" && 0 <= v15 && v15 < 32l);
    int v26;
    v26 = 32l * v24;
    int v27;
    v27 = v26 + v15;
    v5[v27] = v23;
    v6[v27] = v22;
    v7[v27] = v19;
    v8[v27] = v20;
    double v28;
    v28 = (double)v22;
    double v29;
    v29 = log(v28);
    double v30;
    v30 = (double)v21;
    double v31;
    v31 = log(v30);
    assert("Tensor range check" && 0 <= v19 && v19 < 2l);
    double v32;
    v32 = v17[v19];
    double v33;
    v33 = v18[v19];
    double v34;
    v34 = v31 + v32;
    double v35;
    v35 = v29 + v33;
    assert("Tensor range check" && 0 <= v19 && v19 < 2l);
    v17[v19] = v34;
    v18[v19] = v35;
    assert("Tensor range check" && 0 <= v24 && v24 < 16l);
    int v36;
    v36 = 64l * v24;
    assert("Tensor range check" && 0 <= v15 && v15 < 32l);
    int v37;
    v37 = 2l * v15;
    int v38;
    v38 = v37 + v36;
    int v39;
    v39 = 0l;
    while (while_method_0(v39)){
        assert("Tensor range check" && 0 <= v39 && v39 < 2l);
        double v41;
        v41 = v17[v39];
        double v42;
        v42 = v18[v39];
        assert("Tensor range check" && 0 <= v39 && v39 < 2l);
        int v43;
        v43 = v39 + v38;
        v9[v43] = v41;
        v10[v43] = v42;
        v39 += 1l ;
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
extern "C" __global__ void entry0(int * v0, float * v1, int * v2, int * v3, double * v4, double * v5, float * v6, float * v7, float * v8, float * v9, float * v10, float * v11, float * v12, float * v13, float * v14) {
    int v15;
    v15 = threadIdx.x;
    int v16 = 0l;
    double v17[2l];
    double v18[2l];
    int v19;
    v19 = 0l;
    while (while_method_0(v19)){
        assert("Tensor range check" && 0 <= v19 && v19 < 2l);
        v17[v19] = 0.0;
        v18[v19] = 0.0;
        v19 += 1l ;
    }
    int v21;
    v21 = 235l;
    int v22;
    v22 = 0l;
    push_0(v10, v11, v12, v13, v14, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v15, v16, v17, v18, v22, v21);
    int v23;
    v23 = 212l;
    int v24;
    v24 = 1l;
    push_0(v10, v11, v12, v13, v14, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v15, v16, v17, v18, v24, v23);
    int v25;
    v25 = 790l;
    int v26;
    v26 = 0l;
    push_0(v10, v11, v12, v13, v14, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v15, v16, v17, v18, v26, v25);
    int v27;
    v27 = 343l;
    int v28;
    v28 = 1l;
    push_0(v10, v11, v12, v13, v14, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v15, v16, v17, v18, v28, v27);
    int v29;
    v29 = 457l;
    int v30;
    v30 = 0l;
    push_0(v10, v11, v12, v13, v14, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v15, v16, v17, v18, v30, v29);
    int v31;
    v31 = 3447l;
    int v32;
    v32 = 1l;
    push_0(v10, v11, v12, v13, v14, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v15, v16, v17, v18, v32, v31);
    int v33 = v16;
    float v34[2l];
    v34[0l] = 13.0f;
    v34[1l] = -13.0f;
    int v35;
    v35 = v33;
    while (while_method_3(v35)){
        v35 -= 1l ;
        assert("Tensor range check" && 0 <= v35 && v35 < 16l);
        assert("Tensor range check" && 0 <= v15 && v15 < 32l);
        int v37;
        v37 = 32l * v35;
        int v38;
        v38 = v37 + v15;
        int v39;
        v39 = v0[v38];
        float v40;
        v40 = v1[v38];
        int v41;
        v41 = v2[v38];
        int v42;
        v42 = v3[v38];
        assert("Tensor range check" && 0 <= v41 && v41 < 2l);
        float v43;
        v43 = v34[v41];
        assert("Tensor range check" && 0 <= v42 && v42 < 4096l);
        int v44;
        v44 = 4l * v42;
        assert("Tensor range check" && 0 <= v35 && v35 < 16l);
        int v45;
        v45 = 128l * v35;
        assert("Tensor range check" && 0 <= v15 && v15 < 32l);
        int v46;
        v46 = 4l * v15;
        int v47;
        v47 = v46 + v45;
        assert("Tensor range check" && 0 <= v35 && v35 < 16l);
        int v48;
        v48 = 64l * v35;
        double * v49;
        v49 = v4+v48;
        double * v51;
        v51 = v5+v48;
        assert("Tensor range check" && 0 <= v15 && v15 < 32l);
        int v53;
        v53 = 2l * v15;
        double v54[2l];
        int v55;
        v55 = 0l;
        while (while_method_0(v55)){
            assert("Tensor range check" && 0 <= v55 && v55 < 2l);
            int v57;
            v57 = v55 + v53;
            double v58;
            v58 = v49[v57];
            bool v59;
            v59 = v41 == v55;
            double v60;
            if (v59){
                v60 = 0.0;
            } else {
                v60 = v58;
            }
            assert("Tensor range check" && 0 <= v55 && v55 < 2l);
            v54[v55] = v60;
            v55 += 1l ;
        }
        double v61;
        v61 = 0.0;
        int v62;
        v62 = 0l;
        while (while_method_0(v62)){
            assert("Tensor range check" && 0 <= v62 && v62 < 2l);
            double v64;
            v64 = v54[v62];
            double v65;
            v65 = v61 + v64;
            v61 = v65;
            v62 += 1l ;
        }
        double v66;
        v66 = 0.0;
        int v67;
        v67 = 0l;
        while (while_method_0(v67)){
            assert("Tensor range check" && 0 <= v67 && v67 < 2l);
            int v69;
            v69 = v67 + v53;
            double v70;
            v70 = v51[v69];
            double v71;
            v71 = v66 + v70;
            v66 = v71;
            v67 += 1l ;
        }
        double v72;
        v72 = v61 - v66;
        double v73;
        v73 = exp(v72);
        float v74;
        v74 = (float)v73;
        float v75;
        v75 = v43 * v74;
        assert("Tensor range check" && 0 <= v35 && v35 < 16l);
        assert("Tensor range check" && 0 <= v15 && v15 < 32l);
        v8[v38] = v75;
        v9[v38] = v74;
        float * v76;
        v76 = v11+v44;
        float * v78;
        v78 = v13+v44;
        float * v80;
        v80 = v14+v44;
        float * v82;
        v82 = v7+v47;
        __shared__ float v84[32l];
        __shared__ int v85[32l];
        __shared__ float v86[32l];
        __shared__ int v87[32l];
        __shared__ double * v88[32l];
        __shared__ double * v89[32l];
        __shared__ float * v90[32l];
        __shared__ float * v91[32l];
        __shared__ float * v92[32l];
        __shared__ float * v93[32l];
        /* void shared array create v94 */;
        __shared__ float v95[32l];
        int v96;
        v96 = threadIdx.x;
        assert("Tensor range check" && 0 <= v96 && v96 < 32l);
        v84[v96] = v40;
        v85[v96] = v39;
        v86[v96] = v43;
        v87[v96] = v41;
        v88[v96] = v49;
        v89[v96] = v51;
        v90[v96] = v76;
        v91[v96] = v78;
        v92[v96] = v80;
        v93[v96] = v82;
        /* void array set */;
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v97;
        v97 = threadIdx.x;
        bool v98;
        v98 = 0l <= v97;
        bool v99;
        v99 = v98 == false;
        if (v99){
            assert("The index needs to be zero or positive." && v98);
        } else {
        }
        int v101;
        v101 = v97 % 1l;
        bool v102;
        v102 = v97 < 32l;
        bool v103;
        v103 = v102 == false;
        if (v103){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v102);
        } else {
        }
        assert("Tensor range check" && 0 <= v97 && v97 < 32l);
        int v105;
        v105 = 0l;
        while (while_method_1(v105)){
            bool v107;
            v107 = v98 && v102;
            bool v108;
            v108 = v107 == false;
            if (v108){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v107);
            } else {
            }
            bool v110;
            v110 = 0l <= v105;
            bool v112;
            if (v110){
                bool v111;
                v111 = v105 < 1l;
                v112 = v111;
            } else {
                v112 = false;
            }
            bool v113;
            v113 = v112 == false;
            if (v113){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v112);
            } else {
            }
            int v115;
            v115 = v105 * 32l;
            int v116;
            v116 = v115 + v97;
            assert("Tensor range check" && 0 <= v105 && v105 < 1l);
            int v117;
            v117 = 32l * v105;
            int v118;
            v118 = v117 + v97;
            float v119;
            v119 = v84[v118];
            int v120;
            v120 = v85[v118];
            float v121;
            v121 = v86[v118];
            int v122;
            v122 = v87[v118];
            double * v123;
            v123 = v88[v118];
            double * v124;
            v124 = v89[v118];
            float * v125;
            v125 = v90[v118];
            float * v126;
            v126 = v91[v118];
            float * v127;
            v127 = v92[v118];
            float * v128;
            v128 = v93[v118];
            /* void array index */;
            assert("Tensor range check" && 0 <= v101 && v101 < 1l);
            int v129;
            v129 = 4l * v101;
            float v130[4l];
            float v131[4l];
            float v132[4l];
            int v133[4l];
            int v134;
            v134 = 0l;
            while (while_method_1(v134)){
                assert("Tensor range check" && 0 <= v134 && v134 < 1l);
                int v136;
                v136 = 4l * v134;
                assert("Tensor range check" && 0 <= v134 && v134 < 1l);
                int v137;
                v137 = v136 + v129;
                int4* v138;
                v138 = reinterpret_cast<int4*>(v125 + v137);
                int4* v139;
                v139 = reinterpret_cast<int4*>(v130 + v136);
                assert("Pointer alignment check" && (unsigned long long)(v138) % 4l == 0 && (unsigned long long)(v139) % 4l == 0);
                *v139 = *v138;
                int4* v140;
                v140 = reinterpret_cast<int4*>(v126 + v137);
                int4* v141;
                v141 = reinterpret_cast<int4*>(v131 + v136);
                assert("Pointer alignment check" && (unsigned long long)(v140) % 4l == 0 && (unsigned long long)(v141) % 4l == 0);
                *v141 = *v140;
                int4* v142;
                v142 = reinterpret_cast<int4*>(v127 + v137);
                int4* v143;
                v143 = reinterpret_cast<int4*>(v132 + v136);
                assert("Pointer alignment check" && (unsigned long long)(v142) % 4l == 0 && (unsigned long long)(v143) % 4l == 0);
                *v143 = *v142;
                v134 += 1l ;
            }
            int v144;
            v144 = 0l;
            while (while_method_1(v144)){
                int v146;
                v146 = 0l;
                while (while_method_2(v146)){
                    bool v148;
                    v148 = 0l <= v146;
                    bool v150;
                    if (v148){
                        bool v149;
                        v149 = v146 < 4l;
                        v150 = v149;
                    } else {
                        v150 = false;
                    }
                    bool v151;
                    v151 = v150 == false;
                    if (v151){
                        assert("The indices should be inside the range of the dimension." && v150);
                    } else {
                    }
                    bool v153;
                    v153 = 0l <= v101;
                    bool v155;
                    if (v153){
                        bool v154;
                        v154 = v101 < 1l;
                        v155 = v154;
                    } else {
                        v155 = false;
                    }
                    bool v156;
                    v156 = v155 == false;
                    if (v156){
                        assert("The indices should be inside the range of the dimension." && v155);
                    } else {
                    }
                    int v158;
                    v158 = v101 * 4l;
                    int v159;
                    v159 = v146 + v158;
                    bool v160;
                    v160 = 0l <= v144;
                    bool v162;
                    if (v160){
                        bool v161;
                        v161 = v144 < 1l;
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
                    int v165;
                    v165 = v144 * 4l;
                    int v166;
                    v166 = v159 + v165;
                    assert("Tensor range check" && 0 <= v144 && v144 < 1l);
                    assert("Tensor range check" && 0 <= v146 && v146 < 4l);
                    int v167;
                    v167 = 4l * v144;
                    int v168;
                    v168 = v167 + v146;
                    v133[v168] = v166;
                    v146 += 1l ;
                }
                v144 += 1l ;
            }
            float v169[4l];
            int v170;
            v170 = 0l;
            while (while_method_1(v170)){
                int v172;
                v172 = 0l;
                while (while_method_2(v172)){
                    assert("Tensor range check" && 0 <= v170 && v170 < 1l);
                    assert("Tensor range check" && 0 <= v172 && v172 < 4l);
                    int v174;
                    v174 = 4l * v170;
                    int v175;
                    v175 = v174 + v172;
                    float v176;
                    v176 = v131[v175];
                    float v177;
                    v177 = v132[v175];
                    bool v178;
                    v178 = v177 == 0.0f;
                    bool v179;
                    v179 = v178 != true;
                    float v181;
                    if (v179){
                        float v180;
                        v180 = v176 / v177;
                        v181 = v180;
                    } else {
                        v181 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v170 && v170 < 1l);
                    assert("Tensor range check" && 0 <= v172 && v172 < 4l);
                    v169[v175] = v181;
                    v172 += 1l ;
                }
                v170 += 1l ;
            }
            bool v182[4l];
            int v183;
            v183 = 0l;
            while (while_method_1(v183)){
                int v185;
                v185 = 0l;
                while (while_method_2(v185)){
                    assert("Tensor range check" && 0 <= v183 && v183 < 1l);
                    assert("Tensor range check" && 0 <= v185 && v185 < 4l);
                    int v187;
                    v187 = 4l * v183;
                    int v188;
                    v188 = v187 + v185;
                    float v189;
                    v189 = v130[v188];
                    int v190;
                    v190 = v133[v188];
                    bool v191;
                    v191 = v190 < 3l;
                    assert("Tensor range check" && 0 <= v183 && v183 < 1l);
                    assert("Tensor range check" && 0 <= v185 && v185 < 4l);
                    v182[v188] = v191;
                    v185 += 1l ;
                }
                v183 += 1l ;
            }
            float v192[4l];
            int v193;
            v193 = 0l;
            while (while_method_1(v193)){
                int v195;
                v195 = 0l;
                while (while_method_2(v195)){
                    assert("Tensor range check" && 0 <= v193 && v193 < 1l);
                    assert("Tensor range check" && 0 <= v195 && v195 < 4l);
                    int v197;
                    v197 = 4l * v193;
                    int v198;
                    v198 = v197 + v195;
                    float v199;
                    v199 = v130[v198];
                    bool v200;
                    v200 = v182[v198];
                    float v203;
                    if (v200){
                        bool v201;
                        v201 = 0.0f >= v199;
                        if (v201){
                            v203 = 0.0f;
                        } else {
                            v203 = v199;
                        }
                    } else {
                        v203 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v193 && v193 < 1l);
                    assert("Tensor range check" && 0 <= v195 && v195 < 4l);
                    v192[v198] = v203;
                    v195 += 1l ;
                }
                v193 += 1l ;
            }
            float v204;
            v204 = 0.0f;
            int v205;
            v205 = 0l;
            while (while_method_1(v205)){
                int v207;
                v207 = 0l;
                while (while_method_2(v207)){
                    assert("Tensor range check" && 0 <= v205 && v205 < 1l);
                    assert("Tensor range check" && 0 <= v207 && v207 < 4l);
                    int v209;
                    v209 = 4l * v205;
                    int v210;
                    v210 = v209 + v207;
                    float v211;
                    v211 = v192[v210];
                    float v212;
                    v212 = v204 + v211;
                    v204 = v212;
                    v207 += 1l ;
                }
                v205 += 1l ;
            }
            auto v213 = cooperative_groups::coalesced_threads();
            int v214;
            v214 = threadIdx.x;
            auto v215 = cooperative_groups::labeled_partition(v213,v214);
            Closure0 v216{};
            float v217;
            v217 = cooperative_groups::reduce(v215, v204, v216);
            int v218[4l];
            int v219;
            v219 = 0l;
            while (while_method_1(v219)){
                int v221;
                v221 = 0l;
                while (while_method_2(v221)){
                    assert("Tensor range check" && 0 <= v219 && v219 < 1l);
                    assert("Tensor range check" && 0 <= v221 && v221 < 4l);
                    int v223;
                    v223 = 4l * v219;
                    int v224;
                    v224 = v223 + v221;
                    bool v225;
                    v225 = v182[v224];
                    int v226;
                    if (v225){
                        v226 = 1l;
                    } else {
                        v226 = 0l;
                    }
                    assert("Tensor range check" && 0 <= v219 && v219 < 1l);
                    assert("Tensor range check" && 0 <= v221 && v221 < 4l);
                    v218[v224] = v226;
                    v221 += 1l ;
                }
                v219 += 1l ;
            }
            int v227;
            v227 = 0l;
            int v228;
            v228 = 0l;
            while (while_method_1(v228)){
                int v230;
                v230 = 0l;
                while (while_method_2(v230)){
                    assert("Tensor range check" && 0 <= v228 && v228 < 1l);
                    assert("Tensor range check" && 0 <= v230 && v230 < 4l);
                    int v232;
                    v232 = 4l * v228;
                    int v233;
                    v233 = v232 + v230;
                    int v234;
                    v234 = v218[v233];
                    int v235;
                    v235 = v227 + v234;
                    v227 = v235;
                    v230 += 1l ;
                }
                v228 += 1l ;
            }
            auto v236 = cooperative_groups::coalesced_threads();
            int v237;
            v237 = threadIdx.x;
            auto v238 = cooperative_groups::labeled_partition(v236,v237);
            Closure1 v239{};
            int v240;
            v240 = cooperative_groups::reduce(v238, v227, v239);
            float v241;
            v241 = (float)v240;
            float v242;
            v242 = 1.0f / v241;
            float v243[4l];
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
                    float v250;
                    v250 = v192[v249];
                    bool v251;
                    v251 = v182[v249];
                    bool v252;
                    v252 = v251 == false;
                    float v257;
                    if (v252){
                        v257 = 0.0f;
                    } else {
                        bool v253;
                        v253 = v217 == 0.0f;
                        bool v254;
                        v254 = v253 != true;
                        if (v254){
                            float v255;
                            v255 = v250 / v217;
                            v257 = v255;
                        } else {
                            v257 = v242;
                        }
                    }
                    assert("Tensor range check" && 0 <= v244 && v244 < 1l);
                    assert("Tensor range check" && 0 <= v246 && v246 < 4l);
                    v243[v249] = v257;
                    v246 += 1l ;
                }
                v244 += 1l ;
            }
            float v258[4l];
            int v259;
            v259 = 0l;
            while (while_method_1(v259)){
                int v261;
                v261 = 0l;
                while (while_method_2(v261)){
                    assert("Tensor range check" && 0 <= v259 && v259 < 1l);
                    assert("Tensor range check" && 0 <= v261 && v261 < 4l);
                    int v263;
                    v263 = 4l * v259;
                    int v264;
                    v264 = v263 + v261;
                    float v265;
                    v265 = v169[v264];
                    int v266;
                    v266 = v133[v264];
                    bool v267;
                    v267 = v120 == v266;
                    float v270;
                    if (v267){
                        float v268;
                        v268 = v121 - v265;
                        float v269;
                        v269 = v268 / v119;
                        v270 = v269;
                    } else {
                        v270 = 0.0f;
                    }
                    float v271;
                    v271 = v270 + v265;
                    assert("Tensor range check" && 0 <= v259 && v259 < 1l);
                    assert("Tensor range check" && 0 <= v261 && v261 < 4l);
                    v258[v264] = v271;
                    v261 += 1l ;
                }
                v259 += 1l ;
            }
            float v272[4l];
            int v273;
            v273 = 0l;
            while (while_method_1(v273)){
                int v275;
                v275 = 0l;
                while (while_method_2(v275)){
                    assert("Tensor range check" && 0 <= v273 && v273 < 1l);
                    assert("Tensor range check" && 0 <= v275 && v275 < 4l);
                    int v277;
                    v277 = 4l * v273;
                    int v278;
                    v278 = v277 + v275;
                    float v279;
                    v279 = v243[v278];
                    float v280;
                    v280 = v258[v278];
                    float v281;
                    v281 = v279 * v280;
                    assert("Tensor range check" && 0 <= v273 && v273 < 1l);
                    assert("Tensor range check" && 0 <= v275 && v275 < 4l);
                    v272[v278] = v281;
                    v275 += 1l ;
                }
                v273 += 1l ;
            }
            float v282;
            v282 = 0.0f;
            int v283;
            v283 = 0l;
            while (while_method_1(v283)){
                int v285;
                v285 = 0l;
                while (while_method_2(v285)){
                    assert("Tensor range check" && 0 <= v283 && v283 < 1l);
                    assert("Tensor range check" && 0 <= v285 && v285 < 4l);
                    int v287;
                    v287 = 4l * v283;
                    int v288;
                    v288 = v287 + v285;
                    float v289;
                    v289 = v272[v288];
                    float v290;
                    v290 = v282 + v289;
                    v282 = v290;
                    v285 += 1l ;
                }
                v283 += 1l ;
            }
            auto v291 = cooperative_groups::coalesced_threads();
            int v292;
            v292 = threadIdx.x;
            auto v293 = cooperative_groups::labeled_partition(v291,v292);
            float v294;
            v294 = cooperative_groups::reduce(v293, v282, v216);
            assert("Tensor range check" && 0 <= v116 && v116 < 32l);
            int v295;
            v295 = 2l * v116;
            double v296[2l];
            int v297;
            v297 = 0l;
            while (while_method_0(v297)){
                assert("Tensor range check" && 0 <= v297 && v297 < 2l);
                int v299;
                v299 = v297 + v295;
                double v300;
                v300 = v123[v299];
                bool v301;
                v301 = v122 == v297;
                double v302;
                if (v301){
                    v302 = 0.0;
                } else {
                    v302 = v300;
                }
                assert("Tensor range check" && 0 <= v297 && v297 < 2l);
                v296[v297] = v302;
                v297 += 1l ;
            }
            double v303;
            v303 = 0.0;
            int v304;
            v304 = 0l;
            while (while_method_0(v304)){
                assert("Tensor range check" && 0 <= v304 && v304 < 2l);
                double v306;
                v306 = v296[v304];
                double v307;
                v307 = v303 + v306;
                v303 = v307;
                v304 += 1l ;
            }
            double v308;
            v308 = 0.0;
            int v309;
            v309 = 0l;
            while (while_method_0(v309)){
                assert("Tensor range check" && 0 <= v309 && v309 < 2l);
                int v311;
                v311 = v309 + v295;
                double v312;
                v312 = v124[v311];
                double v313;
                v313 = v308 + v312;
                v308 = v313;
                v309 += 1l ;
            }
            double v314;
            v314 = v303 - v308;
            double v315;
            v315 = exp(v314);
            float v316;
            v316 = (float)v315;
            float v317[4l];
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
                    v324 = v258[v323];
                    float v325;
                    v325 = v324 - v294;
                    float v326;
                    v326 = v316 * v325;
                    assert("Tensor range check" && 0 <= v318 && v318 < 1l);
                    assert("Tensor range check" && 0 <= v320 && v320 < 4l);
                    v317[v323] = v326;
                    v320 += 1l ;
                }
                v318 += 1l ;
            }
            int v327;
            v327 = 0l;
            while (while_method_1(v327)){
                assert("Tensor range check" && 0 <= v327 && v327 < 1l);
                int v329;
                v329 = 4l * v327;
                int v330;
                v330 = v329 + v129;
                assert("Tensor range check" && 0 <= v327 && v327 < 1l);
                int4* v331;
                v331 = reinterpret_cast<int4*>(v317 + v329);
                int4* v332;
                v332 = reinterpret_cast<int4*>(v128 + v330);
                assert("Pointer alignment check" && (unsigned long long)(v331) % 4l == 0 && (unsigned long long)(v332) % 4l == 0);
                *v332 = *v331;
                v327 += 1l ;
            }
            assert("Tensor range check" && 0 <= v116 && v116 < 32l);
            v95[v116] = v294;
            v105 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v333;
        v333 = threadIdx.x;
        assert("Tensor range check" && 0 <= v333 && v333 < 32l);
        float v334;
        v334 = v95[v333];
        assert("Tensor range check" && 0 <= v41 && v41 < 2l);
        v34[v41] = v334;
    }
    int v335 = v16;
    int v336;
    v336 = v335;
    while (while_method_3(v336)){
        v336 -= 1l ;
        assert("Tensor range check" && 0 <= v336 && v336 < 16l);
        assert("Tensor range check" && 0 <= v15 && v15 < 32l);
        int v338;
        v338 = 32l * v336;
        int v339;
        v339 = v338 + v15;
        int v340;
        v340 = v0[v339];
        float v341;
        v341 = v1[v339];
        int v342;
        v342 = v2[v339];
        int v343;
        v343 = v3[v339];
        assert("Tensor range check" && 0 <= v336 && v336 < 16l);
        assert("Tensor range check" && 0 <= v15 && v15 < 32l);
        float v344;
        v344 = v8[v339];
        float v345;
        v345 = v9[v339];
        assert("Tensor range check" && 0 <= v343 && v343 < 4096l);
        int v346;
        v346 = 4l * v343;
        float * v347;
        v347 = v10+v346;
        float * v349;
        v349 = v11+v346;
        float * v351;
        v351 = v12+v346;
        float * v353;
        v353 = v13+v346;
        float * v355;
        v355 = v14+v346;
        assert("Tensor range check" && 0 <= v336 && v336 < 16l);
        int v357;
        v357 = 128l * v336;
        assert("Tensor range check" && 0 <= v15 && v15 < 32l);
        int v358;
        v358 = 4l * v15;
        int v359;
        v359 = v358 + v357;
        assert("Tensor range check" && 0 <= v340 && v340 < 4l);
        float * v360;
        v360 = v353+v340;
        float * v362;
        v362 = v355+v340;
        float v364;
        v364 = atomicAdd(v360,v344);
        float v365;
        v365 = atomicAdd(v362,v345);
        float * v366;
        v366 = v7+v359;
        __shared__ float * v368[32l];
        __shared__ float * v369[32l];
        /* void shared array create v370 */;
        /* void shared array create v371 */;
        int v372;
        v372 = threadIdx.x;
        assert("Tensor range check" && 0 <= v372 && v372 < 32l);
        v368[v372] = v351;
        v369[v372] = v366;
        /* void array set */;
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v373;
        v373 = threadIdx.x;
        bool v374;
        v374 = 0l <= v373;
        bool v375;
        v375 = v374 == false;
        if (v375){
            assert("The index needs to be zero or positive." && v374);
        } else {
        }
        int v377;
        v377 = v373 % 1l;
        bool v378;
        v378 = v373 < 32l;
        bool v379;
        v379 = v378 == false;
        if (v379){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v378);
        } else {
        }
        assert("Tensor range check" && 0 <= v373 && v373 < 32l);
        int v381;
        v381 = 0l;
        while (while_method_1(v381)){
            bool v383;
            v383 = v374 && v378;
            bool v384;
            v384 = v383 == false;
            if (v384){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v383);
            } else {
            }
            bool v386;
            v386 = 0l <= v381;
            bool v388;
            if (v386){
                bool v387;
                v387 = v381 < 1l;
                v388 = v387;
            } else {
                v388 = false;
            }
            bool v389;
            v389 = v388 == false;
            if (v389){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v388);
            } else {
            }
            int v391;
            v391 = v381 * 32l;
            int v392;
            v392 = v391 + v373;
            assert("Tensor range check" && 0 <= v381 && v381 < 1l);
            int v393;
            v393 = 32l * v381;
            int v394;
            v394 = v393 + v373;
            float * v395;
            v395 = v368[v394];
            float * v396;
            v396 = v369[v394];
            /* void array index */;
            assert("Tensor range check" && 0 <= v377 && v377 < 1l);
            int v397;
            v397 = 4l * v377;
            float v398[4l];
            int v399[4l];
            int v400;
            v400 = 0l;
            while (while_method_1(v400)){
                assert("Tensor range check" && 0 <= v400 && v400 < 1l);
                int v402;
                v402 = 4l * v400;
                assert("Tensor range check" && 0 <= v400 && v400 < 1l);
                int v403;
                v403 = v402 + v397;
                int4* v404;
                v404 = reinterpret_cast<int4*>(v396 + v403);
                int4* v405;
                v405 = reinterpret_cast<int4*>(v398 + v402);
                assert("Pointer alignment check" && (unsigned long long)(v404) % 4l == 0 && (unsigned long long)(v405) % 4l == 0);
                *v405 = *v404;
                v400 += 1l ;
            }
            int v406;
            v406 = 0l;
            while (while_method_1(v406)){
                int v408;
                v408 = 0l;
                while (while_method_2(v408)){
                    bool v410;
                    v410 = 0l <= v408;
                    bool v412;
                    if (v410){
                        bool v411;
                        v411 = v408 < 4l;
                        v412 = v411;
                    } else {
                        v412 = false;
                    }
                    bool v413;
                    v413 = v412 == false;
                    if (v413){
                        assert("The indices should be inside the range of the dimension." && v412);
                    } else {
                    }
                    bool v415;
                    v415 = 0l <= v377;
                    bool v417;
                    if (v415){
                        bool v416;
                        v416 = v377 < 1l;
                        v417 = v416;
                    } else {
                        v417 = false;
                    }
                    bool v418;
                    v418 = v417 == false;
                    if (v418){
                        assert("The indices should be inside the range of the dimension." && v417);
                    } else {
                    }
                    int v420;
                    v420 = v377 * 4l;
                    int v421;
                    v421 = v408 + v420;
                    bool v422;
                    v422 = 0l <= v406;
                    bool v424;
                    if (v422){
                        bool v423;
                        v423 = v406 < 1l;
                        v424 = v423;
                    } else {
                        v424 = false;
                    }
                    bool v425;
                    v425 = v424 == false;
                    if (v425){
                        assert("The indices should be inside the range of the dimension." && v424);
                    } else {
                    }
                    int v427;
                    v427 = v406 * 4l;
                    int v428;
                    v428 = v421 + v427;
                    assert("Tensor range check" && 0 <= v406 && v406 < 1l);
                    assert("Tensor range check" && 0 <= v408 && v408 < 4l);
                    int v429;
                    v429 = 4l * v406;
                    int v430;
                    v430 = v429 + v408;
                    v399[v430] = v428;
                    v408 += 1l ;
                }
                v406 += 1l ;
            }
            int v431;
            v431 = 0l;
            while (while_method_1(v431)){
                int v433;
                v433 = 0l;
                while (while_method_2(v433)){
                    assert("Tensor range check" && 0 <= v431 && v431 < 1l);
                    assert("Tensor range check" && 0 <= v433 && v433 < 4l);
                    int v435;
                    v435 = 4l * v431;
                    int v436;
                    v436 = v435 + v433;
                    float v437;
                    v437 = v398[v436];
                    int v438;
                    v438 = v399[v436];
                    assert("Tensor range check" && 0 <= v438 && v438 < 4l);
                    float * v439;
                    v439 = v395+v438;
                    float v441;
                    v441 = atomicAdd(v439,v437);
                    v433 += 1l ;
                }
                v431 += 1l ;
            }
            int v442;
            v442 = 0l;
            while (while_method_1(v442)){
                assert("Tensor range check" && 0 <= v442 && v442 < 1l);
                assert("Tensor range check" && 0 <= v442 && v442 < 1l);
                v442 += 1l ;
            }
            assert("Tensor range check" && 0 <= v392 && v392 < 32l);
            /* void array set */;
            v381 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v444;
        v444 = threadIdx.x;
        assert("Tensor range check" && 0 <= v444 && v444 < 32l);
        /* void array index */;
    }
    int v445;
    v445 = threadIdx.x;
    bool v446;
    v446 = 0l <= v445;
    bool v447;
    v447 = v446 == false;
    if (v447){
        assert("The index needs to be zero or positive." && v446);
    } else {
    }
    int v449;
    v449 = v445 % 1l;
    bool v450;
    v450 = v445 < 32l;
    bool v451;
    v451 = v450 == false;
    if (v451){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v450);
    } else {
    }
    assert("Tensor range check" && 0 <= v445 && v445 < 32l);
    assert("Tensor range check" && 0 <= v449 && v449 < 1l);
    int v453;
    v453 = 4l * v449;
    int v454;
    v454 = 4l * v445;
    int v455;
    v455 = v454 + v453;
    assert("Tensor range check" && 0 <= v445 && v445 < 32l);
    assert("Tensor range check" && 0 <= v449 && v449 < 1l);
    int v456;
    v456 = 0l;
    while (while_method_4(v456)){
        assert("Tensor range check" && 0 <= v456 && v456 < 128l);
        int v458;
        v458 = 128l * v456;
        int v459;
        v459 = v458 + v455;
        float v460[4l];
        float v461[4l];
        float v462[4l];
        float v463[4l];
        float v464[4l];
        int v465[4l];
        int v466;
        v466 = 0l;
        while (while_method_1(v466)){
            assert("Tensor range check" && 0 <= v466 && v466 < 1l);
            int v468;
            v468 = 4l * v466;
            assert("Tensor range check" && 0 <= v466 && v466 < 1l);
            int v469;
            v469 = v468 + v459;
            int4* v470;
            v470 = reinterpret_cast<int4*>(v10 + v469);
            int4* v471;
            v471 = reinterpret_cast<int4*>(v460 + v468);
            assert("Pointer alignment check" && (unsigned long long)(v470) % 4l == 0 && (unsigned long long)(v471) % 4l == 0);
            *v471 = *v470;
            int4* v472;
            v472 = reinterpret_cast<int4*>(v11 + v469);
            int4* v473;
            v473 = reinterpret_cast<int4*>(v461 + v468);
            assert("Pointer alignment check" && (unsigned long long)(v472) % 4l == 0 && (unsigned long long)(v473) % 4l == 0);
            *v473 = *v472;
            int4* v474;
            v474 = reinterpret_cast<int4*>(v12 + v469);
            int4* v475;
            v475 = reinterpret_cast<int4*>(v462 + v468);
            assert("Pointer alignment check" && (unsigned long long)(v474) % 4l == 0 && (unsigned long long)(v475) % 4l == 0);
            *v475 = *v474;
            int4* v476;
            v476 = reinterpret_cast<int4*>(v13 + v469);
            int4* v477;
            v477 = reinterpret_cast<int4*>(v463 + v468);
            assert("Pointer alignment check" && (unsigned long long)(v476) % 4l == 0 && (unsigned long long)(v477) % 4l == 0);
            *v477 = *v476;
            int4* v478;
            v478 = reinterpret_cast<int4*>(v14 + v469);
            int4* v479;
            v479 = reinterpret_cast<int4*>(v464 + v468);
            assert("Pointer alignment check" && (unsigned long long)(v478) % 4l == 0 && (unsigned long long)(v479) % 4l == 0);
            *v479 = *v478;
            v466 += 1l ;
        }
        int v480;
        v480 = 0l;
        while (while_method_1(v480)){
            int v482;
            v482 = 0l;
            while (while_method_2(v482)){
                bool v484;
                v484 = 0l <= v482;
                bool v486;
                if (v484){
                    bool v485;
                    v485 = v482 < 4l;
                    v486 = v485;
                } else {
                    v486 = false;
                }
                bool v487;
                v487 = v486 == false;
                if (v487){
                    assert("The indices should be inside the range of the dimension." && v486);
                } else {
                }
                bool v489;
                v489 = 0l <= v449;
                bool v491;
                if (v489){
                    bool v490;
                    v490 = v449 < 1l;
                    v491 = v490;
                } else {
                    v491 = false;
                }
                bool v492;
                v492 = v491 == false;
                if (v492){
                    assert("The indices should be inside the range of the dimension." && v491);
                } else {
                }
                int v494;
                v494 = v449 * 4l;
                int v495;
                v495 = v482 + v494;
                bool v496;
                v496 = 0l <= v480;
                bool v498;
                if (v496){
                    bool v497;
                    v497 = v480 < 1l;
                    v498 = v497;
                } else {
                    v498 = false;
                }
                bool v499;
                v499 = v498 == false;
                if (v499){
                    assert("The indices should be inside the range of the dimension." && v498);
                } else {
                }
                int v501;
                v501 = v480 * 4l;
                int v502;
                v502 = v495 + v501;
                assert("Tensor range check" && 0 <= v480 && v480 < 1l);
                assert("Tensor range check" && 0 <= v482 && v482 < 4l);
                int v503;
                v503 = 4l * v480;
                int v504;
                v504 = v503 + v482;
                v465[v504] = v502;
                v482 += 1l ;
            }
            v480 += 1l ;
        }
        bool v505;
        v505 = v446 && v450;
        bool v506;
        v506 = v505 == false;
        if (v506){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v505);
        } else {
        }
        bool v508;
        v508 = 0l <= v456;
        bool v510;
        if (v508){
            bool v509;
            v509 = v456 < 128l;
            v510 = v509;
        } else {
            v510 = false;
        }
        bool v511;
        v511 = v510 == false;
        if (v511){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v510);
        } else {
        }
        int v513;
        v513 = v456 * 32l;
        int v514;
        v514 = v513 + v445;
        bool v515[4l];
        int v516;
        v516 = 0l;
        while (while_method_1(v516)){
            int v518;
            v518 = 0l;
            while (while_method_2(v518)){
                assert("Tensor range check" && 0 <= v516 && v516 < 1l);
                assert("Tensor range check" && 0 <= v518 && v518 < 4l);
                int v520;
                v520 = 4l * v516;
                int v521;
                v521 = v520 + v518;
                float v522;
                v522 = v462[v521];
                bool v523;
                v523 = v522 == 0.0f;
                bool v524;
                v524 = v523 != true;
                assert("Tensor range check" && 0 <= v516 && v516 < 1l);
                assert("Tensor range check" && 0 <= v518 && v518 < 4l);
                v515[v521] = v524;
                v518 += 1l ;
            }
            v516 += 1l ;
        }
        bool v525;
        v525 = false;
        int v526;
        v526 = 0l;
        while (while_method_1(v526)){
            int v528;
            v528 = 0l;
            while (while_method_2(v528)){
                assert("Tensor range check" && 0 <= v526 && v526 < 1l);
                assert("Tensor range check" && 0 <= v528 && v528 < 4l);
                int v530;
                v530 = 4l * v526;
                int v531;
                v531 = v530 + v528;
                bool v532;
                v532 = v515[v531];
                bool v533;
                v533 = v525 || v532;
                v525 = v533;
                v528 += 1l ;
            }
            v526 += 1l ;
        }
        auto v534 = cooperative_groups::coalesced_threads();
        int v535;
        v535 = threadIdx.x;
        auto v536 = cooperative_groups::labeled_partition(v534,v535);
        Closure7 v537{};
        bool v538;
        v538 = cooperative_groups::reduce(v536, v525, v537);
        if (v538){
            float v539[4l];
            int v540;
            v540 = 0l;
            while (while_method_1(v540)){
                int v542;
                v542 = 0l;
                while (while_method_2(v542)){
                    assert("Tensor range check" && 0 <= v540 && v540 < 1l);
                    assert("Tensor range check" && 0 <= v542 && v542 < 4l);
                    int v544;
                    v544 = 4l * v540;
                    int v545;
                    v545 = v544 + v542;
                    float v546;
                    v546 = v461[v545];
                    float v547;
                    v547 = v462[v545];
                    float v548;
                    v548 = v546 + v547;
                    bool v549;
                    v549 = 0.0f >= v548;
                    float v550;
                    if (v549){
                        v550 = 0.0f;
                    } else {
                        v550 = v548;
                    }
                    assert("Tensor range check" && 0 <= v540 && v540 < 1l);
                    assert("Tensor range check" && 0 <= v542 && v542 < 4l);
                    v539[v545] = v550;
                    v542 += 1l ;
                }
                v540 += 1l ;
            }
            float v551[4l];
            int v552;
            v552 = 0l;
            while (while_method_1(v552)){
                int v554;
                v554 = 0l;
                while (while_method_2(v554)){
                    assert("Tensor range check" && 0 <= v552 && v552 < 1l);
                    assert("Tensor range check" && 0 <= v554 && v554 < 4l);
                    int v556;
                    v556 = 4l * v552;
                    int v557;
                    v557 = v556 + v554;
                    float v558;
                    v558 = v539[v557];
                    bool v559;
                    v559 = 0.0f >= v558;
                    float v560;
                    if (v559){
                        v560 = 0.0f;
                    } else {
                        v560 = v558;
                    }
                    assert("Tensor range check" && 0 <= v552 && v552 < 1l);
                    assert("Tensor range check" && 0 <= v554 && v554 < 4l);
                    v551[v557] = v560;
                    v554 += 1l ;
                }
                v552 += 1l ;
            }
            float v561;
            v561 = 0.0f;
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
                    v568 = v551[v567];
                    float v569;
                    v569 = v561 + v568;
                    v561 = v569;
                    v564 += 1l ;
                }
                v562 += 1l ;
            }
            auto v570 = cooperative_groups::coalesced_threads();
            int v571;
            v571 = threadIdx.x;
            auto v572 = cooperative_groups::labeled_partition(v570,v571);
            Closure0 v573{};
            float v574;
            v574 = cooperative_groups::reduce(v572, v561, v573);
            float v575[4l];
            int v576;
            v576 = 0l;
            while (while_method_1(v576)){
                int v578;
                v578 = 0l;
                while (while_method_2(v578)){
                    assert("Tensor range check" && 0 <= v576 && v576 < 1l);
                    assert("Tensor range check" && 0 <= v578 && v578 < 4l);
                    int v580;
                    v580 = 4l * v576;
                    int v581;
                    v581 = v580 + v578;
                    float v582;
                    v582 = v551[v581];
                    bool v583;
                    v583 = v574 == 0.0f;
                    bool v584;
                    v584 = v583 != true;
                    float v586;
                    if (v584){
                        float v585;
                        v585 = v582 / v574;
                        v586 = v585;
                    } else {
                        v586 = 0.25f;
                    }
                    assert("Tensor range check" && 0 <= v576 && v576 < 1l);
                    assert("Tensor range check" && 0 <= v578 && v578 < 4l);
                    v575[v581] = v586;
                    v578 += 1l ;
                }
                v576 += 1l ;
            }
            float v587[4l];
            int v588;
            v588 = 0l;
            while (while_method_1(v588)){
                int v590;
                v590 = 0l;
                while (while_method_2(v590)){
                    assert("Tensor range check" && 0 <= v588 && v588 < 1l);
                    assert("Tensor range check" && 0 <= v590 && v590 < 4l);
                    int v592;
                    v592 = 4l * v588;
                    int v593;
                    v593 = v592 + v590;
                    float v594;
                    v594 = v460[v593];
                    float v595;
                    v595 = v575[v593];
                    float v596;
                    v596 = v594 + v595;
                    assert("Tensor range check" && 0 <= v588 && v588 < 1l);
                    assert("Tensor range check" && 0 <= v590 && v590 < 4l);
                    v587[v593] = v596;
                    v590 += 1l ;
                }
                v588 += 1l ;
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
                    v604 = v587[v603];
                    float v605;
                    v605 = -v604;
                    bool v606;
                    v606 = v604 >= v605;
                    float v607;
                    if (v606){
                        v607 = v604;
                    } else {
                        v607 = v605;
                    }
                    assert("Tensor range check" && 0 <= v598 && v598 < 1l);
                    assert("Tensor range check" && 0 <= v600 && v600 < 4l);
                    v597[v603] = v607;
                    v600 += 1l ;
                }
                v598 += 1l ;
            }
            float v608;
            v608 = 0.0f;
            int v609;
            v609 = 0l;
            while (while_method_1(v609)){
                int v611;
                v611 = 0l;
                while (while_method_2(v611)){
                    assert("Tensor range check" && 0 <= v609 && v609 < 1l);
                    assert("Tensor range check" && 0 <= v611 && v611 < 4l);
                    int v613;
                    v613 = 4l * v609;
                    int v614;
                    v614 = v613 + v611;
                    float v615;
                    v615 = v597[v614];
                    float v616;
                    v616 = v608 + v615;
                    v608 = v616;
                    v611 += 1l ;
                }
                v609 += 1l ;
            }
            auto v617 = cooperative_groups::coalesced_threads();
            int v618;
            v618 = threadIdx.x;
            auto v619 = cooperative_groups::labeled_partition(v617,v618);
            float v620;
            v620 = cooperative_groups::reduce(v619, v608, v573);
            bool v621;
            v621 = v620 > 100.0f;
            float v623;
            if (v621){
                float v622;
                v622 = 100.0f / v620;
                v623 = v622;
            } else {
                v623 = 1.0f;
            }
            float v624[4l];
            int v625;
            v625 = 0l;
            while (while_method_1(v625)){
                int v627;
                v627 = 0l;
                while (while_method_2(v627)){
                    assert("Tensor range check" && 0 <= v625 && v625 < 1l);
                    assert("Tensor range check" && 0 <= v627 && v627 < 4l);
                    int v629;
                    v629 = 4l * v625;
                    int v630;
                    v630 = v629 + v627;
                    float v631;
                    v631 = v597[v630];
                    float v632;
                    v632 = v623 * v631;
                    assert("Tensor range check" && 0 <= v625 && v625 < 1l);
                    assert("Tensor range check" && 0 <= v627 && v627 < 4l);
                    v624[v630] = v632;
                    v627 += 1l ;
                }
                v625 += 1l ;
            }
            int v633;
            v633 = 0l;
            while (while_method_1(v633)){
                int v635;
                v635 = 0l;
                while (while_method_2(v635)){
                    assert("Tensor range check" && 0 <= v633 && v633 < 1l);
                    assert("Tensor range check" && 0 <= v635 && v635 < 4l);
                    int v637;
                    v637 = 4l * v633;
                    int v638;
                    v638 = v637 + v635;
                    float v639;
                    v639 = v539[v638];
                    float v640;
                    v640 = v624[v638];
                    float v641;
                    v641 = v463[v638];
                    float v642;
                    v642 = v464[v638];
                    assert("Tensor range check" && 0 <= v633 && v633 < 1l);
                    assert("Tensor range check" && 0 <= v635 && v635 < 4l);
                    v460[v638] = v640;
                    v461[v638] = v639;
                    v462[v638] = 0.0f;
                    v463[v638] = v641;
                    v464[v638] = v642;
                    v635 += 1l ;
                }
                v633 += 1l ;
            }
        } else {
        }
        assert("Tensor range check" && 0 <= v456 && v456 < 128l);
        int v643;
        v643 = 0l;
        while (while_method_1(v643)){
            assert("Tensor range check" && 0 <= v643 && v643 < 1l);
            int v645;
            v645 = 4l * v643;
            int v646;
            v646 = v645 + v459;
            assert("Tensor range check" && 0 <= v643 && v643 < 1l);
            int4* v647;
            v647 = reinterpret_cast<int4*>(v460 + v645);
            int4* v648;
            v648 = reinterpret_cast<int4*>(v10 + v646);
            assert("Pointer alignment check" && (unsigned long long)(v647) % 4l == 0 && (unsigned long long)(v648) % 4l == 0);
            *v648 = *v647;
            int4* v649;
            v649 = reinterpret_cast<int4*>(v461 + v645);
            int4* v650;
            v650 = reinterpret_cast<int4*>(v11 + v646);
            assert("Pointer alignment check" && (unsigned long long)(v649) % 4l == 0 && (unsigned long long)(v650) % 4l == 0);
            *v650 = *v649;
            int4* v651;
            v651 = reinterpret_cast<int4*>(v462 + v645);
            int4* v652;
            v652 = reinterpret_cast<int4*>(v12 + v646);
            assert("Pointer alignment check" && (unsigned long long)(v651) % 4l == 0 && (unsigned long long)(v652) % 4l == 0);
            *v652 = *v651;
            int4* v653;
            v653 = reinterpret_cast<int4*>(v463 + v645);
            int4* v654;
            v654 = reinterpret_cast<int4*>(v13 + v646);
            assert("Pointer alignment check" && (unsigned long long)(v653) % 4l == 0 && (unsigned long long)(v654) % 4l == 0);
            *v654 = *v653;
            int4* v655;
            v655 = reinterpret_cast<int4*>(v464 + v645);
            int4* v656;
            v656 = reinterpret_cast<int4*>(v14 + v646);
            assert("Pointer alignment check" && (unsigned long long)(v655) % 4l == 0 && (unsigned long long)(v656) % 4l == 0);
            *v656 = *v655;
            v643 += 1l ;
        }
        v456 += 1l ;
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
    v6 = cp.empty(1024,dtype=cp.float32)
    v7 = cp.empty(2048,dtype=cp.float32)
    v8 = cp.empty(512,dtype=cp.float32)
    v9 = cp.empty(512,dtype=cp.float32)
    v10 = cp.empty(16384,dtype=cp.float32)
    v11 = cp.empty(16384,dtype=cp.float32)
    v12 = cp.empty(16384,dtype=cp.float32)
    v13 = cp.empty(16384,dtype=cp.float32)
    v14 = cp.empty(16384,dtype=cp.float32)
    v15 = 0
    v16 = raw_module.get_function(f"entry{v15}")
    del v15
    v16.max_dynamic_shared_size_bytes = 0 
    v16((1,),(32,),(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14),shared_mem=0)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v16
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
